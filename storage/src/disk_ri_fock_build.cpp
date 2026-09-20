// The disk-backed RI-J Fock builder (disk_ri_fock_build.hpp):
// Create chunk-builds and
// chunk-stores the (uv|P) contraction matrix through BuildRiTensorChunk and
// SaveRiTensorChunk (per-chunk checksum + engine stamp verified on every
// load); BuildFock streams the two RI products as two passes over the
// stored chunks in fixed serial chunk order, with the metric's floored
// eigen-inverse solve in between (the in-memory recipe of ri_engine.cpp,
// duplicated here - its implementation is file-local there - so the solve
// values match). The exchange half runs through the direct builder's
// exchange-only mode with the exact option recipe of the in-memory RI-J
// Create. J accumulates across chunks and drifts in the last ulps - the
// product tolerance is a J-only quantity, measured and
// pinned by the tests, never asserted a priori.

#include "qcx/storage/disk_ri_fock_build.hpp"

#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/linalg/dense_ops.hpp"
#include "qcx/storage/ri_tensor_chunk_store.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::storage {

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// The EXCHANGE half's density precondition, and the criterion the nested
// direct-exchange builder asserts in Debug (DirectJkFockBuilder::BuildFock
// documents it: "Must be SYMMETRIC" - fock_build.hpp). Its K write emits the
// SAME value to the target (a,c) and to its transpose (c,a)
// (internal/fock_contract_kernel.hpp, K1: "the pair-swapped and role-swapped
// twins - density symmetry makes the reads equal"), so the K it returns is
// symmetric BY CONSTRUCTION; that is the correct K only when the density is
// symmetric, since K_uv - K_vu = sum_cd (D_cd - D_dc) (uc|vd) vanishes
// identically only then. The nested check is an assert, which NDEBUG removes
// - so on a Release run a caller that ignored the precondition used to get a
// transpose-locked K and no diagnostic at all. The disk builder's split entry
// points are public, so they check it themselves.
//
// The criterion is the nested builder's own, to the letter: Eigen's
// MatrixXd::isApprox(d, d.transpose()) at NumTraits<double>::dummy_precision
// (1e-12), i.e. the relative-Frobenius test
// ||d - d^T||^2 <= prec^2 * min(||d||^2, ||d^T||^2) with the two norms equal.
// A density assembled as C n C^T is asymmetric at the 1e-16 level and passes
// it; the deliberately asymmetric test fixtures are O(1e-3) and do not.
constexpr double kDensitySymmetryTolerance = 1e-12;

bool DensityIsSymmetric(const CpuTensor2& density, std::size_t n) {
    // The shape refusal belongs to the nested builder (its own
    // kInvalidArgument, "density shape mismatch"), and a shape this loop
    // cannot read is left to it rather than read out of range here.
    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return true;
    }

    double differenceSquared = 0.0;
    double normSquared = 0.0;

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const double value = density(u, v);
            const double difference = value - density(v, u);
            differenceSquared += difference * difference;
            normSquared += value * value;
        }
    }

    return differenceSquared <= kDensitySymmetryTolerance * kDensitySymmetryTolerance * normSquared;
}

// The Create-time chunk plan: ranges of aux SHELLS (in the
// molecule-scoped aux pair-list order - the BuildRiTensorChunk index space)
// whose accumulated function columns stay within the effective chunk size
// max(chunkBytes, 8 * n^2 * maxShellFunctionCount(aux)) - the plan's
// n^2-aware shell floor (a def2-jfit shell column is 8 * n^2 bytes, so the
// floor is the binding term at 5000-function orbital scales). A pure
// function of (basis data, options); chunk boundaries never split a shell.
qcx::Result<std::vector<RiChunkMeta>> PlanChunks(
    const qcx::integrals::ShellPairList& auxPairList,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::size_t n,
    std::size_t chunkBytes) {
    const std::size_t nAuxShells = auxPairList.shells.size();
    std::size_t maxShellFunctions = 0;

    for (const qcx::integrals::ShellInfo& shell : auxPairList.shells)
    {
        maxShellFunctions = std::max(maxShellFunctions, qcx::integrals::ShellFunctionCount(shell));
    }

    const std::size_t n2 = n * n;

    if (n2 == 0 || maxShellFunctions == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the basis sets are empty"});
    }

    const std::size_t effectiveChunkBytes = std::max(chunkBytes, 8 * n2 * maxShellFunctions);
    std::vector<RiChunkMeta> plan;

    for (std::size_t shellStart = 0; shellStart < nAuxShells;)
    {
        std::size_t functions = 0;
        std::size_t shell = shellStart;

        for (; shell < nAuxShells; ++shell)
        {
            const std::size_t shellFunctions =
                qcx::integrals::ShellFunctionCount(auxPairList.shells[shell]);

            if (functions > 0 && 8 * n2 * (functions + shellFunctions) > effectiveChunkBytes)
            {
                break;
            }

            functions += shellFunctions;
        }

        // The floor guarantees the loop above always closes with at least
        // one shell.
        RiChunkMeta meta;
        meta.chunkIndex = plan.size();
        meta.auxShellStart = shellStart;
        meta.auxShellEnd = shell;
        meta.auxFunctionStart = auxPairList.shells[shellStart].functionOffset;
        meta.auxFunctionEnd = auxPairList.shells[shell - 1].functionOffset +
                              qcx::integrals::ShellFunctionCount(auxPairList.shells[shell - 1]);
        meta.orbitalFunctionCount = n;
        plan.push_back(meta);
        shellStart = shell;
    }

    return plan;
}

// The uv-reduced form of one chunk slice: the upper triangle of the bra
// pair, in RiChunkReducedRowIndex order. The chunk build fills BOTH
// triangles of every computed task block with the SAME double (the
// integral is symmetric in its bra pair), so this drops a byte-for-byte
// duplicate half and nothing else - no value is computed, altered or
// approximated here. It is NOT a point-group reduction: the group plays
// no part in it.
Eigen::MatrixXd ReduceToUpperTriangle(const Eigen::MatrixXd& slice, std::size_t n) {
    Eigen::MatrixXd reduced(static_cast<Eigen::Index>(RiChunkReducedRowCount(n)), slice.cols());

    // COLUMN-wise, over contiguous segments: for a fixed u the rows
    // (u, v) with v = u..n-1 are one contiguous run in BOTH layouts
    // (full: u*n+u .. u*n+n-1; reduced: r(u,u) .. r(u,u)+n-1-u), and
    // Eigen is column-major, so a column segment is contiguous too. The
    // row-wise twin - one `reduced.row(r) = slice.row(s)` per pair - is
    // the same copy as 590 strided writes per row, and was measured at
    // 203 ms against this form's 135 ms on the c8h18 chunk, with the two
    // results bit-identical.
    for (std::size_t u = 0; u < n; ++u)
    {
        const Eigen::Index sourceStart = static_cast<Eigen::Index>(u * n + u);
        const Eigen::Index targetStart = static_cast<Eigen::Index>(RiChunkReducedRowIndex(n, u, u));
        const Eigen::Index count = static_cast<Eigen::Index>(n - u);

        for (Eigen::Index column = 0; column < slice.cols(); ++column)
        {
            reduced.col(column).segment(targetStart, count) =
                slice.col(column).segment(sourceStart, count);
        }
    }

    return reduced;
}

// The metric's floored eigen-inverse - the exact recipe of the in-memory
// RI-J Create (ri_engine.cpp): M is SPD but extremely
// ill-conditioned, so the raw factorized solve would amplify the near-null
// directions; eigendecompose once, zero the eigenvalues below
// metricFloorEpsilon * lambdaMax, and keep V diag(invLambda) V^T as the
// per-iteration solve.
qcx::Result<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> MetricEigenInverse(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& auxBasis,
    const qcx::integrals::RiEngineOptions& metricOptions) {
    auto metric = qcx::integrals::BuildAuxMetric(molecule, auxBasis, metricOptions);

    if (!metric.has_value())
    {
        return std::unexpected(metric.error());
    }

    const std::size_t nAux = metric->Shape()[0];
    Eigen::MatrixXd metricMatrix(static_cast<Eigen::Index>(nAux), static_cast<Eigen::Index>(nAux));

    for (std::size_t i = 0; i < nAux; ++i)
    {
        for (std::size_t j = 0; j < nAux; ++j)
        {
            metricMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                (*metric)(i, j);
        }
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(metricMatrix);

    if (solver.info() != Eigen::Success)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the auxiliary-basis metric failed its eigendecomposition"});
    }

    const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
    const double lambdaMax = eigenvalues.maxCoeff();
    const double floor = metricOptions.metricFloorEpsilon * lambdaMax;
    Eigen::VectorXd inverseEigenvalues(eigenvalues.size());

    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i)
    {
        inverseEigenvalues(i) = eigenvalues(i) >= floor ? 1.0 / eigenvalues(i) : 0.0;
    }

    if (lambdaMax <= 0.0 || !(inverseEigenvalues.array() > 0.0).any())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the auxiliary-basis metric is degenerate below the RI-J floor"});
    }

    return std::pair<Eigen::MatrixXd, Eigen::VectorXd>(solver.eigenvectors(),
                                                       std::move(inverseEigenvalues));
}

} // namespace

// The engine-state carrier of the in-memory builder, extended with the
// store identity (the file path, the /molecule names, the chunk plan) the
// two BuildFock passes load against.
struct DiskRiFockBuilder::State {
    State(std::filesystem::path path,
          qcx::molecule::Molecule molecule,
          std::string orbitalName,
          std::string auxName,
          std::vector<RiChunkMeta> chunks,
          qcx::integrals::DirectJkFockBuilder exchange,
          Eigen::MatrixXd metricEigenvectors,
          Eigen::VectorXd inverseMetricEigenvalues,
          std::size_t orbitalFunctionCount) :
        storePath(std::move(path)), molecule(std::move(molecule)),
        orbitalBasisName(std::move(orbitalName)), auxBasisName(std::move(auxName)),
        chunks(std::move(chunks)), exchange(std::move(exchange)),
        metricEigenvectors(std::move(metricEigenvectors)),
        inverseMetricEigenvalues(std::move(inverseMetricEigenvalues)), n(orbitalFunctionCount) {}

    std::filesystem::path storePath;
    qcx::molecule::Molecule molecule;
    std::string orbitalBasisName;
    std::string auxBasisName;
    std::vector<RiChunkMeta> chunks;
    qcx::integrals::DirectJkFockBuilder exchange;
    Eigen::MatrixXd metricEigenvectors;
    Eigen::VectorXd inverseMetricEigenvalues;
    std::size_t n;
};

qcx::Result<DiskRiFockBuilder> DiskRiFockBuilder::Create(
    const std::filesystem::path& storePath,
    qcx::molecule::Molecule molecule,
    const qcx::basisset::BasisSet& orbitalBasis,
    const qcx::basisset::BasisSet& auxBasis,
    std::string_view orbitalBasisName,
    std::string_view auxBasisName,
    const CpuTensor2& coreHamiltonian,
    const DiskRiFockOptions& options) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    auto pairList = qcx::integrals::BuildShellPairs(molecule, orbitalBasis);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto auxPairList = qcx::integrals::BuildShellPairs(molecule, auxBasis);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    if (auxPairList->shells.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the auxiliary basis is empty"});
    }

    const std::size_t n = pairList->functionCount;
    auto plan = PlanChunks(*auxPairList, n, options.chunkBytes);

    if (!plan.has_value())
    {
        return std::unexpected(plan.error());
    }

    // The row layout is a per-store property, not a per-chunk one: chunks
    // partition the aux (column) index only, so every chunk carries the
    // same row space. Stamped on the plan here, so the metas the two
    // BuildFock passes load against carry it too.
    if (options.reducedRowLayout)
    {
        for (RiChunkMeta& meta : *plan)
        {
            meta.storedRowCount = RiChunkReducedRowCount(n);
        }
    }

    // The exchange half - the EXACT option recipe of the in-memory RI-J
    // Create (ri_engine.cpp): exchange-only through the direct builder with
    // density screening on and the fp32 lane off at kTight.
    qcx::integrals::FockBuildOptions exchangeOptions;
    exchangeOptions.accuracy = options.accuracy;
    exchangeOptions.maxBatchBytes = options.maxBatchBytes;
    exchangeOptions.useDensityScreening = true;
    exchangeOptions.useCertifiedMixedPrecision =
        options.accuracy != qcx::integrals::AccuracyPreset::kTight;
    exchangeOptions.buildExchangeOnly = true;
    auto exchange = qcx::integrals::DirectJkFockBuilder::Create(
        molecule, orbitalBasis, coreHamiltonian, exchangeOptions);

    if (!exchange.has_value())
    {
        return std::unexpected(exchange.error());
    }

    // The metric's floored eigen-inverse (the in-memory recipe).
    qcx::integrals::RiEngineOptions metricOptions;
    metricOptions.accuracy = options.accuracy;
    metricOptions.maxBatchBytes = options.maxBatchBytes;
    metricOptions.metricFloorEpsilon = options.metricFloorEpsilon;
    auto metricInverse = MetricEigenInverse(molecule, auxBasis, metricOptions);

    if (!metricInverse.has_value())
    {
        return std::unexpected(metricInverse.error());
    }

    // The one-time chunked build + chunked store (BuildRiTensorChunk, then
    // SaveRiTensorChunk): each chunk's task list is chunk-scoped (the full
    // nPairs x nAuxShells list never exists) and every saved dataset carries
    // its checksum and engine stamp. A store file that already carries chunks
    // refuses (the append-only store contract).
    qcx::integrals::RiEngineOptions chunkOptions;
    chunkOptions.accuracy = options.accuracy;
    chunkOptions.maxBatchBytes = options.maxBatchBytes;

    for (const RiChunkMeta& meta : *plan)
    {
        auto chunk = qcx::integrals::BuildRiTensorChunk(
            molecule, orbitalBasis, auxBasis, meta.auxShellStart, meta.auxShellEnd, chunkOptions);

        if (!chunk.has_value())
        {
            return std::unexpected(chunk.error());
        }

        // The reduced layout stores the slice's upper triangle; the full
        // one stores the slice as built (moved, never copied - a copy here
        // is another n^2 x k_c allocation for nothing). Nothing is
        // recomputed either way.
        Eigen::MatrixXd stored;

        if (options.reducedRowLayout)
        {
            stored = ReduceToUpperTriangle(*chunk, n);
        } else
        {
            stored = std::move(*chunk);
        }

        auto saved =
            SaveRiTensorChunk(storePath, molecule, orbitalBasisName, auxBasisName, meta, stored);

        if (!saved.has_value())
        {
            return std::unexpected(saved.error());
        }
    }

    auto state = std::make_shared<State>(storePath,
                                         std::move(molecule),
                                         std::string(orbitalBasisName),
                                         std::string(auxBasisName),
                                         std::move(*plan),
                                         std::move(*exchange),
                                         std::move(metricInverse->first),
                                         std::move(metricInverse->second),
                                         n);
    return DiskRiFockBuilder(std::move(state));
}

const std::vector<RiChunkMeta>& DiskRiFockBuilder::Chunks() const noexcept {
    return _state->chunks;
}

qcx::Result<Eigen::VectorXd> DiskRiFockBuilder::BuildRiCoulombVector(
    const CpuTensor2& density) const {
    const State& state = *_state;
    const std::size_t n = state.n;
    const std::size_t nAux = static_cast<std::size_t>(state.inverseMetricEigenvalues.size());
    const Eigen::Index n2 = static_cast<Eigen::Index>(n * n);

    Eigen::VectorXd dVec(n2);

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            dVec(static_cast<Eigen::Index>(u * n + v)) = density(u, v);
        }
    }

    // The store's row layout decides the working row count. Under the
    // uv-reduced layout the pass-1 operand is gathered ONCE, here, outside
    // the chunk loop: dRed[u,v] = d(u,v) + d(v,u), which is exactly the sum
    // of the two terms the full layout contributes at that pair - an
    // algebraic identity for ANY density, so no symmetry of the caller's
    // matrix is assumed. The chunk loop below is unchanged in orientation
    // (still d^T * I_c), so no per-chunk transpose copy appears.
    //
    // THAT IS THIS HALF'S PROPERTY ALONE, and it does not travel to the
    // builder: it holds because J is symmetric for any D - J_uv = sum_cd
    // (uv|cd) D_cd and (uv|cd) = (vu|cd) - whereas the exchange half's K is
    // symmetry-locked by construction and therefore REQUIRES a symmetric
    // density (BuildExchangeOnly refuses the other case; the disk-builder
    // header states the requirement per entry point).
    const bool reduced = !state.chunks.empty() && state.chunks.front().storedRowCount != 0;
    const std::size_t nWork = reduced ? RiChunkReducedRowCount(n) : n * n;
    Eigen::VectorXd dWork(static_cast<Eigen::Index>(nWork));

    if (reduced)
    {
        for (std::size_t u = 0; u < n; ++u)
        {
            for (std::size_t v = u; v < n; ++v)
            {
                const Eigen::Index flat = static_cast<Eigen::Index>(u * n + v);
                dWork(static_cast<Eigen::Index>(RiChunkReducedRowIndex(n, u, v))) =
                    u == v ? dVec(flat) : dVec(flat) + dVec(static_cast<Eigen::Index>(v * n + u));
            }
        }
    } else
    {
        dWork = dVec;
    }

    // Pass 1: v_c^T = d^T * I_c per chunk, accumulated serially in chunk
    // order (the no-transpose orientation: no per-chunk
    // transpose copy, the 1x chunk arena). Each load is checksum- and
    // engine-stamp-verified on load. The per-column dot products have the
    // same term order as the in-memory I^T d, so v is bit-identical up to
    // BLAS-internal k-loop blocking.
    Eigen::VectorXd v = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nAux));
    const Eigen::MatrixXd dRow = dWork.transpose();

    for (const RiChunkMeta& meta : state.chunks)
    {
        auto chunk = LoadRiTensorChunk(
            state.storePath, state.molecule, state.orbitalBasisName, state.auxBasisName, meta);

        if (!chunk.has_value())
        {
            return std::unexpected(chunk.error());
        }

        auto partial = qcx::linalg::DenseMultiply(dRow, *chunk);

        if (!partial.has_value())
        {
            return std::unexpected(partial.error());
        }

        v.segment(static_cast<Eigen::Index>(meta.auxFunctionStart),
                  static_cast<Eigen::Index>(meta.auxFunctionEnd - meta.auxFunctionStart)) =
            partial->row(0).transpose();
    }

    // w = V diag(invLambda) V^T v - the in-memory solve verbatim (the
    // eigen-inverse steps stay Eigen: DenseMultiply is for plain products,
    // not solves or decompositions).
    const Eigen::VectorXd w =
        state.metricEigenvectors *
        state.inverseMetricEigenvalues.cwiseProduct(state.metricEigenvectors.transpose() * v);

    // Pass 2: j = sum_c I_c * w_c, accumulated serially in the SAME chunk
    // order as pass 1 (the fixed serial accumulation order is documented -
    // the combine-order precedent). J alone accumulates across chunks
    // and drifts in the last ulps vs the single full-product of the
    // in-memory path. Under the reduced layout `j` holds one entry per
    // unordered pair and is scattered once, after the loop - each pair's
    // dot product sees the identical row the full layout holds twice, so
    // the scattered values are bit-identical to the full layout's.
    Eigen::VectorXd j = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nWork));

    for (const RiChunkMeta& meta : state.chunks)
    {
        auto chunk = LoadRiTensorChunk(
            state.storePath, state.molecule, state.orbitalBasisName, state.auxBasisName, meta);

        if (!chunk.has_value())
        {
            return std::unexpected(chunk.error());
        }

        const Eigen::MatrixXd wChunk =
            w.segment(static_cast<Eigen::Index>(meta.auxFunctionStart),
                      static_cast<Eigen::Index>(meta.auxFunctionEnd - meta.auxFunctionStart));
        auto jChunk = qcx::linalg::DenseMultiply(*chunk, wChunk);

        if (!jChunk.has_value())
        {
            return std::unexpected(jChunk.error());
        }

        j += jChunk->col(0);
    }

    // The reduced layout's scatter, applied here so the returned vector is
    // already in the fused call's flat layout: pair (u,v) and its mirror
    // (v,u) both read the one stored entry. No copy of the n^2 vector is
    // made for it. The factor of two is the family's - this is 2 J_RI(rho),
    // the quantity BuildCoulombOnly returns and BuildFock adds.
    Eigen::VectorXd coulomb(n2);

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const Eigen::Index flat = static_cast<Eigen::Index>(u * n + v);
            const Eigen::Index source = reduced ? static_cast<Eigen::Index>(RiChunkReducedRowIndex(
                                                      n, u < v ? u : v, u < v ? v : u))
                                                : flat;
            coulomb(flat) = 2.0 * j(source);
        }
    }

    return coulomb;
}

qcx::Result<CpuTensor2> DiskRiFockBuilder::BuildExchangeOnly(
    const CpuTensor2& density, qcx::integrals::FockBuildStats* statsOut) const {
    // The exchange half - the ONLY H carrier (the RI path contributes no
    // H), the same accounting as the in-memory BuildFock and the same
    // nested builder: one accuracy preset, one batch cap, one option recipe.
    //
    // The nested builder's density precondition arrives with the delegation,
    // and it is enforced here rather than left to the nested Debug assert:
    // its K is symmetry-locked by construction (see DensityIsSymmetric), so
    // an asymmetric density is a caller error that Release would otherwise
    // answer with a silently transposed K.
    if (!DensityIsSymmetric(density, _state->n))
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the exchange half requires a symmetric density: the nested direct-exchange "
            "builder's K transpose-writes make K symmetric by construction, and its Debug "
            "assert is the only check NDEBUG leaves out (fock_build.hpp, "
            "DirectJkFockBuilder::BuildFock)"});
    }

    return _state->exchange.BuildFock(density, nullptr, statsOut);
}

qcx::Result<CpuTensor2> DiskRiFockBuilder::BuildCoulombOnly(
    const CpuTensor2& density, qcx::integrals::FockBuildStats* statsOut) const {
    const std::size_t n = _state->n;
    auto coulombVector = BuildRiCoulombVector(density);

    if (!coulombVector.has_value())
    {
        return std::unexpected(coulombVector.error());
    }

    // The measured zero (the in-memory split call's rule for this call
    // shape): the streamed contractions evaluate no quartets, so a caller
    // that asked for counts gets this call's structural zeros rather than
    // whatever the previous call left in the struct.
    if (statsOut != nullptr)
    {
        *statsOut = qcx::integrals::FockBuildStats{};
    }

    auto tensor = CpuTensor2::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const Eigen::Index flat = static_cast<Eigen::Index>(u * n + v);
            (*tensor)(u, v) = (*coulombVector)(flat);
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

qcx::Result<CpuTensor2> DiskRiFockBuilder::BuildFock(
    const CpuTensor2& density, qcx::integrals::FockBuildStats* statsOut) const {
    const std::size_t n = _state->n;

    // The exchange half first, so an exchange failure precedes any RI work -
    // the order the fused call has always taken. Both halves come from the
    // SAME code the split entry points run: the fused call is their sum and
    // nothing else, which is what keeps a half and the fused build from
    // disagreeing about the convention's factor of two or about H.
    auto exchangeFock = BuildExchangeOnly(density, statsOut);

    if (!exchangeFock.has_value())
    {
        return std::unexpected(exchangeFock.error());
    }

    auto coulombVector = BuildRiCoulombVector(density);

    if (!coulombVector.has_value())
    {
        return std::unexpected(coulombVector.error());
    }

    auto tensor = CpuTensor2::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const Eigen::Index flat = static_cast<Eigen::Index>(u * n + v);
            (*tensor)(u, v) = (*exchangeFock)(u, v) + (*coulombVector)(flat);
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

DiskRiFockBuilder::DiskRiFockBuilder(std::shared_ptr<const State> state) :
    _state(std::move(state)) {}

} // namespace qcx::storage
