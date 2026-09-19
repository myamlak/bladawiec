#include "qcx/integrals/ri_full_fock.hpp"

#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "internal/tensor_eigen_bridge.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/ri_occ_k.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals {
namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// The rank-2 working class of one BuildFock call: F, K, J, the density and its
// Eigen copy, the two Coulomb products and the exchange bundle - eight n x n
// matrices, the same count the shipped composition's own working set carries.
// Charged so the rung's estimate covers the call, not only the Create.
constexpr std::size_t kRank2WorkingMatrices = 8;

// The auxiliary shell range of the next blocked-rung chunk: whole shells from
// \p firstShell up to (exclusive) the first shell that would push the range's
// FUNCTION width past \p maxFunctions. Always at least one shell, so a range
// that is itself wider than the cap still advances and the chunk count stays
// finite - the loop terminates and the rung's refusal (not a hang) is what
// answers a cap below one shell's slice.
std::size_t ChunkShellEnd(const ShellPairList& auxPairList,
                          // (firstShell, maxFunctions) is the index-then-count pair the sweep is
                          // index-then-count pair
                          // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                          std::size_t firstShell,
                          std::size_t maxFunctions) noexcept {
    const std::size_t nShells = auxPairList.shells.size();
    const std::size_t firstFunction = auxPairList.shells[firstShell].functionOffset;
    std::size_t end = firstShell + 1;

    while (end < nShells)
    {
        const ShellInfo& shell = auxPairList.shells[end];
        const std::size_t width = shell.functionOffset + ShellFunctionCount(shell) - firstFunction;

        if (width > maxFunctions)
        {
            break;
        }

        ++end;
    }

    return end;
}

// The FUNCTION width of the aux shell range [firstShell, end) - the chunk's
// column count, and the number the slice's byte estimate is built from (never
// the shell count: a shell carries 2l+1 functions).
std::size_t ChunkFunctionWidth(const ShellPairList& auxPairList,
                               std::size_t firstShell,
                               std::size_t end) noexcept {
    const ShellInfo& last = auxPairList.shells[end - 1];
    return last.functionOffset + ShellFunctionCount(last) -
           auxPairList.shells[firstShell].functionOffset;
}

// The ladder refusal: neither rung fits the budget, and the text names every
// rung in the ladder's ORDER (direct screened first, then
// batched/blocked, then recompute, then disk LAST - so a refusal may never
// present an exclusion, and it must say which rung the caller is looking at
// and which one this builder does not have).
std::string LadderRefusal(std::size_t fastBytes,
                          std::size_t blockedBytes,
                          std::size_t remaining,
                          std::size_t minimumSliceBytes) {
    return "RiFullFockBuilder: neither rung of the memory ladder fits this workspace budget - the "
           "fast rung's Create-plus-first-call peak is " +
           std::to_string(fastBytes) +
           " bytes, the blocked rung's minimum (one auxiliary "
           "shell range) is " +
           std::to_string(blockedBytes) + " bytes (" + std::to_string(minimumSliceBytes) +
           " bytes of it the slice), and the budget has " + std::to_string(remaining) +
           " bytes remaining. Reinstatement options, in the ladder's order: (1) raise the cap "
           "(memory_cap_gib in [resources]) - a larger budget engages the blocked rung on its "
           "own; (2) the per-iteration 3-center RECOMPUTE rung (the RI-J path's landed light "
           "rung, ri_engine.hpp LightRungPayload) is NOT implemented for this builder: it is "
           "the rung AFTER the blocked one and still absent here; (3) the disk-backed "
           "tensor store (the ri_j_link path's method.ri_tensor_mode = \"disk\" knob).";
}

// The contraction layout's vector form (ri_occ_k.hpp): element u*n + v of the
// vector is element (u, v) of the matrix. Both directions below are written as
// loops rather than as memory maps on purpose - `matrix.data()` is
// column-major, so a row-major view of the same buffer is the TRANSPOSE, and
// the one place that distinction goes wrong is a silently transposed Coulomb
// term. The cost is n^2 against products that are n^2 * nAux, so clarity here
// is free; the same two loops are the ones I1's acceptance test carries.
Eigen::VectorXd ToContractionVector(const Eigen::MatrixXd& matrix) {
    const Eigen::Index n = matrix.rows();
    Eigen::VectorXd vector(n * n);

    for (Eigen::Index u = 0; u < n; ++u)
    {
        for (Eigen::Index v = 0; v < n; ++v)
        {
            vector(u * n + v) = matrix(u, v);
        }
    }

    return vector;
}

Eigen::MatrixXd FromContractionVector(const Eigen::VectorXd& vector, Eigen::Index n) {
    Eigen::MatrixXd matrix(n, n);

    for (Eigen::Index u = 0; u < n; ++u)
    {
        for (Eigen::Index v = 0; v < n; ++v)
        {
            matrix(u, v) = vector(u * n + v);
        }
    }

    return matrix;
}

// The auxiliary-pair density screen's per-call structure (increment I4): the
// surviving (orbital pair block, auxiliary shell) cells, in the CSR shape the
// module's neighbor list already uses (row offsets + indices, fock_screen.hpp
// BuildNeighborList) - one row per auxiliary SHELL, because the auxiliary
// bound is a shell bound and the columns of a shell share it.
//
// The gate is the density-weighted form of the 3-center build's own cutoff:
// a cell survives iff densityWeight * Q_pair * Q_auxShell >= threshold, with
// densityWeight the max |rho| over the pair block
// (internal::BuildShellPairMaxDensity - the direct family's bound, reused
// rather than re-derived), Q_pair the orbital pair's Schwarz bound and
// Q_auxShell the auxiliary shell's. Drop iff strictly less, keep at equality:
// the legacy gate's boundary convention (fock_screen.hpp QuartetDensityGate).
// REBUILT PER BuildFock CALL from the density that call receives, never
// cached across calls - the max-density vector's own contract
// (fock_screen.hpp BuildShellPairMaxDensity), inherited here.
struct AuxScreenStructure {
    std::vector<std::size_t> offsets; ///< One per auxiliary shell, plus a final sentinel.
    std::vector<std::size_t> indices; ///< The surviving canonical pair indices.
    std::size_t totalCells = 0;
    std::size_t survivingCells = 0;

    std::size_t bytes() const noexcept {
        return offsets.size() * sizeof(std::size_t) + indices.size() * sizeof(std::size_t);
    }
};

AuxScreenStructure BuildAuxScreenStructure(
    const Eigen::MatrixXd& density,
    const ShellPairList& pairList,
    // (pairBounds, auxShellBounds) are the two screening sides, one bound table each.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const std::vector<double>& pairBounds,
    const std::vector<double>& auxShellBounds,
    double threshold) {
    AuxScreenStructure structure;
    const std::size_t pairCount = pairList.pairs.size();
    const std::size_t shellCount = auxShellBounds.size();
    // The density weight source: the shell-compressed max-density vector, one
    // entry per canonical pair (the direct family's own bound).
    const std::vector<double> densityMaxima = internal::BuildShellPairMaxDensity(density, pairList);

    structure.offsets.resize(shellCount + 1, 0);
    structure.totalCells = pairCount * shellCount;

    for (std::size_t auxShell = 0; auxShell < shellCount; ++auxShell)
    {
        const double shellBound = auxShellBounds[auxShell];

        for (std::size_t pair = 0; pair < pairCount; ++pair)
        {
            if (densityMaxima[pair] * pairBounds[pair] * shellBound < threshold)
            {
                continue;
            }

            structure.indices.push_back(pair);
            ++structure.survivingCells;
        }

        structure.offsets[auxShell + 1] = structure.indices.size();
    }

    return structure;
}

// J = B (B^T d) through the metric-transformed tensor - the Coulomb half of
// the composition, and the identity I1 anchored against the shipped eigen-path
// J (ri_occ_k.hpp's file comment; measured at 8.7e-14 relative on
// water/def2-SVP).
Eigen::MatrixXd CoulombFromTransformed(const Eigen::MatrixXd& transformed,
                                       const Eigen::MatrixXd& density) {
    const Eigen::VectorXd weights = transformed.transpose() * ToContractionVector(density);
    const Eigen::VectorXd coulomb = transformed * weights;
    return FromContractionVector(coulomb, density.rows());
}

// The screened half of the composition (increment I4): one pass over the
// surviving pair blocks that produces BOTH contractions' inputs - the Coulomb
// weights u_P = sum_uv B^P_uv rho_uv and the occupied transform
// Bocc[(u,i),P] = sum_v B^P_uv C_vi - because both walk the same blocks of
// the same column.
//
// The walk is in the CONTRACTION layout's own view: column P of B is the n x n
// matrix (u, v) in row-major order (ri_occ_k.cpp's RowMajorView), so a
// canonical pair's two blocks are 2-D blocks of it. A canonical pair (I, J)
// with I != J covers the rows u in I, v in J AND - by the (uv|P) symmetry the
// pair list folds - the mirrored rows u in J, v in I, which carry the
// transposed values; both are walked, or the lower triangle's rows would be
// dropped silently and the result would be half a Coulomb term.
//
// The exchange contraction is NOT walked here: `Bocc` stays dense, and the
// rank-nOcc update stays the module's own (BuildRiExchangeMatrix) - see
// BuildFock's contract for why the screen cannot narrow it.
// The parameters are inherently same-typed (three matrix views, two parallel
// shell-range vectors); their order is the layout's and is fixed here.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void SweepScreenedColumns(const Eigen::MatrixXd& transformed,
                          const Eigen::MatrixXd& density,
                          const Eigen::MatrixXd& occupiedOrbitals,
                          const ShellPairList& pairList,
                          const AuxScreenStructure& structure,
                          // (auxShellFirstFunction, auxShellFunctionCount) is the offset-then-count
                          // pair, named at the offset-then-count pair
                          // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                          const std::vector<std::size_t>& auxShellFirstFunction,
                          const std::vector<std::size_t>& auxShellFunctionCount,
                          Eigen::MatrixXd& occTransformed,
                          Eigen::VectorXd& coulombWeights) {
    using RowMajorView =
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using ConstRowMajorView =
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

    const Eigen::Index n = density.rows();
    const Eigen::Index nOcc = occupiedOrbitals.cols();

    for (std::size_t auxShell = 0; auxShell < auxShellFunctionCount.size(); ++auxShell)
    {
        const Eigen::Index firstColumn = static_cast<Eigen::Index>(auxShellFirstFunction[auxShell]);
        const Eigen::Index shellWidth = static_cast<Eigen::Index>(auxShellFunctionCount[auxShell]);
        const std::size_t begin = structure.offsets[auxShell];
        const std::size_t end = structure.offsets[auxShell + 1];

        for (Eigen::Index column = 0; column < shellWidth; ++column)
        {
            const Eigen::Index p = firstColumn + column;
            const ConstRowMajorView block(transformed.col(p).data(), n, n);
            RowMajorView out(occTransformed.col(p).data(), n, nOcc);
            double weight = 0.0;

            for (std::size_t entry = begin; entry < end; ++entry)
            {
                const ShellPairIndex& pair = pairList.pairs[structure.indices[entry]];
                const ShellInfo& bra = pairList.shells[pair.i];
                const ShellInfo& ket = pairList.shells[pair.j];
                const Eigen::Index braFirst = static_cast<Eigen::Index>(bra.functionOffset);
                const Eigen::Index ketFirst = static_cast<Eigen::Index>(ket.functionOffset);
                const Eigen::Index braWidth = static_cast<Eigen::Index>(ShellFunctionCount(bra));
                const Eigen::Index ketWidth = static_cast<Eigen::Index>(ShellFunctionCount(ket));

                // The (u in I, v in J) block: the untransposed half.
                const auto sourceUpper = block.block(braFirst, ketFirst, braWidth, ketWidth);
                weight += (sourceUpper.cwiseProduct(
                               density.block(braFirst, ketFirst, braWidth, ketWidth)))
                              .sum();
                out.block(braFirst, 0, braWidth, nOcc).noalias() +=
                    sourceUpper * occupiedOrbitals.block(ketFirst, 0, ketWidth, nOcc);

                if (pair.i == pair.j)
                {
                    continue;
                }

                // The mirrored (u in J, v in I) block of the same canonical pair.
                const auto sourceLower = block.block(ketFirst, braFirst, ketWidth, braWidth);
                weight += (sourceLower.cwiseProduct(
                               density.block(ketFirst, braFirst, ketWidth, braWidth)))
                              .sum();
                out.block(ketFirst, 0, ketWidth, nOcc).noalias() +=
                    sourceLower * occupiedOrbitals.block(braFirst, 0, braWidth, nOcc);
            }

            coulombWeights(p) = weight;
        }
    }
}

} // namespace

struct RiFullFockBuilder::State {
    Eigen::MatrixXd coreHamiltonian; ///< H = T + V, n x n.
    Eigen::MatrixXd transformed; ///< B, rows u*n + v, columns P.
    /// The Create-time term counters (ri_full_fock.hpp TermCounters).
    RiTermCounters termCounters;
    /// The Create-time rung decision (ri_full_fock.hpp ModeInfo). Default =
    /// not engaged: the no-budget path takes the fast rung without reading a
    /// budget, and a record claiming otherwise would state a read that did not
    /// happen.
    RiFullFockModeInfo modeInfo;
    /// The auxiliary-pair density screen's Create-time stores (increment I4):
    /// the orbital pair list and the pair/aux shell Schwarz bounds the per-call
    /// gate reads, plus the aux shells' function ranges. Empty when the screen
    /// is off (0 threshold): the dense path retains no screening store.
    ShellPairList screenPairs;
    std::vector<double> screenPairBounds;
    std::vector<double> screenAuxShellBounds;
    std::vector<std::size_t> screenAuxShellFirstFunction;
    std::vector<std::size_t> screenAuxShellFunctionCount;
    /// The screen's cutoff (0 = off). Create-time and const, so the per-call
    /// structure is the only thing a call builds.
    double screenThreshold = 0.0;
    /// The seam cap every build of this Create ran at - the caller's
    /// `maxBatchBytes`, or the rung decision's clamped one when a budget
    /// engaged (effectiveOptions.maxBatchBytes; ri_engine.cpp's own rule that
    /// a decision's cap is the cap every later build runs at). The batched
    /// traversal's block width is clamped against it (ri_occ_k.hpp
    /// RiContractionBlockWidth), so the traversal can never take a block wider
    /// than the budget the caller already declared.
    std::size_t maxBatchBytes = 0;
    /// The batched contraction traversal's residency target (increment I5,
    /// RiContractionBatchOptions::blockTargetBytes); 0 - the default -
    /// selects the three-call chain. Create-time and const: the block width
    /// is a function of numbers Create already has, so nothing is retained
    /// for the traversal and no per-call decision is left to make.
    std::size_t contractionBlockBytes = 0;
};

RiFullFockBuilder::RiFullFockBuilder(std::shared_ptr<const State> state) :
    _state(std::move(state)) {}

qcx::Result<RiFullFockBuilder> RiFullFockBuilder::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
    const RiEngineOptions& options,
    const RiAuxScreenOptions& screenOptions,
    const RiContractionBatchOptions& batchOptions) {
    if (options.maxBatchBytes == 0)
    {
        // The same guard ri_engine.cpp's Create runs, and for the same reason:
        // the decision's scratch clamp divides by the batch cap, so a zero cap
        // must be refused here rather than reaching a kernel. (Before this
        // landing the refusal came from BuildRiTensorChunk inside Create, with
        // the same code and the same message - the caller sees no change.)
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    // forceLightRung asks for the per-iteration RECOMPUTE rung by name - the
    // rung that keeps no tensor at all and re-evaluates the 3-center blocks
    // every iteration (ri_engine.hpp LightRungPayload, the RI-J path's landed
    // light rung). This builder's ladder stops one rung earlier: the blocked
    // rung below keeps B, so accepting the knob would state a mode that does
    // not run - the substitution the refusal exists to prevent.
    if (options.forceLightRung)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "RiFullFockBuilder does not implement the per-iteration 3-center recompute "
                       "rung, so forceLightRung cannot be honoured: the rung that runs here is "
                       "the block-batched one (RiFullFockRung::kBlocked), selected by a "
                       "workspaceBudget, and it still retains the metric-transformed tensor. "
                       "Leave the flag false and pass a budget for the memory-bounded rung"});
    }

    // The point-group reduction, refused by name for the same reason as the
    // mode above: this path's 3-center tensor is built through
    // BuildRiTensorChunk, which counts the screened task list (TermCounters)
    // but carries no reduction - BuildRiTensorChunk ignores
    // RiEngineOptions::symmetryOrbitExpansion - so a non-null reduction would
    // be accepted and silently ignored. That is the substitution the refusal
    // forbids, and the honest surface is this refusal: a later increment is
    // aimed by the counters, not by a field that never reaches a kernel.
    if (options.symmetryReduction != nullptr || options.auxSymmetryReduction != nullptr)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "RiFullFockBuilder does not implement the point-group reduction, so a "
                       "non-null symmetryReduction (or auxSymmetryReduction) cannot be honoured: "
                       "the 3-center tensor here is counted but never reduced, and the fields "
                       "would be accepted and silently ignored. Clear both fields for this "
                       "builder (the reduction is the RI-J path's option set, ri_engine.hpp "
                       "RiJkFockBuilder)"});
    }

    // The screen and the batched traversal are ALTERNATIVES here, refused
    // rather than composed - the same rule, one increment on from the two
    // refusals above. The batched traversal walks the whole auxiliary index
    // (its partition is the aux FUNCTION index, not the surviving-cell grid),
    // so it cannot carry the screen's per-call surviving (pair, aux shell)
    // structure: a builder created with both would run the traversal and the
    // screen's threshold would be accepted and silently dropped. The refusal
    // names both options and what each one costs to keep, so a caller reads
    // which knob to clear instead of guessing.
    if (screenOptions.threshold > 0.0 && batchOptions.blockTargetBytes != 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "RiFullFockBuilder cannot combine the auxiliary-pair density screen "
                       "(threshold " +
                           std::to_string(screenOptions.threshold) +
                           ") with the batched contraction traversal (blockTargetBytes " +
                           std::to_string(batchOptions.blockTargetBytes) +
                           "): the batched traversal partitions the AUXILIARY index and reads "
                           "the whole 3-center tensor per block, so it does not carry the "
                           "screen's surviving (pair, aux shell) structure - accepting both "
                           "would silently drop the screen. Ask for one: the screen with "
                           "blockTargetBytes 0, or the batched traversal with threshold 0"});
    }

    if (coreHamiltonian.Shape()[0] != coreHamiltonian.Shape()[1])
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the core Hamiltonian must be a square matrix"});
    }

    // The raw 3-center tensor through the chunked public route over the full
    // auxiliary shell range: it IS the RI contraction layout (rows u*n + v,
    // columns P), so no flattening copy is needed, and the route is the one
    // ri_engine_chunk_test pins against the monolithic BuildRiTensor.
    auto auxPairs = BuildShellPairs(molecule, auxBasisSet);

    if (!auxPairs.has_value())
    {
        return std::unexpected(auxPairs.error());
    }

    const std::size_t n = coreHamiltonian.Shape()[0];
    const std::size_t nAuxFuncs = auxPairs->functionCount;
    // The tensor class: one n^2 x nAux array in doubles. Both rungs build it;
    // what differs is how many copies are live at the peak (see the decision
    // block below), which is why the rung's estimate is a multiple of this.
    const std::size_t tensorBytes = 8 * n * n * nAuxFuncs;
    const std::size_t bytesPerFunction = 8 * n * n;

    // The screen's Create-time stores and the ladder decision's own inputs
    // (the screen's own inputs): the orbital pair list, its
    // Schwarz bounds, the aux shell bounds and the aux shell FUNCTION ranges.
    // Both consumers need them, so they are swept once here and shared - the
    // Schwarz sweep is an integral pass over the pair list, and running it
    // twice for one Create would double a Create-time cost for no result.
    // Built when EITHER consumer is engaged: a builder with no budget and no
    // screen retains none of them (the pre-screen path, byte for byte).
    const bool screenEngaged = screenOptions.threshold > 0.0;
    ShellPairList pairList;
    std::vector<double> orbitalBounds;
    std::vector<double> auxShellBounds;
    std::vector<std::size_t> auxShellFirstFunction;
    std::vector<std::size_t> auxShellFunctionCount;

    if (screenEngaged || options.workspaceBudget != nullptr)
    {
        auto builtPairs = BuildShellPairs(molecule, basisSet);

        if (!builtPairs.has_value())
        {
            return std::unexpected(builtPairs.error());
        }

        pairList = std::move(*builtPairs);
        auto schwarzOrbital = ComputeSchwarzBounds(molecule, basisSet);

        if (!schwarzOrbital.has_value())
        {
            return std::unexpected(schwarzOrbital.error());
        }

        orbitalBounds = std::move(*schwarzOrbital);
        auto schwarzAux = ComputeSchwarzBounds(molecule, auxBasisSet);

        if (!schwarzAux.has_value())
        {
            return std::unexpected(schwarzAux.error());
        }

        // The per-aux-shell bounds: the aux pair list's diagonal (the kets are
        // single shells), the same extraction CountSchwarzSurvivingRiTasks'
        // callers run - so the counted task-grid charge equals the count the
        // chunked builds realize, never the unconditional dense grid (the
        // never-under counted treatment).
        auxShellBounds.resize(auxPairs->shells.size());
        auxShellFirstFunction.resize(auxPairs->shells.size());
        auxShellFunctionCount.resize(auxPairs->shells.size());

        for (std::size_t auxShell = 0; auxShell < auxPairs->shells.size(); ++auxShell)
        {
            auxShellBounds[auxShell] = (*schwarzAux)[PairIndexOf(auxShell, auxShell, *auxPairs)];
            auxShellFirstFunction[auxShell] = auxPairs->shells[auxShell].functionOffset;
            auxShellFunctionCount[auxShell] = ShellFunctionCount(auxPairs->shells[auxShell]);
        }
    }

    // The screen's own class (increment I4): the stores Create retains for it
    // (pair list, Schwarz bounds, aux shell ranges) plus the per-call
    // surviving structure at its WORST case - one entry per (pair, aux shell)
    // cell plus the shell offsets, the never-under form the task-list charge
    // uses. Zero when the screen is off, and recorded either way so a reader
    // can tell a screened builder from an unscreened one.
    const std::size_t screenStoreBytes =
        screenEngaged ? pairList.shells.size() * sizeof(ShellInfo) +
                            pairList.pairs.size() * sizeof(ShellPairIndex) +
                            8 * orbitalBounds.size() + 8 * auxShellBounds.size() +
                            8 * auxShellFirstFunction.size() + 8 * auxShellFunctionCount.size()
                      : 0;
    const std::size_t screenStructureBytes =
        screenEngaged ? (auxShellBounds.size() + 1) * sizeof(std::size_t) +
                            pairList.pairs.size() * auxShellBounds.size() * sizeof(std::size_t)
                      : 0;

    // The memory-ladder rung decision (the
    // workspace-budget seam, decided ONCE from the budget's remaining bytes -
    // ri_engine.cpp's own decision shape, with this builder's rung
    // vocabulary). A builder created without a budget skips it entirely and
    // takes the fast rung, byte for byte as before this landing.
    RiFullFockModeInfo modeInfo;
    // The screen's record is independent of the budget: a builder created
    // with a screen and no budget is screened, and a record that could not say
    // so would leave a reader to infer the treatment from the numbers.
    modeInfo.screenThreshold = screenEngaged ? screenOptions.threshold : 0.0;
    modeInfo.screenStoreBytes = screenStoreBytes;
    modeInfo.screenStructureBytes = screenStructureBytes;
    // The contraction traversal's own record, beside the screen's and
    // independent of the budget for the same reason: a builder created with a
    // batched target and no budget is batched, and a record that could not say
    // so would leave a reader to infer the treatment. The width and the
    // block count are NOT here - they belong to a call, not to a Create, and
    // the per-call sink is their home.
    modeInfo.contractionBlockBytes = batchOptions.blockTargetBytes;
    bool blockedRung = false;
    std::size_t sliceFunctions = 0;
    std::size_t sliceCount = 0;
    std::size_t outputBlockFunctions = 0;
    // The options every build below runs with: identical to the caller's on the
    // no-budget path, and carrying the decision's clamped batch cap when a
    // budget was given (ri_engine.cpp's effectiveOptions).
    RiEngineOptions effectiveOptions = options;

    if (options.workspaceBudget != nullptr)
    {
        qcx::memory::WorkspaceBudget* budget = options.workspaceBudget;
        const std::size_t remaining = budget->Remaining();
        modeInfo.engaged = true;
        modeInfo.budgetBytes = budget->CapacityBytes();
        modeInfo.remainingAtDecision = remaining;

        // The classes both rungs carry, so the decision compares like with
        // like. The SHARED half comes from internal::RiFootprint - footprint.hpp
        // is the one home of the pair-store, screened-task-list, metric/eigen
        // and batch-arena formulas, and this builder consumes the same fields
        // the RI-J engine's decision does. The light-rung variant is the one
        // whose tensor terms are zero (this builder's own tensor terms are
        // stated below, where the two rungs differ); its RI-J first-iteration
        // fields are not read - this builder has no recompute slice, and its
        // own first-call class is the occ transform, charged explicitly. The
        // pair list and the two bound vectors come from the shared sweep above,
        // which this block's own condition also triggers.
        const std::size_t screenedTaskCount =
            internal::CountSchwarzSurvivingRiTasks(orbitalBounds, auxShellBounds, options.accuracy);
        const std::size_t teamSize = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
        const internal::RiFootprintTerms sharedTerms =
            internal::RiFootprint(molecule,
                                  basisSet,
                                  auxBasisSet,
                                  pairList,
                                  *auxPairs,
                                  options.maxBatchBytes,
                                  teamSize,
                                  /*lightRung=*/true,
                                  /*blockedMetricRung=*/false,
                                  screenedTaskCount);
        // The structural class: everything the footprint charges that does NOT
        // scale with the batch cap (the pair stores, the screened task list and
        // the metric/eigen cluster). The batch arena is separated out below,
        // because it is the one term the decision may SHRINK rather than refuse
        // over.
        const std::size_t structuralBytes = sharedTerms.orbitalStoreBytes +
                                            sharedTerms.auxStoreBytes + sharedTerms.taskListBytes +
                                            sharedTerms.metricBytes;
        // The transform's root, live with the metric/eigen cluster while the
        // product runs (the cluster's own count is the footprint's; the root is
        // this rung's fourth live nAux^2 matrix, and it is RETAINED across the
        // blocked loop instead of dying with the product).
        const std::size_t rootBytes = 8 * nAuxFuncs * nAuxFuncs;
        // The occ transform BuildFock allocates: n x nOcc x nAux doubles for
        // the molecule's closed-shell occupied count - the count the shipped
        // RHF caller passes (the driver's own nOccupied = electrons/2), and the
        // only nOcc a Create can know. A caller handing a LARGER block budgets
        // for it itself; the class is stated in BuildFock's contract.
        const std::size_t occupiedCount =
            std::max<std::size_t>(1, static_cast<std::size_t>(molecule.ElectronCount()) / 2);
        const std::size_t occTransformBytes = 8 * n * occupiedCount * nAuxFuncs;
        const std::size_t fockBytes = kRank2WorkingMatrices * 8 * n * n;
        const std::size_t batchIndependentBytes = structuralBytes + rootBytes + occTransformBytes +
                                                  fockBytes + screenStoreBytes +
                                                  screenStructureBytes;
        // The record's own terms (ri_full_fock.hpp RiFullFockModeInfo doubles as
        // the fired estimate's decomposition, the way FockModeInfo's does):
        // filled from the same locals whose sum the decision compares, so a
        // reader can add them up and land on predictedBytes.
        modeInfo.structuralBytes = structuralBytes;
        modeInfo.rootBytes = rootBytes;
        modeInfo.tensorBytes = tensorBytes;
        modeInfo.occTransformBytes = occTransformBytes;
        modeInfo.fockBytes = fockBytes;

        // The batch cap is the FIRST thing to give, not a reason to refuse: the
        // 3c arena is batch x team and batch re-partitioning is value-neutral
        // (the footprint's scratch clamp - ri_engine.cpp's own first move, and
        // the reason its decision never refuses over an arena it could shrink).
        // One attempt per rung: clamp the cap into what the rung's
        // batch-independent class leaves, then read the rung's estimate at the
        // clamped arena. The fast rung is attempted first - its class is the
        // larger one by exactly one tensor copy - and a failed fast attempt
        // falls to the blocked rung rather than to a refusal: the ladder's
        // order, and the reason the refusal below is the last resort.
        //
        // \param blocked True for the blocked rung's class (one tensor copy).
        // \returns The clamped batch cap, or 0 when the rung's class cannot fit
        // at any cap.
        const auto attemptArena = [&](bool blocked, std::size_t budgetBytes) -> std::size_t {
            const std::size_t tensorForRung = blocked ? tensorBytes : 2 * tensorBytes;
            const std::size_t clamped = internal::ClampBatchBytes(
                options.maxBatchBytes,
                teamSize,
                batchIndependentBytes + tensorForRung + options.maxBatchBytes * teamSize,
                budgetBytes,
                /*arenaCount=*/1);

            if (clamped == 0)
            {
                return 0;
            }

            const std::size_t estimate = batchIndependentBytes + tensorForRung + clamped * teamSize;
            return estimate <= budgetBytes ? clamped : 0;
        };

        const std::size_t fastBatch = attemptArena(false, remaining);

        if (fastBatch != 0)
        {
            modeInfo.rung = RiFullFockRung::kFast;
            modeInfo.maxBatchBytes = fastBatch;
            modeInfo.arenaBytes = fastBatch * teamSize;
            modeInfo.predictedBytes = batchIndependentBytes + 2 * tensorBytes + modeInfo.arenaBytes;
        } else
        {
            // The blocked rung's own minimum - one auxiliary shell range's slice
            // plus the accumulation temporary's one-function floor - is reserved
            // BEFORE the arena is clamped. Without the reserve the clamp spends
            // the whole remaining budget on the arena and hands the decision a
            // false refusal: the rung the ladder exists to reach would be
            // unreachable at exactly the budget it is for.
            const std::size_t sliceFloorBytes =
                bytesPerFunction * ChunkFunctionWidth(*auxPairs, 0, ChunkShellEnd(*auxPairs, 0, 1));
            const std::size_t sliceReserve = sliceFloorBytes + bytesPerFunction;
            const std::size_t blockedBudget =
                remaining > sliceReserve ? remaining - sliceReserve : 0;
            const std::size_t blockedBatch = attemptArena(true, blockedBudget);

            if (blockedBatch == 0)
            {
                // Neither rung's class fits: the ladder's refusal, at the
                // smallest arena the stack can reach (a unit batch), so its
                // numbers are the floor for this budget and not an artifact of
                // the caller's batch cap.
                const std::size_t unitArenaBytes = teamSize;
                return std::unexpected(qcx::Error{
                    qcx::ErrorCode::kUnimplemented,
                    LadderRefusal(batchIndependentBytes + unitArenaBytes + 2 * tensorBytes,
                                  batchIndependentBytes + unitArenaBytes + tensorBytes,
                                  remaining,
                                  bytesPerFunction)});
            }

            // The blocked rung: its arena is the clamped one, and the slice and
            // the accumulation's output-block temporary share whatever the
            // fixed class leaves. Every build below runs at this cap, so the
            // realized arena is the charged one (ri_engine.cpp's
            // `effectiveOptions.maxBatchBytes = modeInfo->maxBatchBytes`).
            blockedRung = true;
            modeInfo.rung = RiFullFockRung::kBlocked;
            modeInfo.maxBatchBytes = blockedBatch;
            modeInfo.arenaBytes = blockedBatch * teamSize;
            const std::size_t fixedBytes =
                batchIndependentBytes + tensorBytes + modeInfo.arenaBytes;

            if (remaining <= fixedBytes)
            {
                const std::size_t firstEnd = ChunkShellEnd(*auxPairs, 0, 1);
                const std::size_t sliceBytes =
                    bytesPerFunction * ChunkFunctionWidth(*auxPairs, 0, firstEnd);
                return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                                  LadderRefusal(fixedBytes + tensorBytes,
                                                                fixedBytes + 2 * sliceBytes,
                                                                remaining,
                                                                sliceBytes)});
            }

            const std::size_t allowance = remaining - fixedBytes;
            const std::size_t width = std::max<std::size_t>(1, allowance / (2 * bytesPerFunction));
            const std::size_t firstEnd = ChunkShellEnd(*auxPairs, 0, width);
            sliceFunctions = ChunkFunctionWidth(*auxPairs, 0, firstEnd);
            const std::size_t sliceBytes = bytesPerFunction * sliceFunctions;
            const std::size_t remainder = allowance > sliceBytes ? allowance - sliceBytes : 0;

            if (remainder == 0)
            {
                // One auxiliary shell range's slice plus its temporary does not
                // fit: the blocked rung's quantization is a whole shell (a task
                // block spans a shell), so there is no smaller
                // rung to fall to and the ladder's refusal is the answer.
                return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                                  LadderRefusal(fixedBytes + tensorBytes,
                                                                fixedBytes + 2 * sliceBytes,
                                                                remaining,
                                                                sliceBytes)});
            }

            outputBlockFunctions =
                std::max<std::size_t>(1, std::min(nAuxFuncs, remainder / bytesPerFunction));
            modeInfo.predictedBytes =
                fixedBytes + sliceBytes + bytesPerFunction * outputBlockFunctions;
            modeInfo.sliceFunctions = sliceFunctions;
        }

        // Every build below runs at the decided cap, on both rungs, so the
        // realized arena is the charged one (ri_engine.cpp's
        // `effectiveOptions.maxBatchBytes = modeInfo->maxBatchBytes`).
        effectiveOptions.maxBatchBytes = modeInfo.maxBatchBytes;

        // The reservation: the engaged rung's bytes against the budget, once.
        // A failure here means the budget moved between the decision and the
        // reservation (the seam shares one budget by pointer) - the RI-J
        // engine's own note, and the same reading.
        if (!budget->Reserve(modeInfo.predictedBytes))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "RiFullFockBuilder: the workspace budget moved between the rung "
                           "decision and its reservation - " +
                               std::to_string(modeInfo.predictedBytes >> 20) +
                               " MiB of the engaged rung no longer fits. Retry with an "
                               "uncontended budget"});
        }

        modeInfo.reservedBytes = modeInfo.predictedBytes;
    }

    // The term-counter sink (ri_full_fock.hpp TermCounters): both Create-time
    // builds below accumulate into it, so the builder reports the 3-center
    // terms a reduction would shrink instead of leaving them unpopulated. On
    // the blocked rung the sink is the SAME object across the chunks, and so
    // is the route's cross-chunk dedup state below: the chunk calls partition
    // ONE tensor-pass occurrence, and the orbital pair stamps have to span
    // them for the chunked totals to equal the monolithic build's
    // (ri_engine.hpp RiScreenedPassState) - the rung-identity cell pins it.
    RiTermCounters termCounters;
    RiScreenedPassState passState;

    auto metric = BuildAuxMetric(molecule, auxBasisSet, effectiveOptions, &termCounters);

    if (!metric.has_value())
    {
        return std::unexpected(metric.error());
    }

    // The transform's root: computed once and RETAINED on the blocked rung (the
    // chunk accumulation needs it column-slice by column-slice), computed
    // inside the one-product route on the fast rung - the same rule either way
    // (ri_occ_k.hpp MetricInverseRoot is the single home of it).
    const Eigen::MatrixXd metricMatrix = internal::TensorToEigen(*metric);
    std::optional<Eigen::MatrixXd> root;

    if (blockedRung)
    {
        auto builtRoot = MetricInverseRoot(metricMatrix, effectiveOptions.metricFloorEpsilon);

        if (!builtRoot.has_value())
        {
            return std::unexpected(builtRoot.error());
        }

        root = std::move(*builtRoot);
    }

    Eigen::MatrixXd transformed;

    if (blockedRung)
    {
        // The blocked rung: B += I_chunk s_chunk, one auxiliary shell range at
        // a time, against the retained root. The raw slice is never wider than
        // the decision's allowance, and every chunk multiplies into the whole
        // transform, so the total work is the single product's n^2 nAux^2 - the
        // ladder's batched/blocked rung costs no additional flops and buys the
        // whole raw-tensor copy back.
        transformed = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n * n),
                                            static_cast<Eigen::Index>(nAuxFuncs));
        const Eigen::Index n2 = static_cast<Eigen::Index>(n * n);
        const Eigen::Index nAuxColumns = static_cast<Eigen::Index>(nAuxFuncs);

        std::size_t firstShell = 0;

        while (firstShell < auxPairs->shells.size())
        {
            const std::size_t endShell = ChunkShellEnd(*auxPairs, firstShell, sliceFunctions);
            auto slice = BuildRiTensorChunk(molecule,
                                            basisSet,
                                            auxBasisSet,
                                            firstShell,
                                            endShell,
                                            effectiveOptions,
                                            &termCounters,
                                            &passState);

            if (!slice.has_value())
            {
                return std::unexpected(slice.error());
            }

            ++sliceCount;
            const Eigen::Index chunkColumns = slice->cols();
            const Eigen::Index firstColumn =
                static_cast<Eigen::Index>(auxPairs->shells[firstShell].functionOffset);

            for (Eigen::Index column = 0; column < nAuxColumns;)
            {
                const Eigen::Index width =
                    std::min(static_cast<Eigen::Index>(outputBlockFunctions), nAuxColumns - column);
                // The product's temporary is n^2 x width (Eigen evaluates the
                // product before adding it): that temporary is the term the
                // decision sized `outputBlockFunctions` from, so the peak this
                // loop reaches is the peak the estimate charged.
                transformed.block(0, column, n2, width) +=
                    *slice * (*root).block(firstColumn, column, chunkColumns, width);
                column += width;
            }

            firstShell = endShell;
        }

        modeInfo.sliceCount = sliceCount;
    } else
    {
        auto rawTensor = BuildRiTensorChunk(molecule,
                                            basisSet,
                                            auxBasisSet,
                                            0,
                                            auxPairs->shells.size(),
                                            effectiveOptions,
                                            &termCounters,
                                            &passState);

        if (!rawTensor.has_value())
        {
            return std::unexpected(rawTensor.error());
        }

        auto built = BuildMetricTransformedTensor(
            *rawTensor, metricMatrix, effectiveOptions.metricFloorEpsilon);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        transformed = std::move(*built);
    }

    auto state = std::make_shared<State>();
    state->coreHamiltonian = internal::TensorToEigen(coreHamiltonian);
    state->transformed = std::move(transformed);
    state->termCounters = termCounters;
    state->modeInfo = modeInfo;
    state->screenThreshold = modeInfo.screenThreshold;
    // The seam cap every build below runs at (the budget branch's
    // `effectiveOptions.maxBatchBytes = modeInfo.maxBatchBytes`, or the
    // caller's value when no budget engaged), stashed so BuildFock's batched
    // traversal can clamp its block width against the same number the 3-center
    // route and the blocked rung ran under.
    state->maxBatchBytes = effectiveOptions.maxBatchBytes;
    state->contractionBlockBytes = batchOptions.blockTargetBytes;

    if (screenEngaged)
    {
        // The screen's retained stores. The pair list and the bounds were
        // swept above for the same Create; a builder created without a screen
        // leaves all five empty, which is the dense path's own state.
        state->screenPairs = std::move(pairList);
        state->screenPairBounds = std::move(orbitalBounds);
        state->screenAuxShellBounds = std::move(auxShellBounds);
        state->screenAuxShellFirstFunction = std::move(auxShellFirstFunction);
        state->screenAuxShellFunctionCount = std::move(auxShellFunctionCount);
    }

    if (state->transformed.rows() != state->coreHamiltonian.rows() * state->coreHamiltonian.rows())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the 3-center tensor's orbital dimension does not match the core "
                       "Hamiltonian's"});
    }

    return RiFullFockBuilder(std::move(state));
}

qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> RiFullFockBuilder::BuildHalves(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const Eigen::MatrixXd& occupiedOrbitals,
    FockBuildStats* statsOut,
    RiAuxScreenStats* screenOut,
    RiContractionStats* contractionOut) const {
    if (statsOut != nullptr)
    {
        // The structural zero of the absent direct-exchange half (the file
        // comment's note): this builder calls no quartet kernel, so the
        // sink is written with the default counts rather than left as the
        // caller's struct happened to hold.
        *statsOut = FockBuildStats{};
    }

    const Eigen::MatrixXd rho = internal::TensorToEigen(density);
    const Eigen::Index n = _state->coreHamiltonian.rows();

    if (rho.rows() != n || rho.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the density's shape does not match the basis the builder was created "
                       "with"});
    }

    if (occupiedOrbitals.rows() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the occupied orbital block's row count does not match the basis the "
                       "builder was created with"});
    }

    const std::size_t occupiedCount = static_cast<std::size_t>(occupiedOrbitals.cols());

    if (occupiedOrbitals.cols() == 0)
    {
        // The dense entry points refuse an empty occupied block (and the
        // screened sweep would silently contract nothing); the refusal is
        // hoisted so both paths answer a caller identically.
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the occupied orbital block must be a non-empty matrix"});
    }

    // The screen's per-call structure, built ONCE from this call's density and
    // read by both the record and the contractions below. Off (threshold 0)
    // leaves it empty and the call runs the dense path.
    const bool engaged = _state->screenThreshold > 0.0;
    AuxScreenStructure structure;

    if (engaged)
    {
        structure = BuildAuxScreenStructure(rho,
                                            _state->screenPairs,
                                            _state->screenPairBounds,
                                            _state->screenAuxShellBounds,
                                            _state->screenThreshold);
    }

    if (screenOut != nullptr)
    {
        // The record is written on EVERY call, engaged or not: a sink left
        // untouched would let a caller read a previous call's counts as this
        // one's (the rule FockBuildStats carries).
        RiAuxScreenStats record;
        record.engaged = engaged;
        record.threshold = _state->screenThreshold;
        record.totalCells = _state->screenPairs.pairs.size() * _state->screenAuxShellBounds.size();
        record.survivingCells = engaged ? structure.survivingCells : record.totalCells;
        record.droppedCells = record.totalCells - record.survivingCells;
        record.retainedEntries = structure.indices.size();
        record.structureBytes = structure.bytes();
        record.droppedFraction =
            record.totalCells == 0
                ? 0.0
                : static_cast<double>(record.droppedCells) / static_cast<double>(record.totalCells);
        *screenOut = record;
    }

    Eigen::MatrixXd exchange;
    Eigen::MatrixXd coulomb;
    // The contraction record of THIS call (increment I5). Assembled from the
    // branch that runs below - the two byte fields are the builder's own
    // configuration, and `batched`/`blockWidth`/`blockCount` are set where the
    // traversal is entered, so the record states what executed rather than
    // what was asked for.
    RiContractionStats contraction;
    contraction.blockTargetBytes = _state->contractionBlockBytes;
    contraction.maxBatchBytes = _state->maxBatchBytes;

    if (engaged)
    {
        // The screened path: the surviving pair blocks of THIS call's density
        // drive the occupied transform and the Coulomb weights together, then
        // the exchange contraction runs the module's own dense rank-nOcc
        // update over the transform they produced (see BuildFock's contract
        // for why that product is not narrowed).
        const Eigen::Index n = rho.rows();
        const Eigen::Index nOcc = static_cast<Eigen::Index>(occupiedCount);
        // ZERO-initialized: the sweep writes only the rows a surviving block
        // reaches, and a row no surviving block touches is exactly zero - the
        // one value the contraction may read there (an uninitialized row would
        // enter the exchange product and the result would depend on the
        // allocator's leftovers).
        Eigen::MatrixXd occTransformed =
            Eigen::MatrixXd::Zero(n * nOcc, _state->transformed.cols());
        Eigen::VectorXd weights(_state->transformed.cols());
        SweepScreenedColumns(_state->transformed,
                             rho,
                             occupiedOrbitals,
                             _state->screenPairs,
                             structure,
                             _state->screenAuxShellFirstFunction,
                             _state->screenAuxShellFunctionCount,
                             occTransformed,
                             weights);
        auto builtExchange = BuildRiExchangeMatrix(occTransformed, occupiedCount);

        if (!builtExchange.has_value())
        {
            return std::unexpected(builtExchange.error());
        }

        exchange = std::move(*builtExchange);
        // The Coulomb half's second product, the one the sweep's weights feed:
        // J = B u with u the per-aux density contraction - the J-from-B
        // identity (ri_occ_k.hpp) against the SCREENED u, where the dense path
        // reads it from the density directly.
        coulomb = FromContractionVector(_state->transformed * weights, n);
    } else if (_state->contractionBlockBytes != 0)
    {
        // The batched path (increment I5): ONE traversal of the auxiliary
        // index builds both contractions, and the occ-transformed array the
        // chain allocates is never materialized. The entry point carries the
        // chain's own guards in the chain's own order, and every input it can
        // refuse has been refused above (this call's density shape, the
        // occupied block against the basis, an empty block) or is guaranteed
        // by Create (a positive seam cap, a positive target) - so replacing
        // three calls with one adds no error a caller of this builder can see.
        auto batched = BuildBatchedRiContractions(_state->transformed,
                                                  rho,
                                                  occupiedOrbitals,
                                                  _state->maxBatchBytes,
                                                  _state->contractionBlockBytes);

        if (!batched.has_value())
        {
            return std::unexpected(batched.error());
        }

        // The record of what ran, taken from the build that ran: the width and
        // the block count are the traversal's own report (ri_occ_k.hpp
        // RiBatchedContractions), not a second application of the width rule.
        contraction.batched = true;
        contraction.blockWidth = batched->blockWidth;
        contraction.blockCount = batched->blockCount;
        exchange = std::move(batched->exchange);
        coulomb = std::move(batched->coulomb);
    } else
    {
        // The dense path: the module's own two entry points, called exactly as
        // the I2/I3 composition called them - the absence of the screen and of
        // the batched traversal is the absence of those treatments, not a
        // second implementation of the dense half.
        auto occTransformed = TransformToOccupiedOrbitals(_state->transformed, occupiedOrbitals);

        if (!occTransformed.has_value())
        {
            return std::unexpected(occTransformed.error());
        }

        auto builtExchange = BuildRiExchangeMatrix(*occTransformed, occupiedCount);

        if (!builtExchange.has_value())
        {
            return std::unexpected(builtExchange.error());
        }

        exchange = std::move(*builtExchange);
        coulomb = CoulombFromTransformed(_state->transformed, rho);
    }

    if (contractionOut != nullptr)
    {
        // Written after the branch and before the halves leave this body, so a
        // call that reached a contraction records which one. The guard failures
        // above return before this line and leave the sink untouched, exactly
        // as the screen sink does.
        *contractionOut = contraction;
    }

    // The two halves leave this body SEPARATE: no line above mixes them, and
    // the sum that fuses them lives with the caller - BuildFock's composition
    // line below, or the unrestricted leg's per-spin assembly. That is what
    // makes the pair entry point cost nothing extra: this body already
    // evaluated both halves on every path before either caller asked for one.
    return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{std::move(coulomb), std::move(exchange)};
}

qcx::Result<CpuTensor2> RiFullFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const Eigen::MatrixXd& occupiedOrbitals,
    FockBuildStats* statsOut,
    RiAuxScreenStats* screenOut,
    RiContractionStats* contractionOut) const {
    auto halves = BuildHalves(density, occupiedOrbitals, statsOut, screenOut, contractionOut);

    if (!halves.has_value())
    {
        return std::unexpected(halves.error());
    }

    // F = H + 2 J_RI - K_RI. The factor of two and the sign are the shipped
    // composition's (ri_engine.hpp RiJkFockBuilder, whose BuildFock returns
    // H + 2 J_RI - K with the same spatial-density convention), and H is added
    // HERE and nowhere else on this path: the pair the halves entry point hands
    // out carries neither the core Hamiltonian nor the factor of two
    // (RiFullFockHalves states that accounting, and why the per-spin assembly
    // needs it bare).
    const Eigen::MatrixXd fock = _state->coreHamiltonian + 2.0 * halves->first - halves->second;

    return internal::EigenToTensor(fock);
}

qcx::Result<RiFullFockHalves> RiFullFockBuilder::BuildFockHalves(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const Eigen::MatrixXd& occupiedOrbitals,
    FockBuildStats* statsOut,
    RiAuxScreenStats* screenOut,
    RiContractionStats* contractionOut) const {
    auto halves = BuildHalves(density, occupiedOrbitals, statsOut, screenOut, contractionOut);

    if (!halves.has_value())
    {
        return std::unexpected(halves.error());
    }

    auto coulomb = internal::EigenToTensor(halves->first);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    auto exchange = internal::EigenToTensor(halves->second);

    if (!exchange.has_value())
    {
        return std::unexpected(exchange.error());
    }

    return RiFullFockHalves{std::move(*coulomb), std::move(*exchange)};
}

const Eigen::MatrixXd& RiFullFockBuilder::MetricTransformedTensor() const noexcept {
    return _state->transformed;
}

const RiFullFockModeInfo& RiFullFockBuilder::ModeInfo() const noexcept {
    return _state->modeInfo;
}

const RiTermCounters& RiFullFockBuilder::TermCounters() const noexcept {
    return _state->termCounters;
}

} // namespace qcx::integrals
