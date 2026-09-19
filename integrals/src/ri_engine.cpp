// The 3-center RI-J engine (ri_engine.hpp): the (uv|P) tensor through
// the shared MD kernels with single-shell aux kets (md_vrr_3c.hpp), the
// (P|Q) metric through the shared 4c kernels over the phantom-pair quartets
// (P, s_0 | Q, s_0) - the two-center Coulomb of the aux functions
// themselves - and the RI-J Fock builder - per iteration only the two RI
// contractions (v = I^T D, J = I w with M w = v) plus the direct exchange
// of fock_build.hpp run in its exchange-only mode. [arXiv:2210.03192]

#include "qcx/integrals/ri_engine.hpp"

#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "internal/light_footprint.hpp"
#include "internal/md_attribution.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_vrr_3c.hpp"
#include "internal/ri_orbit_action.hpp"
#include "internal/tensor_eigen_bridge.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/linalg/dense_ops.hpp"

#ifdef QcxHasCuda
#include "qcx/memory/cuda_allocator_traits.hpp"

#include <cublas_v2.h>
#endif

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals {

/// The light rung's retained per-iteration recompute inputs (the fast path
/// keeps the materialized tensor instead): the two shell-pair lists (the
/// block-extraction geometry), the combined orbital + aux pair store, the
/// screened task list and the orbital-pair prefix size of the store. The
/// metric's floored eigen-inverse is shared with the fast path in State.
struct LightRungPayload {
    ShellPairList pairList;
    ShellPairList auxPairList;
    std::vector<internal::MdPairData> combinedStore;
    std::vector<internal::RiTask> taskList;
    std::size_t nOrbitalPairs = 0;
};

namespace {

// The RI-J metric floor: metric eigenvalues below eps * lambdaMax are
// zeroed in the inverse (the standard RI regularization, 2026-08-21). The
// (P|Q) Coulomb metric of jfit-style aux bases is SPD and well-conditioned
// (no eigenvalue of def2-universal-jfit on H2O sits below 1e-10 * lambdaMax
// - the probe measured n < floor = 0), so the floor is a no-op there; it
// guards pathological aux bases the same way every RI-J code does. The
// constant is the canonical default of RiEngineOptions::metricFloorEpsilon;
// Create applies the per-engine value.
inline constexpr double kMetricFloorEpsilon = 1e-10;

qcx::Result<void> CheckSupported(const ShellPairList& pairList) {
    for (const ShellInfo& shell : pairList.shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    return {};
}

// The cuBLAS dispatch: the RI contractions are plain
// col-major double products, so the same m x n x k Dgemm the CPU seam runs
// can go to the device (operands staged through device temporaries - the
// engine made the same discovery on its first integration: raw
// host pointers make cuBLAS return success with garbage or
// CUBLAS_STATUS_INTERNAL_ERROR). The handle is per-call on purpose: the RI
// builder runs one product pair per SCF iteration, and cublasCreate/Destroy
// cost ~10 us against a ~100 us Dgemm at the H2O/jfit sizes - a long-lived
// handle would need ownership plumbing through State for no measurable
// gain. ONLY the handle-init failure falls back to the CPU seam (the device
// may be absent or busy, and the caller asked for acceleration, not for a
// different result); a staging-allocation or staging-copy failure is a hard
// kDeviceError - the device exists, and the staging is the acceleration the
// caller asked for.
#ifdef QcxHasCuda
qcx::Result<Eigen::MatrixXd> CublasProduct(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.cols() != b.rows())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "matrix dimension mismatch"});
    }

    cublasHandle_t handle = nullptr;

    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        return qcx::linalg::DenseMultiply(a, b);
    }

    const int m = static_cast<int>(a.rows());
    const int k = static_cast<int>(a.cols());
    const int n = static_cast<int>(b.cols());
    Eigen::MatrixXd result(m, n);
    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS GEMMs read and write device memory only: the operands are
    // staged through per-call DeviceBuffer temporaries (memory module - the
    // exclusive allocation site; RAII frees and residency-tracked copies,
    // so the first integration's manual cudaMalloc/cudaFree branch is
    // gone). At the RI sizes (n2 x nAux, ~6 KB) the transfers dominate the
    // ~10 us GEMM, but the device lane exists for the fp32/fp64
    // arithmetic-rate asymmetry, not for transfer-bound microbenchmarks.
    auto aDev = qcx::memory::DeviceBuffer<double, qcx::backend::CudaTag>::Create(
        static_cast<std::size_t>(m) * static_cast<std::size_t>(k));

    if (!aDev.has_value())
    {
        cublasDestroy(handle);

        return std::unexpected(aDev.error());
    }

    auto bDev = qcx::memory::DeviceBuffer<double, qcx::backend::CudaTag>::Create(
        static_cast<std::size_t>(k) * static_cast<std::size_t>(n));

    if (!bDev.has_value())
    {
        cublasDestroy(handle);

        return std::unexpected(bDev.error());
    }

    auto cDev = qcx::memory::DeviceBuffer<double, qcx::backend::CudaTag>::Create(
        static_cast<std::size_t>(m) * static_cast<std::size_t>(n));

    if (!cDev.has_value())
    {
        cublasDestroy(handle);

        return std::unexpected(cDev.error());
    }

    aDev->HostView().assign(a.data(),
                            a.data() + static_cast<std::size_t>(m) * static_cast<std::size_t>(k));
    bDev->HostView().assign(b.data(),
                            b.data() + static_cast<std::size_t>(k) * static_cast<std::size_t>(n));

    auto syncA = aDev->SyncToDevice();

    if (!syncA.has_value())
    {
        cublasDestroy(handle);

        return std::unexpected(syncA.error());
    }

    auto syncB = bDev->SyncToDevice();

    if (!syncB.has_value())
    {
        cublasDestroy(handle);

        return std::unexpected(syncB.error());
    }

    const cublasStatus_t status =
        cublasDgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha,
                    static_cast<const double*>(aDev->DeviceHandle().Raw()),
                    m,
                    static_cast<const double*>(bDev->DeviceHandle().Raw()),
                    k,
                    &beta,
                    static_cast<double*>(cDev->DeviceHandle().Raw()),
                    m);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        cublasDestroy(handle);

        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kDeviceError,
                       std::string("cublasDgemm failed: ") + cublasGetStatusName(status)});
    }

    // The Dgemm wrote the device side through the raw handle: mark the
    // buffer dirty so the read-back actually copies.
    cDev->MarkDeviceDirty();

    auto syncC = cDev->SyncToHost();

    if (!syncC.has_value())
    {
        cublasDestroy(handle);

        return std::unexpected(syncC.error());
    }

    cublasDestroy(handle);

    // The Dgemm output is column-major, the same order as the Eigen result.
    Eigen::Map<const Eigen::MatrixXd> map(cDev->HostView().data(), m, n);
    result = map;

    return result;
}
#endif

// 3-center Schwarz screening: |(uv|P)| <= Q_uv * Q_P for
// every function quadruple (Cauchy-Schwarz on the Coulomb bilinear form),
// so a bra-pair x aux-shell block whose bound product sits below the
// preset's Schwarz budget can be skipped before the MD kernels run. The
// density factors (max_d over the block) are per-iteration
// quantities, vacuous here: BuildRiTensor is basis-only, called once at
// Create, and the dense (uv|P) tensor it materializes makes the screening
// budget a preset-level choice. With the kNormal budget (1e-10) no real basis
// screens everything out - the same-center diagonal pairs sit at Q products
// far above the budget; a pathological all-screened task list errors in
// RunRiBatches, by design.
qcx::Result<std::vector<internal::RiTask>> BuildScreenedRiTaskList(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    // pairList (bra/ket pairs) then auxPairList: the Schwarz bound product
    // is Q_uv * Q_P - orbital pair first, auxiliary pair second.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const ShellPairList& pairList,
    const ShellPairList& auxPairList,
    const std::vector<internal::MdShellInput>& auxShells,
    std::size_t nOrbitalPairs,
    const RiEngineOptions& options) {
    auto schwarzOrbital = ComputeSchwarzBounds(molecule, basisSet);

    if (!schwarzOrbital.has_value())
    {
        return std::unexpected(schwarzOrbital.error());
    }

    auto schwarzAux = ComputeSchwarzBounds(molecule, auxBasisSet);

    if (!schwarzAux.has_value())
    {
        return std::unexpected(schwarzAux.error());
    }

    // The aux kets are single shells, not pairs: the per-shell bound is the
    // diagonal pair Q_P of the aux pair list. BuildShellPairs builds the
    // full upper triangle, so the closed-form PairIndexOf is exact, and the
    // bound vector is in ShellPairList::pairs order.
    std::vector<double> schwarzAuxShell(auxShells.size());

    for (std::size_t auxShell = 0; auxShell < auxShells.size(); ++auxShell)
    {
        schwarzAuxShell[auxShell] = (*schwarzAux)[PairIndexOf(auxShell, auxShell, auxPairList)];
    }

    const double schwarzBudget = SchwarzThreshold(options.accuracy);
    // The counted treatment applied to the task grid: the realized reserve is
    // the surviving-cell count, not the unconditional nOrbitalPairs x
    // nAuxShells grid. The footprint's taskListBytes charge
    // (RiFootprint's screenedTaskCount) is
    // the same count x 16 B, so the model and the allocation agree exactly
    // (never-under by equality - the dense all-survive grid would only
    // over-charge). The two-pointer count is bit-identical to the fill's
    // per-cell test below (same doubles, same cutoff), so the fill never
    // exceeds the reserved capacity.
    std::vector<internal::RiTask> tasks;
    tasks.reserve(
        internal::CountSchwarzSurvivingRiTasks(*schwarzOrbital, schwarzAuxShell, options.accuracy));

    for (std::size_t braPair = 0; braPair < nOrbitalPairs; ++braPair)
    {
        for (std::size_t auxShell = 0; auxShell < auxShells.size(); ++auxShell)
        {
            if ((*schwarzOrbital)[braPair] * schwarzAuxShell[auxShell] < schwarzBudget)
            {
                continue;
            }

            tasks.push_back(internal::RiTask{braPair, nOrbitalPairs + auxShell});
        }
    }

    return tasks;
}

// The joint task-grid ORBIT EXPANSION's engaged state (ri_orbit_action.hpp):
// the two-basis action plus the two Schwarz
// bound vectors its representative test and its members' survival test read,
// and the cutoff those tests share with the fill above. Built once, at
// Create, from reductions the caller supplies.
struct RiOrbitReduction {
    internal::RiOrbitAction action;
    /// Q_uv per canonical orbital pair (ComputeSchwarzBounds' own order).
    std::vector<double> orbitalBounds;
    /// Q_P per aux shell (the aux pair list's diagonal).
    std::vector<double> auxShellBounds;
    /// The fill's cutoff: SchwarzThreshold(options.accuracy).
    double cutoff = 0.0;

    /// The fill's own survival test, verbatim (same doubles, same
    /// comparison) - the boundary must not disagree with
    /// BuildScreenedRiTaskList's or the expansion's write set stops matching
    /// the computed set.
    /// \param braPair The cell's canonical orbital pair index.
    /// \param auxShell The cell's aux shell index.
    /// \returns True when the fill would have emitted the cell.
    bool Survives(std::size_t braPair, std::size_t auxShell) const noexcept {
        return orbitalBounds[braPair] * auxShellBounds[auxShell] >= cutoff;
    }
};

// The orbit expansion is DEFINED by the two reductions it expands over
// (lean_fock_build.cpp's refusal of the same shape, one index longer):
// asking for it without them is a caller error, refused rather than silently
// ignored. The flag alone, with both reductions, still stays inert on a
// trivial (order 1) group - with one element every cell is its own orbit and
// the mechanism is the identity.
qcx::Result<void> ValidateRiOrbitExpansion(const RiEngineOptions& options) {
    // The option is a PERMISSION, not a request (it defaults ON - see its
    // documentation for the count analysis that set the default): with no
    // reductions supplied it is inert, not an error, which is what every
    // caller that does not engage symmetry does. Supplying EXACTLY ONE of
    // the two is the caller error the flag still refuses: a joint orbit
    // needs one group acting on both bases, and half of it is not a group
    // action.
    const bool hasOrbital = options.symmetryReduction != nullptr;
    const bool hasAux = options.auxSymmetryReduction != nullptr;

    if (hasOrbital != hasAux)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the orbit expansion takes BOTH the orbital and the auxiliary symmetry reductions or "
            "neither (the joint task-grid orbits are one group acting on two bases)"});
    }

    return {};
}

// Whether the joint task-grid orbit expansion actually runs: the permission
// granted AND both reductions supplied AND a non-trivial group on both bases
// (a group whose every element fixes every canonical shell pair makes the
// mechanism the identity - every cell is its own orbit - so nothing is
// engaged and nothing is charged). THE single predicate: the tensor pass,
// the Create-time charge and the light-rung refusal all read this one, so
// they cannot disagree about what "engaged" means.
//
// "Non-trivial" is the reduction's own isTrivial verdict - the group's
// action on the CANONICAL SHELL PAIRS - and never `groupOrder > 1` (measured
// 2026-09-13). The two differ exactly where it
// costs: a non-C1 sign-only group (every atom on the symmetry element - a
// planar molecule's out-of-plane mirror, the hocl/Cs case) permutes
// no pair, so the expansion removes nothing while still building its
// tables and charging them. The order-1 spelling this predicate was
// written for is the special case of the same statement.
bool RiOrbitExpansionEngaged(const RiEngineOptions& options) noexcept {
    return options.symmetryOrbitExpansion && options.symmetryReduction != nullptr &&
           options.auxSymmetryReduction != nullptr && !options.symmetryReduction->isTrivial &&
           !options.auxSymmetryReduction->isTrivial;
}

// The expansion's per-axis maps are fixed-size (kRiOrbitAxisCap), so a shell
// wider than the cap is refused - a release build would otherwise overflow a
// stack buffer inside RiExpandOrbitMemberBlock. No basis this engine ships
// comes near the cap; a general contraction with an unusual row count is what
// it is for.
//
// A property of the two pair lists ALONE, which is why it is checked twice:
// at the point of use below (it is a precondition of the mechanism), and
// again in BuildRiTensor BEFORE the screened task grid is built - the grid's
// Schwarz sweep runs over the very shell that is about to be refused, and on
// the general contraction this check exists for that sweep is 24^4 primitive
// quartets paid to reach a refusal that needs none of them. The refusal
// returns the same Error either way, so the early call cannot change an
// outcome, only when it is reached.
/// \param pairList The orbital pair list.
/// \param auxPairList The aux pair list.
/// \returns An Error when either list carries a shell wider than the cap.
qcx::Result<void> ValidateRiOrbitActionWidth(const ShellPairList& pairList,
                                             const ShellPairList& auxPairList) {
    if (!internal::RiOrbitActionSupports(pairList, auxPairList))
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kUnimplemented,
            "the symmetry orbit expansion supports shells of up to 64 functions (its per-axis "
            "image maps are fixed-size); this basis carries a wider shell"});
    }

    return {};
}

// Builds the engaged orbit state: the joint action over the two bases plus
// the Schwarz bounds its tests read. Only called when the mechanism is
// engaged AND the group is non-trivial.
qcx::Result<RiOrbitReduction> BuildRiOrbitReduction(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet,
                                                    const qcx::basisset::BasisSet& auxBasisSet,
                                                    const ShellPairList& pairList,
                                                    const ShellPairList& auxPairList,
                                                    const RiEngineOptions& options) {
    auto width = ValidateRiOrbitActionWidth(pairList, auxPairList);

    if (!width.has_value())
    {
        return std::unexpected(width.error());
    }

    auto action = internal::RiOrbitAction::Create(
        *options.symmetryReduction, pairList, *options.auxSymmetryReduction, auxPairList);

    if (!action.has_value())
    {
        return std::unexpected(action.error());
    }

    auto schwarzOrbital = ComputeSchwarzBounds(molecule, basisSet);

    if (!schwarzOrbital.has_value())
    {
        return std::unexpected(schwarzOrbital.error());
    }

    auto schwarzAux = ComputeSchwarzBounds(molecule, auxBasisSet);

    if (!schwarzAux.has_value())
    {
        return std::unexpected(schwarzAux.error());
    }

    RiOrbitReduction engaged;
    engaged.action = std::move(*action);
    engaged.orbitalBounds = std::move(*schwarzOrbital);
    engaged.auxShellBounds.resize(auxPairList.shells.size());

    for (std::size_t auxShell = 0; auxShell < auxPairList.shells.size(); ++auxShell)
    {
        engaged.auxShellBounds[auxShell] =
            (*schwarzAux)[PairIndexOf(auxShell, auxShell, auxPairList)];
    }

    engaged.cutoff = SchwarzThreshold(options.accuracy);
    return engaged;
}

// Filters a screened task grid down to one representative per joint orbit
// (the cells the kernel evaluates; every other surviving cell is filled by
// the expansion at the tensor scatter). The representative is the smallest
// surviving cell of its orbit in the grid's own order, and the candidate set
// is the SURVIVING set - an image the fill would have dropped must not win
// the minimum, or a whole orbit could lose its representative and its block
// would never be computed. In: the screened list, in grid order; out: the
// representatives, in the same order.
std::vector<internal::RiTask> ReduceRiTasksToOrbitReps(const RiOrbitReduction& engaged,
                                                       const std::vector<internal::RiTask>& tasks,
                                                       std::size_t nOrbitalPairs) {
    std::vector<internal::RiTask> representatives;
    representatives.reserve(tasks.size());

    for (const internal::RiTask& task : tasks)
    {
        if (internal::RiCellIsOrbitRep(engaged.action,
                                       engaged.orbitalBounds,
                                       engaged.auxShellBounds,
                                       engaged.cutoff,
                                       task.braPair,
                                       task.auxShell - nOrbitalPairs))
        {
            representatives.push_back(task);
        }
    }

    return representatives;
}

// The light rung's Create-time inputs (the fast path builds the same
// stores transiently inside BuildRiTensor; the light rung retains them):
// the combined orbital + aux pair store, the screened task list and the
// two shell-pair lists for the per-iteration block extraction.
qcx::Result<LightRungPayload> BuildLightRungPayload(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet,
                                                    const qcx::basisset::BasisSet& auxBasisSet,
                                                    const RiEngineOptions& options) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto auxPairList = BuildShellPairs(molecule, auxBasisSet);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    auto orbitalSupported = CheckSupported(*pairList);

    if (!orbitalSupported.has_value())
    {
        return std::unexpected(orbitalSupported.error());
    }

    auto auxSupported = CheckSupported(*auxPairList);

    if (!auxSupported.has_value())
    {
        return std::unexpected(auxSupported.error());
    }

    auto orbitalStore = internal::BuildPairData(molecule, basisSet, *pairList);

    if (!orbitalStore.has_value())
    {
        return std::unexpected(orbitalStore.error());
    }

    auto auxShells = internal::FlattenShells(molecule, auxBasisSet, *auxPairList);

    if (!auxShells.has_value())
    {
        return std::unexpected(auxShells.error());
    }

    std::vector<internal::MdPairData> combinedStore;
    combinedStore.reserve(orbitalStore->size() + auxShells->size());
    combinedStore.insert(combinedStore.end(),
                         std::make_move_iterator(orbitalStore->begin()),
                         std::make_move_iterator(orbitalStore->end()));

    for (const internal::MdShellInput& shell : *auxShells)
    {
        const std::array<double, 3> center = {shell.cx, shell.cy, shell.cz};
        combinedStore.push_back(internal::BuildAuxPairData(shell.contractions, center));
    }

    const std::size_t nOrbitalPairs = combinedStore.size() - auxShells->size();
    auto taskList = BuildScreenedRiTaskList(molecule,
                                            basisSet,
                                            auxBasisSet,
                                            *pairList,
                                            *auxPairList,
                                            *auxShells,
                                            nOrbitalPairs,
                                            options);

    if (!taskList.has_value())
    {
        return std::unexpected(taskList.error());
    }

    return LightRungPayload{std::move(*pairList),
                            std::move(*auxPairList),
                            std::move(combinedStore),
                            std::move(*taskList),
                            nOrbitalPairs};
}

// Runs the assembled 3c batches one at a time against a single shared
// maxBatchBytes-class values buffer (the batch outputs are batch-relative,
// so every batch writes the same buffer from offset 0) and hands each
// batch's tasks to \p sink before the next batch overwrites the buffer.
template <typename Sink>
qcx::Result<void> ConsumeAssembledBatches(const internal::RiBatchAssembly& assembly,
                                          const Sink& sink) {
    // The light rung's per-iteration values buffer (the
    // light_rung_values term - the batch-capped per-call buffer): the
    // scope pins the family inside the runner, which the light rung's two
    // passes call exclusively.
    qcx::memory::AllocationTagScope valuesScope(qcx::memory::AllocationTag::kLightRungValues);
    internal::TaggedVector<double> values(assembly.maxBatchOutput, 0.0);
    std::vector<internal::MdClassBatch> single;
    single.reserve(1);

    for (const internal::MdClassBatch& batch : assembly.batches)
    {
        single.clear();
        single.push_back(batch);
        single[0].outF64 = values.data();
        auto run = internal::RunBatches(single);

        if (!run.has_value())
        {
            return std::unexpected(run.error());
        }

        sink(batch.tasks, values);
    }

    return {};
}

// The light rung's v = I^T d pass: the same sum the materialized path
// computes through the BLAS seam over the flat n2 x nAux layout - each
// tensor entry against d(u, v) for both triangles (the diagonal entry
// once, the flat matrix has one (u, u) entry), accumulated per task block
// in the assembled batch order.
qcx::Result<Eigen::VectorXd> LightRungTransposeProduct(const LightRungPayload& payload,
                                                       const internal::RiBatchAssembly& assembly,
                                                       const Eigen::VectorXd& dVec,
                                                       std::size_t n) {
    Eigen::VectorXd v =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(payload.auxPairList.functionCount));
    auto run = ConsumeAssembledBatches(
        assembly,
        [&](const std::vector<internal::MdQuartetTask>& tasks,
            const internal::TaggedVector<double>& values) {
            for (const internal::MdQuartetTask& task : tasks)
            {
                const ShellPairIndex& braIndex = payload.pairList.pairs[task.braPair];
                const ShellInfo& shellI = payload.pairList.shells[braIndex.i];
                const ShellInfo& shellJ = payload.pairList.shells[braIndex.j];
                const std::size_t nI = ShellFunctionCount(shellI);
                const std::size_t nJ = ShellFunctionCount(shellJ);
                const std::size_t oI = shellI.functionOffset;
                const std::size_t oJ = shellJ.functionOffset;
                const ShellInfo& auxShell =
                    payload.auxPairList.shells[task.ketPair - payload.nOrbitalPairs];
                const std::size_t nP = ShellFunctionCount(auxShell);
                const std::size_t oP = auxShell.functionOffset;

                for (std::size_t fa = 0; fa < nI; ++fa)
                {
                    for (std::size_t fb = 0; fb < nJ; ++fb)
                    {
                        const std::size_t u = oI + fa;
                        const std::size_t vIdx = oJ + fb;

                        for (std::size_t fP = 0; fP < nP; ++fP)
                        {
                            const double value =
                                values[task.outputOffset + (fb * nI + fa) * nP + fP];
                            v(static_cast<Eigen::Index>(oP + fP)) +=
                                value * dVec(static_cast<Eigen::Index>(u * n + vIdx));

                            // The mirror term covers the block's transposed
                            // (v, u) entry. Only the cross-shell pairs need
                            // it: for a diagonal-shell pair (i == j) the
                            // full block already enumerates both orders, so
                            // the mirror would add each off-diagonal twice.
                            if (braIndex.i != braIndex.j && u != vIdx)
                            {
                                v(static_cast<Eigen::Index>(oP + fP)) +=
                                    value * dVec(static_cast<Eigen::Index>(vIdx * n + u));
                            }
                        }
                    }
                }
            }
        });

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    return v;
}

// The light rung's j = I w pass: each tensor entry against w(P),
// accumulated into the flat n2 fock layout for both triangles (the
// diagonal entry once).
qcx::Result<Eigen::VectorXd> LightRungProduct(const LightRungPayload& payload,
                                              const internal::RiBatchAssembly& assembly,
                                              const Eigen::VectorXd& wVec,
                                              std::size_t n) {
    Eigen::VectorXd j = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n * n));
    auto run = ConsumeAssembledBatches(
        assembly,
        [&](const std::vector<internal::MdQuartetTask>& tasks,
            const internal::TaggedVector<double>& values) {
            for (const internal::MdQuartetTask& task : tasks)
            {
                const ShellPairIndex& braIndex = payload.pairList.pairs[task.braPair];
                const ShellInfo& shellI = payload.pairList.shells[braIndex.i];
                const ShellInfo& shellJ = payload.pairList.shells[braIndex.j];
                const std::size_t nI = ShellFunctionCount(shellI);
                const std::size_t nJ = ShellFunctionCount(shellJ);
                const std::size_t oI = shellI.functionOffset;
                const std::size_t oJ = shellJ.functionOffset;
                const ShellInfo& auxShell =
                    payload.auxPairList.shells[task.ketPair - payload.nOrbitalPairs];
                const std::size_t nP = ShellFunctionCount(auxShell);
                const std::size_t oP = auxShell.functionOffset;

                for (std::size_t fa = 0; fa < nI; ++fa)
                {
                    for (std::size_t fb = 0; fb < nJ; ++fb)
                    {
                        const std::size_t u = oI + fa;
                        const std::size_t vIdx = oJ + fb;

                        for (std::size_t fP = 0; fP < nP; ++fP)
                        {
                            const double value =
                                values[task.outputOffset + (fb * nI + fa) * nP + fP];
                            const double weight = wVec(static_cast<Eigen::Index>(oP + fP));
                            j(static_cast<Eigen::Index>(u * n + vIdx)) += value * weight;

                            // The mirror term covers the transposed (v, u)
                            // entry; a diagonal-shell pair's full block
                            // already enumerates both orders (the same
                            // double-count guard as the v pass).
                            if (braIndex.i != braIndex.j && u != vIdx)
                            {
                                j(static_cast<Eigen::Index>(vIdx * n + u)) += value * weight;
                            }
                        }
                    }
                }
            }
        });

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    return j;
}

// The shared-phantom pair list and compact store of the (P|Q) metric
// builds (BuildAuxMetric and the blocked-metric rung's
// BuildBlockedAuxMetric below): one SHARED phantom shell s_0 = {l = 0,
// exponent 0, coefficient 1} at index 0 serves every aux shell - the
// pairs (0, p + 1) and the quartets (0, p + 1, 0, q + 1) reference it as
// the bra/ket partner of the real shells, so the pair list is the phantom
// shell plus the aux shells over the full upper triangle (including the
// real-real pairs the quartets never reference) and the store holds
// exactly one MdPairData per aux shell plus the unused (0, 0) slot. The
// compact layout is store-consistent because the phantom pairs (0, x)
// sit at LINEAR pair index x of the row-major upper triangle (the row
// i = 0 entries come first, one per column). See BuildAuxMetric's comment
// for the phantom's construction and the metric's (P|Q) reading.
struct AuxMetricStore {
    ShellPairList metricPairList; ///< The phantom-shell pair list (full upper triangle).
    std::vector<internal::MdPairData> pairStore; ///< The compact store: entry p + 1 is the
                                                 ///< (phantom, P_p) pair, the entries past
                                                 ///< nAuxShells stay default-constructed.
};

qcx::Result<AuxMetricStore> BuildAuxMetricStore(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& auxBasisSet,
                                                const ShellPairList& auxPairList) {
    auto auxShells = internal::FlattenShells(molecule, auxBasisSet, auxPairList);

    if (!auxShells.has_value())
    {
        return std::unexpected(auxShells.error());
    }

    AuxMetricStore store;
    const std::size_t nAuxShells = auxPairList.shells.size();
    store.metricPairList.shells.reserve(nAuxShells + 1);
    ShellInfo phantom{};
    phantom.angularMomentum = 0;
    phantom.isSpherical = false;
    phantom.contractionCount = 1;
    phantom.functionOffset = auxPairList.functionCount;
    phantom.atomIndex = 0;
    phantom.elementShellIndex = 0;
    store.metricPairList.shells.push_back(phantom);
    store.metricPairList.shells.insert(
        store.metricPairList.shells.end(), auxPairList.shells.begin(), auxPairList.shells.end());
    store.metricPairList.functionCount = auxPairList.functionCount + 1;

    const std::size_t nShells = store.metricPairList.shells.size();

    for (std::size_t i = 0; i < nShells; ++i)
    {
        for (std::size_t j = i; j < nShells; ++j)
        {
            store.metricPairList.pairs.push_back(ShellPairIndex{i, j});
        }
    }

    store.pairStore.resize(nAuxShells + 1);

    // The phantom bra: l = 0 with one row and one primitive, exponent 0,
    // coefficient 1 (the pair (s_0, P_p) is the function phi_P itself).
    internal::MdShellContractions phantomContractions;
    phantomContractions.angularMomentum = 0;
    phantomContractions.isSpherical = false;
    phantomContractions.rows = 1;
    phantomContractions.exponents = {0.0};
    phantomContractions.normalized = {{1.0}};

    for (std::size_t p = 0; p < nAuxShells; ++p)
    {
        const internal::MdShellInput& aux = (*auxShells)[p];
        const std::array<double, 3> center = {aux.cx, aux.cy, aux.cz};
        internal::MdPairData pair;
        pair.la = 0;
        pair.lb = aux.contractions.angularMomentum;
        pair.isSphericalA = false;
        pair.isSphericalB = aux.contractions.isSpherical;
        pair.ax = center[0];
        pair.ay = center[1];
        pair.az = center[2];
        pair.bx = center[0];
        pair.by = center[1];
        pair.bz = center[2];
        internal::BuildContractedPairTransform(
            pair, phantomContractions, aux.contractions, center, center);
        store.pairStore[p + 1] = std::move(pair);
    }

    return store;
}

// The METRIC occurrence's x/g3 accumulation (ri_engine.hpp RiTermCounters -
// the term counters' moments): the distinct phantom-pair
// store entries engaged by the assembled batch tasks, and the
// |primPairs(bra entry)| x |primPairs(ket entry)| kernel weight summed
// over the tasks - the exact set RunBatches evaluates. A store entry
// recurs across the triangle's rows and, in the blocked build, across
// strips, so the caller-held stamp state (one stamp per store entry)
// dedups it per metric build, never per strip: the callers clear the
// stamps once per build and pass the build's single epoch (1) on every
// call, so an entry stamped in an earlier strip stays seen. p3 stays
// untouched: the metric engages no orbital pair.
void AccumulateMetricBatches(RiTermCounters& into,
                             const std::vector<internal::MdPairData>& pairStore,
                             const std::vector<internal::MdClassBatch>& batches,
                             std::vector<std::size_t>& pairStamps,
                             std::size_t epoch) {
    for (const internal::MdClassBatch& batch : batches)
    {
        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            if (pairStamps[task.braPair] != epoch)
            {
                pairStamps[task.braPair] = epoch;
                ++into.x;
            }

            if (pairStamps[task.ketPair] != epoch)
            {
                pairStamps[task.ketPair] = epoch;
                ++into.x;
            }

            into.g3 +=
                pairStore[task.braPair].primPairs.size() * pairStore[task.ketPair].primPairs.size();
        }
    }
}

} // namespace

namespace internal {

// The TENSOR-PASS occurrence's x/p3/g3 accumulation (ri_engine.hpp
// RiTermCounters - the term counters' moments; declared in
// internal/md_vrr_3c.hpp so the chunked build runs the SAME rule): ONE
// kernel evaluation of the screened 3c task list - the distinct aux entries
// with a surviving task (x) and the distinct orbital bra-pair entries with a
// surviving task (p3), each stamped once per pass (an entry recurs across
// tasks), plus the |primPairs(bra)| x |primPairs(aux entry)| kernel
// weight over the surviving tasks (g3). The fast/legacy path evaluates
// the list once, at Create (BuildRiTensor); the light rung re-evaluates
// the same retained list twice per BuildFock call (the recompute's v and
// j passes) - the caller adds this partial once per pass.
//
// The stamps are per evaluation, so the aux entries dedup inside the call
// that reaches them and the orbital pairs inside the call that saves them:
// a caller that partitions ONE evaluation across calls (the chunked build)
// hands over its own state (passState) so the orbital pair stamps span the
// chunks - the chunk ranges partition the aux shells, never the orbital
// pairs, so without it each chunk re-counts every pair it survives with
// (ri_engine.hpp RiScreenedPassState).
void AccumulateScreenedRiPass(RiTermCounters& into,
                              const std::vector<MdPairData>& combinedStore,
                              std::size_t nOrbitalPairs,
                              std::span<const RiTask> tasks,
                              RiScreenedPassState* passState) {
    // One epoch per evaluation: the stamp vectors are fresh per call (one
    // call = one kernel evaluation of the task list), so the fixed epoch 1
    // dedups every entry across the whole list. When the caller holds the
    // state (the chunked build) its stamps stand in for the bra ones and
    // are NOT fresh - the epoch still marks them, and the caller's
    // zero-initialized entries are what a chunk sees as unseen.
    constexpr std::size_t kSeenEpoch = 1;
    std::vector<std::size_t> ownBraStamps;
    std::vector<std::size_t>* braStamps = &ownBraStamps;

    if (passState == nullptr)
    {
        ownBraStamps.assign(nOrbitalPairs, 0);
    } else
    {
        // Grown, never cleared: the entries already stamped belong to the
        // earlier chunks of the same evaluation and must stay seen.
        if (passState->braStamps.size() < nOrbitalPairs)
        {
            passState->braStamps.resize(nOrbitalPairs, 0);
        }

        braStamps = &passState->braStamps;
    }

    std::vector<std::size_t> auxStamps(combinedStore.size() - nOrbitalPairs, 0);

    for (const RiTask& task : tasks)
    {
        if ((*braStamps)[task.braPair] != kSeenEpoch)
        {
            (*braStamps)[task.braPair] = kSeenEpoch;
            ++into.p3;
        }

        const std::size_t auxEntry = task.auxShell - nOrbitalPairs;

        if (auxStamps[auxEntry] != kSeenEpoch)
        {
            auxStamps[auxEntry] = kSeenEpoch;
            ++into.x;
        }

        into.g3 += combinedStore[task.braPair].primPairs.size() *
                   combinedStore[task.auxShell].primPairs.size();
    }
}

} // namespace internal

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildAuxMetric(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& auxBasisSet,
    const RiEngineOptions& options,
    RiTermCounters* termCountersOut) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    auto auxPairList = BuildShellPairs(molecule, auxBasisSet);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    auto supported = CheckSupported(*auxPairList);

    if (!supported.has_value())
    {
        return std::unexpected(supported.error());
    }

    const std::size_t nAux = auxPairList->functionCount;
    const std::size_t nAuxShells = auxPairList->shells.size();

    if (nAuxShells == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the auxiliary basis is empty"});
    }

    // The metric elements are the two-center Coulomb integrals of the aux
    // functions M(fP, fQ) = (P|Q), evaluated through the shared 4c kernel
    // as the phantom-pair quartets (s_0, P | s_0, Q): the phantom shell
    // s_0 = {l = 0, exponent 0, coefficient 1} at the aux center turns the
    // pair (s_0, P) into an ordinary contracted pair of the function phi_P
    // itself (p = zeta, prefactor exp(-a*0/p |A-B|^2) = 1, E tables filled
    // with p = a), so the kernel returns <phi_P | 1/r | phi_Q> exactly.
    // The phantom is not in the basis - the shared BuildAuxMetricStore
    // carries it in the pair list and builds the pair store directly. One
    // SHARED phantom at shell index 0
    // serves every aux shell: the pairs (0, p + 1) and quartets
    // (0, p + 1, 0, q + 1) reference it as the bra/ket partner of the real
    // shells (the pair-internal A/B orientation is one of the 8-fold
    // value-identical partners), so the store holds exactly one entry per
    // aux shell plus the unused (0, 0) slot - the pair-index holes of the
    // former per-shell phantom layout (a ~3 nAuxShells^2 / 2 MdPairData
    // tail, ~1.2 GB at the 2000-shell scale) are gone.
    //
    // The former (PP|QQ) density metric was the RI-J failure (2026-08-21):
    // the half-density ket kappa_P (p = 2 zeta)
    // is proportional to the density rho_P = phi_P^2 only for s shells, so
    // the density metric's near-null directions carried O(1) projections
    // of v = I^T d, amplified by the floored eigen-inverse into the 1e7
    // Coulomb error of RiCoulombMatchesDirectCoulomb. The standard (P|Q)
    // function scheme measures 2.3e-4 max error on the same probe.
    auto metricStore = BuildAuxMetricStore(molecule, auxBasisSet, *auxPairList);

    if (!metricStore.has_value())
    {
        return std::unexpected(metricStore.error());
    }

    ShellPairList& metricPairList = metricStore->metricPairList;
    std::vector<internal::MdPairData>& pairStore = metricStore->pairStore;

    std::vector<ShellQuartet> quartets;
    quartets.reserve(nAuxShells * (nAuxShells + 1) / 2);

    for (std::size_t p = 0; p < nAuxShells; ++p)
    {
        for (std::size_t q = p; q < nAuxShells; ++q)
        {
            quartets.push_back(ShellQuartet{0, p + 1, 0, q + 1});
        }
    }

    std::vector<ShellQuartet> computed;
    auto batches = internal::AssembleClassBatches(
        pairStore, metricPairList, quartets, options.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return std::unexpected(batches.error());
    }

    std::size_t total = 0;

    for (internal::MdClassBatch& batch : *batches)
    {
        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            total += pairStore[task.braPair].nFuncs * pairStore[task.ketPair].nFuncs;
        }
    }

    std::vector<double> values(total);
    std::size_t base = 0;

    for (internal::MdClassBatch& batch : *batches)
    {
        batch.outF64 = values.data() + base;

        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            base += pairStore[task.braPair].nFuncs * pairStore[task.ketPair].nFuncs;
        }
    }

    // The metric occurrence's counts (the x/g3 terms of
    // ri_engine.hpp RiTermCounters), over the assembled tasks - the exact
    // set RunBatches evaluates below. x lands on every aux shell (the full
    // unscreened triangle's phantom-pair entries); the count is carried out
    // only when a sink was given - null keeps the zero-cost path.
    if (termCountersOut != nullptr)
    {
        // One epoch per metric build: the stamps start cleared and this
        // single call is the whole build (AccumulateMetricBatches dedups
        // entries across the tasks against the epoch).
        std::vector<std::size_t> pairStamps(pairStore.size(), 0);
        AccumulateMetricBatches(*termCountersOut, pairStore, *batches, pairStamps, 1);
    }

    auto run = internal::RunBatches(*batches);

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    auto metric = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({nAux, nAux});

    if (!metric.has_value())
    {
        return std::unexpected(metric.error());
    }

    // The metric element M(fP, fQ) = (P|Q) sits in the (0, fP, 0, fQ)
    // entry of the canonical (i, j | k, l) block: the real aux shells are
    // the j/l partners (quartet.j, quartet.l) and the shared phantom
    // occupies i/k with one function each, so the block shape is
    // (1, nP, 1, nQ). The 8-fold canonicalization swaps the bra/ket
    // orientation (pair index and class), so the extraction reads through
    // quartet.j / quartet.l, and the symmetric write fills both triangles.
    std::size_t offset = 0;

    for (std::size_t t = 0; t < computed.size(); ++t)
    {
        const ShellQuartet& quartet = computed[t];
        const ShellInfo& shellP = metricPairList.shells[quartet.j];
        const ShellInfo& shellQ = metricPairList.shells[quartet.l];
        const std::size_t nP = ShellFunctionCount(shellP);
        const std::size_t nQ = ShellFunctionCount(shellQ);
        const std::size_t oP = shellP.functionOffset;
        const std::size_t oQ = shellQ.functionOffset;

        for (std::size_t fP = 0; fP < nP; ++fP)
        {
            for (std::size_t fQ = 0; fQ < nQ; ++fQ)
            {
                const std::size_t element = EriBlockIndex(0, fP, 0, fQ, 1, nP, 1, nQ);
                const double value = values[offset + element];
                (*metric)(oP + fP, oQ + fQ) = value;
                (*metric)(oQ + fQ, oP + fP) = value;
            }
        }

        offset += nP * nQ;
    }

    metric->MarkHostDirty();
    return std::move(*metric);
}

qcx::Result<Eigen::MatrixXd> BuildBlockedAuxMetric(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& auxBasisSet,
                                                   const RiEngineOptions& options,
                                                   RiTermCounters* termCountersOut) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    auto auxPairList = BuildShellPairs(molecule, auxBasisSet);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    auto supported = CheckSupported(*auxPairList);

    if (!supported.has_value())
    {
        return std::unexpected(supported.error());
    }

    const std::size_t nAux = auxPairList->functionCount;
    const std::size_t nAuxShells = auxPairList->shells.size();

    if (nAuxShells == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the auxiliary basis is empty"});
    }

    auto metricStore = BuildAuxMetricStore(molecule, auxBasisSet, *auxPairList);

    if (!metricStore.has_value())
    {
        return std::unexpected(metricStore.error());
    }

    ShellPairList& metricPairList = metricStore->metricPairList;
    std::vector<internal::MdPairData>& pairStore = metricStore->pairStore;

    // The blocked-metric rung's engagement contract:
    // the metric is assembled strip-wise into the eigen-solver's input
    // DIRECTLY - no full values buffer, no memory Tensor and no
    // TensorToEigen copy exist on this path (Create's estimate models the
    // 2 x 8 nAuxFuncs^2 eigen-class peak plus the per-strip arena,
    // internal::BlockedMetricStripBytes - the solver holds one working
    // matrix, m_eivec, which absorbs the input's lower triangle; the
    // eigen-solver reconciliation). The strips are greedy over the
    // metric rows: each row's byte mass (internal::MetricRowStripBytes -
    // the row's (q + 1) phantom quartets at 176 B each plus its 8 B per
    // value) accumulates while the running mass stays at or under the
    // batch cap; a row the cap alone cannot hold becomes a forced
    // single-row strip. The batch machinery's own per-batch cap keeps
    // every class batch inside options.maxBatchBytes regardless (the
    // strip cap is the values buffer's and the assembly lists' bound).
    // The per-strip arena of the blocked-metric rung (the
    // blocked_metric_strip term - BlockedMetricStripBytes): the strip's
    // values buffer is the term's subject; the quartets and the assembly
    // lists cross md_batch.cpp and stay plain. The scope covers the whole
    // strip loop (the buffer's capacity is retained at the largest strip).
    qcx::memory::AllocationTagScope stripScope(qcx::memory::AllocationTag::kBlockedMetricStrip);
    std::vector<ShellQuartet> quartets;
    std::vector<ShellQuartet> computed;
    internal::TaggedVector<double> values;
    Eigen::MatrixXd metric(static_cast<Eigen::Index>(nAux), static_cast<Eigen::Index>(nAux));
    std::size_t stripFirst = 0;

    // The metric occurrence's stamp state (the x/g3 terms of
    // ri_engine.hpp RiTermCounters): sized only when a sink was given (the
    // zero-cost path otherwise), and held OUTSIDE the strip loop so a
    // phantom-pair store entry that recurs across strips counts once per
    // metric build, never once per strip: the stamps are cleared once
    // below and every strip's tasks compare against the same epoch (1),
    // so an entry stamped by an earlier strip stays seen.
    std::vector<std::size_t> pairStamps;

    if (termCountersOut != nullptr)
    {
        pairStamps.assign(pairStore.size(), 0);
    }

    while (stripFirst < nAuxShells)
    {
        std::size_t stripMass = 0;
        std::size_t stripLast = stripFirst;

        for (std::size_t q = stripFirst; q < nAuxShells; ++q)
        {
            const std::size_t rowMass =
                internal::MetricRowStripBytes(q, metricPairList.shells[q + 1]);

            if (stripMass != 0 && stripMass + rowMass > options.maxBatchBytes)
            {
                break;
            }

            stripMass += rowMass;
            stripLast = q;
        }

        // The strip's quartets: the phantom-pair triangle rows
        // (s_0, P_p | s_0, P_q) over p <= q for each row of the strip
        // (BuildAuxMetric pushes the same quartets in the same row
        // order).
        quartets.clear();

        for (std::size_t q = stripFirst; q <= stripLast; ++q)
        {
            for (std::size_t p = 0; p <= q; ++p)
            {
                quartets.push_back(ShellQuartet{0, p + 1, 0, q + 1});
            }
        }

        auto batches = internal::AssembleClassBatches(
            pairStore, metricPairList, quartets, options.maxBatchBytes, computed);

        if (!batches.has_value())
        {
            return std::unexpected(batches.error());
        }

        std::size_t total = 0;

        for (internal::MdClassBatch& batch : *batches)
        {
            for (const internal::MdQuartetTask& task : batch.tasks)
            {
                total += pairStore[task.braPair].nFuncs * pairStore[task.ketPair].nFuncs;
            }
        }

        // The per-strip values buffer: one strip's blocks at a time (the
        // vector's capacity is retained at the largest strip's size).
        values.resize(total);
        std::size_t base = 0;

        for (internal::MdClassBatch& batch : *batches)
        {
            batch.outF64 = values.data() + base;

            for (const internal::MdQuartetTask& task : batch.tasks)
            {
                base += pairStore[task.braPair].nFuncs * pairStore[task.ketPair].nFuncs;
            }
        }

        if (termCountersOut != nullptr)
        {
            AccumulateMetricBatches(*termCountersOut, pairStore, *batches, pairStamps, 1);
        }

        auto run = internal::RunBatches(*batches);

        if (!run.has_value())
        {
            return std::unexpected(run.error());
        }

        // The scatter is BuildAuxMetric's own (the metric element
        // M(fP, fQ) = (P|Q) sits in the (0, fP, 0, fQ) entry of the
        // canonical block - the real aux shells are the j/l partners, the
        // symmetric write fills both triangles): same quartets, same
        // values, same destinations - the strip partition is
        // value-neutral, so the blocked matrix is bit-identical to the
        // unblocked build's.
        std::size_t offset = 0;

        for (std::size_t t = 0; t < computed.size(); ++t)
        {
            const ShellQuartet& quartet = computed[t];
            const ShellInfo& shellP = metricPairList.shells[quartet.j];
            const ShellInfo& shellQ = metricPairList.shells[quartet.l];
            const std::size_t nP = ShellFunctionCount(shellP);
            const std::size_t nQ = ShellFunctionCount(shellQ);
            const std::size_t oP = shellP.functionOffset;
            const std::size_t oQ = shellQ.functionOffset;

            for (std::size_t fP = 0; fP < nP; ++fP)
            {
                for (std::size_t fQ = 0; fQ < nQ; ++fQ)
                {
                    const std::size_t element = EriBlockIndex(0, fP, 0, fQ, 1, nP, 1, nQ);
                    const double value = values[offset + element];
                    metric(static_cast<Eigen::Index>(oP + fP), static_cast<Eigen::Index>(oQ + fQ)) =
                        value;
                    metric(static_cast<Eigen::Index>(oQ + fQ), static_cast<Eigen::Index>(oP + fP)) =
                        value;
                }
            }

            offset += nP * nQ;
        }

        stripFirst = stripLast + 1;
    }

    // Every element is written exactly once per triangle (the rows cover
    // all shells p <= q, each block fully enumerated) - no zero
    // initialization needed, mirroring the unblocked build.
    return metric;
}

qcx::Result<qcx::memory::Tensor<double, 3, qcx::backend::CpuTag>> BuildRiTensor(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const RiEngineOptions& options,
    RiTermCounters* termCountersOut) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto auxPairList = BuildShellPairs(molecule, auxBasisSet);

    if (!auxPairList.has_value())
    {
        return std::unexpected(auxPairList.error());
    }

    auto orbitalSupported = CheckSupported(*pairList);

    if (!orbitalSupported.has_value())
    {
        return std::unexpected(orbitalSupported.error());
    }

    auto auxSupported = CheckSupported(*auxPairList);

    if (!auxSupported.has_value())
    {
        return std::unexpected(auxSupported.error());
    }

    auto orbitalStore = internal::BuildPairData(molecule, basisSet, *pairList);

    if (!orbitalStore.has_value())
    {
        return std::unexpected(orbitalStore.error());
    }

    auto auxShells = internal::FlattenShells(molecule, auxBasisSet, *auxPairList);

    if (!auxShells.has_value())
    {
        return std::unexpected(auxShells.error());
    }

    std::vector<internal::MdPairData> combinedStore;
    combinedStore.reserve(orbitalStore->size() + auxShells->size());
    combinedStore.insert(combinedStore.end(),
                         std::make_move_iterator(orbitalStore->begin()),
                         std::make_move_iterator(orbitalStore->end()));

    for (const internal::MdShellInput& shell : *auxShells)
    {
        const std::array<double, 3> center = {shell.cx, shell.cy, shell.cz};
        combinedStore.push_back(internal::BuildAuxPairData(shell.contractions, center));
    }

    const std::size_t nOrbitalPairs = combinedStore.size() - auxShells->size();

    // The over-wide-shell precondition, refused BEFORE the task grid below
    // (ValidateRiOrbitActionWidth carries the cost argument): the grid's
    // Schwarz sweep is the expensive consumer of the very shell this refusal
    // is about, so reaching the refusal through it pays for an answer already
    // known. Placed after every fallible call above so their errors keep the
    // precedence they had; BuildRiOrbitReduction re-checks at the point of
    // use, under its own engagement condition.
    if (RiOrbitExpansionEngaged(options))
    {
        auto width = ValidateRiOrbitActionWidth(*pairList, *auxPairList);

        if (!width.has_value())
        {
            return std::unexpected(width.error());
        }
    }

    // See BuildScreenedRiTaskList (above) for the screening rationale.
    auto taskList = BuildScreenedRiTaskList(molecule,
                                            basisSet,
                                            auxBasisSet,
                                            *pairList,
                                            *auxPairList,
                                            *auxShells,
                                            nOrbitalPairs,
                                            options);

    if (!taskList.has_value())
    {
        return std::unexpected(taskList.error());
    }

    auto orbitCheck = ValidateRiOrbitExpansion(options);

    if (!orbitCheck.has_value())
    {
        return std::unexpected(orbitCheck.error());
    }

    // The orbit expansion, when engaged: the screened grid is walked to
    // one representative per joint orbit and only those blocks are
    // evaluated. An order-1 (trivial) group leaves the state unbuilt - every
    // cell is its own orbit and the mechanism is the identity, so the plain
    // list is what this call uses. The evaluated list is the representatives;
    // the tensor scatter below fills the rest of each orbit from its
    // representative's block.
    std::optional<RiOrbitReduction> orbitReduction;
    std::vector<internal::RiTask> representativeList;

    if (RiOrbitExpansionEngaged(options))
    {
        auto engaged = BuildRiOrbitReduction(
            molecule, basisSet, auxBasisSet, *pairList, *auxPairList, options);

        if (!engaged.has_value())
        {
            return std::unexpected(engaged.error());
        }

        representativeList = ReduceRiTasksToOrbitReps(*engaged, *taskList, nOrbitalPairs);

        // The note above on the pathological all-screened list applies here
        // unchanged: an empty list errors in RunRiBatches, by design. The
        // reduction cannot empty a non-empty grid - the minimum of a
        // non-empty surviving set is itself surviving.
        orbitReduction = std::move(*engaged);
    }

    // The (uv|P) values buffer and the computed task list of the
    // materialized tensor pass (the rij_fast_tensor term - charged as
    // the tensor bytes): the containers capture the family at their
    // construction, and the scope covers the pass through the tensor copy
    // (the only live traffic under it).
    qcx::memory::AllocationTagScope tensorScope(qcx::memory::AllocationTag::kRijFastTensor);
    internal::TaggedVector<internal::RiTask> computed;
    internal::TaggedVector<double> values;
    const std::vector<internal::RiTask>& evaluated =
        orbitReduction.has_value() ? representativeList : *taskList;
    auto run = internal::RunRiBatches(
        combinedStore, nOrbitalPairs, evaluated, options.maxBatchBytes, computed, values);

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    // The tensor-pass occurrence's counts (ri_engine.hpp
    // RiTermCounters): this BuildRiTensor call evaluated the screened task
    // list once - the x/p3/g3 partial over the surviving tasks, counted
    // only when a sink was given (null keeps the zero-cost path). Under the
    // orbit expansion the "surviving tasks" are the evaluated
    // representatives: the counters report the blocks the engine actually
    // computed (ri_engine.hpp RiEngineOptions::symmetryOrbitExpansion).
    if (termCountersOut != nullptr)
    {
        internal::AccumulateScreenedRiPass(
            *termCountersOut, combinedStore, nOrbitalPairs, computed);
    }

    const std::size_t n = pairList->functionCount;
    const std::size_t nAux = auxPairList->functionCount;
    auto tensor = qcx::memory::Tensor<double, 3, qcx::backend::CpuTag>::Create({n, n, nAux});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    // One task's packed block into the tensor: the kernel returns the
    // (u, v | P) block with u in shell i, v in shell j; the integral is
    // symmetric in (u, v), so the write fills both triangles (the former
    // upper-only fill left the cross-shell lower triangle zero - a 0.6%
    // error on J = I M^-1 I^T d, measured 2026-08-21). Shared by the
    // computed representatives and their expanded members.
    const auto scatterBlock = [&tensor, &pairList, &auxPairList](std::size_t braPair,
                                                                 std::size_t auxShellIndex,
                                                                 const double* block) {
        const ShellPairIndex& braIndex = pairList->pairs[braPair];
        const ShellInfo& shellI = pairList->shells[braIndex.i];
        const ShellInfo& shellJ = pairList->shells[braIndex.j];
        const std::size_t nI = ShellFunctionCount(shellI);
        const std::size_t nJ = ShellFunctionCount(shellJ);
        const std::size_t oI = shellI.functionOffset;
        const std::size_t oJ = shellJ.functionOffset;
        const ShellInfo& auxShell = auxPairList->shells[auxShellIndex];
        const std::size_t nP = ShellFunctionCount(auxShell);
        const std::size_t oP = auxShell.functionOffset;

        for (std::size_t fa = 0; fa < nI; ++fa)
        {
            for (std::size_t fb = 0; fb < nJ; ++fb)
            {
                for (std::size_t fP = 0; fP < nP; ++fP)
                {
                    const double value = block[(fb * nI + fa) * nP + fP];
                    (*tensor)(oI + fa, oJ + fb, oP + fP) = value;
                    (*tensor)(oJ + fb, oI + fa, oP + fP) = value;
                }
            }
        }
    };

    // The expansion's scratch: one member block, sized by the largest block
    // the grid can carry (the largest orbital pair's function product times
    // the largest aux shell's - the members' blocks have their
    // representatives' element count by the shell closure).
    std::vector<double> memberBlock;
    std::size_t largestPairBlock = 0;
    std::size_t largestAuxShell = 0;

    if (orbitReduction.has_value())
    {
        for (const ShellPairIndex& pair : pairList->pairs)
        {
            largestPairBlock = std::max(largestPairBlock,
                                        ShellFunctionCount(pairList->shells[pair.i]) *
                                            ShellFunctionCount(pairList->shells[pair.j]));
        }

        for (const ShellInfo& shell : auxPairList->shells)
        {
            largestAuxShell = std::max(largestAuxShell, ShellFunctionCount(shell));
        }

        memberBlock.assign(largestPairBlock * largestAuxShell, 0.0);
    }

    // Whether a LATER non-identity element carries the same representative
    // onto the SAME member cell - and writes it. LAST WRITE WINS, and the
    // earlier writes are dead stores: the ascending-g walk below writes a
    // member cell once per element that lands on it, and only the LAST one's
    // bytes survive. A cell the group's own stabilizer carries two elements
    // onto is therefore expanded and scattered twice for nothing.
    //
    // Skipping a g that a later element supersedes removes those dead stores
    // and leaves every byte the walk writes exactly as it was, because the
    // survivor is still the largest g that lands on the cell and survives the
    // screen - the two conditions the walk itself applies. Cost is O(order^2)
    // integer comparisons per representative, no allocation.
    const auto supersededByLaterElement = [&orbitReduction](const internal::RiOrbitAction& action,
                                                            std::size_t repPair,
                                                            std::size_t repAux,
                                                            std::size_t g,
                                                            std::size_t memberPair,
                                                            std::size_t memberAux) {
        for (std::size_t later = g + 1; later < action.order; ++later)
        {
            if (action.orbital.pairImage[repPair * action.order + later] != memberPair ||
                action.aux.shellImage[repAux * action.order + later] != memberAux)
            {
                continue;
            }

            // The later element lands on the same cell, so it writes it
            // unless the screen drops it. (It cannot be the representative
            // itself here: the caller skips that case before asking.)
            if (orbitReduction->Survives(memberPair, memberAux))
            {
                return true;
            }
        }

        return false;
    };

    std::size_t offset = 0;

    for (const internal::RiTask& task : computed)
    {
        const std::size_t auxShellIndex = task.auxShell - nOrbitalPairs;
        const double* representativeBlock = values.data() + offset;
        scatterBlock(task.braPair, auxShellIndex, representativeBlock);
        offset += combinedStore[task.braPair].nFuncs * combinedStore[task.auxShell].nFuncs;

        if (!orbitReduction.has_value())
        {
            continue;
        }

        // The orbit's remaining members, expanded from this
        // representative's block. The member cell is the representative's
        // under element g; cells the fill dropped are skipped (they stay
        // zero on the plain path too), as is the representative itself
        // (g's image of the cell can be the cell - a stabilizer - and its
        // block is already written above), and so are the elements a later
        // one supersedes (their writes would be overwritten - the dead
        // stores that made the copies outnumber the evaluations the
        // expansion removed).
        const internal::RiOrbitAction& action = orbitReduction->action;
        const std::size_t order = action.order;

        for (std::size_t g = 1; g < order; ++g)
        {
            const std::size_t memberPair = action.orbital.pairImage[task.braPair * order + g];
            const std::size_t memberAux = action.aux.shellImage[auxShellIndex * order + g];

            if (memberPair == task.braPair && memberAux == auxShellIndex)
            {
                continue;
            }

            if (!orbitReduction->Survives(memberPair, memberAux))
            {
                continue;
            }

            if (supersededByLaterElement(
                    action, task.braPair, auxShellIndex, g, memberPair, memberAux))
            {
                continue;
            }

            internal::RiExpandOrbitMemberBlock(action,
                                               *pairList,
                                               *auxPairList,
                                               g,
                                               task.braPair,
                                               auxShellIndex,
                                               memberPair,
                                               memberAux,
                                               representativeBlock,
                                               memberBlock.data());
            scatterBlock(memberPair, memberAux, memberBlock.data());
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

struct RiJkFockBuilder::State {
    // I as an n^2 x nAux matrix (rows u*n + v, columns P) - the two RI
    // contractions are plain products over this layout - and the metric's
    // floored eigen-inverse (eigenvectors V, inverse eigenvalues with the
    // sub-kMetricFloorEpsilon*lambdaMax components zeroed): per iteration
    // w = V diag(invLambda) V^T v, no Cholesky solve on the raw factor
    // (BUG-2 fix, 2026-08-21 - the unregularized solve was the 1e12
    // RiCoulombMatchesDirectCoulomb failure). The light rung keeps the
    // recompute payload instead of the tensor (_riMatrix stays empty).
    State(Eigen::MatrixXd riMatrixIn,
          Eigen::MatrixXd metricEigenvectorsIn,
          Eigen::VectorXd inverseMetricEigenvaluesIn,
          DirectJkFockBuilder exchangeIn,
          RiEngineOptions optionsIn,
          std::optional<LightRungPayload> lightRungIn) :
        _riMatrix(std::move(riMatrixIn)), _metricEigenvectors(std::move(metricEigenvectorsIn)),
        _inverseMetricEigenvalues(std::move(inverseMetricEigenvaluesIn)),
        _exchange(std::move(exchangeIn)), _options(optionsIn), _lightRung(std::move(lightRungIn)) {}

    Eigen::MatrixXd _riMatrix;
    Eigen::MatrixXd _metricEigenvectors;
    Eigen::VectorXd _inverseMetricEigenvalues;
    DirectJkFockBuilder _exchange;
    RiEngineOptions _options;
    // The adaptive-memory Create-time mode record (fock_build.hpp
    // FockModeInfo); nullopt on the legacy path (no decision - ModeInfo()).
    std::optional<FockModeInfo> _modeInfo;
    // The light rung's retained recompute inputs; nullopt on the fast and
    // legacy paths.
    std::optional<LightRungPayload> _lightRung;
    // Whether Create engaged the joint task-grid orbit expansion
    // (RiEngineOptions::symmetryOrbitExpansion, ri_engine.hpp
    // OrbitExpansionEngaged): the run-record disclosure the option owes.
    bool _orbitExpansionEngaged = false;
    // The run's term counters (ri_engine.hpp RiTermCounters):
    // the Create-time occurrences (the metric build on every rung and the
    // fast path's single tensor pass) accumulate before the State
    // constructor, the per-call occurrences (qx/gx and the light rung's
    // two recompute passes) inside BuildFock. Mutable because BuildFock is
    // const (the seam shares the builder); the SCF seam calls it serially.
    mutable RiTermCounters _termCounters;
    // The light rung's per-pass partial (x/p3/g3): one count
    // over the retained screened task list, made at Create; every
    // successful BuildFock call adds it twice (the recompute's v and j
    // passes each evaluate the list once). Zero on the fast and legacy
    // paths - their single tensor pass was counted at Create.
    RiTermCounters _lightRungPassCounters;
};

RiJkFockBuilder::RiJkFockBuilder(std::shared_ptr<const State> state) : _state(std::move(state)) {}

qcx::Result<RiJkFockBuilder> RiJkFockBuilder::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::basisset::BasisSet& auxBasisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
    const RiEngineOptions& options) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    auto orbitCheck = ValidateRiOrbitExpansion(options);

    if (!orbitCheck.has_value())
    {
        return std::unexpected(orbitCheck.error());
    }

    // The adaptive-memory mode decision: the Create-time footprint estimate
    // (the RI-J's own terms plus the nested direct-exchange half's) against
    // the budget's remaining
    // bytes, decided once. Only the options that carry a budget take this
    // path - the legacy path keeps its exact behavior below. The engine's
    // own BuildRiTensor/BuildAuxMetric rebuild their inputs internally
    // (same inputs, same numbers - the rebuild is Create-time only).
    std::optional<FockModeInfo> modeInfo;
    RiEngineOptions effectiveOptions = options;
    // The light rung engages when the fast path cannot fit (exclusion (ii)
    // or the estimate/reserve flow) or when the option forces it.
    bool lightRung = false;

    if (options.workspaceBudget != nullptr || options.forceLightRung)
    {
        qcx::memory::WorkspaceBudget* budget = options.workspaceBudget;
        const std::size_t remaining = budget != nullptr ? budget->Remaining() : 0;
        ShellPairList pairList;
        ShellPairList auxPairList;
        std::size_t nAuxShells = 0;
        std::size_t tensorBytes = 0;
        bool tensorExcluded = false;
        std::size_t patternCount = 0;
        // The counted peak row width of the nested exchange half's pre-gate (the widest
        // emitted bra row - fock_build.cpp's name for it), carried out of the decision
        // block to priceExchange: the LightPath's peak-chunk ket bound when no sweep runs
        // (a chunk's ket set is the union of its rows', each bounded by the widest row).
        std::size_t excludedPeakRowWidth = 0;
        std::size_t threadCount = 0;
        // The screened RI task-grid count, carried to the attempt lambda's
        // RiFootprint calls: the taskListBytes term
        // charges the surviving (bra pair x aux shell) cells instead of the
        // unconditional grid, and the light/fast builds' BuildScreenedRiTaskList
        // reserves that same count (same pair lists, same Schwarz bounds -
        // same inputs, same count), so charge == realized by equality.
        std::size_t screenedTaskCount = 0;
        // The engaged orbit action's retained tables, carried to the same two
        // RiFootprint calls: the SAME formula
        // and the SAME engagement condition BuildRiOrbitReduction uses, so
        // the charge and the tables cannot drift apart (zero whenever the
        // mechanism does not run - the option off, no reduction, or a
        // trivial group - which is why the default path's envelope is
        // unchanged).
        std::size_t orbitActionBytes = 0;

        if (budget != nullptr)
        {
            auto builtPairList = BuildShellPairs(molecule, basisSet);

            if (!builtPairList.has_value())
            {
                return std::unexpected(builtPairList.error());
            }

            auto builtAuxPairList = BuildShellPairs(molecule, auxBasisSet);

            if (!builtAuxPairList.has_value())
            {
                return std::unexpected(builtAuxPairList.error());
            }

            pairList = std::move(*builtPairList);
            auxPairList = std::move(*builtAuxPairList);
            nAuxShells = auxPairList.shells.size();

            if (nAuxShells == 0)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument, "the auxiliary basis is empty"});
            }

            // The pattern count for exclusion (i) and the exchange half's
            // estimate (the exchange re-sweeps inside its own Create - same
            // inputs, same count; no seam carries a prebuilt pattern). The
            // count is the Schwarz-exact survivor count under the
            // BuildNeighborList cutoff: the all-survive bound over-refused
            // large-sparse systems (the 81,003-pair chain: 24.44 GiB
            // all-survive against a ~38 MB counted pattern), and
            // the sorted two-pointer count replaces the O(nPairs^2) sweep
            // this block used to run for the same number - bit-identical to
            // it (same doubles, same product cutoff), so the nested's
            // re-sweep lands on the same rows.
            auto schwarz = ComputeSchwarzBounds(molecule, basisSet);

            if (!schwarz.has_value())
            {
                return std::unexpected(schwarz.error());
            }

            patternCount = internal::CountSchwarzSurvivingPairs(*schwarz, options.accuracy);

            // A-priori exclusion (i) - the nested exchange half's RUNG, not a refusal
            // (the engine-exclusion-i record, 2026-09-17). The counted pattern is the
            // nested's object, and the fast rung's whole-pattern CSR is only one of the
            // two objects that half can hold: the counted pre-gate of fock_build.cpp's
            // light decision takes the LightPath - a budget-sized chunk, no global CSR -
            // exactly when the counted rows cannot fit the band the outer leaves it.
            // priceExchange below makes that decision and prices the rung it selects;
            // refusing on it instead charged the fast rung's 183.74 GiB against an
            // 11.73 GB grant and refused a stack the same builder constructs one byte
            // under that band (footprint_test.cpp:3122, the direct-builder mirror). The
            // ladder's refusal is unchanged and now sits where it belongs: the light
            // attempt's own estimate (LightRungRefusal below), which fires only when the
            // LightPath cannot fit either. The counted peak row width is taken here for
            // every budgeted Create: the pricing below reads it whenever it selects the
            // light rung, and WHICH band that rung is priced at is only known once the RI
            // terms are estimated (the band is remaining minus those terms, so a test on
            // the remaining does not cover the band - it is the stricter of the two). The
            // count is one pass over the pair space, the same pass the nested's own
            // pre-gate runs when its band is overrun.
            excludedPeakRowWidth = internal::CountSchwarzPeakRowWidth(*schwarz, options.accuracy);

            // The screened task-grid count for the footprint's taskListBytes
            // term: CountSchwarzSurvivingRiTasks over
            // the (orbital pair x aux shell) grid, the count the build's
            // exact reserve realizes. The per-aux-shell bound is the aux
            // pair list's diagonal - the kets are single shells, the same
            // extraction BuildScreenedRiTaskList runs (and the rebuilt pair
            // lists at the light/fast builds are deterministic - the same
            // inputs count the same rows).
            auto schwarzAux = ComputeSchwarzBounds(molecule, auxBasisSet);

            if (!schwarzAux.has_value())
            {
                return std::unexpected(schwarzAux.error());
            }

            std::vector<double> auxShellBounds(nAuxShells);

            for (std::size_t auxShell = 0; auxShell < nAuxShells; ++auxShell)
            {
                auxShellBounds[auxShell] =
                    (*schwarzAux)[PairIndexOf(auxShell, auxShell, auxPairList)];
            }

            screenedTaskCount =
                internal::CountSchwarzSurvivingRiTasks(*schwarz, auxShellBounds, options.accuracy);

            // A-priori exclusion (ii): the (uv|P) values buffer alone cannot
            // fit - the tensor is the RI-J's reason to exist, the light rung
            // is mandatory. nAuxFuncs is the FUNCTION count, never the shell
            // count.
            const std::size_t n = pairList.functionCount;
            const std::size_t nAuxFuncs = auxPairList.functionCount;
            tensorBytes = 8 * n * n * nAuxFuncs;
            tensorExcluded = tensorBytes > remaining;

            threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());

            // The engaged orbit action's charge, from the SAME formula the
            // action reports (RiOrbitActionBytes over both bases' own
            // counts) under the SAME condition BuildRiOrbitReduction
            // engages it, so charge == realized by equality.
            if (RiOrbitExpansionEngaged(options))
            {
                orbitActionBytes =
                    internal::RiOrbitActionBytes(options.symmetryReduction->groupOrder,
                                                 pairList.pairs.size(),
                                                 pairList.shells.size(),
                                                 pairList.functionCount,
                                                 auxPairList.pairs.size(),
                                                 auxPairList.shells.size(),
                                                 auxPairList.functionCount);
            }
        }

        // One attempt of the estimate/clamp/reserve flow. The light attempt
        // zeroes the tensor and the Eigen copy (the recompute rung's terms);
        // its clamp and reservation math is otherwise identical. A nonzero
        // clamp lands the fired estimate at or below the remaining bytes
        // (the two arenas shrink together), so a failed reserve means the
        // budget moved concurrently, never a stale estimate.
        struct Attempt {
            enum class Outcome : std::uint8_t { kReserved, kClampFailed, kReserveFailed };

            Outcome outcome = Outcome::kClampFailed;
            FockModeInfo info;
        };

        // The nested direct-exchange half's charge at the rung its own Create selects.
        // The rung rule is that Create's own (the counted pre-gate of the light decision,
        // fock_build.cpp): the counted pattern cannot fit the band this attempt leaves the
        // nested, so the nested takes the LightPath. The band is CHARGE-INDEPENDENT - the
        // outer reserves everything but the exchange half (outerReservation =
        // predictedBytes - exchangeBytes below), so a charge of any size leaves the nested
        // exactly the remaining bytes less the RI terms, and the nested self-gates on them
        // - which is why the RUNG must be the thing charged, not a bound on the charge:
        // the fast rung's whole-pattern CSR (8 x the counted rows, 183.74 GiB at the
        // 4,974 manifest case) is not an object the light rung ever holds, and charging it
        // refused the stack (the engine-exclusion-i record). The light branch prices with
        // the SAME functions the rung's Create-time light decision calls
        // (internal/light_footprint.hpp): the chunk auto-sizing closed form and
        // LightFootprint, over the band above and the pre-gate's own peak-chunk bounds (a
        // chunk's ket set is the union of its rows', each bounded by the widest row - the
        // exclusion route, no CSR), so the charge is the object the nested half holds.
        struct ExchangeFootprint {
            std::size_t totalBytes = 0;
            std::size_t scratchBytes = 0;
            bool lightRung = false;
        };

        auto priceExchange = [&](std::size_t cap, std::size_t riBytes) -> ExchangeFootprint {
            ExchangeFootprint price;
            const std::size_t band = riBytes < remaining ? remaining - riBytes : 0;
            price.lightRung = patternCount > band / 8;

            if (!price.lightRung)
            {
                const internal::DirectFootprintTerms fast =
                    internal::DirectFootprint(molecule,
                                              basisSet,
                                              pairList,
                                              patternCount,
                                              cap,
                                              threadCount,
                                              false,
                                              0,
                                              options.accuracy,
                                              true);
                price.totalBytes = fast.Total();
                price.scratchBytes = fast.scratchBytes;
                return price;
            }

            const std::size_t nPairs = pairList.pairs.size();
            const std::size_t maxPairPayload =
                internal::MaxPairPayload(molecule, basisSet, pairList);
            // The structural-only query: the light model's shared structural block (the
            // exchange term is not part of it - exchangeEngaged false, the
            // threading), the same term the light decision's own chunk sizing passes.
            const std::size_t structuralBytes = internal::DirectFootprint(molecule,
                                                                          basisSet,
                                                                          pairList,
                                                                          0,
                                                                          cap,
                                                                          threadCount,
                                                                          false,
                                                                          0,
                                                                          options.accuracy,
                                                                          false)
                                                    .structuralBytes;
            // The pre-gate's counted peak row width, carried out of the decision block:
            // the light decision's ket bound when no sweep ran. Zero (no count was made,
            // or no qualifying row) falls back to the pair space, exactly as the
            // decision's own `maxRow = nPairs` initializer does.
            const std::size_t maxRow = excludedPeakRowWidth != 0 ? excludedPeakRowWidth : nPairs;
            const std::size_t chunkPairs = internal::AutoLightChunkPairs(
                band,
                nPairs * sizeof(internal::MdPairData),
                structuralBytes,
                cap * threadCount,
                internal::LightChunkIndexBytes(nPairs) +
                    internal::LightShellsBytes(pairList.shells.size()) +
                    internal::ScreenedQuartetBytes(pairList.functionCount, options.accuracy),
                maxPairPayload,
                maxRow,
                nPairs);
            const std::size_t boundRows = std::min(chunkPairs, nPairs);
            const internal::LightFootprintTerms light =
                internal::LightFootprint(molecule,
                                         basisSet,
                                         pairList,
                                         chunkPairs,
                                         std::min(nPairs, boundRows * maxRow),
                                         std::min(boundRows * nPairs, boundRows * maxRow),
                                         maxPairPayload,
                                         cap,
                                         threadCount,
                                         false,
                                         options.accuracy);
            price.totalBytes = light.Total();
            price.scratchBytes = light.scratchBytes;
            return price;
        };

        auto attempt = [&](bool light, bool blockedMetricRung = false) -> Attempt {
            Attempt result;
            result.info.budgetBytes = budget != nullptr ? budget->CapacityBytes() : 0;
            result.info.remainingAtDecision = remaining;
            result.info.maxBatchBytes = options.maxBatchBytes;
            result.info.tensorExcluded = light && tensorExcluded;

            internal::RiFootprintTerms ri = internal::RiFootprint(molecule,
                                                                  basisSet,
                                                                  auxBasisSet,
                                                                  pairList,
                                                                  auxPairList,
                                                                  options.maxBatchBytes,
                                                                  threadCount,
                                                                  light,
                                                                  blockedMetricRung,
                                                                  screenedTaskCount,
                                                                  orbitActionBytes);
            // The nested direct-exchange half is always engaged (the RI-J
            // runs it in buildExchangeOnly mode) - the exchange term is
            // live here, exchangeEngaged true (the threading).
            // Its charge is the rung its own Create selects (priceExchange
            // above), priced from the band this attempt leaves it.
            const ExchangeFootprint exchange = priceExchange(options.maxBatchBytes, ri.Total());
            result.info.predictedBytes = ri.Total() + exchange.totalBytes;

            // The first-iteration terms ride the folded estimate above,
            // and the two-arena clamp below is the sole
            // refusal mechanism: ClampBatchBytes refuses iff the deficit
            // exceeds the two-arena capacity 2T(batch-1) - a rung whose
            // clamp-ineligible mass cannot fit even after the full shrink.
            // The clamp-IRREDUCIBLE floor check drafted for this zone
            // proved decision-inert (the clamp's refusal region strictly
            // contains the floor's, so every floor refusal is clamp-shadowed)
            // and was deleted.
            const std::size_t clamped = internal::ClampBatchBytes(
                options.maxBatchBytes, threadCount, result.info.predictedBytes, remaining, 2);

            if (clamped == 0)
            {
                return result;
            }

            // Re-estimate with the clamped batch - the fired terms.
            ri = internal::RiFootprint(molecule,
                                       basisSet,
                                       auxBasisSet,
                                       pairList,
                                       auxPairList,
                                       clamped,
                                       threadCount,
                                       light,
                                       blockedMetricRung,
                                       screenedTaskCount,
                                       orbitActionBytes);
            const ExchangeFootprint fired = priceExchange(clamped, ri.Total());
            result.info.predictedBytes = ri.Total() + fired.totalBytes;
            result.info.orbitalAuxBytes = ri.orbitalStoreBytes + ri.auxStoreBytes;
            result.info.taskListBytes = ri.taskListBytes;
            result.info.tensorBytes = ri.tensorBytes;
            result.info.riMatrixBytes = ri.riMatrixBytes;
            result.info.metricBytes = ri.metricBytes;
            result.info.scratchBytes = ri.scratchBytes;
            result.info.exchangeBytes = fired.totalBytes;
            result.info.exchangeScratchBytes = fired.scratchBytes;
            result.info.patternExcluded = fired.lightRung;
            result.info.maxBatchBytes = clamped;
            result.info.mode = light ? FockBuildMode::kLightPath : FockBuildMode::kFastPath;

            // The outer reservation: the full estimate minus the exchange
            // half's own estimate - the nested exchange charges its own
            // parts at its own Create (nesting order = reservation order)
            // from the OPTIONS' cap (effectiveOptions.maxBatchBytes below;
            // this zone's clamped batch is NOT inherited - the nested
            // re-derives its own). The nested now fires k > 1 when its
            // post-reservation band has room for the k live per-slot bounds
            // (the authorization is budget-conditional) and clamps ITS cap to its own
            // saturated fixed point. The fired charge never passes the
            // band - the exchange self-gates on the live remaining like a
            // free-standing run - so the total stack commit never exceeds
            // the budget's capacity and the CWA never loosens. The band is
            // the estimated term whenever the outer decision left no
            // slack (the stack's total then sits at or below
            // predictedBytes, exactly when the saturated fixed point
            // consumes the band); a budget surplus is the exchange's to
            // fold into, still band-bounded.
            const std::size_t outerReservation =
                result.info.predictedBytes - result.info.exchangeBytes;

            if (!budget->Reserve(outerReservation))
            {
                result.info.mode = FockBuildMode::kLightPath;
                result.info.reservedBytes = 0;
                result.outcome = Attempt::Outcome::kReserveFailed;
                return result;
            }

            result.info.reservedBytes = outerReservation;
            result.outcome = Attempt::Outcome::kReserved;
            return result;
        };

        if (budget != nullptr)
        {
            const bool skipFastPath = options.forceLightRung || tensorExcluded;

            if (!skipFastPath)
            {
                Attempt fastAttempt = attempt(false);

                if (fastAttempt.outcome == Attempt::Outcome::kReserved)
                {
                    modeInfo = fastAttempt.info;
                }
            }

            if (!modeInfo.has_value())
            {
                // The light rung: the same flow with the tensor terms
                // zeroed. When it cannot fit either, the blocked-metric
                // rung takes over: the light rung with
                // the metric built strip-wise (BuildBlockedAuxMetric - no
                // full values buffer, no Tensor/TensorToEigen copies, a
                // 2 x 8 nAuxFuncs^2 eigen-class peak against the unblocked
                // 3 x - the blocked path frees its solver input before the
                // State eigenvector copy, the unblocked path does not; the
                // solver's own working-matrix count is one on both, the
                // eigen-solver reconciliation), so its own estimate can
                // fit below the light rung's.
                Attempt lightAttempt = attempt(true);

                if (lightAttempt.outcome != Attempt::Outcome::kReserved)
                {
                    Attempt blockedAttempt = attempt(true, true);

                    if (blockedAttempt.outcome != Attempt::Outcome::kReserved)
                    {
                        // The refusal keeps naming the light-rung estimate
                        // and its standing wording (the four-rung ladder
                        // intact - the next-rung sentence stays as
                        // written, even though the blocked-metric rung now
                        // exists: the sentence documents the ladder shape,
                        // and the wording's "(named, not built)" staleness
                        // is known and left as written for now).
                        std::string message = internal::LightRungRefusal(
                            "RI-J", lightAttempt.info.predictedBytes, remaining, tensorExcluded);

                        if (blockedAttempt.outcome == Attempt::Outcome::kReserveFailed)
                        {
                            message +=
                                " (the budget moved between the decision and the reservation)";
                        }

                        return std::unexpected(
                            qcx::Error{qcx::ErrorCode::kUnimplemented, std::move(message)});
                    }

                    modeInfo = blockedAttempt.info;
                    modeInfo->blockedMetricRung = true;
                    lightRung = true;
                } else
                {
                    modeInfo = lightAttempt.info;
                    lightRung = true;
                }
            }

            effectiveOptions.maxBatchBytes = modeInfo->maxBatchBytes;
            effectiveOptions.workspaceBudget = nullptr;
        } else
        {
            // The forced light rung without a budget: no decision, no
            // reservation, no mode record - the flag's unit-scale
            // validation seam (the budget seam only engages the rung at
            // nt84-class scales).
            lightRung = true;
        }
    }

    // The orbit expansion is wired on the FAST path's materialized tensor
    // pass only (its single Create-time evaluation of the grid). The light
    // rung recomputes the grid per iteration, which is the amortized variant
    // and a separate increment; engaging the flag on a rung that would
    // ignore it is refused rather than silently dropped - the same
    // never-silent discipline the reduction's own validation gets. A budget
    // that selects the light rung under an engaged flag reaches this too.
    if (lightRung && RiOrbitExpansionEngaged(options))
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kUnimplemented,
            "symmetryOrbitExpansion is not wired on the light rung (the per-iteration 3c "
            "recompute): it engages on the fast path's materialized tensor pass only - drop "
            "the flag or the workspace budget that selected the rung"});
    }

    // The fast path's n^2 x nAux Eigen copy of the (uv|P) tensor (rows
    // u*n + v, columns P); empty in the light rung - the recompute mode
    // holds the payload instead (RiMatrix() returns the empty matrix).
    Eigen::MatrixXd riMatrix;

    // The run's Create-time counts (ri_engine.hpp
    // RiTermCounters): the metric occurrence - every rung (fast, light,
    // forced-light and blocked) builds the same full (P|Q) triangle once -
    // plus the fast path's single tensor-pass occurrence (BuildRiTensor
    // below). The light rung replaces the tensor with the retained
    // payload; its per-pass partial is precomputed separately (its
    // BuildFock calls add it twice per call - see _lightRungPassCounters).
    RiTermCounters runCounters;
    RiTermCounters lightRungPassCounters;

    if (!lightRung)
    {
        auto riTensor =
            BuildRiTensor(molecule, basisSet, auxBasisSet, effectiveOptions, &runCounters);

        if (!riTensor.has_value())
        {
            return std::unexpected(riTensor.error());
        }

        const std::size_t n = riTensor->Shape()[0];
        const std::size_t nAux = riTensor->Shape()[2];
        riMatrix.resize(static_cast<Eigen::Index>(n * n), static_cast<Eigen::Index>(nAux));

        for (std::size_t u = 0; u < n; ++u)
        {
            for (std::size_t v = 0; v < n; ++v)
            {
                for (std::size_t p = 0; p < nAux; ++p)
                {
                    riMatrix(static_cast<Eigen::Index>(u * n + v), static_cast<Eigen::Index>(p)) =
                        (*riTensor)(u, v, p);
                }
            }
        }
    }

    // The light rung replaces the materialized tensor with the retained
    // per-iteration recompute inputs (the pair stores, the screened task
    // list and the two shell-pair lists); the metric, its eigendecomposition
    // and the nested exchange below are shared with the fast path.
    std::optional<LightRungPayload> lightPayload;

    if (lightRung)
    {
        auto payload = BuildLightRungPayload(molecule, basisSet, auxBasisSet, effectiveOptions);

        if (!payload.has_value())
        {
            return std::unexpected(payload.error());
        }

        // The light rung's per-pass partial: one count over
        // the retained screened task list (each recompute pass evaluates
        // it once; BuildFock runs two passes per call).
        internal::AccumulateScreenedRiPass(lightRungPassCounters,
                                           payload->combinedStore,
                                           payload->nOrbitalPairs,
                                           payload->taskList);
        lightPayload = std::move(*payload);
    }

    // The metric build's dispatch: the blocked-metric rung (Create's
    // decision above) writes the (P|Q) matrix directly - the strip-wise
    // BuildBlockedAuxMetric, bit-identical to the unblocked build's matrix
    // (the same phantom quartets through the same kernels; the strip
    // partition is value-neutral by contract) - while the fast, light and
    // forced-light paths keep the memory Tensor + TensorToEigen route.
    Eigen::MatrixXd metricMatrix;

    if (modeInfo.has_value() && modeInfo->blockedMetricRung)
    {
        auto blockedMetric =
            BuildBlockedAuxMetric(molecule, auxBasisSet, effectiveOptions, &runCounters);

        if (!blockedMetric.has_value())
        {
            return std::unexpected(blockedMetric.error());
        }

        metricMatrix = std::move(*blockedMetric);
    } else
    {
        auto metric = BuildAuxMetric(molecule, auxBasisSet, effectiveOptions, &runCounters);

        if (!metric.has_value())
        {
            return std::unexpected(metric.error());
        }

        metricMatrix = internal::TensorToEigen(*metric);
    }

    FockBuildOptions exchangeOptions;
    exchangeOptions.accuracy = options.accuracy;
    exchangeOptions.maxBatchBytes = effectiveOptions.maxBatchBytes;
    exchangeOptions.useDensityScreening = true;
    exchangeOptions.useCertifiedMixedPrecision = options.accuracy != AccuracyPreset::kTight;
    exchangeOptions.buildExchangeOnly = true;
    // The SAME budget pointer: the nested exchange charges its own parts at
    // its own Create (the outer reserved the difference - nesting order =
    // reservation order). The counted sweep count is NOT carried: the outer
    // no longer skips the nested's a-priori pre-gate. The skip was written
    // against the ALL-SURVIVE pre-gate, whose bound over-refuses large-sparse
    // systems - but the counted pre-gate can no longer refuse a stack the
    // outer proved
    // feasible, and it is now the route by which the nested reaches its own
    // LightPath: skipping it forced the fast rung's whole-pattern CSR (183.74
    // GiB at the 4,974 manifest case, 16.8x the grant) to be materialized
    // before the light decision could be made, on a rung whose own object is
    // a budget-sized chunk. The nested still runs its own counting pass and
    // estimate, and its charge is priced at that rung (priceExchange).
    exchangeOptions.workspaceBudget = options.workspaceBudget;
    auto exchange =
        DirectJkFockBuilder::Create(molecule, basisSet, coreHamiltonian, exchangeOptions);

    if (!exchange.has_value())
    {
        return std::unexpected(exchange.error());
    }

    // The metric's floored eigen-inverse (BUG-2): M is SPD but extremely
    // ill-conditioned (see kMetricFloorEpsilon above), so the raw
    // factorized solve would amplify the near-null directions into O(1e9)
    // solution components that cancel in J - unusable at double precision.
    // Eigendecompose once (nAux x nAux is tiny - tens to a few hundred),
    // floor the eigenvalues below options.metricFloorEpsilon * lambdaMax
    // (per-engine; the default 1e-10 equals the kMetricFloorEpsilon
    // constant), and keep V diag(invLambda) V^T as the per-iteration solve.
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(metricMatrix);

    if (solver.info() != Eigen::Success)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the auxiliary-basis metric failed its eigendecomposition"});
    }

    const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
    const double lambdaMax = eigenvalues.maxCoeff();
    const double floor = options.metricFloorEpsilon * lambdaMax;
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

    if (modeInfo.has_value() && modeInfo->blockedMetricRung)
    {
        // The blocked rung's estimate charges the 2 x 8 nAuxFuncs^2
        // eigen-class peak (the strip-built solver input plus the
        // solver's m_eivec - the solver keeps no separate copy of its
        // input, compute() copies the lower triangle into m_eivec): the
        // input must die before the State constructor's eigenvector
        // copy, or the live peak would carry a third nAuxFuncs^2 matrix
        // (the unblocked rung's count - it retains its input through
        // this copy; the eigen-solver reconciliation).
        Eigen::MatrixXd empty;
        metricMatrix.swap(empty);
    }

    auto state = std::make_shared<State>(std::move(riMatrix),
                                         solver.eigenvectors(),
                                         std::move(inverseEigenvalues),
                                         std::move(*exchange),
                                         effectiveOptions,
                                         std::move(lightPayload));
    state->_modeInfo = modeInfo;
    // The orbit expansion's engagement, under the SAME condition the tables
    // are built with (and the same one BuildRiOrbitReduction refuses the
    // light rung under, which is why a light-rung run never reaches here):
    // the flag, both reductions, and a non-trivial group on both bases.
    state->_orbitExpansionEngaged = RiOrbitExpansionEngaged(options);
    // The run's term counters ride the shared state (TermCounters()
    // reads them post-run; BuildFock's end-of-call accumulation adds the
    // per-call occurrences).
    state->_termCounters = runCounters;
    state->_lightRungPassCounters = lightRungPassCounters;
    return RiJkFockBuilder(std::move(state));
}

const Eigen::MatrixXd& RiJkFockBuilder::RiMatrix() const noexcept {
    return _state->_riMatrix;
}

const std::optional<FockModeInfo>& RiJkFockBuilder::ModeInfo() const noexcept {
    return _state->_modeInfo;
}

const std::optional<FockModeInfo>& RiJkFockBuilder::ExchangeModeInfo() const noexcept {
    return _state->_exchange.ModeInfo();
}

const RiTermCounters& RiJkFockBuilder::TermCounters() const noexcept {
    return _state->_termCounters;
}

bool RiJkFockBuilder::OrbitExpansionEngaged() const noexcept {
    return _state->_orbitExpansionEngaged;
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> RiJkFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    FockBuildStats* statsOut) const {
    const State& state = *_state;
    const std::size_t n = static_cast<std::size_t>(density.Shape()[0]);
    const Eigen::Index n2 = static_cast<Eigen::Index>(n * n);

    // The exchange half runs through the direct builder in its
    // exchange-only mode, which may route quartets through the certified
    // fp32 lane (Create enables the lane unless accuracy is kTight), but
    // the RI builder cannot consume the delivered certified bound - its
    // BuildFock exposes no certifiedBoundSumOut. The RI-J error is
    // dominated by the metric/RI approximation (measured ~2.3e-4 max,
    // ri_engine_test.cpp), orders above the fp32 lane's class of errors,
    // so the gap is harmless in practice but documented: the certified
    // story covers the direct path only.
    // The RIJCOSX combination: J comes from the RI path below
    // (2j), K from this exchange-only call. This is the same shape as the
    // split UHF combination (MakeDirectUhfFockBuilder, scf/tests/
    // direct_uhf_test.cpp - coulombOnly + exchangeOnly with one H
    // subtracted, because there BOTH split results carry their own H copy)
    // with a different H accounting: the RI path contributes no H, the
    // exchange-only call is the only H carrier, so no subtraction is needed
    // here. Deliberately not unified into a shared helper - the accounting
    // differs; cross-reference both ways so the two stay consistent.
    // The optional per-call stats sink forwards to the exchange half (the
    // RI contractions contribute no quartets); null keeps the zero-cost
    // path of the nested builder.
    auto exchangeFock = state._exchange.BuildFock(density, nullptr, statsOut);

    if (!exchangeFock.has_value())
    {
        return std::unexpected(exchangeFock.error());
    }

    Eigen::VectorXd dVec(n2);
    Eigen::VectorXd fock(n2);

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const Eigen::Index flat = static_cast<Eigen::Index>(u * n + v);
            dVec(flat) = density(u, v);
            fock(flat) = (*exchangeFock)(u, v);
        }
    }

    // The Coulomb half, through the ONE contraction path this call shares
    // with the split BuildCoulombOnly (BuildRiCoulombVector): the fused and
    // the split call cannot disagree about the rung they took, the counters
    // they charged or the convention's factor of two.
    auto coulombVector = BuildRiCoulombVector(dVec, n);

    if (!coulombVector.has_value())
    {
        return std::unexpected(coulombVector.error());
    }

    auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            const Eigen::Index flat = static_cast<Eigen::Index>(u * n + v);
            (*tensor)(u, v) = fock(flat) + (*coulombVector)(flat);
        }
    }

    // The per-call accumulations (ri_engine.hpp
    // RiTermCounters), at one site at the end of the call so a failed call
    // adds nothing: the qx/gx exchange occurrences from the nested
    // exchange's per-call counts (statsOut was given - the exchange's
    // counts exist only on instrumented calls). The light rung's two
    // recompute-pass occurrences belong to the Coulomb half and are counted
    // where that half's contraction ran (BuildRiCoulombVector).
    if (statsOut != nullptr)
    {
        state._termCounters.qx += statsOut->fp64QuartetCount + statsOut->fp32QuartetCount;
        state._termCounters.gx += statsOut->primitiveProductSum;
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>
RiJkFockBuilder::BuildExchangeOnly(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    FockBuildStats* statsOut) const {
    // The nested direct-exchange builder's exchange-only result, verbatim:
    // H - K(rho), the half Create configured from this engine's own options
    // (accuracy, batch cap, the certified-lane default the preset derives,
    // the shared workspace budget and its Create-time reservation). What is
    // exposed here is the SAME builder the fused BuildFock calls, so an
    // unrestricted run's K half IS the restricted run's K half.
    return _state->_exchange.BuildFock(density, nullptr, statsOut);
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> RiJkFockBuilder::BuildCoulombOnly(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    FockBuildStats* statsOut) const {
    const std::size_t n = static_cast<std::size_t>(density.Shape()[0]);
    const Eigen::Index n2 = static_cast<Eigen::Index>(n * n);

    Eigen::VectorXd dVec(n2);

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            dVec(static_cast<Eigen::Index>(u * n + v)) = density(u, v);
        }
    }

    auto coulombVector = BuildRiCoulombVector(dVec, n);

    if (!coulombVector.has_value())
    {
        return std::unexpected(coulombVector.error());
    }

    // The measured zero (ri_full_fock.hpp's rule for this call shape): the
    // RI contractions evaluate no quartets, so a caller that asked for
    // counts gets this call's structural zeros rather than whatever the
    // previous call left in the struct.
    if (statsOut != nullptr)
    {
        *statsOut = FockBuildStats{};
    }

    auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});

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

qcx::Result<Eigen::VectorXd> RiJkFockBuilder::BuildRiCoulombVector(
    const Eigen::VectorXd& densityVector, std::size_t n) const {
    const State& state = *_state;

    // v = I^T d, then M w = v through the floored eigen-inverse (the
    // near-null metric directions were zeroed at Create), then J = I w.
    // The fast path contracts the retained n2 x nAux matrix through the
    // linalg BLAS seam; the light rung recomputes the two
    // products batch-at-a-time against one maxBatchBytes-class buffer (the
    // per-iteration 3c recompute - a different contraction order, not
    // bit-identical to the materialized products, validated within the
    // per-preset accuracy budget). The eigen-inverse solve steps in between stay
    // Eigen - DenseMultiply is for plain products, not solves or
    // decompositions. The transpose operand is materialized once per call
    // (a copy in the same O(n2 x nAux) class as the product itself - the
    // seam has no transpose flag).

    // The two contractions dispatch on useCudaGemm: cuBLAS when
    // requested and the build carries the seam, else the CPU seam - a
    // CUDA-less build compiles the lambda's single return and is byte
    // identical to the CPU-only path. The light rung never calls it.
    const auto product = [&state](const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
#ifdef QcxHasCuda
        if (state._options.useCudaGemm)
        {
            return CublasProduct(a, b);
        }
#endif
        return qcx::linalg::DenseMultiply(a, b);
    };

    Eigen::VectorXd v;
    // One assembly per iteration, shared by the light rung's two passes.
    std::optional<internal::RiBatchAssembly> lightAssembly;

    if (_state->_lightRung.has_value())
    {
        // The light rung's per-iteration recompute slice (the
        // light_rung_slice term): the assembly's containers capture the
        // family tag at their construction inside AssembleRiBatches.
        qcx::memory::AllocationTagScope sliceScope(qcx::memory::AllocationTag::kLightRungSlice);
        auto assembly = internal::AssembleRiBatches(_state->_lightRung->combinedStore,
                                                    _state->_lightRung->nOrbitalPairs,
                                                    _state->_lightRung->taskList,
                                                    _state->_options.maxBatchBytes);

        if (!assembly.has_value())
        {
            return std::unexpected(assembly.error());
        }

        lightAssembly = std::move(*assembly);
        auto vResult =
            LightRungTransposeProduct(*_state->_lightRung, *lightAssembly, densityVector, n);

        if (!vResult.has_value())
        {
            return std::unexpected(vResult.error());
        }

        v = std::move(*vResult);
    } else
    {
        const Eigen::MatrixXd riTranspose = state._riMatrix.transpose();
        const Eigen::MatrixXd dMatrix(densityVector);
        auto vColumn = product(riTranspose, dMatrix);

        if (!vColumn.has_value())
        {
            return std::unexpected(vColumn.error());
        }

        v = vColumn->col(0);
    }

    const Eigen::VectorXd w =
        state._metricEigenvectors *
        state._inverseMetricEigenvalues.cwiseProduct(state._metricEigenvectors.transpose() * v);

    Eigen::VectorXd j;

    if (_state->_lightRung.has_value())
    {
        auto jResult = LightRungProduct(*_state->_lightRung, *lightAssembly, w, n);

        if (!jResult.has_value())
        {
            return std::unexpected(jResult.error());
        }

        j = std::move(*jResult);
    } else
    {
        const Eigen::MatrixXd wMatrix(w);
        auto jColumn = product(state._riMatrix, wMatrix);

        if (!jColumn.has_value())
        {
            return std::unexpected(jColumn.error());
        }

        j = jColumn->col(0);
    }

    // The light rung's two recompute-pass occurrences (the v pass and the j
    // pass above each evaluated the retained screened task list once - the
    // Create-time partial; zero on the fast and legacy paths, whose single
    // tensor pass was counted at Create). Counted HERE because here is where
    // the passes happened: both callers of this helper run exactly one such
    // pair per call.
    if (_state->_lightRung.has_value())
    {
        state._termCounters.x += 2 * state._lightRungPassCounters.x;
        state._termCounters.p3 += 2 * state._lightRungPassCounters.p3;
        state._termCounters.g3 += 2 * state._lightRungPassCounters.g3;
    }

    // The RI-J Fock contribution is 2 J_RI(rho) (the J_uv = sum_P (uv|P)
    // w_P contraction's factor of two in F(D) = H + 2J_RI(rho) - K(rho)),
    // applied here so the two assemblies that consume this vector - the
    // fused BuildFock and the unrestricted split - share one definition of
    // it.
    return 2.0 * j;
}

} // namespace qcx::integrals
