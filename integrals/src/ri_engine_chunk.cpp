// The chunked RI-tensor build entry point (ri_engine.hpp BuildRiTensorChunk,
// (n^2 x k_c) contraction-layout slice of (uv|P) for one contiguous range
// of aux shells, with the CHUNK-SCOPED screened task list - the full
// nPairs x nAuxShells list of the monolithic
// path is never materialized). The chunk runs through the EXISTING
// RunRiBatches machinery (md_vrr_3c.hpp) against a chunk-scoped combined
// store (the orbital pair store plus the chunk's aux pair data); the
// values are bit-identical to the monolithic BuildRiTensor's for the same
// tasks, and the scatter reproduces the monolithic tensor's (u, v, P)
// element space in the flat rows-u*n+v layout - both triangles of every
// computed task block, exactly like the Create-time flatten of the
// monolithic tensor (ri_engine.cpp).

#include "internal/fock_screen.hpp"
#include "internal/md_attribution.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_vrr_3c.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::integrals {

namespace {

// The same l <= kMaxEngineL support check the monolithic paths run
// (ri_engine.cpp CheckSupported).
qcx::Result<void> CheckSupportedChunk(const ShellPairList& pairList) {
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

} // namespace

qcx::Result<Eigen::MatrixXd> BuildRiTensorChunk(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const qcx::basisset::BasisSet& auxBasisSet,
                                                std::size_t auxShellStart,
                                                std::size_t auxShellEnd,
                                                const RiEngineOptions& options,
                                                RiTermCounters* termCountersOut,
                                                RiScreenedPassState* passState) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    // This entry point carries NO point-group reduction, and says so by name
    // rather than by allowing a caller to set one and lose it: RiEngineOptions
    // is the ri_j_link family's option struct, so its three symmetry fields
    // (symmetryReduction, auxSymmetryReduction, symmetryOrbitExpansion) are
    // present in this call's type and are read nowhere below - the chunk's
    // task list is built per aux-shell range and the orbit expansion is the
    // in-memory tensor pass's mechanism (ri_engine.cpp), which the chunked
    // path does not run. A cell a caller supplied and that changed nothing is
    // the substitution the refusal ladder forbids, exactly as RiFullFockBuilder's own
    // Create refusal records for its callers of this same function
    // (ri_full_fock.cpp Create, the "BuildRiTensorChunk ignores
    // RiEngineOptions::symmetryOrbitExpansion" refusal) - that refusal exists
    // so ITS callers never reach here with a non-null reduction; this one
    // closes the surface for every other caller, the disk rung's own Create
    // included. Both fields are refused, so the half-supplied pair
    // ValidateRiOrbitExpansion reports as kInvalidArgument elsewhere is
    // covered by the same sentence.
    if (options.symmetryReduction != nullptr || options.auxSymmetryReduction != nullptr)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "BuildRiTensorChunk does not implement the point-group reduction, so a "
                       "non-null symmetryReduction (or auxSymmetryReduction) cannot be honoured: "
                       "the chunked 3-center build evaluates its chunk-scoped task list and has "
                       "no orbit expansion, so the fields would be accepted and silently ignored. "
                       "Clear both fields for this entry point - the reduction is the ri_j_link "
                       "family's in-memory option set (ri_engine.hpp RiJkFockBuilder)"});
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

    auto orbitalSupported = CheckSupportedChunk(*pairList);

    if (!orbitalSupported.has_value())
    {
        return std::unexpected(orbitalSupported.error());
    }

    auto auxSupported = CheckSupportedChunk(*auxPairList);

    if (!auxSupported.has_value())
    {
        return std::unexpected(auxSupported.error());
    }

    const std::size_t nAuxShells = auxPairList->shells.size();

    if (nAuxShells == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the auxiliary basis is empty"});
    }

    // The chunk range is a half-open interval over the molecule-scoped aux
    // shells; a chunk never splits a shell (a task's block spans a whole
    // shell).
    if (auxShellStart >= auxShellEnd || auxShellEnd > nAuxShells)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the chunk's aux shell range is empty or exceeds the aux shell count"});
    }

    const std::size_t n = pairList->functionCount;
    const std::size_t nOrbitalPairs = pairList->pairs.size();
    const std::size_t chunkShellCount = auxShellEnd - auxShellStart;

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

    // The CHUNK-SCOPED combined store: the orbital pairs followed by the
    // chunk's aux pair data (the full aux pair data is never built on the
    // chunk path - a chunk's tasks only reference its own aux shells).
    std::vector<internal::MdPairData> combinedStore;
    combinedStore.reserve(nOrbitalPairs + chunkShellCount);
    combinedStore.insert(combinedStore.end(),
                         std::make_move_iterator(orbitalStore->begin()),
                         std::make_move_iterator(orbitalStore->end()));

    for (std::size_t shell = auxShellStart; shell < auxShellEnd; ++shell)
    {
        const internal::MdShellInput& input = (*auxShells)[shell];
        const std::array<double, 3> center = {input.cx, input.cy, input.cz};
        combinedStore.push_back(internal::BuildAuxPairData(input.contractions, center));
    }

    // The chunk-scoped screened task list: the Schwarz
    // screen runs over the chunk's aux shells only, with the chunk-scoped
    // exact reserve - the full nPairs x nAuxShells list never exists on
    // this path (the monolithic form now carries the counted treatment too:
    // footprint.hpp taskListBytes charges the surviving-cell count, the
    // counted fix - the chunked build must not inherit the monolithic
    // reserve either way).
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

    std::vector<double> schwarzAuxShell(chunkShellCount);

    for (std::size_t shell = auxShellStart; shell < auxShellEnd; ++shell)
    {
        schwarzAuxShell[shell - auxShellStart] =
            (*schwarzAux)[PairIndexOf(shell, shell, *auxPairList)];
    }

    const double schwarzBudget = SchwarzThreshold(options.accuracy);
    // The counted exact reserve (the counted treatment mirrored on the
    // chunk grid): CountSchwarzSurvivingRiTasks over the full orbital
    // bounds and the chunk's aux bounds is bit-identical to the fill's
    // per-cell test below, so the reserved capacity is exactly the realized
    // task count.
    std::vector<internal::RiTask> tasks;
    tasks.reserve(
        internal::CountSchwarzSurvivingRiTasks(*schwarzOrbital, schwarzAuxShell, options.accuracy));

    for (std::size_t braPair = 0; braPair < nOrbitalPairs; ++braPair)
    {
        for (std::size_t localShell = 0; localShell < chunkShellCount; ++localShell)
        {
            if ((*schwarzOrbital)[braPair] * schwarzAuxShell[localShell] < schwarzBudget)
            {
                continue;
            }

            // The task's ket index is the CHUNK-SCOPED store index: the
            // chunk's aux pair data sits at nOrbitalPairs + localShell.
            tasks.push_back(internal::RiTask{braPair, nOrbitalPairs + localShell});
        }
    }

    const std::size_t chunkFunctionCount =
        auxPairList->shells[auxShellEnd - 1].functionOffset +
        ShellFunctionCount(auxPairList->shells[auxShellEnd - 1]) -
        auxPairList->shells[auxShellStart].functionOffset;

    Eigen::MatrixXd slice = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n * n),
                                                  static_cast<Eigen::Index>(chunkFunctionCount));

    // An all-screened chunk is a valid zero slice (the global monolithic
    // list can still be non-empty elsewhere - the empty-task refusal of
    // RunRiBatches is not a chunk-level error).
    if (tasks.empty())
    {
        return slice;
    }

    // The chunk's (uv|P) values buffer (the rij_fast_tensor term - the
    // same charged tensor bytes as the monolithic pass): the containers
    // capture the family at their construction, and the scope covers the
    // pass through the slice scatter.
    qcx::memory::AllocationTagScope tensorScope(qcx::memory::AllocationTag::kRijFastTensor);
    internal::TaggedVector<internal::RiTask> computed;
    internal::TaggedVector<double> values;
    auto run = internal::RunRiBatches(
        combinedStore, nOrbitalPairs, tasks, options.maxBatchBytes, computed, values);

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    if (termCountersOut != nullptr)
    {
        // The tensor-pass occurrence of THIS chunk's kernel
        // evaluation: the shared rule the monolithic BuildRiTensor runs
        // (internal/md_vrr_3c.hpp AccumulateScreenedRiPass), over the task
        // list the chunk actually evaluated - so a full-range chunk reports
        // the monolithic totals and a chunked caller sums its chunks'. The
        // aux entries and the tasks belong to exactly one chunk, but the
        // orbital pairs do not: they are counted on the caller's state
        // (ri_engine.hpp RiScreenedPassState), which spans the chunks of one
        // build - without it each chunk would re-count every pair it
        // survives with and the sum would be the chunk count times the
        // monolithic p3 (MEASURED: 555 against 15, 37 chunks).
        internal::AccumulateScreenedRiPass(
            *termCountersOut, combinedStore, nOrbitalPairs, computed, passState);
    }

    const std::size_t firstFunction = auxPairList->shells[auxShellStart].functionOffset;
    std::size_t offset = 0;

    for (const internal::RiTask& task : computed)
    {
        const ShellPairIndex& braIndex = pairList->pairs[task.braPair];
        const ShellInfo& shellI = pairList->shells[braIndex.i];
        const ShellInfo& shellJ = pairList->shells[braIndex.j];
        const std::size_t nI = ShellFunctionCount(shellI);
        const std::size_t nJ = ShellFunctionCount(shellJ);
        const std::size_t oI = shellI.functionOffset;
        const std::size_t oJ = shellJ.functionOffset;
        // The task's ket index is chunk-scoped: the local aux store slot
        // maps back to the global shell through the chunk's range.
        const ShellInfo& auxShell =
            auxPairList->shells[auxShellStart + task.auxShell - nOrbitalPairs];
        const std::size_t nP = ShellFunctionCount(auxShell);
        const std::size_t columnBase = auxShell.functionOffset - firstFunction;

        // The kernel returns the (u, v | P) block with u in shell i, v in
        // shell j; the integral is symmetric in (u, v), so the write fills
        // both triangles of the flat rows-u*n+v layout - the exact element
        // space of the monolithic tensor's Create-time flatten.
        for (std::size_t fa = 0; fa < nI; ++fa)
        {
            for (std::size_t fb = 0; fb < nJ; ++fb)
            {
                const std::size_t u = oI + fa;
                const std::size_t v = oJ + fb;

                for (std::size_t fP = 0; fP < nP; ++fP)
                {
                    const double value = values[offset + (fb * nI + fa) * nP + fP];
                    slice(static_cast<Eigen::Index>(u * n + v),
                          static_cast<Eigen::Index>(columnBase + fP)) = value;
                    slice(static_cast<Eigen::Index>(v * n + u),
                          static_cast<Eigen::Index>(columnBase + fP)) = value;
                }
            }
        }

        offset += nI * nJ * nP;
    }

    return slice;
}

} // namespace qcx::integrals
