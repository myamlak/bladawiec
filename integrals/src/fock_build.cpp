// The direct J/K Fock builder (fock_build.hpp): per iteration the
// screened canonical quartets are evaluated through the batched MD engines
// (fp64, with the certified fp32 lane for the quartets whose a-priori
// density-weighted bound fits the preset budget) and contracted with the
// density on the fly. J accumulates on the bra and ket pair blocks of every
// canonical quartet; K accumulates on the four exchange targets with the
// density permuted per block - integrals are never transposed.

#include "qcx/integrals/fock_build.hpp"

#include "internal/fixed_order_reduce.hpp"
#include "internal/fock_contract_kernel.hpp"
#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "internal/light_footprint.hpp"
#include "internal/md_attribution.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_defs.hpp"
#include "internal/md_engine.hpp"
#include "internal/md_vrr.hpp"
#include "internal/precision_ladder.hpp"
#include "internal/tensor_eigen_bridge.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/backend/memory_topology.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace qcx::integrals {

// The screening path shared with the GPU builder lives in
// internal/fock_screen.hpp: DensityBlockMax, the neighbor-list slack,
// ScreeningContext, ScreenOne/ScreenAll, BuildNeighborList and their
// helpers, so both builders evaluate identical screening decisions.

struct DirectJkFockBuilder::State {
    ShellPairList _pairList;
    std::vector<internal::MdPairData> _pairStore;
    std::vector<double> _schwarz;
    Eigen::MatrixXd _coreHamiltonian;
    // The construction-time H kept as the original Tensor (not just the
    // Eigen copy above): CoreHamiltonian() returns it by reference, so this
    // avoids a Tensor->Eigen conversion on every accessor call. Optional
    // because Tensor is move-only with no default constructor (which would
    // otherwise delete State's default constructor); always engaged after
    // Create() populates it.
    std::optional<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> _coreHamiltonianTensor;
    FockBuildOptions _options;
    // Precomputed once here: for each bra pair index, the sorted list of
    // ket <= bra indices whose Schwarz product clears the pair cutoff. The
    // options are fixed at Create() time, so the cutoff is fixed for this
    // instance's whole lifetime; BuildFock iterates this CSR pattern
    // instead of the full nPairs x nPairs Cartesian product - the same
    // Schwarz test, evaluated once instead of per call. The
    // SparsityPattern representation carries the wiring. Optional
    // because SparsityPattern is move-only with no default constructor
    // (which would otherwise delete State's default constructor); always
    // engaged after Create() populates it.
    std::optional<qcx::memory::SparsityPattern<qcx::backend::CpuTag>> _neighborPattern;
    // The class-aware contraction state - the symmetry
    // reduction and its pair-class table, engaged when the options carry a
    // non-trivial reduction (FockBuildOptions::symmetryReduction). Both
    // captured at Create() time: the builder is self-contained (the
    // options pointer need not outlive Create). Disengaged = the plain
    // path.
    std::optional<SymmetryReduction> _reduction;
    // The class table behind the shared_ptr so the on-demand orbit
    // generation (GenerateClassPairOrbits, called from const BuildFock)
    // can fill the screened class pairs' orbit expansions legally - the
    // EriBatchCache precedent ("the shared_ptr keeps the cache mutating
    // legally through const BuildFock").
    std::shared_ptr<PairClassTable> _classTable;
    // The in-memory ERI-value cache, engaged when the options
    // carry a positive maxCacheBytes AND the class path is off (the
    // reduced-element-set interplay is the documented follow-on). The
    // shared_ptr keeps the cache mutating legally through const BuildFock;
    // the cache holds the pair list and the engines capture the pair
    // store - both State members, stable addresses for the State's
    // lifetime, so the cache dies with the State (no dangling).
    std::shared_ptr<EriBatchCache> _eriCache;
    // The LightPath state: the
    // geometry-only light store (one MdPairData per pair, the
    // contracted transforms filled per chunk and cleared after it), the
    // flattened shells it builds from, and the chunk size in bra rows
    // (0 = the FastPath). The shared_ptr keeps the per-chunk mutation
    // legal through const BuildFock (the EriBatchCache precedent); the
    // FastPath leaves the store null and _pairStore full.
    std::shared_ptr<std::vector<internal::MdPairData>> _lightStore;
    std::vector<internal::MdShellInput> _lightShells;
    std::size_t _lightPathChunkPairs = 0;
    // The adaptive-memory Create-time mode record (fock_build.hpp
    // FockModeInfo): populated when the options carry a workspace budget;
    // nullopt on the legacy path (no decision was made - ModeInfo()).
    std::optional<FockModeInfo> _modeInfo;
    // The Create-time authorized concurrent batch slots of the
    // per-half batch loops (the k of the k-factor charge and the k-slot
    // loop) - > 1 only on the budgeted fast path with the exchange engaged
    // and the cache off; 1 on the legacy, cache-engaged, Coulomb-only and
    // LightPath rungs. Carried into the FockContractor at BuildFock so the
    // runtime loop never re-derives k from the live Remaining (the CWA is
    // tight - the charge authorized exactly this many slots' live mass).
    std::size_t _maxConcurrentSlots = 1;
    // The calibration term G (an RI term counter): the per-canonical-pair
    // primitive-pair count - the product of its two shells' primitive
    // counts, the number of (a, b) primitive pairs the pair's store entry
    // enumerates. Filled once at Create from the FastPath store's
    // primPairs (or the LightPath's shell exponents - the light store
    // keeps no primPairs); BuildFock's per-quartet G weight multiplies the
    // bra and ket counts. Mode-independent: the per-call stats accumulate
    // over either path's task lists with the same enumeration.
    std::vector<std::size_t> _pairPrimitivePairCounts;
};

namespace {

// The ParallelReduce element of the RunPass contract loop: the chunk-local
// Fock accumulation and its density-weighted certified
// bound partial, reduced together - one reduction, and never a second
// unprotected shared accumulator. weightMiss carries the certified-routing
// error out of the reduction (the one channel the accumulate lambda cannot
// reach directly).
struct RunPassPartial {
    Eigen::MatrixXd fock; // Chunk-local Fock accumulation.
    double certifiedBoundSum = 0.0; // Chunk-local density-weighted bound sum.
    bool weightMiss = false; // Certified-routing invariant violation.
    // Chunk-local per-element filter drop count (the
    // parallel reduction is order-sensitive for the Fock sum but the drop
    // count combines by exact addition, so the count pins are exact).
    std::size_t elementDrops = 0;
};

// One assembled class batch of a precision half (each batch is
// maxBatchBytes-bounded, so the largest live values allocation of a call
// is ONE batch's buffer, never the whole-call screened mass): the batch's
// computed quartet list, the packed ERI values (the fp64 array, or the
// certified fp32 lane), the per-quartet certified bounds (fp32 only), and
// the prefix-summed block offsets into the values arrays (batch-local,
// starting at 0 within the batch, per batch).
struct AssembledHalf {
    std::vector<ShellQuartet> computed;
    std::vector<double> values; ///< fp64-packed blocks; dead for an fp32 half
                                ///< (never resized or read there).
    std::vector<float> valuesF32;
    std::vector<double> bounds;
    std::vector<std::size_t> quartetOffsets;
};

// The ParallelReduce element of the class-pass contract loop: the
// chunk-local Fock accumulation, its certified-bound
// partial, and the routing-miss flags, reduced together. Each screened
// member quartet contracts in the precision its own routing gate sent it to
// (the same ScreenAll partition the plain path uses): fp64 members read the
// orbit rep's fp64 block, fp32 members the rep's certified fp32 block.
struct ClassPassPartial {
    Eigen::MatrixXd fock; // Chunk-local Fock accumulation.
    double certifiedBoundSum = 0.0; // Chunk-local density-weighted bound sum.
    bool miss = false; // Class-routing invariant violation (no rep block).
    bool certifiedMiss = false; // Certified-routing invariant violation (fp32 member, no weight).
    // Chunk-local per-element filter drop count (see
    // RunPassPartial - exact combine).
    std::size_t elementDrops = 0;
};

// The class-contraction routing invariant violation: the rep block of
// every class task is in the assembly by construction (a task exists only
// for orbits with screened members, and every orbit rep was assembled), so
// a miss can only mean a bookkeeping bug.
qcx::Result<void> ClassRoutingError() {
    return qcx::Result<void>{std::unexpected(
        qcx::Error{qcx::ErrorCode::kInternalError,
                   "class contraction: no assembled block for an orbit representative"})};
}

// The shared front of the cache's engine machinery: the miss
// quartets are canonicalized and grouped into class batches and the packed
// output size is counted (each task's block is the pair store's function
// product).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
qcx::Result<std::pair<std::vector<internal::MdClassBatch>, std::size_t>> AssembleCacheBatches(
    const std::vector<internal::MdPairData>& pairStore,
    const ShellPairList& pairList,
    const std::vector<ShellQuartet>& quartets,
    std::size_t maxBatchBytes,
    std::vector<ShellQuartet>& computed) {
    auto batches =
        internal::AssembleClassBatches(pairStore, pairList, quartets, maxBatchBytes, computed);

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

    return std::make_pair(std::move(*batches), total);
}

// Runs one engine request of the cache's batch machinery:
// AssembleCacheBatches, then the packed output buffer is laid out through
// \p layout - the per-lane difference: the result type, the buffer
// construction (resize for the fp64 lane; the pad-zeroing assign plus the
// error-bound array for the certified fp32 lane) and the per-batch output
// pointers - and the batches run. The two engine lambdas at the call site
// reduce to the captured State members and the layout.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
template <typename ResultT, typename LayoutFn>
qcx::Result<ResultT> RunCacheEngineBatches(const std::vector<internal::MdPairData>& pairStore,
                                           const ShellPairList& pairList,
                                           const std::vector<ShellQuartet>& quartets,
                                           std::size_t maxBatchBytes,
                                           std::vector<ShellQuartet>& computed,
                                           LayoutFn&& layout) {
    auto assembled = AssembleCacheBatches(pairStore, pairList, quartets, maxBatchBytes, computed);

    if (!assembled.has_value())
    {
        return std::unexpected(assembled.error());
    }

    ResultT result;
    result.computed = std::move(computed);
    layout(result, assembled->first, pairStore, assembled->second);

    auto run = internal::RunBatches(assembled->first);

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    return result;
}

// One class-path task: one orbit of member quartets with at
// least one screened member. The orbit representative's ERI block is
// computed once - the petite-list saving (one block per orbit instead of
// one per member quartet) - and expanded per screened member.
struct ClassRepTask {
    std::size_t classPairIndex; // Into the class table's classPairs.
    std::size_t orbitIndex; // Into that pair's orbits.
    std::vector<ClassOrbitMember> screenedMembers; // The orbit's screened members.
    // The screened members' positions in
    // the half's task master, parallel to screenedMembers - recorded only
    // for the fp32 call (recordMemberPositions), whose contraction reads
    // the member's density weight by direct index into the weight vector
    // that parallels the master (fp32DensityWeights[position], the
    // per-batch weight transport replacing the whole-call densityWeightOf
    // routing map). Empty for the fp64 call - the fp64 members never read
    // weights (their contract is weight-free).
    std::vector<std::size_t> screenedMemberPositions;
};

// The outcome of one class-task contract: the task contracted cleanly, the
// rep block is missing from the assembled half (a routing bookkeeping bug -
// ClassRoutingError), or a certified fp32 member has no recorded density
// weight (the class-path mirror of the plain path's CertifiedRoutingError).
enum class ClassTaskOutcome : std::uint8_t { kContracted, kRepMiss, kCertifiedMiss };

// The class-path task list: every orbit whose canonical member quartets
// include at least one of the screened quartets. The screened quartets are
// exactly the plain path's computed set, so the class path contracts the
// same (quartet, block) pairs - the rep blocks plus the per-member
// expansions (the class-vs-plain equivalence pin).
//
// The root restructure's on-demand generation: the Create-time table
// carries the pair classes and the Schwarz-screened class pairs with EMPTY
// orbit vectors. A class pair is reached here only when at least one of
// its canonical member quartets is in the screened set - the member-key
// pre-test below - and GenerateClassPairOrbits fills the pair's orbits the
// first time that happens (the deterministic pure function of the pair's
// class members; idempotent across the fp64/fp32 passes and the LightPath
// chunks). Class pairs whose members are all unscreened never generate
// anything: their orbit expansions are never materialized at all.
std::vector<ClassRepTask> BuildClassRepTasks(PairClassTable& classTable,
                                             const SymmetryReduction& reduction,
                                             const ShellPairList& pairList,
                                             const std::vector<internal::MdQuartetTask>& screened,
                                             std::size_t nPairs,
                                             bool recordMemberPositions = false) {
    std::unordered_set<std::size_t> screenedKeys;
    screenedKeys.reserve(screened.size());

    for (const internal::MdQuartetTask& task : screened)
    {
        screenedKeys.insert(task.braPair * nPairs + task.ketPair);
    }

    // The screened-member position
    // lookup for the fp32 call - one index per screened member so the
    // class contraction's weight read becomes a direct index into the
    // weight vector parallel to the master (fp32DensityWeights[position]).
    // The transient is freed when this function returns, before the class
    // pass runs. The call sites sort the master BEFORE building the class
    // tasks, so the recorded positions index the final (sorted) master and
    // stay valid for the whole call (the sort never runs again).
    std::unordered_map<std::size_t, std::size_t> positionOf;
    positionOf.reserve(recordMemberPositions ? screened.size() : 0);

    if (recordMemberPositions)
    {
        for (std::size_t t = 0; t < screened.size(); ++t)
        {
            positionOf[screened[t].braPair * nPairs + screened[t].ketPair] = t;
        }
    }

    std::vector<ClassRepTask> classTasks;

    for (std::size_t cp = 0; cp < classTable.classPairs.size(); ++cp)
    {
        ClassPair& classPair = classTable.classPairs[cp];

        // The member-key pre-test: does any canonical member quartet of
        // this class pair appear in the screened set? The pair's member
        // quartets are the canonical products of its two classes' members;
        // the screening key is the canonical bra >= ket ordering, the same
        // key the orbit members carry.
        bool reachable = false;
        const internal::TaggedVector<std::size_t>& membersP =
            classTable.classes[classPair.p].members;
        const internal::TaggedVector<std::size_t>& membersQ =
            classTable.classes[classPair.q].members;

        for (const std::size_t m : membersP)
        {
            for (const std::size_t n : membersQ)
            {
                const std::size_t bra = std::max(m, n);
                const std::size_t ket = std::min(m, n);

                if (screenedKeys.contains(bra * nPairs + ket))
                {
                    reachable = true;
                    break;
                }
            }

            if (reachable)
            {
                break;
            }
        }

        if (!reachable)
        {
            // No screened member - the pair contributes no task and its
            // orbit expansions are never materialized.
            continue;
        }

        GenerateClassPairOrbits(classTable, reduction, pairList, cp);

        for (std::size_t o = 0; o < classPair.orbits.size(); ++o)
        {
            const ClassOrbit& orbit = classPair.orbits[o];
            std::vector<ClassOrbitMember> screenedMembers;
            std::vector<std::size_t> memberPositions;

            for (const ClassOrbitMember& member : orbit.members)
            {
                const std::size_t key = member.braPair * nPairs + member.ketPair;

                if (screenedKeys.contains(key))
                {
                    screenedMembers.push_back(member);

                    if (recordMemberPositions)
                    {
                        // Every screened member is in positionOf (the
                        // position map and the key set index the same
                        // master), so the find cannot miss; the unchecked
                        // access is guarded by that construction.
                        memberPositions.push_back(positionOf.find(key)->second);
                    }
                }
            }

            if (!screenedMembers.empty())
            {
                if (recordMemberPositions)
                {
                    classTasks.push_back(ClassRepTask{
                        cp, o, std::move(screenedMembers), std::move(memberPositions)});
                } else
                {
                    classTasks.push_back(ClassRepTask{cp, o, std::move(screenedMembers)});
                }
            }
        }
    }

    return classTasks;
}

// The class-task sort: sorts one
// precision half's class task list IN PLACE into the canonical rep
// emission order - the five-key comparator of SortScreenedTasks over
// each task's orbit-rep canonical form - so the class pass's cursor walk
// over the aligned rep task list emits the eager builder's batch sequence
// for the same rep multiset (AssembleClassBatches canonicalizes its input;
// the pass assembles from the sorted master instead). Orbit reps are
// pair-canonical (their member quartets are canonical (max, min)
// products), and canonical forms are injective on orbit reps (distinct
// reps, distinct forms), so the master order is deterministic. The tasks'
// screened-member positions index the fp32 SCREENED master (the sorted
// master the call site built the tasks from), not this list - the sort
// cannot invalidate them (the positions ride inside the moved task
// structs).
void SortClassRepTasks(const PairClassTable& classTable,
                       const ShellPairList& pairList,
                       std::vector<ClassRepTask>& tasks) {
    if (tasks.size() < 2)
    {
        return;
    }

    std::sort(tasks.begin(), tasks.end(), [&](const ClassRepTask& a, const ClassRepTask& b) {
        const ClassOrbit& orbitA = classTable.classPairs[a.classPairIndex].orbits[a.orbitIndex];
        const ClassOrbit& orbitB = classTable.classPairs[b.classPairIndex].orbits[b.orbitIndex];
        const internal::MdCanonicalTaskForm formA =
            internal::CanonicalTaskFormOf(pairList, orbitA.repBraPair, orbitA.repKetPair);
        const internal::MdCanonicalTaskForm formB =
            internal::CanonicalTaskFormOf(pairList, orbitB.repBraPair, orbitB.repKetPair);

        if (formA.lKet != formB.lKet)
        {
            return formA.lKet < formB.lKet;
        }

        if (formA.lBra != formB.lBra)
        {
            return formA.lBra < formB.lBra;
        }

        if (formA.ketPair != formB.ketPair)
        {
            return formA.ketPair < formB.ketPair;
        }

        const ShellPairIndex& braShellsA = pairList.pairs[formA.braPair];
        const ShellPairIndex& braShellsB = pairList.pairs[formB.braPair];
        const std::size_t rowPairsA = pairList.shells[braShellsA.i].contractionCount *
                                      pairList.shells[braShellsA.j].contractionCount;
        const std::size_t rowPairsB = pairList.shells[braShellsB.i].contractionCount *
                                      pairList.shells[braShellsB.j].contractionCount;

        if (rowPairsA != rowPairsB)
        {
            return rowPairsA < rowPairsB;
        }

        return formA.braPair < formB.braPair;
    });
}

// Expands one member quartet's ERI block from the orbit representative's
// computed block: memberBlock[memberPos] = pattern[srcPos] * repBlock[srcPos]
// with memberPos = pi_g(srcPos) the position permutation induced by the
// member's generator (the image functions of the source position, scattered
// through the member's slot mapping) and pattern[srcPos] the outer product
// of the representative's four shells' per-position sign vectors under g -
// a position-dependent pattern product, NOT a scalar multiple of the rep
// block. The within-pair function sorts of the
// canonicalization are no-ops on the block array (the ERI is symmetric
// under the exchange of a pair's two functions); the whole-quartet role
// swap appears in two places: as the member slot permutation (the target
// assembly) AND as the rep block's flat layout - the assembled block
// follows the COMPUTED quartet's axis order (CanonicalizeQuartetOrder's
// final lBra <= lKet pair swap, a value-identical 8-fold partner), so the
// rep-slot source positions are read through the computed quartet's axes
// (sourceOfComputedAxis). For an orbit rep - already pair-index canonical,
// i <= j and bra >= ket - the canonicalization is identity or the pure pair
// swap, so the map is derived from the two forms; anything else is a
// bookkeeping bug (asserted). The scatter is bijective (g is a bijection),
// so every member position is written exactly once.
// T is the rep block's storage precision: double (the fp64 half) or float
// (the certified fp32 lane - the read widens exactly, so the member block
// carries the rep block's error, covered by the rep's per-quartet certified
// bound).
template <typename T>
void ExpandMemberBlock(const SymmetryReduction& reduction,
                       const ShellPairList& pairList,
                       const ClassOrbitMember& member,
                       const ShellQuartet& repQuartet,
                       const ShellQuartet& computedQuartet,
                       const T* repBlock,
                       std::vector<double>& memberBlock) {
    const std::size_t g = member.generator;
    const ShellInfo& compI = pairList.shells[computedQuartet.i];
    const ShellInfo& compJ = pairList.shells[computedQuartet.j];
    const ShellInfo& compK = pairList.shells[computedQuartet.k];
    const ShellInfo& compL = pairList.shells[computedQuartet.l];
    const std::size_t nComp[4] = {ShellFunctionCount(compI),
                                  ShellFunctionCount(compJ),
                                  ShellFunctionCount(compK),
                                  ShellFunctionCount(compL)};

    // The rep-slot -> computed-axis map: computed axis c receives the
    // position of the rep slot it corresponds to. Identity when the
    // assembler kept the rep's pair order, the pair swap otherwise.
    std::array<std::size_t, 4> sourceOfComputedAxis{0, 1, 2, 3};

    if (computedQuartet.i != repQuartet.i || computedQuartet.j != repQuartet.j ||
        computedQuartet.k != repQuartet.k || computedQuartet.l != repQuartet.l)
    {
        assert(computedQuartet.i == repQuartet.k && computedQuartet.j == repQuartet.l &&
               computedQuartet.k == repQuartet.i && computedQuartet.l == repQuartet.j);
        sourceOfComputedAxis = {2, 3, 0, 1};
    }

    // The source slots follow the REP quartet's shell order: rep slot r's
    // shell is the computed quartet's shell on the axis the slot feeds (the
    // inverse of sourceOfComputedAxis). The loop bounds and the source
    // function offsets below come from these - the source positions are
    // indices within the rep quartet's shells.
    std::array<std::size_t, 4> computedAxisOfRepSlot{};

    for (std::size_t c = 0; c < 4; ++c)
    {
        computedAxisOfRepSlot[sourceOfComputedAxis[c]] = c;
    }

    const std::array<std::size_t, 4> computedShellIndex{
        computedQuartet.i, computedQuartet.j, computedQuartet.k, computedQuartet.l};
    const ShellInfo& braI = pairList.shells[computedShellIndex[computedAxisOfRepSlot[0]]];
    const ShellInfo& braJ = pairList.shells[computedShellIndex[computedAxisOfRepSlot[1]]];
    const ShellInfo& ketK = pairList.shells[computedShellIndex[computedAxisOfRepSlot[2]]];
    const ShellInfo& ketL = pairList.shells[computedShellIndex[computedAxisOfRepSlot[3]]];
    const std::size_t nA = ShellFunctionCount(braI);
    const std::size_t nB = ShellFunctionCount(braJ);
    const std::size_t nC = ShellFunctionCount(ketK);
    const std::size_t nD = ShellFunctionCount(ketL);

    // The member quartet's shell function offsets and counts per member
    // slot (0, 1 = the bra pair's shells, 2, 3 = the ket pair's): the
    // position of an image function within its target slot is its index
    // minus the slot's offset.
    const ShellPairIndex& memberBra = pairList.pairs[member.braPair];
    const ShellPairIndex& memberKet = pairList.pairs[member.ketPair];
    const std::array<std::size_t, 4> memberOffsets{pairList.shells[memberBra.i].functionOffset,
                                                   pairList.shells[memberBra.j].functionOffset,
                                                   pairList.shells[memberKet.i].functionOffset,
                                                   pairList.shells[memberKet.j].functionOffset};
    const std::array<std::size_t, 4> memberCounts{ShellFunctionCount(pairList.shells[memberBra.i]),
                                                  ShellFunctionCount(pairList.shells[memberBra.j]),
                                                  ShellFunctionCount(pairList.shells[memberKet.i]),
                                                  ShellFunctionCount(pairList.shells[memberKet.j])};

    // The source slot of each member slot (the inverse of slotPerm): the
    // target tuple below is ordered by member slot, so the axis gets the
    // position of the source slot the member slot receives.
    std::array<std::size_t, 4> sourceOfSlot{};

    for (std::size_t s = 0; s < 4; ++s)
    {
        sourceOfSlot[member.slotPerm[s]] = s;
    }

    // The member block carries the member quartet's shell structure. The
    // expansion's bijection contract: the image shells of the four rep
    // slots carry the same function counts as their sources (the realized
    // groups map shells to shells with position preservation), so every
    // member position is written exactly once.
    assert(memberCounts[0] * memberCounts[1] * memberCounts[2] * memberCounts[3] ==
           nComp[0] * nComp[1] * nComp[2] * nComp[3]);
    memberBlock.resize(memberCounts[0] * memberCounts[1] * memberCounts[2] * memberCounts[3]);

    for (std::size_t a = 0; a < nA; ++a)
    {
        const std::size_t sourceA = braI.functionOffset + a;
        const std::size_t positionA =
            reduction.permutation[g][sourceA] - memberOffsets[member.slotPerm[0]];
        const int signA = reduction.sign[g][sourceA];

        for (std::size_t b = 0; b < nB; ++b)
        {
            const std::size_t sourceB = braJ.functionOffset + b;
            const std::size_t positionB =
                reduction.permutation[g][sourceB] - memberOffsets[member.slotPerm[1]];
            const int signB = reduction.sign[g][sourceB];

            for (std::size_t c = 0; c < nC; ++c)
            {
                const std::size_t sourceC = ketK.functionOffset + c;
                const std::size_t positionC =
                    reduction.permutation[g][sourceC] - memberOffsets[member.slotPerm[2]];
                const int signC = reduction.sign[g][sourceC];

                for (std::size_t d = 0; d < nD; ++d)
                {
                    const std::size_t sourceD = ketL.functionOffset + d;
                    const std::size_t positionD =
                        reduction.permutation[g][sourceD] - memberOffsets[member.slotPerm[3]];
                    const int signD = reduction.sign[g][sourceD];

                    const std::array<std::size_t, 4> positions{
                        positionA, positionB, positionC, positionD};
                    const std::size_t target[4] = {positions[sourceOfSlot[0]],
                                                   positions[sourceOfSlot[1]],
                                                   positions[sourceOfSlot[2]],
                                                   positions[sourceOfSlot[3]]};

                    memberBlock[EriBlockIndex(target[0],
                                              target[1],
                                              target[2],
                                              target[3],
                                              memberCounts[0],
                                              memberCounts[1],
                                              memberCounts[2],
                                              memberCounts[3])] =
                        static_cast<double>(signA * signB * signC * signD) *
                        static_cast<double>(
                            repBlock[EriBlockIndex(positions[sourceOfComputedAxis[0]],
                                                   positions[sourceOfComputedAxis[1]],
                                                   positions[sourceOfComputedAxis[2]],
                                                   positions[sourceOfComputedAxis[3]],
                                                   nComp[0],
                                                   nComp[1],
                                                   nComp[2],
                                                   nComp[3])]);
                }
            }
        }
    }
}

struct FockContractor;

// Forward declaration of the shared per-quartet contract body: its
// definition follows FockContractor (it calls AccumulateBlock), but
// RunPass's call sites precede it - the declaration keeps the calls legal.
qcx::Result<void> ContractOne(const FockContractor& contractor,
                              const std::vector<ShellQuartet>& computed,
                              const std::vector<double>& values,
                              const std::vector<float>& valuesF32,
                              const std::vector<double>& bounds,
                              std::size_t t,
                              const std::vector<std::size_t>& quartetOffsets,
                              bool fp32,
                              const std::vector<double>& fp32Weights,
                              std::size_t weightBase,
                              std::vector<double>& blockScratch,
                              double& certifiedBoundSum);

// The batch-order
// merge gate. The per-batch deltas of the k concurrent slots enter the
// shared Fock in CANONICAL BATCH-MAJOR ORDER - the k=1 bit-identity
// contract - by serializing the merges through this chain: the holder of
// batch b locks, waits until accumulated == b (every earlier batch's turn
// is done), merges iff no error was recorded, then ALWAYS advances and
// notifies. Error discipline: the first failing holder records the error
// under the lock and still takes its turn (advances without merging), so
// every holder advances exactly once - no deadlock; later holders see the
// error and skip their merges. The slot loops' runHalf returns the
// recorded error after the join (a failed call's partial merges are
// discarded with the call). Null gate = today's un-serialized code path
// (the k=1 loop and the cache branch pass nothing).
struct BatchMergeGate {
    std::mutex mutex; ///< Serializes the turn bookkeeping and the merges.
    std::condition_variable cv; ///< Wakes the holder whose turn arrived.
    std::size_t accumulated = 0; ///< Turns taken so far (batches 0..n-1).
    std::optional<qcx::Error> error; ///< The first failure (any slot), recorded once.

    // The error path of a slot that failed BEFORE its merge point (an
    // assembly error): records the failure (first one wins) and then takes
    // its turn (advances without merging) so the chain never stalls.
    void RecordAndAdvance(std::size_t position, const qcx::Error& e) {
        std::unique_lock<std::mutex> lock(mutex);

        if (!error.has_value())
        {
            error = e;
        }

        cv.wait(lock, [&] { return accumulated == position; });
        ++accumulated;
        cv.notify_all();
    }
};

struct FockContractor {
    const ShellPairList& _pairList;
    const std::vector<internal::MdPairData>& _pairStore;
    const FockBuildOptions& _options;
    const Eigen::MatrixXd& _density;
    Eigen::MatrixXd& _fock;
    // The in-memory ERI-value cache, or nullptr for the plain
    // assembly. Non-const on purpose: the cache is mutable state (its arenas
    // and counters move on every miss), mutated through the const BuildFock
    // via the State's shared_ptr (its lifetime is the State's). The class
    // path contracts with a contractor built WITHOUT the cache (the State's
    // cache is null there by construction). The per-chunk contraction-only
    // contractors default to null - assembly happens before contraction.
    EriBatchCache* _eriCache;

    // The per-call shell-compressed max-density vector (raw max
    // |D_block| per canonical pair), or nullptr when the per-element
    // screening flag is off - the filter never engages (the flag contract,
    // flag contract: false reproduces exactly today's behavior). The vector's
    // lifetime is the BuildFock call that built it - never cached across
    // calls (the incremental wrapper alternates full D and Delta-D).
    const std::vector<double>* _pairMaxDensity = nullptr;
    // The drop-counter target the kernel accumulates into
    // (BuildFock's local on the serial paths, the per-chunk partials' local
    // on the parallel paths). Null = no counting.
    std::size_t* _elementDrops = nullptr;
    // The Create-time authorized concurrent batch slots (the
    // State's _maxConcurrentSlots, modeInfo->concurrentSlots on the budgeted
    // fast path; 1 on the legacy, cache-engaged, Coulomb-only and LightPath
    // rungs). The batch loops (RunPass/RunClassPass runHalf) fire this many
    // pullers when the half has more batches than one - the runtime never
    // re-derives k from the live Remaining (the CWA is tight: the charge
    // authorized exactly this many slots' live mass).
    std::size_t _maxConcurrentSlots = 1;

    FockContractor(const ShellPairList& pairList,
                   const std::vector<internal::MdPairData>& pairStore,
                   const FockBuildOptions& options,
                   const Eigen::MatrixXd& density,
                   Eigen::MatrixXd& fock,
                   EriBatchCache* eriCache = nullptr,
                   const std::vector<double>* pairMaxDensity = nullptr,
                   std::size_t* elementDrops = nullptr,
                   std::size_t maxConcurrentSlots = 1) :
        _pairList(pairList), _pairStore(pairStore), _options(options), _density(density),
        _fock(fock), _eriCache(eriCache), _pairMaxDensity(pairMaxDensity),
        _elementDrops(elementDrops), _maxConcurrentSlots(maxConcurrentSlots) {}

    // Contracts one computed quartet block into the Fock matrix (either
    // precision; the fp32 lane casts per element - covered by the
    // certified bound). The kernel body lives in
    // internal/fock_contract_kernel.hpp - ONE template compiled twice: the
    // scalar copy instantiates here (this TU carries no /arch flag) and an
    // /arch:AVX2 copy in fock_contract_simd.cpp, entered only after the
    // runtime cpuid check (FockAvx2Available - the boys_simd.cpp dispatch
    // shape). forceScalarContract pins the scalar copy even on
    // an AVX2-capable machine (the fallback test pin); both copies must be
    // bit-identical - the A/B tests in fock_build_test.cpp check that.
    void AccumulateBlock(const double* block,
                         const ShellQuartet& quartet,
                         std::size_t pairBra,
                         std::size_t pairKet) const {
        // The per-element filter rides on the pair-max vector
        // (null = filter off, the flag contract) and the per-preset density
        // threshold; the drop counter target is this contractor's - the
        // serial paths point at the BuildFock local, the per-chunk
        // contractors at their partial's local.
        const internal::FockContractContext context{_pairList,
                                                    _density,
                                                    _fock,
                                                    _options.buildExchangeOnly,
                                                    _options.buildCoulombOnly,
                                                    DensityThreshold(_options.accuracy),
                                                    _pairMaxDensity,
                                                    _elementDrops};
        static const bool kAvx2 = internal::FockAvx2Available();

        if (kAvx2 && !_options.forceScalarContract)
        {
            internal::AccumulateBlockAvx2(block, quartet, pairBra, pairKet, context);
        } else
        {
            internal::AccumulateBlockKernel<internal::AccumulateBlockScalarTag>(
                block, quartet, pairBra, pairKet, context);
        }
    }

    // Assembles ONE class batch into batch-scoped buffers - the
    // bound: every batch is maxBatchBytes-bounded, so the largest live
    // values allocation of a call is the single batch's buffer, never
    // the whole-call screened mass (the death-allocation class is
    // gone by construction). The batch's tasks are already in canonical
    // order, so the batch's computed list is built directly from them;
    // the output routing points the batch's outF64/outF32 at the
    // batch-scoped buffer (the tasks' outputOffset is batch-relative, so
    // it starts at 0) and the certified bound slots are batch-local
    // (boundsBase reset to 0 - the kernel writes errorBounds[boundsBase
    // + t]).
    qcx::Result<AssembledHalf> AssembleBatch(internal::MdClassBatch batch,
                                             bool fp32,
                                             std::size_t regionThreads = 0) const {
        AssembledHalf half;
        half.computed.reserve(batch.tasks.size());
        std::size_t total = 0;

        for (const internal::MdQuartetTask& task : batch.tasks)
        {
            const ShellPairIndex& braPair = _pairList.pairs[task.braPair];
            const ShellPairIndex& ketPair = _pairList.pairs[task.ketPair];
            half.computed.push_back(ShellQuartet{braPair.i, braPair.j, ketPair.i, ketPair.j});
            total += _pairStore[task.braPair].nFuncs * _pairStore[task.ketPair].nFuncs;
        }

        // RunBatches over a one-element batch list runs the kernel on the
        // single batch - the ConsumeAssembledBatches discipline (the
        // in-memory batch is the whole request here; the per-call batch
        // fan-out that gave the old single fork/join its parallelism was
        // only ever a memory-bound trick).
        std::vector<internal::MdClassBatch> single;
        single.reserve(1);
        single.push_back(std::move(batch));
        internal::MdClassBatch& run = single[0];

        if (fp32)
        {
            // The certified lane's fp64 vector is dead by construction:
            // no reader touches half.values for an fp32 half (the cache
            // move, the outF64 pointer, the class-pass read, and
            // ContractOne's values parameter - :1402, read at :1451 in
            // the fp64 else-branch; the fp32 branch reads valuesF32 at
            // :1425 - are all fp64-branch-only). Skipping the resize drops the per-call 8*T
            // allocation bit-identically.
            // assign (not resize), like values/bounds: the fp32 lanes
            // are read back in full below, including any pads the batch
            // machinery writes only partially - zero-init so stale data
            // never leaks through a pad.
            half.valuesF32.assign(total, 0.f);
            half.bounds.assign(half.computed.size(), 0.0);
            run.outF32 = half.valuesF32.data();
            run.errorBounds = half.bounds.data();
            run.boundsBase = 0;
        } else
        {
            // resize value-initializes: the fp64 lane is read back in
            // full below, including pads - zero-init so stale data never
            // leaks through a pad.
            half.values.resize(total);
            run.outF64 = half.values.data();
        }

        // The region's explicit thread count (q = max(1, team/k) per
        // slot at k > 1, 0 = team at k = 1) - the bounded-concurrency seam
        // that keeps the k-slot x q-thread product at or under the team.
        auto runBatches = internal::RunBatches(single, regionThreads);

        if (!runBatches.has_value())
        {
            return qcx::Result<AssembledHalf>{std::unexpected(runBatches.error())};
        }

        BuildQuartetOffsets(half);
        return half;
    }

    // Precomputes each computed quartet's starting offset into values/
    // valuesF32 - a prefix sum over blockSize, so the contract loops can
    // index each quartet independently instead of carrying a sequential
    // running offset, which is exactly what lets them run across
    // threads. This pass is O(computed.size()) with trivial per-element
    // cost - negligible next to the contraction work it unblocks - and
    // MUST stay serial (it is a genuine prefix sum). The offsets are
    // batch-local (they start at 0 within the batch).
    void BuildQuartetOffsets(AssembledHalf& half) const {
        half.quartetOffsets.resize(half.computed.size());
        std::size_t runningOffset = 0;

        for (std::size_t t = 0; t < half.computed.size(); ++t)
        {
            half.quartetOffsets[t] = runningOffset;
            const ShellQuartet& quartet = half.computed[t];
            const std::size_t nI = ShellFunctionCount(_pairList.shells[quartet.i]);
            const std::size_t nJ = ShellFunctionCount(_pairList.shells[quartet.j]);
            const std::size_t nK = ShellFunctionCount(_pairList.shells[quartet.k]);
            const std::size_t nL = ShellFunctionCount(_pairList.shells[quartet.l]);
            runningOffset += nI * nJ * nK * nL;
        }
    }

    // Contracts ONE assembled batch's computed quartets into the Fock
    // matrix - the per-batch analog of the pre-split merged contract:
    // the batch's own index space chunked across threads
    // with the RunPass discipline (per-chunk local Fock matrix, fresh
    // contractor, per-chunk scratch, 8-double-aligned interior
    // boundaries; the combine adds the chunk partials in schedule order,
    // within the documented ParallelReduce last-bits tolerance). The
    // serial fallback accumulates in the batch's canonical order -
    // batch-major across the call, bit-identical to the pre-split serial
    // order. The buffers are discarded by the caller before the next
    // batch (the per-call values mass stays bounded at the batch cap).
    // \p gate/\p gatePosition route the merge through the
    // batch-order chain when concurrent slots run (the batch's delta
    // enters the shared Fock at its canonical turn); a null gate executes
    // today's merge exactly (the k=1 loop and the cache branch).
    // \p regionThreads bounds this batch's contract region (q =
    // max(1, team/k) per slot at k > 1; 0 = team at k = 1). \p fp32Weights
    // is the fp32 lane's master-parallel density weights (the
    // positional transport) and \p weightBase the batch's first master
    // position - the certified lane's per-quartet weight index
    // (ContractOne; the fp64 half never touches either). \p slotStats
    // collects the merge-chain timing at the chunked shared-Fock merges
    // (FockBuildStats::mergeWallTime): the per-slot partial at k > 1, the
    // shared statsOut at k = 1; null keeps the zero-cost path.
    qcx::Result<void> ContractBatch(const AssembledHalf& half,
                                    bool fp32,
                                    const std::vector<double>& fp32Weights,
                                    std::size_t weightBase,
                                    std::vector<double>& blockScratch,
                                    double& certifiedBoundSum,
                                    BatchMergeGate* gate = nullptr,
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    std::size_t gatePosition = 0,
                                    std::size_t regionThreads = 0,
                                    FockBuildStats* slotStats = nullptr) const {
        const std::size_t numQuartets = half.computed.size();
        const std::size_t numChunks = internal::ChunkCountFor(numQuartets, _options);

        // The per-index dispatch reads the batch's own arrays (the
        // per-batch offsets start at 0 within the batch). ContractOne
        // touches only the arrays of its own precision.
        auto contractIndex = [&](const FockContractor& contractor,
                                 std::size_t t,
                                 std::vector<double>& scratch,
                                 double& boundSum) -> qcx::Result<void> {
            return ContractOne(contractor,
                               half.computed,
                               half.values,
                               half.valuesF32,
                               half.bounds,
                               t,
                               half.quartetOffsets,
                               fp32,
                               fp32Weights,
                               weightBase,
                               scratch,
                               boundSum);
        };

        if (numChunks <= 1)
        {
            if (gate == nullptr)
            {
                // The k=1 serial fold: today's per-batch canonical-order
                // accumulation, bit-identical batch-major association.
                for (std::size_t t = 0; t < numQuartets; ++t)
                {
                    auto contracted = contractIndex(*this, t, blockScratch, certifiedBoundSum);

                    if (!contracted.has_value())
                    {
                        return contracted;
                    }
                }
            } else
            {
                // At k > 1 the fold is this slot's batch: it runs at the
                // batch's turn, under the gate's lock, so the shared Fock
                // sees the batch's index-order accumulation in canonical
                // batch-major order exactly as at k = 1. A fold error is
                // recorded (the first one wins) and the turn still
                // advances - every holder takes its turn exactly once.
                std::unique_lock<std::mutex> lock(gate->mutex);
                gate->cv.wait(lock, [&] { return gate->accumulated == gatePosition; });

                if (!gate->error.has_value())
                {
                    for (std::size_t t = 0; t < numQuartets; ++t)
                    {
                        auto contracted = contractIndex(*this, t, blockScratch, certifiedBoundSum);

                        if (!contracted.has_value())
                        {
                            gate->error = contracted.error();
                            break;
                        }
                    }
                }

                ++gate->accumulated;
                gate->cv.notify_all();
            }
        } else
        {
            std::vector<std::size_t> chunkStarts(numChunks + 1);
            chunkStarts[0] = 0;
            chunkStarts[numChunks] = numQuartets;

            for (std::size_t c = 1; c < numChunks; ++c)
            {
                // Round interior boundaries down to the nearest
                // 8-double (64 B) multiple so adjacent chunks' per-quartet
                // bounds[]/boundsF32[] writes never share a cache line.
                // Endpoints stay exact; degenerate empty chunks on tiny
                // batches are harmless (the per-chunk loop handles them).
                const std::size_t raw = (numQuartets * c) / numChunks;
                chunkStarts[c] = (raw / 8) * 8;
            }

            const RunPassPartial zero{Eigen::MatrixXd::Zero(
                static_cast<Eigen::Index>(_fock.rows()), static_cast<Eigen::Index>(_fock.cols()))};

            RunPassPartial delta = internal::FixedOrderReduce<RunPassPartial>(
                numChunks,
                zero,
                [&](const RunPassPartial& partial, std::size_t chunkIndex) -> RunPassPartial {
                    RunPassPartial local = partial;
                    // Per-chunk scratch: the shared blockScratch parameter
                    // would be a data race across concurrently running
                    // chunks (the fp32 cast-back writes it). Correctness
                    // first - the per-chunk reallocation is a follow-up
                    // optimization if profiling ever shows it matters.
                    std::vector<double> localBlockScratch;
                    // The plan's per-chunk FRESH contractor, bound to the
                    // chunk's own matrix: AccumulateBlock writes through
                    // _fock, so a contractor bound to the shared builder
                    // matrix would race the accumulation. The chunk-local
                    // drop counter rides the same partial.
                    FockContractor chunkContractor{_pairList,
                                                   _pairStore,
                                                   _options,
                                                   _density,
                                                   local.fock,
                                                   nullptr,
                                                   _pairMaxDensity,
                                                   &local.elementDrops};

                    for (std::size_t t = chunkStarts[chunkIndex]; t < chunkStarts[chunkIndex + 1];
                         ++t)
                    {
                        auto contracted = contractIndex(
                            chunkContractor, t, localBlockScratch, local.certifiedBoundSum);

                        if (!contracted.has_value())
                        {
                            // No error channel out of the reduction - flag
                            // the routing-invariant violation; the caller
                            // converts after the combine.
                            local.weightMiss = true;
                            break;
                        }
                    }

                    return local;
                },
                [](const RunPassPartial& a, const RunPassPartial& b) -> RunPassPartial {
                    return RunPassPartial{a.fock + b.fock,
                                          a.certifiedBoundSum + b.certifiedBoundSum,
                                          a.weightMiss || b.weightMiss,
                                          a.elementDrops + b.elementDrops};
                },
                static_cast<int>(regionThreads));

            if (delta.weightMiss)
            {
                if (gate != nullptr)
                {
                    gate->RecordAndAdvance(gatePosition, internal::CertifiedRoutingError().error());
                }

                return internal::CertifiedRoutingError();
            }

            if (gate == nullptr)
            {
                const auto mergeStarted = slotStats != nullptr
                                              ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};
                _fock += delta.fock;

                if (slotStats != nullptr)
                {
                    // The k = 1 merge-chain accumulation (the batch's delta
                    // enters the shared Fock in canonical batch order) -
                    // FockBuildStats::mergeWallTime, timed per batch.
                    slotStats->mergeWallTime +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - mergeStarted);
                }

                certifiedBoundSum += delta.certifiedBoundSum;

                // The parallel path's drop count combines exactly
                // (the sum over chunks is order-independent); the serial path
                // accumulated directly into the same target through *this.
                if (_elementDrops != nullptr)
                {
                    *_elementDrops += delta.elementDrops;
                }
            } else
            {
                // The delta merges at the batch's turn, in canonical
                // batch-major order (the chain discipline of the gate).
                std::unique_lock<std::mutex> lock(gate->mutex);
                gate->cv.wait(lock, [&] { return gate->accumulated == gatePosition; });

                if (!gate->error.has_value())
                {
                    // The gate's lock is held across the merge, so the
                    // k > 1 merges are strictly serialized: the per-slot
                    // mergeWallTime sums (merged after the join) equal the
                    // call's merge-chain span.
                    const auto mergeStarted = slotStats != nullptr
                                                  ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
                    _fock += delta.fock;

                    if (slotStats != nullptr)
                    {
                        slotStats->mergeWallTime +=
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - mergeStarted);
                    }

                    certifiedBoundSum += delta.certifiedBoundSum;

                    if (_elementDrops != nullptr)
                    {
                        *_elementDrops += delta.elementDrops;
                    }
                }

                ++gate->accumulated;
                gate->cv.notify_all();
            }
        }

        return qcx::Result<void>{};
    }

    // Runs one pass over BOTH precisions' task lists, one
    // maxBatchBytes-bounded batch at a time per half: the fp64 half and
    // the certified fp32 lane (read back
    // through the per-quartet bounds, accumulating the density-weighted
    // certified sum) each assemble and contract their batches in
    // sequence - the per-call values mass is bounded at the batch cap,
    // NEVER the whole-call screened mass (the death-allocation
    // class is gone by construction). The pre-split merged single
    // fork/join becomes per-half, per-batch parallel
    // regions; the serial fallback contracts in batch-major order, which
    // IS the pre-split whole-call canonical order (each batch's tasks are
    // a contiguous slice of the canonical items -
    // CanonicalizeQuartetOrder's sort key is total - so the batch's
    // computed list is the full canonical order restricted to the
    // batch): bit-identical serial accumulation. The parallel path's
    // thread-to-quartet assignment changes with the per-batch regions -
    // within the documented ParallelReduce last-bits tolerance.
    qcx::Result<void> RunPass(const std::vector<internal::MdQuartetTask>& fp64Tasks,
                              const std::vector<internal::MdQuartetTask>& fp32Tasks,
                              const std::vector<double>& fp32DensityWeights,
                              std::vector<double>& blockScratch,
                              double& certifiedBoundSum,
                              FockBuildStats* statsOut = nullptr) const {
        if (fp64Tasks.empty() && fp32Tasks.empty())
        {
            return qcx::Result<void>{};
        }

        const bool wantStats = statsOut != nullptr;

        // One precision half: assemble and contract its batches (or, with
        // the ERI cache engaged, its canonical-order slices) one at a
        // time, discarding each batch's buffers before the next.
        auto runHalf = [&](const std::vector<internal::MdQuartetTask>& tasks,
                           bool fp32) -> qcx::Result<void> {
            // An empty half is a no-op half (no tasks to assemble or
            // contract): the assembler rejects an empty quartet list, so
            // skip it entirely.
            if (tasks.empty())
            {
                return qcx::Result<void>{};
            }

            if (_eriCache != nullptr)
            {
                // The cache decorator over the sorted master: it materializes
                // its request's
                // packed values internally (ComputeEriBatch[Certified]
                // resizes the whole request), so the request is split
                // into maxBatchBytes-bounded canonical-order slices and
                // each slice is assembled, contracted, and discarded
                // before the next. The master positions ARE the eager
                // canonicalizer's output (the canonical sort key is
                // CanonicalizeQuartetOrder's total order over each
                // task's canonical form), so each slice's canonical order
                // is the full order restricted to the slice and the
                // serial contraction stays in the whole-call canonical
                // order - bit-identical to the pre-split cached path. The
                // slice cap counts the lane's OWN request bytes: the
                // fp64 values mass
                // (blockSize * 8 bytes) on the fp64 lane, the certified
                // request's fp32 values plus its per-quartet bounds
                // (blockSize * 4 + 8 bytes) on the fp32 lane - the old
                // fp64-mass cap let a certified s-shell slice reach about
                // 1.5x maxBatchBytes (12 B actual vs 8 B counted per
                // quartet), the one documented overshoot of the fp32
                // lane's per-batch bound; the certified slices now sit at
                // or under the cap like the fp64 half's, so F32Live =
                // maxBatchBytes holds on this path too.
                std::vector<std::size_t> sliceStarts{0};
                std::size_t sliceBytes = 0;

                for (std::size_t pos = 0; pos < tasks.size(); ++pos)
                {
                    const internal::MdCanonicalTaskForm form = internal::CanonicalTaskFormOf(
                        _pairList, tasks[pos].braPair, tasks[pos].ketPair);
                    // The canonical quartet's block size: the canonical
                    // pair functions' product (the same shells the eager
                    // canonicalizer's per-quartet count summed).
                    const std::size_t blockSize =
                        _pairStore[form.braPair].nFuncs * _pairStore[form.ketPair].nFuncs;
                    // The lane's per-quartet request bytes: the packed
                    // values at the lane's element size, plus the
                    // certified lane's 8 B per-quartet bound (fp64
                    // requests carry no bounds).
                    const std::size_t itemBytes = fp32 ? blockSize * sizeof(float) + sizeof(double)
                                                       : blockSize * sizeof(double);

                    if (sliceBytes + itemBytes > _options.maxBatchBytes && sliceStarts.back() < pos)
                    {
                        sliceStarts.push_back(pos);
                        sliceBytes = 0;
                    }

                    sliceBytes += itemBytes;
                }

                sliceStarts.push_back(tasks.size());

                for (std::size_t s = 0; s + 1 < sliceStarts.size(); ++s)
                {
                    // The slice's canonical quartets: each position's
                    // canonical-role shells (the decorator's internal
                    // canonicalization is the identity on them).
                    std::vector<ShellQuartet> sliceQuartets;
                    sliceQuartets.reserve(sliceStarts[s + 1] - sliceStarts[s]);

                    for (std::size_t pos = sliceStarts[s]; pos < sliceStarts[s + 1]; ++pos)
                    {
                        const internal::MdCanonicalTaskForm form = internal::CanonicalTaskFormOf(
                            _pairList, tasks[pos].braPair, tasks[pos].ketPair);
                        const ShellPairIndex& bra = _pairList.pairs[form.braPair];
                        const ShellPairIndex& ket = _pairList.pairs[form.ketPair];
                        sliceQuartets.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
                    }

                    const auto eriSliceStarted = wantStats
                                                     ? std::chrono::steady_clock::now()
                                                     : std::chrono::steady_clock::time_point{};
                    // The per-slice hit delta: each lane's counters move
                    // only on its own assembly, so the delta is the lane's
                    // hits (the accumulated per-call totals are unchanged
                    // by the slicing).
                    const EriCacheStats statsBefore = _eriCache->Stats();
                    AssembledHalf half;

                    if (fp32)
                    {
                        auto batch = _eriCache->ComputeEriBatchCertified(
                            sliceQuartets, {_options.accuracy, _options.maxBatchBytes});

                        if (!batch.has_value())
                        {
                            return qcx::Result<void>{std::unexpected(batch.error())};
                        }

                        half.computed = std::move(batch->computed);
                        half.valuesF32 = std::move(batch->values);
                        half.bounds = std::move(batch->errorBounds);
                    } else
                    {
                        auto batch = _eriCache->ComputeEriBatch(
                            sliceQuartets, {_options.accuracy, _options.maxBatchBytes});

                        if (!batch.has_value())
                        {
                            return qcx::Result<void>{std::unexpected(batch.error())};
                        }

                        half.computed = std::move(batch->computed);
                        half.values = std::move(batch->values);
                    }

                    if (wantStats)
                    {
                        const EriCacheStats statsAfter = _eriCache->Stats();

                        if (fp32)
                        {
                            statsOut->cacheHitFp32QuartetCount +=
                                statsAfter.fp32HitQuartets - statsBefore.fp32HitQuartets;
                        } else
                        {
                            statsOut->cacheHitFp64QuartetCount +=
                                statsAfter.fp64HitQuartets - statsBefore.fp64HitQuartets;
                        }

                        // Accumulated, not assigned (the per-batch note in
                        // the plain branch below).
                        statsOut->eriWallTime +=
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - eriSliceStarted);
                    }

                    BuildQuartetOffsets(half);
                    // The slice's first master position is its weight
                    // transport base (the assembled task at index t IS
                    // master position sliceStarts[s] + t; the fp64 half
                    // never touches the weights).
                    auto contracted = ContractBatch(half,
                                                    fp32,
                                                    fp32DensityWeights,
                                                    sliceStarts[s],
                                                    blockScratch,
                                                    certifiedBoundSum,
                                                    nullptr,
                                                    0,
                                                    0,
                                                    statsOut);

                    if (!contracted.has_value())
                    {
                        return contracted;
                    }
                }

                return qcx::Result<void>{};
            }

            // The plain assembly walks the sorted master with the
            // incremental partitioner: MdClassBatchCursor emits the
            // class batches ONE at a time with AssembleClassBatches'
            // exact run/cut rules over the same per-task payload metrics,
            // so the emitted batch sequence is byte-identical to the
            // eager builder's for the same raw-task multiset (the
            // per-half pin). Each emitted batch is assembled and
            // contracted immediately - today's per-batch unit unchanged.
            internal::MdClassBatchCursor cursor(
                _pairStore, _pairList, tasks, _options.maxBatchBytes);

            // The batch-count pass and the per-batch slice table exist
            // only when the k > 1 slots can run (the k = 1 loop below
            // materializes its batches from the cursor directly and
            // never needs the count): the O(n) dry walk with the starts
            // collector (8 B x numBatches plus one sentinel - the
            // batch-boundary metadata of the streaming-rung charge).
            std::vector<std::size_t> batchStarts;
            std::size_t numBatches = 0;

            if (_maxConcurrentSlots > 1)
            {
                internal::MdClassBatch batch;

                while (cursor.Next(batch, true, &batchStarts))
                {
                    ++numBatches;
                }

                // The sentinel end of the last batch's master range.
                batchStarts.push_back(tasks.size());
                cursor.Reset();
            }

            // The half's batches run on k concurrent slots - kEff = min(authorized,
            // batch count) pullers, each claiming the next unclaimed batch
            // FIFO over the canonical batch order (fetch_add order - the
            // emission order is load-bearing for the k=1 bit-identity
            // contract: at k = 1 the loop below is today's batch sequence
            // verbatim, no gate, no per-slot region-thread
            // change). Each slot's inner regions fork q = max(1,
            // team/kEff) threads (regionThreads 0 = DefaultOmpTeamSize()
            // at k = 1), so the k-slot x q-thread product stays at or
            // under the team policy.
            const std::size_t kEff =
                _maxConcurrentSlots > 1 ? std::min(_maxConcurrentSlots, numBatches) : 1;
            const std::size_t teamSize =
                static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
            const std::size_t regionThreads =
                kEff > 1 ? std::max<std::size_t>(1, teamSize / kEff) : 0;

            // The per-batch slot body: assemble the batch into its bounded
            // buffers, BuildQuartetOffsets (inside AssembleBatch), contract
            // it, discard it - today's exact per-batch sequence. \p batchIndex
            // is the batch's
            // canonical ordinal (the gate's merge position; the k = 1 loop
            // counts its emissions); \p gate engages the batch-order chain
            // at k > 1 (the batch's merge happens at its canonical turn);
            // \p slotStats collects the timing - the shared statsOut at
            // k = 1, the per-slot partials at k > 1 (merged after the
            // join; the sums are order-independent integer nanoseconds).
            auto processBatch = [&](internal::MdClassBatch&& batch,
                                    std::size_t batchIndex,
                                    BatchMergeGate* gate,
                                    std::vector<double>& scratch,
                                    FockBuildStats* slotStats) -> qcx::Result<void> {
                // The fp32 lane's weight transport base: the batch's first
                // master position, captured BEFORE the move - AssembleBatch
                // resets the batch's boundsBase to the batch-local 0 (the
                // fp32 lane's per-batch bounds writes).
                const std::size_t weightBase = batch.boundsBase;
                const auto eriBatchStarted = wantStats && slotStats != nullptr
                                                 ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
                auto half = AssembleBatch(std::move(batch), fp32, regionThreads);

                if (!half.has_value())
                {
                    if (gate != nullptr)
                    {
                        gate->RecordAndAdvance(batchIndex, half.error());
                    }

                    return qcx::Result<void>{std::unexpected(half.error())};
                }

                if (wantStats && slotStats != nullptr)
                {
                    // Accumulated, not assigned: the LightPath chunk loop
                    // calls this body once per chunk, so the per-call
                    // timings sum across the chunks (zeroed at BuildFock;
                    // the FastPath's single pass is unchanged). The
                    // per-batch sums cover the same phases as before.
                    slotStats->eriWallTime += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - eriBatchStarted);
                }

                const auto contractBatchStarted = wantStats && slotStats != nullptr
                                                      ? std::chrono::steady_clock::now()
                                                      : std::chrono::steady_clock::time_point{};
                auto contracted = ContractBatch(*half,
                                                fp32,
                                                fp32DensityWeights,
                                                weightBase,
                                                scratch,
                                                certifiedBoundSum,
                                                gate,
                                                batchIndex,
                                                regionThreads,
                                                slotStats);

                if (!contracted.has_value())
                {
                    return contracted;
                }

                if (wantStats && slotStats != nullptr)
                {
                    // See the eriWallTime note.
                    slotStats->contractWallTime +=
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - contractBatchStarted);
                }

                return qcx::Result<void>{};
            };

            if (kEff <= 1)
            {
                // k = 1: today's per-batch sequence verbatim (no
                // gate, no region-thread change) - the bit-identity
                // baseline. The cursor's emissions ARE the eager
                // partitioner's batch sequence (the per-half pin),
                // materialized one batch at a time.
                std::size_t batchIndex = 0;

                for (;;)
                {
                    internal::MdClassBatch batch;

                    if (!cursor.Next(batch))
                    {
                        break;
                    }

                    auto processed =
                        processBatch(std::move(batch), batchIndex, nullptr, blockScratch, statsOut);

                    if (!processed.has_value())
                    {
                        return processed;
                    }

                    ++batchIndex;
                }

                return qcx::Result<void>{};
            }

            // k > 1: kEff - 1 worker threads pull batches FIFO alongside
            // the caller (each batch is claimed exactly once - the
            // fetch_add order IS the canonical batch order). Every error
            // path of the batch body records into the gate and advances
            // its turn (the chain never stalls), so the workers keep
            // pulling; the recorded error surfaces after the join.
            BatchMergeGate gate;
            std::atomic<std::size_t> nextBatch{0};
            const std::size_t workerCount = kEff - 1;
            std::vector<std::vector<double>> slotScratches(workerCount + 1);
            std::vector<FockBuildStats> slotStats(wantStats ? workerCount + 1 : 0);

            const auto pullAndRun = [&](std::size_t slotIndex) -> void {
                internal::MdClassBatch batch;

                for (;;)
                {
                    const std::size_t batchIndex = nextBatch.fetch_add(1);

                    if (batchIndex >= numBatches)
                    {
                        return;
                    }

                    // The slot slices its claimed batch's master range
                    // directly (FillMdClassBatch is the cursor's exact
                    // fill, so the slot-built descriptors are
                    // byte-identical to its emissions).
                    internal::FillMdClassBatch(batch,
                                               _pairStore,
                                               _pairList,
                                               tasks,
                                               batchStarts[batchIndex],
                                               batchStarts[batchIndex + 1]);

                    auto processed = processBatch(std::move(batch),
                                                  batchIndex,
                                                  &gate,
                                                  slotScratches[slotIndex],
                                                  wantStats ? &slotStats[slotIndex] : nullptr);
                    (void)processed;
                }
            };

            std::vector<std::thread> workers;
            workers.reserve(workerCount);

            for (std::size_t w = 0; w < workerCount; ++w)
            {
                workers.emplace_back([&, w] { pullAndRun(w); });
            }

            // The caller is the last slot (index workerCount).
            pullAndRun(workerCount);

            for (std::thread& worker : workers)
            {
                worker.join();
            }

            if (wantStats)
            {
                // The completed work's timings land in statsOut even on a
                // failed call (the k=1 loop's semantics: the per-batch sums
                // accumulate up to the failure).
                for (const FockBuildStats& slot : slotStats)
                {
                    statsOut->eriWallTime += slot.eriWallTime;
                    statsOut->contractWallTime += slot.contractWallTime;
                    statsOut->mergeWallTime += slot.mergeWallTime;
                }

                // The regime marker: record the
                // actually-run slot count. kEff can sit below the
                // authorized k (kEff = min(k, the pass's batch count)), so
                // the per-pass maximum over the call's passes is the
                // observed concurrency. A call that read > 1 ran
                // overlapping slots: its eri/contract sums above are
                // slot-accumulated (they can exceed the whole-call
                // totalWallTime) - total and merge stay wall-comparable.
                statsOut->concurrentSlots =
                    std::max(statsOut->concurrentSlots, static_cast<std::size_t>(kEff));
            }

            if (gate.error.has_value())
            {
                return qcx::Result<void>{std::unexpected(*gate.error)};
            }

            return qcx::Result<void>{};
        };

        auto pass64 = runHalf(fp64Tasks, false);

        if (!pass64.has_value())
        {
            return pass64;
        }

        return runHalf(fp32Tasks, true);
    }

    // The class-aware contraction pass: each
    // precision half assembles its orbit-rep ERI blocks (one block per
    // orbit per precision: the petite-list saving) and contracts them ONE
    // maxBatchBytes-bounded batch at a time - the per-call rep-values
    // mass is bounded at the batch cap. Each task expands its
    // orbit representative's assembled block per screened member
    // (ExpandMemberBlock: memberBlock[memberPos] = pattern[srcPos] *
    // repBlock[srcPos]) and accumulates under the member quartet - the
    // member-level kMult matches the plain path's shell-level multiplicity
    // exactly, which is what makes the class and plain contractions
    // bit-identical per (quartet, block) pair. The certified fp32
    // lane is engaged exactly as on the plain path: each screened member
    // quartet contracts in the precision its own gate routed it to (the
    // same ScreenAll partition) - fp64 members read the orbit rep's fp64
    // block, fp32 members the rep's certified fp32 block - and the
    // certified sum accumulates weight(member) * bound(orbit rep) (the
    // rep block's per-quartet bound covers every expanded member: the
    // expansion is an exact signed permutation of the rep block, so the
    // member error is the rep error, element-wise). The contraction order
    // is batch-major over the assembled reps (the pre-Fix-1 order was
    // class-task-major): the class-vs-plain equivalence pins hold per
    // (quartet, block) within their pinned machine-precision tolerance.
    qcx::Result<void> RunClassPass(const std::vector<ClassRepTask>& classTasks64,
                                   const std::vector<ClassRepTask>& classTasks32,
                                   const PairClassTable& classTable,
                                   const SymmetryReduction& reduction,
                                   const std::vector<double>& fp32DensityWeights,
                                   std::vector<double>& blockScratch,
                                   double& certifiedBoundSum) const {
        if (classTasks64.empty() && classTasks32.empty())
        {
            return qcx::Result<void>{};
        }

        auto runHalf = [&](const std::vector<ClassRepTask>& tasks, bool fp32) -> qcx::Result<void> {
            if (tasks.empty())
            {
                return qcx::Result<void>{};
            }

            // The aligned rep task list over the CANONICALLY SORTED class
            // tasks (the call sites sorted each precision's list with
            // SortClassRepTasks before the pass ran): position t of repTasks is
            // the orbit rep of
            // classTasks[t] - the positional identity the contraction
            // reads (the batch's assembled rep at index t is the class
            // task at boundsBase + t; no rep-keyed lookup exists any
            // more).
            std::vector<internal::MdQuartetTask> repTasks;
            repTasks.reserve(tasks.size());

            for (const ClassRepTask& task : tasks)
            {
                const ClassOrbit& orbit =
                    classTable.classPairs[task.classPairIndex].orbits[task.orbitIndex];
                repTasks.push_back(internal::MdQuartetTask{orbit.repBraPair, orbit.repKetPair, 0});
            }

            // The k-slot conversion of the class-pass batch
            // loop (the RunPass twin): kEff = min(authorized, batch count)
            // concurrent pullers over the canonical batch order, each
            // slot's inner regions at q = max(1, team/kEff) threads
            // (regionThreads 0 = team at k = 1), the per-batch merges
            // serialized in batch order through the BatchMergeGate. At
            // k = 1 the loop below is today's per-batch sequence verbatim.
            internal::MdClassBatchCursor cursor(
                _pairStore, _pairList, repTasks, _options.maxBatchBytes);

            // The batch-count pass and the per-batch slice table exist
            // only when the k > 1 slots can run (the k = 1 loop
            // materializes its batches from the cursor directly): the
            // O(n) dry walk with the starts collector (8 B x numBatches
            // plus one sentinel - the batch-boundary metadata of the
            // streaming-rung charge).
            std::vector<std::size_t> batchStarts;
            std::size_t numBatches = 0;

            if (_maxConcurrentSlots > 1)
            {
                internal::MdClassBatch batch;

                while (cursor.Next(batch, true, &batchStarts))
                {
                    ++numBatches;
                }

                // The sentinel end of the last batch's master range.
                batchStarts.push_back(repTasks.size());
                cursor.Reset();
            }

            const std::size_t kEff =
                _maxConcurrentSlots > 1 ? std::min(_maxConcurrentSlots, numBatches) : 1;
            const std::size_t teamSize =
                static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
            const std::size_t regionThreads =
                kEff > 1 ? std::max<std::size_t>(1, teamSize / kEff) : 0;

            // The per-batch slot body: assemble the orbit-rep batch into
            // its bounded buffers, then contract it (the serial fold or
            // the chunked reduction), merging through the gate at the
            // batch's canonical turn at k > 1. The batch's boundsBase is
            // the class-task index of its first assembled rep - captured
            // BEFORE the move (AssembleBatch resets it to the batch-local
            // 0) - the positional routing base of every contractRep
            // below.
            auto processBatch = [&](internal::MdClassBatch&& batch,
                                    std::size_t batchIndex,
                                    BatchMergeGate* gate,
                                    std::vector<double>& scratch) -> qcx::Result<void> {
                const std::size_t taskBase = batch.boundsBase;
                auto half = AssembleBatch(std::move(batch), fp32, regionThreads);

                if (!half.has_value())
                {
                    if (gate != nullptr)
                    {
                        gate->RecordAndAdvance(batchIndex, half.error());
                    }

                    return qcx::Result<void>{std::unexpected(half.error())};
                }

                // The per-batch contract: each task's rep block is read at
                // the rep's batch-local offset into the value arrays and
                // its certified bound at the rep's index in the batch's
                // computed list (the fp32 half's per-quartet bounds); the
                // block's flat layout follows ITS axis order, which can
                // differ from the rep's by the assembler's
                // l-canonicalization pair swap (ExpandMemberBlock).
                const std::size_t numReps = half->computed.size();
                const std::size_t numChunks = internal::ChunkCountFor(numReps, _options);

                // Contracts one assembled rep block's task into the target
                // contractor's Fock matrix - the shared body of the serial
                // fallback and every parallel chunk (the chunk-scoped
                // contractor/lambda discipline of RunPass). The task
                // contracts in the precision of its half: the fp32 half's
                // members also accumulate the density-weighted certified
                // bound (weight(member) * bound(orbit rep), the
                // composition).
                auto contractRep = [&](const FockContractor& contractor,
                                       std::size_t t,
                                       std::vector<double>& scratchSlot,
                                       double& boundSum) -> ClassTaskOutcome {
                    const ShellQuartet& repQuartet = half->computed[t];
                    // The positional task identity: the assembled rep at
                    // index t of the batch whose first class-task position
                    // is taskBase IS class task taskBase + t (repTasks is
                    // aligned to the sorted class tasks, and every
                    // assembled batch is a contiguous slice of the sorted
                    // master). An out-of-range index is a routing
                    // bookkeeping bug - the kRepMiss surface, structural.
                    if (taskBase + t >= tasks.size())
                    {
                        return ClassTaskOutcome::kRepMiss;
                    }

                    const ClassRepTask& task = tasks[taskBase + t];
                    const ClassOrbit& orbit =
                        classTable.classPairs[task.classPairIndex].orbits[task.orbitIndex];
                    const ShellPairIndex& repBra = _pairList.pairs[orbit.repBraPair];
                    const ShellPairIndex& repKet = _pairList.pairs[orbit.repKetPair];
                    const ShellQuartet orbitRepQuartet{repBra.i, repBra.j, repKet.i, repKet.j};

                    // The fp32 lane's positional member weights: each
                    // member's weight sits at its SCREENED-MASTER position
                    // (recorded by the fp32 BuildClassRepTasks call), not
                    // at any class-list position. The parallel positions
                    // array is the fp32 call's recording - its absence or
                    // a shortfall on an fp32 task is a routing bookkeeping
                    // bug (the kCertifiedMiss surface, structural). The
                    // fp64 call leaves it empty and never reads it.
                    if (fp32 && task.screenedMemberPositions.size() != task.screenedMembers.size())
                    {
                        return ClassTaskOutcome::kCertifiedMiss;
                    }

                    for (std::size_t memberIndex = 0; memberIndex < task.screenedMembers.size();
                         ++memberIndex)
                    {
                        const ClassOrbitMember& member = task.screenedMembers[memberIndex];
                        const ShellPairIndex& memberBra = _pairList.pairs[member.braPair];
                        const ShellPairIndex& memberKet = _pairList.pairs[member.ketPair];
                        const ShellQuartet memberQuartet{
                            memberBra.i, memberBra.j, memberKet.i, memberKet.j};

                        if (fp32)
                        {
                            // The certified lane's per-member density
                            // weight, read positionally from the
                            // master-parallel weights: the density-
                            // weighted bound sum below is the class-path
                            // mirror of the plain path's
                            // certifiedBoundSum.
                            if (task.screenedMemberPositions[memberIndex] >=
                                fp32DensityWeights.size())
                            {
                                return ClassTaskOutcome::kCertifiedMiss;
                            }

                            ExpandMemberBlock(reduction,
                                              _pairList,
                                              member,
                                              orbitRepQuartet,
                                              repQuartet,
                                              half->valuesF32.data() + half->quartetOffsets[t],
                                              scratchSlot);
                            contractor.AccumulateBlock(
                                scratchSlot.data(), memberQuartet, member.braPair, member.ketPair);
                            boundSum +=
                                fp32DensityWeights[task.screenedMemberPositions[memberIndex]] *
                                half->bounds[t];
                        } else
                        {
                            ExpandMemberBlock(reduction,
                                              _pairList,
                                              member,
                                              orbitRepQuartet,
                                              repQuartet,
                                              half->values.data() + half->quartetOffsets[t],
                                              scratchSlot);
                            contractor.AccumulateBlock(
                                scratchSlot.data(), memberQuartet, member.braPair, member.ketPair);
                        }
                    }

                    return ClassTaskOutcome::kContracted;
                };

                if (numChunks <= 1)
                {
                    if (gate == nullptr)
                    {
                        // The k=1 serial fold: today's per-batch canonical
                        // order, accumulated into the shared Fock directly.
                        for (std::size_t t = 0; t < numReps; ++t)
                        {
                            const ClassTaskOutcome outcome =
                                contractRep(*this, t, scratch, certifiedBoundSum);

                            if (outcome == ClassTaskOutcome::kRepMiss)
                            {
                                return ClassRoutingError();
                            }

                            if (outcome == ClassTaskOutcome::kCertifiedMiss)
                            {
                                return internal::CertifiedRoutingError();
                            }
                        }
                    } else
                    {
                        // The k>1 serial fold runs at the batch's turn,
                        // under the gate's lock - the same canonical-order
                        // association as the k=1 fold. A miss records the
                        // error (first one wins) and breaks; the turn
                        // always advances.
                        std::unique_lock<std::mutex> lock(gate->mutex);
                        gate->cv.wait(lock, [&] { return gate->accumulated == batchIndex; });

                        if (!gate->error.has_value())
                        {
                            for (std::size_t t = 0; t < numReps; ++t)
                            {
                                const ClassTaskOutcome outcome =
                                    contractRep(*this, t, scratch, certifiedBoundSum);

                                if (outcome == ClassTaskOutcome::kRepMiss)
                                {
                                    gate->error = ClassRoutingError().error();
                                    break;
                                }

                                if (outcome == ClassTaskOutcome::kCertifiedMiss)
                                {
                                    gate->error = internal::CertifiedRoutingError().error();
                                    break;
                                }
                            }
                        }

                        ++gate->accumulated;
                        gate->cv.notify_all();
                    }
                } else
                {
                    std::vector<std::size_t> chunkStarts(numChunks + 1);
                    chunkStarts[0] = 0;
                    chunkStarts[numChunks] = numReps;

                    for (std::size_t c = 1; c < numChunks; ++c)
                    {
                        // The RunPass boundary discipline: 8-double-aligned
                        // interior boundaries so adjacent chunks' expanded
                        // member blocks never share a cache line (the
                        // per-member scratch writes).
                        const std::size_t raw = (numReps * c) / numChunks;
                        chunkStarts[c] = (raw / 8) * 8;
                    }

                    const ClassPassPartial zero{
                        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(_fock.rows()),
                                              static_cast<Eigen::Index>(_fock.cols()))};

                    ClassPassPartial delta = internal::FixedOrderReduce<ClassPassPartial>(
                        numChunks,
                        zero,
                        [&](const ClassPassPartial& partial,
                            std::size_t chunkIndex) -> ClassPassPartial {
                            ClassPassPartial local = partial;
                            // Per-chunk scratch and a fresh contractor
                            // bound to the chunk's own matrix: the RunPass
                            // race discipline (ExpandMemberBlock writes the
                            // scratch; AccumulateBlock writes through
                            // _fock). The chunk-local drop counter rides
                            // the same partial.
                            std::vector<double> localBlockScratch;
                            FockContractor chunkContractor{_pairList,
                                                           _pairStore,
                                                           _options,
                                                           _density,
                                                           local.fock,
                                                           nullptr,
                                                           _pairMaxDensity,
                                                           &local.elementDrops};

                            for (std::size_t t = chunkStarts[chunkIndex];
                                 t < chunkStarts[chunkIndex + 1];
                                 ++t)
                            {
                                const ClassTaskOutcome outcome = contractRep(
                                    chunkContractor, t, localBlockScratch, local.certifiedBoundSum);

                                if (outcome == ClassTaskOutcome::kRepMiss)
                                {
                                    local.miss = true;
                                    break;
                                }

                                if (outcome == ClassTaskOutcome::kCertifiedMiss)
                                {
                                    local.certifiedMiss = true;
                                    break;
                                }
                            }

                            return local;
                        },
                        [](const ClassPassPartial& a,
                           const ClassPassPartial& b) -> ClassPassPartial {
                            return ClassPassPartial{a.fock + b.fock,
                                                    a.certifiedBoundSum + b.certifiedBoundSum,
                                                    a.miss || b.miss,
                                                    a.certifiedMiss || b.certifiedMiss,
                                                    a.elementDrops + b.elementDrops};
                        },
                        static_cast<int>(regionThreads));

                    if (delta.miss)
                    {
                        if (gate != nullptr)
                        {
                            gate->RecordAndAdvance(batchIndex, ClassRoutingError().error());
                        }

                        return ClassRoutingError();
                    }

                    if (delta.certifiedMiss)
                    {
                        if (gate != nullptr)
                        {
                            gate->RecordAndAdvance(batchIndex,
                                                   internal::CertifiedRoutingError().error());
                        }

                        return internal::CertifiedRoutingError();
                    }

                    if (gate == nullptr)
                    {
                        _fock += delta.fock;
                        certifiedBoundSum += delta.certifiedBoundSum;

                        // The parallel class path's drop count
                        // combines exactly; the serial path accumulated
                        // into the same target through *this (see RunPass).
                        if (_elementDrops != nullptr)
                        {
                            *_elementDrops += delta.elementDrops;
                        }
                    } else
                    {
                        // The delta merges at the batch's turn (the chain
                        // discipline - see ContractBatch).
                        std::unique_lock<std::mutex> lock(gate->mutex);
                        gate->cv.wait(lock, [&] { return gate->accumulated == batchIndex; });

                        if (!gate->error.has_value())
                        {
                            _fock += delta.fock;
                            certifiedBoundSum += delta.certifiedBoundSum;

                            if (_elementDrops != nullptr)
                            {
                                *_elementDrops += delta.elementDrops;
                            }
                        }

                        ++gate->accumulated;
                        gate->cv.notify_all();
                    }
                }

                return qcx::Result<void>{};
            };

            if (kEff <= 1)
            {
                // k = 1: today's per-batch sequence verbatim (no gate, no
                // region-thread change) - the bit-identity baseline. The
                // cursor's emissions ARE the eager partitioner's batch
                // sequence, materialized one batch at a time.
                std::size_t batchIndex = 0;

                for (;;)
                {
                    internal::MdClassBatch batch;

                    if (!cursor.Next(batch))
                    {
                        break;
                    }

                    auto processed =
                        processBatch(std::move(batch), batchIndex, nullptr, blockScratch);

                    if (!processed.has_value())
                    {
                        return processed;
                    }

                    ++batchIndex;
                }

                return qcx::Result<void>{};
            }

            // k > 1: kEff - 1 worker threads pull batches FIFO alongside
            // the caller (each batch claimed exactly once); every error
            // path of the batch body records into the gate and advances
            // its turn, so the workers keep pulling and the chain never
            // stalls - the recorded error surfaces after the join.
            BatchMergeGate gate;
            std::atomic<std::size_t> nextBatch{0};
            const std::size_t workerCount = kEff - 1;
            std::vector<std::vector<double>> slotScratches(workerCount + 1);

            const auto pullAndRun = [&](std::size_t slotIndex) -> void {
                internal::MdClassBatch batch;

                for (;;)
                {
                    const std::size_t batchIndex = nextBatch.fetch_add(1);

                    if (batchIndex >= numBatches)
                    {
                        return;
                    }

                    // The slot slices its claimed batch's master range
                    // directly (FillMdClassBatch is the cursor's exact
                    // fill, so the slot-built descriptors are
                    // byte-identical to its emissions).
                    internal::FillMdClassBatch(batch,
                                               _pairStore,
                                               _pairList,
                                               repTasks,
                                               batchStarts[batchIndex],
                                               batchStarts[batchIndex + 1]);
                    auto processed =
                        processBatch(std::move(batch), batchIndex, &gate, slotScratches[slotIndex]);
                    (void)processed;
                }
            };

            std::vector<std::thread> workers;
            workers.reserve(workerCount);

            for (std::size_t w = 0; w < workerCount; ++w)
            {
                workers.emplace_back([&, w] { pullAndRun(w); });
            }

            // The caller is the last slot (index workerCount).
            pullAndRun(workerCount);

            for (std::thread& worker : workers)
            {
                worker.join();
            }

            if (gate.error.has_value())
            {
                return qcx::Result<void>{std::unexpected(*gate.error)};
            }

            return qcx::Result<void>{};
        };

        auto pass64 = runHalf(classTasks64, false);

        if (!pass64.has_value())
        {
            return pass64;
        }

        return runHalf(classTasks32, true);
    }
};

// Contracts one computed quartet (index t into `computed`, values at
// quartetOffsets[t]) into the TARGET contractor's Fock matrix - the body
// shared by RunPass's serial fallback and every parallel chunk. A free
// function taking the contractor (not a member of it) because the parallel
// path needs a FRESH contractor per chunk, bound to the chunk's own local
// matrix: AccumulateBlock writes through the contractor's _fock reference,
// so a shared binding would race (the plan's per-chunk-contractor
// requirement). The fp32 lane casts the block back through blockScratch
// (per-thread scratch: parallel chunks must each pass their own vector,
// never the shared one) and accumulates the density-weighted certified
// bound into certifiedBoundSum (per-chunk local when parallel - the
// reduction combines them).
qcx::Result<void> ContractOne(const FockContractor& contractor,
                              const std::vector<ShellQuartet>& computed,
                              const std::vector<double>& values,
                              const std::vector<float>& valuesF32,
                              const std::vector<double>& bounds,
                              std::size_t t,
                              const std::vector<std::size_t>& quartetOffsets,
                              bool fp32,
                              const std::vector<double>& fp32Weights,
                              std::size_t weightBase,
                              std::vector<double>& blockScratch,
                              double& certifiedBoundSum) {
    const ShellQuartet& quartet = computed[t];
    const std::size_t nI = ShellFunctionCount(contractor._pairList.shells[quartet.i]);
    const std::size_t nJ = ShellFunctionCount(contractor._pairList.shells[quartet.j]);
    const std::size_t nK = ShellFunctionCount(contractor._pairList.shells[quartet.k]);
    const std::size_t nL = ShellFunctionCount(contractor._pairList.shells[quartet.l]);
    const std::size_t blockSize = nI * nJ * nK * nL;
    const std::size_t offset = quartetOffsets[t];

    if (fp32)
    {
        blockScratch.resize(blockSize);

        for (std::size_t element = 0; element < blockSize; ++element)
        {
            blockScratch[element] = static_cast<double>(valuesF32[offset + element]);
        }

        const std::size_t braPair = PairIndexOf(quartet.i, quartet.j, contractor._pairList);
        const std::size_t ketPair = PairIndexOf(quartet.k, quartet.l, contractor._pairList);
        contractor.AccumulateBlock(blockScratch.data(), quartet, braPair, ketPair);
        // Each quartet's kernel bound times its max-|D| block weight:
        // the weighted sum bounds every Fock-element error
        // from the fp32 lane - an upper bound on the delivered Fock
        // error, consumable as certified. The weight is transported
        // positionally: fp32Weights is parallel to the canonical-sorted fp32 master
        // and the assembled batch task at index t IS master position
        // weightBase + t (the per-half byte-identity pin), so no key
        // lookup exists - a class-swapped computed quartet still sits
        // at its master position and carries that position's weight
        // (an orientation-free block max). The range guard keeps the
        // CertifiedRoutingError surface: an out-of-range index is a
        // routing bookkeeping bug and must not under-count the
        // certified sum silently.
        if (weightBase + t >= fp32Weights.size())
        {
            return internal::CertifiedRoutingError();
        }

        certifiedBoundSum += fp32Weights[weightBase + t] * bounds[t];
    } else
    {
        const std::size_t braPair = PairIndexOf(quartet.i, quartet.j, contractor._pairList);
        const std::size_t ketPair = PairIndexOf(quartet.k, quartet.l, contractor._pairList);
        contractor.AccumulateBlock(values.data() + offset, quartet, braPair, ketPair);
    }

    return qcx::Result<void>{};
}

// The LightPath light store: one geometry-only MdPairData per
// canonical pair - exactly the
// fields BuildPairData sets from the flattened shells (angular momenta,
// spherical flags, centers), with the transform vectors EMPTY. The chunk
// pass fills the contracted transforms of the chunk's pairs into this same
// vector (BuildChunkPairData; the empty braTransform is the not-built
// marker) and clears them after the chunk, so the vector's capacities
// persist and the per-chunk arena never reallocates past its peak (the
// reused containers are not Reservations - the CWA sees one peak,
// not a sum over chunks).
qcx::Result<std::vector<internal::MdPairData>> BuildLightStore(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const ShellPairList& pairList) {
    auto shells = internal::FlattenShells(molecule, basisSet, pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    std::vector<internal::MdPairData> store;
    store.reserve(pairList.pairs.size());

    for (const ShellPairIndex& pairIndex : pairList.pairs)
    {
        const internal::MdShellInput& shellA = (*shells)[pairIndex.i];
        const internal::MdShellInput& shellB = (*shells)[pairIndex.j];
        internal::MdPairData pair;
        pair.la = shellA.contractions.angularMomentum;
        pair.lb = shellB.contractions.angularMomentum;
        pair.isSphericalA = shellA.contractions.isSpherical;
        pair.isSphericalB = shellB.contractions.isSpherical;
        pair.ax = shellA.cx;
        pair.ay = shellA.cy;
        pair.az = shellA.cz;
        pair.bx = shellB.cx;
        pair.by = shellB.cy;
        pair.bz = shellB.cz;
        store.push_back(std::move(pair));
    }

    return store;
}

} // namespace

DirectJkFockBuilder::DirectJkFockBuilder(std::shared_ptr<const State> state) :
    _state(std::move(state)) {}

const std::optional<FockModeInfo>& DirectJkFockBuilder::ModeInfo() const noexcept {
    return _state->_modeInfo;
}

qcx::Result<DirectJkFockBuilder> DirectJkFockBuilder::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
    const FockBuildOptions& options) {
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

    for (const ShellInfo& shell : pairList->shells)
    {
        if (!SupportsL(shell.angularMomentum))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "shell angular momentum exceeds kMaxEngineL of this build"});
        }
    }

    // The adaptive-memory mode decision: the Create-time footprint estimate
    // against the budget's
    // remaining bytes, decided once. Only the options that carry a budget
    // take this path - the legacy path keeps its exact order and behavior
    // below. The actual BuildNeighborList sweep (needs only the pair list
    // and the Schwarz bounds - nothing built yet) feeds BOTH the decision's
    // pattern term and, later, the state's CSR - no mirror, no second
    // sweep.
    std::optional<FockModeInfo> modeInfo;
    FockBuildOptions effectiveOptions = options;
    std::vector<std::size_t> neighborRowOffsets;
    std::vector<std::size_t> neighborIndices;
    bool neighborListBuilt = false;
    std::optional<std::vector<double>> schwarz;
    // The class-path admission gate (the pair-class root fix of
    // 2026-08-31): the class path builds the pair-class table at Create -
    // the FULL canonical member-quartet space, a theta(N^4) structure the
    // historic estimates never charged (the c60 0xC0000409 deaths at every
    // cap - the 57.1 GiB table at 45,150 pairs). On the budget path
    // the gate below disengages the class path when the table cannot fit
    // the budget's remaining bytes - the plain screened path runs instead
    // (a throughput choice, never a crash: the refusal ladder still fires
    // when even the plain path cannot fit). The legacy null-budget path
    // keeps today's unconditional engagement, bounded only by the
    // escape-hatch host-RAM guard below: without a budget the never-under
    // charge is checked against the node's total physical RAM, and an
    // un-fittable table REFUSES cleanly instead of dying as a raw
    // allocation.
    const bool classPathRequested =
        options.symmetryReduction != nullptr && options.symmetryReduction->groupOrder > 1;
    bool classPath = classPathRequested;
    std::size_t classTableBytes = 0;
    bool classPathDisengaged = false;

    if (options.workspaceBudget != nullptr)
    {
        qcx::memory::WorkspaceBudget& budget = *options.workspaceBudget;
        const std::size_t remaining = budget.Remaining();
        FockModeInfo info;
        info.budgetBytes = budget.CapacityBytes();
        info.remainingAtDecision = remaining;
        info.maxBatchBytes = options.maxBatchBytes;
        const std::size_t nPairs = pairList->pairs.size();
        const std::size_t threadCount =
            static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
        // The team-size read: record the team the
        // decision below clamps k = min(..., DefaultOmpTeamSize()) against
        // (and sizes every per-thread arena with) - a run whose
        // concurrentSlots equals this field was team-clamped, one below it
        // budget-clamped: k == defaultTeamSize means the fired k was the
        // whole ceiling-clamped team, not a budget split.
        info.defaultTeamSize = threadCount;

        // The cache clamp (the sure-fit gate of the
        // O(N^2)-default selection bound): shrink maxCacheBytes until its
        // 3x live footprint - the two payload lanes plus the entry map
        // (DirectFootprint's cacheBytes term) - fits the budget's
        // remaining bytes, else zero it (cache off). Runs before any
        // estimate, so every fired term carries the clamped cap: the
        // cache survives only when it provably fits.
        effectiveOptions.maxCacheBytes = std::min(effectiveOptions.maxCacheBytes, remaining / 3);

        // The class-table admission gate: charge the never-under worst-case
        // table (ClassTableBytes - the singleton-orbit bound, the only
        // shape Create-time-safe without building the table itself) against
        // the remaining budget. When it cannot fit, the class path
        // DISENGAGES: the plain screened path runs instead (the reduction
        // is dropped - bit-identical results, only the throughput
        // optimization is lost), and every term below carries the gated
        // classTableBytes (0 when disengaged). Under-charging would let
        // the theta(N^4) table build blow the budget - the 0xC0000409
        // death this gate removes. The boundary is >=: at exact equality
        // the table would consume the whole remainder and leave zero for
        // the builder's own terms (the downstream refusal would catch it,
        // but the strictly correct fit test disengages here).
        classTableBytes =
            classPathRequested ? internal::ClassTableBytes(nPairs, pairList->functionCount) : 0;

        if (classPathRequested)
        {
            // The charge is the shape the path actually reaches, not the
            // eager ceiling. The ceiling is the singleton-orbit worst case
            // over the WHOLE canonical quartet space - the only shape
            // Create-time-safe without the class decomposition, and a gross
            // over-charge since the root restructure (2026-08-31) began
            // storing the Schwarz-screened class pairs with empty orbit
            // vectors and generating the members on demand per reached class
            // pair. The class decomposition's own bound
            // (CountClassMaterialization) needs only the Schwarz bounds and
            // O(nPairs) bytes: the Schwarz-screened class pairs and their
            // member-quartet mass are counted by a sorted sweep, never
            // materialized. Charging it ALWAYS - not only when the ceiling
            // refuses - keeps the decision and the estimate the same
            // quantity, so the recorded charge cannot describe one shape
            // while the gate decision used another.
            if (!schwarz.has_value())
            {
                auto computedSchwarz = ComputeSchwarzBounds(molecule, basisSet);

                if (!computedSchwarz.has_value())
                {
                    return std::unexpected(computedSchwarz.error());
                }

                schwarz = std::move(*computedSchwarz);
            }

            const auto materialization = CountClassMaterialization(
                *options.symmetryReduction,
                *pairList,
                *schwarz,
                SchwarzThreshold(effectiveOptions.accuracy) * internal::kNeighborListSlack);

            if (!materialization.has_value())
            {
                return std::unexpected(materialization.error());
            }

            // Both charges are upper bounds of the same allocation, so the
            // minimum is the tightest never-under one: the gate engages at
            // least wherever the ceiling alone engaged, and never charges
            // less than the allocation it admits.
            classTableBytes = std::min(classTableBytes,
                                       internal::ClassMaterializationBytes(
                                           *materialization, nPairs, pairList->functionCount));
        }

        if (classPathRequested && classTableBytes >= remaining)
        {
            classPath = false;
            classPathDisengaged = true;
            classTableBytes = 0;
        }

        info.classPathDisengaged = classPathDisengaged;
        info.classTableBytes = classTableBytes;

        // The LightPath chunk knob: non-zero FORCES the light
        // mode with that chunk size - the mode-forcing test surface. The
        // budget-driven LightPath is unreachable at small
        // scales, where the pattern saving is zero and the light rung's
        // floor sits above the fast path's, so the bit-identity pins force
        // the mode through the knob; the knob skips the fast estimate
        // entirely.
        bool lightPath = options.lightPathChunkPairs != 0;
        const std::size_t forcedChunkPairs = options.lightPathChunkPairs;
        // True once the counting sweep has run: the decision's CSR feeds
        // the fast path's pattern term AND the light path's peak-chunk
        // markers (no mirror, no second sweep). At the a-priori exclusion
        // no sweep runs - the counted peak row width below stands in for
        // the markers where the exclusion's own count produced one, and
        // the all-survive bounds remain the fallback for the knob path and
        // for a caller that reached the light decision with no count at
        // all.
        bool patternCounted = false;
        // The exclusion's counted peak row width (CountSchwarzPeakRowWidth
        // - the never-under bound on the widest emitted bra row): the
        // light rung's peak-chunk ket bound when no sweep ran. 0 = no
        // count was made (the knob path, or a pre-gate that did not fire),
        // where the all-survive markers stand.
        std::size_t excludedPeakRowWidth = 0;
        // The fast path's estimate (the fewer-bytes gate's reference; 0
        // when no fast estimate was made - the exclusion or the knob).
        std::size_t fastEstimateBytes = 0;

        // A-priori exclusion (i): the decision's pattern bound against the
        // remaining budget - the LightPath DECISION when the pattern cannot
        // fit (the refusal fires only when the light rung ALSO cannot fit,
        // below). The bound is the COUNTED pattern - the Schwarz-exact
        // survivor count under the BuildNeighborList cutoff: the all-survive
        // form over-refused
        // large-sparse systems (the 81,003-pair chain: 24.44 GiB
        // all-survive against a ~38 MB counted pattern), so it survives
        // only as the cheap PRE-GATE below - when even the all-survive
        // pattern fits, the counted one does a fortiori and no counting
        // pass runs (the small-dense path pays nothing). The exact count
        // is bit-identical to the sweep's counting pass (same doubles,
        // same product cutoff - it never admits a row the sweep would
        // drop), so nothing the estimate models is ever excluded. The count
        // above is also what makes this pre-gate the LightPath's OWN route
        // at a band the all-survive bound already overruns: the counted form
        // cannot refuse a stack the enclosing builder proved feasible (the
        // RI-J's nested exchange half reaches its rung through here - the
        // engine-exclusion-i record: with the count skipped, the sweep below
        // materialized the fast rung's whole-pattern CSR, 183.74 GiB at the
        // 4,974 manifest case, before the light decision could be made). The
        // QFMM outer still carries its counted sweep (validatedSweepCount
        // below) - the skip's original purpose, the all-survive bound that
        // exceeds a post-reservation remaining the counted pattern fits; the
        // nested still runs its own counting pass and estimate, so the
        // decision is not weakened either way.
        if (!lightPath && options.validatedSweepCount == 0 &&
            internal::AllSurvivePatternBytes(nPairs) > remaining)
        {
            if (!schwarz.has_value())
            {
                auto computedSchwarz = ComputeSchwarzBounds(molecule, basisSet);

                if (!computedSchwarz.has_value())
                {
                    return std::unexpected(computedSchwarz.error());
                }

                schwarz = std::move(*computedSchwarz);
            }

            // The restricted rows when a leaf-driven near-field domain is
            // present (the counted block below takes the same branch) - the
            // exclusion must bound the rows the sweep would actually emit.
            const std::size_t patternCount =
                options.leafNearFieldDomain.Empty()
                    ? internal::CountSchwarzSurvivingPairs(*schwarz, options.accuracy)
                    : internal::CountNearFieldPatternEntries(
                          *schwarz, options.accuracy, options.leafNearFieldDomain);
            // The x8 byte form, integer-exact: count > remaining / 8 is
            // 8 x count > remaining without the size overflow.
            if (patternCount > remaining / 8)
            {
                info.patternExcluded = true;
                lightPath = true;
                // The light decision's peak-chunk ket bound, taken from the
                // same bounds the count just read (O(nPairs), no sweep, no
                // CSR - the pattern term is exactly what could not fit). The
                // rows below use it where they would otherwise fall back to
                // the all-survive nPairs, which is the bound that made the
                // rung unreachable at scale (the exclusion fires exactly
                // when the pair space is large, so the fallback is worst
                // where the rung is the only in-memory fit left).
                excludedPeakRowWidth =
                    internal::CountSchwarzPeakRowWidth(*schwarz, options.accuracy);
            }
        }

        // The counted decision terms: the Schwarz bounds (reused when the
        // exclusion's pre-gate computed them) and the actual sweep, then
        // the fast estimate - the budget path's decision when the fast rung
        // can fit.
        if (!lightPath)
        {
            if (!schwarz.has_value())
            {
                auto computedSchwarz = ComputeSchwarzBounds(molecule, basisSet);

                if (!computedSchwarz.has_value())
                {
                    return std::unexpected(computedSchwarz.error());
                }

                schwarz = std::move(*computedSchwarz);
            }

            // The actual sweep - the pattern term of the estimate. The QFMM
            // leaf-driven near-field domain replaces
            // the global sweep with the enumerated near-field candidate
            // rows: the count and the content are the restricted ones - the
            // rows the gated sweep would filter to, same cutoff.
            if (options.leafNearFieldDomain.Empty())
            {
                internal::BuildNeighborList(
                    *pairList, *schwarz, options.accuracy, neighborRowOffsets, neighborIndices);
            } else
            {
                internal::BuildLeafDrivenNeighborList(*pairList,
                                                      *schwarz,
                                                      options.accuracy,
                                                      options.leafNearFieldDomain,
                                                      neighborRowOffsets,
                                                      neighborIndices);
            }

            neighborListBuilt = true;
            patternCounted = true;

            const bool useCache = effectiveOptions.maxCacheBytes > 0 && !classPath;
            // The exchange term is charged only when the exchange pass is
            // live: a Coulomb-only builder (buildCoulombOnly, the UHF J
            // half and the QFMM near-field) carries no exchange mass - the
            // phantom charge the exchange fold-in removes.
            // The exchange term is also the slot-authorized term - k slots only
            // on the budgeted fast path with the exchange engaged and the
            // cache off (the cache-engaged, Coulomb-only and LightPath
            // rungs run k = 1). The authorization extends to a pre-reserved
            // nested half
            // (validatedSweepCount != 0 - the RI-J stack's exchange): the
            // outer reserves only its OWN terms and leaves the exchange
            // band in the budget (nesting order = reservation order,
            // ri_engine.cpp attempt()), and the nested's own decision now
            // fits its k-fold charge into the POST-reservation remaining
            // exactly like a free-standing run - k > 1 fires only where
            // that remaining has room for the k live per-slot bounds
            // (budget-conditional, self-gating), and the exchange's own
            // fixed point clamps ITS cap (never the outer's) when the
            // saturated k-charge would overrun its band. The k=1 paths
            // (no slack, no budget, the cache-engaged/Coulomb-only/
            // LightPath rungs) are unchanged - the k=1 pins stay green by
            // construction. k = min(max(1, floor(remaining /
            // cap)), DefaultOmpTeamSize()) at the CURRENT cap - the
            // authorized count the runtime batch loop carries (never
            // re-derived from the live Remaining; the charge covers
            // exactly this many slots' live mass). Because the k-factor
            // charge k x (2|3) x cap is a RISING SAWTOOTH in the cap (k
            // ticks up as the cap shrinks, so the one-step deficit clamp
            // cannot guarantee the exact re-estimate fits), the fast
            // decision ITERATES to the fixed point: re-estimate at the
            // current cap with the current authorized k, and when the
            // estimate still exceeds the remaining bytes take one clamp
            // step by the charged estimate's exact per-byte-of-cap slope
            // (threadCount + slotCount x (2|3) - the scratch arena at one
            // per thread plus the slot-folded exchange bound). The fitting
            // fixed point is the saturated one (k = team at cap ~
            // (remaining - fixed) / (threadCount + k x (2|3))), reached
            // from any band; the 8-step defensive iteration cap and the
            // unit-batch refusal both descend the ladder to the LightPath
            // decision below.
            const bool slotsAuthorized = !options.buildCoulombOnly && !useCache;
            const std::size_t slotFactor = options.accuracy == AccuracyPreset::kTight ? 2 : 3;
            std::size_t cap = options.maxBatchBytes;
            std::size_t slotCount = 1;
            bool fastFired = false;
            internal::DirectFootprintTerms terms =
                internal::DirectFootprint(molecule,
                                          basisSet,
                                          *pairList,
                                          neighborIndices.size(),
                                          options.maxBatchBytes,
                                          threadCount,
                                          useCache,
                                          effectiveOptions.maxCacheBytes,
                                          options.accuracy,
                                          !options.buildCoulombOnly,
                                          classTableBytes);

            for (std::size_t iteration = 0; iteration < 8 && !fastFired; ++iteration)
            {
                slotCount = slotsAuthorized
                                ? std::min(std::max<std::size_t>(1, remaining / cap), threadCount)
                                : 1;
                terms = internal::DirectFootprint(molecule,
                                                  basisSet,
                                                  *pairList,
                                                  neighborIndices.size(),
                                                  cap,
                                                  threadCount,
                                                  useCache,
                                                  effectiveOptions.maxCacheBytes,
                                                  options.accuracy,
                                                  !options.buildCoulombOnly,
                                                  classTableBytes,
                                                  slotCount);

                if (iteration == 0)
                {
                    // The raw-cap estimate - the fewer-bytes gate's
                    // reference (the light decision below compares against
                    // it).
                    fastEstimateBytes = terms.Total();
                }

                if (terms.Total() <= remaining)
                {
                    fastFired = true;
                    break;
                }

                const std::size_t clamped =
                    internal::ClampBatchBytes(cap,
                                              threadCount,
                                              terms.Total(),
                                              remaining,
                                              1,
                                              threadCount + slotCount * slotFactor);

                if (clamped == 0)
                {
                    break;
                }

                cap = clamped;
            }

            if (!fastFired)
            {
                // The fast rung cannot fit (even a unit batch across the
                // clamp steps, or the defensive iteration cap) - the
                // LightPath DECISION with the counted terms (the
                // decision-9 resolution; the refusal fires only when the
                // light rung also cannot fit, below).
                lightPath = true;
            } else
            {
                // The fired terms at the fixed point: the estimate with
                // the fitted cap and the authorized slot count it carried.
                info.predictedBytes = terms.Total();
                info.pairStoreBytes = terms.pairStoreBytes;
                info.patternBytes = terms.patternBytes;
                info.scratchBytes = terms.scratchBytes;
                info.structuralBytes = terms.structuralBytes;
                info.cacheBytes = terms.cacheBytes;
                info.maxBatchBytes = cap;
                info.concurrentSlots = slotCount;
                info.mode = FockBuildMode::kFastPath;

                if (!budget.Reserve(info.predictedBytes))
                {
                    // A concurrent reservation consumed the budget between
                    // the decision and the reserve (the CWA is monotone -
                    // the failed reserve charges nothing) - the decision is
                    // stale, refuse.
                    info.mode = FockBuildMode::kLightPath;
                    info.reservedBytes = 0;
                    return std::unexpected(qcx::Error{
                        qcx::ErrorCode::kUnimplemented,
                        internal::FastPathRefusal(
                            "direct", info.predictedBytes, remaining, false, false) +
                            " (the budget moved between the decision and the reservation)"});
                }

                info.reservedBytes = info.predictedBytes;
                effectiveOptions.maxBatchBytes = cap;
                effectiveOptions.workspaceBudget = nullptr;
                modeInfo = info;
            }
        }

        if (lightPath)
        {
            // The LightPath decision: the light store
            // plus the per-chunk arena and pattern terms against the
            // remaining budget. The chunk is auto-sized from the budget
            // (or the knob's value); the peak-chunk terms come from the
            // decision's swept CSR (the marker pass, O(pattern) once) or
            // the counted peak row width the exclusion's own count produced
            // (no sweep ran - the pattern is the term that could not fit),
            // or the all-survive bounds when no count exists at all (the
            // knob path).
            const std::size_t maxPairPayload =
                internal::MaxPairPayload(molecule, basisSet, *pairList);
            std::size_t maxRow = nPairs;

            if (patternCounted)
            {
                maxRow = 0;

                for (std::size_t bra = 0; bra < nPairs; ++bra)
                {
                    maxRow =
                        std::max(maxRow, neighborRowOffsets[bra + 1] - neighborRowOffsets[bra]);
                }
            } else if (excludedPeakRowWidth != 0)
            {
                // The byte-identical decision without the sweep: the widest
                // emitted row cannot exceed the widest full qualifying set,
                // and the exclusion's count already walked those bounds.
                maxRow = excludedPeakRowWidth;
            }

            std::size_t chunkPairs = forcedChunkPairs;

            if (chunkPairs == 0)
            {
                const std::size_t lightStoreBytes = nPairs * sizeof(internal::MdPairData);
                // Structural-only query (the exchange term is not part of
                // the light model - exchangeEngaged false, the threading).
                const std::size_t structuralBytes = internal::DirectFootprint(molecule,
                                                                              basisSet,
                                                                              *pairList,
                                                                              0,
                                                                              options.maxBatchBytes,
                                                                              threadCount,
                                                                              false,
                                                                              0,
                                                                              options.accuracy,
                                                                              false)
                                                        .structuralBytes;
                // The class table rides the fixed whole-build terms when
                // the gate engaged it: the auto-sized chunk must leave the
                // table's bytes uncommitted (the gate already proved they
                // fit; the chunk sizes the arena into what is left). The
                // screened-quartet task machinery rides them too (the whole-run
                // charge): the sorted task
                // masters are whole-run mass, not chunk-arena mass, so the
                // auto-sized chunk must leave them uncommitted exactly like
                // the table's bytes.
                chunkPairs = internal::AutoLightChunkPairs(
                    remaining,
                    lightStoreBytes,
                    structuralBytes,
                    options.maxBatchBytes * threadCount,
                    internal::LightChunkIndexBytes(nPairs) +
                        internal::LightShellsBytes(pairList->shells.size()) + classTableBytes +
                        internal::ScreenedQuartetBytes(pairList->functionCount, options.accuracy),
                    maxPairPayload,
                    maxRow,
                    nPairs);
            }

            // The peak-chunk markers: the counted peak row width the
            // exclusion's count produced (a chunk's ket set is the union of
            // its rows' sets, so it is bounded by the rows' count times the
            // widest row - and by the pair space, which is all the count
            // itself was ever bounded by), the marker pass over the
            // decision's CSR otherwise, and the all-survive bounds when
            // neither exists (the knob path).
            std::size_t maxChunkKets = nPairs;
            std::size_t peakChunkPattern = chunkPairs * nPairs;

            if (patternCounted)
            {
                const internal::LightChunkMarkers markers = internal::PeakChunkMarkers(
                    neighborRowOffsets, neighborIndices, nPairs, chunkPairs);
                maxChunkKets = markers.maxChunkKets;
                peakChunkPattern = markers.peakChunkPattern;
            } else if (excludedPeakRowWidth != 0)
            {
                // Rows the chunk can actually hold: a forced chunk wider
                // than the pair space has no extra rows, and the product
                // stays in size_t's range.
                const std::size_t boundRows = std::min(chunkPairs, nPairs);
                maxChunkKets = std::min(nPairs, boundRows * maxRow);
                peakChunkPattern = std::min(boundRows * nPairs, boundRows * maxRow);
            }

            internal::LightFootprintTerms light = internal::LightFootprint(molecule,
                                                                           basisSet,
                                                                           *pairList,
                                                                           chunkPairs,
                                                                           maxChunkKets,
                                                                           peakChunkPattern,
                                                                           maxPairPayload,
                                                                           options.maxBatchBytes,
                                                                           threadCount,
                                                                           classPath,
                                                                           options.accuracy,
                                                                           classTableBytes);
            info.lightStoreBytes = light.lightStoreBytes;
            info.chunkArenaBytes = light.chunkArenaBytes;
            info.chunkPatternBytes = light.chunkPatternBytes;
            info.chunkIndexBytes = light.chunkIndexBytes;
            info.lightShellsBytes = light.lightShellsBytes;
            info.scratchBytes = light.scratchBytes;
            info.structuralBytes = light.structuralBytes;
            info.chunkPairs = chunkPairs;
            info.mode = FockBuildMode::kLightPath;

            // The fewer-bytes gate (the nested contract): the light
            // rung engages only when its estimate does not exceed the fast
            // path's estimate - a light path that cannot beat the fast
            // estimate would only add the light-store overhead (the tiny
            // all-survive end, where the pattern saving is zero). The
            // knob-forced path bypasses the gate (the force is the test
            // surface); at the exclusion no fast estimate exists.
            if (fastEstimateBytes != 0 && light.Total() > fastEstimateBytes)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kUnimplemented,
                               internal::FastPathRefusal(
                                   "direct", fastEstimateBytes, remaining, false, false)});
            }

            // Clamp the batch arena like the fast path; a light rung that
            // still cannot fit at a unit arena is the final refusal.
            const std::size_t clamped = internal::ClampBatchBytes(
                options.maxBatchBytes, threadCount, light.Total(), remaining);

            if (clamped == 0)
            {
                return std::unexpected(qcx::Error{
                    qcx::ErrorCode::kUnimplemented,
                    internal::FastPathRefusal(
                        "direct", light.Total(), remaining, info.patternExcluded, false)});
            }

            // The fired terms at the clamped batch (the chunk terms are
            // batch-independent - only the arena shrinks).
            light.scratchBytes = clamped * threadCount;
            info.scratchBytes = light.scratchBytes;
            info.predictedBytes = light.Total();
            info.maxBatchBytes = clamped;

            if (!budget.Reserve(info.predictedBytes))
            {
                // The race-refusal of the fast path, unchanged in shape.
                info.reservedBytes = 0;
                return std::unexpected(qcx::Error{
                    qcx::ErrorCode::kUnimplemented,
                    internal::FastPathRefusal(
                        "direct", info.predictedBytes, remaining, info.patternExcluded, false) +
                        " (the budget moved between the decision and the reservation)"});
            }

            info.reservedBytes = info.predictedBytes;
            effectiveOptions.maxBatchBytes = clamped;
            // The whole engine tier is disengaged on the LightPath: the
            // miss-assembly reads the full pair store, which the light mode
            // never materializes (the documented follow-on). The decorator
            // goes with the RAM tier rather than around it - the seam is a
            // choice of ENGINE, and the LightPath's chunk loop is the thing
            // that makes the engine unreachable, not the cache's budget. The
            // run's record carries the demotion.
            effectiveOptions.maxCacheBytes = 0;
            effectiveOptions.engineDecorator = std::nullopt;
            effectiveOptions.workspaceBudget = nullptr;
            modeInfo = info;
        }
    }

    // The null-budget escape-hatch host-RAM guard (2026-09-01): the legacy
    // no-budget path (workspaceBudget
    // null - memory_cap_gib = 0, the documented opt-out) engages the class
    // path unconditionally - the admission gate above never runs, so a
    // c60-shaped request would still die as a raw machine-OOM allocation
    // while the on-demand orbit expansions grow into the canonical
    // member-quartet space. The guard charges the same never-under
    // estimate (ClassTableBytes) against the node's TOTAL physical RAM and
    // refuses CLEANLY (kOutOfMemory) when the table cannot fit - never an
    // allocation death. Refusing instead of disengaging is the deliberate
    // decision: the hatch opted out of the admission gate, so no gated
    // fallback remains to run; the refusal names the reinstatement (a
    // positive memory_cap_gib re-engages the gate, which disengages the
    // class path where the plain path still fits). A zero probe (unknown,
    // the memory_topology.hpp contract - the same convention as the
    // maxCacheBytes clamp) never refuses.
    if (options.workspaceBudget == nullptr && classPathRequested)
    {
        const std::size_t tableBytes =
            internal::ClassTableBytes(pairList->pairs.size(), pairList->functionCount);
        const std::size_t hostRamBytes = qcx::backend::DetectHostMemory().totalBytes;

        if (hostRamBytes != 0 && tableBytes > hostRamBytes)
        {
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kOutOfMemory,
                "the class-aware path's never-under class-table estimate (" +
                    std::to_string(tableBytes) + " bytes) exceeds the node's total physical RAM (" +
                    std::to_string(hostRamBytes) +
                    " bytes), and the null-budget escape hatch (memory_cap_gib = 0) skips the "
                    "admission gate that would disengage it: set a positive memory_cap_gib (the "
                    "gate then disengages the class path, the bit-identical plain path runs) or "
                    "drop the symmetry reduction"});
        }
    }

    // The mode's pair data: the FastPath builds the full contracted store
    // once; the LightPath builds the geometry-only light store at Create
    // and fills the chunk's contracted transforms during BuildFock.
    const bool lightMode = modeInfo.has_value() && modeInfo->mode == FockBuildMode::kLightPath;
    std::optional<std::vector<internal::MdPairData>> pairStore;
    std::shared_ptr<std::vector<internal::MdPairData>> lightStore;
    std::vector<internal::MdShellInput> lightShells;

    if (lightMode)
    {
        auto store = BuildLightStore(molecule, basisSet, *pairList);

        if (!store.has_value())
        {
            return std::unexpected(store.error());
        }

        auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

        if (!shells.has_value())
        {
            return std::unexpected(shells.error());
        }

        lightStore = std::make_shared<std::vector<internal::MdPairData>>(std::move(*store));
        lightShells = std::move(*shells);
    } else
    {
        auto store = internal::BuildPairData(molecule, basisSet, *pairList);

        if (!store.has_value())
        {
            return std::unexpected(store.error());
        }

        pairStore.emplace(std::move(*store));
    }

    if (!schwarz.has_value())
    {
        auto computedSchwarz = ComputeSchwarzBounds(molecule, basisSet);

        if (!computedSchwarz.has_value())
        {
            return std::unexpected(computedSchwarz.error());
        }

        schwarz = std::move(*computedSchwarz);
    }

    const std::size_t n = pairList->functionCount;

    if (coreHamiltonian.Shape()[0] != n || coreHamiltonian.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "core Hamiltonian shape mismatch"});
    }

    auto state = std::make_shared<State>();
    // _options must be in place BEFORE the neighbor-list block below: the
    // block's cutoff is preset-dependent, and State is aggregate-
    // initialized - a late assignment would silently pin every preset's
    // slack to kNormal's cutoff (caught 2026-08-23: the kTight slack was
    // void, 1e-12 instead of 1e-14).
    state->_options = effectiveOptions;
    state->_pairList = std::move(*pairList);

    if (lightMode)
    {
        state->_lightStore = std::move(lightStore);
        state->_lightShells = std::move(lightShells);
        state->_lightPathChunkPairs = modeInfo->chunkPairs;

        // The per-pair primitive-pair counts: the counts come from the
        // shells' primitive lists the chunk builder enumerates - one
        // exponent per primitive, the same product the FastPath's primPairs
        // sizes hold. (The light store's own primPairs are populated and
        // retained once the chunk pass has run, so the counts must not be
        // read from them at Create time.)
        state->_pairPrimitivePairCounts.reserve(state->_pairList.pairs.size());

        for (const ShellPairIndex& pair : state->_pairList.pairs)
        {
            state->_pairPrimitivePairCounts.push_back(
                state->_lightShells[pair.i].contractions.exponents.size() *
                state->_lightShells[pair.j].contractions.exponents.size());
        }
    } else
    {
        state->_pairStore = std::move(*pairStore);

        // The per-pair primitive-pair counts: the store entry's primPairs
        // enumeration, one (a, b) primitive pair per (primitive of i,
        // primitive of j) combination.
        state->_pairPrimitivePairCounts.reserve(state->_pairList.pairs.size());

        for (const internal::MdPairData& pair : state->_pairStore)
        {
            state->_pairPrimitivePairCounts.push_back(pair.primPairs.size());
        }
    }

    state->_schwarz = std::move(*schwarz);

    // The class-aware contraction seam engagement. A non-trivial
    // reduction (groupOrder > 1) builds the pair-class table once here and
    // captures the reduction's data - the builder is self-contained (the
    // options pointer need not outlive the Create call). A trivial
    // reduction (or null) keeps the plain path: the class enumeration
    // would be the plain enumeration, evaluated with expansion overhead.
    // On the budget path the admission gate above decides the engagement:
    // when the table's never-under estimate cannot fit the budget's
    // remaining bytes, classPath is false - the reduction is dropped and
    // the plain screened path runs (bit-identical: the reduction is never
    // consulted). The legacy null-budget path engages unconditionally -
    // already refused above when the never-under charge exceeds the
    // node's total physical RAM (the escape-hatch host-RAM guard).
    if (classPath)
    {
        // The Schwarz class-bound screening (the neighbor-list threshold,
        // the same expression the cached CSR below uses): a class pair
        // whose worst member bound product cannot clear it contains no
        // screened member quartet in any iteration, so it is dropped at
        // Create - and the surviving pairs' orbit expansions generate on
        // demand (GenerateClassPairOrbits) when the build first reaches
        // them.
        const auto classTable = BuildPairClasses(*options.symmetryReduction,
                                                 state->_pairList,
                                                 state->_schwarz,
                                                 SchwarzThreshold(state->_options.accuracy) *
                                                     internal::kNeighborListSlack);

        if (!classTable.has_value())
        {
            return std::unexpected(classTable.error());
        }

        state->_reduction = *options.symmetryReduction;
        state->_classTable = std::make_shared<PairClassTable>(*classTable);
    }

    // The cached Schwarz neighbor list: the ket candidates
    // of every bra pair under the pair-cutoff test, precomputed once. The
    // live per-quartet Schwarz check this replaces was static across SCF
    // iterations (Q_ab never changes), so this is an exact precomputation -
    // the same decision, evaluated once per Create() instead of once per
    // BuildFock call. The construction cutoff carries a multiplicative
    // slack (x0.01, niedoida's standard_j_matrix_generator.cpp pattern):
    // the CACHED list is slightly more permissive than the live check was,
    // so a borderline pair near the cutoff can never be permanently
    // excluded by list construction even if a later iteration's density
    // weight would have made it relevant. Costs a few extra entries
    // (constant factor, not asymptotic); the per-iteration density screen
    // still runs inside the loop and decides what is actually contracted.
    // The cached neighbor list itself is the shared internal::BuildNeighborList
    // (internal/fock_screen.hpp) - the GPU builder constructs
    // the identical CSR, so both builders screen identical candidate lists.
    // The LightPath keeps no Create-time CSR: the chunk loop sweeps its
    // windows during BuildFock - the per-chunk CSR peaks at the largest
    // chunk (peak = max chunk, never the sum).
    if (!lightMode)
    {
        if (!neighborListBuilt)
        {
            // The QFMM leaf-driven near-field domain:
            // the enumerated near-field candidate rows replace the global
            // sweep - the same rows the gated sweep would filter to, same
            // cutoff (fock_screen.hpp BuildLeafDrivenNeighborList).
            if (state->_options.leafNearFieldDomain.Empty())
            {
                internal::BuildNeighborList(state->_pairList,
                                            state->_schwarz,
                                            state->_options.accuracy,
                                            neighborRowOffsets,
                                            neighborIndices);
            } else
            {
                internal::BuildLeafDrivenNeighborList(state->_pairList,
                                                      state->_schwarz,
                                                      state->_options.accuracy,
                                                      state->_options.leafNearFieldDomain,
                                                      neighborRowOffsets,
                                                      neighborIndices);
            }
        }

        auto neighborPattern = qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(
            neighborRowOffsets, neighborIndices);

        if (!neighborPattern.has_value())
        {
            return std::unexpected(neighborPattern.error());
        }

        state->_neighborPattern = std::move(*neighborPattern);
    }

    // The engine tier's engagement - the in-memory ERI-value cache and
    // the engine-decorator seam that can stand under
    // it. Engaged by a positive maxCacheBytes OR by a set engineDecorator,
    // on the plain path only: the class-aware path keeps the whole
    // tier disengaged (the reduced-element-set interplay is the documented
    // follow-on), and the LightPath zeroes both at its decision (the
    // miss-assembly would read the full pair store, which the light mode
    // never materializes). The two entries are independent on purpose - the
    // budget is cap-derived and may be clamped to nothing, the decorator is
    // an explicit request that no cap derives, so a store-only run reaches
    // here with maxCacheBytes 0 and still gets its decorator (the RAM tier
    // is then a passthrough, EriBatchCache's zero budget).
    //
    // The cache budget: the caller's explicit cap, clamped to a
    // conservative half of what a FRESH DetectHostMemory() probe reports
    // available - the probe's availableBytes are a point-in-time snapshot,
    // not a reservation (memory_topology.hpp), so this is a sizing hint at
    // Create() time for the single-SCF-run union, not a guarantee that
    // stays valid mid-run (documented at the option). The clamp only ever
    // LOWERS a positive budget, so it cannot resurrect a zero one.
    if ((effectiveOptions.maxCacheBytes > 0 || effectiveOptions.engineDecorator.has_value()) &&
        !state->_classTable)
    {
        const qcx::backend::HostMemoryInfo host = qcx::backend::DetectHostMemory();
        std::size_t budget = effectiveOptions.maxCacheBytes;

        if (host.availableBytes > 0)
        {
            budget = std::min(budget, std::max<std::size_t>(1, host.availableBytes / 2));
        }

        // A decorator with no RAM grant still gets a tier: the smallest
        // budget Create accepts, which admits no block (every stored block
        // is at least one double), so the cache is a PASSTHROUGH - it holds
        // nothing and serves every request through the decorator. The
        // option's zero keeps its meaning everywhere else, the Create-time
        // estimate included (useCache is read from the option, not from
        // this local), so a store-only run is never charged cache RAM.
        if (budget == 0)
        {
            budget = 1;
        }

        // The engines replicate the eri_batch.cpp entry points minus the
        // pair rebuild (the State's pair list/store are fixed for the
        // builder's lifetime): each lane reduces to the captured State
        // members and its buffer layout (RunCacheEngineBatches does the
        // assembly, sizing, and run). The captures are raw pointers to
        // State members - stable addresses (the State is heap-allocated
        // once and never moved) and the cache is a State member that dies
        // with it, so the pointers can never dangle while the cache is
        // live.
        EriBatchCache::UnderlyingEngine fp64Engine =
            [pairList = &state->_pairList,
             pairStore = &state->_pairStore,
             maxBatchBytes = effectiveOptions.maxBatchBytes](
                const std::vector<ShellQuartet>& quartets) -> qcx::Result<EriBatch> {
            std::vector<ShellQuartet> computed;
            const auto layout = [](EriBatch& result,
                                   std::vector<internal::MdClassBatch>& batches,
                                   const std::vector<internal::MdPairData>& pairStore,
                                   std::size_t total) {
                result.values.resize(total);
                std::size_t base = 0;

                for (internal::MdClassBatch& batch : batches)
                {
                    batch.outF64 = result.values.data() + base;

                    for (const internal::MdQuartetTask& task : batch.tasks)
                    {
                        base += pairStore[task.braPair].nFuncs * pairStore[task.ketPair].nFuncs;
                    }
                }
            };
            return RunCacheEngineBatches<EriBatch>(
                *pairStore, *pairList, quartets, maxBatchBytes, computed, layout);
        };

        EriBatchCache::UnderlyingCertifiedEngine fp32Engine =
            [pairList = &state->_pairList,
             pairStore = &state->_pairStore,
             maxBatchBytes = effectiveOptions.maxBatchBytes](
                const std::vector<ShellQuartet>& quartets) -> qcx::Result<CertifiedBatch> {
#if !QcxIntegralsF32
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "this build has no fp32 pipeline (QCX_INTEGRALS_F32=OFF)"});
#else
            std::vector<ShellQuartet> computed;
            const auto layout = [](CertifiedBatch& result,
                                   std::vector<internal::MdClassBatch>& batches,
                                   const std::vector<internal::MdPairData>& pairStore,
                                   std::size_t total) {
                // assign (not resize): zero-init so stale data never leaks
                // through a pad (the fp32 lanes are read back in full,
                // including pads the batch machinery writes only partially -
                // the explicit fill pins the guarantee in code).
                result.values.assign(total, 0.f);
                result.errorBounds.assign(result.computed.size(), 0.0);
                std::size_t base = 0;

                for (internal::MdClassBatch& batch : batches)
                {
                    batch.outF32 = result.values.data() + base;
                    batch.errorBounds = result.errorBounds.data();

                    for (const internal::MdQuartetTask& task : batch.tasks)
                    {
                        base += pairStore[task.braPair].nFuncs * pairStore[task.ketPair].nFuncs;
                    }
                }
            };
            return RunCacheEngineBatches<CertifiedBatch>(
                *pairStore, *pairList, quartets, maxBatchBytes, computed, layout);
#endif
        };

        // The engine-decorator seam, called ONCE and here only:
        // the factory receives the two RAW engines just built and the pair
        // it returns is what every later request runs through. A factory
        // that composes (a RAM tier in front of a store) does so inside
        // itself, in the driver - `integrals` names no tier. A factory that
        // cannot build its decorator returns the reason and the run stops
        // rather than silently computing on the raw engines: an accepted
        // request that is dropped is the run record that can differ from
        // what ran.
        EriBatchEngineFn enginesFp64 = std::move(fp64Engine);
        CertifiedEriBatchEngineFn enginesFp32 = std::move(fp32Engine);

        if (effectiveOptions.engineDecorator.has_value())
        {
            const auto decorated = (*effectiveOptions.engineDecorator)(enginesFp64, enginesFp32);

            if (!decorated.has_value())
            {
                return std::unexpected(decorated.error());
            }

            enginesFp64 = decorated->fp64;
            enginesFp32 = decorated->fp32;
        }

        auto cache = EriBatchCache::Create(
            budget, state->_pairList, std::move(enginesFp64), std::move(enginesFp32));

        if (!cache.has_value())
        {
            return std::unexpected(cache.error());
        }

        state->_eriCache = std::make_shared<EriBatchCache>(std::move(*cache));
    }

    // Keep the original Tensor too - CoreHamiltonian() returns it by
    // reference, so a deep copy here (once, allocation in memory/) beats a
    // Tensor->Eigen conversion on every accessor call. Tensor is move-only,
    // hence Clone() instead of a copy.
    auto coreHamiltonianClone = coreHamiltonian.Clone();

    if (!coreHamiltonianClone.has_value())
    {
        return std::unexpected(coreHamiltonianClone.error());
    }

    state->_coreHamiltonianTensor = std::move(*coreHamiltonianClone);
    state->_coreHamiltonian = internal::TensorToEigen(coreHamiltonian);
    state->_modeInfo = modeInfo;
    // The batch loops' authorized concurrent slots - the fired
    // estimate's k on the budgeted fast path (info.concurrentSlots), 1 on
    // the legacy null-budget path and every k = 1 rung.
    state->_maxConcurrentSlots = modeInfo.has_value() ? modeInfo->concurrentSlots : 1;

    return DirectJkFockBuilder(std::move(state));
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> DirectJkFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    double* certifiedBoundSumOut,
    FockBuildStats* statsOut,
    const PrecisionLadderInputs* ladderInputs,
    CertifiedBudgetOutcome* budgetOut) const {
    const State& state = *_state;
    const std::size_t n = state._pairList.functionCount;
    // The LightPath chunk loop: each chunk screens its bra window over the
    // full ket prefix,
    // builds the chunk's contracted transforms into the light store,
    // assembles and contracts against the chunk store, then clears it -
    // the per-chunk arena peaks at the largest chunk (peak = max chunk,
    // never the sum; the CWA sees one peak, not a sum over chunks). The
    // FastPath runs the identical per-pass bodies once over the full
    // lists and the full store (chunkPairs == 0).
    const std::size_t chunkPairs = state._lightPathChunkPairs;
    const bool lightMode = chunkPairs != 0;
    const bool wantStats = statsOut != nullptr;
    const auto started =
        wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "density shape mismatch"});
    }

    const Eigen::MatrixXd d = internal::TensorToEigen(density);

    // The contractions read both orientations of every unordered pair block
    // (d(c,d) + d(d,c)) and the K transpose-writes assume the symmetry - a
    // documented precondition of BuildFock, checked here in Debug builds
    // (2026-08-21).
    assert(d.isApprox(d.transpose()));

    // The precision ladder (precision_policy.hpp): the
    // per-call dispatch dimension, engaged by the ladder inputs'
    // schedule. The ladder runs the plain FastPath only - the LightPath
    // and the class path return kUnimplemented (their ladder composition
    // is the documented follow-on).
    const bool ladderEngaged = ladderInputs != nullptr && ladderInputs->schedule != nullptr;

    if (ladderEngaged && lightMode)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "the precision ladder does not compose with the LightPath chunk loop yet"});
    }

    if (ladderEngaged && state._classTable != nullptr)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "the precision ladder does not compose with the class-aware reduction yet"});
    }

    // The enforcement's own route refusals: an option that cannot be
    // honoured on this route fails loudly instead of being dropped. The
    // LightPath (the chunked low-memory route) screens per chunk, so its
    // routed sum is a per-window quantity, not the build's; the engaged
    // ladder already carries its own budget contract (the schedule's
    // committed coarse sum against the delivery remainder), and running
    // both would give one build two budgets.
    if (state._options.enforceCertifiedBoundBudget && lightMode)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "the certified-bound budget enforcement does not compose with the "
                       "LightPath chunk loop yet"});
    }

    if (state._options.enforceCertifiedBoundBudget && ladderEngaged)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented,
                       "the certified-bound budget enforcement does not compose with an "
                       "engaged precision ladder (the ladder's own budget contract governs "
                       "this path)"});
    }

    // The screened canonical quartet list: Schwarz (precomputed as the
    // cached neighbor list), then the density weight (the max |D| over the
    // quartet's six blocks), then the certified-mixed-precision
    // routing per quartet. The whole chunked pass is internal::ScreenAll
    // (internal/fock_screen.hpp) - the GPU builder runs the identical
    // decisions on the identical candidate lists.
    const bool classPath = state._classTable != nullptr;
    const double densityThreshold = DensityThreshold(state._options.accuracy);
    const double mixedThreshold = MixedPrecisionThreshold(state._options.accuracy);
    // The certified fp32 lane engages on the class path exactly as on
    // the plain path: the same ScreenAll partition routes
    // each member quartet by its own density-weighted bound, and the class
    // pass accumulates the same certified-bound sum (weight(member) *
    // bound(orbit rep), below).
    // The ladder's band classification replaces the certified routing when
    // engaged (the routing gate stays the ladder-off path, bit-identical).
    // Not const: the budget enforcement below can revoke the lane for the
    // whole build (the fall-back-to-fp64 act).
    // The lane's device-governed default (a device probe's verdict,
    // 2026-09-12) resolves at the ONE point
    // ResolveCertifiedLane owns: an explicit request as written, an unset
    // one from the run's measured fp32/fp64 ratio.
    bool certifiedLane =
        ResolveCertifiedLane(state._options) && mixedThreshold > 0.0 && !ladderEngaged;

    const std::size_t nPairs = state._pairList.pairs.size();
    std::vector<internal::MdQuartetTask> fp64Quartets;
    std::vector<internal::MdQuartetTask> fp32Quartets;
    // The routing weight (max |D| over the quartet's six blocks) per
    // fp32 task, parallel to fp32Quartets - the certified sum accumulates
    // the density-weighted bounds below (2026-08-21: the
    // unweighted sum bounds the block-error sum, not the Fock error).
    std::vector<double> fp32DensityWeights;
    double certifiedBoundSum = 0.0;

    // The per-call shell-compressed max-density vector - raw
    // max |D_block| per canonical pair (rebuilt per call - the
    // incremental wrapper alternates D and delta-D). Present exactly when
    // the per-element screening flag is on; the quartet-level product gate
    // and the kernel per-element filter engage together, and a null vector
    // keeps the legacy six-block gate and no filter (the flag contract).
    std::vector<double> shellPairMaxDensity;
    std::size_t elementDrops = 0;

    // The ladder's classification consumes the per-pair maxima, so the
    // ladder builds the vector even when the per-element screening flag
    // did not (the flag still gates the kernel re-filter alone).
    if (state._options.usePerElementScreening || ladderEngaged)
    {
        shellPairMaxDensity = internal::BuildShellPairMaxDensity(d, state._pairList);
    }

    const std::vector<double>* pairMaxDensity =
        shellPairMaxDensity.empty() ? nullptr : &shellPairMaxDensity;

    // The mode's pair store: the full contracted store on the FastPath,
    // the geometry-only light store on the LightPath. The screening reads
    // only the geometry fields (the angular momenta of the l-sums), so it
    // can run against the light store before the chunk's transforms are
    // built; the assembly and contraction read the chunk's built pairs.
    const std::vector<internal::MdPairData>& pairStoreRef =
        lightMode ? *state._lightStore : state._pairStore;

    const internal::ScreeningContext context{
        state._pairList, state._schwarz, pairStoreRef, state._options, pairMaxDensity};

    // The certified-bound budget enforcement (opt-in, FastPath only): one
    // walk
    // of the same neighbor list the screening pass below sweeps, applying
    // the same two decisions (the density gate, then the routing gate),
    // accumulates the routed candidates' certified bounds and the
    // screened-out ones'. The routed sum is compared against the budget
    // the accuracy preset derives for this build; the comparison DECIDES
    // the routing. A build whose routed sum does not fit the budget
    // revokes the lane for its whole quartet set - the act is the
    // fall-back-to-fp64 branch, taken BEFORE the screening pass runs, so
    // the build IS the lane-disabled build (every candidate lands in the
    // fp64 master and is sorted and assembled exactly as it is with
    // useCertifiedMixedPrecision = false): no novel mixture is ever
    // produced, and the two agree to the builder's own run-to-run
    // reproducibility rather than bit for bit.
    //
    // What the comparison consumes is the ROUTING bound (the gate's own
    // units, the ladder's committed-sum quantity), not the larger kernel
    // bound the run record reports as certified_bound: the routing bound
    // is the only one available before the build runs, and it is the
    // smaller of the two, so the check is the most favorable one the lane
    // can be given.
    CertifiedBudgetOutcome budgetOutcome;

    if (state._options.enforceCertifiedBoundBudget)
    {
        budgetOutcome.enforced = true;

        // A lane that is off anyway (kTight's zero gate, the flag, the
        // ladder) has nothing to enforce: the check runs vacuously and
        // the sums stay at their observed-zero defaults - which is the
        // truth for a build that routed no quartet.
        if (certifiedLane)
        {
            const auto& offsets = state._neighborPattern->RowOffsets().HostView();
            const auto& indices = state._neighborPattern->Indices().HostView();
            const internal::RoutedBoundSums sums = internal::SumRoutingBounds(
                context, d, offsets, indices, densityThreshold, mixedThreshold);

            budgetOutcome.screenedHa = sums.screenedHa;
            budgetOutcome.routedHa = sums.routedHa;
            budgetOutcome.routedQuartets = sums.routedQuartets;
            budgetOutcome.budgetHa = CertifiedBoundBudget(state._options.accuracy, sums.screenedHa);
            budgetOutcome.fellBackToFp64 = sums.routedHa > budgetOutcome.budgetHa;

            if (budgetOutcome.fellBackToFp64)
            {
                certifiedLane = false;
            }
        }
    }

    // The FastPath's single screening pass over the Create-time neighbor
    // pattern. The LightPath keeps no Create-time CSR (the chunk loop
    // below sweeps each chunk's window), so a pre-branch pass would read
    // a pattern that was never built.
    if (!lightMode)
    {
        internal::ScreenAll(context,
                            d,
                            *state._neighborPattern,
                            densityThreshold,
                            mixedThreshold,
                            certifiedLane,
                            fp64Quartets,
                            fp32Quartets,
                            fp32DensityWeights);
    }

    // The per-call measurement seam (FockBuildStats): the
    // counters and the (optional) canonical (bra, ket) keys are zeroed
    // here and ACCUMULATED per pass inside runAssembly below - one pass on
    // the FastPath, one per chunk on the LightPath. The wall-time fields
    // are filled by RunPass and below.
    if (wantStats)
    {
        statsOut->fp64QuartetCount = 0;
        statsOut->fp32QuartetCount = 0;
        statsOut->eriWallTime = {};
        statsOut->contractWallTime = {};
        statsOut->mergeWallTime = {};
        statsOut->concurrentSlots = 1;
        statsOut->totalWallTime = {};
        // The cache-hit fields are filled by the assemblers below (they
        // stay zero on the class path, where the cache is disengaged).
        statsOut->cacheHitFp64QuartetCount = 0;
        statsOut->cacheHitFp32QuartetCount = 0;
        // The calibration terms accumulate per pass inside
        // runAssembly below, like the quartet counts: zeroed here, once
        // per call.
        statsOut->significantPairCount = 0;
        statsOut->primitiveProductSum = 0;

        if (statsOut->quartetKeysOut != nullptr)
        {
            statsOut->quartetKeysOut->clear();
        }
    }

    // The distinct-pair stamping (term P): one stamp per canonical
    // pair, marked with the pass's counter and counted on first touch. A
    // pair re-appears across the LightPath chunk passes (the chunk
    // windows' kets span the overlapping prefix) and across the ladder
    // band passes, so a per-pass add would overcount - the per-call stamp
    // set makes P a per-call distinct count. Sized only when stats are
    // requested (the wantStats seam stays allocation-free otherwise).
    std::vector<std::size_t> pairStamps(wantStats ? nPairs : 0, 0);
    std::size_t pairStampCounter = 0;

    // The fp32 lane's routing weights stay parallel to fp32Quartets (the
    // ScreenAll output) for the whole call and are transported into the
    // passes POSITIONALLY: the fp32 master is sorted ONCE per pass
    // below with the weights permuted alongside, and the contraction
    // reads the weight of an assembled batch task at its master position
    // (batch base + index) - no whole-call key map exists any more. The
    // fp64 lists are sorted into the same canonical emission order (the
    // per-half batch-sequence byte-identity pins).
    // The fp32 read-back scratch (one block at a time, resized per quartet).
    std::vector<double> blockScratch;

    // The ladder pass (Package B): the B_screen co-term from the
    // screening pass's own numbers, the classification budget, and the
    // per-batch band partition. The pair-maximum vector is required (the
    // classification consumes it); the ladder builds it when the
    // per-element screening flag did not.
    std::vector<internal::MdQuartetTask> ladderFp16;
    std::vector<internal::MdQuartetTask> ladderCertified;
    std::vector<internal::MdQuartetTask> ladderMixed;
    // The per-band fp32-lane density weights, parallel to each band
    // list: the positional transport of the band passes (declared next
    // to the lists so the partition's weight vectors outlive the ladder
    // block - the fp32 master stays empty under the ladder).
    std::vector<double> ladderFp16Weights;
    std::vector<double> ladderCertifiedWeights;
    std::vector<double> ladderMixedWeights;

    if (ladderEngaged)
    {
        const double bScreen =
            internal::ScreenBoundSum(context, d, *state._neighborPattern, densityThreshold);
        const EriBudgetInputs budgetInputs{
            bScreen, ladderInputs->riError, ladderInputs->densityError};
        const double classificationBudget =
            ladderInputs->schedule->ClassificationBudget(budgetInputs);
        internal::LadderPartition partition = internal::PartitionBatches(
            context, d, fp64Quartets, classificationBudget, *ladderInputs->schedule);

        // The band lists replace the certified routing's two lists: the
        // fp64 band is the reference pass; the coarse bands run as
        // separate certified passes, each carrying its OWN fp32 task list
        // and parallel weights (the positional transport - no
        // whole-call concatenation exists under the ladder; fp32Quartets
        // and fp32DensityWeights stay empty, so the class-task builds
        // below no-op on the fp32 side) (the dispatch dimension -
        // precision switches happen between class runs, never inside a
        // kernel).
        fp64Quartets = std::move(partition.fp64);
        ladderFp16 = std::move(partition.fp16);
        ladderCertified = std::move(partition.fp32Certified);
        ladderMixed = std::move(partition.fp32Mixed);
        ladderFp16Weights = std::move(partition.fp16Weights);
        ladderCertifiedWeights = std::move(partition.certifiedWeights);
        ladderMixedWeights = std::move(partition.mixedWeights);

        // The band masters sorted ONCE into the canonical emission order,
        // each with its weights permuted alongside - the per-band
        // byte-identity and weight transport of the band passes below.
        internal::SortScreenedTasks(state._pairList, fp64Quartets);
        internal::SortScreenedTasks(state._pairList, ladderFp16, ladderFp16Weights);
        internal::SortScreenedTasks(state._pairList, ladderCertified, ladderCertifiedWeights);
        internal::SortScreenedTasks(state._pairList, ladderMixed, ladderMixedWeights);

        // The schedule commits the A-PRIORI classification sum (the
        // delivery contract's quantity: <= T by construction, one
        // fraction-of-share per batch); the builder's certified-sum
        // out-parameter keeps the delivered kernel-bound sum (the
        // referee's delivered-error quantity).
        ladderInputs->schedule->CommitCoarseBoundSum(kFp16BoundScale * partition.fp16BoundSum +
                                                     partition.certifiedBoundSum +
                                                     partition.mixedBoundSum);
    }

    Eigen::MatrixXd fock = state._coreHamiltonian;

    // The contraction context: the shared state pieces, the density, and
    // the Fock matrix under accumulation - the contract helpers above. The
    // in-memory ERI cache rides along: on the class path the
    // State's cache is null by construction, so the class assembly keeps
    // the plain machinery. On the LightPath the store is the light store
    // and the cache is null (disengaged at the Create decision).
    FockContractor contractor(state._pairList,
                              pairStoreRef,
                              state._options,
                              d,
                              fock,
                              state._eriCache.get(),
                              pairMaxDensity,
                              &elementDrops,
                              state._maxConcurrentSlots);

    // The per-pass assembly and contraction: the FastPath runs it once
    // over the full lists; the LightPath chunk loop once per chunk. The
    // counts and keys accumulate, then the class pass (when the reduction
    // is present - the class tasks are the pass's input, so the chunk
    // loop can mark the orbit representatives into the pair set) or the
    // merged RunPass. An empty class task set is a no-op class pass (the
    // FastPath's guard, unchanged).
    auto runAssembly = [&](const std::vector<internal::MdQuartetTask>& f64Tasks,
                           const std::vector<internal::MdQuartetTask>& f32Tasks,
                           const std::vector<double>& f32Weights,
                           std::vector<double>& scratch,
                           const FockContractor& passContractor,
                           const std::vector<ClassRepTask>& classTasks64,
                           const std::vector<ClassRepTask>& classTasks32) -> qcx::Result<void> {
        // The per-call measurement seam: the screened task lists ARE this
        // iteration's computed-quartet set, so the counts and (when
        // requested) the canonical (bra, ket) keys are the raw recurrence
        // data; the wall-time fields are filled by RunPass and below. The
        // calibration terms accumulate here over the same
        // lists: G sums each task's primitive-pair product weight (the
        // bra pair's prim-pair count times the ket pair's - the pair
        // counts were fixed at Create, so the FastPath, the chunk loop,
        // the ladder bands and the class pass all use one per-pair
        // enumeration), and P counts the distinct pairs the call's tasks
        // touch (stamped per pass; the accumulator above keeps the
        // counter across passes, so P is a per-call distinct count, not a
        // per-pass one).
        if (wantStats)
        {
            statsOut->fp64QuartetCount += f64Tasks.size();
            statsOut->fp32QuartetCount += f32Tasks.size();

            const std::vector<std::size_t>& pairPrimitivePairCounts =
                state._pairPrimitivePairCounts;
            ++pairStampCounter;

            const auto accumulateTermCounters =
                [&](const std::vector<internal::MdQuartetTask>& tasks) {
                    for (const internal::MdQuartetTask& task : tasks)
                    {
                        statsOut->primitiveProductSum += pairPrimitivePairCounts[task.braPair] *
                                                         pairPrimitivePairCounts[task.ketPair];

                        if (pairStamps[task.braPair] != pairStampCounter)
                        {
                            pairStamps[task.braPair] = pairStampCounter;
                            ++statsOut->significantPairCount;
                        }

                        if (pairStamps[task.ketPair] != pairStampCounter)
                        {
                            pairStamps[task.ketPair] = pairStampCounter;
                            ++statsOut->significantPairCount;
                        }
                    }
                };
            accumulateTermCounters(f64Tasks);
            accumulateTermCounters(f32Tasks);

            if (statsOut->quartetKeysOut != nullptr)
            {
                std::vector<std::size_t>& keys = *statsOut->quartetKeysOut;
                keys.reserve(f64Tasks.size() + f32Tasks.size());

                for (const internal::MdQuartetTask& task : f64Tasks)
                {
                    keys.push_back(task.braPair * nPairs + task.ketPair);
                }

                for (const internal::MdQuartetTask& task : f32Tasks)
                {
                    keys.push_back(task.braPair * nPairs + task.ketPair);
                }
            }
        }

        // The reduction rides with the class table (both are set together
        // in Create), but the guard names it explicitly: the class pass
        // below dereferences both optionals, and the unchecked-access
        // check cannot trace the coupling through the classPath flag
        // alone.
        if (classPath && state._reduction.has_value())
        {
            // The class-aware contraction with the
            // certified fp32 lane engaged. The screened set maps to
            // the per-orbit member lists (BuildClassRepTasks) PER
            // PRECISION; every orbit rep with screened members is
            // assembled ONCE per precision (the petite-list saving holds
            // per lane - one ERI block per orbit instead of one per member
            // quartet), and the class pass expands each rep block per
            // screened member: fp64 members read the rep's fp64 block,
            // fp32 members the rep's certified fp32 block, and the
            // certified sum accumulates weight(member) * bound(orbit rep).
            // The contraction result is bit-identical to the plain path's
            // per (quartet, block) pair - the equivalence pin the tests
            // check. The per-batch assembly and contraction live inside
            // RunClassPass.
            if (classTasks64.empty() && classTasks32.empty())
            {
                return qcx::Result<void>{};
            }

            auto pass = passContractor.RunClassPass(classTasks64,
                                                    classTasks32,
                                                    *state._classTable,
                                                    *state._reduction,
                                                    f32Weights,
                                                    scratch,
                                                    certifiedBoundSum);

            if (!pass.has_value())
            {
                return qcx::Result<void>{std::unexpected(pass.error())};
            }
        } else
        {
            // The plain pass over both precisions (restructured per-half,
            // per-batch): each half
            // assembles and contracts one maxBatchBytes-bounded batch at a
            // time.
            auto pass = passContractor.RunPass(
                f64Tasks, f32Tasks, f32Weights, scratch, certifiedBoundSum, statsOut);

            if (!pass.has_value())
            {
                return qcx::Result<void>{std::unexpected(pass.error())};
            }
        }

        return qcx::Result<void>{};
    };

    if (lightMode)
    {
        // The chunk loop: the bra window [rowStart, rowEnd) is
        // swept over the FULL ket prefix (ket <= bra, the canonical
        // enumeration - the chunk's kets span the prefix, not a bounded
        // window). The window CSR is the chunk's own; its capacity peaks
        // at the largest chunk and never reallocates past it.
        std::vector<std::size_t> neighborRowOffsets;
        std::vector<std::size_t> neighborIndices;
        std::vector<std::size_t> chunkPairSet;
        // The stamp vector: the chunk pair set (window bras, distinct
        // kets, class orbit reps) in one pass over the pair space.
        std::vector<std::size_t> stamps(nPairs, 0);
        std::size_t stampCounter = 0;
        const std::size_t numChunks = internal::LightChunkCount(nPairs, chunkPairs);

        for (std::size_t c = 0; c < numChunks; ++c)
        {
            const std::size_t rowStart = internal::LightChunkRowStart(chunkPairs, c);
            const std::size_t rowEnd = internal::LightChunkRowEnd(chunkPairs, c, nPairs);

            // The chunk's screened task lists (the containers are reused
            // across chunks - the capacities persist).
            fp64Quartets.clear();
            fp32Quartets.clear();
            fp32DensityWeights.clear();
            internal::BuildNeighborList(state._pairList,
                                        state._schwarz,
                                        state._options.accuracy,
                                        rowStart,
                                        rowEnd,
                                        neighborRowOffsets,
                                        neighborIndices);
            internal::ScreenAll(context,
                                d,
                                rowStart,
                                rowEnd,
                                neighborRowOffsets,
                                neighborIndices,
                                densityThreshold,
                                mixedThreshold,
                                certifiedLane,
                                fp64Quartets,
                                fp32Quartets,
                                fp32DensityWeights);

            // The chunk's masters sorted into the canonical emission
            // order ONCE (the fp32 weights permuted alongside - the
            // positional transport): every downstream walk of the chunk
            // (the class-task builds below, which record the fp32
            // members' master positions, and the pass) consumes the
            // sorted lists.
            internal::SortScreenedTasks(state._pairList, fp64Quartets);
            internal::SortScreenedTasks(state._pairList, fp32Quartets, fp32DensityWeights);

            // The chunk pair set: the window's bra rows (the assembler
            // reads the store entries of every task pair; the surviving
            // tasks' bras are a subset of the window), the tasks'
            // distinct kets, and the class orbit representatives - the
            // petite-list canonical quartets can sit anywhere in the
            // pair space.
            chunkPairSet.clear();
            ++stampCounter;

            for (std::size_t bra = rowStart; bra < rowEnd; ++bra)
            {
                stamps[bra] = stampCounter;
            }

            auto markTaskPairs = [&](const std::vector<internal::MdQuartetTask>& tasks) {
                for (const internal::MdQuartetTask& task : tasks)
                {
                    stamps[task.braPair] = stampCounter;
                    stamps[task.ketPair] = stampCounter;
                }
            };
            markTaskPairs(fp64Quartets);
            markTaskPairs(fp32Quartets);

            std::vector<ClassRepTask> classTasks64;
            std::vector<ClassRepTask> classTasks32;

            if (classPath && state._reduction.has_value())
            {
                classTasks64 = BuildClassRepTasks(
                    *state._classTable, *state._reduction, state._pairList, fp64Quartets, nPairs);
                // The fp32 call records each screened member's master
                // position (the sorted chunk master; the positional
                // member-weight transport).
                classTasks32 = BuildClassRepTasks(*state._classTable,
                                                  *state._reduction,
                                                  state._pairList,
                                                  fp32Quartets,
                                                  nPairs,
                                                  true);
                auto markRepPairs = [&](const std::vector<ClassRepTask>& tasks) {
                    for (const ClassRepTask& task : tasks)
                    {
                        const ClassOrbit& orbit = state._classTable->classPairs[task.classPairIndex]
                                                      .orbits[task.orbitIndex];
                        stamps[orbit.repBraPair] = stampCounter;
                        stamps[orbit.repKetPair] = stampCounter;
                    }
                };
                markRepPairs(classTasks64);
                markRepPairs(classTasks32);

                // The chunk's class tasks sorted into the canonical rep
                // emission order (SortClassRepTasks, the class-path
                // sort) - the class pass walks the sorted lists.
                SortClassRepTasks(*state._classTable, state._pairList, classTasks64);
                SortClassRepTasks(*state._classTable, state._pairList, classTasks32);
            }

            for (std::size_t p = 0; p < nPairs; ++p)
            {
                if (stamps[p] == stampCounter)
                {
                    chunkPairSet.push_back(p);
                }
            }

            // The chunk's contracted transforms into the capacity-
            // preserved light store and the chunk-scoped contractor (the
            // ERI cache is disengaged on the LightPath - the miss-
            // assembly would read the full pair store, which the light
            // mode never materializes). The chunk's routing weights ride
            // the sorted fp32 master (fp32DensityWeights - the positional
            // positional transport; no per-call key map exists).
            internal::BuildChunkPairData(
                *state._lightStore, state._lightShells, state._pairList, chunkPairSet);
            FockContractor chunkContractor(state._pairList,
                                           pairStoreRef,
                                           state._options,
                                           d,
                                           fock,
                                           nullptr,
                                           pairMaxDensity,
                                           &elementDrops);

            auto pass = runAssembly(fp64Quartets,
                                    fp32Quartets,
                                    fp32DensityWeights,
                                    blockScratch,
                                    chunkContractor,
                                    classTasks64,
                                    classTasks32);

            if (!pass.has_value())
            {
                return std::unexpected(pass.error());
            }

            // The chunk teardown: the transforms cleared, the arena
            // capacities preserved for the next chunk (the reused
            // containers are not Reservations - the CWA sees one peak,
            // not a sum over chunks).
            internal::ClearChunkPairData(*state._lightStore, chunkPairSet);
        }
    } else
    {
        // The masters sorted into the canonical emission order once -
        // under the ladder the lists were already sorted (and the fp32
        // lists emptied) inside the ladder branch above.
        if (!ladderEngaged)
        {
            internal::SortScreenedTasks(state._pairList, fp64Quartets);
            internal::SortScreenedTasks(state._pairList, fp32Quartets, fp32DensityWeights);
        }

        // The class tasks over the full screened lists (the class pass
        // guard, unchanged).
        std::vector<ClassRepTask> classTasks64;
        std::vector<ClassRepTask> classTasks32;

        if (classPath && state._reduction.has_value())
        {
            classTasks64 = BuildClassRepTasks(
                *state._classTable, *state._reduction, state._pairList, fp64Quartets, nPairs);
            // The fp32 call records each screened member's master
            // position (the sorted fp32 master; the positional
            // member-weight transport). Under the ladder the fp32 master
            // is empty (the bands carry their own lists), so this build
            // no-ops there exactly as before.
            classTasks32 = BuildClassRepTasks(
                *state._classTable, *state._reduction, state._pairList, fp32Quartets, nPairs, true);
        }

        if (ladderEngaged)
        {
            // The per-band passes: the fp64 band, then the three coarse
            // bands as separate certified passes with their own sum
            // accumulators (the dispatch dimension - precision switches
            // happen between class runs). The committed ladder sum scales
            // the fp16 band's delivered fp32-lane bounds by
            // kFp16BoundScale (the fp16 band's certified epsilon).
            double fp64Sum = 0.0;
            double fp16Sum = 0.0;
            double certifiedSum = 0.0;
            double mixedSum = 0.0;

            auto runBandPass = [&](const std::vector<internal::MdQuartetTask>& f64Tasks,
                                   const std::vector<internal::MdQuartetTask>& f32Tasks,
                                   const std::vector<double>& f32Weights,
                                   double& bandSum) -> qcx::Result<void> {
                const double before = certifiedBoundSum;
                auto pass =
                    runAssembly(f64Tasks, f32Tasks, f32Weights, blockScratch, contractor, {}, {});

                if (!pass.has_value())
                {
                    return pass;
                }

                bandSum = certifiedBoundSum - before;
                return {};
            };

            auto pass64 = runBandPass(fp64Quartets, {}, {}, fp64Sum);

            if (!pass64.has_value())
            {
                return std::unexpected(pass64.error());
            }

            auto pass16 = runBandPass({}, ladderFp16, ladderFp16Weights, fp16Sum);

            if (!pass16.has_value())
            {
                return std::unexpected(pass16.error());
            }

            auto passCertified =
                runBandPass({}, ladderCertified, ladderCertifiedWeights, certifiedSum);

            if (!passCertified.has_value())
            {
                return std::unexpected(passCertified.error());
            }

            auto passMixed = runBandPass({}, ladderMixed, ladderMixedWeights, mixedSum);

            if (!passMixed.has_value())
            {
                return std::unexpected(passMixed.error());
            }

            // The certified-sum contract generalized to the bands: the
            // fp16 band's delivered fp32-lane bounds scale by the fp16
            // band's certified epsilon; the fp64 band contributes no
            // bound (the two-pass convention).
            certifiedBoundSum = kFp16BoundScale * fp16Sum + certifiedSum + mixedSum;
        } else
        {
            // The class tasks sorted into the canonical rep emission
            // order (SortClassRepTasks, the class-path sort): the
            // pass below walks the sorted lists. The sort sits next to
            // its only consumption - the ladder branch above does not use
            // the class tasks (its bands run the plain pass), and without
            // the class table there is nothing to sort (the class table
            // is null on the plain path).
            if (classPath)
            {
                SortClassRepTasks(*state._classTable, state._pairList, classTasks64);
                SortClassRepTasks(*state._classTable, state._pairList, classTasks32);
            }

            auto pass = runAssembly(fp64Quartets,
                                    fp32Quartets,
                                    fp32DensityWeights,
                                    blockScratch,
                                    contractor,
                                    classTasks64,
                                    classTasks32);

            if (!pass.has_value())
            {
                return std::unexpected(pass.error());
            }
        }
    }

    // Both paths accumulate their certified-bound partial through the same
    // variable (the class path through its pass's own partials, the
    // ladder through the band-scaled committed sum), so the single write
    // below reports both correctly.
    if (certifiedBoundSumOut != nullptr)
    {
        *certifiedBoundSumOut = certifiedBoundSum;
    }

    // The enforcement's outcome rides the same single write (the sums
    // above are the enforcement's own; a build that never ran the check
    // leaves the outcome's zero defaults, which is what its `enforced`
    // flag marks as not-measured).
    if (budgetOut != nullptr)
    {
        *budgetOut = budgetOutcome;
    }

    // The bridge handles the tensor allocation, the copy (i-outer, so the
    // row-major Tensor destination is written contiguously), and
    // MarkHostDirty; the stats timing stays after the conversion so the
    // metric keeps covering the whole build (the copy was inside the timed
    // region before consolidation).
    auto tensor = internal::EigenToTensor(fock);

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    if (wantStats)
    {
        statsOut->totalWallTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started);
        // The drop count accumulated through the contractor (the
        // serial and parallel paths both land here). Zero by construction
        // when the per-element screening flag is off (the flag contract).
        statsOut->elementDrops = elementDrops;
    }

    return std::move(*tensor);
}

const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& DirectJkFockBuilder::CoreHamiltonian()
    const {
    // Create() engages the optional before it returns (the core-Hamiltonian
    // clone moves in during the factory), so a live builder always has a
    // tensor here - the dereference cannot be reached with nullopt.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return *_state->_coreHamiltonianTensor;
}

const EriCacheStats* DirectJkFockBuilder::CacheStats() const noexcept {
    // The stats live inside the cache object (a State member - heap-
    // allocated, stable address), so the returned pointer is valid for the
    // builder's lifetime. Null when the cache was never engaged.
    return _state->_eriCache ? &_state->_eriCache->Stats() : nullptr;
}

} // namespace qcx::integrals
