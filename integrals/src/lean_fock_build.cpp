// The lean direct Fock builder (lean_fock_build.hpp - the 2026-09-08
// lean-direct reframe): a Schwarz-only recompute loop over ONE immutable
// 24-byte record per canonical shell pair - no per-pair task records, no
// retained neighbor CSR, no admission record, no workspace budget. Each
// BuildFock call walks the records in bra-pair (row) order, screens each
// bra row's full ket prefix IN THE LOOP (Q_MN * Q_LS against
// SchwarzThreshold(accuracy) x kNeighborListSlack - the existing
// machinery's neighbor-list cutoff verbatim, fock_screen.hpp), flushes the
// surviving shell quartets through the existing class-batch machinery
// (md_batch.hpp) into bounded per-thread tiles, and contracts every quartet
// immediately with the existing serial kernels' symmetry arithmetic
// (fock_contract_kernel.hpp - J on the bra and ket pair blocks with both
// density orientations, the four K targets with the density permuted per
// block) into per-thread Fock accumulators, reduced once per call.
// Symmetry: the row walk enumerates each unordered pair-pair exactly ONCE
// as its canonical cell (row = the larger pair index, ket <= row - the
// exact cells of the machinery's BuildNeighborList CSR), so every unique
// shell quartet is computed exactly once and its 8-fold symmetry-related
// Fock contributions accumulate in that one pass; the contraction kernel is
// the machinery's verbatim body. No nested parallelism: the parallel
// dimension is the row windows (round-robin rows, one window per chunk),
// and every flush runs single-threaded (regionThreads = 1).
//
// Point-group symmetry reaches this path in two depths (both off by default):
// the
// per-pair CLASSIFICATION packed into each record lets the walk drop cells
// whose block a group element proves exactly zero (lean_pair_record.hpp),
// and - the petite-list increment - the per-pair ORBIT ACTION
// (lean_orbit_action.hpp, options.symmetryOrbitExpansion) lets the walk
// visit one cell per orbit and the contraction expand that representative's
// block over the orbit's members. The first drops exact zeros and cannot
// move a byte; the second replaces a member's evaluation with the
// representative's and therefore does (the ERI's symmetry is exact in exact
// arithmetic, not in IEEE double - a 2026-09-12 amendment). The
// two are independent and neither is engaged unless the caller asks.

#include "qcx/integrals/lean_fock_build.hpp"

#include "internal/fock_contract_kernel.hpp"
#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "internal/lean_orbit_action.hpp"
#include "internal/lean_pair_record.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_kernel_span.hpp"
#include "internal/tensor_eigen_bridge.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals {

namespace {

// The per-flush distinct-pair cap of the lean row walk: a flush's shell
// pairs (its bra rows plus the surviving kets) fill the worker's chunk
// pair data, so this cap bounds the per-thread store-payload arena to ~96 x
// the largest pair payload while keeping each flush large enough to
// amortize the batch assembly. The worker store vector itself is the
// full-size geometry-only template copy (per-thread, created once per
// BuildFock call); only the flush's pairs carry contracted transforms.
constexpr std::size_t kLeanFlushPairCap = 96;

// The generous margin of the Create-time last-resort memory check:
// the builder refuses only when the simple-actuals
// estimate EXCEEDS the configured ceiling by more than this factor -
// headroom for mechanisms that can use extra free memory. No razor
// margins anywhere on the lean path.
constexpr double kRefusalSlack = 1.5;

// The SCF-side O(N^2) matrix count of the refusal estimate: the live n x n
// matrix state of the CALLER, at the worst case the lean builder serves -
// the UHF split adapter (driver/src/run_driver.cpp MakeLeanUhfFockBuilder,
// two half-builders driven from the iteration loop of scf/src/uhf.cpp). The
// count is a sum of named parts, never a round number, and it is an UPPER
// BOUND of that state by construction (a capability must
// never be gated on a never-under bound).
//
// The value this replaced (a flat 16) was read off a decomposition that does
// not add up: "core, overlap, density, Fock, the per-spin halves and the
// DIIS history" needs ~20 n^2 for RHF ALONE, and the measured RHF live set
// is ~28 n^2 at kDiisHistoryLimit = 8 - so the charge sat 12 n^2 (2.2 GiB at
// n = 4,974) UNDER the residency it claimed to bound. The DIIS half was the
// larger error: RHF keeps ONE extrapolator (Fock + error histories,
// rhf.cpp:525) and UHF keeps TWO, one per spin (uhf.cpp:985-988), and the
// joint-system form (JointSystemDiis, scf_common.hpp) keeps the same four
// vectors under one shared coefficient vector - so 4 x kDiisHistoryLimit
// either way, and the histories are live for the WHOLE iteration the builder
// runs inside (they are the acceleration state, not a transient).
//
// DRIFT: kDiisHistoryLimit is scf's own constant (8 - rhf.cpp:161,
// uhf.cpp:257). This module sits BEFORE scf in the module DAG and cannot
// include it, so the number is restated here; a longer scf window is a
// silent under-charge and the two must move together.
constexpr std::size_t kScfDiisHistoryLimit = 8;

// One DIIS history entry per spin channel: the Fock matrix and its error
// vector (the two extrapolators' vectors, or the joint form's two per-spin
// pairs).
constexpr std::size_t kScfDiisVectorCount = 4;

// The enumerated base set of the iteration's live n x n matrices at the
// moment its Fock builder runs - the UHF split adapter's shape, which is the
// worst case the lean builder serves (driver/src/run_driver.cpp
// MakeUhfSplitFockBuilder, whose three half-builds per iteration are:
//   (a) the calling loop's own persistent state, nine (uhf.cpp:691-696,
//       1038-1045, 1501): the core Hamiltonian, the overlap, the
//       orthogonalizer X, the two spin densities, the two coefficient
//       matrices and the two final-Fock captures;
//   (b) the adapter's tensors still live while the LAST half-build of the
//       iteration runs, five (run_driver.cpp:2251-2320): the
//       0.5 x (dAlpha + dBeta) input tensor, the finished Coulomb result,
//       the alpha density tensor, the finished alpha-exchange result and the
//       beta density tensor - each of them an n x n host tensor the adapter
//       holds across the following build;
//   (c) the running half-builder's own output, two: its working Fock matrix
//       and the tensor it returns;
//   (d) one of headroom for the caller's expression temporaries at the
//       fAlpha/fBeta formation.
// 9 + 5 + 2 + 1 = 17. Every part is named and the count is rounded up rather
// than down wherever a term is arguable. The RHF seam is smaller (it builds
// once per iteration: rhf.cpp's eight persistent matrices plus the builder's
// two), so 17 covers it.
//
// SCOPE, stated so the number is not read as proving more than it does: this
// is the SCF loop's matrix state. A caller that composes the builder into a
// larger ledger of its own (the Kohn-Sham split composition, the driver's
// memory accounting) charges that ledger itself.
constexpr std::size_t kScfBaseMatrixCount = 17;

// The two parts summed: 17 + 4 x 8 = 49 n^2 at the current DIIS window.
constexpr std::size_t kScfMatrixCount =
    kScfBaseMatrixCount + kScfDiisVectorCount * kScfDiisHistoryLimit;

// The window oversubscription factor of the auto row-window count (the
// scheduling answer): with auto chunks the row space is cut into
// this many windows per thread, and the window loop is scheduled
// dynamically (chunk = one window), so the per-row cost variance -
// screening density and the ket prefix the row scans - averages out over
// the run instead of landing in one thread's single window, which is what
// the previous one-window-per-thread static split forced. Deliberately the
// floor of the measured 8-16x band: enough windows for the dynamic
// schedule to rebalance, while the per-window scratch (one n x n Fock
// accumulator plus one geometry-store copy per window) stays at eight
// teams' worth. The window count never moves the k = 1 result: that path
// runs exactly one window on the calling thread.
constexpr std::size_t kWindowOversubscription = 8;

// The WAVE-WIDTH FACTOR of the generation-grouped reduction (the
// accumulator-residency bound): the row windows run in waves of this many
// times the region's thread count, and a wave's accumulators fold into the
// running Fock when the wave joins, so at most the wave width of them is
// ever live. The factor is deliberately 2 rather than 1: with the auto
// plan's oversubscription the wave still hands every thread two
// independently scheduled windows to rebalance against (the mechanism the
// oversubscription exists for - whose measured benefit on the alkane
// fixture is an UNCHANGED call wall - while
// its residency cost was the largest term of every lean run: 40 x 8n^2 =
// 7.37 GiB at n = 4,974 against 10 x 8n^2 = 1.84 GiB at this factor).
constexpr std::size_t kWindowWaveFactor = 2;

// The ONE engagement test this path's two point-group mechanisms share
// (measured 2026-09-13): a reduction is trivial when it
// cannot reduce anything. That is NOT "the group is C1" and NOT "the order
// is 1" - the reduction's own isTrivial flag carries the group's action on
// the CANONICAL SHELL PAIRS, which the reduction builder sets from the
// realized action (scf/src/internal/symmetry_blocks.cpp), so a non-C1
// sign-only group (every atom on the symmetry element: hocl/Cs, a planar
// molecule's out-of-plane mirror) reads trivial here. The order test stays
// as a second, independent guard because a hand-built or default reduction
// need not have gone through that builder.
//
// Both mechanisms below are gated on it and neither is weakened by it: the
// classification's masks drop cells a sign-only element proves exactly
// zero, so treating the group as trivial costs those drops and moves no
// byte (the masks are throughput-only by their own contract,
// lean_fock_build.hpp), and the orbit action's tables are pure overhead
// when every orbit is a singleton.
bool IsTrivialReduction(const qcx::integrals::SymmetryReduction* reduction) noexcept {
    return reduction == nullptr || reduction->isTrivial || reduction->groupOrder <= 1;
}

// The bit width of the per-pair point-group classification and the widest
// group the point-group mechanisms carry live in lean_pair_record.hpp
// (kSymmetryMaskBits) - the record's two mask bytes, the shared reduction
// shape check, and the orbit action's own tables all read that one constant.

// One canonical pair's point-group classification: the set of group
// elements under which BOTH of the pair's shells are mapped onto
// themselves function-for-function and carry a CONSTANT sign, split by
// that sign (bit g of plus = the pair's two shells are fixed by element g
// and the pair's sign product under it is +1; bit g of minus = fixed with
// the product -1). An element that moves any function of either shell out
// of its shell - the generic case for these groups, since a non-identity
// element permutes atoms - leaves both bits clear, and so does a shell
// whose functions carry mixed signs under the element.
struct LeanPairSymmetryMask {
    std::uint32_t plus = 0; ///< Elements fixing both shells with product sign +1.
    std::uint32_t minus = 0; ///< Elements fixing both shells with product sign -1.
};

// Derives the per-pair classification from the reduction's own
// signed-permutation arrays.
//
// A canonical cell (row pair, ket pair) has an ERI block that is EXACTLY
// zero when one group element g fixes both pairs - maps every function of
// all four shells onto itself - and the four per-shell signs multiply to
// -1 at every position: the invariance relation (g mu g nu | g lambda g
// sigma) = s(mu)s(nu)s(lambda)s(sigma) (mu nu | lambda sigma) then reads
// A = -A element by element. The two pairs' signs must therefore be
// OPPOSITE under g, and each pair's sign is the product of its two
// shells' signs - which is what the masks encode. The requirement is
// deliberately exact and narrow: when g moves a function of some shell
// (the atom permutation of a mirror that swaps equivalent atoms, say) the
// relation links two DIFFERENT non-zero integrals and nothing may be
// dropped - that boundary is the measured counterexample in
// lean_point_group_test.cpp, not an oversight.
// \param reduction The reduction (its arrays are the classification's
// only input).
// \param pairList The canonical pairs the masks align with (mask p belongs
// to pair index p).
// \returns One mask per canonical pair, or kInvalidArgument for a
// malformed reduction or a group wider than kSymmetryMaskBits.
qcx::Result<std::vector<LeanPairSymmetryMask>> DerivePairSymmetryMasks(
    const SymmetryReduction& reduction, const ShellPairList& pairList) {
    const std::size_t n = pairList.functionCount;
    const std::size_t nShells = pairList.shells.size();
    const std::size_t order = reduction.groupOrder;

    // The shared shape checks (lean_pair_record.hpp): the eight-element
    // width, the row spans, the index ranges, the signs, and the identity
    // row the row walk's zero test rests on. The orbit action's own build
    // calls the same function, so the two consumers of one reduction can
    // never disagree about what a valid reduction is.
    auto shape = internal::ValidateSymmetryReductionShape(reduction, n);

    if (!shape.has_value())
    {
        return std::unexpected(shape.error());
    }

    // The per-shell aggregation: for every element, whether the shell is
    // fixed function-for-function and, if so, whether all its functions
    // carry the same sign.
    std::vector<std::uint32_t> shellPlus(nShells, 0u);
    std::vector<std::uint32_t> shellMinus(nShells, 0u);

    for (std::size_t s = 0; s < nShells; ++s)
    {
        const ShellInfo& shell = pairList.shells[s];
        const std::size_t first = shell.functionOffset;
        const std::size_t count = ShellFunctionCount(shell);

        for (std::size_t g = 0; g < order; ++g)
        {
            bool fixed = true;
            bool plus = true;
            bool minus = true;

            for (std::size_t k = 0; k < count; ++k)
            {
                const std::size_t f = first + k;
                fixed = fixed && (reduction.permutation[g][f] == f);
                plus = plus && (reduction.sign[g][f] == 1);
                minus = minus && (reduction.sign[g][f] == -1);
            }

            const std::uint32_t bit = 1u << g;

            if (fixed && plus)
            {
                shellPlus[s] |= bit;
            }

            if (fixed && minus)
            {
                shellMinus[s] |= bit;
            }
        }
    }

    // The pair masks: a pair's sign under g is the product of its two
    // shells' signs, so both shells must be fixed and both pure.
    std::vector<LeanPairSymmetryMask> masks(pairList.pairs.size());

    for (std::size_t p = 0; p < pairList.pairs.size(); ++p)
    {
        const std::size_t i = pairList.pairs[p].i;
        const std::size_t j = pairList.pairs[p].j;

        masks[p].plus = (shellPlus[i] & shellPlus[j]) | (shellMinus[i] & shellMinus[j]);
        masks[p].minus = (shellPlus[i] & shellMinus[j]) | (shellMinus[i] & shellPlus[j]);
    }

    return masks;
}

// The packed function-pair triangle offset of the shell pair whose first
// functions sit at \p firstA <= \p firstB: the PairIndexOf upper-triangle
// formula over the function count (diagonal included). The record's
// packed-AO-pair offset (the pair's position in the packed function-pair
// triangle of the whole basis).
// \param firstA The first function index of the lower shell.
// \param firstB The first function index of the higher shell.
// \param functionCount The total basis-function count.
// \returns The packed triangle offset (uint32 by record layout; the
// offset exceeds 2^32 only past ~92k basis functions - far beyond any
// build this path can address in memory).
std::uint32_t PackedFunctionPairOffset(std::size_t firstA,
                                       std::size_t firstB,
                                       std::size_t functionCount) noexcept {
    const std::size_t offset = firstA * (2 * functionCount - firstA + 1) / 2 + (firstB - firstA);

    return static_cast<std::uint32_t>(offset);
}

// The fp64 element mass of one raw quartet's ERI block - the product of
// the four shells' function counts (the same block mass the batch layout
// accumulates over the canonical task pairs: role swaps never change the
// product).
// \param pairList The shell list (function counts per shell).
// \param bra The bra-pair record (its shells index the pair list).
// \param ket The ket-pair record.
// \returns The block element count.
std::size_t QuartetElementCount(const ShellPairList& pairList,
                                const internal::LeanShellPairRecord& bra,
                                const internal::LeanShellPairRecord& ket) noexcept {
    const std::size_t nI = ShellFunctionCount(pairList.shells[bra.shellM]);
    const std::size_t nJ = ShellFunctionCount(pairList.shells[bra.shellN]);
    const std::size_t nK = ShellFunctionCount(pairList.shells[ket.shellM]);
    const std::size_t nL = ShellFunctionCount(pairList.shells[ket.shellN]);

    return nI * nJ * nK * nL;
}

// Contracts one computed quartet block into the target Fock matrix with
// the machinery's verbatim dispatch (fock_build.cpp AccumulateBlock): the
// AVX2 copy after the runtime cpuid check unless the scalar copy is
// forced (the fallback pin), both copies bit-identical by contract.
// \param block The quartet's computed fp64 ERI block.
// \param quartet The quartet's canonical-role shells.
// \param pairBra The canonical bra pair index (into pairList).
// \param pairKet The canonical ket pair index.
// \param context The contract context (density, target Fock, mode flags).
// \param forceScalarContract Force the scalar kernel (the options pin).
void ContractBlock(const double* block,
                   const ShellQuartet& quartet,
                   std::size_t pairBra,
                   std::size_t pairKet,
                   const internal::FockContractContext& context,
                   bool forceScalarContract) {
    static const bool kAvx2 = internal::FockAvx2Available();

    if (kAvx2 && !forceScalarContract)
    {
        internal::AccumulateBlockAvx2(block, quartet, pairBra, pairKet, context);
    } else
    {
        internal::AccumulateBlockKernel<internal::AccumulateBlockScalarTag>(
            block, quartet, pairBra, pairKet, context);
    }
}

// The resident contracted payload of one store entry (the retention band's
// byte measure): the capacity - not the size - of every container the pair
// builder fills, because capacity is what a release gives back and what the
// reuse path re-fits. The per-axis E tables ride inside each MdPrimPair as
// vectors, so they are charged by the machinery's own E-table formula
// (footprint.hpp ETableBytes: 24 B per (la+1)(lb+1)(la+lb+1) cell per
// primitive pair).
// \param pair The store entry.
// \returns The resident payload bytes.
std::size_t PairPayloadBytes(const internal::MdPairData& pair) {
    std::size_t bytes = pair.braTransform.capacity() * sizeof(double) +
                        pair.primPairs.capacity() * sizeof(internal::MdPrimPair);
    bytes += internal::ETableBytes(pair.la, pair.lb, pair.primPairs.size());
    bytes += pair.braWeights.capacity() * internal::kVectorHeaderBytes;
    bytes += pair.ketTransforms.capacity() * internal::kVectorHeaderBytes;

    for (const std::vector<double>& weights : pair.braWeights)
    {
        bytes += weights.capacity() * sizeof(double);
    }

    for (const std::vector<double>& transform : pair.ketTransforms)
    {
        bytes += transform.capacity() * sizeof(double);
    }

    return bytes;
}

// Releases the contracted payload of one store entry and returns the bytes
// released: the transforms, the contraction weights and the primitive pairs
// (their per-axis E tables with them) go back to the allocator as empty
// containers, which is also the state the pair builder's CALCULATED fields
// start from (BuildChunkPairData's not-built marker is the empty
// braTransform). The geometry fields - la/lb, the spherical flags and the
// centers - survive: they belong to the skeleton the window's store was
// copied from, and BuildChunkPairData never rewrites them.
// \param pair The store entry.
// \returns The payload bytes released.
std::size_t ReleasePairPayload(internal::MdPairData& pair) {
    const std::size_t bytes = PairPayloadBytes(pair);
    std::vector<internal::MdPrimPair>().swap(pair.primPairs);
    std::vector<std::vector<double>>().swap(pair.braWeights);
    std::vector<double>().swap(pair.braTransform);
    std::vector<std::vector<double>>().swap(pair.ketTransforms);
    return bytes;
}

// The loose per-pair payload bound of one flush arena: the largest pair's
// FastPath store payload (PairStoreBytesPerPair minus the geometry-only
// MdPairData the light template already carries) - the same bound the
// machinery's MaxPairPayload derives, used here by the Create-time
// last-resort estimate only.
// \param molecule Molecule providing the atom coordinates (Bohr).
// \param basisSet Basis set (the primitive-count authority).
// \param pairList The canonical shell pairs.
// \returns The largest per-pair payload in bytes (0 on an empty list).
std::size_t MaxPairPayload(const qcx::molecule::Molecule& molecule,
                           const qcx::basisset::BasisSet& basisSet,
                           const ShellPairList& pairList) {
    std::size_t maxPayload = 0;

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        const ShellInfo& a = pairList.shells[pair.i];
        const ShellInfo& b = pairList.shells[pair.j];
        const std::size_t nPrimA = internal::ShellPrimitiveCount(molecule, basisSet, a);
        const std::size_t nPrimB = internal::ShellPrimitiveCount(molecule, basisSet, b);
        const std::size_t payload = internal::PairStoreBytesPerPair(a.angularMomentum,
                                                                    b.angularMomentum,
                                                                    a.isSpherical,
                                                                    b.isSpherical,
                                                                    a.contractionCount,
                                                                    b.contractionCount,
                                                                    nPrimA * nPrimB);

        if (payload > sizeof(internal::MdPairData))
        {
            maxPayload = std::max(maxPayload, payload - sizeof(internal::MdPairData));
        }
    }

    return maxPayload;
}

// The row-window plan of one build (the row-window decomposition): how many
// windows the row walk is cut into and how many threads the window region
// runs with.
struct WindowPlan {
    std::size_t windowCount; ///< The row-window count (at least 1, at most the row count).
    int threadCount; ///< The region's OpenMP thread count (always at least 1).
};

// Plans the row windows of one build (the row-window scheduling rule).
//
// The window count is the decomposition granularity, NOT the thread count:
// with auto chunks (maxParallelChunks = 0) the row space is cut into
// kWindowOversubscription x the OpenMP team and the window loop runs with
// dynamic scheduling at a chunk of one window, so the row-to-row cost
// variance averages out over the run instead of landing in one thread's
// window - the previous one-window-per-thread static split gave each
// thread exactly its own share of rows with nothing to rebalance with.
// Two counts stay exact rather than oversubscribed: the k = 1 pin
// (maxParallelChunks = 1 - one window, run directly on the calling thread
// with no OpenMP involvement, the byte-pinned path) and an explicit
// maxParallelChunks = N >= 2 (exactly N windows on N threads, the window-
// count experiment seam). An auto count on a one-thread team collapses to
// that same single window: with no second thread to rebalance against,
// oversubscribing would only add reduction terms.
//
// The window count is clamped to the row count: one row per window at
// most, since a window with no rows would still allocate its n x n
// accumulator and contribute nothing.
// \param options The lean options (the chunk override).
// \param nPairs The row count (the canonical shell pairs).
// \returns The window count and the region's thread count.
WindowPlan PlanWindows(const LeanFockBuildOptions& options, std::size_t nPairs) {
    const std::size_t rows = std::max<std::size_t>(1, nPairs);

    if (options.maxParallelChunks == 1)
    {
        return {1, 1};
    }

    if (options.maxParallelChunks != 0)
    {
        const std::size_t windows =
            std::max<std::size_t>(1, std::min(options.maxParallelChunks, rows));

        return {windows, static_cast<int>(windows)};
    }

    const int team = qcx::backend::DefaultOmpTeamSize();

    if (team <= 1)
    {
        return {1, 1};
    }

    const std::size_t oversubscribed = kWindowOversubscription * static_cast<std::size_t>(team);
    const std::size_t windows = std::max<std::size_t>(1, std::min(oversubscribed, rows));
    // Never more threads than windows: a row space too small to fill the
    // team (the clamp above) also collapses the region's thread count, and
    // a one-row system lands on the one-window serial plan exactly like the
    // k = 1 pin does.
    const int threads = static_cast<int>(std::min(windows, static_cast<std::size_t>(team)));

    return {windows, threads};
}

// Plans the wave width of the generation-grouped reduction - the number of
// row windows whose n x n accumulators may be live at once (the
// accumulator-residency bound).
//
// The width is a SCHEDULING and MEMORY knob, never a numerical one: the fold
// in BuildFock (and the estimate below, which charges this same width) runs
// in window index order at every width, so the k > 1 result is bit-identical
// at every width - the wave bit-identity pin. What the width moves is how
// many accumulators are live at once and how many windows the dynamic
// schedule rebalances over at a time (the wave is the schedule's
// granularity, so the width is never taken below the region's thread count
// on purpose - the auto width is twice it).
//
// Auto (requested = 0): kWindowWaveFactor x the region's thread count,
// clamped to the window count. Both EXACT plans keep their historic
// single-wave shape by construction: the k = 1 pin (one window, one thread)
// and an explicit maxParallelChunks = N (N windows on N threads) both get a
// width >= their window count, so their calls are byte for byte the calls
// they were, and the estimator's accumulator charge for them is the same
// windowCount x 8n^2 it always was.
// \param requested The options' wave width (0 = auto).
// \param plan The row-window plan (its thread count scales the auto width).
// \param windowCount The plan's window count (the width's ceiling).
// \returns The wave width in 1..windowCount.
std::size_t PlanWaveWidth(std::size_t requested,
                          const WindowPlan& plan,
                          std::size_t windowCount) noexcept {
    const std::size_t autoWidth =
        kWindowWaveFactor * static_cast<std::size_t>(std::max(plan.threadCount, 1));
    const std::size_t width = requested != 0 ? requested : autoWidth;

    return std::max<std::size_t>(1, std::min(width, std::max<std::size_t>(windowCount, 1)));
}

// The simple actuals sum behind the Create-time last-resort check: the
// SCF O(N^2) matrices and DIIS history, the retained shell-pair state
// (records, pair list, flattened shell inputs), the ONE wave's Fock
// accumulators plus the live windows' scratch (the geometry store copy, the
// flush pair payloads at the flush cap times the largest pair payload, the
// flush tiles) times the live window count, the RETAINED pair-store payload
// of every live window and the run's own pair-store materialization.
// Honest simple terms, no per-class charges, no razor margins - the check
// exists to catch only the grossly-over-ceiling configurations.
//
// The two retained-store terms are the never-under answer to the retention
// the builder holds by construction. The lean flush pass does not tear down
// at all (see the note in processFlush), so a window's store keeps every
// pair it ever built until the window ends; the machinery's chunk pass
// reaches the same retention through ClearChunkPairData (md_batch.cpp),
// whose flat clear() keeps the bra transform's CAPACITY and which
// deliberately leaves the per-prim-pair containers - their nested
// braWeights / ketTransforms vectors and the three per-axis E tables each
// element carries - uncleared. Measured: the c24h50/def2-SVP store retains
// ~41 MB per store, 13.5 MiB of it the E tables alone, against the 1.57 MB
// the flush cap below charges, so
// both terms read the machinery's own per-pair formula (footprint.hpp
// PairStoreBytes - the same ground truth the direct builder's envelope
// uses) and are never-under by construction: the pairs a window actually
// built are a subset of the canonical pairs, so the full store payload
// bounds every window's retention.
//
// What the envelope is NOT: a whole-process model. A lean run's measured
// peak commit carries a floor this estimate cannot see (the driver image
// and its DLLs - measured: H2/def2-SVP, a 0.31 MB estimate, peaks at
// 7.38 MB) and the allocator's committed high-water above the live set,
// which the flush machinery's per-flush allocation churn drives rather than
// any retained allocation (measured: the per-prim-pair and E-slice retention
// added ~3.2 MB of live payload per window at C4H10/def2-SVP and moved the
// measured peak by 0.21 MB). Both are outside a live-bytes envelope's
// reach; the run-context term below is the part of the run's own footprint
// that IS code-visible here.
// \param molecule Molecule providing the atom coordinates (Bohr).
// \param basisSet Basis set (the primitive-count authority).
// \param pairList The canonical shell pairs.
// \param options The lean options (the tile cap and the wave width).
// \param plan The row-window plan the call will run (PlanWindows - the
// window count, whose wave width PlanWaveWidth derives and the accumulator
// term multiplies, and the region's thread count, which is the concurrency
// bound of both the window-local scratch and the auto wave width).
// \returns The estimate in bytes and its largest contributor.
struct PeakEstimate {
    double totalBytes;
    double largestBytes;
    const char* largestName;
};

PeakEstimate EstimatePeakBytes(const qcx::molecule::Molecule& molecule,
                               const qcx::basisset::BasisSet& basisSet,
                               const ShellPairList& pairList,
                               const LeanFockBuildOptions& options,
                               const WindowPlan& plan) {
    const std::size_t n = pairList.functionCount;
    const std::size_t nPairs = pairList.pairs.size();
    const std::size_t nShells = pairList.shells.size();
    const std::size_t windows = plan.windowCount;
    // The concurrently live windows: the store, the stamps vector and every
    // flush arena are WINDOW LOCALS of BuildFock's runWindow (only the
    // accumulator is moved out into workerFocks), so at most the region's
    // thread count of them exist at once - PlanWindows' threadCount, which
    // is min(the window count, the team). The legacy per-window terms below
    // stay at the full window count (never-under, and at one window per
    // thread the two counts coincide).
    const std::size_t liveWindows = static_cast<std::size_t>(plan.threadCount);

    // The SCF-side O(N^2) actuals: kScfMatrixCount matrices of 8n^2 bytes.
    // The footprint envelope is double arithmetic by design; the counts are far below 2^53.
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const double matricesBytes = static_cast<double>(kScfMatrixCount) * 8.0 * n * n;

    // The retained pair state: the lean records (24 B/pair), the pair list
    // (16 B/pair), the flattened shell structs (~64 B/shell) and the shell
    // contraction inputs of the FlattenShells output (per shell: the
    // exponents, one double per primitive, plus the normalized rows, one
    // double per (row, primitive)).
    double shellPrimitiveBytes = 0.0;

    for (const ShellInfo& shell : pairList.shells)
    {
        const std::size_t prims = internal::ShellPrimitiveCount(molecule, basisSet, shell);
        // The footprint envelope is double arithmetic by design; the counts are far below 2^53.
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        shellPrimitiveBytes += 8.0 * prims * (1.0 + static_cast<double>(shell.contractionCount));
    }

    const double pairStateBytes =
        // The footprint envelope is double arithmetic by design; the counts are far below 2^53.
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        40.0 * nPairs + 64.0 * nShells + shellPrimitiveBytes + 128.0 * nShells;

    // The orbit action's retained tables (the petite-list increment): the
    // per-pair image map, the shell images and the function-level signed
    // permutation, charged by the action's OWN formula so the envelope and
    // the tables cannot drift apart. Zero whenever the mechanism is not
    // engaged - the option off, no reduction, or a trivial reduction (one
    // that acts trivially on every shell pair; see the SHAPE below) - which
    // is why the default path's envelope is untouched.
    //
    // SHAPE: a "trivial" reduction is one whose every element fixes every
    // canonical shell pair. That is NOT the same as "C1" - a sign-only
    // group (every atom on the symmetry element, so no atom moves) is
    // non-C1 and still fixes every pair - and the builder engages on the
    // SHAPE, never on the order, because an order-2 group that permutes no
    // pair buys exactly 1.0000x while paying for the tables. The
    // classification's masks are gated the same way: on a pair-trivial
    // group the orbits are singletons, so the flags below describe the
    // plain path and the call is the C1 call byte for byte.
    const std::size_t orbitOrder =
        (options.symmetryOrbitExpansion && !IsTrivialReduction(options.symmetryReduction))
            ? options.symmetryReduction->groupOrder
            : 0;
    const double orbitActionBytes =
        static_cast<double>(internal::LeanOrbitActionBytes(orbitOrder, nPairs, nShells, n));

    // The flush ERI tile's honest bound is the flush structure's own: one
    // flush holds at most kLeanFlushPairCap quartets, each at most the
    // largest pair's function product squared doubles - the maxBatchBytes
    // cap binds only when that product bound would exceed it (min of the
    // two; the tile is the single per-flush values allocation plus the
    // batch-assembly task/quartet records, the small metadata term).
    double maxPairFunctions = 0.0;

    for (const ShellPairIndex& pair : pairList.pairs)
    {
        const double pairFunctions =
            static_cast<double>(ShellFunctionCount(pairList.shells[pair.i])) *
            static_cast<double>(ShellFunctionCount(pairList.shells[pair.j]));
        maxPairFunctions = std::max(maxPairFunctions, pairFunctions);
    }

    const double flushTileBytes = std::min(static_cast<double>(options.maxBatchBytes),
                                           static_cast<double>(kLeanFlushPairCap) *
                                               maxPairFunctions * maxPairFunctions * 8.0) +
                                  static_cast<double>(kLeanFlushPairCap) * 256.0;

    // Per-window scratch: the Fock accumulator (8n^2), the geometry-only
    // store copy (sizeof(MdPairData) x nPairs - the Create-time template
    // is one of the windows' copies), the flush pair payloads (the flush
    // pair cap times the largest pair payload) and the flush tile above.
    // The orbit expansion's per-window member scratch (one member block, the
    // largest block in the system) is inside the flush tile's own bound: the
    // tile term charges kLeanFlushPairCap blocks of the largest pair's
    // function product squared, which dominates one block by that factor.
    const double perWindowBytes =
        // The footprint envelope is double arithmetic by design; the counts are far below 2^53.
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        8.0 * n * n + static_cast<double>(sizeof(internal::MdPairData)) * nPairs +
        static_cast<double>(kLeanFlushPairCap) *
            static_cast<double>(MaxPairPayload(molecule, basisSet, pairList)) +
        flushTileBytes;
    // The per-window terms are split by what is REALLY live how many times
    // (the wave-grouped estimate correction): the Fock accumulator
    // is a WINDOW term, but only one WAVE's worth of accumulators is live at
    // once (the generation-grouped reduction folds a wave's partials into
    // the running Fock as the wave joins), so it is charged at the wave
    // width - PlanWaveWidth, the width BuildFock really runs - while every
    // other per-window term - the store copy, the stamps vector, the flush
    // arenas - is a runWindow local, so at most plan.threadCount of them
    // exist at once. Charging the accumulators at the full oversubscribed
    // window count was TRUE to the one-wave form's residency (all of them
    // were live until the join) and is exactly what the wave grouping
    // removes: 7.37 GiB charged at n = 4,974 before the grouping, 1.84 GiB
    // after, and the one-wave form is still charged what it holds.
    const std::size_t waveWidth = PlanWaveWidth(options.windowWaveCount, plan, windows);
    // The footprint envelope is double arithmetic by design; the counts are far below 2^53.
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
    const double accumulatorBytes = static_cast<double>(waveWidth) * 8.0 * n * n;
    const double windowLocalBytes =
        // The footprint envelope is double arithmetic by design; the counts are far below 2^53.
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        static_cast<double>(liveWindows) * (perWindowBytes - 8.0 * n * n);
    const double windowScratchBytes = accumulatorBytes + windowLocalBytes;

    // The pair store's full byte mass: the machinery's own per-pair formula
    // (footprint.hpp PairStoreBytes - the MdPairData structs, the prim-pair
    // structs, the inner vector headers, the E tables, the bra transform and
    // the per-prim weights and ket transforms). The two run terms below both
    // read it, and both are exact rather than fitted: every component is a
    // sizeof or a per-pair arithmetic product over the canonical pair list.
    const double fullStoreBytes =
        static_cast<double>(internal::PairStoreBytes(molecule, basisSet, pairList));

    // The retained pair-store payload of every LIVE window's store: the
    // pairs that window built and has not released, bounded by the band cap
    // (options.windowStoreBytes - 0 is the legacy unbounded retention, the
    // whole store payload per window). The bands keep their newest entry
    // whatever the cap, so the realized retention is at most the cap plus
    // one flush's pair payload - and that flush payload is already charged
    // by perWindowBytes above, so this term is the cap itself. The pairs a
    // window built are a subset of the canonical pairs, so the whole-store
    // payload bounds the uncapped form.
    const double storePayloadBytes =
        // The footprint envelope is double arithmetic by design; the counts are far below 2^53.
        // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
        fullStoreBytes - static_cast<double>(sizeof(internal::MdPairData)) * nPairs;
    const double retainedStoreBytes =
        static_cast<double>(liveWindows) *
        (options.windowStoreBytes == 0
             ? storePayloadBytes
             : std::min(static_cast<double>(options.windowStoreBytes), storePayloadBytes));

    // The run's own pair-store materialization, outside this builder: the
    // driver's shared setup ramp (BuildCoreHamiltonian's kinetic and nuclear
    // calls plus BuildOverlapMatrix - each through BuildPairMatrix,
    // one_electron.cpp) used to materialize the WHOLE contracted pair store
    // once per call - 9.95 GiB apiece at n = 4,974, three sequential
    // materializations before the SCF's first Fock call. It now runs the
    // chunked sweep's shape (the same
    // three pieces: the geometry-only skeleton, one pair-data chunk, and a
    // COMPLETE release per chunk - md_batch.hpp ReleaseChunkPairData), so the
    // realized residency is the skeleton plus one chunk and NOT the store.
    // Charged as that realized shape rather than as the store: this term sits
    // in the same sum as the builder's own peak, and a never-under bound here
    // would keep the whole path over its ceiling for memory the run no longer
    // touches (the ruling: charge what the algorithm realizes, and
    // justify any oversubscription per term). The chunk is the same 512 MiB
    // the prep's chunk cap names (one_electron.cpp kPairPrepChunkBytes, from
    // this same internal::kPairChunkBytes) and the skeleton is the MdPairData
    // struct count the pair store formula's first component already carries.
    const double prepSkeletonBytes =
        static_cast<double>(sizeof(internal::MdPairData)) * static_cast<double>(nPairs);
    const double prepChunkBytes =
        std::min(static_cast<double>(internal::kPairChunkBytes), storePayloadBytes);
    const double runStoreBytes = prepSkeletonBytes + prepChunkBytes;

    const double totalBytes = matricesBytes + pairStateBytes + orbitActionBytes +
                              windowScratchBytes + retainedStoreBytes + runStoreBytes;

    // The largest contributor, in the historical tie order (the orbit tables
    // are last: they only ever win a strict maximum, so the other four terms'
    // tie preferences - and with them the refusal texts' contributor names -
    // are exactly what they were before the term existed).
    struct Term {
        double bytes;
        const char* name;
    };

    const Term terms[] = {
        {matricesBytes, "the SCF O(N^2) matrices and DIIS history"},
        {pairStateBytes, "the retained shell-pair state"},
        {retainedStoreBytes, "the retained pair-store payload of the live row windows"},
        {windowScratchBytes, "the per-thread scratch (accumulators, stores, flush arenas)"},
        {orbitActionBytes, "the retained pair-orbit tables"}};
    Term largest = terms[0];

    for (const Term& term : terms)
    {
        if (term.bytes > largest.bytes)
        {
            largest = term;
        }
    }

    return {totalBytes, largest.bytes, largest.name};
}

// The last-resort ceiling verdict, the ONE site that decides whether an
// envelope clears its ceiling: the refusal text below is the contract's,
// and both consumers - the builder's Create-time check and the
// driver's pre-ramp check (EstimateLeanEnvelope) - call it, so the early
// decision and the builder's own can never drift apart. A ceiling of 0 is
// no ceiling: no check, no text.
// \param peak The envelope (EstimatePeakBytes).
// \param memoryCapGiB The ceiling in GiB (0 = none).
// \returns The refusal text when the envelope exceeds the ceiling by more
// than the generous slack; nullopt when the ceiling admits it.
std::optional<std::string> LeanCeilingRefusal(const PeakEstimate& peak, double memoryCapGiB) {
    if (memoryCapGiB <= 0.0)
    {
        return std::nullopt;
    }

    const double ceilingBytes = memoryCapGiB * (1ULL << 30);

    if (peak.totalBytes <= ceilingBytes * kRefusalSlack)
    {
        return std::nullopt;
    }

    return std::string("lean-direct last-resort memory refusal: required_peak_bytes ") +
           std::to_string(static_cast<long long>(peak.totalBytes)) + " exceeds available_bytes " +
           std::to_string(static_cast<long long>(ceilingBytes)) +
           " by more than the generous slack (1.5x); largest contributor: " + peak.largestName;
}

} // namespace

qcx::Result<LeanEnvelope> EstimateLeanEnvelope(const qcx::molecule::Molecule& molecule,
                                               const qcx::basisset::BasisSet& basisSet,
                                               const LeanFockBuildOptions& options) {
    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    // The same plan a call will run (PlanWindows - the rule BuildFock
    // applies), so the per-window scratch term and the retention term are
    // the ones the run's own decomposition allocates.
    const WindowPlan plan = PlanWindows(options, pairList->pairs.size());
    const PeakEstimate peak = EstimatePeakBytes(molecule, basisSet, *pairList, options, plan);

    LeanEnvelope envelope;
    envelope.totalBytes = peak.totalBytes;
    const std::optional<std::string> refusal = LeanCeilingRefusal(peak, options.memoryCapGiB);

    if (refusal.has_value())
    {
        envelope.refusal = *refusal;
    }

    return envelope;
}

// The builder's shared implementation state. Everything is immutable after
// Create; each BuildFock call builds its per-window scratch locally, so a
// shared const State makes calls deterministic (the k=1 pin contract) and
// thread-safe.
struct LeanDirectFockBuilder::State {
    ShellPairList pairList; ///< The canonical shell pairs (shared by every machinery entry).
    /// One immutable record per canonical pair (position = pair index),
    /// each carrying its own point-group classification when the builder was
    /// created with a reduction (lean_pair_record.hpp).
    std::vector<internal::LeanShellPairRecord> records;
    /// The per-shell contraction inputs of the pair list (FlattenShells):
    /// the shared read-only input of every window's chunk pair build.
    std::vector<internal::MdShellInput> lightShells;
    /// The geometry-only per-pair store template: every window copies it
    /// once per BuildFock call and fills the flush pair transforms into
    /// its own copy (never shared - chunk pair sets of different windows
    /// overlap on the ket side).
    std::vector<internal::MdPairData> lightTemplate;
    Eigen::MatrixXd coreH; ///< The core Hamiltonian (Eigen copy).
    /// The core Hamiltonian as the caller handed it (CoreHamiltonian()).
    std::optional<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> coreTensor;
    /// The Create-time envelope (EstimatePeakBytes): the number the
    /// last-resort cap check consumed and EstimatedPeakBytes reports.
    double peakBytes = 0.0;
    /// The pair- and function-level orbit action (lean_orbit_action.hpp) when
    /// the caller engaged the orbit expansion (options.symmetryOrbitExpansion
    /// with a non-trivial reduction); empty on every other configuration, so
    /// the walk's representative test and the contraction's expansion are
    /// both inert and the call is the plain path's byte for byte.
    std::optional<internal::LeanOrbitAction> orbitAction;
    LeanFockBuildOptions options;
};

qcx::Result<LeanDirectFockBuilder> LeanDirectFockBuilder::Create(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
    const LeanFockBuildOptions& options) {
    if (options.maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    if (options.memoryCapGiB < 0.0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "memoryCapGiB must not be negative"});
    }

    // The orbit expansion is DEFINED by the group it expands over, so asking
    // for it without a reduction is a caller error, refused rather than
    // silently ignored - the same discipline the reduction's own shape gets
    // below. (The mask option alone stays inert without a reduction, as it
    // always has: nothing about that path changes here.)
    if (options.symmetryOrbitExpansion && options.symmetryReduction == nullptr)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "symmetryOrbitExpansion requires a symmetry reduction (the reduction defines "
            "the orbits the walk enumerates)"});
    }

    auto pairList = BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto bounds = ComputeSchwarzBounds(molecule, basisSet);

    if (!bounds.has_value())
    {
        return std::unexpected(bounds.error());
    }

    auto shells = internal::FlattenShells(molecule, basisSet, *pairList);

    if (!shells.has_value())
    {
        return std::unexpected(shells.error());
    }

    const std::size_t n = pairList->functionCount;

    if (coreHamiltonian.Shape()[0] != n || coreHamiltonian.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "core Hamiltonian shape mismatch"});
    }

    // The point-group classification (the Abelian reduction's classification
    // seam: the machinery's class table and this share the reduction, never
    // the contraction). Derived ONCE here, from the reduction's own
    // signed-permutation arrays, and packed straight into each pair record
    // below - the point-group filter is a property of the records, built at
    // the same moment the records are, not a test bolted onto the walk. The
    // default (null) and a trivial reduction - including a non-C1 group that
    // acts trivially on every shell pair (the IsTrivialReduction shape
    // above) - classify nothing: every record's mask bytes stay zero and no
    // cell ever tests as provably zero.
    std::vector<LeanPairSymmetryMask> pairSymmetry;

    if (!IsTrivialReduction(options.symmetryReduction))
    {
        auto masks = DerivePairSymmetryMasks(*options.symmetryReduction, *pairList);

        if (!masks.has_value())
        {
            return std::unexpected(masks.error());
        }

        pairSymmetry = std::move(*masks);
    }

    // The orbit action (the petite-list increment): the tables the row walk's
    // representative test and the contraction's expansion read. Built at the
    // same Create-time moment as the records and from the same reduction, and
    // retained for the builder's lifetime - the caller keeps ownership of the
    // reduction, and no reference into it outlives this call. A trivial
    // reduction - the order-1 default, the default-constructed empty shape,
    // or a group that acts trivially on every shell pair - leaves the action
    // unbuilt: with every pair fixed by every element each cell is its own
    // orbit and the mechanism is the identity, so the plain path's bytes are
    // what the call produces anyway.
    std::optional<internal::LeanOrbitAction> orbitAction;

    if (options.symmetryOrbitExpansion && !IsTrivialReduction(options.symmetryReduction))
    {
        auto action = internal::LeanOrbitAction::Create(*options.symmetryReduction, *pairList);

        if (!action.has_value())
        {
            return std::unexpected(action.error());
        }

        orbitAction = std::move(*action);
    }

    // The immutable shell-pair records: one per canonical pair, in
    // pair-index order (the record's position IS its pair index - no
    // index stored). Schwarz bounds, the angular class, the packed
    // AO-pair offset and the point-group classification all derive from the
    // pair list (plus the reduction) alone; the bounds come from the counted
    // Schwarz machinery's own input (ComputeSchwarzBounds) and are dropped
    // afterwards - the records are the only retained per-pair Schwarz state
    // and the only retained per-pair symmetry state.
    const std::size_t nPairs = pairList->pairs.size();
    std::vector<internal::LeanShellPairRecord> records;
    records.reserve(nPairs);

    for (const ShellPairIndex& pair : pairList->pairs)
    {
        const ShellInfo& shellA = pairList->shells[pair.i];
        const ShellInfo& shellB = pairList->shells[pair.j];
        const std::uint32_t angularClass =
            static_cast<std::uint32_t>(shellA.angularMomentum + shellB.angularMomentum);
        const std::uint32_t symmetry =
            pairSymmetry.empty()
                ? 0u
                : (pairSymmetry[records.size()].plus << internal::kLeanPairPlusShift) |
                      (pairSymmetry[records.size()].minus << internal::kLeanPairMinusShift);

        records.push_back(internal::LeanShellPairRecord{
            static_cast<std::uint32_t>(pair.i),
            static_cast<std::uint32_t>(pair.j),
            (*bounds)[records.size()],
            angularClass | symmetry,
            PackedFunctionPairOffset(shellA.functionOffset, shellB.functionOffset, n)});
    }

    // The geometry-only light template (the machinery's BuildLightStore
    // shape): la/lb, the spherical flags and the centers per pair - the
    // fields BuildChunkPairData reads (it fills the computed fields and
    // the transforms of exactly the flush's pairs).
    std::vector<internal::MdPairData> lightTemplate;
    lightTemplate.reserve(nPairs);

    for (const ShellPairIndex& pair : pairList->pairs)
    {
        const internal::MdShellInput& shellA = (*shells)[pair.i];
        const internal::MdShellInput& shellB = (*shells)[pair.j];
        internal::MdPairData entry{};
        entry.la = shellA.contractions.angularMomentum;
        entry.lb = shellB.contractions.angularMomentum;
        entry.isSphericalA = shellA.contractions.isSpherical;
        entry.isSphericalB = shellB.contractions.isSpherical;
        entry.ax = shellA.cx;
        entry.ay = shellA.cy;
        entry.az = shellA.cz;
        entry.bx = shellB.cx;
        entry.by = shellB.cy;
        entry.bz = shellB.cz;
        lightTemplate.push_back(std::move(entry));
    }

    auto coreClone = coreHamiltonian.Clone();

    if (!coreClone.has_value())
    {
        return std::unexpected(coreClone.error());
    }

    // The Create-time envelope, always computed (the record the driver's
    // memory accounting reports through EstimatedPeakBytes); the check below
    // consumes the same number, so the record and the refusal can never
    // drift apart. The plan is the one a call will actually run (PlanWindows
    // - the same rule BuildFock applies), so the per-window scratch term is
    // the scratch the call's own decomposition allocates and the retention
    // term rides the same windows' concurrency bound.
    const WindowPlan plan = PlanWindows(options, pairList->pairs.size());
    const PeakEstimate peak = EstimatePeakBytes(molecule, basisSet, *pairList, options, plan);

    // The last-resort memory check: not a budget mechanism - an
    // estimate of the simple actuals sum (SCF matrices + DIIS history +
    // pair state + per-window scratch) that refuses only when the estimate
    // exceeds the ceiling by more than the generous slack. 0 = no ceiling,
    // no check. The verdict is the shared one (LeanCeilingRefusal), so a
    // driver-side pre-ramp check of the same envelope
    // (EstimateLeanEnvelope) decides exactly as this check does.
    const std::optional<std::string> ceilingRefusal =
        LeanCeilingRefusal(peak, options.memoryCapGiB);

    if (ceilingRefusal.has_value())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, *ceilingRefusal});
    }

    auto state = std::make_shared<State>();
    state->pairList = std::move(*pairList);
    state->records = std::move(records);
    state->lightShells = std::move(*shells);
    state->lightTemplate = std::move(lightTemplate);
    state->coreH = internal::TensorToEigen(coreHamiltonian);
    state->coreTensor = std::move(*coreClone);
    state->peakBytes = peak.totalBytes;
    state->orbitAction = std::move(orbitAction);
    state->options = options;

    return LeanDirectFockBuilder{std::move(state)};
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> LeanDirectFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    FockBuildStats* statsOut) const {
    const State& state = *_state;
    const std::size_t n = state.pairList.functionCount;

    if (density.Shape()[0] != n || density.Shape()[1] != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "density shape mismatch"});
    }

    // The per-call stats seam (FockBuildStats): the
    // lean-backed fields (the fp64 survivor count, the assembled batch
    // count, the eri/contract split and the whole-call span) are zeroed
    // here and accumulated over the call below (the machinery's per-call
    // stats contract - a caller-held struct reused across calls carries one
    // call's counts only); a null statsOut keeps the zero-cost path.
    const bool wantStats = statsOut != nullptr;
    const auto callStarted =
        wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    if (wantStats)
    {
        statsOut->fp64QuartetCount = 0;
        statsOut->batchCount = 0;
        statsOut->pairBuildCount = 0;
        statsOut->contractBlockCalls = 0;
        statsOut->eriWallTime = {};
        statsOut->eriPrepWallTime = {};
        statsOut->eriPrepPairDataWallTime = {};
        statsOut->eriPrepAssembleWallTime = {};
        statsOut->eriPrepTileSetupWallTime = {};
        statsOut->contractWallTime = {};
        statsOut->kernelVrrWallTime = {};
        statsOut->kernelKetWallTime = {};
        statsOut->kernelBraWallTime = {};
        statsOut->kernelVrrQuadruples = 0;
        statsOut->kernelGateSeamCalls = 0;
        statsOut->kernelGroupCount = 0;
        statsOut->kernelPrimPasses = 0;
        statsOut->totalWallTime = {};

        // The caller's per-window record (the window probe): refilled
        // after the join when the caller handed one; null keeps the
        // builder free of per-window storage.
        if (statsOut->windowWallTimesOut != nullptr)
        {
            statsOut->windowWallTimesOut->clear();
        }
    }

    const Eigen::MatrixXd d = internal::TensorToEigen(density);

    // The contractions read both orientations of every unordered pair
    // block (d(c,d) + d(d,c)) and the K transpose-writes assume the
    // symmetry - the machinery's documented BuildFock precondition,
    // checked here in Debug builds.
    assert(d.isApprox(d.transpose()));

    const std::size_t nPairs = state.pairList.pairs.size();

    // The in-loop keep cutoff - the machinery's neighbor-list cutoff
    // verbatim (SchwarzThreshold(accuracy) x kNeighborListSlack). The row
    // walk enumerates each unordered pair-pair exactly once as its
    // canonical cell (row = the larger pair index, ket <= row - the exact
    // cells of BuildNeighborList), so every unique shell quartet is
    // computed exactly once with its full 8-fold symmetry folded into the
    // one contraction pass (the kernel's J/K arithmetic, verbatim).
    const double cutoff = SchwarzThreshold(state.options.accuracy) * internal::kNeighborListSlack;
    const std::size_t flushElementCap = state.options.maxBatchBytes / sizeof(double);

    // The orbit expansion's engagement (the petite-list increment): true only
    // when Create built the action, so without the option - and without a
    // non-trivial reduction - both the walk's representative test and the
    // contraction's member expansion below are unreachable and the call is
    // the plain path's, byte for byte. Hoisted out of the loops: the branch
    // cost is once per call, not once per cell.
    const bool expandOrbits = state.orbitAction.has_value();

    // The row windows: the plan PlanWindows derives (auto oversubscribes
    // the team, the k = 1 pin and an explicit override stay exact), rows
    // round-robin, every row screened over its full ket prefix. The window
    // loop below is scheduled dynamically at a chunk of one window, so the
    // oversubscription - not a cost model - is what averages the row-cost
    // variance across the team: round-robin over
    // oversubscribed rows is good enough once the schedule can rebalance).
    const WindowPlan plan = PlanWindows(state.options, nPairs);
    const std::size_t windowCount = plan.windowCount;

    // The wave width of the generation-grouped reduction (PlanWaveWidth): the
    // windows run in waves of this many, each wave's accumulators folding into
    // the running Fock before the next wave starts, so only one wave's worth of
    // them is ever live. The auto width is twice the region's thread count -
    // every thread keeps two windows to rebalance against - and both exact
    // plans (the k = 1 pin, an explicit maxParallelChunks = N) take a
    // one-wave width, so their calls are unchanged.
    const std::size_t waveWidth = PlanWaveWidth(state.options.windowWaveCount, plan, windowCount);

    // The wave's Fock accumulator slots (each window moves its partial sum
    // into the slot it ran in; the fold below adds them into the shared Fock
    // in window order as the wave joins) and the per-window error slots
    // (disjoint writes - no shared mutable state inside the parallel loop).
    // Only `waveWidth` accumulators are live at once, which is the
    // generation-grouped reduction's residency and the term the Create-time
    // estimate charges: 8 n^2 bytes each, so the auto width's 2 x team slots
    // are 16 n^2 bytes of accumulator against the 64 n^2 the one-wave form
    // holds.
    std::vector<Eigen::MatrixXd> workerFocks(waveWidth);
    std::vector<std::optional<qcx::Error>> workerErrors(windowCount);
    // The per-window stats partials (disjoint writes like the slots
    // above): each window's flush survivor counts, phase spans and batch
    // counts land in their own elements and sum into the caller's stats
    // after the join.
    std::vector<std::size_t> workerQuartets(windowCount, 0);
    std::vector<std::size_t> workerBatches(windowCount, 0);
    // The ERI/contract split probe: the two
    // populated spans of the lean path. The ERI span covers one flush's
    // whole block-compute phase - BuildChunkPairData (the flush pairs'
    // contracted transforms), AssembleClassBatches and the tile pointer
    // setup through the end of RunBatches (the kernel dispatch); the
    // contraction span covers the flush's per-task ContractBlock loop. At
    // k = 1 the per-window partials ARE wall spans of the whole call; with
    // more windows the slots' intervals overlap, so the sums are window-
    // accumulated (per-thread) totals, not wall spans - the FockBuildStats
    // k > 1 convention (fock_build.hpp). Observation only: no numerical
    // path reads these.
    // workerEriWall additionally carries the sub-span boundary: workerEriPrepWall
    // holds the part of the ERI span before the RunBatches dispatch (the pair
    // data, the batch assembly and the tile setup), so the kernel time proper
    // is the difference - the field that tells a per-flush setup wall from a
    // per-quartet kernel wall.
    std::vector<std::chrono::nanoseconds> workerEriWall(windowCount);
    std::vector<std::chrono::nanoseconds> workerEriPrepWall(windowCount);
    std::vector<std::chrono::nanoseconds> workerContractWall(windowCount);
    // The prep split's own parts (the prep probe): every flush opens
    // its prep span at the pair-data build exactly as above, and the three
    // parts - the chunk pair data, the batch assembly and the tile setup -
    // are timed into their own per-window partials between the two prep
    // boundaries, so their sum is at most the prep span (the clocks'
    // residual between the parts belongs to no part). workerPairBuilds
    // counts the pair-proportional unit of the chunk pair-data pass
    // alongside: one per pair that pass actually BUILT, read off the store's
    // not-built marker at the flush that built it (a re-listed pair the
    // marker skips is not counted - see the read in processFlush).
    // Observation only, the same window-partial convention as the spans
    // above.
    std::vector<std::chrono::nanoseconds> workerPairDataWall(windowCount);
    std::vector<std::chrono::nanoseconds> workerAssembleWall(windowCount);
    std::vector<std::chrono::nanoseconds> workerTileSetupWall(windowCount);
    std::vector<std::size_t> workerPairBuilds(windowCount, 0);
    // The accumulation-side denominator (measured 2026-09-13): one per
    // ContractBlock call, i.e. per assembled class
    // task plus one per surviving member cell when the orbit expansion
    // is engaged. The ERI side has eight of these; this phase had none.
    std::vector<std::size_t> workerContractBlockCalls(windowCount, 0);
    // The kernel-span probe's own partials: the three
    // sub-spans of the kernel time (eriWallTime - eriPrepWallTime) and their
    // four counters. The spans are taken INSIDE the class kernels, at one
    // primitive pair per group (the VRR/ket boundary) and once around the
    // whole pass-2 bra loop - never per quartet, never per primitive
    // quadruple - so the instrument's clock cost stays orders of magnitude
    // below the span it reports (md_kernel_span.hpp states the argument).
    // Disjoint sub-spans of the kernel span, so at k = 1 they sum to at most
    // it; with more windows the slots' intervals overlap, so the sums are
    // window-accumulated like every other span in this struct's k > 1
    // convention. Observation only: no numerical path reads these.
    std::vector<std::chrono::nanoseconds> workerKernelVrrWall(windowCount);
    std::vector<std::chrono::nanoseconds> workerKernelKetWall(windowCount);
    std::vector<std::chrono::nanoseconds> workerKernelBraWall(windowCount);
    std::vector<std::size_t> workerVrrQuadruples(windowCount, 0);
    std::vector<std::size_t> workerGateSeamCalls(windowCount, 0);
    std::vector<std::size_t> workerGroupCount(windowCount, 0);
    std::vector<std::size_t> workerPrimPasses(windowCount, 0);
    // The per-window elapsed times (the window probe): each window
    // times its own whole body - the accumulator and scratch setup plus the
    // round-robin row walk - into its own slot. The load-imbalance
    // discriminator (max/mean and max/min over the entries) at the
    // dynamic schedule's own granularity: every entry is the work time of
    // one window's rows, so with the schedule grabbing windows one at a
    // time the makespan's tail above sum/team cannot exceed the largest
    // entry - the spread bounds the residual imbalance instead of being it
    // (the caller reading max/mean off a fine-grained oversubscribed count
    // reads the cost variance of the rows, no longer the schedule's error).
    std::vector<std::chrono::nanoseconds> workerWindowWall(windowCount);

    // One window: the round-robin rows [w, w + windowCount, ...), each row
    // screened over its full ket prefix; the surviving quartets flush
    // through the class-batch machinery in bounded tiles (per-thread
    // scratch only). Windows are independent - any thread may run any
    // window - so the dynamic schedule is free to hand them out in any
    // order; every slot below is indexed by the window's own index, and the
    // accumulator's slot is the window's position WITHIN its wave (the wave
    // is the only place the accumulator index differs from the window
    // index; every other slot is window-indexed and windowCount-sized).
    const auto runWindow = [&](std::size_t w, std::size_t accumulatorSlot) {
        // The window's own clock (the window probe): it covers the
        // whole body below - the accumulator and scratch setup and the row
        // walk - so the caller's per-window entries are the work this
        // window's round-robin row share cost, on whichever thread the
        // schedule ran it. Each running window writes only its own slot.
        const auto windowStarted =
            wantStats ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        Eigen::MatrixXd accumulator =
            Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
        // The window's scratch (created per window, per call).
        std::vector<internal::MdPairData> store = state.lightTemplate;
        // The flush membership stamps: stamps[p] == stampCounter marks the
        // pair as part of the open flush (the stamp vector replaces any
        // per-flush pair scan).
        std::vector<std::uint32_t> stamps(nPairs, 0);
        std::uint32_t stampCounter = 1;
        // The open flush's distinct pair set, raw quartets and accumulated
        // element count.
        std::vector<std::size_t> flushPairs;
        std::vector<ShellQuartet> flushQuartets;
        std::vector<ShellQuartet> computed;
        std::vector<internal::MdClassBatch> batches;
        std::vector<double> values;
        // The orbit expansion's member-block scratch (one member at a time,
        // window-local so its capacity survives every flush of the call): the
        // expanded block the member's contraction reads. Sized per member
        // (resize never shrinks the capacity - the largest member block of
        // the call is the scratch's high-water mark).
        std::vector<double> memberScratch;
        std::size_t flushElements = 0;
        // The retention band's state (options.windowStoreBytes): the pairs
        // each recent flush BUILT, oldest entry first, and their resident
        // payload bytes. A pair that a later flush re-lists while its entry
        // is still in the band is not built again (the marker skips it and
        // it belongs to the entry that built it), so an entry is the exact
        // set of pairs the band is holding on that flush's behalf. 0 = the
        // legacy unbounded retention.
        const std::size_t bandCap = state.options.windowStoreBytes;
        // The band is a FIFO of the recent flushes' built-pair lists, walked
        // by a head index: the head advances past an evicted entry instead of
        // erasing it, so the storage is the entry vector alone (an emptied
        // entry costs its 24-byte header and nothing else).
        std::vector<std::vector<std::size_t>> band;
        std::size_t bandHead = 0;
        std::size_t bandBytes = 0;

        // Processes the open flush: builds the flush's chunk pair data,
        // assembles and runs its class batches into one zero-filled fp64
        // tile and contracts every task with the serial kernels' verbatim
        // arithmetic - then opens the next flush. The flush's built pair
        // data joins the retention band, which releases oldest-first once it
        // exceeds options.windowStoreBytes (0 = the legacy unbounded
        // retention; the note at the end of this lambda states the trade).
        const auto processFlush = [&]() -> qcx::Result<void> {
            if (flushQuartets.empty())
            {
                return qcx::Result<void>{};
            }

            // The call's fp64 lane count (the nq64 trace column): the
            // flush's quartets are exactly the blocks the call evaluates
            // (each pushed quartet is computed exactly once), counted into
            // the window's own partial. Without the orbit expansion the
            // pushed set IS the call's screened-in survivors; with it
            // engaged the walk pushes one cell per orbit, so the count is
            // the ERI BLOCK count - the mechanism's work reduction read off
            // a live build.
            workerQuartets[w] += flushQuartets.size();

            // The flush's ERI span opens at the block-compute phase (see the
            // workerEriWall note): the pair data, the batch assembly and the
            // tile setup all belong to the computation this span measures.
            const auto eriStarted = wantStats ? std::chrono::steady_clock::now()
                                              : std::chrono::steady_clock::time_point{};

            // Prep part 1 (the prep probe): the flush's contracted
            // pair transforms, and the count of the builds they ran. Since
            // the teardown was dropped the flush's LISTED set is no longer
            // its BUILT set: BuildChunkPairData skips a re-listed pair whose
            // braTransform marker is still set, so the pass builds exactly
            // the listed pairs whose marker is off. The scan below reads
            // that marker - the call's own predicate, on the call's own
            // store - immediately before the call and outside the pair-data
            // clock (and so outside the prep span that contains it, so the
            // pair-data span stays the call's own span, untimed by the
            // instrument that accounts for it). The count is exact and
            // never over-counts a skip. Its cost is one marker test per
            // listed pair of the flush - at most kLeanFlushPairCap, only
            // when the caller asked for stats, and over the same list the
            // call walks next, against which the built pairs' contracted
            // transforms are the work. The span is that walk plus one
            // transform per build, so this count denominates the span's
            // pair-proportional part, not the whole span.
            std::size_t flushPairBuilds = 0;
            // The pairs this flush BUILDS (the marker off before the call):
            // the stats counter reads their count and the retention band
            // needs their list, so one marker pass serves both.
            std::vector<std::size_t> builtPairs;

            if (wantStats || bandCap != 0)
            {
                for (const std::size_t pairIndex : flushPairs)
                {
                    if (store[pairIndex].braTransform.empty())
                    {
                        ++flushPairBuilds;

                        if (bandCap != 0)
                        {
                            builtPairs.push_back(pairIndex);
                        }
                    }
                }
            }

            const auto pairDataStarted = wantStats ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
            internal::BuildChunkPairData(store, state.lightShells, state.pairList, flushPairs);

            if (wantStats)
            {
                workerPairDataWall[w] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - pairDataStarted);
                workerPairBuilds[w] += flushPairBuilds;
            }

            // Prep part 2 (the prep probe): the flush's class-batch
            // assembly.
            const auto assembleStarted = wantStats ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
            computed.clear();
            auto assembled = internal::AssembleClassBatches(
                store, state.pairList, flushQuartets, state.options.maxBatchBytes, computed);

            if (wantStats)
            {
                workerAssembleWall[w] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - assembleStarted);
            }

            if (!assembled.has_value())
            {
                return std::unexpected(assembled.error());
            }

            // Prep part 3 (the prep probe): the tile setup - the
            // sizing walks, the value arena and the batch copies - closed
            // at the prep boundary below, together with the prep span
            // itself.
            const auto tileSetupStarted = wantStats ? std::chrono::steady_clock::now()
                                                    : std::chrono::steady_clock::time_point{};

            // One fp64 tile per flush: the sum of the batches' element
            // totals. resize (not assign - no zero fill): the fp64 lane is
            // written in full by the kernels (every task block, no pads)
            // and only per-task block ranges are read back - the
            // machinery's own fp64 layout precedent (fock_build.cpp).
            std::size_t tileElements = 0;

            for (const internal::MdClassBatch& batch : *assembled)
            {
                for (const internal::MdQuartetTask& task : batch.tasks)
                {
                    tileElements += store[task.braPair].nFuncs * store[task.ketPair].nFuncs;
                }
            }

            values.resize(tileElements);
            std::size_t batchBase = 0;

            for (internal::MdClassBatch& batch : *assembled)
            {
                std::size_t batchElements = 0;

                for (const internal::MdQuartetTask& task : batch.tasks)
                {
                    batchElements += store[task.braPair].nFuncs * store[task.ketPair].nFuncs;
                }

                batch.outF64 = values.data() + batchBase;
                batchBase += batchElements;
            }

            // The flush's batches run single-threaded (regionThreads = 1):
            // the parallel dimension is the row windows - no nested
            // parallelism anywhere on the lean path.
            batches.clear();
            batches.reserve(assembled->size());

            for (const internal::MdClassBatch& batch : *assembled)
            {
                batches.push_back(batch);
            }

            // The sub-span boundary: everything above (pair data, assembly,
            // tile setup, the batch copies) is prep - the ERI span's kernel
            // part starts here, at the dispatch. The prep split's tile part
            // closes on the same boundary (both clocks close together, so
            // the three parts sum to at most the prep span).
            if (wantStats)
            {
                const auto now = std::chrono::steady_clock::now();

                workerTileSetupWall[w] +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - tileSetupStarted);
                workerEriPrepWall[w] +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(now - eriStarted);
            }

            // The kernel-span probe: the calling
            // thread's phase accumulator is armed around the dispatch and
            // disarmed (never cleared) on return, so the kernels fill it
            // only while this flush's batches run - and this flush runs on
            // this thread, because the lean path dispatches at
            // regionThreads = 1. A stats-free call never arms it, so the
            // kernels pay one thread-local read and one predictable branch
            // per class batch and nothing else (md_kernel_span.hpp).
            // End() is unconditional inside the wantStats branch, so an
            // error-returning dispatch cannot leave the thread armed.
            internal::KernelSpanAccum& span = internal::KernelSpan();

            if (wantStats)
            {
                span.Begin();
            }

            auto run = internal::RunBatches(batches, 1);

            if (wantStats)
            {
                span.End();
                // The flush's ERI span closes after RunBatches: the kernel
                // dispatch is the phase's last stage, so the span covers the
                // whole block computation and nothing of the contraction.
                // The kernel-span probe's sub-spans land in their own
                // per-window partials on the same boundary - disjoint spans
                // inside the kernel span, so their sum is at most the eri
                // span less the prep span.
                workerEriWall[w] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - eriStarted);
                workerKernelVrrWall[w] += std::chrono::nanoseconds{span.vrrNanos};
                workerKernelKetWall[w] += std::chrono::nanoseconds{span.ketNanos};
                workerKernelBraWall[w] += std::chrono::nanoseconds{span.braNanos};
                workerVrrQuadruples[w] += span.vrrQuadruples;
                workerGateSeamCalls[w] += span.gateSeamCalls;
                workerGroupCount[w] += span.groupCount;
                workerPrimPasses[w] += span.primPasses;
                workerBatches[w] += batches.size();
            }

            if (!run.has_value())
            {
                return std::unexpected(run.error());
            }

            // The per-quartet contraction - the machinery's serial fold
            // verbatim: each task's canonical-role quartet contracts its
            // block at the task's batch-relative offset (the batch's
            // output pointers base the tile). J lands on the bra and ket
            // pair blocks, K on the four exchange targets - the full
            // 8-fold symmetry of the quartet in one pass, exactly as the
            // pinned serial machinery contracts it.
            const internal::FockContractContext context{state.pairList,
                                                        d,
                                                        accumulator,
                                                        state.options.buildExchangeOnly,
                                                        state.options.buildCoulombOnly,
                                                        DensityThreshold(state.options.accuracy),
                                                        nullptr,
                                                        nullptr};

            // The flush's contraction span: the per-task ContractBlock loop
            // only (the shared context above is setup, and it is built before
            // the clock opens so the span measures the loop verbatim). With
            // the orbit expansion engaged the span also covers the member
            // pass: the expansion is contraction-phase work (it feeds the
            // member contractions from the representatives' blocks), so it
            // belongs to the phase it is spent in rather than to the ERI span
            // the blocks were computed in.
            const auto contractStarted = wantStats ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};

            // The orbit expansion's member pass (the petite-list increment):
            // the flush computed ONE block per orbit representative, and every
            // other member of that representative's orbit contracts here FROM
            // that block - expanded into the member's own canonical-role frame
            // by the generator's function action (lean_orbit_action.hpp). Two
            // properties make the expanded build accumulate the plain walk's
            // terms exactly, cell for cell: the member's own survival verdict
            // is re-applied (a member the walk would have dropped is dropped
            // here too), and each surviving cell appears exactly once - as its
            // orbit's surviving minimum, which is the task itself, or as one
            // of that minimum's members, which is the loop below. The
            // representative's own contribution is the task's own
            // ContractBlock above, unchanged - the identity member is the
            // representative.
            const auto contractOrbitMembers = [&](const internal::MdQuartetTask& task,
                                                  const double* repBlock) {
                const internal::LeanOrbitAction& action = *state.orbitAction;
                const std::size_t order = action.order;
                const std::size_t cellBra = std::max(task.braPair, task.ketPair);
                const std::size_t cellKet = std::min(task.braPair, task.ketPair);
                // The member set is seeded with the representative's OWN cell:
                // a non-identity element can map the cell onto itself (a
                // within-pair flip does exactly that whenever the two shells of
                // a pair swap under an element that fixes both pairs), and such
                // a "member" is the representative again - contracting it would
                // accumulate the representative's contribution twice. The
                // representative's own contribution is the task's own
                // ContractBlock above, and no stabilizer element duplicates it.
                std::array<std::size_t, internal::kSymmetryMaskBits> seen{};
                seen[0] = internal::LeanCellIndex(cellBra, cellKet);
                std::size_t seenCount = 1;

                for (std::size_t g = 1; g < order; ++g)
                {
                    const std::size_t imageBra = action.pairImage[cellBra * order + g];
                    const std::size_t imageKet = action.pairImage[cellKet * order + g];
                    const std::size_t memberBra = std::max(imageBra, imageKet);
                    const std::size_t memberKet = std::min(imageBra, imageKet);

                    // The member's own walk verdict, re-applied: the
                    // provable-zero test and the Schwarz screen are the two
                    // rules the plain walk drops cells on, so a member they
                    // would have dropped contributes nothing here either - the
                    // expanded build's term set is the plain walk's.
                    if (internal::LeanPairCellIsProvablyZero(state.records[memberBra],
                                                             state.records[memberKet]))
                    {
                        continue;
                    }

                    if (state.records[memberBra].qSchwarz * state.records[memberKet].qSchwarz <
                        cutoff)
                    {
                        continue;
                    }

                    // One contraction per member cell: a generator and its
                    // stabilizer images produce the SAME cell, and the first
                    // generator in group order is the one the expansion uses -
                    // every map of a member produces the identical expanded
                    // block (the machinery's pinned consistency verdict).
                    const std::size_t memberCell = internal::LeanCellIndex(memberBra, memberKet);
                    bool duplicate = false;

                    for (std::size_t s = 0; s < seenCount; ++s)
                    {
                        duplicate = duplicate || seen[s] == memberCell;
                    }

                    if (duplicate)
                    {
                        continue;
                    }

                    seen[seenCount] = memberCell;
                    ++seenCount;

                    // The member's canonical-role task form - the form the
                    // assembler would have emitted for the member's cell, so
                    // the member's contribution lands on the same targets in
                    // the same orientation the plain walk's would.
                    const internal::MdCanonicalTaskForm form =
                        internal::CanonicalTaskFormOf(state.pairList, memberBra, memberKet);
                    const std::size_t memberElements =
                        internal::LeanPairBlockElements(state.pairList, form.braPair) *
                        internal::LeanPairBlockElements(state.pairList, form.ketPair);
                    memberScratch.resize(memberElements);
                    internal::LeanExpandOrbitMemberBlock(action,
                                                         state.pairList,
                                                         g,
                                                         task.braPair,
                                                         task.ketPair,
                                                         form.braPair,
                                                         form.ketPair,
                                                         imageBra,
                                                         imageKet,
                                                         repBlock,
                                                         memberScratch.data());
                    const ShellPairIndex& memberBraPair = state.pairList.pairs[form.braPair];
                    const ShellPairIndex& memberKetPair = state.pairList.pairs[form.ketPair];
                    const ShellQuartet memberQuartet{
                        memberBraPair.i, memberBraPair.j, memberKetPair.i, memberKetPair.j};

                    ++workerContractBlockCalls[w];
                    ContractBlock(memberScratch.data(),
                                  memberQuartet,
                                  form.braPair,
                                  form.ketPair,
                                  context,
                                  state.options.forceScalarContract);
                }
            };

            for (const internal::MdClassBatch& batch : batches)
            {
                // The task's outputOffset is BATCH-relative (cumulative
                // within its own batch), so the task blocks of one batch
                // base the batch's own outF64 - never the tile's start.
                const std::size_t base = batch.outF64 - values.data();

                for (const internal::MdQuartetTask& task : batch.tasks)
                {
                    const ShellPairIndex& bra = state.pairList.pairs[task.braPair];
                    const ShellPairIndex& ket = state.pairList.pairs[task.ketPair];
                    const ShellQuartet quartet{bra.i, bra.j, ket.i, ket.j};
                    const double* repBlock = values.data() + base + task.outputOffset;

                    ++workerContractBlockCalls[w];
                    ContractBlock(repBlock,
                                  quartet,
                                  task.braPair,
                                  task.ketPair,
                                  context,
                                  state.options.forceScalarContract);

                    if (expandOrbits)
                    {
                        contractOrbitMembers(task, repBlock);
                    }
                }
            }

            if (wantStats)
            {
                workerContractWall[w] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - contractStarted);
            }

            // The pair data is deliberately NOT torn down here by the
            // machinery's chunk-pass teardown - the lean path's one
            // divergence from it (ClearChunkPairData, fock_build.cpp) - but
            // the retention is BOUNDED by the band below instead of being
            // unbounded. The band holds the
            // recent flushes' built pairs up to options.windowStoreBytes and
            // releases the oldest entry's payload once it exceeds the cap,
            // so a pair the band dropped is rebuilt when a later flush
            // lists it. Rebuilding is exactly redundant with retaining: the
            // build's only inputs are the pair's own shells, their centers
            // and the store element (fully rewritten, every sum assigned),
            // all window-local, so a skipped build writes the bits a
            // rebuild would have written - bit-identity is structural, no
            // flush composition, batch grouping or accumulation order moves.
            // What the bound costs is pair builds (the workerPairBuilds
            // counter reads them): the band's own size is the memory.
            if (bandCap != 0 && !builtPairs.empty())
            {
                std::size_t builtBytes = 0;

                for (const std::size_t pairIndex : builtPairs)
                {
                    builtBytes += PairPayloadBytes(store[pairIndex]);
                }

                bandBytes += builtBytes;
                band.push_back(std::move(builtPairs));

                while (bandBytes > bandCap && band.size() - bandHead > 1)
                {
                    for (const std::size_t pairIndex : band[bandHead])
                    {
                        const std::size_t released = ReleasePairPayload(store[pairIndex]);
                        bandBytes = released > bandBytes ? 0 : bandBytes - released;
                    }

                    band[bandHead].clear();
                    ++bandHead;
                }
            }

            flushQuartets.clear();
            flushPairs.clear();
            flushElements = 0;
            ++stampCounter;
            return qcx::Result<void>{};
        };

        // The window's row block: CONTIGUOUS rows,
        // not a round-robin residue class. A window's ket working set is its
        // rows' significant partners, and contiguous rows make that set a
        // sliding spatial band - the locality the retention band above needs
        // for its reuse distance to be the band itself rather than the whole
        // pair space. Contiguity costs nothing in balance: the window count
        // is oversubscribed 8x the team and the schedule hands out one
        // window at a time, so a block's row-cost spread (cost grows with
        // the row index) averages over many windows.
        const std::size_t rowsPerWindow = (nPairs + windowCount - 1) / windowCount;
        const std::size_t rowBegin = w * rowsPerWindow;
        const std::size_t rowEnd = std::min(nPairs, rowBegin + rowsPerWindow);

        for (std::size_t row = rowBegin; row < rowEnd; ++row)
        {
            const internal::LeanShellPairRecord& braRecord = state.records[row];

            for (std::size_t q = 0; q <= row; ++q)
            {
                const internal::LeanShellPairRecord& ketRecord = state.records[q];

                // The point-group zero test first: the two records carry the
                // classification the cell's ERI block is dropped on, and a
                // cell whose block is provably an exact zero is skipped
                // before any per-cell arithmetic. Dropping it cannot move a
                // byte - the block's every element is its own negation, so
                // its J and K contributions are exactly-zero products - and
                // the cell is never counted a survivor. Without a reduction
                // both records' masks are zero and the test never fires.
                if (internal::LeanPairCellIsProvablyZero(braRecord, ketRecord))
                {
                    continue;
                }

                if (braRecord.qSchwarz * ketRecord.qSchwarz < cutoff)
                {
                    continue;
                }

                // The orbit representative test (the petite-list increment):
                // with the expansion engaged the walk visits ONE cell per
                // symmetry orbit - its surviving minimum - and the
                // contraction expands that representative's block over the
                // orbit's members, so the ERI evaluation happens once per
                // orbit instead of once per cell (measured: 2,379,972 blocks
                // over 8,811,481 screened cells on c8h18/def2-SVP, 0.2701).
                // The two tests above are the candidate set's own verdicts -
                // a cell this walk would have dropped never wins the minimum
                // (LeanCellIsOrbitRep takes the same two tests on every
                // image). Without the expansion's action the test is
                // unreachable and the walk is the plain path's, step for step.
                if (expandOrbits && !internal::LeanCellIsOrbitRep(
                                        *state.orbitAction, state.records.data(), cutoff, row, q))
                {
                    continue;
                }

                // The quartet survives the pair screen: its canonical cell
                // (row | q) - each unordered pair-pair exactly once, the
                // machinery's CSR cells in row order. The diagonal cell
                // (q == row) is one pair, never two.
                const std::size_t newPairs = (stamps[row] != stampCounter ? 1u : 0u) +
                                             (q != row && stamps[q] != stampCounter ? 1u : 0u);
                const std::size_t blockElements =
                    QuartetElementCount(state.pairList, braRecord, ketRecord);

                // The flush caps cut BEFORE the push (a flush never holds a
                // pushed quartet it cannot keep): the distinct-pair cap
                // bounds the chunk-pair arena, the element cap bounds the
                // flush's fp64 tile to ~maxBatchBytes. A single quartet
                // larger than the element cap still flushes alone (the
                // caps are flush bounds, not per-quartet limits).
                if (!flushPairs.empty() && (flushPairs.size() + newPairs > kLeanFlushPairCap ||
                                            flushElements + blockElements > flushElementCap))
                {
                    auto processed = processFlush();

                    if (!processed.has_value())
                    {
                        workerErrors[w] = processed.error();
                        return;
                    }
                }

                if (stamps[row] != stampCounter)
                {
                    stamps[row] = stampCounter;
                    flushPairs.push_back(row);
                }

                if (stamps[q] != stampCounter)
                {
                    stamps[q] = stampCounter;
                    flushPairs.push_back(q);
                }

                const ShellPairIndex& bra = state.pairList.pairs[row];
                const ShellPairIndex& ket = state.pairList.pairs[q];
                flushQuartets.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
                flushElements += blockElements;
            }
        }

        auto processed = processFlush();

        if (!processed.has_value())
        {
            workerErrors[w] = processed.error();
            return;
        }

        workerFocks[accumulatorSlot] = std::move(accumulator);

        if (wantStats)
        {
            workerWindowWall[w] += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - windowStarted);
        }
    };

    // The k = 1 pin IS a one-window wave run on the calling thread (the
    // branch below); PlanWindows hands a one-window plan exactly when the
    // call is the k = 1 pin (an explicit maxParallelChunks = 1) or an auto
    // call on a one-thread team, so the identity is pinned here instead of
    // assumed: a multi-window plan can never take a serial path, and a
    // one-window plan is never run on a team that could have run windows in
    // parallel.
    assert(windowCount != 1 || plan.threadCount == 1);

    // The reduction: F = H_core + the window accumulators in WINDOW ORDER,
    // folded into the running Fock as each wave joins (the k = 1 result =
    // H_core + the single accumulator - deterministic, the lean pin path).
    // The wave grouping moves only WHEN a partial is added, never its
    // position in the sum: the fold visits slot 0..waveSize-1 of wave 0,
    // then of wave 1, and so on, so the addition sequence is the
    // window-order one at every wave width and the result is bit-identical
    // across widths (the wave pin in LeanDirectFockBuilderTest). The wave
    // width therefore bounds the accumulator residency without touching the
    // arithmetic - the property this path's reproducibility rests on.
    Eigen::MatrixXd fock = state.coreH;
    const qcx::backend::Backend<qcx::backend::CpuTag> backend;

    for (std::size_t waveStart = 0; waveStart < windowCount; waveStart += waveWidth)
    {
        const std::size_t waveSize = std::min(waveWidth, windowCount - waveStart);

        if (waveSize == 1)
        {
            // A one-window wave runs directly on this thread with no OpenMP
            // involvement - the k = 1 pin's own shape, and the shape a tail
            // wave takes when the window count does not divide by the wave
            // width (there is nothing to parallelize).
            runWindow(waveStart, 0);
        } else
        {
            // The wave runs on the backend's dynamic schedule at a chunk of
            // one window: with the auto wave (twice the team) every thread
            // keeps a second window to rebalance against, while the
            // reduction stays window-ordered - each window's accumulator
            // slot is its own; the schedule decides only which thread runs a
            // window, never its index or its position in the reduction.
            backend.ParallelForDynamic(
                waveSize,
                [&](std::size_t slot) { runWindow(waveStart + slot, slot); },
                plan.threadCount);
        }

        // The wave's errors first - an errored window left its accumulator
        // slot empty - then its accumulators into the running Fock, in
        // window order.
        for (std::size_t slot = 0; slot < waveSize; ++slot)
        {
            const std::size_t w = waveStart + slot;

            if (workerErrors[w].has_value())
            {
                return std::unexpected(*workerErrors[w]);
            }

            fock += workerFocks[slot];
        }
    }

    // The bridge handles the tensor allocation and the copy; the stats
    // accumulation stays after it so the wall timing keeps covering the
    // whole build (the machinery's tail convention, fock_build.cpp).
    auto tensor = internal::EigenToTensor(fock);

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    if (wantStats)
    {
        // The fp64 lane count, the batch count and the phase spans: the
        // sums of the per-window partials (each window wrote only its own
        // slots; see the workerEriWall note for the k > 1 span convention).
        // totalWallTime is one steady_clock span over the whole call on the
        // calling thread.
        for (std::size_t w = 0; w < windowCount; ++w)
        {
            statsOut->fp64QuartetCount += workerQuartets[w];
            statsOut->batchCount += workerBatches[w];
            statsOut->pairBuildCount += workerPairBuilds[w];
            statsOut->contractBlockCalls += workerContractBlockCalls[w];
            statsOut->eriWallTime += workerEriWall[w];
            statsOut->eriPrepWallTime += workerEriPrepWall[w];
            statsOut->eriPrepPairDataWallTime += workerPairDataWall[w];
            statsOut->eriPrepAssembleWallTime += workerAssembleWall[w];
            statsOut->eriPrepTileSetupWallTime += workerTileSetupWall[w];
            statsOut->contractWallTime += workerContractWall[w];
            statsOut->kernelVrrWallTime += workerKernelVrrWall[w];
            statsOut->kernelKetWallTime += workerKernelKetWall[w];
            statsOut->kernelBraWallTime += workerKernelBraWall[w];
            statsOut->kernelVrrQuadruples += workerVrrQuadruples[w];
            statsOut->kernelGateSeamCalls += workerGateSeamCalls[w];
            statsOut->kernelGroupCount += workerGroupCount[w];
            statsOut->kernelPrimPasses += workerPrimPasses[w];
        }

        // The caller's per-window record (the window probe): the
        // per-window entries in window order - the caller computing
        // max/mean and max/min over them reads the schedule's balance
        // directly. Null keeps the builder free of per-window storage.
        if (statsOut->windowWallTimesOut != nullptr)
        {
            statsOut->windowWallTimesOut->assign(workerWindowWall.begin(), workerWindowWall.end());
        }

        statsOut->totalWallTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - callStarted);
    }

    return std::move(*tensor);
}

qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> LeanDirectFockBuilder::BuildFock(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const {
    return BuildFock(density, nullptr);
}

const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& LeanDirectFockBuilder::CoreHamiltonian()
    const {
    return *_state->coreTensor;
}

double LeanDirectFockBuilder::EstimatedPeakBytes() const noexcept {
    return _state->peakBytes;
}

LeanDirectFockBuilder::LeanDirectFockBuilder(std::shared_ptr<const State> state) :
    _state(std::move(state)) {}

} // namespace qcx::integrals
