#pragma once

// The shared screening path of the Fock builders: the density-weighted
// screening loop (density weight, density gate, certified mixed-precision
// routing) and the cached Schwarz neighbor-list construction, extracted
// from fock_build.cpp so the DirectJkFockBuilder and the GpuJkFockBuilder
// evaluate EXACTLY the same screening decisions on the same candidate
// lists (the GPU-vs-CPU parity check demands bit-identical lists).
// Header-only: every function here is inline,
// and both consumers are MSVC C++23 host TUs (this header is never included
// from a .cu TU - the device code gets its own CUDA-safe copy of the
// decisions it needs, with the same constants).

#include "md_batch.hpp"
#include "md_defs.hpp"
#include "md_vrr.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/memory/first_touch.hpp"
#include "qcx/memory/sparsity_pattern.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace qcx::integrals::internal {

// The max |density| element over a shell block (offset x extent) - the
// density weight of the screening and routing gates. The four block-geometry
// size_t parameters are inherently same-typed; their order is fixed here.
inline double DensityBlockMax(
    const Eigen::MatrixXd& density,
    std::size_t blockOffsetI,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): size_t quartet.
    std::size_t blockOffsetJ,
    std::size_t blockSizeI,
    std::size_t blockSizeJ) {
    double maximum = 0.0;

    for (std::size_t i = 0; i < blockSizeI; ++i)
    {
        for (std::size_t j = 0; j < blockSizeJ; ++j)
        {
            maximum = std::max(maximum,
                               std::abs(density(static_cast<Eigen::Index>(blockOffsetI + i),
                                                static_cast<Eigen::Index>(blockOffsetJ + j))));
        }
    }

    return maximum;
}

// The canonical pair-list index of the unordered shell pair (i, j): the
// cross blocks of a quartet ((a,c), (b,d), (a,d), (b,c)) are not guaranteed
// to arrive i <= j, and PairIndexOf assumes j >= i (shell_pairs.hpp) -
// calling it with i > j silently reads garbage. Every cross-pair lookup
// (density maxima and Schwarz bounds alike) goes through this first.
inline std::size_t CanonicalPairIndexOf(std::size_t i,
                                        std::size_t j,
                                        const ShellPairList& pairList) noexcept {
    return PairIndexOf(std::min(i, j), std::max(i, j), pairList);
}

// The shell-compressed max-density matrix: max |D_block|
// per canonical pair (i <= j), one entry per pair in pair-index order - the
// density-weight source of the quartet gate and the per-element re-filter.
// REBUILT PER BuildFock CALL from the density that call receives - never
// cached across calls: the incremental wrapper alternates full D and Delta-D
// between calls (incremental_fock.cpp), and a builder-level cache would
// silently screen the wrong density. The vector holds
// the RAW block maxima; the Q-weighted pair bound P_ij = value * Q_ij (Q
// from the pair-indexed Schwarz bounds) is formed at the gate site
// (ScreenOne), and the per-element thresholds divide by the raw value -
// the density block's own scale, which is what a density-weighted
// screening bound needs.
inline std::vector<double> BuildShellPairMaxDensity(const Eigen::MatrixXd& density,
                                                    const ShellPairList& pairList) {
    std::vector<double> maxima(pairList.pairs.size(), 0.0);

    for (std::size_t pair = 0; pair < pairList.pairs.size(); ++pair)
    {
        const ShellPairIndex& index = pairList.pairs[pair];
        maxima[pair] = DensityBlockMax(density,
                                       pairList.shells[index.i].functionOffset,
                                       pairList.shells[index.j].functionOffset,
                                       ShellFunctionCount(pairList.shells[index.i]),
                                       ShellFunctionCount(pairList.shells[index.j]));
    }

    return maxima;
}

// The per-target-block quartet gate (product form CORRECTED 2026-08-29 by
// the C80H162/STO-3G kLoose sweep finding): the
// pure keep/drop decision of the density screen, extracted from ScreenOne
// so it is unit-testable without a builder. pairBounds holds the six raw
// pair density maxima max|D_ij| in ScreenOne's block order (ab, cd, ac, bd,
// ad, bc); pairQ holds the six Schwarz pair values in the same order;
// threshold is the per-preset DensityThreshold. The products are the
// CERTIFIED per-quartet bounds of the mode's work: the J sections target
// (ab) with density block (cd) and (cd) with (ab) (the (ab|cd) = (cd|ab)
// symmetry), so the quartet's J work is bounded by
// (max|D_ab| + max|D_cd|) * Q_ab * Q_cd; the four K sections target (ac),
// (bd), (ad), (bc) with the permuted density blocks, bounding the K work
// by (max|D_ac| + max|D_bd| + max|D_ad| + max|D_bc|) * Q_ab * Q_cd. The
// ORIGINAL product form P_ab * P_cd (the Q-weighted max|D| of BOTH pairs)
// under-bounds whenever a density max is below 1 - the target pair's
// density max never appears in the J contribution, so quartets with a
// sparse target pair were dropped while their true contribution
// (max|D_cd| * Q_ab * Q_cd) exceeded the threshold (measured 1.5e-2 vs the
// 6.6e-9 kLoose budget at n = 562; the H2O diag-0.5 fixture showed the
// same class at 6.29991 with certifiedBoundSum = 0). Drop iff strictly
// less, keep at equality - the six-block-max gate's boundary convention
// (dMax * Q_bra * Q_ket < threshold drops).
inline bool QuartetDensityGate(const std::array<double, 6>& pairBounds,
                               const std::array<double, 6>& pairQ,
                               bool buildExchangeOnly,
                               bool buildCoulombOnly,
                               double threshold) {
    // Full mode: keep iff any mode-relevant certified bound >= threshold.
    const double jProduct = (pairBounds[0] + pairBounds[1]) * pairQ[0] * pairQ[1];
    const double kProduct =
        (pairBounds[2] + pairBounds[3] + pairBounds[4] + pairBounds[5]) * pairQ[0] * pairQ[1];

    if (buildCoulombOnly)
    {
        return jProduct >= threshold;
    }

    if (buildExchangeOnly)
    {
        return kProduct >= threshold;
    }

    return jProduct >= threshold || kProduct >= threshold;
}

// The cached neighbor-list cutoff slack: the list is built with the pair
// cutoff MULTIPLIED by this factor, i.e. more permissive than the live
// check - see the construction comment in BuildNeighborList for the
// rationale.
//
// Criterion: the cached list must contain every quartet the live,
// density-weighted gate could keep, or a cache built once per Create()
// would screen where the per-iteration gate is supposed to decide. Both
// tests compare the same product of two Schwarz pair bounds. The live gate
// keeps a quartet whose product clears tau / W, with W the block-maxima sum
// that mode multiplies the pair product by: W is two blocks in the J form
// (max|D_ab| + max|D_cd|) and four in the K form (the ac, bd, ad and bc
// blocks). The list keeps a quartet whose product clears tau * slack, so
// containment holds whenever tau / W >= tau * slack for every W that could
// have kept a quartet - and the binding one is the largest, the four-block
// K weight. The list is therefore a superset for every density whose
// four-block weight stays at or below 1 / slack = 100. A physical density
// stays orders of magnitude inside that band - an AO density matrix is
// positive semidefinite, so |D_ij| <= sqrt(D_ii D_jj), and the block maxima
// of the densities this repo's fixtures build are O(1) - so the slack never
// binds there and is a candidate-count (cost) parameter: a larger value
// buys containment headroom at the price of a longer list, and only a value
// small enough to cut inside the live gate's own product could change what
// is contracted.
inline constexpr double kNeighborListSlack = 0.01;

// The certified-routing invariant error: an fp32-routed computed quartet
// with no recorded density weight is a screening/bookkeeping bug and must
// not under-count the certified sum silently.
inline qcx::Result<void> CertifiedRoutingError() {
    return qcx::Result<void>{
        std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                   "certified routing: no density weight for a computed quartet"})};
}

// The parallel-build chunk count: the auto mode splits the task count over
// the OpenMP team (DefaultOmpTeamSize, already clamped by the configured
// thread ceiling, so a cap of 1 yields numChunks 1 - the serial path; the
// auto split is team-sized, not hardware_concurrency-sized, so the
// fp64-half/fp32-full split is clamped before it happens);
// FockBuildOptions::maxParallelChunks (0 = auto) lets the
// schedule-independence test force the serial fallback (1) or a fixed
// split (>= 2) on any machine.
inline std::size_t ChunkCountFor(std::size_t taskCount, const FockBuildOptions& options) {
    const std::size_t teamChunks =
        static_cast<std::size_t>(std::max(1, qcx::backend::DefaultOmpTeamSize()));
    const std::size_t maxChunks =
        options.maxParallelChunks != 0 ? options.maxParallelChunks : teamChunks;
    return std::min(taskCount, maxChunks);
}

// The parallel-reduce element of the screening pass: the
// chunk-local screened task lists, concatenated by the combine into the
// final canonical-ish order (see the tolerance note at the ScreenAll
// dispatch site).
struct ScreeningPartial {
    std::vector<MdQuartetTask> fp64Quartets;
    std::vector<MdQuartetTask> fp32Quartets;
    std::vector<double> fp32DensityWeights; // Parallel to fp32Quartets.
};

// The ParallelReduce combine of the screening pass: concatenates two
// chunk-local task-list partials in the schedule's (a, b) order - see the
// order-tolerance note at the ScreenAll dispatch site.
inline ScreeningPartial ConcatenateScreeningPartials(const ScreeningPartial& a,
                                                     const ScreeningPartial& b) {
    ScreeningPartial combined;
    combined.fp64Quartets.reserve(a.fp64Quartets.size() + b.fp64Quartets.size());
    combined.fp64Quartets.insert(
        combined.fp64Quartets.end(), a.fp64Quartets.begin(), a.fp64Quartets.end());
    combined.fp64Quartets.insert(
        combined.fp64Quartets.end(), b.fp64Quartets.begin(), b.fp64Quartets.end());
    combined.fp32Quartets.reserve(a.fp32Quartets.size() + b.fp32Quartets.size());
    combined.fp32Quartets.insert(
        combined.fp32Quartets.end(), a.fp32Quartets.begin(), a.fp32Quartets.end());
    combined.fp32Quartets.insert(
        combined.fp32Quartets.end(), b.fp32Quartets.begin(), b.fp32Quartets.end());
    combined.fp32DensityWeights.reserve(a.fp32DensityWeights.size() + b.fp32DensityWeights.size());
    combined.fp32DensityWeights.insert(combined.fp32DensityWeights.end(),
                                       a.fp32DensityWeights.begin(),
                                       a.fp32DensityWeights.end());
    combined.fp32DensityWeights.insert(combined.fp32DensityWeights.end(),
                                       b.fp32DensityWeights.begin(),
                                       b.fp32DensityWeights.end());
    return combined;
}

// The builder-state pieces the screening path needs, extracted by the
// builders' Create/BuildFock (members, which have access to the private
// State type - the free helpers cannot name it).
struct ScreeningContext {
    const ShellPairList& pairList;
    const std::vector<double>& schwarz;
    const std::vector<MdPairData>& pairStore;
    const FockBuildOptions& options;
    // The per-call shell-compressed max-density vector (raw max |D_block|
    // per canonical pair, BuildShellPairMaxDensity), or nullptr when the
    // per-element screening flag is off. nullptr/empty selects the
    // six-block-max gate in ScreenOne and disables the per-element
    // re-filter - exactly the behavior without the flag.
    // The vector's lifetime is the BuildFock call that built it.
    const std::vector<double>* pairMaxDensity = nullptr;
};

// The screening decision for one neighbor-list candidate (bra, ket): the
// density weight, the density-screening gate, and the certified-mixed-
// precision routing. The weight is dMax, the largest |D| element over the
// six shell-pair blocks this quartet's J/K contractions can read, and it
// is derived here rather than assumed: four shells {a, b, c, d} split
// into two pairs in exactly three ways - ab|cd, ac|bd, ad|bc - and each
// side of each split is read once as a Fock target and once as the
// density block weighting it. J targets (ab) against D_cd and (cd)
// against D_ab (the (ab|cd) = (cd|ab) symmetry); the four exchange
// targets of the K decomposition take (ac) against D_bd, (bd) against
// D_ac, (ad) against D_bc and (bc) against D_ad. The distinct blocks
// across those pairs of roles are the six - ab, cd, ac, bd, ad, bc - and
// each appears as the weight of exactly one target, so the max over all
// six bounds every block any of the quartet's contractions can read. That
// single number is what the routing gate consumes - the certified weight
// must cover whichever block the routed work reads - and what the gate
// without the per-call pair-max vector compares directly. Pushes into the
// caller's lists - every chunk passes its OWN local lists, never shared
// ones (the shared-vector race the parallel dispatch exists to avoid).
// The routing gate sits OUTSIDE the screening block - the certified lane
// must route (and its bound must be delivered) whether or not density
// screening is enabled (it once sat inside, so useDensityScreening=false
// silently disabled the lane).
inline void ScreenOne(std::size_t bra,
                      std::size_t ket,
                      const ScreeningContext& context,
                      const Eigen::MatrixXd& density,
                      // densityThreshold then mixedThreshold: the density gate runs
                      // before the mixed-precision routing consumes it.
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                      double densityThreshold,
                      double mixedThreshold,
                      bool certifiedLane,
                      std::vector<MdQuartetTask>& fp64Quartets,
                      std::vector<MdQuartetTask>& fp32Quartets,
                      std::vector<double>& fp32DensityWeights) {
    // The QFMM near-field restriction: an engaged restrictToPairPairs
    // limits the build to the caller's pair-pair subset - the octree
    // near-field list - before any screening work (the unrestricted path
    // costs one Empty() test). The caller sets both orientations of every
    // near-field leaf pair, so the canonical (bra, ket) iteration of the
    // Schwarz neighbor list matches either way.
    if (!context.options.restrictToPairPairs.Empty() &&
        !context.options.restrictToPairPairs.Contains(bra, ket))
    {
        return;
    }

    const ShellPairIndex& braPair = context.pairList.pairs[bra];
    const ShellPairIndex& ketPair = context.pairList.pairs[ket];
    const std::size_t a = braPair.i;
    const std::size_t b = braPair.j;
    const std::size_t c = ketPair.i;
    const std::size_t dIndex = ketPair.j;
    const std::size_t oA = context.pairList.shells[a].functionOffset;
    const std::size_t oB = context.pairList.shells[b].functionOffset;
    const std::size_t oC = context.pairList.shells[c].functionOffset;
    const std::size_t oD = context.pairList.shells[dIndex].functionOffset;
    const std::size_t nA = ShellFunctionCount(context.pairList.shells[a]);
    const std::size_t nB = ShellFunctionCount(context.pairList.shells[b]);
    const std::size_t nC = ShellFunctionCount(context.pairList.shells[c]);
    const std::size_t nD = ShellFunctionCount(context.pairList.shells[dIndex]);

    // The six density weights come from the per-call pair-max vector (O(1)
    // lookups - the entire screening-pass cost win) or, on the path without
    // it (pairMaxDensity null), from the six block scans exactly as before.
    // The two paths deliver identical values: dMax is the max of the same
    // six blocks either way, so the routing gate below keeps receiving the
    // same weight. The cross-pair lookups are canonicalized before
    // PairIndexOf.
    const bool pairMaxPresent =
        context.pairMaxDensity != nullptr && !context.pairMaxDensity->empty();
    const double dAb =
        pairMaxPresent ? (*context.pairMaxDensity)[bra] : DensityBlockMax(density, oA, oB, nA, nB);
    const double dCd =
        pairMaxPresent ? (*context.pairMaxDensity)[ket] : DensityBlockMax(density, oC, oD, nC, nD);
    const double dAc = pairMaxPresent
                           ? (*context.pairMaxDensity)[CanonicalPairIndexOf(a, c, context.pairList)]
                           : DensityBlockMax(density, oA, oC, nA, nC);
    const double dBd =
        pairMaxPresent
            ? (*context.pairMaxDensity)[CanonicalPairIndexOf(b, dIndex, context.pairList)]
            : DensityBlockMax(density, oB, oD, nB, nD);
    const double dAd =
        pairMaxPresent
            ? (*context.pairMaxDensity)[CanonicalPairIndexOf(a, dIndex, context.pairList)]
            : DensityBlockMax(density, oA, oD, nA, nD);
    const double dBc = pairMaxPresent
                           ? (*context.pairMaxDensity)[CanonicalPairIndexOf(b, c, context.pairList)]
                           : DensityBlockMax(density, oB, oC, nB, nC);
    const double dMax = std::max({dAb, dCd, dAc, dBd, dAd, dBc});

    if (context.options.useDensityScreening)
    {
        const bool keep =
            pairMaxPresent
                ? QuartetDensityGate(
                      {dAb, dCd, dAc, dBd, dAd, dBc},
                      {context.schwarz[bra],
                       context.schwarz[ket],
                       context.schwarz[CanonicalPairIndexOf(a, c, context.pairList)],
                       context.schwarz[CanonicalPairIndexOf(b, dIndex, context.pairList)],
                       context.schwarz[CanonicalPairIndexOf(a, dIndex, context.pairList)],
                       context.schwarz[CanonicalPairIndexOf(b, c, context.pairList)]},
                      context.options.buildExchangeOnly,
                      context.options.buildCoulombOnly,
                      densityThreshold)
                : dMax * context.schwarz[bra] * context.schwarz[ket] >= densityThreshold;

        if (!keep)
        {
            return;
        }
    }

    const int lBra = context.pairStore[bra].la + context.pairStore[bra].lb;
    const int lKet = context.pairStore[ket].la + context.pairStore[ket].lb;
    const double cClass = kClassAmplification[lBra][lKet];
    const double gate =
        cClass * kCertifiedEpsilon * context.schwarz[bra] * context.schwarz[ket] * dMax;

    if (certifiedLane && gate <= mixedThreshold)
    {
        fp32Quartets.push_back(MdQuartetTask{bra, ket, 0});
        fp32DensityWeights.push_back(dMax);
        // The certified sum accumulates the delivered density-weighted
        // per-quartet bounds after the run (below); the gate here is only
        // the routing criterion.
        return;
    }

    fp64Quartets.push_back(MdQuartetTask{bra, ket, 0});
}

// The full density-weighted screening pass of a Fock build: iterates the
// cached Schwarz neighbor list (per bra pair, the ket <= bra candidates
// BuildNeighborList collected), chunked across threads with a static
// bra-row range per chunk - the quartet loop's natural boundary. Row
// lengths vary (compact molecules: s-shell pairs hold most kets, high-l
// pairs few), but on the benchmark molecules the variance is moderate and
// the per-chunk row range still spreads the work evenly; a
// flattened-position split buys nothing here, and is the form to reach for
// only if the rows ever become wildly uneven. Each chunk builds its OWN
// local task lists via ScreenOne - shared vectors would be exactly the
// unsynchronized-accumulation race the per-chunk lists exist to avoid -
// and the combine concatenates them. The combine order is
// thread-schedule dependent, so the concatenated order is not the serial
// canonical (bra, idx) order; every downstream consumer (batch assembly,
// the contraction, the certified read-back) is order-insensitive except
// for last-bit FP sums - within the documented reduction-order tolerance.
// The serial fallback below is the exact same decisions, single-threaded.
inline void ScreenAll(const ScreeningContext& context,
                      const Eigen::MatrixXd& density,
                      const std::vector<std::size_t>& neighborRowOffsets,
                      const std::vector<std::size_t>& neighborIndices,
                      double densityThreshold,
                      double mixedThreshold,
                      bool certifiedLane,
                      std::vector<MdQuartetTask>& fp64Quartets,
                      std::vector<MdQuartetTask>& fp32Quartets,
                      std::vector<double>& fp32DensityWeights) {
    const std::size_t nPairs = context.pairList.pairs.size();
    const std::size_t numChunks = ChunkCountFor(nPairs, context.options);

    if (numChunks <= 1)
    {
        for (std::size_t bra = 0; bra < nPairs; ++bra)
        {
            const std::size_t rowStart = neighborRowOffsets[bra];
            const std::size_t rowEnd = neighborRowOffsets[bra + 1];

            for (std::size_t idx = rowStart; idx < rowEnd; ++idx)
            {
                ScreenOne(bra,
                          neighborIndices[idx],
                          context,
                          density,
                          densityThreshold,
                          mixedThreshold,
                          certifiedLane,
                          fp64Quartets,
                          fp32Quartets,
                          fp32DensityWeights);
            }
        }
    } else
    {
        std::vector<std::size_t> chunkStarts(numChunks + 1);

        for (std::size_t c = 0; c <= numChunks; ++c)
        {
            chunkStarts[c] = (nPairs * c) / numChunks;
        }

        const ScreeningPartial zero{};
        ScreeningPartial screened = qcx::backend::ParallelReduce<ScreeningPartial>(
            numChunks,
            zero,
            [&](const ScreeningPartial& partial, std::size_t chunkIndex) -> ScreeningPartial {
                ScreeningPartial local = partial;

                for (std::size_t bra = chunkStarts[chunkIndex]; bra < chunkStarts[chunkIndex + 1];
                     ++bra)
                {
                    const std::size_t rowStart = neighborRowOffsets[bra];
                    const std::size_t rowEnd = neighborRowOffsets[bra + 1];

                    for (std::size_t idx = rowStart; idx < rowEnd; ++idx)
                    {
                        ScreenOne(bra,
                                  neighborIndices[idx],
                                  context,
                                  density,
                                  densityThreshold,
                                  mixedThreshold,
                                  certifiedLane,
                                  local.fp64Quartets,
                                  local.fp32Quartets,
                                  local.fp32DensityWeights);
                    }
                }

                return local;
            },
            ConcatenateScreeningPartials);

        fp64Quartets = std::move(screened.fp64Quartets);
        fp32Quartets = std::move(screened.fp32Quartets);
        fp32DensityWeights = std::move(screened.fp32DensityWeights);
    }
}

// The pattern-driven form of the screening pass: the cached Schwarz
// neighbor list as the shared qcx::memory::SparsityPattern<CpuTag> instead
// of the raw CSR vectors - the consumer-side half of the LinK migration
// onto the shared sparsity primitive. The pattern is host-canonical (the
// builder wraps BuildNeighborList's output once at Create time via
// BuildSparsityFromAdjacency), so the host views expose exactly the arrays
// the vector form iterates: identical decisions, identical order. The
// builders' State wiring (fock_build.cpp, eri_cuda.cpp) switches to the
// pattern member alongside.
inline void ScreenAll(const ScreeningContext& context,
                      const Eigen::MatrixXd& density,
                      const qcx::memory::SparsityPattern<qcx::backend::CpuTag>& neighborPattern,
                      double densityThreshold,
                      double mixedThreshold,
                      bool certifiedLane,
                      std::vector<MdQuartetTask>& fp64Quartets,
                      std::vector<MdQuartetTask>& fp32Quartets,
                      std::vector<double>& fp32DensityWeights) {
    const auto& offsets = neighborPattern.RowOffsets().HostView();
    const auto& indices = neighborPattern.Indices().HostView();
    ScreenAll(context,
              density,
              offsets,
              indices,
              densityThreshold,
              mixedThreshold,
              certifiedLane,
              fp64Quartets,
              fp32Quartets,
              fp32DensityWeights);
}

// The Schwarz-exact neighbor-pattern count: the
// number of canonical (bra, ket <= bra) pair pairs whose product of the
// two pair Schwarz bounds survives the BuildNeighborList cutoff. The count
// is bit-identical to the build's own counting pass without the
// O(nPairs^2) walk: the cutoff test depends only on the product of the two
// bounds, the IEEE product commutes and is deterministic per operand pair,
// so sorting the bounds descending and walking the per-row prefix boundary
// (rows shrink monotonically as the row value falls - the same-product
// comparison at each boundary step) counts the identical surviving set,
// each unordered pair's product evaluated exactly once plus the diagonals.
// The byte-budget exclusion tests the EXACT pattern bytes (8 x this count)
// against the remaining bytes instead of the all-survive upper bound
// 8 x nPairs(nPairs + 1) / 2: the all-survive form over-refused
// large-sparse systems at ~nPairs^2/2 above their real pattern (measured on
// C80H162/STO-3G: the 26,246,268,048 B bound vs a ~38 MB counted
// pattern). Never under-bounds: a dense system counts every pair and
// reproduces the all-survive bound exactly, so admission only widens.
// \param schwarz The per-canonical-pair Schwarz bounds (the
// ComputeSchwarzBounds output the sweep consumes).
// \param accuracy The screening preset (the pair cutoff is
// SchwarzThreshold(accuracy) x kNeighborListSlack).
// \return The surviving candidate count - BuildNeighborList's
// candidateCount / neighborIndices.size() exactly.
inline std::size_t CountSchwarzSurvivingPairs(const std::vector<double>& schwarz,
                                              AccuracyPreset accuracy) {
    const std::size_t n = schwarz.size();
    const double neighborListThreshold = SchwarzThreshold(accuracy) * kNeighborListSlack;
    std::vector<double> descending = schwarz;
    std::sort(descending.begin(), descending.end(), std::greater<double>());
    // The per-row qualifying set is a prefix [0, boundary) of the
    // descending order (the products with the largest bounds qualify
    // first); the boundary only shrinks as the row value falls. Every
    // off-diagonal surviving pair is counted in two rows - (i, j) and
    // (j, i) - and each surviving diagonal in its own row once, so the
    // canonical upper-triangle count is diagonal + (total - diagonal) / 2.
    std::size_t rowPrefixTotal = 0;
    std::size_t diagonalCount = 0;
    std::size_t boundary = n;

    for (std::size_t i = 0; i < n; ++i)
    {
        while (boundary > 0 && descending[i] * descending[boundary - 1] < neighborListThreshold)
        {
            --boundary;
        }

        rowPrefixTotal += boundary;

        // The diagonal position i qualifies iff its square clears the
        // cutoff; positions at or below the boundary all qualify.
        if (boundary > i)
        {
            ++diagonalCount;
        }
    }

    return diagonalCount + (rowPrefixTotal - diagonalCount) / 2;
}

// The Schwarz-exact peak canonical row width (the LightPath's peak-chunk
// ket bound): the never-under bound on the number of candidate kets the
// WIDEST row of BuildNeighborList emits. It is the counted counterpart of
// the all-survive ket fallback (nPairs) the LightPath's peak-chunk markers
// take when no sweep has run, in exactly the shape the pattern term has:
// the all-survive form survives only as the cheap
// pre-gate, and the counted form decides.
//
// The count is O(nPairs) with no sort and no allocation, because only the
// WIDEST row is needed and the widest row is the one with the LARGEST
// bound: the product test is monotone in each operand, so for any pair
// bound b with b <= b_max the qualifying set {j : b x Q_j >= cutoff} is a
// SUBSET of {j : b_max x Q_j >= cutoff}. Two containments give the
// never-under property, neither of which needs the sweep:
// - every row's FULL symmetric qualifying set is a subset of the largest
//   bound's (the monotonicity above);
// - every EMITTED row is canonical (BuildNeighborList emits ket <= bra),
//   so it is a subset of that row's full symmetric set.
// The product form is kept verbatim (never a divided threshold), so the
// test is the same IEEE comparison the sweep applies.
// \param schwarz The per-canonical-pair Schwarz bounds (the
// ComputeSchwarzBounds output the sweep consumes).
// \param accuracy The screening preset (the pair cutoff is
// SchwarzThreshold(accuracy) x kNeighborListSlack).
// \return The never-under peak canonical row width - at least every
// BuildNeighborList row width, and exactly nPairs on an all-survive bound
// vector (the fallback it stands in for).
inline std::size_t CountSchwarzPeakRowWidth(const std::vector<double>& schwarz,
                                            AccuracyPreset accuracy) {
    if (schwarz.empty())
    {
        return 0;
    }

    const double neighborListThreshold = SchwarzThreshold(accuracy) * kNeighborListSlack;
    const double maxBound = *std::max_element(schwarz.begin(), schwarz.end());

    return static_cast<std::size_t>(std::count_if(
        schwarz.begin(), schwarz.end(), [maxBound, neighborListThreshold](double bound) {
            return maxBound * bound >= neighborListThreshold;
        }));
}

// The RI-J task-grid survivor count (the counted fix): the number of
// (orbital bra pair, aux shell) cells of the BuildScreenedRiTaskList grid
// whose bound product survives the build's cutoff. The count is
// bit-identical to the build's per-cell walk without the O(rows x cols)
// grid walk - the same argument as CountSchwarzSurvivingPairs: the cutoff
// test depends only on the product of the two bounds, the IEEE product is
// deterministic and monotone in each non-negative operand, so sorting both
// bound vectors descending and walking the per-row boundary counts the
// identical surviving set (the qualifying set of every row is a prefix
// [0, boundary) of the descending aux order, and the boundary only
// shrinks as the row value falls). The RI grid is rectangular - no
// canonical-triangle folding, unlike the pair-pair grid above - and its
// cutoff is the plain preset budget (SchwarzThreshold(accuracy), NO
// kNeighborListSlack: the 3c screen is decided by the preset's budget
// alone, the neighbor-list cutoff being where the slack for the
// per-iteration density screen belongs). The footprint's taskListBytes
// charge (RiFootprint's
// screenedTaskCount) is this count x 16 B - exactly the reserve
// BuildScreenedRiTaskList realizes, so the model and the allocation agree
// by equality, the strongest never-under form (the unconditional dense
// grid was the pre-screening reservation, an over-charge at any sparse
// scale). Never under-bounds: on an all-survive grid the count
// reproduces the unconditional rows x cols reservation exactly, so the
// dense reference form is a special case and admission only widens.
// \param orbitalBounds The per-orbital-pair Schwarz bounds (Q_uv - the
// ComputeSchwarzBounds output the build consumes; the row count is its
// size).
// \param auxShellBounds The per-aux-shell Schwarz bounds (Q_P - the
// diagonal elements of the aux pair list, the extraction the build runs).
// \param accuracy The screening preset (the task cutoff is
// SchwarzThreshold(accuracy)).
// \return The surviving task count - BuildScreenedRiTaskList's
// tasks.size() exactly.
inline std::size_t CountSchwarzSurvivingRiTasks(const std::vector<double>& orbitalBounds,
                                                const std::vector<double>& auxShellBounds,
                                                AccuracyPreset accuracy) {
    const double taskThreshold = SchwarzThreshold(accuracy);
    std::vector<double> rowsDesc = orbitalBounds;
    std::vector<double> colsDesc = auxShellBounds;
    std::sort(rowsDesc.begin(), rowsDesc.end(), std::greater<double>());
    std::sort(colsDesc.begin(), colsDesc.end(), std::greater<double>());
    std::size_t total = 0;
    std::size_t boundary = colsDesc.size();

    for (const double rowBound : rowsDesc)
    {
        while (boundary > 0 && rowBound * colsDesc[boundary - 1] < taskThreshold)
        {
            --boundary;
        }

        total += boundary;
    }

    return total;
}

// The cached Schwarz neighbor list construction: the ket candidates of
// every bra pair under the pair-cutoff test, precomputed once per builder
// Create(). The live per-quartet Schwarz check this replaces is static
// across SCF iterations (Q_ab never changes), so this is an exact
// precomputation - the same decision, evaluated once per Create() instead
// of once per BuildFock call. The construction cutoff carries the
// kNeighborListSlack factor (its criterion is stated at the constant): the
// CACHED list is slightly more permissive than the live check was, so a
// borderline pair near the cutoff can never be permanently excluded by
// list construction even if a later iteration's density weight would have
// made it relevant. Costs a few extra entries (constant factor, not
// asymptotic); the per-iteration density screen still runs inside the loop
// and decides what is actually contracted.
inline void BuildNeighborList(const ShellPairList& pairList,
                              const std::vector<double>& schwarz,
                              AccuracyPreset accuracy,
                              std::vector<std::size_t>& neighborRowOffsets,
                              std::vector<std::size_t>& neighborIndices) {
    const std::size_t nPairs = pairList.pairs.size();
    const double neighborListThreshold = SchwarzThreshold(accuracy) * kNeighborListSlack;
    neighborRowOffsets.resize(nPairs + 1, 0);
    // Exact reserve via a counting pass over the same cutoff test: the
    // build is already O(nPairs^2) on compact molecules, so the second
    // pass is free in context, and the count avoids both the repeated
    // growth of an under-guess and the over-allocation of a worst-case
    // guess.
    std::size_t candidateCount = 0;

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
            {
                ++candidateCount;
            }
        }
    }

    neighborIndices.reserve(candidateCount);
    // Place the reserved candidate capacity across the consuming team
    // before the serial fill: an anonymous page stays on the NUMA node of
    // the thread that first accesses it, and every screening thread walks
    // this list each pass. The pass writes raw bytes into the reserved
    // storage - the push_back fill below is the first element access
    // either way.
    qcx::memory::TouchPagesAcrossTeam(neighborIndices.data(),
                                      neighborIndices.capacity() * sizeof(std::size_t));

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
            {
                neighborIndices.push_back(ket);
            }
        }

        neighborRowOffsets[bra + 1] = neighborIndices.size();
    }
}

// The leaf-driven form of the cached neighbor-list construction (the QFMM
// near-field enumeration): the same candidate rows as
// BuildNeighborList restricted to the near-field domain's pair-pair set,
// generated from the near-field leaf pairs instead of the global
// quadratic sweep. For each near-field leaf pair (A, B) the products of
// A's and B's pair indices (the diagonal A == A included) canonicalize to
// (bra, ket) = (max, min) pair index - every unordered pair-pair of the
// near-field set exactly once (the interaction lists carry each unordered
// leaf pair once, a <= b, qfmm_tree.cpp), and the SAME per-pair Schwarz
// cutoff as the sweep applies per product - so the rows are bit-for-bit
// the gated sweep's rows (the full CSR filtered by the
// PairPairRestriction bit test): identical candidates, identical
// ascending per-row order, identical decisions downstream. The emission
// arrives in leaf-pair order, so each row is sorted ascending after the
// fill - the canonical order the serial screening loop iterates.
inline void BuildLeafDrivenNeighborList(const ShellPairList& pairList,
                                        const std::vector<double>& schwarz,
                                        AccuracyPreset accuracy,
                                        const LeafNearFieldDomain& domain,
                                        std::vector<std::size_t>& neighborRowOffsets,
                                        std::vector<std::size_t>& neighborIndices) {
    const std::size_t nPairs = pairList.pairs.size();
    const double neighborListThreshold = SchwarzThreshold(accuracy) * kNeighborListSlack;

    // The per-leaf pair-index lists, derived from leafOfPair in one walk:
    // the pairs are visited in index order, so each leaf's list is
    // ascending by construction.
    std::vector<std::size_t> leafPairOffsets(domain.nLeaves + 1, 0);

    for (const std::size_t leaf : domain.leafOfPair)
    {
        ++leafPairOffsets[leaf + 1];
    }

    for (std::size_t leaf = 0; leaf < domain.nLeaves; ++leaf)
    {
        leafPairOffsets[leaf + 1] += leafPairOffsets[leaf];
    }

    std::vector<std::size_t> leafPairIndices(domain.leafOfPair.size());
    {
        std::vector<std::size_t> cursors = leafPairOffsets;

        for (std::size_t pairIndex = 0; pairIndex < domain.leafOfPair.size(); ++pairIndex)
        {
            const std::size_t leaf = domain.leafOfPair[pairIndex];
            leafPairIndices[cursors[leaf]++] = pairIndex;
        }
    }

    // Pass 1: the candidate count per row over the near-field leaf pairs'
    // products, canonical (bra, ket) = (max, min) index, the same pair
    // cutoff as the sweep.
    neighborRowOffsets.assign(nPairs + 1, 0);

    auto visitProducts = [&](const auto& emit) {
        for (const auto& [leafA, leafB] : domain.nearFieldLeafPairs)
        {
            const std::size_t beginA = leafPairOffsets[leafA];
            const std::size_t beginB = leafPairOffsets[leafB];
            const std::size_t endA = leafPairOffsets[leafA + 1];
            const std::size_t endB = leafPairOffsets[leafB + 1];

            if (leafA == leafB)
            {
                // The diagonal leaf pair: the unordered pairs of the leaf
                // (the same-pair quartets (p, p) included); the ascending
                // leaf list makes (p_i, p_j), i >= j canonical already.
                for (std::size_t i = beginA; i < endA; ++i)
                {
                    const std::size_t bra = leafPairIndices[i];

                    for (std::size_t j = beginA; j <= i; ++j)
                    {
                        const std::size_t ket = leafPairIndices[j];

                        if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
                        {
                            emit(bra, ket);
                        }
                    }
                }
            } else
            {
                for (std::size_t i = beginA; i < endA; ++i)
                {
                    const std::size_t p = leafPairIndices[i];

                    for (std::size_t j = beginB; j < endB; ++j)
                    {
                        const std::size_t q = leafPairIndices[j];
                        const std::size_t bra = std::max(p, q);
                        const std::size_t ket = std::min(p, q);

                        if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
                        {
                            emit(bra, ket);
                        }
                    }
                }
            }
        }
    };

    visitProducts([&](std::size_t bra, std::size_t) { ++neighborRowOffsets[bra + 1]; });

    // The prefix turns the per-row counts into the row offsets.
    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        neighborRowOffsets[bra + 1] += neighborRowOffsets[bra];
    }

    const std::size_t candidateCount = neighborRowOffsets[nPairs];
    neighborIndices.clear();
    // RESIZE, not reserve: the pass-2 fill writes through operator[] into
    // the pre-sized slots (the pass-1 count is authoritative), and the CSR
    // consumers read size() - a reserved-but-empty vector would validate
    // as an empty pattern (a reserved-only fill is the shape trap this
    // sizing avoids).
    neighborIndices.resize(candidateCount);
    // The same NUMA discipline as the sweep form: place the reserved
    // capacity across the consuming team before the fill.
    qcx::memory::TouchPagesAcrossTeam(neighborIndices.data(),
                                      neighborIndices.capacity() * sizeof(std::size_t));
    // Pass 2: the fill, then the per-row sort into the canonical ascending
    // ket order (the emission arrives in leaf-pair order).
    {
        std::vector<std::size_t> cursors = neighborRowOffsets;
        visitProducts(
            [&](std::size_t bra, std::size_t ket) { neighborIndices[cursors[bra]++] = ket; });
    }

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        const std::size_t rowStart = neighborRowOffsets[bra];
        const std::size_t rowEnd = neighborRowOffsets[bra + 1];

        if (rowEnd - rowStart > 1)
        {
            std::sort(neighborIndices.begin() + static_cast<std::ptrdiff_t>(rowStart),
                      neighborIndices.begin() + static_cast<std::ptrdiff_t>(rowEnd));
        }
    }
}

// The count-only form of the leaf-driven construction above (the
// exclusion fix): the exact number of (bra, ket) rows the leaf-driven
// pass would emit - BuildLeafDrivenNeighborList's pass-1 candidateCount
// without building the CSR. The QFMM outer's exclusion (i) and estimate
// need only the count (the nested near-field builder builds the actual
// rows at its own Create - no seam carries a prebuilt pattern), and the
// pre-tree all-survive form could not see the near-field restriction at
// all: the GLOBAL bound over-refused large-sparse systems whose
// near-field pattern is a small fraction of the pair space. Bit-identical
// to the pass-1 count by construction: the same per-leaf pair lists and
// the same product cutoff over the same near-field leaf pairs.
// \param schwarz The per-canonical-pair Schwarz bounds.
// \param accuracy The screening preset (the pair cutoff is
// SchwarzThreshold(accuracy) x kNeighborListSlack).
// \param domain The near-field domain (the carrier of the rows).
// \return The surviving near-field candidate count - the leaf-driven
// CSR's neighborIndices.size() exactly.
inline std::size_t CountNearFieldPatternEntries(const std::vector<double>& schwarz,
                                                AccuracyPreset accuracy,
                                                const LeafNearFieldDomain& domain) {
    const double neighborListThreshold = SchwarzThreshold(accuracy) * kNeighborListSlack;
    std::vector<std::size_t> leafPairOffsets(domain.nLeaves + 1, 0);

    for (const std::size_t leaf : domain.leafOfPair)
    {
        ++leafPairOffsets[leaf + 1];
    }

    for (std::size_t leaf = 0; leaf < domain.nLeaves; ++leaf)
    {
        leafPairOffsets[leaf + 1] += leafPairOffsets[leaf];
    }

    std::vector<std::size_t> leafPairIndices(domain.leafOfPair.size());
    {
        std::vector<std::size_t> cursors = leafPairOffsets;

        for (std::size_t pairIndex = 0; pairIndex < domain.leafOfPair.size(); ++pairIndex)
        {
            const std::size_t leaf = domain.leafOfPair[pairIndex];
            leafPairIndices[cursors[leaf]++] = pairIndex;
        }
    }

    std::size_t count = 0;

    for (const auto& [leafA, leafB] : domain.nearFieldLeafPairs)
    {
        const std::size_t beginA = leafPairOffsets[leafA];
        const std::size_t beginB = leafPairOffsets[leafB];
        const std::size_t endA = leafPairOffsets[leafA + 1];
        const std::size_t endB = leafPairOffsets[leafB + 1];

        if (leafA == leafB)
        {
            for (std::size_t i = beginA; i < endA; ++i)
            {
                const std::size_t bra = leafPairIndices[i];

                for (std::size_t j = beginA; j <= i; ++j)
                {
                    const std::size_t ket = leafPairIndices[j];

                    if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
                    {
                        ++count;
                    }
                }
            }
        } else
        {
            for (std::size_t i = beginA; i < endA; ++i)
            {
                const std::size_t p = leafPairIndices[i];

                for (std::size_t j = beginB; j < endB; ++j)
                {
                    const std::size_t q = leafPairIndices[j];
                    const std::size_t bra = std::max(p, q);
                    const std::size_t ket = std::min(p, q);

                    if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
                    {
                        ++count;
                    }
                }
            }
        }
    }

    return count;
}

// The windowed form of the neighbor-list construction (the chunked
// LightPath pass): builds the ket candidates of the bra rows
// [rowStart, rowEnd) over the FULL ket range (ket <= bra,
// the canonical enumeration - the chunk's kets span the prefix, not a
// bounded window). The offsets are WINDOW-RELATIVE: neighborRowOffsets[0] =
// 0 and neighborRowOffsets[k + 1] holds the accumulated count after row
// rowStart + k, so the windowed ScreenAll reads the same (offsets[k],
// offsets[k + 1]) row range shape as the full form. The vectors are
// cleared and refilled (the caller reuses them across chunks); the reserve
// after the counting pass keeps the capacity at the largest chunk seen -
// the "peak = max chunk, never Σ chunks" invariant, with the buffers never
// re-allocated once the peak chunk has been built.
inline void BuildNeighborList(const ShellPairList& pairList,
                              const std::vector<double>& schwarz,
                              AccuracyPreset accuracy,
                              std::size_t rowStart,
                              std::size_t rowEnd,
                              std::vector<std::size_t>& neighborRowOffsets,
                              std::vector<std::size_t>& neighborIndices) {
    const std::size_t nPairs = pairList.pairs.size();
    const double neighborListThreshold = SchwarzThreshold(accuracy) * kNeighborListSlack;
    neighborRowOffsets.resize(rowEnd - rowStart + 1, 0);
    std::size_t candidateCount = 0;

    for (std::size_t bra = rowStart; bra < rowEnd; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
            {
                ++candidateCount;
            }
        }
    }

    neighborIndices.clear();
    neighborIndices.reserve(candidateCount);

    for (std::size_t bra = rowStart; bra < rowEnd; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (schwarz[bra] * schwarz[ket] >= neighborListThreshold)
            {
                neighborIndices.push_back(ket);
            }
        }

        neighborRowOffsets[bra - rowStart + 1] = neighborIndices.size();
    }
}

// The windowed form of the density-weighted screening pass (the LightPath
// chunk pass): the same decisions as the full CSR form -
// the same ScreenOne, the same chunking pattern - over the bra rows
// [rowStart, rowEnd) with the WINDOW-RELATIVE offsets the windowed
// BuildNeighborList produced. The tasks carry FULL pair indices (the
// chunk's store is full-space-indexed), so the downstream assembly and
// contraction are untouched - this is the pairStore-seam half of the
// chunk pass.
inline void ScreenAll(const ScreeningContext& context,
                      const Eigen::MatrixXd& density,
                      std::size_t rowStart,
                      std::size_t rowEnd,
                      const std::vector<std::size_t>& neighborRowOffsets,
                      const std::vector<std::size_t>& neighborIndices,
                      double densityThreshold,
                      double mixedThreshold,
                      bool certifiedLane,
                      std::vector<MdQuartetTask>& fp64Quartets,
                      std::vector<MdQuartetTask>& fp32Quartets,
                      std::vector<double>& fp32DensityWeights) {
    const std::size_t windowRows = rowEnd - rowStart;
    const std::size_t numChunks = ChunkCountFor(windowRows, context.options);

    if (numChunks <= 1)
    {
        for (std::size_t k = 0; k < windowRows; ++k)
        {
            const std::size_t rowStartOffset = neighborRowOffsets[k];
            const std::size_t rowEndOffset = neighborRowOffsets[k + 1];

            for (std::size_t idx = rowStartOffset; idx < rowEndOffset; ++idx)
            {
                ScreenOne(rowStart + k,
                          neighborIndices[idx],
                          context,
                          density,
                          densityThreshold,
                          mixedThreshold,
                          certifiedLane,
                          fp64Quartets,
                          fp32Quartets,
                          fp32DensityWeights);
            }
        }
    } else
    {
        std::vector<std::size_t> chunkStarts(numChunks + 1);

        for (std::size_t c = 0; c <= numChunks; ++c)
        {
            chunkStarts[c] = rowStart + (windowRows * c) / numChunks;
        }

        const ScreeningPartial zero{};
        ScreeningPartial screened = qcx::backend::ParallelReduce<ScreeningPartial>(
            numChunks,
            zero,
            [&](const ScreeningPartial& partial, std::size_t chunkIndex) -> ScreeningPartial {
                ScreeningPartial local = partial;
                const std::size_t firstRow = chunkStarts[chunkIndex];
                const std::size_t lastRow = chunkStarts[chunkIndex + 1];

                for (std::size_t bra = firstRow; bra < lastRow; ++bra)
                {
                    const std::size_t rowStartOffset = neighborRowOffsets[bra - rowStart];
                    const std::size_t rowEndOffset = neighborRowOffsets[bra - rowStart + 1];

                    for (std::size_t idx = rowStartOffset; idx < rowEndOffset; ++idx)
                    {
                        ScreenOne(bra,
                                  neighborIndices[idx],
                                  context,
                                  density,
                                  densityThreshold,
                                  mixedThreshold,
                                  certifiedLane,
                                  local.fp64Quartets,
                                  local.fp32Quartets,
                                  local.fp32DensityWeights);
                    }
                }

                return local;
            },
            ConcatenateScreeningPartials);

        fp64Quartets = std::move(screened.fp64Quartets);
        fp32Quartets = std::move(screened.fp32Quartets);
        fp32DensityWeights = std::move(screened.fp32DensityWeights);
    }
}

} // namespace qcx::integrals::internal
