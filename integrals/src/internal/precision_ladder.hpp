#pragma once

// The builder-side ladder pass: the per-quartet certified
// bound in the fp32 lane's bound units, the B_screen sum over the
// screening pass's own numbers, and the per-batch band partition (the
// Valeev constraint) of the screened quartets. Header-only, consumed by
// the host TUs of both Fock builders (fock_build.cpp, eri_cuda.cpp) -
// the same shared-pass shape as fock_screen.hpp. The classification is
// a dispatch dimension BETWEEN class runs: the partition assembles the
// class batches first, classifies each batch by its aggregate bound, and
// hands the builders per-band task lists - never a branch inside a
// kernel, and the certified lane's bit-exactness is untouched (the
// pre-ladder two-pass build is the ladder-off path).

#include "fock_screen.hpp"
#include "md_batch.hpp"
#include "md_vrr.hpp"
#include "qcx/integrals/precision_policy.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <vector>

namespace qcx::integrals::internal {

// The certified bound of one quartet in the fp32 lane's bound units:
// eps * cClass * (1 + rounding) * Q_bra * Q_ket * dMax - the ScreenOne
// gate formula plus the class rounding factor, the conservative form the
// certified sum accumulates.
inline double QuartetCertifiedBound(int lBra, int lKet, double qBra, double qKet, double dMax) {
    return kCertifiedBandEpsilon * kClassAmplification[lBra][lKet] *
           (1.0 + static_cast<double>(kClassRounding[lBra][lKet])) * qBra * qKet * dMax;
}

// The six-block density weight of one candidate (bra, ket) - the
// ScreenOne computation shared with the ladder pass: the pair-maximum
// vector's six lookups, or the six block scans on the legacy path. The
// two paths deliver identical values (the fock_screen.hpp contract).
inline double QuartetDensityWeight(const ScreeningContext& context,
                                   const Eigen::MatrixXd& density,
                                   std::size_t bra,
                                   std::size_t ket) {
    const ShellPairIndex& braPair = context.pairList.pairs[bra];
    const ShellPairIndex& ketPair = context.pairList.pairs[ket];
    const std::size_t a = braPair.i;
    const std::size_t b = braPair.j;
    const std::size_t c = ketPair.i;
    const std::size_t dIndex = ketPair.j;
    const bool pairMaxPresent =
        context.pairMaxDensity != nullptr && !context.pairMaxDensity->empty();

    if (pairMaxPresent)
    {
        const std::vector<double>& maxima = *context.pairMaxDensity;
        return std::max({maxima[bra],
                         maxima[ket],
                         maxima[CanonicalPairIndexOf(a, c, context.pairList)],
                         maxima[CanonicalPairIndexOf(b, dIndex, context.pairList)],
                         maxima[CanonicalPairIndexOf(a, dIndex, context.pairList)],
                         maxima[CanonicalPairIndexOf(b, c, context.pairList)]});
    }

    const std::size_t oA = context.pairList.shells[a].functionOffset;
    const std::size_t oB = context.pairList.shells[b].functionOffset;
    const std::size_t oC = context.pairList.shells[c].functionOffset;
    const std::size_t oD = context.pairList.shells[dIndex].functionOffset;
    const std::size_t nA = ShellFunctionCount(context.pairList.shells[a]);
    const std::size_t nB = ShellFunctionCount(context.pairList.shells[b]);
    const std::size_t nC = ShellFunctionCount(context.pairList.shells[c]);
    const std::size_t nD = ShellFunctionCount(context.pairList.shells[dIndex]);
    return std::max({DensityBlockMax(density, oA, oB, nA, nB),
                     DensityBlockMax(density, oC, oD, nC, nD),
                     DensityBlockMax(density, oA, oC, nA, nC),
                     DensityBlockMax(density, oB, oD, nB, nD),
                     DensityBlockMax(density, oA, oD, nA, nD),
                     DensityBlockMax(density, oB, oC, nB, nC)});
}

// The gate's six pair density maxima in ScreenOne's block order (ab, cd,
// ac, bd, ad, bc) - the QuartetDensityGate input of one candidate
// (requires the pair-maximum vector: the product gate exists only on the
// pair-max path).
inline std::array<double, 6> DensityGateWeights(const ScreeningContext& context,
                                                std::size_t bra,
                                                std::size_t ket) {
    const ShellPairIndex& braPair = context.pairList.pairs[bra];
    const ShellPairIndex& ketPair = context.pairList.pairs[ket];
    const std::size_t a = braPair.i;
    const std::size_t b = braPair.j;
    const std::size_t c = ketPair.i;
    const std::size_t dIndex = ketPair.j;
    const std::vector<double>& maxima = *context.pairMaxDensity;
    return {maxima[bra],
            maxima[ket],
            maxima[CanonicalPairIndexOf(a, c, context.pairList)],
            maxima[CanonicalPairIndexOf(b, dIndex, context.pairList)],
            maxima[CanonicalPairIndexOf(a, dIndex, context.pairList)],
            maxima[CanonicalPairIndexOf(b, c, context.pairList)]};
}

// The gate's six Schwarz pair values in the same order - the
// QuartetDensityGate input of one candidate.
inline std::array<double, 6> DensityGateSchwarz(const ScreeningContext& context,
                                                std::size_t bra,
                                                std::size_t ket) {
    const ShellPairIndex& braPair = context.pairList.pairs[bra];
    const ShellPairIndex& ketPair = context.pairList.pairs[ket];
    const std::size_t a = braPair.i;
    const std::size_t b = braPair.j;
    const std::size_t c = ketPair.i;
    const std::size_t dIndex = ketPair.j;
    return {context.schwarz[bra],
            context.schwarz[ket],
            context.schwarz[CanonicalPairIndexOf(a, c, context.pairList)],
            context.schwarz[CanonicalPairIndexOf(b, dIndex, context.pairList)],
            context.schwarz[CanonicalPairIndexOf(a, dIndex, context.pairList)],
            context.schwarz[CanonicalPairIndexOf(b, c, context.pairList)]};
}

// The B_screen term: the per-iteration sum of the certified
// bounds of the quartets the density gate screened out - the same gate
// ScreenOne applies over the same neighbor list, so the sum is the
// screening pass's own numbers in the certified sum's bound units. The
// ladder consumes it as one of the budget's co-terms.
inline double ScreenBoundSum(
    const ScreeningContext& context,
    const Eigen::MatrixXd& density,
    const qcx::memory::SparsityPattern<qcx::backend::CpuTag>& neighborPattern,
    double densityThreshold) {
    const auto& offsets = neighborPattern.RowOffsets().HostView();
    const auto& indices = neighborPattern.Indices().HostView();
    const bool pairMaxPresent =
        context.pairMaxDensity != nullptr && !context.pairMaxDensity->empty();
    double droppedSum = 0.0;

    for (std::size_t bra = 0; bra < context.pairList.pairs.size(); ++bra)
    {
        const std::size_t rowStart = offsets[bra];
        const std::size_t rowEnd = offsets[bra + 1];

        for (std::size_t idx = rowStart; idx < rowEnd; ++idx)
        {
            const std::size_t ket = indices[idx];
            const double dMax = QuartetDensityWeight(context, density, bra, ket);
            const bool keep =
                pairMaxPresent
                    ? QuartetDensityGate(DensityGateWeights(context, bra, ket),
                                         DensityGateSchwarz(context, bra, ket),
                                         context.options.buildExchangeOnly,
                                         context.options.buildCoulombOnly,
                                         densityThreshold)
                    : dMax * context.schwarz[bra] * context.schwarz[ket] >= densityThreshold;

            if (!keep)
            {
                const int lBra = context.pairStore[bra].la + context.pairStore[bra].lb;
                const int lKet = context.pairStore[ket].la + context.pairStore[ket].lb;
                droppedSum += QuartetCertifiedBound(
                    lBra, lKet, context.schwarz[bra], context.schwarz[ket], dMax);
            }
        }
    }

    return droppedSum;
}

// The routing budget's two sums over one screening walk (the
// enforcement pass of the production path): the certified bounds of the
// candidates the density gate screens OUT (the B_screen co-term) and of
// the candidates the routing gate ADMITS into the fp32 lane (the routed
// bound sum the budget is compared against), both in the routing gate's
// own bound units and by the gate's own decision - the same candidate
// walk, the same dMax weights, the same threshold comparisons
// ScreenOne applies. One walk yields both, so the enforcement costs one
// registration-free sweep of the neighbor list rather than two.
struct RoutedBoundSums {
    /// Σ over the density-screened-out candidates of their certified
    /// bounds (Eh, the ladder's B_screen co-term).
    double screenedHa = 0.0;
    /// Σ over the candidates the routing gate admits (Eh) - the build's
    /// delivered fp32-lane bound in the gate's units, compared against
    /// the preset's budget.
    double routedHa = 0.0;
    /// The admitted candidate count (the fp32 task-list size the routing
    /// pass will produce).
    std::size_t routedQuartets = 0;
};

// The enforcement walk: every candidate of one screening pass, in the
// pass's own enumeration order, classified by the SAME two decisions
// ScreenOne applies (fock_screen.hpp) - the density gate, then the
// certified routing gate - with the bound of each candidate accumulated
// into the side its decision sends it to. The routing expression is
// ScreenOne's gate verbatim (cClass * kCertifiedEpsilon * Q_bra *
// Q_ket * dMax, that multiplication order) so the two cannot disagree
// about a candidate: the sum is the exact bound sum the routed lists
// carry, not a re-derivation of it. The per-candidate bound adds the
// class rounding factor (QuartetCertifiedBound - the conservative form
// the ladder's committed sum uses).
//
// The QFMM near-field restriction is honoured first (ScreenOne's step,
// before any screening work) so a restricted build counts only the
// candidates its routing pass sees.
inline RoutedBoundSums SumRoutingBounds(const ScreeningContext& context,
                                        const Eigen::MatrixXd& density,
                                        const std::vector<std::size_t>& neighborRowOffsets,
                                        const std::vector<std::size_t>& neighborIndices,
                                        // densityThreshold then mixedThreshold: the density gate
                                        // runs before the routing gate consumes the mixed bound
                                        // (the ScreenOne order).
                                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                        double densityThreshold,
                                        double mixedThreshold) {
    RoutedBoundSums sums;
    const bool pairMaxPresent =
        context.pairMaxDensity != nullptr && !context.pairMaxDensity->empty();
    const bool restricted = !context.options.restrictToPairPairs.Empty();

    for (std::size_t bra = 0; bra < context.pairList.pairs.size(); ++bra)
    {
        for (std::size_t idx = neighborRowOffsets[bra]; idx < neighborRowOffsets[bra + 1]; ++idx)
        {
            const std::size_t ket = neighborIndices[idx];

            if (restricted && !context.options.restrictToPairPairs.Contains(bra, ket))
            {
                continue;
            }

            const double dMax = QuartetDensityWeight(context, density, bra, ket);
            const bool keep =
                pairMaxPresent
                    ? QuartetDensityGate(DensityGateWeights(context, bra, ket),
                                         DensityGateSchwarz(context, bra, ket),
                                         context.options.buildExchangeOnly,
                                         context.options.buildCoulombOnly,
                                         densityThreshold)
                    : dMax * context.schwarz[bra] * context.schwarz[ket] >= densityThreshold;
            const int lBra = context.pairStore[bra].la + context.pairStore[bra].lb;
            const int lKet = context.pairStore[ket].la + context.pairStore[ket].lb;
            const double bound =
                QuartetCertifiedBound(lBra, lKet, context.schwarz[bra], context.schwarz[ket], dMax);

            if (!keep)
            {
                sums.screenedHa += bound;
                continue;
            }

            // ScreenOne's routing gate, the identical expression and
            // multiplication order (the drift guard).
            const double gate = kClassAmplification[lBra][lKet] * kCertifiedEpsilon *
                                context.schwarz[bra] * context.schwarz[ket] * dMax;

            if (gate <= mixedThreshold)
            {
                sums.routedHa += bound;
                ++sums.routedQuartets;
            }
        }
    }

    return sums;
}

// One ladder build's band partition.
struct LadderPartition {
    std::vector<MdQuartetTask> fp64;
    std::vector<MdQuartetTask> fp16;
    std::vector<MdQuartetTask> fp32Certified;
    std::vector<MdQuartetTask> fp32Mixed;
    // The density weights (dMax) parallel to each coarse-band list - the
    // certified-sum weight map sources.
    std::vector<double> fp16Weights;
    std::vector<double> certifiedWeights;
    std::vector<double> mixedWeights;
    // The a-priori certified bound sums per coarse band (fp32 bound
    // units; the committed sum scales the fp16 band by kFp16BoundScale).
    double fp16BoundSum = 0.0;
    double certifiedBoundSum = 0.0;
    double mixedBoundSum = 0.0;
};

// Partitions the screened quartets into the ladder's bands (the Valeev
// constraint): the kept quartets assemble into class batches first, each
// batch classifies by its aggregate certified bound against
// perBatchBudget (T / nBatches), and the schedule's ratchet keys on the
// assembled batch index. The per-batch granularity is deliberate - the
// band lists dispatch between class runs, never inside a kernel.
inline LadderPartition PartitionBatches(const ScreeningContext& context,
                                        const Eigen::MatrixXd& density,
                                        const std::vector<MdQuartetTask>& keptQuartets,
                                        double classificationBudget,
                                        EscalationSchedule& schedule) {
    LadderPartition partition;

    if (keptQuartets.empty())
    {
        return partition;
    }

    std::vector<ShellQuartet> quartets;
    quartets.reserve(keptQuartets.size());

    for (const MdQuartetTask& task : keptQuartets)
    {
        const ShellPairIndex& braPair = context.pairList.pairs[task.braPair];
        const ShellPairIndex& ketPair = context.pairList.pairs[task.ketPair];
        quartets.push_back(ShellQuartet{braPair.i, braPair.j, ketPair.i, ketPair.j});
    }

    std::vector<ShellQuartet> computed;
    auto batches = AssembleClassBatches(
        context.pairStore, context.pairList, quartets, context.options.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        // An assembly failure is not the ladder's to swallow: degrade by
        // routing everything fp64 (the builder re-assembles per pass and
        // surfaces the failure there).
        partition.fp64 = keptQuartets;
        return partition;
    }

    const std::size_t nBatches = batches->size();
    const double perBatchBudget =
        nBatches == 0 ? 0.0 : classificationBudget / static_cast<double>(nBatches);

    for (std::size_t batchIndex = 0; batchIndex < nBatches; ++batchIndex)
    {
        const MdClassBatch& batch = (*batches)[batchIndex];
        double batchBound = 0.0;

        for (const MdQuartetTask& task : batch.tasks)
        {
            const int lBra =
                context.pairStore[task.braPair].la + context.pairStore[task.braPair].lb;
            const int lKet =
                context.pairStore[task.ketPair].la + context.pairStore[task.ketPair].lb;
            batchBound += QuartetCertifiedBound(
                lBra,
                lKet,
                context.schwarz[task.braPair],
                context.schwarz[task.ketPair],
                QuartetDensityWeight(context, density, task.braPair, task.ketPair));
        }

        const PrecisionBand band = schedule.RatchetBand(batchIndex, batchBound, perBatchBudget);

        for (const MdQuartetTask& task : batch.tasks)
        {
            const double dMax = QuartetDensityWeight(context, density, task.braPair, task.ketPair);

            switch (band)
            {
            case PrecisionBand::kFp16:
                partition.fp16.push_back(task);
                partition.fp16Weights.push_back(dMax);
                break;
            case PrecisionBand::kFp32Certified:
                partition.fp32Certified.push_back(task);
                partition.certifiedWeights.push_back(dMax);
                break;
            case PrecisionBand::kFp32EvalFp64Accumulate:
                partition.fp32Mixed.push_back(task);
                partition.mixedWeights.push_back(dMax);
                break;
            case PrecisionBand::kFp64:
                partition.fp64.push_back(task);
                break;
            }
        }

        switch (band)
        {
        case PrecisionBand::kFp16:
            partition.fp16BoundSum += batchBound;
            break;
        case PrecisionBand::kFp32Certified:
            partition.certifiedBoundSum += batchBound;
            break;
        case PrecisionBand::kFp32EvalFp64Accumulate:
            partition.mixedBoundSum += batchBound;
            break;
        case PrecisionBand::kFp64:
            break;
        }
    }

    return partition;
}

} // namespace qcx::integrals::internal
