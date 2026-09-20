// The algorithm-selection cost table (selection_heuristic.hpp): the gates -
// the estimator returns a total order over the candidates,
// is monotone in nBasis for each candidate, and never selects a candidate
// with a budget below its admission floor - plus the documented priority
// behavior (GPU presence first, then the budgets, then threads)
// on hand-computed fixtures.

#include "qcx/driver/selection_heuristic.hpp"

#include <algorithm>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>

namespace {

using qcx::driver::CandidateEstimate;
using qcx::driver::CostCandidate;
using qcx::driver::SelectionInput;
using qcx::driver::SelectionRanking;

// The anchor fixture the seed cites (2026-08-29): C80H162/STO-3G, n = 562,
// nPairs = 81003 (402 shells). The capture was never committed (no git
// object, not gitignored) and the 5717 ms row it records is the superseded
// timing the repo disowns - the corrected same-fixture row is 2478 ms, a
// 2.31x gap. The fixture's SIZES are still what the pins below measure
// against; only the cited TIMING is disqualified.
constexpr std::size_t kAlkaneN = 562;
constexpr std::size_t kAlkanePairs = 81003;

const CandidateEstimate& Find(const SelectionRanking& ranking, CostCandidate candidate) {
    const auto it =
        std::find_if(ranking.candidates.begin(),
                     ranking.candidates.end(),
                     [candidate](const CandidateEstimate& e) { return e.candidate == candidate; });
    EXPECT_NE(it, ranking.candidates.end());
    return *it;
}

bool IsSortedAscendingByCost(const SelectionRanking& ranking, bool selectable) {
    double previous = -1.0;

    for (const CandidateEstimate& estimate : ranking.candidates)
    {
        if (estimate.selectable != selectable)
        {
            continue;
        }

        if (estimate.costSeconds < previous)
        {
            return false;
        }

        previous = estimate.costSeconds;
    }

    return true;
}

TEST(SelectionHeuristicTest, RankingIsATotalOrderOverTheCandidates) {
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);

    // All six candidates present exactly once - the total order over the
    // candidate set, selectable first then not-selectable, each group
    // ascending by cost (exact ties break by the fixed candidate order).
    ASSERT_EQ(ranking.candidates.size(), 6u);
    EXPECT_TRUE(IsSortedAscendingByCost(ranking, /* selectable */ true));
    EXPECT_TRUE(IsSortedAscendingByCost(ranking, /* selectable */ false));
}

TEST(SelectionHeuristicTest, SmallNoDevicePicksDirect) {
    // h2o-class: n = 24, no device tier, 12 threads, default cap. Hand
    // computation: direct ~4.77e-4 s (the n^3 law at the re-anchored
    // constant, its measured team of 5 folded in), qfmm ~1.0 s (the fixed
    // octree overhead), ri_j_link ~40 s (the recorded Create cost dominates
    // at this size) - direct wins, which is the v1 default for small
    // systems.
    SelectionInput input;
    input.nBasis = 24;
    input.nPairs = 21;
    input.effectiveThreads = 12;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    ASSERT_TRUE(ranking.bestSelectable.has_value());
    EXPECT_EQ(*ranking.bestSelectable, qcx::io::BuilderKind::kDirect);

    const CandidateEstimate& direct = Find(ranking, CostCandidate::kDirect);
    EXPECT_TRUE(direct.selectable);
    EXPECT_NEAR(direct.costSeconds, 4.76926e-4, 1e-6);

    const CandidateEstimate& riJ = Find(ranking, CostCandidate::kRiJLink);
    EXPECT_TRUE(riJ.selectable);
    EXPECT_GT(riJ.costSeconds, 39.0);

    const CandidateEstimate& gpu = Find(ranking, CostCandidate::kGpu);
    EXPECT_FALSE(gpu.selectable);
    EXPECT_NE(gpu.reason.find("no CUDA device"), std::string::npos);

    // The reasoning string names the pick and why.
    EXPECT_NE(ranking.reasoning.find("direct: cheapest selectable path at 12 threads"),
              std::string::npos);
}

TEST(SelectionHeuristicTest, UnwireableRiJIsNotSelectable) {
    // The aux-less case: no [basis].aux and no auto-selectable aux
    // exists, so the wiring would refuse ri_j at the seam. The estimator
    // must mark the candidate not-selectable with the reason - never
    // price it at the phantom zero-aux cost and floor, which would pick
    // it and then refuse the run where a selectable direct path ran.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.riJUnavailableReason = "no aux basis for \"cc-pVDZ\"; specify [basis].aux explicitly";

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    ASSERT_TRUE(ranking.bestSelectable.has_value());

    const CandidateEstimate& riJ = Find(ranking, CostCandidate::kRiJLink);
    EXPECT_FALSE(riJ.selectable);
    EXPECT_EQ(riJ.reason, "no aux basis for \"cc-pVDZ\"; specify [basis].aux explicitly");

    const CandidateEstimate& direct = Find(ranking, CostCandidate::kDirect);
    EXPECT_TRUE(direct.selectable);
}

TEST(SelectionHeuristicTest, DevicePresentPicksGpuFirst) {
    // The design priority order: GPU presence wins when the residency
    // footprint fits the device budget. n = 562 with a T1000-class device
    // budget (3.1 GiB free minus the 0.5 GiB headroom): gpu ~1.58 s vs
    // direct 6.12 s (the re-anchored constant) vs qfmm ~5.3 s.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.deviceBudgetGiB = 2.6;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    ASSERT_TRUE(ranking.bestSelectable.has_value());
    EXPECT_EQ(*ranking.bestSelectable, qcx::io::BuilderKind::kGpu);

    const CandidateEstimate& gpu = Find(ranking, CostCandidate::kGpu);
    EXPECT_TRUE(gpu.selectable);
    // The row anchors the re-anchored direct cost at the GPU constant's own
    // team (12), not the team the measurement fired at: the anchor wall at
    // 5 threads carried to 12, then the speedup and the fixed launch cost.
    EXPECT_NEAR(gpu.costSeconds, 14.6973 * 5.0 / 12.0 / 4.0 + 0.05, 1e-3);

    // The reasoning string names the pick and why: the footprint (0.25 GiB
    // base + the matrix and pair-store terms) fits the device budget.
    EXPECT_NE(ranking.reasoning.find("gpu: device present, footprint 0.3 GiB fits 2.6 GiB"),
              std::string::npos);
    EXPECT_NE(ranking.reasoning.find("threads 12"), std::string::npos);
}

TEST(SelectionHeuristicTest, DeviceBudgetBelowFootprintDropsGpu) {
    // The second priority driver: GPU memory available. A 0.2 GiB device
    // budget cannot hold the ~0.27 GiB footprint, so the GPU candidate
    // drops with the reason recorded and the cheapest CPU path wins.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.deviceBudgetGiB = 0.2;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    ASSERT_TRUE(ranking.bestSelectable.has_value());
    EXPECT_EQ(*ranking.bestSelectable, qcx::io::BuilderKind::kQfmm);

    const CandidateEstimate& gpu = Find(ranking, CostCandidate::kGpu);
    EXPECT_FALSE(gpu.selectable);
    EXPECT_NE(gpu.reason.find("device footprint 0.3 GiB does not fit 0.2 GiB free"),
              std::string::npos);
}

TEST(SelectionHeuristicTest, NoCandidateIsSelectableBelowItsDeviceBudget) {
    // BEHAVIOUR CHANGE, 2026-09-17 (the memory model's deletion): this
    // cell used to assert that a 0.3 GiB cap made EVERY CPU candidate
    // unselectable, because the model's admission floor (~1.5 GiB at
    // n = 562) exceeded the cap. The estimator no longer refuses a
    // candidate for its modeled size - the job-object cap is the
    // admission and the engine's Create-time estimates decide the rung -
    // so at 0.3 GiB the CPU candidates are selectable and only the DEVICE
    // budget excludes the GPU family. Old expectation: bestSelectable
    // empty and every candidate refused. New: the CPU family is
    // selectable, the device family is not.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.cpuBudgetGiB = 0.3;
    input.deviceBudgetGiB = 0.0;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    ASSERT_TRUE(ranking.bestSelectable.has_value());

    for (const CandidateEstimate& estimate : ranking.candidates)
    {
        if (estimate.candidate == CostCandidate::kGpu)
        {
            EXPECT_FALSE(estimate.selectable) << "the device tier has no device budget";
            continue;
        }

        if (estimate.candidate == CostCandidate::kRiJk)
        {
            // Opted in by name, never proposed by the ranking.
            EXPECT_FALSE(estimate.selectable);
            continue;
        }

        // The CPU family is no longer refused for its modeled size; the
        // gpu_split row is the CPU-only fallback at the default zero
        // device share, so it is selectable here too.
        EXPECT_TRUE(estimate.selectable) << "no candidate is refused on CPU memory any more";
    }

    // The reasoning names the PICK. The old "no candidate fits the budgets"
    // text belongs to the branch where nothing is selectable, which is no
    // longer this input's state - keeping the assertion there would pin a
    // sentence the surviving behaviour cannot produce.
    EXPECT_EQ(ranking.reasoning.find("no candidate fits the budgets"), std::string::npos);
    EXPECT_FALSE(ranking.reasoning.empty());
}

TEST(SelectionHeuristicTest, CostsAreMonotoneInNBasis) {
    // The gate: monotone in nBasis for every candidate. The
    // sweep uses a fixed aux count so the tensor-driven candidates (ri_j,
    // ri_jk) exercise their n^2 terms too.
    constexpr std::size_t kSizes[] = {24, 100, 300, 562};

    for (const std::size_t n : kSizes)
    {
        SelectionInput input;
        input.nBasis = n;
        input.nAux = 1600;
        input.nPairs = n * n / 8;
        input.effectiveThreads = 12;

        const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
        ASSERT_EQ(ranking.candidates.size(), 6u);
    }

    for (int c = 0; c < 6; ++c)
    {
        const auto candidate = static_cast<CostCandidate>(c);
        double previous = -1.0;

        for (const std::size_t n : kSizes)
        {
            SelectionInput input;
            input.nBasis = n;
            input.nAux = 1600;
            input.nPairs = n * n / 8;
            input.effectiveThreads = 12;

            const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
            const double cost = Find(ranking, candidate).costSeconds;
            EXPECT_GT(cost, previous) << "cost must be strictly monotone in nBasis";
            previous = cost;
        }
    }
}

TEST(SelectionHeuristicTest, ThreadCountScalesTheCpuCosts) {
    // The fourth priority driver: thread count. The direct cost halves the
    // per-thread law exactly; the CPU ranking scales with the team while
    // the GPU estimate does not (low thread counts favor the GPU path).
    SelectionInput twelve;
    twelve.nBasis = kAlkaneN;
    twelve.nPairs = kAlkanePairs;
    twelve.effectiveThreads = 12;
    twelve.deviceBudgetGiB = 2.6;

    SelectionInput one = twelve;
    one.effectiveThreads = 1;

    const SelectionRanking twelveRanking = qcx::driver::EstimateCandidates(twelve);
    const SelectionRanking oneRanking = qcx::driver::EstimateCandidates(one);
    const double directTwelve = Find(twelveRanking, CostCandidate::kDirect).costSeconds;
    const double directOne = Find(oneRanking, CostCandidate::kDirect).costSeconds;
    EXPECT_NEAR(directOne, directTwelve * 12.0, directTwelve * 1e-6);

    const double gpuTwelve = Find(twelveRanking, CostCandidate::kGpu).costSeconds;
    const double gpuOne = Find(oneRanking, CostCandidate::kGpu).costSeconds;
    EXPECT_NEAR(gpuOne, gpuTwelve, gpuTwelve * 1e-6);
    EXPECT_EQ(*oneRanking.bestSelectable, qcx::io::BuilderKind::kGpu);
}

TEST(SelectionHeuristicTest, NonSingleNodeColumnsAreStubs) {
    // Beyond one node every column reports "not selectable": MPI is not
    // wired and nothing distributes a Fock build today.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.nodeCount = 2;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    EXPECT_FALSE(ranking.bestSelectable.has_value());

    for (const CandidateEstimate& estimate : ranking.candidates)
    {
        EXPECT_FALSE(estimate.selectable);
        EXPECT_NE(estimate.reason.find("distributed execution is not wired"), std::string::npos);
    }

    EXPECT_NE(ranking.reasoning.find("distributed execution is not wired"), std::string::npos);
}

TEST(SelectionHeuristicTest, RiJkIsDeliberatelyExcludedFromTheRanking) {
    // NOT "unwired": ri_jk's builder exists since it landed and runs on both legs
    // since 2026-09-16. It is the only candidate the RANKING excludes, and
    // that is an owner ruling rather than a missing path - it is an
    // approximated-exchange builder, so it is opted into by name and never
    // proposed for a user who did not ask for one. gpu_split is the other
    // candidate that once reported unwired; the batch-partition slice-A scaffold gave
    // it an io kind and the zero device share wires the CPU-only fallback,
    // so ri_jk is alone here for a different reason than it used to be.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.deviceBudgetGiB = 2.6;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    EXPECT_FALSE(Find(ranking, CostCandidate::kRiJk).selectable);
    EXPECT_NE(Find(ranking, CostCandidate::kRiJk).reason.find("deliberately excluded"),
              std::string::npos);
}

TEST(SelectionHeuristicTest, GpuSplitAtTheDefaultZeroFractionIsTheCpuOnlyFallback) {
    // The batch-partition slice-A gate pin: at the default zero device fraction the
    // gpu_split row IS the CPU-only fallback - the k = 1 path - so it
    // is selectable at exactly the direct row's cost, needs no device, and
    // ranks immediately after direct (the exact tie breaks by the fixed
    // candidate order): the pick can never be gpu_split by construction.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.deviceBudgetGiB = 2.6;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    const CandidateEstimate& gpuSplit = Find(ranking, CostCandidate::kGpuSplit);
    EXPECT_TRUE(gpuSplit.selectable);
    EXPECT_NE(gpuSplit.reason.find("fits the budgets"), std::string::npos);
    EXPECT_DOUBLE_EQ(gpuSplit.costSeconds, Find(ranking, CostCandidate::kDirect).costSeconds);

    // The rank directly after direct (a device is present, so gpu itself
    // wins the pick - and it is never gpu_split).
    const auto rankOf = [&ranking](CostCandidate candidate) {
        for (std::size_t i = 0; i < ranking.candidates.size(); ++i)
        {
            if (ranking.candidates[i].candidate == candidate)
            {
                return i;
            }
        }

        ADD_FAILURE() << "candidate not present";
        return ranking.candidates.size();
    };
    EXPECT_EQ(rankOf(CostCandidate::kGpuSplit), rankOf(CostCandidate::kDirect) + 1u);
    ASSERT_TRUE(ranking.bestSelectable.has_value());
    EXPECT_EQ(*ranking.bestSelectable, qcx::io::BuilderKind::kGpu);

    // The no-device case: gpu_split stays selectable (it needs no device
    // at the zero fraction) at exactly the direct row's cost, and the pick
    // can never be gpu_split (the exact tie breaks by the fixed candidate
    // order) nor the GPU tier (no device). The pick itself is whatever CPU
    // row undercuts direct - QFMM at this scale - not a slice-A concern.
    input.deviceBudgetGiB = 0.0;

    const SelectionRanking cpuOnly = qcx::driver::EstimateCandidates(input);
    const CandidateEstimate& fallback = Find(cpuOnly, CostCandidate::kGpuSplit);
    EXPECT_TRUE(fallback.selectable);
    EXPECT_DOUBLE_EQ(fallback.costSeconds, Find(cpuOnly, CostCandidate::kDirect).costSeconds);
    ASSERT_TRUE(cpuOnly.bestSelectable.has_value());
    EXPECT_NE(*cpuOnly.bestSelectable, qcx::io::BuilderKind::kGpuSplit);
    EXPECT_NE(*cpuOnly.bestSelectable, qcx::io::BuilderKind::kGpu);
}

TEST(SelectionHeuristicTest, EngagedGpuSplitIsPricedByTheDeviceShare) {
    // The batch-partition cost semantics once engaged: at a nonzero device fraction
    // g the row needs the device, admits against the GPU floor, and prices
    // max((1 - g) x direct, g x direct / gpuSpeedup) x (1 + overhead) -
    // the two sides' shares of the direct row at their rates, shared
    // completion time plus the coordination overhead.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.deviceBudgetGiB = 2.6;
    input.constants.gpuSplitDeviceFraction = 0.5;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    const CandidateEstimate& gpuSplit = Find(ranking, CostCandidate::kGpuSplit);
    EXPECT_TRUE(gpuSplit.selectable);

    // The share arithmetic on the live direct row (LightPath-inclusive):
    // max(0.5 x X, 0.5 x X / gpuSpeedup) x (1 + gpuSplitOverhead).
    const double directRow = Find(ranking, CostCandidate::kDirect).costSeconds;
    const double expected =
        std::max(0.5 * directRow, 0.5 * directRow / input.constants.gpuSpeedup) *
        (1.0 + input.constants.gpuSplitOverhead);
    EXPECT_NEAR(gpuSplit.costSeconds, expected, 1e-6);

    // Engaged without a device the row refuses like the GPU tier.
    input.deviceBudgetGiB = 0.0;

    const SelectionRanking noDeviceRanking = qcx::driver::EstimateCandidates(input);
    const CandidateEstimate& noDevice = Find(noDeviceRanking, CostCandidate::kGpuSplit);
    EXPECT_FALSE(noDevice.selectable);
    EXPECT_NE(noDevice.reason.find("no CUDA device"), std::string::npos);
}

TEST(SelectionHeuristicTest, EngagedGpuSplitPricingFollowsTheDeviceShare) {
    // The batch-partition slice-A pricing sweep on the live direct row X (the
    // LightPath-inclusive directRow both rows are pushed with): every
    // engaged row prices max((1 - g) x X, g x X / gpuSpeedup) x (1 +
    // overhead) and, from g >= 0.1 up, undercuts the pure-CPU direct row
    // (the device share does real work). The GPU tier keeps the pick at the
    // low shares: its row anchors on the FAST direct anchor (the device
    // path never needs the CPU light path) while the partition's shares
    // ride the LightPath-inflated row - while that row IS inflated, which
    // the corrected direct floor no longer makes true at this fixture's
    // 16 GiB budget (the per-share branch below carries the arithmetic).
    // The pick vocabulary names the kind; the execution is the flag-gated
    // slice-C work.
    SelectionInput input;
    input.nBasis = kAlkaneN;
    input.nPairs = kAlkanePairs;
    input.effectiveThreads = 12;
    input.deviceBudgetGiB = 2.6;

    for (const double share : {0.1, 0.5, 0.9})
    {
        input.constants.gpuSplitDeviceFraction = share;

        const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
        const CandidateEstimate& gpuSplit = Find(ranking, CostCandidate::kGpuSplit);
        ASSERT_TRUE(gpuSplit.selectable) << "share " << share;

        const double directRow = Find(ranking, CostCandidate::kDirect).costSeconds;
        const double expected =
            std::max((1.0 - share) * directRow, share * directRow / input.constants.gpuSpeedup) *
            (1.0 + input.constants.gpuSplitOverhead);
        EXPECT_NEAR(gpuSplit.costSeconds, expected, 1e-6) << "share " << share;
        EXPECT_LT(gpuSplit.costSeconds, directRow) << "share " << share;

        const CandidateEstimate& gpu = Find(ranking, CostCandidate::kGpu);

        // The device tier keeps the pick while its own row stays under the
        // partition's, and at 0.9 it does not: the partition stops riding
        // the direct row and undercuts the pure device row at the high
        // share. The flip was first measured under the 2026-09-17
        // memory-model correction, which moved this fixture's direct floor
        // 8.5533 -> 1.8548 GiB and so took the 16 GiB budget off the
        // LightPath threshold. The floor and the penalty are BOTH gone with
        // the model (see SelectionCostConstants), so no cap reproduces the
        // penalised row now. Pinned as measured behaviour; the correction
        // moved the model's floor, not this test's expectation.
        if (share > 0.5)
        {
            EXPECT_LT(gpuSplit.costSeconds, gpu.costSeconds) << "share " << share;
            ASSERT_TRUE(ranking.bestSelectable.has_value()) << "share " << share;
            EXPECT_EQ(*ranking.bestSelectable, qcx::io::BuilderKind::kGpuSplit)
                << "share " << share;
        } else
        {
            EXPECT_GT(gpuSplit.costSeconds, gpu.costSeconds) << "share " << share;
            ASSERT_TRUE(ranking.bestSelectable.has_value()) << "share " << share;
            EXPECT_EQ(*ranking.bestSelectable, qcx::io::BuilderKind::kGpu) << "share " << share;
        }
    }
}

// The name says what the body asserts. It was `BuilderKindOfMapsEveryWiredCandidate`,
// which claimed completeness while asserting one candidate is NOT wired - true
// before the full-RI builder landed, false since. ri_jk's nullopt is a RULING (an
// approximated-exchange path is opted into by name, never chosen by the ranking, owner 2026-09-16),
// not a statement that its builder is missing.
TEST(SelectionHeuristicTest, BuilderKindOfMapsEveryKindExceptTheDeliberatelyManualOne) {
    using qcx::driver::BuilderKindOf;
    using qcx::io::BuilderKind;
    EXPECT_EQ(BuilderKindOf(CostCandidate::kDirect),
              std::optional<BuilderKind>(BuilderKind::kDirect));
    EXPECT_EQ(BuilderKindOf(CostCandidate::kRiJLink),
              std::optional<BuilderKind>(BuilderKind::kRiJLink));
    EXPECT_EQ(BuilderKindOf(CostCandidate::kQfmm), std::optional<BuilderKind>(BuilderKind::kQfmm));
    EXPECT_EQ(BuilderKindOf(CostCandidate::kGpu), std::optional<BuilderKind>(BuilderKind::kGpu));
    EXPECT_EQ(BuilderKindOf(CostCandidate::kGpuSplit),
              std::optional<BuilderKind>(BuilderKind::kGpuSplit));
    EXPECT_EQ(BuilderKindOf(CostCandidate::kRiJk), std::nullopt);
}

TEST(SelectionHeuristicTest, EmptyBasisIsTotalNotAnError) {
    SelectionInput input;
    input.effectiveThreads = 12;

    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    EXPECT_FALSE(ranking.bestSelectable.has_value());
    EXPECT_NE(ranking.reasoning.find("no basis functions"), std::string::npos);

    for (const CandidateEstimate& estimate : ranking.candidates)
    {
        EXPECT_FALSE(estimate.selectable);
        EXPECT_NE(estimate.reason.find("no basis functions"), std::string::npos);
    }
}

TEST(SelectionHeuristicTest, GpuProbeFloorIsTheRuntimeCommitAlone) {
    // The calibration probe-gate side of the admission rule: the driver gates
    // its device probe on GpuProbeFloorGiB, because the CUDA runtime's
    // load-time commit is already held against the job-object cap when
    // the probe initializes its runtime - below the floor the init is a
    // hard access violation, never a clean CUDA error.
    //
    // BEHAVIOUR CHANGE, 2026-09-17 (the memory model's deletion): the
    // floor used to be the model's base term at n = 0 (0.056 GiB) PLUS
    // this runtime commit (0.35), i.e. 0.406 GiB. The model term is gone
    // with the model, so the floor is the runtime's measured commit
    // alone - a real, small loosening of a safety gate, recorded rather
    // than hidden.
    constexpr double kRuntimeCommitGiB = 0.35;

    EXPECT_NEAR(qcx::driver::GpuProbeFloorGiB(qcx::driver::SelectionCostConstants{}),
                kRuntimeCommitGiB,
                1e-6);

    SelectionInput input;
    input.nBasis = 24;
    input.nPairs = 21;
    input.effectiveThreads = 12;
    input.deviceBudgetGiB = 2.6;
    input.cpuBudgetGiB = 0.4;

    // The estimator itself no longer carries a host admission floor: the
    // GPU candidate's own verdict comes from the DEVICE budget, so with a
    // device present this cell prices it like any other candidate and the
    // direct row keeps the pick on cost.
    const SelectionRanking ranking = qcx::driver::EstimateCandidates(input);
    ASSERT_TRUE(ranking.bestSelectable.has_value());

    const CandidateEstimate& gpu = Find(ranking, CostCandidate::kGpu);
    EXPECT_TRUE(gpu.selectable) << "the device path is no longer excluded by a host memory floor";
}

} // namespace
