// The precision-ladder policy tests (Package B steps 2-4 and 6): the
// budget arithmetic with hand-computed budgets on a fixed fixture (the
// step-2 gate), the band classification and its profile pricing, the
// escalation ratchet's monotonicity, the referee cadence and checks, and
// the method.precision cap. Pure policy - no builders, no device.

#include "qcx/integrals/precision_policy.hpp"

#include <gtest/gtest.h>

namespace {

using qcx::integrals::AccuracyPreset;
using qcx::integrals::ApplyCap;
using qcx::integrals::BandRank;
using qcx::integrals::ClassifyBatch;
using qcx::integrals::ComputeEriBudget;
using qcx::integrals::EriBudgetInputs;
using qcx::integrals::EscalationSchedule;
using qcx::integrals::MorePreciseBand;
using qcx::integrals::PrecisionBand;
using qcx::integrals::PrecisionBandProfile;
using qcx::integrals::PrecisionCap;
using qcx::integrals::PresetEnergyBudget;
using qcx::integrals::ProfileForRatio;
using qcx::integrals::RefereeErrorCheck;

TEST(PrecisionPolicyTest, PresetEnergyBudgetsArePinned) {
    EXPECT_DOUBLE_EQ(PresetEnergyBudget(AccuracyPreset::kLoose), 1e-6);
    EXPECT_DOUBLE_EQ(PresetEnergyBudget(AccuracyPreset::kNormal), 1e-10);
    EXPECT_DOUBLE_EQ(PresetEnergyBudget(AccuracyPreset::kTight), 1e-12);
}

TEST(PrecisionPolicyTest, EriBudgetHandComputed) {
    // The step-2 gate: hand-computed budgets on a fixed fixture.
    // B(kNormal) = 1e-10, slack = kLadderSlackFraction * B = 1e-11, so
    // B_eri = 1e-10 - bScreen - bRi - bDensity - 1e-11.
    const EriBudgetInputs fixture{2.0e-11, 1.0e-11, 3.0e-11};
    const double bEri = ComputeEriBudget(AccuracyPreset::kNormal, fixture);
    EXPECT_NEAR(bEri, 3.0e-11, 1e-26);

    // kLoose: B = 1e-6, slack = 1e-7; B_eri = 1e-6 - 6e-8 - 1e-7 =
    // 8.4e-7.
    EXPECT_NEAR(ComputeEriBudget(AccuracyPreset::kLoose, {1.0e-8, 2.0e-8, 3.0e-8}), 8.4e-7, 1e-21);

    // kTight: B = 1e-12, slack = 1e-13; B_eri = 1e-12 - 1.5e-12 - 1e-13 =
    // -6e-13 (negative - the co-terms exceed the preset budget).
    EXPECT_NEAR(ComputeEriBudget(AccuracyPreset::kTight, {1.0e-12, 5.0e-13, 0.0}), -6.0e-13, 1e-27);

    // Zero co-terms: B_eri = 0.9 * B (the slack alone).
    EXPECT_NEAR(ComputeEriBudget(AccuracyPreset::kNormal, {}), 9.0e-11, 1e-25);
}

TEST(PrecisionPolicyTest, Fp16BoundScaleRelation) {
    EXPECT_NEAR(qcx::integrals::kFp16BoundScale,
                qcx::integrals::kFp16CertifiedEpsilon / qcx::integrals::kCertifiedBandEpsilon,
                1e-9);
}

TEST(PrecisionPolicyTest, ClassifyBatchBands) {
    // The profile with tensor cores: fp16 is reachable.
    PrecisionBandProfile profile;
    profile.fp16TensorCores = true;

    // No budget routes the reference lane.
    EXPECT_EQ(ClassifyBatch(1e-20, 0.0, profile), PrecisionBand::kFp64);
    EXPECT_EQ(ClassifyBatch(1e-20, -1.0, profile), PrecisionBand::kFp64);

    // Delivered-bound comparisons: the fp16 test scales the fp32-unit
    // bound by kFp16BoundScale (~2442.5).
    const double budget = 1e-12;
    const double scale = qcx::integrals::kFp16BoundScale;
    EXPECT_EQ(ClassifyBatch(0.5e-2 * budget / scale, budget, profile), PrecisionBand::kFp16);
    EXPECT_EQ(ClassifyBatch(0.5 * profile.certifiedFraction * budget, budget, profile),
              PrecisionBand::kFp32Certified);
    EXPECT_EQ(ClassifyBatch(0.5 * budget, budget, profile), PrecisionBand::kFp32EvalFp64Accumulate);
    EXPECT_EQ(ClassifyBatch(2.0 * budget, budget, profile), PrecisionBand::kFp64);

    // Boundary conventions: <= enters the band (the certified-lane gate's
    // convention). The fp16 boundary is probed just inside the threshold
    // (the bound round-trips through the division by the band scale, so
    // an exact-boundary probe could land a hair over); the certified and
    // mixed boundaries compare the same product twice and are exact.
    EXPECT_EQ(ClassifyBatch(0.99 * 0.01 * budget / scale, budget, profile), PrecisionBand::kFp16);
    EXPECT_EQ(ClassifyBatch(profile.certifiedFraction * budget, budget, profile),
              PrecisionBand::kFp32Certified);
    EXPECT_EQ(ClassifyBatch(budget, budget, profile), PrecisionBand::kFp32EvalFp64Accumulate);

    // Without tensor cores the fp16 band is unreachable (the CPU rule):
    // the same tiny bound takes the certified band.
    PrecisionBandProfile noTensorCores;
    EXPECT_EQ(ClassifyBatch(0.5e-2 * budget / scale, budget, noTensorCores),
              PrecisionBand::kFp32Certified);
}

TEST(PrecisionPolicyTest, ProfilePricing) {
    // The 1/32 consumer ratio narrows the certified comfort zone (the
    // mixed band's domain shrinks - fp64 accumulation is expensive).
    const PrecisionBandProfile consumer = ProfileForRatio(32.0, true);
    EXPECT_TRUE(consumer.fp16TensorCores);
    EXPECT_NEAR(consumer.certifiedFraction, 0.0125, 1e-12);

    // The half-ratio datacenter class keeps the full comfort width.
    const PrecisionBandProfile datacenter = ProfileForRatio(2.0, true);
    EXPECT_NEAR(datacenter.certifiedFraction, 0.1, 1e-12);

    // The unknown ratio (1.0) does not price and has no fp16 band.
    const PrecisionBandProfile unknown = ProfileForRatio(1.0, false);
    EXPECT_FALSE(unknown.fp16TensorCores);
    EXPECT_NEAR(unknown.certifiedFraction, 0.1, 1e-12);
}

TEST(PrecisionPolicyTest, BandOrderAndRatchetStep) {
    EXPECT_LT(BandRank(PrecisionBand::kFp16), BandRank(PrecisionBand::kFp32Certified));
    EXPECT_LT(BandRank(PrecisionBand::kFp32Certified),
              BandRank(PrecisionBand::kFp32EvalFp64Accumulate));
    EXPECT_LT(BandRank(PrecisionBand::kFp32EvalFp64Accumulate), BandRank(PrecisionBand::kFp64));

    EXPECT_EQ(MorePreciseBand(PrecisionBand::kFp16, PrecisionBand::kFp64), PrecisionBand::kFp64);
    EXPECT_EQ(MorePreciseBand(PrecisionBand::kFp64, PrecisionBand::kFp16), PrecisionBand::kFp64);
    EXPECT_EQ(MorePreciseBand(PrecisionBand::kFp16, PrecisionBand::kFp16), PrecisionBand::kFp16);
}

TEST(PrecisionPolicyTest, ApplyCapConstrains) {
    EXPECT_EQ(ApplyCap(PrecisionBand::kFp16, PrecisionCap::kAuto), PrecisionBand::kFp16);
    EXPECT_EQ(ApplyCap(PrecisionBand::kFp64, PrecisionCap::kAuto), PrecisionBand::kFp64);

    // fp32-certified: fp16 and the mixed band clamp up; the certified and
    // fp64 bands stay.
    EXPECT_EQ(ApplyCap(PrecisionBand::kFp16, PrecisionCap::kFp32Certified),
              PrecisionBand::kFp32Certified);
    EXPECT_EQ(ApplyCap(PrecisionBand::kFp32EvalFp64Accumulate, PrecisionCap::kFp32Certified),
              PrecisionBand::kFp32Certified);
    EXPECT_EQ(ApplyCap(PrecisionBand::kFp32Certified, PrecisionCap::kFp32Certified),
              PrecisionBand::kFp32Certified);
    EXPECT_EQ(ApplyCap(PrecisionBand::kFp64, PrecisionCap::kFp32Certified), PrecisionBand::kFp64);

    // fp64: everything runs the reference lane.
    for (PrecisionBand band : {PrecisionBand::kFp16,
                               PrecisionBand::kFp32Certified,
                               PrecisionBand::kFp32EvalFp64Accumulate,
                               PrecisionBand::kFp64})
    {
        EXPECT_EQ(ApplyCap(band, PrecisionCap::kFp64), PrecisionBand::kFp64);
    }
}

TEST(PrecisionPolicyTest, ScheduleBudgetHandComputed) {
    // kNormal, TC profile, auto cap: B = 1e-10, slack = 1e-11.
    PrecisionBandProfile profile;
    profile.fp16TensorCores = true;
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);

    EXPECT_TRUE(schedule.CoarseBandsAvailable());

    // bScreen 2e-11, bRi 1e-11: the delivery remainder = 1e-10 - 2e-11 -
    // 1e-11 - 1e-11 = 6e-11. bDensity 1e-9: the allowance = min(1e-9,
    // 1e3 * 1e-10 = 1e-7) = 1e-9. T = 6e-11 + 1e-9 = 1.06e-9.
    const EriBudgetInputs inputs{2.0e-11, 1.0e-11, 1.0e-9};
    EXPECT_NEAR(schedule.ClassificationBudget(inputs), 1.06e-9, 1e-22);
    EXPECT_NEAR(schedule.DeliveryRemainder(inputs), 6.0e-11, 1e-24);

    // The allowance caps at the preset multiple.
    const EriBudgetInputs huge{0.0, 0.0, 1e-3};
    EXPECT_NEAR(schedule.ClassificationBudget(huge), 9.0e-11 + 1.0e-7, 1e-21);

    // A negative remainder routes everything fp64 (budget 0).
    const EriBudgetInputs exhausted{9.5e-11, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(schedule.ClassificationBudget(exhausted), 0.0);
    EXPECT_DOUBLE_EQ(schedule.DeliveryRemainder(exhausted), 0.0);

    // kTight keeps the coarse bands off entirely (the strict-pins
    // contract).
    EscalationSchedule tight(AccuracyPreset::kTight, profile, PrecisionCap::kAuto);
    EXPECT_FALSE(tight.CoarseBandsAvailable());
    EXPECT_DOUBLE_EQ(tight.ClassificationBudget({}), 0.0);
    EXPECT_DOUBLE_EQ(tight.DeliveryRemainder({}), 0.0);
}

TEST(PrecisionPolicyTest, RatchetIsMonotoneUpward) {
    PrecisionBandProfile profile;
    profile.fp16TensorCores = true;
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);

    const double scale = qcx::integrals::kFp16BoundScale;
    // The bound fits the fp16 band at the generous budget (bound * scale
    // <= 0.01 * 1e-10) but exceeds every coarse band at the tight one.
    const double bound = 2e-16;

    // A generous first budget classifies fp16.
    const PrecisionBand first = schedule.RatchetBand(0, bound, 1e-10);
    EXPECT_EQ(first, PrecisionBand::kFp16);

    // A tight second budget classifies fp64 - the ratchet steps up.
    const PrecisionBand second = schedule.RatchetBand(0, bound, 1e-16);
    EXPECT_EQ(second, PrecisionBand::kFp64);

    // A generous budget again (the DIIS bounce) must HOLD the fp64 band,
    // never switch back - monotone downward in error.
    const PrecisionBand third = schedule.RatchetBand(0, bound, 1e-10);
    EXPECT_EQ(third, PrecisionBand::kFp64);

    // Fresh batches classify independently.
    const PrecisionBand other = schedule.RatchetBand(1, 100.0, 1e-14);
    EXPECT_EQ(other, PrecisionBand::kFp64);
}

TEST(PrecisionPolicyTest, RatchetAppliesTheCap) {
    PrecisionBandProfile profile;
    profile.fp16TensorCores = true;

    EscalationSchedule capped(AccuracyPreset::kNormal, profile, PrecisionCap::kFp32Certified);
    EXPECT_EQ(capped.RatchetBand(0, 1e-30, 1e-8), PrecisionBand::kFp32Certified);

    EscalationSchedule strict(AccuracyPreset::kNormal, profile, PrecisionCap::kFp64);
    EXPECT_EQ(strict.RatchetBand(0, 1e-30, 1e-8), PrecisionBand::kFp64);
}

TEST(PrecisionPolicyTest, RefereeCadence) {
    PrecisionBandProfile profile; // refereeEvery 0 = the preset default.
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);

    // kNormal's default interval is 5: the referee fires on 5, 10, ... and
    // on the final iteration regardless.
    for (int iteration = 1; iteration <= 12; ++iteration)
    {
        EXPECT_EQ(schedule.RefereeDue(iteration, false), iteration % 5 == 0);
    }

    EXPECT_TRUE(schedule.RefereeDue(3, true));
    EXPECT_FALSE(schedule.RefereeDue(0, true));
    EXPECT_FALSE(schedule.RefereeDue(0, false));

    // A profile override wins over the preset default.
    profile.refereeEvery = 2;
    EscalationSchedule frequent(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);
    EXPECT_TRUE(frequent.RefereeDue(2, false));
    EXPECT_FALSE(frequent.RefereeDue(3, false));
    EXPECT_TRUE(frequent.RefereeDue(4, false));
}

TEST(PrecisionPolicyTest, RefereeDeliveryContract) {
    PrecisionBandProfile profile;
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);

    const EriBudgetInputs inputs{0.0, 0.0, 0.0}; // remainder = 9e-11.

    // A committed sum inside the delivery remainder passes.
    EXPECT_TRUE(schedule.RefereeDelivery(8.0e-11, inputs).has_value());

    // Over the remainder is a hard error, never a silent drift.
    EXPECT_FALSE(schedule.RefereeDelivery(9.0e-11 + 1e-15, inputs).has_value());
}

TEST(PrecisionPolicyTest, RefereeErrorCheckContract) {
    // The delivered Fock deviation must stay inside the committed bound.
    EXPECT_TRUE(RefereeErrorCheck(1.0e-12, 2.0e-12).has_value());
    EXPECT_FALSE(RefereeErrorCheck(2.0e-12, 1.0e-12).has_value());
    // A committed sum of 0 demands the bitwise fp64 build.
    EXPECT_TRUE(RefereeErrorCheck(0.0, 0.0).has_value());
    EXPECT_FALSE(RefereeErrorCheck(1e-30, 0.0).has_value());
}

TEST(PrecisionPolicyTest, ScheduleResetClearsRunState) {
    PrecisionBandProfile profile;
    profile.fp16TensorCores = true;
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);

    EXPECT_EQ(schedule.RatchetBand(0, 1e-30, 1e-8), PrecisionBand::kFp16);
    schedule.CommitCoarseBoundSum(1.0);
    EXPECT_DOUBLE_EQ(schedule.LastCommittedSum(), 1.0);

    schedule.Reset();
    EXPECT_DOUBLE_EQ(schedule.LastCommittedSum(), 0.0);
    // The ratchet state is cleared: the same batch re-classifies from
    // scratch under the tight budget (the bound exceeds every coarse
    // band there).
    EXPECT_EQ(schedule.RatchetBand(0, 2e-16, 1e-16), PrecisionBand::kFp64);
}

} // namespace
