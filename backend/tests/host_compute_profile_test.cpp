// The host compute-profile probe's contract.
//
// The assertions here are STRUCTURAL and load-tolerant by construction: the
// measured flag, the GFLOP/s identities, and a wide plausibility band on the
// ratio. The probe is a TIMING measurement, so a tight band on its value
// would be a pin on this laptop's clock rather than on the probe, and the
// absolute GFLOP/s are clock- and core-dependent by design (the header says
// so) - they are RECORDED here, never banded. The numbers ride the passing
// record through RecordProperty, the pattern
// integrals/tests/certified_lane_device_default_test.cpp:307-321 uses for the
// device probe: a passing gate must still emit its numbers (operating-model
// 9.16).

#include "qcx/backend/cpu_features.hpp"
#include "qcx/backend/host_compute_profile.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <string>

namespace {

using qcx::backend::HostComputeProfile;

// The documented unknown profile: what a failed probe returns, and what every
// field of a default-constructed profile is. The 1.0 ratio is the load-bearing
// member - it is below the certified lane's threshold by construction, so the
// failure path needs no second branch to get the verdict right.
TEST(HostComputeProfileTest, UnknownProfileIsTheUnmeasuredDefault) {
    const HostComputeProfile unknown;

    EXPECT_FALSE(unknown.measured);
    EXPECT_FALSE(unknown.simdLane);
    EXPECT_DOUBLE_EQ(unknown.fp32ToFp64Ratio, 1.0);
    EXPECT_DOUBLE_EQ(unknown.fp32Gflops, 0.0);
    EXPECT_DOUBLE_EQ(unknown.fp64Gflops, 0.0);
    EXPECT_EQ(unknown.pairs, 0);
    EXPECT_DOUBLE_EQ(unknown.ratioSpread, 0.0);
}

// The real probe, measured rather than assumed. This test does NOT skip on a
// machine without AVX2: the scalar reference lane is the designed answer
// there, and "no SIMD lane" is itself a case the instrument names (its ~1.0
// ratio says "this machine has no fp32 premium", which is a reading, not an
// absence of one).
TEST(HostComputeProfileTest, TheProbeMeasurementIsRecorded) {
    const HostComputeProfile probed = qcx::backend::DetectHostComputeProfile();

    RecordProperty("probe_measured", probed.measured ? "yes" : "no");
    RecordProperty("probe_simd_lane", probed.simdLane ? "yes" : "no");
    RecordProperty("probe_fp32_gflops", std::to_string(probed.fp32Gflops));
    RecordProperty("probe_fp64_gflops", std::to_string(probed.fp64Gflops));
    RecordProperty("probe_ratio", std::to_string(probed.fp32ToFp64Ratio));
    RecordProperty("probe_pairs", std::to_string(probed.pairs));
    RecordProperty("probe_ratio_spread", std::to_string(probed.ratioSpread));

    // The lane choice is the shared predicate's answer, never a second
    // detector: the probe and the AVX2 kernels must agree on what this
    // machine may execute.
    EXPECT_EQ(probed.simdLane, qcx::backend::CpuHasAvx2Fma());

    // A machine whose lanes can be timed must produce a reading. A false
    // here is the defect this pin exists for: a probe that silently reports
    // "unknown" on every machine is a recorded number nobody measured.
    ASSERT_TRUE(probed.measured) << "the probe failed to measure on a machine whose FMA lanes can "
                                    "be timed; the certified lane's default is then assumed, not "
                                    "measured";

    EXPECT_GT(probed.fp32Gflops, 0.0);
    EXPECT_GT(probed.fp64Gflops, 0.0);
    EXPECT_TRUE(std::isfinite(probed.fp32Gflops));
    EXPECT_TRUE(std::isfinite(probed.fp64Gflops));
    EXPECT_TRUE(std::isfinite(probed.fp32ToFp64Ratio));
    EXPECT_GT(probed.fp32ToFp64Ratio, 0.0);
    EXPECT_GE(probed.ratioSpread, 0.0);
    EXPECT_TRUE(std::isfinite(probed.ratioSpread));
    EXPECT_GE(probed.pairs, 1) << "a measured profile carries the pairs its ratio came from";

    // The ratio IS the two lanes' quotient - not a third quantity that could
    // drift away from them.
    EXPECT_DOUBLE_EQ(probed.fp32ToFp64Ratio, probed.fp32Gflops / probed.fp64Gflops);

    // A wide plausibility band, not a machine pin: every possible machine
    // width lands inside it (a scalar host reads ~1, AVX2 and AVX-512 hosts
    // read ~2), so a reading outside it is a broken instrument rather than an
    // unusual machine.
    EXPECT_GT(probed.fp32ToFp64Ratio, 0.2);
    EXPECT_LT(probed.fp32ToFp64Ratio, 16.0);
}

} // namespace
