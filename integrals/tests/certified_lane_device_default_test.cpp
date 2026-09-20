// The certified fp32 lane's DEVICE-GOVERNED DEFAULT (the
// owner's ruling of 2026-09-12: fp32 "may be far more useful on GPUs and
// AVX machines", so the lane's default is HARDWARE-AWARE, never a blanket
// switch). The probe's verdict is precision_policy.hpp
// CertifiedLaneDefaultForRatio; the lane's ONE resolution point is
// fock_build.hpp ResolveCertifiedLane. Both are pinned here, and the
// numbers every case reads are emitted through RecordProperty so a PASSING
// gate still carries them rather than only a
// failing one - including the real device probe's own measurement, which
// is recorded on every host (the non-CUDA build's documented "unknown"
// profile included).

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/backend/gpu_compute_profile.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <string>

namespace {

using qcx::backend::GpuComputeProfile;
using qcx::integrals::AccuracyPreset;
using qcx::integrals::CertifiedLaneDefaultForRatio;
using qcx::integrals::FockBuildOptions;
using qcx::integrals::kCertifiedLaneMinRatio;
using qcx::integrals::ResolveCertifiedLane;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The measured ratios this suite reasons about, named so the provenance
// travels with the assertion: the local Quadro T1000's measurement
// (2416 / 77.5 GFLOP/s), the datacenter reference class, and the
// "unknown" value a failed or absent probe reports.
constexpr double kT1000Ratio = 31.2;
constexpr double kDatacenterRatio = 2.0;
constexpr double kUnknownRatio = 1.0;

// RecordProperty is a static member of ::testing::Test, so it resolves
// inside a TEST body and not from a free helper: this formats what the
// bodies record.
std::string RatioKey(const char* const label) {
    return std::string("verdict_") + label + "_ratio";
}

std::string DefaultKey(const char* const label) {
    return std::string("verdict_") + label + "_default";
}

// The fixture density: the same seeded generator fock_build_test.cpp's
// PhysicalDensity uses (its seed and distribution, so the kLoose runs here
// route the same s-pair quartets the certified-lane pins there do). Kept
// local - the two suites pin different contracts and must not share a
// mutable fixture.
Eigen::MatrixXd FixtureDensity(std::size_t n) {
    std::mt19937_64 rng(20260817);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

// H = T + V (the gpu_precision_ladder_test.cpp helper, test-local).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoreHamiltonian(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return std::move(*kinetic);
}

// One kLoose H2O/STO-3G build at the given options; the certified bound sum
// it delivers and the Fock it produces. Serial, so the comparison is
// deterministic (the builder's own reproducibility floor needs no slack
// here).
struct LaneProbe {
    double boundSum = -1.0;
    Eigen::MatrixXd fock;
};

LaneProbe RunLooseLane(const qcx::molecule::Molecule& molecule,
                       const qcx::basisset::BasisSet& basisSet,
                       const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& core,
                       qcx::integrals::FockBuildOptions options) {
    options.accuracy = AccuracyPreset::kLoose;
    options.maxParallelChunks = 1;

    auto builder = qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, core, options);
    EXPECT_TRUE(builder.has_value()) << (builder.has_value() ? "" : builder.error().message);

    if (!builder.has_value())
    {
        return {};
    }

    auto density = ToTensor(FixtureDensity(7));
    EXPECT_TRUE(density.has_value());

    if (!density.has_value())
    {
        return {};
    }

    double boundSum = -1.0;
    auto fock = builder->BuildFock(*density, &boundSum);
    EXPECT_TRUE(fock.has_value()) << (fock.has_value() ? "" : fock.error().message);

    if (!fock.has_value())
    {
        return {};
    }

    return LaneProbe{boundSum, ToMatrix(*fock)};
}

} // namespace

// The rule itself: the measured ratio decides, and the threshold is the
// repo's own fp64-premium boundary (ProfileForRatio's 4x knee) rather than
// an invented number. The two measured anchors of the ruling bracket it -
// the AVX2 host case (ratio ~2) whose lane measured 13-16% SLOWER, and the
// local T1000 (31.2) - and the unknown value sits below it by construction.
TEST(CertifiedLaneDeviceDefaultTest, TheVerdictFollowsTheMeasuredRatio) {
    RecordProperty("kCertifiedLaneMinRatio", std::to_string(kCertifiedLaneMinRatio));

    const struct Case {
        const char* label;
        double ratio;
        bool expected;
    } cases[] = {
        {"t1000", kT1000Ratio, true},
        {"datacenter", kDatacenterRatio, false},
        {"unknown_no_probe", kUnknownRatio, false},
        {"at_threshold", kCertifiedLaneMinRatio, false},
        {"above_threshold", kCertifiedLaneMinRatio + 1e-9, true},
    };

    for (const Case& testCase : cases)
    {
        const bool verdict = CertifiedLaneDefaultForRatio(testCase.ratio);
        RecordProperty(RatioKey(testCase.label), std::to_string(testCase.ratio));
        RecordProperty(DefaultKey(testCase.label), verdict ? "on" : "off");
        EXPECT_EQ(verdict, testCase.expected) << testCase.label;
    }
}

// The conservative interim: a build whose options carry no probe result
// lands on OFF, so a caller that never asked and never probed does not get
// the lane. This is the "no probe applies" case of the ruling.
TEST(CertifiedLaneDeviceDefaultTest, TheUnseededDefaultIsTheConservativeInterim) {
    const FockBuildOptions defaults;

    EXPECT_FALSE(defaults.useCertifiedMixedPrecision.has_value())
        << "the request must be UNSET by default, not set to a value: the device decides";
    EXPECT_EQ(defaults.deviceComputeProfile.fp32ToFp64Ratio, kUnknownRatio)
        << "the unseeded profile must be the documented unknown profile";

    const bool requested = ResolveCertifiedLane(defaults);
    RecordProperty("unseeded_profile_ratio",
                   std::to_string(defaults.deviceComputeProfile.fp32ToFp64Ratio));
    RecordProperty("unseeded_default", requested ? "on" : "off");
    EXPECT_FALSE(requested) << "no probe applies -> the lane is off";
}

// The device leads the default: with a probe result supplied and the
// request left unset, the lane's default IS the verdict - on for a device
// whose fp32 premium pays, off for one whose ratio says it does not.
TEST(CertifiedLaneDeviceDefaultTest, TheSeededDeviceProfileGovernsTheUnsetDefault) {
    FockBuildOptions fast;
    fast.deviceComputeProfile.fp32ToFp64Ratio = kT1000Ratio;
    EXPECT_TRUE(ResolveCertifiedLane(fast)) << "ratio 31.2: the lane's default is on";

    FockBuildOptions slow;
    slow.deviceComputeProfile.fp32ToFp64Ratio = kDatacenterRatio;
    EXPECT_FALSE(ResolveCertifiedLane(slow)) << "ratio 2.0: the lane's default is off";

    RecordProperty("seeded_t1000_default", ResolveCertifiedLane(fast) ? "on" : "off");
    RecordProperty("seeded_ratio2_default", ResolveCertifiedLane(slow) ? "on" : "off");
}

// An explicit request resolves at one point and is honoured as
// written. The device verdict moves the DEFAULT; it never overrides a
// caller who asked.
TEST(CertifiedLaneDeviceDefaultTest, TheExplicitRequestWinsOverTheDevice) {
    FockBuildOptions forcedOn;
    forcedOn.useCertifiedMixedPrecision = true;
    forcedOn.deviceComputeProfile.fp32ToFp64Ratio = kDatacenterRatio;
    EXPECT_TRUE(ResolveCertifiedLane(forcedOn))
        << "an explicit request is not lowered by the device";

    FockBuildOptions forcedOff;
    forcedOff.useCertifiedMixedPrecision = false;
    forcedOff.deviceComputeProfile.fp32ToFp64Ratio = kT1000Ratio;
    EXPECT_FALSE(ResolveCertifiedLane(forcedOff))
        << "an explicit request is not raised by the device";
}

// The ruling's OFF half, end to end: a kLoose build whose unset request
// meets a low-ratio device runs the fp64 lane - the certified bound sum is
// the OBSERVED zero the lane-off path produces, and the Fock is the
// laneless build's, bit for bit.
TEST(CertifiedLaneDeviceDefaultTest, TheDeviceVerdictOffIsTheLaneDisabledBuild) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    FockBuildOptions byDevice;
    byDevice.deviceComputeProfile.fp32ToFp64Ratio = kDatacenterRatio;
    const LaneProbe deviceOff = RunLooseLane(*molecule, *basis, *core, byDevice);

    FockBuildOptions explicitly = byDevice;
    explicitly.useCertifiedMixedPrecision = false;
    const LaneProbe laneOff = RunLooseLane(*molecule, *basis, *core, explicitly);

    RecordProperty("device_off_bound_sum", std::to_string(deviceOff.boundSum));
    RecordProperty("lane_off_bound_sum", std::to_string(laneOff.boundSum));
    EXPECT_DOUBLE_EQ(deviceOff.boundSum, 0.0)
        << "the device verdict off must deliver the observed zero, not a placeholder";
    EXPECT_EQ(deviceOff.fock, laneOff.fock) << "the verdict-off build IS the lane-disabled build";
}

// The ruling's ON half, end to end: the same kLoose build whose unset
// request meets a high-ratio device routes quartets through the lane and
// delivers a strictly positive certified bound - the same build an explicit
// request produces.
TEST(CertifiedLaneDeviceDefaultTest, TheDeviceVerdictOnIsTheLaneEngagedBuild) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    FockBuildOptions byDevice;
    byDevice.deviceComputeProfile.fp32ToFp64Ratio = kT1000Ratio;
    const LaneProbe deviceOn = RunLooseLane(*molecule, *basis, *core, byDevice);

    FockBuildOptions unseeded;
    const LaneProbe noProbe = RunLooseLane(*molecule, *basis, *core, unseeded);

    RecordProperty("device_on_bound_sum", std::to_string(deviceOn.boundSum));
    RecordProperty("unseeded_bound_sum", std::to_string(noProbe.boundSum));
    EXPECT_GT(deviceOn.boundSum, 0.0) << "the ratio-31.2 device must route quartets";
    EXPECT_DOUBLE_EQ(noProbe.boundSum, 0.0) << "the unseeded build must stay off";
}

// The real probe, recorded rather than assumed: this host's own measured
// numbers go into the PASSING record, and the verdict read off them is
// required to be the one the rule gives. On a device-less host (or a
// non-CUDA build) the probe reports its documented unknown profile and the
// assertion is the interim's - the suite does not skip, because "no probe
// applies -> off" is itself a case the ruling names.
TEST(CertifiedLaneDeviceDefaultTest, TheHostProbeMeasurementIsRecorded) {
    const GpuComputeProfile probed = qcx::backend::DetectGpuComputeProfile(0);

    RecordProperty("probe_fp32_gflops", std::to_string(probed.fp32Gflops));
    RecordProperty("probe_fp64_gflops", std::to_string(probed.fp64Gflops));
    RecordProperty("probe_ratio", std::to_string(probed.fp32ToFp64Ratio));
    RecordProperty("probe_tensor_cores", probed.tensorCores ? "yes" : "no");
    RecordProperty("probe_verdict",
                   CertifiedLaneDefaultForRatio(probed.fp32ToFp64Ratio) ? "on" : "off");

    EXPECT_EQ(CertifiedLaneDefaultForRatio(probed.fp32ToFp64Ratio),
              probed.fp32ToFp64Ratio > kCertifiedLaneMinRatio)
        << "the verdict must be the rule applied to the probe's own number";
    EXPECT_GT(probed.fp32ToFp64Ratio, 0.0) << "the probe's ratio is never zero or negative";
}
