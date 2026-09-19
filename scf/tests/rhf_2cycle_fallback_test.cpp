// The RHF period-2 seed-fallback tests: the M2 verification found
// 4/96 records where the GWH-seeded Roothaan map locks a parity-pinned
// period-2 limit cycle instead of the fixed point (CO/6-311G** alternates
// -124.8077797352 <-> -102.0254233499 vs psi4's -112.7680161861, pinned to
// 2e-13 at iterations 100-103). This file carries (a) the detector
// truth-table (pure-function unit tests on the internal seam, the
// diis_test.cpp precedent) and (b) the 4 M2 records as ctest fixtures
// pinning the post-fix contract: the GWH-primary 2-cycle is detected and
// the loop restarts exactly once from the average of the cycle's two
// member densities (the measured basin escape; the P = 0 restart locks the
// same cycle), converging within the 2e-6 budget.
#include "fast_test_mode.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/rhf.hpp"
#include "scf_common.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::scf::internal::DetectPeriodTwoCycle;
using qcx::scf::internal::DetectStationaryCycle;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The M2 CO/6-311G** cycle members: the parity-pinned 2-cycle pair.
constexpr double kCoCycleOdd = -124.8077797352;
constexpr double kCoCycleEven = -102.0254233499;

TEST(PeriodTwoDetectorTest, FiresOnParityPinnedHistory) {
    // The CO/6-311G** pair repeated three times: two full periods, the
    // verified signature.
    const std::array<double, 6> history{
        kCoCycleEven, kCoCycleOdd, kCoCycleEven, kCoCycleOdd, kCoCycleEven, kCoCycleOdd};
    EXPECT_TRUE(DetectPeriodTwoCycle(history,
                                     qcx::scf::internal::kTwoCycleParityTolerance,
                                     qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, SilentOnConvergedFixedPoint) {
    // A converged fixed point is constant: the parity gaps pass trivially
    // but the alternation floor discriminates (class gap ~ 0).
    const std::array<double, 6> history{
        -76.041844, -76.041844, -76.041844, -76.041844, -76.041844, -76.041844};
    EXPECT_FALSE(DetectPeriodTwoCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, SilentOnGeometricallyDampedOscillation) {
    // A damped oscillation's parity gaps are large while the amplitude is
    // still meaningful - never parity-stable with a live alternation.
    const std::array<double, 6> history{1.0, -1.0, 0.5, -0.5, 0.25, -0.25};
    EXPECT_FALSE(DetectPeriodTwoCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, SilentOnAperiodicSequence) {
    // A chaotic trajectory is not parity-stable.
    const std::array<double, 6> history{-96.66, -58.30, -93.23, -58.38, -90.11, -61.90};
    EXPECT_FALSE(DetectPeriodTwoCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, SilentOnShortHistory) {
    // The window IS the confirmation: five energies cannot confirm a period.
    const std::array<double, 5> history{
        kCoCycleEven, kCoCycleOdd, kCoCycleEven, kCoCycleOdd, kCoCycleEven};
    EXPECT_FALSE(DetectPeriodTwoCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, SilentOnBelowFloorCycle) {
    // A "quiet" 2-cycle alternating by 1e-8 sits far below the 1e-6
    // alternation floor - deliberately NOT rescued.
    const std::array<double, 6> history{1.0, 1.0 + 1e-8, 1.0, 1.0 + 1e-8, 1.0, 1.0 + 1e-8};
    EXPECT_FALSE(DetectPeriodTwoCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, ParityBoundaryIsStrict) {
    // One parity gap exactly AT the tolerance must not fire (strict <).
    // Even class drifts by exactly 1e-8 per step; the odd class is pinned
    // far away, so the class gap is huge and only the strict parity
    // comparison can veto.
    const std::array<double, 6> history{0.0, 10.0, 1e-8, 10.0, 2e-8, 10.0};
    EXPECT_FALSE(DetectPeriodTwoCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, AlternationBoundaryIsStrict) {
    // A perfect 2-cycle whose class gap is exactly AT the floor must not
    // fire (strict >).
    const std::array<double, 6> history{0.0, 1e-6, 0.0, 1e-6, 0.0, 1e-6};
    EXPECT_FALSE(DetectPeriodTwoCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor));
}

TEST(PeriodTwoDetectorTest, FiresWithSmallNonzeroParityGap) {
    // A real cycle pinned to 5e-9 (well inside the tolerance) fires.
    const std::array<double, 6> history{0.0, 1.0, 5e-9, 1.0, 1e-8, 1.0};
    EXPECT_TRUE(DetectPeriodTwoCycle(history,
                                     qcx::scf::internal::kTwoCycleParityTolerance,
                                     qcx::scf::internal::kTwoCycleAlternationFloor));
}

// The generalized stationarity watchdog: DetectStationaryCycle keeps
// DetectPeriodTwoCycle as its k = 2 branch (bit-identical firing) and scans
// the higher lags 3..4 behind the same flag. The fixture system: a
// C12H26-scale alternation spread (0.3-0.4 Ha between the members, the
// floor's regime) so the member gaps dwarf the 1e-6 alternation floor.
TEST(StationaryCycleDetectorTest, FiresPeriodThreeAsLagThree) {
    // Two full periods of (a, b, c): the lag-3 gaps vanish while the
    // shorter lags mix the distinct members (above the floor).
    const std::array<double, 6> history{-469.2, -469.5, -469.8, -469.2, -469.5, -469.8};
    int period = 0;
    EXPECT_TRUE(DetectStationaryCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor,
                                      period));
    EXPECT_EQ(period, 3);
}

TEST(StationaryCycleDetectorTest, FiresPeriodFourAsLagFour) {
    // One and a half periods of (a, b, c, d): the lag-4 gaps vanish over
    // the window's two phase checks; lags 2 and 3 mix the members.
    const std::array<double, 6> history{-469.2, -469.4, -469.6, -469.8, -469.2, -469.4};
    int period = 0;
    EXPECT_TRUE(DetectStationaryCycle(history,
                                      qcx::scf::internal::kTwoCycleParityTolerance,
                                      qcx::scf::internal::kTwoCycleAlternationFloor,
                                      period));
    EXPECT_EQ(period, 4);
}

TEST(StationaryCycleDetectorTest, KeepsPeriodTwoBranchBitIdentical) {
    // Every history of the truth table fires (or stays silent) exactly
    // as DetectPeriodTwoCycle does, and a fire reports period 2: the
    // generalized detector is a superset only at the higher lags, never at
    // k = 2.
    const std::array<std::array<double, 6>, 6> histories{{
        {kCoCycleEven, kCoCycleOdd, kCoCycleEven, kCoCycleOdd, kCoCycleEven, kCoCycleOdd},
        {-76.041844, -76.041844, -76.041844, -76.041844, -76.041844, -76.041844},
        {1.0, -1.0, 0.5, -0.5, 0.25, -0.25},
        {-96.66, -58.30, -93.23, -58.38, -90.11, -61.90},
        {1.0, 1.0 + 1e-8, 1.0, 1.0 + 1e-8, 1.0, 1.0 + 1e-8},
        {0.0, 1.0, 5e-9, 1.0, 1e-8, 1.0},
    }};

    for (const auto& history : histories)
    {
        int period = 0;
        const bool fires = DetectStationaryCycle(history,
                                                 qcx::scf::internal::kTwoCycleParityTolerance,
                                                 qcx::scf::internal::kTwoCycleAlternationFloor,
                                                 period);
        const bool firesAtTwo = DetectPeriodTwoCycle(history,
                                                     qcx::scf::internal::kTwoCycleParityTolerance,
                                                     qcx::scf::internal::kTwoCycleAlternationFloor);
        EXPECT_EQ(fires, firesAtTwo);
        EXPECT_EQ(period, fires ? 2 : 0);
    }
}

TEST(StationaryCycleDetectorTest, SilentOnShortHistory) {
    // The full-window discipline holds at every lag: five energies cannot
    // confirm any period.
    const std::array<double, 5> history{-469.2, -469.5, -469.8, -469.2, -469.5};
    int period = 7;
    EXPECT_FALSE(DetectStationaryCycle(history,
                                       qcx::scf::internal::kTwoCycleParityTolerance,
                                       qcx::scf::internal::kTwoCycleAlternationFloor,
                                       period));
    EXPECT_EQ(period, 7); // untouched on no fire.
}

TEST(StationaryCycleDetectorTest, SilentOnQuietPeriodThree) {
    // A "quiet" 3-cycle alternating by 2e-9 per member: the lag-3 gaps
    // vanish, but the shorter-lag veto applies the floor discipline -
    // the lag-2 median sits at the member-spacing scale, far below the
    // alternation floor, so no lag fires (a sub-floor cycle is never
    // rescued; the quiet-cycle rule extended to every k).
    const std::array<double, 6> history{1.0 - 2e-9, 1.0, 1.0 + 2e-9, 1.0 - 2e-9, 1.0, 1.0 + 2e-9};
    int period = 0;
    EXPECT_FALSE(DetectStationaryCycle(history,
                                       qcx::scf::internal::kTwoCycleParityTolerance,
                                       qcx::scf::internal::kTwoCycleAlternationFloor,
                                       period));
    EXPECT_EQ(period, 0);
}

TEST(StationaryCycleDetectorTest, SilentOnQuietPeriodTwoAtAnyLag) {
    // The quiet 2-cycle (the deliberately-not-rescued case) must not
    // fire at a higher lag either: the lag-2 median sits at the floor and
    // vetoes every longer lag even though the lag-3 gaps are tiny.
    const std::array<double, 6> history{1.0, 1.0 + 5e-9, 1.0, 1.0 + 5e-9, 1.0, 1.0 + 5e-9};
    int period = 0;
    EXPECT_FALSE(DetectStationaryCycle(history,
                                       qcx::scf::internal::kTwoCycleParityTolerance,
                                       qcx::scf::internal::kTwoCycleAlternationFloor,
                                       period));
    EXPECT_EQ(period, 0);
}

TEST(StationaryCycleDetectorTest, SilentOnMonotoneDrift) {
    // A monotone drift never fires at any lag: its gaps grow with the lag,
    // so no lag-k median sits below the parity tolerance (fast drift), and
    // a slow drift's short lags sit at the floor and veto the scan.
    const std::array<double, 6> fastDrift{1.0, 1.01, 1.02, 1.03, 1.04, 1.05};
    const std::array<double, 6> slowDrift{
        1.0, 1.0 + 1e-5, 1.0 + 2e-5, 1.0 + 3e-5, 1.0 + 4e-5, 1.0 + 5e-5};
    int period = 0;
    EXPECT_FALSE(DetectStationaryCycle(fastDrift,
                                       qcx::scf::internal::kTwoCycleParityTolerance,
                                       qcx::scf::internal::kTwoCycleAlternationFloor,
                                       period));
    EXPECT_EQ(period, 0);
    EXPECT_FALSE(DetectStationaryCycle(slowDrift,
                                       qcx::scf::internal::kTwoCycleParityTolerance,
                                       qcx::scf::internal::kTwoCycleAlternationFloor,
                                       period));
    EXPECT_EQ(period, 0);
}

TEST(StationaryCycleDetectorTest, SilentOnUnconfirmableLongPeriod) {
    // A 6-member cycle cannot be confirmed inside a 6-energy window: no
    // lag has two full periods of phase checks, so the sequence reads as a
    // wander and no lag fires (the window bounds the confirmable periods).
    const std::array<double, 6> history{-469.2, -469.3, -469.4, -469.5, -469.6, -469.7};
    int period = 0;
    EXPECT_FALSE(DetectStationaryCycle(history,
                                       qcx::scf::internal::kTwoCycleParityTolerance,
                                       qcx::scf::internal::kTwoCycleAlternationFloor,
                                       period));
    EXPECT_EQ(period, 0);
}

TEST(StationaryCycleDetectorTest, ParityBoundaryIsStrictAtLagThree) {
    // The strict < boundary holds at every lag: a lag-3 gap median sitting
    // exactly AT the parity tolerance must not fire. The three lag-3 gaps
    // are {0, 1e-8, 1.0} - the median is the exact 1e-8 double (E3 - E0
    // with E0 = 0), which is not strictly below the tolerance.
    const std::array<double, 6> history{0.0, 1.0, 2.0, 1e-8, 1.0, 3.0};
    int period = 0;
    EXPECT_FALSE(DetectStationaryCycle(history,
                                       qcx::scf::internal::kTwoCycleParityTolerance,
                                       qcx::scf::internal::kTwoCycleAlternationFloor,
                                       period));
    EXPECT_EQ(period, 0);
}

// H = T + V from the one-electron engines (the direct_rhf_test pattern).
qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
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

// The FockBuilderFn adapter (the direct_rhf_test pattern): the seam hands
// over the spin-summed density D = 2 C_occ C_occ^T, the builder contracts
// the spatial density rho = D/2.
qcx::scf::FockBuilderFn MakeDirectFockBuilder(const qcx::integrals::DirectJkFockBuilder& builder) {
    return [builder](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto densityTensor = ToTensor(0.5 * density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        auto fock = builder.BuildFock(*densityTensor);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        return ToMatrix(*fock);
    };
}

// The M2 record geometries (H&H 1979 constants, angstrom -> bohr; the
// M2 manifest's cartesian_angstrom, verbatim).
qcx::Result<qcx::molecule::Molecule> MakeCo() {
    constexpr double kAngstromToBohr = qcx::molecule::kAngstromToBohr;
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.1283 * kAngstromToBohr;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"C", 6, 0.0}, {"O", 8, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

qcx::Result<qcx::molecule::Molecule> MakeHcn() {
    constexpr double kAngstromToBohr = qcx::molecule::kAngstromToBohr;
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.0655 * kAngstromToBohr;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = 2.2187 * kAngstromToBohr;
    (*coordinates)(2, 1) = 0.0;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"C", 6, 0.0}, {"N", 7, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

qcx::Result<qcx::molecule::Molecule> MakeH2co() {
    constexpr double kAngstromToBohr = qcx::molecule::kAngstromToBohr;
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({4, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 1.208 * kAngstromToBohr;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -0.9489930830950483 * kAngstromToBohr;
    (*coordinates)(2, 1) = 0.5872547387954865 * kAngstromToBohr;
    (*coordinates)(2, 2) = 0.0;
    (*coordinates)(3, 0) = 0.9489930830950483 * kAngstromToBohr;
    (*coordinates)(3, 1) = 0.5872547387954865 * kAngstromToBohr;
    (*coordinates)(3, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{
            {"C", 6, 0.0}, {"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// Runs the DEFAULT-path direct RHF (no initialScfState - the loop's own
// GWH start, exactly the M2 runner's start) at kNormal on the direct Fock
// path and returns the (converged, iterations, energy) triple.
struct DirectRhfRun {
    bool converged;
    int iterations;
    double totalEnergy;
};

qcx::Result<DirectRhfRun> RunDefaultRhf(const qcx::molecule::Molecule& molecule,
                                        const std::string& basisDir) {
    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / basisDir).string());

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto overlapTensor = qcx::integrals::BuildOverlapMatrix(molecule, *basis);

    if (!overlapTensor.has_value())
    {
        return std::unexpected(overlapTensor.error());
    }

    auto coreTensor = BuildCoreHamiltonian(molecule, *basis);

    if (!coreTensor.has_value())
    {
        return std::unexpected(coreTensor.error());
    }

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, *basis, *coreTensor, fockOptions);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    const Eigen::MatrixXd overlap = ToMatrix(*overlapTensor);
    const Eigen::MatrixXd core = ToMatrix(*coreTensor);

    auto scf =
        qcx::scf::RunRhfScf(molecule, overlap, core, {}, MakeDirectFockBuilder(*builder), &*basis);

    if (!scf.has_value())
    {
        return std::unexpected(scf.error());
    }

    return DirectRhfRun{scf->converged, scf->iterations, scf->totalEnergy};
}

// The 4 M2 2-cycle records as fixtures: psi4 references baked
// from the M2 gate record at the 2e-6 budget. The pre-fix failure mode is
// pinned by the record itself (parity-locked energies at 100-300
// iterations); the fixtures pin the post-fix contract - the GWH-primary
// 2-cycle is detected and the exactly-one restart from the cycle-members
// average (measured) converges in budget. References: the M2 gate
// record (m2_energies.json, puream true) at the 2e-6 budget.
TEST(RhfTwoCycleFallbackTest, Co6311GStarStarConvergesToThePsi4Pin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP();
    }

    auto molecule = MakeCo();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto run = RunDefaultRhf(*molecule, "6-311g-dstar");
    ASSERT_TRUE(run.has_value()) << run.error().message;

    EXPECT_TRUE(run->converged) << "CO/6-311G** RHF did not converge (GWH-primary 2-cycle)";
    EXPECT_NEAR(run->totalEnergy, -112.7680161861, 2e-6);
    EXPECT_LT(run->iterations, 200);
}

TEST(RhfTwoCycleFallbackTest, Hcn6311GStarStarConvergesToThePsi4Pin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP();
    }

    auto molecule = MakeHcn();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto run = RunDefaultRhf(*molecule, "6-311g-dstar");
    ASSERT_TRUE(run.has_value()) << run.error().message;

    EXPECT_TRUE(run->converged) << "HCN/6-311G** RHF did not converge (GWH-primary 2-cycle)";
    EXPECT_NEAR(run->totalEnergy, -92.89804562663593, 2e-6);
    EXPECT_LT(run->iterations, 200);
}

TEST(RhfTwoCycleFallbackTest, H2co6311GStarStarConvergesToThePsi4Pin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP();
    }

    auto molecule = MakeH2co();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto run = RunDefaultRhf(*molecule, "6-311g-dstar");
    ASSERT_TRUE(run.has_value()) << run.error().message;

    EXPECT_TRUE(run->converged) << "H2CO/6-311G** RHF did not converge (GWH-primary 2-cycle)";
    EXPECT_NEAR(run->totalEnergy, -113.54366422665336, 2e-6);
    EXPECT_LT(run->iterations, 200);
}

TEST(RhfTwoCycleFallbackTest, Hcn321GConvergesToThePsi4Pin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP();
    }

    auto molecule = MakeHcn();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto run = RunDefaultRhf(*molecule, "3-21g");
    ASSERT_TRUE(run.has_value()) << run.error().message;

    EXPECT_TRUE(run->converged) << "HCN/3-21G RHF did not converge (GWH-primary 2-cycle)";
    EXPECT_NEAR(run->totalEnergy, -92.35323159409685, 2e-6);
    EXPECT_LT(run->iterations, 200);
}

// The synthetic re-entry test (the C12H26 discriminator's test-side
// counterpart): a scripted alternating Fock map locks the trajectory into a
// parity-pinned 2-cycle deterministically, so the seam contract is pinned
// directly - the detector fires, exactly one restart follows, its seed (the
// cycle-members average) is the ONLY non-idempotent density the Fock builder
// ever receives, the cycle persists past the restart, and the run reports
// non-convergence. DIIS and the labeling stage are off: their behavior on a
// synthetic map is not the contract under test (DIIS could mix the fixed
// Fock pair into a convergent combination), and the restart orchestration is
// independent of both. No fast-only skip: milliseconds.
TEST(RhfTwoCycleFallbackTest, SyntheticCycleRestartsExactlyOnceWithTheAveragedSeed) {
    auto molecule = MakeHcn();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The RunDefaultRhf setup (3-21g), but the FockBuilderFn is scripted
    // instead of the direct-JK builder: FockA is the core Hamiltonian,
    // FockB is the core plus 0.5 on the (0,0) diagonal entry, alternating
    // per call. The map has no fixed point - the trajectory locks the
    // same 2-cycle from any seed (the measured C12H26 pattern, reproduced
    // synthetically).
    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / "3-21g").string());
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto overlapTensor = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlapTensor.has_value()) << overlapTensor.error().message;

    auto coreTensor = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;

    const Eigen::MatrixXd overlap = ToMatrix(*overlapTensor);
    const Eigen::MatrixXd core = ToMatrix(*coreTensor);

    int builderCalls = 0;
    int nonIdempotentInputs = 0;
    int firstNonIdempotentCall = -1;
    bool evenCall = true;

    // The probe: record the builder traffic and flag idempotency
    // violations of each INCOMING density (D S D == 2 D for a valid
    // closed-shell idempotent density; the averaged restart seed violates
    // it grossly). The FIRST violated call is kept: the restart is pinned
    // by WHERE its seed arrives, not by a total traffic count.
    qcx::scf::FockBuilderFn scriptedFock = [&](const Eigen::MatrixXd& density) {
        ++builderCalls;
        const double maxViolation =
            (density * overlap * density - 2.0 * density).cwiseAbs().maxCoeff();

        if (maxViolation > 1e-9 * static_cast<double>(density.rows()))
        {
            ++nonIdempotentInputs;

            if (firstNonIdempotentCall < 0)
            {
                firstNonIdempotentCall = builderCalls;
            }
        }

        Eigen::MatrixXd fock = core;

        if (evenCall)
        {
            fock(0, 0) += 0.5;
        }

        evenCall = !evenCall;
        return fock;
    };

    qcx::scf::RhfOptions options;
    options.useDiis = false;
    options.fullGroupLabeling = false;

    auto run = qcx::scf::RunRhfScf(*molecule, overlap, core, options, scriptedFock, &*basis);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    // The contract: the cycle never converges, the restart ran exactly
    // once (the primary's 100 iterations plus the restart's 100 - the
    // engine is structurally bound to one restart), and the ONLY
    // non-idempotent density the builder received is the restart seed.
    //
    // The restart is pinned by its SEED's position, a count-insensitive
    // form (the density-consistent rule): the density-consistent exit-path
    // recompute builds the Fock of the density the run RETURNS, one extra
    // trailing call per run, so this run's total is 201 builds where it
    // was 200 - a number that must move again for the next by-design
    // build, while the SEED's position (the primary's 100 iteration
    // builds, then the seed) does not. A mid-loop restart, a doubled
    // build rate inside an iteration or a seed from anywhere but the
    // budget's boundary still moves it and fails here.
    EXPECT_FALSE(run->converged) << "the scripted 2-cycle must not converge";
    EXPECT_EQ(run->iterations, 100) << "the no-rescue result reports the primary's budget";
    EXPECT_EQ(firstNonIdempotentCall, run->iterations + 1)
        << "the restart seed must follow the primary's full budget";
    EXPECT_EQ(nonIdempotentInputs, 1) << "the averaged seed is the only non-idempotent input";
    EXPECT_GE(builderCalls, 2 * run->iterations)
        << "the primary and its restart must both consume their iteration budgets";
}

} // namespace
