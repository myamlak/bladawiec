// The QfmmHfFockBuilder acceptance tests (the RIJCOSX-style composition
// point applied to QFMM):
// the composed builder F = H + 2J_QFMM(rho) - K(rho) (the Coulomb half
// through QfmmJBuilder, the exchange half through the exchange-only direct
// builder) must
//   1. wire a workspaceBudget through to BOTH nested Creates (nesting
//      order = reservation order - the QFMM half's estimate, outer store
//      plus its near-field builder, reserves first) and expose each
//      half's Create-time mode record through ModeInfo() /
//      ExchangeModeInfo() - the engine seam whose numbers the composed
//      runs' budget/mode records carry, never invented at the call site,
//   2. equal the fused direct builder (H + 2J - K in ONE pass) at the
//      theta -> 0 degenerate gate up to the fp-pairing scale - the split
//      passes accumulate per-element in a different order than the fused
//      pass's interleaved one, so the composed gate is tolerance-based,
//      NOT bit-exact like the J-only gate of qfmm_fock_build_test (a
//      composition bug - a dropped or mis-signed half, the double-H trap -
//      shows at chemical scale, 1e-2+),
//   3. stay inside the committed QFMM budgets {1e-5, 1e-7, 1e-8} Eh at
//      the per-preset winners on the C12 chain - compared against a
//      fused direct reference at the SAME preset, per the equal-preset
//      comparison protocol (a kNormal-screened
//      reference would carry its own ~3.3e-9 screening floor and spend the
//      kTight budget on the reference's error, not the QFMM's - the
//      qfmm_fixture BuildDirectFock is Coulomb-only at the kNormal
//      default, so this test builds its own reference), with the
//      FarFieldPairCount liveness flag asserted per rung (at the ladder's
//      pinned recorded model, kLoose C12 is far-alive - 14 far
//      pairs at (0.45, 1e-6); kNormal C12 is the VACUOUS tooth - all-near
//      at (0.3, 1e-8), so its budget row measures the split-pass fp
//      pairing. That vacuity belongs to that model, not to the rung: under
//      the current default the same cell records 517 far pairs. The rung's
//      real pin lives on the C24 chain at the driver level),
//   4. keep the C24 far-alive rungs alive at Create time (counts 73785 /
//      33404 far pairs at kLoose and kNormal under the current default -
//      the product-ball geometry with the centre-to-width test at each
//      rung's own angle, measured 2026-09-20, MSVC Release; the earlier
//      1997 / 499 are the retired midpoint-extent winners' counts.
//      The C24 BUDGET rows run at the driver level, threaded, where the
//      suite's serial-C12 / threaded-C24 division of the evidence lives).

#include "alkane_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_hf_build.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "tensor_conversions.hpp"

// The qfmm_fixture.hpp harness calls ToTensor / ToMatrix
// unqualified from inside qcx::integrals::test, while they live in
// qcx::testing - ADL cannot see them there (their arguments are Eigen /
// qcx::memory types), so they must be in scope BEFORE the fixture is
// included: lookup in the fixture's inline definitions happens at include
// time, not at use time.
namespace {

using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

} // namespace

#include "qfmm_fixture.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <utility>

namespace {

using qcx::integrals::test::BuildCoreHamiltonian;
using qcx::integrals::test::PhysicalDensity;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The fp64-only serial configuration of every comparison (the
// measurement configuration of the qfmm_fock_build_test: the serial pin
// makes every pass deterministic and reproducible).
qcx::integrals::QfmmOptions MeasurementOptions() {
    qcx::integrals::QfmmOptions options;
    options.useCertifiedMixedPrecision = false;
    options.maxParallelChunks = 1;
    return options;
}

// The composed builder under test: F = H + 2J_QFMM(rho) - K(rho) at the
// given preset (the winners resolve from the preset - theta 0, lMult -1).
qcx::Result<Eigen::MatrixXd> BuildComposedFock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::integrals::test::CpuTensor2& coreHamiltonian,
    const Eigen::MatrixXd& density,
    qcx::integrals::AccuracyPreset preset) {
    qcx::integrals::QfmmOptions options = MeasurementOptions();
    options.accuracy = preset;
    auto builder =
        qcx::integrals::QfmmHfFockBuilder::Create(molecule, basisSet, coreHamiltonian, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        return std::unexpected(fock.error());
    }

    return ToMatrix(*fock);
}

// The fused direct ground truth of the composed comparison: H + 2J - K in
// ONE DirectJkFockBuilder pass, built at the SAME preset as the composed
// side (the equal-preset comparison protocol). The
// qfmm_fixture's BuildDirectFock cannot serve: it is Coulomb-only AND
// kNormal-default, and against it the kTight composed build would measure
// the reference's own 1e-10 screening floor (~3.3e-9 on C12, measured
// 2026-09-13) - inside the re-derived 1e-8 kTight budget, but that number
// is the reference's error and not the QFMM's, so it cannot judge the
// composed build.
qcx::Result<Eigen::MatrixXd> BuildFusedDirectFock(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::integrals::test::CpuTensor2& coreHamiltonian,
    const Eigen::MatrixXd& density,
    qcx::integrals::AccuracyPreset preset) {
    qcx::integrals::FockBuildOptions options;
    options.accuracy = preset;
    options.useCertifiedMixedPrecision = false;
    options.maxParallelChunks = 1;
    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, coreHamiltonian, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        return std::unexpected(fock.error());
    }

    return ToMatrix(*fock);
}

// The fused direct UHF ground truth of the composed-UHF comparison:
// the driver's MakeDirectUhfFockBuilder assembly (run_driver.cpp) -
// a J-only coulomb pass on the half-summed density 0.5 (P_a + P_b) plus
// per-spin exchange-only passes on the raw spin densities, with one
// core-Hamiltonian copy subtracted back per channel (the double-H trap) -
// built here from the split-mode direct builders at the SAME preset (the
// equal-preset comparison protocol, like
// BuildFusedDirectFock).
qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> BuildFusedDirectUhfFocks(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::integrals::test::CpuTensor2& coreHamiltonian,
    const Eigen::MatrixXd& alphaDensity,
    const Eigen::MatrixXd& betaDensity,
    qcx::integrals::AccuracyPreset preset) {
    qcx::integrals::FockBuildOptions coulombOptions;
    coulombOptions.accuracy = preset;
    coulombOptions.buildCoulombOnly = true;
    coulombOptions.useCertifiedMixedPrecision = false;
    coulombOptions.maxParallelChunks = 1;
    auto coulomb = qcx::integrals::DirectJkFockBuilder::Create(
        molecule, basisSet, coreHamiltonian, coulombOptions);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    qcx::integrals::FockBuildOptions exchangeOptions;
    exchangeOptions.accuracy = preset;
    exchangeOptions.buildExchangeOnly = true;
    exchangeOptions.useCertifiedMixedPrecision = false;
    exchangeOptions.maxParallelChunks = 1;
    auto exchange = qcx::integrals::DirectJkFockBuilder::Create(
        molecule, basisSet, coreHamiltonian, exchangeOptions);

    if (!exchange.has_value())
    {
        return std::unexpected(exchange.error());
    }

    auto totalTensor = ToTensor(0.5 * (alphaDensity + betaDensity));

    if (!totalTensor.has_value())
    {
        return std::unexpected(totalTensor.error());
    }

    auto coulombFock = coulomb->BuildFock(*totalTensor);

    if (!coulombFock.has_value())
    {
        return std::unexpected(coulombFock.error());
    }

    auto alphaTensor = ToTensor(alphaDensity);

    if (!alphaTensor.has_value())
    {
        return std::unexpected(alphaTensor.error());
    }

    auto exchangeAlpha = exchange->BuildFock(*alphaTensor);

    if (!exchangeAlpha.has_value())
    {
        return std::unexpected(exchangeAlpha.error());
    }

    auto betaTensor = ToTensor(betaDensity);

    if (!betaTensor.has_value())
    {
        return std::unexpected(betaTensor.error());
    }

    auto exchangeBeta = exchange->BuildFock(*betaTensor);

    if (!exchangeBeta.has_value())
    {
        return std::unexpected(exchangeBeta.error());
    }

    const Eigen::MatrixXd core = ToMatrix(coreHamiltonian);
    return std::pair<Eigen::MatrixXd, Eigen::MatrixXd>{
        ToMatrix(*coulombFock) + ToMatrix(*exchangeAlpha) - core,
        ToMatrix(*coulombFock) + ToMatrix(*exchangeBeta) - core};
}

// The largest absolute element difference.
double MaxAbsoluteDifference(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    double maximum = 0.0;

    for (Eigen::Index i = 0; i < a.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < a.cols(); ++j)
        {
            maximum = std::max(maximum, std::abs(a(i, j) - b(i, j)));
        }
    }

    return maximum;
}

// The energy error: for the full Fock both sides carry the core Hamiltonian
// on the diagonal, so 0.5 * |Tr(D * (F_composed - F_fused))| is the
// budget metric (Eh) of the composed approximation (the preset's budget shape).
double EnergyError(const Eigen::MatrixXd& composed,
                   // (composed, fused) is the compared pair, density the fixed weight.
                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                   const Eigen::MatrixXd& fused,
                   const Eigen::MatrixXd& density) {
    const Eigen::MatrixXd delta = composed - fused;
    return 0.5 * std::abs((density.cwiseProduct(delta)).sum());
}

// This row's acceptance bounds. They MIRROR the committed QFMM ladder
// (benchmarks/data/qfmm_ladder_sweep_full.csv) rather than reading it,
// and deliberately do NOT follow the 2026-09-13 kTight re-derivation
// 1e-9 -> 1e-8: the assertion below is an acceptance row, and pinning it
// TIGHTER than the contract it names is the safe direction (measured
// 9.83143e-13 on C12 2026-09-13, ~1000x inside). The kTight entry is
// therefore 10x stricter than qcx::integrals::QfmmBudgetForPreset(kTight)
// by design, not by drift - do not "sync" it without re-measuring, and do
// not read it as the ladder's value.
double BudgetForPreset(qcx::integrals::AccuracyPreset preset) {
    switch (preset)
    {
    case qcx::integrals::AccuracyPreset::kLoose:
        return 1e-5;
    case qcx::integrals::AccuracyPreset::kNormal:
        return 1e-7;
    case qcx::integrals::AccuracyPreset::kTight:
        return 1e-9;
    }

    return 1e-7;
}

TEST(QfmmHfBuildTest, CreateWiresTheWorkspaceBudgetAcrossBothHalves) {
    // The composed path's adaptive-memory wiring (the composed-UHF
    // budget/mode records gap fix, interrogation 6): a workspaceBudget is
    // shared with both nested Creates - the QFMM half's Create-time
    // estimate (outer store plus its near-field builder) reserves first,
    // the exchange half's own estimate second - and each half's decision
    // is exposed through ModeInfo() / ExchangeModeInfo(). The budget is
    // never silently dropped onto the legacy path.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // The capacity must clear BOTH halves at their natural 512 MiB batch
    // caps - the near-field's per-thread arena (T x maxBatchBytes, the
    // arena-dominated end) plus the exchange half's arena and its
    // slot-folded exchange mass (<= 4 x T x maxBatchBytes at the full
    // team) - so the cap is derived from the process team size. A fixed
    // 1 GiB cap would NOT be a wiring failure: the QFMM half's budget
    // clamp sizes its near-field arena to fill the entire remaining
    // budget (the standalone clamp semantics - the estimate is
    // arena-dominated at this scale), the exchange half then finds ~0
    // bytes and refuses honestly (nothing invented; the refusal text
    // reports the real remainder). Coordinating a split band across the
    // two halves at arena-dominated caps is engine-model work beyond what
    // this test pins; it runs in the natural-cap regime the
    // driver-level composed runs live in.
    const std::size_t teamSize = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    constexpr std::size_t kDefaultBatchBytes = std::size_t{512} * 1024 * 1024;
    const std::size_t capacityBytes =
        6 * teamSize * kDefaultBatchBytes + std::size_t{4} * 1024 * 1024 * 1024;
    auto budget = qcx::memory::WorkspaceBudget::Create(capacityBytes);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;

    qcx::integrals::QfmmOptions options;
    options.workspaceBudget = &*budget;
    auto builder = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const auto& coulombMode = builder->ModeInfo();
    ASSERT_TRUE(coulombMode.has_value());
    EXPECT_EQ(coulombMode->mode, qcx::integrals::FockBuildMode::kFastPath);
    EXPECT_EQ(coulombMode->budgetBytes, capacityBytes);
    EXPECT_GT(coulombMode->outerStoreBytes, 0u);
    EXPECT_GT(coulombMode->reservedBytes, 0u);

    const auto& exchangeMode = builder->ExchangeModeInfo();
    ASSERT_TRUE(exchangeMode.has_value());
    EXPECT_EQ(exchangeMode->mode, qcx::integrals::FockBuildMode::kFastPath);
    EXPECT_GT(exchangeMode->reservedBytes, 0u);

    // Both halves charged the one shared counter; the QFMM half reserved
    // first (nesting order = reservation order).
    EXPECT_GT(budget->CommittedBytes(), 0u);
    EXPECT_LE(budget->CommittedBytes(), capacityBytes);
}

TEST(QfmmHfBuildTest, DegenerateGateCompositionEqualsTheFusedDirectFock) {
    // The theta -> 0 gate on the composed builder: everything is near field,
    // the far field contributes nothing, and the composed Fock must equal
    // the fused direct Fock at the same preset up to the fp pairing of the
    // split passes (NOT bit-exact - the two halves accumulate per-element
    // in their own order before the double-H subtraction, while the fused
    // pass interleaves 2J and -K - the tolerance below is the fp-pairing
    // scale on this 7-function fixture, and a composition bug would land
    // at chemical scale instead).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);

    qcx::integrals::QfmmOptions options = MeasurementOptions();
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    options.theta = -1.0;
    auto builder = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;
    EXPECT_EQ(builder->FarFieldPairCount(), 0u);

    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto composedTensor = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(composedTensor.has_value()) << composedTensor.error().message;
    const Eigen::MatrixXd composed = ToMatrix(*composedTensor);

    auto fused = BuildFusedDirectFock(
        *molecule, *basis, *core, density, qcx::integrals::AccuracyPreset::kTight);
    ASSERT_TRUE(fused.has_value()) << fused.error().message;

    const double maxDifference = MaxAbsoluteDifference(composed, *fused);
    const double energy = EnergyError(composed, *fused, density);
    std::cout << "H2O degenerate gate: max abs diff " << maxDifference << ", energy error "
              << energy << "\n";

    EXPECT_LT(maxDifference, 1e-12);
    EXPECT_LT(energy, 1e-13);
}

TEST(QfmmHfBuildTest, PresetLadderStaysInsideTheCommittedBudgetsAtEqualPreset) {
    // The preset ladder on the C12 chain, each rung at its recorded winner, each
    // measured against a fused direct reference at the SAME preset. The
    // kLoose rung is far-ALIVE (14 far pairs at the (0.45,
    // tau 1e-6) winner - the multipole truncation is exercised and lands
    // far inside the 1e-5 budget at the equal-preset protocol). The kNormal
    // rung is the VACUOUS C12 tooth AT THIS PINNED MODEL (all-near at (0.3,
    // tau 1e-8) - zero far pairs, count printed per rung below; the row
    // measures the split-pass fp pairing, and the kNormal rung's real pin
    // lives on C24 at the driver level). The current default is not this
    // model: at the same cell it measures 517 far pairs, so the vacuity here
    // is the pinned record's, not the rung's. The kTight rung is the theta ->
    // 0 degenerate gate (near-field-only by design).
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(86);

    using qcx::integrals::AccuracyPreset;
    const AccuracyPreset presets[] = {
        AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight};

    for (const AccuracyPreset preset : presets)
    {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = preset;
        // The preset ladder is a record of the RECORDED model (midpoint
        // extents + the width test); pin it so the record keeps measuring what
        // it recorded. Since the 2026-09-15 flip the DEFAULT is the corrected
        // pair; its ladder accuracy is a re-calibration job (the corrections'
        // error at kNormal/C12 measures 1.4e-9 relative Frobenius and 2.1e-8
        // Eh J-energy, inside the kNormal budget, and 5.3e-3 in the kTight
        // cell only if the kTight rung were NOT the gate - which it still is).
        options.extentModel = qcx::integrals::QfmmExtentMode::kMidpointBound;
        options.separationMode = qcx::integrals::QfmmSeparationMode::kWidthTheta;
        options.separationK = 0.0;
        auto builder = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(builder.has_value()) << builder.error().message;

        auto densityTensor = ToTensor(density);
        ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
        auto composedTensor = builder->BuildFock(*densityTensor);
        ASSERT_TRUE(composedTensor.has_value()) << composedTensor.error().message;
        const Eigen::MatrixXd composed = ToMatrix(*composedTensor);

        auto fused = BuildFusedDirectFock(*molecule, *basis, *core, density, preset);
        ASSERT_TRUE(fused.has_value()) << fused.error().message;

        const double energy = EnergyError(composed, *fused, density);
        std::cout << "C12 preset " << static_cast<int>(preset) << ": far pairs "
                  << builder->FarFieldPairCount() << ", energy error " << energy << "\n";

        EXPECT_LT(energy, BudgetForPreset(preset)) << "preset " << static_cast<int>(preset);
    }

    // The liveness flags: the kLoose rung must be far-ALIVE (a vacuous far
    // field would hide the multipole truncation - the fixture gap noted
    // when the far field was first wired), the
    // kNormal rung is far-ALIVE too since the 2026-09-15 flip (the recorded
    // vacuous-by-design count was 0 at kNormal on this fixture; the current
    // default - the product ball with the centre-to-width test at the rung's
    // own 0.3 - measures 517 far pairs here (measured 2026-09-20, MSVC
    // Release)), and the kTight rung stays degenerate
    // BY DESIGN: its own rung is the theta -> 0 gate, and the ruling
    // keeps a preset whose rung IS the gate exact rather than handing it a
    // live far field at L = 0.
    {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = AccuracyPreset::kLoose;
        auto loose = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(loose.has_value()) << loose.error().message;
        EXPECT_GT(loose->FarFieldPairCount(), 0u);
    }

    {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = AccuracyPreset::kNormal;
        auto normal = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(normal.has_value()) << normal.error().message;
        EXPECT_GT(normal->FarFieldPairCount(), 0u);
    }

    {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = AccuracyPreset::kTight;
        auto tight = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(tight.has_value()) << tight.error().message;
        EXPECT_EQ(tight->FarFieldPairCount(), 0u);
    }
}

TEST(QfmmHfBuildTest, UhfSpinChannelsEqualTheRhfCompositionAtEqualSpinDensities) {
    // The closed-shell consistency gate: with P_alpha = P_beta = P the
    // per-spin UHF assembly must collapse onto the RHF composition - the
    // shared Coulomb call sees 0.5 (P + P) = P and each exchange call sees
    // P, exactly the BuildFock(rho = P) inputs, so both spin channels are
    // bit-identical to the RHF result (same split calls, same arithmetic).
    // A wrong halving anywhere (a missing 0.5 on the total, a halved spin
    // density) would break the equality at chemical scale instead.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);

    qcx::integrals::QfmmOptions options = MeasurementOptions();
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    options.theta = -1.0;
    auto builder = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto alphaTensor = ToTensor(density);
    ASSERT_TRUE(alphaTensor.has_value()) << alphaTensor.error().message;
    auto betaTensor = ToTensor(density);
    ASSERT_TRUE(betaTensor.has_value()) << betaTensor.error().message;
    auto uhfTensor = builder->BuildUhfFock(*alphaTensor, *betaTensor);
    ASSERT_TRUE(uhfTensor.has_value()) << uhfTensor.error().message;
    const Eigen::MatrixXd fockAlpha = ToMatrix(uhfTensor->first);
    const Eigen::MatrixXd fockBeta = ToMatrix(uhfTensor->second);

    auto rhfTensor = builder->BuildFock(*alphaTensor);
    ASSERT_TRUE(rhfTensor.has_value()) << rhfTensor.error().message;
    const Eigen::MatrixXd rhf = ToMatrix(*rhfTensor);

    std::cout << "UHF equal-spin gate: max abs diff alpha-vs-beta "
              << MaxAbsoluteDifference(fockAlpha, fockBeta) << ", alpha-vs-rhf "
              << MaxAbsoluteDifference(fockAlpha, rhf) << "\n";

    EXPECT_LT(MaxAbsoluteDifference(fockAlpha, fockBeta), 1e-12);
    EXPECT_LT(MaxAbsoluteDifference(fockAlpha, rhf), 1e-12);
}

TEST(QfmmHfBuildTest, UhfPolarizedFocksMatchTheFusedDirectAssembly) {
    // The composition gate on a SPIN-POLARIZED input (P_alpha !=
    // P_beta): each composed channel must equal the driver's
    // MakeDirectUhfFockBuilder assembly (J-only direct coulomb pass on the
    // half-summed density plus per-spin exchange-only passes, one H copy
    // subtracted back per channel) at the theta -> 0 degenerate gate, up
    // to the fp-pairing scale of the split passes (a dropped or mis-signed
    // spin channel - the double-H trap per channel - would land at chemical
    // scale instead). The alpha/beta distinction rides the exchange half:
    // unequal spin densities must produce unequal Fock channels, K_a from
    // P_alpha only.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd alphaDensity = PhysicalDensity(7);
    const Eigen::MatrixXd betaDensity = 0.5 * PhysicalDensity(7);

    qcx::integrals::QfmmOptions options = MeasurementOptions();
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    options.theta = -1.0;
    auto builder = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto alphaTensor = ToTensor(alphaDensity);
    ASSERT_TRUE(alphaTensor.has_value()) << alphaTensor.error().message;
    auto betaTensor = ToTensor(betaDensity);
    ASSERT_TRUE(betaTensor.has_value()) << betaTensor.error().message;
    auto uhfTensor = builder->BuildUhfFock(*alphaTensor, *betaTensor);
    ASSERT_TRUE(uhfTensor.has_value()) << uhfTensor.error().message;
    const Eigen::MatrixXd fockAlpha = ToMatrix(uhfTensor->first);
    const Eigen::MatrixXd fockBeta = ToMatrix(uhfTensor->second);

    auto fused = BuildFusedDirectUhfFocks(*molecule,
                                          *basis,
                                          *core,
                                          alphaDensity,
                                          betaDensity,
                                          qcx::integrals::AccuracyPreset::kTight);
    ASSERT_TRUE(fused.has_value()) << fused.error().message;

    const double alphaDifference = MaxAbsoluteDifference(fockAlpha, fused->first);
    const double betaDifference = MaxAbsoluteDifference(fockBeta, fused->second);
    const double channelSplit = MaxAbsoluteDifference(fockAlpha, fockBeta);
    std::cout << "UHF polarized gate: max abs diff alpha " << alphaDifference << ", beta "
              << betaDifference << ", alpha-vs-beta " << channelSplit << "\n";

    EXPECT_LT(alphaDifference, 1e-12);
    EXPECT_LT(betaDifference, 1e-12);
    EXPECT_GT(channelSplit, 1e-3);
}

TEST(QfmmHfBuildTest, C24KeepsItsFarAliveRungsAliveAtCreateTime) {
    // The C24 chain carries the kNormal far-field pin (the C12 tooth's
    // vacuity at (0.3, 1e-8) is the pinned recorded model's - and
    // this fixture is where the driver's threaded budget rows read it) and
    // the strongest kLoose exercise. At CREATE time the liveness flag must
    // report the current default's far-alive rungs (73785 far pairs at
    // kLoose, 33404 at kNormal, measured 2026-09-20, MSVC Release; the earlier
    // 1997 / 499 are the retired midpoint-extent winners' counts)
    // and the degenerate kTight rung must be empty -
    // the C24 BUDGET rows themselves run at the driver level, where the
    // threaded Fock builds keep them inside the suite's runtime envelope.
    auto molecule = MakeAlkaneSto3g(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    using qcx::integrals::AccuracyPreset;
    const AccuracyPreset presets[] = {
        AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight};

    for (const AccuracyPreset preset : presets)
    {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = preset;
        auto builder = qcx::integrals::QfmmHfFockBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(builder.has_value()) << builder.error().message;
        std::cout << "C24 preset " << static_cast<int>(preset) << ": far pairs "
                  << builder->FarFieldPairCount() << "\n";

        // kLoose and kNormal are far-ALIVE at the current default (73785 /
        // 33404 far pairs); kTight is the degenerate theta -> 0 gate (empty).
        if (preset == AccuracyPreset::kTight)
        {
            EXPECT_EQ(builder->FarFieldPairCount(), 0u);
        } else
        {
            EXPECT_GT(builder->FarFieldPairCount(), 0u);
        }
    }
}

} // namespace
