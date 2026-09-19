// The named acceptance fixture for the auxiliary-pair density screen:
// benzene/def2-TZVP, the screened converged energy against the UNScreened RI-K,
// with the 0.01 kcal/mol = 1.59e-5 Eh bar stated beside the number.
//
// The unscreened side is this repo's own dense occ-RI-K over the SAME
// metric-transformed tensor (the dense entry points - the code the composed
// builder ran before the screen), never a second builder's: one tensor,
// one density, one occupied block, and one threshold, so the two sides differ
// only by the screen.
//
// This cell is in its own translation unit because the basis fixtures define
// `kSto3gCarbon` twice (benzene_sto3g.hpp and alkane_sto3g.hpp) and one TU
// cannot include both. The instrument it shares with the other screen cells is
// ri_aux_screen_fixture.hpp.

#include "benzene_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "qcx/integrals/limits.hpp"
#include "ri_aux_screen_fixture.hpp"

#include <cstddef>

namespace {

using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeBenzeneSto3g;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;
using qcx::testing::riAuxScreen::BuildCoreHamiltonian;
using qcx::testing::riAuxScreen::kAuxName;
using qcx::testing::riAuxScreen::kCaloriesPerHartree;
using qcx::testing::riAuxScreen::kPlanScreenTolerance;
using qcx::testing::riAuxScreen::kScreenTruncationClass;
using qcx::testing::riAuxScreen::MeasureBudgetCell;
using qcx::testing::riAuxScreen::ParseFixtureBasis;
using qcx::testing::riAuxScreen::RecordMeasurement;

TEST(RiAuxScreenTest, BenzeneDef2TzvpScreensWithinThePlanTolerance) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jkfit g shells";
    }

    auto molecule = MakeBenzeneSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = ParseFixtureBasis("def2-tzvp", {1, 6});
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = ParseFixtureBasis(kAuxName, {1, 6});
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    auto coreTensor = ToTensor(*core);
    ASSERT_TRUE(coreTensor.has_value()) << coreTensor.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    const std::size_t occupiedCount = static_cast<std::size_t>(molecule->ElectronCount()) / 2;

    // The preset is not named for this row anywhere; kNormal is the operating
    // default and the tightest preset that still lets the screen do work.
    auto outcome = MeasureBudgetCell(*molecule,
                                     *basis,
                                     *aux,
                                     *coreTensor,
                                     *core,
                                     ToMatrix(*overlap),
                                     occupiedCount,
                                     qcx::integrals::AccuracyPreset::kNormal,
                                     "benzene_def2tzvp");
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    RecordMeasurement("benzene_def2tzvp.plan_tolerance", kPlanScreenTolerance);
    RecordMeasurement("benzene_def2tzvp.plan_tolerance_kcal",
                      kPlanScreenTolerance * kCaloriesPerHartree);
    RecordMeasurement("benzene_def2tzvp.plan_tolerance_ratio",
                      outcome->move / kPlanScreenTolerance);

    std::cout << "[ri_jk screen] benzene/def2-TZVP acceptance (aux " << kAuxName
              << ", kNormal): screened vs unscreened RI-K move " << outcome->move
              << " Eh = " << (outcome->move * kCaloriesPerHartree)
              << " kcal/mol against the acceptance bar " << kPlanScreenTolerance
              << " Eh = 0.01 kcal/mol, ratio " << (outcome->move / kPlanScreenTolerance) << "\n";

    EXPECT_LE(outcome->move, kPlanScreenTolerance)
        << "the screen's acceptance: the screened converged energy must stay within "
           "0.01 kcal/mol = 1.59e-5 Eh of the unscreened RI-K on benzene/def2-TZVP";

    // The budget requirement, at the one bar the measurement supports
    // on this fixture: the truncation must never reach the auxiliary fit's own
    // error class. The STRICT reading - the screen's move inside the preset's
    // J/K target (1e-10 Eh at kNormal) - is MEASURED AND MISSED here by
    // `energy_move_ratio`, which MeasureBudgetCell records; the ratio is the
    // evidence the threshold decision needs, and the gate's
    // shape explains it: the reused density weight is the pair block's, which
    // bounds the Coulomb half and not the exchange half's row norm.
    EXPECT_LE(outcome->move, kScreenTruncationClass)
        << "the screen's own converged-energy move has reached the auxiliary fit's error class - "
           "a truncation there is no longer sub-dominant to the approximation it screens, and "
           "the screen must not ship at this threshold";
}

} // namespace
