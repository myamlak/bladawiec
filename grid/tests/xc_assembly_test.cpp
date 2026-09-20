// The XC engine's energy and potential must be consistent with each other and
// with the closed-form relations the functionals obey. Two independent
// channels pin the assembly:
//   - finite difference: E(D + t P) - E(D - t P) over 2t must equal
//     Tr(P V(D)) for every path (the potential is the energy's derivative);
//   - the LDA exchange Euler identity: pure Slater exchange is homogeneous of
//     degree 4/3 in each spin density, so sum_s Tr(D_s V_s) = (4/3) E_x holds
//     pointwise and the quadrature cannot break it.
// The per-point kernel values themselves are excgrid's own suite's business.

#include "h2_sto3g.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/xc_grid_engine.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

// The H2/STO-3G closed-shell density: two electrons in the bonding
// combination c = (1, 1) of the two hydrogen 1s AOs, so D = 2 c c^T and
// rho = (phi_1 + phi_2)^2 >= 0 on the whole grid. The combination is not
// S-normalized, so this density carries 2 + 2S_12 = 3.3186 electrons rather
// than 2 - irrelevant for the energy/potential identities below, which hold
// for any fixed D.
Eigen::MatrixXd H2ClosedShellDensity() {
    Eigen::MatrixXd density(2, 2);
    density << 1.0, 1.0, 1.0, 1.0;

    return density;
}

// One hydrogen at the origin: a single AO, so int rho dr = D_00 * int phi^2
// = D_00 by the basis module's unit-norm convention.
qcx::Result<qcx::molecule::Molecule> MakeSingleHydrogen() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}}, std::move(*coordinates), 0, 1);
}

// A symmetric perturbation for the finite-difference channel, small enough
// that D +- t P stays a valid density.
Eigen::MatrixXd SymmetricPerturbation() {
    Eigen::MatrixXd perturbation(2, 2);
    perturbation << 0.1, -0.05, -0.05, 0.1;

    return perturbation;
}

// An unrestricted spin-density pair: alpha and beta differ (both matrices are
// positive semidefinite, so rho stays non-negative on the grid), which puts
// the sigma_ab cross terms and two different spin gradients on the test path.
Eigen::MatrixXd H2PolarizedAlpha() {
    Eigen::MatrixXd density(2, 2);
    density << 0.8, 0.7, 0.7, 0.8;

    return density;
}

Eigen::MatrixXd H2PolarizedBeta() {
    Eigen::MatrixXd density(2, 2);
    density << 0.4, 0.3, 0.3, 0.4;

    return density;
}

// The finite-difference pin. Measured central-difference residuals at step 1e-4
// are ~1e-10 on potentials of order 1e-1, which is the roundoff floor of the
// energy difference rather than truncation (the scaling test below shows the
// truncation term is negligible there), so the tolerance carries roughly
// twentyfold headroom over the floor.
constexpr double kFiniteDifferenceTolerance = 2e-9;

double TraceProduct(const Eigen::MatrixXd& left, const Eigen::MatrixXd& right) {
    return (left.array() * right.array()).sum();
}

// Creates the engine for H2/STO-3G with the named functional.
qcx::Result<XcGridEngine> MakeEngine(std::string_view functionalName) {
    const auto molecule = qcx::testing::MakeH2Sto3g();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    const auto basis = qcx::testing::MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    return XcGridEngine::Create(*molecule, *basis, functionalName);
}

// The finite-difference check for the closed-shell path: a central difference
// of the energy in the direction P against the potential the engine returns
// at the unperturbed density. Both spins move (D_alpha = D_beta = D/2), so
// the analytic term carries the half weight and both potentials.
void ExpectPotentialMatchesEnergyDerivative(const XcGridEngine& engine, double step) {
    const Eigen::MatrixXd density = H2ClosedShellDensity();
    const Eigen::MatrixXd perturbation = SymmetricPerturbation();

    const auto center = engine.EvaluateClosedShell(density);
    const auto plus = engine.EvaluateClosedShell(density + step * perturbation);
    const auto minus = engine.EvaluateClosedShell(density - step * perturbation);
    ASSERT_TRUE(center.has_value());
    ASSERT_TRUE(plus.has_value());
    ASSERT_TRUE(minus.has_value());

    const double numeric = (plus->energy - minus->energy) / (2.0 * step);
    const double analytic =
        0.5 * TraceProduct(perturbation, center->potentialAlpha + center->potentialBeta);

    EXPECT_NEAR(numeric, analytic, kFiniteDifferenceTolerance)
        << "finite difference " << numeric << " vs potential " << analytic;
}

// The unrestricted finite-difference check: each spin density is perturbed on
// its own, so the analytic term is that spin's potential alone (no half
// weight). At this step the residual is the roundoff floor of the energy
// difference, not truncation - the energy is a sum over tens of thousands of
// grid points, so its own error (~sqrt(N) * eps * |E|, of order 1e-13) divided
// by 2 * step lands at ~1e-10, and shrinking the step makes it worse rather
// than sharper. The quadratic behaviour of the truncation term is pinned
// separately, at a step coarse enough for truncation to dominate.
void ExpectUnrestrictedPotentialMatchesEnergyDerivative(const XcGridEngine& engine,
                                                        double step,
                                                        const Eigen::MatrixXd& densityAlpha,
                                                        const Eigen::MatrixXd& densityBeta) {
    const Eigen::MatrixXd perturbation = SymmetricPerturbation();

    const auto center = engine.Evaluate(densityAlpha, densityBeta);
    const auto alphaPlus = engine.Evaluate(densityAlpha + step * perturbation, densityBeta);
    const auto alphaMinus = engine.Evaluate(densityAlpha - step * perturbation, densityBeta);
    const auto betaPlus = engine.Evaluate(densityAlpha, densityBeta + step * perturbation);
    const auto betaMinus = engine.Evaluate(densityAlpha, densityBeta - step * perturbation);
    ASSERT_TRUE(center.has_value());
    ASSERT_TRUE(alphaPlus.has_value());
    ASSERT_TRUE(alphaMinus.has_value());
    ASSERT_TRUE(betaPlus.has_value());
    ASSERT_TRUE(betaMinus.has_value());

    const double numericAlpha = (alphaPlus->energy - alphaMinus->energy) / (2.0 * step);
    const double analyticAlpha = TraceProduct(perturbation, center->potentialAlpha);
    EXPECT_NEAR(numericAlpha, analyticAlpha, kFiniteDifferenceTolerance)
        << "alpha spin: finite difference " << numericAlpha << " vs potential " << analyticAlpha;

    const double numericBeta = (betaPlus->energy - betaMinus->energy) / (2.0 * step);
    const double analyticBeta = TraceProduct(perturbation, center->potentialBeta);
    EXPECT_NEAR(numericBeta, analyticBeta, kFiniteDifferenceTolerance)
        << "beta spin: finite difference " << numericBeta << " vs potential " << analyticBeta;
}

} // namespace

TEST(XcGridEngineTest, GridBlocksAreWellFormed) {
    const auto engine = MakeEngine("slater");
    ASSERT_TRUE(engine.has_value());
    EXPECT_EQ(engine->AOCount(), 2u);
    EXPECT_GT(engine->BlockCount(), 0u);
    EXPECT_GT(engine->PointCount(), 0u);

    std::size_t countedPoints = 0;

    for (const excgrid::Block& block : engine->Grid().Blocks())
    {
        EXPECT_EQ(block.points.size(), block.pointCount);
        EXPECT_EQ(block.weights.size(), block.pointCount);
        EXPECT_EQ(block.atomIndex.size(), block.pointCount);
        countedPoints += block.pointCount;
    }

    EXPECT_EQ(countedPoints, engine->PointCount());
}

TEST(XcGridEngineTest, GridWeightsIntegrateAnAtomicDensity) {
    // One hydrogen, one AO, D = [[2]]: int rho dr = D_00 * int phi^2 = 2, by
    // the basis module's unit-norm convention - an exact target that needs no
    // overlap matrix (the grid module cannot reach the integrals module). What
    // this pins is the block weights against the AO normalization over the
    // whole point set; the contraction itself is written inline here.
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value());
    const auto evaluator = AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value());
    const auto engine = XcGridEngine::Create(*molecule, *basis, "slater");
    ASSERT_TRUE(engine.has_value());
    ASSERT_EQ(engine->AOCount(), 1u);

    std::vector<double> values(1);
    double electrons = 0.0;

    for (const excgrid::Block& block : engine->Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            evaluator->Evaluate(block.points[point], values);
            electrons += block.weights[point] * 2.0 * values[0] * values[0];
        }
    }

    // Measured deviation 2.0e-12: the Lebedev/MHL quadrature's accuracy floor
    // for a 1s density, not an assembly error.
    EXPECT_NEAR(electrons, 2.0, 1e-10) << "grid-integrated electron count " << electrons;
}

TEST(XcGridEngineTest, SlaterExchangeSatisfiesTheEulerIdentity) {
    const auto engine = MakeEngine("slater");
    ASSERT_TRUE(engine.has_value());

    const Eigen::MatrixXd density = H2ClosedShellDensity();
    const auto evaluation = engine->EvaluateClosedShell(density);
    ASSERT_TRUE(evaluation.has_value());
    ASSERT_LT(evaluation->energy, 0.0);

    const double potentialEnergy =
        0.5 * TraceProduct(density, evaluation->potentialAlpha + evaluation->potentialBeta);

    // Measured deviation 1.1e-14 on -1.53 Hartree: the identity holds pointwise,
    // so only roundoff separates the two sides.
    EXPECT_NEAR(potentialEnergy, (4.0 / 3.0) * evaluation->energy, 1e-12)
        << "Tr(D V) " << potentialEnergy << " vs (4/3) E " << (4.0 / 3.0) * evaluation->energy;
}

TEST(XcGridEngineTest, SlaterPotentialIsTheEnergyDerivative) {
    const auto engine = MakeEngine("slater");
    ASSERT_TRUE(engine.has_value());
    EXPECT_FALSE(engine->UsesGradient());

    ExpectPotentialMatchesEnergyDerivative(*engine, 1e-4);
}

TEST(XcGridEngineTest, PbePotentialIsTheEnergyDerivative) {
    // The GGA path: this fails if the sigma terms are missing from the
    // potential, since the energy's gradient dependence is still perturbed.
    const auto engine = MakeEngine("pbe");
    ASSERT_TRUE(engine.has_value());
    EXPECT_TRUE(engine->UsesGradient());

    ExpectPotentialMatchesEnergyDerivative(*engine, 1e-4);
}

TEST(XcGridEngineTest, UnrestrictedPotentialIsTheEnergyDerivative) {
    // Spin-polarized: alpha and beta differ, so sigma_ab and both spins'
    // sigma-weighted cross terms are on the path. "b3lyp" is the load-bearing
    // name here: it carries the LYP correlation term, whose vsigmaAb is
    // nonzero, whereas "pbe" denotes PBE exchange alone and returns a zero
    // vsigmaAb, so the LDA and exchange-only legs would pass with that term
    // dropped.
    for (const std::string_view name : {"slater", "pbe", "b3lyp"})
    {
        const auto engine = MakeEngine(name);
        ASSERT_TRUE(engine.has_value()) << name;

        ExpectUnrestrictedPotentialMatchesEnergyDerivative(
            *engine, 1e-4, H2PolarizedAlpha(), H2PolarizedBeta());
    }
}

TEST(XcGridEngineTest, NonSymmetricDensityUsesTheFullContraction) {
    // The contraction is documented as exact for a non-symmetric D, because
    // the double sum is written out rather than factored through a symmetric
    // operator. The symmetric part stays positive semidefinite so that rho
    // stays non-negative on the grid; the antisymmetric part leaves rho
    // untouched but does move the gradient.
    const auto engine = MakeEngine("pbe");
    ASSERT_TRUE(engine.has_value());

    Eigen::MatrixXd nonSymmetricAlpha(2, 2);
    nonSymmetricAlpha << 0.8, 0.75, 0.65, 0.8;

    ExpectUnrestrictedPotentialMatchesEnergyDerivative(
        *engine, 1e-4, nonSymmetricAlpha, H2PolarizedBeta());
}

TEST(XcGridEngineTest, SettingsReachTheGridBuild) {
    // The settings mirror the grid build parameters; a dropped assignment would
    // pass every other test in this file, so each observable field is moved
    // here. Pinned: the radial and angular counts and the trim threshold, all
    // of which move the surviving point count. NOT pinned: the block target
    // (excgrid's spatial re-batching caps blocks well below any target tried -
    // a 64-point target still yields 369 blocks against the default grid's
    // 1216, so the block count does not respond to it), the mapping scale, and
    // the mapping exponent.
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value());

    const auto defaults = XcGridEngine::Create(*molecule, *basis, "slater");
    ASSERT_TRUE(defaults.has_value());

    XcGridSettings coarse;
    coarse.radialPoints = 20;
    coarse.angularPoints = 74;
    const auto small = XcGridEngine::Create(*molecule, *basis, "slater", coarse);
    ASSERT_TRUE(small.has_value());
    EXPECT_LT(small->PointCount(), defaults->PointCount());
    EXPECT_EQ(small->AOCount(), defaults->AOCount());

    XcGridSettings trimmed;
    trimmed.trimWeight = 1e-3;
    const auto trimmedEngine = XcGridEngine::Create(*molecule, *basis, "slater", trimmed);
    ASSERT_TRUE(trimmedEngine.has_value());
    EXPECT_LT(trimmedEngine->PointCount(), defaults->PointCount());
}

TEST(XcGridEngineTest, FiniteDifferenceResidualGrowsWithTheStepSquared) {
    // The consistency checks above run at a step small enough that their
    // residual is the roundoff floor of the energy difference. This one runs at
    // a coarse step, where the central difference's truncation term dominates
    // instead, and there the residual must grow roughly fourfold when the step
    // doubles. That is what identifies the residual as truncation rather than an
    // assembly error, which would not move with the step at all.
    const auto engine = MakeEngine("slater");
    ASSERT_TRUE(engine.has_value());

    const Eigen::MatrixXd density = H2ClosedShellDensity();
    const Eigen::MatrixXd perturbation = SymmetricPerturbation();
    const auto center = engine->EvaluateClosedShell(density);
    ASSERT_TRUE(center.has_value());
    const double analytic =
        0.5 * TraceProduct(perturbation, center->potentialAlpha + center->potentialBeta);

    const auto residual = [&engine, &density, &perturbation, analytic](double step) {
        const auto plus = engine->EvaluateClosedShell(density + step * perturbation);
        const auto minus = engine->EvaluateClosedShell(density - step * perturbation);

        if (!plus.has_value() || !minus.has_value())
        {
            ADD_FAILURE() << "the finite-difference evaluations must succeed";

            return 0.0;
        }

        return std::abs((plus->energy - minus->energy) / (2.0 * step) - analytic);
    };

    const double coarse = residual(1e-2);
    const double fine = residual(5e-3);
    EXPECT_GT(coarse, 2.0 * fine) << "residual at step 1e-2 (" << coarse
                                  << ") did not exceed twice the residual at 5e-3 (" << fine
                                  << "): it is not growing with the step";
    EXPECT_LT(coarse, 8.0 * fine) << "residual at step 1e-2 (" << coarse
                                  << ") exceeded eight times the residual at 5e-3 (" << fine
                                  << "): it is growing faster than the step squared";
}

TEST(XcGridEngineTest, BadGridSettingsAreReportedThroughTheErrorSeam) {
    // A quadrature failure inside excgrid must surface as a translated qcx
    // Error naming the failed step - never a silent empty grid.
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value());

    XcGridSettings notALebedevSize;
    notALebedevSize.angularPoints = 51; // not a shipped Lebedev point count
    const auto badAngular = XcGridEngine::Create(*molecule, *basis, "slater", notALebedevSize);
    ASSERT_FALSE(badAngular.has_value());
    EXPECT_EQ(badAngular.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(badAngular.error().message.find("block grid"), std::string::npos);

    XcGridSettings noRadialPoints;
    noRadialPoints.radialPoints = 0;
    const auto badRadial = XcGridEngine::Create(*molecule, *basis, "slater", noRadialPoints);
    ASSERT_FALSE(badRadial.has_value());
    EXPECT_EQ(badRadial.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(XcGridEngineTest, PotentialsAreSymmetricAndSpinSymmetric) {
    for (const std::string_view name : {"slater", "pbe", "b3lyp"})
    {
        const auto engine = MakeEngine(name);
        ASSERT_TRUE(engine.has_value()) << name;

        const Eigen::MatrixXd halfDensity = 0.5 * H2ClosedShellDensity();
        const auto evaluation = engine->Evaluate(halfDensity, halfDensity);
        ASSERT_TRUE(evaluation.has_value()) << name;

        EXPECT_LT((evaluation->potentialAlpha - evaluation->potentialAlpha.transpose())
                      .cwiseAbs()
                      .maxCoeff(),
                  1e-12)
            << name;
        EXPECT_LT((evaluation->potentialAlpha - evaluation->potentialBeta).cwiseAbs().maxCoeff(),
                  1e-12)
            << name;
    }
}

TEST(XcGridEngineTest, FunctionalPlumbingMatchesTheRegistry) {
    const auto slater = MakeEngine("slater");
    ASSERT_TRUE(slater.has_value());
    EXPECT_FALSE(slater->UsesGradient());
    EXPECT_DOUBLE_EQ(slater->ExchangeFraction(), 0.0);

    const auto pbe = MakeEngine("pbe");
    ASSERT_TRUE(pbe.has_value());
    EXPECT_TRUE(pbe->UsesGradient());

    const auto b3lyp = MakeEngine("b3lyp");
    ASSERT_TRUE(b3lyp.has_value());
    EXPECT_NEAR(b3lyp->ExchangeFraction(), 0.20, 1e-12);
}

TEST(XcGridEngineTest, UnknownFunctionalIsRejected) {
    const auto engine = MakeEngine("not_a_functional");
    ASSERT_FALSE(engine.has_value());
    EXPECT_EQ(engine.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(engine.error().message.find("not_a_functional"), std::string::npos);
    EXPECT_NE(engine.error().message.find("slater"), std::string::npos);
}

TEST(XcGridEngineTest, DensityShapeIsChecked) {
    const auto engine = MakeEngine("slater");
    ASSERT_TRUE(engine.has_value());

    Eigen::MatrixXd wrongSize = Eigen::MatrixXd::Zero(3, 3);

    const auto closedShell = engine->EvaluateClosedShell(wrongSize);
    ASSERT_FALSE(closedShell.has_value());
    EXPECT_EQ(closedShell.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto unrestricted = engine->Evaluate(wrongSize, H2ClosedShellDensity());
    ASSERT_FALSE(unrestricted.has_value());
    EXPECT_EQ(unrestricted.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(GeometryTranslationTest, CarriesSpeciesAndBohrCoordinatesInAtomOrder) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    const auto geometry = ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value());
    ASSERT_EQ(geometry->atoms.size(), molecule->AtomCount());

    for (std::size_t atom = 0; atom < molecule->AtomCount(); ++atom)
    {
        EXPECT_EQ(geometry->atoms[atom].atomicNumber, molecule->Atoms()[atom].atomicNumber);

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            EXPECT_DOUBLE_EQ(geometry->atoms[atom].position[axis],
                             molecule->CoordinatesBohr()(atom, axis));
        }
    }

    // The fixture's H2 sits at R = 1.4 bohr.
    const std::array<double, 3>& first = geometry->atoms[0].position;
    const std::array<double, 3>& second = geometry->atoms[1].position;
    const double dx = second[0] - first[0];
    const double dy = second[1] - first[1];
    const double dz = second[2] - first[2];

    EXPECT_NEAR(std::sqrt(dx * dx + dy * dy + dz * dz), 1.4, 1e-12);
}

TEST(GeometryTranslationTest, MapsExcgridErrorCodes) {
    EXPECT_EQ(TranslateExcgridError(excgrid::ErrorCode::kInvalidArgument, "probe").code,
              qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(TranslateExcgridError(excgrid::ErrorCode::kUnsupported, "probe").code,
              qcx::ErrorCode::kUnimplemented);

    const qcx::Error mapped = TranslateExcgridError(excgrid::ErrorCode::kUnsupported, "probe");
    EXPECT_NE(mapped.message.find("probe"), std::string::npos);
}

} // namespace qcx::grid
