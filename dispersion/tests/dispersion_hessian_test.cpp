// The dispersion module's coordinate-Hessian acceptance: the D4 arm's
// frozen-charge Hessian against central finite differences of the
// frozen-charge gradient, its symmetry and translation sum rule, and the two
// second derivatives the module refuses by name.
//
// The finite-difference reference is the frozen-charge gradient expression
// reconstructed from the module's own reference pair list, not the gradient
// entry called on displaced molecules: that entry recomputes the pair C6, C8
// and damping radius at the displaced geometry, so differentiating it
// measures the charge-relaxed derivative, which is a different second
// derivative and is refused by name.

#include "dispersion_fixture.hpp"
#include "qcx/dispersion/dispersion.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <excgrid/d4.hpp>
#include <string>
#include <vector>

using namespace qcx::dispersion::testing;

namespace {

qcx::Result<qcx::dispersion::ResolvedDispersion> ResolvedWater(
    qcx::dispersion::DispersionForm form) {
    return qcx::dispersion::ResolveDispersion(PresetOptions(form, "b3lyp"), "hf");
}

} // namespace

TEST(DispersionHessianTest, D4FrozenChargeHessianMatchesFiniteDifferenceOfTheGradient) {
    auto molecule = MakeMolecule(kWater, Flatten(kWater), 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = ResolvedWater(qcx::dispersion::DispersionForm::kD4);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto analytic = qcx::dispersion::EvaluateDispersionHessian(*molecule, *resolved);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    const auto dimension = static_cast<Eigen::Index>(3 * molecule->AtomCount());
    ASSERT_EQ(analytic->hessian.rows(), dimension);
    ASSERT_EQ(analytic->hessian.cols(), dimension);

    // The Hessian entry reports the energy entry's own number: one model.
    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_TRUE(energy.has_value()) << energy.error().message;
    EXPECT_EQ(analytic->energy, *energy);

    // The frozen-charge reference: the module's own pair list at the
    // reference geometry, with each pair's C6, C8 and damping radius held,
    // summed as a function of the displaced coordinates. It is the same
    // expression the gradient entry differentiates, and the same definition
    // the Hessian implements.
    auto geometry = qcx::grid::ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;

    const auto parameters = qcx::dispersion::ParameterMap(*resolved);
    const double s6 = parameters.at("s6");
    const double s8 = parameters.at("s8");
    const excgrid::D4Parameters kernel{s6, s8, parameters.at("a1"), parameters.at("a2")};

    auto pairs = excgrid::D4TwoBodyPairs(*geometry, kernel);
    ASSERT_TRUE(pairs.has_value()) << "the D4 pair list was refused";

    const Eigen::VectorXd coordinates = CanonicalCoordinates(*molecule);

    const auto frozenGradient = [&pairs, s6, s8](const Eigen::VectorXd& positions) {
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(positions.size());

        for (const excgrid::D4TwoBodyPair& pair : *pairs)
        {
            const double dx = positions(static_cast<Eigen::Index>(3 * pair.first)) -
                              positions(static_cast<Eigen::Index>(3 * pair.second));
            const double dy = positions(static_cast<Eigen::Index>(3 * pair.first + 1)) -
                              positions(static_cast<Eigen::Index>(3 * pair.second + 1));
            const double dz = positions(static_cast<Eigen::Index>(3 * pair.first + 2)) -
                              positions(static_cast<Eigen::Index>(3 * pair.second + 2));
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (!(r > 0.0))
            {
                continue;
            }

            const double r6 = std::pow(r, 6.0);
            const double r8 = r6 * r * r;
            const double radius6 = std::pow(pair.radius, 6.0);
            const double radius8 = radius6 * pair.radius * pair.radius;
            const double denominator6 = r6 + radius6;
            const double denominator8 = r8 + radius8;
            const double radial =
                6.0 * s6 * pair.c6 * (r6 / (r * r)) / (denominator6 * denominator6) +
                8.0 * s8 * pair.c8 * r6 / (denominator8 * denominator8);

            const std::array<double, 3> displacement{dx, dy, dz};

            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                const double term = radial * displacement[direction];
                gradient(static_cast<Eigen::Index>(3 * pair.first + direction)) += term;
                gradient(static_cast<Eigen::Index>(3 * pair.second + direction)) -= term;
            }
        }

        return gradient;
    };

    // At the reference geometry the reconstructed expression is the gradient
    // entry's number, bit for bit: the same operands in the same order.
    auto referenceGradient = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_TRUE(referenceGradient.has_value()) << referenceGradient.error().message;
    EXPECT_TRUE(frozenGradient(coordinates) == referenceGradient->gradient);

    // h = 1e-6 Bohr: the expression is smooth at the separation scale here, so
    // the central difference's truncation error is far below the 1e-9 gate,
    // and a Hessian built from a different expression would miss it by the
    // size of the entry.
    constexpr double kStep = 1e-6;
    constexpr double kTolerance = 1e-9;

    for (Eigen::Index coordinate = 0; coordinate < coordinates.size(); ++coordinate)
    {
        Eigen::VectorXd plus = coordinates;
        Eigen::VectorXd minus = coordinates;
        plus(coordinate) += kStep;
        minus(coordinate) -= kStep;

        const Eigen::VectorXd finiteDifference =
            (frozenGradient(plus) - frozenGradient(minus)) / (2.0 * kStep);

        for (Eigen::Index component = 0; component < dimension; ++component)
        {
            EXPECT_NEAR(
                analytic->hessian(component, coordinate), finiteDifference(component), kTolerance)
                << "component " << component << ", coordinate " << coordinate;
        }
    }
}

TEST(DispersionHessianTest, TheD4FrozenChargeHessianIsSymmetric) {
    auto molecule = MakeMolecule(kWater, Flatten(kWater), 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = ResolvedWater(qcx::dispersion::DispersionForm::kD4);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto analytic = qcx::dispersion::EvaluateDispersionHessian(*molecule, *resolved);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    // Second derivatives commute, and the pair assembly writes each off-
    // diagonal block with the same value it writes its transpose: the matrix
    // is symmetric to the rounding of the sum.
    const Eigen::MatrixXd asymmetry = analytic->hessian - analytic->hessian.transpose();
    EXPECT_LT(asymmetry.cwiseAbs().maxCoeff(), 1e-14) << asymmetry;
}

TEST(DispersionHessianTest, TheD4FrozenChargeHessianSumsToZeroOverTranslations) {
    auto molecule = MakeMolecule(kWater, Flatten(kWater), 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = ResolvedWater(qcx::dispersion::DispersionForm::kD4);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto analytic = qcx::dispersion::EvaluateDispersionHessian(*molecule, *resolved);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    // The correction depends on interatomic distances alone, so a uniform
    // translation is not a deformation: for every atom the sum of its row of
    // 3x3 blocks over all atoms is the zero matrix.
    const auto atoms = static_cast<Eigen::Index>(molecule->AtomCount());

    for (Eigen::Index atom = 0; atom < atoms; ++atom)
    {
        Eigen::Matrix3d total = Eigen::Matrix3d::Zero();

        for (Eigen::Index other = 0; other < atoms; ++other)
        {
            total += analytic->hessian.block<3, 3>(3 * atom, 3 * other);
        }

        EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-12) << "atom " << atom << "\n" << total;
    }
}

TEST(DispersionHessianTest, TheD3HessianIsRefusedByName) {
    auto molecule = MakeMolecule(kWater, Flatten(kWater), 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = ResolvedWater(qcx::dispersion::DispersionForm::kD3);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto analytic = qcx::dispersion::EvaluateDispersionHessian(*molecule, *resolved);
    ASSERT_FALSE(analytic.has_value());
    EXPECT_EQ(analytic.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(analytic.error().message.find("d3"), std::string::npos) << analytic.error().message;
    EXPECT_NE(analytic.error().message.find("Hessian"), std::string::npos)
        << analytic.error().message;
}

TEST(DispersionHessianTest, TheChargeRelaxedD4HessianIsRefusedByName) {
    auto molecule = MakeMolecule(kWater, Flatten(kWater), 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = ResolvedWater(qcx::dispersion::DispersionForm::kD4);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto analytic = qcx::dispersion::EvaluateDispersionHessian(
        *molecule, *resolved, qcx::dispersion::DispersionDerivativeTier::kChargeRelaxed);
    ASSERT_FALSE(analytic.has_value());
    EXPECT_EQ(analytic.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(analytic.error().message.find("charge-relaxed"), std::string::npos)
        << analytic.error().message;
}
