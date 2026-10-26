// The grid's own second-order contribution to the XC Hessian, against the
// central difference of the analytic gradient it differentiates:
//   - a constant energy density switches every electronic channel off exactly,
//     so the difference can see nothing but the quadrature weights' curvature;
//   - the density matrix is held fixed in the AO basis while the nuclei move,
//     which is the convention both walks claim;
//   - a provider carrying a prescribed position curvature pins the term that
//     the default grid's rigidly translating points leave at zero.
//
// The full-density finite-difference gate is the Hessian arc's acceptance. A
// density-dependent integrand responds to moving nuclei at fixed density
// through a channel the evaluator cannot supply without a third-derivative AO
// tier, so it is switched off here exactly rather than approximated.

#include "h2o_sto3g.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/grid/xc_grid_engine.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <excgrid/grid_derivatives.hpp>
#include <excgrid/kernel.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

constexpr std::size_t kRadialPoints = 20;
constexpr std::size_t kAngularPoints = 50;

// Keep every shell: the finite difference moves the nuclei, and a shell
// switching out between the reference and a displaced point would make the
// difference discontinuous.
constexpr double kScreeningTolerance = -1.0;

// The central difference's step and its half, combined by Richardson. The gate
// is relative: the constant integrand weights every point, far-tail curvature
// included, so entries reach 1e7 and an absolute 1e-7 would sit below the
// difference's own rounding floor.
constexpr double kFiniteDifferenceStep = 1e-4;
constexpr double kFiniteDifferenceTolerance = 1e-7;

// A unit-scale energy density that does not depend on the density at all, so
// every term that carries a partial of it vanishes and the weights' curvature
// is the whole of the second derivative.
constexpr double kEnergyDensityValue = 1.7;

// The partial the position-curvature test's functional carries on each spin
// density.
constexpr double kRhoPartial = 0.75;

// The scale of the prescribed position curvature.
constexpr double kPositionCurvatureScale = 1.0;

std::string Scientific(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6e", value);

    return buffer;
}

// The prescribed position curvature, one distinct value per (direction, row,
// column) and symmetric in the last two so the walk's upper-triangle mirror is
// exact. Shared by the provider that supplies it and the test that integrates
// it, so the two cannot drift.
double PrescribedPositionCurvature(double scale,
                                   std::size_t direction,
                                   std::size_t row,
                                   std::size_t column) {
    return scale * static_cast<double>((direction + 1) * (row + 1) * (column + 1));
}

// An integrand with no density dependence: e(rho, sigma, tau) = value.
class ConstantFunctional final : public excgrid::XcFunctional {
public:
    explicit ConstantFunctional(double value) : _value(value) {}

    [[nodiscard]] bool UsesGradient() const override {
        return false;
    }

    [[nodiscard]] double ExchangeFraction() const override {
        return 0.0;
    }

    [[nodiscard]] excgrid::ComponentMask RequiredMask() const override {
        return {};
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePoint(const excgrid::PointInputs&,
                                                      excgrid::PointResult& result) const override {
        result.exc = _value;
        return excgrid::KernelStatus::kOk;
    }

private:
    double _value;
};

// e = partial * (rho_a + rho_b): the integrand's spatial gradient is the density
// gradient's, which is what the position curvature multiplies, and its energy
// density is zero so the weight channel drops out of the comparison.
class RhoFunctional final : public excgrid::XcFunctional {
public:
    explicit RhoFunctional(double partial) : _partial(partial) {}

    [[nodiscard]] bool UsesGradient() const override {
        return false;
    }

    [[nodiscard]] double ExchangeFraction() const override {
        return 0.0;
    }

    [[nodiscard]] excgrid::ComponentMask RequiredMask() const override {
        excgrid::ComponentMask mask;
        mask.Set(excgrid::Component::RhoA);
        mask.Set(excgrid::Component::RhoB);
        return mask;
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePoint(const excgrid::PointInputs&,
                                                      excgrid::PointResult& result) const override {
        result.first[excgrid::IndexOf(excgrid::Component::RhoA)] = _partial;
        result.first[excgrid::IndexOf(excgrid::Component::RhoB)] = _partial;
        return excgrid::KernelStatus::kOk;
    }

private:
    double _partial;
};

// A provider that delegates to the real one and then overwrites the position
// curvature. The real grid rides each point on its owning atom, so its position
// curvature is identically zero and the carry-through has no other way to be
// observed.
class CurvedPositionProvider final : public excgrid::GridDerivativeProvider {
public:
    CurvedPositionProvider(const excgrid::GridDerivativeProvider& inner, double scale) :
        _inner(inner), _scale(scale) {}

    [[nodiscard]] excgrid::DerivativeOrder MaxOrder() const noexcept override {
        return excgrid::DerivativeOrder::kSecond;
    }

    [[nodiscard]] excgrid::DerivativeStatus Evaluate(
        const excgrid::Block& block,
        const excgrid::DerivativeRequest& request,
        const excgrid::BlockDerivatives& derivatives) const override {
        const excgrid::DerivativeStatus status = _inner.Evaluate(block, request, derivatives);

        if (status != excgrid::DerivativeStatus::kOk ||
            request.order != excgrid::DerivativeOrder::kSecond)
        {
            return status;
        }

        const std::size_t coordinates = request.nuclearCoordinateCount;
        const std::size_t stride = coordinates * coordinates;

        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            double* row = derivatives.positionSecond.data() + point * 3 * stride;

            for (std::size_t direction = 0; direction < 3; ++direction)
            {
                for (std::size_t rowIndex = 0; rowIndex < coordinates; ++rowIndex)
                {
                    for (std::size_t column = rowIndex; column < coordinates; ++column)
                    {
                        const double value =
                            PrescribedPositionCurvature(_scale, direction, rowIndex, column);
                        row[direction * stride + rowIndex * coordinates + column] = value;
                        row[direction * stride + column * coordinates + rowIndex] = value;
                    }
                }
            }
        }

        return status;
    }

private:
    const excgrid::GridDerivativeProvider& _inner;
    double _scale;
};

std::optional<qcx::molecule::Molecule> DisplacedMolecule(const qcx::molecule::Molecule& reference,
                                                         std::size_t atom,
                                                         std::size_t axis,
                                                         double delta) {
    const std::size_t atomCount = reference.AtomCount();
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomCount, 3});

    if (!coordinates.has_value())
    {
        return std::nullopt;
    }

    for (std::size_t row = 0; row < atomCount; ++row)
    {
        for (std::size_t column = 0; column < 3; ++column)
        {
            (*coordinates)(row, column) = reference.CoordinatesBohr()(row, column);
        }
    }

    (*coordinates)(atom, axis) += delta;
    coordinates->MarkHostDirty();

    auto displaced = qcx::molecule::Molecule::Create(
        reference.Atoms(), std::move(*coordinates), reference.Charge(), reference.Multiplicity());

    if (!displaced.has_value() || displaced->AtomCount() != atomCount)
    {
        return std::nullopt;
    }

    // The molecule renumbers canonically; a reordering would make the caller's
    // atom index name a different atom.
    for (std::size_t index = 0; index < atomCount; ++index)
    {
        if (displaced->Atoms()[index].atomicNumber != reference.Atoms()[index].atomicNumber)
        {
            return std::nullopt;
        }
    }

    return std::move(*displaced);
}

struct HessianFixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    XcGridSettings settings;
    std::string functionalName;
    XcGridEngine engine;
    std::unique_ptr<excgrid::GridDerivativeProvider> provider;
    Eigen::MatrixXd density; ///< 2 on the occupied diagonal, zero elsewhere.
};

std::optional<HessianFixture> MakeHessianFixture(
    std::string_view functionalName,
    excgrid::DerivativeOrder order = excgrid::DerivativeOrder::kSecond) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();

    if (!basis.has_value())
    {
        return std::nullopt;
    }

    auto molecule = qcx::testing::MakeH2oSto3g();

    if (!molecule.has_value())
    {
        return std::nullopt;
    }

    XcGridSettings settings;
    settings.radialPoints = kRadialPoints;
    settings.angularPoints = kAngularPoints;
    settings.blockTarget = 256;

    auto engine = XcGridEngine::Create(*molecule, *basis, functionalName, settings);

    if (!engine.has_value())
    {
        return std::nullopt;
    }

    auto geometry = ToExcgridGeometry(*molecule);

    if (!geometry.has_value())
    {
        return std::nullopt;
    }

    auto provider =
        excgrid::CreateGridDerivativeProvider(*geometry, ToExcgridParams(settings), order);

    if (!provider.has_value())
    {
        return std::nullopt;
    }

    const std::size_t aoCount = engine->Evaluator().AOCount();
    const std::size_t occupied = static_cast<std::size_t>(molecule->ElectronCount()) / 2;
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(aoCount),
                                                    static_cast<Eigen::Index>(aoCount));

    for (std::size_t orbital = 0; orbital < occupied; ++orbital)
    {
        density(static_cast<Eigen::Index>(orbital), static_cast<Eigen::Index>(orbital)) = 2.0;
    }

    return HessianFixture{std::move(*molecule),
                          std::move(*basis),
                          settings,
                          std::string(functionalName),
                          std::move(*engine),
                          std::move(*provider),
                          std::move(density)};
}

// The analytic gradient one displaced geometry owes, at the fixed density.
std::optional<Eigen::VectorXd> DisplacedGradient(const HessianFixture& fixture,
                                                 const excgrid::XcFunctional& functional,
                                                 std::size_t atom,
                                                 std::size_t axis,
                                                 double delta,
                                                 const XcIntegrationThresholds& thresholds) {
    auto displaced = DisplacedMolecule(fixture.molecule, atom, axis, delta);

    if (!displaced.has_value())
    {
        return std::nullopt;
    }

    auto engine =
        XcGridEngine::Create(*displaced, fixture.basis, fixture.functionalName, fixture.settings);

    if (!engine.has_value() || engine->PointCount() != fixture.engine.PointCount())
    {
        return std::nullopt;
    }

    auto geometry = ToExcgridGeometry(*displaced);

    if (!geometry.has_value())
    {
        return std::nullopt;
    }

    auto provider =
        excgrid::CreateGridDerivativeProvider(*geometry, ToExcgridParams(fixture.settings));

    if (!provider.has_value())
    {
        return std::nullopt;
    }

    const auto evaluated = EvaluateXcGradient(engine->Grid().Blocks(),
                                              **provider,
                                              engine->Evaluator(),
                                              functional,
                                              engine->ShellEnvelopes(),
                                              thresholds,
                                              XcBatchSettings{},
                                              fixture.density,
                                              fixture.density);

    if (!evaluated.has_value())
    {
        return std::nullopt;
    }

    return evaluated->gradient;
}

// One row of the Hessian by a Richardson-extrapolated central difference of the
// analytic gradient: the step's h^2 term is removed by the half-step value, so
// the comparison is the walk's arithmetic and not the step's truncation.
std::optional<Eigen::VectorXd> NumericalGradient(const HessianFixture& fixture,
                                                 const excgrid::XcFunctional& functional,
                                                 std::size_t atom,
                                                 std::size_t axis,
                                                 const XcIntegrationThresholds& thresholds) {
    const double step = kFiniteDifferenceStep;
    const auto plus = DisplacedGradient(fixture, functional, atom, axis, step, thresholds);
    const auto minus = DisplacedGradient(fixture, functional, atom, axis, -step, thresholds);
    const auto halfPlus =
        DisplacedGradient(fixture, functional, atom, axis, 0.5 * step, thresholds);
    const auto halfMinus =
        DisplacedGradient(fixture, functional, atom, axis, -0.5 * step, thresholds);

    if (!plus.has_value() || !minus.has_value() || !halfPlus.has_value() || !halfMinus.has_value())
    {
        return std::nullopt;
    }

    const Eigen::VectorXd coarse = (*plus - *minus) / (2.0 * step);
    const Eigen::VectorXd fine = (*halfPlus - *halfMinus) / step;

    return (4.0 * fine - coarse) / 3.0;
}

// The density's gradient at one point, in the walk's own contraction and over
// every slot: the tolerance below zero keeps every shell, so the walk's
// selection is the dense one and this is the same sum.
std::array<double, 3> DensityGradient(const AoEvaluator& evaluator,
                                      const Eigen::MatrixXd& density,
                                      const std::array<double, 3>& point) {
    const std::size_t aoCount = evaluator.AOCount();
    std::vector<double> values(aoCount);
    std::vector<double> gradients(3 * aoCount);
    evaluator.EvaluateGradients(point, values, gradients);

    std::vector<double> contractedValues(aoCount);
    std::vector<double> contractedGradients(3 * aoCount);

    for (std::size_t mu = 0; mu < aoCount; ++mu)
    {
        for (std::size_t nu = 0; nu < aoCount; ++nu)
        {
            const double element =
                density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));
            contractedValues[mu] += element * values[nu];

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                contractedGradients[3 * mu + axis] += element * gradients[3 * nu + axis];
            }
        }
    }

    std::array<double, 3> result{};

    for (std::size_t mu = 0; mu < aoCount; ++mu)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            result[axis] += gradients[3 * mu + axis] * contractedValues[mu] +
                            values[mu] * contractedGradients[3 * mu + axis];
        }
    }

    return result;
}

TEST(XcGradientHessianTest, TheGridMotionHessianMatchesTheGradientDifference) {
    const auto fixture = MakeHessianFixture("slater");
    ASSERT_TRUE(fixture.has_value());

    const ConstantFunctional functional(kEnergyDensityValue);
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);
    const std::size_t coordinateCount = 3 * fixture->molecule.AtomCount();

    const auto hessian = EvaluateXcGridMotionHessian(fixture->engine.Grid().Blocks(),
                                                     *fixture->provider,
                                                     fixture->engine.Evaluator(),
                                                     functional,
                                                     fixture->engine.ShellEnvelopes(),
                                                     thresholds,
                                                     fixture->density,
                                                     fixture->density);
    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;
    ASSERT_EQ(hessian->hessian.rows(), static_cast<Eigen::Index>(coordinateCount));
    ASSERT_EQ(hessian->hessian.cols(), static_cast<Eigen::Index>(coordinateCount));
    EXPECT_GT(hessian->counts.points, 0U);

    // The gradient this walk accumulated is the first-order walk's: the two
    // group the same terms differently, so they agree to rounding.
    const auto analytic = EvaluateXcGradient(fixture->engine.Grid().Blocks(),
                                             *fixture->provider,
                                             fixture->engine.Evaluator(),
                                             functional,
                                             fixture->engine.ShellEnvelopes(),
                                             thresholds,
                                             XcBatchSettings{},
                                             fixture->density,
                                             fixture->density);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    const double gradientDeviation = (hessian->gradient - analytic->gradient).cwiseAbs().maxCoeff();
    ::testing::Test::RecordProperty("gradient agreement (max deviation, Hartree/Bohr)",
                                    Scientific(gradientDeviation));
    EXPECT_LT(gradientDeviation, 1e-10 * (1.0 + analytic->gradient.cwiseAbs().maxCoeff()));

    // One triangle is accumulated and mirrored, so the symmetry is exact: an
    // absolute value is never negative, and this pins it at zero.
    const Eigen::MatrixXd asymmetry = hessian->hessian - hessian->hessian.transpose();
    EXPECT_LE(asymmetry.cwiseAbs().maxCoeff(), 0.0);

    // A grid whose weights did not move with the geometry would satisfy the
    // gate below with two zeros; the curvature has to be here first.
    EXPECT_GT(hessian->hessian.cwiseAbs().maxCoeff(), 1e-8);

    double maxDeviation = 0.0;
    double maxNumerical = 0.0;

    for (std::size_t atom = 0; atom < fixture->molecule.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const std::size_t coordinate = 3 * atom + axis;
            const auto numerical = NumericalGradient(*fixture, functional, atom, axis, thresholds);
            ASSERT_TRUE(numerical.has_value()) << "displacement " << coordinate;
            maxNumerical = std::max(maxNumerical, numerical->cwiseAbs().maxCoeff());

            for (std::size_t row = 0; row < coordinateCount; ++row)
            {
                const Eigen::Index entry = static_cast<Eigen::Index>(row);
                const double value = hessian->hessian(entry, static_cast<Eigen::Index>(coordinate));
                maxDeviation = std::max(maxDeviation, std::abs((*numerical)[entry] - value));
            }
        }
    }

    ::testing::Test::RecordProperty("grid motion Hessian agreement (max deviation, Hartree/Bohr^2)",
                                    Scientific(maxDeviation));
    const double scale = 1.0 + std::max(hessian->hessian.cwiseAbs().maxCoeff(), maxNumerical);
    EXPECT_LT(maxDeviation, kFiniteDifferenceTolerance * scale)
        << "the second-order walk and the displaced-geometry gradients disagree";
}

TEST(XcGradientHessianTest, ThePositionCurvatureReachesTheHessian) {
    const auto fixture = MakeHessianFixture("pbe");
    ASSERT_TRUE(fixture.has_value());

    const RhoFunctional functional(kRhoPartial);
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);
    const std::size_t coordinateCount = 3 * fixture->molecule.AtomCount();
    const CurvedPositionProvider curved(*fixture->provider, kPositionCurvatureScale);

    const auto plain = EvaluateXcGridMotionHessian(fixture->engine.Grid().Blocks(),
                                                   *fixture->provider,
                                                   fixture->engine.Evaluator(),
                                                   functional,
                                                   fixture->engine.ShellEnvelopes(),
                                                   thresholds,
                                                   fixture->density,
                                                   fixture->density);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    ASSERT_EQ(plain->counts.aoSlotsEvaluated,
              plain->counts.points * fixture->engine.Evaluator().AOCount())
        << "the tolerance must keep every shell for the expected value below";

    const auto perturbed = EvaluateXcGridMotionHessian(fixture->engine.Grid().Blocks(),
                                                       curved,
                                                       fixture->engine.Evaluator(),
                                                       functional,
                                                       fixture->engine.ShellEnvelopes(),
                                                       thresholds,
                                                       fixture->density,
                                                       fixture->density);
    ASSERT_TRUE(perturbed.has_value()) << perturbed.error().message;

    // The cross section the position curvature multiplies: the weighted sum of
    // the integrand's spatial gradient, which is the two spins' partials times
    // the one density's gradient.
    std::array<double, 3> crossSection{};

    for (const excgrid::Block& block : fixture->engine.Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            if (block.weights[point] == 0.0)
            {
                continue;
            }

            const std::array<double, 3> gradient =
                DensityGradient(fixture->engine.Evaluator(), fixture->density, block.points[point]);

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                crossSection[axis] += block.weights[point] * 2.0 * kRhoPartial * gradient[axis];
            }
        }
    }

    double maxDeviation = 0.0;
    double maxExpected = 0.0;

    for (std::size_t row = 0; row < coordinateCount; ++row)
    {
        for (std::size_t column = row; column < coordinateCount; ++column)
        {
            double expected = 0.0;

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                const double curvature =
                    PrescribedPositionCurvature(kPositionCurvatureScale, axis, row, column);
                expected += curvature * crossSection[axis];
            }

            const Eigen::Index rowEntry = static_cast<Eigen::Index>(row);
            const Eigen::Index columnEntry = static_cast<Eigen::Index>(column);
            const double actual =
                perturbed->hessian(rowEntry, columnEntry) - plain->hessian(rowEntry, columnEntry);
            maxDeviation = std::max(maxDeviation, std::abs(actual - expected));
            maxExpected = std::max(maxExpected, std::abs(expected));
        }
    }

    ::testing::Test::RecordProperty("position curvature agreement (max deviation, Hartree/Bohr^2)",
                                    Scientific(maxDeviation));
    EXPECT_GT(maxExpected, 1e-6) << "the prescribed curvature must be visible";
    EXPECT_LT(maxDeviation, 1e-9 * (1.0 + maxExpected));
}

TEST(XcGradientHessianTest, AFirstOrderProviderIsRefused) {
    const auto fixture = MakeHessianFixture("pbe", excgrid::DerivativeOrder::kFirst);
    ASSERT_TRUE(fixture.has_value());

    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);
    const auto hessian = EvaluateXcGridMotionHessian(
        fixture->engine, fixture->density, fixture->density, *fixture->provider, thresholds);

    ASSERT_FALSE(hessian.has_value());
    EXPECT_EQ(hessian.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace qcx::grid
