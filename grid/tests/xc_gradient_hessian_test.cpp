// The grid's own second-order contribution to the XC Hessian, against the
// central difference of the analytic gradient it differentiates:
//   - a constant energy density switches every electronic channel off exactly,
//     so the difference can see nothing but the quadrature weights' curvature;
//   - the density matrix is held fixed in the AO basis while the nuclei move,
//     which is the convention both walks claim;
//   - a provider carrying a prescribed position curvature pins the term that
//     the default grid's rigidly translating points leave at zero.
//
// The full-density finite-difference gate is the Hessian arc's acceptance. For a
// functional that reads only the spin densities the density's second response is
// expressible with the tiers the evaluator already has, so that Hessian is
// complete; a functional reading sigma or tau is refused by name, because its
// second response would need a third-derivative AO tier.

#include "h2o_sto3g.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/grid/xc_grid_engine.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <excgrid/grid_derivatives.hpp>
#include <excgrid/kernel.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <span>
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
// density is zero so the weight channel drops out of the comparison. The two
// classes below differ in one thing only - whether they declare the
// second-derivative tier the completed walk owes a density-reading functional -
// so their difference isolates what the tier is for.
class LinearRhoFunctional : public excgrid::XcFunctional {
public:
    explicit LinearRhoFunctional(double partial) : _partial(partial) {}

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

protected:
    double _partial;
};

// The zero second-derivative tier. e is linear in the two densities, so every
// second partial is zero; the tier exists to be spanned, because the walk
// refuses a functional that reads a density and declares none.
class RhoFunctional : public LinearRhoFunctional {
public:
    using LinearRhoFunctional::LinearRhoFunctional;

    [[nodiscard]] excgrid::ComponentMask SecondDerivativeMask() const noexcept override {
        return RequiredMask();
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePointWithSecondDerivatives(
        const excgrid::PointInputs& inputs,
        std::span<const double> rightHandSide,
        excgrid::PointResult& result,
        excgrid::PointSecondDerivative& second) const override {
        // The contract's own shape rules, so a request the registry would refuse
        // is refused here rather than silently answered with zeros.
        const std::size_t active =
            excgrid::ComponentMask{inputs.mask.bits & RequiredMask().bits}.ActiveCount();

        if (active == 0 || rightHandSide.size() % active != 0 ||
            rightHandSide.size() > excgrid::kSecondDerivativeCapacity)
        {
            return excgrid::KernelStatus::kRefusedUnsupportedCombination;
        }

        second = excgrid::PointSecondDerivative{};
        second.mask = excgrid::ComponentMask{inputs.mask.bits & RequiredMask().bits};
        second.rightHandSides = rightHandSide.size() / active;

        return EvaluatePoint(inputs, result);
    }
};

// e = c (rhoA + rhoB): the design's synthetic exact pin. The energy density is
// carried as well as the partials, so the weight terms have a nonzero integrand
// to differentiate, and the second partials stay zero.
class LinearRhoEnergyFunctional final : public RhoFunctional {
public:
    using RhoFunctional::RhoFunctional;

    [[nodiscard]] excgrid::KernelStatus EvaluatePoint(const excgrid::PointInputs& inputs,
                                                      excgrid::PointResult& result) const override {
        const excgrid::KernelStatus status = RhoFunctional::EvaluatePoint(inputs, result);
        result.exc = _partial *
                     (inputs.Get(excgrid::Component::RhoA) + inputs.Get(excgrid::Component::RhoB));
        return status;
    }
};

// The same energy density with no tier: what the walk must refuse rather than
// answer with the density channel's terms missing their second partials.
class UntieredRhoFunctional final : public LinearRhoFunctional {
public:
    using LinearRhoFunctional::LinearRhoFunctional;
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

// The synthetic pin's bar. Its expected Hessian is analytic and both sides read
// the same AO tier and provider, so only the summation order differs; the
// measured residual is recorded on the test.
constexpr double kSyntheticTolerance = 1e-10;

// One entry of the AO Hessian's packed six, in the evaluator's order
// (xx, xy, xz, yy, yz, zz).
double PackedHessian(std::span<const double> hessians,
                     std::size_t ao,
                     std::size_t row,
                     std::size_t column) {
    constexpr std::size_t slots[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return hessians[6 * ao + slots[row][column]];
}

// One spin density's value, spatial gradient and first and second total
// responses at a point, from the AO value, gradient and Hessian tiers alone.
// The test's own instrument: the product rule over AO pairs is written out with
// each AO's centre-relative displacement
//     S_q(mu) = X_q - delta(atom(mu), atom(q)) e_direction(q),
// so it shares no atom-block decomposition with the walk it pins.
struct DensityResponse {
    double value = 0.0;
    std::array<double, 3> gradient{};
    std::vector<double> first; ///< Per nuclear coordinate, D_q rho.
    std::vector<double> second; ///< Per coordinate pair, row-major, D_qD_r rho.
};

DensityResponse EvaluateDensityResponse(const AoEvaluator& evaluator,
                                        const Eigen::MatrixXd& density,
                                        std::span<const std::size_t> atomOfAo,
                                        std::span<const std::array<double, 3>> positionFirst,
                                        const std::array<double, 3>& point) {
    const std::size_t aoCount = evaluator.AOCount();
    const std::size_t coordinates = positionFirst.size();

    std::vector<double> values(aoCount);
    std::vector<double> gradients(3 * aoCount);
    std::vector<double> hessians(6 * aoCount);
    evaluator.EvaluateDerivatives(point, values, gradients, hessians);

    std::vector<std::array<double, 3>> shifts(coordinates * aoCount);

    for (std::size_t q = 0; q < coordinates; ++q)
    {
        for (std::size_t ao = 0; ao < aoCount; ++ao)
        {
            std::array<double, 3> shift = positionFirst[q];

            if (atomOfAo[ao] == q / 3)
            {
                shift[q % 3] -= 1.0;
            }

            shifts[q * aoCount + ao] = shift;
        }
    }

    // D_q phi_mu = S_q . grad phi_mu.
    std::vector<double> dPhi(coordinates * aoCount);

    for (std::size_t q = 0; q < coordinates; ++q)
    {
        for (std::size_t ao = 0; ao < aoCount; ++ao)
        {
            double value = 0.0;

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                value += shifts[q * aoCount + ao][axis] * gradients[3 * ao + axis];
            }

            dPhi[q * aoCount + ao] = value;
        }
    }

    DensityResponse response;
    response.first.assign(coordinates, 0.0);
    response.second.assign(coordinates * coordinates, 0.0);

    for (std::size_t ao = 0; ao < aoCount; ++ao)
    {
        for (std::size_t other = 0; other < aoCount; ++other)
        {
            const double element =
                density(static_cast<Eigen::Index>(ao), static_cast<Eigen::Index>(other));
            response.value += element * values[ao] * values[other];

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                response.gradient[axis] += element * (gradients[3 * ao + axis] * values[other] +
                                                      values[ao] * gradients[3 * other + axis]);
            }
        }
    }

    for (std::size_t q = 0; q < coordinates; ++q)
    {
        for (std::size_t ao = 0; ao < aoCount; ++ao)
        {
            for (std::size_t other = 0; other < aoCount; ++other)
            {
                const double element =
                    density(static_cast<Eigen::Index>(ao), static_cast<Eigen::Index>(other));
                response.first[q] += element * (dPhi[q * aoCount + ao] * values[other] +
                                                values[ao] * dPhi[q * aoCount + other]);
            }
        }
    }

    // D_qD_r(phi_mu phi_nu) = (D_qD_r phi_mu) phi_nu + (D_q phi_mu)(D_r phi_nu)
    //                       + (D_r phi_mu)(D_q phi_nu) + phi_mu (D_qD_r phi_nu),
    // with D_qD_r phi_mu = S_q^T hess phi_mu S_r.
    for (std::size_t q = 0; q < coordinates; ++q)
    {
        for (std::size_t r = 0; r < coordinates; ++r)
        {
            double value = 0.0;

            for (std::size_t ao = 0; ao < aoCount; ++ao)
            {
                const std::size_t qAo = q * aoCount + ao;
                const std::size_t rAo = r * aoCount + ao;
                double secondAo = 0.0;

                for (std::size_t row = 0; row < 3; ++row)
                {
                    for (std::size_t column = 0; column < 3; ++column)
                    {
                        secondAo += shifts[qAo][row] * PackedHessian(hessians, ao, row, column) *
                                    shifts[rAo][column];
                    }
                }

                for (std::size_t other = 0; other < aoCount; ++other)
                {
                    const std::size_t qOther = q * aoCount + other;
                    const std::size_t rOther = r * aoCount + other;
                    double secondOther = 0.0;

                    for (std::size_t row = 0; row < 3; ++row)
                    {
                        for (std::size_t column = 0; column < 3; ++column)
                        {
                            secondOther += shifts[qOther][row] *
                                           PackedHessian(hessians, other, row, column) *
                                           shifts[rOther][column];
                        }
                    }

                    const double element =
                        density(static_cast<Eigen::Index>(ao), static_cast<Eigen::Index>(other));
                    value += element * (secondAo * values[other] + dPhi[qAo] * dPhi[rOther] +
                                        dPhi[rAo] * dPhi[qOther] + values[ao] * secondOther);
                }
            }

            response.second[q * coordinates + r] = value;
        }
    }

    return response;
}

excgrid::DerivativeRequest SecondOrderRequest(std::size_t nuclearCoordinateCount) {
    excgrid::DerivativeRequest request;
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    request.parts = static_cast<excgrid::DerivativePart>(
        static_cast<std::uint8_t>(excgrid::DerivativePart::kCoordinates) |
        static_cast<std::uint8_t>(excgrid::DerivativePart::kWeights));
    request.order = excgrid::DerivativeOrder::kSecond;
    request.nuclearCoordinateCount = nuclearCoordinateCount;

    return request;
}

// The synthetic pin's expected Hessian, from the test-local field evaluator:
//     H_qr = sum_p [ W_,qr e + W_,q T_r + W_,r T_q
//                    + W (de/dx^c) X^c_,qr + W e_,qr ],
// with e = c (rhoA + rhoB), T its total first response and e_,qr its second.
// The upper triangle is filled and mirrored, as the walk does.
std::optional<Eigen::MatrixXd> ExpectedFixedDensityHessian(
    const XcGridEngine& engine,
    const excgrid::GridDerivativeProvider& provider,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    // partial is the pin's scale and atomCount the size of the coordinate block;
    // the call site reads them in that order.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    double partial,
    std::size_t atomCount,
    std::span<const std::size_t> atomOfAo) {
    const std::size_t coordinates = 3 * atomCount;
    const excgrid::DerivativeRequest request = SecondOrderRequest(coordinates);
    std::size_t largestBlock = 0;

    for (const excgrid::Block& block : engine.Grid().Blocks())
    {
        largestBlock = std::max(largestBlock, block.pointCount);
    }

    const excgrid::BlockDerivativeSizes sizes = excgrid::SizeOf(request, largestBlock);
    std::vector<double> positionFirst(sizes.positionFirst);
    std::vector<double> weightFirst(sizes.weightFirst);
    std::vector<double> positionSecond(sizes.positionSecond);
    std::vector<double> weightSecond(sizes.weightSecond);

    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(coordinates),
                                                     static_cast<Eigen::Index>(coordinates));
    std::vector<std::array<double, 3>> xFirst(coordinates);
    std::vector<double> firstResponse(coordinates);

    for (const excgrid::Block& block : engine.Grid().Blocks())
    {
        excgrid::BlockDerivatives derivatives;
        derivatives.positionFirst = std::span<double>(positionFirst.data(), sizes.positionFirst);
        derivatives.weightFirst = std::span<double>(weightFirst.data(), sizes.weightFirst);
        derivatives.positionSecond = std::span<double>(positionSecond.data(), sizes.positionSecond);
        derivatives.weightSecond = std::span<double>(weightSecond.data(), sizes.weightSecond);
        derivatives.atomIndex = block.atomIndex;

        if (provider.Evaluate(block, request, derivatives) != excgrid::DerivativeStatus::kOk)
        {
            return std::nullopt;
        }

        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const double weight = block.weights[point];

            if (weight == 0.0)
            {
                continue;
            }

            const double* positionRow = positionFirst.data() + point * 3 * coordinates;

            for (std::size_t q = 0; q < coordinates; ++q)
            {
                xFirst[q] = {
                    positionRow[q], positionRow[coordinates + q], positionRow[2 * coordinates + q]};
            }

            const DensityResponse alpha = EvaluateDensityResponse(
                engine.Evaluator(), densityAlpha, atomOfAo, xFirst, block.points[point]);
            const DensityResponse beta = EvaluateDensityResponse(
                engine.Evaluator(), densityBeta, atomOfAo, xFirst, block.points[point]);

            const double energy = partial * (alpha.value + beta.value);
            std::array<double, 3> spatial{};

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                spatial[axis] = partial * (alpha.gradient[axis] + beta.gradient[axis]);
            }

            for (std::size_t q = 0; q < coordinates; ++q)
            {
                firstResponse[q] = partial * (alpha.first[q] + beta.first[q]);
            }

            const double* weightRow = weightFirst.data() + point * coordinates;
            const double* weightCurvature = weightSecond.data() + point * coordinates * coordinates;

            for (std::size_t q = 0; q < coordinates; ++q)
            {
                for (std::size_t r = q; r < coordinates; ++r)
                {
                    double value =
                        weightCurvature[q * coordinates + r] * energy +
                        weightRow[q] * firstResponse[r] + weightRow[r] * firstResponse[q] +
                        weight * partial *
                            (alpha.second[q * coordinates + r] + beta.second[q * coordinates + r]);

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        value += weight * spatial[axis] *
                                 positionSecond[(point * 3 + axis) * coordinates * coordinates +
                                                q * coordinates + r];
                    }

                    expected(static_cast<Eigen::Index>(q), static_cast<Eigen::Index>(r)) += value;
                }
            }
        }
    }

    for (std::size_t q = 0; q < coordinates; ++q)
    {
        for (std::size_t r = q + 1; r < coordinates; ++r)
        {
            expected(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(q)) =
                expected(static_cast<Eigen::Index>(q), static_cast<Eigen::Index>(r));
        }
    }

    return expected;
}

// The acceptance gate: with the density channel in place, the fixed-density LDA
// Hessian differentiates the displaced analytic gradient.
TEST(XcGradientHessianTest, TheFixedDensityLdaHessianMatchesTheGradientDifference) {
    const auto fixture = MakeHessianFixture("slater");
    ASSERT_TRUE(fixture.has_value());

    const excgrid::XcFunctional& functional = fixture->engine.Functional();
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);
    const std::size_t coordinateCount = 3 * fixture->molecule.AtomCount();

    const auto hessian = EvaluateXcGridMotionHessian(
        fixture->engine, fixture->density, fixture->density, *fixture->provider, thresholds);
    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;
    ASSERT_EQ(hessian->hessian.rows(), static_cast<Eigen::Index>(coordinateCount));
    ASSERT_EQ(hessian->hessian.cols(), static_cast<Eigen::Index>(coordinateCount));

    // The density channel ran, in the chunks the tier's capacity allows: two
    // active components, so 16 / 2 right-hand sides per call.
    const std::size_t sides = excgrid::kSecondDerivativeCapacity / 2;
    EXPECT_EQ(hessian->counts.kernelSecondDerivativeCalls,
              ((coordinateCount + sides - 1) / sides) * hessian->counts.points);

    // The tolerance keeps every shell, so every atom block is filled at every
    // point and the screened column meets its dense counterpart.
    EXPECT_GT(hessian->counts.electronicCurvatureTerms, 0U);
    EXPECT_EQ(hessian->counts.electronicCurvatureTerms,
              hessian->counts.electronicCurvatureTermsDense);

    // The gradient this walk accumulated is the first-order walk's.
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

    // The two walks' counted cost, side by side. The first-order walk fetches
    // the value and gradient tiers; the second-order walk fetches the Hessian
    // tier as well, which is the electronic channel's visible per-point cost.
    ::testing::Test::RecordProperty("points", std::to_string(hessian->counts.points));
    ::testing::Test::RecordProperty("gradient walk AO slots",
                                    std::to_string(analytic->counts.aoSlotsEvaluated));
    ::testing::Test::RecordProperty("Hessian walk AO slots",
                                    std::to_string(hessian->counts.aoSlotsEvaluated));
    ::testing::Test::RecordProperty("electronic curvature terms",
                                    std::to_string(hessian->counts.electronicCurvatureTerms));
    ::testing::Test::RecordProperty("electronic curvature terms dense",
                                    std::to_string(hessian->counts.electronicCurvatureTermsDense));
    ::testing::Test::RecordProperty("kernel second-derivative calls",
                                    std::to_string(hessian->counts.kernelSecondDerivativeCalls));

    // One triangle is accumulated and mirrored, so the symmetry is exact.
    const Eigen::MatrixXd asymmetry = hessian->hessian - hessian->hessian.transpose();
    EXPECT_LE(asymmetry.cwiseAbs().maxCoeff(), 0.0);

    // The channel has to be visible: a walk that skipped it would leave only the
    // weight terms, and the difference below would then see the missing ones.
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

    ::testing::Test::RecordProperty(
        "fixed-density LDA Hessian agreement (max deviation, Hartree/Bohr^2)",
        Scientific(maxDeviation));
    const double scale = 1.0 + std::max(hessian->hessian.cwiseAbs().maxCoeff(), maxNumerical);
    EXPECT_LT(maxDeviation, kFiniteDifferenceTolerance * scale)
        << "the completed fixed-density Hessian and the displaced-geometry gradients disagree";
}

// The synthetic exact pin: e = c (rhoA + rhoB) is linear in the two densities,
// so its fixed-density Hessian is c d2(rhoA + rhoB)/dR2 exactly. The densities
// are general symmetric rather than the fixture's occupied-diagonal one, which
// is what makes the atom-pair contraction and the cross terms' row-and-column
// order visible, and the geometry is displaced so both the point's and the
// orbitals' motion carry weight.
TEST(XcGradientHessianTest, TheSyntheticLinearDensityHessianMatchesTheFieldEvaluator) {
    const auto fixture = MakeHessianFixture("slater");
    ASSERT_TRUE(fixture.has_value());

    const auto displaced = DisplacedMolecule(fixture->molecule, 0, 1, 0.13);
    ASSERT_TRUE(displaced.has_value()) << "the fixture must keep its atom order";

    auto engine = XcGridEngine::Create(
        *displaced, fixture->basis, fixture->functionalName, fixture->settings);
    ASSERT_TRUE(engine.has_value()) << engine.error().message;

    const auto geometry = ToExcgridGeometry(*displaced);
    ASSERT_TRUE(geometry.has_value());

    auto provider = excgrid::CreateGridDerivativeProvider(
        *geometry, ToExcgridParams(fixture->settings), excgrid::DerivativeOrder::kSecond);
    ASSERT_TRUE(provider.has_value());

    const std::size_t aoCount = engine->Evaluator().AOCount();
    Eigen::MatrixXd densityAlpha(static_cast<Eigen::Index>(aoCount),
                                 static_cast<Eigen::Index>(aoCount));
    Eigen::MatrixXd densityBeta(static_cast<Eigen::Index>(aoCount),
                                static_cast<Eigen::Index>(aoCount));

    for (std::size_t row = 0; row < aoCount; ++row)
    {
        for (std::size_t column = 0; column < aoCount; ++column)
        {
            const double i = static_cast<double>(row + 1);
            const double j = static_cast<double>(column + 1);
            densityAlpha(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
                0.05 * std::cos(0.7 * i * j);
            densityBeta(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) =
                -0.03 * std::sin(0.5 * i * j + 0.2);
        }
    }

    std::vector<std::size_t> atomOfAo(aoCount);

    for (const ShellEnvelope& shell : engine->ShellEnvelopes())
    {
        for (std::size_t ao = 0; ao < shell.functionCount; ++ao)
        {
            atomOfAo[shell.aoOffset + ao] = shell.atomIndex;
        }
    }

    constexpr double kPartial = 0.37;
    const LinearRhoEnergyFunctional functional(kPartial);
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);

    const auto hessian = EvaluateXcGridMotionHessian(engine->Grid().Blocks(),
                                                     **provider,
                                                     engine->Evaluator(),
                                                     functional,
                                                     engine->ShellEnvelopes(),
                                                     thresholds,
                                                     densityAlpha,
                                                     densityBeta);
    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    // The tolerance keeps every shell, so the test-local evaluator's dense walk
    // reads the same slots the screened walk kept.
    EXPECT_EQ(hessian->counts.aoSlotsEvaluated, hessian->counts.points * aoCount);

    const auto expected = ExpectedFixedDensityHessian(
        *engine, **provider, densityAlpha, densityBeta, kPartial, displaced->AtomCount(), atomOfAo);
    ASSERT_TRUE(expected.has_value());

    double maxDeviation = 0.0;
    double maxEntry = 0.0;

    for (Eigen::Index row = 0; row < expected->rows(); ++row)
    {
        for (Eigen::Index column = 0; column < expected->cols(); ++column)
        {
            maxDeviation = std::max(
                maxDeviation, std::abs(hessian->hessian(row, column) - (*expected)(row, column)));
            maxEntry = std::max(maxEntry, std::abs((*expected)(row, column)));
        }
    }

    ::testing::Test::RecordProperty(
        "synthetic linear-density Hessian agreement (max deviation, Hartree/Bohr^2)",
        Scientific(maxDeviation));
    EXPECT_GT(maxEntry, 1e-8) << "the pin must carry a visible Hessian";
    EXPECT_LT(maxDeviation, kSyntheticTolerance * (1.0 + maxEntry))
        << "the walk and the test-local field evaluator disagree";
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

    // A functional that reads a density is owed its second response, so the
    // zero tier is requested rather than skipped.
    EXPECT_GT(plain->counts.kernelSecondDerivativeCalls, 0U);

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

// A GGA reads the sigma invariants, whose second response differentiates the AO
// Hessian once more. The walk refuses by name instead of answering with the
// density channel alone, which would be a plausible wrong Hessian.
TEST(XcGradientHessianTest, AGgaFunctionalIsRefusedByComponentName) {
    const auto fixture = MakeHessianFixture("pbe");
    ASSERT_TRUE(fixture.has_value());

    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);
    const auto hessian = EvaluateXcGridMotionHessian(
        fixture->engine, fixture->density, fixture->density, *fixture->provider, thresholds);

    ASSERT_FALSE(hessian.has_value());
    EXPECT_EQ(hessian.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(hessian.error().message.find("sigmaAa"), std::string::npos)
        << hessian.error().message;
    EXPECT_NE(hessian.error().message.find("sigmaAb"), std::string::npos)
        << hessian.error().message;
    EXPECT_NE(hessian.error().message.find("sigmaBb"), std::string::npos)
        << hessian.error().message;
    EXPECT_NE(hessian.error().message.find("third-derivative"), std::string::npos)
        << "the refusal must say what would be needed, not only that it declined";
}

// The tier is how a functional says it can be differentiated twice. A
// density-reading functional that declares none is refused, which is the reason
// RhoFunctional carries a zero one.
TEST(XcGradientHessianTest, ADensityFunctionalWithoutATierIsRefused) {
    const auto fixture = MakeHessianFixture("slater");
    ASSERT_TRUE(fixture.has_value());

    const UntieredRhoFunctional functional(kRhoPartial);
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);

    const auto hessian = EvaluateXcGridMotionHessian(fixture->engine.Grid().Blocks(),
                                                     *fixture->provider,
                                                     fixture->engine.Evaluator(),
                                                     functional,
                                                     fixture->engine.ShellEnvelopes(),
                                                     thresholds,
                                                     fixture->density,
                                                     fixture->density);

    ASSERT_FALSE(hessian.has_value());
    EXPECT_EQ(hessian.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(hessian.error().message.find("no second-derivative tier"), std::string::npos)
        << hessian.error().message;
}

// A functional that reads nothing has no second response and no tier, and is
// answered rather than refused: the refusal belongs to functionals whose
// integrand depends on a field.
TEST(XcGradientHessianTest, AConstantIntegrandNeedsNoTier) {
    const auto fixture = MakeHessianFixture("slater");
    ASSERT_TRUE(fixture.has_value());

    const ConstantFunctional functional(kEnergyDensityValue);
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kScreeningTolerance);

    const auto hessian = EvaluateXcGridMotionHessian(fixture->engine.Grid().Blocks(),
                                                     *fixture->provider,
                                                     fixture->engine.Evaluator(),
                                                     functional,
                                                     fixture->engine.ShellEnvelopes(),
                                                     thresholds,
                                                     fixture->density,
                                                     fixture->density);

    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;
    EXPECT_EQ(hessian->counts.kernelSecondDerivativeCalls, 0U);
    EXPECT_EQ(hessian->counts.electronicCurvatureTerms, 0U);
}

} // namespace
} // namespace qcx::grid
