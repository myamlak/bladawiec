// The fixed-density XC Hessian's exact-condition pins: the second total
// responses of all seven collinear components, the kernel's contracted second
// derivatives, the two bookkeeping checks that name a component and a
// coordinate, the screening sweep at second order, and the refusal paths.
//
// The walk under test is EvaluateXcGridMotionHessian. At fixed density its
// electronic channel is
//     D_q D_r e = sum_i (de/du_i) D_q D_r u_i
//               + sum_ij (d2e/du_i du_j) (D_q u_i)(D_r u_j),
// with u = (rhoA, rhoB, sigmaAa, sigmaAb, sigmaBb, tauA, tauB). Every field's
// total derivative is computed here from the shift identity
//     D_q phi_mu     = S_q(mu) . grad phi_mu,
//     D_r D_q phi_mu = S_q(mu)^T hess phi_mu S_r(mu),
// with S_q(mu) = X_q - delta(atom(mu), atom(q)) e_direction(q): the point's own
// motion plus the AO centre's displacement. The identity carries one more
// spatial derivative for a gradient-valued field, which is where the AO's third
// derivative enters. The evaluator is exact, not an approximation, because the
// AO centre translates rigidly, and it shares no atom-block decomposition with
// the walk it pins.
//
// Every numeric pin is paired with a deliberately broken value computed
// alongside it. The correct value must pass, the broken one must fall outside
// the same bar, and the two must be separated at that bar: a check that has
// never been seen to fail is not evidence.

#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/memory/tensor.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <excgrid/grid.hpp>
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

// A coarse grid: the pins compare two evaluations of the same point set, so the
// point count changes no number and only the walk's cost.
constexpr std::size_t kRadialPoints = 8;
constexpr std::size_t kAngularPoints = 14;
constexpr std::size_t kBlockTarget = 128;

// The tolerance that keeps every shell at every point. A displaced geometry
// must not change the selection between the reference and the walk.
constexpr double kDenseScreeningTolerance = -1.0;

// The synthetic pins' bar. Both sides read the same AO tier and the same
// provider, so only the summation order differs; the measured residual is
// recorded on each case.
constexpr double kExactPinTolerance = 1e-10;

// Pins 4-5 and the bookkeeping checks. Both sides read the same AO tier and the
// same materialised second-partial matrix, so only the contraction's order
// differs; the bookkeeping's compared value is itself a difference of two
// Hessians and carries their cancellation, which is why this bar is looser than
// the synthetic pins'.
constexpr double kKernelPinTolerance = 1e-9;

// The position-curvature pin's bar: the same one the density channel's
// curvature test uses, since the term is the same chain rule.
constexpr double kPositionCurvatureTolerance = 1e-9;

// The screening sweep's coarse grid and the tolerances it walks. The list is
// fixed before the measurement: tuning it to make a deviation pass is the
// defect the gate exists to prevent.
constexpr std::size_t kScreeningRadialPoints = 20;
constexpr std::size_t kScreeningAngularPoints = 50;
constexpr double kNearDenseTolerance = 1e-12;
constexpr double kTightTolerance = 1e-10;
constexpr double kProductionTolerance = 1e-8;

std::string Scientific(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6e", value);

    return buffer;
}

// One AO Hessian entry in the evaluator's packed six (xx, xy, xz, yy, yz, zz).
double PackedHessian(std::span<const double> hessians,
                     std::size_t ao,
                     std::size_t row,
                     std::size_t column) {
    constexpr std::size_t kSlots[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return hessians[6 * ao + kSlots[row][column]];
}

// One AO third-derivative entry in the evaluator's packed ten
// (xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz).
double PackedThird(std::span<const double> thirds,
                   std::size_t ao,
                   std::size_t first,
                   std::size_t second,
                   std::size_t third) {
    constexpr std::size_t kSlots[3][3][3] = {{{0, 1, 2}, {1, 3, 4}, {2, 4, 5}},
                                             {{1, 3, 4}, {3, 6, 7}, {4, 7, 8}},
                                             {{2, 4, 5}, {4, 7, 8}, {5, 8, 9}}};
    return thirds[10 * ao + kSlots[first][second][third]];
}

// The upper-triangle index the second-derivative contract uses over `active`
// components: (i, j), i <= j, at i*active - i*(i-1)/2 + (j-i).
std::size_t UpperIndex(std::size_t row, std::size_t column, std::size_t active) {
    return row * (2 * active - row + 1) / 2 + (column - row);
}

// The prescribed position curvature: one distinct value per (direction, row,
// column), symmetric in the last two so the walk's upper-triangle mirror is
// exact. Shared by the provider and the test that integrates it.
double PrescribedPositionCurvature(double scale,
                                   std::size_t direction,
                                   std::size_t row,
                                   std::size_t column) {
    return scale * static_cast<double>((direction + 1) * (row + 1) * (column + 1));
}

// A provider that delegates to the real one and then overwrites the position
// curvature. The default grid rides each point on its owning atom, so its
// position curvature is identically zero and the carry-through has no other way
// to be observed.
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

// A row-major copy of a density matrix, the layout ShellDensityWeights reads.
std::vector<double> RowMajorCopy(const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
    std::vector<double> flat(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
        flat.data(), n, n);
    map = density;

    return flat;
}

// ---------------------------------------------------------------------------
// The test-local functionals
// ---------------------------------------------------------------------------

// What one functional records, so a case can name the component, the
// coordinate and the side whose value was wrong rather than only the Hessian
// entry that moved.
struct FunctionalRecording {
    std::vector<excgrid::PointInputs> pointInputs;
    std::vector<excgrid::PointInputs> curvatureInputs;
    std::vector<std::vector<double>> curvatureRightHandSides;
};

// e(u) = sum_i linear_i u_i + 1/2 sum_ij quadratic_ij u_i u_j over the seven
// collinear components. The first derivative is linear + quadratic u; the
// second is quadratic. `tier` is what the second-derivative tier answers, which
// may be a different matrix from `quadratic`: the bookkeeping check isolates the
// walk's kernel contraction by comparing two functional objects whose first-order
// tiers are identical and whose second-derivative tiers differ.
struct PolynomialSpec {
    std::array<double, 7> linear{};
    std::array<std::array<double, 7>, 7> quadratic{};
    std::array<std::array<double, 7>, 7> tier{};
    excgrid::ComponentMask required{};
    excgrid::ComponentMask tierMask{};
    FunctionalRecording* recording = nullptr;
};

class PolynomialFunctional final : public excgrid::XcFunctional {
public:
    explicit PolynomialFunctional(PolynomialSpec spec) : _spec(spec) {}

    [[nodiscard]] bool UsesGradient() const override {
        return _spec.required.Test(excgrid::Component::SigmaAa) ||
               _spec.required.Test(excgrid::Component::SigmaAb) ||
               _spec.required.Test(excgrid::Component::SigmaBb);
    }

    [[nodiscard]] double ExchangeFraction() const override {
        return 0.0;
    }

    [[nodiscard]] excgrid::ComponentMask RequiredMask() const override {
        return _spec.required;
    }

    [[nodiscard]] excgrid::ComponentMask SecondDerivativeMask() const noexcept override {
        return _spec.tierMask;
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePoint(const excgrid::PointInputs& inputs,
                                                      excgrid::PointResult& result) const override {
        if (_spec.recording != nullptr)
        {
            _spec.recording->pointInputs.push_back(inputs);
        }

        FillFirstOrder(inputs, result);

        return excgrid::KernelStatus::kOk;
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePointMaterialising(
        const excgrid::PointInputs& inputs,
        excgrid::PointResult& result,
        excgrid::PointSecondDerivativeMatrix& matrix) const override {
        FillFirstOrder(inputs, result);

        matrix = excgrid::PointSecondDerivativeMatrix{};
        matrix.mask = ActiveMask(inputs);

        std::array<std::size_t, 7> slots{};
        const std::size_t activeCount = ActiveSlots(matrix.mask, slots);

        if (activeCount == 0)
        {
            return excgrid::KernelStatus::kRefusedUnsupportedCombination;
        }

        for (std::size_t row = 0; row < activeCount; ++row)
        {
            for (std::size_t column = row; column < activeCount; ++column)
            {
                matrix.upper[UpperIndex(row, column, activeCount)] =
                    _spec.tier[slots[row]][slots[column]];
            }
        }

        return excgrid::KernelStatus::kOk;
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePointWithSecondDerivatives(
        const excgrid::PointInputs& inputs,
        std::span<const double> rightHandSide,
        excgrid::PointResult& result,
        excgrid::PointSecondDerivative& second) const override {
        if (_spec.recording != nullptr)
        {
            _spec.recording->curvatureInputs.push_back(inputs);
            _spec.recording->curvatureRightHandSides.emplace_back(rightHandSide.begin(),
                                                                  rightHandSide.end());
        }

        const excgrid::ComponentMask active = ActiveMask(inputs);
        std::array<std::size_t, 7> slots{};
        const std::size_t activeCount = ActiveSlots(active, slots);

        if (activeCount == 0 || rightHandSide.size() % activeCount != 0 ||
            rightHandSide.size() > excgrid::kSecondDerivativeCapacity)
        {
            return excgrid::KernelStatus::kRefusedUnsupportedCombination;
        }

        FillFirstOrder(inputs, result);

        second = excgrid::PointSecondDerivative{};
        second.mask = active;
        second.rightHandSides = rightHandSide.size() / activeCount;

        // The documented layout: the component varies fastest, and the returned
        // entry is F times the caller's side.
        for (std::size_t side = 0; side < second.rightHandSides; ++side)
        {
            for (std::size_t row = 0; row < activeCount; ++row)
            {
                double value = 0.0;

                for (std::size_t column = 0; column < activeCount; ++column)
                {
                    value += _spec.tier[slots[row]][slots[column]] *
                             rightHandSide[side * activeCount + column];
                }

                second.contracted[side * activeCount + row] = value;
            }
        }

        return excgrid::KernelStatus::kOk;
    }

private:
    [[nodiscard]] excgrid::ComponentMask ActiveMask(const excgrid::PointInputs& inputs) const {
        return excgrid::ComponentMask{inputs.mask.bits & _spec.required.bits};
    }

    static std::size_t ActiveSlots(const excgrid::ComponentMask& mask,
                                   std::array<std::size_t, 7>& slots) {
        std::size_t count = 0;

        for (std::size_t index = 0; index < 7; ++index)
        {
            if (mask.Test(static_cast<excgrid::Component>(index)))
            {
                slots[count++] = index;
            }
        }

        return count;
    }

    void FillFirstOrder(const excgrid::PointInputs& inputs, excgrid::PointResult& result) const {
        std::array<double, 7> u{};

        for (std::size_t index = 0; index < 7; ++index)
        {
            u[index] = inputs.Get(static_cast<excgrid::Component>(index));
        }

        result.first = {};
        double energy = 0.0;

        for (std::size_t index = 0; index < 7; ++index)
        {
            if (!_spec.required.Test(static_cast<excgrid::Component>(index)))
            {
                continue;
            }

            double first = _spec.linear[index];

            for (std::size_t other = 0; other < 7; ++other)
            {
                first += _spec.quadratic[index][other] * u[other];
            }

            result.first[index] = first;
            energy += _spec.linear[index] * u[index];

            for (std::size_t other = 0; other < 7; ++other)
            {
                if (_spec.required.Test(static_cast<excgrid::Component>(other)))
                {
                    energy += 0.5 * _spec.quadratic[index][other] * u[index] * u[other];
                }
            }
        }

        result.exc = energy;
        result.mask = inputs.mask;
    }

    PolynomialSpec _spec;
};

// ---------------------------------------------------------------------------
// The brute-force field evaluator
// ---------------------------------------------------------------------------

// The knobs a deliberately broken case needs. Every knob acts on the fields'
// second total responses only, so the energy, the first responses and the
// weight terms of the broken value stay correct and the difference isolates the
// defect the knob names.
struct InstrumentOptions {
    /// Whether the AO third-derivative tier contributes. Clearing the thirds is
    /// exactly the second response a walk without the tier would compute.
    bool thirdTier = true;
    /// The sign of the pure orbital part of every second total response. The
    /// correct value is +1; the tau pin's broken case writes it at -1.
    double orbitalSecondSign = 1.0;
    /// Whether the kernel's contracted second partials enter.
    bool kernelContraction = true;
    /// Whether the second side's component order is reversed, which is the
    /// layout error the mixed-partial pin and the bookkeeping check break.
    bool transposeSecondSide = false;
    /// A scale on the sigma_ab partial inside the field second responses only:
    /// the cross pair's doubled weight, with the energy and first responses left
    /// correct.
    double crossPairResponseScale = 1.0;
};

// The seven components at one point, in identifier order, with their spatial
// gradients and their first and second total responses.
struct FieldResponses {
    std::array<double, 7> value{};
    std::array<std::array<double, 3>, 7> spatialGradient{};
    std::vector<std::array<double, 7>> first; ///< Per nuclear coordinate, D_q u.
    std::vector<std::array<double, 7>> second; ///< Per pair, row-major, D_qD_r u.
};

// One spin's field and its total responses, built from the double sums of the
// definitions rather than from the walk's factored contractions.
struct SpinField {
    double rho = 0.0;
    double tau = 0.0;
    std::array<double, 3> gradient{};
    std::array<double, 9> hessian{}; ///< Row-major spatial Hessian of rho.
    std::array<double, 3> tauGradient{};
    std::vector<double> rhoFirst;
    std::vector<double> rhoSecond;
    std::array<std::vector<double>, 3> gradientFirst; ///< [direction][coordinate].
    std::array<std::vector<double>, 3> gradientSecond; ///< [direction][pair].
    std::vector<double> tauFirst;
    std::vector<double> tauSecond;
};

FieldResponses EvaluateFieldResponses(const AoEvaluator& evaluator,
                                      std::span<const std::size_t> atomOfAo,
                                      const Eigen::MatrixXd& densityAlpha,
                                      const Eigen::MatrixXd& densityBeta,
                                      std::span<const std::array<double, 3>> positionFirst,
                                      const std::array<double, 3>& point,
                                      const InstrumentOptions& options) {
    const std::size_t aoCount = evaluator.AOCount();
    const std::size_t coordinates = positionFirst.size();

    std::vector<double> values(aoCount);
    std::vector<double> gradients(3 * aoCount);
    std::vector<double> hessians(6 * aoCount);
    std::vector<double> thirds(10 * aoCount);
    evaluator.EvaluateDerivativesToThird(point, values, gradients, hessians, thirds);

    if (!options.thirdTier)
    {
        std::fill(thirds.begin(), thirds.end(), 0.0);
    }

    // The two shift vectors of (2), per nuclear coordinate and AO. The orbital
    // part is kept apart because one broken case needs its sign.
    std::vector<std::array<double, 3>> shifts(coordinates * aoCount);
    std::vector<std::array<double, 3>> orbitalShifts(coordinates * aoCount);

    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
    {
        for (std::size_t ao = 0; ao < aoCount; ++ao)
        {
            std::array<double, 3> orbital{};

            if (atomOfAo[ao] == coordinate / 3)
            {
                orbital[coordinate % 3] = -1.0;
            }

            std::array<double, 3> shift = positionFirst[coordinate];

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                shift[axis] += orbital[axis];
            }

            shifts[coordinate * aoCount + ao] = shift;
            orbitalShifts[coordinate * aoCount + ao] = orbital;
        }
    }

    // The AO value's Hessian and one component of the AO gradient's Hessian,
    // each contracted with two directions.
    const auto valueForm =
        [&](std::size_t ao, const std::array<double, 3>& left, const std::array<double, 3>& right) {
            double value = 0.0;

            for (std::size_t row = 0; row < 3; ++row)
            {
                for (std::size_t column = 0; column < 3; ++column)
                {
                    value += left[row] * PackedHessian(hessians, ao, row, column) * right[column];
                }
            }

            return value;
        };

    const auto gradientForm = [&](std::size_t ao,
                                  std::size_t axis,
                                  const std::array<double, 3>& left,
                                  const std::array<double, 3>& right) {
        double value = 0.0;

        for (std::size_t row = 0; row < 3; ++row)
        {
            for (std::size_t column = 0; column < 3; ++column)
            {
                value += left[row] * PackedThird(thirds, ao, axis, row, column) * right[column];
            }
        }

        return value;
    };

    // First and second total responses of every AO quantity the fields read.
    std::vector<double> valueFirst(coordinates * aoCount);
    std::vector<std::array<double, 3>> gradientFirst(coordinates * aoCount);

    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
    {
        for (std::size_t ao = 0; ao < aoCount; ++ao)
        {
            const std::array<double, 3>& shift = shifts[coordinate * aoCount + ao];
            double value = 0.0;

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                value += shift[axis] * gradients[3 * ao + axis];
                gradientFirst[coordinate * aoCount + ao][axis] = 0.0;

                for (std::size_t other = 0; other < 3; ++other)
                {
                    gradientFirst[coordinate * aoCount + ao][axis] +=
                        shift[other] * PackedHessian(hessians, ao, axis, other);
                }
            }

            valueFirst[coordinate * aoCount + ao] = value;
        }
    }

    // The second total responses, upper triangle only: every reader of this
    // instrument takes D_qD_r with q <= r and the walk mirrors its own triangle,
    // so the lower triangle is left at zero. With the shift written as X + O,
    //     S_q^T T S_r = X_q^T T X_r + X_q^T T O_r + O_q^T T X_r + O_q^T T O_r,
    // and the last term is the pure orbital second response whose sign the tau
    // pin's broken case flips.
    const double orbitalCorrection = options.orbitalSecondSign - 1.0;
    std::vector<double> valueSecond(coordinates * coordinates * aoCount);
    std::vector<std::array<double, 3>> gradientSecond(coordinates * coordinates * aoCount);

    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
    {
        for (std::size_t other = coordinate; other < coordinates; ++other)
        {
            for (std::size_t ao = 0; ao < aoCount; ++ao)
            {
                const std::size_t index = (coordinate * coordinates + other) * aoCount + ao;
                const std::array<double, 3>& shiftLeft = shifts[coordinate * aoCount + ao];
                const std::array<double, 3>& shiftRight = shifts[other * aoCount + ao];
                const std::array<double, 3>& orbitalLeft = orbitalShifts[coordinate * aoCount + ao];
                const std::array<double, 3>& orbitalRight = orbitalShifts[other * aoCount + ao];

                valueSecond[index] = valueForm(ao, shiftLeft, shiftRight) +
                                     orbitalCorrection * valueForm(ao, orbitalLeft, orbitalRight);

                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    gradientSecond[index][axis] =
                        gradientForm(ao, axis, shiftLeft, shiftRight) +
                        orbitalCorrection * gradientForm(ao, axis, orbitalLeft, orbitalRight);
                }
            }
        }
    }

    const auto evaluateSpin = [&](const Eigen::MatrixXd& density) {
        SpinField field;
        field.rhoFirst.assign(coordinates, 0.0);
        field.rhoSecond.assign(coordinates * coordinates, 0.0);
        field.tauFirst.assign(coordinates, 0.0);
        field.tauSecond.assign(coordinates * coordinates, 0.0);

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            field.gradientFirst[axis].assign(coordinates, 0.0);
            field.gradientSecond[axis].assign(coordinates * coordinates, 0.0);
        }

        for (std::size_t mu = 0; mu < aoCount; ++mu)
        {
            for (std::size_t nu = 0; nu < aoCount; ++nu)
            {
                const double element =
                    density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));

                if (element == 0.0)
                {
                    continue;
                }

                field.rho += element * values[mu] * values[nu];

                for (std::size_t axis = 0; axis < 3; ++axis)
                {
                    field.gradient[axis] += element * (gradients[3 * mu + axis] * values[nu] +
                                                       values[mu] * gradients[3 * nu + axis]);

                    for (std::size_t column = 0; column < 3; ++column)
                    {
                        field.hessian[3 * axis + column] +=
                            element * (PackedHessian(hessians, mu, axis, column) * values[nu] +
                                       gradients[3 * mu + axis] * gradients[3 * nu + column] +
                                       gradients[3 * mu + column] * gradients[3 * nu + axis] +
                                       values[mu] * PackedHessian(hessians, nu, axis, column));
                    }

                    field.tau +=
                        0.5 * element * gradients[3 * mu + axis] * gradients[3 * nu + axis];

                    for (std::size_t column = 0; column < 3; ++column)
                    {
                        field.tauGradient[axis] += 0.5 * element *
                                                   (PackedHessian(hessians, mu, column, axis) *
                                                        gradients[3 * nu + column] +
                                                    gradients[3 * mu + column] *
                                                        PackedHessian(hessians, nu, column, axis));
                    }
                }
            }
        }

        for (std::size_t mu = 0; mu < aoCount; ++mu)
        {
            for (std::size_t nu = 0; nu < aoCount; ++nu)
            {
                const double element =
                    density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));

                if (element == 0.0)
                {
                    continue;
                }

                for (std::size_t q = 0; q < coordinates; ++q)
                {
                    const std::size_t qMu = q * aoCount + mu;
                    const std::size_t qNu = q * aoCount + nu;
                    field.rhoFirst[q] +=
                        element * (valueFirst[qMu] * values[nu] + values[mu] * valueFirst[qNu]);

                    double tauFirst = 0.0;

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        field.gradientFirst[axis][q] +=
                            element * (gradientFirst[qMu][axis] * values[nu] +
                                       gradients[3 * mu + axis] * valueFirst[qNu] +
                                       valueFirst[qMu] * gradients[3 * nu + axis] +
                                       values[mu] * gradientFirst[qNu][axis]);
                        tauFirst += gradientFirst[qMu][axis] * gradients[3 * nu + axis] +
                                    gradients[3 * mu + axis] * gradientFirst[qNu][axis];
                    }

                    field.tauFirst[q] += 0.5 * element * tauFirst;
                }

                for (std::size_t q = 0; q < coordinates; ++q)
                {
                    for (std::size_t r = q; r < coordinates; ++r)
                    {
                        const std::size_t pair = q * coordinates + r;
                        const std::size_t secondMu = pair * aoCount + mu;
                        const std::size_t secondNu = pair * aoCount + nu;
                        const std::size_t qMu = q * aoCount + mu;
                        const std::size_t qNu = q * aoCount + nu;
                        const std::size_t rMu = r * aoCount + mu;
                        const std::size_t rNu = r * aoCount + nu;

                        field.rhoSecond[pair] += element * (valueSecond[secondMu] * values[nu] +
                                                            valueFirst[qMu] * valueFirst[rNu] +
                                                            valueFirst[rMu] * valueFirst[qNu] +
                                                            values[mu] * valueSecond[secondNu]);

                        double tauSecond = 0.0;

                        for (std::size_t axis = 0; axis < 3; ++axis)
                        {
                            field.gradientSecond[axis][pair] +=
                                element * (gradientSecond[secondMu][axis] * values[nu] +
                                           gradientFirst[qMu][axis] * valueFirst[rNu] +
                                           gradientFirst[rMu][axis] * valueFirst[qNu] +
                                           gradients[3 * mu + axis] * valueSecond[secondNu] +
                                           valueSecond[secondMu] * gradients[3 * nu + axis] +
                                           valueFirst[qMu] * gradientFirst[rNu][axis] +
                                           valueFirst[rMu] * gradientFirst[qNu][axis] +
                                           values[mu] * gradientSecond[secondNu][axis]);

                            tauSecond += gradientSecond[secondMu][axis] * gradients[3 * nu + axis] +
                                         gradientFirst[qMu][axis] * gradientFirst[rNu][axis] +
                                         gradientFirst[rMu][axis] * gradientFirst[qNu][axis] +
                                         gradients[3 * mu + axis] * gradientSecond[secondNu][axis];
                        }

                        field.tauSecond[pair] += 0.5 * element * tauSecond;
                    }
                }
            }
        }

        return field;
    };

    const SpinField alpha = evaluateSpin(densityAlpha);
    const SpinField beta = evaluateSpin(densityBeta);

    FieldResponses responses;
    responses.first.assign(coordinates, std::array<double, 7>{});
    responses.second.assign(coordinates * coordinates, std::array<double, 7>{});

    responses.value[0] = alpha.rho;
    responses.value[1] = beta.rho;
    responses.value[2] = alpha.gradient[0] * alpha.gradient[0] +
                         alpha.gradient[1] * alpha.gradient[1] +
                         alpha.gradient[2] * alpha.gradient[2];
    responses.value[3] = alpha.gradient[0] * beta.gradient[0] +
                         alpha.gradient[1] * beta.gradient[1] +
                         alpha.gradient[2] * beta.gradient[2];
    responses.value[4] = beta.gradient[0] * beta.gradient[0] + beta.gradient[1] * beta.gradient[1] +
                         beta.gradient[2] * beta.gradient[2];
    responses.value[5] = alpha.tau;
    responses.value[6] = beta.tau;

    responses.spatialGradient[0] = alpha.gradient;
    responses.spatialGradient[1] = beta.gradient;
    responses.spatialGradient[5] = alpha.tauGradient;
    responses.spatialGradient[6] = beta.tauGradient;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        double sigmaAa = 0.0;
        double sigmaAb = 0.0;
        double sigmaBb = 0.0;

        for (std::size_t column = 0; column < 3; ++column)
        {
            sigmaAa += alpha.hessian[3 * axis + column] * alpha.gradient[column];
            sigmaAb += alpha.hessian[3 * axis + column] * beta.gradient[column] +
                       beta.hessian[3 * axis + column] * alpha.gradient[column];
            sigmaBb += beta.hessian[3 * axis + column] * beta.gradient[column];
        }

        responses.spatialGradient[2][axis] = 2.0 * sigmaAa;
        responses.spatialGradient[3][axis] = sigmaAb;
        responses.spatialGradient[4][axis] = 2.0 * sigmaBb;
    }

    for (std::size_t q = 0; q < coordinates; ++q)
    {
        double sigmaAa = 0.0;
        double sigmaAb = 0.0;
        double sigmaBb = 0.0;

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            sigmaAa += alpha.gradient[axis] * alpha.gradientFirst[axis][q];
            sigmaAb += alpha.gradientFirst[axis][q] * beta.gradient[axis] +
                       alpha.gradient[axis] * beta.gradientFirst[axis][q];
            sigmaBb += beta.gradient[axis] * beta.gradientFirst[axis][q];
        }

        responses.first[q][0] = alpha.rhoFirst[q];
        responses.first[q][1] = beta.rhoFirst[q];
        responses.first[q][2] = 2.0 * sigmaAa;
        responses.first[q][3] = sigmaAb;
        responses.first[q][4] = 2.0 * sigmaBb;
        responses.first[q][5] = alpha.tauFirst[q];
        responses.first[q][6] = beta.tauFirst[q];
    }

    for (std::size_t q = 0; q < coordinates; ++q)
    {
        for (std::size_t r = 0; r < coordinates; ++r)
        {
            const std::size_t pair = q * coordinates + r;
            double sigmaAa = 0.0;
            double sigmaAb = 0.0;
            double sigmaBb = 0.0;

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                // The product rule over two fields: two second-response terms
                // and two first-response products. The diagonal pairs carry the
                // factor two outside the sum; the cross pair does not.
                sigmaAa += alpha.gradient[axis] * alpha.gradientSecond[axis][pair] +
                           alpha.gradientFirst[axis][q] * alpha.gradientFirst[axis][r];
                sigmaAb += alpha.gradient[axis] * beta.gradientSecond[axis][pair] +
                           alpha.gradientSecond[axis][pair] * beta.gradient[axis] +
                           alpha.gradientFirst[axis][q] * beta.gradientFirst[axis][r] +
                           beta.gradientFirst[axis][q] * alpha.gradientFirst[axis][r];
                sigmaBb += beta.gradient[axis] * beta.gradientSecond[axis][pair] +
                           beta.gradientFirst[axis][q] * beta.gradientFirst[axis][r];
            }

            responses.second[pair][0] = alpha.rhoSecond[pair];
            responses.second[pair][1] = beta.rhoSecond[pair];
            responses.second[pair][2] = 2.0 * sigmaAa;
            responses.second[pair][3] = sigmaAb;
            responses.second[pair][4] = 2.0 * sigmaBb;
            responses.second[pair][5] = alpha.tauSecond[pair];
            responses.second[pair][6] = beta.tauSecond[pair];
        }
    }

    return responses;
}

// ---------------------------------------------------------------------------
// The fixtures
// ---------------------------------------------------------------------------

struct PinFixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    XcGridSettings settings;
    XcGridEngine engine;
    std::unique_ptr<excgrid::GridDerivativeProvider> provider;
    Eigen::MatrixXd densityAlpha; ///< General symmetric, so every pair is visible.
    Eigen::MatrixXd densityBeta;
    std::vector<std::size_t> atomOfAo;
};

std::optional<PinFixture> MakePinFixture() {
    auto basis = qcx::testing::MakeH2oSto3gBasis();

    if (!basis.has_value())
    {
        return std::nullopt;
    }

    auto reference = qcx::testing::MakeH2oSto3g();

    if (!reference.has_value())
    {
        return std::nullopt;
    }

    // A displaced geometry carries both the point's and the orbitals' motion,
    // which is what makes the cross terms and the atom-pair sums visible.
    auto molecule = DisplacedMolecule(*reference, 0, 1, 0.13);

    if (!molecule.has_value())
    {
        return std::nullopt;
    }

    XcGridSettings settings;
    settings.radialPoints = kRadialPoints;
    settings.angularPoints = kAngularPoints;
    settings.blockTarget = kBlockTarget;

    auto engine = XcGridEngine::Create(*molecule, *basis, "slater", settings);

    if (!engine.has_value())
    {
        return std::nullopt;
    }

    auto geometry = ToExcgridGeometry(*molecule);

    if (!geometry.has_value())
    {
        return std::nullopt;
    }

    auto provider = excgrid::CreateGridDerivativeProvider(
        *geometry, ToExcgridParams(settings), excgrid::DerivativeOrder::kSecond);

    if (!provider.has_value())
    {
        return std::nullopt;
    }

    const std::size_t aoCount = engine->AOCount();
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

    return PinFixture{std::move(*molecule),
                      std::move(*basis),
                      settings,
                      std::move(*engine),
                      std::move(*provider),
                      std::move(densityAlpha),
                      std::move(densityBeta),
                      std::move(atomOfAo)};
}

qcx::Result<XcGridMotionHessian> RunWalk(const PinFixture& fixture,
                                         const excgrid::XcFunctional& functional) {
    return EvaluateXcGridMotionHessian(
        fixture.engine.Grid().Blocks(),
        *fixture.provider,
        fixture.engine.Evaluator(),
        functional,
        fixture.engine.ShellEnvelopes(),
        XcIntegrationThresholds::FromEnergyPath(fixture.settings, kDenseScreeningTolerance),
        fixture.densityAlpha,
        fixture.densityBeta);
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

// The expected Hessian, from the field evaluator and the functional's own
// partials:
//     H_qr = sum_p [ W_,qr e + W_,q T_r + W_,r T_q
//                    + W (de/dx) . X^c_,qr + W M_qr ],
//     M_qr = sum_i (de/du_i) D_qD_r u_i
//          + sum_ij (d2e/du_i du_j) (D_q u_i)(D_r u_j).
// The second partials are read from the functional's materialising tier, so the
// kernel's own packing and the walk's right-hand side layout are what the
// comparison sees, not a second copy of either.
std::optional<Eigen::MatrixXd> ExpectedHessian(const PinFixture& fixture,
                                               const excgrid::XcFunctional& functional,
                                               const InstrumentOptions& options) {
    std::size_t atomCount = 0;

    for (const ShellEnvelope& shell : fixture.engine.ShellEnvelopes())
    {
        atomCount = std::max(atomCount, shell.atomIndex + 1);
    }

    if (atomCount == 0)
    {
        return std::nullopt;
    }

    const std::size_t coordinates = 3 * atomCount;
    const excgrid::DerivativeRequest request = SecondOrderRequest(coordinates);
    std::size_t largestBlock = 0;

    for (const excgrid::Block& block : fixture.engine.Grid().Blocks())
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

    for (const excgrid::Block& block : fixture.engine.Grid().Blocks())
    {
        excgrid::BlockDerivatives derivatives;
        derivatives.positionFirst = std::span<double>(positionFirst.data(), sizes.positionFirst);
        derivatives.weightFirst = std::span<double>(weightFirst.data(), sizes.weightFirst);
        derivatives.positionSecond = std::span<double>(positionSecond.data(), sizes.positionSecond);
        derivatives.weightSecond = std::span<double>(weightSecond.data(), sizes.weightSecond);
        derivatives.atomIndex = block.atomIndex;

        if (fixture.provider->Evaluate(block, request, derivatives) !=
            excgrid::DerivativeStatus::kOk)
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

            const FieldResponses fields = EvaluateFieldResponses(fixture.engine.Evaluator(),
                                                                 fixture.atomOfAo,
                                                                 fixture.densityAlpha,
                                                                 fixture.densityBeta,
                                                                 xFirst,
                                                                 block.points[point],
                                                                 options);

            excgrid::PointInputs inputs;

            for (std::size_t index = 0; index < 7; ++index)
            {
                inputs.Set(static_cast<excgrid::Component>(index), fields.value[index]);
            }

            excgrid::PointResult result;

            if (functional.EvaluatePoint(inputs, result) != excgrid::KernelStatus::kOk)
            {
                return std::nullopt;
            }

            excgrid::PointSecondDerivativeMatrix matrix;

            if (functional.EvaluatePointMaterialising(inputs, result, matrix) !=
                excgrid::KernelStatus::kOk)
            {
                return std::nullopt;
            }

            const excgrid::ComponentMask required = functional.RequiredMask();

            if (matrix.mask.bits != required.bits)
            {
                return std::nullopt;
            }

            std::array<std::size_t, 7> active{};
            std::size_t activeCount = 0;

            for (std::size_t index = 0; index < 7; ++index)
            {
                if (required.Test(static_cast<excgrid::Component>(index)))
                {
                    active[activeCount++] = index;
                }
            }

            if (activeCount == 0)
            {
                return std::nullopt;
            }

            std::array<double, 7> partial{};
            std::array<double, 7> secondPartial{};
            std::array<std::array<double, 7>, 7> curvature{};

            for (std::size_t slot = 0; slot < activeCount; ++slot)
            {
                partial[active[slot]] = result.first[active[slot]];
                secondPartial[active[slot]] = result.first[active[slot]];
            }

            secondPartial[static_cast<std::size_t>(excgrid::Component::SigmaAb)] *=
                options.crossPairResponseScale;

            for (std::size_t row = 0; row < activeCount; ++row)
            {
                for (std::size_t column = row; column < activeCount; ++column)
                {
                    const double value = matrix.upper[UpperIndex(row, column, activeCount)];
                    curvature[active[row]][active[column]] = value;
                    curvature[active[column]][active[row]] = value;
                }
            }

            std::array<double, 3> spatial{};
            std::fill(firstResponse.begin(), firstResponse.end(), 0.0);

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                for (std::size_t index = 0; index < 7; ++index)
                {
                    spatial[axis] += partial[index] * fields.spatialGradient[index][axis];
                }
            }

            for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
            {
                for (std::size_t index = 0; index < 7; ++index)
                {
                    firstResponse[coordinate] += partial[index] * fields.first[coordinate][index];
                }
            }

            const double* weightRow = weightFirst.data() + point * coordinates;
            const double* weightCurvature = weightSecond.data() + point * coordinates * coordinates;
            const double* positionCurvature =
                positionSecond.data() + point * 3 * coordinates * coordinates;

            for (std::size_t q = 0; q < coordinates; ++q)
            {
                for (std::size_t r = q; r < coordinates; ++r)
                {
                    double secondResponse = 0.0;

                    for (std::size_t index = 0; index < 7; ++index)
                    {
                        secondResponse +=
                            secondPartial[index] * fields.second[q * coordinates + r][index];
                    }

                    if (options.kernelContraction)
                    {
                        for (std::size_t row = 0; row < activeCount; ++row)
                        {
                            for (std::size_t column = 0; column < activeCount; ++column)
                            {
                                // The matrix is always indexed by the active
                                // order's identifiers. The broken case reads the
                                // second side's response in the reversed order,
                                // which is what a right-hand-side layout error
                                // does: reversing the matrix index as well would
                                // only re-order the same sum.
                                const std::size_t left = active[row];
                                const std::size_t right = active[column];
                                const std::size_t response = options.transposeSecondSide
                                                                 ? active[activeCount - 1 - column]
                                                                 : right;
                                secondResponse += curvature[left][right] * fields.first[q][left] *
                                                  fields.first[r][response];
                            }
                        }
                    }

                    double value = weightCurvature[q * coordinates + r] * result.exc +
                                   weightRow[q] * firstResponse[r] +
                                   weightRow[r] * firstResponse[q];

                    for (std::size_t axis = 0; axis < 3; ++axis)
                    {
                        value += weight * spatial[axis] *
                                 positionCurvature[axis * coordinates * coordinates +
                                                   q * coordinates + r];
                    }

                    value += weight * secondResponse;

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

// The pin's two-sided assertion: the walk must match the expected value inside
// the bar, the deliberate defect must move the expected value outside it, and
// the walk must therefore be separated from the defect by that same bar.
void ExpectWalkMatches(std::string_view label,
                       const Eigen::MatrixXd& walk,
                       const Eigen::MatrixXd& expected,
                       const Eigen::MatrixXd& broken,
                       double tolerance) {
    const double scale = 1.0 + expected.cwiseAbs().maxCoeff();
    const double bar = tolerance * scale;
    const double deviation = (walk - expected).cwiseAbs().maxCoeff();
    const double margin = (expected - broken).cwiseAbs().maxCoeff();
    const double separation = (walk - broken).cwiseAbs().maxCoeff();

    ::testing::Test::RecordProperty(std::string(label) + ": walk against expected",
                                    Scientific(deviation));
    ::testing::Test::RecordProperty(std::string(label) + ": expected against broken",
                                    Scientific(margin));
    ::testing::Test::RecordProperty(std::string(label) + ": walk against broken",
                                    Scientific(separation));

    EXPECT_GT(expected.cwiseAbs().maxCoeff(), 1e-8) << label << ": the pin carries no Hessian";
    EXPECT_LT(deviation, bar) << label << ": the walk and the field evaluator disagree";
    EXPECT_GT(margin, bar)
        << label << ": the deliberate defect is below the pin's own bar, so it cannot be seen";
    EXPECT_GT(separation, bar) << label << ": the walk does not separate the deliberate defect";
}

// ---------------------------------------------------------------------------
// Pins 1-3: the field second responses alone
// ---------------------------------------------------------------------------

// e = c sigma_aa. The sigma second response differentiates the AO Hessian once
// more, and the term it adds is the third-derivative tier's whole contribution.
// Clearing the thirds is exactly the value a walk without that tier would
// produce.
TEST(XcHessianExactnessTest, TheSigmaSecondResponseNeedsTheThirdTier) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    constexpr double kScale = 0.31;

    PolynomialSpec spec;
    spec.linear[static_cast<std::size_t>(excgrid::Component::SigmaAa)] = kScale;
    spec.required.Set(excgrid::Component::SigmaAa);
    spec.tierMask = spec.required;

    const PolynomialFunctional functional(spec);
    const auto walk = RunWalk(*fixture, functional);
    ASSERT_TRUE(walk.has_value()) << walk.error().message;

    const auto expected = ExpectedHessian(*fixture, functional, InstrumentOptions{});
    ASSERT_TRUE(expected.has_value()) << "the field evaluator refused the pin";

    InstrumentOptions withoutThirds;
    withoutThirds.thirdTier = false;
    const auto broken = ExpectedHessian(*fixture, functional, withoutThirds);
    ASSERT_TRUE(broken.has_value());

    ExpectWalkMatches(
        "sigma second response", walk->hessian, *expected, *broken, kExactPinTolerance);
}

// e = c tau_a. The first response is -S, but the pure second orbital response
// is +T2: the sign comes from the second derivative of a translation. Writing
// it at -1 flips that half.
TEST(XcHessianExactnessTest, TheTauSecondResponsePinsItsOrbitalSign) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    constexpr double kScale = 0.23;

    PolynomialSpec spec;
    spec.linear[static_cast<std::size_t>(excgrid::Component::TauA)] = kScale;
    spec.required.Set(excgrid::Component::TauA);
    spec.tierMask = spec.required;

    const PolynomialFunctional functional(spec);
    const auto walk = RunWalk(*fixture, functional);
    ASSERT_TRUE(walk.has_value()) << walk.error().message;

    const auto expected = ExpectedHessian(*fixture, functional, InstrumentOptions{});
    ASSERT_TRUE(expected.has_value());

    InstrumentOptions flipped;
    flipped.orbitalSecondSign = -1.0;
    const auto broken = ExpectedHessian(*fixture, functional, flipped);
    ASSERT_TRUE(broken.has_value());

    ExpectWalkMatches("tau second response", walk->hessian, *expected, *broken, kExactPinTolerance);
}

// e = c sigma_ab. The cross pair enters the weight once, not twice: only the
// diagonal pairs carry the doubled coefficient. Doubling the cross half is a
// defect no internally consistent derivative check sees.
TEST(XcHessianExactnessTest, TheCrossPairWeightIsNotDoubled) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    constexpr double kScale = 0.19;

    PolynomialSpec spec;
    spec.linear[static_cast<std::size_t>(excgrid::Component::SigmaAb)] = kScale;
    spec.required.Set(excgrid::Component::SigmaAb);
    spec.tierMask = spec.required;

    const PolynomialFunctional functional(spec);
    const auto walk = RunWalk(*fixture, functional);
    ASSERT_TRUE(walk.has_value()) << walk.error().message;

    const auto expected = ExpectedHessian(*fixture, functional, InstrumentOptions{});
    ASSERT_TRUE(expected.has_value());

    InstrumentOptions doubled;
    doubled.crossPairResponseScale = 2.0;
    const auto broken = ExpectedHessian(*fixture, functional, doubled);
    ASSERT_TRUE(broken.has_value());

    ExpectWalkMatches("cross pair weight", walk->hessian, *expected, *broken, kExactPinTolerance);
}

// ---------------------------------------------------------------------------
// Pins 4-5: the kernel's contracted second partials
// ---------------------------------------------------------------------------

// e = 1/2 k sigma_aa^2. The kernel contributes k sigma D2 sigma; the field's
// own second response contributes k (D sigma)(D sigma). Dropping the kernel
// contraction leaves only the second and moves the pin by the first. A
// transposed right-hand side is invisible here, because this block is a single
// symmetric entry; the mixed pin below is the one that catches it.
TEST(XcHessianExactnessTest, TheKernelContractionPinsTheSigmaSecondPartial) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    constexpr double kCurvature = 0.47;

    PolynomialSpec spec;
    const auto sigma = static_cast<std::size_t>(excgrid::Component::SigmaAa);
    spec.quadratic[sigma][sigma] = kCurvature;
    spec.tier[sigma][sigma] = kCurvature;
    spec.required.Set(excgrid::Component::SigmaAa);
    spec.tierMask = spec.required;

    const PolynomialFunctional functional(spec);
    const auto walk = RunWalk(*fixture, functional);
    ASSERT_TRUE(walk.has_value()) << walk.error().message;

    const auto expected = ExpectedHessian(*fixture, functional, InstrumentOptions{});
    ASSERT_TRUE(expected.has_value());

    InstrumentOptions withoutKernel;
    withoutKernel.kernelContraction = false;
    const auto broken = ExpectedHessian(*fixture, functional, withoutKernel);
    ASSERT_TRUE(broken.has_value());

    ExpectWalkMatches("kernel contraction", walk->hessian, *expected, *broken, kKernelPinTolerance);
}

// e = 1/2 a s^2 + 1/2 b t^2 + k s t with s = sigma_aa and t = tau_a, a != b.
// The second-partial matrix is two by two and its diagonal entries differ, so
// an ordering error that exchanges the sides changes the value. The two sides
// of a symmetric matrix are otherwise indistinguishable, which is why the
// single-block pin above cannot catch this.
TEST(XcHessianExactnessTest, TheMixedSecondPartialPinsTheRightHandSideLayout) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    constexpr double kSigmaCurvature = 0.61;
    constexpr double kTauCurvature = 0.17;
    constexpr double kMixed = 0.29;

    PolynomialSpec spec;
    const auto sigma = static_cast<std::size_t>(excgrid::Component::SigmaAa);
    const auto tau = static_cast<std::size_t>(excgrid::Component::TauA);
    spec.quadratic[sigma][sigma] = kSigmaCurvature;
    spec.quadratic[tau][tau] = kTauCurvature;
    spec.quadratic[sigma][tau] = kMixed;
    spec.quadratic[tau][sigma] = kMixed;
    spec.tier = spec.quadratic;
    spec.required.Set(excgrid::Component::SigmaAa);
    spec.required.Set(excgrid::Component::TauA);
    spec.tierMask = spec.required;

    const PolynomialFunctional functional(spec);
    const auto walk = RunWalk(*fixture, functional);
    ASSERT_TRUE(walk.has_value()) << walk.error().message;

    const auto expected = ExpectedHessian(*fixture, functional, InstrumentOptions{});
    ASSERT_TRUE(expected.has_value());

    InstrumentOptions transposed;
    transposed.transposeSecondSide = true;
    const auto broken = ExpectedHessian(*fixture, functional, transposed);
    ASSERT_TRUE(broken.has_value());

    ExpectWalkMatches(
        "mixed second partial", walk->hessian, *expected, *broken, kKernelPinTolerance);
}

// ---------------------------------------------------------------------------
// Pin 6: the position curvature is carried once
// ---------------------------------------------------------------------------

// A functional linear in rho, sigma and tau has constant partials and a zero
// second-derivative tier. Its whole spatial gradient is the chain rule over the
// seven fields, which is the cross section the position curvature multiplies.
// The electronic channel must not carry the curvature itself: the difference
// between the curved and plain walks is exactly sum_axis curvature x cross
// section, and carrying it inside the field responses would double it.
TEST(XcHessianExactnessTest, ThePositionCurvatureIsCarriedOnce) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    constexpr double kCurvatureScale = 1.0;
    constexpr std::array<double, 7> kPartials = {0.3, 0.2, 0.11, -0.07, 0.05, 0.13, -0.04};

    PolynomialSpec spec;
    spec.linear = kPartials;
    spec.required = excgrid::ComponentMask::Collinear();
    spec.tierMask = spec.required;

    const PolynomialFunctional functional(spec);
    const CurvedPositionProvider curved(*fixture->provider, kCurvatureScale);

    const auto plain = EvaluateXcGridMotionHessian(
        fixture->engine.Grid().Blocks(),
        *fixture->provider,
        fixture->engine.Evaluator(),
        functional,
        fixture->engine.ShellEnvelopes(),
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kDenseScreeningTolerance),
        fixture->densityAlpha,
        fixture->densityBeta);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    const auto perturbed = EvaluateXcGridMotionHessian(
        fixture->engine.Grid().Blocks(),
        curved,
        fixture->engine.Evaluator(),
        functional,
        fixture->engine.ShellEnvelopes(),
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kDenseScreeningTolerance),
        fixture->densityAlpha,
        fixture->densityBeta);
    ASSERT_TRUE(perturbed.has_value()) << perturbed.error().message;

    // The cross section: the weighted sum of the integrand's spatial gradient.
    // The dense tolerance keeps every shell, so the instrument's all-AO walk and
    // the walk's kept-slot walk read the same data.
    const std::size_t coordinates = 3 * fixture->molecule.AtomCount();
    std::vector<std::array<double, 3>> xFirst(coordinates);
    std::array<double, 3> crossSection{};
    std::size_t visited = 0;

    for (const excgrid::Block& block : fixture->engine.Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const double weight = block.weights[point];

            if (weight == 0.0)
            {
                continue;
            }

            ++visited;
            const FieldResponses fields = EvaluateFieldResponses(fixture->engine.Evaluator(),
                                                                 fixture->atomOfAo,
                                                                 fixture->densityAlpha,
                                                                 fixture->densityBeta,
                                                                 xFirst,
                                                                 block.points[point],
                                                                 InstrumentOptions{});

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                for (std::size_t index = 0; index < 7; ++index)
                {
                    crossSection[axis] +=
                        weight * kPartials[index] * fields.spatialGradient[index][axis];
                }
            }
        }
    }

    ASSERT_EQ(visited, plain->counts.points);

    Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(coordinates),
                                                     static_cast<Eigen::Index>(coordinates));

    for (std::size_t row = 0; row < coordinates; ++row)
    {
        for (std::size_t column = row; column < coordinates; ++column)
        {
            double value = 0.0;

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                value += PrescribedPositionCurvature(kCurvatureScale, axis, row, column) *
                         crossSection[axis];
            }

            expected(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column)) += value;
        }
    }

    for (std::size_t row = 0; row < coordinates; ++row)
    {
        for (std::size_t column = row + 1; column < coordinates; ++column)
        {
            expected(static_cast<Eigen::Index>(column), static_cast<Eigen::Index>(row)) =
                expected(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(column));
        }
    }

    const Eigen::MatrixXd actual = perturbed->hessian - plain->hessian;
    const Eigen::MatrixXd doubled = 2.0 * expected;

    ExpectWalkMatches("position curvature", actual, expected, doubled, kPositionCurvatureTolerance);
}

// ---------------------------------------------------------------------------
// The bookkeeping checks
// ---------------------------------------------------------------------------

// The walk's kernel term, isolated by comparing two functional objects whose
// first-order tiers are identical and whose second-derivative tiers differ.
// The difference of their Hessians is exactly
//     sum_p W G1(q)^T C G1(r),
// and the right-hand side here is that sum computed from the field evaluator's
// own first responses. The materialised matrix is checked entry by entry
// against the analytic one at every visited point, which pins the identifier
// order and the packing; the reversed component order is the broken case.
struct ContractionReference {
    Eigen::MatrixXd expected;
    Eigen::MatrixXd reversed;
    double maxMatrixDeviation = 0.0;
};

std::optional<ContractionReference> ReferenceKernelContraction(
    const PinFixture& fixture,
    const excgrid::XcFunctional& functional,
    const std::array<std::array<double, 7>, 7>& curvature) {
    std::size_t atomCount = 0;

    for (const ShellEnvelope& shell : fixture.engine.ShellEnvelopes())
    {
        atomCount = std::max(atomCount, shell.atomIndex + 1);
    }

    const std::size_t coordinates = 3 * atomCount;
    const excgrid::DerivativeRequest request = SecondOrderRequest(coordinates);
    std::size_t largestBlock = 0;

    for (const excgrid::Block& block : fixture.engine.Grid().Blocks())
    {
        largestBlock = std::max(largestBlock, block.pointCount);
    }

    const excgrid::BlockDerivativeSizes sizes = excgrid::SizeOf(request, largestBlock);
    std::vector<double> positionFirst(sizes.positionFirst);
    std::vector<double> weightFirst(sizes.weightFirst);
    std::vector<double> positionSecond(sizes.positionSecond);
    std::vector<double> weightSecond(sizes.weightSecond);
    std::vector<std::array<double, 3>> xFirst(coordinates);

    ContractionReference reference;
    reference.expected = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(coordinates),
                                               static_cast<Eigen::Index>(coordinates));
    reference.reversed = reference.expected;

    for (const excgrid::Block& block : fixture.engine.Grid().Blocks())
    {
        excgrid::BlockDerivatives derivatives;
        derivatives.positionFirst = std::span<double>(positionFirst.data(), sizes.positionFirst);
        derivatives.weightFirst = std::span<double>(weightFirst.data(), sizes.weightFirst);
        derivatives.positionSecond = std::span<double>(positionSecond.data(), sizes.positionSecond);
        derivatives.weightSecond = std::span<double>(weightSecond.data(), sizes.weightSecond);
        derivatives.atomIndex = block.atomIndex;

        if (fixture.provider->Evaluate(block, request, derivatives) !=
            excgrid::DerivativeStatus::kOk)
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

            const FieldResponses fields = EvaluateFieldResponses(fixture.engine.Evaluator(),
                                                                 fixture.atomOfAo,
                                                                 fixture.densityAlpha,
                                                                 fixture.densityBeta,
                                                                 xFirst,
                                                                 block.points[point],
                                                                 InstrumentOptions{});

            excgrid::PointInputs inputs;

            for (std::size_t index = 0; index < 7; ++index)
            {
                inputs.Set(static_cast<excgrid::Component>(index), fields.value[index]);
            }

            excgrid::PointResult result;
            excgrid::PointSecondDerivativeMatrix matrix;

            if (functional.EvaluatePointMaterialising(inputs, result, matrix) !=
                excgrid::KernelStatus::kOk)
            {
                return std::nullopt;
            }

            if (matrix.mask.bits != excgrid::ComponentMask::Collinear().bits)
            {
                return std::nullopt;
            }

            for (std::size_t row = 0; row < 7; ++row)
            {
                for (std::size_t column = row; column < 7; ++column)
                {
                    const double value = matrix.upper[UpperIndex(row, column, 7)];
                    reference.maxMatrixDeviation = std::max(
                        reference.maxMatrixDeviation, std::abs(value - curvature[row][column]));
                }
            }

            for (std::size_t q = 0; q < coordinates; ++q)
            {
                for (std::size_t r = q; r < coordinates; ++r)
                {
                    double value = 0.0;
                    double reversed = 0.0;

                    for (std::size_t row = 0; row < 7; ++row)
                    {
                        for (std::size_t column = 0; column < 7; ++column)
                        {
                            value += curvature[row][column] * fields.first[q][row] *
                                     fields.first[r][column];
                            reversed += curvature[row][column] * fields.first[q][row] *
                                        fields.first[r][6 - column];
                        }
                    }

                    const Eigen::Index qEntry = static_cast<Eigen::Index>(q);
                    const Eigen::Index rEntry = static_cast<Eigen::Index>(r);
                    reference.expected(qEntry, rEntry) += weight * value;
                    reference.reversed(qEntry, rEntry) += weight * reversed;
                }
            }
        }
    }

    for (std::size_t q = 0; q < coordinates; ++q)
    {
        for (std::size_t r = q + 1; r < coordinates; ++r)
        {
            const Eigen::Index qEntry = static_cast<Eigen::Index>(q);
            const Eigen::Index rEntry = static_cast<Eigen::Index>(r);
            reference.expected(rEntry, qEntry) = reference.expected(qEntry, rEntry);
            reference.reversed(rEntry, qEntry) = reference.reversed(qEntry, rEntry);
        }
    }

    return reference;
}

TEST(XcHessianExactnessTest, TheMaterialisedMatrixMatchesTheWalkContractionTerm) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    // A generic symmetric matrix with distinct entries, so an ordering error
    // cannot hide behind a repeated coefficient.
    std::array<std::array<double, 7>, 7> curvature{};

    for (std::size_t row = 0; row < 7; ++row)
    {
        for (std::size_t column = row; column < 7; ++column)
        {
            const double i = static_cast<double>(row + 1);
            const double j = static_cast<double>(column + 1);
            const double value = 0.31 * std::cos(0.9 * i * j) + 0.07 * (i + j);
            curvature[row][column] = value;
            curvature[column][row] = value;
        }
    }

    PolynomialSpec spec;
    spec.linear = {0.13, -0.09, 0.17, 0.05, -0.11, 0.07, 0.03};
    spec.quadratic = curvature;
    spec.tier = curvature;
    spec.required = excgrid::ComponentMask::Collinear();
    spec.tierMask = spec.required;

    const PolynomialFunctional full(spec);

    // The control answers the same energy and the same first partials and a
    // zero second-derivative matrix, so subtracting its Hessian leaves the
    // walk's kernel contraction and nothing else.
    PolynomialSpec controlSpec = spec;
    controlSpec.tier = {};

    const PolynomialFunctional control(controlSpec);

    const auto fullWalk = RunWalk(*fixture, full);
    ASSERT_TRUE(fullWalk.has_value()) << fullWalk.error().message;

    const auto controlWalk = RunWalk(*fixture, control);
    ASSERT_TRUE(controlWalk.has_value()) << controlWalk.error().message;

    const auto reference = ReferenceKernelContraction(*fixture, full, curvature);
    ASSERT_TRUE(reference.has_value());

    // The materialising entry point is the kernel's own packing; the analytic
    // matrix is written verbatim, so the two agree to the last bit.
    EXPECT_LE(reference->maxMatrixDeviation, 0.0)
        << "the materialised matrix is not the analytic one over the active components";

    const Eigen::MatrixXd difference = fullWalk->hessian - controlWalk->hessian;

    ExpectWalkMatches("contraction term",
                      difference,
                      reference->expected,
                      reference->reversed,
                      kExactPinTolerance);
}

// The spy: every point's curvature inputs and right-hand sides, compared
// component by component against the definitions the field evaluator computes.
// This is the check a finite difference cannot make - it names the component,
// the coordinate and the side rather than only the Hessian entry that moved.
TEST(XcHessianExactnessTest, TheSpySeesTheComponentsAndResponsesTheDefinitionsName) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    PolynomialSpec spec;
    spec.linear = {0.11, -0.07, 0.15, 0.04, -0.09, 0.06, 0.02};
    spec.required = excgrid::ComponentMask::Collinear();
    spec.tierMask = spec.required;

    FunctionalRecording recording;
    spec.recording = &recording;
    const PolynomialFunctional functional(spec);

    const auto walk = RunWalk(*fixture, functional);
    ASSERT_TRUE(walk.has_value()) << walk.error().message;

    const std::size_t coordinates = 3 * fixture->molecule.AtomCount();
    const std::size_t sides = std::max<std::size_t>(
        1, excgrid::kSecondDerivativeCapacity / excgrid::ComponentMask::Collinear().ActiveCount());
    const std::size_t callsPerPoint = (coordinates + sides - 1) / sides;

    ASSERT_EQ(recording.pointInputs.size(), walk->counts.points);
    ASSERT_EQ(recording.curvatureInputs.size(), walk->counts.points * callsPerPoint);
    ASSERT_EQ(recording.curvatureRightHandSides.size(), walk->counts.points * callsPerPoint);

    const excgrid::DerivativeRequest request = SecondOrderRequest(coordinates);
    std::size_t largestBlock = 0;

    for (const excgrid::Block& block : fixture->engine.Grid().Blocks())
    {
        largestBlock = std::max(largestBlock, block.pointCount);
    }

    const excgrid::BlockDerivativeSizes sizes = excgrid::SizeOf(request, largestBlock);
    std::vector<double> positionFirst(sizes.positionFirst);
    std::vector<double> weightFirst(sizes.weightFirst);
    std::vector<double> positionSecond(sizes.positionSecond);
    std::vector<double> weightSecond(sizes.weightSecond);
    std::vector<std::array<double, 3>> xFirst(coordinates);
    std::size_t record = 0;
    double maxFieldDeviation = 0.0;
    double maxResponseDeviation = 0.0;
    double maxBrokenSeparation = 0.0;
    double fieldScale = 0.0;

    for (const excgrid::Block& block : fixture->engine.Grid().Blocks())
    {
        excgrid::BlockDerivatives derivatives;
        derivatives.positionFirst = std::span<double>(positionFirst.data(), sizes.positionFirst);
        derivatives.weightFirst = std::span<double>(weightFirst.data(), sizes.weightFirst);
        derivatives.positionSecond = std::span<double>(positionSecond.data(), sizes.positionSecond);
        derivatives.weightSecond = std::span<double>(weightSecond.data(), sizes.weightSecond);
        derivatives.atomIndex = block.atomIndex;

        ASSERT_EQ(fixture->provider->Evaluate(block, request, derivatives),
                  excgrid::DerivativeStatus::kOk);

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

            const FieldResponses fields = EvaluateFieldResponses(fixture->engine.Evaluator(),
                                                                 fixture->atomOfAo,
                                                                 fixture->densityAlpha,
                                                                 fixture->densityBeta,
                                                                 xFirst,
                                                                 block.points[point],
                                                                 InstrumentOptions{});

            const excgrid::PointInputs& pointInputs = recording.pointInputs[record];
            ASSERT_EQ(pointInputs.mask.bits, excgrid::ComponentMask::Collinear().bits);

            for (std::size_t index = 0; index < 7; ++index)
            {
                const auto id = static_cast<excgrid::Component>(index);
                maxFieldDeviation = std::max(maxFieldDeviation,
                                             std::abs(pointInputs.Get(id) - fields.value[index]));
                fieldScale = std::max(fieldScale, std::abs(fields.value[index]));
            }

            for (std::size_t call = 0; call < callsPerPoint; ++call)
            {
                const std::size_t start = call * sides;
                const std::size_t count = std::min(sides, coordinates - start);
                const std::size_t recordIndex = record * callsPerPoint + call;
                const excgrid::PointInputs& curvatureInputs =
                    recording.curvatureInputs[recordIndex];
                const std::vector<double>& rightHandSide =
                    recording.curvatureRightHandSides[recordIndex];

                ASSERT_EQ(curvatureInputs.mask.bits, excgrid::ComponentMask::Collinear().bits)
                    << "the curvature inputs must carry exactly the active components";
                ASSERT_EQ(rightHandSide.size(), count * 7);

                for (std::size_t index = 0; index < 7; ++index)
                {
                    const auto id = static_cast<excgrid::Component>(index);
                    maxFieldDeviation = std::max(
                        maxFieldDeviation, std::abs(curvatureInputs.Get(id) - fields.value[index]));
                }

                for (std::size_t side = 0; side < count; ++side)
                {
                    for (std::size_t index = 0; index < 7; ++index)
                    {
                        const double recorded = rightHandSide[side * 7 + index];
                        const double expected = fields.first[start + side][index];
                        maxResponseDeviation =
                            std::max(maxResponseDeviation, std::abs(recorded - expected));

                        // The broken definition: a sigma_aa first response without
                        // its two-slot factor. Half of it is a defect the pair
                        // check would miss if it only compared the total.
                        if (index == static_cast<std::size_t>(excgrid::Component::SigmaAa))
                        {
                            maxBrokenSeparation =
                                std::max(maxBrokenSeparation, std::abs(recorded - 0.5 * expected));
                        }
                    }
                }
            }

            ++record;
        }
    }

    EXPECT_EQ(record, walk->counts.points);

    const double bar = 1e-10 * (1.0 + fieldScale);

    ::testing::Test::RecordProperty("inputs against the definitions",
                                    Scientific(maxFieldDeviation));
    ::testing::Test::RecordProperty("responses against the definitions",
                                    Scientific(maxResponseDeviation));
    ::testing::Test::RecordProperty("half-response separation", Scientific(maxBrokenSeparation));

    EXPECT_LT(maxFieldDeviation, 1e-12 * (1.0 + fieldScale))
        << "the walk's inputs and the definitions disagree";
    EXPECT_LT(maxResponseDeviation, bar)
        << "the walk's right-hand sides and the definitions disagree";
    EXPECT_GT(maxBrokenSeparation, bar)
        << "the comparison cannot separate a response missing its two-slot factor";
}

// ---------------------------------------------------------------------------
// The screening sweep at second order
// ---------------------------------------------------------------------------

// An oxygen STO-3G core (its 1s exponent is 130.7) beside a diffuse hydrogen s
// function (exponent 0.03). The sigma and tau fields carry gradients, so their
// tails scale as gradient squared and are not bounded by the density-weight
// envelope; the second response is the largest extrapolation of that bound.
// The fixture is the one the first-order tau gate names.
constexpr std::string_view kTightCoreDiffuseBasis = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      0.1307093214E+03       0.1543289673E+00
      0.2380886605E+02       0.5353281423E+00
      0.6443608313E+01       0.4446345422E+00
O    SP
      0.5033151319E+01      -0.9996722919E-01       0.1559162750E+00
      0.1169596125E+01       0.3995128261E+00       0.6076837186E+00
      0.3803889600E+00       0.7001154689E+00       0.3919573931E+00
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
H    S
      3.0000000000E-02       1.0000000000E+00
END
)";

qcx::Result<qcx::molecule::Molecule> MakeTightCoreDiffuseMolecule() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.8;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// O/STO-3G carries five AOs and the hydrogen contributes the STO-3G s plus the
// diffuse s: seven in all. Both hydrogen slots carry density, so the diffuse
// shell is populated whichever order the canonical atom list puts them in.
constexpr std::size_t kDiffuseAOCount = 7;

Eigen::MatrixXd DiffuseAlphaDensity() {
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(kDiffuseAOCount),
                                                    static_cast<Eigen::Index>(kDiffuseAOCount));
    density.diagonal() << 0.6, 0.6, 0.4, 0.4, 0.4, 0.5, 0.8;

    return density;
}

Eigen::MatrixXd DiffuseBetaDensity() {
    Eigen::MatrixXd density = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(kDiffuseAOCount),
                                                    static_cast<Eigen::Index>(kDiffuseAOCount));
    density.diagonal() << 0.3, 0.3, 0.2, 0.2, 0.2, 0.25, 0.4;

    return density;
}

XcGridSettings ScreeningSettings() {
    XcGridSettings settings;
    settings.radialPoints = kScreeningRadialPoints;
    settings.angularPoints = kScreeningAngularPoints;
    settings.blockTarget = kBlockTarget;

    return settings;
}

struct ScreeningFixture {
    XcGridSettings settings;
    XcGridEngine engine;
    std::unique_ptr<excgrid::GridDerivativeProvider> provider;
};

std::optional<ScreeningFixture> MakeScreeningFixture(std::string_view functionalName) {
    const auto basis = qcx::basisset::ParseNwchemText(kTightCoreDiffuseBasis);

    if (!basis.has_value())
    {
        return std::nullopt;
    }

    const auto molecule = MakeTightCoreDiffuseMolecule();

    if (!molecule.has_value())
    {
        return std::nullopt;
    }

    const XcGridSettings settings = ScreeningSettings();
    auto engine = XcGridEngine::Create(*molecule, *basis, functionalName, settings);

    if (!engine.has_value())
    {
        return std::nullopt;
    }

    const auto geometry = ToExcgridGeometry(*molecule);

    if (!geometry.has_value())
    {
        return std::nullopt;
    }

    auto provider = excgrid::CreateGridDerivativeProvider(
        *geometry, ToExcgridParams(settings), excgrid::DerivativeOrder::kSecond);

    if (!provider.has_value())
    {
        return std::nullopt;
    }

    return ScreeningFixture{settings, std::move(*engine), std::move(*provider)};
}

// The selection the public rule makes, rebuilt from the envelopes and the
// predicate: the same point set the energy path and the Hessian walk must both
// visit.
struct SelectionCounts {
    std::size_t points = 0;
    std::size_t shellTests = 0;
    std::size_t shellKeeps = 0;
    std::size_t aoSlots = 0;
};

SelectionCounts ReproduceSelection(const ScreeningFixture& fixture,
                                   std::span<const double> weightsAlpha,
                                   std::span<const double> weightsBeta,
                                   double tolerance) {
    SelectionCounts counts;

    for (const excgrid::Block& block : fixture.engine.Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            if (block.weights[point] == 0.0)
            {
                continue;
            }

            ++counts.points;

            for (std::size_t shell = 0; shell < fixture.engine.ShellEnvelopes().size(); ++shell)
            {
                ++counts.shellTests;

                if (!ShellIsSignificant(fixture.engine.ShellEnvelopes()[shell],
                                        block.points[point],
                                        std::max(weightsAlpha[shell], weightsBeta[shell]),
                                        tolerance))
                {
                    continue;
                }

                ++counts.shellKeeps;
                counts.aoSlots += fixture.engine.ShellEnvelopes()[shell].functionCount;
            }
        }
    }

    return counts;
}

// The sweep: for pbe and tau_x on the tight-core/diffuse fixture, the walk at
// three tolerances against the dense one. The production bar is the order the
// first-order tau gate reaches at the same tolerance; a deviation above it is
// a finding about the density-weight envelope at second order, recorded rather
// than absorbed by loosening the tolerance.
TEST(XcHessianExactnessTest, TheSecondOrderScreeningSweepStaysWithinItsBar) {
    constexpr double kProductionBar = 1e-4;
    constexpr double kNearDenseBar = 1e-6;
    constexpr std::array<double, 3> kTolerances = {
        kNearDenseTolerance, kTightTolerance, kProductionTolerance};

    for (const std::string_view name : {"pbe", "tau_x"})
    {
        const auto fixture = MakeScreeningFixture(name);
        ASSERT_TRUE(fixture.has_value()) << name;

        const Eigen::MatrixXd alpha = DiffuseAlphaDensity();
        const Eigen::MatrixXd beta = DiffuseBetaDensity();

        ASSERT_EQ(fixture->engine.AOCount(), kDiffuseAOCount)
            << "the tight-core/diffuse fixture's AO layout changed";

        const auto weightsAlphaResult = ShellDensityWeights(
            fixture->engine.ShellEnvelopes(), RowMajorCopy(alpha), fixture->engine.AOCount());
        const auto weightsBetaResult = ShellDensityWeights(
            fixture->engine.ShellEnvelopes(), RowMajorCopy(beta), fixture->engine.AOCount());
        ASSERT_TRUE(weightsAlphaResult.has_value());
        ASSERT_TRUE(weightsBetaResult.has_value());

        const auto dense = EvaluateXcGridMotionHessian(
            fixture->engine.Grid().Blocks(),
            *fixture->provider,
            fixture->engine.Evaluator(),
            fixture->engine.Functional(),
            fixture->engine.ShellEnvelopes(),
            XcIntegrationThresholds::FromEnergyPath(fixture->settings, kDenseScreeningTolerance),
            alpha,
            beta);
        ASSERT_TRUE(dense.has_value()) << std::string(name) << ": " << dense.error().message;

        // The third tier is fetched wherever the Hessian tier is, and only for a
        // functional that reads sigma or tau.
        EXPECT_EQ(dense->counts.aoSlotsEvaluated, dense->counts.points * kDiffuseAOCount);
        EXPECT_EQ(dense->counts.aoThirdSlotsEvaluated, dense->counts.points * kDiffuseAOCount)
            << std::string(name) << ": the third tier is not fetched at the dense setting";
        EXPECT_EQ(dense->counts.aoThirdSlotsDense, dense->counts.points * kDiffuseAOCount);
        EXPECT_GT(dense->counts.electronicThirdTerms, 0U);

        const double denseScale = std::max(dense->hessian.cwiseAbs().maxCoeff(), 1e-300);
        std::array<double, 3> deviation{};
        std::array<SelectionCounts, 3> selection{};

        for (std::size_t index = 0; index < kTolerances.size(); ++index)
        {
            const double tolerance = kTolerances[index];

            const auto walk = EvaluateXcGridMotionHessian(
                fixture->engine.Grid().Blocks(),
                *fixture->provider,
                fixture->engine.Evaluator(),
                fixture->engine.Functional(),
                fixture->engine.ShellEnvelopes(),
                XcIntegrationThresholds::FromEnergyPath(fixture->settings, tolerance),
                alpha,
                beta);
            ASSERT_TRUE(walk.has_value()) << std::string(name) << ": " << walk.error().message;

            const auto energyPath = fixture->engine.EvaluateScreened(alpha, beta, tolerance);
            ASSERT_TRUE(energyPath.has_value()) << std::string(name);

            selection[index] =
                ReproduceSelection(*fixture, *weightsAlphaResult, *weightsBetaResult, tolerance);

            deviation[index] = (walk->hessian - dense->hessian).cwiseAbs().maxCoeff() / denseScale;

            ::testing::Test::RecordProperty(std::string(name) + " tolerance " +
                                                std::to_string(index) + ": deviation",
                                            Scientific(deviation[index]));
            ::testing::Test::RecordProperty(std::string(name) + " tolerance " +
                                                std::to_string(index) + ": shells kept",
                                            std::to_string(walk->counts.shellKeeps) + " / " +
                                                std::to_string(walk->counts.shellTests));

            // The walk visits the selection the public rule makes, and the
            // energy path's own counts agree: the Hessian is the second
            // derivative of the energy that was actually integrated.
            EXPECT_EQ(walk->counts.points, selection[index].points);
            EXPECT_EQ(walk->counts.shellTests, selection[index].shellTests);
            EXPECT_EQ(walk->counts.shellKeeps, selection[index].shellKeeps);
            EXPECT_EQ(walk->counts.aoSlotsEvaluated, selection[index].aoSlots);
            EXPECT_EQ(energyPath->counts.points, walk->counts.points);
            EXPECT_EQ(energyPath->counts.shellTests, walk->counts.shellTests);
            EXPECT_EQ(energyPath->counts.shellKeeps, walk->counts.shellKeeps);

            // The two walks read the same point set, so the energy each
            // integrates is the same object summed in the same order.
            EXPECT_DOUBLE_EQ(walk->energy, energyPath->energy)
                << std::string(name)
                << ": the Hessian walk integrated an energy the energy path does not recognise";
        }

        // The sweep must bite at its loose end, or it measures nothing.
        EXPECT_LT(selection[2].shellKeeps, selection[2].shellTests)
            << std::string(name) << ": the production tolerance keeps every shell";

        EXPECT_LT(deviation[0], kNearDenseBar)
            << std::string(name) << ": the near-dense tolerance is not near dense";
        EXPECT_LT(deviation[2], kProductionBar)
            << std::string(name)
            << ": the selection's second-order deviation exceeds the first-order gate's order; "
               "this is a finding about the density-weight envelope, not a tolerance to loosen";
    }
}

// ---------------------------------------------------------------------------
// The refusal paths and the density channel's counted cost
// ---------------------------------------------------------------------------

// A component outside the seven has no second response at all, so the walk must
// refuse it by name rather than answer with the density channel's terms.
TEST(XcHessianExactnessTest, ALaplacianFunctionalIsRefusedByName) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    PolynomialSpec spec;
    spec.linear[static_cast<std::size_t>(excgrid::Component::RhoA)] = 0.4;
    spec.required.Set(excgrid::Component::RhoA);
    spec.required.Set(excgrid::Component::LaplA);
    spec.tierMask = spec.required;

    const PolynomialFunctional functional(spec);
    const auto hessian = RunWalk(*fixture, functional);

    ASSERT_FALSE(hessian.has_value());
    EXPECT_EQ(hessian.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(hessian.error().message.find("laplA"), std::string::npos) << hessian.error().message;

    // The control: the same first-order tier without the Laplacian component is
    // answered, so the refusal above is the component's and nothing else.
    PolynomialSpec answered = spec;
    answered.required.Clear(excgrid::Component::LaplA);
    answered.tierMask = answered.required;

    const PolynomialFunctional controllable(answered);
    const auto result = RunWalk(*fixture, controllable);
    ASSERT_TRUE(result.has_value()) << result.error().message;
}

// A five-component tier under a seven-component mask leaves tau unspanned. The
// walk must name the uncovered components; answering with the five it spans
// would be a plausible wrong Hessian.
TEST(XcHessianExactnessTest, ATierThatDoesNotSpanTheMaskIsRefusedByName) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    PolynomialSpec spec;
    spec.linear = {0.2, 0.1, 0.05, 0.03, -0.02, 0.07, 0.04};
    spec.required = excgrid::ComponentMask::Collinear();

    for (const excgrid::Component id : {excgrid::Component::RhoA,
                                        excgrid::Component::RhoB,
                                        excgrid::Component::SigmaAa,
                                        excgrid::Component::SigmaAb,
                                        excgrid::Component::SigmaBb})
    {
        spec.tierMask.Set(id);
    }

    const PolynomialFunctional functional(spec);
    const auto hessian = RunWalk(*fixture, functional);

    ASSERT_FALSE(hessian.has_value());
    EXPECT_EQ(hessian.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(hessian.error().message.find("tauA"), std::string::npos) << hessian.error().message;
    EXPECT_NE(hessian.error().message.find("tauB"), std::string::npos) << hessian.error().message;

    // The control: the same mask with a tier that does span it is answered.
    PolynomialSpec answered = spec;
    answered.tierMask = excgrid::ComponentMask::Collinear();

    const PolynomialFunctional controllable(answered);
    const auto result = RunWalk(*fixture, controllable);
    ASSERT_TRUE(result.has_value()) << result.error().message;
}

// The density channel is unchanged: an LDA functional fetches no third tier,
// and its counters are the ones the density-only walk has always reported.
TEST(XcHessianExactnessTest, TheDensityChannelFetchesNoThirdTier) {
    const auto fixture = MakePinFixture();
    ASSERT_TRUE(fixture.has_value());

    const auto hessian = EvaluateXcGridMotionHessian(
        fixture->engine,
        fixture->densityAlpha,
        fixture->densityBeta,
        *fixture->provider,
        XcIntegrationThresholds::FromEnergyPath(fixture->settings, kDenseScreeningTolerance));

    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    const std::size_t aoCount = fixture->engine.AOCount();
    EXPECT_GT(hessian->counts.points, 0U);
    EXPECT_EQ(hessian->counts.aoSlotsEvaluated, hessian->counts.points * aoCount);
    EXPECT_EQ(hessian->counts.aoThirdSlotsEvaluated, 0U)
        << "an LDA functional has no third-derivative term to fetch";
    EXPECT_EQ(hessian->counts.aoThirdSlotsDense, 0U);
    EXPECT_EQ(hessian->counts.electronicThirdTerms, 0U);

    // Two active components, so the contraction's capacity carries eight
    // right-hand sides per call.
    const std::size_t coordinates = 3 * fixture->molecule.AtomCount();
    const std::size_t sides = excgrid::kSecondDerivativeCapacity / 2;
    EXPECT_EQ(hessian->counts.kernelSecondDerivativeCalls,
              ((coordinates + sides - 1) / sides) * hessian->counts.points);

    ::testing::Test::RecordProperty("LDA points", std::to_string(hessian->counts.points));
    ::testing::Test::RecordProperty("LDA AO slots",
                                    std::to_string(hessian->counts.aoSlotsEvaluated));
    ::testing::Test::RecordProperty("LDA third-tier slots",
                                    std::to_string(hessian->counts.aoThirdSlotsEvaluated));
}

} // namespace
} // namespace qcx::grid
