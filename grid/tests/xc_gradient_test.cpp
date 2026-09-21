// The exchange-correlation gradient, on a fixture whose right answer is known
// without asking the code under test:
//   - the point set is this file's, with closed-form positions and weights, so
//     the grid's own motion is exact rather than borrowed from the library's
//     partition;
//   - the energy is this file's too, written as the sums themselves, so the
//     finite difference the gradient is checked against is not a difference of
//     the assembly it checks.
// Three numbers are kept apart on purpose, because a gradient can be wrong in a
// way only one of them sees:
//   - the agreement with the finite difference;
//   - the thresholds it reused, which are the energy path's or the gradient is
//     the derivative of another energy;
//   - the energy the walk integrated, which is where a gradient that screens
//     differently from the energy it differentiates shows up and nowhere else.

#include "h2_sto3g.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/memory/tensor.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <excgrid/geometry.hpp>
#include <excgrid/grid.hpp>
#include <excgrid/grid_derivatives.hpp>
#include <excgrid/kernel.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

// ---------------------------------------------------------------------------
// The fixture's geometry and point set
// ---------------------------------------------------------------------------

// H2 in STO-3G, off the axis: the bond points along a direction no Cartesian
// axis lies on, so every off-diagonal component of the field's second
// derivative is in play rather than three of them.
constexpr std::array<double, 3> kAtomZeroPosition = {0.0, 0.0, 0.0};
constexpr std::array<double, 3> kAtomOnePosition = {1.40, 0.18, -0.12};

// One grid point: the atom it rides and its fixed offset from that atom, Bohr.
// The offset is geometry-independent, which is what makes a grid point a rigid
// translate of its owner and its position's derivative the owner's identity
// block.
struct PointSpec {
    std::array<double, 3> offset = {};
};

// Atom zero's points, then atom one's. The last point of each atom's set sits
// far enough out that a loose tolerance drops shells there, which is what puts
// the significance test on the path rather than leaving the walk dense.
constexpr std::array<PointSpec, 3> kPointsAtomZero = {
    PointSpec{{0.55, 0.31, -0.22}}, PointSpec{{-0.47, 0.28, 0.35}}, PointSpec{{0.25, -0.62, 0.44}}};
constexpr std::array<PointSpec, 3> kPointsAtomOne = {PointSpec{{0.32, -0.24, 0.29}},
                                                     PointSpec{{-0.41, 0.36, -0.18}},
                                                     PointSpec{{4.30, 0.40, -0.30}}};

// ---------------------------------------------------------------------------
// The weight law
// ---------------------------------------------------------------------------

// A point owned by atom a is weighted by exp(-k |x - R_b|^2), with b the other
// atom of this two-atom fixture. The law sees the geometry only through a
// difference, so the whole fixture is translation invariant and the gradient's
// components must sum to zero; and it is a function of BOTH atoms' positions,
// so the weight's motion reaches coordinates the point's own motion does not.
constexpr double kWeightDecay = 0.6;

std::size_t OtherAtom(const excgrid::Geometry& geometry, std::size_t owner) noexcept {
    return (owner + 1U) % geometry.atoms.size();
}

std::array<double, 3> Difference(const std::array<double, 3>& left,
                                 const std::array<double, 3>& right) noexcept {
    return {left[0] - right[0], left[1] - right[1], left[2] - right[2]};
}

double SquaredNorm(const std::array<double, 3>& vector) noexcept {
    return vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
}

double PointWeight(const excgrid::Geometry& geometry,
                   std::size_t owner,
                   const std::array<double, 3>& position) noexcept {
    const std::array<double, 3> delta =
        Difference(position, geometry.atoms[OtherAtom(geometry, owner)].position);

    return std::exp(-kWeightDecay * SquaredNorm(delta));
}

// The weight's derivative with respect to the two atoms that carry it: the
// gradient of exp(-k d^2) is -2 k exp(...) times the gradient of d^2, and d^2
// moves with both ends in opposite directions. Every other atom's entries are
// zero, and the two blocks below sum to zero, which is the law's translation
// invariance seen at the derivative.
std::array<double, 3> WeightDerivative(const excgrid::Geometry& geometry,
                                       std::size_t owner,
                                       const std::array<double, 3>& position) noexcept {
    const std::array<double, 3> delta =
        Difference(position, geometry.atoms[OtherAtom(geometry, owner)].position);

    return {2.0 * kWeightDecay * PointWeight(geometry, owner, position) * delta[0],
            2.0 * kWeightDecay * PointWeight(geometry, owner, position) * delta[1],
            2.0 * kWeightDecay * PointWeight(geometry, owner, position) * delta[2]};
}

// ---------------------------------------------------------------------------
// The fixture's derivative provider
// ---------------------------------------------------------------------------

// The point set's own derivatives, in closed form, behind the contract's
// interface: a point rides its atom, so its position's derivative is the owning
// atom's identity block and nothing else. A block whose weights do not agree
// with the law is not this geometry's grid, and is refused rather than
// differentiated.
class FixtureProvider final : public excgrid::GridDerivativeProvider {
public:
    explicit FixtureProvider(excgrid::Geometry geometry) noexcept :
        _geometry(std::move(geometry)) {}

    void RefuseEverything() noexcept {
        _refuse = true;
    }

    [[nodiscard]] excgrid::DerivativeOrder MaxOrder() const noexcept override {
        return excgrid::DerivativeOrder::kFirst;
    }

    [[nodiscard]] excgrid::DerivativeStatus Evaluate(
        const excgrid::Block& block,
        const excgrid::DerivativeRequest& request,
        const excgrid::BlockDerivatives& derivatives) const override {
        const std::size_t width = request.nuclearCoordinateCount;

        if (request.order != excgrid::DerivativeOrder::kFirst ||
            derivatives.weightFirst.size() < block.pointCount * width ||
            derivatives.positionFirst.size() < block.pointCount * 3U * width)
        {
            return excgrid::DerivativeStatus::kRefusedBufferTooSmall;
        }

        if (_refuse)
        {
            return excgrid::DerivativeStatus::kRefusedUndifferentiableGrid;
        }

        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            const std::size_t owner = block.atomIndex[point];
            const std::size_t other = OtherAtom(_geometry, owner);
            const std::array<double, 3> derivative =
                WeightDerivative(_geometry, owner, block.points[point]);

            if (PointWeight(_geometry, owner, block.points[point]) != block.weights[point])
            {
                return excgrid::DerivativeStatus::kRefusedUndifferentiableGrid;
            }

            for (std::size_t coordinate = 0; coordinate < width; ++coordinate)
            {
                const std::size_t carrier = coordinate / 3U;
                double weight = 0.0;

                if (carrier == owner)
                {
                    weight = -derivative[coordinate % 3U];
                }

                if (carrier == other)
                {
                    weight = derivative[coordinate % 3U];
                }

                derivatives.weightFirst[point * width + coordinate] = weight;

                for (std::size_t direction = 0; direction < 3; ++direction)
                {
                    derivatives.positionFirst[point * 3U * width + direction * width + coordinate] =
                        coordinate == 3U * owner + direction ? 1.0 : 0.0;
                }
            }
        }

        return excgrid::DerivativeStatus::kOk;
    }

private:
    excgrid::Geometry _geometry;
    bool _refuse = false;
};

// ---------------------------------------------------------------------------
// One geometry's state
// ---------------------------------------------------------------------------

// Everything a geometry needs to be walked: the molecule, its AO tier, its
// screening envelopes and the library's own view of its atoms. A displaced
// geometry needs all four rebuilt, because the AO tier and the envelopes carry
// the shell centers.
struct GeometryState {
    qcx::molecule::Molecule molecule;
    AoEvaluator evaluator;
    std::vector<ShellEnvelope> envelopes;
    excgrid::Geometry geometry;
};

qcx::Result<qcx::molecule::Molecule> MakeMolecule(
    const std::array<std::array<double, 3>, 2>& positions) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t atom = 0; atom < 2; ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(static_cast<Eigen::Index>(atom), static_cast<Eigen::Index>(axis)) =
                positions[atom][axis];
        }
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// The state for one geometry, or nothing when a construction fails: the caller
// reports which step failed, since a test that returns a bare nullopt says
// nothing about what went wrong.
std::optional<GeometryState> MakeGeometryState(
    const std::array<std::array<double, 3>, 2>& positions, const qcx::basisset::BasisSet& basis) {
    auto molecule = MakeMolecule(positions);

    if (!molecule.has_value())
    {
        return std::nullopt;
    }

    // The molecule renumbers its atoms canonically, so the fixture's own atom
    // order is the caller's only if the sort agrees with it. The fixture's two
    // hydrogen positions are ordered along x, and a displacement is far too
    // small to reorder them - asserted rather than assumed, because a reorder
    // would make every finite difference below nonsense.
    for (std::size_t atom = 0; atom < 2; ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (molecule->CoordinatesBohr()(static_cast<Eigen::Index>(atom),
                                            static_cast<Eigen::Index>(axis)) !=
                positions[atom][axis])
            {
                return std::nullopt;
            }
        }
    }

    auto evaluator = AoEvaluator::Create(*molecule, basis);

    if (!evaluator.has_value())
    {
        return std::nullopt;
    }

    auto geometry = ToExcgridGeometry(*molecule);

    if (!geometry.has_value())
    {
        return std::nullopt;
    }

    auto envelopes = BuildShellEnvelopes(*molecule, basis, evaluator->ShellRanges());

    if (!envelopes.has_value())
    {
        return std::nullopt;
    }

    return GeometryState{
        std::move(*molecule), std::move(*evaluator), std::move(*envelopes), std::move(*geometry)};
}

// ---------------------------------------------------------------------------
// The fixture's energy, written here
// ---------------------------------------------------------------------------

// One spin's density and its gradient at a point, as the sums themselves rather
// than through the contractions the assembly factors them into: the point of
// this file is an implementation the gradient is not.
void FieldsAtPoint(const Eigen::MatrixXd& density,
                   std::span<const double> values,
                   std::span<const double> gradients,
                   std::span<const std::size_t> slots,
                   double& rho,
                   std::array<double, 3>& gradient) {
    rho = 0.0;
    gradient = {};

    for (const std::size_t mu : slots)
    {
        for (const std::size_t nu : slots)
        {
            const double element =
                density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));
            rho += element * values[mu] * values[nu];

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                gradient[axis] += element * (gradients[3 * mu + axis] * values[nu] +
                                             values[mu] * gradients[3 * nu + axis]);
            }
        }
    }
}

// The fixture's energy at one geometry, under one tolerance, with the selection
// reproduced from the same public rule the engine makes its own with.
struct ReferenceResult {
    double energy = 0.0;
    std::size_t keptShells = 0;
    std::size_t visitedPoints = 0;
};

ReferenceResult ReferenceEnergy(const GeometryState& state,
                                std::span<const PointSpec> points,
                                std::size_t owner,
                                const excgrid::XcFunctional& functional,
                                const Eigen::MatrixXd& densityAlpha,
                                const Eigen::MatrixXd& densityBeta,
                                double tolerance) {
    const std::size_t aoCount = state.evaluator.AOCount();

    // The per-shell density weights, from the same public rule.
    std::vector<double> weightsAlpha(state.envelopes.size(), 0.0);
    std::vector<double> weightsBeta(state.envelopes.size(), 0.0);

    for (std::size_t shell = 0; shell < state.envelopes.size(); ++shell)
    {
        for (std::size_t mu = 0; mu < state.envelopes[shell].functionCount; ++mu)
        {
            const std::size_t row = state.envelopes[shell].aoOffset + mu;

            for (std::size_t nu = 0; nu < aoCount; ++nu)
            {
                weightsAlpha[shell] += std::abs(
                    densityAlpha(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(nu)));
                weightsBeta[shell] += std::abs(
                    densityBeta(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(nu)));
            }
        }
    }

    std::vector<double> values(aoCount);
    std::vector<double> gradients(3 * aoCount);

    ReferenceResult result;

    for (const PointSpec& spec : points)
    {
        const std::array<double, 3> position = [&] {
            const std::array<double, 3>& center = state.geometry.atoms[owner].position;
            return std::array<double, 3>{
                center[0] + spec.offset[0], center[1] + spec.offset[1], center[2] + spec.offset[2]};
        }();

        state.evaluator.EvaluateGradients(position, values, gradients);

        std::vector<std::size_t> slots;

        for (std::size_t shell = 0; shell < state.envelopes.size(); ++shell)
        {
            const double weight = std::max(weightsAlpha[shell], weightsBeta[shell]);

            if (!ShellIsSignificant(state.envelopes[shell], position, weight, tolerance))
            {
                continue;
            }

            ++result.keptShells;

            for (std::size_t ao = 0; ao < state.envelopes[shell].functionCount; ++ao)
            {
                slots.push_back(state.envelopes[shell].aoOffset + ao);
            }
        }

        if (slots.empty())
        {
            continue;
        }

        ++result.visitedPoints;

        double rhoAlpha = 0.0;
        double rhoBeta = 0.0;
        std::array<double, 3> gradientAlpha{};
        std::array<double, 3> gradientBeta{};

        FieldsAtPoint(densityAlpha, values, gradients, slots, rhoAlpha, gradientAlpha);
        FieldsAtPoint(densityBeta, values, gradients, slots, rhoBeta, gradientBeta);

        const Eigen::Vector3d alpha(gradientAlpha[0], gradientAlpha[1], gradientAlpha[2]);
        const Eigen::Vector3d beta(gradientBeta[0], gradientBeta[1], gradientBeta[2]);

        const excgrid::XcKernelValue kernel = functional.Evaluate(
            rhoAlpha, rhoBeta, alpha.dot(alpha), alpha.dot(beta), beta.dot(beta));

        result.energy += PointWeight(state.geometry, owner, position) * kernel.exc;
    }

    return result;
}

// Both atoms' points, as the walk walks them: one block per atom, in atom order.
std::vector<excgrid::Block> MakeBlocks(const GeometryState& state) {
    const std::array<std::span<const PointSpec>, 2> pointSets = {
        std::span<const PointSpec>(kPointsAtomZero), std::span<const PointSpec>(kPointsAtomOne)};
    std::vector<excgrid::Block> blocks;

    for (std::size_t atom = 0; atom < state.geometry.atoms.size(); ++atom)
    {
        excgrid::Block block;
        block.pointCount = pointSets[atom].size();

        for (const PointSpec& spec : pointSets[atom])
        {
            const std::array<double, 3>& center = state.geometry.atoms[atom].position;
            const std::array<double, 3> position = {
                center[0] + spec.offset[0], center[1] + spec.offset[1], center[2] + spec.offset[2]};
            block.points.push_back(position);
            block.weights.push_back(PointWeight(state.geometry, atom, position));
            block.atomIndex.push_back(atom);
        }

        blocks.push_back(std::move(block));
    }

    return blocks;
}

// The energy the fixture integrates at one geometry, over both blocks.
ReferenceResult ReferenceEnergyAll(const GeometryState& state,
                                   const excgrid::XcFunctional& functional,
                                   const Eigen::MatrixXd& densityAlpha,
                                   const Eigen::MatrixXd& densityBeta,
                                   double tolerance) {
    const ReferenceResult zero = ReferenceEnergy(state,
                                                 std::span<const PointSpec>(kPointsAtomZero),
                                                 0,
                                                 functional,
                                                 densityAlpha,
                                                 densityBeta,
                                                 tolerance);
    const ReferenceResult one = ReferenceEnergy(state,
                                                std::span<const PointSpec>(kPointsAtomOne),
                                                1,
                                                functional,
                                                densityAlpha,
                                                densityBeta,
                                                tolerance);

    return ReferenceResult{zero.energy + one.energy,
                           zero.keptShells + one.keptShells,
                           zero.visitedPoints + one.visitedPoints};
}

// ---------------------------------------------------------------------------
// Shared fixture values
// ---------------------------------------------------------------------------

constexpr std::array<std::array<double, 3>, 2> kPositions = {kAtomZeroPosition, kAtomOnePosition};

// The H2/STO-3G densities this fixture differentiates. They are deliberately
// NOT the symmetric closed-shell pair: with D = c c^T both AOs contract
// identically, every sum over the kept block becomes symmetric, and an indexing
// error in the orbital-motion sums would be invisible on it. Both matrices are
// positive semidefinite, so rho >= 0 wherever the selection leaves a shell, and
// they differ between the spins, which puts the sigma_ab cross term on the path.
Eigen::MatrixXd AlphaDensity() {
    Eigen::MatrixXd density(2, 2);
    density << 0.70, 0.30, 0.30, 0.45;

    return density;
}

Eigen::MatrixXd BetaDensity() {
    Eigen::MatrixXd density(2, 2);
    density << 0.35, 0.15, 0.15, 0.40;

    return density;
}

// The screened walk's tolerance. It is a fixture value, not a production one:
// the significance values on this point set run from 1.07 to 1.92, so 1.1 drops
// exactly the one shell that the test needs to see dropped and keeps every
// other, and 1.5 below drops four of the nine. The reuse of a tolerance means
// nothing unless the tolerance bites.
constexpr double kTolerance = 0.7;
constexpr double kLooserTolerance = 0.9;

// The step of the central difference. At 1e-5 the truncation term is far below
// the roundoff floor of a difference of energies of order 1e-2.
constexpr double kFiniteDifferenceStep = 1e-5;

// The same step on the engine's own grid, where the energy is of order one: the
// roundoff floor of the difference is a hundred times higher there, so the two
// terms balance elsewhere and the step is chosen for that grid rather than
// shared.
constexpr double kEngineFiniteDifferenceStep = 1e-4;

// A number as the test records it. The recorded values are the numbers whoever
// reports this run quotes, so they are written with the precision the
// measurement deserves rather than the six decimals a default conversion keeps:
// an agreement that reads "0.000000" says nothing about how good it is.
std::string Scientific(double value) {
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.3e", value);

    return std::string(buffer.data());
}

} // namespace

// The selected second-derivative tier against the dense one: the same numbers on
// the slots it is given, and untouched slots everywhere else. The gradient walk
// reads a point's curvature through this tier, so a selection that wrote the
// wrong slots would put another atom's second derivative on this atom's
// coordinates, and a tier that wrote nothing would leave the walk reading
// whatever the buffer held.
TEST(XcGradientTest, TheSelectedSecondDerivativeTierMatchesTheDenseOne) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const auto state = MakeGeometryState(kPositions, *basis);
    ASSERT_TRUE(state.has_value());

    const std::size_t aoCount = state->evaluator.AOCount();
    const double sentinel = -12345.678;
    std::vector<double> denseValues(aoCount);
    std::vector<double> denseGradients(3 * aoCount);
    std::vector<double> denseHessians(6 * aoCount);
    std::vector<double> values(aoCount, sentinel);
    std::vector<double> gradients(3 * aoCount, sentinel);
    std::vector<double> hessians(6 * aoCount, sentinel);
    const std::array<double, 3> point = state->geometry.atoms[0].position;

    state->evaluator.EvaluateDerivatives(point, denseValues, denseGradients, denseHessians);

    // One slot only, and it is the one the selection names: every other slot of
    // every array must still hold the sentinel.
    const std::array<std::size_t, 1> selection = {1};
    state->evaluator.EvaluateDerivativesSelected(point, selection, values, gradients, hessians);

    for (std::size_t slot = 0; slot < 3 * aoCount; ++slot)
    {
        if (slot >= 3 && slot < 6)
        {
            EXPECT_DOUBLE_EQ(gradients[slot], denseGradients[slot]) << "slot " << slot;
        } else
        {
            EXPECT_DOUBLE_EQ(gradients[slot], sentinel) << "slot " << slot;
        }
    }

    for (std::size_t slot = 0; slot < 6 * aoCount; ++slot)
    {
        if (slot >= 6 && slot < 12)
        {
            EXPECT_DOUBLE_EQ(hessians[slot], denseHessians[slot]) << "slot " << slot;
        } else
        {
            EXPECT_DOUBLE_EQ(hessians[slot], sentinel) << "slot " << slot;
        }
    }

    EXPECT_DOUBLE_EQ(values[1], denseValues[1]);
    EXPECT_DOUBLE_EQ(values[0], sentinel);
}

// I1: the assembled gradient against a central difference of the fixture's own
// energy, on the dense walk (tolerance zero keeps every shell whose weight and
// decay are nonzero, which on this fixture is every shell at every near point).
TEST(XcGradientTest, TheAssembledGradientMatchesAFiniteDifferenceOfTheFixturesEnergy) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    for (const std::string_view name : {std::string_view{"pbe"}, std::string_view{"slater"}})
    {
        const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);
        ASSERT_NE(functional, nullptr) << name;

        const auto state = MakeGeometryState(kPositions, *basis);
        ASSERT_TRUE(state.has_value()) << name;

        const auto blocks = MakeBlocks(*state);
        const Eigen::MatrixXd alpha = AlphaDensity();
        const Eigen::MatrixXd beta = BetaDensity();

        const auto assembled = EvaluateXcGradient(blocks,
                                                  FixtureProvider(state->geometry),
                                                  state->evaluator,
                                                  *functional,
                                                  state->envelopes,
                                                  XcIntegrationThresholds::FromEnergyPath({}, 0.0),
                                                  {},
                                                  alpha,
                                                  beta);
        ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

        const std::size_t coordinates = 3U * kPositions.size();
        ASSERT_EQ(assembled->gradient.size(), static_cast<Eigen::Index>(coordinates));

        double maxDeviation = 0.0;

        for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
        {
            const std::size_t atom = coordinate / 3U;
            const std::size_t axis = coordinate % 3U;

            auto plus = kPositions;
            plus[atom][axis] += kFiniteDifferenceStep;
            auto minus = kPositions;
            minus[atom][axis] -= kFiniteDifferenceStep;

            const auto plusState = MakeGeometryState(plus, *basis);
            const auto minusState = MakeGeometryState(minus, *basis);
            ASSERT_TRUE(plusState.has_value() && minusState.has_value()) << name;

            const ReferenceResult plusEnergy =
                ReferenceEnergyAll(*plusState, *functional, alpha, beta, 0.0);
            const ReferenceResult minusEnergy =
                ReferenceEnergyAll(*minusState, *functional, alpha, beta, 0.0);

            // The selection must not move with the displacement: a point or a
            // shell switching in or out makes the difference discontinuous and
            // the comparison meaningless, and nothing else in this test would
            // say so.
            ASSERT_EQ(plusEnergy.keptShells, minusEnergy.keptShells) << name;
            ASSERT_EQ(plusEnergy.visitedPoints, minusEnergy.visitedPoints) << name;

            const double difference =
                (plusEnergy.energy - minusEnergy.energy) / (2.0 * kFiniteDifferenceStep);
            maxDeviation = std::max(
                maxDeviation,
                std::abs(assembled->gradient[static_cast<Eigen::Index>(coordinate)] - difference));
        }

        RecordProperty(std::string(name) + " agreement (max deviation, Hartree/Bohr)",
                       Scientific(maxDeviation));

        EXPECT_LT(maxDeviation, 1e-9) << name
                                      << ": the assembled gradient and the difference of "
                                         "the fixture's own energy disagree";
    }
}

// I1 again, on the walk that ships: the screened selection, with the reference
// energy's selection reproduced from the same public rule. A shell the walk
// drops is a term the energy drops too, so the two stay the derivative of one
// another rather than of different energies.
TEST(XcGradientTest, TheScreenedWalkIsTheDerivativeOfItsOwnScreenedEnergy) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const excgrid::XcFunctional* functional = excgrid::FindFunctional("pbe");
    ASSERT_NE(functional, nullptr);

    const auto state = MakeGeometryState(kPositions, *basis);
    ASSERT_TRUE(state.has_value());

    const auto blocks = MakeBlocks(*state);
    const Eigen::MatrixXd alpha = AlphaDensity();
    const Eigen::MatrixXd beta = BetaDensity();

    const auto assembled =
        EvaluateXcGradient(blocks,
                           FixtureProvider(state->geometry),
                           state->evaluator,
                           *functional,
                           state->envelopes,
                           XcIntegrationThresholds::FromEnergyPath({}, kTolerance),
                           {},
                           alpha,
                           beta);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    // The fixture is only a fixture if the tolerance actually drops something:
    // a screened walk that keeps every shell would prove nothing about
    // screening.
    const ReferenceResult reference =
        ReferenceEnergyAll(*state, *functional, alpha, beta, kTolerance);
    EXPECT_LT(reference.keptShells, 6U * state->envelopes.size())
        << "the fixture keeps every shell at every point: the tolerance does not bite";
    EXPECT_LT(reference.visitedPoints, 6U) << "the fixture visits every point";
    EXPECT_EQ(assembled->counts.shellKeeps, reference.keptShells);
    EXPECT_EQ(assembled->counts.aoFetches, reference.visitedPoints);

    const std::size_t coordinates = 3U * kPositions.size();
    double maxDeviation = 0.0;

    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
    {
        const std::size_t atom = coordinate / 3U;
        const std::size_t axis = coordinate % 3U;

        auto plus = kPositions;
        plus[atom][axis] += kFiniteDifferenceStep;
        auto minus = kPositions;
        minus[atom][axis] -= kFiniteDifferenceStep;

        const auto plusState = MakeGeometryState(plus, *basis);
        const auto minusState = MakeGeometryState(minus, *basis);
        ASSERT_TRUE(plusState.has_value() && minusState.has_value());

        const ReferenceResult plusEnergy =
            ReferenceEnergyAll(*plusState, *functional, alpha, beta, kTolerance);
        const ReferenceResult minusEnergy =
            ReferenceEnergyAll(*minusState, *functional, alpha, beta, kTolerance);

        ASSERT_EQ(plusEnergy.keptShells, reference.keptShells);
        ASSERT_EQ(minusEnergy.keptShells, reference.keptShells);

        const double difference =
            (plusEnergy.energy - minusEnergy.energy) / (2.0 * kFiniteDifferenceStep);
        maxDeviation = std::max(
            maxDeviation,
            std::abs(assembled->gradient[static_cast<Eigen::Index>(coordinate)] - difference));
    }

    RecordProperty("screened agreement (max deviation, Hartree/Bohr)", Scientific(maxDeviation));
    EXPECT_LT(maxDeviation, 1e-9);
}

// I2: the walk runs under the energy path's thresholds, asserted rather than
// assumed. The echo is the assertion's left side; the counts and the energy
// below are its behavioural side, because a walk that echoes a threshold it did
// not use would pass a comparison of labels and fail a comparison of results.
TEST(XcGradientTest, TheGradientWalkReusesTheEnergyPathsThresholds) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const excgrid::XcFunctional* functional = excgrid::FindFunctional("pbe");
    ASSERT_NE(functional, nullptr);

    const auto state = MakeGeometryState(kPositions, *basis);
    ASSERT_TRUE(state.has_value());

    const auto blocks = MakeBlocks(*state);
    const Eigen::MatrixXd alpha = AlphaDensity();
    const Eigen::MatrixXd beta = BetaDensity();

    // What the energy path ran under: the grid settings it was built with and
    // the tolerance its caller passed.
    XcGridSettings settings;
    settings.trimWeight = 1e-14;
    settings.radialPoints = 75;
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(settings, kTolerance);

    const auto assembled = EvaluateXcGradient(blocks,
                                              FixtureProvider(state->geometry),
                                              state->evaluator,
                                              *functional,
                                              state->envelopes,
                                              thresholds,
                                              {},
                                              alpha,
                                              beta);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    EXPECT_DOUBLE_EQ(assembled->thresholds.trimWeight, settings.trimWeight);
    EXPECT_DOUBLE_EQ(assembled->thresholds.screeningTolerance, kTolerance);

    // The behavioural half: with the SAME thresholds the reference walks, the
    // walk's shell selection and its energy are the reference's, so the reuse is
    // of the numbers and not of a label.
    const ReferenceResult reference =
        ReferenceEnergyAll(*state, *functional, alpha, beta, kTolerance);
    EXPECT_EQ(assembled->counts.shellKeeps, reference.keptShells);
    EXPECT_EQ(assembled->counts.aoFetches, reference.visitedPoints);
    EXPECT_NEAR(assembled->energy, reference.energy, 1e-12 * std::abs(reference.energy) + 1e-18);

    // And the counter-check the test exists for: running the same walk under a
    // DIFFERENT tolerance moves both, which is what makes the identity above a
    // statement about the run rather than about the fixture.
    const ReferenceResult loose =
        ReferenceEnergyAll(*state, *functional, alpha, beta, kLooserTolerance);
    EXPECT_NE(loose.keptShells, reference.keptShells)
        << "a looser tolerance selects the same shells: this fixture cannot see the difference the "
           "assertion is about";

    const XcIntegrationThresholds looser =
        XcIntegrationThresholds::FromEnergyPath(settings, kLooserTolerance);
    const auto underLooser = EvaluateXcGradient(blocks,
                                                FixtureProvider(state->geometry),
                                                state->evaluator,
                                                *functional,
                                                state->envelopes,
                                                looser,
                                                {},
                                                alpha,
                                                beta);
    ASSERT_TRUE(underLooser.has_value()) << underLooser.error().message;
    EXPECT_EQ(underLooser->counts.shellKeeps, loose.keptShells);
    EXPECT_NE(underLooser->counts.shellKeeps, assembled->counts.shellKeeps)
        << "the tolerance reaches the reference but not the walk";
    EXPECT_DOUBLE_EQ(underLooser->thresholds.screeningTolerance, kLooserTolerance);
}

// I3: the energy path against the gradient path, as its own line. The gradient
// walk reports the energy it integrated so that the two can be compared
// directly, and this test reports the difference rather than folding it into the
// agreement number above.
TEST(XcGradientTest, TheConsistencyLineIsReportedSeparately) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const excgrid::XcFunctional* functional = excgrid::FindFunctional("pbe");
    ASSERT_NE(functional, nullptr);

    const auto state = MakeGeometryState(kPositions, *basis);
    ASSERT_TRUE(state.has_value());

    const auto blocks = MakeBlocks(*state);
    const Eigen::MatrixXd alpha = AlphaDensity();
    const Eigen::MatrixXd beta = BetaDensity();

    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(XcGridSettings{}, kTolerance);

    const auto assembled = EvaluateXcGradient(blocks,
                                              FixtureProvider(state->geometry),
                                              state->evaluator,
                                              *functional,
                                              state->envelopes,
                                              thresholds,
                                              {},
                                              alpha,
                                              beta);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    const ReferenceResult energyPath =
        ReferenceEnergyAll(*state, *functional, alpha, beta, kTolerance);
    const double consistency = std::abs(energyPath.energy - assembled->energy);

    RecordProperty("consistency line (energy path against gradient path, Hartree)",
                   Scientific(consistency));
    EXPECT_LT(consistency, 1e-12 * std::abs(energyPath.energy) + 1e-18)
        << "the gradient path integrated an energy the energy path does not recognise";

    // The line has teeth: under a threshold the energy path did not use, the
    // same comparison is large. A consistency line that cannot move is not a
    // measurement.
    const ReferenceResult otherPath =
        ReferenceEnergyAll(*state, *functional, alpha, beta, kLooserTolerance);
    EXPECT_GT(std::abs(otherPath.energy - assembled->energy), 1e-3 * std::abs(energyPath.energy))
        << "the consistency line does not move when the thresholds do";
}

// The provider's refusal is returned, not answered with zeros: a gradient built
// on a block the geometry does not produce is a wrong force that looks
// plausible, which is what the refusal exists to prevent.
TEST(XcGradientTest, AProviderRefusalIsReturnedByName) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const excgrid::XcFunctional* functional = excgrid::FindFunctional("pbe");
    ASSERT_NE(functional, nullptr);

    const auto state = MakeGeometryState(kPositions, *basis);
    ASSERT_TRUE(state.has_value());

    const auto blocks = MakeBlocks(*state);
    const Eigen::MatrixXd alpha = AlphaDensity();
    const Eigen::MatrixXd beta = BetaDensity();

    FixtureProvider provider(state->geometry);
    provider.RefuseEverything();

    const auto assembled =
        EvaluateXcGradient(blocks,
                           provider,
                           state->evaluator,
                           *functional,
                           state->envelopes,
                           XcIntegrationThresholds::FromEnergyPath({}, kTolerance),
                           {},
                           alpha,
                           beta);

    ASSERT_FALSE(assembled.has_value());
    EXPECT_NE(assembled.error().message.find("not a point set this geometry's grid produces"),
              std::string::npos)
        << assembled.error().message;
}

// The whole fixture is translation invariant - the point set rides its atoms,
// the weights see the geometry only through a difference, and the AOs are
// centered on the atoms - so the gradient's components must sum to zero. An
// invariant the assembly cannot see is a check the finite difference of one
// coordinate cannot make.
TEST(XcGradientTest, TheGradientSumsToZeroUnderTranslation) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const excgrid::XcFunctional* functional = excgrid::FindFunctional("pbe");
    ASSERT_NE(functional, nullptr);

    const auto state = MakeGeometryState(kPositions, *basis);
    ASSERT_TRUE(state.has_value());

    const auto blocks = MakeBlocks(*state);
    const Eigen::MatrixXd alpha = AlphaDensity();
    const Eigen::MatrixXd beta = BetaDensity();

    const auto assembled =
        EvaluateXcGradient(blocks,
                           FixtureProvider(state->geometry),
                           state->evaluator,
                           *functional,
                           state->envelopes,
                           XcIntegrationThresholds::FromEnergyPath({}, kTolerance),
                           {},
                           alpha,
                           beta);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    double sum = 0.0;

    for (Eigen::Index component = 0; component < assembled->gradient.size(); ++component)
    {
        sum += assembled->gradient[component];
    }

    EXPECT_LT(std::abs(sum), 1e-14) << "the assembled gradient is not translation invariant";
}

// A real run's grid: the engine, and the derivative provider built for the same
// geometry and the same build parameters. The two come from one pair, which is
// what makes the provider answer for exactly the blocks the engine walks.
struct RealGrid {
    XcGridEngine engine;
    std::unique_ptr<excgrid::GridDerivativeProvider> provider;
};

// A coarse grid: the engine's own blocks, small enough to walk many times in one
// test, and coarse enough that the screens bite at its outer points.
XcGridSettings RealGridSettings() {
    XcGridSettings settings;
    settings.radialPoints = 20;
    settings.blockTarget = 256;

    return settings;
}

constexpr double kEngineTolerance = 1e-8;

std::optional<RealGrid> MakeRealGrid(const std::array<std::array<double, 3>, 2>& positions,
                                     const qcx::basisset::BasisSet& basis,
                                     const XcGridSettings& settings) {
    auto state = MakeGeometryState(positions, basis);

    if (!state.has_value())
    {
        return std::nullopt;
    }

    auto engine = XcGridEngine::Create(state->molecule, basis, "pbe", settings);

    if (!engine.has_value())
    {
        return std::nullopt;
    }

    auto provider =
        excgrid::CreateGridDerivativeProvider(state->geometry, ToExcgridParams(settings));

    if (!provider.has_value())
    {
        return std::nullopt;
    }

    return RealGrid{std::move(*engine), std::move(*provider)};
}

// A finite difference of the engine's own energy, at the grid the engine rebuilds
// for each displaced geometry: the displacement moves the atoms, the product
// grid rides them, and the partition reweighs every point, which is exactly what
// the assembled gradient claims to differentiate.
double EngineEnergyDifference(const std::array<std::array<double, 3>, 2>& positions,
                              std::size_t atom,
                              std::size_t axis,
                              const qcx::basisset::BasisSet& basis,
                              const XcGridSettings& settings,
                              const Eigen::MatrixXd& alpha,
                              const Eigen::MatrixXd& beta,
                              std::size_t& keptShells) {
    auto plus = positions;
    plus[atom][axis] += kEngineFiniteDifferenceStep;
    auto minus = positions;
    minus[atom][axis] -= kEngineFiniteDifferenceStep;

    const auto plusGrid = MakeRealGrid(plus, basis, settings);
    const auto minusGrid = MakeRealGrid(minus, basis, settings);
    EXPECT_TRUE(plusGrid.has_value() && minusGrid.has_value());

    if (!plusGrid.has_value() || !minusGrid.has_value())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const auto plusEnergy = plusGrid->engine.EvaluateScreened(alpha, beta, kEngineTolerance, {});
    const auto minusEnergy = minusGrid->engine.EvaluateScreened(alpha, beta, kEngineTolerance, {});
    EXPECT_TRUE(plusEnergy.has_value() && minusEnergy.has_value());

    if (!plusEnergy.has_value() || !minusEnergy.has_value())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    // The grid and the selection must not move with the displacement: a point
    // trimmed in or out, or a shell switching on or off, makes the difference
    // discontinuous, and nothing else here would say so.
    EXPECT_EQ(plusGrid->engine.PointCount(), minusGrid->engine.PointCount());
    EXPECT_EQ(plusEnergy->counts.shellKeeps, minusEnergy->counts.shellKeeps);
    keptShells = plusEnergy->counts.shellKeeps;

    return (plusEnergy->energy - minusEnergy->energy) / (2.0 * kEngineFiniteDifferenceStep);
}

// I3 on the assembly a run uses: the engine's own grid, its own envelopes, its
// own functional, the library's own provider, and the thresholds its energy path
// ran under. The walk has to pick the same shells and integrate the same energy
// the energy path does, to the last bit: a walk that screened or truncated
// differently would integrate a different number, and no finite-difference
// agreement would say so.
TEST(XcGradientTest, TheEngineWalkIntegratesTheEnergyPathsOwnEnergy) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const XcGridSettings settings = RealGridSettings();
    const auto grid = MakeRealGrid(kPositions, *basis, settings);
    ASSERT_TRUE(grid.has_value());

    const Eigen::MatrixXd alpha = AlphaDensity();
    const Eigen::MatrixXd beta = BetaDensity();
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(settings, kEngineTolerance);
    const XcBatchSettings batch;

    const auto energyPath =
        grid->engine.EvaluateScreened(alpha, beta, thresholds.screeningTolerance, batch);
    ASSERT_TRUE(energyPath.has_value()) << energyPath.error().message;

    const auto gradient =
        EvaluateXcGradient(grid->engine, alpha, beta, *grid->provider, thresholds, batch);
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    ASSERT_EQ(gradient->gradient.size(), 3 * static_cast<Eigen::Index>(2));

    // The selection, point for point: the walk and the energy path have to agree
    // about which shells were dropped, or the energy below is a coincidence.
    EXPECT_EQ(gradient->counts.points, energyPath->counts.points);
    EXPECT_EQ(gradient->counts.shellKeeps, energyPath->counts.shellKeeps);
    EXPECT_EQ(gradient->counts.shellTests, energyPath->counts.shellTests);
    EXPECT_EQ(gradient->counts.aoFetches, energyPath->counts.aoFetches);
    EXPECT_EQ(gradient->counts.aoSlotsEvaluated, energyPath->counts.aoSlotsEvaluated);
    EXPECT_EQ(gradient->counts.contractionPairs, energyPath->counts.contractionPairs);
    EXPECT_GT(gradient->counts.shellKeeps, 0U) << "the walk selected nothing";
    EXPECT_LT(gradient->counts.shellKeeps, gradient->counts.shellTests)
        << "the tolerance drops nothing on this grid: the walk is dense here";

    const double consistency = std::abs(energyPath->energy - gradient->energy);
    RecordProperty("the engine's own consistency line (Hartree)", Scientific(consistency));
    EXPECT_DOUBLE_EQ(energyPath->energy, gradient->energy)
        << "the gradient path integrated an energy the energy path does not recognise";

    EXPECT_DOUBLE_EQ(gradient->thresholds.screeningTolerance, thresholds.screeningTolerance);
    EXPECT_DOUBLE_EQ(gradient->thresholds.trimWeight, thresholds.trimWeight);
}

// I1 on the assembly a run uses: the engine's own grid and the library's own
// provider, against a central difference of the engine's own energy at displaced
// geometries. This is the number the assembly exists to produce.
TEST(XcGradientTest, TheRealGridGradientMatchesAFiniteDifferenceOfTheRealEnergy) {
    const auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const XcGridSettings settings = RealGridSettings();
    const auto grid = MakeRealGrid(kPositions, *basis, settings);
    ASSERT_TRUE(grid.has_value());

    const Eigen::MatrixXd alpha = AlphaDensity();
    const Eigen::MatrixXd beta = BetaDensity();
    const XcIntegrationThresholds thresholds =
        XcIntegrationThresholds::FromEnergyPath(settings, kEngineTolerance);
    const XcBatchSettings batch;

    const auto assembled =
        EvaluateXcGradient(grid->engine, alpha, beta, *grid->provider, thresholds, batch);
    ASSERT_TRUE(assembled.has_value()) << assembled.error().message;

    const std::size_t coordinates = 3U * kPositions.size();
    ASSERT_EQ(assembled->gradient.size(), static_cast<Eigen::Index>(coordinates));

    const auto centralEnergy =
        grid->engine.EvaluateScreened(alpha, beta, thresholds.screeningTolerance, batch);
    ASSERT_TRUE(centralEnergy.has_value()) << centralEnergy.error().message;

    double maxDeviation = 0.0;

    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate)
    {
        const std::size_t atom = coordinate / 3U;
        const std::size_t axis = coordinate % 3U;
        std::size_t keptShells = 0;
        const double difference = EngineEnergyDifference(
            kPositions, atom, axis, *basis, settings, alpha, beta, keptShells);
        ASSERT_FALSE(std::isnan(difference));
        ASSERT_EQ(keptShells, centralEnergy->counts.shellKeeps)
            << "the selection moved with the displacement";

        maxDeviation = std::max(
            maxDeviation,
            std::abs(assembled->gradient[static_cast<Eigen::Index>(coordinate)] - difference));
    }

    RecordProperty("real grid agreement (max deviation, Hartree/Bohr)", Scientific(maxDeviation));
    EXPECT_LT(maxDeviation, 1e-8) << "the assembled gradient and the engine's own energy disagree";

    // An invariant no finite difference of this test can see: the grid rides its
    // atoms and its weights see the geometry only through differences, so the
    // gradient carries no net force. A systematic error in either of the
    // provider's arrays leaves one, and the finite difference above would still
    // look reasonable.
    double sum = 0.0;

    for (Eigen::Index component = 0; component < assembled->gradient.size(); ++component)
    {
        sum += assembled->gradient[component];
    }

    RecordProperty("real grid net force (Hartree/Bohr)", Scientific(std::abs(sum)));
    EXPECT_LT(std::abs(sum), 1e-12) << "the real grid's gradient carries a net force";
}

} // namespace qcx::grid
