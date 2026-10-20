// The fixed-density derivative of the XC potential in the occupied-virtual MO
// block, against displaced-geometry potentials:
//   - the density matrix is held fixed in the AO basis while the nuclei move,
//     which is the convention the block claims;
//   - the reference MO coefficients contract both sides, so the comparison is
//     the potential's own derivative and not an orbital-relaxation effect;
//   - every atom and axis is walked, so no explicit coordinate channel can be
//     left out silently.
// The GGA run exercises the sigma channels and the tau run the kinetic-energy
// density's; the LDA run is the control with no gradient channel at all.

#include "h2o_sto3g.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/grid/xc_kernel_gradient.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <excgrid/grid_derivatives.hpp>
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

// The central difference's step; the Richardson combination of the step and its
// half removes the second-order term. The potential and its low derivatives are
// of order one, so this leaves a floor far below the gate.
constexpr double kFiniteDifferenceStep = 1e-4;
constexpr double kFiniteDifferenceTolerance = 1e-7;

std::string Scientific(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6e", value);

    return buffer;
}

// The same deterministic rotation the action test uses: an orthogonal basis
// with no canonical structure. The block reads only the orbitals' coefficients.
Eigen::MatrixXd RotationMatrix(std::size_t size) {
    Eigen::MatrixXd rotation =
        Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(size), static_cast<Eigen::Index>(size));
    double angle = 0.4;

    for (std::size_t left = 0; left < size; ++left)
    {
        for (std::size_t right = left + 1; right < size; ++right)
        {
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            angle += 0.37;

            for (std::size_t row = 0; row < size; ++row)
            {
                const double first =
                    rotation(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(left));
                const double second =
                    rotation(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(right));
                rotation(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(left)) =
                    cosine * first - sine * second;
                rotation(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(right)) =
                    sine * first + cosine * second;
            }
        }
    }

    return rotation;
}

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

    if (!displaced.has_value())
    {
        return std::nullopt;
    }

    // The molecule renumbers canonically; a reordering would make the caller's
    // atom index name a different atom, so it is refused here rather than
    // silently differentiated.
    if (displaced->AtomCount() != atomCount)
    {
        return std::nullopt;
    }

    for (std::size_t index = 0; index < atomCount; ++index)
    {
        if (displaced->Atoms()[index].atomicNumber != reference.Atoms()[index].atomicNumber)
        {
            return std::nullopt;
        }
    }

    return std::move(*displaced);
}

struct GradientFixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    XcGridSettings settings;
    std::string functionalName;
    XcGridEngine engine;
    std::unique_ptr<excgrid::GridDerivativeProvider> provider;
    Eigen::MatrixXd orbitals; ///< AOCount() x (o + v), occupied columns first.
    Eigen::MatrixXd density; ///< 2 C_o C_o^T.
    std::size_t numOccupied = 0;
    std::size_t numVirtual = 0;
};

std::optional<GradientFixture> MakeGradientFixture(std::string_view functionalName) {
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

    auto provider = excgrid::CreateGridDerivativeProvider(*geometry, ToExcgridParams(settings));

    if (!provider.has_value())
    {
        return std::nullopt;
    }

    const std::size_t aoCount = engine->Evaluator().AOCount();
    const std::size_t numOccupied = static_cast<std::size_t>(molecule->ElectronCount()) / 2;
    const std::size_t numVirtual = aoCount - numOccupied;

    Eigen::MatrixXd orbitals = RotationMatrix(aoCount);
    const Eigen::MatrixXd occupied = orbitals.leftCols(static_cast<Eigen::Index>(numOccupied));
    const Eigen::MatrixXd density = 2.0 * occupied * occupied.transpose();

    return GradientFixture{std::move(*molecule),
                           std::move(*basis),
                           settings,
                           std::string(functionalName),
                           std::move(*engine),
                           std::move(*provider),
                           std::move(orbitals),
                           density,
                           numOccupied,
                           numVirtual};
}

// The occupied-virtual potential block at one displaced geometry, at the fixed
// density and against the reference orbitals.
std::optional<Eigen::MatrixXd> DisplacedBlock(const GradientFixture& fixture,
                                              std::size_t atom,
                                              std::size_t axis,
                                              double delta) {
    auto displaced = DisplacedMolecule(fixture.molecule, atom, axis, delta);

    if (!displaced.has_value())
    {
        return std::nullopt;
    }

    auto engine =
        XcGridEngine::Create(*displaced, fixture.basis, fixture.functionalName, fixture.settings);

    if (!engine.has_value())
    {
        return std::nullopt;
    }

    if (engine->PointCount() != fixture.engine.PointCount())
    {
        return std::nullopt;
    }

    const auto evaluated = engine->EvaluateClosedShell(fixture.density);

    if (!evaluated.has_value())
    {
        return std::nullopt;
    }

    const Eigen::MatrixXd potential = 0.5 * (evaluated->potentialAlpha + evaluated->potentialBeta);
    const Eigen::MatrixXd moPotential = fixture.orbitals.transpose() * potential * fixture.orbitals;

    Eigen::MatrixXd block(static_cast<Eigen::Index>(fixture.numOccupied),
                          static_cast<Eigen::Index>(fixture.numVirtual));

    for (std::size_t occupied = 0; occupied < fixture.numOccupied; ++occupied)
    {
        for (std::size_t virtualOrbital = 0; virtualOrbital < fixture.numVirtual; ++virtualOrbital)
        {
            block(static_cast<Eigen::Index>(occupied), static_cast<Eigen::Index>(virtualOrbital)) =
                moPotential(static_cast<Eigen::Index>(fixture.numOccupied + virtualOrbital),
                            static_cast<Eigen::Index>(occupied));
        }
    }

    return block;
}

// The block's derivative by a Richardson-extrapolated central difference.
std::optional<Eigen::MatrixXd> NumericalBlock(const GradientFixture& fixture,
                                              std::size_t atom,
                                              std::size_t axis) {
    const double step = kFiniteDifferenceStep;
    const auto plus = DisplacedBlock(fixture, atom, axis, step);
    const auto minus = DisplacedBlock(fixture, atom, axis, -step);
    const auto halfPlus = DisplacedBlock(fixture, atom, axis, 0.5 * step);
    const auto halfMinus = DisplacedBlock(fixture, atom, axis, -0.5 * step);

    if (!plus.has_value() || !minus.has_value() || !halfPlus.has_value() || !halfMinus.has_value())
    {
        return std::nullopt;
    }

    const Eigen::MatrixXd coarse = (*plus - *minus) / (2.0 * step);
    const Eigen::MatrixXd fine = (*halfPlus - *halfMinus) / step;

    return (4.0 * fine - coarse) / 3.0;
}

// The gate: every coordinate channel of every entry against the displaced
// potentials.
void CheckFunctional(std::string_view name) {
    const auto fixture = MakeGradientFixture(name);
    ASSERT_TRUE(fixture.has_value()) << name;

    const auto block = EvaluateXcKernelDerivativeBlock(fixture->engine,
                                                       *fixture->provider,
                                                       fixture->orbitals,
                                                       fixture->density,
                                                       fixture->numOccupied,
                                                       fixture->numVirtual,
                                                       XcKernelSettings{kScreeningTolerance});
    ASSERT_TRUE(block.has_value()) << name << ": " << block.error().message;
    ASSERT_EQ(block->values.size(),
              3 * fixture->molecule.AtomCount() * fixture->numOccupied * fixture->numVirtual);

    double maxDeviation = 0.0;

    for (std::size_t atom = 0; atom < fixture->molecule.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const std::size_t coordinate = 3 * atom + axis;
            const auto numerical = NumericalBlock(*fixture, atom, axis);
            ASSERT_TRUE(numerical.has_value()) << name << ": displacement " << coordinate;

            for (std::size_t occupied = 0; occupied < fixture->numOccupied; ++occupied)
            {
                for (std::size_t virtualOrbital = 0; virtualOrbital < fixture->numVirtual;
                     ++virtualOrbital)
                {
                    const double difference =
                        (*numerical)(static_cast<Eigen::Index>(occupied),
                                     static_cast<Eigen::Index>(virtualOrbital));
                    const double analytic = block->At(coordinate, occupied, virtualOrbital);
                    maxDeviation = std::max(maxDeviation, std::abs(difference - analytic));
                }
            }
        }
    }

    ::testing::Test::RecordProperty(std::string(name) + " agreement (max deviation, Hartree/Bohr)",
                                    Scientific(maxDeviation));
    EXPECT_LT(maxDeviation, kFiniteDifferenceTolerance)
        << name << ": the block and the displaced-geometry potential disagree";
}

TEST(XcKernelGradientTest, TheBlockMatchesTheDisplacedPotentialForALda) {
    CheckFunctional("slater");
}

TEST(XcKernelGradientTest, TheBlockMatchesTheDisplacedPotentialForAGga) {
    CheckFunctional("pbe");
}

TEST(XcKernelGradientTest, TheBlockMatchesTheDisplacedPotentialForATauFunctional) {
    CheckFunctional("tau_x");
}

TEST(XcKernelGradientTest, AFunctionalWithoutATierIsRefusedByName) {
    const auto fixture = MakeGradientFixture("lyp");
    ASSERT_TRUE(fixture.has_value());

    const auto block = EvaluateXcKernelDerivativeBlock(fixture->engine,
                                                       *fixture->provider,
                                                       fixture->orbitals,
                                                       fixture->density,
                                                       fixture->numOccupied,
                                                       fixture->numVirtual,
                                                       XcKernelSettings{kScreeningTolerance});

    ASSERT_FALSE(block.has_value());
    EXPECT_EQ(block.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(block.error().message.find("no second-derivative tier"), std::string::npos)
        << block.error().message;
}

TEST(XcKernelGradientTest, AShapeMismatchIsRefused) {
    const auto fixture = MakeGradientFixture("pbe");
    ASSERT_TRUE(fixture.has_value());

    const Eigen::MatrixXd shortOrbitals =
        fixture->orbitals.leftCols(static_cast<Eigen::Index>(fixture->numOccupied));

    const auto block = EvaluateXcKernelDerivativeBlock(fixture->engine,
                                                       *fixture->provider,
                                                       shortOrbitals,
                                                       fixture->density,
                                                       fixture->numOccupied,
                                                       fixture->numVirtual);

    ASSERT_FALSE(block.has_value());
    EXPECT_EQ(block.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace qcx::grid
