// The XC kernel action, on the linear density parameterisation it is defined
// against:
//   - the finite difference moves the density linearly, D + sum kappa_ai
//     deltaD_ai, so the comparison is the second derivative the action claims
//     and not a renormalised path's;
//   - the energy is the engine's own, integrated on the same grid the walk
//     reads, so the kernel algebra is the only thing under test.
// The tier scan and the two synthetic refusals come first: a functional with no
// second-derivative tier is answered by name, never by a zero term.

#include "h2o_sto3g.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/grid/xc_kernel_action.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <excgrid/kernel.hpp>
#include <gtest/gtest.h>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

// A coarse grid: the same quadrature on both sides of the comparison, so
// exactness is irrelevant and only consistency is measured.
constexpr std::size_t kRadialPoints = 20;
constexpr std::size_t kAngularPoints = 50;

// Keep every shell: the finite difference moves the density, not the grid, and
// the selection is frozen at the reference density on both sides. Keeping all
// shells is what makes that freezing invisible to the comparison.
constexpr double kScreeningTolerance = -1.0;

// The central difference's step, and its half. The four-point difference is
// second order; the Richardson combination of the two steps removes that term
// and leaves a fourth-order result.
constexpr double kFiniteDifferenceStep = 2e-3;
constexpr double kFiniteDifferenceTolerance = 1e-6;

std::string Scientific(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6e", value);

    return buffer;
}

// The names this excgrid pin ships a second-derivative tier for. The remaining
// registry entries refuse the tier by name; they are recorded by the scan and
// re-open as a gate when the pin moves, not here.
constexpr std::array<std::string_view, 7> kTieredNames = {
    "slater", "becke88", "pbe", "revpbe", "rpbe", "mpw91", "tau_x"};

// A deterministic orthogonal matrix. The action needs an orbital basis, not a
// canonical one: the density and its response are all it reads, so a rotation
// of the AO basis exercises every occupied-virtual pair without borrowing an
// eigensolver from a higher module.
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

// One functional's engine and the reference state the action is built on.
struct KernelFixture {
    XcGridEngine engine;
    Eigen::MatrixXd orbitals; ///< AOCount() x (o + v), occupied columns first.
    Eigen::MatrixXd density; ///< 2 C_o C_o^T.
    std::size_t numOccupied = 0;
    std::size_t numVirtual = 0;
};

std::optional<KernelFixture> MakeKernelFixture(std::string_view functionalName) {
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

    const std::size_t aoCount = engine->Evaluator().AOCount();
    const std::size_t numOccupied = static_cast<std::size_t>(molecule->ElectronCount()) / 2;
    const std::size_t numVirtual = aoCount - numOccupied;

    Eigen::MatrixXd orbitals =
        RotationMatrix(aoCount).leftCols(static_cast<Eigen::Index>(numOccupied + numVirtual));
    const Eigen::MatrixXd occupied = orbitals.leftCols(static_cast<Eigen::Index>(numOccupied));
    const Eigen::MatrixXd density = 2.0 * occupied * occupied.transpose();

    return KernelFixture{std::move(*engine), std::move(orbitals), density, numOccupied, numVirtual};
}

// The engine's XC energy along the linear density path
//     D(kappa) = D + sum_ai kappa_ai 2 (C_a C_i^T + C_i C_a^T).
double XcEnergy(const KernelFixture& fixture, std::span<const double> amplitudes) {
    Eigen::MatrixXd density = fixture.density;

    for (std::size_t occupied = 0; occupied < fixture.numOccupied; ++occupied)
    {
        for (std::size_t virtualOrbital = 0; virtualOrbital < fixture.numVirtual; ++virtualOrbital)
        {
            const double amplitude = amplitudes[occupied * fixture.numVirtual + virtualOrbital];

            if (amplitude == 0.0)
            {
                continue;
            }

            const Eigen::VectorXd virtualColumn = fixture.orbitals.col(
                static_cast<Eigen::Index>(fixture.numOccupied + virtualOrbital));
            const Eigen::VectorXd occupiedColumn =
                fixture.orbitals.col(static_cast<Eigen::Index>(occupied));

            // The rank-two update is symmetric by construction: the density
            // the engine reads is symmetric, and a column-wise write would
            // perturb a different, non-symmetric path.
            density += 2.0 * amplitude *
                       (virtualColumn * occupiedColumn.transpose() +
                        occupiedColumn * virtualColumn.transpose());
        }
    }

    const auto evaluated = fixture.engine.EvaluateClosedShell(density);

    if (!evaluated.has_value())
    {
        return std::nan("");
    }

    return evaluated->energy;
}

// The four-point central second difference in two amplitudes.
double SecondDifference(const KernelFixture& fixture,
                        std::size_t first,
                        std::size_t second,
                        double step) {
    std::vector<double> amplitudes(fixture.numOccupied * fixture.numVirtual, 0.0);
    amplitudes[first] += step;
    amplitudes[second] += step;
    const double plusPlus = XcEnergy(fixture, amplitudes);

    amplitudes[second] -= 2.0 * step;
    const double plusMinus = XcEnergy(fixture, amplitudes);

    amplitudes[first] -= 2.0 * step;
    const double minusMinus = XcEnergy(fixture, amplitudes);

    amplitudes[second] += 2.0 * step;
    const double minusPlus = XcEnergy(fixture, amplitudes);

    return (plusPlus - plusMinus - minusPlus + minusMinus) / (4.0 * step * step);
}

// A functional whose masks are the test's: the action's refusal path reads the
// capability report and nothing else.
class SyntheticFunctional final : public excgrid::XcFunctional {
public:
    SyntheticFunctional(excgrid::ComponentMask required, excgrid::ComponentMask spanned) :
        _required(required), _spanned(spanned) {}

    [[nodiscard]] bool UsesGradient() const override {
        return false;
    }

    [[nodiscard]] double ExchangeFraction() const override {
        return 0.0;
    }

    [[nodiscard]] excgrid::ComponentMask RequiredMask() const override {
        return _required;
    }

    [[nodiscard]] excgrid::ComponentMask SecondDerivativeMask() const noexcept override {
        return _spanned;
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePoint(const excgrid::PointInputs&,
                                                      excgrid::PointResult&) const override {
        return excgrid::KernelStatus::kOk;
    }

private:
    excgrid::ComponentMask _required;
    excgrid::ComponentMask _spanned;
};

// The first five identifiers: a tier that spans densities and gradient
// invariants but not the kinetic-energy densities.
excgrid::ComponentMask DensityAndGradientTier() {
    excgrid::ComponentMask mask;
    mask.Set(excgrid::Component::RhoA);
    mask.Set(excgrid::Component::RhoB);
    mask.Set(excgrid::Component::SigmaAa);
    mask.Set(excgrid::Component::SigmaAb);
    mask.Set(excgrid::Component::SigmaBb);

    return mask;
}

TEST(XcKernelActionTest, ThePinnedRegistryCarriesTheKnownSecondDerivativeTiers) {
    for (const std::string_view name : kTieredNames)
    {
        const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);
        ASSERT_NE(functional, nullptr) << name;
        EXPECT_NE(functional->SecondDerivativeMask().bits, 0U)
            << name << ": the tier this acceptance runs on is missing";
    }

    // Recorded, not asserted: at this pin the composite and correlation names
    // refuse the tier, and the acceptance runs on the tiered pure components.
    // The full scan becomes a gate when the pin moves to the generation that
    // carries all twenty-four.
    std::string untiered;

    for (const std::string_view name : excgrid::FunctionalNames())
    {
        const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);
        ASSERT_NE(functional, nullptr) << name;

        if (functional->SecondDerivativeMask().bits == 0)
        {
            if (!untiered.empty())
            {
                untiered += ", ";
            }

            untiered += std::string(name);
        }
    }

    RecordProperty("untiered names at this excgrid pin", untiered);
}

TEST(XcKernelActionTest, AFunctionalWithNoTierIsRefusedByName) {
    const auto fixture = MakeKernelFixture("slater");
    ASSERT_TRUE(fixture.has_value());

    const SyntheticFunctional functional{excgrid::ComponentMask::Collinear(),
                                         excgrid::ComponentMask{}};

    const auto action = XcKernelAction::Create(fixture->engine.Grid().Blocks(),
                                               fixture->engine.Evaluator(),
                                               functional,
                                               fixture->engine.ShellEnvelopes(),
                                               fixture->orbitals,
                                               fixture->density,
                                               fixture->numOccupied,
                                               fixture->numVirtual,
                                               XcKernelSettings{kScreeningTolerance});

    ASSERT_FALSE(action.has_value());
    EXPECT_EQ(action.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(action.error().message.find("no second-derivative tier"), std::string::npos)
        << action.error().message;
}

TEST(XcKernelActionTest, ATierThatDoesNotSpanTheRequiredComponentsIsRefusedByName) {
    const auto fixture = MakeKernelFixture("slater");
    ASSERT_TRUE(fixture.has_value());

    const SyntheticFunctional functional{excgrid::ComponentMask::Collinear(),
                                         DensityAndGradientTier()};

    const auto action = XcKernelAction::Create(fixture->engine.Grid().Blocks(),
                                               fixture->engine.Evaluator(),
                                               functional,
                                               fixture->engine.ShellEnvelopes(),
                                               fixture->orbitals,
                                               fixture->density,
                                               fixture->numOccupied,
                                               fixture->numVirtual,
                                               XcKernelSettings{kScreeningTolerance});

    ASSERT_FALSE(action.has_value());
    EXPECT_EQ(action.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(action.error().message.find("tauA, tauB"), std::string::npos)
        << action.error().message;
}

// The gate: for each tiered kind the acceptance names - an LDA, a GGA and a
// tau functional - four times the action's matrix is the second difference of
// the same engine's energy along the linear density path.
TEST(XcKernelActionTest, TheActionMatchesTheSecondDifferenceOfTheXcEnergy) {
    for (const std::string_view name : std::array<std::string_view, 3>{"slater", "pbe", "tau_x"})
    {
        const auto fixture = MakeKernelFixture(name);
        ASSERT_TRUE(fixture.has_value()) << name;

        const auto action = XcKernelAction::Create(fixture->engine,
                                                   fixture->orbitals,
                                                   fixture->density,
                                                   fixture->numOccupied,
                                                   fixture->numVirtual,
                                                   XcKernelSettings{kScreeningTolerance});
        ASSERT_TRUE(action.has_value()) << name << ": " << action.error().message;

        const std::size_t dimension = action->Dimension();
        double maxDeviation = 0.0;

        for (std::size_t column = 0; column < dimension; ++column)
        {
            std::vector<double> basisVector(dimension, 0.0);
            basisVector[column] = 1.0;

            std::vector<double> columnResult(dimension, 0.0);
            const auto applied = action->Apply(basisVector, columnResult);
            ASSERT_TRUE(applied.has_value()) << name << ": " << applied.error().message;

            for (std::size_t row = 0; row < dimension; ++row)
            {
                const double coarse =
                    SecondDifference(*fixture, row, column, kFiniteDifferenceStep);
                const double fine =
                    SecondDifference(*fixture, row, column, 0.5 * kFiniteDifferenceStep);
                ASSERT_FALSE(std::isnan(coarse) || std::isnan(fine)) << name;

                const double numerical = (4.0 * fine - coarse) / 3.0;
                const double analytic = 4.0 * columnResult[row];
                maxDeviation = std::max(maxDeviation, std::abs(numerical - analytic));
            }
        }

        RecordProperty(std::string(name) + " agreement (max deviation, Hartree)",
                       Scientific(maxDeviation));
        EXPECT_LT(maxDeviation, kFiniteDifferenceTolerance)
            << name << ": the action and the second difference of the XC energy disagree";
    }
}

} // namespace
} // namespace qcx::grid
