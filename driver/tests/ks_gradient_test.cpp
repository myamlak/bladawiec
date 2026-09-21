// — the driver-side Kohn-Sham gradient: the run's own grid, differenced against
// the energy that grid integrates.
//
// Three numbers are kept apart, because a gradient can be wrong in a way only
// one of them sees:
//
//   - the agreement with a central difference of the engine's own energy at
//     displaced geometries;
//   - the thresholds the walk ran under, which are the energy path's or the
//     gradient is the derivative of a different energy;
//   - the energy the walk integrated, reported on its own line and never
//     merged into the agreement, which is where a walk that truncated or
//     screened differently from the energy it differentiates shows up and
//     nowhere else.
//
// The density is a real run's: H2/STO-3G through the driver's own composition,
// converged, then held FIXED for the gradient and for the difference. That is
// the term this walk computes - the density's response to the displacement is
// the SCF's and the response layer's - so a difference that let the density
// relax would be measuring another layer's contribution.
#include "h2_sto3g.hpp"
#include "internal/ks_composition.hpp"
#include "internal/ks_grid.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using qcx::driver::internal::AddXcGradientContribution;
using qcx::driver::internal::CreateKsGrid;
using qcx::driver::internal::HalfFockFn;
using qcx::driver::internal::KsGrid;
using qcx::driver::internal::MakeRksSeam;
using qcx::driver::internal::XcEvaluatorFn;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;

// The geometry the difference walks: H2 in STO-3G, off every axis, so the
// density's curvature reaches the off-diagonal components of the integrand's
// second derivative rather than three of them.
constexpr std::array<std::array<double, 3>, 2> kPositions = {
    {{0.0, 0.0, 0.0}, {1.40, 0.18, -0.12}}};

// The functional: a GGA, so the walk reads the density's gradient and the AO
// tier's second derivatives - the longest chain the assembly has.
constexpr std::string_view kFunctional = "pbe";

// The run's grid: the production defaults, because they are what a run that
// writes no `[grid]` block gets.
constexpr qcx::grid::XcGridSettings kSettings{};

// The screen this file's runs resolve. It is deliberately NOT the schema
// default: io's kDefaultScreeningTolerance is 1e-10, so a gradient that
// re-derived its own thresholds would carry the default and the reuse
// assertions below would catch it.
constexpr double kScreen = 1e-8;
constexpr double kLooserScreen = 1e-4;

// The difference's step and the bar the agreement is held to.
constexpr double kStep = 1e-4;
constexpr double kAgreement = 1e-8;

// A number as the test records it. The recorded values are the numbers whoever
// reports this run quotes, so they carry the precision the measurement deserves
// rather than the six decimals a default conversion keeps.
std::string Scientific(double value) {
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%.3e", value);

    return std::string(buffer.data());
}

// The largest absolute component of a difference of two vectors.
double MaxAbsDifference(const Eigen::VectorXd& left, const Eigen::VectorXd& right) {
    return (left - right).cwiseAbs().maxCoeff();
}

// The fixture's molecule at one set of positions.
//
// The molecule renumbers its atoms canonically, so the caller's atom order is
// the fixture's only if the sort agrees with it - asserted here rather than
// assumed, because a reorder would make every displacement below address the
// wrong atom.
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
    auto molecule = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    for (std::size_t atom = 0; atom < 2; ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            if (molecule->CoordinatesBohr()(static_cast<Eigen::Index>(atom),
                                            static_cast<Eigen::Index>(axis)) !=
                positions[atom][axis])
            {
                return std::unexpected(qcx::Error{
                    qcx::ErrorCode::kInternalError,
                    "the fixture's atom order is not the molecule's: the canonical sort moved an "
                    "atom, so a displacement would not address the atom it names"});
            }
        }
    }

    return std::move(*molecule);
}

// The one-electron integrals, the basis and the exact Coulomb operator of one
// geometry. Only the SCF's density needs the integrals: the gradient's own walk
// reads the engine.
struct Fixture {
    qcx::basisset::BasisSet basis;
    qcx::molecule::Molecule molecule;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd core;
    Eigen::MatrixXd coulombSuper;
};

// The Coulomb operator as a flat supermatrix over the fixture's two-electron
// tensor, in the row-major AO order a flat density is written in: J = E d.
Eigen::MatrixXd BuildCoulombSuper(const CpuTensor4& eri, std::size_t n) {
    const Eigen::Index n2 = static_cast<Eigen::Index>(n * n);
    Eigen::MatrixXd coulomb = Eigen::MatrixXd::Zero(n2, n2);

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            for (std::size_t lambda = 0; lambda < n; ++lambda)
            {
                for (std::size_t sigma = 0; sigma < n; ++sigma)
                {
                    coulomb(static_cast<Eigen::Index>(mu * n + nu),
                            static_cast<Eigen::Index>(lambda * n + sigma)) =
                        eri(mu, nu, lambda, sigma);
                }
            }
        }
    }

    return coulomb;
}

Eigen::MatrixXd FlattenRowMajor(const Eigen::MatrixXd& matrix) {
    Eigen::MatrixXd flat(matrix.rows() * matrix.cols(), 1);

    for (Eigen::Index mu = 0; mu < matrix.rows(); ++mu)
    {
        for (Eigen::Index nu = 0; nu < matrix.cols(); ++nu)
        {
            flat(mu * matrix.cols() + nu, 0) = matrix(mu, nu);
        }
    }

    return flat;
}

Eigen::MatrixXd UnflattenRowMajor(const Eigen::MatrixXd& flat, Eigen::Index n) {
    Eigen::MatrixXd matrix(n, n);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            matrix(mu, nu) = flat(mu * n + nu, 0);
        }
    }

    return matrix;
}

qcx::Result<Fixture> MakeFixture(const std::array<std::array<double, 3>, 2>& positions) {
    auto basis = MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto molecule = MakeMolecule(positions);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);

    if (!overlap.has_value() || !kinetic.has_value() || !nuclear.has_value() || !eri.has_value())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the H2/STO-3G fixture integrals did not build"});
    }

    const std::size_t n = static_cast<std::size_t>(ToMatrix(*overlap).rows());

    return Fixture{std::move(*basis),
                   std::move(*molecule),
                   ToMatrix(*overlap),
                   ToMatrix(*kinetic) + ToMatrix(*nuclear),
                   BuildCoulombSuper(*eri, n)};
}

// The Coulomb half the composition's seam consumes: H + 2 J(rho), the shape a
// buildCoulombOnly member of the integrals family returns.
HalfFockFn MakeCoulombHalf(const Eigen::MatrixXd& coulombSuper, const Eigen::MatrixXd& core) {
    return [coulombSuper, core](const Eigen::MatrixXd& rho) -> qcx::Result<Eigen::MatrixXd> {
        return core + 2.0 * UnflattenRowMajor(coulombSuper * FlattenRowMajor(rho), rho.rows());
    };
}

// The XC evaluator a run's energy path uses: the context's own engine, screened
// by the context's own threshold object.
XcEvaluatorFn MakeEvaluator(const KsGrid& grid) {
    return [engine = grid.engine, thresholds = grid.thresholds](
               const Eigen::MatrixXd& densityAlpha,
               const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        return engine->EvaluateScreened(densityAlpha, densityBeta, thresholds.screeningTolerance);
    };
}

// The run's converged closed-shell density: the driver's own composition, the
// engine's real grid, and the fixture's exact Coulomb operator.
qcx::Result<Eigen::MatrixXd> ConvergedDensity(const Fixture& fixture, const KsGrid& grid) {
    auto seam = MakeRksSeam(MakeCoulombHalf(fixture.coulombSuper, fixture.core),
                            HalfFockFn{},
                            fixture.core,
                            0.0,
                            MakeEvaluator(grid));

    if (!seam.has_value())
    {
        return std::unexpected(seam.error());
    }

    const qcx::scf::RhfOptions options;
    auto run = qcx::scf::RunRhfScf(fixture.molecule,
                                   fixture.overlap,
                                   fixture.core,
                                   options,
                                   seam->fock,
                                   seam->coulomb,
                                   seam->contribution);

    if (!run.has_value())
    {
        return std::unexpected(run.error());
    }

    if (!run->converged)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kConvergenceFailure,
                                          "the fixture's Kohn-Sham run did not converge, so its "
                                          "density is not a run's"});
    }

    return run->density;
}

// The engine's own energy at one displaced geometry, with the density held at
// the caller's.
//
// The context is rebuilt the way a run builds it - one settings object, one
// tolerance - so the displaced geometry's grid is the same rule's grid, and the
// difference below is a difference of the energy this walk differentiates.
struct Displaced {
    double energy = 0.0;
    std::size_t points = 0;
    std::size_t shellKeeps = 0;
};

qcx::Result<Displaced> DisplacedEnergy(const std::array<std::array<double, 3>, 2>& positions,
                                       const Eigen::MatrixXd& spinDensity) {
    auto basis = MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto molecule = MakeMolecule(positions);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto grid = CreateKsGrid(*molecule, *basis, kFunctional, kSettings, kScreen);

    if (!grid.has_value())
    {
        return std::unexpected(grid.error());
    }

    auto energy = grid->engine->EvaluateScreened(
        spinDensity, spinDensity, grid->thresholds.screeningTolerance, grid->batch);

    if (!energy.has_value())
    {
        return std::unexpected(energy.error());
    }

    return Displaced{energy->energy, energy->counts.points, energy->counts.shellKeeps};
}

// I1 through the driver's own wiring: the run's grid, the library's provider,
// the run's density, against a central difference of the engine's own energy at
// displaced geometries.
TEST(KsGradientTest, TheRunGradientMatchesAFiniteDifferenceOfItsOwnEnergy) {
    const auto fixture = MakeFixture(kPositions);
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;

    const auto grid =
        CreateKsGrid(fixture->molecule, fixture->basis, kFunctional, kSettings, kScreen);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    const auto density = ConvergedDensity(*fixture, *grid);
    ASSERT_TRUE(density.has_value()) << density.error().message;

    // The closed-shell spin split the engine's conventions use, D_s = D/2, and
    // the pair the driver's evaluator is handed for a closed-shell run.
    const Eigen::MatrixXd spinDensity = 0.5 * *density;

    const Eigen::Index coordinates = static_cast<Eigen::Index>(3U * kPositions.size());
    Eigen::VectorXd total = Eigen::VectorXd::Zero(coordinates);
    const auto walked = AddXcGradientContribution(*grid, spinDensity, spinDensity, total);
    ASSERT_TRUE(walked.has_value()) << walked.error().message;
    ASSERT_EQ(total.size(), coordinates);

    // The term went INTO the total rather than replacing it: a total already
    // carrying the other contributions keeps them, term for term, and a walk's
    // gradient added twice would show up here as a doubling.
    Eigen::VectorXd carried = Eigen::VectorXd::Ones(coordinates) * 0.25;
    const Eigen::VectorXd before = carried;
    const auto again = AddXcGradientContribution(*grid, spinDensity, spinDensity, carried);
    ASSERT_TRUE(again.has_value()) << again.error().message;
    EXPECT_EQ(MaxAbsDifference(carried, before + walked->gradient), 0.0)
        << "the walk's vector was not added into the accumulator";
    EXPECT_EQ(MaxAbsDifference(total, walked->gradient), 0.0)
        << "the accumulator holds a multiple of the walk's vector";

    double maxDeviation = 0.0;

    for (Eigen::Index coordinate = 0; coordinate < coordinates; ++coordinate)
    {
        const std::size_t atom = static_cast<std::size_t>(coordinate) / 3U;
        const std::size_t axis = static_cast<std::size_t>(coordinate) % 3U;
        auto plus = kPositions;
        auto minus = kPositions;
        plus[atom][axis] += kStep;
        minus[atom][axis] -= kStep;

        const auto plusEnergy = DisplacedEnergy(plus, spinDensity);
        ASSERT_TRUE(plusEnergy.has_value()) << plusEnergy.error().message;
        const auto minusEnergy = DisplacedEnergy(minus, spinDensity);
        ASSERT_TRUE(minusEnergy.has_value()) << minusEnergy.error().message;

        // The grid and the selection must not move with the displacement: a
        // point trimmed in or out, or a shell switching on or off, makes the
        // difference discontinuous, and nothing else here would say so.
        ASSERT_EQ(plusEnergy->points, minusEnergy->points)
            << "the grid's point set moved with the displacement";
        ASSERT_EQ(plusEnergy->shellKeeps, walked->counts.shellKeeps)
            << "the selection moved with the displacement";

        const double difference = (plusEnergy->energy - minusEnergy->energy) / (2.0 * kStep);
        maxDeviation = std::max(maxDeviation, std::abs(total[coordinate] - difference));
    }

    RecordProperty("driver gradient agreement (max deviation, Hartree/Bohr)",
                   Scientific(maxDeviation));
    EXPECT_LT(maxDeviation, kAgreement) << "the driver's gradient and the engine's own energy "
                                           "disagree at the difference's own step";

    // An invariant no finite difference of this file sees: the grid rides its
    // atoms and its weights see the geometry only through differences, so the
    // gradient carries no net force.
    double sum = 0.0;

    for (Eigen::Index component = 0; component < total.size(); ++component)
    {
        sum += total[component];
    }

    RecordProperty("driver gradient net force (Hartree/Bohr)", Scientific(std::abs(sum)));
    EXPECT_LT(std::abs(sum), 1e-12) << "the driver's gradient carries a net force";
}

// The reuse, asserted rather than assumed: the thresholds the walk ran under
// are the run's own object, field for field, and the walk's selection and
// energy are the energy path's - which is the behaviour the values claim.
TEST(KsGradientTest, TheWalkRunsUnderTheEnergyPathsOwnThresholds) {
    const auto fixture = MakeFixture(kPositions);
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;

    const auto grid =
        CreateKsGrid(fixture->molecule, fixture->basis, kFunctional, kSettings, kScreen);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    const auto density = ConvergedDensity(*fixture, *grid);
    ASSERT_TRUE(density.has_value()) << density.error().message;
    const Eigen::MatrixXd spinDensity = 0.5 * *density;

    // The case only proves anything if the screen it resolves is not the one a
    // re-derivation from defaults would produce.
    const qcx::grid::XcIntegrationThresholds expected =
        qcx::grid::XcIntegrationThresholds::FromEnergyPath(kSettings, kScreen);
    EXPECT_NE(kScreen, qcx::io::kDefaultScreeningTolerance)
        << "the fixture resolved the schema default, so a re-derived threshold would pass";
    EXPECT_DOUBLE_EQ(grid->thresholds.trimWeight, expected.trimWeight);
    EXPECT_DOUBLE_EQ(grid->thresholds.screeningTolerance, expected.screeningTolerance);

    const auto energyPath = grid->engine->EvaluateScreened(
        spinDensity, spinDensity, grid->thresholds.screeningTolerance, grid->batch);
    ASSERT_TRUE(energyPath.has_value()) << energyPath.error().message;

    Eigen::VectorXd total = Eigen::VectorXd::Zero(3 * static_cast<Eigen::Index>(2));
    const auto walked = AddXcGradientContribution(*grid, spinDensity, spinDensity, total);
    ASSERT_TRUE(walked.has_value()) << walked.error().message;

    // The thresholds the walk echoes are the ones it was handed.
    EXPECT_DOUBLE_EQ(walked->thresholds.trimWeight, grid->thresholds.trimWeight);
    EXPECT_DOUBLE_EQ(walked->thresholds.screeningTolerance, grid->thresholds.screeningTolerance);

    // And the same object, seen in behaviour: one selection, one point set, one
    // cost.
    EXPECT_EQ(walked->counts.points, energyPath->counts.points);
    EXPECT_EQ(walked->counts.shellTests, energyPath->counts.shellTests);
    EXPECT_EQ(walked->counts.shellKeeps, energyPath->counts.shellKeeps);
    EXPECT_EQ(walked->counts.aoFetches, energyPath->counts.aoFetches);
    EXPECT_EQ(walked->counts.aoSlotsEvaluated, energyPath->counts.aoSlotsEvaluated);
    EXPECT_EQ(walked->counts.contractionPairs, energyPath->counts.contractionPairs);
    EXPECT_GT(walked->counts.shellKeeps, 0U) << "the walk selected nothing";
    EXPECT_LT(walked->counts.shellKeeps, walked->counts.shellTests)
        << "the screen drops nothing at this tolerance: the walk is dense here, so the reuse it "
           "asserts is not the reuse a screened run makes";

    // The consistency line, on its own: a walk that screened or truncated
    // differently from the energy it differentiates differs here and nowhere
    // else, and a gradient that agrees with the difference while the energy
    // does not move with it is exactly the defect the reuse exists to prevent.
    const double consistency = std::abs(energyPath->energy - walked->energy);
    RecordProperty("energy path vs gradient path (Hartree)", Scientific(consistency));
    EXPECT_DOUBLE_EQ(energyPath->energy, walked->energy)
        << "the gradient path integrated an energy the energy path does not recognise";
}

// The tolerance the run resolved is the one the walk screens by: two contexts
// built from one settings object with two tolerances must each carry their own
// and each screen with it, so a walk that reached for a default of its own
// would not track the run it belongs to.
TEST(KsGradientTest, TheScreenTheRunResolvedIsTheOneTheWalkScreensBy) {
    const auto fixture = MakeFixture(kPositions);
    ASSERT_TRUE(fixture.has_value()) << fixture.error().message;

    const auto tight =
        CreateKsGrid(fixture->molecule, fixture->basis, kFunctional, kSettings, kScreen);
    ASSERT_TRUE(tight.has_value()) << tight.error().message;
    const auto loose =
        CreateKsGrid(fixture->molecule, fixture->basis, kFunctional, kSettings, kLooserScreen);
    ASSERT_TRUE(loose.has_value()) << loose.error().message;

    const auto density = ConvergedDensity(*fixture, *tight);
    ASSERT_TRUE(density.has_value()) << density.error().message;
    const Eigen::MatrixXd spinDensity = 0.5 * *density;

    Eigen::VectorXd tightTotal = Eigen::VectorXd::Zero(3 * static_cast<Eigen::Index>(2));
    const auto tightWalk = AddXcGradientContribution(*tight, spinDensity, spinDensity, tightTotal);
    ASSERT_TRUE(tightWalk.has_value()) << tightWalk.error().message;

    Eigen::VectorXd looseTotal = Eigen::VectorXd::Zero(3 * static_cast<Eigen::Index>(2));
    const auto looseWalk = AddXcGradientContribution(*loose, spinDensity, spinDensity, looseTotal);
    ASSERT_TRUE(looseWalk.has_value()) << looseWalk.error().message;

    EXPECT_DOUBLE_EQ(tightWalk->thresholds.screeningTolerance, kScreen);
    EXPECT_DOUBLE_EQ(looseWalk->thresholds.screeningTolerance, kLooserScreen);
    EXPECT_DOUBLE_EQ(looseWalk->thresholds.trimWeight, tightWalk->thresholds.trimWeight)
        << "the truncation is the settings object's, shared by both runs";

    RecordProperty("kept shells at 1e-8 (count)", std::to_string(tightWalk->counts.shellKeeps));
    RecordProperty("kept shells at 1e-4 (count)", std::to_string(looseWalk->counts.shellKeeps));
    EXPECT_GT(looseWalk->counts.shellKeeps, 0U) << "the loose screen selected nothing";
    EXPECT_LT(looseWalk->counts.shellKeeps, tightWalk->counts.shellKeeps)
        << "the tolerance did not reach the walk: the two runs kept the same shells";
}

} // namespace
