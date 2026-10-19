// The one-electron derivative surface's tests (one_electron.hpp): the S/T/V
// derivative entry against the CENTRAL FINITE DIFFERENCE of the unchanged
// production builders - BuildOverlapMatrix, BuildKineticMatrix and
// BuildNuclearAttractionMatrix - on a water fixture with one hydrogen moved
// off the molecular plane. Equilibrium water is C2v, and a symmetric fixture
// hides a transposed or misplaced pair block; the displaced geometry has no
// symmetry operation at all.
//
// The entry is gated against the builders, not against the internal derivative
// tier: the tier carries its own order-1/order-2 finite-difference coverage
// (md_derivative_test.cpp), and what this file checks is that the public entry
// differentiates what the production path computes.

#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The central-difference step: the truncation is O(h^2) and the cancellation
/// error O(eps E / h), both near 1e-11 at this fixture's O(1) magnitudes -
/// comfortably inside the 1e-8 acceptance threshold.
constexpr double kFiniteDifferenceStep = 1e-5;

/// The analytic-versus-finite-difference threshold (the acceptance gate).
constexpr double kFiniteDifferenceTolerance = 1e-8;

/// The reported contraction against the production builders' own: the same
/// terms summed in a different order, so the gap is the accumulation's
/// roundoff alone.
constexpr double kContractionTolerance = 1e-12;

/// The translation sums' gate: each is exactly zero in exact arithmetic (a
/// rigid translation of every nucleus leaves every integral alone), and this
/// is the roundoff floor a fixed-order accumulation leaves.
constexpr double kTranslationTolerance = 1e-12;

/// A double as a short scientific string, for the recorded properties.
std::string ShortDouble(double value) {
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
}

/// H2O in STO-3G with one hydrogen displaced off the molecular plane: the
/// equilibrium fixture's geometry (r(OH) = 0.9572 A, angle = 104.52 deg, in
/// the xy plane) with the second hydrogen moved 0.437 Bohr along z. The two
/// hydrogens then sit at different z, so no mirror plane and no rotation maps
/// the molecule onto itself - a pair block placed on the wrong atom or
/// transposed has nothing to cancel against.
qcx::Result<qcx::molecule::Molecule> MakeLowSymmetryWater() {
    auto coordinates = CpuTensor2::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    // O at the origin; H at +-d, h (d = 1.430428808474167, h =
    // 1.107157044080814 Bohr); the second H carries the off-plane shift.
    constexpr std::array<std::array<double, 3>, 3> kCoordinates = {
        {{0.0, 0.0, 0.0},
         {1.430428808474167, 1.107157044080814, 0.0},
         {-1.430428808474167, 1.107157044080814, 0.437}}};

    for (std::size_t atom = 0; atom < kCoordinates.size(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(atom, axis) = kCoordinates[atom][axis];
        }
    }

    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

/// The geometry with one flat coordinate displaced. Errors when the canonical
/// renumbering would move an atom: the finite-difference loop reads the
/// gradient's row for the same atom, which the reordering would break - and
/// the fixture's separating distances are far above the step, so a reordering
/// is a defect in the fixture, not an expected outcome.
qcx::Result<qcx::molecule::Molecule> MakeDisplaced(const qcx::molecule::Molecule& base,
                                                   std::size_t flatCoordinate,
                                                   double delta) {
    auto coordinates = CpuTensor2::Create({base.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(atom, axis) = base.CoordinatesBohr()(atom, axis);
        }
    }

    (*coordinates)(flatCoordinate / 3, flatCoordinate % 3) += delta;
    coordinates->MarkHostDirty();
    auto displaced = qcx::molecule::Molecule::Create(
        base.Atoms(), std::move(*coordinates), base.Charge(), base.Multiplicity());

    if (!displaced.has_value())
    {
        return std::unexpected(displaced.error());
    }

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            double expected = base.CoordinatesBohr()(atom, axis);

            if (atom == flatCoordinate / 3 && axis == flatCoordinate % 3)
            {
                expected += delta;
            }

            if (displaced->CoordinatesBohr()(atom, axis) != expected)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "the displaced geometry is not the base geometry with one "
                               "coordinate moved - the atom rows were reordered"});
            }
        }
    }

    return displaced;
}

/// The density the derivative walk is contracted with: the core-Hamiltonian
/// eigenvector density, not a converged SCF density. The gradient is at fixed
/// density by contract, so any density exercises the same code, and this one
/// needs no SCF module - which the integrals tests cannot depend on.
qcx::Result<Eigen::MatrixXd> CoreDensity(const qcx::molecule::Molecule& molecule,
                                         const qcx::basisset::BasisSet& basisSet) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto attraction = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!attraction.has_value())
    {
        return std::unexpected(attraction.error());
    }

    const Eigen::MatrixXd core =
        qcx::testing::ToMatrix(*kinetic) + qcx::testing::ToMatrix(*attraction);
    const Eigen::MatrixXd metric = qcx::testing::ToMatrix(*overlap);
    const Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(core, metric);

    if (solver.info() != Eigen::Success)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "the core-Hamiltonian density solve failed"});
    }

    int electrons = 0;

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        electrons += molecule.Atoms()[atom].atomicNumber;
    }

    electrons -= molecule.Charge();
    const Eigen::Index occupied = static_cast<Eigen::Index>(std::max(electrons, 0) / 2);
    const Eigen::MatrixXd coefficients = solver.eigenvectors().leftCols(occupied);
    return 2.0 * coefficients * coefficients.transpose();
}

/// The density contraction Tr[D X] of one symmetric matrix.
double WeightedTrace(const CpuTensor2& matrix, const Eigen::MatrixXd& density) {
    const std::size_t n = matrix.Shape()[0];
    double sum = 0.0;

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            sum +=
                density(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) * matrix(i, j);
        }
    }

    return sum;
}

/// The S, T and V contractions at one geometry, through the unchanged
/// production builders - the finite difference's sample point.
qcx::Result<std::array<double, 3>> OneElectronContractions(const qcx::molecule::Molecule& molecule,
                                                           const qcx::basisset::BasisSet& basisSet,
                                                           const Eigen::MatrixXd& density) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    return std::array<double, 3>{WeightedTrace(*overlap, density),
                                 WeightedTrace(*kinetic, density),
                                 WeightedTrace(*nuclear, density)};
}

/// Records and gates one gradient's translation sums: moving every atom
/// together leaves every integral alone, so each Cartesian sum is exactly
/// zero.
/// \param prefix The property-name prefix.
/// \param gradient The 3N gradient.
/// \param atomCount The molecule's atom count.
void ExpectZeroTranslationSums(const char* prefix,
                               const Eigen::VectorXd& gradient,
                               std::size_t atomCount) {
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        double sum = 0.0;

        for (std::size_t atom = 0; atom < atomCount; ++atom)
        {
            sum += gradient(static_cast<Eigen::Index>(3 * atom + axis));
        }

        ::testing::Test::RecordProperty(std::string(prefix) + "_axis_" + std::to_string(axis),
                                        ShortDouble(sum));
        EXPECT_LE(std::abs(sum), kTranslationTolerance)
            << prefix << " sum over axis " << axis << " is " << sum;
    }
}

} // namespace

/// The acceptance gate: the entry's three gradients against the central finite
/// difference of the production builders' contractions, every nuclear
/// coordinate, on the low-symmetry fixture. The density is held fixed at the
/// base geometry - the derivative's contract - while the builders see the
/// displaced geometries.
TEST(OneElectronGradientTest, TheDerivativeSurfaceMatchesCentralDifferences) {
    auto molecule = MakeLowSymmetryWater();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto density = CoreDensity(*molecule, *basis);
    ASSERT_TRUE(density.has_value()) << density.error().message;
    auto densityTensor = qcx::testing::ToTensor(*density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto analytic = qcx::integrals::ComputeOneElectronGradient(*molecule, *basis, *densityTensor);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    ASSERT_EQ(analytic->overlapGradient.size(),
              static_cast<Eigen::Index>(3 * molecule->AtomCount()));
    ASSERT_EQ(analytic->kineticGradient.size(), analytic->overlapGradient.size());
    ASSERT_EQ(analytic->nuclearGradient.size(), analytic->overlapGradient.size());

    std::array<double, 3> worst = {0.0, 0.0, 0.0};
    std::array<std::size_t, 3> worstCoordinate = {0, 0, 0};

    for (std::size_t flat = 0; flat < 3 * molecule->AtomCount(); ++flat)
    {
        auto plus = MakeDisplaced(*molecule, flat, kFiniteDifferenceStep);
        auto minus = MakeDisplaced(*molecule, flat, -kFiniteDifferenceStep);
        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusValues = OneElectronContractions(*plus, *basis, *density);
        auto minusValues = OneElectronContractions(*minus, *basis, *density);
        ASSERT_TRUE(plusValues.has_value()) << plusValues.error().message;
        ASSERT_TRUE(minusValues.has_value()) << minusValues.error().message;

        const std::array<double, 3> numeric = {
            ((*plusValues)[0] - (*minusValues)[0]) / (2.0 * kFiniteDifferenceStep),
            ((*plusValues)[1] - (*minusValues)[1]) / (2.0 * kFiniteDifferenceStep),
            ((*plusValues)[2] - (*minusValues)[2]) / (2.0 * kFiniteDifferenceStep)};
        const std::array<double, 3> analyticValues = {
            analytic->overlapGradient(static_cast<Eigen::Index>(flat)),
            analytic->kineticGradient(static_cast<Eigen::Index>(flat)),
            analytic->nuclearGradient(static_cast<Eigen::Index>(flat))};

        for (std::size_t op = 0; op < 3; ++op)
        {
            const double deviation = std::abs(numeric[op] - analyticValues[op]);

            if (deviation > worst[op])
            {
                worst[op] = deviation;
                worstCoordinate[op] = flat;
            }
        }
    }

    const std::array<const char*, 3> names = {"overlap", "kinetic", "nuclear"};

    for (std::size_t op = 0; op < 3; ++op)
    {
        RecordProperty(std::string("worst_") + names[op] + "_deviation", ShortDouble(worst[op]));
        EXPECT_LT(worst[op], kFiniteDifferenceTolerance)
            << names[op] << " derivative at coordinate " << worstCoordinate[op];
    }
}

/// The contractions the entry reports are the production builders' own: the
/// values its gradients are the derivatives of. A walk that contracted a
/// different density or a different block would differ here before any finite
/// difference is involved.
TEST(OneElectronGradientTest, TheReportedContractionsAreTheProductionBuilders) {
    auto molecule = MakeLowSymmetryWater();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto density = CoreDensity(*molecule, *basis);
    ASSERT_TRUE(density.has_value()) << density.error().message;
    auto densityTensor = qcx::testing::ToTensor(*density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto analytic = qcx::integrals::ComputeOneElectronGradient(*molecule, *basis, *densityTensor);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    auto reference = OneElectronContractions(*molecule, *basis, *density);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    RecordProperty("overlap_contraction", ShortDouble(analytic->overlapContraction));
    RecordProperty("kinetic_contraction", ShortDouble(analytic->kineticContraction));
    RecordProperty("nuclear_contraction", ShortDouble(analytic->nuclearContraction));
    EXPECT_NEAR(analytic->overlapContraction, (*reference)[0], kContractionTolerance);
    EXPECT_NEAR(analytic->kineticContraction, (*reference)[1], kContractionTolerance);
    EXPECT_NEAR(analytic->nuclearContraction, (*reference)[2], kContractionTolerance);

    // The counts the walk reports: one entry per canonical pair walked, and
    // the two coordinate columns the operators reach - the pair's own atoms
    // for S/T (six per hetero pair, three per same-atom pair) and every
    // nucleus for V.
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    EXPECT_EQ(analytic->counts.orbitalPairs, pairList->pairs.size());
    EXPECT_GT(analytic->counts.overlapTuples, 0u);
    EXPECT_EQ(analytic->counts.kineticTuples, analytic->counts.overlapTuples);
    EXPECT_EQ(analytic->counts.nuclearTuples, pairList->pairs.size() * 3 * molecule->AtomCount());
}

/// The translation sums are exactly zero: a rigid translation of every nucleus
/// leaves every integral alone. The gate is the accumulation's roundoff floor,
/// and the raw sums are recorded beside it.
TEST(OneElectronGradientTest, TheTranslationalSumsAreExactlyZero) {
    auto molecule = MakeLowSymmetryWater();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto density = CoreDensity(*molecule, *basis);
    ASSERT_TRUE(density.has_value()) << density.error().message;
    auto densityTensor = qcx::testing::ToTensor(*density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto analytic = qcx::integrals::ComputeOneElectronGradient(*molecule, *basis, *densityTensor);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;

    ExpectZeroTranslationSums(
        "overlap_translation", analytic->overlapGradient, molecule->AtomCount());
    ExpectZeroTranslationSums(
        "kinetic_translation", analytic->kineticGradient, molecule->AtomCount());
    ExpectZeroTranslationSums(
        "nuclear_translation", analytic->nuclearGradient, molecule->AtomCount());
}

/// The chunk loop's own pin: on this fixture the whole pair store fits ONE
/// chunk, so the default walk never runs the loop as a loop. A one-pair chunk
/// cap forces every boundary, and the arithmetic must not move - the pairs are
/// built by the same builder and accumulated in the same pair order, so the
/// result is bit-identical.
TEST(OneElectronGradientTest, AOnePairChunkWalkIsBitIdentical) {
    auto molecule = MakeLowSymmetryWater();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto density = CoreDensity(*molecule, *basis);
    ASSERT_TRUE(density.has_value()) << density.error().message;
    auto densityTensor = qcx::testing::ToTensor(*density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto whole = qcx::integrals::ComputeOneElectronGradient(*molecule, *basis, *densityTensor);
    ASSERT_TRUE(whole.has_value()) << whole.error().message;

    qcx::integrals::OneElectronGradientOptions options;
    options.maxPairChunkBytes = 0;
    auto chunked =
        qcx::integrals::ComputeOneElectronGradient(*molecule, *basis, *densityTensor, options);
    ASSERT_TRUE(chunked.has_value()) << chunked.error().message;

    EXPECT_EQ((chunked->overlapGradient - whole->overlapGradient).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((chunked->kineticGradient - whole->kineticGradient).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ((chunked->nuclearGradient - whole->nuclearGradient).cwiseAbs().maxCoeff(), 0.0);
    EXPECT_EQ(chunked->overlapContraction, whole->overlapContraction);
    EXPECT_EQ(chunked->kineticContraction, whole->kineticContraction);
    EXPECT_EQ(chunked->nuclearContraction, whole->nuclearContraction);
    EXPECT_EQ(chunked->counts.orbitalPairs, whole->counts.orbitalPairs);
    EXPECT_EQ(chunked->counts.overlapTuples, whole->counts.overlapTuples);
    EXPECT_EQ(chunked->counts.kineticTuples, whole->counts.kineticTuples);
    EXPECT_EQ(chunked->counts.nuclearTuples, whole->counts.nuclearTuples);
}

/// The refusal paths: a density that is not nBasis x nBasis, and an atom with
/// no basis entry (the shell-pair build's own error).
TEST(OneElectronGradientTest, RejectsInvalidInputs) {
    auto molecule = MakeLowSymmetryWater();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto wrongDensity = CpuTensor2::Create({2, 2});
    ASSERT_TRUE(wrongDensity.has_value()) << wrongDensity.error().message;
    wrongDensity->MarkHostDirty();
    auto badDensity = qcx::integrals::ComputeOneElectronGradient(*molecule, *basis, *wrongDensity);
    ASSERT_FALSE(badDensity.has_value());
    EXPECT_EQ(badDensity.error().code, qcx::ErrorCode::kInvalidArgument);

    auto helium = qcx::testing::MakeHeAtom();
    ASSERT_TRUE(helium.has_value()) << helium.error().message;
    auto density = CpuTensor2::Create({1, 1});
    ASSERT_TRUE(density.has_value()) << density.error().message;
    density->MarkHostDirty();
    auto missing = qcx::integrals::ComputeOneElectronGradient(*helium, *basis, *density);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, qcx::ErrorCode::kInvalidArgument);
}
