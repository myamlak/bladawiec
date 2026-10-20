// The one-electron derivative Fock blocks' tests (one_electron.hpp).
// ComputeOneElectronFockDerivative returns C^T dS/dR C over the whole MO
// basis and dH/dR in the occupied-virtual block, H = T + V, at fixed MO
// coefficients. The checks, in the order the surface's acceptance rests on
// them:
//
//  - each block against the CENTRAL FINITE DIFFERENCE of the corresponding
//    matrix transformed with the same fixed coefficients - the operator
//    matrices rebuilt at the displaced geometry, so the reference carries the
//    derivative through the basis functions and through the nuclear charges'
//    centres alike;
//  - the translation identity: moving every atom together changes no matrix,
//    so each block's sum over the atoms of an axis vanishes;
//  - the empty-orbital-block edge and the documented refusals.
//
// The fixture is the displaced H2O/STO-3G geometry (the water's symmetry
// planes broken so no entry is zero by symmetry alone) with the
// core-Hamiltonian eigenvectors as the fixed MO coefficients.
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
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
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The central-difference step: the overlap and Hamiltonian elements are O(1)
/// and smooth, so the truncation and the cancellation error both sit far
/// inside the 1e-8 threshold.
constexpr double kFiniteDifferenceStep = 1e-5;

/// The analytic-versus-finite-difference threshold of both blocks.
constexpr double kFiniteDifferenceTolerance = 1e-8;

/// The translation sum's threshold: the identity is exact in arithmetic, and
/// the sum's roundoff is orders of magnitude below this gate.
constexpr double kTranslationTolerance = 1e-12;

/// The symmetry-breaking displacement of the fixture's three atoms, one
/// triple per atom in (x, y, z) order.
constexpr std::array<double, 9> kBaseShifts = {
    0.013, -0.007, 0.019, -0.011, 0.017, -0.023, 0.029, -0.013, 0.011};

/// A double as a short scientific string, for the recorded properties.
std::string ShortDouble(double value) {
    char text[32] = {};
    std::snprintf(text, sizeof(text), "%.3e", value);
    return text;
}

/// The geometry with the listed flat coordinates (3 * atom + axis) displaced
/// by the listed deltas.
/// \param base The base molecule.
/// \param flatCoordinates The displaced coordinates.
/// \param deltas The displacements.
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> Moved(const qcx::molecule::Molecule& base,
                                           std::span<const std::size_t> flatCoordinates,
                                           std::span<const double> deltas) {
    if (flatCoordinates.size() != deltas.size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "displacement list shape mismatch"});
    }

    auto coordinates = CpuTensor2::Create({base.AtomCount(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    const auto& source = base.CoordinatesBohr();
    std::vector<std::array<double, 3>> expected;
    expected.reserve(base.AtomCount());

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        std::array<double, 3> row{};

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            row[axis] = source(atom, axis);
            (*coordinates)(atom, axis) = source(atom, axis);
        }

        expected.push_back(row);
    }

    for (std::size_t k = 0; k < flatCoordinates.size(); ++k)
    {
        const std::size_t flat = flatCoordinates[k];

        if (flat >= 3 * base.AtomCount())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "displaced coordinate out of range"});
        }

        (*coordinates)(flat / 3, flat % 3) += deltas[k];
        expected[flat / 3][flat % 3] += deltas[k];
    }

    coordinates->MarkHostDirty();
    auto displaced = qcx::molecule::Molecule::Create(
        base.Atoms(), std::move(*coordinates), base.Charge(), base.Multiplicity());

    if (!displaced.has_value())
    {
        return std::unexpected(displaced.error());
    }

    for (std::size_t atom = 0; atom < base.AtomCount(); ++atom)
    {
        if (displaced->Atoms()[atom].atomicNumber != base.Atoms()[atom].atomicNumber ||
            displaced->CoordinatesBohr()(atom, 0) != expected[atom][0] ||
            displaced->CoordinatesBohr()(atom, 1) != expected[atom][1] ||
            displaced->CoordinatesBohr()(atom, 2) != expected[atom][2])
        {
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kInvalidArgument,
                "the displaced geometry is not the base geometry with one row moved - the shell "
                "indices of the finite-difference loop would name other shells"});
        }
    }

    return displaced;
}

/// The fixture's water with the symmetry-breaking displacement applied.
/// \param base The base molecule.
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> DisplacedBase(const qcx::molecule::Molecule& base) {
    if (base.AtomCount() != kBaseShifts.size() / 3)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "the fixture is not a three-atom water"});
    }

    std::array<std::size_t, 9> coordinates{};

    for (std::size_t flat = 0; flat < coordinates.size(); ++flat)
    {
        coordinates[flat] = flat;
    }

    return Moved(base, coordinates, kBaseShifts);
}

/// The geometry with one flat coordinate displaced.
/// \param base The base molecule.
/// \param flatCoordinate The coordinate (3 * atom + axis) to displace.
/// \param delta The displacement (Bohr).
/// \returns The displaced molecule, or an Error.
qcx::Result<qcx::molecule::Molecule> MakeDisplaced(const qcx::molecule::Molecule& base,
                                                   std::size_t flatCoordinate,
                                                   double delta) {
    const std::array<std::size_t, 1> coordinates = {flatCoordinate};
    const std::array<double, 1> deltas = {delta};
    return Moved(base, coordinates, deltas);
}

/// The fixture's water, its basis set and the fixed MO coefficients.
struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basisSet;
    Eigen::MatrixXd coefficients; ///< nBasis x nMo, columns the orbitals.
    std::vector<double> rowMajor; ///< The same coefficients in the entry's layout.
    std::size_t numOccupied = 0;
    std::size_t numVirtual = 0;
};

/// Builds the fixture and its core-Hamiltonian eigenvectors: the occupied
/// count is the electron pairs, the virtual count is every remaining
/// eigenvector, so the MO basis is the whole basis.
/// \returns The fixture, or an Error.
qcx::Result<Fixture> MakeFixture() {
    auto base = qcx::testing::MakeH2oSto3g();

    if (!base.has_value())
    {
        return std::unexpected(base.error());
    }

    auto molecule = DisplacedBase(*base);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto basisSet = qcx::testing::MakeH2oSto3gBasis();

    if (!basisSet.has_value())
    {
        return std::unexpected(basisSet.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto attraction = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basisSet);

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
                                          "the core-Hamiltonian eigenvector solve failed"});
    }

    const std::size_t numOccupied =
        static_cast<std::size_t>(std::max(molecule->ElectronCount(), 0) / 2);
    const std::size_t numVirtual = static_cast<std::size_t>(metric.rows()) - numOccupied;
    const Eigen::MatrixXd coefficients =
        solver.eigenvectors().leftCols(static_cast<Eigen::Index>(numOccupied + numVirtual));

    const Eigen::Index n = coefficients.rows();
    const Eigen::Index nMo = coefficients.cols();
    std::vector<double> rowMajor(static_cast<std::size_t>(n * nMo));

    for (Eigen::Index p = 0; p < n; ++p)
    {
        for (Eigen::Index q = 0; q < nMo; ++q)
        {
            rowMajor[static_cast<std::size_t>(p * nMo + q)] = coefficients(p, q);
        }
    }

    return Fixture{std::move(*molecule),
                   std::move(*basisSet),
                   coefficients,
                   std::move(rowMajor),
                   numOccupied,
                   numVirtual};
}

/// One built operator matrix in the MO basis, with the fixture's fixed
/// coefficients: C^T X C.
/// \param matrix The builder's result.
/// \param coefficients The fixed MO coefficients.
/// \returns The MO-basis matrix, or the builder's error.
qcx::Result<Eigen::MatrixXd> TransformAt(qcx::Result<CpuTensor2> matrix,
                                         const Eigen::MatrixXd& coefficients) {
    if (!matrix.has_value())
    {
        return std::unexpected(matrix.error());
    }

    return coefficients.transpose() * qcx::testing::ToMatrix(*matrix) * coefficients;
}

/// One coordinate's central-difference MO matrix of a two-matrix operator:
/// (X(+h) - X(-h)) / 2h in the MO basis.
/// \param plus The MO matrix at +h.
/// \param minus The MO matrix at -h.
/// \returns The central difference.
Eigen::MatrixXd CentralDifference(const Eigen::MatrixXd& plus, const Eigen::MatrixXd& minus) {
    return (plus - minus) / (2.0 * kFiniteDifferenceStep);
}

} // namespace

/// The overlap derivative in the whole MO basis is the finite difference of
/// C^T S C at the fixed coefficients.
TEST(OneElectronFockDerivativeTest, TheOverlapBlockIsTheFiniteDifference) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto derivative = qcx::integrals::ComputeOneElectronFockDerivative(fixture->molecule,
                                                                       fixture->basisSet,
                                                                       fixture->rowMajor,
                                                                       fixture->numOccupied,
                                                                       fixture->numVirtual);

    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;

    const std::size_t nMo = fixture->numOccupied + fixture->numVirtual;
    ASSERT_EQ(derivative->moOverlapDerivative.size(),
              static_cast<std::size_t>(3 * fixture->molecule.AtomCount()) * nMo * nMo);

    double worst = 0.0;
    std::size_t worstCoordinate = 0;

    for (std::size_t flat = 0; flat < 3 * fixture->molecule.AtomCount(); ++flat)
    {
        auto plus = MakeDisplaced(fixture->molecule, flat, kFiniteDifferenceStep);
        auto minus = MakeDisplaced(fixture->molecule, flat, -kFiniteDifferenceStep);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusBlock = TransformAt(qcx::integrals::BuildOverlapMatrix(*plus, fixture->basisSet),
                                     fixture->coefficients);
        auto minusBlock = TransformAt(qcx::integrals::BuildOverlapMatrix(*minus, fixture->basisSet),
                                      fixture->coefficients);

        ASSERT_TRUE(plusBlock.has_value()) << plusBlock.error().message;
        ASSERT_TRUE(minusBlock.has_value()) << minusBlock.error().message;

        const Eigen::MatrixXd numeric = CentralDifference(*plusBlock, *minusBlock);

        for (std::size_t p = 0; p < nMo; ++p)
        {
            for (std::size_t q = 0; q < nMo; ++q)
            {
                const double analytic = derivative->moOverlapDerivative[(flat * nMo + p) * nMo + q];
                const double deviation = std::abs(
                    analytic - numeric(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)));

                if (deviation > worst)
                {
                    worst = deviation;
                    worstCoordinate = flat;
                }
            }
        }
    }

    RecordProperty("worst_deviation", ShortDouble(worst));
    RecordProperty("block_scale",
                   ShortDouble(*std::max_element(
                       derivative->moOverlapDerivative.begin(),
                       derivative->moOverlapDerivative.end(),
                       [](double a, double b) { return std::abs(a) < std::abs(b); })));
    RecordProperty("orbital_pairs", std::to_string(derivative->counts.orbitalPairs));
    EXPECT_LT(worst, kFiniteDifferenceTolerance) << "coordinate " << worstCoordinate;
}

/// The core-Hamiltonian derivative in the occupied-virtual block is the
/// finite difference of C^T (T + V) C at the fixed coefficients. The nuclear
/// attraction's derivative carries the nucleus and the basis functions it
/// attracts alike, so a walk that differentiated only one of the two fails
/// here.
TEST(OneElectronFockDerivativeTest, TheHamiltonianBlockIsTheFiniteDifference) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto derivative = qcx::integrals::ComputeOneElectronFockDerivative(fixture->molecule,
                                                                       fixture->basisSet,
                                                                       fixture->rowMajor,
                                                                       fixture->numOccupied,
                                                                       fixture->numVirtual);

    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;
    ASSERT_EQ(derivative->occupiedVirtual.rows(),
              static_cast<Eigen::Index>(3 * fixture->molecule.AtomCount() * fixture->numVirtual));
    ASSERT_EQ(derivative->occupiedVirtual.cols(), static_cast<Eigen::Index>(fixture->numOccupied));

    double worst = 0.0;
    std::size_t worstCoordinate = 0;

    for (std::size_t flat = 0; flat < 3 * fixture->molecule.AtomCount(); ++flat)
    {
        auto plus = MakeDisplaced(fixture->molecule, flat, kFiniteDifferenceStep);
        auto minus = MakeDisplaced(fixture->molecule, flat, -kFiniteDifferenceStep);

        ASSERT_TRUE(plus.has_value()) << plus.error().message;
        ASSERT_TRUE(minus.has_value()) << minus.error().message;

        auto plusKinetic = TransformAt(qcx::integrals::BuildKineticMatrix(*plus, fixture->basisSet),
                                       fixture->coefficients);
        auto minusKinetic = TransformAt(
            qcx::integrals::BuildKineticMatrix(*minus, fixture->basisSet), fixture->coefficients);
        auto plusAttraction =
            TransformAt(qcx::integrals::BuildNuclearAttractionMatrix(*plus, fixture->basisSet),
                        fixture->coefficients);
        auto minusAttraction =
            TransformAt(qcx::integrals::BuildNuclearAttractionMatrix(*minus, fixture->basisSet),
                        fixture->coefficients);

        ASSERT_TRUE(plusKinetic.has_value()) << plusKinetic.error().message;
        ASSERT_TRUE(minusKinetic.has_value()) << minusKinetic.error().message;
        ASSERT_TRUE(plusAttraction.has_value()) << plusAttraction.error().message;
        ASSERT_TRUE(minusAttraction.has_value()) << minusAttraction.error().message;

        const Eigen::MatrixXd numeric = CentralDifference(*plusKinetic, *minusKinetic) +
                                        CentralDifference(*plusAttraction, *minusAttraction);

        for (std::size_t a = 0; a < fixture->numVirtual; ++a)
        {
            for (std::size_t i = 0; i < fixture->numOccupied; ++i)
            {
                const Eigen::Index row = static_cast<Eigen::Index>(flat * fixture->numVirtual + a);
                const Eigen::Index column = static_cast<Eigen::Index>(i);
                const Eigen::Index moRow = static_cast<Eigen::Index>(fixture->numOccupied + a);
                const double deviation =
                    std::abs(derivative->occupiedVirtual(row, column) - numeric(moRow, column));

                if (deviation > worst)
                {
                    worst = deviation;
                    worstCoordinate = flat;
                }
            }
        }
    }

    RecordProperty("worst_deviation", ShortDouble(worst));
    RecordProperty("block_scale", ShortDouble(derivative->occupiedVirtual.cwiseAbs().maxCoeff()));
    RecordProperty("hamiltonian_tuples", std::to_string(derivative->counts.hamiltonianTuples));
    EXPECT_LT(worst, kFiniteDifferenceTolerance) << "coordinate " << worstCoordinate;
}

/// Translating every atom together changes no matrix, so for each element of
/// either block the sum over the atoms of one axis is zero. A block
/// derivative put on the wrong coordinate cannot satisfy it.
TEST(OneElectronFockDerivativeTest, TheBlocksAreTranslationInvariant) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto derivative = qcx::integrals::ComputeOneElectronFockDerivative(fixture->molecule,
                                                                       fixture->basisSet,
                                                                       fixture->rowMajor,
                                                                       fixture->numOccupied,
                                                                       fixture->numVirtual);

    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;

    const std::size_t atomCount = fixture->molecule.AtomCount();
    const std::size_t nMo = fixture->numOccupied + fixture->numVirtual;
    double worstOverlap = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (std::size_t p = 0; p < nMo; ++p)
        {
            for (std::size_t q = 0; q < nMo; ++q)
            {
                double sum = 0.0;

                for (std::size_t atom = 0; atom < atomCount; ++atom)
                {
                    const std::size_t flat = 3 * atom + axis;
                    sum += derivative->moOverlapDerivative[(flat * nMo + p) * nMo + q];
                }

                worstOverlap = std::max(worstOverlap, std::abs(sum));
            }
        }
    }

    double worstHamiltonian = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (std::size_t a = 0; a < fixture->numVirtual; ++a)
        {
            for (std::size_t i = 0; i < fixture->numOccupied; ++i)
            {
                double sum = 0.0;

                for (std::size_t atom = 0; atom < atomCount; ++atom)
                {
                    const std::size_t flat = 3 * atom + axis;
                    sum += derivative->occupiedVirtual(
                        static_cast<Eigen::Index>(flat * fixture->numVirtual + a),
                        static_cast<Eigen::Index>(i));
                }

                worstHamiltonian = std::max(worstHamiltonian, std::abs(sum));
            }
        }
    }

    RecordProperty("worst_overlap_sum", ShortDouble(worstOverlap));
    RecordProperty("worst_hamiltonian_sum", ShortDouble(worstHamiltonian));
    EXPECT_LT(worstOverlap, kTranslationTolerance);
    EXPECT_LT(worstHamiltonian, kTranslationTolerance);
}

/// The screening structure the entry accepts is inert by contract: this walk
/// forms no pair screen, so a carried structure does not change a number.
TEST(OneElectronFockDerivativeTest, TheWarmStartIsInert) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto warmStart =
        qcx::integrals::BuildFockDerivativeWarmStart(fixture->molecule, fixture->basisSet);
    ASSERT_TRUE(warmStart.has_value()) << warmStart.error().message;
    warmStart->maxAtomMotionBohr = 0.25;
    qcx::integrals::OneElectronFockDerivativeOptions options;
    options.warmStart = *warmStart;

    auto lifted = qcx::integrals::ComputeOneElectronFockDerivative(fixture->molecule,
                                                                   fixture->basisSet,
                                                                   fixture->rowMajor,
                                                                   fixture->numOccupied,
                                                                   fixture->numVirtual,
                                                                   options);
    auto plain = qcx::integrals::ComputeOneElectronFockDerivative(fixture->molecule,
                                                                  fixture->basisSet,
                                                                  fixture->rowMajor,
                                                                  fixture->numOccupied,
                                                                  fixture->numVirtual);

    ASSERT_TRUE(lifted.has_value()) << lifted.error().message;
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_EQ(lifted->moOverlapDerivative, plain->moOverlapDerivative);
    EXPECT_EQ((lifted->occupiedVirtual - plain->occupiedVirtual).cwiseAbs().maxCoeff(), 0.0);
}

/// An empty orbital block is a successful zero, not a walk: the entry returns
/// the right shapes and no numbers.
TEST(OneElectronFockDerivativeTest, AnEmptyOrbitalBlockIsZero) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    auto derivative = qcx::integrals::ComputeOneElectronFockDerivative(
        fixture->molecule, fixture->basisSet, std::span<const double>(), 0, 0);

    ASSERT_TRUE(derivative.has_value()) << derivative.error().message;
    EXPECT_TRUE(derivative->moOverlapDerivative.empty());
    EXPECT_EQ(derivative->occupiedVirtual.rows(), 0);
    EXPECT_EQ(derivative->occupiedVirtual.cols(), 0);
    EXPECT_EQ(derivative->counts.orbitalPairs, 0);
}

/// The entry's two documented refusals: a virtual count that leaves the
/// occupied block past the basis, and a coefficient span of the wrong size.
TEST(OneElectronFockDerivativeTest, TheEntryNamesItsRefusals) {
    auto fixture = MakeFixture();

    if (!fixture.has_value())
    {
        ADD_FAILURE() << fixture.error().message;
        return;
    }

    const std::size_t n = static_cast<std::size_t>(fixture->coefficients.rows());

    {
        const std::size_t occupied = n - 1;
        const std::size_t virtualOrbitals = 2;
        std::vector<double> coefficients(n * (occupied + virtualOrbitals), 0.0);
        auto refused = qcx::integrals::ComputeOneElectronFockDerivative(
            fixture->molecule, fixture->basisSet, coefficients, occupied, virtualOrbitals);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("exceed the"), std::string::npos)
            << refused.error().message;
    }

    {
        std::vector<double> coefficients(fixture->rowMajor.begin(), fixture->rowMajor.end() - 1);
        auto refused = qcx::integrals::ComputeOneElectronFockDerivative(fixture->molecule,
                                                                        fixture->basisSet,
                                                                        coefficients,
                                                                        fixture->numOccupied,
                                                                        fixture->numVirtual);
        ASSERT_FALSE(refused.has_value());
        EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(refused.error().message.find("MO coefficient span holds"), std::string::npos)
            << refused.error().message;
    }
}
