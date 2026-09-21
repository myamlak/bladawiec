#include "h2_sto3g.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/response/response_operator.hpp"
#include "qcx/response/response_solver.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <vector>

// The synthetic band of the response solver's suite: every fixture below is a
// made-up matrix with a known answer, so a failure here names linear algebra
// before any chemistry is involved. Two checks in the file are chemistry-shaped:
// the orbital-Hessian action against a numerical energy Hessian of made-up
// integrals (the operator's formula is the piece every downstream derivative
// sits on), and the finite-field polarizability of real H2/STO-3G at the end,
// where the solver meets a converged SCF.

namespace {

using qcx::response::DenseResponseOperator;
using qcx::response::MoTwoElectronTensor;
using qcx::response::OrbitalEnergies;
using qcx::response::OrbitalHessianOperator;
using qcx::response::PreconditionerKind;
using qcx::response::PreparePreconditioner;
using qcx::response::ResponseLayout;
using qcx::response::ResponseSolveOptions;
using qcx::response::ResponseSolver;
using qcx::response::ToString;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

/// A fixture's dimension: the count of orbitals, rows or columns it is drawn at.
///
/// Its own type so that the seed and the size a fixture is drawn with cannot be
/// handed over in each other's place.
struct Dimension {
    std::size_t value = 0; ///< The count.
};

/// The seed a fixture's generator is started from.
struct Seed {
    std::uint32_t value = 0; ///< The generator seed.
};

/// A reproducible symmetric matrix with entries drawn uniformly from
/// [-scale, scale].
Eigen::MatrixXd MakeSymmetricMatrix(Dimension dimension, Seed seed, double scale) {
    std::mt19937 generator(seed.value);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    const int size = static_cast<int>(dimension.value);
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(size, size);

    for (int row = 0; row < size; ++row)
    {
        for (int column = row; column < size; ++column)
        {
            const double value = scale * distribution(generator);
            matrix(row, column) = value;
            matrix(column, row) = value;
        }
    }

    return matrix;
}

/// A strictly increasing diagonal spanning [1, largest] geometrically.
std::vector<double> MakeSpreadDiagonal(Dimension dimension, double largest) {
    std::vector<double> diagonal(dimension.value);

    for (std::size_t i = 0; i < dimension.value; ++i)
    {
        const double exponent = static_cast<double>(i) / static_cast<double>(dimension.value - 1);
        diagonal[i] = std::pow(largest, exponent);
    }

    return diagonal;
}

/// The 1D chain Laplacian with Dirichlet ends, used as a coupling block.
Eigen::MatrixXd MakeChainLaplacian(std::size_t dimension) {
    const int size = static_cast<int>(dimension);
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(size, size);

    for (int i = 0; i < size; ++i)
    {
        laplacian(i, i) = 2.0;

        if (i > 0)
        {
            laplacian(i, i - 1) = -1.0;
            laplacian(i - 1, i) = -1.0;
        }
    }

    return laplacian;
}

/// Row-major flattening of a dense matrix.
std::vector<double> Flatten(const Eigen::MatrixXd& matrix) {
    const int rows = static_cast<int>(matrix.rows());
    const int columns = static_cast<int>(matrix.cols());
    std::vector<double> flat(static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns));

    for (int row = 0; row < rows; ++row)
    {
        for (int column = 0; column < columns; ++column)
        {
            // The stride widens before the multiply: a large matrix's flat index
            // would otherwise be computed in int and overflow there.
            const std::size_t offset =
                static_cast<std::size_t>(row) * static_cast<std::size_t>(columns) +
                static_cast<std::size_t>(column);
            flat[offset] = matrix(row, column);
        }
    }

    return flat;
}

/// Builds the dense member of the solver ladder from an Eigen matrix, with a
/// preconditioner of the caller's choosing.
qcx::Result<DenseResponseOperator> MakeDenseOperator(std::size_t numOccupied,
                                                     std::size_t numVirtual,
                                                     const Eigen::MatrixXd& matrix,
                                                     std::span<const double> preconditioner) {
    const std::vector<double> flat = Flatten(matrix);

    return DenseResponseOperator::Create(
        ResponseLayout{numOccupied, numVirtual}, flat, preconditioner);
}

/// Builds an operator whose preconditioner is its own diagonal.
qcx::Result<DenseResponseOperator> MakeDenseOperator(std::size_t numOccupied,
                                                     std::size_t numVirtual,
                                                     const Eigen::MatrixXd& matrix) {
    const std::vector<double> flat = Flatten(matrix);

    return DenseResponseOperator::Create(ResponseLayout{numOccupied, numVirtual}, flat);
}

/// Right-hand sides, contiguous and right-hand-side-major.
std::vector<double> MakeRightHandSides(Dimension dimension, std::size_t count, Seed seed) {
    std::mt19937 generator(seed.value);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    std::vector<double> sides(count * dimension.value);

    for (double& value : sides)
    {
        value = distribution(generator);
    }

    return sides;
}

/// The diagonal of a dense matrix, as a vector.
std::vector<double> Diagonal(const Eigen::MatrixXd& matrix) {
    std::vector<double> diagonal(static_cast<std::size_t>(matrix.rows()));

    for (int i = 0; i < static_cast<int>(matrix.rows()); ++i)
    {
        diagonal[static_cast<std::size_t>(i)] = matrix(i, i);
    }

    return diagonal;
}

// --- the orbital-Hessian formula -------------------------------------------

/// Made-up MO integrals with the eight-fold index symmetry a real two-electron
/// tensor has, a reference that is a converged canonical Hartree-Fock solution
/// of those integrals and that one-electron Hamiltonian, and the energy
/// functional the orbital-Hessian action is the second derivative of.
struct HessianFixture {
    std::size_t numOccupied = 0;
    std::size_t numVirtual = 0;
    std::vector<double> orbitalEnergies;
    std::vector<double> moTwoElectron;
    std::vector<double> coreHamiltonian;

    std::size_t NumOrbitals() const {
        return numOccupied + numVirtual;
    }

    double Integral(std::size_t p, std::size_t q, std::size_t r, std::size_t s) const {
        const std::size_t n = NumOrbitals();
        return moTwoElectron[((p * n + q) * n + r) * n + s];
    }

    double Core(std::size_t p, std::size_t q) const {
        return coreHamiltonian[p * NumOrbitals() + q];
    }

    /// The closed-shell Hartree-Fock energy as a function of the occupied-virtual
    /// rotation amplitudes in the MO frame: the occupied block is rotated by the
    /// anti-Hermitian generator the amplitudes parametrise, and renormalised so
    /// the orbitals stay orthonormal. The reference is stationary, because the
    /// one-electron Hamiltonian was built to make it so.
    double Energy(std::span<const double> amplitudes) const {
        const int nOrb = static_cast<int>(NumOrbitals());

        const Eigen::MatrixXd mixing = Mixing(amplitudes);
        const Eigen::MatrixXd overlap = mixing.transpose() * mixing;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> root(overlap);
        const Eigen::MatrixXd inverseRoot =
            root.eigenvectors() * root.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
            root.eigenvectors().transpose();
        const Eigen::MatrixXd orbitals = mixing * inverseRoot;
        const Eigen::MatrixXd density = 2.0 * orbitals * orbitals.transpose();

        // The one-electron part is a full contraction against the core
        // Hamiltonian, which is not diagonal: it is a core Hamiltonian, not the
        // orbital energies.
        double energy = 0.0;

        for (int p = 0; p < nOrb; ++p)
        {
            for (int q = 0; q < nOrb; ++q)
            {
                energy +=
                    density(p, q) * Core(static_cast<std::size_t>(p), static_cast<std::size_t>(q));
            }
        }

        double twoElectron = 0.0;

        for (int p = 0; p < nOrb; ++p)
        {
            for (int q = 0; q < nOrb; ++q)
            {
                for (int r = 0; r < nOrb; ++r)
                {
                    for (int s = 0; s < nOrb; ++s)
                    {
                        const auto P = static_cast<std::size_t>(p);
                        const auto Q = static_cast<std::size_t>(q);
                        const auto R = static_cast<std::size_t>(r);
                        const auto S = static_cast<std::size_t>(s);
                        twoElectron += density(p, q) * density(r, s) *
                                       (Integral(P, Q, R, S) - 0.5 * Integral(P, R, Q, S));
                    }
                }
            }
        }

        return energy + 0.5 * twoElectron;
    }

private:
    /// The mixing block [I_occ ; kappa], one column per occupied orbital.
    Eigen::MatrixXd Mixing(std::span<const double> amplitudes) const {
        const int nOcc = static_cast<int>(numOccupied);
        const int nVirt = static_cast<int>(numVirtual);
        const int nOrb = static_cast<int>(NumOrbitals());
        Eigen::MatrixXd mixing = Eigen::MatrixXd::Zero(nOrb, nOcc);

        for (int i = 0; i < nOcc; ++i)
        {
            mixing(i, i) = 1.0;
        }

        for (int a = 0; a < nVirt; ++a)
        {
            for (int i = 0; i < nOcc; ++i)
            {
                const std::size_t offset =
                    static_cast<std::size_t>(i) * static_cast<std::size_t>(nVirt) +
                    static_cast<std::size_t>(a);
                mixing(nOcc + a, i) = amplitudes[offset];
            }
        }

        return mixing;
    }
};

/// The action as a dense matrix, one unit vector at a time.
Eigen::MatrixXd DenseAction(const OrbitalHessianOperator& op) {
    const std::size_t dimension = op.Dimension();
    const int size = static_cast<int>(dimension);
    Eigen::MatrixXd action = Eigen::MatrixXd::Zero(size, size);
    std::vector<double> trial(dimension, 0.0);
    std::vector<double> product(dimension, 0.0);

    for (std::size_t column = 0; column < dimension; ++column)
    {
        std::fill(trial.begin(), trial.end(), 0.0);
        trial[column] = 1.0;
        EXPECT_TRUE(op.Apply(trial, product).has_value());

        for (std::size_t row = 0; row < dimension; ++row)
        {
            action(static_cast<int>(row), static_cast<int>(column)) = product[row];
        }
    }

    return action;
}

/// Builds the fixture with random integrals carrying the eight-fold symmetry
/// (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq), occupied orbital energies below the
/// virtual ones, and the core Hamiltonian those two determine.
HessianFixture MakeHessianFixture(std::size_t numOccupied, std::size_t numVirtual, Seed seed) {
    const std::size_t n = numOccupied + numVirtual;
    std::mt19937 generator(seed.value);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    // One value per unordered pair of unordered pairs: the symmetric pattern.
    const std::size_t pairCount = n * (n + 1) / 2;
    std::vector<double> patterns(pairCount * pairCount, 0.0);

    for (std::size_t first = 0; first < pairCount; ++first)
    {
        for (std::size_t second = first; second < pairCount; ++second)
        {
            const double value = distribution(generator);
            patterns[first * pairCount + second] = value;
            patterns[second * pairCount + first] = value;
        }
    }

    const auto pairIndex = [n](std::size_t p, std::size_t q) {
        const std::size_t low = p < q ? p : q;
        const std::size_t high = p < q ? q : p;
        return low * (2 * n - low + 1) / 2 + (high - low);
    };

    HessianFixture fixture;
    fixture.numOccupied = numOccupied;
    fixture.numVirtual = numVirtual;
    fixture.moTwoElectron.assign(n * n * n * n, 0.0);

    for (std::size_t p = 0; p < n; ++p)
    {
        for (std::size_t q = 0; q < n; ++q)
        {
            for (std::size_t r = 0; r < n; ++r)
            {
                for (std::size_t s = 0; s < n; ++s)
                {
                    const std::size_t first = pairIndex(p, q);
                    const std::size_t second = pairIndex(r, s);
                    fixture.moTwoElectron[((p * n + q) * n + r) * n + s] =
                        patterns[first * pairCount + second];
                }
            }
        }
    }

    fixture.orbitalEnergies.assign(n, 0.0);

    for (std::size_t i = 0; i < numOccupied; ++i)
    {
        fixture.orbitalEnergies[i] = -0.4 - 0.2 * static_cast<double>(i);
    }

    for (std::size_t a = 0; a < numVirtual; ++a)
    {
        fixture.orbitalEnergies[numOccupied + a] = 0.3 + 0.4 * static_cast<double>(a);
    }

    // The one-electron Hamiltonian is determined rather than drawn: it is
    // whatever makes the reference - the first numOccupied MOs, with the
    // density those orbitals give - a converged canonical Hartree-Fock
    // solution carrying exactly the orbital energies above. The Fock matrix of
    // that reference is diagonal in this basis, so the Brillouin condition
    // holds, the energy is stationary there, and the orbital energies are Fock
    // eigenvalues with the reference's own two-electron terms already inside
    // them. That is the (A + B) form's domain of validity; with a core
    // Hamiltonian and orbital energies chosen independently, the second
    // derivative of the same functional is some other matrix.
    fixture.coreHamiltonian.assign(n * n, 0.0);

    for (std::size_t p = 0; p < n; ++p)
    {
        for (std::size_t q = 0; q < n; ++q)
        {
            double meanField = 0.0;

            for (std::size_t r = 0; r < numOccupied; ++r)
            {
                for (std::size_t s = 0; s < numOccupied; ++s)
                {
                    // The reference density is 2 on the occupied diagonal.
                    const double density = r == s ? 2.0 : 0.0;
                    meanField += density * (fixture.Integral(p, q, r, s) -
                                            0.5 * fixture.Integral(p, r, q, s));
                }
            }

            const double fock = p == q ? fixture.orbitalEnergies[p] : 0.0;
            fixture.coreHamiltonian[p * n + q] = fock - meanField;
        }
    }

    return fixture;
}

} // namespace

// --- C5: the synthetic suite -----------------------------------------------

TEST(ResponseSolverTest, DenseOperatorAppliesItsMatrixAndKeepsItsOwnPreconditioner) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 3;
    const std::size_t dimension = kOccupied * kVirtual;
    const Eigen::MatrixXd matrix = MakeSymmetricMatrix(Dimension{dimension}, Seed{20260920u}, 1.0);
    const std::vector<double> preconditioner = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const auto operatorResult = MakeDenseOperator(kOccupied, kVirtual, matrix, preconditioner);

    ASSERT_TRUE(operatorResult.has_value());
    const DenseResponseOperator& op = *operatorResult;
    EXPECT_EQ(op.Dimension(), dimension);
    EXPECT_EQ(op.Layout().numOccupied, kOccupied);
    EXPECT_EQ(op.Layout().numVirtual, kVirtual);

    std::vector<double> product(dimension);
    std::vector<double> trial(dimension, 0.0);

    for (std::size_t column = 0; column < dimension; ++column)
    {
        std::fill(trial.begin(), trial.end(), 0.0);
        trial[column] = 1.0;
        ASSERT_TRUE(op.Apply(trial, product).has_value());

        for (std::size_t row = 0; row < dimension; ++row)
        {
            EXPECT_NEAR(
                product[row], matrix(static_cast<int>(row), static_cast<int>(column)), 1e-12);
        }
    }

    // The preconditioner is what it was handed, not the matrix's own diagonal
    // (which here is a different vector): the distinction a physical response
    // operator depends on.
    std::vector<double> reported(dimension);
    ASSERT_TRUE(op.Preconditioner(reported).has_value());

    for (std::size_t i = 0; i < dimension; ++i)
    {
        EXPECT_DOUBLE_EQ(reported[i], preconditioner[i]);
    }
}

TEST(ResponseSolverTest, OperatorConstructionRefusesMismatchedShapes) {
    const std::vector<double> energies = {-0.5, -0.3, 0.2, 0.4};
    constexpr std::size_t kOrbitals = 4;
    const std::vector<double> integrals(kOrbitals * kOrbitals * kOrbitals * kOrbitals, 0.0);
    const std::vector<double> shortEnergies = {1.0, 2.0};
    const std::vector<double> shortIntegrals = {1.0, 2.0};

    EXPECT_EQ(OrbitalHessianOperator::Create(
                  ResponseLayout{0, 2}, OrbitalEnergies{energies}, MoTwoElectronTensor{integrals})
                  .error()
                  .code,
              qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(OrbitalHessianOperator::Create(
                  ResponseLayout{2, 0}, OrbitalEnergies{energies}, MoTwoElectronTensor{integrals})
                  .error()
                  .code,
              qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(OrbitalHessianOperator::Create(ResponseLayout{2, 2},
                                             OrbitalEnergies{shortEnergies},
                                             MoTwoElectronTensor{integrals})
                  .error()
                  .code,
              qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(OrbitalHessianOperator::Create(ResponseLayout{2, 2},
                                             OrbitalEnergies{energies},
                                             MoTwoElectronTensor{shortIntegrals})
                  .error()
                  .code,
              qcx::ErrorCode::kInvalidArgument);

    const std::vector<double> goodMatrix(16, 0.0);
    const std::vector<double> shortMatrix(9, 0.0);
    const std::vector<double> shortPreconditioner = {1.0};
    EXPECT_EQ(DenseResponseOperator::Create(ResponseLayout{2, 2}, shortMatrix).error().code,
              qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(DenseResponseOperator::Create(ResponseLayout{2, 2}, goodMatrix, shortPreconditioner)
                  .error()
                  .code,
              qcx::ErrorCode::kInvalidArgument);

    const auto op = DenseResponseOperator::Create(ResponseLayout{2, 2}, goodMatrix);
    ASSERT_TRUE(op.has_value());
    std::vector<double> wrongSize(3);
    EXPECT_EQ(op->Apply(wrongSize, wrongSize).error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(op->Preconditioner(wrongSize).error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(ResponseSolverTest, OrbitalHessianPreconditionerIsTheOrbitalEnergyDifference) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 3;
    const std::vector<double> energies = {-0.9, -0.5, 0.2, 0.4, 0.8};
    constexpr std::size_t kOrbitals = kOccupied + kVirtual;
    const std::vector<double> integrals(kOrbitals * kOrbitals * kOrbitals * kOrbitals, 0.0);
    const auto op = OrbitalHessianOperator::Create(ResponseLayout{kOccupied, kVirtual},
                                                   OrbitalEnergies{energies},
                                                   MoTwoElectronTensor{integrals});

    ASSERT_TRUE(op.has_value());
    std::vector<double> diagonal(kOccupied * kVirtual);
    ASSERT_TRUE(op->Preconditioner(diagonal).has_value());

    for (std::size_t i = 0; i < kOccupied; ++i)
    {
        for (std::size_t a = 0; a < kVirtual; ++a)
        {
            EXPECT_DOUBLE_EQ(diagonal[i * kVirtual + a], energies[kOccupied + a] - energies[i]);
        }
    }
}

TEST(ResponseSolverTest, OrbitalHessianActionMatchesTheNumericalEnergyHessian) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 2;
    const std::size_t dimension = kOccupied * kVirtual;
    const HessianFixture fixture = MakeHessianFixture(kOccupied, kVirtual, Seed{20260921u});
    const auto op = OrbitalHessianOperator::Create(ResponseLayout{kOccupied, kVirtual},
                                                   OrbitalEnergies{fixture.orbitalEnergies},
                                                   MoTwoElectronTensor{fixture.moTwoElectron});

    ASSERT_TRUE(op.has_value()) << op.error().message;
    const Eigen::MatrixXd action = DenseAction(*op);

    // The action is symmetric, as the closed-shell (A + B) combination is.
    for (std::size_t row = 0; row < dimension; ++row)
    {
        for (std::size_t column = row + 1; column < dimension; ++column)
        {
            EXPECT_NEAR(action(static_cast<int>(row), static_cast<int>(column)),
                        action(static_cast<int>(column), static_cast<int>(row)),
                        1e-12);
        }
    }

    // The energy's second derivative is four times the action. At a stationary
    // reference the energy's first derivative in an amplitude is four times the
    // corresponding off-diagonal Fock element - a factor of two from the
    // closed-shell density and another from the symmetric sum - and the second
    // derivative of that Fock element with respect to the rotation is the
    // (A + B) combination the action applies.
    const double step = 1e-4;
    std::vector<double> perturbations(dimension, 0.0);

    for (std::size_t alpha = 0; alpha < dimension; ++alpha)
    {
        for (std::size_t beta = 0; beta < dimension; ++beta)
        {
            double energies[4] = {0.0, 0.0, 0.0, 0.0};

            for (std::size_t corner = 0; corner < 4; ++corner)
            {
                // Both corners accumulate: for alpha == beta the two
                // displacements add to a displacement of 2 * step along one
                // amplitude, which is what makes the difference below a second
                // difference in that direction rather than a first one.
                std::fill(perturbations.begin(), perturbations.end(), 0.0);
                perturbations[alpha] += (corner == 0 || corner == 1) ? step : -step;
                perturbations[beta] += (corner == 0 || corner == 2) ? step : -step;
                energies[corner] = fixture.Energy(perturbations);
            }

            const double numerical =
                (energies[0] - energies[1] - energies[2] + energies[3]) / (4.0 * step * step);
            EXPECT_NEAR(
                numerical, 4.0 * action(static_cast<int>(alpha), static_cast<int>(beta)), 1e-6)
                << "amplitude pair (" << alpha << ", " << beta << ")";
        }
    }
}

TEST(ResponseSolverTest, SolverMatchesADenseReferenceAcrossSeveralRightHandSides) {
    constexpr std::size_t kOccupied = 5;
    constexpr std::size_t kVirtual = 6;
    const std::size_t dimension = kOccupied * kVirtual;
    const Eigen::MatrixXd generator =
        MakeSymmetricMatrix(Dimension{dimension}, Seed{20260922u}, 1.0);
    const Eigen::MatrixXd matrix =
        generator.transpose() * generator +
        5.0 * Eigen::MatrixXd::Identity(static_cast<int>(dimension), static_cast<int>(dimension));
    const auto op = MakeDenseOperator(kOccupied, kVirtual, matrix);

    ASSERT_TRUE(op.has_value());
    const std::vector<double> rhs = MakeRightHandSides(Dimension{dimension}, 3, Seed{20260923u});
    std::vector<double> solution(3 * dimension);

    ResponseSolveOptions options;
    options.tolerance = 1e-9;
    ResponseSolver<DenseResponseOperator> solver(options);
    const auto report = solver.Solve(*op, 3, rhs, solution);

    ASSERT_TRUE(report.has_value()) << report.error().message;
    EXPECT_EQ(report->rhsCount, 3u);
    EXPECT_EQ(report->warmStarted, 0u);
    EXPECT_EQ(report->preconditioner, PreconditionerKind::kDiagonal);
    EXPECT_LE(report->worstResidual, 1e-9);

    Eigen::MatrixXd rightHandSide(static_cast<int>(dimension), 3);

    for (std::size_t k = 0; k < 3; ++k)
    {
        for (std::size_t i = 0; i < dimension; ++i)
        {
            rightHandSide(static_cast<int>(i), static_cast<int>(k)) = rhs[k * dimension + i];
        }
    }

    const Eigen::MatrixXd reference = matrix.ldlt().solve(rightHandSide);

    for (std::size_t k = 0; k < 3; ++k)
    {
        for (std::size_t i = 0; i < dimension; ++i)
        {
            EXPECT_NEAR(solution[k * dimension + i],
                        reference(static_cast<int>(i), static_cast<int>(k)),
                        1e-6)
                << "right-hand side " << k << ", element " << i;
        }
    }
}

TEST(ResponseSolverTest, SolverRefusesShapesAndEmptyInputs) {
    constexpr std::size_t kOccupied = 2;
    constexpr std::size_t kVirtual = 2;
    const std::size_t dimension = kOccupied * kVirtual;
    const Eigen::MatrixXd matrix =
        4.0 * Eigen::MatrixXd::Identity(static_cast<int>(dimension), static_cast<int>(dimension));
    const auto op = MakeDenseOperator(kOccupied, kVirtual, matrix);

    ASSERT_TRUE(op.has_value());
    const std::vector<double> rhs = MakeRightHandSides(Dimension{dimension}, 1, Seed{20260924u});
    std::vector<double> solution(2 * dimension);
    ResponseSolver<DenseResponseOperator> solver;

    const auto noSides = solver.Solve(*op, 0, rhs, solution);
    ASSERT_FALSE(noSides.has_value());
    EXPECT_EQ(noSides.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto shortRhs = solver.Solve(*op, 2, rhs, solution);
    ASSERT_FALSE(shortRhs.has_value());
    EXPECT_EQ(shortRhs.error().code, qcx::ErrorCode::kInvalidArgument);

    std::vector<double> shortBuffer(dimension - 1);
    const auto shortSolution = solver.Solve(*op, 1, rhs, shortBuffer);
    ASSERT_FALSE(shortSolution.has_value());
    EXPECT_EQ(shortSolution.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(ResponseSolverTest, PreconditionerRefusalsAreNamedRatherThanSilent) {
    EXPECT_EQ(ToString(PreconditionerKind::kNone), "none");
    EXPECT_EQ(ToString(PreconditionerKind::kDiagonal), "diagonal");
    EXPECT_EQ(ToString(PreconditionerKind::kRegularised), "regularised");

    ResponseSolveOptions options;
    const std::vector<double> nothing;
    const auto empty = PreparePreconditioner(nothing, options);
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, qcx::ErrorCode::kInvalidArgument);

    const std::vector<double> zeroed(3, 0.0);
    const auto zero = PreparePreconditioner(zeroed, options);
    ASSERT_FALSE(zero.has_value());
    EXPECT_EQ(zero.error().code, qcx::ErrorCode::kInvalidArgument);

    // A healthy diagonal with the regularised kind requested by name is a
    // refusal: nothing here needs the fallback, and substituting one silently
    // is what the project's rule forbids.
    const std::vector<double> healthy = {1.0, 0.9, 1.1};
    options.preconditioner = PreconditionerKind::kRegularised;
    const auto notNeeded = PreparePreconditioner(healthy, options);
    ASSERT_FALSE(notNeeded.has_value());
    EXPECT_EQ(notNeeded.error().code, qcx::ErrorCode::kInvalidArgument);

    // A small gap with the fallback disallowed is refused by name, and the
    // message says which fallback it refused.
    const std::vector<double> gapped = {1.0, 1e-9, 1.0};
    options.preconditioner = PreconditionerKind::kDiagonal;
    options.allowSmallGapFallback = false;
    const auto refused = PreparePreconditioner(gapped, options);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(refused.error().message.find("small-gap fallback"), std::string::npos)
        << refused.error().message;

    // The same diagonal with the fallback allowed is clamped to the default
    // floor of one thousandth of the largest entry, and it reports the ratio.
    options.allowSmallGapFallback = true;
    const auto clamped = PreparePreconditioner(gapped, options);
    ASSERT_TRUE(clamped.has_value());
    EXPECT_EQ(clamped->kind, PreconditionerKind::kRegularised);
    EXPECT_NEAR(clamped->gapRatio, 1e-9, 1e-15);
    EXPECT_NEAR(clamped->inverse[1], 1e3, 1e-6);

    // kNone inverts nothing, so the gap is measured and reported and nothing is
    // refused.
    options.preconditioner = PreconditionerKind::kNone;
    const auto identity = PreparePreconditioner(gapped, options);
    ASSERT_TRUE(identity.has_value());
    EXPECT_EQ(identity->kind, PreconditionerKind::kNone);
    EXPECT_DOUBLE_EQ(identity->inverse[1], 1.0);
}

// --- C2: the preconditioner earns its place ---------------------------------

TEST(ResponseSolverTest, DiagonalPreconditionerReducesTheIterationCount) {
    constexpr std::size_t kOccupied = 6;
    constexpr std::size_t kVirtual = 10;
    const std::size_t dimension = kOccupied * kVirtual;
    const std::vector<double> spread = MakeSpreadDiagonal(Dimension{dimension}, 200.0);
    Eigen::MatrixXd matrix = 0.5 * MakeChainLaplacian(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        matrix(static_cast<int>(i), static_cast<int>(i)) += spread[i];
    }

    // The operator's own diagonal - here the exact one, which is what the
    // solver's diagonal preconditioner is allowed to be handed.
    const std::vector<double> diagonal = Diagonal(matrix);
    const auto op = MakeDenseOperator(kOccupied, kVirtual, matrix, diagonal);
    ASSERT_TRUE(op.has_value());

    const std::vector<double> rhs = MakeRightHandSides(Dimension{dimension}, 1, Seed{20260925u});
    std::vector<double> withSolution(dimension);
    std::vector<double> withoutSolution(dimension);

    ResponseSolveOptions withDiagonal;
    ResponseSolveOptions without;
    without.preconditioner = PreconditionerKind::kNone;

    ResponseSolver<DenseResponseOperator> withSolver(withDiagonal);
    ResponseSolver<DenseResponseOperator> withoutSolver(without);

    const auto withReport = withSolver.Solve(*op, 1, rhs, withSolution);
    const auto withoutReport = withoutSolver.Solve(*op, 1, rhs, withoutSolution);

    ASSERT_TRUE(withReport.has_value()) << withReport.error().message;
    ASSERT_TRUE(withoutReport.has_value()) << withoutReport.error().message;
    EXPECT_EQ(withReport->preconditioner, PreconditionerKind::kDiagonal);
    EXPECT_EQ(withoutReport->preconditioner, PreconditionerKind::kNone);

    EXPECT_LT(withReport->iterations, withoutReport->iterations)
        << "preconditioned " << withReport->iterations << " iterations, unpreconditioned "
        << withoutReport->iterations;

    // The cost, printed so a green run still states what the preconditioner
    // bought.
    std::cout << "[preconditioner] diagonal " << withReport->iterations << " iterations, none "
              << withoutReport->iterations << "\n";

    // The preconditioner speeds the solve up; it does not change the answer.
    // The comparison is against the solve's own tolerance times the operator's
    // conditioning, not against its tolerance alone.
    for (std::size_t i = 0; i < dimension; ++i)
    {
        EXPECT_NEAR(withSolution[i], withoutSolution[i], 1e-5);
    }
}

// --- C3: the small-gap fallback ---------------------------------------------

TEST(ResponseSolverTest, SmallGapFallbackConvergesWhereThePlainPreconditionerStalls) {
    constexpr std::size_t kOccupied = 40;
    constexpr std::size_t kVirtual = 10;
    constexpr std::size_t kSides = 3;
    const std::size_t dimension = kOccupied * kVirtual;

    // The physical situation the fallback exists for: the preconditioner is the
    // orbital-energy difference across a dense virtual manifold, so its entries
    // span ten decades, while the operator's own diagonal stays flat near one -
    // the two-electron terms carry the curvature wherever an energy difference
    // nearly vanishes. The spread is continuous on purpose: one lone vanishing
    // entry is deflated by the first Krylov step, a decade-spanning manifold of
    // them is not, and the fallback is for the manifold.
    const std::vector<double> diagonal = MakeSpreadDiagonal(Dimension{dimension}, 1e-10);
    Eigen::MatrixXd matrix =
        Eigen::MatrixXd::Identity(static_cast<int>(dimension), static_cast<int>(dimension)) +
        0.2 * MakeChainLaplacian(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        matrix(static_cast<int>(i), static_cast<int>(i)) += diagonal[i];
    }

    const auto op = MakeDenseOperator(kOccupied, kVirtual, matrix, diagonal);

    ASSERT_TRUE(op.has_value());
    const std::vector<double> rhs =
        MakeRightHandSides(Dimension{dimension}, kSides, Seed{20260926u});
    std::vector<double> solution(kSides * dimension);

    ResponseSolveOptions options;
    options.maxIterations = 200;

    // The operator's true diagonal is about one, so a floor at half the largest
    // entry lands near it: the clamp recovers a preconditioner, which is the
    // claim under test. The library's default floor is far more conservative.
    options.regularisationRatio = 0.5;
    ResponseSolver<DenseResponseOperator> fallbackSolver(options);
    const auto fallback = fallbackSolver.Solve(*op, kSides, rhs, solution);

    ASSERT_TRUE(fallback.has_value()) << fallback.error().message;
    EXPECT_EQ(fallback->preconditioner, PreconditionerKind::kRegularised);
    EXPECT_EQ(fallback->regularised, kSides);
    EXPECT_NEAR(fallback->gapRatio, 1e-10, 1e-14);

    // The answer is an answer: check its residual independently of the solver's
    // own report.
    for (std::size_t k = 0; k < kSides; ++k)
    {
        Eigen::VectorXd side(static_cast<int>(dimension));
        Eigen::VectorXd answer(static_cast<int>(dimension));

        for (std::size_t i = 0; i < dimension; ++i)
        {
            side(static_cast<int>(i)) = rhs[k * dimension + i];
            answer(static_cast<int>(i)) = solution[k * dimension + i];
        }

        EXPECT_LT((matrix * answer - side).norm() / side.norm(), 1e-7);
    }

    // The same system with the gap undetected - the plain (e_a - e_i) inverse,
    // whose smallest entry is ten decades below its largest - does not reach the
    // tolerance in the same budget. That is the stall the fallback is for, and it
    // is asserted as the budget the plain run fails to meet, not as a number it
    // returns.
    ResponseSolveOptions plain = options;
    plain.smallGapRatio = 0.0;
    ResponseSolver<DenseResponseOperator> plainSolver(plain);
    std::vector<double> stalledSolution(kSides * dimension);
    const auto stalled = plainSolver.Solve(*op, kSides, rhs, stalledSolution);

    ASSERT_FALSE(stalled.has_value()) << "the plain (e_a - e_i) inverse converged in "
                                      << stalled->iterations << " iterations, inside the budget of "
                                      << options.maxIterations << " the fixture needs it to exceed";
    EXPECT_EQ(stalled.error().code, qcx::ErrorCode::kConvergenceFailure) << stalled.error().message;

    // The cost of both runs, printed so a green run still states what the
    // fallback bought.
    std::cout << "[small gap] fallback " << fallback->iterations
              << " iterations, plain did not reach the tolerance in " << options.maxIterations
              << " iterations\n";

    // And with the fallback disallowed outright, the refusal is by name.
    ResponseSolveOptions refused = options;
    refused.allowSmallGapFallback = false;
    ResponseSolver<DenseResponseOperator> refusingSolver(refused);
    const auto named = refusingSolver.Solve(*op, kSides, rhs, solution);
    ASSERT_FALSE(named.has_value());
    EXPECT_EQ(named.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(named.error().message.find("small-gap fallback"), std::string::npos)
        << named.error().message;
}

// --- C4: the reuse, asserted as a cost and a counter ------------------------

TEST(ResponseSolverTest, SecondSolveReusesTheStoredVectorsAndKeepsTheHistory) {
    constexpr std::size_t kOccupied = 8;
    constexpr std::size_t kVirtual = 5;
    constexpr std::size_t kSides = 2;
    const std::size_t dimension = kOccupied * kVirtual;
    const std::vector<double> spread = MakeSpreadDiagonal(Dimension{dimension}, 50.0);
    const Eigen::MatrixXd coupling =
        MakeSymmetricMatrix(Dimension{dimension}, Seed{20260927u}, 0.3);
    Eigen::MatrixXd first = coupling;

    for (std::size_t i = 0; i < dimension; ++i)
    {
        first(static_cast<int>(i), static_cast<int>(i)) += spread[i] + 5.0;
    }

    // The displaced operator: the same problem, slowly changed. In a real sweep
    // this is the orbitals, the density and the Fock matrix, all of which live
    // above this seam; what the solver owns is the solution it last produced.
    const Eigen::MatrixXd second =
        first + 1e-5 * MakeSymmetricMatrix(Dimension{dimension}, Seed{20260928u}, 1.0);
    const std::vector<double> diagonal = Diagonal(first);
    const auto firstOp = MakeDenseOperator(kOccupied, kVirtual, first, diagonal);
    const auto secondOp = MakeDenseOperator(kOccupied, kVirtual, second, diagonal);

    ASSERT_TRUE(firstOp.has_value());
    ASSERT_TRUE(secondOp.has_value());

    const std::vector<double> rhs =
        MakeRightHandSides(Dimension{dimension}, kSides, Seed{20260929u});
    std::vector<double> firstSolution(kSides * dimension);
    std::vector<double> warmSolution(kSides * dimension);
    std::vector<double> coldSolution(kSides * dimension);

    ResponseSolver<DenseResponseOperator> sweep;
    const auto cold = sweep.Solve(*firstOp, kSides, rhs, firstSolution);

    ASSERT_TRUE(cold.has_value()) << cold.error().message;
    EXPECT_EQ(cold->warmStarted, 0u);
    EXPECT_EQ(sweep.SolveCount(), kSides);
    EXPECT_EQ(sweep.ConvergenceHistory().size(), kSides);

    const auto warm = sweep.Solve(*secondOp, kSides, rhs, warmSolution);

    ASSERT_TRUE(warm.has_value()) << warm.error().message;

    // The state the solver kept: both sides started from a stored vector, the
    // history accumulated rather than restarted, and the stored vector is the
    // one this solve produced.
    EXPECT_EQ(warm->warmStarted, kSides);
    EXPECT_LT(warm->iterations, cold->iterations)
        << "warm " << warm->iterations << " iterations against a fresh " << cold->iterations;
    EXPECT_EQ(sweep.SolveCount(), 2 * kSides);
    ASSERT_EQ(sweep.ConvergenceHistory().size(), 2 * kSides);
    EXPECT_EQ(sweep.ConvergenceHistory()[2] + sweep.ConvergenceHistory()[3], warm->iterations);

    for (std::size_t k = 0; k < kSides; ++k)
    {
        const std::vector<double>& stored = sweep.PreviousSolution(k);
        ASSERT_EQ(stored.size(), dimension);

        for (std::size_t i = 0; i < dimension; ++i)
        {
            EXPECT_DOUBLE_EQ(stored[i], warmSolution[k * dimension + i]);
        }
    }

    // The control: a fresh solver against the SAME displaced operator, so the
    // saving is the warm start and not the perturbation. A stateless
    // implementation reports the cold numbers here.
    ResponseSolver<DenseResponseOperator> control;
    const auto controlReport = control.Solve(*secondOp, kSides, rhs, coldSolution);

    ASSERT_TRUE(controlReport.has_value()) << controlReport.error().message;
    EXPECT_EQ(controlReport->warmStarted, 0u);
    EXPECT_EQ(control.SolveCount(), kSides);
    EXPECT_LT(warm->iterations, controlReport->iterations)
        << "warm " << warm->iterations << " iterations against an unwarmed "
        << controlReport->iterations;

    // The cost, printed so a green run still states what the reuse bought.
    std::cout << "[reuse] fresh " << cold->iterations << " iterations, warm " << warm->iterations
              << ", unwarmed control on the same operator " << controlReport->iterations << "\n";

    for (std::size_t i = 0; i < kSides * dimension; ++i)
    {
        EXPECT_NEAR(warmSolution[i], coldSolution[i], 1e-5);
    }

    // A caller who wants a cold solve against the same operator says so.
    ResponseSolveOptions coldOptions;
    coldOptions.reusePrevious = false;
    ResponseSolver<DenseResponseOperator> coldSolver(coldOptions);
    std::vector<double> explicitSolution(kSides * dimension);
    ASSERT_TRUE(coldSolver.Solve(*firstOp, kSides, rhs, explicitSolution).has_value());
    const auto explicitCold = coldSolver.Solve(*secondOp, kSides, rhs, explicitSolution);
    ASSERT_TRUE(explicitCold.has_value());
    EXPECT_EQ(explicitCold->warmStarted, 0u);

    // A change of shape invalidates every stored vector rather than reusing
    // state that no longer describes the problem: after four recorded solves,
    // the history holds only the new ones.
    constexpr std::size_t kNarrowVirtual = 4;
    const std::size_t narrowed = kOccupied * kNarrowVirtual;
    Eigen::MatrixXd narrowMatrix =
        6.0 * Eigen::MatrixXd::Identity(static_cast<int>(narrowed), static_cast<int>(narrowed));
    const std::vector<double> narrowSpread = MakeSpreadDiagonal(Dimension{narrowed}, 20.0);

    for (std::size_t i = 0; i < narrowed; ++i)
    {
        narrowMatrix(static_cast<int>(i), static_cast<int>(i)) += narrowSpread[i];
    }

    const auto narrower = MakeDenseOperator(kOccupied, kNarrowVirtual, narrowMatrix);
    ASSERT_TRUE(narrower.has_value());
    const std::vector<double> narrowRhs =
        MakeRightHandSides(Dimension{narrowed}, kSides, Seed{20260930u});
    std::vector<double> narrowSolution(kSides * narrowed);
    const auto reshaped = sweep.Solve(*narrower, kSides, narrowRhs, narrowSolution);

    ASSERT_TRUE(reshaped.has_value()) << reshaped.error().message;
    EXPECT_EQ(reshaped->warmStarted, 0u);
    EXPECT_EQ(sweep.SolveCount(), kSides);
    EXPECT_EQ(sweep.ConvergenceHistory().size(), kSides);

    // Reset drops the state on demand, too.
    sweep.Reset();
    EXPECT_EQ(sweep.SolveCount(), 0u);
    EXPECT_TRUE(sweep.PreviousSolution(0).empty());
}

// --- C1: the finite-field anchor (H2/STO-3G in an applied electric field) ----

/// One MO coefficient, with the Eigen index cast kept in one place.
double Coefficient(const Eigen::MatrixXd& coefficients, std::size_t row, std::size_t column) {
    return coefficients(static_cast<int>(row), static_cast<int>(column));
}

/// The MO-basis two-electron tensor,
///     (pq|rs) = sum_{mu nu lam sig} C(mu,p) C(nu,q) C(lam,r) C(sig,s)
///                                   (mu nu|lam sig),
/// in chemists' notation and flattened the way OrbitalHessianOperator::Create
/// consumes it. Written as the explicit index sum rather than a chain of matrix
/// products: the anchor is H2/STO-3G, where n = 2, and the definition above is
/// what a reader checks this function against.
std::vector<double> TransformEriToMo(
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& coefficients) {
    const std::size_t n = static_cast<std::size_t>(coefficients.rows());
    std::vector<double> moEri(n * n * n * n, 0.0);

    for (std::size_t p = 0; p < n; ++p)
    {
        for (std::size_t q = 0; q < n; ++q)
        {
            for (std::size_t r = 0; r < n; ++r)
            {
                for (std::size_t s = 0; s < n; ++s)
                {
                    double value = 0.0;

                    for (std::size_t mu = 0; mu < n; ++mu)
                    {
                        for (std::size_t nu = 0; nu < n; ++nu)
                        {
                            for (std::size_t lambda = 0; lambda < n; ++lambda)
                            {
                                for (std::size_t sigma = 0; sigma < n; ++sigma)
                                {
                                    value += Coefficient(coefficients, mu, p) *
                                             Coefficient(coefficients, nu, q) *
                                             Coefficient(coefficients, lambda, r) *
                                             Coefficient(coefficients, sigma, s) *
                                             eri(mu, nu, lambda, sigma);
                                }
                            }
                        }
                    }

                    moEri[((p * n + q) * n + r) * n + s] = value;
                }
            }
        }
    }

    return moEri;
}

// The first physical case, and the one that needs no derivative integrals: a
// uniform electric field along the H2 bond axis. The field enters the core
// Hamiltonian as F * D, so the same SCF runs at three field strengths, and the
// analytic side is the response solver on the reference's own MO quantities.
//
// SCOPE, stated so this is not read as proving more than it does: both sides
// consume the same dipole matrix, so this pins the solver, the operator, the MO
// transform and the field-to-right-hand-side relation against the SCF - it does
// not re-validate the dipole integrals, which carry their own pin in
// integrals/tests.
TEST(ResponseSolverTest, H2Sto3gPolarizabilityMatchesTheFiniteFieldSecondDifference) {
    const auto basis = MakeSto3gBasis();
    const auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    const auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);
    const auto dipole = qcx::integrals::BuildDipoleMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;
    ASSERT_TRUE(eri.has_value()) << eri.error().message;
    ASSERT_TRUE(dipole.has_value()) << dipole.error().message;

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    // Component 0 is x: the H2 fixture puts both atoms on the x axis, so the
    // bond axis is the one direction with a nonzero response.
    const Eigen::MatrixXd fieldMatrix = ToMatrix((*dipole)[0]);
    const Eigen::MatrixXd coreHamiltonian = ToMatrix(*kinetic) + ToMatrix(*nuclear);

    // The converged wavefunction the response consumes.
    const auto reference = qcx::scf::RunRhfScf(*molecule, overlapMatrix, coreHamiltonian, *eri);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    ASSERT_TRUE(reference->converged);

    const std::size_t numOrbitals = static_cast<std::size_t>(reference->coefficients.rows());
    constexpr std::size_t kNumOccupied = 1; // H2/STO-3G: two electrons, one closed shell.
    const std::size_t numVirtual = numOrbitals - kNumOccupied;
    const ResponseLayout layout{kNumOccupied, numVirtual};

    // X: the occupied-virtual block of the dipole in the MO basis, flattened
    // occupied-major - the layout the operator acts on.
    const Eigen::MatrixXd dipoleMo =
        reference->coefficients.transpose() * fieldMatrix * reference->coefficients;
    std::vector<double> dipoleOv(layout.Dimension(), 0.0);

    for (std::size_t i = 0; i < kNumOccupied; ++i)
    {
        for (std::size_t a = 0; a < numVirtual; ++a)
        {
            dipoleOv[i * numVirtual + a] = Coefficient(dipoleMo, kNumOccupied + a, i);
        }
    }

    // The equation the field's first-order response satisfies is A x = -X, with
    // A the orbital Hessian this module applies.
    std::vector<double> rhs(dipoleOv.size(), 0.0);

    for (std::size_t k = 0; k < dipoleOv.size(); ++k)
    {
        rhs[k] = -dipoleOv[k];
    }

    const std::vector<double> moEri = TransformEriToMo(*eri, reference->coefficients);
    const std::span<const double> orbitalEnergies(
        reference->orbitalEnergies.data(),
        static_cast<std::size_t>(reference->orbitalEnergies.size()));
    const auto hessian = OrbitalHessianOperator::Create(
        layout, OrbitalEnergies{orbitalEnergies}, MoTwoElectronTensor{moEri});
    ASSERT_TRUE(hessian.has_value()) << hessian.error().message;

    ResponseSolver<OrbitalHessianOperator> solver;
    std::vector<double> solution(layout.Dimension(), 0.0);
    const auto solved = solver.Solve(*hessian, 1, rhs, solution);
    ASSERT_TRUE(solved.has_value()) << solved.error().message;

    // alpha_xx = -4 X^T x with A x = -X. The 4 is the closed-shell factor: the
    // energy Hessian is 4 A (the C5 relation above), the field couples to two
    // electrons through a factor 2 twice, and the two minus signs - the CPHF
    // right-hand side and d^2E/dF^2 = -alpha - cancel. Checked in the uncoupled
    // limit against second-order perturbation theory, where it reads
    // alpha = 4 X^2 / (e_a - e_i).
    double contraction = 0.0;

    for (std::size_t k = 0; k < dipoleOv.size(); ++k)
    {
        contraction += dipoleOv[k] * solution[k];
    }

    const double analytic = -4.0 * contraction;

    // The finite-field side: the same SCF at +F and -F. A central second
    // difference annihilates every term linear in the field, so the
    // field-nucleus coupling and the origin dependence of the dipole integrals
    // drop out, and what is left is alpha_xx = -d^2E/dF^2, written below as a
    // function of the field so the same expression is read at two strengths.
    const auto runAt = [&](double field) {
        return qcx::scf::RunRhfScf(
            *molecule, overlapMatrix, coreHamiltonian + field * fieldMatrix, *eri);
    };
    const auto centralSecondDifference = [&](double field) {
        const auto plus = runAt(field);
        const auto minus = runAt(-field);
        EXPECT_TRUE(plus.has_value()) << plus.error().message;
        EXPECT_TRUE(minus.has_value()) << minus.error().message;
        EXPECT_TRUE(plus->converged);
        EXPECT_TRUE(minus->converged);

        return -(plus->totalEnergy - 2.0 * reference->totalEnergy + minus->totalEnergy) /
               (field * field);
    };

    // A field of 1e-2 rather than 1e-3 on purpose. The second difference divides
    // the SCF's own convergence error by F^2, so a small field is the noisy
    // choice and a large one truncates: at F = 1e-2 the quartic term's
    // contribution to the second difference is of order 1e-3, well above the
    // 1e-4 this check asserts. The pair of strengths removes it: E(F) = E(0) -
    // (a/2) F^2 - (g/24) F^4 - ... makes the difference a + (g/12) F^2, so
    // (4 a(F) - a(2F)) / 3 cancels the quartic term exactly and leaves the next
    // order, which is smaller than the assertion by orders of magnitude.
    constexpr double kField = 1e-2;
    const double coarse = centralSecondDifference(kField);
    const double fine = centralSecondDifference(2.0 * kField);
    const double finiteField = (4.0 * coarse - fine) / 3.0;

    std::cout << "[finite field] analytic alpha_xx " << analytic << ", finite field " << coarse
              << " at F = " << kField << " and " << fine << " at 2F, extrapolated " << finiteField
              << " (" << solved->iterations << " iterations, " << solved->matvecs << " matvecs)\n";

    EXPECT_NEAR(analytic, finiteField, 1e-4);
}
