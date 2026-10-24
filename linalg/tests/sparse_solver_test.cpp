#include "qcx/error.hpp"
#include "qcx/linalg/sparse_solver.hpp"

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

constexpr std::size_t kGridSize = 10;
constexpr std::size_t kUnknownCount = kGridSize * kGridSize;

/// 2D Poisson problem on a kGridSize x kGridSize grid with Dirichlet
/// boundaries: diagonal 4, -1 to the four grid neighbors.
struct PoissonSystem {
    std::vector<std::size_t> rowOffsets;
    std::vector<std::size_t> columnIndices;
    std::vector<double> values;
    std::vector<double> rhs;
};

PoissonSystem MakePoissonSystem() {
    PoissonSystem system;
    system.rowOffsets.push_back(0);

    for (std::size_t i = 0; i < kGridSize; ++i)
    {
        for (std::size_t j = 0; j < kGridSize; ++j)
        {
            const auto node = i * kGridSize + j;
            system.columnIndices.push_back(node);
            system.values.push_back(4.0);

            if (i > 0)
            {
                system.columnIndices.push_back((i - 1) * kGridSize + j);
                system.values.push_back(-1.0);
            }

            if (i + 1 < kGridSize)
            {
                system.columnIndices.push_back((i + 1) * kGridSize + j);
                system.values.push_back(-1.0);
            }

            if (j > 0)
            {
                system.columnIndices.push_back(i * kGridSize + j - 1);
                system.values.push_back(-1.0);
            }

            if (j + 1 < kGridSize)
            {
                system.columnIndices.push_back(i * kGridSize + j + 1);
                system.values.push_back(-1.0);
            }

            system.rowOffsets.push_back(system.values.size());
        }
    }

    // Manufactured solution x_j = j + 1; rhs = A x.
    Eigen::MatrixXd dense =
        Eigen::MatrixXd::Zero(static_cast<int>(kUnknownCount), static_cast<int>(kUnknownCount));

    for (std::size_t row = 0; row < kUnknownCount; ++row)
    {
        for (std::size_t entry = system.rowOffsets[row]; entry < system.rowOffsets[row + 1];
             ++entry)
        {
            dense(static_cast<int>(row), static_cast<int>(system.columnIndices[entry])) =
                system.values[entry];
        }
    }

    Eigen::VectorXd exact(static_cast<int>(kUnknownCount));

    for (std::size_t i = 0; i < kGridSize; ++i)
    {
        for (std::size_t j = 0; j < kGridSize; ++j)
        {
            exact(static_cast<int>(i * kGridSize + j)) = static_cast<double>(j + 1);
        }
    }

    const Eigen::VectorXd denseRhs = dense * exact;
    system.rhs.assign(denseRhs.data(), denseRhs.data() + denseRhs.size());

    return system;
}

} // namespace

TEST(SparseSolverTest, SolvesPoissonAgainstDenseReference) {
    const PoissonSystem system = MakePoissonSystem();
    // 1e-8 rather than the 1e-12 this call used to pass: the solution is
    // asserted to 1e-7 below, so the tighter residual bought the comparison
    // nothing, and no convergence gate at or below 1e-10 is offered.
    const auto result = qcx::linalg::SolveSparseSystem(
        system.rowOffsets, system.columnIndices, system.values, system.rhs, 1e-8);
    ASSERT_TRUE(result.has_value());
    const auto& solution = *result;

    ASSERT_EQ(solution.size(), kUnknownCount);

    for (std::size_t i = 0; i < kGridSize; ++i)
    {
        for (std::size_t j = 0; j < kGridSize; ++j)
        {
            EXPECT_NEAR(solution[i * kGridSize + j], static_cast<double>(j + 1), 1e-7);
        }
    }
}

TEST(SparseSolverTest, RejectsShapeMismatches) {
    const std::vector<double> one = {1.0};
    const std::vector<double> rhs = {1.0};

    const auto offsetSizeResult = qcx::linalg::SolveSparseSystem({0}, {0}, one, rhs, 1e-10);
    ASSERT_FALSE(offsetSizeResult.has_value());
    EXPECT_EQ(offsetSizeResult.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto columnSizeResult = qcx::linalg::SolveSparseSystem({0, 1}, {0, 1}, one, rhs, 1e-10);
    ASSERT_FALSE(columnSizeResult.has_value());
    EXPECT_EQ(columnSizeResult.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto emptyResult = qcx::linalg::SolveSparseSystem({}, {}, {}, {}, 1e-10);
    ASSERT_FALSE(emptyResult.has_value());
    EXPECT_EQ(emptyResult.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto toleranceResult = qcx::linalg::SolveSparseSystem({0, 1}, {0}, one, rhs, 0.0);
    ASSERT_FALSE(toleranceResult.has_value());
    EXPECT_EQ(toleranceResult.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto columnRangeResult = qcx::linalg::SolveSparseSystem({0, 1}, {5}, one, rhs, 1e-10);
    ASSERT_FALSE(columnRangeResult.has_value());
    EXPECT_EQ(columnRangeResult.error().code, qcx::ErrorCode::kInvalidArgument);
}

// Coverage gap: amgcl reports the relative residual, and "NaN >=
// tolerance" is false, so a NaN residual (NaN matrix entry or RHS, or a CG
// breakdown) used to sail through as a successful solve with a garbage
// solution. The n=1 system is the coarse direct-solve path: x = rhs / NaN =
// NaN and the relative residual is NaN, which must be reported as
// kConvergenceFailure, not success.
TEST(SparseSolverTest, RejectsNonFiniteResiduals) {
    const std::vector<double> nanValue = {std::numeric_limits<double>::quiet_NaN()};
    const std::vector<double> one = {1.0};

    const auto nanMatrixResult = qcx::linalg::SolveSparseSystem({0, 1}, {0}, nanValue, one, 1e-6);
    ASSERT_FALSE(nanMatrixResult.has_value());
    EXPECT_EQ(nanMatrixResult.error().code, qcx::ErrorCode::kConvergenceFailure);

    const auto nanRhsResult = qcx::linalg::SolveSparseSystem(
        {0, 1}, {0}, one, {std::numeric_limits<double>::quiet_NaN()}, 1e-6);
    ASSERT_FALSE(nanRhsResult.has_value());
    EXPECT_EQ(nanRhsResult.error().code, qcx::ErrorCode::kConvergenceFailure);
}

namespace {

/// One manufactured solution per right-hand side, distinct enough that a
/// solution block returned in the wrong place would not match.
std::vector<double> MakeExactSides() {
    std::vector<double> exact(kUnknownCount * 3);

    for (std::size_t i = 0; i < kGridSize; ++i)
    {
        for (std::size_t j = 0; j < kGridSize; ++j)
        {
            const std::size_t node = i * kGridSize + j;
            const double index = static_cast<double>(j + 1);
            exact[node] = index;
            exact[kUnknownCount + node] = std::cos(0.1 * static_cast<double>(node));
            exact[2 * kUnknownCount + node] =
                (node % 2 == 0 ? 1.0 : -1.0) * static_cast<double>(node + 1);
        }
    }

    return exact;
}

/// The relative residual of one side, recomputed from the CSR triple and that
/// side's own right-hand side rather than read back from the solver's report.
double CsrRelativeResidual(const PoissonSystem& system,
                           const std::vector<double>& rhs,
                           const std::vector<double>& solution,
                           std::size_t side) {
    double residualSquared = 0.0;
    double rhsSquared = 0.0;

    for (std::size_t row = 0; row < kUnknownCount; ++row)
    {
        double product = 0.0;

        for (std::size_t entry = system.rowOffsets[row]; entry < system.rowOffsets[row + 1];
             ++entry)
        {
            product +=
                system.values[entry] * solution[side * kUnknownCount + system.columnIndices[entry]];
        }

        const double expected = rhs[side * kUnknownCount + row];
        const double residual = expected - product;
        residualSquared += residual * residual;
        rhsSquared += expected * expected;
    }

    return std::sqrt(residualSquared / rhsSquared);
}

/// The right-hand sides of the Poisson system for the manufactured solutions.
std::vector<double> MakeMultipleRhs(const PoissonSystem& system) {
    std::vector<double> exact = MakeExactSides();
    std::vector<double> rhs(3 * kUnknownCount);

    for (std::size_t side = 0; side < 3; ++side)
    {
        for (std::size_t row = 0; row < kUnknownCount; ++row)
        {
            double product = 0.0;

            for (std::size_t entry = system.rowOffsets[row]; entry < system.rowOffsets[row + 1];
                 ++entry)
            {
                product += system.values[entry] *
                           exact[side * kUnknownCount + system.columnIndices[entry]];
            }

            rhs[side * kUnknownCount + row] = product;
        }
    }

    return rhs;
}

/// The same operator as a dense Eigen matrix, for the reference solve.
Eigen::MatrixXd DensePoissonMatrix(const PoissonSystem& system) {
    Eigen::MatrixXd dense =
        Eigen::MatrixXd::Zero(static_cast<int>(kUnknownCount), static_cast<int>(kUnknownCount));

    for (std::size_t row = 0; row < kUnknownCount; ++row)
    {
        for (std::size_t entry = system.rowOffsets[row]; entry < system.rowOffsets[row + 1];
             ++entry)
        {
            dense(static_cast<int>(row), static_cast<int>(system.columnIndices[entry])) =
                system.values[entry];
        }
    }

    return dense;
}

} // namespace

// One hierarchy, a loop over the right-hand sides: the solve is checked
// against the manufactured solutions it was built from, against a dense Eigen
// factorisation of the same operator, and against the true residual
// recomputed from the CSR triple, because AMGCL's own reported residual is the
// recurrence's, not the operator's.
TEST(SparseSolverTest, MultipleRightHandSidesMatchTheirManufacturedSolutions) {
    const PoissonSystem system = MakePoissonSystem();
    const std::vector<double> exact = MakeExactSides();
    const std::vector<double> rhs = MakeMultipleRhs(system);

    // The dense reference: the same operator solved directly.
    const Eigen::MatrixXd dense = DensePoissonMatrix(system);
    const Eigen::LDLT<Eigen::MatrixXd> reference(dense);
    ASSERT_EQ(reference.info(), Eigen::Success);

    // 1e-9 is the requested residual; the true residual is asserted at 1e-8
    // below, so the request keeps a decade of margin.
    const auto result =
        qcx::linalg::SolveSparseSystemMultipleRhs(system.rowOffsets,
                                                  system.columnIndices,
                                                  system.values,
                                                  qcx::linalg::RightHandSides{rhs, 3},
                                                  1e-9);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    ASSERT_EQ(result->solution.size(), 3 * kUnknownCount);
    EXPECT_GT(result->iterations, 0u);
    EXPECT_LE(result->worstResidual, 1e-9);

    for (std::size_t side = 0; side < 3; ++side)
    {
        Eigen::VectorXd denseRhs(static_cast<int>(kUnknownCount));

        for (std::size_t node = 0; node < kUnknownCount; ++node)
        {
            denseRhs(static_cast<int>(node)) = rhs[side * kUnknownCount + node];
        }

        const Eigen::VectorXd referenceSolution = reference.solve(denseRhs);

        for (std::size_t node = 0; node < kUnknownCount; ++node)
        {
            const double expected = exact[side * kUnknownCount + node];
            const double referenceValue = referenceSolution(static_cast<int>(node));

            // The manufactured solutions span magnitudes up to 100; a 1e-9
            // residual maps to roughly 1e-7 absolute error there, so the gate
            // is 1e-7 RELATIVE with an absolute floor of 1e-7.
            EXPECT_NEAR(result->solution[side * kUnknownCount + node],
                        expected,
                        1e-7 * std::max(1.0, std::fabs(expected)));
            EXPECT_NEAR(result->solution[side * kUnknownCount + node],
                        referenceValue,
                        1e-7 * std::max(1.0, std::fabs(referenceValue)))
                << "right-hand side " << side << " node " << node;
        }

        EXPECT_LT(CsrRelativeResidual(system, rhs, result->solution, side), 1e-8)
            << "right-hand side " << side;
    }
}

// The single-right-hand-side entry delegates to the multi entry, so the two
// must return the same bits - not merely both satisfy the tolerance.
TEST(SparseSolverTest, TheSingleRhsEntryMatchesTheMultiRhsEntry) {
    const PoissonSystem system = MakePoissonSystem();
    const auto single = qcx::linalg::SolveSparseSystem(
        system.rowOffsets, system.columnIndices, system.values, system.rhs, 1e-9);
    const auto multiple =
        qcx::linalg::SolveSparseSystemMultipleRhs(system.rowOffsets,
                                                  system.columnIndices,
                                                  system.values,
                                                  qcx::linalg::RightHandSides{system.rhs, 1},
                                                  1e-9);

    ASSERT_TRUE(single.has_value()) << single.error().message;
    ASSERT_TRUE(multiple.has_value()) << multiple.error().message;
    ASSERT_EQ(single->size(), multiple->solution.size());

    for (std::size_t i = 0; i < single->size(); ++i)
    {
        EXPECT_DOUBLE_EQ((*single)[i], multiple->solution[i]);
    }
}

TEST(SparseSolverTest, MultipleRightHandSidesRejectShapeMismatches) {
    const PoissonSystem system = MakePoissonSystem();
    const std::vector<double> rhs = MakeMultipleRhs(system);

    const auto noSides =
        qcx::linalg::SolveSparseSystemMultipleRhs(system.rowOffsets,
                                                  system.columnIndices,
                                                  system.values,
                                                  qcx::linalg::RightHandSides{rhs, 0},
                                                  1e-8);
    ASSERT_FALSE(noSides.has_value());
    EXPECT_EQ(noSides.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto shortRhs =
        qcx::linalg::SolveSparseSystemMultipleRhs(system.rowOffsets,
                                                  system.columnIndices,
                                                  system.values,
                                                  qcx::linalg::RightHandSides{rhs, 4},
                                                  1e-8);
    ASSERT_FALSE(shortRhs.has_value());
    EXPECT_EQ(shortRhs.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto badTolerance =
        qcx::linalg::SolveSparseSystemMultipleRhs(system.rowOffsets,
                                                  system.columnIndices,
                                                  system.values,
                                                  qcx::linalg::RightHandSides{rhs, 3},
                                                  0.0);
    ASSERT_FALSE(badTolerance.has_value());
    EXPECT_EQ(badTolerance.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto emptyOffsets = qcx::linalg::SolveSparseSystemMultipleRhs(
        {}, {}, {}, qcx::linalg::RightHandSides{rhs, 3}, 1e-8);
    ASSERT_FALSE(emptyOffsets.has_value());
    EXPECT_EQ(emptyOffsets.error().code, qcx::ErrorCode::kInvalidArgument);
}
