#include "qcx/error.hpp"
#include "qcx/linalg/sparse_solver.hpp"

#include <Eigen/Core>
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
    // nothing, and a gate at or below 1e-10 is not accepted.
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

// lin-1 coverage gap: amgcl reports the relative residual, and "NaN >=
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
