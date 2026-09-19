#include "qcx/error.hpp"
#include "qcx/linalg/iterative_eigensolver.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

constexpr std::size_t kChainLength = 20;
constexpr std::size_t kNumEigenpairs = 3;

/// 1D chain Laplacian (Dirichlet ends): eigenvalues 2 - 2 cos(k pi / (n + 1)).
Eigen::MatrixXd MakeChainLaplacian(std::size_t n) {
    Eigen::MatrixXd laplacian = Eigen::MatrixXd::Zero(static_cast<int>(n), static_cast<int>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        laplacian(static_cast<int>(i), static_cast<int>(i)) = 2.0;

        if (i > 0)
        {
            laplacian(static_cast<int>(i), static_cast<int>(i - 1)) = -1.0;
            laplacian(static_cast<int>(i - 1), static_cast<int>(i)) = -1.0;
        }
    }

    return laplacian;
}

} // namespace

TEST(IterativeEigenSolverTest, LowestEigenpairsMatchDenseReference) {
    const auto laplacian = MakeChainLaplacian(kChainLength);
    const auto result = qcx::linalg::ComputeLowestEigenpairs(laplacian, kNumEigenpairs);
    ASSERT_TRUE(result.has_value());
    const auto& pairs = *result;

    ASSERT_EQ(pairs.size(), kNumEigenpairs);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> reference(laplacian);

    for (std::size_t i = 0; i < pairs.size(); ++i)
    {
        EXPECT_NEAR(pairs[i].eigenvalue, reference.eigenvalues()[static_cast<int>(i)], 1e-9);
        EXPECT_NEAR(
            std::abs(pairs[i].eigenvector.dot(reference.eigenvectors().col(static_cast<int>(i)))),
            1.0,
            1e-9);
    }
}

TEST(IterativeEigenSolverTest, RejectsNonSquareMatrix) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(2, 3);
    const auto result = qcx::linalg::ComputeLowestEigenpairs(matrix, 1);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(IterativeEigenSolverTest, RejectsNonSymmetricMatrix) {
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(3, 3);
    matrix(0, 1) = 1.0;
    const auto result = qcx::linalg::ComputeLowestEigenpairs(matrix, 1);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(IterativeEigenSolverTest, RejectsInvalidEigenpairCounts) {
    const auto matrix = Eigen::MatrixXd::Identity(4, 4);
    const auto zeroResult = qcx::linalg::ComputeLowestEigenpairs(matrix, 0);
    ASSERT_FALSE(zeroResult.has_value());
    EXPECT_EQ(zeroResult.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto oversizedResult = qcx::linalg::ComputeLowestEigenpairs(matrix, 4);
    ASSERT_FALSE(oversizedResult.has_value());
    EXPECT_EQ(oversizedResult.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(IterativeEigenSolverTest, RejectsInvalidBudgets) {
    const auto matrix = Eigen::MatrixXd::Identity(4, 4);
    const auto iterationsResult = qcx::linalg::ComputeLowestEigenpairs(matrix, 1, 0, 0, 1e-10);
    ASSERT_FALSE(iterationsResult.has_value());
    EXPECT_EQ(iterationsResult.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto toleranceResult = qcx::linalg::ComputeLowestEigenpairs(matrix, 1, 0, 100, 0.0);
    ASSERT_FALSE(toleranceResult.has_value());
    EXPECT_EQ(toleranceResult.error().code, qcx::ErrorCode::kInvalidArgument);
}
