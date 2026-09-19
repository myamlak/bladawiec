#include "qcx/linalg/dense_ops.hpp"

#include <Eigen/Dense>
#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

TEST(DenseOpsTest, SmallBlockUsesEigenPath) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(3, 3);
    auto result = qcx::linalg::DenseMultiply(a, b);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->isApprox(b));
}

#ifdef QcxHasVendorBlas
TEST(DenseOpsTest, LargeBlockMatchesEigenReference) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(100, 100);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(100, 100);
    auto viaBlas = qcx::linalg::BlasMultiply(a, b);
    Eigen::MatrixXd viaEigen = a * b;
    ASSERT_TRUE(viaBlas.has_value());
    EXPECT_TRUE(viaBlas->isApprox(viaEigen, 1e-9));
}

TEST(DenseOpsTest, LargeBlockDispatchesToBlas) {
    Eigen::MatrixXd a = Eigen::MatrixXd::Random(80, 80);
    Eigen::MatrixXd b = Eigen::MatrixXd::Random(80, 80);
    auto result = qcx::linalg::DenseMultiply(a, b);
    Eigen::MatrixXd expected = a * b;
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->isApprox(expected, 1e-9));
}

TEST(DenseOpsTest, BlasMultiplyRejectsDimensionMismatch) {
    Eigen::MatrixXd a(2, 3);
    Eigen::MatrixXd b(4, 2);
    auto result = qcx::linalg::BlasMultiply(a, b);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}
#endif

TEST(DenseOpsTest, RotationMatrixIsOrthogonal) {
    Eigen::Matrix3d rotation = Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    EXPECT_NEAR(rotation.determinant(), 1.0, 1e-12);
    EXPECT_TRUE((rotation * rotation.transpose()).isApprox(Eigen::Matrix3d::Identity()));
}

TEST(DenseOpsTest, DimensionMismatchReturnsError) {
    Eigen::MatrixXd a(2, 3);
    Eigen::MatrixXd b(4, 2);
    auto result = qcx::linalg::DenseMultiply(a, b);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

#if defined(QcxHasVendorBlas) && !defined(QcxVendorBlasIsMkl)
TEST(DenseOpsTest, FallbackVendorBlasThreadingPinIsApplied) {
    // OpenBLAS and AOCL(-BLIS) thread by default, unlike the sequential MKL
    // link. The qcx-linalg-tests discovery in linalg/CMakeLists.txt delivers
    // the one-thread pin (OPENBLAS_NUM_THREADS=1 / BLIS_NUM_THREADS=1) to
    // every ctest-launched process on the fallback vendors - assert it
    // reached this process (fails when run outside ctest without the pin
    // exported).
    const char* pin = std::getenv("OPENBLAS_NUM_THREADS");

    if (pin == nullptr)
    {
        pin = std::getenv("BLIS_NUM_THREADS");
    }

    ASSERT_NE(pin, nullptr) << "non-MKL vendor build without the one-thread pin: export "
                               "OPENBLAS_NUM_THREADS=1 (or BLIS_NUM_THREADS=1 for AOCL) at launch";
    EXPECT_EQ(std::string(pin), "1");
}
#endif
