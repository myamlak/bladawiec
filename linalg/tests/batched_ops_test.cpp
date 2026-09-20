// Batched GEMM seam tests: GemmKernel and MultiplyBatched against
// Eigen products on random inputs, the stride-0 shared-matrix contract of
// the MD ket transform, the JIT-vs-cblas cross-check on the vendor lane,
// and every error path.

#include "qcx/linalg/batched_ops.hpp"
#include "qcx/linalg/dense_ops.hpp"

#include <Eigen/Dense>
#include <climits>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

std::vector<double> RandomMatrix(std::size_t rows, std::size_t cols, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> values(rows * cols);

    for (double& value : values)
    {
        value = dist(rng);
    }

    return values;
}

// The independent dense reference: C_i = alpha A_i B_i + beta C_i per batch
// element, honoring leading dimensions and the stride-0 shared-matrix
// contract. Deliberately Eigen-free - the engine's vendor-free lane is
// itself Eigen, so an Eigen-based reference would be self-consistency only.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): mirrors the MultiplyBatched order.
void ReferenceBatched(std::size_t batch,
                      std::size_t m,
                      std::size_t k,
                      std::size_t n,
                      double alpha,
                      const double* a,
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): see above.
                      std::size_t lda,
                      std::ptrdiff_t strideA,
                      const double* b,
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): see above.
                      std::size_t ldb,
                      std::ptrdiff_t strideB,
                      double beta,
                      double* c,
                      // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): see above.
                      std::size_t ldc,
                      std::ptrdiff_t strideC) {
    for (std::size_t i = 0; i < batch; ++i)
    {
        const double* ai = a + static_cast<std::ptrdiff_t>(i) * strideA;
        const double* bi = b + static_cast<std::ptrdiff_t>(i) * strideB;
        double* ci = c + static_cast<std::ptrdiff_t>(i) * strideC;

        for (std::size_t row = 0; row < m; ++row)
        {
            for (std::size_t col = 0; col < n; ++col)
            {
                double sum = beta * ci[row * ldc + col];

                for (std::size_t t = 0; t < k; ++t)
                {
                    sum += alpha * ai[row * lda + t] * bi[t * ldb + col];
                }

                ci[row * ldc + col] = sum;
            }
        }
    }
}

TEST(BatchedOpsTest, GemmKernelMatchesEigen) {
    std::mt19937_64 rng(20260817);
    const std::size_t shapes[][3] = {{3, 5, 4}, {17, 23, 19}, {64, 64, 64}, {80, 97, 71}};

    for (const auto& shape : shapes)
    {
        const std::size_t m = shape[0];
        const std::size_t n = shape[1];
        const std::size_t k = shape[2];
        auto kernel = qcx::linalg::GemmKernel::Create(m, n, k, 0.7, 0.3);
        ASSERT_TRUE(kernel.has_value()) << kernel.error().message;
        const std::vector<double> a = RandomMatrix(m, k, rng);
        const std::vector<double> b = RandomMatrix(k, n, rng);
        const std::vector<double> cIn = RandomMatrix(m, n, rng);
        std::vector<double> c = cIn;
        // Row-major leading dimensions: lda = k, ldb = n, ldc = n.
        auto result = kernel->Apply(a.data(), k, b.data(), n, c.data(), n);
        ASSERT_TRUE(result.has_value()) << result.error().message;

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            aMap(a.data(), static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            bMap(b.data(), static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            cInMap(cIn.data(), static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));

        const Eigen::MatrixXd expected = 0.7 * (aMap * bMap) + 0.3 * cInMap;

        for (std::size_t i = 0; i < m * n; ++i)
        {
            // c is row-major (ldc = n): element i is (i / n, i % n).
            EXPECT_NEAR(
                c[i],
                expected(static_cast<Eigen::Index>(i / n), static_cast<Eigen::Index>(i % n)),
                1e-12)
                << "shape " << m << "x" << n << "x" << k;
        }
    }
}

TEST(BatchedOpsTest, GemmKernelErrorPaths) {
    auto zeroM = qcx::linalg::GemmKernel::Create(0, 3, 2);
    EXPECT_FALSE(zeroM.has_value());
    EXPECT_EQ(zeroM.error().code, qcx::ErrorCode::kInvalidArgument);

    auto zeroN = qcx::linalg::GemmKernel::Create(4, 0, 2);
    EXPECT_FALSE(zeroN.has_value());
    EXPECT_EQ(zeroN.error().code, qcx::ErrorCode::kInvalidArgument);

    auto zeroK = qcx::linalg::GemmKernel::Create(4, 3, 0);
    EXPECT_FALSE(zeroK.has_value());
    EXPECT_EQ(zeroK.error().code, qcx::ErrorCode::kInvalidArgument);

    auto kernel = qcx::linalg::GemmKernel::Create(4, 4, 4);
    ASSERT_TRUE(kernel.has_value()) << kernel.error().message;
    std::vector<double> data(16, 1.0);
    auto badLda = kernel->Apply(data.data(), 3, data.data(), 4, data.data(), 4); // lda 3 < k 4
    EXPECT_FALSE(badLda.has_value());
    EXPECT_EQ(badLda.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badLdb = kernel->Apply(data.data(), 4, data.data(), 3, data.data(), 4); // ldb 3 < n 4
    EXPECT_FALSE(badLdb.has_value());
    EXPECT_EQ(badLdb.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badLdc = kernel->Apply(data.data(), 4, data.data(), 4, data.data(), 3); // ldc 3 < n 4
    EXPECT_FALSE(badLdc.has_value());
    EXPECT_EQ(badLdc.error().code, qcx::ErrorCode::kInvalidArgument);

    auto nullA = kernel->Apply(nullptr, 4, data.data(), 4, data.data(), 4);
    EXPECT_FALSE(nullA.has_value());
    EXPECT_EQ(nullA.error().code, qcx::ErrorCode::kInvalidArgument);

    auto nullB = kernel->Apply(data.data(), 4, nullptr, 4, data.data(), 4);
    EXPECT_FALSE(nullB.has_value());
    EXPECT_EQ(nullB.error().code, qcx::ErrorCode::kInvalidArgument);

    auto nullC = kernel->Apply(data.data(), 4, data.data(), 4, nullptr, 4);
    EXPECT_FALSE(nullC.has_value());
    EXPECT_EQ(nullC.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BatchedOpsTest, MultiplyBatchedMatchesEigen) {
    std::mt19937_64 rng(20260818);
    const std::size_t m = 13;
    const std::size_t k = 11;
    const std::size_t n = 7;
    const std::size_t batch = 6;
    const std::vector<double> a = RandomMatrix(batch * m * k, 1, rng);
    const std::vector<double> b = RandomMatrix(batch * k * n, 1, rng);
    std::vector<double> c = RandomMatrix(batch * m * n, 1, rng);
    std::vector<double> expected = c;
    ReferenceBatched(batch,
                     m,
                     k,
                     n,
                     0.5,
                     a.data(),
                     k,
                     static_cast<std::ptrdiff_t>(m * k),
                     b.data(),
                     n,
                     static_cast<std::ptrdiff_t>(k * n),
                     1.25,
                     expected.data(),
                     n,
                     static_cast<std::ptrdiff_t>(m * n));

    auto result = qcx::linalg::MultiplyBatched(batch,
                                               m,
                                               k,
                                               n,
                                               0.5,
                                               a.data(),
                                               k,
                                               static_cast<std::ptrdiff_t>(m * k),
                                               b.data(),
                                               n,
                                               static_cast<std::ptrdiff_t>(k * n),
                                               1.25,
                                               c.data(),
                                               n,
                                               static_cast<std::ptrdiff_t>(m * n));
    ASSERT_TRUE(result.has_value()) << result.error().message;

    for (std::size_t i = 0; i < c.size(); ++i)
    {
        EXPECT_NEAR(c[i], expected[i], 1e-12) << "element " << i;
    }
}

TEST(BatchedOpsTest, MultiplyBatchedSharedMatrices) {
    // The stride-0 contract: the MD ket transform shares its per-primitive-
    // pair matrix across the batch, the bra transform its T_ab^ctd.
    std::mt19937_64 rng(20260819);
    const std::size_t m = 5;
    const std::size_t k = 6;
    const std::size_t n = 4;
    const std::size_t batch = 3;
    const std::vector<double> sharedA = RandomMatrix(m * k, 1, rng);
    const std::vector<double> b = RandomMatrix(batch * k * n, 1, rng);
    std::vector<double> c(batch * m * n, 0.0);
    std::vector<double> expected = c;
    ReferenceBatched(batch,
                     m,
                     k,
                     n,
                     1.0,
                     sharedA.data(),
                     k,
                     0,
                     b.data(),
                     n,
                     static_cast<std::ptrdiff_t>(k * n),
                     0.0,
                     expected.data(),
                     n,
                     static_cast<std::ptrdiff_t>(m * n));

    auto result = qcx::linalg::MultiplyBatched(batch,
                                               m,
                                               k,
                                               n,
                                               1.0,
                                               sharedA.data(),
                                               k,
                                               0,
                                               b.data(),
                                               n,
                                               static_cast<std::ptrdiff_t>(k * n),
                                               0.0,
                                               c.data(),
                                               n,
                                               static_cast<std::ptrdiff_t>(m * n));
    ASSERT_TRUE(result.has_value()) << result.error().message;

    for (std::size_t i = 0; i < c.size(); ++i)
    {
        EXPECT_NEAR(c[i], expected[i], 1e-12) << "element " << i;
    }
}

TEST(BatchedOpsTest, MultiplyBatchedEngineShapes) {
    // The MD engine's transform shapes above the 16-wide micro kernel
    // (md_transform.hpp) - the first batch-API call sites the engine hits:
    // the class-(6,6), -(8,8), and -(12,12) ket GEMMs. These went untested
    // until the class-(6,6) benchmark exposed the gap (2026-08-18).
    std::mt19937_64 rng(20260821);
    const std::size_t shapes[][3] = {{28, 28, 15}, {45, 45, 5}, {28, 45, 15}, {91, 91, 13}};

    for (const auto& shape : shapes)
    {
        const std::size_t m = shape[0];
        const std::size_t k = shape[1];
        const std::size_t n = shape[2];

        // The ket-transform contract: shared B (stride 0), beta = 1 accumulate.
        const std::size_t batch = 3;
        const std::vector<double> a = RandomMatrix(batch * m * k, 1, rng);
        const std::vector<double> b = RandomMatrix(k * n, 1, rng);
        std::vector<double> c = RandomMatrix(batch * m * n, 1, rng);
        std::vector<double> expected = c;
        ReferenceBatched(batch,
                         m,
                         k,
                         n,
                         1.0,
                         a.data(),
                         k,
                         static_cast<std::ptrdiff_t>(m * k),
                         b.data(),
                         n,
                         0,
                         1.0,
                         expected.data(),
                         n,
                         static_cast<std::ptrdiff_t>(m * n));

        auto result = qcx::linalg::MultiplyBatched(batch,
                                                   m,
                                                   k,
                                                   n,
                                                   1.0,
                                                   a.data(),
                                                   k,
                                                   static_cast<std::ptrdiff_t>(m * k),
                                                   b.data(),
                                                   n,
                                                   0,
                                                   1.0,
                                                   c.data(),
                                                   n,
                                                   static_cast<std::ptrdiff_t>(m * n));
        ASSERT_TRUE(result.has_value()) << result.error().message;

        for (std::size_t i = 0; i < c.size(); ++i)
        {
            EXPECT_NEAR(c[i], expected[i], 1e-12)
                << "element " << i << " of shape " << m << "x" << n << "x" << k;
        }
    }
}

TEST(BatchedOpsTest, MultiplyBatchedPaddedLeadingDimensions) {
    // The regression guard for the BLAS-less lane (2026-08-22): padded
    // leading dimensions (lda > k etc.) must give the same result on every
    // lane. The Eigen fallback used plain contiguous maps that ignored the
    // leading dimensions, silently computing wrong values on the CI lane
    // while the vendor lanes were correct - invisible until a padded-ld
    // caller arrived.
    std::mt19937_64 rng(20260822);
    const std::size_t m = 4;
    const std::size_t k = 3;
    const std::size_t n = 2;
    const std::size_t batch = 2;
    const std::size_t lda = k + 3;
    const std::size_t ldb = n + 2;
    const std::size_t ldc = n + 1;
    const std::vector<double> a = RandomMatrix(batch * m * lda, 1, rng);
    const std::vector<double> b = RandomMatrix(batch * k * ldb, 1, rng);
    std::vector<double> c = RandomMatrix(batch * m * ldc, 1, rng);
    std::vector<double> expected = c;
    ReferenceBatched(batch,
                     m,
                     k,
                     n,
                     0.6,
                     a.data(),
                     lda,
                     static_cast<std::ptrdiff_t>(m * lda),
                     b.data(),
                     ldb,
                     static_cast<std::ptrdiff_t>(k * ldb),
                     1.1,
                     expected.data(),
                     ldc,
                     static_cast<std::ptrdiff_t>(m * ldc));

    auto result = qcx::linalg::MultiplyBatched(batch,
                                               m,
                                               k,
                                               n,
                                               0.6,
                                               a.data(),
                                               lda,
                                               static_cast<std::ptrdiff_t>(m * lda),
                                               b.data(),
                                               ldb,
                                               static_cast<std::ptrdiff_t>(k * ldb),
                                               1.1,
                                               c.data(),
                                               ldc,
                                               static_cast<std::ptrdiff_t>(m * ldc));
    ASSERT_TRUE(result.has_value()) << result.error().message;

    for (std::size_t i = 0; i < c.size(); ++i)
    {
        EXPECT_NEAR(c[i], expected[i], 1e-12) << "element " << i;
    }
}

TEST(BatchedOpsTest, MultiplyBatchedEdgeSizes) {
    // batch = 1, a unit dimension, and the stride-0 shared-C contract - the
    // boundaries the error paths do not cover.
    std::mt19937_64 rng(20260822);

    {
        // batch = 1 with a unit row dimension (m = 1).
        const std::size_t m = 1;
        const std::size_t k = 2;
        const std::size_t n = 3;
        const std::size_t batch = 1;
        const std::vector<double> a = RandomMatrix(m * k, 1, rng);
        const std::vector<double> b = RandomMatrix(k * n, 1, rng);
        std::vector<double> c = RandomMatrix(m * n, 1, rng);
        std::vector<double> expected = c;
        ReferenceBatched(batch,
                         m,
                         k,
                         n,
                         0.9,
                         a.data(),
                         k,
                         static_cast<std::ptrdiff_t>(m * k),
                         b.data(),
                         n,
                         static_cast<std::ptrdiff_t>(k * n),
                         0.4,
                         expected.data(),
                         n,
                         static_cast<std::ptrdiff_t>(m * n));

        auto result = qcx::linalg::MultiplyBatched(batch,
                                                   m,
                                                   k,
                                                   n,
                                                   0.9,
                                                   a.data(),
                                                   k,
                                                   static_cast<std::ptrdiff_t>(m * k),
                                                   b.data(),
                                                   n,
                                                   static_cast<std::ptrdiff_t>(k * n),
                                                   0.4,
                                                   c.data(),
                                                   n,
                                                   static_cast<std::ptrdiff_t>(m * n));
        ASSERT_TRUE(result.has_value()) << result.error().message;

        for (std::size_t i = 0; i < c.size(); ++i)
        {
            EXPECT_NEAR(c[i], expected[i], 1e-12) << "element " << i;
        }
    }

    {
        // Shared C (strideC = 0): every element writes into the same
        // buffer, so the last element's result wins (the documented
        // contract, element i does read-modify-write on the shared C).
        const std::size_t m = 2;
        const std::size_t k = 2;
        const std::size_t n = 2;
        const std::size_t batch = 2;
        const std::vector<double> a = RandomMatrix(batch * m * k, 1, rng);
        const std::vector<double> b = RandomMatrix(batch * k * n, 1, rng);
        std::vector<double> c = RandomMatrix(m * n, 1, rng);
        std::vector<double> expected = c;
        ReferenceBatched(batch,
                         m,
                         k,
                         n,
                         1.0,
                         a.data(),
                         k,
                         static_cast<std::ptrdiff_t>(m * k),
                         b.data(),
                         n,
                         static_cast<std::ptrdiff_t>(k * n),
                         1.0,
                         expected.data(),
                         n,
                         0);

        auto result = qcx::linalg::MultiplyBatched(batch,
                                                   m,
                                                   k,
                                                   n,
                                                   1.0,
                                                   a.data(),
                                                   k,
                                                   static_cast<std::ptrdiff_t>(m * k),
                                                   b.data(),
                                                   n,
                                                   static_cast<std::ptrdiff_t>(k * n),
                                                   1.0,
                                                   c.data(),
                                                   n,
                                                   0);
        ASSERT_TRUE(result.has_value()) << result.error().message;

        for (std::size_t i = 0; i < c.size(); ++i)
        {
            EXPECT_NEAR(c[i], expected[i], 1e-12) << "element " << i;
        }
    }
}

TEST(BatchedOpsTest, MultiplyBatchedErrorPaths) {
    std::vector<double> data(8, 1.0);

    auto zeroBatch = qcx::linalg::MultiplyBatched(
        0, 2, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(zeroBatch.has_value());
    EXPECT_EQ(zeroBatch.error().code, qcx::ErrorCode::kInvalidArgument);

    auto zeroM = qcx::linalg::MultiplyBatched(
        2, 0, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(zeroM.has_value());
    EXPECT_EQ(zeroM.error().code, qcx::ErrorCode::kInvalidArgument);

    auto zeroK = qcx::linalg::MultiplyBatched(
        2, 2, 0, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(zeroK.has_value());
    EXPECT_EQ(zeroK.error().code, qcx::ErrorCode::kInvalidArgument);

    auto zeroN = qcx::linalg::MultiplyBatched(
        2, 2, 2, 0, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(zeroN.has_value());
    EXPECT_EQ(zeroN.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badLda = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 1, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(badLda.has_value());
    EXPECT_EQ(badLda.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badLdb = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 2, 8, data.data(), 1, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(badLdb.has_value());
    EXPECT_EQ(badLdb.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badLdc = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 1, 8);
    EXPECT_FALSE(badLdc.has_value());
    EXPECT_EQ(badLdc.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badStrideA = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 2, -1, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(badStrideA.has_value());
    EXPECT_EQ(badStrideA.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badStrideB = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, -1, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(badStrideB.has_value());
    EXPECT_EQ(badStrideB.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badStrideC = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, -1);
    EXPECT_FALSE(badStrideC.has_value());
    EXPECT_EQ(badStrideC.error().code, qcx::ErrorCode::kInvalidArgument);

    auto nullA = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, nullptr, 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(nullA.has_value());
    EXPECT_EQ(nullA.error().code, qcx::ErrorCode::kInvalidArgument);

    auto nullB = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 2, 8, nullptr, 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(nullB.has_value());
    EXPECT_EQ(nullB.error().code, qcx::ErrorCode::kInvalidArgument);

    auto nullC = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, nullptr, 2, 8);
    EXPECT_FALSE(nullC.has_value());
    EXPECT_EQ(nullC.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(BatchedOpsTest, DimensionsAboveIntRangeAreRejected) {
    // Every vendor lane passes batch and the dimensions to 32-bit
    // BLAS parameters (MKL_INT under this build's LP64 interface, int
    // elsewhere); a size_t value above INT_MAX would truncate silently.
    // The guard rejects it up front, before any pointer arithmetic, so the
    // huge dimension never reaches a vendor call or an allocation.
    std::vector<double> data(8, 1.0);
    const std::size_t huge = static_cast<std::size_t>(INT_MAX) + 1;

    auto kernel = qcx::linalg::GemmKernel::Create(huge, 2, 2);
    EXPECT_FALSE(kernel.has_value());
    EXPECT_EQ(kernel.error().code, qcx::ErrorCode::kInvalidArgument);

    auto tooMany = qcx::linalg::MultiplyBatched(
        huge, 2, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(tooMany.has_value());
    EXPECT_EQ(tooMany.error().code, qcx::ErrorCode::kInvalidArgument);

    auto tooWide = qcx::linalg::MultiplyBatched(
        2, huge, 2, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(tooWide.has_value());
    EXPECT_EQ(tooWide.error().code, qcx::ErrorCode::kInvalidArgument);

    auto tooDeep = qcx::linalg::MultiplyBatched(
        2, 2, huge, 2, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(tooDeep.has_value());
    EXPECT_EQ(tooDeep.error().code, qcx::ErrorCode::kInvalidArgument);

    auto tooLong = qcx::linalg::MultiplyBatched(
        2, 2, 2, huge, 1.0, data.data(), 2, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(tooLong.has_value());
    EXPECT_EQ(tooLong.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badLd = qcx::linalg::MultiplyBatched(
        2, 2, 2, 2, 1.0, data.data(), huge, 8, data.data(), 2, 8, 0.0, data.data(), 2, 8);
    EXPECT_FALSE(badLd.has_value());
    EXPECT_EQ(badLd.error().code, qcx::ErrorCode::kInvalidArgument);
}

#ifdef QcxHasVendorBlas

TEST(BatchedOpsTest, JitMatchesCblas) {
    // The MKL lane: the JIT-compiled fixed-shape kernel and the cblas path
    // must agree (the JIT-vs-cblas cross-check).
    std::mt19937_64 rng(20260820);
    const std::size_t m = 43;
    const std::size_t n = 37;
    const std::size_t k = 29;
    auto kernel = qcx::linalg::GemmKernel::Create(m, n, k);
    ASSERT_TRUE(kernel.has_value()) << kernel.error().message;
    const std::vector<double> a = RandomMatrix(m, k, rng);
    const std::vector<double> b = RandomMatrix(k, n, rng);
    std::vector<double> c(m * n, 0.0);
    auto result = kernel->Apply(a.data(), k, b.data(), n, c.data(), n);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> aMap(
        a.data(), static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> bMap(
        b.data(), static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
    auto expected = qcx::linalg::BlasMultiply(aMap, bMap);
    ASSERT_TRUE(expected.has_value()) << expected.error().message;

    for (std::size_t i = 0; i < m * n; ++i)
    {
        // c is row-major (ldc = n): element i is (i / n, i % n).
        EXPECT_NEAR(c[i],
                    (*expected)(static_cast<Eigen::Index>(i / n), static_cast<Eigen::Index>(i % n)),
                    1e-12)
            << "element " << i;
    }
}

#endif

} // namespace
