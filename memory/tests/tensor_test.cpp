#include "qcx/backend/tags.hpp"
#include "qcx/memory/tensor.hpp"

#include <concepts>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <utility>

using CpuTensor2D = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

TEST(TensorTest, CreateAndAccessRoundTrips) {
    auto t = CpuTensor2D::Create({3, 4});
    ASSERT_TRUE(t.has_value());
    (*t)(1, 2) = 5.5;
    EXPECT_DOUBLE_EQ((*t)(1, 2), 5.5);
    EXPECT_EQ(t->Size(), 12u);
    EXPECT_EQ(t->Shape()[0], 3u);
    EXPECT_EQ(t->Shape()[1], 4u);
}

TEST(TensorTest, RowMajorFlatteningIsConsistent) {
    auto t = CpuTensor2D::Create({2, 3});
    ASSERT_TRUE(t.has_value());
    int counter = 0;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            (*t)(i, j) = counter++;
        }
    }

    EXPECT_DOUBLE_EQ((*t)(1, 2), 5.0);
}

TEST(TensorTest, CreateRejectsZeroExtent) {
    auto t = CpuTensor2D::Create({0, 4});
    ASSERT_FALSE(t.has_value());
    EXPECT_EQ(t.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(TensorTest, MoveTransfersOwnership) {
    auto t = CpuTensor2D::Create({2, 3});
    ASSERT_TRUE(t.has_value());
    (*t)(1, 2) = 7.5;
    CpuTensor2D moved(std::move(*t));
    EXPECT_EQ(moved.Size(), 6u);
    EXPECT_DOUBLE_EQ(moved(1, 2), 7.5);
}

TEST(TensorTest, CloneProducesIndependentCopy) {
    auto t = CpuTensor2D::Create({2, 3});
    ASSERT_TRUE(t.has_value());
    (*t)(1, 2) = 7.5;
    auto clone = t->Clone();
    ASSERT_TRUE(clone.has_value());
    EXPECT_EQ(clone->Size(), 6u);
    EXPECT_DOUBLE_EQ((*clone)(1, 2), 7.5);
    (*clone)(1, 2) = -1.0;
    EXPECT_DOUBLE_EQ((*t)(1, 2), 7.5); // source unchanged
}

TEST(TensorTest, CloneWorksOnConstTensor) {
    auto t = CpuTensor2D::Create({2, 3});
    ASSERT_TRUE(t.has_value());
    (*t)(1, 2) = 7.5;
    const CpuTensor2D& constRef = *t;
    auto clone = constRef.Clone();
    ASSERT_TRUE(clone.has_value());
    EXPECT_EQ(clone->Size(), 6u);
    EXPECT_DOUBLE_EQ((*clone)(1, 2), 7.5);
}

TEST(TensorTest, CreateRejectsElementCountOverflow) {
    constexpr auto kMax = std::numeric_limits<std::size_t>::max();
    auto t = CpuTensor2D::Create({kMax, 2});
    ASSERT_FALSE(t.has_value());
    EXPECT_EQ(t.error().code, qcx::ErrorCode::kInvalidArgument);
}

// Compile-time check: operator() must NOT be callable for a non-CPU backend -
// this is the constraint itself, not just a runtime behavior test.
// (Requires-expression probes are unusable here: clang hard-errors and MSVC
// false-positives on constrained-out member calls; std::invocable works on
// both. The positive assertion guards against the check being vacuous.)
using CudaTensor1D = qcx::memory::Tensor<float, 1, qcx::backend::CudaTag>;
static_assert(std::invocable<CpuTensor2D&, std::size_t, std::size_t>,
              "operator() must be callable for the CPU backend");
static_assert(!std::invocable<CudaTensor1D&, std::size_t>,
              "operator() must not compile for non-CPU backends");

static_assert(CpuTensor2D::TensorRank() == 2, "TensorRank must match the Rank parameter");
