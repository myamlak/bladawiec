#include "qcx/backend/execution_context.hpp"

#include <gtest/gtest.h>

// This file runs unchanged against both build modes: the serial stub of a
// build without MPI support and the real MPI context of a build with it. ctest
// launches the binary as one process, so the world is rank 0 of 1 there in
// either mode; every assertion is written against the reported rank and rank
// count, so the same source also checks a multi-rank launch under mpiexec.
namespace {

qcx::Result<std::shared_ptr<qcx::backend::ExecutionContext>> CreateContext() {
    return qcx::backend::ExecutionContext::Create();
}

} // namespace

TEST(ExecutionContextTest, ReportsThisProcessRankAndSize) {
    const auto context = CreateContext();
    ASSERT_TRUE(context.has_value()) << context.error().message;

    EXPECT_GE((*context)->Size(), 1);
    EXPECT_GE((*context)->Rank(), 0);
    EXPECT_LT((*context)->Rank(), (*context)->Size());
}

// Create() is the process's only entry point: every call returns the one
// context, so a second caller never initializes MPI a second time.
TEST(ExecutionContextTest, ReturnsTheSameContextOnEveryCall) {
    const auto first = CreateContext();
    ASSERT_TRUE(first.has_value()) << first.error().message;

    const auto second = CreateContext();
    ASSERT_TRUE(second.has_value()) << second.error().message;

    EXPECT_EQ(first->get(), second->get());
}

TEST(ExecutionContextTest, BarrierSucceeds) {
    const auto context = CreateContext();
    ASSERT_TRUE(context.has_value()) << context.error().message;

    EXPECT_TRUE((*context)->Barrier().has_value());
}

TEST(ExecutionContextTest, IntegerAllReduceSumsOverTheRankCount) {
    const auto context = CreateContext();
    ASSERT_TRUE(context.has_value()) << context.error().message;

    const int size = (*context)->Size();
    const int local = (*context)->Rank() + 1;

    const auto sum = (*context)->AllReduceSum(local);
    ASSERT_TRUE(sum.has_value()) << sum.error().message;
    EXPECT_EQ(*sum, size * (size + 1) / 2);
}

TEST(ExecutionContextTest, DoubleAllReduceSumsOverTheRankCount) {
    const auto context = CreateContext();
    ASSERT_TRUE(context.has_value()) << context.error().message;

    const int size = (*context)->Size();
    const double local = static_cast<double>((*context)->Rank() + 1);

    const auto sum = (*context)->AllReduceSum(local);
    ASSERT_TRUE(sum.has_value()) << sum.error().message;
    EXPECT_DOUBLE_EQ(*sum, static_cast<double>(size * (size + 1) / 2));
}

TEST(ExecutionContextTest, AllReduceSumsNegativeValues) {
    const auto context = CreateContext();
    ASSERT_TRUE(context.has_value()) << context.error().message;

    const int size = (*context)->Size();
    const int local = -((*context)->Rank() + 1);

    const auto sum = (*context)->AllReduceSum(local);
    ASSERT_TRUE(sum.has_value()) << sum.error().message;
    EXPECT_EQ(*sum, -(size * (size + 1) / 2));
}

TEST(ExecutionContextTest, AllReduceOfZeroIsZero) {
    const auto context = CreateContext();
    ASSERT_TRUE(context.has_value()) << context.error().message;

    const auto integerSum = (*context)->AllReduceSum(0);
    ASSERT_TRUE(integerSum.has_value()) << integerSum.error().message;
    EXPECT_EQ(*integerSum, 0);

    const auto doubleSum = (*context)->AllReduceSum(0.0);
    ASSERT_TRUE(doubleSum.has_value()) << doubleSum.error().message;
    EXPECT_DOUBLE_EQ(*doubleSum, 0.0);
}
