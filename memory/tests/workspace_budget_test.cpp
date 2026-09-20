#include "qcx/memory/workspace_budget.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

using qcx::memory::WorkspaceBudget;

TEST(WorkspaceBudgetTest, CreateRejectsZeroCapacity) {
    auto budget = WorkspaceBudget::Create(0);
    ASSERT_FALSE(budget.has_value());
    EXPECT_EQ(budget.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(WorkspaceBudgetTest, FreshBudgetReportsFullCapacity) {
    auto budget = WorkspaceBudget::Create(1024);
    ASSERT_TRUE(budget.has_value());
    EXPECT_EQ(budget->CapacityBytes(), 1024u);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 1024u);
}

TEST(WorkspaceBudgetTest, ReserveChargesBytesInFull) {
    auto budget = WorkspaceBudget::Create(1024);
    ASSERT_TRUE(budget.has_value());
    EXPECT_TRUE(budget->Reserve(256));
    EXPECT_EQ(budget->CommittedBytes(), 256u);
    EXPECT_EQ(budget->Remaining(), 768u);
    EXPECT_TRUE(budget->Reserve(512));
    EXPECT_EQ(budget->CommittedBytes(), 768u);
    EXPECT_EQ(budget->Remaining(), 256u);
}

TEST(WorkspaceBudgetTest, ReleaseDoesNotRestoreCapacity) {
    auto budget = WorkspaceBudget::Create(200);
    ASSERT_TRUE(budget.has_value());
    ASSERT_TRUE(budget->Reserve(100));
    budget->Release(100);
    // Cumulative high-water: the release is diagnostics only.
    EXPECT_EQ(budget->CommittedBytes(), 100u);
    EXPECT_EQ(budget->Remaining(), 100u);
    // The released 100 is still gone: this charge consumes the last headroom.
    EXPECT_TRUE(budget->Reserve(100));
    EXPECT_EQ(budget->CommittedBytes(), 200u);
    EXPECT_EQ(budget->Remaining(), 0u);
    EXPECT_FALSE(budget->Reserve(1));
}

TEST(WorkspaceBudgetTest, ReleaseLeavesCounterUntouchedWhenOversized) {
    auto budget = WorkspaceBudget::Create(64);
    ASSERT_TRUE(budget.has_value());
    ASSERT_TRUE(budget->Reserve(32));
    // Releasing more than was reserved cannot underflow the counter: the
    // cumulative commit is monotone and Release never moves it.
    budget->Release(std::numeric_limits<std::size_t>::max());
    EXPECT_EQ(budget->CommittedBytes(), 32u);
    EXPECT_EQ(budget->Remaining(), 32u);
}

TEST(WorkspaceBudgetTest, OverReserveFailsCleanly) {
    auto budget = WorkspaceBudget::Create(100);
    ASSERT_TRUE(budget.has_value());
    EXPECT_TRUE(budget->Reserve(100));
    EXPECT_FALSE(budget->Reserve(1));
    EXPECT_EQ(budget->CommittedBytes(), 100u);
    EXPECT_EQ(budget->Remaining(), 0u);
}

TEST(WorkspaceBudgetTest, OversizedReserveFailsWithoutUnderflow) {
    auto budget = WorkspaceBudget::Create(100);
    ASSERT_TRUE(budget.has_value());
    EXPECT_FALSE(budget->Reserve(std::numeric_limits<std::size_t>::max()));
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 100u);
    EXPECT_FALSE(budget->Reserve(101));
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 100u);
}

TEST(WorkspaceBudgetTest, ZeroReserveSucceedsWithoutCharging) {
    auto budget = WorkspaceBudget::Create(100);
    ASSERT_TRUE(budget.has_value());
    EXPECT_TRUE(budget->Reserve(0));
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 100u);
}

TEST(WorkspaceBudgetTest, RemainingIsMonotoneNonIncreasing) {
    auto budget = WorkspaceBudget::Create(1000);
    ASSERT_TRUE(budget.has_value());
    std::size_t lastRemaining = budget->Remaining();
    std::size_t lastCommitted = budget->CommittedBytes();

    for (std::size_t i = 0; i < 64; ++i)
    {
        EXPECT_TRUE(budget->Reserve(7));
        budget->Release(3);
        EXPECT_LE(budget->Remaining(), lastRemaining);
        EXPECT_GE(budget->CommittedBytes(), lastCommitted);
        lastRemaining = budget->Remaining();
        lastCommitted = budget->CommittedBytes();
    }

    EXPECT_EQ(budget->CommittedBytes(), 448u);
}

TEST(WorkspaceBudgetTest, ConcurrentReservesChargeExactlyOnce) {
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kPerThread = 250;
    auto budget = WorkspaceBudget::Create(kThreads * kPerThread);
    ASSERT_TRUE(budget.has_value());
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (std::size_t t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([&budget] {
            for (std::size_t i = 0; i < kPerThread; ++i)
            {
                EXPECT_TRUE(budget->Reserve(1));
            }
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(budget->CommittedBytes(), kThreads * kPerThread);
    EXPECT_EQ(budget->Remaining(), 0u);
}

TEST(WorkspaceBudgetTest, ConcurrentOversubscriptionFailsExactlyOncePerShortfall) {
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kPerThread = 100;
    // One byte short of the total demand: exactly one reservation fails.
    auto budget = WorkspaceBudget::Create(kThreads * kPerThread - 1);
    ASSERT_TRUE(budget.has_value());
    std::atomic<std::size_t> successes{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (std::size_t t = 0; t < kThreads; ++t)
    {
        threads.emplace_back([&budget, &successes] {
            std::size_t local = 0;

            for (std::size_t i = 0; i < kPerThread; ++i)
            {
                if (budget->Reserve(1))
                {
                    ++local;
                }
            }

            successes.fetch_add(local, std::memory_order_relaxed);
        });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(successes.load(), kThreads * kPerThread - 1);
    EXPECT_EQ(budget->CommittedBytes(), kThreads * kPerThread - 1);
    EXPECT_EQ(budget->Remaining(), 0u);
}

static_assert(!std::is_copy_constructible_v<WorkspaceBudget>,
              "budgets are shared by pointer, never copied");
static_assert(!std::is_copy_assignable_v<WorkspaceBudget>,
              "budgets are shared by pointer, never copied");
static_assert(std::is_move_constructible_v<WorkspaceBudget>,
              "move construction exists for the Result-creating house pattern");
