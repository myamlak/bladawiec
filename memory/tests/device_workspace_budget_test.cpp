#include "qcx/memory/device_workspace_budget.hpp"

#include <atomic>
#include <gtest/gtest.h>
#include <limits>
#include <thread>
#include <type_traits>
#include <vector>

using qcx::memory::DeviceWorkspaceBudget;

TEST(DeviceWorkspaceBudgetTest, CreateRejectsProbeWithoutHeadroom) {
    auto budget = DeviceWorkspaceBudget::Create(0, 100, 100);
    ASSERT_FALSE(budget.has_value());
    EXPECT_EQ(budget.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(budget.error().message.find("device 0"), std::string::npos);
}

TEST(DeviceWorkspaceBudgetTest, CreateRejectsProbeBelowHeadroom) {
    auto budget = DeviceWorkspaceBudget::Create(1, 100, 200);
    ASSERT_FALSE(budget.has_value());
    EXPECT_EQ(budget.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(budget.error().message.find("device 1"), std::string::npos);
}

TEST(DeviceWorkspaceBudgetTest, CreateRejectsZeroProbe) {
    auto budget = DeviceWorkspaceBudget::Create(0, 0, 0);
    ASSERT_FALSE(budget.has_value());
    EXPECT_EQ(budget.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DeviceWorkspaceBudgetTest, CapacityIsProbeMinusHeadroom) {
    auto budget = DeviceWorkspaceBudget::Create(0, 4096, 512);
    ASSERT_TRUE(budget.has_value());
    EXPECT_EQ(budget->DeviceId(), 0);
    EXPECT_EQ(budget->FreeBytesAtCreate(), 4096u);
    EXPECT_EQ(budget->HeadroomBytes(), 512u);
    EXPECT_EQ(budget->CapacityBytes(), 3584u);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 3584u);
}

TEST(DeviceWorkspaceBudgetTest, ZeroHeadroomKeepsFullProbe) {
    auto budget = DeviceWorkspaceBudget::Create(2, 2048, 0);
    ASSERT_TRUE(budget.has_value());
    EXPECT_EQ(budget->DeviceId(), 2);
    EXPECT_EQ(budget->CapacityBytes(), 2048u);
    EXPECT_EQ(budget->Remaining(), 2048u);
}

TEST(DeviceWorkspaceBudgetTest, ReserveChargesBytesInFull) {
    auto budget = DeviceWorkspaceBudget::Create(0, 4096, 512);
    ASSERT_TRUE(budget.has_value());
    EXPECT_TRUE(budget->Reserve(256));
    EXPECT_EQ(budget->CommittedBytes(), 256u);
    EXPECT_EQ(budget->Remaining(), 3328u);
    EXPECT_TRUE(budget->Reserve(512));
    EXPECT_EQ(budget->CommittedBytes(), 768u);
    EXPECT_EQ(budget->Remaining(), 2816u);
}

TEST(DeviceWorkspaceBudgetTest, ReleaseDoesNotRestoreCapacity) {
    auto budget = DeviceWorkspaceBudget::Create(0, 700, 500);
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

TEST(DeviceWorkspaceBudgetTest, ReleaseLeavesCounterUntouchedWhenOversized) {
    auto budget = DeviceWorkspaceBudget::Create(0, 1064, 1000);
    ASSERT_TRUE(budget.has_value());
    ASSERT_TRUE(budget->Reserve(32));
    // Releasing more than was reserved cannot underflow the counter: the
    // cumulative commit is monotone and Release never moves it.
    budget->Release(std::numeric_limits<std::size_t>::max());
    EXPECT_EQ(budget->CommittedBytes(), 32u);
    EXPECT_EQ(budget->Remaining(), 32u);
}

TEST(DeviceWorkspaceBudgetTest, OverReserveFailsCleanly) {
    auto budget = DeviceWorkspaceBudget::Create(0, 600, 500);
    ASSERT_TRUE(budget.has_value());
    EXPECT_TRUE(budget->Reserve(100));
    EXPECT_FALSE(budget->Reserve(1));
    EXPECT_EQ(budget->CommittedBytes(), 100u);
    EXPECT_EQ(budget->Remaining(), 0u);
}

TEST(DeviceWorkspaceBudgetTest, OversizedReserveFailsWithoutUnderflow) {
    auto budget = DeviceWorkspaceBudget::Create(0, 600, 500);
    ASSERT_TRUE(budget.has_value());
    EXPECT_FALSE(budget->Reserve(std::numeric_limits<std::size_t>::max()));
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 100u);
    EXPECT_FALSE(budget->Reserve(101));
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 100u);
}

TEST(DeviceWorkspaceBudgetTest, ZeroReserveSucceedsWithoutCharging) {
    auto budget = DeviceWorkspaceBudget::Create(0, 600, 500);
    ASSERT_TRUE(budget.has_value());
    EXPECT_TRUE(budget->Reserve(0));
    EXPECT_EQ(budget->CommittedBytes(), 0u);
    EXPECT_EQ(budget->Remaining(), 100u);
}

TEST(DeviceWorkspaceBudgetTest, RemainingIsMonotoneNonIncreasing) {
    auto budget = DeviceWorkspaceBudget::Create(0, 1500, 500);
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

TEST(DeviceWorkspaceBudgetTest, ConcurrentReservesChargeExactlyOnce) {
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kPerThread = 250;
    auto budget = DeviceWorkspaceBudget::Create(0, kThreads * kPerThread, 0);
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

TEST(DeviceWorkspaceBudgetTest, ConcurrentOversubscriptionFailsExactlyOncePerShortfall) {
    constexpr std::size_t kThreads = 8;
    constexpr std::size_t kPerThread = 100;
    // One byte short of the total demand: exactly one reservation fails.
    auto budget = DeviceWorkspaceBudget::Create(0, kThreads * kPerThread - 1, 0);
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

static_assert(!std::is_copy_constructible_v<DeviceWorkspaceBudget>,
              "budgets are shared by pointer, never copied");
static_assert(!std::is_copy_assignable_v<DeviceWorkspaceBudget>,
              "budgets are shared by pointer, never copied");
static_assert(std::is_move_constructible_v<DeviceWorkspaceBudget>,
              "move construction exists for the Result-creating house pattern");
