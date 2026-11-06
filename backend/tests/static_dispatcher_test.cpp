#include "qcx/backend/static_dispatcher.hpp"

#include <array>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

using Bounds = std::array<std::size_t, 2>;

void ExpectRange(const qcx::Result<qcx::backend::IndexRange>& actual, Bounds expected) {
    ASSERT_TRUE(actual.has_value()) << actual.error().message;
    EXPECT_EQ(actual->begin, expected[0]);
    EXPECT_EQ(actual->end, expected[1]);
}

} // namespace

// The split is fixed arithmetic: each rank takes count / size items, the last
// rank takes the remainder. The cases below are the worked examples of that
// rule, one per shape of the remainder.
TEST(StaticDispatcherTest, LastRankTakesTheRemainderForFiveOverTwo) {
    ExpectRange(qcx::backend::PartitionRange(5, 0, 2), Bounds{0, 2});
    ExpectRange(qcx::backend::PartitionRange(5, 1, 2), Bounds{2, 5});
}

TEST(StaticDispatcherTest, LastRankTakesTheRemainderForThreeOverTwo) {
    ExpectRange(qcx::backend::PartitionRange(3, 0, 2), Bounds{0, 1});
    ExpectRange(qcx::backend::PartitionRange(3, 1, 2), Bounds{1, 3});
}

TEST(StaticDispatcherTest, LastRankTakesTheOnlyRemainderForSixOverFour) {
    ExpectRange(qcx::backend::PartitionRange(6, 0, 4), Bounds{0, 1});
    ExpectRange(qcx::backend::PartitionRange(6, 1, 4), Bounds{1, 2});
    ExpectRange(qcx::backend::PartitionRange(6, 2, 4), Bounds{2, 3});
    ExpectRange(qcx::backend::PartitionRange(6, 3, 4), Bounds{3, 6});
}

TEST(StaticDispatcherTest, OneItemPerRankWhenEveryRankFitsExactly) {
    for (int rank = 0; rank < 8; ++rank)
    {
        ExpectRange(qcx::backend::PartitionRange(8, rank, 8),
                    Bounds{static_cast<std::size_t>(rank), static_cast<std::size_t>(rank) + 1});
    }
}

TEST(StaticDispatcherTest, SingleRankTakesEverything) {
    ExpectRange(qcx::backend::PartitionRange(7, 0, 1), Bounds{0, 7});
    ExpectRange(qcx::backend::PartitionRange(1, 0, 1), Bounds{0, 1});
}

// The blocks of a partition are contiguous, disjoint and cover every item.
TEST(StaticDispatcherTest, BlocksCoverEveryItemExactlyOnce) {
    constexpr std::size_t kCount = 17;
    constexpr int kSize = 4;

    std::size_t covered = 0;

    for (int rank = 0; rank < kSize; ++rank)
    {
        const auto range = qcx::backend::PartitionRange(kCount, rank, kSize);
        ASSERT_TRUE(range.has_value());
        EXPECT_EQ(range->begin, covered);
        EXPECT_GT(range->end, range->begin);
        covered = range->end;
    }

    EXPECT_EQ(covered, kCount);
}

// A rank must never receive an empty block. The refusal is computed from the
// count and the rank count alone, so it is identical on every rank and can be
// raised before the first collective.
TEST(StaticDispatcherTest, RefusesWhenARankWouldGetNoWork) {
    const auto noItems = qcx::backend::PartitionRange(0, 0, 1);
    ASSERT_FALSE(noItems.has_value());
    EXPECT_EQ(noItems.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto fewerItemsThanRanks = qcx::backend::PartitionRange(3, 0, 4);
    ASSERT_FALSE(fewerItemsThanRanks.has_value());
    EXPECT_EQ(fewerItemsThanRanks.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto oneItemForTwoRanks = qcx::backend::PartitionRange(1, 1, 2);
    ASSERT_FALSE(oneItemForTwoRanks.has_value());
    EXPECT_EQ(oneItemForTwoRanks.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(StaticDispatcherTest, RefusalMessageIsIdenticalOnEveryRank) {
    constexpr std::size_t kCount = 3;
    constexpr int kSize = 4;

    const auto first = qcx::backend::PartitionRange(kCount, 0, kSize);
    ASSERT_FALSE(first.has_value());

    for (int rank = 1; rank < kSize; ++rank)
    {
        const auto other = qcx::backend::PartitionRange(kCount, rank, kSize);
        ASSERT_FALSE(other.has_value());
        EXPECT_EQ(other.error().message, first.error().message);
    }
}

TEST(StaticDispatcherTest, RefusesAnInvalidRankOrRankCount) {
    const auto noRanks = qcx::backend::PartitionRange(5, 0, 0);
    ASSERT_FALSE(noRanks.has_value());
    EXPECT_EQ(noRanks.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto negativeRank = qcx::backend::PartitionRange(5, -1, 2);
    ASSERT_FALSE(negativeRank.has_value());
    EXPECT_EQ(negativeRank.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto rankPastTheEnd = qcx::backend::PartitionRange(5, 2, 2);
    ASSERT_FALSE(rankPastTheEnd.has_value());
    EXPECT_EQ(rankPastTheEnd.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(StaticDispatcherTest, BindsRankAndSize) {
    const qcx::backend::StaticDispatcher last(1, 2);
    EXPECT_EQ(last.Rank(), 1);
    EXPECT_EQ(last.Size(), 2);
    ExpectRange(last.Partition(5), Bounds{2, 5});

    const qcx::backend::StaticDispatcher first(0, 2);
    ExpectRange(first.Partition(3), Bounds{0, 1});
}
