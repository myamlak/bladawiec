#include "qcx/backend/tags.hpp"
#include "qcx/memory/device_buffer.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>

using CpuBuffer = qcx::memory::DeviceBuffer<float, qcx::backend::CpuTag>;

TEST(DeviceBufferTest, CreateRejectsZeroCount) {
    auto buf = CpuBuffer::Create(0);
    ASSERT_FALSE(buf.has_value());
    EXPECT_EQ(buf.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DeviceBufferTest, HostViewStartsCanonical) {
    auto buf = CpuBuffer::Create(100);
    ASSERT_TRUE(buf.has_value());
    auto& host = buf->HostView();
    host[0] = 3.14f;
    EXPECT_FLOAT_EQ(buf->HostView()[0], 3.14f);
    EXPECT_EQ(buf->CurrentResidency(), qcx::memory::Residency::kHost);
}

TEST(DeviceBufferTest, ExplicitSyncRoundTripsOnCpu) {
    auto buf = CpuBuffer::Create(8);
    ASSERT_TRUE(buf.has_value());
    auto& host = buf->HostView();

    for (std::size_t i = 0; i < host.size(); ++i)
    {
        host[i] = static_cast<float>(i);
    }

    buf->MarkHostDirty();
    EXPECT_EQ(buf->CurrentResidency(), qcx::memory::Residency::kHostDirty);
    auto syncOut = buf->SyncToDevice(); // host -> device copy
    ASSERT_TRUE(syncOut.has_value());
    EXPECT_EQ(buf->CurrentResidency(), qcx::memory::Residency::kBothValid);

    auto dptr = buf->DeviceHandle();
    EXPECT_TRUE(static_cast<bool>(dptr));

    buf->MarkDeviceDirty();
    auto syncBack = buf->SyncToHost(); // device -> host copy
    ASSERT_TRUE(syncBack.has_value());

    auto& back = buf->HostView();

    for (std::size_t i = 0; i < back.size(); ++i)
    {
        EXPECT_FLOAT_EQ(back[i], static_cast<float>(i));
    }
}

TEST(DeviceBufferTest, MoveTransfersOwnership) {
    auto buf = CpuBuffer::Create(100);
    ASSERT_TRUE(buf.has_value());
    buf->HostView()[0] = 1.0f;
    CpuBuffer moved(std::move(*buf));
    EXPECT_EQ(moved.Size(), 100u);
    EXPECT_FLOAT_EQ(moved.HostView()[0], 1.0f);
}

TEST(DeviceBufferTest, MoveAssignmentTransfersOwnership) {
    auto buf = CpuBuffer::Create(100);
    ASSERT_TRUE(buf.has_value());
    buf->HostView()[0] = 2.0f;
    auto other = CpuBuffer::Create(7);
    ASSERT_TRUE(other.has_value());
    *other = std::move(*buf);
    EXPECT_EQ(other->Size(), 100u);
    EXPECT_FLOAT_EQ(other->HostView()[0], 2.0f);
}

TEST(DeviceBufferTest, CreateRejectsCountOverflow) {
    constexpr auto kMax = std::numeric_limits<std::size_t>::max();
    auto buf = CpuBuffer::Create(kMax / sizeof(float) + 1);
    ASSERT_FALSE(buf.has_value());
    EXPECT_EQ(buf.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DeviceBufferTest, SyncsAreNoOpsWhenBothValid) {
    auto buf = CpuBuffer::Create(4);
    ASSERT_TRUE(buf.has_value());
    buf->HostView()[0] = 1.0f;
    auto out = buf->SyncToDevice();
    ASSERT_TRUE(out.has_value());
    EXPECT_EQ(buf->CurrentResidency(), qcx::memory::Residency::kBothValid);

    auto back = buf->SyncToHost();
    ASSERT_TRUE(back.has_value());
    EXPECT_EQ(buf->CurrentResidency(), qcx::memory::Residency::kBothValid);

    auto again = buf->SyncToDevice();
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(buf->CurrentResidency(), qcx::memory::Residency::kBothValid);
}

TEST(DeviceBufferTest, ConstBufferCanSync) {
    auto buf = CpuBuffer::Create(4);
    ASSERT_TRUE(buf.has_value());
    buf->HostView()[0] = 2.5f;
    buf->MarkHostDirty();
    const CpuBuffer& constBuf = *buf;
    auto sync = constBuf.SyncToDevice();
    ASSERT_TRUE(sync.has_value());
    EXPECT_EQ(constBuf.CurrentResidency(), qcx::memory::Residency::kBothValid);
}

static_assert(!std::is_copy_constructible_v<CpuBuffer>, "DeviceBuffer must be movable-only");
static_assert(!std::is_copy_assignable_v<CpuBuffer>, "DeviceBuffer must be movable-only");
static_assert(std::is_nothrow_move_constructible_v<CpuBuffer>, "moves must not throw");
