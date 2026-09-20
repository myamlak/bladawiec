#include "qcx/error.hpp"

#include <gtest/gtest.h>

TEST(ErrorTest, ResultHoldsValueOnSuccess) {
    qcx::Result<int> r = 42;
    ASSERT_TRUE(r.has_value());
    EXPECT_EQ(*r, 42);
}

TEST(ErrorTest, ResultHoldsErrorOnFailure) {
    qcx::Result<int> r = std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "bad input"});
    ASSERT_FALSE(r.has_value());
    EXPECT_EQ(r.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(r.error().message, "bad input");
}

TEST(ErrorTest, ErrorCodeHasHumanReadableName) {
    EXPECT_EQ(qcx::ToString(qcx::ErrorCode::kInvalidArgument), "InvalidArgument");
    EXPECT_EQ(qcx::ToString(qcx::ErrorCode::kIOError), "IOError");
    EXPECT_EQ(qcx::ToString(qcx::ErrorCode::kDeviceError), "DeviceError");
}
