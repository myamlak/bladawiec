// The host-side compute-profile contract (the non-CUDA half): without a
// CUDA build the probe is the documented
// "unknown" profile - the fallback path, compiled only when the CUDA
// build does not (the QcxHasCuda guard convention of
// topology_profile_test.cpp).

#include "qcx/backend/gpu_compute_profile.hpp"

#include <gtest/gtest.h>

namespace {

#ifndef QcxHasCuda

TEST(GpuComputeProfileTest, UnknownProfileWithoutCuda) {
    const qcx::backend::GpuComputeProfile profile = qcx::backend::DetectGpuComputeProfile(0);
    EXPECT_FALSE(profile.tensorCores);
    EXPECT_FALSE(profile.fp16TensorCores);
    EXPECT_DOUBLE_EQ(profile.fp32ToFp64Ratio, 1.0);
    EXPECT_DOUBLE_EQ(profile.fp32Gflops, 0.0);
    EXPECT_DOUBLE_EQ(profile.fp64Gflops, 0.0);
}

#endif // QcxHasCuda

} // namespace
