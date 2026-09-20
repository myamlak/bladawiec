// The CUDA-gated compute-profile sweep: the probe
// returns sane values on the local device - the tensor-core flag matches
// the device properties, the measured ratio lands inside the shipping-
// hardware band (fp64 is at most the fp32 rate on every CUDA device:
// near 1/2 on the datacenter class, 1/32-1/64 consumer) - and the
// documented "unknown" profile for a device that does not exist. Skipped
// without a device (the self-skip convention of gpu_topology_test.cpp);
// the host-side fallback is pinned by
// GpuComputeProfileTest.UnknownProfileWithoutCuda, which compiles exactly
// when this test's CUDA build does not.

#include "qcx/backend/gpu_compute_profile.hpp"

#include <cmath>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

namespace {

TEST(GpuDeviceProfileTest, DetectComputeProfileMatchesDeviceProps) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    cudaDeviceProp prop{};
    ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);

    const qcx::backend::GpuComputeProfile profile = qcx::backend::DetectGpuComputeProfile(0);
    EXPECT_EQ(profile.tensorCores, prop.major >= 7);
    EXPECT_EQ(profile.fp16TensorCores, profile.tensorCores);

    // Sane-value bands: every shipping CUDA device evaluates fp32 at or
    // above its fp64 rate, and a register-bound FMA kernel on a real
    // device lands between the toy and the theoretical ceilings.
    EXPECT_GT(profile.fp32Gflops, 100.0);
    EXPECT_LT(profile.fp32Gflops, 200000.0);
    EXPECT_GT(profile.fp64Gflops, 1.0);
    EXPECT_LE(profile.fp64Gflops, profile.fp32Gflops);
    EXPECT_GE(profile.fp32ToFp64Ratio, 1.5);
    EXPECT_LE(profile.fp32ToFp64Ratio, 100.0);

    std::cout << "device 0: tensorCores " << profile.tensorCores << ", fp32 " << profile.fp32Gflops
              << " GFLOP/s, fp64 " << profile.fp64Gflops << " GFLOP/s, ratio "
              << profile.fp32ToFp64Ratio << '\n';
}

TEST(GpuDeviceProfileTest, UnknownProfileForMissingDevice) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    const qcx::backend::GpuComputeProfile profile = qcx::backend::DetectGpuComputeProfile(1024);
    EXPECT_FALSE(profile.tensorCores);
    EXPECT_FALSE(profile.fp16TensorCores);
    EXPECT_DOUBLE_EQ(profile.fp32ToFp64Ratio, 1.0);
    EXPECT_DOUBLE_EQ(profile.fp32Gflops, 0.0);
    EXPECT_DOUBLE_EQ(profile.fp64Gflops, 0.0);
}

TEST(GpuDeviceProfileTest, ProbeRestoresCurrentDevice) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    int currentDevice = 0;
    ASSERT_EQ(cudaGetDevice(&currentDevice), cudaSuccess);

    (void)qcx::backend::DetectGpuComputeProfile(0);
    (void)qcx::backend::DetectGpuComputeProfile(1024);

    int afterProbe = -1;
    ASSERT_EQ(cudaGetDevice(&afterProbe), cudaSuccess);
    EXPECT_EQ(afterProbe, currentDevice);
}

} // namespace
