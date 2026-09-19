// The CUDA-gated GPU memory probe: succeeds and returns
// plausible values when a device is present; skipped without one (the
// no-device zeroed-fallback path is exercised by the host-only
// MemoryTopologyTest.GpuProbeIsZeroedWithoutCuda in memory_topology_test.cpp,
// which compiles exactly when this test's CUDA build does not).

#include "qcx/backend/memory_topology.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>

namespace {

TEST(GpuMemoryTest, ProbeReportsPlausibleDeviceMemory) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    const qcx::backend::GpuMemoryInfo gpu = qcx::backend::DetectGpuMemory();
    std::cout << "DetectGpuMemory: total " << gpu.totalBytes << " bytes ("
              << gpu.totalBytes / 1048576 << " MiB), free " << gpu.freeBytes << " bytes ("
              << gpu.freeBytes / 1048576 << " MiB)\n";
    EXPECT_GT(gpu.totalBytes, 0u);
    EXPECT_GT(gpu.freeBytes, 0u);
    EXPECT_LE(gpu.freeBytes, gpu.totalBytes);
}

} // namespace
