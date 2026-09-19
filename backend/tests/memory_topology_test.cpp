// Host RAM probe: succeeds and returns plausible values on
// any supported OS - the total must be positive on a real machine, and the
// available share can never exceed it. The GPU probe's no-CUDA fallback
// returns the documented zeroed struct; the real CUDA-gated probe has its
// own test in the CUDA target (gpu_memory_test.cpp) because it needs the
// cudart link that only that target has.

#include "qcx/backend/memory_topology.hpp"

#include <gtest/gtest.h>

namespace {

TEST(MemoryTopologyTest, HostProbeSucceeds) {
    const qcx::backend::HostMemoryInfo memory = qcx::backend::DetectHostMemory();
    EXPECT_GT(memory.totalBytes, 0u);
    EXPECT_GT(memory.availableBytes, 0u);
    EXPECT_LE(memory.availableBytes, memory.totalBytes);
}

#ifndef QcxHasCuda

TEST(MemoryTopologyTest, GpuProbeIsZeroedWithoutCuda) {
    const qcx::backend::GpuMemoryInfo gpu = qcx::backend::DetectGpuMemory();
    EXPECT_EQ(gpu.totalBytes, 0u);
    EXPECT_EQ(gpu.freeBytes, 0u);
}

#endif

} // namespace
