// The topology-profile probe (the device-abstraction design's section 7):
// the single-node bundle is consistent, and the device sweep
// is the documented empty fallback without a CUDA tier. The device sweep's
// real per-device assertions live in the CUDA-gated
// gpu_topology_test.cpp, which compiles exactly when this test's CUDA
// build does not (the same split as MemoryTopologyTest.GpuProbeIsZeroed
// without Cuda) - the fallback symbol exists only in non-CUDA builds.

#ifndef QcxHasCuda

#include "qcx/backend/topology_profile.hpp"

#include <gtest/gtest.h>

namespace {

TEST(TopologyProfileTest, ProbeReportsSingleNode) {
    const qcx::backend::TopologyProfile profile = qcx::backend::DetectTopologyProfile();

    // The single-node class: nothing distributes a build across nodes today,
    // so the bundle is one node with an unknown interconnect by construction.
    EXPECT_EQ(profile.nodeCount, 1u);
    EXPECT_EQ(profile.interconnect, qcx::backend::InterconnectClass::kUnknown);

    // The CPU half is the DetectCpuTopology contract: the P/E split sums
    // to the physical cores and the hyperthreaded flag tracks the logical
    // excess (the probe's own consistency test, re-asserted here on the
    // bundle).
    EXPECT_GE(profile.cpu.logicalProcessors, 1u);
    EXPECT_GE(profile.cpu.physicalCores, 1u);
    EXPECT_LE(profile.cpu.physicalCores, profile.cpu.logicalProcessors);
    EXPECT_EQ(profile.cpu.performanceCores + profile.cpu.efficiencyCores,
              profile.cpu.physicalCores);
    EXPECT_EQ(profile.cpu.hyperthreaded, profile.cpu.logicalProcessors > profile.cpu.physicalCores);

    // The host-RAM half is the DetectHostMemory contract.
    EXPECT_GT(profile.hostMemory.totalBytes, 0u);
    EXPECT_LE(profile.hostMemory.availableBytes, profile.hostMemory.totalBytes);
}

TEST(TopologyProfileTest, DeviceSweepIsEmptyWithoutCuda) {
    // Compiled only in non-CUDA builds (the file guard): the empty sweep
    // is the "no GPU tier" contract, never an error.
    EXPECT_TRUE(qcx::backend::DetectDevices().empty());
}

} // namespace

#endif // QcxHasCuda
