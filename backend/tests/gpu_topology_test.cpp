// The CUDA-gated topology sweep: the
// device count and per-device VRAM of the probe cross-checked against the
// CUDA runtime and against nvidia-smi on the local machine. Skipped
// without a device (the host-only fallback path is exercised by
// TopologyProfileTest.DeviceSweepIsEmptyWithoutCuda in topology_profile_test.cpp,
// which compiles exactly when this test's CUDA build does not).

#include "qcx/backend/topology_profile.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

namespace {

TEST(GpuTopologyTest, DetectDevicesMatchesCudaRuntime) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    const std::vector<qcx::backend::DeviceProfile> devices = qcx::backend::DetectDevices();
    ASSERT_EQ(devices.size(), static_cast<std::size_t>(deviceCount));

    for (int i = 0; i < deviceCount; ++i)
    {
        const qcx::backend::DeviceProfile& device = devices[static_cast<std::size_t>(i)];
        EXPECT_EQ(device.deviceId, i);
        EXPECT_GT(device.totalBytes, 0u);
        EXPECT_GT(device.freeBytes, 0u);
        EXPECT_LE(device.freeBytes, device.totalBytes);
        std::cout << "device " << i << ": total " << device.totalBytes / 1048576 << " MiB, free "
                  << device.freeBytes / 1048576 << " MiB\n";
    }

    // The probe never leaves the caller's current device changed.
    int currentDevice = 0;
    ASSERT_EQ(cudaGetDevice(&currentDevice), cudaSuccess);
    const std::vector<qcx::backend::DeviceProfile> again = qcx::backend::DetectDevices();
    int afterProbe = -1;
    ASSERT_EQ(cudaGetDevice(&afterProbe), cudaSuccess);
    EXPECT_EQ(afterProbe, currentDevice);
    EXPECT_EQ(again.size(), devices.size());
}

TEST(GpuTopologyTest, DeviceGatedProbeSkipsTheCudaInit) {
    // The D-A9 probe gate (driver run_driver.cpp): a caller whose memory
    // cap cannot hold the CUDA build's host footprint (the runtime DLLs'
    // load-time commit) asks for the CPU and host halves only. The device
    // sweep must not start - initializing CUDA under a sub-footprint
    // job-object cap is a hard access violation, never a clean error - so
    // the profile reports the empty "no GPU tier" sweep while the CPU and
    // host halves still probe.
    const qcx::backend::TopologyProfile profile = qcx::backend::DetectTopologyProfile(false);
    EXPECT_TRUE(profile.devices.empty());
    EXPECT_EQ(profile.nodeCount, 1u);
    EXPECT_EQ(profile.interconnect, qcx::backend::InterconnectClass::kUnknown);
    EXPECT_GE(profile.cpu.physicalCores, 1u);
    EXPECT_GT(profile.hostMemory.totalBytes, 0u);
}

TEST(GpuTopologyTest, DetectTopologyProfileBundlesDevices) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    const qcx::backend::TopologyProfile profile = qcx::backend::DetectTopologyProfile();
    EXPECT_EQ(profile.nodeCount, 1u);
    EXPECT_EQ(profile.interconnect, qcx::backend::InterconnectClass::kUnknown);
    EXPECT_EQ(profile.devices.size(), static_cast<std::size_t>(deviceCount));
    EXPECT_GE(profile.cpu.physicalCores, 1u);
    EXPECT_GT(profile.hostMemory.totalBytes, 0u);
}

TEST(GpuTopologyTest, DetectDevicesMatchesNvidiaSmi) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device available";
    }

    // The gate: device count + VRAM match nvidia-smi within
    // tolerance. The total comparison pins the device identity and VRAM
    // size (1%). The free values differ by construction: the probe runs in
    // the same process that owns the CUDA context - exactly the driver's
    // usage, where the context footprint is NOT free - while nvidia-smi is
    // a separate process reporting the device without that footprint
    // (~590 MiB measured on the T1000-class 4 GiB device). The free
    // tolerance therefore adds a fixed context-footprint allowance to the
    // point-in-time fraction.
    constexpr double kTotalToleranceFraction = 0.01;
    constexpr double kFreeToleranceFraction = 0.10;
    constexpr double kContextFootprintAllowanceMiB = 1024.0;

    // nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits
    // emits one "TOTAL, FREE" row per device, MiB.
    std::string command = "nvidia-smi --query-gpu=memory.total,memory.free "
                          "--format=csv,noheader,nounits";
    FILE* pipe = _popen(command.c_str(), "r");

    if (pipe == nullptr)
    {
        GTEST_SKIP() << "nvidia-smi not available";
    }

    std::vector<std::pair<double, double>> smiRows;

    char line[256];

    while (std::fgets(line, sizeof(line), pipe) != nullptr)
    {
        double totalMiB = 0.0;
        double freeMiB = 0.0;

        if (std::sscanf(line, "%lf,%lf", &totalMiB, &freeMiB) == 2)
        {
            smiRows.push_back({totalMiB, freeMiB});
        }
    }

    const int closeResult = _pclose(pipe);

    if (closeResult != 0 || smiRows.empty())
    {
        GTEST_SKIP() << "nvidia-smi produced no rows";
    }

    const std::vector<qcx::backend::DeviceProfile> devices = qcx::backend::DetectDevices();
    ASSERT_EQ(devices.size(), smiRows.size());

    for (std::size_t i = 0; i < devices.size(); ++i)
    {
        const double smiTotalMiB = smiRows[i].first;
        const double smiFreeMiB = smiRows[i].second;
        const double probeTotalMiB = static_cast<double>(devices[i].totalBytes) / 1048576.0;
        const double probeFreeMiB = static_cast<double>(devices[i].freeBytes) / 1048576.0;
        EXPECT_NEAR(probeTotalMiB, smiTotalMiB, smiTotalMiB * kTotalToleranceFraction);
        EXPECT_NEAR(probeFreeMiB,
                    smiFreeMiB,
                    smiFreeMiB * kFreeToleranceFraction + kContextFootprintAllowanceMiB);
        std::cout << "nvidia-smi device " << i << ": total " << smiTotalMiB << " MiB, free "
                  << smiFreeMiB << " MiB; probe " << probeTotalMiB << " MiB, free " << probeFreeMiB
                  << " MiB\n";
    }
}

} // namespace
