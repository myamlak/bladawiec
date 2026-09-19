// Device-footprint formula tests (device_footprint.hpp): the per-call output
// envelope against hand arithmetic (the gate re-pinned n^-1 screened-exchange
// law of device_footprint.hpp, floored at the fractions of footprint.hpp), and
// the GpuDeviceFootprint record - the pass-through of the caller's exact table
// bytes, the per-call matrices, the kV2-only batch arena and cuBLAS workspace,
// and the driver-pool allowance that is always charged (the per-Create module
// loads are fixture-independent). The VRAM-vs-estimate gate itself is the CUDA
// test (gpu_device_budget_test.cpp in qcx-cuda-md-tests); this suite pins the
// arithmetic on the host.

#include "internal/device_footprint.hpp"
#include "internal/footprint.hpp"

#include <cstddef>
#include <gtest/gtest.h>

using qcx::integrals::GpuDeviceFootprint;
using qcx::integrals::internal::ComputeGpuDeviceFootprint;
using qcx::integrals::internal::GpuPerCallOutputBytes;
using qcx::integrals::internal::kGpuCublasWorkspaceBytes;
using qcx::integrals::internal::kGpuDriverPoolBytes;

TEST(DeviceFootprintTest, PerCallOutputMatchesHandArithmetic) {
    // 8*T + 24*Q with the gate re-pinned n^-1 envelope (device_footprint.hpp:
    // survival = max(5.8765/n, 0.0088), quartets = max(0.09412/n, 1.42e-4);
    // the fp64 pass is the peak: 8T > 4T + 8Q since Q/T ~= 0.016, so the
    // fp32-only bounds never make the peak pass; the 24 B per screened
    // quartet rides the task/meta/ranges buffers). The values are the
    // truncations of the double arithmetic: 3153, 16899, 31339038.
    EXPECT_EQ(GpuPerCallOutputBytes(4), 3153u);
    EXPECT_EQ(GpuPerCallOutputBytes(7), 16899u);
    EXPECT_EQ(GpuPerCallOutputBytes(86), 31339038u);
}

TEST(DeviceFootprintTest, PerCallOutputGrowsWithFunctionCount) {
    std::size_t last = GpuPerCallOutputBytes(4);

    for (std::size_t n = 5; n <= 64; ++n)
    {
        const std::size_t next = GpuPerCallOutputBytes(n);
        EXPECT_GT(next, last);
        last = next;
    }
}

TEST(DeviceFootprintTest, RecordPassesExactTableBytesThrough) {
    const GpuDeviceFootprint terms = ComputeGpuDeviceFootprint(86, 512u, 1234u, 5678u, false);
    EXPECT_EQ(terms.tablesBytes, 1234u);
    EXPECT_EQ(terms.boysBytes, 5678u);
    EXPECT_EQ(terms.matricesBytes, 118336u); // 2 x 8 x 86^2.
    EXPECT_EQ(terms.batchScratchBytes, 0u); // No kV2 class.
    EXPECT_EQ(terms.perCallOutputBytes, GpuPerCallOutputBytes(86));
    EXPECT_EQ(terms.riResidencyBytes, 0u); // The named slot; zero until RI residency lands.
    EXPECT_EQ(terms.structuralBytes, kGpuDriverPoolBytes);
    EXPECT_EQ(terms.Total(),
              terms.tablesBytes + terms.boysBytes + terms.matricesBytes + terms.perCallOutputBytes +
                  terms.structuralBytes);
}

TEST(DeviceFootprintTest, Kv2ClassesChargeTheBatchArenaAndCublasWorkspace) {
    const std::size_t batch = std::size_t{512} * 1024 * 1024;
    const GpuDeviceFootprint terms = ComputeGpuDeviceFootprint(562, batch, 1000u, 2000u, true);
    EXPECT_EQ(terms.batchScratchBytes, batch);
    EXPECT_EQ(terms.structuralBytes, kGpuCublasWorkspaceBytes + kGpuDriverPoolBytes);
    EXPECT_EQ(terms.Total(),
              terms.tablesBytes + terms.boysBytes + terms.matricesBytes + batch +
                  terms.perCallOutputBytes + terms.structuralBytes);
}

TEST(DeviceFootprintTest, SmallFixtureRecordIsSmall) {
    // H2O/STO-3G-class input: n = 7, a fused-only class set.
    const GpuDeviceFootprint terms = ComputeGpuDeviceFootprint(7, 512u, 1000u, 2000u, false);
    EXPECT_EQ(terms.matricesBytes, 784u); // 2 x 8 x 7^2.
    EXPECT_EQ(terms.batchScratchBytes, 0u);
    EXPECT_EQ(terms.structuralBytes, kGpuDriverPoolBytes);
}

TEST(DeviceFootprintTest, NeverUnderTheBatchArenaWithKv2) {
    // The kV2 arena is bounded by the batch cap (one batch at a time); a
    // large cap must not leak beyond the cap into the estimate.
    const std::size_t batch = std::size_t{64} * 1024 * 1024;
    const GpuDeviceFootprint terms = ComputeGpuDeviceFootprint(300, batch, 0u, 0u, true);
    EXPECT_EQ(terms.batchScratchBytes, batch);
    EXPECT_EQ(terms.Total(),
              terms.matricesBytes + terms.perCallOutputBytes + terms.batchScratchBytes +
                  terms.structuralBytes);
}
