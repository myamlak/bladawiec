// The GPU device-budget gate: the device workspace budget mirror
// (memory/device_workspace_budget.hpp - the free-VRAM probe minus the
// display headroom at Create) and the Create-time device footprint estimate
// (GpuDeviceFootprint, eri_cuda.cpp) match the MEASURED VRAM usage on the
// Quadro fixtures (H2O/STO-3G and C12H26/STO-3G): the per-call growth of
// one BuildFock call (peak minus create, measured by a polling thread)
// and the create / post-call resident slices, every slice one-sided (the
// measured must never exceed the estimate by more than 10% - the driver
// maps the create's allocations granularity-lazily, so the measured can
// under-run the estimate's byte truth; it errs conservative). The driver's
// first-mapping quantum is pre-paid by a kept-alive 64 KiB buffer before
// the baseline (measured 2 MiB on the T1000). The estimate must fit the
// budget's capacity on both fixtures. Self-skips without a CUDA device
// (the RequireCudaDevice convention of the CUDA suite).

#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/backend/topology_profile.hpp"
#include "qcx/integrals/eri_cuda.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/memory/device_workspace_budget.hpp"
#include "shellset_fixture.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <random>
#include <thread>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

/// The display headroom the driver's device budget reserves (the
/// SelectionCostConstants value the driver layer uses).
constexpr std::size_t kDisplayHeadroomBytes = 512 * 1024 * 1024;

// H = T + V from the one-electron engines (the gpu_fock_build_test.cpp
// helper, duplicated test-locally).
qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return std::move(*kinetic);
}

// A physical-density-shaped symmetric matrix (the fock_build_test.cpp
// helper, duplicated test-locally): 0.5 on the diagonal (D ~ N/2 in the
// spatial density), U(-0.25, 0.25) deviations everywhere.
Eigen::MatrixXd PhysicalDensity(std::size_t n) {
    std::mt19937_64 rng(20260817);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

/// The free-VRAM probe of THIS process (backend::DetectDevices - the same
/// probe the driver's topology sweep uses; the first call creates the CUDA
/// context, so callers warm the context before taking any baseline).
std::size_t FreeVramBytes() {
    const auto devices = qcx::backend::DetectDevices();
    return devices.empty() ? 0 : devices.front().freeBytes;
}

/// Runs the device-budget gate on one fixture: probes free VRAM before Create,
/// creates the builder (the engine's Create uploads the tables), polls the
/// free VRAM through one BuildFock call (the peak), probes after the call,
/// and asserts the measured deltas against the estimated slices one-sided
/// within 10% (the measured never exceeds the estimate).
void RunVramGate(const qcx::molecule::Molecule& molecule,
                 const qcx::basisset::BasisSet& basisSet,
                 const char* fixtureName) {
    // Warm the CUDA context before any baseline: the first runtime call
    // maps the context (hundreds of MiB on a small card), which must not
    // land inside a measured delta. The allocate/free cycle settles the
    // driver's lazy mapping growth.
    for (int i = 0; i < 4; ++i)
    {
        (void)FreeVramBytes();
    }

    // Pre-pay the driver's first-allocation mapping quantum (measured 2 MiB
    // on the T1000 at this gate): the FIRST small cudaMalloc maps 2 MiB
    // regardless of its size. The buffer stays alive through the whole
    // measurement, so the baseline sits on the mapped region and every
    // engine allocation of the run maps its true size (exact at >= 4 MiB,
    // riding the region below that). Without this, the 2 MiB quantum lands
    // inside the H2O fixture's create delta and no estimate can match it
    // within 10% (12.0 MiB measured for 84 KB of tables + a 9.6 MiB pool).
    void* quantumBuffer = nullptr;
    ASSERT_EQ(cudaMalloc(&quantumBuffer, 64 * 1024), cudaSuccess);
    std::unique_ptr<void, void (*)(void*)> quantum(quantumBuffer,
                                                   [](void* p) { (void)cudaFree(p); });

    auto coreHamiltonian = BuildCoreHamiltonian(molecule, basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    const std::size_t n = coreHamiltonian->Shape()[0];
    const Eigen::MatrixXd d = PhysicalDensity(n);
    const auto density = ToTensor(d);
    ASSERT_TRUE(density.has_value());

    const std::size_t freeBefore = FreeVramBytes();
    ASSERT_GT(freeBefore, kDisplayHeadroomBytes);

    // The budget seam: the free-VRAM probe minus the display headroom at
    // Create.
    auto budget = qcx::memory::DeviceWorkspaceBudget::Create(0, freeBefore, kDisplayHeadroomBytes);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;

    auto builder = qcx::integrals::GpuJkFockBuilder::Create(molecule, basisSet, *coreHamiltonian);

    if (!builder.has_value() && builder.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << builder.error().message;
    }

    ASSERT_TRUE(builder.has_value());
    const qcx::integrals::GpuDeviceFootprint estimate = builder->DeviceFootprint();
    // The estimate must fit the probed budget on the fixture.
    EXPECT_TRUE(budget->Reserve(estimate.Total()));

    const std::size_t freeAfterCreate = FreeVramBytes();

    // The peak of one BuildFock call: a polling thread records the minimum
    // free VRAM while the call runs (the per-call buffers are freed at the
    // call's return, so a post-call probe alone would miss the peak).
    std::atomic<bool> done{false};
    std::atomic<std::size_t> minFree{freeBefore};
    std::thread poll([&done, &minFree] {
        while (!done.load(std::memory_order_relaxed))
        {
            const std::size_t free = FreeVramBytes();

            for (std::size_t current = minFree.load(std::memory_order_relaxed);
                 free < current &&
                 !minFree.compare_exchange_weak(current, free, std::memory_order_relaxed);)
            {
            }
        }
    });

    auto fock = builder->BuildFock(*density);
    done.store(true, std::memory_order_relaxed);
    poll.join();
    ASSERT_TRUE(fock.has_value());

    const std::size_t freeAfterCall = FreeVramBytes();

    const std::size_t createDelta = freeBefore - freeAfterCreate;
    const std::size_t peakDelta = freeBefore - minFree.load(std::memory_order_relaxed);
    const std::size_t callDelta = freeBefore - freeAfterCall;
    const std::size_t residentBytes = estimate.tablesBytes + estimate.boysBytes;
    // The create slice carries the structural allowance too: the cuBLAS
    // module and the ERI cubins are loaded by Create (measured at the gate:
    // an 8 MiB per-Create pool plus the 2 MiB first-mapping granule) and
    // retained for the process lifetime.
    const std::size_t residentAndStructuralBytes = residentBytes + estimate.structuralBytes;
    const std::size_t perCallGrowthDelta = peakDelta - createDelta;

    std::cout << "[" << fixtureName << "] n = " << n << ": tables " << estimate.tablesBytes
              << " B, boys " << estimate.boysBytes << " B, matrices " << estimate.matricesBytes
              << " B, batchScratch " << estimate.batchScratchBytes << " B, perCallOutput "
              << estimate.perCallOutputBytes << " B, riResidency " << estimate.riResidencyBytes
              << " B, structural " << estimate.structuralBytes << " B; estimate Total "
              << estimate.Total() << " B; measured: create delta " << createDelta << " B, peak "
              << peakDelta << " B, post-call " << callDelta << " B" << std::endl;

    // The gate, one-sided in every slice (the measured must never exceed
    // the estimate by more than 10% - the direction that keeps the device
    // budget's refusal safe). The PER-CALL GROWTH slice (peak minus create)
    // is the state-free, byte-exact one: the driver's mapping of the
    // per-call allocations is exact at these sizes, and the C12 measurement
    // pins the estimate to 0.0003% (measured growth 31,457,280 B vs
    // perCallOutput + matrices 31,457,374 B; H2O's growth is 0 - its
    // per-call buffers ride the mapped region). The CREATE and POST-CALL
    // slices are pool-state-bound: the driver maps an 8 MiB per-Create
    // pool plus a 2 MiB first-mapping granule (measured on the T1000), so
    // the measured create can under-run the estimate's byte-truth charge -
    // the estimate errs conservative there (H2O create 8,388,608-10,485,760
    // B vs 10,569,344 B estimated depending on the suite's prior state;
    // C12 create 12,582,912 B vs 14,861,048 B estimated - the uploads'
    // mapping rode the pool). The estimate never under-charges the
    // measured usage in any slice; the evidence is in the D-record.
    const double perCallEstimate =
        static_cast<double>(estimate.perCallOutputBytes + estimate.matricesBytes);
    EXPECT_LE(static_cast<double>(perCallGrowthDelta), 1.10 * perCallEstimate);
    EXPECT_LE(static_cast<double>(createDelta),
              1.10 * static_cast<double>(residentAndStructuralBytes));
    EXPECT_LE(static_cast<double>(callDelta),
              1.10 * static_cast<double>(residentAndStructuralBytes));
}

} // namespace

TEST(GpuDeviceBudgetTest, H2oSto3gEstimateMatchesMeasuredVram) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    RunVramGate(*molecule, *basisSet, "H2O/STO-3G");
}

TEST(GpuDeviceBudgetTest, C12H26Sto3gEstimateMatchesMeasuredVram) {
    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    RunVramGate(*molecule, *basisSet, "C12H26/STO-3G");
}
