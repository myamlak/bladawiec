// The per-device mode-selection gate: the FastPath/LightPath decision
// generalized to the device context.
// The engine's Create-time decision runs against the device workspace
// budget mirror (memory/device_workspace_budget.hpp), the forced-LightPath
// knob (FockBuildOptions::forceGpuLightPath) forces the per-call-statics
// rung, and the light rung uploads the statics per call (EnsureStatics)
// and releases them at the call end (ReleaseStatics, the RAII guard).
//
// The gate: (1) the default mode is FastPath and the knob reports
// LightPath with the byte terms (fast = the full footprint, light = the
// per-call working set, remaining = the budget's Remaining at the
// decision); (2) the budget-driven decision picks LightPath when the
// capacity sits between the rungs and keeps FastPath when it covers the
// full footprint - without the knob; (3) a LightPath BuildFock matches the
// FastPath BuildFock within the fp64 last-bit budget (1e-10 - the
// cross-quartet atomicAdd accumulation order is not deterministic between
// two launches, the existing GPU parity pins' budget; anything beyond it
// is a mode-switch effect), and the engine per-call paths (ComputeBatch
// and ComputeBatchCertified) are bit-identical, values and certified
// bounds included; (4) the light rung's per-call commit carries the
// statics (the peak dips by the table bytes) and the post-call residency
// returns to the baseline within one 2 MiB mapping granule (the statics
// are released -
// the retained-vs-per-call residency contract). The per-call commit is
// the state-free slice of the create gate (peak minus create), so the
// suite's prior pool state does not affect the assertions. (5) The
// refusal before OOM: an artificially tiny device budget refuses at Create
// with kInvalidArgument and the device context named - the four-rung
// ladder (direct screened, batched/blocked, recompute, disk LAST) fires
// before the first device allocation, so no CUDA allocation fails for a
// budget the caller declared, and the forced-LightPath knob cannot bypass
// the refusal (a forced light rung under a budget that cannot hold it is a
// caller contradiction). (6) The residency verification: the per-call
// transfer accounting - the Fock-path traffic by
// class (density up, Fock up/down, the screened lists, the certified lane's
// bounds read-back, and the statics: 0 per call on the FastPath, the full
// table + Boys bytes per call on the LightPath), exact by construction (the
// byte counts are the buffer sizes at the transfer sites) and asserted on
// the C12H26/STO-3G fixture - the residency contract's per-iteration
// statement (density up, Fock down, screened lists - nothing else, at any
// system size), including the screened-lists-only pin (a looser preset
// screens strictly fewer quartets: the lists shrink, nothing else moves)
// and the lane-off shape (kTight disables the certified lane: the bounds
// class is exactly 0). Self-skips without a CUDA device (the
// RequireCudaDevice convention of the CUDA suite).

#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/backend/topology_profile.hpp"
#include "qcx/integrals/eri_cuda.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/device_workspace_budget.hpp"
#include "shellset_fixture.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <memory>
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

/// The driver's allocation-mapping granule on the T1000 (measured 2 MiB;
/// every allocation rounds up to it).
constexpr std::size_t kMappingGranuleBytes = 2 * 1024 * 1024;

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

/// Warm the CUDA context and pre-pay the driver's first-allocation mapping
/// quantum before any measured sequence (the settle cycle: the
/// first runtime call maps the context, which must not land inside a
/// measured delta, and the FIRST small cudaMalloc maps 2 MiB regardless of
/// its size - the kept-alive buffer rides the mapped region through the
/// whole measurement).
void SettleContextAndPrepayQuantum() {
    for (int i = 0; i < 4; ++i)
    {
        (void)FreeVramBytes();
    }

    void* quantumBuffer = nullptr;
    EXPECT_EQ(cudaMalloc(&quantumBuffer, 64 * 1024), cudaSuccess);
    static std::unique_ptr<void, void (*)(void*)> quantum(quantumBuffer,
                                                          [](void* p) { (void)cudaFree(p); });
}

/// The light rung's per-call estimate: the per-call working set - the
/// matrices, the batch scratch and the per-call output.
std::size_t LightEstimate(const qcx::integrals::GpuDeviceFootprint& footprint) {
    return footprint.matricesBytes + footprint.batchScratchBytes + footprint.perCallOutputBytes;
}

/// Asserts one builder's mode record against the footprint (the terms the
/// decision ran on): fast = the full footprint, light = the per-call
/// working set, remaining = the budget's Remaining at the decision.
void ExpectModeInfo(const qcx::integrals::GpuJkFockBuilder& builder,
                    qcx::integrals::FockBuildMode expectedMode,
                    std::size_t expectedRemaining) {
    const qcx::integrals::GpuDeviceFootprint footprint = builder.DeviceFootprint();
    const qcx::integrals::EriCudaModeInfo info = builder.ModeInfo();

    EXPECT_EQ(builder.Mode(), expectedMode);
    EXPECT_EQ(info.mode, expectedMode);
    EXPECT_EQ(info.fastEstimateBytes, footprint.Total());
    EXPECT_EQ(info.lightEstimateBytes, LightEstimate(footprint));
    EXPECT_EQ(info.remainingAtDecision, expectedRemaining);
}

} // namespace

TEST(GpuModeSelectionTest, DefaultModeIsFastPath) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());

    // The engine level: no budget, no knob - the legacy behavior, the
    // statics retained from Create.
    auto engine = qcx::integrals::EriCudaEngine::Create(*molecule, *basisSet);

    if (!engine.has_value() && engine.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << engine.error().message;
    }

    ASSERT_TRUE(engine.has_value());
    EXPECT_EQ(engine->Mode(), qcx::integrals::FockBuildMode::kFastPath);
    EXPECT_EQ(engine->ModeInfo().mode, qcx::integrals::FockBuildMode::kFastPath);
    EXPECT_EQ(engine->ModeInfo().fastEstimateBytes, engine->DeviceFootprint().Total());
    EXPECT_EQ(engine->ModeInfo().lightEstimateBytes, LightEstimate(engine->DeviceFootprint()));
    EXPECT_EQ(engine->ModeInfo().remainingAtDecision, 0u);

    // The builder level: the same default (the knob surface is the
    // builder's FockBuildOptions).
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());
    auto builder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian);

    if (!builder.has_value() && builder.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << builder.error().message;
    }

    ASSERT_TRUE(builder.has_value());
    ExpectModeInfo(*builder, qcx::integrals::FockBuildMode::kFastPath, 0);
}

TEST(GpuModeSelectionTest, ForcedKnobReportsLightPath) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    // The forced-LightPath knob (the mode-forcing test surface): no budget
    // in play, so the remaining-at-decision term is 0 - the knob is the
    // entire decision.
    qcx::integrals::FockBuildOptions forced;
    forced.forceGpuLightPath = true;
    auto builder =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian, forced);

    if (!builder.has_value() && builder.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << builder.error().message;
    }

    ASSERT_TRUE(builder.has_value());
    ExpectModeInfo(*builder, qcx::integrals::FockBuildMode::kLightPath, 0);
}

TEST(GpuModeSelectionTest, BudgetBetweenRungsSelectsLightPath) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    // The reference footprint first (a no-budget builder; the decision
    // surface needs the rung byte terms the engine computes at Create).
    auto reference =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian);

    if (!reference.has_value() && reference.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << reference.error().message;
    }

    ASSERT_TRUE(reference.has_value());
    const qcx::integrals::GpuDeviceFootprint footprint = reference->DeviceFootprint();
    const std::size_t lightEstimate = LightEstimate(footprint);
    ASSERT_GT(footprint.Total(), lightEstimate);

    // A capacity between the rungs: the light rung fits, the fast rung
    // does not - the decision must pick the light rung WITHOUT the knob.
    auto budget = qcx::memory::DeviceWorkspaceBudget::Create(
        0, lightEstimate + kDisplayHeadroomBytes, kDisplayHeadroomBytes);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    ASSERT_EQ(budget->Remaining(), lightEstimate);

    qcx::integrals::FockBuildOptions constrained;
    constrained.deviceWorkspaceBudget = &*budget;
    auto builder = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, constrained);
    ASSERT_TRUE(builder.has_value());
    ExpectModeInfo(*builder, qcx::integrals::FockBuildMode::kLightPath, lightEstimate);
}

TEST(GpuModeSelectionTest, CapacityAboveTotalKeepsFastPath) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    auto reference =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian);

    if (!reference.has_value() && reference.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << reference.error().message;
    }

    ASSERT_TRUE(reference.has_value());
    const qcx::integrals::GpuDeviceFootprint footprint = reference->DeviceFootprint();

    // A capacity covering the FULL footprint: the fast rung fits, so the
    // decision stays on the fast rung - the statics retained from Create.
    auto budget = qcx::memory::DeviceWorkspaceBudget::Create(
        0, footprint.Total() + kDisplayHeadroomBytes, kDisplayHeadroomBytes);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    ASSERT_EQ(budget->Remaining(), footprint.Total());

    qcx::integrals::FockBuildOptions unconstrained;
    unconstrained.deviceWorkspaceBudget = &*budget;
    auto builder = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, unconstrained);
    ASSERT_TRUE(builder.has_value());
    ExpectModeInfo(*builder, qcx::integrals::FockBuildMode::kFastPath, footprint.Total());
}

TEST(GpuModeSelectionTest, TinyBudgetRefusesWithKInvalidArgument) {
    // An artificially tiny device budget - below the light rung's per-call
    // working set - refuses at Create with kInvalidArgument and the device
    // context named, before the first device allocation of the Create (the
    // statics upload and cublasCreate): no CUDA allocation can fail for a
    // budget the caller declared.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    if (qcx::backend::DetectDevices().empty())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    SettleContextAndPrepayQuantum();

    // Capacity 4 KiB: far below the light rung at H2O/STO-3G (the
    // per-call working set is ~17.7 KB), so both device rungs refuse. The
    // reference footprint pins that the tiny budget undercuts the light
    // rung (the assertion fails loudly if the formulas ever shrink below
    // the constant).
    constexpr std::size_t budgetCapacity = 4 * 1024;
    auto reference =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian);
    ASSERT_TRUE(reference.has_value());
    ASSERT_LT(budgetCapacity, LightEstimate(reference->DeviceFootprint()));

    auto budget = qcx::memory::DeviceWorkspaceBudget::Create(0, 2 * budgetCapacity, budgetCapacity);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    ASSERT_EQ(budget->Remaining(), budgetCapacity);

    int deviceOrdinal = 0;
    ASSERT_EQ(cudaGetDevice(&deviceOrdinal), cudaSuccess);
    // Clear any error residue from the settle cycle before the Create.
    (void)cudaGetLastError();

    qcx::integrals::FockBuildOptions refused;
    refused.deviceWorkspaceBudget = &*budget;
    auto builder =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian, refused);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(builder.error().message.find("device " + std::to_string(deviceOrdinal)),
              std::string::npos)
        << builder.error().message;
    EXPECT_NE(builder.error().message.find("light rung"), std::string::npos)
        << builder.error().message;
    EXPECT_NE(builder.error().message.find("direct screened"), std::string::npos)
        << builder.error().message;
    // The ladder fired before the first device allocation: the runtime's
    // last error is still clean - no CUDA allocation was attempted.
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);

    // The engine level refuses identically (the builder forwards the
    // budget to the engine's Create).
    qcx::integrals::EriCudaOptions engineRefused;
    engineRefused.deviceWorkspaceBudget = &*budget;
    auto engine = qcx::integrals::EriCudaEngine::Create(*molecule, *basisSet, engineRefused);
    ASSERT_FALSE(engine.has_value());
    EXPECT_EQ(engine.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(engine.error().message.find("device " + std::to_string(deviceOrdinal)),
              std::string::npos)
        << engine.error().message;
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(GpuModeSelectionTest, ForcedKnobStillRefusesWhenTheLightRungCannotFit) {
    // The forced-LightPath knob cannot bypass the refusal: a forced light
    // rung under a budget that cannot hold it is a caller contradiction,
    // refused at Create with the ladder named - never deferred to the
    // per-call uploads, where the allocation would fail after the caller's
    // work had already committed.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    if (qcx::backend::DetectDevices().empty())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    SettleContextAndPrepayQuantum();

    constexpr std::size_t budgetCapacity = 4 * 1024;
    auto budget = qcx::memory::DeviceWorkspaceBudget::Create(0, 2 * budgetCapacity, budgetCapacity);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;

    qcx::integrals::FockBuildOptions refused;
    refused.forceGpuLightPath = true;
    refused.deviceWorkspaceBudget = &*budget;
    auto builder =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian, refused);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(builder.error().message.find("light rung"), std::string::npos)
        << builder.error().message;
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(GpuModeSelectionTest, LightPathBuildFockMatchesFastPath) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    const std::size_t n = coreHamiltonian->Shape()[0];
    const auto density = ToTensor(PhysicalDensity(n));
    ASSERT_TRUE(density.has_value());

    // The same options in both modes; the only difference is the mode
    // decision (the knob). The screening, the batches and the kernels run
    // identically.
    qcx::integrals::FockBuildOptions fastOptions;
    qcx::integrals::FockBuildOptions lightOptions;
    lightOptions.forceGpuLightPath = true;

    auto fast = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, fastOptions);

    if (!fast.has_value() && fast.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << fast.error().message;
    }

    ASSERT_TRUE(fast.has_value());
    auto light = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, lightOptions);
    ASSERT_TRUE(light.has_value());
    ASSERT_EQ(light->Mode(), qcx::integrals::FockBuildMode::kLightPath);

    auto fastFock = fast->BuildFock(*density);
    ASSERT_TRUE(fastFock.has_value()) << fastFock.error().message;
    auto lightFock = light->BuildFock(*density);
    ASSERT_TRUE(lightFock.has_value()) << lightFock.error().message;

    // The fp64 last-bit budget of the GPU parity pins (gpu_fock_build_test
    // line ~152, 1e-10): the contraction accumulates cross-quartet with
    // atomicAdd, whose block scheduling order is not deterministic between
    // two launches, so the last bit can differ between the fast and the
    // light calls even with identical inputs. Anything beyond the last-bit
    // budget is a mode-switch effect.
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            EXPECT_NEAR((*fastFock)(i, j), (*lightFock)(i, j), 1e-10)
                << "Fock mismatch at (" << i << ", " << j << ") - the mode switch "
                << "changed the results";
        }
    }
}

TEST(GpuModeSelectionTest, LightPathEngineBatchMatchesFastPath) {
    // The engine per-call paths (ComputeBatch and ComputeBatchCertified)
    // read the statics through the active copy: the mode switch must be
    // invisible there too, values and certified bounds included.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeH2oSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());

    // Every canonical pair, all (bra, ket) combinations: 15 pairs on
    // H2O/STO-3G, 225 quartets through the s/p classes.
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basisSet);
    ASSERT_TRUE(pairList.has_value());
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (std::size_t a = 0; a < pairList->pairs.size(); ++a)
    {
        for (std::size_t b = 0; b < pairList->pairs.size(); ++b)
        {
            const qcx::integrals::ShellPairIndex& bra = pairList->pairs[a];
            const qcx::integrals::ShellPairIndex& ket = pairList->pairs[b];
            quartets.push_back(qcx::integrals::ShellQuartet{bra.i, bra.j, ket.i, ket.j});
        }
    }

    auto fast = qcx::integrals::EriCudaEngine::Create(*molecule, *basisSet);

    if (!fast.has_value() && fast.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << fast.error().message;
    }

    ASSERT_TRUE(fast.has_value());
    qcx::integrals::EriCudaOptions forced;
    forced.forceLightPath = true;
    auto light = qcx::integrals::EriCudaEngine::Create(*molecule, *basisSet, forced);
    ASSERT_TRUE(light.has_value());
    ASSERT_EQ(light->Mode(), qcx::integrals::FockBuildMode::kLightPath);

    auto fastBatch = fast->ComputeBatch(quartets);
    ASSERT_TRUE(fastBatch.has_value()) << fastBatch.error().message;
    auto lightBatch = light->ComputeBatch(quartets);
    ASSERT_TRUE(lightBatch.has_value()) << lightBatch.error().message;
    ASSERT_EQ(lightBatch->values.size(), fastBatch->values.size());

    for (std::size_t t = 0; t < fastBatch->values.size(); ++t)
    {
        EXPECT_EQ(fastBatch->values[t], lightBatch->values[t]) << "fp64 batch mismatch at " << t;
    }

    auto fastCertified = fast->ComputeBatchCertified(quartets);
    ASSERT_TRUE(fastCertified.has_value()) << fastCertified.error().message;
    auto lightCertified = light->ComputeBatchCertified(quartets);
    ASSERT_TRUE(lightCertified.has_value()) << lightCertified.error().message;
    ASSERT_EQ(lightCertified->values.size(), fastCertified->values.size());
    ASSERT_EQ(lightCertified->errorBounds.size(), fastCertified->errorBounds.size());

    for (std::size_t t = 0; t < fastCertified->values.size(); ++t)
    {
        EXPECT_EQ(fastCertified->values[t], lightCertified->values[t])
            << "certified fp32 value mismatch at " << t;
    }

    for (std::size_t t = 0; t < fastCertified->errorBounds.size(); ++t)
    {
        EXPECT_EQ(fastCertified->errorBounds[t], lightCertified->errorBounds[t])
            << "certified bound mismatch at " << t;
    }
}

TEST(GpuModeSelectionTest, LightPathCommitsAndReleasesTheStaticsPerCall) {
    // The residency contract on the C12H26/STO-3G fixture (the statics are
    // 12.6 MB there - far above the 2 MiB mapping granule, so the per-call
    // commit and the post-call release are both observable). The per-call
    // commit (peak minus the create baseline) is the state-free slice of
    // the create gate: the suite's prior pool state does not affect it.
    // The fast rung's call commits the working set only; the light rung's
    // call commits the working set PLUS the statics (the per-call upload),
    // and its post-call residency returns to the baseline within one
    // granule (the statics are released at the call end - the retained-
    // vs-per-call residency contract).
    if (qcx::backend::DetectDevices().empty())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    SettleContextAndPrepayQuantum();

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    const std::size_t n = coreHamiltonian->Shape()[0];
    const auto density = ToTensor(PhysicalDensity(n));
    ASSERT_TRUE(density.has_value());

    const std::size_t freeBefore = FreeVramBytes();
    ASSERT_GT(freeBefore, kDisplayHeadroomBytes);

    qcx::integrals::FockBuildOptions lightOptions;
    lightOptions.forceGpuLightPath = true;

    auto fast = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basisSet, *coreHamiltonian);

    if (!fast.has_value() && fast.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << fast.error().message;
    }

    ASSERT_TRUE(fast.has_value());
    const qcx::integrals::GpuDeviceFootprint fastFootprint = fast->DeviceFootprint();
    const std::size_t staticsBytes = fastFootprint.tablesBytes + fastFootprint.boysBytes;
    ASSERT_GT(staticsBytes, kMappingGranuleBytes);

    // The peak of one call, polled (the per-call buffers are freed at the
    // call's return, so a post-call probe alone would miss the peak). The
    // baseline is probed right before the call (after Create, so the
    // create's commit - if any - is already in it).
    auto runWithPeakPolling = [&](const qcx::integrals::GpuJkFockBuilder& builder,
                                  const CpuTensor2& densityIn,
                                  std::size_t* commitOut,
                                  std::size_t* postCallFreeOut,
                                  std::size_t* baselineOut) {
        const std::size_t baseline = FreeVramBytes();
        std::atomic<bool> done{false};
        std::atomic<std::size_t> minFree{baseline};
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

        auto fock = builder.BuildFock(densityIn);
        done.store(true, std::memory_order_relaxed);
        poll.join();
        ASSERT_TRUE(fock.has_value());
        *commitOut = baseline - minFree.load(std::memory_order_relaxed);
        *postCallFreeOut = FreeVramBytes();
        *baselineOut = baseline;
    };

    std::size_t fastCommit = 0;
    std::size_t fastPostCallFree = 0;
    std::size_t fastBaseline = 0;
    runWithPeakPolling(*fast, *density, &fastCommit, &fastPostCallFree, &fastBaseline);

    auto light = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, lightOptions);
    ASSERT_TRUE(light.has_value());
    ASSERT_EQ(light->Mode(), qcx::integrals::FockBuildMode::kLightPath);

    std::size_t lightCommit = 0;
    std::size_t lightPostCallFree = 0;
    std::size_t lightBaseline = 0;
    runWithPeakPolling(*light, *density, &lightCommit, &lightPostCallFree, &lightBaseline);

    const std::size_t lightEstimate = LightEstimate(fastFootprint);

    std::cout << "[C12H26/STO-3G] n = " << n << ": statics " << staticsBytes << " B, light "
              << "estimate " << lightEstimate << " B; fast per-call commit " << fastCommit
              << " B, light per-call commit " << lightCommit << " B; fast post-call free "
              << fastPostCallFree << " B, light post-call free " << lightPostCallFree << " B"
              << std::endl;

    // The fast rung's per-call commit is the working set only (the
    // per-call law, re-asserted here).
    EXPECT_LE(static_cast<double>(fastCommit), 1.10 * static_cast<double>(lightEstimate));

    // The light rung's per-call commit carries the statics: it must clear
    // the fast rung's commit by the table bytes, within one mapping granule
    // (the state-free distinguishing slice).
    EXPECT_GE(lightCommit, fastCommit + staticsBytes - kMappingGranuleBytes);

    // The light rung's per-call commit stays bounded by the working set
    // plus the statics (nothing else leaks into the call).
    EXPECT_LE(static_cast<double>(lightCommit),
              1.10 * static_cast<double>(lightEstimate + staticsBytes));

    // The post-call release: the light rung returns to its own baseline
    // within one granule - the statics are released at the call end, so the
    // light rung's residency outside a call is exactly the per-call working
    // set (released too), not the retained tables.
    EXPECT_GE(lightPostCallFree, lightBaseline - kMappingGranuleBytes);
}

TEST(GpuModeSelectionTest, PerCallTrafficIsDensityFockAndScreenedListsOnly) {
    // The residency contract:
    // per-iteration bus traffic = density up, Fock up/down, the screened
    // lists - and nothing else, at any system size. The accounting is exact
    // by construction (every byte count is the buffer size at the transfer
    // site), so the C12 fixture's numbers are deterministic: no VRAM
    // polling, no mapping granules - the gate reads the builder's own
    // record of the last call.
    if (qcx::backend::DetectDevices().empty())
    {
        GTEST_SKIP() << "no CUDA device";
    }

    SettleContextAndPrepayQuantum();

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value());
    auto basisSet = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basisSet.has_value());
    auto coreHamiltonian = BuildCoreHamiltonian(*molecule, *basisSet);
    ASSERT_TRUE(coreHamiltonian.has_value());

    const std::size_t n = coreHamiltonian->Shape()[0];
    const std::size_t matrixBytes = n * n * sizeof(double);
    const auto density = ToTensor(PhysicalDensity(n));
    ASSERT_TRUE(density.has_value());

    qcx::integrals::FockBuildOptions fastOptions;
    qcx::integrals::FockBuildOptions lightOptions;
    lightOptions.forceGpuLightPath = true;

    auto fast = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, fastOptions);

    if (!fast.has_value() && fast.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << fast.error().message;
    }

    ASSERT_TRUE(fast.has_value());
    const qcx::integrals::GpuDeviceFootprint footprint = fast->DeviceFootprint();
    const std::size_t staticsBytes = footprint.tablesBytes + footprint.boysBytes;

    double certifiedSum = 0.0;
    auto fastFock = fast->BuildFock(*density, &certifiedSum);
    ASSERT_TRUE(fastFock.has_value()) << fastFock.error().message;
    const qcx::integrals::EriCudaTransferAccounting fastAcc = fast->TransferAccounting();

    // The certified lane provably runs at kNormal (MixedPrecisionThreshold
    // 1e-10): the far-apart terminal pairs of the C12 chain
    // carry density-weighted certified bounds far below the gate, so the
    // fp32 lane routes quartets and the bounds read-back is provably
    // non-empty - a missing bounds increment would read 0 here (the kTight
    // run below pins the lane-off shape separately).
    ASSERT_GT(certifiedSum, 0.0);

    // The matrix classes: exactly n^2 doubles per direction (the per-call
    // fresh buffers sync once).
    EXPECT_EQ(fastAcc.densityUploadBytes, matrixBytes);
    EXPECT_EQ(fastAcc.fockUploadBytes, matrixBytes);
    EXPECT_EQ(fastAcc.fockDownloadBytes, matrixBytes);

    // The screened lists: non-empty (C12 screens to thousands of quartets).
    EXPECT_GT(fastAcc.listUploadBytes, 0u);

    // The certified lane's per-quartet bound-sum read-back.
    EXPECT_GT(fastAcc.boundsDownloadBytes, 0u);

    // The statics: 0 per call on the FastPath - uploaded once at Create,
    // the retained copy (the per-iteration contract: statics move once per
    // run, never per iteration).
    EXPECT_EQ(fastAcc.staticsUploadBytes, 0u);

    // The Total identity: the sum of the named classes.
    EXPECT_EQ(fastAcc.Total(),
              fastAcc.densityUploadBytes + fastAcc.fockUploadBytes + fastAcc.fockDownloadBytes +
                  fastAcc.listUploadBytes + fastAcc.boundsDownloadBytes +
                  fastAcc.staticsUploadBytes);

    // The LightPath: the same screening, so the same lists; the statics
    // upload once per call (the per-call statics class - the retained copy
    // is released at the call end).
    auto light = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, lightOptions);
    ASSERT_TRUE(light.has_value());
    ASSERT_EQ(light->Mode(), qcx::integrals::FockBuildMode::kLightPath);

    double lightCertifiedSum = 0.0;
    auto lightFock = light->BuildFock(*density, &lightCertifiedSum);
    ASSERT_TRUE(lightFock.has_value()) << lightFock.error().message;
    const qcx::integrals::EriCudaTransferAccounting lightAcc = light->TransferAccounting();

    EXPECT_EQ(lightAcc.densityUploadBytes, matrixBytes);
    EXPECT_EQ(lightAcc.fockUploadBytes, matrixBytes);
    EXPECT_EQ(lightAcc.fockDownloadBytes, matrixBytes);
    EXPECT_EQ(lightAcc.listUploadBytes, fastAcc.listUploadBytes);
    EXPECT_EQ(lightAcc.boundsDownloadBytes, fastAcc.boundsDownloadBytes);
    EXPECT_EQ(lightAcc.staticsUploadBytes, staticsBytes);

    // The screened-lists-only pin: a looser preset screens strictly fewer
    // quartets (DensityThreshold 1e-8 vs 1e-10), so the lists shrink
    // strictly - the only class that varies with the screening, at any
    // system size.
    qcx::integrals::FockBuildOptions looseOptions;
    looseOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    auto loose = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, looseOptions);
    ASSERT_TRUE(loose.has_value());
    auto looseFock = loose->BuildFock(*density);
    ASSERT_TRUE(looseFock.has_value()) << looseFock.error().message;
    const qcx::integrals::EriCudaTransferAccounting looseAcc = loose->TransferAccounting();
    ASSERT_LT(looseAcc.listUploadBytes, fastAcc.listUploadBytes);

    // The lane-off shape: kTight disables the certified lane
    // (MixedPrecisionThreshold 0.0), so the bounds class is exactly 0 and
    // the certified sum stays 0 - the accounting's lane-off statement.
    qcx::integrals::FockBuildOptions tightOptions;
    tightOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto tight = qcx::integrals::GpuJkFockBuilder::Create(
        *molecule, *basisSet, *coreHamiltonian, tightOptions);
    ASSERT_TRUE(tight.has_value());
    double tightCertifiedSum = 0.0;
    auto tightFock = tight->BuildFock(*density, &tightCertifiedSum);
    ASSERT_TRUE(tightFock.has_value()) << tightFock.error().message;
    const qcx::integrals::EriCudaTransferAccounting tightAcc = tight->TransferAccounting();
    EXPECT_EQ(tightAcc.boundsDownloadBytes, 0u);
    EXPECT_EQ(tightCertifiedSum, 0.0);

    std::cout << "[C12H26/STO-3G] n = " << n << " (matrix " << matrixBytes
              << " B per direction, statics " << staticsBytes << " B):" << std::endl
              << "  fast:  density " << fastAcc.densityUploadBytes << " B, fock up "
              << fastAcc.fockUploadBytes << " B, fock down " << fastAcc.fockDownloadBytes
              << " B, lists " << fastAcc.listUploadBytes << " B, bounds "
              << fastAcc.boundsDownloadBytes << " B, statics " << fastAcc.staticsUploadBytes
              << " B, total " << fastAcc.Total() << " B" << std::endl
              << "  light: lists " << lightAcc.listUploadBytes << " B, bounds "
              << lightAcc.boundsDownloadBytes << " B, statics " << lightAcc.staticsUploadBytes
              << " B, total " << lightAcc.Total() << " B" << std::endl
              << "  loose: lists " << looseAcc.listUploadBytes << " B (screened only)" << std::endl
              << "  tight: bounds " << tightAcc.boundsDownloadBytes << " B (lane off)" << std::endl;
}
