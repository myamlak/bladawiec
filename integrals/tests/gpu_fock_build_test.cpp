// The GPU Fock builder tests:
// the GpuJkFockBuilder (gpu_fock_build.hpp) against the
// DirectJkFockBuilder parity gate - the same screened list, the same
// contraction formula, the same certified-bound bookkeeping - and the RHF
// pins wired through RunRhfScf with the GPU builder as the Fock source.
// The device tests self-skip without a CUDA device (the engine's Create
// probes the device count - the RequireCudaDevice convention of
// eri_cuda_test.cpp). The parity loops run the GPU build several times:
// the atomicAdd accumulations make the device sums order-dependent at the
// last bits, so every run must land inside the budget, not just one.

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_cuda.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/scf/rhf.hpp"
#include "shellset_fixture.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeHfSto3g;
using qcx::testing::MakeHfSto3gBasis;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines (the direct_rhf_test.cpp helper,
// test-local).
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

// The self-skip convention of the CUDA suite: no device -> the builders
// report kDeviceError from Create and the test skips instead of failing.
void RequireCudaDevice(const qcx::Result<qcx::integrals::GpuJkFockBuilder>& builder) {
    if (!builder.has_value() && builder.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << builder.error().message;
    }
}

// The H2 GPU-vs-CPU parity harness: the SAME builder options on both sides
// (the shared ScreenAll path then defines the same screened list), a modest
// symmetric density that clears the density gate at kNormal, and several
// GPU runs - the atomicAdd accumulation order varies per launch, so every
// run must stay inside the budget (the run-multiple-times gate).
void H2ParityAgainstCpuWith(qcx::integrals::FockBuildOptions options,
                            double* cpuBoundSumOut = nullptr,
                            double* gpuBoundSumOut = nullptr) {
    // The GPU parity pins run the per-element screening flag OFF:
    // the device kernel predates the per-element re-filter,
    // so a flag-on CPU side would filter what the GPU side cannot - the
    // bit-parity budget then holds only if both sides run the legacy
    // single-level screen.
    options.usePerElementScreening = false;
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto cpuBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(cpuBuilder.has_value()) << cpuBuilder.error().message;

    auto gpuBuilder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);
    RequireCudaDevice(gpuBuilder);
    ASSERT_TRUE(gpuBuilder.has_value()) << gpuBuilder.error().message;

    // A modest symmetric density: at kNormal every H2 quartet clears the
    // density gate, so the parity compares the full screened list on both
    // sides (the same comment as the DirectRhfTest H2 pin).
    auto density = ToTensor(0.5 * Eigen::MatrixXd::Identity(2, 2));
    ASSERT_TRUE(density.has_value()) << density.error().message;

    auto cpuFock = cpuBuilder->BuildFock(*density, cpuBoundSumOut);
    ASSERT_TRUE(cpuFock.has_value()) << cpuFock.error().message;
    const Eigen::MatrixXd cpuMatrix = ToMatrix(*cpuFock);

    for (int run = 0; run < 3; ++run)
    {
        auto gpuFock = gpuBuilder->BuildFock(*density, gpuBoundSumOut);
        ASSERT_TRUE(gpuFock.has_value()) << gpuFock.error().message;
        const Eigen::MatrixXd gpuMatrix = ToMatrix(*gpuFock);

        EXPECT_LE((gpuMatrix - cpuMatrix).cwiseAbs().maxCoeff(), 1e-10)
            << "GPU Fock deviates from the CPU Fock beyond the parity budget (run " << run << ")";
    }
}

// The FockBuilderFn adapter for the GPU builder (the direct_rhf_test.cpp
// adapter pattern): the seam hands over the spin-summed D = 2 rho, the
// builder contracts the spatial density rho = D/2 (rhf.hpp documents the
// seam convention).
qcx::scf::FockBuilderFn MakeGpuFockBuilder(const qcx::integrals::GpuJkFockBuilder& builder) {
    return [builder](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto densityTensor = ToTensor(0.5 * density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        auto fock = builder.BuildFock(*densityTensor);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        return ToMatrix(*fock);
    };
}

qcx::Result<qcx::scf::HfResult> RunGpuRhf(const qcx::molecule::Molecule& molecule,
                                          const qcx::basisset::BasisSet& basisSet,
                                          const qcx::integrals::FockBuildOptions& fockOptions,
                                          const qcx::scf::RhfOptions& scfOptions) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto core = BuildCoreHamiltonian(molecule, basisSet);

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    auto builder = qcx::integrals::GpuJkFockBuilder::Create(molecule, basisSet, *core, fockOptions);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    return qcx::scf::RunRhfScf(
        molecule, ToMatrix(*overlap), ToMatrix(*core), scfOptions, MakeGpuFockBuilder(*builder));
}

} // namespace

TEST(GpuFockBuildTest, H2ParityAgainstCpu) {
    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    double cpuBoundSum = 0.0;
    double gpuBoundSum = 0.0;
    H2ParityAgainstCpuWith(options, &cpuBoundSum, &gpuBoundSum);

    // kNormal keeps the H2 quartets above the certified gate: both lanes are
    // all-fp64 here, both certified sums are zero (the H2 pin comment).
    EXPECT_DOUBLE_EQ(cpuBoundSum, 0.0);
    EXPECT_DOUBLE_EQ(gpuBoundSum, 0.0);
}

TEST(GpuFockBuildTest, ExchangeOnlyParityAgainstCpu) {
    // The exchange-only split mode: only the K phases run. The
    // device kernel's buildExchangeOnly branch (eri_cuda_fock.cu) is
    // otherwise unreachable - every other test builds the full J+K Fock -
    // so the split mode needs its own parity gate against the CPU builder
    // with the same option.
    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.buildExchangeOnly = true;
    H2ParityAgainstCpuWith(options);
}

TEST(GpuFockBuildTest, CoulombOnlyParityAgainstCpu) {
    // The J-only split mode: only the J phases run. Same rationale:
    // the device buildCoulombOnly branch needs its own parity gate.
    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.buildCoulombOnly = true;
    H2ParityAgainstCpuWith(options);
}

// The GPU-builder error paths (the FockBuildTest.ErrorPaths mirror): the
// Create-time guards (zero batch cap, wrong-shaped core) and the BuildFock
// density-shape guard. The wrong-shape tensors are built DIRECTLY from the
// Tensor factory - never through qcx::testing::ToTensor (the 5cf0231
// lesson: that helper's tensor builder is square-only).
TEST(GpuFockBuildTest, ErrorPaths) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // A zero batch cap is rejected at Create - the guard runs BEFORE any
    // device work, so this path is testable without a CUDA device.
    qcx::integrals::FockBuildOptions zeroCap;
    zeroCap.maxBatchBytes = 0;
    auto zeroCapBuilder =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, zeroCap);
    EXPECT_FALSE(zeroCapBuilder.has_value());
    EXPECT_EQ(zeroCapBuilder.error().code, qcx::ErrorCode::kInvalidArgument);

    // A wrong-shaped core tensor is rejected at Create (after the engine
    // create: no device means the engine reports kDeviceError first, and
    // the test self-skips like the other device tests).
    qcx::integrals::FockBuildOptions options;
    auto wrongCore = CpuTensor2::Create({1, 1});
    ASSERT_TRUE(wrongCore.has_value()) << wrongCore.error().message;
    auto wrongCoreBuilder =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *wrongCore, options);
    RequireCudaDevice(wrongCoreBuilder);
    EXPECT_FALSE(wrongCoreBuilder.has_value());
    EXPECT_EQ(wrongCoreBuilder.error().code, qcx::ErrorCode::kInvalidArgument);

    // A wrong-shaped density is rejected at BuildFock.
    auto builder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);
    RequireCudaDevice(builder);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    auto wrongShape = CpuTensor2::Create({1, 1});
    ASSERT_TRUE(wrongShape.has_value()) << wrongShape.error().message;
    auto wrongFock = builder->BuildFock(*wrongShape);
    EXPECT_FALSE(wrongFock.has_value());
    EXPECT_EQ(wrongFock.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The GPU side cannot filter per-element (the device kernel predates the
// re-filter), so the parity pins run the flag
// off on BOTH sides; this test instead runs the GPU on the legacy screen
// and the CPU on the full density-gated screen (product gate + per-element
// filter) and bounds the deviation: each element the CPU drops carries at
// most 2*tau to a touched Fock element (the recorded bound), and each
// quartet the CPU's product gate drops that the GPU contracted has its
// elements bounded by the six-block keep condition times the block-max
// ratios of the physical density (~2-3 on this fixture - no near-zero
// blocks) - so tau * (total canonical quartet elements) is a conservative
// envelope for the whole deviation at kLoose. The certified lane is off
// (fp64-only both sides) so the residual is exactly the screen difference,
// not the lane's delivered error. The GPU side runs 3 times (the
// atomicAdd-reduction convention: every run must stay inside the budget).
TEST(GpuFockBuildTest, GpuNoFilterVsCpuFilterBounded) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions cpuOptions;
    cpuOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    cpuOptions.useDensityScreening = true;
    cpuOptions.useCertifiedMixedPrecision = false;
    cpuOptions.usePerElementScreening = true;
    auto cpuBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, cpuOptions);
    ASSERT_TRUE(cpuBuilder.has_value()) << cpuBuilder.error().message;

    qcx::integrals::FockBuildOptions gpuOptions = cpuOptions;
    gpuOptions.usePerElementScreening = false; // legacy screen (see H2ParityAgainstCpuWith).
    auto gpuBuilder =
        qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, gpuOptions);
    RequireCudaDevice(gpuBuilder);
    ASSERT_TRUE(gpuBuilder.has_value()) << gpuBuilder.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The conservative envelope: tau times the total canonical quartet
    // element count (an upper bound on the elements either side could have
    // contracted).
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    std::size_t totalElements = 0;
    const std::size_t nShells = pairList->shells.size();

    for (std::size_t a = 0; a < nShells; ++a)
    {
        for (std::size_t b = a; b < nShells; ++b)
        {
            for (std::size_t c = 0; c < nShells; ++c)
            {
                for (std::size_t d = c; d < nShells; ++d)
                {
                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    totalElements += ShellFunctionCount(pairList->shells[a]) *
                                     ShellFunctionCount(pairList->shells[b]) *
                                     ShellFunctionCount(pairList->shells[c]) *
                                     ShellFunctionCount(pairList->shells[d]);
                }
            }
        }
    }

    const double tau = qcx::integrals::DensityThreshold(qcx::integrals::AccuracyPreset::kLoose);
    const double budget = tau * static_cast<double>(totalElements);

    auto cpuFock = cpuBuilder->BuildFock(*densityTensor);
    ASSERT_TRUE(cpuFock.has_value()) << cpuFock.error().message;
    const Eigen::MatrixXd cpuMatrix = ToMatrix(*cpuFock);

    for (int run = 0; run < 3; ++run)
    {
        auto gpuFock = gpuBuilder->BuildFock(*densityTensor);
        ASSERT_TRUE(gpuFock.has_value()) << gpuFock.error().message;
        const Eigen::MatrixXd gpuMatrix = ToMatrix(*gpuFock);

        EXPECT_LE((gpuMatrix - cpuMatrix).cwiseAbs().maxCoeff(), budget)
            << "GPU (legacy screen) deviates from the CPU (the two-level density screen) beyond "
               "tau*"
               "totalElements (tau "
            << tau << ", totalElements " << totalElements << ", budget " << budget << ", run "
            << run << ")";
    }
}

TEST(GpuFockBuildTest, H2oCertifiedLaneParity) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    options.useDensityScreening = true;
    options.useCertifiedMixedPrecision = true;
    // The GPU parity pins run the per-element screening flag off (the
    // device kernel predates the re-filter - see H2ParityAgainstCpuWith).
    options.usePerElementScreening = false;

    auto cpuBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(cpuBuilder.has_value()) << cpuBuilder.error().message;

    auto gpuBuilder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);
    RequireCudaDevice(gpuBuilder);
    ASSERT_TRUE(gpuBuilder.has_value()) << gpuBuilder.error().message;

    // A physical density: converge the plain (no-DIIS) kLoose loop on the
    // CPU and capture the last spin-summed iterate (D = 2 rho), like the
    // MixedLaneEnergyBoundDominates test's loop.
    qcx::scf::RhfOptions scfOptions;
    scfOptions.useDiis = false;
    Eigen::MatrixXd capturedDensity;
    qcx::scf::FockBuilderFn capturing = [&](const Eigen::MatrixXd& density) {
        capturedDensity = density;
        auto densityTensor = ToTensor(0.5 * density);

        if (!densityTensor.has_value())
        {
            return qcx::Result<Eigen::MatrixXd>{std::unexpected(densityTensor.error())};
        }

        auto fock = cpuBuilder->BuildFock(*densityTensor);

        if (!fock.has_value())
        {
            return qcx::Result<Eigen::MatrixXd>{std::unexpected(fock.error())};
        }

        return qcx::Result<Eigen::MatrixXd>{ToMatrix(*fock)};
    };
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto run =
        qcx::scf::RunRhfScf(*molecule, ToMatrix(*overlap), ToMatrix(*core), scfOptions, capturing);
    ASSERT_TRUE(run.has_value()) << run.error().message;
    ASSERT_TRUE(run->converged) << "iterations: " << run->iterations;
    ASSERT_GT(capturedDensity.size(), 0);

    auto density = ToTensor(0.5 * capturedDensity);
    ASSERT_TRUE(density.has_value()) << density.error().message;

    double cpuBoundSum = 0.0;
    double gpuBoundSum = 0.0;
    auto cpuFock = cpuBuilder->BuildFock(*density, &cpuBoundSum);
    ASSERT_TRUE(cpuFock.has_value()) << cpuFock.error().message;
    const Eigen::MatrixXd cpuMatrix = ToMatrix(*cpuFock);

    EXPECT_GT(cpuBoundSum, 0.0); // the s-pair quartets must route to the lane
    auto gpuFock = gpuBuilder->BuildFock(*density, &gpuBoundSum);
    ASSERT_TRUE(gpuFock.has_value()) << gpuFock.error().message;
    const Eigen::MatrixXd gpuMatrix = ToMatrix(*gpuFock);
    EXPECT_GT(gpuBoundSum, 0.0);

    // The certified sums: the same routing, the same bound formula, the
    // same accumulation order - the device and CPU kernels multiply the
    // same factors, so the sums agree to the last few bits (NEAR with a
    // relative budget instead of DOUBLE_EQ: the per-quartet products may
    // be ordered differently inside the device bound finalize).
    EXPECT_NEAR(gpuBoundSum, cpuBoundSum, 1e-9 * std::max(1.0, std::abs(cpuBoundSum)));

    // The Fock parity: every fp32-lane element error is bounded by its
    // side's certified sum, so the two sides differ by at most the sum of
    // both budgets (plus the fp64-pipeline last-bit tolerance).
    const double budget = 2.0 * std::max(cpuBoundSum, gpuBoundSum) + 1e-10;
    EXPECT_LE((gpuMatrix - cpuMatrix).cwiseAbs().maxCoeff(), budget)
        << "GPU Fock deviates beyond the certified budget (cpuBoundSum " << cpuBoundSum
        << ", gpuBoundSum " << gpuBoundSum << ")";
}

TEST(GpuFockBuildTest, H2RhfPin) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    fockOptions.usePerElementScreening =
        false; // GPU pin: the legacy screen (see H2ParityAgainstCpuWith).
    auto result = RunGpuRhf(*molecule, *basis, fockOptions, qcx::scf::RhfOptions{});
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    EXPECT_NEAR(result->totalEnergy, -1.1167143252, 1e-8);
}

TEST(GpuFockBuildTest, HfRhfPin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kTight disables the fp32 lane, so the GPU path reproduces the fp64
    // dense pin like the direct path.
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    fockOptions.usePerElementScreening =
        false; // GPU pin: the legacy screen (see H2ParityAgainstCpuWith).
    auto result = RunGpuRhf(*molecule, *basis, fockOptions, qcx::scf::RhfOptions{});
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    EXPECT_NEAR(result->totalEnergy, -98.57075766, 1e-5);
}

TEST(GpuFockBuildTest, H2oRhfPin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    fockOptions.usePerElementScreening =
        false; // GPU pin: the legacy screen (see H2ParityAgainstCpuWith).
    auto result = RunGpuRhf(*molecule, *basis, fockOptions, qcx::scf::RhfOptions{});
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged) << "iterations: " << result->iterations;
    EXPECT_NEAR(result->totalEnergy, -74.96292827, 1e-5);
}

// The device-path sweep: the certified
// bound walk of certified_bound_sweep_test.cpp over the device pipeline - the
// same canonical quartet list through EriCudaEngine::ComputeBatch (fp64) and
// ComputeBatchCertified (fp32), asserting the a-priori bound dominates the
// measured deviation, and the GPU-fp32-vs-CPU-fp64 cross-check (catches a
// device-specific systematic bias the GPU-vs-CPU fp64 parity alone would
// not surface). The CPU lane here is the host reference for the fp32 lane,
// the gate this sweep exists to run.
void RequireCudaEngine(const qcx::Result<qcx::integrals::EriCudaEngine>& engine) {
    if (!engine.has_value() && engine.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << engine.error().message;
    }
}

void SweepFixtureGpu(const qcx::molecule::Molecule& molecule,
                     const qcx::basisset::BasisSet& basisSet) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<qcx::integrals::ShellQuartet> quartets;
    const std::size_t nShells = pairList->shells.size();

    for (std::size_t a = 0; a < nShells; ++a)
    {
        for (std::size_t b = a; b < nShells; ++b)
        {
            for (std::size_t c = 0; c < nShells; ++c)
            {
                for (std::size_t d = c; d < nShells; ++d)
                {
                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    auto engine = qcx::integrals::EriCudaEngine::Create(molecule, basisSet);
    RequireCudaEngine(engine);
    ASSERT_TRUE(engine.has_value()) << engine.error().message;

    auto fp64Gpu = engine->ComputeBatch(quartets);
    ASSERT_TRUE(fp64Gpu.has_value()) << fp64Gpu.error().message;
    auto fp32Gpu = engine->ComputeBatchCertified(quartets);
    ASSERT_TRUE(fp32Gpu.has_value()) << fp32Gpu.error().message;
    auto fp64Cpu = qcx::integrals::ComputeEriBatch(molecule, basisSet, quartets);
    ASSERT_TRUE(fp64Cpu.has_value()) << fp64Cpu.error().message;

    // Both pipelines canonicalize with the identical AssembleClassBatches
    // pass, so the computed lists align positionally (the CPU sweep relies
    // on the same alignment).
    ASSERT_EQ(fp64Gpu->computed.size(), fp64Cpu->computed.size());
    ASSERT_EQ(fp32Gpu->computed.size(), fp64Gpu->computed.size());
    ASSERT_EQ(fp32Gpu->errorBounds.size(), fp32Gpu->computed.size());

    std::size_t offsetGpu = 0;
    std::size_t offsetCpu = 0;

    for (std::size_t t = 0; t < fp64Gpu->computed.size(); ++t)
    {
        const qcx::integrals::ShellQuartet& quartet = fp64Gpu->computed[t];
        const std::size_t nI = ShellFunctionCount(pairList->shells[quartet.i]);
        const std::size_t nJ = ShellFunctionCount(pairList->shells[quartet.j]);
        const std::size_t nK = ShellFunctionCount(pairList->shells[quartet.k]);
        const std::size_t nL = ShellFunctionCount(pairList->shells[quartet.l]);
        const std::size_t blockSize = nI * nJ * nK * nL;
        const double bound = fp32Gpu->errorBounds[t];
        EXPECT_GT(bound, 0.0) << "quartet " << t << " has no certified bound";

        double maxDeviationVsGpu = 0.0;
        double maxDeviationVsCpu = 0.0;

        for (std::size_t element = 0; element < blockSize; ++element)
        {
            const double fp64GpuValue = fp64Gpu->values[offsetGpu + element];
            const double fp32Value = static_cast<double>(fp32Gpu->values[offsetGpu + element]);
            const double fp64CpuValue = fp64Cpu->values[offsetCpu + element];
            maxDeviationVsGpu = std::max(maxDeviationVsGpu, std::abs(fp32Value - fp64GpuValue));
            maxDeviationVsCpu = std::max(maxDeviationVsCpu, std::abs(fp32Value - fp64CpuValue));
        }

        EXPECT_LE(maxDeviationVsGpu, bound)
            << "quartet " << t << " shells (" << quartet.i << "," << quartet.j << "," << quartet.k
            << "," << quartet.l << "): GPU deviation " << maxDeviationVsGpu
            << " exceeds the certified bound " << bound;
        EXPECT_LE(maxDeviationVsCpu, bound)
            << "quartet " << t << " shells (" << quartet.i << "," << quartet.j << "," << quartet.k
            << "," << quartet.l << "): CPU-reference deviation " << maxDeviationVsCpu
            << " exceeds the certified bound " << bound;

        offsetGpu += blockSize;
        offsetCpu += blockSize;
    }
}

TEST(GpuFockBuildTest, DeviceSweepShellsetBoundsDominate) {
    auto basis = qcx::basisset::ParseNwchemText(qcx::testing::kShellsetBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    SweepFixtureGpu(*molecule, *basis);
}

TEST(GpuFockBuildTest, DeviceSweepHfBoundsDominate) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    SweepFixtureGpu(*molecule, *basis);
}

TEST(GpuFockBuildTest, DeviceSweepH2oBoundsDominate) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    SweepFixtureGpu(*molecule, *basis);
}
