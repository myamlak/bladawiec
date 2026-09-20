// The GPU-gated precision-ladder tests (the ladder gate and the
// device-side smoke): the fp64 cap reproduces the pre-ladder fp64 build
// within the GPU pins' tolerance (the device atomicAdd accumulations are
// order-dependent at the last bits, so the device pins are tolerance pins,
// never bitwise), and the ladder runs on the device with the schedule's
// profile seeded from the device compute profile (the T1000-class device
// has fp16 tensor cores). The suite self-skips without a CUDA device (the
// RequireCudaDevice convention of gpu_fock_build_test.cpp).

#include "h2o_sto3g.hpp"
#include "qcx/backend/gpu_compute_profile.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/precision_policy.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::AccuracyPreset;
using qcx::integrals::EscalationSchedule;
using qcx::integrals::FockBuildOptions;
using qcx::integrals::PrecisionBandProfile;
using qcx::integrals::PrecisionCap;
using qcx::integrals::PrecisionLadderInputs;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V (the gpu_fock_build_test.cpp helper, test-local).
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

void RequireCudaDevice(const qcx::Result<qcx::integrals::GpuJkFockBuilder>& builder) {
    if (!builder.has_value() && builder.error().code == qcx::ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << builder.error().message;
    }
}

TEST(GpuPrecisionLadderTest, Fp64CapMatchesPlainFp64Build) {
    // The ladder gate (GPU half): the fp64 cap routes every batch to the
    // fp64 kernel, reproducing the pre-ladder fp64 build within the GPU
    // pins' parity tolerance (the device atomicAdd order varies per
    // launch, so the device pins are tolerance pins, never bitwise).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    FockBuildOptions options;
    options.accuracy = AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false;
    options.usePerElementScreening = false; // The device-pin convention.

    auto plain = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);
    RequireCudaDevice(plain);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    auto ladder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(ladder.has_value()) << ladder.error().message;

    const std::size_t n = core->Shape()[0];
    const auto density = ToTensor(
        Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
    ASSERT_TRUE(density.has_value()) << density.error().message;

    double plainSum = -1.0;
    auto plainFock = plain->BuildFock(*density, &plainSum);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;

    PrecisionBandProfile profile;
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kFp64);
    const PrecisionLadderInputs inputs{0.0, 0.0, &schedule};

    double ladderSum = -1.0;
    auto ladderFock = ladder->BuildFock(*density, &ladderSum, &inputs);
    ASSERT_TRUE(ladderFock.has_value()) << ladderFock.error().message;

    // The parity tolerance of the GPU pins (1e-12 * scale, the device
    // pow/exp 1-2 ulp convention).
    const Eigen::MatrixXd delta = ToMatrix(*plainFock) - ToMatrix(*ladderFock);
    EXPECT_NEAR(delta.cwiseAbs().maxCoeff(), 0.0, 1e-12) << "fp64 cap vs plain fp64 build";
    EXPECT_DOUBLE_EQ(plainSum, 0.0);
    EXPECT_DOUBLE_EQ(ladderSum, 0.0);
    EXPECT_DOUBLE_EQ(schedule.LastCommittedSum(), 0.0);
}

TEST(GpuPrecisionLadderTest, LadderRunsOnDeviceWithTensorCores) {
    // The device-side smoke: the schedule's band profile seeded from the
    // device compute profile (the driver's one-line composition - the
    // device probe is public), and a ladder build with
    // a large early-iteration density error completes with a non-negative
    // committed sum.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    FockBuildOptions options;
    options.accuracy = AccuracyPreset::kNormal;

    auto builder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);
    RequireCudaDevice(builder);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::size_t n = core->Shape()[0];
    const auto density = ToTensor(
        Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n)));
    ASSERT_TRUE(density.has_value()) << density.error().message;

    // The device-profile composition (the driver's seeding rule): the
    // T1000-class device reports fp16 tensor cores and the measured
    // 1/32-class ratio, so the schedule prices its bands per the device.
    const qcx::backend::GpuComputeProfile device = qcx::backend::DetectGpuComputeProfile(0);
    const PrecisionBandProfile profile =
        qcx::integrals::ProfileForRatio(device.fp32ToFp64Ratio, device.fp16TensorCores);
    EscalationSchedule schedule(AccuracyPreset::kNormal, profile, PrecisionCap::kAuto);
    const PrecisionLadderInputs inputs{1e-3, 0.0, &schedule};

    double kernelSum = -1.0;
    auto fock = builder->BuildFock(*density, &kernelSum, &inputs);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;

    EXPECT_GE(kernelSum, 0.0);
    EXPECT_GE(schedule.LastCommittedSum(), 0.0);
    EXPECT_GT(ToMatrix(*fock).rows(), 0);
}

} // namespace
