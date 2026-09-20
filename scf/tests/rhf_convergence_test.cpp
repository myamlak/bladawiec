// RHF diffuse-basis convergence regression tests:
// the CCCBDB validation rehearsal exposed a real scf-core bug - qcx RHF
// on H2O/aug-cc-pVDZ (41 functions) failed the 100-iteration budget with a
// chaotic DIIS trajectory (-96.66 -> -58.30 -> -93.23 -> -58.38 Eh vs the
// correct -76.041844) while the cc-pVDZ control converged in 22 iterations.
// These tests pin the fixed behavior: both bases converge to the CCCBDB
// references within the claim budget on the direct Fock path (the
// production path the runner uses), at both accuracy presets, with the
// dense supermatrix path's shared loop covered by the direct path (same
// RunScfLoop core, seam_test.cpp pins the builders' equivalence).
#include "fast_test_mode.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines (the direct_rhf_test pattern).
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

// The FockBuilderFn adapter (the direct_rhf_test pattern): the seam hands
// over the spin-summed density D = 2 C_occ C_occ^T, the builder contracts
// the spatial density rho = D/2.
qcx::scf::FockBuilderFn MakeDirectFockBuilder(const qcx::integrals::DirectJkFockBuilder& builder) {
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

// H2O at the CCCBDB HF/aug-cc-pVDZ optimized geometry (angstrom ->
// bohr; the same geometry the runner passes to the driver).
qcx::Result<qcx::molecule::Molecule> MakeH2oAtAugCcpvdzGeometry() {
    constexpr double kAngstromToBohr = qcx::molecule::kAngstromToBohr;
    constexpr double kOZ = 0.1137 * kAngstromToBohr;
    constexpr double kHX = 0.7533 * kAngstromToBohr;
    constexpr double kHZ = -0.4547 * kAngstromToBohr;
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = kOZ;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = kHX;
    (*coordinates)(1, 2) = kHZ;
    (*coordinates)(2, 0) = 0.0;
    (*coordinates)(2, 1) = -kHX;
    (*coordinates)(2, 2) = kHZ;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// H2O at the CCCBDB HF/cc-pVDZ optimized geometry (the control pin).
qcx::Result<qcx::molecule::Molecule> MakeH2oAtCcpvdzGeometry() {
    constexpr double kAngstromToBohr = qcx::molecule::kAngstromToBohr;
    constexpr double kOZ = 0.1157 * kAngstromToBohr;
    constexpr double kHX = 0.7488 * kAngstromToBohr;
    constexpr double kHZ = -0.4629 * kAngstromToBohr;
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = kOZ;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = kHX;
    (*coordinates)(1, 2) = kHZ;
    (*coordinates)(2, 0) = 0.0;
    (*coordinates)(2, 1) = -kHX;
    (*coordinates)(2, 2) = kHZ;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// Runs the direct-path RHF at the given basis directory and returns the
// (converged, iterations, energy) triple.
struct DirectRhfRun {
    bool converged;
    int iterations;
    double totalEnergy;
};

qcx::Result<DirectRhfRun> RunDirectRhf(const qcx::molecule::Molecule& molecule,
                                       const std::string& basisDir,
                                       qcx::integrals::AccuracyPreset accuracy) {
    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / basisDir).string());

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto overlapTensor = qcx::integrals::BuildOverlapMatrix(molecule, *basis);

    if (!overlapTensor.has_value())
    {
        return std::unexpected(overlapTensor.error());
    }

    auto coreTensor = BuildCoreHamiltonian(molecule, *basis);

    if (!coreTensor.has_value())
    {
        return std::unexpected(coreTensor.error());
    }

    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = accuracy;
    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, *basis, *coreTensor, fockOptions);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    const Eigen::MatrixXd overlap = ToMatrix(*overlapTensor);
    const Eigen::MatrixXd core = ToMatrix(*coreTensor);

    auto scf =
        qcx::scf::RunRhfScf(molecule, overlap, core, {}, MakeDirectFockBuilder(*builder), &*basis);

    if (!scf.has_value())
    {
        return std::unexpected(scf.error());
    }

    return DirectRhfRun{scf->converged, scf->iterations, scf->totalEnergy};
}

TEST(RhfConvergenceTest, AugCcpvdzConvergesToThePyscfPin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP();
    }

    auto molecule = MakeH2oAtAugCcpvdzGeometry();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kTight: the fp64 direct path (fp32 lane disabled), the pin
    // convention - the energy assertion is against the CCCBDB reference
    // -76.041844 at 1e-6 (pyscf reproduces it to 4.5e-7 at this geometry).
    auto run = RunDirectRhf(*molecule, "aug-cc-pvdz", qcx::integrals::AccuracyPreset::kTight);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    EXPECT_TRUE(run->converged) << "H2O/aug-cc-pVDZ RHF did not converge (kTight)";
    EXPECT_NEAR(run->totalEnergy, -76.041844, 1e-6);
    EXPECT_LE(run->iterations, 60);
}

TEST(RhfConvergenceTest, AugCcpvdzConvergesAtTheClaimPreset) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP();
    }

    auto molecule = MakeH2oAtAugCcpvdzGeometry();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kNormal: the claim preset - the fp32 certified lane is engaged,
    // so the energy budget is the claim's own 2e-6 (the certified bound
    // keeps the delivered error below it; the pre-fix tree failed to
    // converge entirely here).
    auto run = RunDirectRhf(*molecule, "aug-cc-pvdz", qcx::integrals::AccuracyPreset::kNormal);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    EXPECT_TRUE(run->converged) << "H2O/aug-cc-pVDZ RHF did not converge (kNormal)";
    EXPECT_NEAR(run->totalEnergy, -76.041844, 2e-6);
    EXPECT_LE(run->iterations, 60);
}

TEST(RhfConvergenceTest, CcpvdzControlConvergesToThePyscfPin) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP();
    }

    auto molecule = MakeH2oAtCcpvdzGeometry();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The cc-pVDZ control (the reproduction baseline): the same shared
    // iteration core must keep converging the no-diffuse case. The +-3e-6
    // band is the loose-gate trajectory class: at the default density
    // gate (1e-6) DIIS trajectories stop on nearby iterates, with inter-route
    // energy differences 8.6e-7..3.3e-6 (a re-pin precedent). The
    // engagement-gate removal (rhf.cpp: the
    // kDiisStartThreshold = 1e-1 commutator-norm gate deleted, so CDIIS
    // appends and extrapolates from iteration 2) moved this landing +1.197e-6
    // above the pin. A/B-verified (2026-09-03): with the seven scf engine and
    // test files restored to their HEAD content the test passes at +-1e-6;
    // with the revised engine the same test code converges to -76.027052802530704
    // (converged, <= 40 iterations); nothing else differs. The SCF fixed point
    // is unchanged - all 12 explicit-gate pins (1e-10/1e-10 tolerances;
    // the reading dates from 2026-09-03, when that family still carried the
    // explicit gates; the 2026-09-15 gate conversion re-decided them)
    // stay bit-identical under the revised engine.
    auto run = RunDirectRhf(*molecule, "cc-pvdz", qcx::integrals::AccuracyPreset::kTight);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    EXPECT_TRUE(run->converged) << "H2O/cc-pVDZ RHF did not converge (control)";
    EXPECT_NEAR(run->totalEnergy, -76.027054, 3e-6);
    EXPECT_LE(run->iterations, 40);
}

} // namespace
