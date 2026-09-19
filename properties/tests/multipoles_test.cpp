// Multipole-moment pins: dipole and traceless quadrupole on the
// H2/HF/H2O/O2 STO-3G fixtures, run on the REAL converged SCF densities
// and pinned against pyscf 2.14.0 (WSL 2026-08-25 cross-check; the
// convention there is identical to the port: dipole d = sum_a Z_a R_a -
// sum_ij P(i,j) <j|r|i> with P = P_alpha + P_beta, and the symmetrized
// Theta = 3M - Tr(M) I, all relative to the coordinate origin as given).
//
// Hand checks (stated provenance):
//   - H2: the nuclei sit at x = 0 and x = 1.4 (the fixture geometry), so
//     the nuclear moment cancels the electronic one exactly by inversion
//     symmetry through the midpoint: d = 0 at 1e-10.
//   - O2 triplet: the two oxygen atoms are equivalent, so the dipole
//     vanishes and Theta is diag(xx, yy, -2xx) with xx = yy.
//   - Origin-shift identities (exact algebra, verified at 1e-12): for a
//     neutral molecule the dipole is origin-invariant, and the quadrupole
//     shifts as Theta' = Theta - 3(Delta d^T + d Delta^T) + 2(Delta . d) I
//     where d is the dipole at the original origin. The HF case exercises
//     both terms with d != 0; the identity is the end-to-end proof that
//     the origin parameter plumbed through to the integrals.
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "o2_sad_guess.hpp"
#include "o2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/multipoles.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <utility>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::properties::testing::RunDenseUhfO2;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeHfSto3g;
using qcx::testing::MakeHfSto3gBasis;
using qcx::testing::MakeO2Sto3gBasis;
using qcx::testing::MakeO2Sto3gTriplet;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// The pins are density-limited: qcx and pyscf each converge to their own
// DIIS fixed point, so density-derived moments agree to ~1e-8 (dipoles),
// but the trace-zero quadrupole Theta = 3M - Tr(M) I amplifies the tiny
// density difference: the diagonals are sums over all electrons' worth of
// moment integrals, and the ~3.5e-8 energy drift of the dense-ERI SCF
// recipe vs pyscf lands ~1.2e-7 on the diagonals (HF and O2 observed
// 2026-08-25; the direct kTight recipe reproduces pyscf at 3.5e-11,
// the module test recipe converges by its own 1e-10 tolerance instead).
// The quadrupole diagonals therefore get a looser tolerance.
constexpr double kReferenceTolerance = 1e-7;
constexpr double kQuadrupoleTolerance = 5e-7;
// The O2/STO-3G triplet UHF runs land at different points of the
// near-degenerate DIIS fixed-point cluster (documented inter-route
// spread 8.6e-7..3.3e-6); the Fix-1 per-batch parallel Fock layout
// flips this test's landing (observed
// 1.6e-6..3.2e-6), so the O2 quadrupole pins widen to the cluster scale.
constexpr double kO2ClusterQuadrupoleTolerance = 5e-6;
constexpr double kIdentityTolerance = 1e-12;

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

// The dense general-l SCF path (eri_dense.hpp), shared with the population
// tests: the moments are computed from the real converged densities.
qcx::Result<qcx::scf::HfResult> RunDenseRhf(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet) {
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

    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basisSet);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return qcx::scf::RunRhfScf(molecule, ToMatrix(*overlap), ToMatrix(*core), *eri);
}

void ExpectSymmetric(const Eigen::Matrix3d& matrix) {
    EXPECT_NEAR(matrix(0, 1), matrix(1, 0), kIdentityTolerance);
    EXPECT_NEAR(matrix(0, 2), matrix(2, 0), kIdentityTolerance);
    EXPECT_NEAR(matrix(1, 2), matrix(2, 1), kIdentityTolerance);
}

void ExpectTraceless(const Eigen::Matrix3d& matrix) {
    EXPECT_NEAR(matrix.trace(), 0.0, kIdentityTolerance);
}

} // namespace

TEST(MultipolesTest, H2Sto3gDipoleAndQuadrupole) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    // HfResult::density is the spin-summed D = 2 rho; the per-spin
    // convention is D/2 per channel (multipoles.hpp).
    auto moments = qcx::properties::AnalyzeMultipoles(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density);
    ASSERT_TRUE(moments.has_value()) << moments.error().message;

    // Inversion symmetry through the bond midpoint: exact zero.
    EXPECT_NEAR(moments->dipole(0), 0.0, 1e-10);
    EXPECT_NEAR(moments->dipole(1), 0.0, 1e-10);
    EXPECT_NEAR(moments->dipole(2), 0.0, 1e-10);
    EXPECT_NEAR(moments->quadrupole(0, 0), 0.6258891042, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(1, 1), -0.3129445521, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(2, 2), -0.3129445521, kReferenceTolerance);
    ExpectSymmetric(moments->quadrupole);
    ExpectTraceless(moments->quadrupole);
}

TEST(MultipolesTest, HfSto3gDipoleAndQuadrupole) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto moments = qcx::properties::AnalyzeMultipoles(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density);
    ASSERT_TRUE(moments.has_value()) << moments.error().message;

    EXPECT_NEAR(moments->dipole(0), 0.50700628073, kReferenceTolerance);
    EXPECT_NEAR(moments->dipole(1), 0.0, kReferenceTolerance);
    EXPECT_NEAR(moments->dipole(2), 0.0, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(0, 0), 2.0227565527, kQuadrupoleTolerance);
    EXPECT_NEAR(moments->quadrupole(1, 1), -1.0113782764, kQuadrupoleTolerance);
    EXPECT_NEAR(moments->quadrupole(2, 2), -1.0113782764, kQuadrupoleTolerance);
    ExpectSymmetric(moments->quadrupole);
    ExpectTraceless(moments->quadrupole);
}

TEST(MultipolesTest, H2oSto3gDipoleAndQuadrupole) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto moments = qcx::properties::AnalyzeMultipoles(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density);
    ASSERT_TRUE(moments.has_value()) << moments.error().message;

    EXPECT_NEAR(moments->dipole(0), 0.0, kReferenceTolerance);
    EXPECT_NEAR(moments->dipole(1), 0.67898079210, kReferenceTolerance);
    EXPECT_NEAR(moments->dipole(2), 0.0, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(0, 0), 1.7609360829, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(0, 1), 0.0, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(0, 2), 0.0, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(1, 1), 0.35190091054, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(1, 2), 0.0, kReferenceTolerance);
    EXPECT_NEAR(moments->quadrupole(2, 2), -2.1128369935, kReferenceTolerance);
    ExpectSymmetric(moments->quadrupole);
    ExpectTraceless(moments->quadrupole);
}

TEST(MultipolesTest, O2TripletDipoleAndQuadrupole) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeO2Sto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeO2Sto3gTriplet();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The SAD-seeded dense UHF (the default P = 0 start locks the
    // higher-lying saddle solution instead - see o2_sad_guess.hpp).
    auto scf = RunDenseUhfO2(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    // UHF: the per-spin densities pass through unchanged (no halving).
    auto moments =
        qcx::properties::AnalyzeMultipoles(*molecule, *basis, scf->densityAlpha, scf->densityBeta);
    ASSERT_TRUE(moments.has_value()) << moments.error().message;

    // Equivalent atoms: the dipole vanishes; Theta is diag(xx, xx, -2xx)
    // with the bond along z.
    EXPECT_NEAR(moments->dipole(0), 0.0, 1e-10);
    EXPECT_NEAR(moments->dipole(1), 0.0, 1e-10);
    EXPECT_NEAR(moments->dipole(2), 0.0, 1e-10);
    EXPECT_NEAR(moments->quadrupole(0, 0), 0.93063166671, kO2ClusterQuadrupoleTolerance);
    EXPECT_NEAR(moments->quadrupole(1, 1), 0.93063166671, kO2ClusterQuadrupoleTolerance);
    EXPECT_NEAR(moments->quadrupole(2, 2), -1.8612633334, kO2ClusterQuadrupoleTolerance);
    ExpectSymmetric(moments->quadrupole);
    ExpectTraceless(moments->quadrupole);
}

TEST(MultipolesTest, OriginShiftIdentitiesOnHf) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    const std::array<double, 3> origin = {0.0, 0.0, 0.0};
    const std::array<double, 3> shifted = {0.3, 0.2, -0.1};
    const Eigen::Vector3d delta(0.3, 0.2, -0.1);

    auto atOrigin = qcx::properties::AnalyzeMultipoles(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density, origin);
    ASSERT_TRUE(atOrigin.has_value()) << atOrigin.error().message;
    auto atShift = qcx::properties::AnalyzeMultipoles(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density, shifted);
    ASSERT_TRUE(atShift.has_value()) << atShift.error().message;

    // Neutral molecule: the dipole is origin-invariant.
    for (std::size_t k = 0; k < 3; ++k)
    {
        EXPECT_NEAR(atShift->dipole(k), atOrigin->dipole(k), kIdentityTolerance);
    }

    // Theta' = Theta - 3(Delta d^T + d Delta^T) + 2(Delta . d) I.
    const Eigen::Vector3d d = atOrigin->dipole;
    const Eigen::Matrix3d predicted = atOrigin->quadrupole -
                                      3.0 * (delta * d.transpose() + d * delta.transpose()) +
                                      2.0 * delta.dot(d) * Eigen::Matrix3d::Identity();

    for (std::size_t k = 0; k < 3; ++k)
    {
        for (std::size_t l = 0; l < 3; ++l)
        {
            EXPECT_NEAR(atShift->quadrupole(k, l), predicted(k, l), kIdentityTolerance);
        }
    }
}

TEST(MultipolesTest, RejectsDensityShapeMismatch) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const Eigen::MatrixXd wrongSize = Eigen::MatrixXd::Identity(3, 3);
    auto moments = qcx::properties::AnalyzeMultipoles(*molecule, *basis, wrongSize, wrongSize);
    EXPECT_FALSE(moments.has_value());
    EXPECT_EQ(moments.error().code, qcx::ErrorCode::kInvalidArgument);
}
