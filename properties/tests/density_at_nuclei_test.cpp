// Density-at-nuclei tests: the electron
// density at every nucleus from the spin-summed AO density.
//
// The defining pins:
//   (a) a single s contraction on an atom: rho(0) = D * phi(0)^2 with
//       phi(0) = sum_i d_i N_0(zeta_i), N_0(zeta) = (2 zeta/pi)^(3/4) -
//       the evaluator's exact l = 0 normalization, reimplemented here
//   (b) the tr(D S) = N_e invariant and H2 inversion symmetry on a real
//       converged RHF density
//   (c) the H2/STO-3G known-value row from the pyscf 2.14.0 cross-check
//       (unit='Bohr', tools/properties/density_at_nuclei_pyscf_probe.py)
//   (d) the deferred-Bader seed: rho(r) sampled on a fine radial grid
//       must match the closed form at every point and approach rho(0) as
//       r -> 0 - the consistency between the point evaluator and the
//       quadrature path the Bader increment will build on.

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/radial_grid.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/density_at_nuclei.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

// The H2/STO-3G reference value from the pyscf 2.14.0 cross-check (WSL
// 2026-08-29, unit='Bohr'; probe in tools/properties/): rho(R_A) =
// 0.354892073081 electrons/bohr^3 at each nucleus, N_e = 2.000000000000.
constexpr double kH2Sto3gRhoAtNucleus = 0.354892073081;

// The evaluator's l = 0 normalization (ao_evaluator.cpp RadialNormalization,
// reimplemented for the reference): N_0(zeta) = (2 zeta/pi)^(3/4).
double SOrbitalNormalization(double exponent) {
    return std::pow(2.0 * exponent / kPi, 0.75);
}

using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// One hydrogen at the origin (the fixtures only offer H2/He).
qcx::Result<qcx::molecule::Molecule> MakeSingleHydrogen() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}}, std::move(*coordinates), 0, 1);
}

// H = T + V from the one-electron engines (the dense_rhf_test.cpp pattern,
// shared with the sibling properties tests).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoreHamiltonian(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
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

// The dense general-l RHF path - the density-at-nuclei tests run the real
// converged density, then evaluate it at the nuclei.
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

TEST(DensityAtNucleiTest, SingleSPrimitiveMatchesClosedForm) {
    // Pin (a): two electrons in the single STO-3G s function on a hydrogen
    // at the origin.  rho(0) = D phi(0)^2 with phi(0) = sum_i d_i N_0(zeta_i)
    // (the evaluator's exact normalization) - no quadrature involved.
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    Eigen::MatrixXd density(1, 1);
    density(0, 0) = 2.0;

    auto values = qcx::properties::AnalyzeDensityAtNuclei(*molecule, *basis, density);
    ASSERT_TRUE(values.has_value()) << values.error().message;
    ASSERT_EQ(values->size(), 1);

    const auto& shells = basis->Find(1)->shells;
    ASSERT_EQ(shells.size(), 1u);
    ASSERT_EQ(shells[0].coefficients.size(), 1u);

    double phiAtOrigin = 0.0;

    for (std::size_t i = 0; i < shells[0].exponents.size(); ++i)
    {
        phiAtOrigin += shells[0].coefficients[0][i] * SOrbitalNormalization(shells[0].exponents[i]);
    }

    EXPECT_NEAR((*values)(0), 2.0 * phiAtOrigin * phiAtOrigin, 1e-10);
}

TEST(DensityAtNucleiTest, RadialGridConsistencySeed) {
    // Pin (d): the deferred-Bader seed.  rho(r) along a ray from the
    // nucleus, sampled at the points of a fine radial quadrature, must
    // match the closed form phi(r)^2 (single s contraction, D = 2) at
    // every point - the consistency between the point evaluator and the
    // quadrature path - and the innermost point must approach rho(0).
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto evaluator = qcx::grid::AoEvaluator::Create(*molecule, *basis);
    ASSERT_TRUE(evaluator.has_value()) << evaluator.error().message;

    auto radial = qcx::grid::RadialGrid::Create(200, 4.0);
    ASSERT_TRUE(radial.has_value()) << radial.error().message;

    const auto& shells = basis->Find(1)->shells;

    for (std::size_t k = 0; k < radial->Size(); ++k)
    {
        const double radius = radial->Point(k);
        std::vector<double> phi(1);
        evaluator->Evaluate({0.0, 0.0, radius}, phi);

        double phiReference = 0.0;

        for (std::size_t i = 0; i < shells[0].exponents.size(); ++i)
        {
            phiReference += shells[0].coefficients[0][i] *
                            SOrbitalNormalization(shells[0].exponents[i]) *
                            std::exp(-shells[0].exponents[i] * radius * radius);
        }

        EXPECT_NEAR(phi[0], phiReference, 1e-12);
    }

    // The r -> 0 limit: the innermost radial point (alpha / N^2 ~ 1e-4
    // bohr) carries essentially rho(0) - the quadrature path sees the
    // same density the nucleus evaluation returns.  The residual is the
    // Gaussian drop over the innermost radius: rho(r1) / rho(0) =
    // exp(-2 * zeta * r1^2), bounded by the tightest contraction
    // exponent (largest zeta) - at r1 = 1e-4 bohr that is ~3e-8, so a
    // literal 1e-9 would be tighter than the physics at this radius.
    const double r1 = radial->Point(0);
    double tightestExponent = 0.0;

    for (std::size_t i = 0; i < shells[0].exponents.size(); ++i)
    {
        if (shells[0].exponents[i] > tightestExponent)
        {
            tightestExponent = shells[0].exponents[i];
        }
    }

    auto values = qcx::properties::AnalyzeDensityAtNuclei(
        *molecule, *basis, Eigen::MatrixXd::Constant(1, 1, 2.0));
    ASSERT_TRUE(values.has_value()) << values.error().message;

    std::vector<double> phiInner(1);
    evaluator->Evaluate({0.0, 0.0, r1}, phiInner);
    const double rhoInner = 2.0 * phiInner[0] * phiInner[0];
    const double dropBound = rhoInner * (1.0 - std::exp(-2.0 * tightestExponent * r1 * r1));
    EXPECT_NEAR(rhoInner, (*values)(0), dropBound + 1e-12);
}

TEST(DensityAtNucleiTest, H2Sto3gTrDsInvariantAndSymmetry) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    // Pin (b): the converged H2/STO-3G RHF density integrates to N_e = 2
    // (tr(D S)) and the inversion symmetry forces rho(H1) = rho(H2).
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    const Eigen::MatrixXd overlapMatrix = ToMatrix(*overlap);
    const double electronCount = (scf->density * overlapMatrix).trace();
    EXPECT_NEAR(electronCount, 2.0, 1e-9);

    auto values = qcx::properties::AnalyzeDensityAtNuclei(*molecule, *basis, scf->density);
    ASSERT_TRUE(values.has_value()) << values.error().message;
    ASSERT_EQ(values->size(), 2);

    EXPECT_GT((*values)(0), 0.0);
    EXPECT_NEAR((*values)(0), (*values)(1), 1e-12);
}

TEST(DensityAtNucleiTest, H2Sto3gKnownValueRow) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    // Pin (c): the pyscf cross-check row (unit='Bohr', R = 1.4 bohr; the
    // probe prints 0.354892073081 at each nucleus).  Same convention, so
    // the tolerance is the tight pyscf-vs-qcx agreement.
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;

    auto values = qcx::properties::AnalyzeDensityAtNuclei(*molecule, *basis, scf->density);
    ASSERT_TRUE(values.has_value()) << values.error().message;

    EXPECT_NEAR((*values)(0), kH2Sto3gRhoAtNucleus, 1e-8);
    EXPECT_NEAR((*values)(1), kH2Sto3gRhoAtNucleus, 1e-8);
}

TEST(DensityAtNucleiTest, RejectsDensityShapeMismatch) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    Eigen::MatrixXd density(2, 2);
    density.setZero();

    auto values = qcx::properties::AnalyzeDensityAtNuclei(*molecule, *basis, density);
    ASSERT_FALSE(values.has_value());
    EXPECT_EQ(values.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
