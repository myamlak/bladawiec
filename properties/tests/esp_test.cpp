// ESP charge-fit tests: CHELPG and MK charges of H2O
// and H2 in STO-3G from the real converged RHF density, pinned against the
// pyscf reference (tools/properties/esp_pyscf_probe.py, WSL 2026-08-26,
// unit='Bohr'; the probe replicates the point sets exactly and fits with
// the same constrained normal equations).
//
// The defining identities:
//   sum_A q_A = Z - N (the molecular charge; 0 for the neutral fixtures)
//   the MK point set is C2v-symmetric, so q_H1 = q_H2 exactly
//   the CHELPG fit reproduces the dipole (its design goal)
//   the H2 fits vanish (monopole-dominated far-field point sets)
//
// Atom order: H2O sorts to [H(-x), H(+x), O]; the pyscf input order
// [O, H(-x), H(+x)] is permuted in the probe.
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/esp.hpp"
#include "qcx/properties/multipoles.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// The pyscf pins (probe output above); the fit runs through different
// solvers (Eigen LDLT vs LAPACK LU) and the point potentials through
// different integral engines (MD vs pyscf), so the tolerance is loose.
constexpr double kPinTolerance = 1e-6;

// H = T + V from the one-electron engines (the dense_rhf_test.cpp
// pattern, shared with the sibling properties tests).
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

// The dense general-l RHF path (eri_dense.hpp) - the ESP tests run the
// real converged density, then fit charges to it.
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

// The dipole of the fitted charges, sum_A q_A R_A (e*a0), against the
// quantum dipole of the density (properties::AnalyzeMultipoles).
double DipoleDifference(const qcx::molecule::Molecule& molecule,
                        const qcx::basisset::BasisSet& basisSet,
                        const Eigen::MatrixXd& density,
                        const Eigen::VectorXd& charges) {
    auto moments =
        qcx::properties::AnalyzeMultipoles(molecule, basisSet, density / 2.0, density / 2.0);
    EXPECT_TRUE(moments.has_value());
    const auto& coordinates = molecule.CoordinatesBohr();
    Eigen::Vector3d fitted = Eigen::Vector3d::Zero();

    for (std::size_t a = 0; a < molecule.AtomCount(); ++a)
    {
        fitted += charges(static_cast<Eigen::Index>(a)) *
                  Eigen::Vector3d(coordinates(a, 0), coordinates(a, 1), coordinates(a, 2));
    }

    return (fitted - moments->dipole).norm();
}

} // namespace

TEST(EspTest, H2oSto3gChelpgPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + lattice: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto result = qcx::properties::AnalyzeEspCharges(
        *molecule, *basis, scf->density, qcx::properties::EspFitScheme::kChelpg);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_EQ(result->pointCount, 3910u);
    EXPECT_NEAR(result->charges(0), 0.307317743072, kPinTolerance) << "q(H(-x))";
    EXPECT_NEAR(result->charges(1), 0.307210416720, kPinTolerance) << "q(H(+x))";
    EXPECT_NEAR(result->charges(2), -0.614528159792, kPinTolerance) << "q(O)";

    // The defining identities: neutral sum, CHELPG dipole reproduction.
    EXPECT_NEAR(result->charges.sum(), 0.0, 1e-8);
    EXPECT_LT(result->rmsError, 1e-3);
    EXPECT_LT(DipoleDifference(*molecule, *basis, scf->density, result->charges), 0.05);
}

TEST(EspTest, H2oSto3gMerzKollmanPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + shells: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto result = qcx::properties::AnalyzeEspCharges(
        *molecule, *basis, scf->density, qcx::properties::EspFitScheme::kMerzKollman);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_EQ(result->pointCount, 3624u);
    EXPECT_NEAR(result->charges(0), 0.485866618952, kPinTolerance) << "q(H(-x))";
    EXPECT_NEAR(result->charges(1), 0.485866618952, kPinTolerance) << "q(H(+x))";
    EXPECT_NEAR(result->charges(2), -0.971733237904, kPinTolerance) << "q(O)";

    // The C2v-symmetric shell set forces the H charges exactly equal.
    EXPECT_NEAR(result->charges(0), result->charges(1), 1e-12);
    EXPECT_NEAR(result->charges.sum(), 0.0, 1e-8);
}

TEST(EspTest, H2Sto3gChelpgVanishes) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + lattice: Release-only";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto result = qcx::properties::AnalyzeEspCharges(
        *molecule, *basis, scf->density, qcx::properties::EspFitScheme::kChelpg);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The CHELPG band is far outside H2, so the potential is
    // monopole-dominated and the fit vanishes (inversion symmetry forces
    // q_H1 = -q_H2).
    EXPECT_EQ(result->pointCount, 3345u);
    EXPECT_NEAR(result->charges(0), -0.000409207723, kPinTolerance);
    EXPECT_NEAR(result->charges(1), 0.000409207723, kPinTolerance);
    EXPECT_NEAR(result->charges.sum(), 0.0, 1e-12);
}

// Coverage: the fit must refuse
// degenerate inputs up front instead of silently returning garbage
// charges.  Two mechanisms, both pinned by probe replication of the
// exact CHELPG lattice: (a) hydrogens 5e-7 bohr apart (just past the
// molecule module's 1e-7 coincident-atom rejection floor) make the A
// block nearly rank-1 (1/(r_A r_B) ~ 1e14 entries) - the LDLT pivot
// ratio (a condition lower bound) is 1.2e13, far past the gate, and the
// un-gated solve returns charges far beyond the sanity cap while LDLT
// reports Success; (b) a fit window with inner > outer yields zero
// points for any molecule - fewer points than atoms leaves the normal
// equations rank-deficient.  No SCF is needed: an identity density
// gives N_e = tr(D S) = 2 and a zero molecular charge.
TEST(EspTest, RejectsPathologicalFitGeometries) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto coordinates = CpuTensor2::Create({2, 3});
    ASSERT_TRUE(coordinates.has_value()) << coordinates.error().message;
    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 5e-7;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();

    auto molecule = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const Eigen::MatrixXd density = Eigen::MatrixXd::Identity(2, 2);

    auto nearCoincident = qcx::properties::AnalyzeEspCharges(
        *molecule, *basis, density, qcx::properties::EspFitScheme::kChelpg);
    ASSERT_FALSE(nearCoincident.has_value());
    EXPECT_EQ(nearCoincident.error().code, qcx::ErrorCode::kInvalidArgument);

    // A CHELPG window with inner > outer: the lattice produces zero fit
    // points for any molecule, so the point-count gate rejects it.
    auto ordinary = MakeH2Sto3g();
    ASSERT_TRUE(ordinary.has_value()) << ordinary.error().message;

    qcx::properties::EspFitOptions degenerate;
    degenerate.chelpgInnerRadiusBohr = 3.0;
    degenerate.chelpgOuterRadiusBohr = 2.0;

    auto noPoints = qcx::properties::AnalyzeEspCharges(
        *ordinary, *basis, density, qcx::properties::EspFitScheme::kChelpg, degenerate);
    ASSERT_FALSE(noPoints.has_value());
    EXPECT_EQ(noPoints.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(EspTest, RejectsInvalidInputs) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // Wrong density shape: the H2O/STO-3G AO count is 7.
    Eigen::MatrixXd badDensity = Eigen::MatrixXd::Zero(3, 3);
    auto badShape = qcx::properties::AnalyzeEspCharges(
        *molecule, *basis, badDensity, qcx::properties::EspFitScheme::kChelpg);
    ASSERT_FALSE(badShape.has_value());
    EXPECT_EQ(badShape.error().code, qcx::ErrorCode::kInvalidArgument);

    // A non-Lebedev MK angular count.
    Eigen::MatrixXd density = Eigen::MatrixXd::Identity(7, 7);
    qcx::properties::EspFitOptions options;
    options.mkLebedevPoints = 7;
    auto badAngular = qcx::properties::AnalyzeEspCharges(
        *molecule, *basis, density, qcx::properties::EspFitScheme::kMerzKollman, options);
    ASSERT_FALSE(badAngular.has_value());
    EXPECT_EQ(badAngular.error().code, qcx::ErrorCode::kInvalidArgument);
}
