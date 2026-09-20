// Fukui-condensation tests: the condensed Fukui indices
// (f^+, f^-, f^0) of H2O/STO-3G from the three real converged SCF
// densities - RHF for the neutral singlet, UHF doublets for the anion
// (N + 1) and cation (N - 1) at the same geometry and basis.
//
// The condensation is defined through Mulliken population differences
// ([Parr1984], [YangMortier1986]); the defining identities are
//     sum_A f_A^+ = sum_A f_A^- = sum_A f_A^0 = 1
// (each difference changes the total population by exactly one electron),
// and f^0 = (f^+ + f^-) / 2 by construction.  The H2O/STO-3G chemistry is
// the textbook case: the oxygen lone pairs make O the electrophilic site
// (largest f^-) and the sigma* OH orbital makes the hydrogens the
// nucleophilic site (largest f^+).
//
// Atom order note: H2O sorts to [H(-x), H(+x), O], so the O atom is
// index 2.
#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/fukui.hpp"
#include "qcx/properties/populations.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;

// The H2O geometry of the MakeH2oSto3g fixture with a different
// charge/multiplicity - the charged species share geometry and basis
// with the neutral (the defining condition of a Fukui response).
qcx::Result<qcx::molecule::Molecule> MakeChargedH2o(int charge, int multiplicity) {
    auto coordinates = CpuTensor2::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.430428808474167;
    (*coordinates)(1, 1) = 1.107157044080814;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -1.430428808474167;
    (*coordinates)(2, 1) = 1.107157044080814;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        charge,
        multiplicity);
}

// H = T + V from the one-electron engines (the dense_rhf_test.cpp
// pattern, shared with populations_test.cpp).
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

// The dense general-l SCF path (eri_dense.hpp) - the Fukui tests run the
// real converged densities, then condense them.
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

qcx::Result<qcx::scf::UhfResult> RunDenseUhf(const qcx::molecule::Molecule& molecule,
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

    return qcx::scf::RunUhfScf(molecule, ToMatrix(*overlap), ToMatrix(*core), *eri);
}

// The three converged densities for the H2O/STO-3G Fukui response: RHF
// neutral singlet, UHF anion (charge -1, doublet), UHF cation (charge +1,
// doublet), plus the shared overlap matrix.  HfResult::density is the
// spin-summed D = 2 rho; the per-spin convention is D/2 per channel
// (populations.hpp), while the UHF densities are already per-spin.
struct H2oFukuiSystem {
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd neutralAlpha;
    Eigen::MatrixXd neutralBeta;
    Eigen::MatrixXd anionAlpha;
    Eigen::MatrixXd anionBeta;
    Eigen::MatrixXd cationAlpha;
    Eigen::MatrixXd cationBeta;
    std::vector<qcx::properties::AoRange> aoRanges;
};

qcx::Result<H2oFukuiSystem> RunH2oFukuiSystem() {
    auto basis = MakeH2oSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto neutral = MakeH2oSto3g();

    if (!neutral.has_value())
    {
        return std::unexpected(neutral.error());
    }

    auto anionMolecule = MakeChargedH2o(-1, 2);

    if (!anionMolecule.has_value())
    {
        return std::unexpected(anionMolecule.error());
    }

    auto cationMolecule = MakeChargedH2o(1, 2);

    if (!cationMolecule.has_value())
    {
        return std::unexpected(cationMolecule.error());
    }

    auto neutralScf = RunDenseRhf(*neutral, *basis);

    if (!neutralScf.has_value())
    {
        return std::unexpected(neutralScf.error());
    }

    auto anionScf = RunDenseUhf(*anionMolecule, *basis);

    if (!anionScf.has_value())
    {
        return std::unexpected(anionScf.error());
    }

    auto cationScf = RunDenseUhf(*cationMolecule, *basis);

    if (!cationScf.has_value())
    {
        return std::unexpected(cationScf.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*neutral, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*neutral, *basis);

    if (!aoRanges.has_value())
    {
        return std::unexpected(aoRanges.error());
    }

    return H2oFukuiSystem{ToMatrix(*overlap),
                          0.5 * neutralScf->density,
                          0.5 * neutralScf->density,
                          anionScf->densityAlpha,
                          anionScf->densityBeta,
                          cationScf->densityAlpha,
                          cationScf->densityBeta,
                          *aoRanges};
}

} // namespace

TEST(FukuiTest, H2oSto3gSumRules) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "three real SCFs: Release-only";
    }

    auto system = RunH2oFukuiSystem();
    ASSERT_TRUE(system.has_value()) << system.error().message;

    auto fukui = qcx::properties::AnalyzeFukui(system->overlap,
                                               system->neutralAlpha,
                                               system->neutralBeta,
                                               system->anionAlpha,
                                               system->anionBeta,
                                               system->cationAlpha,
                                               system->cationBeta,
                                               system->aoRanges);
    ASSERT_TRUE(fukui.has_value()) << fukui.error().message;

    // The defining identities: each vector sums to exactly one.
    EXPECT_NEAR(fukui->nucleophilic.sum(), 1.0, 1e-8);
    EXPECT_NEAR(fukui->electrophilic.sum(), 1.0, 1e-8);
    EXPECT_NEAR(fukui->radical.sum(), 1.0, 1e-8);

    // The textbook H2O response: O is the electrophilic site (the lone
    // pairs - largest f^-), the H's the nucleophilic site (the sigma* OH
    // orbital - largest f^+).  Atom order [H, H, O].
    EXPECT_GT(fukui->electrophilic(2), fukui->electrophilic(0));
    EXPECT_GT(fukui->nucleophilic(0), fukui->nucleophilic(2));
}

TEST(FukuiTest, H2oSto3gMatchesManualMullikenDifferences) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "three real SCFs: Release-only";
    }

    auto system = RunH2oFukuiSystem();
    ASSERT_TRUE(system.has_value()) << system.error().message;

    auto fukui = qcx::properties::AnalyzeFukui(system->overlap,
                                               system->neutralAlpha,
                                               system->neutralBeta,
                                               system->anionAlpha,
                                               system->anionBeta,
                                               system->cationAlpha,
                                               system->cationBeta,
                                               system->aoRanges);
    ASSERT_TRUE(fukui.has_value()) << fukui.error().message;

    auto neutral = qcx::properties::AnalyzeMulliken(
        system->overlap, system->neutralAlpha, system->neutralBeta, system->aoRanges);
    ASSERT_TRUE(neutral.has_value()) << neutral.error().message;

    auto anion = qcx::properties::AnalyzeMulliken(
        system->overlap, system->anionAlpha, system->anionBeta, system->aoRanges);
    ASSERT_TRUE(anion.has_value()) << anion.error().message;

    auto cation = qcx::properties::AnalyzeMulliken(
        system->overlap, system->cationAlpha, system->cationBeta, system->aoRanges);
    ASSERT_TRUE(cation.has_value()) << cation.error().message;

    const Eigen::VectorXd plus = anion->total - neutral->total;
    const Eigen::VectorXd minus = neutral->total - cation->total;

    EXPECT_NEAR((fukui->nucleophilic - plus).norm(), 0.0, 1e-12);
    EXPECT_NEAR((fukui->electrophilic - minus).norm(), 0.0, 1e-12);
    EXPECT_NEAR((fukui->radical - 0.5 * (plus + minus)).norm(), 0.0, 1e-12);
}

TEST(FukuiTest, IdenticalDensitiesGiveZero) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*molecule, *basis);
    ASSERT_TRUE(aoRanges.has_value()) << aoRanges.error().message;

    // No response: all three species share the same densities, so every
    // difference vanishes.
    const Eigen::MatrixXd density = 0.5 * scf->density;
    auto fukui = qcx::properties::AnalyzeFukui(
        ToMatrix(*overlap), density, density, density, density, density, density, *aoRanges);
    ASSERT_TRUE(fukui.has_value()) << fukui.error().message;

    EXPECT_NEAR(fukui->nucleophilic.norm(), 0.0, 1e-12);
    EXPECT_NEAR(fukui->electrophilic.norm(), 0.0, 1e-12);
    EXPECT_NEAR(fukui->radical.norm(), 0.0, 1e-12);
}

TEST(FukuiTest, RejectsShapeMismatch) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*molecule, *basis);
    ASSERT_TRUE(aoRanges.has_value()) << aoRanges.error().message;

    const Eigen::MatrixXd density = 0.5 * scf->density;
    const Eigen::MatrixXd wrongShape =
        Eigen::MatrixXd::Zero(density.rows() + 1, density.rows() + 1);
    auto fukui = qcx::properties::AnalyzeFukui(
        ToMatrix(*overlap), density, density, wrongShape, density, density, density, *aoRanges);
    ASSERT_FALSE(fukui.has_value()) << "expected a rejected anion density";
    EXPECT_EQ(fukui.error().code, qcx::ErrorCode::kInvalidArgument);
}
