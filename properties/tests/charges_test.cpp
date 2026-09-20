// Hirshfeld/Voronoi charge tests: the charges of H2O and H2 in
// STO-3G from the real converged RHF density and the SAD promolecular
// fragments ([Hirshfeld1977], [FonsecaGuerra2004]).
//
// The quadrature runs on the MolecularGrid (Euler-Maclaurin radial x
// Lebedev angular, Becke partition); charges converge with the grid
// density, so the pins carry a grid- and quadrature-dependent tolerance
// and the grid-trend test asserts the monotone-in-grid-size behavior.
//
// The defining identities:
//   sum_A Q_A = Z - N (the molecular charge; 0 for the neutral fixtures)
//   H2 inversion symmetry forces Q_H1 = Q_H2 and hence Q_H1 = Q_H2 = 0
//   the SAD promolecular integrates to the molecular electron count
//
// Atom order note: H2O sorts to [H(-x), H(+x), O].
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/grid/molecular_grid.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/charges.hpp"
#include "qcx/properties/populations.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <gtest/gtest.h>
#include <map>
#include <utility>
#include <vector>

namespace {

// The H2O/STO-3G Hirshfeld reference charges from the pyscf 2.14.0
// cross-check (WSL 2026-08-26, unit='Bohr', grid level 4;
// pyscf [O,H,H] = [-0.44558, +0.22279, +0.22279]).  The grid and SAD
// quadrature conventions differ, so the pins carry a loose tolerance.
// Atom order [H, H, O]; pyscf's natural [O, H, H] order reversed.
constexpr double kHirshfeldHRef = 0.22279;
constexpr double kHirshfeldORef = -0.44558;

// The Voronoi reference charges from the same cross-check: pyscf
// [O,H,H] = [+0.65282, -0.32928, -0.32354]; a single (loose) H pin
// covers the ~5e-3 H1/H2 cell asymmetry of the nearest-atom partition.
constexpr double kVoronoiHRef = -0.32928;
constexpr double kVoronoiORef = 0.65282;

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

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

// The dense general-l RHF path (eri_dense.hpp) - the charge tests run the
// real converged density, then partition it.
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

// The dense per-element atomic integrals of the SAD guess: one atom at
// the origin with the given charge/multiplicity, integrals computed in
// the molecular basis (the fragment run; the geometry never affects its
// own integrals).  The pattern is the scf module's own uhf_test.cpp
// assembly (mirrored by properties/tests/o2_sad_guess.hpp).
qcx::Result<qcx::scf::AtomicUhfInputs> BuildAtomicInputs(const char* symbol,
                                                         int atomicNumber,
                                                         int multiplicity,
                                                         const qcx::basisset::BasisSet& basisSet) {
    auto coordinates = CpuTensor2::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    auto atom = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{symbol, atomicNumber, 0.0}},
        std::move(*coordinates),
        0,
        multiplicity);

    if (!atom.has_value())
    {
        return std::unexpected(atom.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*atom, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*atom, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*atom, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*atom, basisSet);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return qcx::scf::AtomicUhfInputs{
        ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), std::move(*eri)};
}

// The H2O/STO-3G SAD guess: O (triplet fragment) + H (doublet fragment),
// the standard fragment spin multiplicities.
qcx::Result<qcx::scf::SadGuess> BuildSadGuessForH2o(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet) {
    auto oxygen = BuildAtomicInputs("O", 8, 3, basisSet);

    if (!oxygen.has_value())
    {
        return std::unexpected(oxygen.error());
    }

    auto hydrogen = BuildAtomicInputs("H", 1, 2, basisSet);

    if (!hydrogen.has_value())
    {
        return std::unexpected(hydrogen.error());
    }

    std::map<int, qcx::scf::AtomicUhfInputs> atomicMap;
    atomicMap.emplace(8, std::move(*oxygen));
    atomicMap.emplace(1, std::move(*hydrogen));
    return qcx::scf::BuildSadGuess(molecule, basisSet, atomicMap);
}

// The H2/STO-3G SAD guess: H doublet fragment only.
qcx::Result<qcx::scf::SadGuess> BuildSadGuessForH2(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet) {
    auto hydrogen = BuildAtomicInputs("H", 1, 2, basisSet);

    if (!hydrogen.has_value())
    {
        return std::unexpected(hydrogen.error());
    }

    std::map<int, qcx::scf::AtomicUhfInputs> atomicMap;
    atomicMap.emplace(1, std::move(*hydrogen));
    return qcx::scf::BuildSadGuess(molecule, basisSet, atomicMap);
}

} // namespace

TEST(ChargesTest, H2oSto3gHirshfeldSumRules) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + quadrature: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto sad = BuildSadGuessForH2o(*molecule, *basis);
    ASSERT_TRUE(sad.has_value()) << sad.error().message;

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*molecule, *basis);
    ASSERT_TRUE(aoRanges.has_value()) << aoRanges.error().message;

    auto grid = qcx::grid::MolecularGrid::Create(*molecule, 60, 110);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    auto charges = qcx::properties::AnalyzeHirshfeld(
        *molecule, *basis, *grid, scf->density, sad->densityAlpha, sad->densityBeta, *aoRanges);
    ASSERT_TRUE(charges.has_value()) << charges.error().message;

    // sum_A Q_A = Z - N = 0 for the neutral molecule, at quadrature
    // resolution (the grid integrates rho to N).
    EXPECT_NEAR(charges->sum(), 0.0, 1e-3);

    // The electronegativity ordering: O carries negative charge, H
    // positive (atom order [H, H, O]).
    EXPECT_LT((*charges)(2), 0.0);
    EXPECT_GT((*charges)(0), 0.0);
    EXPECT_GT((*charges)(1), 0.0);
}

TEST(ChargesTest, H2oSto3gHirshfeldPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + quadrature: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto sad = BuildSadGuessForH2o(*molecule, *basis);
    ASSERT_TRUE(sad.has_value()) << sad.error().message;

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*molecule, *basis);
    ASSERT_TRUE(aoRanges.has_value()) << aoRanges.error().message;

    auto grid = qcx::grid::MolecularGrid::Create(*molecule, 80, 194);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    auto charges = qcx::properties::AnalyzeHirshfeld(
        *molecule, *basis, *grid, scf->density, sad->densityAlpha, sad->densityBeta, *aoRanges);
    ASSERT_TRUE(charges.has_value()) << charges.error().message;

    // Pyscf cross-check: the pin constants are loose because the grid
    // quadrature and the SAD promolecular conventions differ.
    EXPECT_NEAR((*charges)(0), kHirshfeldHRef, 2e-2);
    EXPECT_NEAR((*charges)(1), kHirshfeldHRef, 2e-2);
    EXPECT_NEAR((*charges)(2), kHirshfeldORef, 2e-2);
}

TEST(ChargesTest, H2oSto3gVoronoiSumRules) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + quadrature: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto grid = qcx::grid::MolecularGrid::Create(*molecule, 80, 194);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    auto charges = qcx::properties::AnalyzeVoronoi(*molecule, *basis, *grid, scf->density);
    ASSERT_TRUE(charges.has_value()) << charges.error().message;

    EXPECT_NEAR(charges->sum(), 0.0, 1e-3);

    // Voronoi charges the O cell POSITIVELY (the pyscf cross-check shows
    // the same pattern): each H cell extends past the O-H bisector into
    // the O density tail and so integrates more than the H electron, while
    // the O cell integrates 8 - Q_O electrons.
    EXPECT_GT((*charges)(2), 0.0);
    EXPECT_LT((*charges)(0), 0.0);
    EXPECT_LT((*charges)(1), 0.0);
}

TEST(ChargesTest, H2oSto3gVoronoiPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + quadrature: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto grid = qcx::grid::MolecularGrid::Create(*molecule, 80, 194);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    auto charges = qcx::properties::AnalyzeVoronoi(*molecule, *basis, *grid, scf->density);
    ASSERT_TRUE(charges.has_value()) << charges.error().message;

    // Pyscf cross-check, same loose tolerance rationale as the Hirshfeld
    // pins (the nearest-atom partition differs slightly between grids).
    EXPECT_NEAR((*charges)(0), kVoronoiHRef, 2e-2);
    EXPECT_NEAR((*charges)(1), kVoronoiHRef, 2e-2);
    EXPECT_NEAR((*charges)(2), kVoronoiORef, 2e-2);
}

TEST(ChargesTest, H2oSto3gHirshfeldGridTrend) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "two quadratures + real SCF: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto sad = BuildSadGuessForH2o(*molecule, *basis);
    ASSERT_TRUE(sad.has_value()) << sad.error().message;

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*molecule, *basis);
    ASSERT_TRUE(aoRanges.has_value()) << aoRanges.error().message;

    // The converged-with-grid-size trend: as the grid
    // densities, the charges must move toward the converged reference
    // (the cross-checked pins), so the fine-grid deviation from the
    // reference must be smaller than the coarse-grid deviation.
    auto coarseGrid = qcx::grid::MolecularGrid::Create(*molecule, 30, 26);
    ASSERT_TRUE(coarseGrid.has_value()) << coarseGrid.error().message;

    auto fineGrid = qcx::grid::MolecularGrid::Create(*molecule, 80, 194);
    ASSERT_TRUE(fineGrid.has_value()) << fineGrid.error().message;

    auto coarse = qcx::properties::AnalyzeHirshfeld(*molecule,
                                                    *basis,
                                                    *coarseGrid,
                                                    scf->density,
                                                    sad->densityAlpha,
                                                    sad->densityBeta,
                                                    *aoRanges);
    ASSERT_TRUE(coarse.has_value()) << coarse.error().message;

    auto fine = qcx::properties::AnalyzeHirshfeld(
        *molecule, *basis, *fineGrid, scf->density, sad->densityAlpha, sad->densityBeta, *aoRanges);
    ASSERT_TRUE(fine.has_value()) << fine.error().message;

    const std::array<double, 3> reference{kHirshfeldHRef, kHirshfeldHRef, kHirshfeldORef};

    for (std::size_t atom = 0; atom < molecule->AtomCount(); ++atom)
    {
        const double coarseDeviation =
            std::abs((*coarse)(static_cast<Eigen::Index>(atom)) - reference[atom]);
        const double fineDeviation =
            std::abs((*fine)(static_cast<Eigen::Index>(atom)) - reference[atom]);

        EXPECT_LT(fineDeviation, coarseDeviation) << "atom " << atom;
        EXPECT_LT(fineDeviation, 1e-2) << "atom " << atom;
    }
}

TEST(ChargesTest, H2Sto3gHirshfeldAndVoronoiSymmetric) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF + quadrature: Release-only";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto sad = BuildSadGuessForH2(*molecule, *basis);
    ASSERT_TRUE(sad.has_value()) << sad.error().message;

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*molecule, *basis);
    ASSERT_TRUE(aoRanges.has_value()) << aoRanges.error().message;

    auto grid = qcx::grid::MolecularGrid::Create(*molecule, 60, 110);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    auto hirshfeld = qcx::properties::AnalyzeHirshfeld(
        *molecule, *basis, *grid, scf->density, sad->densityAlpha, sad->densityBeta, *aoRanges);
    ASSERT_TRUE(hirshfeld.has_value()) << hirshfeld.error().message;

    auto voronoi = qcx::properties::AnalyzeVoronoi(*molecule, *basis, *grid, scf->density);
    ASSERT_TRUE(voronoi.has_value()) << voronoi.error().message;

    // Inversion symmetry: both atoms are equivalent, so each carries
    // exactly half of the total charge, which is zero.
    EXPECT_NEAR((*hirshfeld)(0), 0.0, 1e-3);
    EXPECT_NEAR((*hirshfeld)(1), 0.0, 1e-3);
    EXPECT_NEAR((*voronoi)(0), 0.0, 1e-3);
    EXPECT_NEAR((*voronoi)(1), 0.0, 1e-3);
}

TEST(ChargesTest, RejectsShapeMismatch) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;

    auto aoRanges = qcx::properties::AoIndexRangesByAtom(*molecule, *basis);
    ASSERT_TRUE(aoRanges.has_value()) << aoRanges.error().message;

    auto grid = qcx::grid::MolecularGrid::Create(*molecule, 30, 26);
    ASSERT_TRUE(grid.has_value()) << grid.error().message;

    const Eigen::MatrixXd wrongShape =
        Eigen::MatrixXd::Zero(scf->density.rows() + 1, scf->density.rows() + 1);

    auto hirshfeld = qcx::properties::AnalyzeHirshfeld(
        *molecule, *basis, *grid, wrongShape, wrongShape, wrongShape, *aoRanges);
    EXPECT_FALSE(hirshfeld.has_value());

    auto voronoi = qcx::properties::AnalyzeVoronoi(*molecule, *basis, *grid, wrongShape);
    EXPECT_FALSE(voronoi.has_value());

    // The fragment densities are validated independently of the molecular
    // density: a wrong-shaped fragment pair is rejected even when the
    // density itself is fine (the sum fragmentAlpha + fragmentBeta is
    // only reached after both pass the n x n check).
    auto badFragments = qcx::properties::AnalyzeHirshfeld(
        *molecule, *basis, *grid, scf->density, wrongShape, wrongShape, *aoRanges);
    EXPECT_FALSE(badFragments.has_value());

    // The per-atom ranges index the AO vector directly, so a range
    // leaking past n is rejected before the grid pass reads out of
    // bounds (DensitiesOnGrid's range validation).
    auto sad = BuildSadGuessForH2o(*molecule, *basis);
    ASSERT_TRUE(sad.has_value()) << sad.error().message;

    auto badRanges = *aoRanges;
    badRanges[0].functionCount += 100;

    auto outOfBoundsRanges = qcx::properties::AnalyzeHirshfeld(
        *molecule, *basis, *grid, scf->density, sad->densityAlpha, sad->densityBeta, badRanges);
    EXPECT_FALSE(outOfBoundsRanges.has_value());
}
