// EDDB tests: the Bond-Orbital Projection analysis of
// [Szczepanik2014]/[Szczepanik2017BOP] over the Gopinathan-Jug forward-root
// density (P_ort = S^{1/2} P S^{1/2}), on real converged RHF densities.
//
// What is pinned EXACTLY is what the method guarantees by construction:
//   the H2 molecule has no central atom (2 atoms, 1 neighbor each), so the
//     raw BOP layer vanishes and N_EDDB = 0 - delocalization needs at least
//     three atoms;
//   the H2O and benzene fixture geometries carry exact point-group
//     symmetry, so the converged density is symmetric and equivalent atoms
//     must receive EQUAL delocalized populations;
//   the delocalized density is symmetric positive semidefinite, its trace
//     is N_EDDB, and the per-atom diagonal block traces sum to it.
// What is pinned LOOSELY (relative) is the absolute magnitude against the
// runEDDB reference program (aromaticity.uj.edu.pl/runeddb, build
// 2026-Aug-15, run on the pyscf-generated .molden inputs of
// tools/properties/eddb_runeddb_probe.py, WSL 2026-08-26): the reference
// works in the NAO representation (its own pre-orthogonalization plus
// per-atom density diagonalization) while qcx works in the GJ forward-root
// basis. The two bases are NOT related by per-atom rotations (S^{-1/2}
// does not commute with the per-atom block rotations of the NAO two-cycle),
// so the mask texture and the projections differ: the benzene totals agree
// within ~10%, but the H2O H-H pair falls below the qcx Wiberg gate
// (3.3e-4 < 1e-3) while runEDDB's NAO basis admits it, changing the H2O
// central-atom set (1 vs 3) and hence the EDDB magnitude by ~3x. The
// per-atom reference pins therefore check magnitude class and partition
// ordering, not values; the tight pins are the numpy cross-validation
// values (tools/properties/eddb_numpy_reference.py reproduces eddb.cpp
// bit-for-bit).
#include "benzene_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/eddb.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeBenzeneSto3g;
using qcx::testing::MakeBenzeneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// The runEDDB reference pins: the 2026-Aug-15 Windows binary on the
// pyscf-generated molden inputs, WSL 2026-08-26. Relative tolerance only -
// the reference's NAO working basis and qcx's GJ forward-root basis are
// related by a congruence, not a per-atom rotation (see the file comment).
constexpr double kReferenceTolerance = 0.15;

// The numpy cross-validation values (eddb_numpy_reference.py = eddb.cpp
// bit-for-bit): these pin the implementation exactly.
constexpr double kH2oTotal = 0.013065;
constexpr double kH2oOxygen = 0.00736;
constexpr double kH2oHydrogen = 0.00285;
constexpr double kBenzeneTotal = 6.150054;
constexpr double kBenzeneCarbon = 0.991752;
constexpr double kBenzeneHydrogen = 0.033257;

// H = T + V from the one-electron engines (the dense_rhf_test.cpp pattern,
// shared with the sibling properties tests).
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

// The dense general-l RHF path (eri_dense.hpp): the EDDB tests run the real
// converged density, then analyze it.
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

// The per-spin densities from a converged RHF result (the populations.hpp /
// eddb.hpp density convention: P_sigma = C C^T, D = 2 P_alpha = 2 P_beta).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> SpinDensities(const qcx::scf::HfResult& scf) {
    const Eigen::MatrixXd perSpin = scf.density / 2.0;
    return {perSpin, perSpin};
}

// Mean of atomicPopulations over the atoms of the given element (the
// canonical order is a sort, so tests must never assume an index layout).
double MeanPopulation(const qcx::molecule::Molecule& molecule,
                      const Eigen::VectorXd& atomicPopulations,
                      int atomicNumber) {
    const auto& atoms = molecule.Atoms();
    double sum = 0.0;
    std::size_t count = 0;

    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        if (atoms[i].atomicNumber == atomicNumber)
        {
            sum += atomicPopulations(static_cast<Eigen::Index>(i));
            ++count;
        }
    }

    EXPECT_GT(count, 0u);
    return sum / static_cast<double>(count);
}

} // namespace

TEST(EddbTest, H2Sto3gHasNoDelocalization) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto [alpha, beta] = SpinDensities(*scf);
    auto result = qcx::properties::AnalyzeEddb(*molecule, *basis, alpha, beta);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Two atoms, one bond: no atom has two selected neighbors, so the raw
    // BOP layer is empty and the EDDB population vanishes identically.
    EXPECT_EQ(result->centralAtomCount, 0u);
    EXPECT_EQ(result->twoCenterOrbitalCount, 0u);
    EXPECT_EQ(result->totalPopulation, 0.0);
    EXPECT_EQ(result->atomicPopulations.size(), 2);
    EXPECT_EQ(result->atomicPopulations.sum(), 0.0);
    EXPECT_EQ(result->nobdOccupations.size(), 2);
    EXPECT_EQ(result->nobdOccupations.sum(), 0.0);
    EXPECT_EQ(result->delocalizedDensity.norm(), 0.0);
}

TEST(EddbTest, H2oSto3gInvariants) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto [alpha, beta] = SpinDensities(*scf);
    auto result = qcx::properties::AnalyzeEddb(*molecule, *basis, alpha, beta);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Delocalization exists (O is central with two H neighbors) but is a
    // fraction of the 10 electrons. Only O is central: the GJ-basis H-H
    // Wiberg index (3.3e-4) stays below the 1e-3 gate.
    EXPECT_EQ(result->centralAtomCount, 1u);
    EXPECT_EQ(result->twoCenterOrbitalCount, 2u);
    EXPECT_GT(result->totalPopulation, 0.0);
    EXPECT_LT(result->totalPopulation, 2.0);

    // The C2v-symmetric converged density: the two H atoms are equivalent,
    // so their delocalized populations match exactly, and the block traces
    // sum to the trace.
    EXPECT_NEAR(result->atomicPopulations(0), result->atomicPopulations(1), 1e-10);
    EXPECT_NEAR(result->atomicPopulations.sum(), result->totalPopulation, 1e-9);
    EXPECT_GT(result->atomicPopulations(2), result->atomicPopulations(0));

    // The delocalized density is symmetric PSD and its diagonalization is
    // the NOBD spectrum: nonnegative, largest occupation at most 2.
    EXPECT_LT((result->delocalizedDensity - result->delocalizedDensity.transpose()).norm(), 1e-12);
    EXPECT_EQ(result->nobdOccupations.size(), 7);
    EXPECT_GE(result->nobdOccupations(0), 0.0);
    EXPECT_LE(result->nobdOccupations(0), 2.0 + 1e-9);
    EXPECT_GE(result->nobdOccupations(result->nobdOccupations.size() - 1), -1e-12);
}

TEST(EddbTest, H2oSto3gRunEddbReferencePins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto [alpha, beta] = SpinDensities(*scf);
    auto result = qcx::properties::AnalyzeEddb(*molecule, *basis, alpha, beta);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The numpy cross-validation pins (eddb_numpy_reference.py reproduces
    // eddb.cpp exactly): H2O/STO-3G N_EDDB = 0.0131, per-atom [H, H, O] in
    // the canonical (Z, x, y, z) order.
    EXPECT_NEAR(result->totalPopulation, kH2oTotal, 1e-4);
    EXPECT_NEAR(result->atomicPopulations(0), kH2oHydrogen, 1e-5);
    EXPECT_NEAR(result->atomicPopulations(1), kH2oHydrogen, 1e-5);
    EXPECT_NEAR(result->atomicPopulations(2), kH2oOxygen, 1e-5);

    // runEDDB (NAO basis, 2026-Aug-15 build) on the same molecule:
    // N_EDDB = 0.03933, per-atom O 0.02359, H 0.00787. The GJ-basis mask
    // drops the H-H pair (3.3e-4 < 1e-3) that the NAO basis admits, so the
    // reference is ~3x larger; the pins check the magnitude class and the
    // partition ordering only.
    EXPECT_NEAR(result->totalPopulation, 0.03933, 0.03);
    EXPECT_NEAR(result->atomicPopulations(2), 0.02359, 0.02);
    EXPECT_NEAR(result->atomicPopulations(0), 0.00787, 0.01);
    EXPECT_GT(result->atomicPopulations(2), 2.0 * result->atomicPopulations(0));
}

TEST(EddbTest, BenzeneSto3gSymmetryAndReferencePins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    auto basis = MakeBenzeneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeBenzeneSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto [alpha, beta] = SpinDensities(*scf);
    auto result = qcx::properties::AnalyzeEddb(*molecule, *basis, alpha, beta);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The aromatic ring: substantial delocalization (a several-electron
    // fraction of the 42 electrons), every atom central through the loose
    // Wiberg gate, and the D6h-symmetric density assigns every carbon the
    // same delocalized population and every hydrogen the same (smaller)
    // one.
    EXPECT_GT(result->totalPopulation, 2.0);
    EXPECT_LT(result->totalPopulation, 42.0);
    EXPECT_EQ(result->atomicPopulations.size(), 12);
    EXPECT_EQ(result->centralAtomCount, 12u);
    EXPECT_GT(result->twoCenterOrbitalCount, 0u);

    // Grouping by element (the canonical order is a sort - the H's land
    // first for this fixture, but the test must not assume an index layout).
    double carbon = MeanPopulation(*molecule, result->atomicPopulations, 6);
    double hydrogen = MeanPopulation(*molecule, result->atomicPopulations, 1);
    EXPECT_GT(carbon, hydrogen);
    EXPECT_NEAR(result->atomicPopulations.sum(), result->totalPopulation, 1e-9);

    // The numpy cross-validation pins: per-atom C 0.99175, H 0.03326,
    // total 6.1501.
    EXPECT_NEAR(carbon, kBenzeneCarbon, 5e-4);
    EXPECT_NEAR(hydrogen, kBenzeneHydrogen, 5e-4);
    EXPECT_NEAR(result->totalPopulation, kBenzeneTotal, 1e-3);

    // The NOBD spectrum: a threefold-degenerate leading delocalization
    // orbital near doubly occupied, the tail decaying.
    EXPECT_NEAR(result->nobdOccupations(0), 1.785, 5e-3);
    EXPECT_LE(result->nobdOccupations(0), 2.0 + 1e-9);
    EXPECT_NEAR(result->nobdOccupations(0), result->nobdOccupations(1), 1e-4);
    EXPECT_NEAR(result->nobdOccupations(1), result->nobdOccupations(2), 1e-4);
    EXPECT_GT(result->nobdOccupations(2), result->nobdOccupations(3));

    // runEDDB (NAO basis) on the same molecule: N_EDDB = 5.69119, per-atom
    // C 0.93325, H 0.01528. Same mask topology (all 12 atoms central), so
    // the totals agree within ~10%; the H's are inflated ~2x in the
    // delocalized GJ basis (documented working-space difference).
    EXPECT_NEAR(result->totalPopulation, 5.69119, kReferenceTolerance * 5.69119);
    EXPECT_NEAR(carbon, 0.93325, kReferenceTolerance * 0.93325);
    EXPECT_NEAR(hydrogen, 0.01528, 0.02);
}

TEST(EddbTest, RejectsInvalidInputs) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // Wrong density shape: the H2O/STO-3G AO count is 7.
    Eigen::MatrixXd badDensity = Eigen::MatrixXd::Zero(3, 3);
    auto badShape = qcx::properties::AnalyzeEddb(*molecule, *basis, badDensity, badDensity);
    ASSERT_FALSE(badShape.has_value());
    EXPECT_EQ(badShape.error().code, qcx::ErrorCode::kInvalidArgument);

    // Mismatched spin channels are rejected too.
    auto mixed = qcx::properties::AnalyzeEddb(
        *molecule, *basis, Eigen::MatrixXd::Identity(7, 7), Eigen::MatrixXd::Zero(8, 8));
    ASSERT_FALSE(mixed.has_value());
    EXPECT_EQ(mixed.error().code, qcx::ErrorCode::kInvalidArgument);
}
