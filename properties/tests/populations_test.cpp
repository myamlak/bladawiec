// Population-analysis pins: Mulliken, Lowdin, Mayer, and
// Gopinathan-Jug on the H2/HF/H2O/O2 STO-3G fixtures, run on the REAL
// converged SCF densities (dense general-l ERI path) and pinned against
// pyscf 2.14.0 (WSL 2026-08-25 cross-check; conventions identical to the
// port - per-spin densities, forward S^{1/2}, the trapezoid-free Mayer
// formula).
//
// Hand checks (stated provenance):
//   - H2: both atoms equivalent by inversion symmetry, so every per-atom
//     population is exactly half; the Mayer bond order is 1.0 at this
//     level of theory (a textbook value), and the free valence is
//     identically zero for RHF (P_alpha = P_beta).
//   - O2 triplet: the Mayer bond order is 2.0 (the O2 triple bond in the
//     valence sense), the free valence 0.5 per atom, and the
//     Gopinathan-Jug diagonal 7 (five electrons per spin channel squared
//     block sums). Note: an interim numpy cross-check that reported
//     1.6019685 was WRONG - it transposed the 5x5 cross REGION instead of
//     pairing PS(mu,nu) with the full-matrix transpose mirror PS(nu,mu)
//     (the two pairings coincide only when PS is symmetric); the pairing
//     bug in ComputeMayer is what made 1.46 look
//     "measured" and 2.0 look like an assertion.
//   - The H2O off-diagonal Gopinathan-Jug H-H entry is ~3e-4, an order of
//     magnitude below the O-H entries: the H's are not bonded to each
//     other at this level of theory.
//
// Atom order note: the molecule renumbering sorts atoms by
// (Z, x, y, z), so HF is [H, F] here (pyscf's input order [F, H] is
// reversed in the pins), H2O is [H(-x), H(+x), O] (the O has the largest
// Z, so it sorts last), and O2 is [O(0,0,0), O(0,0,z)].
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "o2_sad_guess.hpp"
#include "o2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
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
using qcx::testing::ToTensor;

// The pins are density-limited: qcx and pyscf each converge to their own
// DIIS fixed point (both energy/RMS-gated), so density-derived quantities
// agree to ~1e-8 and 1e-7 is the honest precision statement (still seven
// significant digits on values in [0.3, 9.2]).
constexpr double kReferenceTolerance = 1e-7;

// H = T + V from the one-electron engines (the dense_rhf_test.cpp pattern).
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

// The dense general-l SCF path (eri_dense.hpp) - the properties tests run
// the real converged densities, then analyze them.
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

} // namespace

TEST(PopulationsTest, H2Sto3gMullikenLowdinMayerAndGj) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    // HfResult::density is the spin-summed D = 2 rho; the per-spin
    // convention is D/2 per channel (populations.hpp).
    auto populations = qcx::properties::AnalyzePopulations(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density);
    ASSERT_TRUE(populations.has_value()) << populations.error().message;

    // Inversion symmetry: each atom carries exactly half of everything.
    EXPECT_NEAR(populations->mulliken.total(0), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.total(1), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.alpha(0), 0.5, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.spin(0), 0.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.orbitalAlpha(0), 0.5, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.orbitalAlpha(1), 0.5, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.orbitalTotal(0), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.total(0), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.total(1), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.spin(0), 0.0, kReferenceTolerance);

    EXPECT_NEAR(populations->mayer.bondOrders(0, 1), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.bondOrders(1, 0), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.bondOrders(0, 0), 0.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.freeValences(0), 0.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.totalValences(0), 1.0, kReferenceTolerance);

    // Gopinathan-Jug: the off-diagonal equals the diagonal here (one s
    // function per atom, and P_ort is symmetric).
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 0), 0.5, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 1), 0.5, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(1, 1), 0.5, kReferenceTolerance);
}

TEST(PopulationsTest, HfSto3gPins) {
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

    auto populations = qcx::properties::AnalyzePopulations(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density);
    ASSERT_TRUE(populations.has_value()) << populations.error().message;

    // Atom order [H, F]; pyscf's [F, H] values reversed.
    EXPECT_NEAR(populations->mulliken.total(0), 0.7888476268, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.total(1), 9.2111523732, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.total(0), 0.8477431017, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.total(1), 9.1522568983, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.bondOrders(0, 1), 0.9554146753, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.freeValences(0), 0.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.totalValences(1), 0.9554146753, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 0), 0.3593341832, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 1), 0.4884089185, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(1, 1), 8.6638479798, kReferenceTolerance);
}

TEST(PopulationsTest, H2oSto3gPins) {
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

    auto populations = qcx::properties::AnalyzePopulations(
        *molecule, *basis, 0.5 * scf->density, 0.5 * scf->density);
    ASSERT_TRUE(populations.has_value()) << populations.error().message;

    // Atom order [H(-x), H(+x), O] (the (Z, x, y, z) sort puts the
    // H's first; pyscf's input order [O, H, H] values reversed).
    EXPECT_NEAR(populations->mulliken.total(0), 0.8168220177, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.total(1), 0.8168220177, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.total(2), 8.3663559645, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.total(0), 0.8733083820, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.total(1), 0.8733083820, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.total(2), 8.2533832360, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.bondOrders(0, 1), 0.0124910345, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.bondOrders(0, 2), 0.9539547924, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.bondOrders(1, 2), 0.9539547924, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.freeValences(1), 0.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.totalValences(0), 0.9664458268, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.totalValences(1), 0.9664458268, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.totalValences(2), 1.9079095847, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 0), 0.3813337650, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 1), 0.0003323959591, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 2), 0.4916422210, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(1, 1), 0.3813337650, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(1, 2), 0.4916422210, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(2, 2), 7.2700987940, kReferenceTolerance);
}

TEST(PopulationsTest, O2TripletPins) {
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
    auto populations =
        qcx::properties::AnalyzePopulations(*molecule, *basis, scf->densityAlpha, scf->densityBeta);
    ASSERT_TRUE(populations.has_value()) << populations.error().message;

    // 9 alpha + 7 beta electrons, split evenly across the equivalent O's.
    EXPECT_NEAR(populations->mulliken.alpha(0), 4.5, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.alpha(1), 4.5, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.beta(0), 3.5, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.beta(1), 3.5, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.spin(0), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mulliken.spin(1), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.alpha(0), 4.5, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.beta(0), 3.5, kReferenceTolerance);
    EXPECT_NEAR(populations->lowdin.spin(0), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.bondOrders(0, 1), 2.0, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.freeValences(0), 0.5017054288, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.freeValences(1), 0.5017054288, kReferenceTolerance);
    EXPECT_NEAR(populations->mayer.totalValences(0), 2.5017054288, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 0), 7.0, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(0, 1), 1.0, kReferenceTolerance);
    EXPECT_NEAR(populations->gopinathanJug.bondOrders(1, 1), 7.0, kReferenceTolerance);
}

TEST(PopulationsTest, AoRangesMatchShellLayout) {
    // The function order is atom order, then shell order, then angular
    // component (shell_pairs.hpp); the (Z, x, y, z) renumbering puts
    // H before F in HF and the x = -1.43 H before the x = +1.43 H in H2O.
    auto h2Basis = MakeSto3gBasis();
    ASSERT_TRUE(h2Basis.has_value()) << h2Basis.error().message;
    auto h2 = MakeH2Sto3g();
    ASSERT_TRUE(h2.has_value()) << h2.error().message;
    auto h2Ranges = qcx::properties::AoIndexRangesByAtom(*h2, *h2Basis);
    ASSERT_TRUE(h2Ranges.has_value()) << h2Ranges.error().message;
    EXPECT_EQ((*h2Ranges)[0].firstFunction, 0u);
    EXPECT_EQ((*h2Ranges)[0].functionCount, 1u);
    EXPECT_EQ((*h2Ranges)[1].firstFunction, 1u);
    EXPECT_EQ((*h2Ranges)[1].functionCount, 1u);

    auto hfBasis = MakeHfSto3gBasis();
    ASSERT_TRUE(hfBasis.has_value()) << hfBasis.error().message;
    auto hf = MakeHfSto3g();
    ASSERT_TRUE(hf.has_value()) << hf.error().message;
    auto hfRanges = qcx::properties::AoIndexRangesByAtom(*hf, *hfBasis);
    ASSERT_TRUE(hfRanges.has_value()) << hfRanges.error().message;
    EXPECT_EQ((*hfRanges)[0].firstFunction, 0u);
    EXPECT_EQ((*hfRanges)[0].functionCount, 1u);
    EXPECT_EQ((*hfRanges)[1].firstFunction, 1u);
    EXPECT_EQ((*hfRanges)[1].functionCount, 5u);

    auto h2oBasis = MakeH2oSto3gBasis();
    ASSERT_TRUE(h2oBasis.has_value()) << h2oBasis.error().message;
    auto h2o = MakeH2oSto3g();
    ASSERT_TRUE(h2o.has_value()) << h2o.error().message;
    auto h2oRanges = qcx::properties::AoIndexRangesByAtom(*h2o, *h2oBasis);
    ASSERT_TRUE(h2oRanges.has_value()) << h2oRanges.error().message;
    // [H(-x), H(+x), O] per the (Z, x, y, z) sort: 1 + 1 + 5 functions.
    EXPECT_EQ((*h2oRanges)[0].firstFunction, 0u);
    EXPECT_EQ((*h2oRanges)[0].functionCount, 1u);
    EXPECT_EQ((*h2oRanges)[1].firstFunction, 1u);
    EXPECT_EQ((*h2oRanges)[1].functionCount, 1u);
    EXPECT_EQ((*h2oRanges)[2].firstFunction, 2u);
    EXPECT_EQ((*h2oRanges)[2].functionCount, 5u);
}

TEST(PopulationsTest, RejectsShapeMismatches) {
    // A density that disagrees with the overlap dimension is an invalid
    // argument for every analysis.
    Eigen::MatrixXd overlap = Eigen::MatrixXd::Identity(2, 2);
    Eigen::MatrixXd wrongSize = Eigen::MatrixXd::Identity(3, 3);
    const std::vector<qcx::properties::AoRange> ranges = {{0u, 1u}, {1u, 1u}};

    auto mulliken = qcx::properties::AnalyzeMulliken(overlap, wrongSize, wrongSize, ranges);
    EXPECT_FALSE(mulliken.has_value());
    EXPECT_EQ(mulliken.error().code, qcx::ErrorCode::kInvalidArgument);

    auto lowdin = qcx::properties::AnalyzeLowdin(overlap, wrongSize, wrongSize, ranges);
    EXPECT_FALSE(lowdin.has_value());
    EXPECT_EQ(lowdin.error().code, qcx::ErrorCode::kInvalidArgument);

    auto mayer = qcx::properties::AnalyzeMayer(overlap, wrongSize, wrongSize, ranges);
    EXPECT_FALSE(mayer.has_value());
    EXPECT_EQ(mayer.error().code, qcx::ErrorCode::kInvalidArgument);

    auto gj = qcx::properties::AnalyzeGopinathanJug(overlap, wrongSize, wrongSize, ranges);
    EXPECT_FALSE(gj.has_value());
    EXPECT_EQ(gj.error().code, qcx::ErrorCode::kInvalidArgument);

    // Ranges beyond the matrix dimension are rejected too.
    const std::vector<qcx::properties::AoRange> outOfBounds = {{0u, 3u}};
    auto outOfBoundsMulliken =
        qcx::properties::AnalyzeMulliken(overlap, overlap, overlap, outOfBounds);
    EXPECT_FALSE(outOfBoundsMulliken.has_value());
    EXPECT_EQ(outOfBoundsMulliken.error().code, qcx::ErrorCode::kInvalidArgument);
}
