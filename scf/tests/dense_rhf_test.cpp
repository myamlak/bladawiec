// The three pinned SCF energies through the retained dense path: the
// general-l MD 1e matrices + BuildEriTensorGeneral
// into the unchanged RunRhfScf. H2 runs always; the HF/H2O sweeps are
// fast-mode-gated (the MSVC Debug cost of the full dense builds).
//
// Provenance: H2 -1.1167143252 has a three-chain record (mpmath
// grid SCF -1.116714325176, pyscf -1.116714325062551, pin at 1e-8).
// HF -98.570757591618 and H2O -74.962928246436 are the pyscf RHF/STO-3G
// values at the fixture geometries (hf_sto3g.hpp / h2o_sto3g.hpp),
// re-anchored 2026-08-22, pinned at 1e-5.
//
// The HF (n = 6) and H2O (n = 7) pins double as the RMS-gate regression of
// rhf.cpp: with the / n^2 divisor the density gate was ~n x too lenient and
// both converged ~1e-5 off the pins (2026-08-21, BUG-1); H2 (n = 2) is too
// small to catch it, which is why the gate bug survived.

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <gtest/gtest.h>

namespace {
namespace {

qcx::Result<qcx::scf::HfResult> RunDenseRhf(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet) {
    auto overlapTensor = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlapTensor.has_value())
    {
        return std::unexpected(overlapTensor.error());
    }

    auto kineticTensor = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kineticTensor.has_value())
    {
        return std::unexpected(kineticTensor.error());
    }

    auto nuclearTensor = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclearTensor.has_value())
    {
        return std::unexpected(nuclearTensor.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basisSet);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    const Eigen::MatrixXd overlap = qcx::testing::ToMatrix(*overlapTensor);
    const Eigen::MatrixXd coreHamiltonian =
        qcx::testing::ToMatrix(*kineticTensor) + qcx::testing::ToMatrix(*nuclearTensor);
    return qcx::scf::RunRhfScf(molecule, overlap, coreHamiltonian, *eri);
}

} // namespace

TEST(DenseRhfTest, H2Sto3gPinned) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto result = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged);
    EXPECT_NEAR(result->totalEnergy, -1.1167143252, 1e-8);
}

TEST(DenseRhfTest, HfSto3gPinned) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = qcx::testing::MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto result = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged);
    // pyscf-anchored 2026-08-22: pyscf RHF/STO-3G at this geometry gives
    // -98.570757591618; agrees with DirectRhfTest at 1e-11.
    EXPECT_NEAR(result->totalEnergy, -98.57075766, 1e-5);
}

TEST(DenseRhfTest, H2oSto3gPinned) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto result = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_TRUE(result->converged);
    // pyscf-anchored 2026-08-22: pyscf RHF/STO-3G at this geometry gives
    // -74.962928246436; agrees with DirectRhfTest at 1e-13.
    EXPECT_NEAR(result->totalEnergy, -74.96292827, 1e-5);
}

} // namespace
