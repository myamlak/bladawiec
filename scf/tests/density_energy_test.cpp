// The arbitrary-density Fock/energy evaluator tests (density_energy.hpp,
// the ETS-NOCV embedding seam): the shape-mismatch
// rejections of both evaluators, the Coulomb-only includeExchange=false
// pin (F = H + J against an explicit contraction of the ERI tensor), and
// the SCF-level energy mirror - the evaluator at a converged density
// reproduces the loop's total energy to the convergence residue.
#include "h2_sto3g.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/two_electron.hpp"
#include "qcx/scf/density_energy.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// The H2/STO-3G one-electron matrices and the dense ERI tensor, the shared
// inputs of every test below.
struct H2Inputs {
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd coreHamiltonian;
    qcx::memory::Tensor<double, 4, qcx::backend::CpuTag> eri;
};

qcx::Result<H2Inputs> BuildH2Inputs() {
    auto basis = MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto molecule = MakeH2Sto3g();

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto eri = qcx::integrals::BuildEriTensor(*molecule, *basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return H2Inputs{ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), std::move(*eri)};
}

// J_uv = sum_ws (uv|ws) D_ws, an explicit test-side contraction of the
// host-canonical ERI tensor (the pin for includeExchange=false).
Eigen::MatrixXd TestCoulomb(const CpuTensor4& eri, const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
    Eigen::MatrixXd j = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            double value = 0.0;

            for (Eigen::Index w = 0; w < n; ++w)
            {
                for (Eigen::Index s = 0; s < n; ++s)
                {
                    value += eri(mu, nu, w, s) * density(w, s);
                }
            }

            j(mu, nu) = value;
        }
    }

    return j;
}

TEST(DensityEnergyTest, ShapeMismatchIsRejected) {
    // The evaluators validate like the SCF loops: every matrix must be
    // n x n against the ERI tensor's function count. A 3x3 density cannot
    // match n = 2 and must be rejected, not silently contracted.
    auto inputs = BuildH2Inputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto rhf = qcx::scf::EvaluateRhfDensityEnergy(
        *molecule, inputs->coreHamiltonian, inputs->eri, Eigen::MatrixXd::Zero(3, 3));
    EXPECT_FALSE(rhf.has_value());
    EXPECT_EQ(rhf.error().code, qcx::ErrorCode::kInvalidArgument);

    auto uhf = qcx::scf::EvaluateUhfDensityEnergy(*molecule,
                                                  inputs->coreHamiltonian,
                                                  inputs->eri,
                                                  Eigen::MatrixXd::Identity(2, 2),
                                                  Eigen::MatrixXd::Identity(3, 3));
    EXPECT_FALSE(uhf.has_value());
    EXPECT_EQ(uhf.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(DensityEnergyTest, CoulombOnlyFockIsCorePlusCoulomb) {
    // includeExchange=false must give exactly F = H + J at the given
    // density: the classical Coulomb-only functional whose cross-fragment
    // pieces form the ETS electrostatic term. The J contraction is pinned
    // against an explicit test-side sum over the ERI tensor.
    auto inputs = BuildH2Inputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const Eigen::MatrixXd density = Eigen::MatrixXd::Constant(2, 2, 0.25);
    const Eigen::MatrixXd expected = inputs->coreHamiltonian + TestCoulomb(inputs->eri, density);

    auto result = qcx::scf::EvaluateRhfDensityEnergy(
        *molecule, inputs->coreHamiltonian, inputs->eri, density, false);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR((result->fock - expected).norm(), 0.0, 1e-12);
}

TEST(DensityEnergyTest, EvaluatorMirrorsTheConvergedScfEnergy) {
    // The evaluator at a converged SCF density reproduces the loop's own
    // Fock/energy to the convergence residue (the ETS energy-identity pin
    // depends on that): the RHF loop and the evaluator share the
    // supermatrix contraction and the 1/2 Tr[D (H + F)] bookkeeping.
    auto inputs = BuildH2Inputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto scf =
        qcx::scf::RunRhfScf(*molecule, inputs->overlap, inputs->coreHamiltonian, inputs->eri);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    ASSERT_TRUE(scf->converged);

    auto evaluated = qcx::scf::EvaluateRhfDensityEnergy(
        *molecule, inputs->coreHamiltonian, inputs->eri, scf->density);
    ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
    EXPECT_NEAR(evaluated->totalEnergy, scf->totalEnergy, 1e-9);
    EXPECT_NEAR(evaluated->electronicEnergy, scf->electronicEnergy, 1e-9);
}

TEST(DensityEnergyTest, UhfEvaluatorMirrorsTheConvergedScfEnergy) {
    // The per-spin mirror: the UHF evaluator at the converged (D_a, D_b)
    // reproduces the UHF loop's total energy to the convergence residue.
    auto inputs = BuildH2Inputs();
    ASSERT_TRUE(inputs.has_value()) << inputs.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const auto scf =
        qcx::scf::RunUhfScf(*molecule, inputs->overlap, inputs->coreHamiltonian, inputs->eri);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    ASSERT_TRUE(scf->converged);

    auto evaluated = qcx::scf::EvaluateUhfDensityEnergy(
        *molecule, inputs->coreHamiltonian, inputs->eri, scf->densityAlpha, scf->densityBeta);
    ASSERT_TRUE(evaluated.has_value()) << evaluated.error().message;
    EXPECT_NEAR(evaluated->totalEnergy, scf->totalEnergy, 1e-9);
    EXPECT_NEAR(evaluated->electronicEnergy, scf->electronicEnergy, 1e-9);
}

} // namespace
