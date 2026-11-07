// ETS-NOCV energy-decomposition tests: the
// [Mitoraj2009] partition of the molecular binding energy into the
// classical electrostatic interaction of the frozen fragment densities,
// the Pauli repulsion of the antisymmetrized combined determinant, and
// the orbital relaxation resolved into the paired NOCV channels
// (nocv.hpp, on real converged RHF densities - properties links scf
// PRIVATE; the fragment SCFs run internally in the full basis).
//
// What is pinned EXACTLY is what the decomposition guarantees by
// construction:
//   the partition identity E_mol - sum_i E_frag_i = E_elstat + E_Pauli
//     + E_orb (E_Pauli absorbs the remainder by definition);
//   the electron-conservation pin tr(S Delta P) = 0 (the S-weighted trace
//     IS the density integral; the plain trace is basis-dependent) and
//     the PER-SPIN-CHANNEL idempotence D_a S D_a = D_a (the 1x projector
//     convention; the combined P_orth is S-idempotent only for closed-shell
//     fragments, whose equal channels sum to the 2x density);
//   E_orb = sum_k Delta E_orb^k over the magnitude-ordered channels with
//     |nu_k| <= 2 (the eigenvalues of the difference of two equal-rank
//     projectors are +-sin theta_i; the closed-shell-fragment spectra are
//     exactly paired, the spin-polarized ones only approximately - the
//     own-sign channel formula and the unpaired tail carry the defect
//     exactly);
//   the H2/STO-3G channel: the two H doublets in the full two-function
//     basis give the exactly paired spectrum {+1, -1} - one channel with
//     nu = 1, the textbook sigma pairing (E_orb itself is NOT the
//     physical bond term: the doublet fragment states are not
//     closed-shell-representable, and the restricted functional
//     mis-evaluates the alpha-only P_orth - see the caveat in nocv.hpp);
//   the arbitrary-density evaluator reproduces the SCF loop's energy at
//     the converged density (the ETS energy identity's anchor).
// What is pinned LOOSELY is the nonzero orbital relaxation of the
// H2O {O},{H,H} partitions in both basis sizes: the frozen O triplet
// and H2 singlet leave virtual space for the relaxation to fill (the
// magnitudes cross-validate against the numpy reference later).
//
// The unrestricted (UHF) and the two Kohn-Sham references are pinned by
// their own gates: the closed-shell limit (the unrestricted entry on
// D/2 = D/2 reproduces the closed-shell per-spin resolution; the
// unrestricted Kohn-Sham entry on the same D/2 = D/2 reproduces the
// unrestricted entry; the closed-shell Kohn-Sham entry on the
// zero-XC/c_HF = 1 seam reproduces the Hartree-Fock decomposition), the sum
// rule against each functional (channel sums = E[D] - E[P_orth]), the +-nu
// pairing of the O2/STO-3G triplet's beta spectrum, and the three-point
// transition state's trace on a real Slater functional in both the
// closed-shell and the per-spin form.
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_ccpvdz.hpp"
#include "h2o_sto3g.hpp"
#include "o2_sad_guess.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/nocv.hpp"
#include "qcx/properties/populations.hpp"
#include "qcx/scf/density_energy.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::properties::NocvEtsEntryPoint;
using qcx::testing::MakeH2oCcpvdz;
using qcx::testing::MakeH2oCcpvdzBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeO2Sto3gBasis;
using qcx::testing::MakeO2Sto3gTriplet;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

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

// The dense general-l RHF path (eri_dense.hpp): the NOCV tests run the
// real converged density, then analyze it.
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

// The full decomposition inputs of one molecule (overlap in Eigen form
// for the projector pin).
struct NocvInputs {
    qcx::Result<qcx::scf::HfResult> scf;
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd core;
    CpuTensor4 eri;
};

qcx::Result<NocvInputs> BuildNocvInputs(const qcx::molecule::Molecule& molecule,
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

    auto scf = qcx::scf::RunRhfScf(molecule, ToMatrix(*overlap), ToMatrix(*core), *eri);

    if (!scf.has_value())
    {
        return std::unexpected(scf.error());
    }

    return NocvInputs{std::move(scf), ToMatrix(*overlap), ToMatrix(*core), std::move(*eri)};
}

// The construction pins shared by every partition: the partition
// identity, the trace-zero density difference, and the projector
// idempotence of P_orth.
void ExpectConstructionPins(const qcx::properties::NocvEtsDecomposition& result,
                            const qcx::scf::HfResult& scf,
                            const Eigen::MatrixXd& overlap,
                            const qcx::molecule::Molecule& molecule,
                            const Eigen::MatrixXd& core,
                            const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri) {
    const double fragmentSum = result.fragmentEnergies.sum();
    const double binding = scf.totalEnergy - fragmentSum;
    // The identity holds exactly at the SCF fixed point; at the gate-converged
    // iterate it holds up to the last-iterate residual, which is
    // route-dependent (the GWH seed removes the zero-pair-pinned trajectory
    // alignment; observed 4.05e-9 against the 1e-9 assert). The per-batch
    // parallel Fock layout moves the cc-pVDZ fragment/supermolecule landings
    // (observed 1.14e-7), so 5e-7 kept the construction-consistency meaning at
    // the 1e-7 anchor scale. The CDIIS commutator-norm engagement gate was
    // deleted, so CDIIS extrapolates unconditionally from iteration 2 and
    // moves the loose-gate (densityTolerance 1e-6) supermolecule stopping
    // iterate; the DIIS-Fock-vs-density-evaluator residual is 7.579e-7 now
    // (3x bit-identical) vs green under 5e-7 before the change (3x), in the
    // documented default-tolerance class (8.6e-7..3.3e-6; an A/B run proved
    // the same engine moved the H2O cc-pVDZ RHF control +1.197e-6). 2e-6
    // keeps the construction-consistency meaning above the 1e-7 anchor scale
    // while covering the post-change stopping class.
    EXPECT_NEAR(result.bindingEnergy, binding, 2e-6);
    EXPECT_NEAR(result.electrostatic + result.pauli + result.orbital, binding, 2e-6);
    EXPECT_NEAR(result.orbital, result.orbitalComponents.sum(), 1e-12);

    // The unrestricted resolution: E_orb_U = sum of the per-spin channels,
    // and the spin-resolved correction identity E_orb_U - E_orb_R =
    // 1/4 Tr[(P_a - P_b) K[P_a - P_b]] (the two functionals coincide on
    // spin-symmetric densities; K[D] recovered as the exchange-free-minus-
    // full UHF Fock at (D, 0)).
    EXPECT_NEAR(result.orbitalUnrestricted,
                result.orbitalComponentsAlpha.sum() + result.orbitalComponentsBeta.sum(),
                1e-12);
    const Eigen::MatrixXd deltaSpin =
        result.orthogonalizedDensityAlpha - result.orthogonalizedDensityBeta;
    auto coulombOnly = qcx::scf::EvaluateUhfDensityEnergy(
        molecule,
        core,
        eri,
        deltaSpin,
        Eigen::MatrixXd::Zero(deltaSpin.rows(), deltaSpin.cols()),
        /*includeExchange=*/false);
    ASSERT_TRUE(coulombOnly.has_value());
    auto fullSpin = qcx::scf::EvaluateUhfDensityEnergy(
        molecule, core, eri, deltaSpin, Eigen::MatrixXd::Zero(deltaSpin.rows(), deltaSpin.cols()));
    ASSERT_TRUE(fullSpin.has_value());
    const Eigen::MatrixXd kDelta = coulombOnly->fockAlpha - fullSpin->fockAlpha;
    const double correction = 0.25 * (deltaSpin.array() * kDelta.array()).sum();
    EXPECT_NEAR(result.orbitalUnrestricted - result.orbital - correction, 0.0, 1e-9);

    // Electron-conservation pin: tr(S Delta P) = tr(S D_mol) - tr(S P_orth)
    // = N - N = 0 (tr(S D) = integral of the density = the electron count;
    // the PLAIN trace is not conserved - tr(D_mol) = 2 tr(C^T C) is
    // basis-dependent, only the S-weighted trace is the density integral).
    const Eigen::MatrixXd deltaP = scf.density - result.orthogonalizedDensity;
    EXPECT_NEAR((overlap * deltaP).trace(), 0.0, 1e-10);

    // Per-channel idempotence: D_a S D_a = D_a (the 1x projector
    // convention - the 2x closed-shell density appears only as the sum of
    // two equal channels, never as a per-channel factor). The combined
    // P_orth is NOT S-idempotent for spin-polarized fragments (the alpha
    // and beta occupied spaces overlap), so the pin runs per channel.
    const Eigen::MatrixXd alphaSquared =
        result.orthogonalizedDensityAlpha * overlap * result.orthogonalizedDensityAlpha;
    EXPECT_LT((alphaSquared - result.orthogonalizedDensityAlpha).norm(), 1e-9);

    const Eigen::MatrixXd betaSquared =
        result.orthogonalizedDensityBeta * overlap * result.orthogonalizedDensityBeta;
    EXPECT_LT((betaSquared - result.orthogonalizedDensityBeta).norm(), 1e-9);

    EXPECT_LT((result.orthogonalizedDensityAlpha + result.orthogonalizedDensityBeta -
               result.orthogonalizedDensity)
                  .norm(),
              1e-12);

    // The (+-nu_k) channels: |nu_k| <= 2 and nu_k >= -2, the bound for a
    // difference of two N-dimensional densities in the Lowdin basis
    // (holds for the sum of two channels too). The exact +-2 sin theta_i
    // PAIRING and the consequent non-negative nu_k are closed-shell
    // properties; for spin-polarized fragments the pairing is approximate,
    // so the signed pins are dropped here.
    EXPECT_TRUE((result.nocvEigenvalues.array() <= 2.0 + 1e-8).all());
    EXPECT_TRUE((result.nocvEigenvalues.array() >= -2.0 - 1e-8).all());
}

// The fragment SCFs of the ETS partition analysis run at the OPERATING rung
// (kNormal, 1e-8/1e-6) - the value set here explicitly rather than inherited
// (the defaults were loosened to 1e-8/1e-6,
// and these unit call sites — unlike the driver path, which threads the input
// [scf] block, run_driver.cpp:1434 — silently ran the loosened defaults,
// shifting the pins by ~5e-6). The gate was 1e-10/1e-10, an off-ladder
// equal-legs pair: it is RETIRED (no
// convergence gate at or below 1e-10). The ~5e-6 movement above is the
// finding this site carried. It is now answered (2026-09-20), and the answer
// is that the PINS stay and the BANDS move, for a reason that is checkable
// rather than convenient: tools/properties/nocv_ref.py, the independent
// reference these pins are cross-validated against, runs its molecular and
// fragment SCFs at pyscf conv_tol = 1e-12 - the reference side of this
// comparison is CONVERGED, not gated. So the pinned values are the
// reference-confirmed ones, not a record of where our own gate happened to
// stop, and re-taking them from our rung landing would replace the
// cross-validation with a comparison of our landing against itself. What the
// rung moves is the OTHER side of the comparison: each fragment SCF now
// stops where a 1e-6 density leg stops it, and every component of the
// decomposition is a functional of those stopped densities, so every pin
// carries that landing as a floor. Measured here at the rung on this tree
// against these pins (MSVC Release, 12 threads): worst 2.17e-7 over the
// STO-3G partition and 5.80e-7 over the cc-pVDZ one. The magnitude bands are
// therefore stated at 2e-6 - twice the gate's own density leg, and the same
// number this file's construction pins already carry for the same rung-era
// stopping class - where they had been left at the 1e-7/5e-7/1e-6 calibrated
// at the retired gate. Plainly: the magnitude bands are LOOSENED, the pins
// are NOT re-taken, and the construction identities in
// ExpectConstructionPins keep their own bands. The driver test fixture
// kH2oNocvToml is the other home of this gate.
const qcx::scf::UhfOptions kNocvFragmentOptions = [] {
    qcx::scf::UhfOptions options;
    options.energyTolerance = 1e-8;
    options.densityTolerance = 1e-6;
    return options;
}();

// A comparison between TWO independently converged decompositions carries both
// sides' stopping landings, so its floor is the fragment gate twice over. The
// Kohn-Sham/Hartree-Fock limit test converges both sides below its asserted
// bands instead of widening them: the fragments are STO-3G atoms, and a
// 1e-10 energy over a 1e-9 density leg costs a handful of iterations. The
// operating tight rung's 1e-8 density leg left a 2.33e-6 electrostatic
// difference, above the 2e-6 cross-path band; 1e-9 lands below it, and the
// density leg stays above the retired 1e-10/1e-10 gate.
const qcx::scf::UhfOptions kNocvTightFragmentOptions = [] {
    qcx::scf::UhfOptions options;
    options.energyTolerance = 1e-10;
    options.densityTolerance = 1e-9;
    options.maxIterations = 300;
    return options;
}();

// The reference words, short enough for the call sites.
constexpr qcx::properties::NocvEtsReference kClosedShellReference =
    qcx::properties::NocvEtsReference::kClosedShell;
constexpr qcx::properties::NocvEtsReference kUnrestrictedReference =
    qcx::properties::NocvEtsReference::kUnrestricted;
constexpr qcx::properties::NocvEtsReference kRestrictedOpenShellReference =
    qcx::properties::NocvEtsReference::kRestrictedOpenShell;

TEST(NocvEtsTest, H2Sto3gExactlyPairedChannel) {
    auto molecule = MakeH2Sto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    // The two identical H doublets in the full two-function basis: the
    // combined ALPHA occupied space spans the whole space, so P_orth =
    // S^-1 and Delta P = D_mol - S^-1 has the EXACTLY paired spectrum
    // {+1, -1} in the Lowdin basis: one NOCV channel with nu = 1, the
    // textbook sigma pairing of the two 1s orbitals (the closed-shell
    // two-projector limit). The pins are the exact pairing, the exact
    // channel resolution, and the construction pins - NOT the magnitude
    // of E_orb (the restricted functional on the alpha-only density, see
    // the caveat in nocv.hpp).
    auto result = qcx::properties::AnalyzeNocvEts(*molecule,
                                                  *basis,
                                                  inputs->core,
                                                  inputs->eri,
                                                  inputs->scf->density,
                                                  {{0}, {1}},
                                                  kNocvFragmentOptions);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_EQ(result->orbitalComponents.size(), 1);
    EXPECT_NEAR(result->nocvEigenvalues(0), 1.0, 1e-8);
    EXPECT_NEAR(result->fragmentEnergies(0), result->fragmentEnergies(1), 1e-12);

    // The magnitudes, cross-validated against the independent numpy/pyscf
    // reference tools/properties/nocv_ref.py (the fragment H atoms in the
    // full two-function basis, the restricted functional on the alpha-only
    // P_orth - the caveat of nocv.hpp). The fragment UHF energies agree to
    // 1e-9; the E_elstat/E_Pauli/E_orb split sits on their difference and
    // inherits the fragment SCF convergence floor (~8e-9 measured), so
    // those pins run at 1e-7.
    EXPECT_NEAR(result->fragmentEnergies(0), -0.470988242977, 1e-9);
    EXPECT_NEAR(result->electrostatic, -0.060440900418, 1e-7);
    EXPECT_NEAR(result->pauli, 0.904261131022, 1e-7);
    EXPECT_NEAR(result->orbital, -1.018558069713, 1e-7);
    EXPECT_NEAR(result->orbitalComponents(0), -1.018558069713, 1e-7);

    // The unrestricted resolution: the alpha difference Delta D_a = D/2 -
    // S^-1 has the spectrum {0, -1} (the full-space beta difference is
    // empty), the beta {1, 0} - the per-spin halves of the spin-summed
    // {+1, -1}. The spin-resolved value is the physical sigma-bond
    // relaxation.
    EXPECT_NEAR(result->orbitalUnrestricted, -0.584906754566, 1e-7);
    EXPECT_NEAR(result->orbitalComponentsAlpha(0), -0.338485772663, 1e-7);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(0), 0.0, 1e-8);
    EXPECT_NEAR(result->orbitalComponentsBeta(0), -0.246420981902, 1e-7);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(0), 1.0, 1e-8);
    ExpectConstructionPins(
        *result, *inputs->scf, inputs->overlap, *molecule, inputs->core, inputs->eri);
}

TEST(NocvEtsTest, H2oSto3gPartitionPins) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    // The open-shell O triplet plus the H2 singlet in 7 functions: the
    // occupied spaces leave virtual room, so the orbital relaxation is
    // nonzero (and the Hund path of the fragment SCFs runs). The molecule
    // canonicalizes to [H, H, O] (Molecule::Create sorts by Z, x, y, z), so
    // the partition is O = {2}, H2 = {0, 1} in CANONICAL order.
    auto result = qcx::properties::AnalyzeNocvEts(*molecule,
                                                  *basis,
                                                  inputs->core,
                                                  inputs->eri,
                                                  inputs->scf->density,
                                                  {{2}, {0, 1}},
                                                  kNocvFragmentOptions);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_GT(std::abs(result->orbital), 1e-4);
    EXPECT_EQ(result->orbitalComponents.size(), 4); // 7 functions -> 3 +-nu pairs + the tail.

    // The magnitudes, cross-validated against the independent numpy/pyscf
    // reference tools/properties/nocv_ref.py. The beta spectrum's +-nu
    // magnitudes tie at the 1e-15 level, so the pairs are taken BY SIGN
    // (the k-th largest positive against the k-th largest-magnitude
    // negative): a magnitude-adjacent pairing would be decided by solver
    // noise and land on different channel values per code (C++ std::sort
    // vs numpy argsort). The reference applies the identical sign-pairing;
    // these are the deterministic values. The bands are 2e-6 on the
    // rung-era criterion stated at kNocvFragmentOptions (the fragment SCFs
    // run at kNormal, so every component of the decomposition inherits
    // their 1e-6 density leg as a landing floor): measured here at the
    // rung, the worst pin in this test reads 2.17e-7 away.
    EXPECT_NEAR(result->fragmentEnergies(0), -73.834215719824, 2e-6);
    EXPECT_NEAR(result->fragmentEnergies(1), -0.921901167759, 2e-6);
    EXPECT_NEAR(result->electrostatic, -0.724475877668, 2e-6);
    EXPECT_NEAR(result->pauli, 1.885101541776, 2e-6);
    EXPECT_NEAR(result->orbital, -1.367437022962, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(0), -0.9259217636957, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(1), -0.1967735230470, 2e-6);
    // The sign-paired remainder: the 0.6227 partner of the imbalanced -nu
    // side lands as the odd tail channel, the null-space pairs at ~0.
    EXPECT_NEAR(result->orbitalComponents(2), 0.0, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(3), -0.2447417362197, 2e-6);
    EXPECT_NEAR(result->nocvEigenvalues(1), 0.7754514517966, 2e-6);
    EXPECT_NEAR(result->nocvEigenvalues(2), 0.0, 2e-6);
    EXPECT_NEAR(result->nocvEigenvalues(3), 0.6227084420696, 2e-6);

    // The unrestricted resolution (the O triplet leaves the per-spin
    // spectra imbalanced: the alpha channel of the beta-only density
    // polarization carries nu = 0.3978, the beta 1.0 - the open-shell
    // pairing defect of the spin-summed spectrum). The -0.3978 alpha
    // partner is the odd tail channel, the null-space pairs at ~0.
    EXPECT_NEAR(result->orbitalUnrestricted, -0.897936277657, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(0), -0.3880788090645, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(1), 0.0, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(2), 0.0, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(3), -0.04096882842779, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(0), 0.3978282716368, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(3), -0.3978282716368, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(0), -0.3052112589200, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(1), -0.1179241614879, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(2), 0.0, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(3), -0.04575321975669, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(0), 1.0, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(1), 0.7108003397207, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(2), 0.0, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(3), 0.3776695148433, 2e-6);
    ExpectConstructionPins(
        *result, *inputs->scf, inputs->overlap, *molecule, inputs->core, inputs->eri);
}

TEST(NocvEtsTest, H2oCcpvdzLargerBasisNonzeroOrbital) {
    auto molecule = MakeH2oCcpvdz();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oCcpvdzBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    // Canonical order [H, H, O]: O = {2}, H2 = {0, 1}.  The open-shell O
    // fragment (molecular overlap, full basis) needs more than the
    // default 100-iteration budget to satisfy the SCF's formal RMS-density
    // criterion - its iterate at iteration 100 was already converged for
    // every observable (the pins below were validated from that iterate
    // before the convergence gate), so the converged-gate fix raises the
    // budget here rather than weakening the gate.
    qcx::scf::UhfOptions fragmentOptions = kNocvFragmentOptions;
    fragmentOptions.maxIterations = 200;
    auto result = qcx::properties::AnalyzeNocvEts(*molecule,
                                                  *basis,
                                                  inputs->core,
                                                  inputs->eri,
                                                  inputs->scf->density,
                                                  {{2}, {0, 1}},
                                                  fragmentOptions);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_GT(std::abs(result->orbital), 1e-3);
    EXPECT_EQ(result->orbitalComponents.size(), 12); // 24 functions -> 12 +-nu pairs.

    // The magnitudes, cross-validated against the independent numpy/pyscf
    // reference tools/properties/nocv_ref.py (the same sign-pairing as
    // H2oSto3gPartitionPins: the beta spectrum's +-nu magnitudes tie at
    // the 1e-15 level and a magnitude-adjacent pairing would be
    // noise-dependent).
    // fragmentEnergies(0) and pauli carry the cross-platform band per the
    // user's 2026-09-07 ruling (the platform-sensitive pins widen to the
    // documented cross-platform band): the re-pinned landings,
    // -74.79547332557577 and 1.655386189129473, are the deterministic MSVC
    // Release values (3/3 standalone runs, bit-identical); the boys v1.1.0
    // per-range-seed landing's within-certified-budget (3e-14)
    // band shifts flipped the O-fragment tight-gate trajectory's stopping
    // iterate), while the clang Debug CI leg lands 1.478e-7 / 1.459e-7
    // away (run 34155831480: -74.795473473409402 / 1.6553863350408733) -
    // the trajectory class spreads a few 1e-7 per platform/team-size
    // The trajectory class spreads a few 1e-7 per platform/team-size
    // stopping-iterate flip, so the strict 1e-7 gate cannot hold both legs.
    // Every magnitude pin in this test now carries the same 2e-6 band, on
    // the rung-era criterion stated at kNocvFragmentOptions (the fragment
    // SCFs run at kNormal and every decomposition component inherits their
    // 1e-6 density leg as a landing floor); this supersedes the 5e-7 that
    // had been given to fragmentEnergies(0) and pauli alone. The MSVC
    // anchors stay, and the 2e-6 is the band this file's construction pins
    // already carried for the same reason.
    EXPECT_NEAR(result->fragmentEnergies(0), -74.79547332557577, 2e-6);
    EXPECT_NEAR(result->fragmentEnergies(1), -0.999801674280, 2e-6);
    EXPECT_NEAR(result->electrostatic, -0.499398236690, 2e-6);
    EXPECT_NEAR(result->pauli, 1.655386189129473, 2e-6);
    EXPECT_NEAR(result->orbital, -1.387511669102, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(0), -0.7152319134284, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(1), -0.4571110159018, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(2), -0.1218753696263, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(3), -0.0654505293736, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(4), -0.03661672366563, 2e-6);
    // The null-space pairs land at ~0; the post-core remainder runs down to
    // the sub-milli-Hartree channels: the last two partners of the
    // nine-negative spectrum pair (extra, extra), the LARGER eigenvalue
    // carrying the + role even when both are negative (the signed nu of
    // the tail).
    EXPECT_NEAR(result->orbitalComponents(5), 0.0, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(10), -0.0002848653554205, 2e-6);
    EXPECT_NEAR(result->orbitalComponents(11), 0.009058747249369, 2e-6);
    EXPECT_NEAR(result->nocvEigenvalues(0), 1.018059963957, 2e-6);
    EXPECT_NEAR(result->nocvEigenvalues(1), 0.7897806736933, 2e-6);
    EXPECT_NEAR(result->nocvEigenvalues(10), -5.633531775580e-07, 2e-6);
    EXPECT_NEAR(result->nocvEigenvalues(11), -0.001066051242712, 2e-6);

    // The unrestricted resolution (per-spin leading channels; the beta
    // spectrum carries the open-shell nu = 1.0 of the O-triplet hole).
    EXPECT_NEAR(result->orbitalUnrestricted, -0.929213938171, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(0), -0.2158400077125, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(0), 0.3435101792597, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(1), 0.1452083332008, 2e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(0), -0.3290592962919, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(0), 1.0, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(1), 0.8043036618282, 2e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(2), 0.3195756040997, 2e-6);
    ExpectConstructionPins(
        *result, *inputs->scf, inputs->overlap, *molecule, inputs->core, inputs->eri);
}

TEST(NocvEtsTest, RejectsInvalidPartitions) {
    auto molecule = MakeH2Sto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    // Overlapping groups.
    auto overlapPartition = qcx::properties::AnalyzeNocvEts(
        *molecule, *basis, inputs->core, inputs->eri, inputs->scf->density, {{0}, {0}});
    ASSERT_FALSE(overlapPartition.has_value());
    EXPECT_EQ(overlapPartition.error().code, qcx::ErrorCode::kInvalidArgument);

    // Out-of-range atom index.
    auto outOfRange = qcx::properties::AnalyzeNocvEts(
        *molecule, *basis, inputs->core, inputs->eri, inputs->scf->density, {{0}, {7}});
    ASSERT_FALSE(outOfRange.has_value());
    EXPECT_EQ(outOfRange.error().code, qcx::ErrorCode::kInvalidArgument);

    // Incomplete coverage (an atom belongs to no fragment).
    auto incomplete = qcx::properties::AnalyzeNocvEts(
        *molecule, *basis, inputs->core, inputs->eri, inputs->scf->density, {{0}});
    ASSERT_FALSE(incomplete.has_value());
    EXPECT_EQ(incomplete.error().code, qcx::ErrorCode::kInvalidArgument);

    // A density that does not match the ERI function count.
    const Eigen::MatrixXd mismatched = Eigen::MatrixXd::Identity(3, 3);
    auto badDensity = qcx::properties::AnalyzeNocvEts(
        *molecule, *basis, inputs->core, inputs->eri, mismatched, {{2}, {0, 1}});
    ASSERT_FALSE(badDensity.has_value());
    EXPECT_EQ(badDensity.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The convergence-gate coverage gap: the fragment SCFs feed the
// decomposition, so a fragment run that exhausts its budget (returns the
// last iterate with converged == false) must be rejected, not silently
// analyzed as if it were the fragment ground state.
TEST(NocvEtsTest, UnconvergedFragmentScfIsRejected) {
    auto molecule = MakeH2Sto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    qcx::scf::UhfOptions tinyBudget;
    tinyBudget.maxIterations = 1;

    auto result = qcx::properties::AnalyzeNocvEts(
        *molecule, *basis, inputs->core, inputs->eri, inputs->scf->density, {{0}, {1}}, tinyBudget);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(NocvEtsTest, EvaluatorMatchesScfEnergy) {
    // The ETS energy identity anchors on the arbitrary-density evaluator
    // reproducing the SCF loop's energy at the converged density.
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    auto evaluated = qcx::scf::EvaluateRhfDensityEnergy(
        *molecule, inputs->core, inputs->eri, inputs->scf->density);

    ASSERT_TRUE(evaluated.has_value());

    // Not bit-for-bit: the SCF loop's final Fock is the DIIS-extrapolated
    // one, while the evaluator rebuilds F at the converged density - the
    // residue is ~1e-11, far below the decomposition's 1e-9 pins.
    EXPECT_NEAR(evaluated->totalEnergy, inputs->scf->totalEnergy, 1e-9);
    EXPECT_NEAR(evaluated->electronicEnergy, inputs->scf->electronicEnergy, 1e-9);
}

// ---- the unrestricted (UHF) and Kohn-Sham references ---------------------

// The UHF total energy of a density pair through the arbitrary-density
// evaluator (the ETS identity's right-hand side).
qcx::Result<double> UhfTotalEnergy(const qcx::molecule::Molecule& molecule,
                                   const Eigen::MatrixXd& core,
                                   const CpuTensor4& eri,
                                   const Eigen::MatrixXd& densityAlpha,
                                   const Eigen::MatrixXd& densityBeta) {
    auto evaluated =
        qcx::scf::EvaluateUhfDensityEnergy(molecule, core, eri, densityAlpha, densityBeta);

    if (!evaluated.has_value())
    {
        return std::unexpected(evaluated.error());
    }

    return evaluated->totalEnergy;
}

// The Hartree-Fock limit of the Kohn-Sham seam: zero XC energy and potential
// with c_HF = 1, which makes the seam's Fock and energy the Hartree-Fock
// ones. The closed-shell-limit gate runs the Kohn-Sham entry point on it and
// compares against AnalyzeNocvEts.
qcx::properties::NocvEtsFunctional ZeroXcFunctional() {
    qcx::properties::NocvEtsFunctional functional;
    functional.exchangeFraction = 1.0;
    functional.evaluate =
        [](const Eigen::MatrixXd& densityAlpha,
           const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        qcx::grid::XcEvaluation evaluation;
        evaluation.energy = 0.0;
        evaluation.potentialAlpha = Eigen::MatrixXd::Zero(densityAlpha.rows(), densityAlpha.cols());
        evaluation.potentialBeta = Eigen::MatrixXd::Zero(densityBeta.rows(), densityBeta.cols());
        return evaluation;
    };
    return functional;
}

// The grid engine's pair evaluation as the module's functional seam. The
// callback holds a POINTER to the caller-owned engine, so the seam outlives
// this frame exactly as the caller's engine does.
qcx::properties::NocvEtsFunctional EngineFunctional(const qcx::grid::XcGridEngine& engine) {
    qcx::properties::NocvEtsFunctional functional;
    functional.exchangeFraction = engine.ExchangeFraction();
    const qcx::grid::XcGridEngine* enginePointer = &engine;
    functional.evaluate =
        [enginePointer](
            const Eigen::MatrixXd& densityAlpha,
            const Eigen::MatrixXd& densityBeta) -> qcx::Result<qcx::grid::XcEvaluation> {
        return enginePointer->Evaluate(densityAlpha, densityBeta);
    };
    return functional;
}

// The closed-shell Kohn-Sham Fock matrix and total energy at an arbitrary
// density, recomputed in the test from the dense evaluator and the seam -
// the independent definition the three-point trace check compares against.
struct KsDensity {
    Eigen::MatrixXd fock;
    double totalEnergy = 0.0;
};

qcx::Result<KsDensity> EvaluateTestKs(const qcx::molecule::Molecule& molecule,
                                      const Eigen::MatrixXd& core,
                                      const CpuTensor4& eri,
                                      const Eigen::MatrixXd& density,
                                      const qcx::properties::NocvEtsFunctional& functional) {
    auto full = qcx::scf::EvaluateRhfDensityEnergy(molecule, core, eri, density);

    if (!full.has_value())
    {
        return std::unexpected(full.error());
    }

    auto coulomb = qcx::scf::EvaluateRhfDensityEnergy(molecule, core, eri, density, false);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    auto xc = functional.evaluate(0.5 * density, 0.5 * density);

    if (!xc.has_value())
    {
        return std::unexpected(xc.error());
    }

    const double fraction = functional.exchangeFraction;
    KsDensity result;
    result.fock = fraction * full->fock + (1.0 - fraction) * coulomb->fock +
                  0.5 * (xc->potentialAlpha + xc->potentialBeta);
    result.totalEnergy =
        fraction * full->totalEnergy + (1.0 - fraction) * coulomb->totalEnergy + xc->energy;
    return result;
}

// The unrestricted Kohn-Sham Fock pair and total energy at an arbitrary
// spin-density pair, recomputed in the test from the dense evaluator and the
// seam - the independent definition the per-spin three-point trace check
// compares against (the per-spin twin of EvaluateTestKs).
struct KsSpinDensity {
    Eigen::MatrixXd fockAlpha;
    Eigen::MatrixXd fockBeta;
    double totalEnergy = 0.0;
};

qcx::Result<KsSpinDensity> EvaluateTestKsSpin(
    const qcx::molecule::Molecule& molecule,
    const Eigen::MatrixXd& core,
    const CpuTensor4& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    const qcx::properties::NocvEtsFunctional& functional) {
    auto full = qcx::scf::EvaluateUhfDensityEnergy(molecule, core, eri, densityAlpha, densityBeta);

    if (!full.has_value())
    {
        return std::unexpected(full.error());
    }

    auto coulomb =
        qcx::scf::EvaluateUhfDensityEnergy(molecule, core, eri, densityAlpha, densityBeta, false);

    if (!coulomb.has_value())
    {
        return std::unexpected(coulomb.error());
    }

    auto xc = functional.evaluate(densityAlpha, densityBeta);

    if (!xc.has_value())
    {
        return std::unexpected(xc.error());
    }

    const double fraction = functional.exchangeFraction;
    KsSpinDensity result;
    result.fockAlpha =
        fraction * full->fockAlpha + (1.0 - fraction) * coulomb->fockAlpha + xc->potentialAlpha;
    result.fockBeta =
        fraction * full->fockBeta + (1.0 - fraction) * coulomb->fockBeta + xc->potentialBeta;
    result.totalEnergy =
        fraction * full->totalEnergy + (1.0 - fraction) * coulomb->totalEnergy + xc->energy;
    return result;
}

// The RKS loop a Kohn-Sham test fixture converges: the closure's three
// callbacks over the dense evaluator and the seam (the driver's RksSeam
// shape, composed test-side).
qcx::Result<qcx::scf::HfResult> RunDenseRks(const qcx::molecule::Molecule& molecule,
                                            const Eigen::MatrixXd& overlap,
                                            const Eigen::MatrixXd& core,
                                            const CpuTensor4& eri,
                                            const qcx::properties::NocvEtsFunctional& functional) {
    const auto fock = [&](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto evaluated = EvaluateTestKs(molecule, core, eri, density, functional);

        if (!evaluated.has_value())
        {
            return std::unexpected(evaluated.error());
        }

        return Eigen::MatrixXd(evaluated->fock);
    };
    const auto coulomb = [&](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto evaluated = qcx::scf::EvaluateRhfDensityEnergy(molecule, core, eri, density, false);

        if (!evaluated.has_value())
        {
            return std::unexpected(evaluated.error());
        }

        return Eigen::MatrixXd(evaluated->fock - core);
    };
    const auto contribution = [&](const Eigen::MatrixXd& density) -> qcx::Result<double> {
        auto full = qcx::scf::EvaluateRhfDensityEnergy(molecule, core, eri, density);

        if (!full.has_value())
        {
            return std::unexpected(full.error());
        }

        auto coulombOnly = qcx::scf::EvaluateRhfDensityEnergy(molecule, core, eri, density, false);

        if (!coulombOnly.has_value())
        {
            return std::unexpected(coulombOnly.error());
        }

        auto xc = functional.evaluate(0.5 * density, 0.5 * density);

        if (!xc.has_value())
        {
            return std::unexpected(xc.error());
        }

        // K[D]/2 = (H + J) - F_HF, so the exact-exchange piece
        // -c_HF 1/2 Tr[D K[D]/2] reads -0.5 c_HF Tr[D (F_coul - F_HF)].
        const double exchangeTrace = density.cwiseProduct(coulombOnly->fock - full->fock).sum();
        return xc->energy - 0.5 * functional.exchangeFraction * exchangeTrace;
    };

    const qcx::scf::RhfOptions options;
    return qcx::scf::RunRhfScf(molecule, overlap, core, options, fock, coulomb, contribution);
}

// The unrestricted entry point on a closed-shell density pair must reproduce
// the closed-shell entry point's per-spin resolution exactly: same fragment
// SCFs, same projectors, same transition Focks, same channels. The E_Pauli /
// E_orb SPLIT is not comparable (the restricted functional mis-evaluates a
// spin-polarized P_orth), but the two splits sum to the same binding energy.
TEST(NocvEtsTest, UnrestrictedReproducesTheClosedShellResolution) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    auto restricted = qcx::properties::AnalyzeNocvEts(*molecule,
                                                      *basis,
                                                      inputs->core,
                                                      inputs->eri,
                                                      inputs->scf->density,
                                                      {{2}, {0, 1}},
                                                      kNocvFragmentOptions);

    ASSERT_TRUE(restricted.has_value()) << restricted.error().message;

    const Eigen::MatrixXd half = 0.5 * inputs->scf->density;
    auto unrestricted = qcx::properties::AnalyzeNocvEtsUnrestricted(*molecule,
                                                                    *basis,
                                                                    inputs->core,
                                                                    inputs->eri,
                                                                    half,
                                                                    half,
                                                                    kUnrestrictedReference,
                                                                    {{2}, {0, 1}},
                                                                    kNocvFragmentOptions);

    ASSERT_TRUE(unrestricted.has_value()) << unrestricted.error().message;

    // Same fragment runs: the shared reference's energies are identical.
    ASSERT_EQ(unrestricted->fragmentEnergies.size(), restricted->fragmentEnergies.size());

    for (Eigen::Index i = 0; i < restricted->fragmentEnergies.size(); ++i)
    {
        EXPECT_NEAR(unrestricted->fragmentEnergies(i), restricted->fragmentEnergies(i), 1e-12);
    }

    EXPECT_NEAR(unrestricted->electrostatic, restricted->electrostatic, 1e-12);

    // The per-spin resolution is the SAME computation on both sides.
    EXPECT_NEAR(unrestricted->orbitalUnrestricted, restricted->orbitalUnrestricted, 1e-12);
    ASSERT_EQ(unrestricted->orbitalComponentsAlpha.size(),
              restricted->orbitalComponentsAlpha.size());
    ASSERT_EQ(unrestricted->orbitalComponentsBeta.size(), restricted->orbitalComponentsBeta.size());

    for (Eigen::Index k = 0; k < restricted->orbitalComponentsAlpha.size(); ++k)
    {
        EXPECT_NEAR(
            unrestricted->orbitalComponentsAlpha(k), restricted->orbitalComponentsAlpha(k), 1e-12);
        EXPECT_NEAR(
            unrestricted->nocvEigenvaluesAlpha(k), restricted->nocvEigenvaluesAlpha(k), 1e-12);
        EXPECT_NEAR(unrestricted->nocvEigenvaluesAlphaNegative(k),
                    restricted->nocvEigenvaluesAlphaNegative(k),
                    1e-12);
    }

    for (Eigen::Index k = 0; k < restricted->orbitalComponentsBeta.size(); ++k)
    {
        EXPECT_NEAR(
            unrestricted->orbitalComponentsBeta(k), restricted->orbitalComponentsBeta(k), 1e-12);
        EXPECT_NEAR(
            unrestricted->nocvEigenvaluesBeta(k), restricted->nocvEigenvaluesBeta(k), 1e-12);
        EXPECT_NEAR(unrestricted->nocvEigenvaluesBetaNegative(k),
                    restricted->nocvEigenvaluesBetaNegative(k),
                    1e-12);
    }

    // The total deformation density of the unrestricted entry is the sum of
    // its spin problems, and it is the closed-shell difference on D/2.
    EXPECT_LT((unrestricted->deformationDensity - restricted->deformationDensity).norm(), 1e-12);
    EXPECT_LT((unrestricted->deformationDensityAlpha - restricted->deformationDensityAlpha).norm(),
              1e-12);
    EXPECT_LT((unrestricted->deformationDensityBeta - restricted->deformationDensityBeta).norm(),
              1e-12);

    // The binding energy is method-independent; the Pauli/orbital split is
    // the restricted-vs-spin-resolved functional difference.
    EXPECT_NEAR(unrestricted->bindingEnergy, restricted->bindingEnergy, 1e-9);
    EXPECT_NEAR(
        unrestricted->pauli + unrestricted->orbital, restricted->pauli + restricted->orbital, 1e-9);

    // The unrestricted reference has no spin-summed NOCV problem.
    EXPECT_EQ(unrestricted->orbitalComponents.size(), 0);
    EXPECT_EQ(unrestricted->nocvEigenvalues.size(), 0);
    EXPECT_NEAR(unrestricted->orbital, unrestricted->orbitalUnrestricted, 1e-12);
}

// The sum rule against the functional itself: the per-spin channel sums are
// exactly E_UHF[D] - E_UHF[P_orth], and the binding energy is exactly the
// molecular UHF energy minus the fragment energies (up to the molecular
// loop's DIIS residue, the documented 2e-6 class).
TEST(NocvEtsTest, UnrestrictedSumRuleMatchesTheUhfFunctional) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    const Eigen::MatrixXd half = 0.5 * inputs->scf->density;
    auto result = qcx::properties::AnalyzeNocvEtsUnrestricted(*molecule,
                                                              *basis,
                                                              inputs->core,
                                                              inputs->eri,
                                                              half,
                                                              half,
                                                              kUnrestrictedReference,
                                                              {{2}, {0, 1}},
                                                              kNocvFragmentOptions);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    auto molecular = UhfTotalEnergy(*molecule, inputs->core, inputs->eri, half, half);

    ASSERT_TRUE(molecular.has_value());

    auto orthogonalized = UhfTotalEnergy(*molecule,
                                         inputs->core,
                                         inputs->eri,
                                         result->orthogonalizedDensityAlpha,
                                         result->orthogonalizedDensityBeta);

    ASSERT_TRUE(orthogonalized.has_value());

    EXPECT_NEAR(result->orbitalUnrestricted, *molecular - *orthogonalized, 1e-10);
    EXPECT_NEAR(result->orbitalComponentsAlpha.sum() + result->orbitalComponentsBeta.sum(),
                result->orbitalUnrestricted,
                1e-12);
    EXPECT_NEAR(result->bindingEnergy, *molecular - result->fragmentEnergies.sum(), 2e-6);
}

// The open-shell O2/STO-3G triplet decomposition: two independent per-spin
// NOCV problems on real UHF densities. The beta channel's spectra are
// differences of two projectors, so its sign core is exactly +-nu paired;
// the alpha fragment space covers the whole function space (P_orth,a = I in
// the Lowdin basis), so its spectrum is the rank imbalance.
TEST(NocvEtsTest, UnrestrictedOpenShellO2Partition) {
    auto molecule = MakeO2Sto3gTriplet();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeO2Sto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto uhf = qcx::properties::testing::RunDenseUhfO2(*molecule, *basis);

    ASSERT_TRUE(uhf.has_value()) << uhf.error().message;
    ASSERT_TRUE(uhf->converged);

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value());

    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);

    ASSERT_TRUE(eri.has_value());

    qcx::scf::UhfOptions fragmentOptions = kNocvFragmentOptions;
    fragmentOptions.maxIterations = 200;
    auto result = qcx::properties::AnalyzeNocvEtsUnrestricted(*molecule,
                                                              *basis,
                                                              ToMatrix(*core),
                                                              *eri,
                                                              uhf->densityAlpha,
                                                              uhf->densityBeta,
                                                              kUnrestrictedReference,
                                                              {{0}, {1}},
                                                              fragmentOptions);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_GT(uhf->spinSquared, 1.5);

    // The total deformation density is the sum of the two spin problems.
    const Eigen::MatrixXd spinSum = uhf->densityAlpha + uhf->densityBeta;
    EXPECT_LT((result->deformationDensity - (spinSum - result->orthogonalizedDensity)).norm(),
              1e-10);
    EXPECT_LT((result->deformationDensity -
               (result->deformationDensityAlpha + result->deformationDensityBeta))
                  .norm(),
              1e-12);

    // The sum rule against the UHF functional.
    auto molecular =
        UhfTotalEnergy(*molecule, ToMatrix(*core), *eri, uhf->densityAlpha, uhf->densityBeta);

    ASSERT_TRUE(molecular.has_value());

    auto orthogonalized = UhfTotalEnergy(*molecule,
                                         ToMatrix(*core),
                                         *eri,
                                         result->orthogonalizedDensityAlpha,
                                         result->orthogonalizedDensityBeta);

    ASSERT_TRUE(orthogonalized.has_value());

    EXPECT_NEAR(result->orbitalUnrestricted, *molecular - *orthogonalized, 1e-9);
    EXPECT_NEAR(result->orbitalComponentsAlpha.sum() + result->orbitalComponentsBeta.sum(),
                result->orbitalUnrestricted,
                1e-12);
    EXPECT_NEAR(result->bindingEnergy, *molecular - result->fragmentEnergies.sum(), 2e-6);

    // The +-nu pairing of a difference of two projectors: every channel whose
    // BOTH partners are non-negligible is a genuine +- pair. The beta
    // spectrum (rank 7 against rank 6) has at least one such pair; the
    // ~0/rank-imbalance leftovers are skipped.
    ASSERT_EQ(result->nocvEigenvaluesBeta.size(), result->nocvEigenvaluesBetaNegative.size());
    int genuinePairs = 0;

    for (Eigen::Index k = 0; k < result->nocvEigenvaluesBeta.size(); ++k)
    {
        const double plus = result->nocvEigenvaluesBeta(k);
        const double minus = result->nocvEigenvaluesBetaNegative(k);
        EXPECT_LE(plus, 2.0 + 1e-8);
        EXPECT_GE(plus, -2.0 - 1e-8);

        if (std::abs(plus) > 1e-6 && std::abs(minus) > 1e-6)
        {
            EXPECT_NEAR(plus + minus, 0.0, 1e-8);
            ++genuinePairs;
        }
    }

    EXPECT_GE(genuinePairs, 1);
}

// The Hartree-Fock limit of the unrestricted Kohn-Sham seam: zero XC with
// c_HF = 1 is the unrestricted Hartree-Fock functional, so the UKS entry
// point must reproduce the UHF entry point's per-spin decomposition (the
// fragment SCFs run through the callback seam instead of the supermatrix
// path - the tolerance is that seam's rounding, not a physics band).
TEST(NocvEtsTest, UnrestrictedKohnShamZeroXcReproducesTheUhfEntry) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    const Eigen::MatrixXd half = 0.5 * inputs->scf->density;
    auto uhf = qcx::properties::AnalyzeNocvEtsUnrestricted(*molecule,
                                                           *basis,
                                                           inputs->core,
                                                           inputs->eri,
                                                           half,
                                                           half,
                                                           kUnrestrictedReference,
                                                           {{2}, {0, 1}},
                                                           kNocvTightFragmentOptions);

    ASSERT_TRUE(uhf.has_value()) << uhf.error().message;

    auto uks = qcx::properties::AnalyzeNocvEtsUnrestrictedKohnSham(*molecule,
                                                                   *basis,
                                                                   inputs->core,
                                                                   inputs->eri,
                                                                   half,
                                                                   half,
                                                                   kUnrestrictedReference,
                                                                   ZeroXcFunctional(),
                                                                   {{2}, {0, 1}},
                                                                   kNocvTightFragmentOptions);

    ASSERT_TRUE(uks.has_value()) << uks.error().message;

    ASSERT_EQ(uks->fragmentEnergies.size(), uhf->fragmentEnergies.size());

    // Each path runs its OWN fragment SCFs, so the cross-path band is the
    // 2e-6 magnitude band the file's other seam gates carry.
    for (Eigen::Index i = 0; i < uhf->fragmentEnergies.size(); ++i)
    {
        EXPECT_NEAR(uks->fragmentEnergies(i), uhf->fragmentEnergies(i), 2e-6);
    }

    EXPECT_NEAR(uks->electrostatic, uhf->electrostatic, 2e-6);
    EXPECT_NEAR(uks->pauli, uhf->pauli, 2e-6);
    EXPECT_NEAR(uks->orbitalUnrestricted, uhf->orbitalUnrestricted, 2e-6);
    EXPECT_NEAR(uks->orbital, uks->orbitalUnrestricted, 1e-12);
    EXPECT_NEAR(uks->bindingEnergy, uhf->bindingEnergy, 2e-6);
    ASSERT_EQ(uks->orbitalComponentsAlpha.size(), uhf->orbitalComponentsAlpha.size());
    ASSERT_EQ(uks->orbitalComponentsBeta.size(), uhf->orbitalComponentsBeta.size());

    for (Eigen::Index k = 0; k < uhf->orbitalComponentsAlpha.size(); ++k)
    {
        EXPECT_NEAR(uks->orbitalComponentsAlpha(k), uhf->orbitalComponentsAlpha(k), 2e-6);
        EXPECT_NEAR(uks->nocvEigenvaluesAlpha(k), uhf->nocvEigenvaluesAlpha(k), 2e-6);
        EXPECT_NEAR(
            uks->nocvEigenvaluesAlphaNegative(k), uhf->nocvEigenvaluesAlphaNegative(k), 2e-6);
    }

    for (Eigen::Index k = 0; k < uhf->orbitalComponentsBeta.size(); ++k)
    {
        EXPECT_NEAR(uks->orbitalComponentsBeta(k), uhf->orbitalComponentsBeta(k), 2e-6);
        EXPECT_NEAR(uks->nocvEigenvaluesBeta(k), uhf->nocvEigenvaluesBeta(k), 2e-6);
        EXPECT_NEAR(uks->nocvEigenvaluesBetaNegative(k), uhf->nocvEigenvaluesBetaNegative(k), 2e-6);
    }

    // The per-spin reference reports no restricted resolution.
    EXPECT_EQ(uks->orbitalComponents.size(), 0);
    EXPECT_EQ(uks->nocvEigenvalues.size(), 0);
    EXPECT_NEAR(uks->orbital, uks->orbitalUnrestricted, 1e-12);
}

// The real-functional gate of the per-spin Kohn-Sham seam: the O2/STO-3G
// triplet with the Slater LDA exchange through the grid engine. Each spin's
// channels must be its own THREE-point transition state's trace - which
// differs from the one-point midpoint trace for a non-linear functional -
// and the two per-spin channel sums must be E_KS[D] - E_KS[P_orth].
TEST(NocvEtsTest, UnrestrictedKohnShamThreePointStateSlaterO2) {
    auto molecule = MakeO2Sto3gTriplet();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeO2Sto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto uhf = qcx::properties::testing::RunDenseUhfO2(*molecule, *basis);

    ASSERT_TRUE(uhf.has_value()) << uhf.error().message;
    ASSERT_TRUE(uhf->converged);

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value());

    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);

    ASSERT_TRUE(eri.has_value());

    auto engine = qcx::grid::XcGridEngine::Create(*molecule, *basis, "slater");

    ASSERT_TRUE(engine.has_value()) << engine.error().message;
    EXPECT_DOUBLE_EQ(engine->ExchangeFraction(), 0.0);

    const auto functional = EngineFunctional(*engine);
    qcx::scf::UhfOptions fragmentOptions = kNocvFragmentOptions;
    fragmentOptions.maxIterations = 300;
    auto result = qcx::properties::AnalyzeNocvEtsUnrestrictedKohnSham(*molecule,
                                                                      *basis,
                                                                      ToMatrix(*core),
                                                                      *eri,
                                                                      uhf->densityAlpha,
                                                                      uhf->densityBeta,
                                                                      kUnrestrictedReference,
                                                                      functional,
                                                                      {{0}, {1}},
                                                                      fragmentOptions);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_GT(std::abs(result->orbitalUnrestricted), 1e-4);
    EXPECT_NEAR(result->orbitalComponentsAlpha.sum() + result->orbitalComponentsBeta.sum(),
                result->orbitalUnrestricted,
                1e-12);
    EXPECT_NEAR(
        result->electrostatic + result->pauli + result->orbital, result->bindingEnergy, 1e-12);
    EXPECT_EQ(result->orbitalComponents.size(), 0);
    EXPECT_NEAR(result->orbital, result->orbitalUnrestricted, 1e-12);

    // The independent per-spin three-point trace the module's E_orb claims
    // to be: 1/6 F[P_orth] + 2/3 F[P_TS] + 1/6 F[D] on each spin. The trace
    // is basis-independent, so it reads in the AO matrices directly.
    const auto orth = EvaluateTestKsSpin(*molecule,
                                         ToMatrix(*core),
                                         *eri,
                                         result->orthogonalizedDensityAlpha,
                                         result->orthogonalizedDensityBeta,
                                         functional);

    ASSERT_TRUE(orth.has_value());

    const auto middle =
        EvaluateTestKsSpin(*molecule,
                           ToMatrix(*core),
                           *eri,
                           0.5 * (result->orthogonalizedDensityAlpha + uhf->densityAlpha),
                           0.5 * (result->orthogonalizedDensityBeta + uhf->densityBeta),
                           functional);

    ASSERT_TRUE(middle.has_value());

    const auto molecular = EvaluateTestKsSpin(
        *molecule, ToMatrix(*core), *eri, uhf->densityAlpha, uhf->densityBeta, functional);

    ASSERT_TRUE(molecular.has_value());

    const Eigen::MatrixXd fockAlphaTransition = (1.0 / 6.0) * orth->fockAlpha +
                                                (2.0 / 3.0) * middle->fockAlpha +
                                                (1.0 / 6.0) * molecular->fockAlpha;
    const Eigen::MatrixXd fockBetaTransition = (1.0 / 6.0) * orth->fockBeta +
                                               (2.0 / 3.0) * middle->fockBeta +
                                               (1.0 / 6.0) * molecular->fockBeta;
    const Eigen::MatrixXd deltaAlpha = uhf->densityAlpha - result->orthogonalizedDensityAlpha;
    const Eigen::MatrixXd deltaBeta = uhf->densityBeta - result->orthogonalizedDensityBeta;
    const double threePointTrace = deltaAlpha.cwiseProduct(fockAlphaTransition).sum() +
                                   deltaBeta.cwiseProduct(fockBetaTransition).sum();
    EXPECT_NEAR(result->orbitalUnrestricted, threePointTrace, 1e-12);

    // The midpoint-only trace is a DIFFERENT number: the XC potential really
    // is sampled at all three per-spin pairs, not just the midpoint.
    const double midpointTrace = deltaAlpha.cwiseProduct(middle->fockAlpha).sum() +
                                 deltaBeta.cwiseProduct(middle->fockBeta).sum();
    EXPECT_GT(std::abs(threePointTrace - midpointTrace), 1e-9);

    // The per-spin Simpson residual: the three-point E_orb is the exact
    // E_KS[D] - E_KS[P_orth] up to the rule's residual on this functional.
    const double exactRelaxation = molecular->totalEnergy - orth->totalEnergy;
    EXPECT_NEAR(result->orbitalUnrestricted, exactRelaxation, 1e-4);
}

// The closed-shell-limit gate of the Kohn-Sham seam: zero XC with c_HF = 1 is
// the Hartree-Fock functional, so the RKS entry point must reproduce the
// closed-shell HF decomposition's numbers (the fragment SCFs run through the
// callback seam instead of the supermatrix path - the tolerance is that
// seam's rounding, not a physics band).
TEST(NocvEtsTest, KohnShamZeroXcReproducesTheHartreeFockDecomposition) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    const auto functional = ZeroXcFunctional();
    auto rks = RunDenseRks(*molecule, inputs->overlap, inputs->core, inputs->eri, functional);

    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    ASSERT_TRUE(rks->converged);
    EXPECT_NEAR(rks->totalEnergy, inputs->scf->totalEnergy, 1e-7);

    auto restricted = qcx::properties::AnalyzeNocvEts(*molecule,
                                                      *basis,
                                                      inputs->core,
                                                      inputs->eri,
                                                      inputs->scf->density,
                                                      {{2}, {0, 1}},
                                                      kNocvTightFragmentOptions);

    ASSERT_TRUE(restricted.has_value()) << restricted.error().message;

    auto kohnSham = qcx::properties::AnalyzeNocvEtsKohnSham(*molecule,
                                                            *basis,
                                                            inputs->core,
                                                            inputs->eri,
                                                            inputs->scf->density,
                                                            kClosedShellReference,
                                                            functional,
                                                            {{2}, {0, 1}},
                                                            kNocvTightFragmentOptions);

    ASSERT_TRUE(kohnSham.has_value()) << kohnSham.error().message;

    ASSERT_EQ(kohnSham->fragmentEnergies.size(), restricted->fragmentEnergies.size());

    // The fragment energies carry the same 1e-6 stopping leg as every other
    // rung-era pin in this file; the two paths land separately.
    for (Eigen::Index i = 0; i < restricted->fragmentEnergies.size(); ++i)
    {
        EXPECT_NEAR(kohnSham->fragmentEnergies(i), restricted->fragmentEnergies(i), 2e-6);
    }

    // Each path runs its OWN fragment SCFs, and the file's rung-era rule
    // applies to both: a decomposition is a functional of the densities its
    // SCFs stopped at (1e-6 density leg), so a cross-path band is the same
    // 2e-6 magnitude band the pins above carry - not a 1e-7 band the gate
    // cannot deliver and not a 1e-12 band on a quantity linear in the
    // fragment densities.
    EXPECT_NEAR(kohnSham->electrostatic, restricted->electrostatic, 2e-6);
    EXPECT_NEAR(kohnSham->pauli, restricted->pauli, 2e-6);
    EXPECT_NEAR(kohnSham->orbital, restricted->orbital, 2e-6);
    EXPECT_NEAR(kohnSham->bindingEnergy, restricted->bindingEnergy, 2e-6);
    ASSERT_EQ(kohnSham->orbitalComponents.size(), restricted->orbitalComponents.size());

    for (Eigen::Index k = 0; k < restricted->orbitalComponents.size(); ++k)
    {
        EXPECT_NEAR(kohnSham->orbitalComponents(k), restricted->orbitalComponents(k), 2e-6);
        EXPECT_NEAR(kohnSham->nocvEigenvalues(k), restricted->nocvEigenvalues(k), 2e-6);
        EXPECT_NEAR(
            kohnSham->nocvEigenvaluesNegative(k), restricted->nocvEigenvaluesNegative(k), 2e-6);
    }

    EXPECT_NEAR(kohnSham->orbital, kohnSham->orbitalComponents.sum(), 1e-12);
    EXPECT_GT(kohnSham->deformationDensityAlpha.norm(), 0.0);
}

// The real-functional gate: H2/STO-3G with the Slater LDA exchange through
// the grid engine. The per-pair energies must be the THREE-point transition
// state's trace (1/6 F[rho_orth] + 2/3 F[rho_TS] + 1/6 F[rho_mol]), which
// differs from the one-point midpoint trace for a non-linear functional.
TEST(NocvEtsTest, KohnShamThreePointTransitionStateSlaterH2) {
    auto molecule = MakeH2Sto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    auto engine = qcx::grid::XcGridEngine::Create(*molecule, *basis, "slater");

    ASSERT_TRUE(engine.has_value()) << engine.error().message;
    EXPECT_DOUBLE_EQ(engine->ExchangeFraction(), 0.0);

    const auto functional = EngineFunctional(*engine);
    auto rks = RunDenseRks(*molecule, inputs->overlap, inputs->core, inputs->eri, functional);

    ASSERT_TRUE(rks.has_value()) << rks.error().message;
    ASSERT_TRUE(rks->converged);

    auto result = qcx::properties::AnalyzeNocvEtsKohnSham(*molecule,
                                                          *basis,
                                                          inputs->core,
                                                          inputs->eri,
                                                          rks->density,
                                                          kClosedShellReference,
                                                          functional,
                                                          {{0}, {1}},
                                                          kNocvFragmentOptions);

    ASSERT_TRUE(result.has_value()) << result.error().message;

    EXPECT_GT(std::abs(result->orbital), 1e-4);
    EXPECT_NEAR(result->orbitalComponents.sum(), result->orbital, 1e-12);
    EXPECT_NEAR(
        result->electrostatic + result->pauli + result->orbital, result->bindingEnergy, 1e-12);

    // The module's E_orb is the three-point trace, and it is NOT the
    // midpoint-only trace (the XC potential really is sampled at all three
    // densities).
    const Eigen::MatrixXd deltaP = rks->density - result->orthogonalizedDensity;
    auto fockOrth = EvaluateTestKs(
        *molecule, inputs->core, inputs->eri, result->orthogonalizedDensity, functional);
    auto fockTransition = EvaluateTestKs(*molecule,
                                         inputs->core,
                                         inputs->eri,
                                         0.5 * (result->orthogonalizedDensity + rks->density),
                                         functional);
    auto fockMolecular =
        EvaluateTestKs(*molecule, inputs->core, inputs->eri, rks->density, functional);

    ASSERT_TRUE(fockOrth.has_value());
    ASSERT_TRUE(fockTransition.has_value());
    ASSERT_TRUE(fockMolecular.has_value());

    const Eigen::MatrixXd threePointFock = (1.0 / 6.0) * fockOrth->fock +
                                           (2.0 / 3.0) * fockTransition->fock +
                                           (1.0 / 6.0) * fockMolecular->fock;
    EXPECT_NEAR(result->orbital, (deltaP.cwiseProduct(threePointFock)).sum(), 1e-10);
    EXPECT_GT(std::abs(result->orbital - (deltaP.cwiseProduct(fockTransition->fock)).sum()), 1e-9);

    // The exact Kohn-Sham binding energy is E_KS[rho_mol] - sum E_frag; the
    // three-point E_orb differs from the exact orbital relaxation by the
    // Simpson residual. Printed as evidence; the band is an order-of-
    // magnitude sanity bound, not a precision claim.
    auto molecular = EvaluateTestKs(*molecule, inputs->core, inputs->eri, rks->density, functional);
    auto orthogonalized = EvaluateTestKs(
        *molecule, inputs->core, inputs->eri, result->orthogonalizedDensity, functional);

    ASSERT_TRUE(molecular.has_value());
    ASSERT_TRUE(orthogonalized.has_value());

    const double exactBinding = molecular->totalEnergy - result->fragmentEnergies.sum();
    std::printf("Slater H2/STO-3G: E_KS = %.12f, E_orb(3pt) = %.12f, "
                "E_orb(exact) = %.12f, binding = %.12f, exact binding = %.12f\n",
                molecular->totalEnergy,
                result->orbital,
                molecular->totalEnergy - orthogonalized->totalEnergy,
                result->bindingEnergy,
                exactBinding);
    EXPECT_LT(std::abs(result->bindingEnergy - exactBinding), 5e-2);
}

// The reference refusals. The restricted open-shell reference is refused by
// name (its per-spin densities are idempotent, so the unrestricted path
// would run silently); the mismatched closed-shell/unrestricted references
// are refused with the entry point that serves them.
TEST(NocvEtsTest, RefusesUnservedReferences) {
    auto molecule = MakeH2Sto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto inputs = BuildNocvInputs(*molecule, *basis);

    ASSERT_TRUE(inputs.has_value());

    const auto functional = ZeroXcFunctional();

    auto rohfUnrestricted =
        qcx::properties::AnalyzeNocvEtsUnrestricted(*molecule,
                                                    *basis,
                                                    inputs->core,
                                                    inputs->eri,
                                                    0.5 * inputs->scf->density,
                                                    0.5 * inputs->scf->density,
                                                    kRestrictedOpenShellReference,
                                                    {{0}, {1}},
                                                    kNocvFragmentOptions);

    ASSERT_FALSE(rohfUnrestricted.has_value());
    EXPECT_EQ(rohfUnrestricted.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(rohfUnrestricted.error().message.find("restricted open-shell"), std::string::npos);
    EXPECT_NE(rohfUnrestricted.error().message.find("Fock"), std::string::npos);

    auto rohfKohnSham = qcx::properties::AnalyzeNocvEtsKohnSham(*molecule,
                                                                *basis,
                                                                inputs->core,
                                                                inputs->eri,
                                                                inputs->scf->density,
                                                                kRestrictedOpenShellReference,
                                                                functional,
                                                                {{0}, {1}},
                                                                kNocvFragmentOptions);

    ASSERT_FALSE(rohfKohnSham.has_value());
    EXPECT_EQ(rohfKohnSham.error().code, qcx::ErrorCode::kUnimplemented);

    auto closedShellUnrestricted =
        qcx::properties::AnalyzeNocvEtsUnrestricted(*molecule,
                                                    *basis,
                                                    inputs->core,
                                                    inputs->eri,
                                                    0.5 * inputs->scf->density,
                                                    0.5 * inputs->scf->density,
                                                    kClosedShellReference,
                                                    {{0}, {1}},
                                                    kNocvFragmentOptions);

    ASSERT_FALSE(closedShellUnrestricted.has_value());
    EXPECT_EQ(closedShellUnrestricted.error().code, qcx::ErrorCode::kInvalidArgument);

    auto unrestrictedKohnSham = qcx::properties::AnalyzeNocvEtsKohnSham(*molecule,
                                                                        *basis,
                                                                        inputs->core,
                                                                        inputs->eri,
                                                                        inputs->scf->density,
                                                                        kUnrestrictedReference,
                                                                        functional,
                                                                        {{0}, {1}},
                                                                        kNocvFragmentOptions);

    // The capability now exists in AnalyzeNocvEtsUnrestrictedKohnSham, so the
    // closed-shell entry's answer is a wrong-reference argument error naming
    // that entry, not a missing-capability refusal.
    ASSERT_FALSE(unrestrictedKohnSham.has_value());
    EXPECT_EQ(unrestrictedKohnSham.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(unrestrictedKohnSham.error().message.find("AnalyzeNocvEtsUnrestrictedKohnSham"),
              std::string::npos)
        << unrestrictedKohnSham.error().message;

    // The unrestricted Kohn-Sham entry's own reference refusals.
    const Eigen::MatrixXd halfDensity = 0.5 * inputs->scf->density;
    auto rohfUnrestrictedKohnSham =
        qcx::properties::AnalyzeNocvEtsUnrestrictedKohnSham(*molecule,
                                                            *basis,
                                                            inputs->core,
                                                            inputs->eri,
                                                            halfDensity,
                                                            halfDensity,
                                                            kRestrictedOpenShellReference,
                                                            functional,
                                                            {{0}, {1}},
                                                            kNocvFragmentOptions);

    ASSERT_FALSE(rohfUnrestrictedKohnSham.has_value());
    EXPECT_EQ(rohfUnrestrictedKohnSham.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(rohfUnrestrictedKohnSham.error().message.find("restricted open-shell"),
              std::string::npos);

    auto closedShellUnrestrictedKohnSham =
        qcx::properties::AnalyzeNocvEtsUnrestrictedKohnSham(*molecule,
                                                            *basis,
                                                            inputs->core,
                                                            inputs->eri,
                                                            halfDensity,
                                                            halfDensity,
                                                            kClosedShellReference,
                                                            functional,
                                                            {{0}, {1}},
                                                            kNocvFragmentOptions);

    ASSERT_FALSE(closedShellUnrestrictedKohnSham.has_value());
    EXPECT_EQ(closedShellUnrestrictedKohnSham.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(closedShellUnrestrictedKohnSham.error().message.find("AnalyzeNocvEtsKohnSham"),
              std::string::npos)
        << closedShellUnrestrictedKohnSham.error().message;

    // The shared gate answers every entry point's reference the same way the
    // entry does, which is what lets a caller refuse before the dense setup.
    const auto gateServes = [](qcx::properties::NocvEtsReference reference,
                               NocvEtsEntryPoint entryPoint) {
        return qcx::properties::CheckNocvEtsReference(reference, entryPoint).has_value();
    };

    for (const auto entryPoint : {NocvEtsEntryPoint::kClosedShellHartreeFock,
                                  NocvEtsEntryPoint::kUnrestrictedHartreeFock,
                                  NocvEtsEntryPoint::kClosedShellKohnSham,
                                  NocvEtsEntryPoint::kUnrestrictedKohnSham})
    {
        auto restrictedGate =
            qcx::properties::CheckNocvEtsReference(kRestrictedOpenShellReference, entryPoint);
        ASSERT_FALSE(restrictedGate.has_value());
        EXPECT_EQ(restrictedGate.error().code, qcx::ErrorCode::kUnimplemented);
    }

    EXPECT_TRUE(gateServes(kClosedShellReference, NocvEtsEntryPoint::kClosedShellHartreeFock));
    EXPECT_TRUE(gateServes(kClosedShellReference, NocvEtsEntryPoint::kClosedShellKohnSham));
    EXPECT_TRUE(gateServes(kUnrestrictedReference, NocvEtsEntryPoint::kUnrestrictedHartreeFock));
    EXPECT_TRUE(gateServes(kUnrestrictedReference, NocvEtsEntryPoint::kUnrestrictedKohnSham));
    EXPECT_FALSE(gateServes(kUnrestrictedReference, NocvEtsEntryPoint::kClosedShellHartreeFock));
    EXPECT_FALSE(gateServes(kClosedShellReference, NocvEtsEntryPoint::kUnrestrictedHartreeFock));
    EXPECT_FALSE(gateServes(kUnrestrictedReference, NocvEtsEntryPoint::kClosedShellKohnSham));
    EXPECT_FALSE(gateServes(kClosedShellReference, NocvEtsEntryPoint::kUnrestrictedKohnSham));

    qcx::properties::NocvEtsFunctional emptyFunctional;
    auto emptySeam = qcx::properties::AnalyzeNocvEtsKohnSham(*molecule,
                                                             *basis,
                                                             inputs->core,
                                                             inputs->eri,
                                                             inputs->scf->density,
                                                             kClosedShellReference,
                                                             emptyFunctional,
                                                             {{0}, {1}},
                                                             kNocvFragmentOptions);

    ASSERT_FALSE(emptySeam.has_value());
    EXPECT_EQ(emptySeam.error().code, qcx::ErrorCode::kInvalidArgument);

    qcx::properties::NocvEtsFunctional badFraction = ZeroXcFunctional();
    badFraction.exchangeFraction = 2.0;
    auto badFractionResult = qcx::properties::AnalyzeNocvEtsKohnSham(*molecule,
                                                                     *basis,
                                                                     inputs->core,
                                                                     inputs->eri,
                                                                     inputs->scf->density,
                                                                     kClosedShellReference,
                                                                     badFraction,
                                                                     {{0}, {1}},
                                                                     kNocvFragmentOptions);

    ASSERT_FALSE(badFractionResult.has_value());
    EXPECT_EQ(badFractionResult.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
