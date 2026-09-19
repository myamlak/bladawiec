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
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_ccpvdz.hpp"
#include "h2o_sto3g.hpp"
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
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeH2oCcpvdz;
using qcx::testing::MakeH2oCcpvdzBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
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
    // route-dependent (the GWH seed removes the pre-fix zero-pair-pinned
    // trajectory alignment; observed 4.05e-9 against the 1e-9 assert). The
    // Fix-1 per-batch parallel Fock layout moves
    // the cc-pVDZ fragment/supermolecule landings (observed 1.14e-7), so 5e-7
    // kept the construction-consistency meaning at the 1e-7 anchor scale.
    // The RHF CDIIS kDiisStartThreshold = 1e-1 commutator-norm
    // engagement gate was deleted, so CDIIS extrapolates unconditionally from
    // iteration 2 and moves the loose-gate (densityTolerance 1e-6)
    // supermolecule stopping iterate; the DIIS-Fock-vs-density-evaluator
    // residual is 7.579e-7 at HEAD (3x bit-identical) vs green under 5e-7 at
    // the pre-deletion commit (3x), in the documented tolerance-change class
    // (8.6e-7..3.3e-6; an A/B on the same engine moved H2O cc-pVDZ
    // RHF control +1.197e-6). 2e-6 keeps the construction-consistency meaning
    // above the 1e-7 anchor scale while covering that stopping class.
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
// (the exposure: the default tolerances loosened to 1e-8/1e-6,
// and these unit call sites — unlike the driver path, which threads the input
// [scf] block, run_driver.cpp:1434 — silently ran the loosened defaults,
// shifting the pins by ~5e-6). The gate was 1e-10/1e-10, an off-ladder
// equal-legs pair: it is RETIRED (no
// convergence gate at or below 1e-10). The ~5e-6 movement above is the
// finding this site carries - the recorded partition pins were taken at the
// retired gate, and the pins are NOT re-taken here. The driver test fixture
// kH2oNocvToml is the other home of this gate.
const qcx::scf::UhfOptions kNocvFragmentOptions = [] {
    qcx::scf::UhfOptions options;
    options.energyTolerance = 1e-8;
    options.densityTolerance = 1e-6;
    return options;
}();

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
    // these are the deterministic values.
    EXPECT_NEAR(result->fragmentEnergies(0), -73.834215719824, 1e-7);
    EXPECT_NEAR(result->fragmentEnergies(1), -0.921901167759, 1e-7);
    EXPECT_NEAR(result->electrostatic, -0.724475877668, 1e-7);
    EXPECT_NEAR(result->pauli, 1.885101541776, 1e-7);
    EXPECT_NEAR(result->orbital, -1.367437022962, 1e-7);
    EXPECT_NEAR(result->orbitalComponents(0), -0.9259217636957, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(1), -0.1967735230470, 1e-6);
    // The sign-paired remainder: the 0.6227 partner of the imbalanced -nu
    // side lands as the odd tail channel, the null-space pairs at ~0.
    EXPECT_NEAR(result->orbitalComponents(2), 0.0, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(3), -0.2447417362197, 1e-6);
    EXPECT_NEAR(result->nocvEigenvalues(1), 0.7754514517966, 1e-6);
    EXPECT_NEAR(result->nocvEigenvalues(2), 0.0, 1e-6);
    EXPECT_NEAR(result->nocvEigenvalues(3), 0.6227084420696, 1e-6);

    // The unrestricted resolution (the O triplet leaves the per-spin
    // spectra imbalanced: the alpha channel of the beta-only density
    // polarization carries nu = 0.3978, the beta 1.0 - the open-shell
    // pairing defect of the spin-summed spectrum). The -0.3978 alpha
    // partner is the odd tail channel, the null-space pairs at ~0.
    EXPECT_NEAR(result->orbitalUnrestricted, -0.897936277657, 1e-7);
    EXPECT_NEAR(result->orbitalComponentsAlpha(0), -0.3880788090645, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(1), 0.0, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(2), 0.0, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsAlpha(3), -0.04096882842779, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(0), 0.3978282716368, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(3), -0.3978282716368, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(0), -0.3052112589200, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(1), -0.1179241614879, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(2), 0.0, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(3), -0.04575321975669, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(0), 1.0, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(1), 0.7108003397207, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(2), 0.0, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(3), 0.3776695148433, 1e-6);
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
    // before the prop-2 gate), so the converged-gate fix raises the budget
    // here rather than weakening the gate.
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
    // fragmentEnergies(0) and pauli carry the cross-platform band (the
    // platform-sensitive pins widen to the
    // documented cross-platform band): the re-pinned landings,
    // -74.79547332557577 and 1.655386189129473, are the deterministic MSVC
    // Release values (3/3 standalone runs, bit-identical), while the O2-UHF
    // precedent holds here: the boys v1.1.0 per-range-seed landing's
    // within-certified-budget (3e-14) band shifts flipped the O-fragment
    // tight-gate trajectory's stopping
    // iterate), while the clang Debug CI leg lands 1.478e-7 / 1.459e-7
    // away (run 34155831480: -74.795473473409402 / 1.6553863350408733) -
    // the trajectory class spreads a few 1e-7 per platform/team-size
    // stopping-iterate flip, so the strict 1e-7 gate cannot hold both
    // legs. The gates widen to 5e-7 (the measured 1.48e-7 spread plus the
    // margin); the MSVC anchors stay. orbital carries the same band by the
    // same criterion rather than by a new one: the arm64 legs land 2.2e-7 and
    // 2.5e-7 away (2026-09-19, CI run 35451596080), which the 1e-7 gate it was
    // left at cannot hold - the missed application of the widening above, not
    // a separate decision. 5e-7 covers the 2.5e-7 spread with a factor of two,
    // the same margin fragmentEnergies(0) and pauli carry. The remaining pins
    // keep the 1e-7/1e-6 gates they held on every leg.
    EXPECT_NEAR(result->fragmentEnergies(0), -74.79547332557577, 5e-7);
    EXPECT_NEAR(result->fragmentEnergies(1), -0.999801674280, 1e-7);
    EXPECT_NEAR(result->electrostatic, -0.499398236690, 1e-7);
    EXPECT_NEAR(result->pauli, 1.655386189129473, 5e-7);
    EXPECT_NEAR(result->orbital, -1.387511669102, 5e-7);
    EXPECT_NEAR(result->orbitalComponents(0), -0.7152319134284, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(1), -0.4571110159018, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(2), -0.1218753696263, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(3), -0.0654505293736, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(4), -0.03661672366563, 1e-6);
    // The null-space pairs land at ~0; the post-core remainder runs down to
    // the sub-milli-Hartree channels: the last two partners of the
    // nine-negative spectrum pair (extra, extra), the LARGER eigenvalue
    // carrying the + role even when both are negative (the signed nu of
    // the tail).
    EXPECT_NEAR(result->orbitalComponents(5), 0.0, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(10), -0.0002848653554205, 1e-6);
    EXPECT_NEAR(result->orbitalComponents(11), 0.009058747249369, 1e-6);
    EXPECT_NEAR(result->nocvEigenvalues(0), 1.018059963957, 1e-6);
    EXPECT_NEAR(result->nocvEigenvalues(1), 0.7897806736933, 1e-6);
    EXPECT_NEAR(result->nocvEigenvalues(10), -5.633531775580e-07, 1e-6);
    EXPECT_NEAR(result->nocvEigenvalues(11), -0.001066051242712, 1e-6);

    // The unrestricted resolution (per-spin leading channels; the beta
    // spectrum carries the open-shell nu = 1.0 of the O-triplet hole).
    EXPECT_NEAR(result->orbitalUnrestricted, -0.929213938171, 1e-7);
    EXPECT_NEAR(result->orbitalComponentsAlpha(0), -0.2158400077125, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(0), 0.3435101792597, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesAlpha(1), 0.1452083332008, 1e-6);
    EXPECT_NEAR(result->orbitalComponentsBeta(0), -0.3290592962919, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(0), 1.0, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(1), 0.8043036618282, 1e-6);
    EXPECT_NEAR(result->nocvEigenvaluesBeta(2), 0.3195756040997, 1e-6);
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

// Coverage gap: the fragment SCFs feed the
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

} // namespace
