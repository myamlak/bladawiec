#pragma once

/// \file
/// Unrestricted Hartree-Fock: independent alpha/beta spin channels over
/// the same molecular orbital basis. See rhf.hpp for the closed-shell
/// counterpart and the shared step-functions in src/internal/scf_common.hpp.

#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/scf_state.hpp"
#include "qcx/scf/symmetry_labels.hpp"

#include <Eigen/Dense>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

// The SAD guess consumes a basis set only through a reference (the
// per-atom function offsets); the full definition stays in scf/src/uhf.cpp
// (qcx-basisset is a PRIVATE dependency of qcx-scf).
namespace qcx::basisset {
class BasisSet;
} // namespace qcx::basisset

namespace qcx::scf {

/// The provisional virtual-space level-shift value (Hartree) applied per
/// spin under UhfOptions::useLevelShift: F += (S·C)·B·(S·C)^T with
/// B = shift·I and zeros on the occupied block. Provisional pending the
/// shift-value sweep of the robustness harness.
/// \ingroup qcx-scf
inline constexpr double kLevelShiftDefault = 1.0;

/// Convergence settings of one UHF run. Deliberately a separate struct
/// from RhfOptions (not a shared base) - the two may diverge (UHF-only
/// robustness knobs land here) and a shared base buys nothing today.
/// \ingroup qcx-scf
struct UhfOptions {
    int maxIterations = 100; ///< Iteration budget.
    double energyTolerance = 1e-8; ///< Total-energy change (Hartree).
    double densityTolerance = 1e-6; ///< RMS change, summed over both spins - see
                                    ///< the BuildDensity note in uhf.cpp.
    bool useDiis = true; ///< Independent CDIIS per spin channel (two DiisExtrapolator
                         ///< instances).
    bool fullGroupLabeling = true; ///< The post-SCF full-group labeling stage runs
                                   ///< per spin when the caller supplies the basis set;
                                   ///< false restores the behavior without the stage
                                   ///< bit-identically (nothing else in the run reads
                                   ///< this flag).

    /// Two-phase per-spin DIIS: a short 2-pair history until
    /// the spin's error norm drops below the phase threshold, then a
    /// one-way handoff to the full 8-pair window. Requires useDiis (the
    /// combination without it is rejected); off by default - the
    /// single-window trajectory stays bit-identical.
    bool useTwoPhaseDiis = false;

    /// Joint-system DIIS: one DIIS subspace over the
    /// combined alpha+beta error vectors - the B-matrix entries are the
    /// symmetrized sum 0.5·(<eAlpha_i, eAlpha_j> + <eBeta_i, eBeta_j>),
    /// one Lagrange constraint, and a single coefficient vector applied to
    /// both spins' Fock histories (the joint form). Replaces the two
    /// per-spin extrapolators when useDiis is also set; the two-phase cap
    /// does not combine with the joint form (the combination is rejected).
    /// Off by default - per-spin CDIIS stays the default.
    bool useJointDiis = false;

    /// Virtual-space level shift: each iteration's
    /// DIIS-extrapolated Fock gains (S·C)·B·(S·C)^T before the
    /// diagonalization - after the extrapolation and outside the energy
    /// computation and the DIIS error, so the shifted matrix never reaches
    /// the energy path and the converged state is the unshifted fixed
    /// point. The shift is skipped on the first
    /// iteration (no previous coefficients). Off by default.
    bool useLevelShift = false;

    /// Per-spin shift values (Hartree) of the virtual-space level shift;
    /// must be non-negative (0.0 is a legal no-op) - validated regardless
    /// of useLevelShift, applied only when it is set.
    double levelShiftAlpha = kLevelShiftDefault; ///< Alpha-spin shift value.
    double levelShiftBeta = kLevelShiftDefault; ///< Beta-spin shift value.

    /// Three-way convergence gate: adds the max-density-change
    /// leg (at the full densityTolerance) and the per-spin DIIS-error leg
    /// (kDiisErrorGateTolerance) to the two-way gate, and tightens the RMS
    /// density leg to densityTolerance/100. Off by default - the two-way
    /// predicate stays bit-identical.
    bool useRobustGate = false;

    /// Optional per-spin starting densities (the GWH and SAD initial
    /// guesses). Empty matrices mean the core guess P = 0; a non-empty
    /// pair must match the function count.
    Eigen::MatrixXd initialDensityAlpha;
    /// Optional per-spin starting densities; see initialDensityAlpha.
    Eigen::MatrixXd initialDensityBeta;

    /// Maximum-overlap occupation selection [Besley2009]: when
    /// true, each iteration occupies the \p momReferenceAlpha / \p
    /// momReferenceBeta orbitals most overlapping the current MOs instead
    /// of the lowest-energy ones (the target-state/broken-symmetry driver).
    bool useMom = false;
    /// Reference alpha orbitals for MOM (n x nMo); the first nAlpha
    /// columns are the reference occupied set.
    Eigen::MatrixXd momReferenceAlpha;
    /// Reference beta orbitals for MOM (n x nMo); the first nBeta columns
    /// are the reference occupied set.
    Eigen::MatrixXd momReferenceBeta;

    ScfRestartState initialScfState; ///< Optional restart seed: non-empty
                                     ///< densities/DIIS replace the initial guess.

    /// Per-iteration trace side-channel (diagnostics; the memory-anomaly
    /// investigation): when non-empty, the loop appends one
    /// `iter=<n> wall=<s> e=<E> dE=<dE> rmsd=<rms> conv=<0|1>` line per
    /// iteration after its convergence decision and flushes it. Empty
    /// (the default) keeps the zero-cost path - no file, no allocation,
    /// no numerical effect (the bit-parity pins are absolute).
    std::string traceFile;
};

/// What a run did with the point-group symmetry blocking it was in a position
/// to use (the blocked diagonalization, requested by supplying the basis
/// set to RunUhfScf). The honoured / refused-by-name / demoted vocabulary,
/// graded where the decision is made.
/// \ingroup qcx-scf
enum class SymmetryBlockingAction {
    kNotRequested, ///< No basis set was supplied, or the detected group is C1/unrealizable:
                   ///< nothing was asked and every diagonalization ran the plain solve.
    kUsed, ///< Every diagonalization ran on the irrep blocks.
    kDemoted, ///< Blocking ran for at least one diagonalization and was refused for at least
              ///< one other - the run walked both paths.
    kRefused, ///< Every diagonalization was refused: blocking was requested and never ran.
    kUnavailable, ///< The decomposition could not serve a blocked solve at all - the
                  ///< linear-dependence removal left a non-square orthogonalizer, whose
                  ///< disengagement DiagonalizeFockBlocked documents. Distinct from
                  ///< kRefused by name: no Fock was measured, no guard decision was taken.
};

/// One spin channel's symmetry-blocking DISCLOSURE: what was requested, what
/// the guard decided, and the numbers it decided on.
///
/// The guard exists because the blocked equivalence is conditional. "U^T (X^T F X) U
/// is block-diagonal up to integral noise ... equivalent to the plain n x n
/// solve" holds while the Fock commutes with the group - a property of the
/// SOLUTION. The point group is a property of the NUCLEAR FRAMEWORK; the group
/// that may legitimately constrain a solution is the invariance group OF THAT
/// SOLUTION, and an unrestricted solution may have a smaller one. Blocking a
/// Fock that has broken the group's symmetry does not accelerate that
/// solution's diagonalization - it projects the solution out, and silently:
/// the blocked path still returns a converged, variational-looking answer,
/// just the wrong one. The guard measures the precondition
/// (internal::MeasureSymmetryAdaptation) and refuses by name when it fails,
/// per spin and per iteration, because the alpha and beta Focks are different
/// matrices.
///
/// This report is the record of that decision (the number a reader can check
/// rather than take). The per-generator norms are the WORST measurement of the
/// channel, so they always correspond to maxGeneratorCommutatorNorm.
/// \ingroup qcx-scf
struct SymmetryBlockingReport {
    /// The action taken over the whole run; see SymmetryBlockingAction.
    SymmetryBlockingAction action = SymmetryBlockingAction::kNotRequested;
    /// Diagonalizations that ran on the irrep blocks.
    int blockedSolveCount = 0;
    /// Diagonalizations that ran the plain n x n solve (the guard's refusals,
    /// and the finalizer's re-diagonalization, which is counted here too).
    int plainSolveCount = 0;
    /// The relative commutator norms || [F_orth, A(g_j)] ||_F / || F_orth ||_F
    /// of the worst measurement, one per generator of the computational group.
    /// Empty when nothing was ever measured (kNotRequested, kUnavailable).
    std::vector<double> generatorCommutatorNorms;
    /// The largest relative commutator norm the channel ever measured; compare
    /// it with tolerance. 0 when nothing was measured.
    double maxGeneratorCommutatorNorm = 0.0;
    /// The threshold the norms above were compared against, recorded so the
    /// numbers are readable without the source (the guard's constant is
    /// internal to the scf module).
    double tolerance = 0.0;
};

/// Result of one UHF run.
/// \ingroup qcx-scf
struct UhfResult {
    double totalEnergy = 0.0; ///< Electronic plus nuclear repulsion (Hartree) of the densities
                              ///< this result RETURNS: the exit re-forms the trace from them,
                              ///< so it is not the DIIS-extrapolated Focks' trace of
                              ///< the last iteration that the loop reported per iterate.
    double electronicEnergy = 0.0; ///< 1/2 sum_sigma Tr[D_sigma (H + F_sigma[D])] (Hartree) at
                                   ///< the returned densities.
    bool converged = false; ///< True when a convergence criterion fired.
    int iterations = 0; ///< Iterations spent; the budget when not converged.
    double spinSquared = 0.0; ///< <S^2> spin-contamination diagnostic.

    // The converged (or last-iterate) SCF state, exposed for the
    // properties/ module - the per-spin analog of HfResult's
    // trailing fields. Deliberately distinct from the SAD-guess struct's
    // same-named fields (SadGuess, above): those are the INITIAL-guess
    // fragment embedding, these are the CONVERGED final densities. On a
    // non-converged result they hold the FINAL iterate (the converged flag
    // stays the gate), matching HfResult.
    Eigen::MatrixXd densityAlpha; ///< Converged alpha density D_alpha = C_alpha,occ
                                  ///< C_alpha,occ^T (the UhfFockBuilderFn convention - no
                                  ///< factor of 2 anywhere). The occupied MOs are the
                                  ///< returned \c coefficientsAlpha's occupied columns, as the
                                  ///< run's own occupation rule selects them (aufbau, or the
                                  ///< MOM mask) - the exit builds this matrix FROM the
                                  ///< returned coefficients, so the two describe
                                  ///< one state. Not the trajectory's final density.
    Eigen::MatrixXd densityBeta; ///< Converged beta density; see densityAlpha.
    Eigen::MatrixXd coefficientsAlpha; ///< Alpha MO coefficients (n x n) of the returned alpha
                                       ///< density: the eigenvectors of the Focks built from the
                                       ///< trajectory's returned per-spin densities (the seed
                                       ///< pair D_seed of the polish), columns ascending by
                                       ///< orbital energy - the returned densities are built from
                                       ///< THESE coefficients, so the pair is one state. Not
                                       ///< the DIIS-extrapolated Focks'. EXCEPT the two-callback
                                       ///< (seam/KS) path, which returns the loop's own
                                       ///< diagonalization as before - the same carve-out the
                                       ///< energy fix-A took.
    Eigen::MatrixXd coefficientsBeta; ///< Beta MO coefficients; see
                                      ///< coefficientsAlpha.
    Eigen::VectorXd orbitalEnergiesAlpha; ///< Eigenvalues of the seed pair's Focks, ascending:
                                          ///< the returned coefficients' orbital energies. Not the
                                          ///< final (DIIS-extrapolated) Focks' MO diagonal. Same
                                          ///< seam/KS carve-out as coefficientsAlpha.
    Eigen::VectorXd orbitalEnergiesBeta; ///< Beta orbital energies; see
                                         ///< orbitalEnergiesAlpha.

    ScfRestartState restart; ///< The final-iterate state: the checkpoint
                             ///< writer's source; on a non-converged result it holds the
                             ///< last iterate (converged stays the gate).

    std::optional<SymmetryLabels> symmetryLabelsAlpha;
    ///< The full-group labeling of the returned alpha channel (the
    ///< symmetrized alpha density D_sym,alpha feeds the properties
    ///< consumers); empty exactly like HfResult::symmetryLabels.
    std::optional<SymmetryLabels> symmetryLabelsBeta;
    ///< The beta channel's labeling; see symmetryLabelsAlpha.

    /// The convergence gate's own operands at the returned iterate, recorded
    /// for the reader and never acted on (the HfResult::achievedEnergyDelta
    /// pair, same meaning): the true |E_n - E_{n-1}| and the two-spin RMS
    /// density change the gate compared. Unlike the RHF path, BOTH of this
    /// gate's legs are live (the UHF loop runs IsUhfConverged before the
    /// previous-energy bookkeeping), so a converged result here has
    /// achievedEnergyDelta < UhfOptions::energyTolerance by construction -
    /// the recorded value is the evidence of that, not a qualification.
    double achievedEnergyDelta = 0.0;
    /// The two-spin RMS density change the gate compared ((||dD_a|| +
    /// ||dD_b||)/n) - the quantity that decided the stop beside the energy.
    double achievedRmsDensityChange = 0.0;

    /// The returned alpha state's PROJECTOR residual: the Frobenius
    /// norm ||D_alpha - C_alpha,occ C_alpha,occ^T|| between the returned alpha
    /// density and the density the returned alpha MOs define, with the occupied
    /// set the run's own rule selects (aufbau, or the MOM mask - both are 0/1
    /// masks). The per-spin HfResult::stateConsistencyResidual, and the
    /// quantity the O2 triplet pin asserted while the return path handed out
    /// one state's density beside another's coefficients: 3.4e-9 (alpha) and
    /// 1.3e-6 (beta) on that fixture, both with the run reporting converged.
    ///
    /// Zero by construction on the polished path. A fractional-occupation run
    /// (the SAD atomic fragments' averaged shells) does NOT satisfy the
    /// projector identity at all - its density is a weighted sum, not a
    /// projector - so a nonzero value there is a property of the occupation
    /// rule, not a defect signal. Same seam/KS carve-out as
    /// HfResult::stateConsistencyResidual.
    double stateConsistencyResidualAlpha = 0.0;
    /// The beta channel's projector residual; see
    /// stateConsistencyResidualAlpha.
    double stateConsistencyResidualBeta = 0.0;

    /// The returned alpha state's FIXED-POINT residual: the
    /// Frobenius norm ||D_alpha,returned - D_alpha,seed|| between the returned
    /// alpha density and the alpha density the Focks that produced the returned
    /// coefficients were built from. The per-spin
    /// HfResult::stateFixedPointResidual, and the density-space measure of the
    /// contract's remaining leg F_sigma = F_sigma[D_alpha, D_beta]. On the
    /// pre-fix path this quantity WAS the projector residual above, because the
    /// returned density was the seed.
    ///
    /// On the seam/KS carve-out path no polish runs, so the returned densities
    /// ARE the densities the returned coefficients were built with and these
    /// are zero by identity rather than by measurement - unlike the projector
    /// residuals beside them, which that path does measure.
    double stateFixedPointResidualAlpha = 0.0;
    /// The beta channel's fixed-point residual; see
    /// stateFixedPointResidualAlpha.
    double stateFixedPointResidualBeta = 0.0;

    /// The linear-dependence removal's DISCLOSURE (scf_common.hpp
    /// OrthogonalizeOverlap): how many overlap directions were dropped
    /// because they fell below kOverlapEigenvalueFloorTolerance times the
    /// largest overlap eigenvalue. Zero for every well-conditioned system,
    /// and identical to the RHF value for the same molecule and basis - the
    /// removal is a SYSTEM property (basis and molecule), not a per-method
    /// switch. See HfResult::numRemovedOverlapDirections for what a nonzero
    /// value does and does not mean - including the symmetry stages it
    /// disengages, which is the same on both paths, and the same count.
    std::size_t numRemovedOverlapDirections = 0;

    /// The alpha channel's symmetry-blocking disclosure (the blocking guard; see
    /// SymmetryBlockingReport). The per-spin pair, not one run-level flag:
    /// the alpha and beta Focks are different matrices and one may be
    /// symmetry-adapted while the other has broken the group.
    SymmetryBlockingReport symmetryBlockingAlpha;
    /// The beta channel's disclosure; see symmetryBlockingAlpha.
    SymmetryBlockingReport symmetryBlockingBeta;
};

/// Per-iteration Fock-builder callback pair for UHF (the two-Fock analog
/// of rhf.hpp's FockBuilderFn): called once per iteration with the
/// CURRENT (dAlpha, dBeta) actual per-spin densities (C_sigma,occ *
/// C_sigma,occ^T - no factor of 2 anywhere, unlike RHF's spin-summed
/// convention), returns (fockAlpha, fockBeta).
/// \ingroup qcx-scf
using UhfFockBuilderFn = std::function<qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>(
    const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)>;

/// Supplies the energy the UHF trace formula cannot cover: the two-spin analog
/// of rhf.hpp's EnergyContributionFn.
///
/// Called with BOTH per-spin densities of the iterate. For a density functional
/// this is Exc(dAlpha, dBeta) - a functional of the pair, not of the total,
/// because the spin densities enter it separately; for a hybrid it also carries
/// the exact-exchange piece -c_HF 1/2 sum_sigma Tr[d_sigma K(d_sigma)], which
/// the loop cannot form: it sees the densities and an opaque Fock builder, never
/// K. The two-spin signature is what makes the pair available: a
/// spin-summed-only callback could not express either term.
/// \ingroup qcx-scf
using UhfEnergyContributionFn =
    std::function<qcx::Result<double>(const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)>;

/// Runs UHF over a dense (uv|ws) repulsion tensor - the direct analog of
/// rhf.hpp's first RunRhfScf overload. No electron-parity restriction:
/// molecule.Multiplicity() together with molecule.ElectronCount() fixes
/// nAlpha/nBeta (see DeriveSpinOccupations in uhf.cpp); an inconsistent
/// pair (e.g. an even electron count with an even multiplicity) is
/// rejected with kInvalidArgument.
///
/// Per iteration: F_sigma = H + J(D_alpha + D_beta) - K(D_sigma) built
/// through the shared J/K supermatrices (two n^2-by-n^2 matrix products
/// per spin channel), per-spin CDIIS extrapolation, diagonalization in
/// the orthogonal basis, per-spin density rebuild (no factor of 2), and
/// a convergence gate on the energy change plus the summed RMS density
/// change. The <S^2> diagnostic is computed from the final occupied
/// orbitals.
/// \param molecule Validated molecule; the electron count and spin
/// multiplicity fix nAlpha/nBeta, and the nuclear repulsion comes from
/// molecule::NuclearRepulsionEnergy.
/// \param overlap Overlap matrix S (n x n).
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical.
/// \param options Convergence settings; the iteration budget and both
/// tolerances must be positive. Optional initial densities seed the run.
/// \returns The converged result, or the final iterate with
/// \p UhfResult::converged false when the budget runs out; an Error
/// (kInvalidArgument for an inconsistent electron count/multiplicity, a
/// shape mismatch, or non-positive options).
qcx::Result<UhfResult> RunUhfScf(const qcx::molecule::Molecule& molecule,
                                 const Eigen::MatrixXd& overlap,
                                 const Eigen::MatrixXd& coreHamiltonian,
                                 const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
                                 const UhfOptions& options = {});

/// Runs UHF with a caller-provided two-Fock builder (the callback seam - the
/// UHF analog of rhf.hpp's FockBuilderFn overload).
///
/// Same procedure as the supermatrix overload, but each iteration's
/// (fockAlpha, fockBeta) pair comes from \p fockBuilder:
/// the direct/RI Fock builders of qcx-integrals plug in here (tests today,
/// the driver). Convergence gating, the per-spin CDIIS path,
/// and the result shape are identical to the supermatrix overload.
/// \param molecule Validated molecule; the electron count and spin
/// multiplicity fix nAlpha/nBeta.
/// \param overlap Overlap matrix S (n x n).
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param options Convergence settings; the iteration budget and both
/// tolerances must be positive. Optional initial densities seed the run.
/// \param fockBuilder Per-iteration Fock-builder pair, called with the
/// current actual per-spin densities; must be callable (an empty callback
/// is rejected).
/// \param basisSet Optional AO basis of the system. When provided, the
/// per-iteration Fock diagonalizations run in the symmetry-adapted basis
/// (blocked diagonalization): the point group is detected from the
/// molecule once, the AO-space decomposition is built once, and every
/// iteration diagonalizes the irrep blocks separately - equivalent to the
/// plain n x n solve up to rounding, cheaper in aggregate for molecules
/// with real symmetry. C1 molecules and unrealizable groups fall back to
/// the plain path; null (the default) always takes the plain path.
/// \returns The converged result, or the final iterate with
/// \p UhfResult::converged false when the budget runs out; an Error
/// (kInvalidArgument for an inconsistent electron count/multiplicity, an
/// empty \p fockBuilder, a shape mismatch, non-positive options, or a
/// detected symmetry inconsistent with the basis).
qcx::Result<UhfResult> RunUhfScf(const qcx::molecule::Molecule& molecule,
                                 const Eigen::MatrixXd& overlap,
                                 const Eigen::MatrixXd& coreHamiltonian,
                                 const UhfOptions& options,
                                 const UhfFockBuilderFn& fockBuilder,
                                 const qcx::basisset::BasisSet* basisSet = nullptr);

/// Runs UHF with an explicit energy: the trace of the core Hamiltonian and the
/// Coulomb operator plus a caller-supplied contribution - the two-spin analog of
/// rhf.hpp's three-callback overload.
///
/// The electronic energy of every iterate is
///     Tr[D_total H] + 1/2 Tr[D_total J[D_total]] + energyContribution(D_a, D_b),
/// where D_total = D_alpha + D_beta. That is the same total as the Hartree-Fock
/// trace identity 1/2 sum_sigma Tr[D_sigma (H + F_sigma)] when the contribution
/// carries the exchange, and it is the Kohn-Sham energy when the contribution
/// carries Exc (plus the exact-exchange piece for a hybrid). The overall 1/2 of
/// the trace identity is absorbed into the closed-shell-shaped Coulomb trace;
/// the derivation is on the helper in uhf.cpp.
///
/// The existing overloads are untouched: this one is additive, and a caller who
/// supplies a Fock builder, a Coulomb provider and a contribution that accounts
/// for the exchange reproduces the Hartree-Fock result of the overload above
/// (algebraically - the two trace expressions sum in different orders, so
/// equality is to rounding, not bit for bit).
/// \param molecule Validated molecule; the electron count and spin multiplicity
/// fix nAlpha/nBeta, and the nuclear repulsion comes from
/// molecule::NuclearRepulsionEnergy.
/// \param overlap The AO overlap matrix S (n x n).
/// \param coreHamiltonian The one-electron core Hamiltonian H (n x n).
/// \param options Iteration budget, tolerances, DIIS, restart seed.
/// \param fockBuilder Builds (F_alpha, F_beta) each iteration from the current
/// pair of per-spin densities (see UhfFockBuilderFn above); must be callable.
/// \param coulomb Supplies J[D_total] for the iterate the energy is formed at:
/// the callback receives the SPIN-SUMMED total density D_alpha + D_beta, which
/// is the same object rhf.hpp's CoulombFn receives (a spin-summed density in the
/// real, unhalved convention), so one adapter serves both runs. Must be callable.
/// \param energyContribution Supplies the energy outside the trace, as
/// described on UhfEnergyContributionFn; must be callable.
/// \param basisSet Optional AO basis (blocked diagonalization), as above.
/// \returns The converged result, or an Error: as the overload above, plus
/// kInvalidArgument for an empty \p coulomb or \p energyContribution, and any
/// error either callback returns.
/// \ingroup qcx-scf
qcx::Result<UhfResult> RunUhfScf(const qcx::molecule::Molecule& molecule,
                                 const Eigen::MatrixXd& overlap,
                                 const Eigen::MatrixXd& coreHamiltonian,
                                 const UhfOptions& options,
                                 const UhfFockBuilderFn& fockBuilder,
                                 const CoulombFn& coulomb,
                                 const UhfEnergyContributionFn& energyContribution,
                                 const qcx::basisset::BasisSet* basisSet = nullptr);

/// Generalized Wolfsberg-Helmholtz initial guess [Wolfsberg1952]: the
/// guess Fock is H on the diagonal and 1.75 * (H_ii + H_jj) * S_ij / 2
/// off it, diagonalized against the overlap; the densities come from the
/// lowest nAlpha/nBeta orbitals. Needs only H and S - the cheapest
/// non-trivial seed for the guess tiers.
/// \param overlap Overlap matrix S (n x n).
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param nAlpha Occupied alpha orbitals; must be non-negative and within
/// the function count.
/// \param nBeta Occupied beta orbitals; must be non-negative and within
/// the function count.
/// \returns (densityAlpha, densityBeta), or an Error (kInvalidArgument
/// for a shape mismatch or occupation counts outside the function count).
qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> BuildGwhGuess(
    const Eigen::MatrixXd& overlap, const Eigen::MatrixXd& coreHamiltonian, int nAlpha, int nBeta);

/// The dense per-element atomic integrals one SAD fragment run consumes:
/// built by the caller (which may link integrals - scf cannot, module
/// DAG; the same seam as the Fock builders), one entry per distinct element of the
/// molecule, keyed by atomic number. The single atom sits at the origin
/// with charge 0 and the element's own basis functions.
/// \ingroup qcx-scf
struct AtomicUhfInputs {
    Eigen::MatrixXd overlap; ///< Single-atom overlap (nZ x nZ).
    Eigen::MatrixXd coreHamiltonian; ///< Single-atom H = T + V (nZ x nZ).
    qcx::memory::Tensor<double, 4, qcx::backend::CpuTag> eri; ///< Single-atom dense
                                                              ///< (uv|ws) tensor.
};

/// The superposition-of-atomic-densities (SAD) guess: per-spin densities
/// ready to seed RunUhfScf, plus the embedded fragment orbital
/// coefficients (the MOM reference handoff).
/// \ingroup qcx-scf
struct SadGuess {
    Eigen::MatrixXd densityAlpha; ///< Block-diagonal fragment alpha density.
    Eigen::MatrixXd densityBeta; ///< Block-diagonal fragment beta density.
    Eigen::MatrixXd coefficientsAlpha; ///< Embedded fragment alpha orbitals (MOM reference).
    Eigen::MatrixXd coefficientsBeta; ///< Embedded fragment beta orbitals (MOM reference).
};

/// Builds the SAD initial guess: one atomic UHF run per distinct element
/// (deduplicated by atomic number - never per atom instance), with the
/// spherical-averaging procedure (fractional occupations spread evenly over
/// the highest partially filled shell, core 1.0) and the deliberate
/// Deliberate deviation: the atomic spin polarization is PRESERVED (each
/// fragment's P_alpha/P_beta scaled by the molecular N_alpha/N_beta over
/// the atomic totals), rather than spin-unpolarized.
/// \param molecule The molecular system the guess seeds.
/// \param basisSet The molecular basis set; must carry an entry for every
/// element (kInvalidArgument otherwise).
/// \param atomicInputs The atomic integrals per distinct element, keyed by
/// atomic number; a missing element is kInvalidArgument. The caller can
/// cache this map across geometries (the integrals do not change).
/// \param atomicOptions Options of the per-element atomic runs; defaults
/// are fine for the tiny atomic systems.
/// \returns The assembled guess, or an Error.
qcx::Result<SadGuess> BuildSadGuess(const qcx::molecule::Molecule& molecule,
                                    const qcx::basisset::BasisSet& basisSet,
                                    const std::map<int, AtomicUhfInputs>& atomicInputs,
                                    const UhfOptions& atomicOptions = {});

} // namespace qcx::scf
