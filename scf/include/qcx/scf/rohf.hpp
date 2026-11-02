#pragma once

/// \file
/// Restricted open-shell Hartree-Fock and Kohn-Sham: ONE spatial orbital set
/// carrying a closed block (two electrons per orbital) and an open block (one
/// electron each), driven by the Roothaan effective Fock. See rhf.hpp for the
/// closed-shell counterpart, uhf.hpp for the per-spin seam this path consumes
/// unchanged, and src/internal/scf_common.hpp for the shared step-functions.
///
/// The loop is the restricted open-shell one, not a repurposed UHF loop: the
/// two physical spin Focks do not share an eigenbasis, so the equations are
/// solved through the effective Fock assembled from them (Roothaan's coupling
/// matrix). Diagonalizing the alpha Fock and filling by eigenvalue would
/// solve the UHF equations instead, and the UHF convergence gate would wait
/// for two commutators that do not vanish at a restricted open-shell fixed
/// point.

#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"

#include <Eigen/Dense>
#include <string>

namespace qcx::basisset {
class BasisSet;
} // namespace qcx::basisset

namespace qcx::scf {

/// Convergence settings of one restricted open-shell run. Deliberately its
/// own struct rather than a shared base: the axes that do not exist here
/// (per-spin DIIS histories, the two-phase handoff, the joint system, the
/// maximum-overlap occupation selection) stay unrepresentable instead of
/// being present and ignored.
/// \ingroup qcx-scf
struct RohfOptions {
    int maxIterations = 100; ///< Iteration budget.
    double energyTolerance = 1e-8; ///< Total-energy change (Hartree).
    double densityTolerance = 1e-6; ///< RMS change of the ALPHA density
                                    ///< (D_alpha = D_c/2 + D_o, the occupied-space
                                    ///< projector), divided by the function count.
                                    ///< Not the per-spin pair the UHF loop reads:
                                    ///< the closed/open split is not unique when the
                                    ///< closed and open boundary orbitals are
                                    ///< degenerate in the energy (a linear radical's
                                    ///< pi shell), and the alpha density is the
                                    ///< object that alternation leaves invariant -
                                    ///< the beta density carries the split. The
                                    ///< energy leg refuses a split change that is
                                    ///< not degenerate.

    /// One CDIIS channel on the effective Fock (the Roothaan assembly is a
    /// single matrix, so there is one history, not one per spin).
    bool useDiis = true;

    /// Three-way convergence gate: adds the max-density-change leg (at the
    /// full densityTolerance) and the single channel's DIIS-error leg
    /// (internal::kDiisErrorGateTolerance) to the two-way gate, and tightens
    /// the RMS density leg to densityTolerance / internal::kDiisDensityGateFactor.
    /// Off by default - the two-way predicate is the plain loop's.
    bool useRobustGate = false;

    /// Level shift of the open and virtual blocks relative to the closed one:
    /// the diagonalization input gains
    ///     levelShiftOpen * P_o + levelShiftVirtual * P_v
    /// after the DIIS extrapolation and outside the energy path, so the
    /// shifted matrix never reaches the energy computation and the converged
    /// state is the unshifted fixed point. The pair is the closed-relative
    /// form of the two-shift recipe documented on
    /// internal::ApplyVirtualSpaceLevelShift (shift_1 on the virtual space
    /// over the alpha occupied set, then shift_2 - shift_1 over the beta
    /// set): open = shift_2 - shift_1, virtual = shift_2. Skipped on the
    /// first iterate, whose projectors do not come from a diagonalization.
    /// Off by default.
    bool useLevelShift = false;

    /// Shift of the open block relative to the closed one (Hartree); must be
    /// non-negative (0.0 is a legal no-op) - validated regardless of
    /// useLevelShift, applied only when it is set.
    double levelShiftOpen = 0.0;
    /// Shift of the virtual block relative to the closed one (Hartree); see
    /// levelShiftOpen.
    double levelShiftVirtual = kLevelShiftDefault;

    /// Optional starting pair of per-spin densities (the GWH and SAD guesses
    /// produce it). The pair must describe a restricted determinant for the
    /// open block to mean anything: the loop reads the closed density as
    /// 2 D_beta and the open one as D_alpha - D_beta. A polarized start whose
    /// difference is not a projector is accepted and corrected by the first
    /// aufbau assignment. Empty matrices mean the zero start; a non-empty pair
    /// must match the function count.
    Eigen::MatrixXd initialDensityAlpha;
    /// Optional starting per-spin density; see initialDensityAlpha.
    Eigen::MatrixXd initialDensityBeta;

    /// Optional restart seed: non-empty densities replace the initial guess,
    /// the single DIIS history in ScfRestartState::diis replaces the stored
    /// one, and the previous energies seed the convergence gate. The
    /// per-spin DIIS fields are unused - this loop has one channel.
    ScfRestartState initialScfState;

    /// Per-iteration trace side-channel (diagnostics; the memory-anomaly
    /// investigation): when non-empty, the loop appends one
    /// `iter=<n> wall=<s> e=<E> dE=<dE> rmsd=<rms> conv=<0|1>` line per
    /// iteration after its convergence decision and flushes it. Empty (the
    /// default) keeps the zero-cost path - no file, no allocation, no
    /// numerical effect.
    std::string traceFile;
};

/// Result of one restricted open-shell run: the UHF result, because every
/// field describes the same objects - per-spin densities, coefficients,
/// orbital energies, <S^2>, the restart state, the returned-state residuals
/// and the linear-dependence disclosure.
///
/// The one semantics note is the orbital spectrum: both channels carry the
/// EFFECTIVE Fock's eigenvalues, because that is the matrix the loop
/// diagonalized. They are not the eigenvalues of the physical alpha and beta
/// Focks (which do not share an eigenbasis). The coefficients are the same
/// spatial set in both channels, ordered [closed, open, virtual], with alpha
/// occupations 1 over the closed and open blocks and beta occupations 1 over
/// the closed block, so D_sigma = C occ_sigma C^T reconstructs the restricted
/// pair exactly.
///
/// The second is the recorded gate operand: \p
/// UhfResult::achievedRmsDensityChange carries the ALPHA-density RMS change
/// this loop's gate compared (see RohfOptions::densityTolerance), not the
/// two-spin sum the UHF field's own doc describes.
/// \ingroup qcx-scf
using RohfResult = UhfResult;

/// Runs restricted open-shell Hartree-Fock with a caller-provided two-Fock
/// builder, the same per-spin seam the UHF loop consumes: the builder is
/// called with the current (D_alpha, D_beta) pair (the UHF density
/// convention, no factor of 2), and the pair it returns is the pair of
/// physical spin Focks.
///
/// The electron count and the multiplicity fix the blocks: nO = multiplicity
/// - 1 open orbitals, nC = (electronCount - nO)/2 closed ones, nAlpha =
/// nC + nO and nBeta = nC. Each iteration assembles the Roothaan effective
/// Fock in the orthonormal basis from the spin pair and the closed/open
/// projectors, appends one CDIIS error to the single history, diagonalizes
/// the effective Fock, and assigns the lowest nC orbitals closed and the
/// next nO open. The energy is 1/2 sum_sigma Tr[D_sigma (H + F_sigma)].
/// \param molecule Validated molecule; the electron count and spin
/// multiplicity fix the block occupations.
/// \param overlap Overlap matrix S (n x n).
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param options Convergence settings; the iteration budget and both
/// tolerances must be positive, and the level shifts non-negative.
/// \param fockBuilder Per-iteration Fock-builder pair, called with the
/// current actual per-spin densities; must be callable.
/// \returns The converged result, or the final iterate with
/// \p RohfResult::converged false when the budget runs out; an Error
/// (kInvalidArgument for an inconsistent electron count/multiplicity, an
/// empty \p fockBuilder, a shape mismatch, or non-positive options).
/// \ingroup qcx-scf
qcx::Result<RohfResult> RunRohfScf(const qcx::molecule::Molecule& molecule,
                                   const Eigen::MatrixXd& overlap,
                                   const Eigen::MatrixXd& coreHamiltonian,
                                   const RohfOptions& options,
                                   const UhfFockBuilderFn& fockBuilder);

/// Runs restricted open-shell Kohn-Sham: the same loop with an explicit
/// energy, the two-spin analog of rhf.hpp's three-callback overload.
///
/// The electronic energy of every iterate is
///     Tr[D_total H] + 1/2 Tr[D_total J[D_total]] + energyContribution(D_a, D_b),
/// with D_total = D_alpha + D_beta, which is the ROHF trace identity when the
/// contribution carries the exchange and the Kohn-Sham energy when it carries
/// Exc (plus the hybrid's exact-exchange piece). The effective Fock is the
/// same Roothaan assembly of the pair the builder returns, so the closed
/// orbitals see the average of the two spin potentials and the open orbitals
/// the alpha one - the physical restricted open-shell potential structure.
/// No new seam is required: the per-spin composition is already this object.
/// \param molecule Validated molecule.
/// \param overlap The AO overlap matrix S (n x n).
/// \param coreHamiltonian The one-electron core Hamiltonian H (n x n).
/// \param options Iteration budget, tolerances, DIIS, restart seed.
/// \param fockBuilder Builds (F_alpha, F_beta) each iteration from the current
/// pair of per-spin densities; must be callable.
/// \param coulomb Supplies J[D_total] for the iterate the energy is formed at;
/// the callback receives the spin-summed density D_alpha + D_beta, the same
/// object rhf.hpp's CoulombFn receives. Must be callable.
/// \param energyContribution Supplies the energy outside the trace (see
/// UhfEnergyContributionFn); must be callable.
/// \returns The converged result, or an Error: as the overload above, plus
/// kInvalidArgument for an empty \p coulomb or \p energyContribution, and any
/// error either callback returns.
/// \ingroup qcx-scf
qcx::Result<RohfResult> RunRohfScf(const qcx::molecule::Molecule& molecule,
                                   const Eigen::MatrixXd& overlap,
                                   const Eigen::MatrixXd& coreHamiltonian,
                                   const RohfOptions& options,
                                   const UhfFockBuilderFn& fockBuilder,
                                   const CoulombFn& coulomb,
                                   const UhfEnergyContributionFn& energyContribution);

} // namespace qcx::scf
