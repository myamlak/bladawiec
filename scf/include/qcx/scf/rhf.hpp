#pragma once

/// \file
/// Closed-shell restricted Hartree-Fock.
///
/// The Roothaan-Hall iteration [Roothaan1951] over the dense four-index
/// repulsion tensor of the integrals module, with symmetric (Loewdin)
/// orthogonalization [Lowdin1950] and the supermatrix Fock build described
/// on the run function.

#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/scf_state.hpp"
#include "qcx/scf/symmetry_labels.hpp"

#include <Eigen/Dense>
#include <functional>
#include <optional>

// Forward declaration only: the basis set is an optional pointer parameter
// of the FockBuilderFn overloads - no basisset symbols are exposed by
// this header.
namespace qcx::basisset {
class BasisSet;
} // namespace qcx::basisset

namespace qcx::scf {

/// \defgroup qcx-scf Scf module
/// Self-consistent-field iterations: closed-shell RHF over the s-function
/// integral engines (integrals/). The loop steps are written as named
/// functions that map one-to-one onto the future FockBuilder<Derived> hooks
/// - the CRTP lands when UHF/DFT make the hook set
/// real, and it reuses these steps unchanged.
/// \{

/// Convergence settings of one RHF run.
struct RhfOptions {
    int maxIterations = 100; ///< Iteration budget.
    /// Total-energy change (Hartree). The RHF stop
    /// requires BOTH legs: |E_n - E_{n-1}| below this AND the RMS density
    /// change below densityTolerance. The energy leg was INERT on the RHF
    /// path until the deferred energy-leg fix landed - the loop assigned its
    /// previous-iteration energy before the gate read it, so the leg
    /// compared the fresh energy against itself and every positive value
    /// passed. Both legs of the gate are live now, so a tighter value buys
    /// more iterations and a lower energy (the run-schema note carries the
    /// measurement).
    double energyTolerance = 1e-8;
    double densityTolerance =
        1e-6; ///< RMS density change (gates both DIIS and plain paths).
    bool useDiis = true; ///< CDIIS acceleration (diis.hpp); false is plain Roothaan.
    bool fullGroupLabeling = true; ///< The post-SCF full-group labeling stage runs
                                   ///< when the caller supplies the basis set; false restores
                                   ///< the behavior without the stage bit-identically
                                   ///< (nothing else in the run reads this flag).

    ScfRestartState initialScfState; ///< Optional restart seed: when the density
                                     ///< field is non-empty it replaces the default GWH
                                     ///< starting guess; the DIIS history and the
                                     ///< previous energies seed the accelerator and the
                                     ///< convergence gate.

    /// Per-iteration trace side-channel (diagnostics; memory
    /// investigation): when non-empty, the loop appends one
    /// `iter=<n> wall=<s> e=<E> dE=<dE> rmsd=<rms> conv=<0|1>` line per
    /// iteration after its convergence decision and flushes it. DIIS-on
    /// runs append the diis-status suffix at the line's end
    /// (append-only: consumers tokenize key=value pairs and ignore unknown
    /// keys): `diis=floor|ext|wait` (the per-iteration gate state - floor =
    /// the machine-precision floor guard skipped the pair, wait = the pair
    /// entered the history and the two-pair minimum is not reached, ext = an
    /// extrapolation replaced the Fock), `diiserr=<norm>` (the commutator
    /// error norm of the decision), `diiscoef=<c1,c2,...>` (the
    /// extrapolation's coefficients, present only when ext), and `negMass=`
    /// plus `parityImb=` (the extrapolation's monitor-only coefficient
    /// measures, present only when ext: (sum|c_i| - 1)/2 and the
    /// even/odd-position imbalance |sum_even c - sum_odd c| / sum|c_i|;
    /// logged for the calibration population - nothing acts on
    /// them). DIIS-off runs keep the six-field line shape. Empty (the
    /// default) keeps the zero-cost path - no file, no allocation, no
    /// numerical effect (the bit-parity pins are absolute).
    std::string traceFile;

    /// Per-iteration density/Fock binary dump side-channel (the C12H26
    /// discriminating-experiment diagnostics): when non-empty, the loop
    /// writes one binary stream - the magic "QXCDFDMP", the matrix
    /// dimension, and per-iteration (density, physical-Fock) pair records
    /// plus the overlap and core Hamiltonian once (format in
    /// scf_common.hpp's ScfDensityDumpWriter; consumed by
    /// tools/amf_density_invariants.py). Empty (the default) keeps the
    /// zero-cost path - no file, no allocation, no numerical effect (the
    /// bit-parity pins are absolute).
    std::string densityDumpFile;
};

/// Result of one RHF run.
struct HfResult {
    double totalEnergy = 0.0; ///< Electronic plus nuclear repulsion (Hartree) of the density this
                              ///< result RETURNS: the exit re-forms the trace from \c density
                              ///< (density-consistent reporting), so it is not the
                              ///< DIIS-extrapolated Fock's trace of
                              ///< the last iteration that the loop reported per iterate.
    double electronicEnergy = 0.0; ///< 1/2 Tr[D (H + F[D])] (Hartree) at the returned density.
    bool converged = false; ///< True when a convergence criterion fired.
    int iterations = 0; ///< Iterations spent; the budget when not converged.

    // The converged (or last-iterate) SCF state, exposed for the
    // properties/ module: the loop computes these on every
    // iteration and previously dropped them at the return. On a
    // non-converged result they hold the FINAL iterate, so a caller
    // inspecting a property of a non-converged run gets the last density
    // rather than an empty matrix - the converged flag stays the gate.
    Eigen::MatrixXd density; ///< Spin-summed closed-shell density D = 2 C_occ C_occ^T -
                             ///< the FockBuilderFn convention (rhf.hpp's FockBuilderFn note),
                             ///< NOT the spatial density rho = D/2. The occupied MOs are the
                             ///< returned \c coefficients' first numOccupied columns: the exit
                             ///< matrix FROM the returned coefficients, so the two describe
                             ///< one state and \c stateConsistencyResidual is zero by
                             ///< construction. It is NOT the trajectory's final density, and
                             ///< not a DIIS linear combination of densities.
    Eigen::MatrixXd coefficients; ///< MO coefficients C (n x n) of the density this result
                                  ///< RETURNS: the eigenvectors of F[D_seed], the Fock built
                                  ///< from the density the polish was seeded with - the
                                  ///< trajectory's returned density (the polish of the
                                  ///< converged-state contract). Columns ascending
                                  ///< by orbital energy. Not the eigenvectors of the
                                  ///< DIIS-extrapolated Fock the loop diagonalized per iterate.
                                  ///< EXCEPT the two-callback (seam/KS) path, which returns the
                                  ///< loop's own diagonalization as before - the same carve-out
                                  ///< the energy fix took, kept because re-diagonalizing
                                  ///< there costs a Fock build per KS run and is a separate
                                  ///< decision (see the finalizer note).
    Eigen::VectorXd orbitalEnergies; ///< Eigenvalues of F[D_seed] in the MO basis, C^T F[D] C
                                     ///< (the density-consistent reporting contract): the returned
                                     ///< coefficients' orbital energies, ascending, so HOMO/LUMO
                                     ///< and the orbital ordering are route-INDEPENDENT and agree
                                     ///< with the returned totalEnergy. Not the final
                                     ///< (DIIS-extrapolated) Fock's MO diagonal. Same seam/KS
                                     ///< carve-out as coefficients.

    ScfRestartState restart; ///< The final-iterate state: the checkpoint
                             ///< writer's source; on a non-converged result it holds the
                             ///< last iterate (converged stays the gate).

    std::optional<SymmetryLabels> symmetryLabels;
    ///< The full-group labeling of the returned state (per-MO irrep
    ///< labels, the canonicalization/straddle record, and the symmetrized
    ///< density D_sym the properties consumers use). Empty when the run was
    ///< a C1 molecule, the caller passed no basis set, the option was off,
    ///< or the group was unrealizable; the converged flag stays the gate.

    /// The convergence gate's OWN operands at the returned iterate, recorded
    /// for the reader and never acted on: the true |E_n - E_{n-1}| and the
    /// RMS density change the gate compared (both of the last iterate on a
    /// non-converged result).
    ///
    /// These exist because a bare `converged` does not say what it stood on.
    /// Both gate legs are live on the RHF and the UHF path, so a converged
    /// result satisfies achievedEnergyDelta < energyTolerance and
    /// achievedRmsDensityChange < densityTolerance by construction. The
    /// achieved pair is what makes that checkable from outside: it was the
    /// instrument that showed the RHF energy leg to be inert while the
    /// density leg alone decided every RHF stop (the deferral, since closed).
    double achievedEnergyDelta = 0.0;
    /// The RMS density change the gate compared (Frobenius/n) - the leg that
    /// decided an RHF stop on its own while the energy leg was inert.
    double achievedRmsDensityChange = 0.0;

    /// The returned state's PROJECTOR residual (state consistency): the Frobenius norm
    /// ||D - 2 C_occ C_occ^T|| between the returned density and the density the
    /// returned occupied MOs define. This is the consistency check the
    /// converged-state contract needed and did not have: a returned density and
    /// a returned coefficient set one SCF step apart are not one state, and no
    /// convergence criterion bounds the gap (the density gate compares two
    /// trajectory iterates, not the returned objects). It measured 1.3e-6 on
    /// the O2 triplet's beta channel while the run reported converged - the
    /// defect the polish below closes.
    ///
    /// Zero by construction on the polished path (the exit builds \c density
    /// from \c coefficients). A NONZERO value is a real defect signal, not a
    /// tolerance: it would mean the two returned objects drifted apart. The
    /// seam/KS carve-out path returns the loop's own pair, which is also
    /// projector-consistent, so this field is zero there too - it does NOT
    /// cover that path's remaining gap (its returned C and eps are the
    /// DIIS-extrapolated Fock's eigenpairs, not F[returned density]'s).
    double stateConsistencyResidual = 0.0;

    /// The returned state's FIXED-POINT residual (state consistency): the Frobenius norm
    /// ||D_returned - D_seed|| between the returned density and the density the
    /// Fock that produced the returned coefficients was built from. The
    /// contract's remaining leg on the polished HF path is F = F[D_returned],
    /// and this is its density-space measure: F is affine in D, so
    /// F[D_returned] - F[D_seed] is linear in this gap and bounded by it (the
    /// contract's one-pass exactness for HF).
    ///
    /// This is the number a CONSISTENCY tolerance would gate, and it is what
    /// the projector residual above measured BEFORE the polish existed: the
    /// pre-fix return path handed out D_seed as the density and F[D_seed]'s
    /// eigenvectors as the coefficients, so the old
    /// ||D - 2 C_occ C_occ^T|| and this quantity are the same two objects. The
    /// KS leg needs the iterate-until-consistent form for exactly this leg (Vxc
    /// is nonlinear); see the finalizer note.
    ///
    /// On the seam/KS carve-out path no polish runs, so the returned density IS
    /// the density the returned coefficients were built with and this is zero
    /// by identity rather than by measurement - unlike the projector residual
    /// beside it, which that path does measure.
    double stateFixedPointResidual = 0.0;

    /// The linear-dependence removal's DISCLOSURE (scf_common.hpp
    /// OrthogonalizeOverlap): how many overlap directions were dropped
    /// because they fell below kOverlapEigenvalueFloorTolerance times the
    /// largest overlap eigenvalue. Zero for every well-conditioned system.
    ///
    /// A nonzero value is not a defect and not a smaller calculation: the
    /// run happened in the (n - numRemovedOverlapDirections)-dimensional
    /// orthonormal space and was mapped back, so density, Fock and energy
    /// are all in the full AO dimension. It is reported because a silent
    /// truncation would be indistinguishable from a correct run (disclosed,
    /// never silent).
    ///
    /// It is ALSO the disclosure of what the removal disengaged, and that is
    /// the only record of it: a nonzero value means the symmetry stages stood
    /// down by name rather than by failure. The blocked diagonalization falls
    /// back to the plain n-dimensional solve (X is rectangular, so U_b^T X U
    /// has no square per-irrep block), and the full-group labeling stage does
    /// not run at all - `symmetryLabels` is then empty, exactly as it is for a
    /// C1 molecule or with \c fullGroupLabeling off. The two facts are not
    /// independent and are not recorded twice: nothing else removes
    /// directions, so this count is what says why the labeling block is absent.
    std::size_t numRemovedOverlapDirections = 0;
};

/// Per-iteration Fock-matrix builder callback (the Fock-builder seam).
/// scf cannot link integrals (module DAG), so the direct/RI Fock builders of
/// qcx-integrals are wired in by the caller (tests today, the driver in Phase 1).
///
/// Density convention: the callback receives the SCF's own iterate density -
/// the SPIN-SUMMED closed-shell density D = 2 C_occ C_occ^T (the supermatrix
/// path of rhf.cpp contracts it with F = H + J[D] - K[D]/2). The direct/RI
/// builders of qcx-integrals instead contract the SPATIAL density rho = D/2
/// (fock_build.hpp / ri_engine.hpp document their convention), so adapters
/// that wire those builders in must pass density / 2 - the direct_rhf_test /
/// ri_rhf_test adapters demonstrate the scaling.
using FockBuilderFn = std::function<qcx::Result<Eigen::MatrixXd>(const Eigen::MatrixXd& density)>;

/// Supplies the Coulomb matrix J[D] of the current iterate.
///
/// Density convention: the same as \p FockBuilderFn - the callback receives the
/// SCF's own iterate density, the spin-summed closed-shell D = 2 C_occ C_occ^T,
/// and an adapter that wires a builder contracting the SPATIAL density rho = D/2
/// passes density / 2 exactly as it does for the Fock build.
///
/// The two-callback overload needs J as a separate operator because the
/// Hartree-Fock trace identity 1/2 Tr[D (H + F)] does not survive a Kohn-Sham
/// Fock: the identity is a property of F = H + J - K/2, and it is not
/// recoverable from an opaque builder that returns F alone.
/// \ingroup qcx-scf
using CoulombFn = std::function<qcx::Result<Eigen::MatrixXd>(const Eigen::MatrixXd& density)>;

/// Supplies the energy the trace formula cannot cover.
///
/// For a density functional this is Exc[D]; for a hybrid it also carries the
/// exact-exchange piece -c_HF 1/2 Tr[D K[D]], because the exchange is a
/// bilinear form the caller has and the loop does not. The loop cannot compute
/// either from D and an opaque Fock builder: Exc is a generically nonlinear
/// functional of the density, not the bilinear form the Hartree-Fock trace
/// recovers, so a Kohn-Sham Fock composed into the existing seam converges to
/// the right density while the trace reports an energy that does not belong to
/// what was built. This callback is what closes that gap.
/// \ingroup qcx-scf
using EnergyContributionFn = std::function<qcx::Result<double>(const Eigen::MatrixXd& density)>;

/// Runs closed-shell RHF over s functions.
///
/// Symmetric orthogonalization, then the iterate-until-converged Roothaan
/// procedure: build the Fock matrix from the current density, transform into
/// the orthogonal basis, diagonalize, rebuild the density from the occupied
/// orbitals, check convergence. The Fock build contracts the density with
/// the repulsion tensor through the supermatrix form: two n^2-by-n^2 matrix
/// products per iteration through linalg::DenseMultiply (Eigen for small
/// blocks, vendor BLAS above the threshold) instead of a four-index
/// scalar loop. With \p RhfOptions::useDiis (the default), the Fock matrix
/// is CDIIS-extrapolated [Pulay1980] before diagonalization. Convergence is
/// gated by the energy change and the RMS density change on both paths -
/// DIIS accelerates, it does not define convergence. Both legs are live on
/// both paths: the RHF energy leg was inert until the deferred fix was
/// closed, because the loop assigned its previous-iteration energy before
/// the gate read it, which left the density leg deciding every RHF stop on
/// its own. With no restart seed (empty
/// \p RhfOptions::initialScfState), the loop starts from the GWH guess
/// (uhf.hpp BuildGwhGuess): the zero matrix is a known-bad start on
/// diffuse bases, locking the iteration in a non-converging cycle instead
/// of the true fixed point.
/// \param molecule Validated molecule; the electron count comes from its
/// atoms and charge, and the nuclear repulsion from
/// molecule::NuclearRepulsionEnergy.
/// \param overlap Overlap matrix S (n x n).
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical (the integral engines return it that way).
/// \param options Convergence settings; the iteration budget and both
/// tolerances must be positive.
/// \returns The converged result, or the final iterate with
/// \p HfResult::converged false when the budget runs out; an Error
/// (kUnimplemented for an odd electron count, kInvalidArgument for a shape
/// mismatch or non-positive options).
qcx::Result<HfResult> RunRhfScf(const qcx::molecule::Molecule& molecule,
                                const Eigen::MatrixXd& overlap,
                                const Eigen::MatrixXd& coreHamiltonian,
                                const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
                                const RhfOptions& options = {});

/// Runs closed-shell RHF over s functions with a caller-provided Fock
/// builder (the Fock-builder seam).
///
/// Same procedure as the supermatrix overload - symmetric orthogonalization,
/// then the iterate-until-converged Roothaan loop - but each iteration's Fock
/// matrix comes from \p fockBuilder instead of a supermatrix contraction of
/// the dense ERI tensor: scf cannot link integrals (module DAG), so the
/// direct/RI Fock builders of qcx-integrals are wired in by the caller (tests
/// today, the driver in Phase 1). Convergence gating, the CDIIS path, and the
/// result shape are identical to the supermatrix overload.
/// \param molecule Validated molecule; the electron count comes from its
/// atoms and charge, and the nuclear repulsion from
/// molecule::NuclearRepulsionEnergy.
/// \param overlap Overlap matrix S (n x n).
/// \param coreHamiltonian Core Hamiltonian H = T + V (n x n).
/// \param options Convergence settings; the iteration budget and both
/// tolerances must be positive.
/// \param fockBuilder Per-iteration Fock-matrix builder, called with the
/// current spin-summed density D = 2 C_occ C_occ^T (see the FockBuilderFn
/// convention note above); must be callable (an empty callback is rejected).
/// \param basisSet Optional AO basis of the system. When provided, the
/// per-iteration Fock diagonalization runs in the symmetry-adapted basis
/// (blocked diagonalization): the point group is detected from the
/// molecule once, the AO-space decomposition is built once, and every
/// iteration diagonalizes the irrep blocks separately - equivalent to the
/// plain n x n solve up to rounding, cheaper in aggregate for molecules
/// with real symmetry. C1 molecules and unrealizable groups fall back to
/// the plain path; null (the default) always takes the plain path.
/// \returns The converged result, or the final iterate with
/// \p HfResult::converged false when the budget runs out; an Error
/// (kUnimplemented for an odd electron count, kInvalidArgument for an empty
/// \p fockBuilder, a shape mismatch, non-positive options, or a detected
/// symmetry inconsistent with the basis).
qcx::Result<HfResult> RunRhfScf(const qcx::molecule::Molecule& molecule,
                                const Eigen::MatrixXd& overlap,
                                const Eigen::MatrixXd& coreHamiltonian,
                                const RhfOptions& options,
                                const FockBuilderFn& fockBuilder,
                                const qcx::basisset::BasisSet* basisSet = nullptr);

/// Runs closed-shell SCF with an explicit energy: the trace of the core
/// Hamiltonian and the Coulomb operator plus a caller-supplied contribution.
///
/// The electronic energy of every iterate is
///     Tr[D H] + 1/2 Tr[D J[D]] + energyContribution(D),
/// which is the same total as the Hartree-Fock trace identity when the
/// contribution carries the exchange, and is the Kohn-Sham energy when it
/// carries Exc (and the exact-exchange piece for a hybrid). The Fock matrix is
/// the caller's through \p fockBuilder exactly as in the overload above; the
/// two callbacks are what separate the energy the loop can form itself from the
/// energy only the caller knows.
///
/// The existing overloads are untouched: this one is additive, and a caller
/// who supplies a Fock builder, a Coulomb provider and a contribution that
/// accounts for the exchange reproduces the Hartree-Fock result of the
/// overload above (algebraically - the two trace expressions sum in different
/// orders, so equality is to rounding, not bit for bit).
/// \param molecule Validated molecule; the electron count comes from its atoms
/// and charge, and the nuclear repulsion from molecule::NuclearRepulsionEnergy.
/// \param overlap The AO overlap matrix S (n x n).
/// \param coreHamiltonian The one-electron core Hamiltonian H (n x n).
/// \param options Iteration budget, tolerances, DIIS, restart seed.
/// \param fockBuilder Builds F[D] each iteration from the spin-summed density D
/// (see the FockBuilderFn convention note above); must be callable.
/// \param coulomb Supplies J[D] for the same density; must be callable.
/// \param energyContribution Supplies the energy outside the trace, as
/// described on EnergyContributionFn; must be callable.
/// \param basisSet Optional AO basis (blocked diagonalization), as above.
/// \returns The converged result, or an Error: as the overload above, plus
/// kInvalidArgument for an empty \p coulomb or \p energyContribution, and any
/// error either callback returns.
/// \ingroup qcx-scf
qcx::Result<HfResult> RunRhfScf(const qcx::molecule::Molecule& molecule,
                                const Eigen::MatrixXd& overlap,
                                const Eigen::MatrixXd& coreHamiltonian,
                                const RhfOptions& options,
                                const FockBuilderFn& fockBuilder,
                                const CoulombFn& coulomb,
                                const EnergyContributionFn& energyContribution,
                                const qcx::basisset::BasisSet* basisSet = nullptr);

/// \}
} // namespace qcx::scf
