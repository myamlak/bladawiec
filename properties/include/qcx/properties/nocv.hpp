#pragma once

/// \file
/// The ETS-NOCV energy decomposition: the fragment-based
/// total-energy decomposition of [Mitoraj2009] (J. Chem. Theory Comput.
/// 5, 962-975), pairing the extended transition state (ETS) partitioning
/// with the Natural Orbitals for Chemical Valence (NOCV).
///
/// The decomposition of the binding energy of the fragments into
///
///   E_mol - sum_i E_frag_i = E_elstat + E_Pauli + E_orb,
///
/// where E_elstat is the classical Coulomb interaction of the frozen
/// fragment densities (the exchange-free functional), E_Pauli the
/// repulsion from antisymmetrizing and orthogonalizing the combined
/// fragment determinant, and E_orb the relaxation energy of the frozen
/// combined determinant to the converged molecular density, resolved
/// into the paired NOCV channels:
///
///   E_orb = sum_k (lambda_+ F^TS_{++} + lambda_- F^TS_{--})_k,
///
/// the transition-state Fock matrix F^TS evaluated at
/// 1/2(P_orth + P_mol) and the SIGN-PAIRED (+-nu_k) eigenpairs of the
/// orthogonalized density difference Delta P = P_mol - P_orth: the k-th
/// largest positive eigenvalue pairs with the k-th largest-magnitude
/// negative one (an odd function count leaves one unpaired tail
/// eigenvalue). The pairing is decided by the sign, not by a |lambda|
/// comparison - the paired magnitudes of an open-shell pairing defect tie
/// at the 1e-15 level, where a magnitude-adjacent pairing would be
/// noise-dependent (an open-shell beta spectrum [1.0, 0.7108, 0.3777] vs
/// [-0.7108, -0.3777, ~0] has NO both-sign tie: the exact two-projector
/// pairing is the sign pairing). This is the EXACT resolution E[P_mol] -
/// E[P_orth] = Tr[Delta P F^TS] of the quadratic RHF functional - each
/// eigenpair contributes its own diagonal element with its own eigenvalue.
/// The textbook form nu_k (F^TS_{++} - F^TS_{--}) follows only for the
/// exactly-paired spectra of equal-rank two-projector differences (the
/// closed-shell fragments); the off-diagonal bilinear form nu F_-+
/// coincides only in that same limit.
/// E_orb is ALSO resolved per spin (the unrestricted ETS-NOCV): the
/// per-spin differences Delta D_sigma = D/2 - P_orth,sigma against the
/// per-spin transition Focks F_sigma^TS = h + J[P^TS_total] -
/// K[D_sigma^TS], whose channels and spin-resolved value are the physical
/// ones for open-shell fragments.
///
/// The fragments run their own SCFs at their molecular geometries in the
/// FULL molecular basis set (the ETS convention - fragment orbitals span
/// the whole space, only the occupied block differs). The molecular
/// density is the converged closed-shell RHF D = 2 C_occ C_occ^T; the
/// energy functional values come from the scf arbitrary-density evaluator
/// (qcx/scf/density_energy.hpp), so the ETS energy identity pins against
/// the SCF loops' energies (1e-9: the loops' final Fock is the
/// DIIS-extrapolated one, so the evaluator's fresh F at the converged
/// density differs at the residue level, ~1e-11).
///
/// AnalyzeNocvEts, the closed-shell Hartree-Fock entry point, admits no
/// density functional: its fragment SCFs are UHF and no functional enters
/// it, so a Kohn-Sham run's total energy is not the energy it partitions.
/// The Kohn-Sham entry points take the run's own functional seam instead
/// (AnalyzeNocvEtsKohnSham, AnalyzeNocvEtsUnrestrictedKohnSham).
///
/// E_orb is resolved twice. The restricted resolution evaluates every
/// density with the closed-shell functional (self-consistent - the
/// partition identity holds exactly - but the restricted functional
/// over-estimates spin-polarized P_orth energies by
/// 1/4 Tr[(D_a - D_b) K[D_a - D_b]] relative to the spin-resolved
/// functional). The unrestricted resolution computes the per-spin
/// differences Delta D_sigma = D/2 - P_orth,sigma (the molecular density
/// is the closed-shell D = 2 C_occ C_occ^T, so D_sigma = D/2) with the
/// per-spin transition Focks F_sigma^TS = h + J[P^TS_total] - K[D_sigma^TS]
/// at D_sigma^TS = 1/2(P_orth,sigma + D/2), whose J argument sums to the
/// total transition density 1/2(P_orth + D): the spin-resolved identity
/// E[P_mol] - E[P_orth] = sum_sigma Tr[Delta D_sigma F_sigma^TS] is exact,
/// and the per-spin channels are the physical ones (for the exactly-paired
/// closed-shell fragments the per-spin spectra halve the spin-summed one
/// and the two resolutions coincide).
///
/// THE OTHER THREE REFERENCES (the closed-shell Hartree-Fock entry point is
/// AnalyzeNocvEts above):
///
/// * AnalyzeNocvEtsUnrestricted runs an UNRESTRICTED Hartree-Fock
///   decomposition on a UHF determinant: two INDEPENDENT NOCV problems,
///   one on Delta P_sigma = D_sigma - P_orth,sigma per spin. Each spin
///   density of a UHF determinant is a projector, so each difference is a
///   difference of two projectors and its spectrum keeps the sign-paired
///   (+-nu_k) structure; each spin's transition Fock is that spin's own
///   F_sigma^TS = h + J[P^TS_total] - K[D_sigma^TS], evaluated at
///   D_sigma^TS = 1/2(P_orth,sigma + D_sigma). The exact identity
///   E_UHF[D_a, D_b] - E_UHF[P_orth,a, P_orth,b]
///     = sum_sigma Tr[Delta D_sigma F_sigma^TS]
///   holds for the quadratic UHF functional (the midpoint is exact), so
///   the per-spin channel sums ARE the total orbital relaxation. The total
///   deformation density is Delta P_a + Delta P_b.
///
/// * AnalyzeNocvEtsKohnSham runs a closed-shell (RKS) decomposition with
///   the Ziegler-Rauk three-point transition state. The orbital relaxation
///   is E_orb = E[rho_mol] - E[rho_orth] = integral_0^1 dlambda
///   Tr[Delta rho F[rho_lambda]] with rho_lambda = rho_orth +
///   lambda Delta rho; Simpson's rule on [0,1] gives
///     F^TS = 1/6 F[rho_orth] + 2/3 F[rho_TS] + 1/6 F[rho_mol],
///     rho_TS = 1/2 (rho_orth + rho_mol),
///   (Simpson's rule is exact when the integrand Tr[Delta rho
///   F[rho_lambda]] is cubic in lambda, i.e. when the energy is quartic
///   along the path) and it collapses to the midpoint Fock EXACTLY for a
///   quadratic (Hartree-Fock) functional. The XC potential is
///   evaluated at each of the three densities, so a Kohn-Sham run's
///   per-pair channels are its own functional's, not a Hartree-Fock
///   functional wearing the run's label. The three-point E_orb differs from
///   the exact E[rho_mol] - E[rho_orth] by the Simpson residual (zero for
///   Hartree-Fock), which the result records through the binding-energy
///   identity's own accuracy.
///
/// * AnalyzeNocvEtsUnrestrictedKohnSham runs an UNRESTRICTED Kohn-Sham
///   (UKS) decomposition: the two per-spin NOCV problems of
///   AnalyzeNocvEtsUnrestricted, each resolved against its own spin's
///   three-point transition Fock
///     F_sigma^TS = 1/6 F_sigma[P_orth] + 2/3 F_sigma[P_TS]
///                  + 1/6 F_sigma[D],
///   every F_sigma from the run's functional seam at the per-spin density
///   pair. E_Pauli is that functional's value at the orthogonalized spin
///   pair, so the partition identity
///     E_KS[D_a, D_b] - sum_i E_frag,i = E_elstat + E_Pauli + E_orb
///   holds by construction with the run's own functional throughout, and
///   E_orb differs from the exact E[D] - E[P_orth] by the Simpson residual
///   of the spin-pair path (zero when the functional is quadratic). The
///   entry collapses to AnalyzeNocvEtsUnrestricted exactly on the
///   Hartree-Fock seam (zero XC, c_HF = 1) and to AnalyzeNocvEtsKohnSham
///   when the two molecular densities are equal.
///
/// * The RESTRICTED open-shell reference (ROHF/ROKS) is NOT served. Each
///   spin density of a restricted open-shell determinant is a projector,
///   so the per-spin path would run on it unchanged - which is exactly
///   why it must be refused by name rather than left to run: the restricted
///   reference is ONE spatial orbital set, the unrestricted resolution
///   diagonalizes the two channels independently and returns NOCV pairs of
///   different spatial shapes, which that reference does not permit; and
///   the ROHF/ROKS energy is not a functional of the total density (it
///   depends on the core/open-shell split), so a transition-state Fock has
///   to be defined from the ROHF effective Fock operators (F_c and F_o)
///   together with a channel pairing rule for a spectrum that is not
///   +-nu paired. Neither exists in this module; every entry point refuses
///   NocvEtsReference::kRestrictedOpenShell with that statement, and
///   CheckNocvEtsReference is its one home.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/uhf.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace qcx::properties {

/// The ETS-NOCV decomposition of one molecular density over one fragment
/// partition.
struct NocvEtsDecomposition {
    /// E_elstat: the classical (Coulomb-only) interaction energy of the
    /// frozen fragment densities with each other's nuclei and charge
    /// clouds (Hartree).
    double electrostatic = 0.0;
    /// E_Pauli: the repulsion from antisymmetrizing the combined frozen
    /// fragment determinant and orthogonalizing its occupied space
    /// (Hartree). Defined so that
    /// E[P_orth] - sum_i E_frag_i = E_elstat + E_Pauli.
    double pauli = 0.0;
    /// E_orb of the MOLECULAR METHOD. For the closed-shell Hartree-Fock
    /// reference this is the RESTRICTED (closed-shell) functional's value,
    /// equal to the sum of orbitalComponents (the identity pin). For the
    /// unrestricted reference it is the spin-resolved value, equal to
    /// orbitalUnrestricted (there is no restricted functional on a UHF
    /// determinant). For the closed-shell Kohn-Sham reference it is the
    /// three-point transition-state value, equal to the sum of
    /// orbitalComponents. bindingEnergy is formed with this member.
    double orbital = 0.0;
    /// E_orb of the SPIN-RESOLVED (UHF) functional: the relaxation energy
    /// resolved per spin,
    /// E[P_mol] - E[P_orth] = sum_sigma Tr[Delta D_sigma F_sigma^TS],
    /// equal to the sum of the alpha and beta channel components by
    /// construction. On the closed-shell reference it differs from orbital
    /// by 1/4 Tr[(P_a - P_b) K[P_a - P_b]] on open-shell fragments (the two
    /// functionals coincide on spin-symmetric densities). On the
    /// unrestricted reference it is the method's own E_orb and equals
    /// orbital. On the closed-shell Kohn-Sham reference it is not separately
    /// resolved (the functional is spin-summed) and equals orbital.
    double orbitalUnrestricted = 0.0;
    /// The binding energy E_mol - sum_i E_frag_i = E_elstat + E_Pauli +
    /// E_orb (the partition identity; Hartree).
    double bindingEnergy = 0.0;
    /// The isolated fragment SCF total energies E_frag_i, one per fragment
    /// in the caller's fragment order (Hartree).
    Eigen::VectorXd fragmentEnergies;
    /// Delta E_orb^k per NOCV channel (Hartree): the sign-paired (+-nu_k)
    /// pairs, each carrying lambda_+ F_++ + lambda_- F_-- with its own
    /// eigenvalues, plus the unpaired tail eigenvalue of an odd function
    /// count (zero for exactly-paired spectra); one entry per ceil(n/2),
    /// orbital = sum over this vector. EMPTY on the unrestricted reference:
    /// a UHF determinant's total deformation density is not a difference of
    /// two projectors, so it has no spin-summed NOCV problem; the physical
    /// channels are the per-spin ones (orbitalComponentsAlpha/Beta).
    Eigen::VectorXd orbitalComponents;
    /// The +nu_k of each sign-paired pair (electrons); for an odd function
    /// count the last entry holds the unpaired tail eigenvalue (signed,
    /// zero for exactly-paired spectra).
    Eigen::VectorXd nocvEigenvalues;
    /// The -nu_k partner of each closed-shell channel (electrons): the
    /// negative eigenvalue of the same sign-paired pair, in nocvEigenvalues'
    /// order. For an exactly paired spectrum this is -nocvEigenvalues; the
    /// partner of a rank-imbalance leftover is ~0 (a null-space
    /// eigenvalue), and the unpaired tail channel of an odd function count
    /// reports 0 here (it has no partner). The restricted open-shell refusal
    /// is NOT about this field: it exists because a restricted reference
    /// cannot consume independently diagonalized per-spin channels.
    Eigen::VectorXd nocvEigenvaluesNegative;
    /// The alpha-channel resolution of E_orb (Hartree per channel): the
    /// own-sign channels of Delta D_a = D/2 - P_orth,a in the Lowdin
    /// basis, one entry per ceil(n/2); the sum over this vector and
    /// orbitalComponentsBeta is orbitalUnrestricted.
    Eigen::VectorXd orbitalComponentsAlpha;
    /// The beta-channel resolution of E_orb (Hartree per channel): the
    /// own-sign channels of Delta D_b = D/2 - P_orth,b in the Lowdin
    /// basis, one entry per ceil(n/2); the sum over this vector and
    /// orbitalComponentsAlpha is orbitalUnrestricted.
    Eigen::VectorXd orbitalComponentsBeta;
    /// The +nu_k of each alpha-channel pair (electrons; the last entry of
    /// an odd function count holds the unpaired tail eigenvalue, signed).
    Eigen::VectorXd nocvEigenvaluesAlpha;
    /// The +nu_k of each beta-channel pair (electrons; the last entry of
    /// an odd function count holds the unpaired tail eigenvalue, signed).
    Eigen::VectorXd nocvEigenvaluesBeta;
    /// The -nu_k partner of each alpha-channel pair (electrons), in
    /// nocvEigenvaluesAlpha's order; the leftover conventions are those of
    /// nocvEigenvaluesNegative.
    Eigen::VectorXd nocvEigenvaluesAlphaNegative;
    /// The -nu_k partner of each beta-channel pair (electrons), in
    /// nocvEigenvaluesBeta's order; the leftover conventions are those of
    /// nocvEigenvaluesNegative.
    Eigen::VectorXd nocvEigenvaluesBetaNegative;
    /// The combined frozen fragment density P_frag = sum_i P_frag_i
    /// (n x n, spin-summed).
    Eigen::MatrixXd fragmentDensity;
    /// The orthogonalized combined fragment density P_orth = P_a + P_b,
    /// the spin-summed sum of the per-channel projectors onto the combined
    /// occupied spaces (n x n). Each channel is S-idempotent; the combined
    /// matrix is S-idempotent only for closed-shell fragments (equal
    /// channels), which is why the idempotence pins run per channel.
    Eigen::MatrixXd orthogonalizedDensity;
    /// The alpha-channel projector onto the combined alpha occupied space,
    /// C_a (C_a^T S C_a)^-1 C_a^T (n x n, the 1x UHF-density convention).
    Eigen::MatrixXd orthogonalizedDensityAlpha;
    /// The beta-channel projector onto the combined beta occupied space,
    /// C_b (C_b^T S C_b)^-1 C_b^T (n x n, the 1x UHF-density convention).
    Eigen::MatrixXd orthogonalizedDensityBeta;
    /// The spin-summed deformation density Delta P = P_mol - P_orth, the
    /// sum of the per-spin differences (n x n). For the closed-shell
    /// references P_mol is the closed-shell density; for the unrestricted
    /// one it is D_alpha + D_beta of the UHF determinant.
    Eigen::MatrixXd deformationDensity;
    /// The alpha-spin deformation density D_alpha - P_orth,alpha (n x n).
    /// The closed-shell references split their molecular density as
    /// D_alpha = D_beta = D/2, the same convention the per-spin resolution
    /// uses.
    Eigen::MatrixXd deformationDensityAlpha;
    /// The beta-spin deformation density D_beta - P_orth,beta (n x n).
    Eigen::MatrixXd deformationDensityBeta;
};

/// The reference determinant an ETS-NOCV decomposition is resolved against.
/// \ingroup qcx-properties
enum class NocvEtsReference {
    /// A closed-shell determinant (RHF or RKS): D_alpha = D_beta = D/2.
    kClosedShell,
    /// An unrestricted determinant (UHF or UKS): each spin density is its
    /// own projector, and the two NOCV problems are independent.
    kUnrestricted,
    /// A restricted open-shell determinant (ROHF or ROKS). NOT SERVED:
    /// every entry point refuses it by name (see the file comment for what
    /// the restricted formulation still needs).
    kRestrictedOpenShell,
};

/// The ETS-NOCV entry point a reference is checked against.
///
/// The four served entry points cover the two spin classes crossed with the
/// two functional classes: closed-shell Hartree-Fock (AnalyzeNocvEts),
/// unrestricted Hartree-Fock (AnalyzeNocvEtsUnrestricted), closed-shell
/// Kohn-Sham (AnalyzeNocvEtsKohnSham) and unrestricted Kohn-Sham
/// (AnalyzeNocvEtsUnrestrictedKohnSham). AnalyzeNocvEts takes no reference
/// argument; its entry-point value is declared here so the gate's table is
/// total, and it serves the closed-shell reference only.
/// \ingroup qcx-properties
enum class NocvEtsEntryPoint : std::uint8_t {
    /// AnalyzeNocvEts: closed-shell Hartree-Fock.
    kClosedShellHartreeFock,
    /// AnalyzeNocvEtsUnrestricted: unrestricted Hartree-Fock.
    kUnrestrictedHartreeFock,
    /// AnalyzeNocvEtsKohnSham: closed-shell Kohn-Sham.
    kClosedShellKohnSham,
    /// AnalyzeNocvEtsUnrestrictedKohnSham: unrestricted Kohn-Sham.
    kUnrestrictedKohnSham,
};

/// The reference gate the ETS-NOCV entry points share: returns the same
/// Error the named entry point returns for \p reference, or success when
/// that entry point serves it.
///
/// A caller that prepares an expensive analysis setup calls this first, so
/// an unserved reference is refused with the module's own sentence instead
/// of a later allocation or integrals refusal. The restricted open-shell
/// reference is refused by every entry point; a reference of the other spin
/// class than the entry point's is refused with a pointer to the entry
/// point that serves it.
/// \param reference The reference determinant the density came from.
/// \param entryPoint The entry point the caller intends to call.
/// \returns The entry point's refusal, or success.
/// \ingroup qcx-properties
qcx::Result<void> CheckNocvEtsReference(NocvEtsReference reference, NocvEtsEntryPoint entryPoint);

/// The density-functional seam of a Kohn-Sham decomposition: the XC energy
/// and both spin potentials at an arbitrary spin-density pair, plus the
/// functional's exact-exchange fraction.
///
/// The module reaches the XC potential of an arbitrary density THROUGH this
/// seam, and the caller owns its construction (the driver composes it from
/// its XcGridEngine; a test may supply an analytic functional). The seam is
/// a pair callback because the fragment SCFs may be open-shell: a fragment
/// with the Hund multiplicity runs its alpha and beta densities separately,
/// so the XC energy is a functional of the pair, not of the total. Every
/// matrix is in the FULL molecular basis (the ETS fragment convention), so
/// the caller's engine must be the full molecule's - the fragment densities
/// integrate over the full grid with their own AO contraction.
/// \ingroup qcx-properties
struct NocvEtsFunctional {
    /// E_xc and (dE_xc/dD_alpha, dE_xc/dD_beta) at the given per-spin
    /// densities, in the engine's closed/open-shell conventions. Must be
    /// callable for a Kohn-Sham decomposition.
    std::function<qcx::Result<qcx::grid::XcEvaluation>(const Eigen::MatrixXd& densityAlpha,
                                                       const Eigen::MatrixXd& densityBeta)>
        evaluate;
    /// c_HF, the functional's exact-exchange fraction in [0, 1]: the share
    /// of the exchange that stays on the Hartree-Fock path. The XC energy
    /// already carries the (1 - c_HF) share.
    double exchangeFraction = 0.0;
};

/// Runs the ETS-NOCV energy decomposition of the converged molecular
/// density over the given fragment partition.
///
/// The fragments are the disjoint atom-index groups of \p fragments
/// (every atom of the molecule must appear in exactly one group); each
/// runs its own SCF at its molecular geometry in the full molecular basis
/// set (UHF internally - fragments may be open-shell), and the per-spin
/// fragment densities are summed into P_frag_i. The overlap, fragment
/// core Hamiltonians, and the E_elstat/E_Pauli/E_orb values are built
/// internally; the caller supplies the converged molecular density, the
/// full core Hamiltonian and the dense repulsion tensor - the same
/// integrals the molecular SCF used, so the energy identity
/// E_elstat + E_Pauli + E_orb = E_mol - sum_i E_frag_i pins against the
/// molecular run's own total energy on the RHF run this function supports
/// (a Kohn-Sham total energy is not the decomposed energy).
///
/// The decomposition is Hartree-Fock: the fragments run UHF and every
/// energy functional value (E_elstat, E_Pauli, the transition Fock, both
/// E_orb resolutions) is the Hartree-Fock functional. The function takes no
/// density functional, so a Kohn-Sham run's total energy is not the energy
/// it partitions; AnalyzeNocvEtsKohnSham is the functional-aware entry point,
/// and AnalyzeNocvEtsUnrestricted the unrestricted (UHF) one. The supported
/// caller here is an RHF run.
/// \param molecule Validated molecule; the canonical atom order indexes
/// the fragment groups.
/// \param basisSet The molecular basis set (function counts; the fragment
/// SCFs run in this full basis).
/// \param coreHamiltonian The molecular core Hamiltonian H = T + V
/// (n x n) of the converged molecular run.
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical.
/// \param density The converged molecular density (n x n, the spin-summed
/// closed-shell D = 2 C_occ C_occ^T convention).
/// \param fragments The fragment partition: each group holds the atom
/// indices of one fragment; the groups must be disjoint and cover every
/// atom exactly once.
/// \param fragmentOptions Convergence settings of every fragment SCF; a
/// non-converged fragment state is rejected (kInvalidArgument - the ETS
/// identity holds at any density, but the fragment energies and channels
/// would describe a wrong state). The default runs the standard budget.
/// \returns The decomposition, or an Error (kInvalidArgument for a shape
/// mismatch or an invalid fragment partition, or when a fragment SCF does
/// not converge; a fragment atom with Z > 19 returns kUnimplemented - the
/// Aufbau table covers Z = 1..19).
qcx::Result<NocvEtsDecomposition> AnalyzeNocvEts(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions = {});

/// Runs the ETS-NOCV decomposition of an UNRESTRICTED Hartree-Fock
/// determinant: two independent NOCV problems, one per spin, on
/// Delta P_sigma = D_sigma - P_orth,sigma.
///
/// The fragments run their own UHF SCFs exactly as AnalyzeNocvEts's do (the
/// fragment spin states follow the same Hund/closed-shell convention), and
/// the per-spin fragment occupied spaces are joined into P_orth,alpha and
/// P_orth,beta. The energy identity this entry pins is the UHF one,
///     E_UHF[D_a, D_b] - sum_i E_frag,i = E_elstat + E_Pauli + E_orb,
/// with E_Pauli = E_UHF[P_orth,a, P_orth,b] - sum_i E_frag,i - E_elstat and
/// E_orb = sum_sigma Tr[Delta D_sigma F_sigma^TS] resolved into each spin's
/// own sign-paired channels.
///
/// \param molecule Validated molecule; the canonical atom order indexes the
/// fragment groups.
/// \param basisSet The molecular basis set.
/// \param coreHamiltonian The molecular core Hamiltonian H = T + V (n x n).
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical.
/// \param densityAlpha, densityBeta The converged UHF molecular densities
/// (n x n each, the C_sigma,occ C_sigma,occ^T convention - no factor of 2).
/// They need not be converged for the identity to hold, but the reported
/// channels then describe a non-stationary state.
/// \param reference The reference the densities came from; must be
/// NocvEtsReference::kUnrestricted. kRestrictedOpenShell returns
/// kUnimplemented with the missing restricted formulation named, and
/// kClosedShell is refused with a pointer to AnalyzeNocvEts (the restricted
/// path reports the closed-shell functional's split, which this entry does
/// not compute). The declaration is the caller's: a restricted open-shell
/// determinant's per-spin densities are idempotent too, so the module cannot
/// tell one from a UHF determinant by inspection.
/// \param fragments The fragment partition, as AnalyzeNocvEts's.
/// \param fragmentOptions Convergence settings of every fragment SCF; a
/// non-converged fragment state is rejected (kInvalidArgument).
/// \returns The decomposition, or an Error: as AnalyzeNocvEts, plus
/// kInvalidArgument for a shape mismatch or a reference this entry does not
/// serve.
/// \ingroup qcx-properties
qcx::Result<NocvEtsDecomposition> AnalyzeNocvEtsUnrestricted(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    NocvEtsReference reference,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions = {});

/// Runs the ETS-NOCV decomposition of a CLOSED-SHELL Kohn-Sham (RKS) run
/// through the Ziegler-Rauk three-point transition state.
///
/// The orbital relaxation is
///     E_orb = E[rho_mol] - E[rho_orth]
///           = integral_0^1 Tr[Delta rho F[rho_orth + lambda Delta rho]]
///             dlambda,
/// and Simpson's rule on [0, 1] replaces the integrand's three samples by
///     F^TS = 1/6 F[rho_orth] + 2/3 F[rho_TS] + 1/6 F[rho_mol],
///     rho_TS = 1/2 (rho_orth + rho_mol).
/// The rule is exact through third order and reproduces the midpoint
/// exactly for a quadratic (Hartree-Fock) functional, which is the
/// closed-shell limit the module's tests pin. Every Fock and every energy
/// comes from the supplied functional seam: the XC potential is evaluated
/// at each of the three densities, and each fragment runs its own SCF with
/// the same seam (its spin state follows AnalyzeNocvEts's convention), so
/// the decomposition partitions the run's own Kohn-Sham energy rather than a
/// Hartree-Fock energy under the run's label.
///
/// The partition identity this entry pins is
///     E_KS[rho_mol] - sum_i E_frag,i = E_elstat + E_Pauli + E_orb
/// up to the Simpson residual of E_orb (zero on the Hartree-Fock limit; the
/// result's bindingEnergy is formed from the three-point E_orb, so the
/// residual shows as its distance from the run's own binding energy).
///
/// E_Pauli evaluates the CLOSED-SHELL functional at the spin-summed
/// orthogonalized density P_orth = P_orth,alpha + P_orth,beta (the D/2 spin
/// split), the same convention AnalyzeNocvEts uses. The fragments themselves
/// keep their own spin states, so their energies are spin-resolved; the
/// restricted evaluation at P_orth is what makes the zero-XC/c_HF = 1 limit
/// reproduce AnalyzeNocvEts exactly.
/// \param molecule Validated molecule; the canonical atom order indexes the
/// fragment groups.
/// \param basisSet The molecular basis set.
/// \param coreHamiltonian The molecular core Hamiltonian H = T + V (n x n).
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical.
/// \param density The converged RKS molecular density (n x n, the
/// spin-summed closed-shell D = 2 C_occ C_occ^T convention).
/// \param reference The reference the density came from; must be
/// NocvEtsReference::kClosedShell. kRestrictedOpenShell returns
/// kUnimplemented with the missing restricted formulation named;
/// kUnrestricted returns kInvalidArgument naming the per-spin (UKS) entry
/// point that serves it.
/// \param functional The density-functional seam (see NocvEtsFunctional);
/// its evaluate callback must be callable and exchangeFraction in [0, 1].
/// \param fragments The fragment partition, as AnalyzeNocvEts's.
/// \param fragmentOptions Convergence settings of every fragment SCF.
/// \returns The decomposition, or an Error: as AnalyzeNocvEts, plus
/// kInvalidArgument for an empty or invalid functional seam or a reference
/// this entry does not serve.
/// \ingroup qcx-properties
qcx::Result<NocvEtsDecomposition> AnalyzeNocvEtsKohnSham(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& density,
    NocvEtsReference reference,
    const NocvEtsFunctional& functional,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions = {});

/// Runs the ETS-NOCV decomposition of an UNRESTRICTED Kohn-Sham (UKS) run:
/// the two per-spin NOCV problems of AnalyzeNocvEtsUnrestricted, each
/// resolved against its own spin's Ziegler-Rauk three-point transition
/// state through the run's functional seam.
///
/// The orbital relaxation is
///     E_orb = E[D_a, D_b] - E[P_orth,a, P_orth,b]
///           = sum_sigma integral_0^1 Tr[Delta D_sigma
///             F_sigma[P_orth + lambda Delta D]] dlambda,
/// and Simpson's rule on [0, 1] replaces each spin's integrand samples by
///     F_sigma^TS = 1/6 F_sigma[P_orth] + 2/3 F_sigma[P_TS]
///                  + 1/6 F_sigma[D],
///     P_TS = 1/2 (P_orth + D) per spin.
/// Every Fock and every energy comes from the supplied functional seam: the
/// XC potential is evaluated at all three per-spin density pairs, and each
/// fragment runs its own SCF with the same seam (its spin state follows
/// AnalyzeNocvEts's Hund/closed-shell convention, so an open-shell fragment
/// is a UKS fragment). E_Pauli is the functional's value at the
/// orthogonalized spin pair, so the partition identity
///     E_KS[D_a, D_b] - sum_i E_frag,i = E_elstat + E_Pauli + E_orb
/// holds by construction up to the Simpson residual of E_orb (zero on the
/// Hartree-Fock seam, where the three-point Fock collapses to the midpoint).
///
/// The spin channels the closed-shell resolution reports are absent here:
/// the result carries the per-spin channel arrays and leaves
/// orbitalComponents/nocvEigenvalues empty, exactly as
/// AnalyzeNocvEtsUnrestricted does.
/// \param molecule Validated molecule; the canonical atom order indexes the
/// fragment groups.
/// \param basisSet The molecular basis set.
/// \param coreHamiltonian The molecular core Hamiltonian H = T + V (n x n).
/// \param eri Dense (uv|ws) repulsion tensor with shape {n, n, n, n},
/// host-canonical.
/// \param densityAlpha, densityBeta The converged UKS molecular densities
/// (n x n each, the C_sigma,occ C_sigma,occ^T convention - no factor of 2).
/// \param reference The reference the densities came from; must be
/// NocvEtsReference::kUnrestricted. kRestrictedOpenShell returns
/// kUnimplemented with the missing restricted formulation named;
/// kClosedShell returns kInvalidArgument naming the closed-shell (RKS)
/// entry point that serves it.
/// \param functional The density-functional seam (see NocvEtsFunctional);
/// its evaluate callback must be callable and exchangeFraction in [0, 1].
/// \param fragments The fragment partition, as AnalyzeNocvEts's.
/// \param fragmentOptions Convergence settings of every fragment SCF.
/// \returns The decomposition, or an Error: as AnalyzeNocvEts, plus
/// kInvalidArgument for an empty or invalid functional seam or a reference
/// this entry does not serve.
/// \ingroup qcx-properties
qcx::Result<NocvEtsDecomposition> AnalyzeNocvEtsUnrestrictedKohnSham(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& coreHamiltonian,
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    NocvEtsReference reference,
    const NocvEtsFunctional& functional,
    const std::vector<std::vector<std::size_t>>& fragments,
    const qcx::scf::UhfOptions& fragmentOptions = {});

} // namespace qcx::properties
