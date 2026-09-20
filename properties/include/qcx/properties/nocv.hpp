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

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/uhf.hpp"

#include <Eigen/Core>
#include <cstddef>
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
    /// E_orb of the RESTRICTED (closed-shell) functional: the relaxation
    /// energy of the frozen combined determinant to the molecular density,
    /// equal to the sum of orbitalComponents by construction (the identity
    /// pin of the decomposition). The open-shell fragments' spin-polarized
    /// P_orth makes this the restricted-functional value - the
    /// spin-resolved E_orb is orbitalUnrestricted.
    double orbital = 0.0;
    /// E_orb of the SPIN-RESOLVED (UHF) functional: the relaxation energy
    /// resolved per spin,
    /// E[P_mol] - E[P_orth] = sum_sigma Tr[Delta D_sigma F_sigma^TS],
    /// equal to the sum of the alpha and beta channel components by
    /// construction. Differs from orbital by
    /// 1/4 Tr[(P_a - P_b) K[P_a - P_b]] on the open-shell fragments (the
    /// two functionals coincide on spin-symmetric densities).
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
    /// orbital = sum over this vector.
    Eigen::VectorXd orbitalComponents;
    /// The +nu_k of each sign-paired pair (electrons); for an odd function
    /// count the last entry holds the unpaired tail eigenvalue (signed,
    /// zero for exactly-paired spectra).
    Eigen::VectorXd nocvEigenvalues;
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
/// molecular run's own total energy.
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

} // namespace qcx::properties
