#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace qcx::properties {

/// \file
/// Electron Density of Delocalized Bonds (EDDB) - the Bond-Orbital
/// Projection (BOP) algorithm [Szczepanik2014], in the implementation-level
/// formulation of the current runEDDB program [Szczepanik2017BOP]:
/// two-center bond-order orbitals (Jug's bond-order orbitals) and their
/// three-center, through-bridge projections define the part of the density
/// coherently shared through multicenter delocalization channels.
///
/// Working basis. The BOP construction requires an atom-respecting
/// orthonormal basis. The reference program uses natural atomic orbitals;
/// qcx uses the Gopinathan-Jug orthogonalized basis P_ort = S^{1/2} P
/// S^{1/2} (the FORWARD square root, shared with the Lowdin populations and
/// the Gopinathan-Jug bond orders of populations.hpp) - the same
/// "well-localized orthonormalized atomic orbital representation" the
/// method allows in principle, and exactly the representation this
/// implementation builds ("block eigen-decompositions of the Gopinathan-Jug
/// P_ort blocks"). The atomic blocks of P_ort replace the NAO blocks of the
/// reference; all bond-order and projection expressions below carry no
/// overlap matrix because the metric is absorbed in the representation.
///
/// Density convention (identical to populations.hpp): the analyses take
/// the per-spin densities P_sigma = C_sigma,occ C_sigma,occ^T with NO
/// factor of 2 anywhere. An RHF caller passes HfResult::density / 2 per
/// spin; a UHF caller passes UhfResult::densityAlpha / densityBeta
/// unchanged. The two spin channels are processed independently and
/// combined in the final congruence projection.
///
/// The algorithm. For each spin channel independently:
///   1. the connectivity mask M: W_AB = sum_{mu in A, nu in B} P^2_mu,nu
///      (the Wiberg-type bond orders of the orthogonalized density; the
///      closed-shell factor of 1/2 in the D-based definition is absorbed
///      by the per-spin convention). Pairs with W_AB > options.wibergThreshold
///      enter the analysis; atoms with fewer than two selected neighbors are
///      not central and are skipped.
///   2. for every central atom X and selected neighbor A, the two-center
///      coupling P^(2)_AX = [[0, P_AX], [P_XA, 0]] (P_AX the rectangular
///      A-X block of P) is diagonalized; the eigenvectors define the
///      two-center bond-order orbitals (2cBOOs), the n_AX = min(n_A, n_X)
///      largest-positive and most-negative pairs their bonding and
///      antibonding members, with occupations eta = lambda^2.
///   3. for every admissible triplet A-X-B (both A and B selected neighbors
///      of X), the through-bridge coupling P^(3)_AXB (the A-B block
///      deliberately zero) is diagonalized; the leading n_X bonding
///      eigenvectors define the three-center bond-order orbitals (3cBOOs)
///      with occupations theta = lambda^2.
///   4. the 3cBOOs are projected onto the adjacent 2cBOO space, C = G2
///      [B_AX^T; B_XB^T] T_AXB G3 with the occupation-dependent scaling
///      g_tau(x) = x/tau_BOP for x >= tau_BOP (else 1), followed by a
///      thresholded Lowdin orthogonalization. The signed coefficients are
///      accumulated Q_k = Q_{k-1} + c_AX,k c_XB,k^T BEFORE squaring, so
///      phase-coherent (conjugating) and phase-reversed (anticonjugating)
///      channels interfere; the incremental coherence
///      Delta_theta_k = 4 (||Q_k||_F^2 - ||Q_{k-1}||_F^2) weighted by the
///      3cBOO occupation gives the channel populations d_k, with small
///      negative residuals redistributed to the most closely related
///      positive channel (eq. 23-24 of [Szczepanik2017BOP]).
///   5. each 2cBOO of X keeps the largest phase-coherence population any
///      of its triplets assigns it, capped by its own occupation eta
///      (the upper-envelope selection of the through-bridge formalism).
///   6. the raw BOP layer K = sum_X (B_X X_X B_X^T + B^a_X X_X (B^a_X)^T)
///      (bonding and antibonding 2cBOOs share the same retained
///      populations) is converted into the EDDB density by the congruence
///      projection with the molecular density,
///      D~_EDDB = 2 P_alpha K_alpha P_alpha + 2 P_beta K_beta P_beta
///      (D K D for the closed shell), which removes the virtual-space
///      components the off-diagonal blocks introduced.
/// The EDDB population N_EDDB = Tr D~_EDDB and the per-atom diagonal block
/// traces are the delocalization descriptors; the NOBD occupations (the
/// eigenvalues of D~_EDDB) resolve the layer into delocalization orbitals.

/// BOP options (the runEDDB defaults where the reference defines one).
/// \ingroup qcx-properties
struct EddbOptions {
    /// The Wiberg pair gate tau_w: an atom pair enters the analysis when
    /// its Wiberg-type bond order (over the orthogonalized density) exceeds
    /// this value. runEDDB's default is deliberately loose (0.001) - it is
    /// a BOP gate, not a covalent-connectivity criterion.
    double wibergThreshold = 0.001;
    /// The internal BOP threshold tau_BOP controlling the numerical
    /// conditioning of the orbital-projection problem (the occupation
    /// scaling g_tau and the Lowdin orthogonalization's thresholded
    /// inverse root). Not exposed by runEDDB; the historical default of
    /// the earlier implementations is used.
    double bopThreshold = 0.03;
};

/// The EDDB analysis result (per-spin combined, spinless population).
/// \ingroup qcx-properties
struct EddbAnalysis {
    /// N_EDDB = Tr D~_EDDB, the total delocalized population (electrons).
    double totalPopulation = 0.0;
    /// Per-atom delocalized populations, diagonal block traces of
    /// D~_EDDB, in the molecule's canonical atom order (electrons).
    Eigen::VectorXd atomicPopulations;
    /// The EDDB density matrix D~_EDDB in the P_ort (orthogonalized)
    /// working basis; symmetric positive semidefinite. Needed by the
    /// NOBD diagonalization and any real-space EDDB evaluation.
    Eigen::MatrixXd delocalizedDensity;
    /// Occupations of the natural orbitals for bond delocalization (the
    /// eigenvalues of D~_EDDB, descending).
    Eigen::VectorXd nobdOccupations;
    /// Number of central atoms (atoms with >= 2 selected neighbors);
    /// diagnostic.
    std::size_t centralAtomCount = 0;
    /// Number of retained two-center bond-order orbitals; diagnostic.
    std::size_t twoCenterOrbitalCount = 0;
};

/// Runs the EDDB (Bond-Orbital Projection) analysis for a molecule: the
/// overlap matrix, the per-spin orthogonalized densities, the connectivity
/// mask, the two- and three-center bond-order-orbital problems, and the
/// occupied-space congruence projection.
/// \param molecule Molecule providing the atoms.
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \param options BOP thresholds (see EddbOptions).
/// \returns The EDDB analysis, or an Error (the integrals builders' errors,
/// or kInvalidArgument when the densities disagree with the basis function
/// count).
/// \ingroup qcx-properties
qcx::Result<EddbAnalysis> AnalyzeEddb(const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basisSet,
                                      const Eigen::MatrixXd& densityAlpha,
                                      const Eigen::MatrixXd& densityBeta,
                                      const EddbOptions& options = {});

} // namespace qcx::properties
