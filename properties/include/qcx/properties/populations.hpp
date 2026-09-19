#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace qcx::properties {

/// \defgroup qcx-properties Properties module
/// Electron-structure analyses over converged SCF densities: populations
/// and bond orders, multipole moments, density-partitioning charges,
/// Fukui indices, ESP fits, EDDB delocalization, and the ETS-NOCV
/// energy decomposition.
/// \{

/// \file
/// Atomic population and bond-order analyses: Mulliken
/// [Mulliken1955], Lowdin [Lowdin1950], Mayer [Mayer1983], and
/// Gopinathan-Jug [Gopinathan1983], ported from the read-only niedoida
/// reference repo (property_kit/mulliken_analysis.cpp, lowdin_analysis.cpp,
/// mayer_analysis.cpp, gopinathan_jug_analysis.cpp - algorithms and traps
/// only, never code).
///
/// Density convention (matching the scf module's per-spin convention): the
/// analyses take the per-spin densities P_sigma = C_sigma,occ
/// C_sigma,occ^T with NO factor of 2 anywhere. An RHF caller passes
/// HfResult::density / 2 per spin - that field is the spin-summed
/// D = 2 rho = 2 (P_alpha + P_beta), and the two spins coincide for the
/// closed shell. A UHF caller passes UhfResult::densityAlpha /
/// densityBeta unchanged. <S^2> needs no computation here: UHF already
/// reports it as UhfResult::spinSquared, and the RHF path has no spin
/// contamination to report - consumers forward that field as-is.
///
/// All populations are in electrons; every result is per atom in the
/// molecule's canonical atom order (molecule.hpp), which is also the order
/// the basis set's shells were assigned (shell_pairs.hpp).
///
/// Shape contract: overlap, densityAlpha, and densityBeta must be n x n
/// with n the total basis-function count, and aoRanges must partition
/// [0, n) per atom (see AoIndexRangesByAtom).

/// Per-atom AO index range.
struct AoRange {
    std::size_t firstFunction = 0; ///< First basis-function index of the atom.
    std::size_t functionCount = 0; ///< Number of basis functions on the atom.
};

/// Mulliken gross populations ([Mulliken1955]): the atomic population of
/// atom a in spin channel sigma is q_sigma(a) = sum_{i in a, j}
/// P_sigma(i,j) S(i,j) - the sum of the diagonal of P_sigma S over the
/// atom's functions (the "overlap population" row sums of niedoida's
/// mulliken_analysis.cpp). The per-AO gross orbital populations are the
/// full diagonal of P_sigma S.
struct MullikenPopulations {
    Eigen::VectorXd alpha; ///< Per-atom gross alpha population (electrons).
    Eigen::VectorXd beta; ///< Per-atom gross beta population (electrons).
    Eigen::VectorXd total; ///< alpha + beta (electrons).
    Eigen::VectorXd spin; ///< alpha - beta (the Mulliken spin population).
    Eigen::VectorXd orbitalAlpha; ///< Per-AO gross alpha population, diagonal of P_alpha S.
    Eigen::VectorXd orbitalBeta; ///< Per-AO gross beta population, diagonal of P_beta S.
    Eigen::VectorXd orbitalTotal; ///< Per-AO gross total population.
};

/// Lowdin populations ([Lowdin1950], niedoida's lowdin_analysis.cpp
/// contract): with P_ort = S^{1/2} P S^{1/2} (the FORWARD square root -
/// distinct from scf's S^{-1/2} symmetric orthogonalization), the atomic
/// population of atom a is the sum of the P_ort diagonal over the atom's
/// functions.
struct LowdinPopulations {
    Eigen::VectorXd alpha; ///< Per-atom Lowdin alpha population (electrons).
    Eigen::VectorXd beta; ///< Per-atom Lowdin beta population (electrons).
    Eigen::VectorXd total; ///< alpha + beta (electrons).
    Eigen::VectorXd spin; ///< alpha - beta (the Lowdin spin population).
};

/// Mayer bond orders and valences ([Mayer1983], niedoida's
/// mayer_analysis.cpp). With PS_sigma = P_sigma S:
///   B(a,b) = 2 sum_{i in a, j in b} [PS_alpha(i,j) PS_alpha(j,i) +
///                                   PS_beta(i,j) PS_beta(j,i)]   (a != b)
///   F(a) = sum_{i,j in a} [PS_alpha - PS_beta](i,j) [PS_alpha - PS_beta](j,i)
///   V(a) = F(a) + sum_{b != a} B(a,b)
/// The free valence is identically zero for RHF (P_alpha = P_beta there);
/// it is a meaningful UHF diagnostic only.
struct MayerAnalysis {
    Eigen::MatrixXd bondOrders; ///< B(a,b), symmetric, zero diagonal (electrons).
    Eigen::VectorXd freeValences; ///< F(a); zero for RHF by construction.
    Eigen::VectorXd totalValences; ///< V(a) = F(a) + sum_{b != a} B(a,b).
};

/// Gopinathan-Jug bond orders ([Gopinathan1983], niedoida's
/// gopinathan_jug_analysis.cpp): B(a,b) = sum_{i in a, j in b}
/// [P_ort_alpha(i,j)^2 + P_ort_beta(i,j)^2] over ALL pairs including the
/// a == b diagonal. The self-term is a mathematical artifact of the
/// formula's definition, not a "bond to itself"; it is included here
/// exactly as the reference computes it, and consumers that mean a
/// bond-order matrix proper (e.g. the EDDB stage) drop the diagonal
/// themselves.
struct GopinathanJugAnalysis {
    Eigen::MatrixXd bondOrders; ///< B(a,b), symmetric, diagonal included.
};

/// The four analyses together (the population block): the overlap
/// matrix and the symmetric square root are computed once and shared.
/// \ingroup qcx-properties
struct PopulationAnalysis {
    MullikenPopulations mulliken; ///< The Mulliken gross populations.
    LowdinPopulations lowdin; ///< The Lowdin populations.
    MayerAnalysis mayer; ///< The Mayer bond orders and valences.
    GopinathanJugAnalysis gopinathanJug; ///< The Gopinathan-Jug bond orders.
};

/// Builds the per-atom AO index ranges from the molecule and basis set:
/// for every atom, the half-open [firstFunction, firstFunction +
/// functionCount) interval into the canonical function order. Shells are
/// assigned in atom order (integrals' shell_pairs.hpp), so each atom's
/// functions form one contiguous range.
/// \param molecule Molecule providing the atoms.
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \returns One AoRange per atom, in atom order, or an Error
/// (kUnimplemented for a shell beyond the parser cap, kInvalidArgument
/// when an atom has no basis entry).
/// \ingroup qcx-properties
qcx::Result<std::vector<AoRange>> AoIndexRangesByAtom(const qcx::molecule::Molecule& molecule,
                                                      const qcx::basisset::BasisSet& basisSet);

/// The forward square root S^{1/2} (SymmetricOrthogonalization's inverse-
/// root cousin): U diag(sqrt(s)) U^T from the overlap's eigendecomposition.
/// Needed by Lowdin populations and Gopinathan-Jug bond orders; compute it
/// ONCE and share it between them. Note this is NOT scf's S^{-1/2}
/// (scf_common's OrthogonalizeOverlap): the two are different matrices,
/// and this module lives after scf in the module DAG anyway.
/// \param overlap Symmetric positive-definite n x n overlap matrix.
/// \returns The symmetric square root S^{1/2}.
/// \ingroup qcx-properties
Eigen::MatrixXd SymmetricSquareRootOfOverlap(const Eigen::MatrixXd& overlap);

/// Mulliken gross population analysis.
/// \param overlap The n x n overlap matrix S.
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \param aoRanges Per-atom function ranges partitioning [0, n).
/// \returns The Mulliken populations, or an Error (kInvalidArgument when
/// the matrices disagree in shape or the ranges exceed n).
/// \ingroup qcx-properties
qcx::Result<MullikenPopulations> AnalyzeMulliken(const Eigen::MatrixXd& overlap,
                                                 const Eigen::MatrixXd& densityAlpha,
                                                 const Eigen::MatrixXd& densityBeta,
                                                 const std::vector<AoRange>& aoRanges);

/// Lowdin population analysis through the forward root (S^{1/2} P S^{1/2}).
/// \param overlap The n x n overlap matrix S.
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \param aoRanges Per-atom function ranges partitioning [0, n).
/// \returns The Lowdin populations, or an Error (kInvalidArgument when
/// the matrices disagree in shape or the ranges exceed n).
/// \ingroup qcx-properties
qcx::Result<LowdinPopulations> AnalyzeLowdin(const Eigen::MatrixXd& overlap,
                                             const Eigen::MatrixXd& densityAlpha,
                                             const Eigen::MatrixXd& densityBeta,
                                             const std::vector<AoRange>& aoRanges);

/// Mayer bond-order and valence analysis.
/// \param overlap The n x n overlap matrix S.
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \param aoRanges Per-atom function ranges partitioning [0, n).
/// \returns The Mayer analysis, or an Error (kInvalidArgument when
/// the matrices disagree in shape or the ranges exceed n).
/// \ingroup qcx-properties
qcx::Result<MayerAnalysis> AnalyzeMayer(const Eigen::MatrixXd& overlap,
                                        const Eigen::MatrixXd& densityAlpha,
                                        const Eigen::MatrixXd& densityBeta,
                                        const std::vector<AoRange>& aoRanges);

/// Gopinathan-Jug bond-order analysis through the forward root.
/// \param overlap The n x n overlap matrix S.
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \param aoRanges Per-atom function ranges partitioning [0, n).
/// \returns The Gopinathan-Jug bond orders, or an Error (kInvalidArgument
/// when the matrices disagree in shape or the ranges exceed n).
/// \ingroup qcx-properties
qcx::Result<GopinathanJugAnalysis> AnalyzeGopinathanJug(const Eigen::MatrixXd& overlap,
                                                        const Eigen::MatrixXd& densityAlpha,
                                                        const Eigen::MatrixXd& densityBeta,
                                                        const std::vector<AoRange>& aoRanges);

/// Runs the full population block (Mulliken, Lowdin, Mayer,
/// Gopinathan-Jug) for a molecule, building the overlap matrix and the
/// per-atom ranges internally.
/// \param molecule Molecule providing the atoms.
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \returns The four analyses, or an Error (the integrals builders'
/// errors, or kInvalidArgument when the densities disagree with the basis
/// function count).
/// \ingroup qcx-properties
qcx::Result<PopulationAnalysis> AnalyzePopulations(const qcx::molecule::Molecule& molecule,
                                                   const qcx::basisset::BasisSet& basisSet,
                                                   const Eigen::MatrixXd& densityAlpha,
                                                   const Eigen::MatrixXd& densityBeta);

/// \}

} // namespace qcx::properties
