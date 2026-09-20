#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace qcx::properties {

/// \file
/// Nalewajski-Mrozek bond orders [Nalewajski1996] - the quadratic valence
/// indices (QVIs) of the one-determinantal difference approach: the bond
/// orders and atomic valences measure the second-order change of the
/// charge-and-bond-order (CBO) matrix between the molecule and a
/// promolecular reference.
///
/// Reference state. The frozen superposition-of-isolated-atoms (SAL)
/// reference: each atom is represented by its isolated-atom SCF with the
/// degenerate shell's occupations averaged (here the qcx UHF equivalent -
/// the alpha channel of the isolated atom, whose highest partially filled
/// shell is degenerate so any spanning set yields the same averaged
/// density). The per-atom valence contributions are averaged over all
/// configurations of the degenerate shell (every occupation pattern of the
/// shell's orbitals), always with the closed-shell core states doubly
/// occupied.
///
/// Working basis. Both densities live in the Lowdin-orthogonalized AO
/// basis: the molecular P1 = S^{1/2} P S^{1/2} with the molecular forward
/// root (the Gopinathan-Jug basis shared with populations.hpp and eddb.hpp),
/// the fragment P0 with the atom's own overlap matrix (the frozen-SAL
/// "separately orthogonalized per fragment"). The per-spin difference
/// Delta = P1 - P0 (atom's block) gives the one-center valences:
///   v_i_a = 1/2 sum_i (Delta_ii)^2            (atomic ionic valence),
///   v_c_a = sum_{i<j} (Delta_ij)^2             (atomic covalent valence),
/// both averaged over the fragment configurations; the interatomic
/// quadratic contributions
///   v_ab = sum_{i in a, j in b} (P1_alpha^2 + P1_beta^2)
/// pair the molecular orthogonalized density's off-diagonal blocks. The
/// weighted division of the one-center valences (the paper's Scheme III)
/// yields the bond orders
///   B(a,b) = v_ab (1 + (v_i_a + v_c_a)/total_ab(a) + (v_i_a + v_c_a)/total_ab(b))
/// with total_ab(a) = sum_{b != a} v_ab.
///
/// Sign convention: absolute values throughout (the paper's bonding
/// valences are negative; the squares are what is accumulated here). The
/// config averaging is invariant under rotations of the degenerate-shell
/// spanning set (v_i_a + v_c_a = 1/2 <||Delta||_F^2>_cfg), so the bond
/// orders depend only on the shell, not on the SCF's arbitrary choice of
/// spanning orbitals within it.
///
/// Density convention (identical to populations.hpp / eddb.hpp): the
/// analyses take the per-spin densities P_sigma = C_sigma,occ
/// C_sigma,occ^T with NO factor of 2 anywhere. An RHF caller passes
/// HfResult::density / 2 per spin; a UHF caller passes
/// UhfResult::densityAlpha / densityBeta unchanged.
///
/// The isolated-atom fragment SCFs run internally (properties links
/// qcx-scf; the module DAG allows it) with the molecular basis set and a
/// one-atom molecule at the origin, per distinct element - the same
/// construction the SAD guess uses. The Aufbau state classification covers
/// Z <= 19 ("less than 20 shells"); heavier elements are kUnimplemented.

/// The Nalewajski-Mrozek analysis result.
struct NalewajskiBondOrders {
    /// B(a,b): the Scheme-III bond orders (electrons), symmetric, zero on
    /// the diagonal, in the molecule's canonical atom order.
    Eigen::MatrixXd bondOrders;
    /// v_ab: the interatomic quadratic contributions of the orthogonalized
    /// molecular density, symmetric, zero on the diagonal.
    Eigen::MatrixXd diatomicCovalent;
    /// v_i_a: the atomic ionic valences (the diagonal one-center
    /// displacements, config-averaged).
    Eigen::VectorXd atomicIonicValence;
    /// v_c_a: the atomic covalent valences (the off-diagonal one-center
    /// displacements, config-averaged).
    Eigen::VectorXd atomicCovalentValence;
    /// total_ab(a) = sum_{b != a} v_ab(a, b), the interatomic valence of
    /// atom a (the denominator of the Scheme-III weighting).
    Eigen::VectorXd totalValence;
};

/// Runs the Nalewajski-Mrozek bond-order analysis for a molecule: the
/// overlap matrix, the per-spin orthogonalized densities, the isolated-atom
/// fragment SCFs (one per distinct element, cached), the fragment
/// configuration averaging, and the Scheme-III weighted bond orders.
/// \param molecule Molecule providing the atoms.
/// \param basisSet Basis set; every shell must satisfy l <= 6 and the
/// fragment atoms Z <= 19 (kUnimplemented otherwise).
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \returns The bond-order analysis, or an Error (the integrals/scf
/// builders' errors, kInvalidArgument when the densities disagree with the
/// basis function count, kUnimplemented for Z > 19).
/// \ingroup qcx-properties
qcx::Result<NalewajskiBondOrders> AnalyzeNalewajskiBondOrders(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta);

} // namespace qcx::properties
