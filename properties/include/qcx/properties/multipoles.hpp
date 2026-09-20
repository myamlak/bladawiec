#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>

namespace qcx::properties {

/// \file
/// Electric multipole moments: the dipole and the traceless
/// quadrupole. With the total density P = P_alpha + P_beta:
///   d_k = sum_a Z_a (R_a - R0)_k - sum_ij P(i,j) D_k(j,i)
///   M_kl = sum_a Z_a (R_a - R0)_k (R_a - R0)_l - sum_ij P(i,j) Q_kl(j,i)
///   Theta = 3 M - Tr(M) I     (the traceless quadrupole, [Helgaker2000])
/// where D_k / Q_kl are the dipole / quadrupole integral matrices of
/// qcx::integrals (one_electron.hpp), and R0 is the reference origin.
///
/// Origin convention (shared with the integrals builders): all moments are
/// relative to the coordinate origin AS GIVEN - the caller is responsible
/// for recentering the molecule first if a center-of-mass or
/// center-of-charge dipole is wanted. Units are atomic: e*a0 for the
/// dipole, e*a0^2 for the quadrupole.
///
/// Density convention: P is the SPIN-SUMMED density P_alpha + P_beta with
/// the per-spin normalization P_sigma = C_sigma,occ C_sigma,occ^T. An RHF
/// caller passes HfResult::density / 2 per spin (that field is the
/// doubled spin-sum D = 2 rho); a UHF caller passes
/// UhfResult::densityAlpha / densityBeta unchanged - the same convention
/// as the population analyses (populations.hpp).
///
/// Symmetrization: the returned quadrupole is explicitly symmetrized
/// rather than left with a partial fill; every qcx caller reads complete
/// matrix types, and a caller reading the lower half must get the real
/// value, not a stale zero.

/// The electric multipole moments of one density.
struct MultipoleMoments {
    Eigen::Vector3d dipole; ///< (d_x, d_y, d_z), e*a0 relative to the origin.
    Eigen::Matrix3d quadrupole; ///< Traceless quadrupole Theta = 3M - Tr(M) I,
                                ///< e*a0^2, symmetric (explicitly symmetrized).
};

/// Computes the dipole and traceless quadrupole moments of a density.
/// \param molecule Molecule providing the atoms and coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= 6.
/// \param densityAlpha The n x n per-spin alpha density P_alpha.
/// \param densityBeta The n x n per-spin beta density P_beta.
/// \param origin The reference point R0 (Bohr) the moments are measured
/// from; defaults to the coordinate origin. Same convention as the
/// integrals builders (one_electron.hpp).
/// \returns The moments, or an Error (the integrals builders' errors, or
/// kInvalidArgument when the densities disagree with the basis function
/// count).
/// \ingroup qcx-properties
qcx::Result<MultipoleMoments> AnalyzeMultipoles(const qcx::molecule::Molecule& molecule,
                                                const qcx::basisset::BasisSet& basisSet,
                                                const Eigen::MatrixXd& densityAlpha,
                                                const Eigen::MatrixXd& densityBeta,
                                                const std::array<double, 3>& origin = {
                                                    0.0, 0.0, 0.0});

} // namespace qcx::properties
