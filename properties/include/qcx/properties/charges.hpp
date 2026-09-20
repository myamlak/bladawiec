#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/molecular_grid.hpp"
#include "qcx/properties/populations.hpp"

#include <Eigen/Core>
#include <vector>

namespace qcx::properties {

/// Hirshfeld (stockholder) charges [Hirshfeld1977]: each atom A shares
/// the molecular density at a point proportionally to its free-atom
/// density at that point,
///     Q_A = Z_A - int w_A(r) rho(r) d^3r,
///     w_A(r) = rho_A(r) / sum_B rho_B(r),
/// with rho_A the spherically-averaged free-atom density of A (the
/// promolecular fragment) and rho the molecular density.  The quadrature
/// runs on the caller's molecular grid (the point weights already carry
/// the Becke partition, which merely makes the quadrature converge
/// faster - the charge is the plain ratio-weighted integral).
/// \param molecule The molecule whose atoms partition the density.
/// \param basisSet The molecular basis set (must cover every element).
/// \param grid The molecular integration grid (MolecularGrid).
/// \param density The spin-summed molecular density D = D_alpha +
/// D_beta, as converged by the SCF (HfResult::density).
/// \param fragmentAlpha, fragmentBeta The promolecular per-spin fragment
/// densities, assembled block-diagonal in the molecular AO ordering -
/// exactly the SAD guess's densityAlpha/densityBeta (scf::BuildSadGuess):
/// the fragments are the per-element atomic UHF densities, spherically
/// averaged and scaled to the molecular electron count, embedded per
/// atom.
/// \param aoRanges Per-atom function ranges partitioning [0, n), in the
/// same ordering (AoIndexRangesByAtom).
/// \returns The per-atom charges in molecule atom order, or an Error
/// (kInvalidArgument for a shape mismatch or ranges exceeding n).
/// \ingroup qcx-properties
qcx::Result<Eigen::VectorXd> AnalyzeHirshfeld(const qcx::molecule::Molecule& molecule,
                                              const qcx::basisset::BasisSet& basisSet,
                                              const qcx::grid::MolecularGrid& grid,
                                              const Eigen::MatrixXd& density,
                                              const Eigen::MatrixXd& fragmentAlpha,
                                              const Eigen::MatrixXd& fragmentBeta,
                                              const std::vector<AoRange>& aoRanges);

/// Voronoi (nearest-atom cell) charges: the density is partitioned by
/// nearest-nucleus cells, the plane-bisector Voronoi tessellation
/// ([FonsecaGuerra2004]),
///     Q_A = Z_A - int_{cell A} rho(r) d^3r,
/// with cell A the set of points closer to nucleus A than to any other.
/// Points on a cell boundary go to the lowest-index atom.  The charges
/// are purely geometric - no promolecular density is involved.
/// \param molecule The molecule whose atoms define the cells.
/// \param basisSet The molecular basis set (must cover every element).
/// \param grid The molecular integration grid.
/// \param density The spin-summed molecular density (HfResult::density).
/// \returns The per-atom charges in molecule atom order, or an Error
/// (kInvalidArgument when the density is not n x n with n the AO count).
/// \ingroup qcx-properties
qcx::Result<Eigen::VectorXd> AnalyzeVoronoi(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const qcx::grid::MolecularGrid& grid,
                                            const Eigen::MatrixXd& density);

} // namespace qcx::properties
