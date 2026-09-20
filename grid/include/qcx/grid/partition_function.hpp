#pragma once

#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::grid {

/// Bragg-Slater covalent radii in Bohr (used for the Becke partition's
/// interatomic distances R_ab).  Slater, J. Chem. Phys. 41 (1964) 3199.
/// \param atomicNumber Nuclear charge Z.
/// \returns The radius in Bohr; elements without a tabulated value get
/// the 1.0 A fallback radius.
/// \ingroup qcx-grid
double BraggSlaterRadiusBohr(int atomicNumber);

/// Becke fuzzy-cell partition weights (Becke, J. Chem. Phys. 88 (1988)
/// 2547, with the SSF smooth step: Stratmann, Scuseria, Frisch, Chem.
/// Phys. Lett. 257 (1996) 213).
///
/// For every atom a the raw weight is the product over b != a of the
/// switching function s(mu_ab) with mu_ab = (r_a - r_b) / R_ab, where
/// r_a is the distance from the point to atom a and R_ab = rho_a + rho_b
/// is built from the Bragg-Slater radii.  The switch s(mu) is 1 deep
/// inside atom a's cell (mu << -1), 0 deep inside atom b's (mu >> 1),
/// and passes through 1/2 at the cell boundary; the SSF polynomial step
/// is applied over the |mu| <= a window (a = 0.64) and the analytic
/// cutoff beyond it.  The returned weights are normalized to sum to 1 at
/// every point (the partition of unity that the molecular grid needs).
/// Near a nucleus - r_a <= 0.5 (1 - a) d_nearest - the product shortcut
/// gives weight 1 without touching the other atoms.
/// \param pointBohr The quadrature point, in Bohr.
/// \param molecule The molecule whose atoms define the cells.
/// \returns One partition weight per atom, summing to 1.
/// \ingroup qcx-grid
std::vector<double> BeckePartitionWeights(const std::array<double, 3>& pointBohr,
                                          const qcx::molecule::Molecule& molecule);

/// The raw (unnormalized) Becke product weights - see
/// BeckePartitionWeights.  Exposed for tests of the switching function
/// itself.
/// \param pointBohr The quadrature point, in Bohr.
/// \param molecule The molecule whose atoms define the cells.
/// \returns One raw weight per atom (not normalized to sum to 1).
/// \ingroup qcx-grid
std::vector<double> BeckeRawWeights(const std::array<double, 3>& pointBohr,
                                    const qcx::molecule::Molecule& molecule);

/// The raw Becke product weights with the pair-loop work reported.  The
/// extra parameter is a diagnostic for scaling studies and tests: it
/// receives the number of switching-function evaluations the point's pair
/// loop performed.  The exact distance cutoff excludes atoms beyond the
/// partition's decay radius (their weight vanishes analytically), so this
/// count is independent of atoms far away from the point.
/// \param pointBohr The quadrature point, in Bohr.
/// \param molecule The molecule whose atoms define the cells.
/// \param switchCount Receives the number of switching-function evaluations.
/// \returns One raw weight per atom (not normalized to sum to 1).
/// \ingroup qcx-grid
std::vector<double> BeckeRawWeights(const std::array<double, 3>& pointBohr,
                                    const qcx::molecule::Molecule& molecule,
                                    std::size_t& switchCount);

} // namespace qcx::grid
