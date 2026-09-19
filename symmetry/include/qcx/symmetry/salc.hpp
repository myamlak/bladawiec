#pragma once

#include "qcx/molecule/molecule.hpp"
#include "qcx/symmetry/detection.hpp"
#include "qcx/symmetry/point_group.hpp"

#include <Eigen/Core>
#include <string_view>
#include <vector>

namespace qcx::symmetry {

/// \defgroup qcx-symmetry-salc Symmetry-adapted linear combinations
/// Basis transformation onto the irreps of the computational group.

/// Symmetry-adapted linear combinations over the atom space of a molecule.
///
/// Row r of \ref coefficients is one SALC: coefficients(r, i) is the
/// contribution of atom i; rows are orthonormal and each belongs to one
/// irrep of the computational (Abelian) group.
/// \ingroup qcx-symmetry-salc
struct SalcSet {
    PointGroup group; ///< The computational group the SALCs span.
    std::vector<std::string_view> irrepLabels; ///< Irrep of each row.
    Eigen::MatrixXd coefficients; ///< N x N orthonormal rows (SALCs x atoms).
};

/// Builds the SALCs of a molecule for its detected computational group.
///
/// The projection uses the verified operations from the detection analysis: each
/// orbit of symmetry-equivalent atoms is projected onto the group's irreps
/// with chi-weighted sums. Because the computational groups are Abelian,
/// every orbit carries each occurring irrep at most once (the transitive
/// permutation representation is multiplicity-free), so the projections are
/// automatically orthogonal - no Gram-Schmidt step. Distinct orbits have
/// disjoint supports, so their projections are orthogonal as well; the
/// result is asserted orthonormal to 1e-8 by the tests, not recomputed.
///
/// Axis labeling convention (only the counts are ever asserted - B1/B2
/// labels are label-only): for D2/D2h/C2v/C2h/C2 the 'z' axis is the C2
/// axis with the largest projection onto the highest principal moment; for
/// D2/D2h 'y' is the remaining C2 axis best aligned with the middle moment
/// and perpendicular to 'z', and 'x' is z x y (which must itself be a
/// detected C2 axis - C2 powers of recorded even-order rotations count,
/// e.g. C6^3 of benzene). Mirrors are labeled by their normals with the
/// same rule; a mirror column whose letter matches a fixed axis uses that
/// axis, so molecules with inversion but no detected mirrors (Ih) realize
/// sigma as i x C2. The convention is deterministic: ties break by
/// canonicalized direction comparison. Before projecting, the concrete
/// operations are validated to close under composition - a degenerate
/// inertia tensor (spherical/symmetric top) cannot silently corrupt the
/// multiplicities. Linear molecules are supported: the detector records
/// the C2/mirror elements CinfV/DinfH need.
/// \param molecule Validated molecule.
/// \param analysis Detection result; must have been computed for molecule.
/// \param toleranceBohr Matching tolerance for the permutations; use the
/// same value the detection ran with.
/// \returns The SALC set, or an Error (kInvalidArgument) when the detected
/// elements cannot realize the computational group's operations - an
/// operation does not permute the atoms, a required element is missing, or
/// the concrete operations fail the closure check.
qcx::Result<SalcSet> GenerateSalcs(const qcx::molecule::Molecule& molecule,
                                   const SymmetryAnalysis& analysis,
                                   double toleranceBohr = 1e-4);

} // namespace qcx::symmetry
