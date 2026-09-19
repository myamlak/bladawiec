#pragma once

#include "qcx/molecule/molecule.hpp"
#include "qcx/symmetry/character_tables.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <Eigen/Core>
#include <vector>

namespace qcx::symmetry {

/// Tolerances of the detector.
/// \ingroup qcx-symmetry
struct DetectOptions {
    /// Absolute position tolerance in Bohr for a permutation match. The
    /// per-fit rotation-angle classification tolerance is DERIVED from it
    /// and the geometry (a position error of ~positionToleranceBohr on an
    /// atom at radius r from the fit's axis perturbs the recovered angle
    /// by ~positionToleranceBohr / r, so near-axis atoms need a wider
    /// angle tolerance than any fixed value).
    double positionToleranceBohr = 1e-4;
    /// Moment ratio below which a molecule counts as linear.
    double linearRatio = 1e-6;
};

/// One verified symmetry operation with its concrete geometry.
/// \ingroup qcx-symmetry
struct SymmetryElement {
    OperationKind kind; ///< What the operation is.
    int order; ///< n for Cn/Sn; 1 for identity, inversion, and mirrors.
    Eigen::Vector3d axis; ///< Normalized rotation axis (mirrors: plane
                          ///< normal); zero for identity and inversion.
};

/// Result of point-group detection.
/// \ingroup qcx-symmetry
struct SymmetryAnalysis {
    PointGroupName group; ///< The full detected group.
    PointGroup computational; ///< group reduced via LargestAbelianSubgroup.
    /// The verified operations: identity, inversion (when present), one
    /// mirror per verified normal, the maximal rotation per axis, and the
    /// verified improper rotations (S2n). Day 8's SALC generator selects the
    /// computational group's subset from this list; powers of a recorded Cn
    /// (e.g. C2 of a C6 axis) are derivable as rotations about the same axis.
    std::vector<SymmetryElement> elements;
};

/// Detects the point group of a molecule, psi4-style [Turney2012].
///
/// The detector centers the molecule on its center of mass and classifies
/// the inertia tensor (linear / spherical / symmetric / asymmetric top),
/// then enumerates the automorphisms of the labeled atom set (same element,
/// all pair distances preserved within tolerance) with a deterministic
/// depth-first search, capped at 256. Each permutation is fit to its
/// proper and improper least-squares orthogonal transformation (Kabsch with
/// nullspace completion for planar sets), and a fit is kept when it maps the
/// atom set onto itself. Finding operations from the structure itself - not
/// from guessed axes - is what catches D2d's diagonal C2' axes, which no
/// pair-direction or plane-normal candidate set contains. Degenerate inertia
/// eigenvectors are never used (their basis is arbitrary), which makes the
/// result deterministic across compiler/library versions; per-fit
/// classification runs in parallel.
///
/// Detection limits: improper rotations S2n are tested for 2n = 4, 6, 8, so
/// D5d/D6d/D7d are not reachable (they need S10/S12/S14); the permutation
/// enumeration is O(order x N^2), fine to ~10^3 atoms. Any three-atom
/// molecule is planar and therefore at least Cs - the in-plane mirror is a
/// genuine symmetry element.
/// \param molecule Validated molecule (coincident atoms already rejected).
/// \param options Tolerances; the defaults match psi4's strategy.
/// \returns The detected group, its Abelian reduction, and the verified
/// elements.
SymmetryAnalysis DetectPointGroup(const qcx::molecule::Molecule& molecule,
                                  const DetectOptions& options = {});

} // namespace qcx::symmetry
