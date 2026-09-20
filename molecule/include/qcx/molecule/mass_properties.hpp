#pragma once

#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>

namespace qcx::molecule {

/// Center-of-mass and mass-weighted inertia properties of a molecule.
/// \ingroup qcx-molecule
struct MassProperties {
    double totalMass; ///< Sum of isotopic masses (u).
    Eigen::Vector3d centerOfMass; ///< CoM in Bohr (not subtracted from coordinates).
    Eigen::Matrix3d inertiaTensor; ///< Mass-weighted inertia about the CoM (Bohr^2 u).
    Eigen::Vector3d principalMoments; ///< Descending eigenvalues of the inertia tensor.
    Eigen::Matrix3d principalAxes; ///< Eigenvectors as columns, matching principalMoments.
    bool isLinear; ///< Smallest moment is below the linearity threshold; false for a single atom.
};

/// Computes mass properties from the molecule's isotopes and coordinates.
///
/// Parallel blocked reductions over the atoms (deterministic up to
/// floating-point merge order).
/// \param molecule Validated molecule.
/// \param linearityThreshold Smallest-to-largest moment ratio below which the
/// molecule is considered linear.
/// \returns The mass properties.
MassProperties ComputeMassProperties(const Molecule& molecule, double linearityThreshold = 1e-6);

/// Closed-form nuclear repulsion energy: sum of Z_i Z_j / r_ij over pairs.
///
/// Parallel blocked reduction; coordinates must be validated (no coincident
/// atoms - Molecule::Create guarantees it).
/// \param molecule Validated molecule.
/// \returns The repulsion energy in Hartree (Bohr distances, elementary charge).
double NuclearRepulsionEnergy(const Molecule& molecule);

} // namespace qcx::molecule
