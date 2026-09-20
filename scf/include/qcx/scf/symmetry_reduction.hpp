#pragma once

/// \file
/// The AO-space symmetry reduction of a molecule's basis:
/// the action of the detected computational point group on
/// the basis functions as signed permutations - the data the integrals
/// module's Fock builders consume to evaluate symmetry-unique ERI classes
/// once. Built here because the scf module owns the group realization
/// (internal/symmetry_blocks.cpp); consumed by qcx::integrals through the
/// reduction type.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/molecule/molecule.hpp"

namespace qcx::scf {

/// Builds the AO-space symmetry reduction of a molecule's basis.
///
/// Detects the point group of the molecule (the computational group),
/// realizes its elements from the detected symmetry, and extracts the
/// signed-permutation action on the basis functions. The petite list
/// requires every element's action to be a signed permutation in the global
/// function basis, which holds exactly when the molecule's symmetry
/// planes/axes coincide with the global coordinate planes/axes (the
/// standard fixture orientation).
/// \param molecule The molecule; its point group is detected internally.
/// \param basisSet The basis; every shell must satisfy l <= 6 (the parser
/// cap).
/// \returns The reduction (trivial for C1 molecules), or an Error:
/// kUnimplemented when the group is not realizable or its action is not a
/// signed permutation in the global function basis - callers fall back to
/// the plain path; kInvalidArgument when the detection is inconsistent with
/// the molecule or a basis entry is missing.
/// \ingroup qcx-scf
qcx::Result<qcx::integrals::SymmetryReduction> BuildSymmetryReduction(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet);

} // namespace qcx::scf
