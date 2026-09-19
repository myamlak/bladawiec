#pragma once

// Synthetic test systems for the grid module's screening gates.
//
// A screening gate only means something on a system that is spatially
// extended: on a compact molecule every shell pair survives the screen and
// the gate measures nothing.  The generator here builds the simplest such
// system - a collinear hydrogen chain - so a caller can ask for any size
// and get a geometry whose extent tracks that size.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>

namespace qcx::grid {

/// A generated test system: a molecule plus the basis it is meant to run in.
///
/// The fields are the public API (aggregate-struct exemption).  The
/// struct is move-only, because Molecule is.
/// \ingroup qcx-grid
struct SyntheticSystem {
    qcx::molecule::Molecule molecule; ///< The generated geometry, in canonical atom order.
    qcx::basisset::BasisSet basis; ///< The matching basis: one s shell per atom.
};

/// Builds a collinear hydrogen chain of \p atomCount atoms, 1.4 Bohr apart.
///
/// Atom i sits at (1.4 * i, 0, 0) Bohr, coordinates generated from the
/// index rather than stored as a table, so the chain spans exactly
/// 1.4 * (atomCount - 1) Bohr end to end and grows with every requested
/// atom.  The basis is STO-3G hydrogen (three contracted primitives), whose
/// single s shell gives one basis function per atom: \p atomCount atoms
/// yield \p atomCount functions in \p atomCount shells.
///
/// The system is neutral (charge 0) with multiplicity 1, and is fully
/// determined by \p atomCount - repeated calls produce bit-identical
/// coordinates.
/// \param atomCount Number of hydrogen atoms; at least 1.
/// \returns The molecule and its basis, or an Error (kInvalidArgument when
///          \p atomCount is zero; otherwise the construction errors of the
///          molecule and basis-set factories).
/// \ingroup qcx-grid
qcx::Result<SyntheticSystem> MakeSyntheticChain(std::size_t atomCount);

} // namespace qcx::grid
