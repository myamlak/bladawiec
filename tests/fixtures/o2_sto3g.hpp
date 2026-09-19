#pragma once

// Shared O2/STO-3G construction fixtures for the UHF tests: the published
// STO-3G O NWChem text (the same literal values as h2o_sto3g.hpp), and the
// triplet O2 molecule of the UHF pin at R = 2.2818443 Bohr on the
// z axis. Test-only helper - not part of the public API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

namespace qcx::testing {

/// The pyscf UHF/STO-3G total-energy pin of the O2 triplet (2026-08-22,
/// tolerance 1e-7) - the record
/// both the supermatrix and the direct paths assert against.
inline constexpr double kO2PinnedTotalEnergy = -147.63394678545018;

/// The <S^2> pin of the converged O2 triplet (2026-08-22, tolerance 1e-6).
inline constexpr double kO2PinnedSpinSquared = 2.0034108576810308;

/// The STO-3G O basis of the O2 molecule (kSto3gOxygen of h2o_sto3g.hpp).
qcx::Result<qcx::basisset::BasisSet> MakeO2Sto3gBasis();

/// O2 in STO-3G, triplet: O at the origin, O at (0, 0, 2.2818443) Bohr
/// (R = 2.2818443 Bohr = 1.2075 A), charge 0, multiplicity 3.
qcx::Result<qcx::molecule::Molecule> MakeO2Sto3gTriplet();

} // namespace qcx::testing
