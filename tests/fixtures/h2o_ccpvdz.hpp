#pragma once

// Shared H2O/cc-pVDZ construction fixtures for the storage tests:
// the vendored cc-pvdz corpus (data/basis/cc-pvdz, ParseNwchemDirectory)
// for the orbital basis, the vendored cc-pvdz-rifit auxiliary corpus for
// the RI-3c tests, and the water molecule (same experimental geometry as
// MakeH2oSto3g - the molecule is basis-independent). Test-only helper -
// not part of the public API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

namespace qcx::testing {

/// The cc-pVDZ basis of the water molecule, parsed from the vendored
/// data/basis/cc-pvdz corpus. Requires the QcxBasisDataDir compile
/// definition (the fixture library carries it).
qcx::Result<qcx::basisset::BasisSet> MakeH2oCcpvdzBasis();

/// The cc-pvdz-rifit auxiliary basis (RI-3c tests, the ri_engine_test
/// pattern), parsed from the vendored data/basis/cc-pvdz-rifit corpus.
qcx::Result<qcx::basisset::BasisSet> MakeH2oCcpvdzRifitBasis();

/// H2O at the experimental geometry (same molecule as MakeH2oSto3g:
/// O at the origin, H at (+-d, h, 0), d = 1.430428808 Bohr,
/// h = 1.107157044 Bohr), singlet, charge 0.
qcx::Result<qcx::molecule::Molecule> MakeH2oCcpvdz();

} // namespace qcx::testing
