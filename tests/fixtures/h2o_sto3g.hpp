#pragma once

// Shared H2O/STO-3G construction fixtures for the MD-engine and SCF tests:
// the published STO-3G O and H NWChem texts (the same literal values the
// vendored data/basis/sto-3g corpus carries), a water molecule at the
// experimental geometry (r(OH) = 0.9572 A, angle = 104.52 deg, in the xy
// plane), and the merged basis. Test-only helper - not part of the public
// API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <string_view>

namespace qcx::testing {

/// STO-3G O, published values (identical to data/basis/sto-3g/O.nwchem).
inline constexpr std::string_view kSto3gOxygen = R"(BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (6s,3p) -> [2s,1p]
O    S
      0.1307093214E+03       0.1543289673E+00
      0.2380886605E+02       0.5353281423E+00
      0.6443608313E+01       0.4446345422E+00
O    SP
      0.5033151319E+01      -0.9996722919E-01       0.1559162750E+00
      0.1169596125E+01       0.3995128261E+00       0.6076837186E+00
      0.3803889600E+00       0.7001154689E+00       0.3919573931E+00
END
)";

/// The merged O + H STO-3G basis of the water molecule.
qcx::Result<qcx::basisset::BasisSet> MakeH2oSto3gBasis();

/// H2O in STO-3G: O at the origin, the H atoms at (+-d, h, 0) with
/// d = 1.430428808 Bohr, h = 1.107157044 Bohr (r = 0.9572 A,
/// angle = 104.52 deg), singlet.
qcx::Result<qcx::molecule::Molecule> MakeH2oSto3g();

} // namespace qcx::testing
