#pragma once

// Shared HF/STO-3G construction fixtures for the MD-engine and SCF tests:
// the published STO-3G F and H NWChem texts (the same literal values the
// vendored data/basis/sto-3g corpus carries), an HF molecule at the
// experimental bond length R = 0.9168 A, and the merged basis. Test-only
// helper - not part of the public API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <string_view>

namespace qcx::testing {

/// STO-3G F, published values (identical to data/basis/sto-3g/F.nwchem).
inline constexpr std::string_view kSto3gFluorine = R"(BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (6s,3p) -> [2s,1p]
F    S
      0.1666791340E+03       0.1543289673E+00
      0.3036081233E+02       0.5353281423E+00
      0.8216820672E+01       0.4446345422E+00
F    SP
      0.6464803249E+01      -0.9996722919E-01       0.1559162750E+00
      0.1502281245E+01       0.3995128261E+00       0.6076837186E+00
      0.4885884864E+00       0.7001154689E+00       0.3919573931E+00
END
)";

/// The merged F + H STO-3G basis of the HF molecule.
qcx::Result<qcx::basisset::BasisSet> MakeHfSto3gBasis();

/// HF in STO-3G: F at the origin, H on the +x axis at R = 0.9168 A
/// (1.732500911 Bohr), singlet.
qcx::Result<qcx::molecule::Molecule> MakeHfSto3g();

} // namespace qcx::testing
