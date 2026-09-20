#pragma once

// Shared benzene/STO-3G construction fixture for the properties tests: the
// published STO-3G C NWChem text (identical to data/basis/sto-3g/C.nwchem),
// benzene at the D6h experimental geometry (r(CC) = 1.395 A, r(CH) = 1.085 A,
// in the xy plane), and the merged C + H basis. Test-only helper - not part
// of the public API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <string_view>

namespace qcx::testing {

/// STO-3G C, published values (identical to data/basis/sto-3g/C.nwchem).
inline constexpr std::string_view kSto3gCarbon = R"(BASIS "ao basis" SPHERICAL PRINT
#BASIS SET: (6s,3p) -> [2s,1p]
C    S
      0.7161683735E+02       0.1543289673E+00
      0.1304509632E+02       0.5353281423E+00
      0.3530512160E+01       0.4446345422E+00
C    SP
      0.2941249355E+01      -0.9996722919E-01       0.1559162750E+00
      0.6834830964E+00       0.3995128261E+00       0.6076837186E+00
      0.2222899159E+00       0.7001154689E+00       0.3919573931E+00
END
)";

/// The merged C + H STO-3G basis of benzene (the H text is the same
/// kSto3gHydrogen literal the water fixture uses).
qcx::Result<qcx::basisset::BasisSet> MakeBenzeneSto3gBasis();

/// Benzene in STO-3G: D6h in the xy plane, r(CC) = 1.395 A
/// (R_C = 2.636168 Bohr), r(CH) = 1.085 A (R_H = 4.686530 Bohr), one
/// carbon/hydrogen pair per 60-degree spoke (C_k at R_C, H_k at R_H on the
/// same ray), singlet. Atom order: the six carbons then the six hydrogens,
/// both counterclockwise from the +x spoke.
qcx::Result<qcx::molecule::Molecule> MakeBenzeneSto3g();

/// The same geometry turned 30 degrees in-plane: the in-plane C2 axes of
/// the D6h fixture land 30 degrees off the global axes, so only the
/// C6-axis C2, the inversion, and the molecular-plane mirror stay signed
/// coordinate permutations of the global frame (the C2h subgroup the
/// symmetry gate falls back to). Test-only helper - not part of the public
/// API.
qcx::Result<qcx::molecule::Molecule> MakeRotatedBenzeneSto3g();

} // namespace qcx::testing
