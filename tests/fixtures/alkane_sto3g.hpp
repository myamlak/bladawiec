#pragma once

// The STO-3G alkane fixtures (the spatially-extended molecule and the
// acceptance suite of the QFMM stage):
// C_nH_{2n+2} built programmatically for any carbon count at a PERFECTLY
// tetrahedral geometry - the backbone C-C-C angle is the tetrahedral
// angle (109.47 deg, so the planar trans zigzag's internal carbons host
// exact tetrahedral C-H directions out of the backbone plane) and the
// terminal carbons get the exact tetrahedral 3-H fan. The carbon text is
// the vendored data/basis/sto-3g/C.nwchem values (the same basis set as
// niedoida's reference g94-sto-3g.base, which prints 8 significant
// digits); hydrogen comes from
// h2_sto3g.hpp. Test-only helper - not part of the public API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <string_view>

namespace qcx::testing {

/// STO-3G C, the vendored data/basis/sto-3g/C.nwchem values.
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

/// The merged C + H STO-3G basis of the alkane fixture.
qcx::Result<qcx::basisset::BasisSet> MakeAlkaneSto3gBasis();

/// A linear alkane C_nH_{2n+2}: the carbon backbone is a planar trans
/// zigzag in the xy plane at C-C 1.538 A (2.9066 Bohr) and the
/// tetrahedral C-C-C angle; each carbon carries C-H 1.09 A bonds - the
/// exact tetrahedral 3-fan on the terminals, the exact tetrahedral pair
/// (out of the backbone plane) on the internal carbons. Neutral singlet.
/// \param carbonCount Number of carbon atoms, must be >= 1.
/// \returns The molecule, or an Error (kInvalidArgument for zero carbons).
qcx::Result<qcx::molecule::Molecule> MakeAlkaneSto3g(std::size_t carbonCount);

} // namespace qcx::testing
