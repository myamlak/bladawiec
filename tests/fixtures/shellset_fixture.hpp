#pragma once

// Shared synthetic shellset fixtures for the integral-engine tests: an
// H (s, p, d) + He (s, p) basis on two atoms at R = 1.4 bohr - the
// molecule matching tools/gen_md_reference.py's shellset grid. Test-only
// helper - not part of the public API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <string_view>

namespace qcx::testing {

/// The synthetic shellset basis: H (s, p, d) at the origin, He (s, p) at
/// (1.4, 0, 0) - the molecule matching tools/gen_md_reference.py's
/// shellset grid.
inline constexpr std::string_view kShellsetBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
H    P
      1.0000000000E+00       1.0000000000E+00
      3.0000000000E-01       1.0000000000E+00
H    D
      8.0000000000E-01       1.0000000000E+00
He    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
He    P
      1.0000000000E+00       1.0000000000E+00
      3.0000000000E-01       1.0000000000E+00
END
)";

/// The synthetic shellset molecule: H at the origin, He at (1.4, 0, 0).
qcx::Result<qcx::molecule::Molecule> MakeShellsetMolecule();

} // namespace qcx::testing
