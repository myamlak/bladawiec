#pragma once

// Shared H2/STO-3G construction fixtures for the integral-engine and SCF
// tests: the published STO-3G H NWChem text (the same literal values the
// basisset module tests), an H2 molecule at R = 1.4 bohr, and the error-path
// fixtures. Test-only helper - not part of the public API.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <string_view>

namespace qcx::testing {

/// STO-3G H, published values (same literal text as
/// basisset/tests/basis_set_test.cpp).
inline constexpr std::string_view kSto3gHydrogen = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
END
)";

/// Minimal p-shell basis: exercises the non-s rejection path of every engine.
inline constexpr std::string_view kPOrbitalBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    P
      1.0000000000E+00       1.0000000000E+00
END
)";

/// Parses the STO-3G H text into a BasisSet.
qcx::Result<qcx::basisset::BasisSet> MakeSto3gBasis();

/// H2 in STO-3G: two H atoms on the x axis at R = 1.4 bohr.
qcx::Result<qcx::molecule::Molecule> MakeH2Sto3g();

/// One He atom at the origin: exercises the missing-basis-entry path.
qcx::Result<qcx::molecule::Molecule> MakeHeAtom();

} // namespace qcx::testing
