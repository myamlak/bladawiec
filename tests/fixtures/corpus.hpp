#pragma once

// Regression corpus for point-group detection: molecules whose coordinates
// are exact symmetries by construction (seeds transformed by the group
// generators), so the detector is tested against ground truth, not
// hand-typed approximations. Test-only - not part of the public API.

#include "qcx/molecule/molecule.hpp"

namespace qcx::symmetry::testing {

/// H2O, bent in the yz plane: C2v.
qcx::Result<qcx::molecule::Molecule> MakeWater();

/// Planar C2H4: D2h.
qcx::Result<qcx::molecule::Molecule> MakeEthene();

/// Regular hexagon, C6H6: D6h.
qcx::Result<qcx::molecule::Molecule> MakeBenzene();

/// Pyramidal NH3: C3v.
qcx::Result<qcx::molecule::Molecule> MakeAmmonia();

/// Tetrahedral CH4: Td.
qcx::Result<qcx::molecule::Molecule> MakeMethane();

/// Octahedral SF6: Oh.
qcx::Result<qcx::molecule::Molecule> MakeSulfurHexafluoride();

/// Linear CO2: D infinity h.
qcx::Result<qcx::molecule::Molecule> MakeCarbonDioxide();

/// Linear HF: C infinity v.
qcx::Result<qcx::molecule::Molecule> MakeHydrogenFluoride();

/// Twisted (non-planar) H2O2: C2.
qcx::Result<qcx::molecule::Molecule> MakeTwistedHydrogenPeroxide();

/// Planar trans-HN=NH: C2h.
qcx::Result<qcx::molecule::Molecule> MakeTransDiazene();

/// Allene H2C=C=CH2 with perpendicular end planes: D2d.
qcx::Result<qcx::molecule::Molecule> MakeAllene();

/// meso-CHFCl-CHFCl (substituents mirrored through the center): Ci.
qcx::Result<qcx::molecule::Molecule> MakeMesoDifluoroethane();

/// Planar HOF: Cs.
qcx::Result<qcx::molecule::Molecule> MakeHypofluorousAcid();

/// Eclipsed C2H6: D3h.
qcx::Result<qcx::molecule::Molecule> MakeEclipsedEthane();

/// Staggered C2H6: D3d.
qcx::Result<qcx::molecule::Molecule> MakeStaggeredEthane();

/// S8 crown: D4d.
qcx::Result<qcx::molecule::Molecule> MakeS8Crown();

/// No symmetry at all: C1.
qcx::Result<qcx::molecule::Molecule> MakeGenericC1();

/// Four identical atoms on the D2 orbit of a generic seed (a, b, c): D2.
qcx::Result<qcx::molecule::Molecule> MakeD2Synthetic();

/// Eight atoms on the D2d orbit pair of two seeds: four identical atoms on
/// the C2' axes (the xy diagonals, z = 0) plus four in the allene ring
/// pattern: D2d. The S4 axis (z) carries no atom, so the C2 on it fixes
/// nothing while each C2' fixes its two axis atoms.
qcx::Result<qcx::molecule::Molecule> MakeD2dSynthetic();

/// Four identical atoms on the S4 orbit of a generic seed: pure S4.
qcx::Result<qcx::molecule::Molecule> MakeS4Synthetic();

/// Six identical atoms on the S6 orbit of a generic seed: pure S6.
qcx::Result<qcx::molecule::Molecule> MakeS6Synthetic();

/// Eight identical atoms on the S8 orbit of a generic seed: pure S8.
qcx::Result<qcx::molecule::Molecule> MakeS8Synthetic();

/// Twelve identical atoms on the T orbit of a generic seed: pure T.
qcx::Result<qcx::molecule::Molecule> MakeTSynthetic();

/// The T orbit plus its inverted copy (24 atoms): Th.
qcx::Result<qcx::molecule::Molecule> MakeThSynthetic();

} // namespace qcx::symmetry::testing
