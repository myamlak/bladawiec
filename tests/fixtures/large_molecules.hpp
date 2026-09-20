#pragma once

// Large, analytically-generated molecule fixtures used for scaling
// measurements and symmetry checks: buckminsterfullerene (Ih) and zigzag
// carbon nanotubes (Dnh). Test-only helper - not part of the public API.

#include "qcx/molecule/molecule.hpp"

#include <cstddef>

namespace qcx::molecule::testing {

/// C60 (Ih): the 60 vertices of the truncated icosahedron, the even
/// permutations of (0, +-1, +-3 phi), (+-1, +-(2+phi), +-2 phi) and
/// (+-phi, +-2, +-(2 phi + 1)), scaled to a 1.40 Angstrom edge length.
qcx::Result<Molecule> MakeBuckminsterfullerene();

/// (n, 0) zigzag carbon nanotube: graphene rolled along n*a1, with 4n atoms
/// per translational cell - four rings of n atoms each (A/B sublattices of
/// n2 = 2c and 2c+1) at z = 3cb, 3cb + b/2, 3cb + 3b/2, 3cb + 2b with
/// theta offsets 0, pi/n, pi/n, 2pi/n (b = 1.42 Angstrom); a total of
/// 4*n*cells atoms. n must be >= 8 - smaller tubes are too strained for the
/// bond heuristic.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): n and cells are the
// standard chiral-index/cell-count pair of nanotube naming.
qcx::Result<Molecule> MakeZigzagNanotube(std::size_t n, std::size_t cells);

} // namespace qcx::molecule::testing
