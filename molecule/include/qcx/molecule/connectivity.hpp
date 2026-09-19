#pragma once

#include "qcx/molecule/molecule.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <cstddef>
#include <vector>

namespace qcx::molecule {

/// Default bonding tolerance: 0.3 Angstrom in Bohr (see BuildConnectivity).
inline constexpr double kDefaultConnectivityToleranceBohr = 0.3 * kAngstromToBohr;

/// Undirected atom graph (used later for internal-coordinate generation).
/// \ingroup qcx-molecule
using ConnectivityGraph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;

/// CSR-style neighbor list: the neighbors of atom i are
/// neighbors[offsets[i] .. offsets[i+1]) (sorted).
/// \ingroup qcx-molecule
struct ConnectivityCsr {
    std::vector<std::size_t> offsets; ///< Size atomCount + 1.
    std::vector<std::size_t> neighbors; ///< Packed, per-atom sorted neighbor indices.
};

/// Bond connectivity and its CSR index representation.
/// \ingroup qcx-molecule
struct Connectivity {
    ConnectivityGraph graph; ///< Boost.Graph view of the same edges.
    ConnectivityCsr csr; ///< CSR view of the same edges.
};

/// Builds connectivity from the covalent-radius-sum heuristic.
///
/// Atoms i and j are bonded when distance(i,j) <= r_i + r_j + tolerance,
/// where r are the single-bond covalent radii from the element table;
/// elements with no published radius (0) never bond. The pair loop runs in
/// parallel with one neighbor list per atom (each atom's row is written by
/// exactly one loop iteration - no merge step); a deterministic
/// finalization mirrors and sorts the lists.
/// \param molecule Validated molecule.
/// \param toleranceBohr Extra allowance beyond the radius sum; the default
/// 0.3 Angstrom (in Bohr) catches F2 at its equilibrium distance without
/// bonding next-nearest carbon pairs (C-C 2.42 Angstrom stays unbonded).
/// \returns The graph and its CSR representation.
Connectivity BuildConnectivity(const Molecule& molecule,
                               double toleranceBohr = kDefaultConnectivityToleranceBohr);

} // namespace qcx::molecule
