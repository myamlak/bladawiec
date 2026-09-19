// The Molden-format F-file writer:
// the [Molden Format] / [Atoms] (AU) / [5D] / [7F] / [GTO] / [MO] sections
// of one converged (or last-iterate) SCF result, for Molden, ORCA, Jmol and
// pyscf consumers. Two conventions are encoded deliberately, both
// documented on WriteMoldenFile below: the [GTO] contraction coefficients
// are emitted AS STORED (N_l(zeta) applies at evaluation only), and the
// [MO] coefficient rows are permuted from the qcx spherical m order
// (m = -l..+l, m < 0 sine, m > 0 cosine) to the molden 5D/7F order
// (0, +1, -1, +2, -2, ...) - the l = 1..3 tables in the spec and pyscf's
// order_ao_index are the external-convention referees (cited, not executed).

#pragma once

#include "qcx/error.hpp"

#include <Eigen/Dense>
#include <span>
#include <string_view>

namespace qcx::molecule {
class Molecule;
}

namespace qcx::basisset {
class BasisSet;
}

namespace qcx::io {

/// One [MO] block: the orbitals of one spin channel.
/// \ingroup qcx-io
struct MoldenMolecularOrbitals {
    std::string_view spin; ///< "Alpha" | "Beta".
    const Eigen::MatrixXd& coefficients; ///< n x n, columns ascending by energy.
    const Eigen::VectorXd& energies; ///< n, hartree.
    const Eigen::VectorXd& occupations; ///< n, aufbau (2/0 or 1/0).
};

/// Writes the [Molden Format] [Atoms] (AU) [5D] [7F] [GTO] [MO] sections
/// of one SCF result to `path` (overwritten when present, standard tool
/// behavior). The [Atoms] coordinates are the molecule's Bohr coordinates
/// (the Bohr contract - the driver's molecule is already in Bohr),
/// 12 decimals. The [GTO] coefficients are the STORED spherical
/// coefficients as-is: they pair with the consumer's own normalization and
/// equal the qcx basis exactly. The stored coefficients are not
/// recoverable from the file when the parse-time rescale s != 1 (the
/// BasisSet parse normalizes each contraction to unit norm; the map back
/// is not injective - named limitation; the s = 1 class covers
/// STO-3G/6-31G*/def2-class segmented sets). A cartesian shell is not
/// exportable (v1 is spherical everywhere). The [MO] rows carry the
/// 1-based molden function order: m = 0 -> slot 1, m > 0 -> slot 2m,
/// m < 0 -> slot 2|m|+1 (l = 1: [3,1,2], l = 2: [5,3,1,2,4],
/// l = 3: [7,5,3,1,2,4,6] - the spec's 5D/7F tables, matching pyscf's
/// order_ao_index). Occupations are taken from the input vectors verbatim
/// (the driver's aufbau responsibility), all n rows per block.
/// \param molecule The result's molecule; atoms in canonical order.
/// \param basis The AO basis: per atom the element's shells in file order,
///        functions in the shell's m order.
/// \param moBlocks One block per spin channel (RHF: one, Spin= Alpha;
///        UHF: alpha then beta).
/// \param path The output file; overwritten when present.
/// \returns An Error: kIOError when the file cannot be written,
///          kUnimplemented for a cartesian shell, kInvalidArgument for a
///          shape mismatch (coefficients not n x n / energies /
///          occupations not n) or a basis missing an element of the
///          molecule.
/// \ingroup qcx-io
qcx::Result<void> WriteMoldenFile(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basis,
                                  std::span<const MoldenMolecularOrbitals> moBlocks,
                                  std::string_view path);

} // namespace qcx::io
