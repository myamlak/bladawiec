#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>

namespace qcx::driver {

/// Counts the molecule-scoped basis functions of a parsed set: the shell
/// function count per atom (2l + 1 spherical, (l + 1)(l + 2) / 2 cartesian)
/// summed over the molecule's atoms. The driver parses filtered per the
/// molecule's elements (ParseNwchemDirectoryFiltered), so the parsed set
/// carries exactly the molecule's elements and this sum equals the
/// integrals engine's molecule-scoped count by construction.
/// \param molecule The molecule the basis belongs to.
/// \param basis The parsed basis set.
/// \returns The basis-function count.
/// \ingroup qcx-driver
std::size_t CountBasisFunctions(const qcx::molecule::Molecule& molecule,
                                const qcx::basisset::BasisSet& basis) noexcept;

/// Counts the molecule-scoped shell-pair count (nShell (nShell + 1) / 2),
/// the size the Create-time Schwarz neighbor CSR is bounded by.
/// \param molecule The molecule the basis belongs to.
/// \param basis The parsed basis set.
/// \returns The shell-pair count.
/// \ingroup qcx-driver
std::size_t CountShellPairs(const qcx::molecule::Molecule& molecule,
                            const qcx::basisset::BasisSet& basis) noexcept;

} // namespace qcx::driver
