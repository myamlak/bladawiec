#pragma once

/// \file
/// Shell pairs and the canonical indexing of the batched integral engines.
///
/// Function ordering (shared by every engine): atoms in molecule-canonical
/// order, then element shells in file order, then contraction rows, then the
/// angular components - spherical functions in m ascending -l..+l order
/// (cos(m phi) for m >= 0, sin(m phi) for m < 0), Cartesian components
/// z-slowest (p = x, y, z; d = xx, xy, xz, yy, yz, zz; ...).

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <vector>

namespace qcx::integrals {

/// One flattened shell of the molecule's basis.
/// \ingroup qcx-integrals
struct ShellInfo {
    int angularMomentum; ///< l, 0..6 (the parser cap).
    bool isSpherical; ///< True unless the basis file requested CARTESIAN.
    std::size_t contractionCount; ///< Rows in Shell::coefficients.
    std::size_t functionOffset; ///< Index of the shell's first global function.
    std::size_t atomIndex; ///< The owning atom (molecule order).
    std::size_t elementShellIndex; ///< The shell's index in its ElementBasis.
};

/// The number of basis functions of one shell: the contraction rows times
/// the angular components (2l+1 spherical, (l+1)(l+2)/2 Cartesian). The
/// single definition shared by every engine (the former per-module copies
/// drifted apart).
/// \param shell The shell.
/// \returns The function count.
/// \ingroup qcx-integrals
inline std::size_t ShellFunctionCount(const ShellInfo& shell) noexcept {
    const std::size_t angular = shell.isSpherical
                                    ? static_cast<std::size_t>(2 * shell.angularMomentum + 1)
                                    : static_cast<std::size_t>((shell.angularMomentum + 1) *
                                                               (shell.angularMomentum + 2) / 2);
    return shell.contractionCount * angular;
}

/// A canonical shell pair: i <= j.
/// \ingroup qcx-integrals
struct ShellPairIndex {
    std::size_t i; ///< First shell.
    std::size_t j; ///< Second shell, j >= i.
};

/// A shell quartet (i, j | k, l); not necessarily canonical on input - the
/// batch machinery canonicalizes (i <= j, k <= l, pair index (i,j) >= (k,l))
/// and reports the computed form.
/// \ingroup qcx-integrals
struct ShellQuartet {
    std::size_t i; ///< First bra shell.
    std::size_t j; ///< Second bra shell.
    std::size_t k; ///< First ket shell.
    std::size_t l; ///< Second ket shell.

    /// Defaulted comparison (C++23): keeps ShellQuartet an aggregate, lets
    /// tests and the decorator assert on whole computed sets.
    /// \returns True when all four shell indices are equal.
    bool operator==(const ShellQuartet&) const noexcept = default;
};

/// A 3-center shell triple (i, j | k) with i <= j and k the auxiliary shell
/// (consumed by the RI engine).
/// \ingroup qcx-integrals
struct ShellTriple {
    std::size_t i; ///< First bra shell.
    std::size_t j; ///< Second bra shell, j >= i.
    std::size_t k; ///< Auxiliary shell.
};

/// The flattened shell list plus the canonical pair list of one molecule.
/// \ingroup qcx-integrals
struct ShellPairList {
    std::vector<ShellInfo> shells; ///< Flattened shells (atom, shell, row order).
    std::vector<ShellPairIndex> pairs; ///< Canonical pairs in pair-index order.
    std::size_t functionCount = 0; ///< Total number of basis functions.
};

/// The pair-list index of the canonical pair (i, j): the upper-triangle
/// position i*(2n-i+1)/2 + (j-i).
/// \param i First shell.
/// \param j Second shell, j >= i.
/// \param pairList The pair list the index addresses.
/// \returns The index into ShellPairList::pairs.
/// \ingroup qcx-integrals
inline constexpr std::size_t PairIndexOf(std::size_t i,
                                         std::size_t j,
                                         const ShellPairList& pairList) noexcept {
    const std::size_t n = pairList.shells.size();
    return i * (2 * n - i + 1) / 2 + (j - i);
}

/// Flattens the molecule's basis into the shell list and the canonical pair
/// list (upper triangle, pair-index order).
/// \param molecule Molecule providing the atoms; every element must have a
/// basis entry.
/// \param basisSet Basis set; every shell must satisfy l <= 6 (the parser
/// cap).
/// \returns The pair list, or an Error (kInvalidArgument when an atom has no
/// basis entry, kUnimplemented for a shell beyond the parser cap).
/// \ingroup qcx-integrals
qcx::Result<ShellPairList> BuildShellPairs(const qcx::molecule::Molecule& molecule,
                                           const qcx::basisset::BasisSet& basisSet);

} // namespace qcx::integrals
