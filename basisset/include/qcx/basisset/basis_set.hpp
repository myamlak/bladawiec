#pragma once

#include "qcx/error.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::molecule {
class Molecule;
}

namespace qcx::basisset {

/// \defgroup qcx-basisset Basisset module
/// Gaussian basis sets parsed from NWChem-format data (Basis Set Exchange).

/// One contracted Gaussian shell.
///
/// General contractions (aug-cc and friends) carry several contracted
/// functions sharing one exponent set; the common segmented form has
/// exactly one contraction.
/// \ingroup qcx-basisset
struct Shell {
    int angularMomentum; ///< 0 = s, 1 = p, 2 = d, ...
    bool isSpherical; ///< True unless the file requested CARTESIAN.
    std::vector<double> exponents; ///< One per primitive.
    std::vector<std::vector<double>> coefficients; ///< [contraction][primitive].
};

/// The shells of one element.
/// \ingroup qcx-basisset
struct ElementBasis {
    int atomicNumber; ///< Z.
    std::string symbol; ///< IUPAC symbol, exact case.
    std::vector<Shell> shells; ///< In file order.
};

/// One Gaussian radial term of an ECP channel: c r^n exp(-zeta r^2).
///
/// The radial coordinate r is measured from the ECP centre. The corpus stores
/// n = 2 throughout; the parser keeps the file's integer n unchanged so a term
/// of another power survives a round trip.
/// \ingroup qcx-basisset
struct EcpTerm {
    int power; ///< n, the integer power of r.
    double exponent; ///< zeta, the positive Gaussian exponent.
    double coefficient; ///< c, the term coefficient.
};

/// The angular-momentum value EcpChannel uses for the local (`ul`) channel.
/// \ingroup qcx-basisset
inline constexpr int kEcpLocalChannel = -1;

/// One angular channel of an ECP, including the local channel.
///
/// The local channel is the `ul` block of the NWChem text; the projectors are
/// the S, P, D, F, ... blocks. The terms are kept in file order.
/// \ingroup qcx-basisset
struct EcpChannel {
    int angularMomentum; ///< l, or kEcpLocalChannel for the local `ul` channel.
    std::vector<EcpTerm> terms; ///< In file order.
};

/// An effective-core potential: the core it replaces and its radial channels.
///
/// The local channel is the Gaussian expansion
///
///     V_loc(r) = sum_k c_k r^n_k exp(-zeta_k r^2);
///
/// each projector channel carries its own expansion. The channels and their
/// terms are kept in file order, so the file's term counts and order survive
/// the parse. A file that declared no channels (a bare `SYMBOL nelec N`
/// block) parses to an empty channel list.
/// \ingroup qcx-basisset
struct EcpDefinition {
    int atomicNumber; ///< Z of the element the ECP replaces.
    int coreElectronCount; ///< Electrons removed into the core.
    std::vector<EcpChannel> channels; ///< In file order; empty when none were declared.

    /// The channel for one angular momentum, or nullptr when it is absent.
    /// \param angularMomentum l, or kEcpLocalChannel for the local channel.
    /// \returns The channel, or nullptr when the ECP has no such channel.
    const EcpChannel* FindChannel(int angularMomentum) const noexcept;
};

/// A parsed basis set: one ElementBasis per element, ordered by Z.
/// \ingroup qcx-basisset
class BasisSet {
public:
    /// The element entry for Z, or nullptr when the set has none.
    /// \param atomicNumber Z of the element to find.
    /// \returns The entry, or nullptr when the set has no such element.
    const ElementBasis* Find(int atomicNumber) const noexcept;

    /// The element entry for an exact-case IUPAC symbol, or nullptr.
    /// \param symbol IUPAC symbol (exact case).
    /// \returns The entry, or nullptr when the set has no such element.
    const ElementBasis* Find(std::string_view symbol) const noexcept;

    /// All elements, ascending Z.
    /// \returns The element list.
    const std::vector<ElementBasis>& Elements() const noexcept {
        return _elements;
    }

    /// The ECP definitions, ascending Z.
    /// \returns The ECP list.
    const std::vector<EcpDefinition>& Ecps() const noexcept {
        return _ecps;
    }

    /// Parses one NWChem basis block and merges it into this set.
    /// Elements and ECPs are kept sorted by Z. Repeating an element that is
    /// already in the set, or an ECP for an element whose ECP is already in
    /// the set, is an Error (kInvalidArgument). Spherical contractions are
    /// renormalized to unit norm exactly as ParseNwchemText does, so the
    /// merge is idempotent for already-normalized sets (a second append of
    /// the same block rescales by exactly 1).
    /// \param text The whole file content.
    /// \returns An Error (kInvalidArgument with a line number) on any
    /// grammar or content violation.
    qcx::Result<void> AppendNwchemText(std::string_view text);

    /// Merges another basis set into this one. An element already present,
    /// or an ECP for an element whose ECP is already present, is an Error
    /// (kInvalidArgument). Elements and ECPs stay sorted by Z.
    /// \param other The set to absorb.
    /// \returns An Error (kInvalidArgument) on a duplicate element or ECP.
    qcx::Result<void> Merge(const BasisSet& other);

private:
    std::vector<ElementBasis> _elements;
    std::vector<EcpDefinition> _ecps;
};

/// Number of electrons an effective-core-potential treatment leaves in the
/// valence space.
///
/// Sums (Z - core) over the molecule's atoms and subtracts the molecular
/// charge. The core is the ECP core-electron count of the atom's element in
/// \p basis; an element the basis describes all-electron has no ECP entry and
/// contributes all Z electrons. A basis with no ECPs therefore returns the
/// same count as Molecule::ElectronCount(). The count is basis-dependent: the
/// same element is all-electron in one basis and core-replaced in another.
/// \param molecule The molecule.
/// \param basis The basis set; its ECP entries decide each atom's core.
/// \returns The valence electron count, or an Error (kInvalidArgument) naming
///          the element when an atom is described neither by an orbital basis
///          entry nor by an ECP, or when an ECP's core count is not smaller
///          than Z.
/// \ingroup qcx-basisset
qcx::Result<int> ValenceElectronCount(const qcx::molecule::Molecule& molecule,
                                      const BasisSet& basis);

/// Parses NWChem-format basis text into a new BasisSet.
///
/// Strict grammar: `BASIS "ao basis" [SPHERICAL|CARTESIAN] PRINT`, then one
/// `SYMBOL SHLTYPE` line per element (shell types S, P, D, F, G, H, SP) with
/// one primitive per line - two columns for pure shells (exponent,
/// coefficient), three for SP (exponent, s-coefficient, p-coefficient; an SP
/// shell flattens to two shells whose exponents and coefficients are copied,
/// not shared). An ECP block (`ECP`, then one `SYMBOL nelec N` line per
/// element and one `SYMBOL ul|S|P|D|F|G|H|I` header per channel, each channel
/// followed by `n zeta c` term lines) fills one EcpDefinition per element,
/// channels and terms in file order. `END` terminates the block. Blank lines
/// and lines starting with '#' or '!' are ignored. Unreadable input is
/// kIOError; every grammar/content violation is kInvalidArgument with the
/// offending line number.
///
/// Every spherical-shell contraction is renormalized to unit norm on parse
/// (as all QC codes do): the norm is measured under the module's convention -
/// the (2a/pi)^(3/4) primitive times the solid-harmonic factor of the
/// integrals engine - and an unnormalized basis breaks the RI-J metric (a
/// ~1e20 condition number in (P|Q)). Cartesian shells keep their raw file
/// coefficients (the engine's Cartesian path applies only the (2a/pi)^(3/4)
/// factor; see NormalizeContractions in basis_set.cpp).
/// \param text The whole file content.
/// \returns The parsed basis set, or an Error.
qcx::Result<BasisSet> ParseNwchemText(std::string_view text);

/// Reads and parses one NWChem basis file.
/// \param path Path to the file.
/// \returns The parsed basis set, or an Error (kIOError when unreadable).
qcx::Result<BasisSet> ParseNwchemFile(std::string_view path);

/// Reads and parses every `.nwchem` file in a directory, in parallel, and
/// merges them into one basis set (deterministic: elements end up sorted by
/// Z; the first error in filename order is reported).
/// \param directory Path to a directory of NWChem basis files.
/// \returns The merged basis set, or an Error.
qcx::Result<BasisSet> ParseNwchemDirectory(std::string_view directory);

/// Reads and parses only the `.nwchem` files of the given elements, in
/// parallel, and merges them into one basis set (deterministic, sorted by Z;
/// the first error in filename order is reported).
///
/// The Z values map to file names through qcx::molecule::FindElement's
/// exact-case IUPAC symbol; duplicates are tolerated. The surviving files go
/// through the identical per-file parse as ParseNwchemDirectory, so the
/// entries of the requested elements are bit-identical to the unfiltered
/// parse. The engine (BuildShellPairs) therefore only ever sees the
/// molecule's elements: the parsed set and the engine's molecule-scoped aux
/// function count are equal by construction, which keeps any pre-Create
/// model that sums the parsed set honest (a directory-wide parse would
/// over-count elements the molecule does not contain).
/// \param directory Path to a directory of NWChem basis files.
/// \param atomicNumbers The Z values to keep (duplicates tolerated).
/// \returns The filtered set, or an Error (kInvalidArgument naming the
///          element when no file exists for a requested Z).
qcx::Result<BasisSet> ParseNwchemDirectoryFiltered(std::string_view directory,
                                                   std::span<const int> atomicNumbers);

} // namespace qcx::basisset
