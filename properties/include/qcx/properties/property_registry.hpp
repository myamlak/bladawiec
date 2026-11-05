#pragma once

#include "qcx/io/run_input.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace qcx::properties {

/// One analysis the driver can run over an SCF result.
///
/// The order is the registry's order: the declaration table holds exactly one
/// entry per enumerator, and the coverage test walks the table against the
/// driver outcomes transcribed from its code paths (the properties module
/// cannot link the driver, so the transcription is the test's input and the
/// driver tests named in each row's evidence are what makes it checkable).
/// The names are the stable words of the coverage record; ToString is their
/// one home, and a test holds every declaration's copy equal to it.
/// \ingroup qcx-properties
enum class PropertyKind : std::uint8_t {
    kPopulations, ///< Mulliken/Lowdin/Mayer/Gopinathan-Jug (always on).
    kMultipoles, ///< Dipole and traceless quadrupole (always on).
    kHirshfeld, ///< Hirshfeld stockholder charges (`properties.hirshfeld`).
    kVoronoi, ///< Voronoi nearest-atom charges (`properties.voronoi`).
    kEsp, ///< CHELPG / Merz-Kollman ESP fit (`properties.esp`).
    kEddb, ///< EDDB delocalization analysis (`properties.eddb`).
    kFukui, ///< Condensed Fukui indices (`properties.fukui`); needs the
            ///< N +- 1 species, so it is not a function of one wavefunction.
    kNalewajski, ///< Nalewajski-Mrozek bond orders (`properties.nalewajski`).
    kDensityAtNuclei, ///< rho(R_A) at every nucleus
                      ///< (`properties.density_at_nuclei`).
    kQtaim, ///< Bader QTAIM critical points (`properties.qtaim`).
    kEtsNocv, ///< ETS-NOCV energy decomposition (`properties.nocv_fragments`).
    kMolden, ///< Molden-format MO export (`properties.molden`).
    kXcGradient, ///< Fixed-density exchange-correlation gradient
                 ///< (`properties.xc_gradient`).
    kTotalGradient, ///< Total analytic nuclear gradient
                    ///< (`properties.total_gradient`).
};

/// Why a property refuses a method kind, by a stable name.
///
/// kNone marks a pair the driver runs. Every other enumerator names one
/// refusal the driver already makes by name, and its comment quotes the
/// sentence that refusal carries, so a reader can match the declaration to the
/// driver text without guessing. kUnclassifiedMethod is the registry's own
/// guard for a method value no enumerator names; it is never a declaration.
/// \ingroup qcx-properties
enum class RefusalReason : std::uint8_t {
    kNone, ///< Supported: the driver runs the property on this method kind.
    /// ETS-NOCV on an unrestricted run. The per-spin resolution assumes the
    /// closed-shell D_sigma = D/2, so a molecular density with two different
    /// spin channels would feed it the wrong ones (`"the ETS-NOCV analysis is
    /// closed-shell in v1"`).
    kEtsNocvClosedShellOnly,
    /// ETS-NOCV on a method naming a functional. No functional enters the
    /// fragment SCFs or the density-energy evaluator, so a DFT total energy
    /// would sit beside Hartree-Fock fragment energies (`"the ETS-NOCV
    /// analysis is Hartree-Fock in v1"`).
    kEtsNocvHartreeFockOnly,
    /// Hirshfeld beyond the SAD atomic-multiplicity table. The promolecular
    /// fragment densities are the SAD guess's, and the v1 table covers
    /// Z <= 10 (`"the SAD atomic-multiplicity table (v1, Z <= 10) does not
    /// cover element Z = <N>"`).
    kHirshfeldSadTableZLimit,
    /// Nalewajski-Mrozek beyond its isolated-atom Aufbau table, which covers
    /// Z <= 19 (`"the Nalewajski-Mrozek Aufbau table covers Z = 1..19"`).
    kNalewajskiFragmentTableZLimit,
    /// ETS-NOCV beyond the fragment Aufbau table for a fragment built as one
    /// atom, so the limit bites only where the partition makes a single-atom
    /// fragment of the element, which covers Z <= 19 (`"the Aufbau table
    /// covers Z = 1..19"`).
    kEtsNocvFragmentTableZLimit,
    /// The fixed-density exchange-correlation gradient on a method that
    /// names no functional, so there is nothing to differentiate
    /// (`"properties.xc_gradient: <method> does not take an
    /// exchange-correlation gradient"`).
    kXcGradientRequiresFunctional,
    /// The total analytic gradient on an unrestricted lane. The assembly
    /// differentiates a converged closed-shell reference (`"properties.
    /// total_gradient: <method> does not take the total analytic gradient"`).
    kTotalGradientRequiresClosedShell,
    /// A method value no qcx::io::MethodType enumerator names. The run path's
    /// own classifier refuses such a word before the registry is consulted;
    /// the value keeps the lookup total and fail-closed.
    kUnclassifiedMethod,
};

/// The number of method kinds a declaration states, one per
/// qcx::io::MethodType enumerator.
/// \ingroup qcx-properties
inline constexpr std::size_t kMethodKindCount = 6;

/// Where a declaration's element limit applies.
/// \ingroup qcx-properties
enum class ElementScope : std::uint8_t {
    /// Any atom of the molecule above the limit refuses the property.
    kMolecule,
    /// Only a fragment the analysis builds as one atom above the limit
    /// refuses it; a multi-atom fragment carries no element table.
    kSingleAtomFragment,
};

/// One property's declared coverage: the driver's current outcome on every
/// method kind, plus the conditions that cut across the kinds.
///
/// This is a coverage RECORD, not a promise about what each analysis could do.
/// The declarations match what the driver does today, including the refusals
/// that exist because the analysis behind the property is not yet wired for a
/// kind. A supported cell is backed by the driver code path named in
/// \c evidence; a refused cell names its reason.
/// \ingroup qcx-properties
struct PropertyDeclaration {
    /// The analysis this declaration describes.
    PropertyKind kind = PropertyKind::kPopulations;
    /// The stable word of the coverage record (ToString(kind)).
    std::string_view name;
    /// The outcome on each method kind, in the method-kind order ToString
    /// documents for qcx::io::MethodType: rhf, uhf, rks, uks, rohf, roks.
    /// kNone means the driver runs the property on that kind.
    std::array<RefusalReason, kMethodKindCount> byMethod{
        RefusalReason::kNone,
        RefusalReason::kNone,
        RefusalReason::kNone,
        RefusalReason::kNone,
        RefusalReason::kNone,
        RefusalReason::kNone,
    };
    /// True when a method that names a density functional refuses the
    /// property even on a kind the method table admits (ETS-NOCV: the
    /// decomposition is Hartree-Fock in v1).
    bool refusesWhenFunctionalNamed = false;
    /// The largest atomic number the analysis covers; 0 means no element
    /// limit.
    std::uint16_t maxAtomicNumber = 0;
    /// What maxAtomicNumber bounds: the molecule's atoms, or only the
    /// fragments the analysis builds as one atom.
    ElementScope elementScope = ElementScope::kMolecule;
    /// The named reason reported for a fragment or molecule above
    /// maxAtomicNumber; kNone when maxAtomicNumber is 0.
    RefusalReason elementRefusal = RefusalReason::kNone;
    /// The driver code path (and, where one exists, the driver test) that
    /// backs the declaration. Required: a supported cell with no evidence is
    /// a claim nobody checked.
    std::string_view evidence;
};

/// The declaration for one property.
/// \param kind The analysis; every PropertyKind enumerator has exactly one
/// declaration.
/// \returns The declaration, owned by the static registry.
/// \ingroup qcx-properties
const PropertyDeclaration& DeclarationFor(PropertyKind kind) noexcept;

/// Every declaration, in PropertyKind order.
/// \returns A view over the static registry table.
/// \ingroup qcx-properties
std::span<const PropertyDeclaration> AllDeclarations() noexcept;

/// The declared outcome for one property on one method kind.
/// \param kind The analysis.
/// \param method The method kind the driver ran.
/// \returns kNone when the driver runs the property on this kind; otherwise
/// the named refusal reason. An out-of-range method value returns
/// kUnclassifiedMethod, never a supported answer.
/// \ingroup qcx-properties
RefusalReason RefusalFor(PropertyKind kind, qcx::io::MethodType method) noexcept;

/// The stable word for a property, e.g. "hirshfeld".
/// \param kind The analysis.
/// \returns The word the coverage record uses.
/// \ingroup qcx-properties
std::string_view ToString(PropertyKind kind) noexcept;

/// The stable word for a refusal reason, e.g. "hirshfeld-sad-table-z-limit".
/// \param reason The refusal reason.
/// \returns The word the coverage record uses.
/// \ingroup qcx-properties
std::string_view ToString(RefusalReason reason) noexcept;

} // namespace qcx::properties
