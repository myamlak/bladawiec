#pragma once

#include <optional>
#include <string>

namespace qcx::dispersion {

/// \defgroup qcx-dispersion Dispersion corrections
///
/// The geometry-only London dispersion corrections: Grimme's D3
/// zero-damping scheme and the two-body part of the D4 default model. The
/// module owns the request vocabulary, the parameter resolution and the
/// evaluation seam onto excgrid. It depends on neither `io/` nor `scf/`: an
/// input parser fills the request type below, and the run driver resolves
/// and evaluates it.
/// \{

/// The dispersion correction a run selected.
enum class DispersionForm {
    kD3, ///< The D3 zero-damping scheme.
    kD4, ///< The D4 default model, its two-body energy (Becke-Johnson damping).
};

/// Where a resolved parameter set came from.
enum class DispersionParameterSource {
    kMethod, ///< The run method's own family, looked up in the form's table.
    kPreset, ///< A set named by `dispersion.preset`.
    kExplicit, ///< Every value of the form's numeric set, written by the run.
};

/// The D3 numeric parameters as an input wrote them.
///
/// One optional per published key, so a validator can name exactly which key
/// is missing or belongs to the other form; ResolveDispersion consumes the
/// complete request.
/// \ingroup qcx-dispersion
struct D3ParameterRequest {
    std::optional<double> s6; ///< Global C6 scale.
    std::optional<double> s8; ///< C8 scale.
    std::optional<double> rs6; ///< The a1 damping scale of the R^6 term.
    std::optional<double> alpha6; ///< The a2 exponent of the R^6 term.
    std::optional<double> rs8; ///< The a1 damping scale of the R^8 term.
    std::optional<double> alpha8; ///< The a2 exponent of the R^8 term.

    /// Whether the input wrote any D3 numeric key.
    /// \returns True when at least one member is engaged.
    bool Any() const noexcept {
        return s6.has_value() || s8.has_value() || rs6.has_value() || alpha6.has_value() ||
               rs8.has_value() || alpha8.has_value();
    }

    /// Whether the input wrote the complete D3 numeric set.
    /// \returns True when every member is engaged.
    bool Complete() const noexcept {
        return s6.has_value() && s8.has_value() && rs6.has_value() && alpha6.has_value() &&
               rs8.has_value() && alpha8.has_value();
    }
};

/// The D4 numeric parameters as an input wrote them.
///
/// One optional per published key; `s9` and `alp` are deliberately absent:
/// the shipped D4 energy is the two-body term and consumes neither, so the
/// request type cannot represent them.
/// \ingroup qcx-dispersion
struct D4ParameterRequest {
    std::optional<double> s6; ///< The dipole-dipole scale.
    std::optional<double> s8; ///< The dipole-quadrupole scale.
    std::optional<double> a1; ///< The Becke-Johnson damping scale.
    std::optional<double> a2; ///< The Becke-Johnson damping offset, Bohr.

    /// Whether the input wrote any D4 numeric key.
    /// \returns True when at least one member is engaged.
    bool Any() const noexcept {
        return s6.has_value() || s8.has_value() || a1.has_value() || a2.has_value();
    }

    /// Whether the input wrote the complete D4 numeric set.
    /// \returns True when every member is engaged.
    bool Complete() const noexcept {
        return s6.has_value() && s8.has_value() && a1.has_value() && a2.has_value();
    }
};

/// The `[dispersion]` block as an input parser read it.
///
/// Presence is preserved per key rather than resolved here: which keys were
/// written is what the cross-key rules act on, and the form's table is what
/// turns a name or the method family into numbers. A default-constructed
/// value describes a block that wrote nothing.
/// \ingroup qcx-dispersion
struct DispersionOptions {
    std::optional<DispersionForm> form; ///< The selected form; absent when the block wrote none.
    std::optional<std::string> preset; ///< A preset name, left unresolved.
    D3ParameterRequest d3; ///< The D3 numeric keys written.
    D4ParameterRequest d4; ///< The D4 numeric keys written.
};

/// \}
} // namespace qcx::dispersion
