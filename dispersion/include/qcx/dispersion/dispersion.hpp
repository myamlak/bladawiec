#pragma once

#include "qcx/dispersion/options.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace qcx::dispersion {

/// The resolved D3 parameter values, one member per published key.
/// \ingroup qcx-dispersion
struct D3ParameterValues {
    double s6 = 1.0; ///< Global C6 scale.
    double s8 = 1.0; ///< C8 scale.
    double rs6 = 1.0; ///< The a1 damping scale of the R^6 term.
    double alpha6 = 14.0; ///< The a2 exponent of the R^6 term.
    double rs8 = 1.0; ///< The a1 damping scale of the R^8 term.
    double alpha8 = 16.0; ///< The a2 exponent of the R^8 term.
};

/// The resolved D4 parameter values, one member per published key.
/// \ingroup qcx-dispersion
struct D4ParameterValues {
    double s6 = 1.0; ///< The dipole-dipole scale.
    double s8 = 0.0; ///< The dipole-quadrupole scale.
    double a1 = 0.0; ///< The Becke-Johnson damping scale.
    double a2 = 0.0; ///< The Becke-Johnson damping offset, Bohr.
};

/// A dispersion request resolved against its form's parameter table.
/// \ingroup qcx-dispersion
struct ResolvedDispersion {
    DispersionForm form = DispersionForm::kD3; ///< The selected form.
    DispersionParameterSource source = DispersionParameterSource::kMethod; ///< The value source.
    /// The preset name the values came from; absent for kExplicit, where the
    /// input stated every value itself.
    std::optional<std::string> parameterSet;
    /// The values the correction consumes, under the form they were resolved
    /// for.
    std::variant<D3ParameterValues, D4ParameterValues> parameters;
};

/// The input word of a dispersion form.
/// \param form The form.
/// \returns "d3" or "d4".
/// \ingroup qcx-dispersion
std::string_view ToString(DispersionForm form) noexcept;

/// The record word of a parameter source.
/// \param source The source.
/// \returns "method", "preset" or "explicit".
/// \ingroup qcx-dispersion
std::string_view ToString(DispersionParameterSource source) noexcept;

/// Resolves a run's dispersion request into the values the correction will
/// consume.
///
/// Exactly one source is honoured: the complete numeric set when the input
/// wrote one, otherwise the named preset, otherwise the run method's own
/// family looked up in the selected form's table. A name the table does not
/// carry is refused naming the name, the form and the `preset` remedy; a
/// partial numeric set is refused naming every missing key.
/// \param options The request as the parser read it.
/// \param methodFamily The family the no-parameter default reads: the
/// `[method] functional` on the Kohn-Sham lanes, "hf" on rhf/uhf. Empty when
/// the run names none, which makes the default unresolvable.
/// \returns The resolved values, or kInvalidArgument for a request the
/// form's contract does not admit.
/// \ingroup qcx-dispersion
qcx::Result<ResolvedDispersion> ResolveDispersion(const DispersionOptions& options,
                                                  std::string_view methodFamily);

/// Evaluates the resolved correction on a molecule.
///
/// The D3 arm is the zero-damping energy. The D4 arm is the shipped
/// two-body energy; because that energy evaluates its
/// electronegativity-equilibration charges at total charge 0, a charged
/// molecule is refused rather than given a number computed at the wrong
/// charge.
/// \param molecule The molecule, its coordinates in Bohr.
/// \param resolved The resolved form and values.
/// \returns The energy in Hartree (negative for attraction), or an Error: a
/// charged molecule on the D4 arm and an element the form does not cover are
/// kUnimplemented, and a kernel-level refusal is translated from excgrid.
/// \ingroup qcx-dispersion
qcx::Result<double> EvaluateDispersion(const molecule::Molecule& molecule,
                                       const ResolvedDispersion& resolved);

/// The resolved values under the form's own key names.
///
/// The map is what a run record carries, so a reader sees the numbers the
/// correction consumed under the vocabulary the input would use to state
/// them: d3 gives s6, s8, rs6, alpha6, rs8 and alpha8; d4 gives s6, s8, a1
/// and a2.
/// \param resolved The resolved request.
/// \returns The values, keyed by the form's parameter names.
/// \ingroup qcx-dispersion
std::map<std::string, double> ParameterMap(const ResolvedDispersion& resolved);

} // namespace qcx::dispersion
