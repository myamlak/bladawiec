// Dispersion request resolution and the excgrid evaluation seam.

#include "qcx/dispersion/dispersion.hpp"

#include "qcx/grid/geometry_translation.hpp"

#include <excgrid/d4.hpp>
#include <excgrid/dispersion.hpp>
#include <string>
#include <utility>

namespace qcx::dispersion {
namespace {

qcx::Error MakeError(std::string message) {
    return qcx::Error{qcx::ErrorCode::kInvalidArgument, std::move(message)};
}

// The keys of a request, each with the value the input wrote, as
// "dispersion.<key> = <value>" and joined by ", ". The resolver's refusals
// name the offending key AND its value, so the two cannot drift apart.
void KeyValues(std::string& output, const char* key, const std::optional<double>& value) {
    if (!value.has_value())
    {
        return;
    }

    if (!output.empty())
    {
        output += ", ";
    }

    output += std::string("dispersion.") + key + " = " + std::to_string(*value);
}

std::string D3KeyValues(const D3ParameterRequest& request) {
    std::string output;
    KeyValues(output, "s6", request.s6);
    KeyValues(output, "s8", request.s8);
    KeyValues(output, "rs6", request.rs6);
    KeyValues(output, "alpha6", request.alpha6);
    KeyValues(output, "rs8", request.rs8);
    KeyValues(output, "alpha8", request.alpha8);
    return output;
}

std::string D4KeyValues(const D4ParameterRequest& request) {
    std::string output;
    KeyValues(output, "s6", request.s6);
    KeyValues(output, "s8", request.s8);
    KeyValues(output, "a1", request.a1);
    KeyValues(output, "a2", request.a2);
    return output;
}

// The D3 keys the request did not write, as "dispersion.<key>" and joined by
// ", "; empty when the request is complete.
std::string MissingD3Keys(const D3ParameterRequest& request) {
    std::string output;

    const auto add = [&output](const char* key, const std::optional<double>& value) {
        if (value.has_value())
        {
            return;
        }

        if (!output.empty())
        {
            output += ", ";
        }

        output += std::string("dispersion.") + key;
    };

    add("s6", request.s6);
    add("s8", request.s8);
    add("rs6", request.rs6);
    add("alpha6", request.alpha6);
    add("rs8", request.rs8);
    add("alpha8", request.alpha8);
    return output;
}

std::string MissingD4Keys(const D4ParameterRequest& request) {
    std::string output;

    const auto add = [&output](const char* key, const std::optional<double>& value) {
        if (value.has_value())
        {
            return;
        }

        if (!output.empty())
        {
            output += ", ";
        }

        output += std::string("dispersion.") + key;
    };

    add("s6", request.s6);
    add("s8", request.s8);
    add("a1", request.a1);
    add("a2", request.a2);
    return output;
}

D3ParameterValues ToValues(const D3ParameterRequest& request) {
    return D3ParameterValues{
        *request.s6, *request.s8, *request.rs6, *request.alpha6, *request.rs8, *request.alpha8};
}

D4ParameterValues ToValues(const D4ParameterRequest& request) {
    return D4ParameterValues{*request.s6, *request.s8, *request.a1, *request.a2};
}

} // namespace

std::string_view ToString(DispersionForm form) noexcept {
    switch (form)
    {
    case DispersionForm::kD3:
        return "d3";
    case DispersionForm::kD4:
        return "d4";
    }

    return "unknown";
}

std::string_view ToString(DispersionParameterSource source) noexcept {
    switch (source)
    {
    case DispersionParameterSource::kMethod:
        return "method";
    case DispersionParameterSource::kPreset:
        return "preset";
    case DispersionParameterSource::kExplicit:
        return "explicit";
    }

    return "unknown";
}

qcx::Result<ResolvedDispersion> ResolveDispersion(const DispersionOptions& options,
                                                  std::string_view methodFamily) {
    if (!options.form.has_value())
    {
        return std::unexpected(
            MakeError("dispersion: a parameter was written without a form (set dispersion.form to "
                      "\"d3\" or \"d4\")"));
    }

    const DispersionForm form = *options.form;
    const std::string formWord(ToString(form));

    // A key of the other form cannot be silently dropped: it names a
    // parameter the selected form does not read, so the run would consume a
    // set the file did not write.
    if (form == DispersionForm::kD3 && options.d4.Any())
    {
        return std::unexpected(MakeError(
            "dispersion: " + D4KeyValues(options.d4) +
            " cannot accompany form = \"d3\" (d3 takes s6, s8, rs6, alpha6, rs8 and alpha8)"));
    }

    if (form == DispersionForm::kD4 && options.d3.Any())
    {
        return std::unexpected(
            MakeError("dispersion: " + D3KeyValues(options.d3) +
                      " cannot accompany form = \"d4\" (d4 takes s6, s8, a1 and a2)"));
    }

    const bool explicitValues = form == DispersionForm::kD3 ? options.d3.Any() : options.d4.Any();

    // One parameter source: a preset names a published set and numeric keys
    // state the set, so writing both would leave the run choosing between
    // them.
    if (options.preset.has_value() && explicitValues)
    {
        const std::string numeric =
            form == DispersionForm::kD3 ? D3KeyValues(options.d3) : D4KeyValues(options.d4);
        return std::unexpected(
            MakeError("dispersion.preset = \"" + *options.preset + "\" and " + numeric +
                      " are two parameter sources for form \"" + formWord +
                      "\"; write either the preset or the complete numeric set"));
    }

    ResolvedDispersion resolved;
    resolved.form = form;

    if (explicitValues)
    {
        if (form == DispersionForm::kD3)
        {
            if (!options.d3.Complete())
            {
                return std::unexpected(
                    MakeError("dispersion: form \"d3\" takes the complete numeric set; missing " +
                              MissingD3Keys(options.d3)));
            }

            resolved.source = DispersionParameterSource::kExplicit;
            resolved.parameters = ToValues(options.d3);
            return resolved;
        }

        if (!options.d4.Complete())
        {
            return std::unexpected(
                MakeError("dispersion: form \"d4\" takes the complete numeric set; missing " +
                          MissingD4Keys(options.d4)));
        }

        resolved.source = DispersionParameterSource::kExplicit;
        resolved.parameters = ToValues(options.d4);
        return resolved;
    }

    // The named preset, resolved in the selected form's own table. The D3 and
    // D4 name spaces deliberately differ, so a name from the other one is not
    // found here.
    if (options.preset.has_value())
    {
        if (form == DispersionForm::kD3)
        {
            auto parameters = excgrid::D3Preset(*options.preset);

            if (!parameters.has_value())
            {
                return std::unexpected(MakeError(
                    "dispersion.preset = \"" + *options.preset +
                    "\" is not a d3 parameter set (the form's table carries its own published "
                    "names); set dispersion.preset to one of them, or write the complete numeric "
                    "set"));
            }

            resolved.source = DispersionParameterSource::kPreset;
            resolved.parameterSet = *options.preset;
            resolved.parameters = D3ParameterValues{parameters->s6,
                                                    parameters->s8,
                                                    parameters->rs6,
                                                    parameters->alpha6,
                                                    parameters->rs8,
                                                    parameters->alpha8};
            return resolved;
        }

        auto parameters = excgrid::D4Preset(*options.preset);

        if (!parameters.has_value())
        {
            return std::unexpected(MakeError(
                "dispersion.preset = \"" + *options.preset +
                "\" is not a d4 parameter set (the form's table carries its own published "
                "names); set dispersion.preset to one of them, or write the complete numeric "
                "set"));
        }

        resolved.source = DispersionParameterSource::kPreset;
        resolved.parameterSet = *options.preset;
        resolved.parameters =
            D4ParameterValues{parameters->s6, parameters->s8, parameters->a1, parameters->a2};
        return resolved;
    }

    // No parameters were written: the default is the run method's own family,
    // which is the only default that cannot silently pair a functional with
    // another method's dispersion parameters.
    if (methodFamily.empty())
    {
        return std::unexpected(MakeError(
            "dispersion: form \"" + formWord +
            "\" was selected with no parameters and the run names no method family to default "
            "to; set dispersion.preset"));
    }

    if (form == DispersionForm::kD3)
    {
        auto parameters = excgrid::D3Preset(methodFamily);

        if (!parameters.has_value())
        {
            return std::unexpected(MakeError(
                "dispersion: the run method's family \"" + std::string(methodFamily) +
                "\" has no d3 parameter set, so form \"d3\" has no default parameters; set "
                "dispersion.preset to name the d3 parameters"));
        }

        resolved.source = DispersionParameterSource::kMethod;
        resolved.parameterSet = std::string(methodFamily);
        resolved.parameters = D3ParameterValues{parameters->s6,
                                                parameters->s8,
                                                parameters->rs6,
                                                parameters->alpha6,
                                                parameters->rs8,
                                                parameters->alpha8};
        return resolved;
    }

    auto parameters = excgrid::D4Preset(methodFamily);

    if (!parameters.has_value())
    {
        return std::unexpected(
            MakeError("dispersion: the run method's family \"" + std::string(methodFamily) +
                      "\" has no d4 parameter set, so form \"d4\" has no default parameters; set "
                      "dispersion.preset to name the d4 parameters"));
    }

    resolved.source = DispersionParameterSource::kMethod;
    resolved.parameterSet = std::string(methodFamily);
    resolved.parameters =
        D4ParameterValues{parameters->s6, parameters->s8, parameters->a1, parameters->a2};
    return resolved;
}

qcx::Result<double> EvaluateDispersion(const molecule::Molecule& molecule,
                                       const ResolvedDispersion& resolved) {
    auto geometry = qcx::grid::ToExcgridGeometry(molecule);

    if (!geometry.has_value())
    {
        return std::unexpected(geometry.error());
    }

    if (resolved.form == DispersionForm::kD3)
    {
        const auto* values = std::get_if<D3ParameterValues>(&resolved.parameters);

        if (values == nullptr)
        {
            return std::unexpected(
                MakeError("dispersion: the resolved values do not match form \"d3\""));
        }

        const excgrid::D3Parameters parameters{
            values->s6, values->s8, values->rs6, values->alpha6, values->rs8, values->alpha8};
        auto energy = excgrid::GrimmeD3Energy(*geometry, parameters);

        if (!energy.has_value())
        {
            return std::unexpected(
                qcx::grid::TranslateExcgridError(energy.error(), "dispersion: the d3 kernel"));
        }

        return *energy;
    }

    const auto* values = std::get_if<D4ParameterValues>(&resolved.parameters);

    if (values == nullptr)
    {
        return std::unexpected(
            MakeError("dispersion: the resolved values do not match form \"d4\""));
    }

    // The shipped D4 energy evaluates its EEQ charges at total charge 0, so a
    // charged molecule's number would be computed at the wrong charge. The
    // charge-aware energy is the debt, not a boundary: until it exists the
    // request is refused by name rather than answered with the wrong number.
    if (molecule.Charge() != 0)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kUnimplemented,
            "dispersion: form \"d4\" evaluates its electronegativity-equilibration charges at "
            "total charge 0, so a charged molecule cannot use it (use form \"d3\", which is "
            "charge-independent; a charge-aware D4 energy is not implemented)"});
    }

    for (const auto& atom : molecule.Atoms())
    {
        if (!excgrid::D4CoversElement(atom.atomicNumber))
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kUnimplemented,
                           "dispersion: form \"d4\" does not cover element Z = " +
                               std::to_string(atom.atomicNumber) +
                               " (the published parameterization ends at radon, Z = 86)"});
        }
    }

    const excgrid::D4Parameters parameters{values->s6, values->s8, values->a1, values->a2};
    auto energy = excgrid::D4TwoBodyEnergy(*geometry, parameters);

    if (!energy.has_value())
    {
        return std::unexpected(
            qcx::grid::TranslateExcgridError(energy.error(), "dispersion: the d4 kernel"));
    }

    return *energy;
}

std::map<std::string, double> ParameterMap(const ResolvedDispersion& resolved) {
    std::map<std::string, double> values;

    if (const auto* d3 = std::get_if<D3ParameterValues>(&resolved.parameters))
    {
        values = {{"s6", d3->s6},
                  {"s8", d3->s8},
                  {"rs6", d3->rs6},
                  {"alpha6", d3->alpha6},
                  {"rs8", d3->rs8},
                  {"alpha8", d3->alpha8}};
        return values;
    }

    const auto& d4 = std::get<D4ParameterValues>(resolved.parameters);
    values = {{"s6", d4.s6}, {"s8", d4.s8}, {"a1", d4.a1}, {"a2", d4.a2}};
    return values;
}

} // namespace qcx::dispersion
