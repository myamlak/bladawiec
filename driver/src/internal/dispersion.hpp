#pragma once

// The driver-side dispersion seam: the run's [dispersion] request resolved to
// numbers and evaluated once, on the molecule, before any SCF work.
//
// The module (qcx::dispersion) owns the request vocabulary, the parameter
// tables and the excgrid kernel. The two facts only the driver can supply are
// here: the method family the no-parameter default reads ([method]
// functional on the Kohn-Sham lanes, "hf" on the Hartree-Fock lanes), and the
// packaging of the module's resolved answer into the io record's shape.
//
// Driver-internal (src/internal/, the ks_grid.hpp footprint): the io layer
// parses and validates into qcx::dispersion::DispersionOptions and never sees
// a molecule; the driver resolves and evaluates.

#include "qcx/dispersion/dispersion.hpp"
#include "qcx/error.hpp"
#include "qcx/io/result_json.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/molecule/molecule.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace qcx::driver::internal {

/// The method family a run with no dispersion parameters defaults to.
///
/// The Kohn-Sham lanes read the input's own `[method] functional` name; the
/// Hartree-Fock lanes name no functional and use "hf", which both forms'
/// tables carry. An rks/uks run with no functional yields an empty family:
/// that run is refused by the Kohn-Sham composition for the same missing key,
/// and a dispersion default cannot be invented ahead of it.
/// \param input The run input.
/// \returns The family name, or an empty view when the run names none.
/// \ingroup qcx-driver
inline std::string_view DispersionMethodFamily(const qcx::io::RunInput& input) {
    switch (input.method.method)
    {
    case qcx::io::MethodType::kRhf:
    case qcx::io::MethodType::kUhf:
        return "hf";
    case qcx::io::MethodType::kRks:
    case qcx::io::MethodType::kUks:
        return input.method.functional.has_value() ? std::string_view(*input.method.functional)
                                                   : std::string_view{};
    }

    return std::string_view{};
}

/// Resolves and evaluates a run's `[dispersion]` block.
///
/// An absent request yields an unset optional and no correction is added. A
/// present request is resolved to exactly one parameter set, evaluated on the
/// molecule, and packaged as the record block.
/// \param input The run input: its `[dispersion]` block and its method.
/// \param molecule The molecule whose geometry the correction is evaluated on.
/// \returns The record block, or an Error from the resolution or the kernel.
/// \ingroup qcx-driver
inline qcx::Result<std::optional<qcx::io::RunDispersion>> EvaluateRunDispersion(
    const qcx::io::RunInput& input, const qcx::molecule::Molecule& molecule) {
    if (!input.dispersion.has_value())
    {
        return std::optional<qcx::io::RunDispersion>{};
    }

    auto resolved =
        qcx::dispersion::ResolveDispersion(*input.dispersion, DispersionMethodFamily(input));

    if (!resolved.has_value())
    {
        return std::unexpected(resolved.error());
    }

    auto energy = qcx::dispersion::EvaluateDispersion(molecule, *resolved);

    if (!energy.has_value())
    {
        return std::unexpected(energy.error());
    }

    qcx::io::RunDispersion record;
    record.form = std::string(qcx::dispersion::ToString(resolved->form));
    record.parameterSource = std::string(qcx::dispersion::ToString(resolved->source));
    record.parameterSet = resolved->parameterSet;
    record.parameters = qcx::dispersion::ParameterMap(*resolved);
    record.energyHartree = *energy;

    return std::optional<qcx::io::RunDispersion>(std::move(record));
}

} // namespace qcx::driver::internal
