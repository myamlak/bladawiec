// The human-readable run report (see FormatRunReport): the companion a
// `qcx run` writes beside every input file, next to the JSON document that
// result_json.hpp serializes.
//
// The report states what the run OBSERVED and nothing else. A section whose
// data the result does not carry is omitted entirely - no zero, no dash, no
// placeholder stands in for a number the run never produced - and an unset
// optional member prints no row at all. The result type carries those
// members as std::optional for exactly that reason, so absence is stated by
// absence here too.
//
// Two facts the report's lines need are in neither the input file nor the
// result record, because they are facts of the RUN rather than of the file
// or of the record: the molecule-scoped basis-function count, and the
// permutation between the input's atom rows and the order the result's
// per-atom vectors use (the molecule renumbers its atoms canonically on
// construction). The caller hands both in - RunReportFacts - rather than
// having the report resolve them a second time.
//
// Layout: fixed-width fields, never tabs, one blank line between sections.
// Energies in Hartree and times in milliseconds, with the unit written out.

#pragma once

#include "qcx/io/result_json.hpp"
#include "qcx/io/run_input.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace qcx::io {

/// The facts the report needs that neither the parsed input nor the run
/// result carries.
/// \ingroup qcx-io
struct RunReportFacts {
    /// The molecule-scoped basis-function count the run resolved. Unset when
    /// no count was resolved: the basis line then states no count rather than
    /// a number nobody measured.
    std::optional<std::size_t> basisFunctionCount;

    /// The permutation between the input's atom rows and the order the
    /// result's per-atom vectors use: element c names the row of
    /// `RunInput::molecule.atoms` that atom c of those vectors (the
    /// population block, the charge fits) belongs to.
    ///
    /// It is NOT the identity for every input. The molecule renumbers its
    /// atoms canonically on construction (by element, then by position), so a
    /// file that writes its rows in another order indexes its own atoms and
    /// the result's vectors differently - and a report that paired the rows
    /// directly would print one atom's charge beside another atom's symbol.
    ///
    /// Empty when no permutation was resolved. The charges section then
    /// labels each atom by its 1-based index in the result's own order rather
    /// than borrowing a symbol that may belong to another atom.
    std::vector<std::size_t> canonicalAtomOrder;
};

/// Formats the report of one run: the molecule, the basis and the method the
/// file asked for, what the SCF reached, the Mulliken charges of the density
/// it produced, and the wall times.
///
/// The header block reads the input's own resolved values - the formula is
/// built from the atom rows, and the basis name, the method word and the
/// accuracy preset are the file's request as the parser resolved it - while
/// the builder line reads the result's resolved selection, with the
/// vocabulary that chose it spelled out beside the axes.
/// \param input The parsed run description.
/// \param result The structured run record.
/// \param facts The resolved facts that neither of the two carries.
/// \returns The report text, ending in a newline. A section whose data is
/// absent from the result is absent from the report.
/// \ingroup qcx-io
std::string FormatRunReport(const RunInput& input,
                            const RunResult& result,
                            const RunReportFacts& facts);

} // namespace qcx::io
