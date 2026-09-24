// The qcx run driver: turns a parsed RunInput into the
// JSON run result end to end - molecule (Angstrom -> Bohr), basis, the
// one-electron integrals, the Fock builder wired per the method/accuracy
// selection, the SCF seam call (rhf.hpp/uhf.hpp), and the timings.
//
// Module-DAG note: the architecture's final placement of io/driver sits
// after gradients/hessian/properties/optimizer; v1 registers right after
// scf because that is where the dependency actually is today - the driver
// only needs RHF/UHF Fock builds. The final position
// is TBD once the later modules exist.

#pragma once

#include "qcx/error.hpp"
#include "qcx/io/result_json.hpp"
#include "qcx/io/result_report.hpp"
#include "qcx/io/run_input.hpp"

#include <string>

namespace qcx::driver {

/// \defgroup qcx-driver module
/// The run driver: turns one parsed RunInput into the JSON run
/// result end to end (molecule, basis, integrals, Fock builder,
/// SCF seam, properties, timings).
/// \{

/// The full outcome of one run: the structured record, its JSON document,
/// and the facts the human-readable report needs that neither the input nor
/// the record carries.
/// \ingroup qcx-driver
struct RunOutcome {
    qcx::io::RunResult result; ///< The structured run record.
    std::string json; ///< The JSON document of `result`.
    /// The report's own facts: the molecule-scoped basis-function count and
    /// the atom canonicalization permutation the run resolved, so a caller
    /// that writes the report never re-derives either one.
    qcx::io::RunReportFacts reportFacts;
};

/// Runs one input end to end and returns the full outcome.
///
/// This is the form for a caller that needs more than the JSON text - the
/// command line writes a report beside the input file, and the report's
/// header and atom labels read facts that live outside the record. `RunDriver`
/// below is the JSON-only form of the same run.
/// \param input The parsed run description.
/// \returns The outcome, or the same Errors `RunDriver` documents.
/// \ingroup qcx-driver
qcx::Result<RunOutcome> RunDriverOutcome(const qcx::io::RunInput& input);

/// Runs one input end to end and serializes the JSON result.
/// \param input The parsed run description.
/// \returns The result JSON document, or an Error (kInvalidArgument for an
/// unknown element symbol, a missing bundled basis or aux basis, and a
/// method the driver does not classify as a run path;
/// kUnimplemented for the not-yet-wired combinations - UHF with any
/// non-direct builder, ri_jk, the guess/method
/// combinations the schema documents, SAD beyond Z <= 10, and UHF with
/// nocv_fragments; kDeviceError from the GPU builder without a device;
/// kInvalidArgument for an out-of-range nocv_fragments atom index). The
/// parsed input is semantically validated first (qcx::io::ValidateInput);
/// invalid input fails fast with the first issue (kInvalidArgument) before
/// any construction. Which method words run is decided by one exhaustive
/// classification (run_driver.cpp ResolveScfPath), never by a per-site
/// equality test, so a word with no run path cannot fall through to another
/// method's physics.
/// \ingroup qcx-driver
qcx::Result<std::string> RunDriver(const qcx::io::RunInput& input);

/// \}

} // namespace qcx::driver
