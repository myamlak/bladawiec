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
#include "qcx/io/run_input.hpp"

#include <string>

namespace qcx::driver {

/// \defgroup qcx-driver Driver module
/// The run driver: turns one parsed RunInput into the JSON run
/// result end to end (molecule, basis, integrals, Fock builder,
/// SCF seam, properties, timings).
/// \{

/// Runs one input end to end and serializes the JSON result.
/// \param input The parsed run description.
/// \returns The result JSON document, or an Error (kInvalidArgument for an
/// unknown element symbol, a missing bundled basis or aux basis, and a
/// method the driver does not classify as a run path;
/// kUnimplemented for the not-yet-wired combinations - UHF with any
/// non-direct builder, ri_jk, the Kohn-Sham methods (no run path folds the
/// exchange-correlation potential into the Fock build), the guess/method
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
