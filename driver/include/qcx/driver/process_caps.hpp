// The process-level resource caps of the qcx run driver: the hard memory
// cap, applied to the driver's own process at run start. The implementation
// reuses the benchmark harness gate's job-object pattern in-process: a
// Windows job object with JOB_OBJECT_LIMIT_PROCESS_MEMORY at the cap kills
// the process at the cap instead of the machine dying. The refusal policy
// has two cases: when the cap is requested but cannot be created, the run
// refuses (RunDriver returns an error) — an uncapped run must never start
// by its own doing; when the process is already inside an external job
// (the harness gate or a CI runner), the driver proceeds uncapped and
// records the fact — the external job's limits are the caller's authority.
//
// The job handles are process-lifetime (never released): a released handle
// on a KILL_ON_JOB_CLOSE job would kill the process, and nested jobs'
// limits combine, so keeping every applied cap alive preserves the most
// restrictive one.

#pragma once

#include "qcx/error.hpp"
#include "qcx/io/run_input.hpp"

#include <cstdint>
#include <optional>
#include <string>

namespace qcx::driver {

/// The enforcement record of one ApplyProcessCaps call.
struct AppliedProcessCaps {
    /// True when the driver's own job-object process-memory cap is in
    /// force in-process (Windows) — applied by this call or by an earlier
    /// one whose cap was at least as tight (a reuse). False when the cap
    /// could not be applied — see note.
    bool inProcessCapApplied = false;
    /// Why the in-process cap was not applied; nullopt when it was.
    std::optional<std::string> note;
};

/// Applies the run's hard memory cap to the calling process.
/// memoryCapGiB > 0 (the default 16.0 included) puts the process under a
/// job object with JOB_OBJECT_LIMIT_PROCESS_MEMORY at the cap and assigns
/// the current process; the OS terminates the process the moment it would
/// exceed.
/// The cap is process-wide and its job handles are process-lifetime, so a
/// call whose cap is not tighter than the tightest cap already applied
/// REUSES that job — a nested job at an equal-or-looser cap would enforce
/// exactly the same limit — and only a strictly tighter cap nests one
/// more. A process can hold about 100 job objects; reaching that ceiling
/// is a hard error that names it. memoryCapGiB == 0 (the documented escape
/// hatch) applies nothing.
///
/// Fail-closed semantics: a CreateJobObjectW or SetInformationJobObject
/// failure is a hard error (the run must refuse — an uncapped run must
/// never start), and so is an AssignProcessToJobObject failure that is not
/// the external-job case below — the job-object ceiling included.
/// AssignProcessToJobObject failing with ERROR_ACCESS_DENIED means the
/// process is already in an external job (the benchmark harness gate or a
/// CI runner); the driver cannot and must not fight the caller's job, so it
/// proceeds and records the fact — the external job's limits are the
/// caller's authority. On platforms without job objects (Linux CI) the cap
/// cannot be applied in-process: the record says so, and NOTHING enforces
/// it there. No predictive model stands in for the missing
/// limit, so a Linux run is bounded by the
/// machine alone.
///
/// \param resources The parsed [resources] block.
/// \returns The enforcement record, or an Error when the cap was requested
/// and could not be applied.
/// \ingroup qcx-driver
qcx::Result<AppliedProcessCaps> ApplyProcessCaps(const qcx::io::RunResourcesInput& resources);

/// Is the run's memory cap a REAL limit right now? True when an OS-enforced process-memory limit at or below
/// memoryCapGiB is in force for THIS process, so a run that needs more
/// than the cap is stopped by the OS instead of by a prediction. Two
/// sources count, and both are the same metric the cap itself uses
/// (JOB_OBJECT_LIMIT_PROCESS_MEMORY):
///
/// 1. The driver's own applied cap (ApplyProcessCaps) when the tightest
///    cap it has applied is at or below memoryCapGiB - nested job limits
///    combine and the tightest binds.
/// 2. The enclosing Windows job object's own limit, read live with
///    QueryInformationJobObject on the calling process: the benchmark
///    harness gate (tools/bench/memory_gate.py ApplyToChild) and a CI
///    runner both cap the child this way, and ApplyProcessCaps reports
///    that case as its external-job no-op - the driver can see the limit
///    even though it did not apply it.
///
/// Deliberately conservative: a job whose limit is LOOSER than
/// memoryCapGiB, a job with no process-memory limit flag, a zero or
/// negative cap, and every non-Windows platform all read false. A nested
/// chain whose outer job is tighter is not walked: the immediate job's
/// limit is the verdict.
/// \param memoryCapGiB The cap in effect for the run (0 = no input cap).
/// \returns true when a real process-memory limit at or below the cap is
/// in force for this process.
/// \ingroup qcx-driver
bool ProcessMemoryCapEnforced(double memoryCapGiB) noexcept;

/// The process's own peak committed bytes, read from the same job-object
/// metric the cap and the allocation instrument use (QueryInformationJobObject,
/// JobObjectExtendedLimitInformation.PeakProcessMemoryUsed) - the ACTUAL
/// number the run is checked against at
/// run end. Fail-open: nullopt when the read
/// fails, on a platform without job objects, or for a process in no job at
/// all, and a nullopt logs nothing rather than a zero that would read as a
/// measurement.
/// \returns The peak committed bytes, or nullopt when it cannot be read.
/// \ingroup qcx-driver
std::optional<std::uint64_t> PeakProcessMemoryBytes() noexcept;

} // namespace qcx::driver
