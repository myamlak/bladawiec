// The hard memory cap implementation: the benchmark harness gate's
// job-object pattern applied to the driver's own process. The Windows API
// constants mirror the Python struct layout byte for byte
// (JOB_OBJECT_EXTENDED_LIMIT_INFORMATION class 9 with ProcessMemoryLimit
// and the two limit flags). The applied cap's job handle is also bound to
// the allocation-attribution instrument: its watchdog samples
// PeakProcessMemoryUsed from exactly this job (class 9 again — the
// cap metric), so an instrumented run's trace carries the same peak the
// harness gate would have killed at.
//
// One cap job in force, not one per call: a process may carry only about
// 100 job objects, the applied handles are process-lifetime (never
// released), and nested jobs' limits combine - so a call whose cap is not
// tighter than the one in force REUSES it, and only a strictly tighter cap
// nests one more. The ceiling itself is a refusal that names it, never a
// silent uncapped run or a bare error 50 that reads as an environment
// problem (the defect: one job per call, ~100 calls per
// process).

#include "qcx/driver/process_caps.hpp"

#include "qcx/memory/allocation_instrument.hpp"

#include <cstdio>
#include <vector>

#ifdef _WIN32
#include <windows.h>
// The committed-bytes readout of the one-shot apply log below: psapi's
// GetProcessMemoryInfo maps to K32GetProcessMemoryInfo on this SDK
// (PSAPI_VERSION 2), which kernel32.lib imports - no extra link needed.
#include <psapi.h>
#endif

namespace qcx::driver {
namespace {

qcx::Error Err(qcx::ErrorCode code, std::string message) {
    return qcx::Error{code, std::move(message)};
}

// The job handles of every applied cap, process-lifetime: never released.
// KILL_ON_JOB_CLOSE would kill the process if the last handle closed, and
// nested jobs' limits combine, so each applied cap's handle must outlive
// the process. A function-local static keeps the accessor tidy-clean.
#ifdef _WIN32
std::vector<HANDLE>& AppliedJobHandles() {
    static std::vector<HANDLE> handles;

    return handles;
}
#endif

#ifdef _WIN32
// The cap state of this process, process-lifetime like the handles above.
// Only the tightest cap is remembered, and it is the minimum of every cap
// this path has applied: a job is created only for a cap strictly tighter
// than that minimum, so the nesting sequence descends and the LAST handle
// in AppliedJobHandles() is always the binding job - the one enforcing the
// tightest limit and the one the allocation instrument samples.
struct ProcessCapState {
    bool applied = false;
    double tightestCapGiB = 0.0;
};

ProcessCapState& CapState() {
    static ProcessCapState state;

    return state;
}

// The Windows job-object ceiling of one process, measured 2026-09-12:
// 100 applied cap jobs bind inside the benchmark harness job, 101 in a
// job-free process, and the assignment past either number fails with
// ERROR_NOT_SUPPORTED (50). The floor below that measured pair is what
// makes error 50 legible AS the ceiling - under it the refusal could be
// something else and is reported as the plain API failure.
constexpr std::size_t kJobCeilingFloor = 90;
#endif

#ifdef _WIN32
// The one-shot stderr instrument of the spawn-site hunt: the FIRST
// cap application of each process reports the requested cap, the outcome
// and its GetLastError, and the process's committed bytes at apply time -
// a handful of lines per suite (the host's first apply: bound, or the
// external-job no-op at whatever commit it sat; plus one line per cap-
// child), never a line per run. A function-local static gates it.
void LogFirstCapApply(double capGiB, const char* outcome, DWORD lastError) {
    static bool logged = false;

    if (logged)
    {
        return;
    }

    logged = true;

    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    GetProcessMemoryInfo(GetCurrentProcess(),
                         reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                         sizeof(counters));
    std::fprintf(stderr,
                 "[qcx-cap] apply %.4g GiB: %s (GetLastError %lu), process committed %.1f MiB\n",
                 capGiB,
                 outcome,
                 lastError,
                 static_cast<double>(counters.PrivateUsage) / (1ULL << 20));
}
#endif

} // namespace

qcx::Result<AppliedProcessCaps> ApplyProcessCaps(const qcx::io::RunResourcesInput& resources) {
#ifdef _WIN32
    if (resources.memoryCapGiB <= 0.0)
    {
        // The documented escape hatch: 0 = no input cap. Nothing to apply.
        return AppliedProcessCaps{};
    }

    // Reuse, do not nest, when the request is not tighter than the cap
    // already in force. Nested job limits combine and the tightest one
    // binds, so a second job at an equal-or-looser cap would enforce
    // exactly what the job in force already enforces while spending one of
    // the ~100 job-object slots this process has - and the handles are
    // process-lifetime, so a host that runs many capped runs in one process
    // (the driver suite: 117 in-process RunDriver call sites against that
    // ~100-slot budget) accumulates one job per run until the assignment
    // fails with error 50. Reusing the job in force also keeps the allocation
    // instrument bound to the binding job, since the last handle is the
    // tightest by construction.
    if (CapState().applied && resources.memoryCapGiB >= CapState().tightestCapGiB)
    {
        const HANDLE bindingJob = AppliedJobHandles().back();
        qcx::memory::AllocationInstrumentBindCapJob(bindingJob);
        // LogFirstCapApply is deliberately not called: it reports the FIRST
        // apply of the process, and a first call can never take this branch
        // (nothing is applied yet), so the line would only ever be a lie.
        return AppliedProcessCaps{true, std::nullopt};
    }

    const HANDLE job = CreateJobObjectW(nullptr, nullptr);

    if (job == nullptr)
    {
        const DWORD error = GetLastError();
        LogFirstCapApply(resources.memoryCapGiB, "CreateJobObjectW failed", error);
        return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                   "resources.memory_cap_gib could not be applied: "
                                   "CreateJobObjectW failed (error " +
                                       std::to_string(error) +
                                       "); the run refuses "
                                       "rather than starting uncapped"));
    }

    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
    // The same limit flags as the benchmark harness gate (memory_gate.py):
    // the process-memory limit itself plus kill-on-close (the handle is
    // process-lifetime, so the flag never fires in practice - it is the
    // harness pattern kept verbatim).
    info.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    info.ProcessMemoryLimit = static_cast<std::size_t>(resources.memoryCapGiB * (1ULL << 30));

    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info)))
    {
        const DWORD error = GetLastError();
        LogFirstCapApply(resources.memoryCapGiB, "SetInformationJobObject failed", error);
        CloseHandle(job);
        return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                   "resources.memory_cap_gib could not be applied: "
                                   "SetInformationJobObject failed (error " +
                                       std::to_string(error) +
                                       "); the run refuses rather than "
                                       "starting uncapped"));
    }

    if (!AssignProcessToJobObject(job, GetCurrentProcess()))
    {
        const DWORD error = GetLastError();

        if (error == ERROR_ACCESS_DENIED)
        {
            CloseHandle(job);
            LogFirstCapApply(resources.memoryCapGiB, "no-op (external job, access denied)", error);
            return AppliedProcessCaps{false,
                                      "the process is already in an external job (the benchmark "
                                      "harness gate or a CI runner); its limits are the caller's "
                                      "authority and the in-process cap was not applied"};
        }

        CloseHandle(job);

        // The ceiling, named instead of surfaced as a bare error 50 that
        // reads as an environment problem. Still a hard refusal, NEVER the
        // ACCESS_DENIED no-op above: folding a resource-exhaustion failure
        // into that no-op would start the run uncapped, the one outcome
        // fail-closed exists to prevent. The floor on the applied count is
        // what makes error 50 legible as the ceiling (kJobCeilingFloor).
        if (error == ERROR_NOT_SUPPORTED && AppliedJobHandles().size() >= kJobCeilingFloor)
        {
            LogFirstCapApply(resources.memoryCapGiB, "job-object ceiling reached", error);
            return std::unexpected(
                Err(qcx::ErrorCode::kInternalError,
                    "resources.memory_cap_gib could not be applied: the process has reached the "
                    "Windows job-object ceiling - AssignProcessToJobObject failed with error 50 "
                    "(ERROR_NOT_SUPPORTED) after " +
                        std::to_string(AppliedJobHandles().size()) +
                        " in-process cap job objects (one process can hold about 100; measured 100 "
                        "binds inside the harness job and 101 in a job-free process, 2026-09-12). "
                        "Each strictly tighter cap nests one more job while a cap at or above the "
                        "one in force reuses it; the run refuses rather than starting uncapped"));
        }

        LogFirstCapApply(resources.memoryCapGiB, "AssignProcessToJobObject failed", error);
        return std::unexpected(Err(qcx::ErrorCode::kInternalError,
                                   "resources.memory_cap_gib could not be applied: "
                                   "AssignProcessToJobObject failed (error " +
                                       std::to_string(error) +
                                       "); the run refuses rather "
                                       "than starting uncapped"));
    }

    AppliedJobHandles().push_back(job);
    CapState().applied = true;
    CapState().tightestCapGiB = resources.memoryCapGiB;

    // The allocation-attribution instrument's job-peak sampler binds the applied
    // cap's handle (the instrument lives in the memory module and cannot
    // see the driver's handle, so the driver binds it here): the watchdog
    // queries PeakProcessMemoryUsed from the MOST RECENTLY applied job —
    // nested jobs' limits combine, so the innermost cap is the binding
    // one. The bind precedes any AllocationInstrumentEnable by
    // construction (caps go on first in RunDriver). When the process is
    // already inside an external job and nothing was applied, nothing is
    // bound and the sampler reports 0 — the external job's limits are the
    // caller's authority there.
    qcx::memory::AllocationInstrumentBindCapJob(job);

    LogFirstCapApply(resources.memoryCapGiB, "bound", 0);

    return AppliedProcessCaps{true, std::nullopt};
#else
    // No job objects on this platform (Linux CI): the hard cap cannot be
    // applied in-process and NOTHING enforces it here - no predictive
    // model stands in for the missing limit. The record says so honestly
    // instead of pretending a cap
    // exists.
    (void)resources;
    return AppliedProcessCaps{false,
                              "this platform has no Windows job objects; the hard cap is "
                              "Windows-only and nothing enforces a limit here"};
#endif
}

bool ProcessMemoryCapEnforced(double memoryCapGiB) noexcept {
#ifdef _WIN32
    if (memoryCapGiB <= 0.0)
    {
        // No cap in the input (the documented escape hatch): there is no
        // limit to be in force, so nothing is lost by reading false.
        return false;
    }

    // 1. The driver's own applied cap. The state is process-lifetime and
    //    holds the TIGHTEST cap applied, and nested job limits combine with
    //    the tightest binding - so a tightest cap at or below this run's cap
    //    is OS-enforced by construction.
    if (CapState().applied && CapState().tightestCapGiB > 0.0 &&
        CapState().tightestCapGiB <= memoryCapGiB)
    {
        return true;
    }

    // 2. The enclosing job object. A NULL handle asks for the job associated
    //    with the CALLING process, which is the only way the driver can see a
    //    limit it did not apply: the benchmark harness gate caps the qcx child
    //    exactly this way (tools/bench/memory_gate.py ApplyToChild), and
    //    ApplyProcessCaps reports that case as its external-job no-op at
    //    ERROR_ACCESS_DENIED - the limit is real and the driver can read it.
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};

    if (!QueryInformationJobObject(
            nullptr, JobObjectExtendedLimitInformation, &info, sizeof(info), nullptr))
    {
        return false;
    }

    if ((info.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_PROCESS_MEMORY) == 0)
    {
        return false;
    }

    const double limitGiB = static_cast<double>(info.ProcessMemoryLimit) / (1ULL << 30);

    return limitGiB > 0.0 && limitGiB <= memoryCapGiB;
#else
    // No job objects on this platform (Linux CI), so no real limit can be in
    // force - the same statement ApplyProcessCaps' own Linux arm makes.
    (void)memoryCapGiB;

    return false;
#endif
}

std::optional<std::uint64_t> PeakProcessMemoryBytes() noexcept {
#ifdef _WIN32
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};

    if (!QueryInformationJobObject(
            nullptr, JobObjectExtendedLimitInformation, &info, sizeof(info), nullptr))
    {
        return std::nullopt;
    }

    if (info.PeakProcessMemoryUsed == 0)
    {
        return std::nullopt;
    }

    return static_cast<std::uint64_t>(info.PeakProcessMemoryUsed);
#else
    return std::nullopt;
#endif
}

} // namespace qcx::driver
