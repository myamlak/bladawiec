// The small-cap tests of the hard job-object memory cap. The cap applies
// to the whole process, so these spawn a dedicated child binary
// (qcx-driver-cap-child, the team_size_test pattern) instead of capping
// the test binary itself. The child runs the H2 STO-3G fixture through
// the real driver under the given cap and writes a marker file only when
// the run completes.
//
// The two halves of the contract: a small-but-sufficient cap does not
// disturb a small run (0.1 GiB on H2 completes normally), and a cap
// smaller than the run's own footprint ends the child without completing
// the run (0.1 MiB on H2: either the job object kills the child at the
// cap or a refusal lands first - the parent and the machine surviving is
// the assertion itself).
//
// The accumulation tests drive ApplyProcessCaps directly. A
// Windows process may carry only about 100 job objects and the applied cap
// handles are process-lifetime (never released), so a cap that is not
// tighter than the one already in force must REUSE its job: one job in
// force per process, not one per call. The old one-job-per-call rule made
// the 102nd assignment fail with ERROR_NOT_SUPPORTED (50), which the driver
// suite hit on its 118th case - its ~117 in-process RunDriver call sites sat
// exactly on the ceiling. Both tests below are the by-design version of
// that accident, and the ceiling is probed in a spawned COPY of this binary
// (selected by QCX_CAPS_CEILING_PROBE): exhausting the job budget is
// unrecoverable for the process that spends it, so the suite's own process
// must never be the one to drive past the ceiling.

#include "cap_child_harness.hpp"
#include "fast_test_mode.hpp"
#include "qcx/driver/process_caps.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

TEST(ProcessCapsTest, SmallCapDoesNotDisturbASmallRun) {
#ifdef QcxHasCuda
    // CUDA build only: the cuBLAS runtime's load-time footprint
    // (~0.25-0.3 GiB, measured for the cap probe gate; the environmental floor) exceeds any
    // small cap, so a CUDA-linked child dies inside the runtime before the
    // run starts - no qcx ordering can fire first, the driver's probe gate
    // never even runs below the floor. The clean-run contract is CPU-side;
    // the CUDA-less lanes carry this fixture's coverage.
    GTEST_SKIP() << "CUDA build: the runtime's load-time footprint (~0.25-0.3 GiB) exceeds "
                    "the 0.1 GiB cap; this clean-run contract is verified on the CUDA-less "
                    "builds";
#else
    // The hard cap is a ceiling, not a floor: 0.1 GiB on the ~tens-of-MB
    // H2 fixture must not disturb the run (the "cap 0.1 GiB on H2"
    // clean-run branch), and the resources_resolved block must say the
    // in-process cap was applied on Windows.
    const auto marker = qcx::testing::CapChildMarkerPath(1);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(0.1, marker);

    EXPECT_EQ(exitCode, 0);
    EXPECT_NE(qcx::testing::ReadCapChildMarker(marker).find("completed"), std::string::npos)
        << "the run must have completed under a 0.1 GiB cap";
    std::filesystem::remove(marker);
#endif
}

#ifdef _WIN32
TEST(ProcessCapsTest, TinyCapKillsTheChildAtTheCap) {
    // 0.1 MiB is far below the H2 fixture's own footprint, so the run can
    // never complete. Two outcomes race, and both are fine here: the job
    // object terminates the child at the next commit past the limit (no
    // marker), or the child reaches a refusal first - the basis load or
    // the memory seam - and exits 1 with a "refused:" marker (the refusal
    // path writes the marker by contract, cap_child.cpp). "Never machine
    // death" is the assertion itself - the parent process and the machine
    // survive, and the completion marker must never appear.
    const auto marker = qcx::testing::CapChildMarkerPath(2);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(0.0001, marker);

    EXPECT_NE(exitCode, 0) << "the capped child must not complete";
    EXPECT_EQ(qcx::testing::ReadCapChildMarker(marker).find("completed"), std::string::npos)
        << "the capped child must never write the completion marker";
    std::filesystem::remove(marker);
}
#else
TEST(ProcessCapsTest, TinyCapKillsTheChildAtTheCap) {
    GTEST_SKIP() << "no Windows job objects on this platform; the child's cap is a no-op";
}
#endif

#ifdef _WIN32

namespace {

// The probe child's selection AND its output path in one mechanism: the
// parent puts the marker path in the child's environment (CreateProcessW
// inherits the current block when lpEnvironment is null) and clears the
// variable right after the spawn, so nothing but the spawned copy takes the
// probe branch.
constexpr const wchar_t* kProbeEnvName = L"QCX_CAPS_CEILING_PROBE";
constexpr const wchar_t* kProbeFilter =
    L"--gtest_filter=ProcessCapsTest.JobObjectCeilingRefusalNamesTheCeiling";

// io's [resources] default: memory_cap_gib absent means 16.0 GiB
// (parse_input.cpp), which is also what RunResourcesInput holds.
constexpr double kDefaultCapGiB = 16.0;
// More than the whole job budget of the process, so these applies only fit
// by reuse.
constexpr std::size_t kReuseCalls = 200;
// The bound on the probe child's run (see RunCeilingProbe).
constexpr DWORD kProbeTimeoutMs = 300000;
// The descent walks strictly tighter caps until the assignment refuses. The
// bound keeps the walk finite - and the caps in the tens of GiB, far above
// the probe's own footprint - should a platform ever have no ceiling.
constexpr std::size_t kDescentLimit = 200;
constexpr double kDescentStepGiB = 0.01;

std::filesystem::path SelfExePath() {
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);

    // The Source constructor, not the iterator one (see cap_child_harness):
    // path(first, last) treats each character as a path component.
    return std::filesystem::path(buffer);
}

std::filesystem::path ProbeMarkerFromEnvironment() {
    wchar_t buffer[MAX_PATH];
    const DWORD length = GetEnvironmentVariableW(kProbeEnvName, buffer, MAX_PATH);

    if (length == 0 || length >= MAX_PATH)
    {
        return {};
    }

    return std::filesystem::path(buffer);
}

// Runs this binary as the ceiling probe; returns the child's exit code.
int RunCeilingProbe(const std::filesystem::path& marker) {
    SetEnvironmentVariableW(kProbeEnvName, marker.wstring().c_str());

    std::wstring command = L"\"" + SelfExePath().wstring() + L"\" " + kProbeFilter;

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};

    // A quoted command line is mandatory: CreateProcessW parses it itself
    // (cap_child_harness.hpp carries the rejected-command-line history).
    const BOOL created = CreateProcessW(nullptr,
                                        command.data(),
                                        nullptr,
                                        nullptr,
                                        FALSE,
                                        CREATE_NO_WINDOW,
                                        nullptr,
                                        nullptr,
                                        &startup,
                                        &process);

    SetEnvironmentVariableW(kProbeEnvName, nullptr);

    if (!created)
    {
        return -1;
    }

    CloseHandle(process.hThread);

    // Bounded, unlike the cap-child harness's INFINITE wait: the probe's
    // whole job is a few hundred job-object calls (well under a second), and
    // a test that can hang the suite is worse than one that fails. 300 s is
    // ~30x the measured cost, so a spurious timeout is not a real risk.
    DWORD exitCode = 0;

    if (WaitForSingleObject(process.hProcess, kProbeTimeoutMs) == WAIT_TIMEOUT)
    {
        TerminateProcess(process.hProcess, 1);
        exitCode = static_cast<DWORD>(-2);
    } else
    {
        GetExitCodeProcess(process.hProcess, &exitCode);
    }

    CloseHandle(process.hProcess);

    return static_cast<int>(exitCode);
}

// The probe protocol, as a record the parent asserts on:
//   1. the first apply of the process at the run default - must bind (a
//      context that refuses the assignment outright ends the probe here);
//   2. kReuseCalls applies at that same cap - all must bind, which only
//      reuse makes possible;
//   3. descending caps, each strictly tighter than the one in force, so
//      each nests one more job until the assignment refuses;
//   4. one more apply at the default cap - must bind, because the refusal
//      must leave the caps already in force alone.
// Nothing is asserted here: the child only records, so the parent's failure
// output can carry the whole picture.
std::string DriveTheCapPastTheCeiling() {
    // The launch context, recorded as evidence rather than asserted on: it
    // is what makes the descent's bind count legible (a process already
    // inside a job - the benchmark harness gate or a CI runner - starts
    // with one of the ~100 slots spent, so it takes one cap job fewer).
    BOOL inJob = FALSE;
    IsProcessInJob(GetCurrentProcess(), nullptr, &inJob);

    std::string record = std::string("launchContext: inJob=") + (inJob ? "1" : "0") + "\n";

    qcx::io::RunResourcesInput resources{};
    resources.memoryCapGiB = kDefaultCapGiB;

    const auto first = qcx::driver::ApplyProcessCaps(resources);

    if (!first.has_value())
    {
        return record + "first: error\nfirstMessage: " + first.error().message + "\n";
    }

    if (!first->inProcessCapApplied)
    {
        return record + "first: noop\nfirstNote: " + first->note.value_or("") + "\n";
    }

    record += "first: bound\n";

    std::size_t reuseBinds = 0;
    std::string reuseRefusal = "none";

    for (std::size_t i = 0; i < kReuseCalls; ++i)
    {
        const auto applied = qcx::driver::ApplyProcessCaps(resources);

        if (!applied.has_value())
        {
            reuseRefusal = "error: " + applied.error().message;
            break;
        }

        if (!applied->inProcessCapApplied)
        {
            reuseRefusal = "noop: " + applied->note.value_or("");
            break;
        }

        ++reuseBinds;
    }

    record += "reuseBinds: " + std::to_string(reuseBinds) + "\n";
    record += "reuseRefusal: " + reuseRefusal + "\n";

    std::size_t descentBinds = 0;
    std::string refusalKind = "none";
    std::string refusalMessage;

    for (std::size_t i = 1; i <= kDescentLimit; ++i)
    {
        qcx::io::RunResourcesInput tighter{};
        tighter.memoryCapGiB = kDefaultCapGiB - kDescentStepGiB * static_cast<double>(i);
        const auto applied = qcx::driver::ApplyProcessCaps(tighter);

        if (!applied.has_value())
        {
            refusalKind = "error";
            refusalMessage = applied.error().message;
            break;
        }

        if (!applied->inProcessCapApplied)
        {
            refusalKind = "noop";
            refusalMessage = applied->note.value_or("");
            break;
        }

        ++descentBinds;
    }

    record += "refusalKind: " + refusalKind + "\n";
    record += "descentBinds: " + std::to_string(descentBinds) + "\n";
    record += "refusalMessage: " + refusalMessage + "\n";

    const auto after = qcx::driver::ApplyProcessCaps(resources);
    const bool afterBinds = after.has_value() && after->inProcessCapApplied;
    record += std::string("postRefusalReuseBinds: ") + (afterBinds ? "1" : "0") + "\n";

    return record;
}

// The number on a "key: N" record line (0 when the line is absent).
std::size_t NumberOnLine(const std::string& record, const std::string& key) {
    const std::size_t at = record.find(key);

    if (at == std::string::npos)
    {
        return 0;
    }

    return static_cast<std::size_t>(std::strtoull(record.c_str() + at + key.size(), nullptr, 10));
}

std::string ReadRecord(const std::filesystem::path& marker) {
    std::ifstream stream(marker);

    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void WriteRecord(const std::filesystem::path& marker, const std::string& record) {
    std::ofstream stream(marker);
    stream << record;
}

} // namespace

TEST(ProcessCapsTest, EqualCapAppliesReuseTheJobInForce) {
    // The suite's own process is where the accident ran: ~117 in-process
    // RunDriver call sites against a ~100-job budget, so the equal-cap
    // reuse is asserted in-process. kReuseCalls applies at one constant cap
    // only fit if they reuse - one job per call exhausts the budget at
    // about the hundredth (measured: the 100th binds inside the harness
    // job, the 101st in a job-free process, the next assignment fails with
    // error 50).
    qcx::io::RunResourcesInput resources{};
    resources.memoryCapGiB = kDefaultCapGiB;

    for (std::size_t i = 0; i < kReuseCalls; ++i)
    {
        const auto applied = qcx::driver::ApplyProcessCaps(resources);

        ASSERT_TRUE(applied.has_value())
            << "apply " << i << " at the constant default cap refused: " << applied.error().message;

        ASSERT_TRUE(applied->inProcessCapApplied)
            << "apply " << i << " at the constant default cap returned the external-job no-op: "
            << applied->note.value_or("");
    }
}

TEST(ProcessCapsTest, JobObjectCeilingRefusalNamesTheCeiling) {
    const std::filesystem::path probeMarker = ProbeMarkerFromEnvironment();

    if (!probeMarker.empty())
    {
        // The probe child: record and return - the assertions live in the
        // parent, whose failure output carries the record.
        WriteRecord(probeMarker, DriveTheCapPastTheCeiling());
        return;
    }

    const std::filesystem::path marker =
        std::filesystem::temp_directory_path() /
        ("qcx-caps-ceiling-" + std::to_string(GetCurrentProcessId()) + ".marker");
    std::filesystem::remove(marker);

    const int exitCode = RunCeilingProbe(marker);
    const std::string record = ReadRecord(marker);
    std::filesystem::remove(marker);

    ASSERT_FALSE(record.empty())
        << "the probe child wrote no record (exit code " << exitCode
        << "; -1 = the spawn failed, -2 = the child did not finish within the probe timeout, "
           "0 = the child ran but the probe test matched nothing - renamed?)";

    EXPECT_EQ(exitCode, 0) << "the probe child exited non-zero: " << record;

    if (record.find("first: noop") != std::string::npos)
    {
        GTEST_SKIP() << "this context refuses the assignment into the driver's own job (an "
                        "external job that forbids nesting), so no in-process cap is applied "
                        "and the ceiling cannot be reached here: "
                     << record;
    }

    ASSERT_NE(record.find("first: bound"), std::string::npos)
        << "the probe's first apply at the default cap did not bind: " << record;

    // Reuse: the kReuseCalls equal-cap applies all bound, and the descent
    // bound about a hundred more - roughly 300 binds in a process whose
    // whole job budget is about 100. Only reuse can do that; one job per
    // call stops at the ceiling (the accident: error 50 on the 102nd).
    EXPECT_NE(record.find("reuseRefusal: none"), std::string::npos)
        << "an equal-cap apply must reuse the job in force, not refuse or no-op: " << record;
    EXPECT_NE(record.find("reuseBinds: " + std::to_string(kReuseCalls)), std::string::npos)
        << "all " << kReuseCalls << " equal-cap applies must bind: " << record;

    // Fail-closed: the ceiling is an Error, never the ACCESS_DENIED no-op
    // that would let the run start uncapped.
    EXPECT_NE(record.find("refusalKind: error"), std::string::npos)
        << "reaching the ceiling must refuse, not proceed uncapped: " << record;

    // And the refusal names the ceiling it reached, with the number of cap
    // jobs behind it (one per nested descent step plus the first bind) - not
    // a bare "AssignProcessToJobObject failed (error 50)".
    const std::size_t descentBinds = NumberOnLine(record, "descentBinds: ");
    EXPECT_GE(descentBinds, 50) << "the descent must reach the ~100-job ceiling: " << record;
    EXPECT_NE(record.find("job-object ceiling"), std::string::npos)
        << "the refusal must name the job-object ceiling: " << record;
    EXPECT_NE(
        record.find("after " + std::to_string(descentBinds + 1) + " in-process cap job objects"),
        std::string::npos)
        << "the refusal must report the " << descentBinds + 1 << " cap jobs it applied: " << record;

    // The refusal must leave the caps in force alone.
    EXPECT_NE(record.find("postRefusalReuseBinds: 1"), std::string::npos)
        << "a refused apply must not disturb the cap already in force: " << record;
}

TEST(ProcessCapsTest, CappedProcessReadsItsCapAsEnforced) {
    // The cap-READING pin, in the one context where it can differ from the
    // host's: a process whose cap is REAL. (The advisory-verdict half of
    // this cell - the memory model's over-cap verdict under a real cap -
    // was deleted with the model on 2026-09-17.)
    //
    // The cap is 2 GiB, far above this child's own footprint (it applies
    // one job object and reads its own cap).
    constexpr double kCapReadingGiB = 2.0;

    const auto marker = qcx::testing::CapChildMarkerPath(25);
    std::filesystem::remove(marker);

    const int exitCode = qcx::testing::RunCapChild(kCapReadingGiB, marker, "cap_verdict");
    const std::string record = qcx::testing::ReadCapChildMarker(marker);
    std::filesystem::remove(marker);

    ASSERT_FALSE(record.empty()) << "the cap-verdict child wrote no marker (exit code " << exitCode
                                 << "; 2 = bad arguments)";
    ASSERT_NE(record.find("applied: bound"), std::string::npos)
        << "the child could not put itself under its own cap: " << record;

    EXPECT_NE(record.find("enforcedAtCap: 1"), std::string::npos)
        << "a cap the driver applied must read as enforced: " << record;
    EXPECT_NE(record.find("enforcedAtTighter: 0"), std::string::npos)
        << "a cap tighter than the one in force must read false (the conservative read): "
        << record;
}

#else

TEST(ProcessCapsTest, EqualCapAppliesReuseTheJobInForce) {
    GTEST_SKIP() << "no Windows job objects on this platform; the cap is a no-op here";
}

TEST(ProcessCapsTest, JobObjectCeilingRefusalNamesTheCeiling) {
    GTEST_SKIP() << "no Windows job objects on this platform; the cap is a no-op here";
}

TEST(ProcessCapsTest, CappedProcessReadsItsCapAsEnforced) {
    GTEST_SKIP() << "no Windows job objects on this platform; the cap is a no-op here";
}

#endif
