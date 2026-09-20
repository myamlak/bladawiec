// The small-cap test harness shared by the driver test files: spawns
// qcx-driver-cap-child (a dedicated binary, because a memory cap applies
// to the whole process — a test that capped its own host test binary
// would kill the suite) and reads its marker file. See cap_child.cpp for
// the child's contract.

#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#ifdef _WIN32
// NOMINMAX: the windows min/max macros break std::min in the driver test
// TUs that include this harness before cpu_backend.hpp (C2589 at
// cpu_backend.hpp:55). Nothing in this header needs the macros.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <chrono>
#include <csignal>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#endif

#if defined(__APPLE__)
// dyld, for the running executable's path in CapChildExePath below: macOS
// has no /proc for the Linux route to read.
#include <mach-o/dyld.h>
#endif

namespace qcx::testing {

// The value RunCapChild returns when a wait bound expired and it stopped the
// child: the suite's established vocabulary for a probe that did not finish
// in time (-1 = the spawn failed, -2 = the child did not finish within its
// bound - process_caps_test.cpp's ceiling probe returns the same pair). No
// real child produces it: the child's own codes are 0 (the run completed),
// 1 (a refusal) and 2 (bad arguments), a Windows cap kill or fast-fail
// reports a large negative int (0xC0000409 = -1073740791) and a POSIX wait
// status is a small non-negative number - so a caller can tell a leg that
// never returned from one that decided.
inline constexpr int kCapChildWaitExpired = -2;

// How long an expired child is given to die before RunCapChild stops waiting
// for it: the function returns bounded whether or not the kill lands at once.
inline constexpr unsigned long kCapChildTerminationGraceMs = 5000;

// How long the bounded POSIX wait sleeps between two waitpid checks. A sleep,
// not a spin: the wait costs the platform's own timer.
inline constexpr long kCapChildWaitTickMs = 50;

// The child exe sits next to this test exe (both build into driver/<cfg>).
inline std::filesystem::path CapChildExePath() {
#ifdef _WIN32
    wchar_t buffer[MAX_PATH];
    GetModuleFileNameW(nullptr, buffer, MAX_PATH);
    // The Source constructor, not the iterator one: path(first, last)
    // treats each character as a path component, which would turn the
    // buffer into "C\:\U\s\e\..." garbage (caught when cmd rejected the
    // spawned command line, 2026-08-29).
    std::filesystem::path self(buffer);
#elif defined(__APPLE__)
    // No /proc on macOS, so the Linux route below has nothing to read: dyld
    // is the interface that knows. The first call is the documented size
    // query - it answers -1 and writes the length the path needs into size -
    // and the second fills the buffer that length asked for.
    std::uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::string exePath(size, '\0');
    _NSGetExecutablePath(exePath.data(), &size);
    std::filesystem::path self(exePath.c_str());
#else
    std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe");
#endif
    std::filesystem::path name = "qcx-driver-cap-child";
#ifdef _WIN32
    name += ".exe";
#endif
    return self.parent_path() / name;
}

inline std::filesystem::path CapChildMarkerPath(std::size_t id) {
#ifdef _WIN32
    const std::size_t processId = static_cast<std::size_t>(GetCurrentProcessId());
#else
    const std::size_t processId = static_cast<std::size_t>(getpid());
#endif
    return std::filesystem::temp_directory_path() /
           ("qcx-cap-child-" + std::to_string(processId) + "-" + std::to_string(id) + ".marker");
}

// Spawns the child with the given cap, a marker path, and an optional
// fixture selector ("direct" default, "ri_j" - see cap_child.cpp); returns
// the child's exit code (the OS status when the cap killed it). When a wait
// bound is given and expires, the child is stopped, whatever it flushed to
// the marker stays there for ReadCapChildMarker to read, and the return is
// kCapChildWaitExpired.
//
// The wait is UNBOUNDED by default - the behaviour every existing caller
// relies on - because a fixture under a cap that ADMITS its work legitimately
// runs for minutes (the ri_jk fixtures ran 150 s to 16+ minutes under the
// caps that completed). The bound is therefore per call and never global: a
// caller that wants one hands in its own, and the default keeps the call the
// call it was.
//
// Deliberately NOT std::system: the CRT's system() routes through the
// command processor, whose quote handling varies by locale and shell
// environment (the spawned command line "exe" 0.1 "marker" was rejected
// with ERROR_INVALID_NAME on this machine, 2026-08-29). A direct spawn
// has no shell in the middle and nothing to quote wrong. On POSIX the
// raw wait status is normalized to the child's plain exit code.
//
// \param capGib The memory cap in GiB handed to the child.
// \param marker The marker file the child writes its outcome to.
// \param fixture The child's fixture selector.
// \param timeoutMilliseconds How long to wait for the child before stopping
//        it. 0 (the default, and any non-positive value) means unbounded.
inline int RunCapChild(double capGib,
                       const std::filesystem::path& marker,
                       const std::string& fixture = "direct",
                       int timeoutMilliseconds = 0) {
#ifdef _WIN32
    std::wstring command = L"\"" + CapChildExePath().wstring() + L"\" " + std::to_wstring(capGib) +
                           L" \"" + marker.wstring() + L"\" " +
                           std::wstring(fixture.begin(), fixture.end());

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    // A quoted command line is mandatory: CreateProcessW parses it itself.
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

    if (!created)
    {
        return -1;
    }

    CloseHandle(process.hThread);
    const DWORD waited = WaitForSingleObject(
        process.hProcess,
        timeoutMilliseconds > 0 ? static_cast<DWORD>(timeoutMilliseconds) : INFINITE);

    if (waited == WAIT_TIMEOUT)
    {
        // Not coming back: stop it and report the sentinel, which is also the
        // termination code - a caller that reads the OS status instead of the
        // return value sees it too. The second wait is bounded as well, so
        // this function returns whether or not the kill lands at once (the
        // handle is closed either way; that only drops this reference).
        TerminateProcess(process.hProcess, static_cast<UINT>(kCapChildWaitExpired));
        WaitForSingleObject(process.hProcess, kCapChildTerminationGraceMs);
        CloseHandle(process.hProcess);
        return kCapChildWaitExpired;
    }

    DWORD exitCode = 0;
    GetExitCodeProcess(process.hProcess, &exitCode);
    CloseHandle(process.hProcess);
    return static_cast<int>(exitCode);
#else
    const std::string exe = CapChildExePath().string();
    const std::string markerText = marker.string();
    const std::string capText = std::to_string(capGib);
    const std::string command = exe + " " + capText + " " + markerText + " " + fixture;

    if (timeoutMilliseconds <= 0)
    {
        // Unbounded: the std::system call every existing caller has always
        // made, wait status normalization included.
        const int status = std::system(command.c_str());
        return WIFEXITED(status) ? WEXITSTATUS(status) : status;
    }

    // Bounded: std::system's wait cannot be given a deadline, so the spawn is
    // direct (fork/exec - no shell, the same reason the Windows path spawns
    // directly) and the PID is in hand, which addresses THIS child exactly
    // where a command-line pattern search would only describe it. Everything
    // the child arms is read before the fork: only async-signal-safe calls
    // run between fork and exec.
    const pid_t pid = fork();

    if (pid < 0)
    {
        return -1;
    }

    if (pid == 0)
    {
        execl(exe.c_str(),
              exe.c_str(),
              capText.c_str(),
              markerText.c_str(),
              fixture.c_str(),
              static_cast<char*>(nullptr));
        _exit(127);
    }

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMilliseconds);
    int status = 0;

    while (true)
    {
        const pid_t reaped = waitpid(pid, &status, WNOHANG);

        if (reaped == pid)
        {
            return WIFEXITED(status) ? WEXITSTATUS(status) : status;
        }

        if (reaped < 0 && errno != EINTR)
        {
            return -1;
        }

        if (std::chrono::steady_clock::now() >= deadline)
        {
            // The deadline is the whole point: stop the child, reap it (a
            // SIGKILLed child leaves no zombie), and report the sentinel.
            kill(pid, SIGKILL);
            waitpid(pid, &status, 0);
            return kCapChildWaitExpired;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(kCapChildWaitTickMs));
    }
#endif
}

inline std::string ReadCapChildMarker(const std::filesystem::path& marker) {
    std::ifstream stream(marker);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

} // namespace qcx::testing
