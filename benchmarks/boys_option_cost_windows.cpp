// The option-cost probe for the platform that ships: what the fp64 evaluation
// options cost, measured from outside the process that runs them.
//
// benchmarks/boys_option_count.cpp answers "how much work" and refuses to
// answer "how much does that work cost", because the counter it needs is the
// platform's and this platform has no perf. The option probe's own answer to
// cost is a wall clock, and on the development machine that clock could not
// separate the shapes: its resolution across admitted passes, 5.2% to 56.2%,
// was never tighter than the margin it was asked to order. Everything here
// exists to move that resolution.
//
// The instrument is QueryProcessCycleTime, with QueryThreadCycleTime recorded
// alongside it. A process's cycle count accrues only while one of its threads
// is running, so a pass that is descheduled mid-flight contributes nothing to
// it - that is the whole difference from a wall clock, and it is the term that
// made the wall clock's resolution wander with how steady the machine was
// rather than with what it was running. Wall time is recorded too, so the
// ratio of the two says how much of a run was descheduled rather than how much
// it worked.
//
// Whether the count is the core's own cycles or the timestamp counter is a
// property of this host and not of the interface, and the two rank differently
// when a clock changes frequency mid-run. --clock-check settles it here: it
// runs the fixed-work canary many times and reports the relation between its
// cycle count and its wall time. A core-cycle count of work that never changes
// barely moves while the wall time does; a timestamp counter is the wall time
// in other units and tracks it exactly. The ratio of the two, read against the
// nominal clock, names which one is in force.
//
// The protocol is the option-count probe's, at the granularity of a whole
// process. One option per process - a process's count carries its start-up,
// its accuracy pass and its workload construction, so two options never share
// one - and the per-pass figure is a two-point subtraction over two differing
// repetition counts:
//
//     (C(R2) - C(R1)) / (R2 - R1)   is the cost one pass pays
//
// which removes everything in the process that is not the work. Options
// alternate within a round and the order reverses between rounds, so a drift
// that is monotone in time lands on every option equally.
//
// Every pass carries a reading of a fixed-work integer spin, as the library's
// own option probe does: the fastest run of that work here is the machine's
// quiet floor, and a reading is the percentage by which a run of it exceeded
// that floor. The spin holds no floating-point state, so the arithmetic
// question this probe is about cannot change the cost of the instrument.
//
// Usage:
//   qcx-bench-boys-option-cost --exe=PATH [--rounds=N] [--reps-low=N]
//                              [--reps-high=N] [--nmax=32,16,8,4] [--csv=PATH]
//                              [--clock-check]
//
// The exe is the option-count driver, built by the same configuration as this
// probe. This one measures; that one counts.
#include <algorithm>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <windows.h>

namespace {

/// Iterations of the canary spin. Fixed, because the reading is a ratio of two
/// times of the same work and the work must not vary between them.
constexpr std::uint32_t kCanaryRounds = 1u << 23;

/// The nominal clock, in Hz, of the part this project develops on. Read only
/// to name which of the two counters the platform handed back; a host that is
/// not this one needs its own figure before that reading means anything.
constexpr double kNominalClockHz = 2.592e9;

struct Config {
    std::string exe;
    std::string csv = "boys_option_cost.csv";
    std::size_t rounds = 9;
    std::size_t repsLow = 1;
    std::size_t repsHigh = 257;
    std::size_t count = 16384;
    std::size_t seed = 47;
    std::string xrange = "1e-3,40";
    std::vector<int> nmax{32, 16, 8, 4};
    bool clockCheck = false;
};

const char* const kOptionNames[] = {"batch-fp64", "grouped-fp64", "tagged-fp64"};
constexpr std::size_t kOptionCount = 3;

/// The spin: a xorshift chain, one dependent step per iteration, no memory
/// beyond the accumulator and no floating point.
double CanaryMilliseconds() {
    volatile std::uint64_t sink = 0;
    LARGE_INTEGER frequency;
    LARGE_INTEGER from;
    LARGE_INTEGER to;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&from);

    std::uint64_t state = 0x243F6A8885A308D3ull;

    for (std::uint32_t i = 0; i < kCanaryRounds; ++i)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
    }

    // Inside the timed region on purpose: the store is the observable use of
    // the chain, so neither the chain nor the region can be moved past the
    // other.
    sink = state;
    QueryPerformanceCounter(&to);
    (void)sink;

    return static_cast<double>(to.QuadPart - from.QuadPart) * 1000.0 /
           static_cast<double>(frequency.QuadPart);
}

double NowMilliseconds() {
    LARGE_INTEGER frequency;
    LARGE_INTEGER now;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&now);
    return static_cast<double>(now.QuadPart) * 1000.0 / static_cast<double>(frequency.QuadPart);
}

/// One child run: the whole process, from its creation to its exit.
struct RunResult {
    std::uint64_t cycles = 0;
    std::uint64_t threadCycles = 0;
    double wallMs = 0.0;
    double canaryBeforeMs = 0.0;
    double canaryAfterMs = 0.0;
    unsigned long exitCode = 0;
    bool launched = false;
};

std::string Format(const char* format, ...) {
    char buffer[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return std::string(buffer);
}

/// Runs the option-count driver once, one option, and measures it from here.
///
/// The child's own output is left on the console: its accuracy line is the
/// verdict that the values this cost is being paid for are the right ones, and
/// a cost figure for the wrong values costs nothing.
RunResult RunOnce(const Config& config, const char* option, int nmax, std::size_t reps) {
    RunResult result;
    result.canaryBeforeMs = CanaryMilliseconds();

    const std::string command =
        Format("\"%s\" --option=%s --reps=%zu --nmax=%d --count=%zu --xrange=%s --seed=%zu",
               config.exe.c_str(),
               option,
               reps,
               nmax,
               config.count,
               config.xrange.c_str(),
               config.seed);

    std::vector<char> mutableCommand(command.begin(), command.end());
    mutableCommand.push_back('\0');

    STARTUPINFOA startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};

    const double wallFrom = NowMilliseconds();
    const BOOL created = CreateProcessA(nullptr,
                                        mutableCommand.data(),
                                        nullptr,
                                        nullptr,
                                        FALSE,
                                        CREATE_NO_WINDOW,
                                        nullptr,
                                        nullptr,
                                        &startup,
                                        &process);

    if (created == FALSE)
    {
        std::fprintf(stderr,
                     "qcx-bench-boys-option-cost: cannot start '%s' (error %lu)\n",
                     config.exe.c_str(),
                     GetLastError());
        return result;
    }

    WaitForSingleObject(process.hProcess, INFINITE);
    const double wallTo = NowMilliseconds();
    GetExitCodeProcess(process.hProcess, &result.exitCode);

    ULONG64 cycles = 0;
    ULONG64 threadCycles = 0;

    if (QueryProcessCycleTime(process.hProcess, &cycles) != FALSE)
    {
        result.cycles = cycles;
    }

    if (QueryThreadCycleTime(process.hThread, &threadCycles) != FALSE)
    {
        result.threadCycles = threadCycles;
    }

    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);

    result.wallMs = wallTo - wallFrom;
    result.launched = true;
    result.canaryAfterMs = CanaryMilliseconds();
    return result;
}

/// One measured pair: the low and high repetition runs of one option at one
/// order mix, and the per-pass figure their difference gives.
struct Pair {
    std::uint64_t cyclesPerPass = 0;
    double wallPerPassMs = 0.0;
    std::uint64_t cyclesLow = 0;
    std::uint64_t cyclesHigh = 0;
    std::uint64_t threadCyclesLow = 0;
    std::uint64_t threadCyclesHigh = 0;
    double wallLowMs = 0.0;
    double wallHighMs = 0.0;
    double canaryBeforeMs = 0.0;
    double canaryAfterMs = 0.0;
    unsigned long exitLow = 0;
    unsigned long exitHigh = 0;
    bool valid = false;
};

Pair MeasurePair(const Config& config, const char* option, int nmax) {
    Pair pair;

    // The low run first, then the high one: adjacent, so both halves of the
    // subtraction meet the same conditions, and the subtraction is what
    // removes the process's own fixed cost from the figure.
    const RunResult low = RunOnce(config, option, nmax, config.repsLow);
    const RunResult high = RunOnce(config, option, nmax, config.repsHigh);

    if (!low.launched || !high.launched || low.exitCode != 0 || high.exitCode != 0)
    {
        return pair;
    }

    if (high.cycles <= low.cycles)
    {
        return pair;
    }

    const std::size_t passes = config.repsHigh - config.repsLow;
    pair.cyclesPerPass = (high.cycles - low.cycles) / passes;
    pair.wallPerPassMs = (high.wallMs - low.wallMs) / static_cast<double>(passes);
    pair.cyclesLow = low.cycles;
    pair.cyclesHigh = high.cycles;
    pair.threadCyclesLow = low.threadCycles;
    pair.threadCyclesHigh = high.threadCycles;
    pair.wallLowMs = low.wallMs;
    pair.wallHighMs = high.wallMs;
    pair.canaryBeforeMs = std::min(low.canaryBeforeMs, high.canaryBeforeMs);
    pair.canaryAfterMs = std::max(low.canaryAfterMs, high.canaryAfterMs);
    pair.exitLow = low.exitCode;
    pair.exitHigh = high.exitCode;
    pair.valid = true;
    return pair;
}

/// The clock check: does the platform's cycle count follow the core's work or
/// the wall clock?
///
/// Runs the same fixed work many times and reports, per run, its cycle count,
/// its wall time, and the clock the ratio implies. A count that tracks the
/// wall clock exactly is the timestamp counter, and the implied clock sits at
/// the part's nominal figure whatever the core is doing. A count that holds
/// still while the wall time moves is the core's own cycles, and the implied
/// clock wanders with the core's frequency.
int RunClockCheck() {
    constexpr std::size_t kRuns = 40;
    std::vector<double> implied;
    std::vector<double> walls;
    std::vector<std::uint64_t> cycles;
    implied.reserve(kRuns);

    std::printf("clock-check | fixed-work spin, %zu runs\n", kRuns);

    for (std::size_t i = 0; i < kRuns; ++i)
    {
        ULONG64 before = 0;
        ULONG64 after = 0;
        QueryThreadCycleTime(GetCurrentThread(), &before);
        const double wallMs = CanaryMilliseconds();
        QueryThreadCycleTime(GetCurrentThread(), &after);

        const std::uint64_t spent = after - before;
        const double seconds = wallMs / 1000.0;
        const double clock = (seconds > 0.0) ? static_cast<double>(spent) / seconds : 0.0;

        cycles.push_back(spent);
        walls.push_back(wallMs);
        implied.push_back(clock);

        std::printf("clock-check run | %zu | cycles %llu | wall_ms %.6f | implied_clock_hz %.0f\n",
                    i,
                    static_cast<unsigned long long>(spent),
                    wallMs,
                    clock);
    }

    const auto minmaxCycles = std::minmax_element(cycles.begin(), cycles.end());
    const auto minmaxWalls = std::minmax_element(walls.begin(), walls.end());
    const auto minmaxImplied = std::minmax_element(implied.begin(), implied.end());

    const double cycleSpread = 100.0 *
                               static_cast<double>(*minmaxCycles.second - *minmaxCycles.first) /
                               static_cast<double>(*minmaxCycles.first);
    const double wallSpread =
        100.0 * (*minmaxWalls.second - *minmaxWalls.first) / *minmaxWalls.first;
    const double ratio = static_cast<double>(*minmaxImplied.second) / kNominalClockHz;

    std::printf("clock-check summary | cycles min %llu max %llu spread %.3f%% | wall_ms min %.4f "
                "max %.4f spread %.3f%% | implied_clock_hz min %.0f max %.0f | "
                "max_over_nominal %.3f\n",
                static_cast<unsigned long long>(*minmaxCycles.first),
                static_cast<unsigned long long>(*minmaxCycles.second),
                cycleSpread,
                *minmaxWalls.first,
                *minmaxWalls.second,
                wallSpread,
                *minmaxImplied.first,
                *minmaxImplied.second,
                ratio);
    return 0;
}

void Usage() {
    std::fputs("qcx-bench-boys-option-cost - the fp64 options' cost, measured from outside\n"
               "\n"
               "usage: qcx-bench-boys-option-cost --exe=PATH [options]\n"
               "\n"
               "  --exe=PATH         the option-count driver to measure (required)\n"
               "  --rounds=N         measured rounds, one pair per option and mix (default 9)\n"
               "  --reps-low=N       the subtraction's low repetition count (default 1)\n"
               "  --reps-high=N      the subtraction's high repetition count (default 257)\n"
               "  --nmax=LIST        order mixes, comma separated (default 32,16,8,4)\n"
               "  --count=N          arguments per pass (default 16384)\n"
               "  --seed=N           workload generator seed (default 47)\n"
               "  --xrange=LO,HI     log-uniform argument range (default 1e-3,40)\n"
               "  --csv=PATH         where the per-pair rows go (default boys_option_cost.csv)\n"
               "  --clock-check      settle which counter the platform returned, then stop\n",
               stdout);
}

std::vector<int> ParseNmax(const std::string& text) {
    std::vector<int> mixes;
    const char* cursor = text.c_str();

    while (*cursor != '\0')
    {
        mixes.push_back(std::atoi(cursor));
        const char* comma = std::strchr(cursor, ',');

        if (comma == nullptr)
        {
            break;
        }

        cursor = comma + 1;
    }

    if (mixes.empty())
    {
        mixes.push_back(32);
    }

    return mixes;
}

/// One option's per-pass figures over the rounds, and what they say about the
/// resolution this instrument reached.
struct Summary {
    std::vector<std::uint64_t> cycles;
    std::vector<double> walls;
    double canaryPeakPercent = 0.0;
};

void PrintSummary(const Config& config,
                  const std::vector<Summary>& summaries,
                  double canaryFloorMs,
                  FILE* csv) {
    std::printf("\nper-pass cost | cycles = QueryProcessCycleTime, the two-point subtraction over "
                "reps %zu..%zu\n",
                config.repsLow,
                config.repsHigh);

    const std::vector<int>& mixes = config.nmax;

    for (std::size_t m = 0; m < mixes.size(); ++m)
    {
        std::printf("\nnmax %d\n", mixes[m]);
        std::printf("%-14s %12s %12s %12s %10s %10s\n",
                    "option",
                    "min",
                    "median",
                    "max",
                    "spread",
                    "wall_ms");

        for (std::size_t o = 0; o < kOptionCount; ++o)
        {
            const Summary& summary = summaries[m * kOptionCount + o];

            if (summary.cycles.empty())
            {
                std::printf("%-14s %12s\n", kOptionNames[o], "no valid pair");
                continue;
            }

            std::vector<std::uint64_t> sorted = summary.cycles;
            std::sort(sorted.begin(), sorted.end());
            const std::uint64_t least = sorted.front();
            const std::uint64_t most = sorted.back();
            const std::uint64_t median = sorted[sorted.size() / 2];
            const auto minmaxWall = std::minmax_element(summary.walls.begin(), summary.walls.end());
            const double spread =
                100.0 * static_cast<double>(most - least) / static_cast<double>(least);

            std::printf("%-14s %12llu %12llu %12llu %9.3f%% %10.4f\n",
                        kOptionNames[o],
                        static_cast<unsigned long long>(least),
                        static_cast<unsigned long long>(median),
                        static_cast<unsigned long long>(most),
                        spread,
                        *minmaxWall.first);

            if (csv != nullptr)
            {
                std::fprintf(csv,
                             "# summary nmax %d %-14s cycles_per_pass_min %llu median %llu max "
                             "%llu spread %.3f%% wall_ms_min %.4f\n",
                             mixes[m],
                             kOptionNames[o],
                             static_cast<unsigned long long>(least),
                             static_cast<unsigned long long>(median),
                             static_cast<unsigned long long>(most),
                             spread,
                             *minmaxWall.first);
            }
        }
    }

    if (canaryFloorMs > 1e29)
    {
        std::printf("\ncanary floor | no admitted pair, so no reading was taken\n");
    } else
    {
        std::printf("\ncanary floor | %.4f ms | the fixed-work spin's best run this session\n",
                    canaryFloorMs);
    }

    std::printf("resolution | the spread above is the instrument's; two options are separated only "
                "by more than the wider of the two\n");
}

} // namespace

int main(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg.rfind("--exe=", 0) == 0)
        {
            config.exe = arg.substr(6);
        } else if (arg.rfind("--csv=", 0) == 0)
        {
            config.csv = arg.substr(6);
        } else if (arg.rfind("--rounds=", 0) == 0)
        {
            config.rounds = static_cast<std::size_t>(std::strtoull(arg.c_str() + 9, nullptr, 10));
        } else if (arg.rfind("--reps-low=", 0) == 0)
        {
            config.repsLow = static_cast<std::size_t>(std::strtoull(arg.c_str() + 11, nullptr, 10));
        } else if (arg.rfind("--reps-high=", 0) == 0)
        {
            config.repsHigh =
                static_cast<std::size_t>(std::strtoull(arg.c_str() + 12, nullptr, 10));
        } else if (arg.rfind("--count=", 0) == 0)
        {
            config.count = static_cast<std::size_t>(std::strtoull(arg.c_str() + 8, nullptr, 10));
        } else if (arg.rfind("--seed=", 0) == 0)
        {
            config.seed = static_cast<std::size_t>(std::strtoull(arg.c_str() + 7, nullptr, 10));
        } else if (arg.rfind("--xrange=", 0) == 0)
        {
            config.xrange = arg.substr(9);
        } else if (arg.rfind("--nmax=", 0) == 0)
        {
            config.nmax = ParseNmax(arg.substr(7));
        } else if (arg == "--clock-check")
        {
            config.clockCheck = true;
        } else
        {
            std::fprintf(
                stderr, "qcx-bench-boys-option-cost: unknown argument '%s'\n", arg.c_str());
            return 2;
        }
    }

    if (config.clockCheck)
    {
        return RunClockCheck();
    }

    if (config.exe.empty())
    {
        Usage();
        return 2;
    }

    // The child's name is resolved here rather than left to the launch: this
    // probe takes the driver's path from its own caller, who need not have
    // written it the way CreateProcess reads a relative one.
    {
        char resolved[MAX_PATH];
        const DWORD length = GetFullPathNameA(config.exe.c_str(), MAX_PATH, resolved, nullptr);

        if (length == 0 || length >= MAX_PATH)
        {
            std::fprintf(stderr,
                         "qcx-bench-boys-option-cost: cannot resolve '%s' to a path\n",
                         config.exe.c_str());
            return 2;
        }

        config.exe.assign(resolved, length);
    }

    if (GetFileAttributesA(config.exe.c_str()) == INVALID_FILE_ATTRIBUTES)
    {
        std::fprintf(
            stderr, "qcx-bench-boys-option-cost: no such file: '%s'\n", config.exe.c_str());
        return 2;
    }

    if (config.repsHigh <= config.repsLow)
    {
        std::fprintf(stderr,
                     "qcx-bench-boys-option-cost: --reps-high must exceed --reps-low; the "
                     "per-pass figure is their difference\n");
        return 2;
    }

    FILE* csv = nullptr;

    if (fopen_s(&csv, config.csv.c_str(), "w") != 0 || csv == nullptr)
    {
        std::fprintf(stderr, "qcx-bench-boys-option-cost: cannot write '%s'\n", config.csv.c_str());
        return 2;
    }

    std::fprintf(
        csv,
        "# pair | option | nmax | round | cycles_per_pass | wall_per_pass_ms | cycles_low | "
        "cycles_high | wall_low_ms | wall_high_ms | thread_cycles_low | "
        "thread_cycles_high | canary_before_ms | canary_after_ms\n");
    std::fprintf(csv,
                 "# exe | %s | rounds | %zu | reps | %zu..%zu | count | %zu | seed | %zu | "
                 "xrange | %s\n",
                 config.exe.c_str(),
                 config.rounds,
                 config.repsLow,
                 config.repsHigh,
                 config.count,
                 config.seed,
                 config.xrange.c_str());

    std::vector<Summary> summaries(config.nmax.size() * kOptionCount);
    double canaryFloorMs = 1e30;

    for (std::size_t round = 0; round < config.rounds; ++round)
    {
        // The option order reverses between rounds, so a drift that is
        // monotone in time lands on every option equally.
        const bool forward = (round % 2) == 0;

        for (std::size_t m = 0; m < config.nmax.size(); ++m)
        {
            const int nmax = config.nmax[m];

            for (std::size_t step = 0; step < kOptionCount; ++step)
            {
                const std::size_t o = forward ? step : (kOptionCount - 1 - step);
                const Pair pair = MeasurePair(config, kOptionNames[o], nmax);

                if (!pair.valid)
                {
                    std::fprintf(stderr,
                                 "qcx-bench-boys-option-cost: %s at nmax %d, round %zu: no valid "
                                 "pair (exit %lu/%lu)\n",
                                 kOptionNames[o],
                                 nmax,
                                 round,
                                 pair.exitLow,
                                 pair.exitHigh);
                    continue;
                }

                canaryFloorMs =
                    std::min(canaryFloorMs, std::min(pair.canaryBeforeMs, pair.canaryAfterMs));
                Summary& summary = summaries[m * kOptionCount + o];
                summary.cycles.push_back(pair.cyclesPerPass);
                summary.walls.push_back(pair.wallPerPassMs);
                summary.canaryPeakPercent =
                    std::max(summary.canaryPeakPercent,
                             100.0 * (pair.canaryAfterMs - canaryFloorMs) / canaryFloorMs);

                std::fprintf(csv,
                             "pair | %-14s | %d | %zu | %llu | %.6f | %llu | %llu | %.4f | "
                             "%.4f | %llu | %llu | %.4f | %.4f\n",
                             kOptionNames[o],
                             nmax,
                             round,
                             static_cast<unsigned long long>(pair.cyclesPerPass),
                             pair.wallPerPassMs,
                             static_cast<unsigned long long>(pair.cyclesLow),
                             static_cast<unsigned long long>(pair.cyclesHigh),
                             pair.wallLowMs,
                             pair.wallHighMs,
                             static_cast<unsigned long long>(pair.threadCyclesLow),
                             static_cast<unsigned long long>(pair.threadCyclesHigh),
                             pair.canaryBeforeMs,
                             pair.canaryAfterMs);
                std::fflush(csv);
            }
        }
    }

    PrintSummary(config, summaries, canaryFloorMs, csv);
    std::fclose(csv);

    std::printf("\nrows written to %s\n", config.csv.c_str());
    return 0;
}
