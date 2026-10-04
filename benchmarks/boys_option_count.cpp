// The option-count probe: what the fp64 evaluation options cost, counted
// rather than timed.
//
// The option probe asks this question against the clock, over this same
// workload, and refuses to answer it: the shapes differ by a percent or two,
// while the leading one's own spread across its admitted passes is wider than
// that, because a machine's clock state differs from pass to pass. An idle
// machine wanders more than a loaded one rather than less, so more passes make
// the resolution worse. A stopwatch cannot separate these options on such a
// machine.
//
// A count can. Options that return the same values over the same arguments do
// the same work, so the one retiring fewer micro-operations is the cheaper
// one, and that number is a property of the code and the CPU rather than of
// the clock or of what else the machine is running.
//
// This driver therefore reports the work a run did - entry calls, returned
// values, and the checksum that proves the values came back - and no time at
// all. The counters are the platform's, read from outside the process:
//
//     perf stat -e instructions:u,uops_retired.retire_slots:u
//         qcx-bench-boys-option-count --option=tagged-fp64 --reps=64
//
// One option per run, because a process's own count carries its start-up, its
// accuracy pass and its workload construction. The per-pass figure is a
// two-point subtraction over two differing repetition counts:
//
//     (C(R2) - C(R1)) / (R2 - R1)   is the work one pass retires
//
// Read both counters, and rank by retired slots. A microcoded instruction is
// one instruction and many retirement slots, so an instruction count can put
// the slower of two shapes first: the library's own record has the gathered
// coefficient fetch retiring the fewest instructions and the most slots of the
// three routes it compares, and losing on the counter that decides.
//
// The workload is the option probe's: --count arguments, x log-uniform over
// --xrange, each argument's order the sum of two shell angular momenta drawn
// uniformly over 0..--nmax/2 and capped at --nmax. The options are its fp64
// call shapes:
//
//     batch-fp64    BoysAllOrders per argument
//     grouped-fp64  BoysAllN per order run, on the caller's workspace
//     tagged-fp64   the same call over BoysSortedArgs
//
// Every run verifies its own values against the certified per-order entry
// before it counts anything, at the fp64 budget, and prints the worst
// deviation: a ranking means something only inside one accuracy class, so the
// class is measured here rather than assumed. --self-check stops after the
// verification, and its exit status carries the verdict.
//
// Usage:
//   qcx-bench-boys-option-count --list
//   qcx-bench-boys-option-count --option=NAME [--reps=N] [--count=N]
//                               [--nmax=N] [--xrange=LO,HI] [--seed=N]
//                               [--self-check]
#include "qcx/integrals/boys.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using qcx::integrals::BoysAllN;
using qcx::integrals::BoysAllNWorkspaceSize;
using qcx::integrals::BoysAllOrders;
using qcx::integrals::BoysSingle;
using qcx::integrals::BoysSortedArgs;
using qcx::integrals::kMaxBoysOrder;

/// The fp64 budget the library documents for its loosest region: the class
/// every option this driver ranks sits in, and the bar the accuracy pass
/// checks each one against.
constexpr double kFp64Budget = 5.5e-14;

/// The fp64 call shapes, under the names the option probe gives them.
enum class OptionKind {
    kBatchFp64,
    kGroupedFp64,
    kTaggedFp64,
};

struct Option {
    const char* name;
    OptionKind kind;
    const char* shape;
};

constexpr std::array<Option, 3> kOptions{{
    {"batch-fp64", OptionKind::kBatchFp64, "BoysAllOrders(n, x, out) per argument"},
    {"grouped-fp64",
     OptionKind::kGroupedFp64,
     "BoysAllN(n, x, out, count, workspace) per order run"},
    {"tagged-fp64",
     OptionKind::kTaggedFp64,
     "BoysAllN(n, x, out, count, BoysSortedArgs) per order run"},
}};

struct Config {
    std::string option;
    std::size_t count = 16384;
    int nmax = kMaxBoysOrder;
    double xLo = 1e-3;
    double xHi = 40.0;
    std::size_t seed = 47;
    std::size_t reps = 1;
    bool selfCheck = false;
    bool list = false;
};

const Option* FindOption(const std::string& name) {
    for (const Option& option : kOptions)
    {
        if (name == option.name)
        {
            return &option;
        }
    }

    return nullptr;
}

/// The arguments, their own orders, and the runs the grouped entries take,
/// built as the option probe builds them so the two measure one workload.
struct Workload {
    /// One argument per entry.
    std::vector<double> x;

    /// Each argument's own highest order, 0..nmax.
    std::vector<int> order;

    /// Argument indices by (order, x): each run of equal order is a batch one
    /// all-N call can take, and each run is non-decreasing in x, which is what
    /// makes the sorted overload's declaration true of it.
    std::vector<std::size_t> sorted;

    /// Run boundaries into `sorted`; one more entry than there are runs.
    std::vector<std::size_t> runBegin;

    /// Each run's order.
    std::vector<int> runOrder;

    /// Arguments in the largest run.
    std::size_t largestRun = 0;

    /// Values one pass returns: the orders each argument asks for, over the
    /// whole array. The grouped shapes return the same count, run by run.
    std::size_t valuesPerPass = 0;
};

Workload BuildWorkload(const Config& config) {
    Workload work;
    work.x.resize(config.count);
    work.order.resize(config.count);

    std::mt19937_64 generator(config.seed);
    std::uniform_real_distribution<double> logX(std::log10(config.xLo), std::log10(config.xHi));
    std::uniform_int_distribution<int> shell(0, config.nmax / 2);

    for (std::size_t i = 0; i < config.count; ++i)
    {
        work.x[i] = std::pow(10.0, logX(generator));

        // The order a shell pair presents: the sum of two shell angular
        // momenta, which is what an integral engine's batches are grouped by.
        const int first = shell(generator);
        const int second = shell(generator);
        work.order[i] = std::min(config.nmax, first + second);
    }

    work.sorted.resize(config.count);
    std::iota(work.sorted.begin(), work.sorted.end(), std::size_t{0});
    std::sort(work.sorted.begin(), work.sorted.end(), [&work](std::size_t a, std::size_t b) {
        if (work.order[a] != work.order[b])
        {
            return work.order[a] < work.order[b];
        }

        return work.x[a] < work.x[b];
    });

    work.runBegin.push_back(0);

    for (std::size_t i = 1; i < config.count; ++i)
    {
        if (work.order[work.sorted[i]] != work.order[work.sorted[i - 1]])
        {
            work.runBegin.push_back(i);
            work.runOrder.push_back(work.order[work.sorted[i - 1]]);
        }
    }

    work.runOrder.push_back(work.order[work.sorted[config.count - 1]]);
    work.runBegin.push_back(config.count);

    for (std::size_t r = 0; r + 1 < work.runBegin.size(); ++r)
    {
        work.largestRun = std::max(work.largestRun, work.runBegin[r + 1] - work.runBegin[r]);
    }

    for (const int order : work.order)
    {
        work.valuesPerPass += static_cast<std::size_t>(order) + 1;
    }

    return work;
}

/// The scratch the grouped shapes need, allocated once so no pass pays for an
/// allocation.
struct Buffers {
    std::vector<double> runX;
    std::vector<double> runOut;
    std::vector<std::size_t> workspace;
};

Buffers MakeBuffers(const Workload& work) {
    Buffers buffers;
    const std::size_t run = std::max<std::size_t>(work.largestRun, 1);

    buffers.runX.resize(run);
    buffers.runOut.resize(run * static_cast<std::size_t>(kMaxBoysOrder + 1));
    buffers.workspace.resize(BoysAllNWorkspaceSize(run));
    return buffers;
}

/// Runs one option's own calls over the workload and hands every value it
/// returns to `visit(x, order, value)`.
///
/// The single body shared by the accuracy pass and the counted pass: a second
/// implementation of an option would be a second chance to count something
/// other than what was verified.
template <typename Visit>
void VisitValues(const Workload& work, const Option& option, Buffers& buffers, Visit&& visit) {
    if (option.kind == OptionKind::kBatchFp64)
    {
        double values[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];
            BoysAllOrders(n, work.x[i], values);

            for (int k = 0; k <= n; ++k)
            {
                visit(work.x[i], k, values[k]);
            }
        }

        return;
    }

    for (std::size_t r = 0; r + 1 < work.runBegin.size(); ++r)
    {
        const std::size_t from = work.runBegin[r];
        const std::size_t to = work.runBegin[r + 1];
        const std::size_t run = to - from;
        const int n = work.runOrder[r];

        for (std::size_t j = 0; j < run; ++j)
        {
            buffers.runX[j] = work.x[work.sorted[from + j]];
        }

        if (option.kind == OptionKind::kGroupedFp64)
        {
            // The allocation-free form the entry documents for a hot loop: the
            // caller's own workspace, so a pass measures the grouping and not
            // the allocator.
            BoysAllN(n, buffers.runX.data(), buffers.runOut.data(), run, buffers.workspace.data());
        } else
        {
            // Every run is non-decreasing in x by construction, which is
            // exactly what this overload declares.
            BoysAllN(n, buffers.runX.data(), buffers.runOut.data(), run, BoysSortedArgs{});
        }

        for (std::size_t j = 0; j < run; ++j)
        {
            for (int k = 0; k <= n; ++k)
            {
                visit(buffers.runX[j], k, buffers.runOut[static_cast<std::size_t>(k) * run + j]);
            }
        }
    }
}

/// The worst difference from the certified per-order entry, over the whole
/// workload, and whether the two ever differed at all.
///
/// The reference is `BoysSingle`, walked upward per argument and memoized: it
/// is the only anchor that is the same value for every option, because a batch
/// entry seeds region A at the batch's own highest order, so its F_k below it
/// is a recurrence's value rather than a per-order walk's.
class AccuracyWalker {
public:
    explicit AccuracyWalker(double budget) : _budget(budget) {}

    void operator()(double x, int k, double value) {
        const double expected = Reference(x, k);
        const double difference = std::fabs(value - expected);

        _maxError = std::max(_maxError, difference);
        _bitIdentical = _bitIdentical && (value == expected);
    }

    /// \returns the largest absolute deviation from the reference
    double MaxError() const {
        return _maxError;
    }

    /// \returns whether every value compared equal to the reference
    bool BitIdentical() const {
        return _bitIdentical;
    }

    /// \returns whether the worst deviation is inside the accuracy budget
    bool Inside() const {
        return _maxError <= _budget;
    }

private:
    double Reference(double x, int k) {
        if (x != _lastX)
        {
            _lastX = x;
            _filled = 0;
        }

        if (k >= _filled)
        {
            for (int j = _filled; j <= k; ++j)
            {
                _reference[j] = BoysSingle(j, x);
            }

            _filled = k + 1;
        } else
        {
            _reference[k] = BoysSingle(k, x);
        }

        return _reference[k];
    }

    double _budget = 0.0;
    double _reference[kMaxBoysOrder + 1] = {};
    double _lastX = -1.0; // arguments are >= 0, so this is never a match
    int _filled = 0;
    double _maxError = 0.0;
    bool _bitIdentical = true;
};

/// One pass, returning the checksum of every value the option returned: one
/// addition per value, so the compiler cannot drop work whose result nothing
/// reads, and the same addition for every option.
double CountPass(const Workload& work, const Option& option, Buffers& buffers) {
    double sum = 0.0;
    VisitValues(work, option, buffers, [&sum](double, int, double value) { sum += value; });
    return sum;
}

std::size_t RunCount(const Workload& work) {
    return work.runBegin.size() - 1;
}

void Usage() {
    std::fputs("qcx-bench-boys-option-count - the fp64 options' work, counted\n"
               "\n"
               "usage: qcx-bench-boys-option-count --option=NAME [options]\n"
               "\n"
               "  --option=NAME      batch-fp64 | grouped-fp64 | tagged-fp64\n"
               "  --count=N          arguments per pass (default 16384)\n"
               "  --nmax=N           highest order any argument carries (default 32)\n"
               "  --xrange=LO,HI     log-uniform argument range (default 1e-3,40)\n"
               "  --seed=N           workload generator seed (default 47)\n"
               "  --reps=N           passes counted (default 1)\n"
               "  --self-check       verify the values and stop\n"
               "  --list             name the options and exit\n"
               "\n"
               "The count is the platform's, read from outside the process; this\n"
               "driver prints the work a run did and no time. One option per run,\n"
               "and the per-pass figure is the difference of two runs at two\n"
               "repetition counts. See the file preamble for the command.\n",
               stdout);
}

} // namespace

int main(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--list")
        {
            config.list = true;
        } else if (arg == "--self-check")
        {
            config.selfCheck = true;
        } else if (arg.rfind("--option=", 0) == 0)
        {
            config.option = arg.substr(9);
        } else if (arg.rfind("--count=", 0) == 0)
        {
            config.count = static_cast<std::size_t>(std::strtoull(arg.c_str() + 8, nullptr, 10));
        } else if (arg.rfind("--nmax=", 0) == 0)
        {
            config.nmax = std::atoi(arg.c_str() + 7);
        } else if (arg.rfind("--seed=", 0) == 0)
        {
            config.seed = static_cast<std::size_t>(std::strtoull(arg.c_str() + 7, nullptr, 10));
        } else if (arg.rfind("--reps=", 0) == 0)
        {
            config.reps = static_cast<std::size_t>(std::strtoull(arg.c_str() + 7, nullptr, 10));
        } else if (arg.rfind("--xrange=", 0) == 0)
        {
            config.xLo = std::strtod(arg.c_str() + 9, nullptr);
            const char* comma = std::strchr(arg.c_str() + 9, ',');

            if (comma != nullptr)
            {
                config.xHi = std::strtod(comma + 1, nullptr);
            } else
            {
                config.xHi = config.xLo;
            }
        } else
        {
            std::fprintf(
                stderr, "qcx-bench-boys-option-count: unknown argument '%s'\n", arg.c_str());
            return 2;
        }
    }

    if (config.list)
    {
        for (const Option& option : kOptions)
        {
            std::printf("%-14s %s\n", option.name, option.shape);
        }

        return 0;
    }

    if (config.option.empty())
    {
        Usage();
        return 2;
    }

    const Option* option = FindOption(config.option);

    if (option == nullptr)
    {
        std::fprintf(stderr,
                     "qcx-bench-boys-option-count: no option named '%s' (--list shows them)\n",
                     config.option.c_str());
        return 2;
    }

    if (config.nmax < 0 || config.nmax > kMaxBoysOrder)
    {
        std::fprintf(stderr, "qcx-bench-boys-option-count: nmax must be 0..%d\n", kMaxBoysOrder);
        return 2;
    }

    const Workload work = BuildWorkload(config);
    Buffers buffers = MakeBuffers(work);

    // The accuracy pass, always and before anything is counted: what a count
    // of the wrong values would mean is nothing, and the class an option sits
    // in is what makes a ranking across options legitimate.
    AccuracyWalker walker(kFp64Budget);
    VisitValues(work, *option, buffers, walker);

    std::printf(
        "qcx-bench-boys-option-count | option %s | shape %s\n", option->name, option->shape);
    std::printf("workload | count %zu | nmax %d | x log-uniform %g..%g | seed %zu | avx2 %d\n",
                config.count,
                config.nmax,
                config.xLo,
                config.xHi,
                config.seed,
                qcx::integrals::BoysAvx2Available() ? 1 : 0);
    std::printf("accuracy | worst absolute %.6e | budget %.6e | inside %d | bit identical %d | "
                "reference BoysSingle per (x, order)\n",
                walker.MaxError(),
                kFp64Budget,
                walker.Inside() ? 1 : 0,
                walker.BitIdentical() ? 1 : 0);

    if (!walker.Inside())
    {
        std::fprintf(stderr,
                     "qcx-bench-boys-option-count: %s is outside the fp64 budget on this "
                     "workload; no count of it means anything\n",
                     option->name);
        return 1;
    }

    if (config.selfCheck)
    {
        return 0;
    }

    if (config.reps == 0)
    {
        config.reps = 1;
    }

    // Warm-up: one pass, so the tables and the workspace pages are resident
    // before the first counted pass. It is a fixed cost of the process rather
    // than a per-pass one, and the two-point subtraction is what removes it.
    (void)CountPass(work, *option, buffers);

    double checksum = 0.0;

    for (std::size_t rep = 0; rep < config.reps; ++rep)
    {
        checksum += CountPass(work, *option, buffers);
    }

    const std::size_t callsPerPass =
        (option->kind == OptionKind::kBatchFp64) ? config.count : RunCount(work);

    std::printf("work | reps %zu | entry calls %zu | entry calls per pass %zu | "
                "values %zu | values per pass %zu | checksum %.6e\n",
                config.reps,
                config.reps * callsPerPass,
                callsPerPass,
                config.reps * work.valuesPerPass,
                work.valuesPerPass,
                checksum);
    std::printf("note | the counters are the platform's and are read from outside this process; "
                "a count is a property of this build and this CPU, and no time is reported\n");
    return 0;
}
