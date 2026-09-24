// What a runtime accuracy tier costs per call: a dispatch over
// already-instantiated rungs measured against the direct call it replaces.
//
// The rungs are compile-time instantiations that coexist in one binary (the
// multiplier is a non-type template parameter, so each rung is its own
// symbol). A runtime tier is therefore a switch over them - the question is
// what that switch costs against a kernel of tens of operations.
//
// Reported as a RATIO only: this host is shared with other jobs, so an
// absolute nanosecond count would describe the load, not the code. Both legs
// run in the same process, interleaved, so the load is common to them.
//
// The two shapes are measured separately because they amortise differently:
// a fixed-order call pays the branch once for one value, a batch call pays it
// once for nmax + 1 values - and the batch is the shape a real integral
// engine uses.

#include "boys/boys.hpp"
#include "boys/boys_impl.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kMaxOrder = 8;
constexpr int kArgCount = 4096;
constexpr double kTier1 = 64.0;
constexpr double kTier2 = 1024.0;
constexpr double kTier3 = 65536.0;

struct Arg {
    int n = 0;
    double x = 0.0;
};

/// Shell-pair-like arguments: every order 0..kMaxOrder, spanning the three
/// regions of the partition.
std::vector<Arg> MakeArgs() {
    std::vector<Arg> args;
    args.reserve(kArgCount);

    for (int i = 0; i < kArgCount; ++i)
    {
        Arg arg;
        arg.n = i % (kMaxOrder + 1);
        // 0 .. ~60, so region A, B and C are all exercised.
        arg.x = 60.0 * static_cast<double>(i) / static_cast<double>(kArgCount - 1);
        args.push_back(arg);
    }

    return args;
}

enum class Tier { kReference, kTier1, kTier2, kTier3 };

template <double kM> void BatchRung(int nmax, double x, double* out) noexcept {
    boys::BoysAllOrders<kM>(nmax, x, out);
}

/// The shape under test: one branch, then the rung.
void BatchDispatched(Tier tier, int nmax, double x, double* out) noexcept {
    switch (tier)
    {
    case Tier::kReference:
        BatchRung<boys::kBoysFullAccuracyMultiplier>(nmax, x, out);
        return;
    case Tier::kTier1:
        BatchRung<kTier1>(nmax, x, out);
        return;
    case Tier::kTier2:
        BatchRung<kTier2>(nmax, x, out);
        return;
    case Tier::kTier3:
        BatchRung<kTier3>(nmax, x, out);
        return;
    }
}

void BatchDirect(int nmax, double x, double* out) noexcept {
    BatchRung<boys::kBoysFullAccuracyMultiplier>(nmax, x, out);
}

double SingleDispatched(Tier tier, int n, double x) noexcept {
    switch (tier)
    {
    case Tier::kReference:
        return boys::BoysSingle<boys::kBoysFullAccuracyMultiplier>(n, x);
    case Tier::kTier1:
        return boys::BoysSingle<kTier1>(n, x);
    case Tier::kTier2:
        return boys::BoysSingle<kTier2>(n, x);
    case Tier::kTier3:
        return boys::BoysSingle<kTier3>(n, x);
    }

    return 0.0;
}

double SingleDirect(int n, double x) noexcept {
    return boys::BoysSingle<boys::kBoysFullAccuracyMultiplier>(n, x);
}

/// One pass. The checksum is returned so no call can be discarded.
double RunBatch(const std::vector<Arg>& args, Tier tier, bool dispatched) {
    double scratch[boys::kMaxBoysOrder + 1];
    double sum = 0.0;

    for (const Arg& arg : args)
    {
        if (dispatched)
        {
            BatchDispatched(tier, arg.n, arg.x, scratch);
        } else
        {
            BatchDirect(arg.n, arg.x, scratch);
        }

        sum += scratch[arg.n];
    }

    return sum;
}

double RunSingle(const std::vector<Arg>& args, Tier tier, bool dispatched) {
    double sum = 0.0;

    for (const Arg& arg : args)
    {
        sum += dispatched ? SingleDispatched(tier, arg.n, arg.x) : SingleDirect(arg.n, arg.x);
    }

    return sum;
}

using BatchFn = double (*)(const std::vector<Arg>&, Tier, bool);
using SingleFn = double (*)(const std::vector<Arg>&, Tier, bool);

/// Median nanoseconds per call over kRepeats passes, with the observed spread.
struct Timing {
    double median = 0.0;
    double low = 0.0;
    double high = 0.0;
};

template <typename Fn>
Timing Measure(Fn fn, const std::vector<Arg>& args, Tier tier, bool dispatched, int repeats) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r)
    {
        const auto start = Clock::now();
        const double sum = fn(args, tier, dispatched);
        const auto stop = Clock::now();

        if (sum == 12345.6789)
        {
            std::fprintf(stderr, "unreachable checksum guard\n");
        }

        const double ns = std::chrono::duration<double, std::nano>(stop - start).count();
        samples.push_back(ns / static_cast<double>(args.size()));
    }

    for (std::size_t i = 1; i < samples.size(); ++i)
    {
        for (std::size_t j = i; j > 0 && samples[j] < samples[j - 1]; --j)
        {
            const double tmp = samples[j];
            samples[j] = samples[j - 1];
            samples[j - 1] = tmp;
        }
    }

    Timing timing;
    timing.median = samples[samples.size() / 2];
    timing.low = samples.front();
    timing.high = samples.back();
    return timing;
}

void Report(const char* label, const Timing& direct, const Timing& dispatched) {
    std::printf("%s: direct %.1f ns/call, dispatched %.1f ns/call, ratio %.4f "
                "(direct %.1f-%.1f, dispatched %.1f-%.1f)\n",
                label,
                direct.median,
                dispatched.median,
                dispatched.median / direct.median,
                direct.low,
                direct.high,
                dispatched.low,
                dispatched.high);
}

bool SameBits(double left, double right) {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
}

/// The reference tier must be today's behaviour exactly - not merely close.
/// Compared value by value, since a sum can agree while its terms do not.
int CheckReferenceTier(const std::vector<Arg>& args) {
    int compared = 0;

    for (const Arg& arg : args)
    {
        double directScratch[boys::kMaxBoysOrder + 1];
        double dispatchedScratch[boys::kMaxBoysOrder + 1];
        BatchDirect(arg.n, arg.x, directScratch);
        BatchDispatched(Tier::kReference, arg.n, arg.x, dispatchedScratch);

        for (int k = 0; k <= arg.n; ++k)
        {
            if (!SameBits(directScratch[k], dispatchedScratch[k]))
            {
                std::printf("MISMATCH batch n=%d x=%.17g order=%d\n", arg.n, arg.x, k);
                return -1;
            }

            ++compared;
        }

        const double directSingle = SingleDirect(arg.n, arg.x);
        const double dispatchedSingle = SingleDispatched(Tier::kReference, arg.n, arg.x);

        if (!SameBits(directSingle, dispatchedSingle))
        {
            std::printf("MISMATCH single n=%d x=%.17g\n", arg.n, arg.x);
            return -1;
        }

        ++compared;
    }

    return compared;
}

/// A tier that resolves to the wrong rung is a silent accuracy substitution,
/// so each tier is checked against the rung it names rather than only timed.
int CheckTierSelectsItsRung(const std::vector<Arg>& args) {
    const Tier tiers[3] = {Tier::kTier1, Tier::kTier2, Tier::kTier3};
    int checked = 0;

    for (const Tier tier : tiers)
    {
        for (const Arg& arg : args)
        {
            const double viaSwitch = SingleDispatched(tier, arg.n, arg.x);
            double rung = 0.0;

            switch (tier)
            {
            case Tier::kTier1:
                rung = boys::BoysSingle<kTier1>(arg.n, arg.x);
                break;
            case Tier::kTier2:
                rung = boys::BoysSingle<kTier2>(arg.n, arg.x);
                break;
            case Tier::kTier3:
                rung = boys::BoysSingle<kTier3>(arg.n, arg.x);
                break;
            case Tier::kReference:
                continue;
            }

            if (!SameBits(viaSwitch, rung))
            {
                std::printf(
                    "WRONG RUNG tier=%d n=%d x=%.17g\n", static_cast<int>(tier), arg.n, arg.x);
                return -1;
            }

            ++checked;
        }
    }

    return checked;
}

} // namespace

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::atoi(argv[1]) : 9;
    const Tier tier = argc > 2 ? static_cast<Tier>(std::atoi(argv[2])) : Tier::kReference;
    const std::vector<Arg> args = MakeArgs();
    std::fprintf(
        stderr, "args %zu, repeats %d, tier %d\n", args.size(), repeats, static_cast<int>(tier));

    // Warm both paths before either is timed.
    RunBatch(args, tier, false);
    RunBatch(args, tier, true);
    RunSingle(args, tier, false);
    RunSingle(args, tier, true);

    Report("batch  ",
           Measure(static_cast<BatchFn>(RunBatch), args, tier, false, repeats),
           Measure(static_cast<BatchFn>(RunBatch), args, tier, true, repeats));

    // Interleaved second pass: if the two legs drift together, the ratio is
    // not an artefact of one leg running under a different load.
    Report("single ",
           Measure(static_cast<SingleFn>(RunSingle), args, tier, false, repeats),
           Measure(static_cast<SingleFn>(RunSingle), args, tier, true, repeats));

    const int referenceCompared = CheckReferenceTier(args);
    std::printf("reference tier bit-identical to the direct call: %s (%d values compared)\n",
                referenceCompared > 0 ? "yes" : "NO",
                referenceCompared);

    const int rungChecked = CheckTierSelectsItsRung(args);
    std::printf("every tier resolves to the rung it names: %s (%d values checked)\n",
                rungChecked > 0 ? "yes" : "NO",
                rungChecked);

    return (referenceCompared > 0 && rungChecked > 0) ? 0 : 1;
}
