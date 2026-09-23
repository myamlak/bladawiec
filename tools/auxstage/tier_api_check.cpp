// Validates the run-time accuracy tier API against the rungs it names, and
// measures what selecting a tier per call costs against calling the rung
// directly. Ratios only: the machine is shared.

#include "boys/boys.hpp"
#include "boys_impl.hpp" // the rungs themselves, for the comparisons

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

struct Arg {
    int n;
    double x;
};

std::vector<Arg> MakeArgs() {
    std::vector<Arg> args;
    args.reserve(4096);

    // nmax sweeps 0..8 and x sweeps the whole argument range, so all three
    // regions are exercised.
    for (int i = 0; i < 4096; ++i)
    {
        args.push_back({i % 9, 60.0 * i / 4095.0});
    }

    return args;
}

bool SameBits(double left, double right) {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
}

/// A tier that resolves to the wrong rung is a silent accuracy substitution,
/// so each tier is compared against the rung it names, value by value.
template <double kM>
int CheckTierAgainstRung(const std::vector<Arg>& args, boys::AccuracyTier tier, const char* name) {
    double got[boys::kMaxBoysOrder + 1];
    double want[boys::kMaxBoysOrder + 1];
    int bad = 0;
    int checked = 0;

    for (const Arg& arg : args)
    {
        boys::BoysBatchAtTier(tier, arg.n, arg.x, got);
        boys::BoysBatch<kM>(arg.n, arg.x, want);

        for (int k = 0; k <= arg.n; ++k)
        {
            ++checked;

            if (!SameBits(got[k], want[k]))
            {
                if (bad == 0)
                {
                    std::printf("  first mismatch n=%d x=%.17g k=%d: %.17g vs %.17g\n",
                                arg.n,
                                arg.x,
                                k,
                                got[k],
                                want[k]);
                }

                ++bad;
            }
        }
    }

    std::printf("  %-10s m=%-8.0f %d of %d values differ from its rung\n",
                name,
                boys::AccuracyMultiplier(tier),
                bad,
                checked);

    return bad;
}

/// The reference tier must be today's behaviour exactly, not merely close.
int CheckReferenceIsToday(const std::vector<Arg>& args) {
    return CheckTierAgainstRung<boys::kBoysFullAccuracyMultiplier>(
        args, boys::AccuracyTier::kReference, "reference");
}

void ShowCoverage(const char* label, boys::AccuracyTier tier) {
    const boys::AccuracyRegion regions[3] = {
        boys::AccuracyRegion::kA, boys::AccuracyRegion::kB, boys::AccuracyRegion::kC};
    const char* names[3] = {"A", "B", "C"};
    const char* components[3] = {"region A seed fit", "region B F0 fit", "region C asymptotic"};

    for (int r = 0; r < 3; ++r)
    {
        const boys::TierCoverage coverage = boys::QueryTier(tier, regions[r], 1e-14);

        std::printf("  %-9s region %s: %s, reachable %.3e, limited by %s\n",
                    label,
                    names[r],
                    coverage.meets ? "meets 1e-14" : "cannot meet 1e-14",
                    coverage.reachable,
                    components[static_cast<int>(coverage.limiting)]);
    }
}

/// Median of several repeats, as a ratio against the direct call.
template <typename Call> double TimeRatio(Call call, const std::vector<Arg>& args, int repeats) {
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));

    for (int r = 0; r < repeats; ++r)
    {
        const auto start = std::chrono::steady_clock::now();

        call(args);

        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
    }

    for (std::size_t i = 1; i < samples.size(); ++i)
    {
        for (std::size_t j = i; j > 0 && samples[j] < samples[j - 1]; --j)
        {
            const double swap = samples[j];
            samples[j] = samples[j - 1];
            samples[j - 1] = swap;
        }
    }

    return samples[samples.size() / 2];
}

void TimeTiers(const std::vector<Arg>& args) {
    double out[boys::kMaxBoysOrder + 1];
    const int repeats = 7;

    const double direct = TimeRatio(
        [&](const std::vector<Arg>& as) {
            for (const Arg& arg : as)
            {
                boys::BoysBatch<boys::kBoysFullAccuracyMultiplier>(arg.n, arg.x, out);
            }
        },
        args,
        repeats);

    struct Named {
        const char* name;
        boys::AccuracyTier tier;
    };

    const Named named[7] = {{"reference", boys::AccuracyTier::kReference},
                            {"m=64", boys::AccuracyTier::kRelaxed64},
                            {"m=256", boys::AccuracyTier::kRelaxed256},
                            {"m=1024", boys::AccuracyTier::kRelaxed1024},
                            {"m=4096", boys::AccuracyTier::kRelaxed4096},
                            {"m=16384", boys::AccuracyTier::kRelaxed16384},
                            {"m=65536", boys::AccuracyTier::kRelaxed65536}};

    std::printf("  direct call median %.0f ns over %d repeats, %zu arguments\n",
                direct,
                repeats,
                args.size());

    for (const Named& entry : named)
    {
        const boys::AccuracyTier tier = entry.tier;
        const double atTier = TimeRatio(
            [&](const std::vector<Arg>& as) {
                for (const Arg& arg : as)
                {
                    boys::BoysBatchAtTier(tier, arg.n, arg.x, out);
                }
            },
            args,
            repeats);

        std::printf("  %-9s ratio %.4f of the direct call\n", entry.name, atTier / direct);
    }
}

} // namespace

int main() {
    const std::vector<Arg> args = MakeArgs();
    int bad = 0;

    std::printf("=== each tier against the rung it names ===\n");
    bad += CheckReferenceIsToday(args);
    bad += CheckTierAgainstRung<64.0>(args, boys::AccuracyTier::kRelaxed64, "m=64");
    bad += CheckTierAgainstRung<256.0>(args, boys::AccuracyTier::kRelaxed256, "m=256");
    bad += CheckTierAgainstRung<1024.0>(args, boys::AccuracyTier::kRelaxed1024, "m=1024");
    bad += CheckTierAgainstRung<4096.0>(args, boys::AccuracyTier::kRelaxed4096, "m=4096");
    bad += CheckTierAgainstRung<16384.0>(args, boys::AccuracyTier::kRelaxed16384, "m=16384");
    bad += CheckTierAgainstRung<65536.0>(args, boys::AccuracyTier::kRelaxed65536, "m=65536");

    std::printf("=== what a 1e-14 request gets, per region ===\n");
    ShowCoverage("reference", boys::AccuracyTier::kReference);
    ShowCoverage("m=65536", boys::AccuracyTier::kRelaxed65536);

    std::printf("=== cost of selecting the tier per call ===\n");
    TimeTiers(args);

    std::printf("MISMATCHED_VALUES=%d\n", bad);
    return 0;
}
