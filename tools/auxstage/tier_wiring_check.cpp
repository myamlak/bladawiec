// The product's own headers, and nothing else: this translation unit proves a
// caller in this tree can select the accuracy of one call without reaching
// into the submodule's namespace. Numbers only; no timing.

#include "internal/md_boys.hpp"
#include "qcx/integrals/boys.hpp"

#include <cmath>
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

    for (int i = 0; i < 4096; ++i)
    {
        args.push_back({i % 9, 60.0 * i / 4095.0});
    }

    return args;
}

bool SameBits(double left, double right) {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
}

/// The reference tier must be the certified lane exactly, so it is compared
/// against the forwarding name every existing call site already uses.
int CheckReferenceIsTheForwardedName(const std::vector<Arg>& args) {
    double got[qcx::integrals::kMaxBoysOrder + 1];
    double want[qcx::integrals::kMaxBoysOrder + 1];
    int bad = 0;
    int checked = 0;

    for (const Arg& arg : args)
    {
        qcx::integrals::BoysAllOrdersAtTier(
            qcx::integrals::AccuracyTier::kReference, arg.n, arg.x, got);
        qcx::integrals::BoysBatch(arg.n, arg.x, want);

        for (int k = 0; k <= arg.n; ++k)
        {
            ++checked;

            if (!SameBits(got[k], want[k]))
            {
                ++bad;
            }
        }
    }

    std::printf("reference tier against BoysBatch: %d of %d values differ\n", bad, checked);

    return bad;
}

/// A relaxed tier must move the value and stay inside its own bound, m times
/// the certified 5.5e-14 batch bound.
int CheckRelaxedTierStaysInsideItsBound(const std::vector<Arg>& args,
                                        qcx::integrals::AccuracyTier tier) {
    double got[qcx::integrals::kMaxBoysOrder + 1];
    double want[qcx::integrals::kMaxBoysOrder + 1];
    const double m = qcx::integrals::AccuracyMultiplier(tier);
    double worst = 0.0;
    int moved = 0;
    int checked = 0;

    for (const Arg& arg : args)
    {
        qcx::integrals::BoysAllOrdersAtTier(tier, arg.n, arg.x, got);
        qcx::integrals::BoysBatch(arg.n, arg.x, want);

        for (int k = 0; k <= arg.n; ++k)
        {
            ++checked;

            const double delta = std::fabs(got[k] - want[k]);

            if (delta != 0.0)
            {
                ++moved;
            }

            if (delta > worst)
            {
                worst = delta;
            }
        }
    }

    std::printf("m=%-6.0f moved %d of %d values, worst %.3e, bound %.3e, %s\n",
                m,
                moved,
                checked,
                worst,
                m * 5.5e-14,
                worst <= m * 5.5e-14 ? "inside" : "OUTSIDE");

    return worst <= m * 5.5e-14 ? 0 : 1;
}

/// The md engine's own seam must accept a tier and agree with the facade.
int CheckMdSeam(const std::vector<Arg>& args) {
    double got[qcx::integrals::kMaxBoysOrder + 1];
    double want[qcx::integrals::kMaxBoysOrder + 1];
    int bad = 0;

    for (const Arg& arg : args)
    {
        qcx::integrals::internal::MdBoysBatchAtTier(
            qcx::integrals::AccuracyTier::kRelaxed1024, arg.n, arg.x, got);
        qcx::integrals::BoysAllOrdersAtTier(
            qcx::integrals::AccuracyTier::kRelaxed1024, arg.n, arg.x, want);

        for (int k = 0; k <= arg.n; ++k)
        {
            if (!SameBits(got[k], want[k]))
            {
                ++bad;
            }
        }
    }

    std::printf("md seam against the facade at m=1024: %d values differ\n", bad);

    return bad;
}

} // namespace

int main() {
    const std::vector<Arg> args = MakeArgs();
    int bad = 0;

    bad += CheckReferenceIsTheForwardedName(args);
    bad += CheckRelaxedTierStaysInsideItsBound(args, qcx::integrals::AccuracyTier::kRelaxed256);
    bad += CheckRelaxedTierStaysInsideItsBound(args, qcx::integrals::AccuracyTier::kRelaxed65536);
    bad += CheckMdSeam(args);

    const qcx::integrals::TierCoverage coverage = qcx::integrals::QueryTier(
        qcx::integrals::AccuracyTier::kReference, qcx::integrals::AccuracyRegion::kC, 1e-14);

    std::printf("a 1e-14 request in region C: meets=%d reachable=%.3e limiting=%d\n",
                coverage.meets ? 1 : 0,
                coverage.reachable,
                static_cast<int>(coverage.limiting));

    std::printf("WIRING_MISMATCHED_VALUES=%d\n", bad);

    return 0;
}
