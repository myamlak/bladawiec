// The link-mode hot-call instrument: the wall time of ONE direct Fock build on
// a fixed fixture, with the fixed-work canary this tree's timing protocol uses.
//
// THE FIXTURE IS FIXED, so the two arms of a comparison do the same work:
// C6H14/STO-3G (44 contracted functions) at a fixed diagonal density - the
// alkane the memory-gate case and the Boys probes use. It is sized so one
// Fock build is a few hundred milliseconds: comfortably above the canary's
// resolution, while a full comparison (each arm launched twice per round)
// stays minutes rather than an afternoon. C12H26/STO-3G was measured at
// about a second per call, which made the protocol's ten rounds cost longer
// than the change under test deserves. It also runs under the 16 GiB bench
// cap without a workspace argument.
//
// THE SERIAL LANE IS THE DEFAULT (maxParallelChunks = 1): the instrument
// measures what a call pays, and thread scheduling is a second variance term
// the canary would then attribute to the machine. --parallel selects the
// production lane; the comparison protocol fixes one choice per run.
//
// EVERY MEASURED CALL IS WRAPPED IN THE FIXED-WORK INTEGER SPIN from
// benchmarks/boys_option_cost_windows.cpp: one dependent xorshift step per
// round, no memory beyond the accumulator, no floating point, so the reading's
// cost cannot move with the work being measured. The driver reports the raw
// spin times instead of a verdict - a single process cannot know the session's
// quiet floor; tools/bench/link_mode_comparison.py takes the floor across the
// arms and reports the peak excess as the run's canary.
//
// The first --warmup calls are not measured: the first call of a process pays
// page faults and table construction that later calls do not, and the mean of
// a run that includes it is not the cost of the call. Each reported figure is
// one call's steady_clock interval; the comparison script reads the minimum
// (the robust number the Fock acceptance benchmark documents) and the
// checksum that proves both arms computed the same matrix to the printed
// precision.
//
// Usage:
//   qcx-bench-fock-hot [--iterations N] [--warmup N] [--parallel] [--json]
//
// --json writes the machine record to stdout and the human lines to stderr;
// without it the human summary goes to stdout.

#include "alkane_sto3g.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

/// Iterations of the canary spin. Fixed, because a reading is compared against
/// the best run of the same work and the work must not vary between them.
constexpr std::uint32_t kCanaryRounds = 1u << 23;

/// The fixed-work spin: a xorshift chain, one dependent step per iteration, no
/// memory beyond the accumulator and no floating point.
double CanaryMilliseconds() {
    volatile std::uint64_t sink = 0;
    const auto from = std::chrono::steady_clock::now();
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
    const auto to = std::chrono::steady_clock::now();
    (void)sink;

    return std::chrono::duration<double, std::milli>(to - from).count();
}

struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd core;
    CpuTensor2 density;
    qcx::integrals::DirectJkFockBuilder builder;
};

/// The fixed input: C6H14/STO-3G and a diagonal density of 0.5. Every arm of
/// a comparison constructs this same fixture, so a reading difference is the
/// call and not the geometry, the basis or the density.
bool BuildFixture(std::unique_ptr<Fixture>& out, bool serial) {
    auto molecule = MakeAlkaneSto3g(6);
    auto basis = MakeAlkaneSto3gBasis();

    if (!molecule.has_value() || !basis.has_value())
    {
        return false;
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    if (!kinetic.has_value())
    {
        return false;
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    if (!nuclear.has_value())
    {
        return false;
    }

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();

    Eigen::MatrixXd densityMatrix =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        densityMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 0.5;
    }

    auto density = ToTensor(densityMatrix);

    if (!density.has_value())
    {
        return false;
    }

    qcx::integrals::FockBuildOptions options;
    // kLoose is the lane the Fock acceptance benchmark measures: its
    // permissive MixedPrecisionThreshold routes a meaningful share of
    // quartets through the certified fp32 path, which is the regime a link
    // mode can plausibly move.
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;

    if (serial)
    {
        // 0/absent is the auto (parallel) lane; 1 is the single-chunk
        // reference lane.
        options.maxParallelChunks = 1;
    }

    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *kinetic, options);

    if (!builder.has_value())
    {
        return false;
    }

    out = std::make_unique<Fixture>(Fixture{std::move(*molecule),
                                            std::move(*basis),
                                            ToMatrix(*kinetic),
                                            std::move(*density),
                                            std::move(*builder)});
    return true;
}

void PrintDoubleArray(const char* key, const std::vector<double>& values) {
    std::printf("\"%s\": [", key);

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
        {
            std::printf(", ");
        }

        std::printf("%.6f", values[i]);
    }

    std::printf("]");
}

} // namespace

int main(int argc, char** argv) {
    int iterations = 21;
    int warmup = 3;
    bool serial = true;
    bool json = false;

    for (int a = 1; a < argc; ++a)
    {
        const std::string arg(argv[a]);

        if (arg == "--parallel")
        {
            serial = false;
            continue;
        }

        if (arg == "--json")
        {
            json = true;
            continue;
        }

        if (arg == "--iterations" && a + 1 < argc)
        {
            iterations = std::atoi(argv[++a]);
            continue;
        }

        if (arg == "--warmup" && a + 1 < argc)
        {
            warmup = std::atoi(argv[++a]);
            continue;
        }

        std::fprintf(stderr,
                     "usage: qcx-bench-fock-hot [--iterations N] [--warmup N] "
                     "[--parallel] [--json]\n");
        return 2;
    }

    if (iterations < 1 || warmup < 0)
    {
        std::fprintf(stderr, "qcx-bench-fock-hot: iterations must be >= 1 and warmup >= 0\n");
        return 2;
    }

    std::unique_ptr<Fixture> fixture;

    if (!BuildFixture(fixture, serial))
    {
        std::fprintf(stderr, "fixture construction failed\n");
        return 1;
    }

    for (int i = 0; i < warmup; ++i)
    {
        auto fock = fixture->builder.BuildFock(fixture->density);

        if (!fock.has_value())
        {
            std::fprintf(stderr, "warm-up BuildFock failed: %s\n", fock.error().message.c_str());
            return 1;
        }
    }

    std::vector<double> perCallMs(static_cast<std::size_t>(iterations));
    std::vector<double> canaryBeforeMs(static_cast<std::size_t>(iterations));
    std::vector<double> canaryAfterMs(static_cast<std::size_t>(iterations));
    double fockSum = 0.0;

    for (int i = 0; i < iterations; ++i)
    {
        const std::size_t index = static_cast<std::size_t>(i);
        canaryBeforeMs[index] = CanaryMilliseconds();
        const auto start = std::chrono::steady_clock::now();
        auto fock = fixture->builder.BuildFock(fixture->density);

        if (!fock.has_value())
        {
            std::fprintf(stderr, "BuildFock failed: %s\n", fock.error().message.c_str());
            return 1;
        }

        const auto stop = std::chrono::steady_clock::now();
        canaryAfterMs[index] = CanaryMilliseconds();
        perCallMs[index] = std::chrono::duration<double, std::milli>(stop - start).count();
        fockSum = ToMatrix(*fock).sum();
    }

    double totalMs = 0.0;
    double minMs = perCallMs.front();

    for (double ms : perCallMs)
    {
        totalMs += ms;

        if (ms < minMs)
        {
            minMs = ms;
        }
    }

    if (json)
    {
        std::printf("{\"schema\": \"qcx-fock-hot/1\", ");
        std::printf("\"fixture\": \"C6H14/STO-3G\", \"preset\": \"kLoose\", ");
        std::printf("\"serial\": %s, \"warmup\": %d, \"iterations\": %d, ",
                    serial ? "true" : "false",
                    warmup,
                    iterations);
        std::printf("\"canary_rounds\": %u, \"fock_sum\": %.17g, ", kCanaryRounds, fockSum);
        PrintDoubleArray("per_call_ms", perCallMs);
        std::printf(", ");
        PrintDoubleArray("canary_before_ms", canaryBeforeMs);
        std::printf(", ");
        PrintDoubleArray("canary_after_ms", canaryAfterMs);
        std::printf("}\n");
        std::fprintf(stderr,
                     "qcx-bench-fock-hot: C6H14/STO-3G %s, %d calls, "
                     "mean %.3f ms, min %.3f ms\n",
                     serial ? "serial" : "parallel",
                     iterations,
                     totalMs / static_cast<double>(iterations),
                     minMs);
        return 0;
    }

    std::printf("fixture=C6H14/STO-3G preset=kLoose serial=%s warmup=%d iterations=%d\n",
                serial ? "yes" : "no",
                warmup,
                iterations);
    std::printf("total_ms=%.3f mean_ms=%.3f min_ms=%.3f sum=%.17g\n",
                totalMs,
                totalMs / static_cast<double>(iterations),
                minMs,
                fockSum);
    return 0;
}
