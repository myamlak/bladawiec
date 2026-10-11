// The A/B probe of the Boys swap on the device lane: one process, one
// fixture, one card, both sides.
//
// The engine's VRR seeds its slice triangle from a Boys ladder formed per
// primitive quadruple. That ladder used to be the engine's own port of the
// library's host arithmetic (eri_cuda.cu, retired); it is now the library's
// device lane (boys::BoysDeviceAllNF64 / AllNF32, read through the handle
// BoysCuda::DeviceTables fills). This probe keeps both, asks the engine for
// one and then the other - the arm is GpuJkFockBuilder::Create's
// boysProbeArm, recorded at Create - and reports the same spans from both, so
// the comparison is one fixture, one card, one process, and an alternating
// order rather than two builds of the engine.
//
// What is measured, and in whose terms:
//   - the whole GPU Fock build (BuildFock), wall-clock, per arm - the
//     consumer's own unit;
//   - the device timeline span the engine already records (cudaEvents around
//     the device passes), per arm - the same call without the host's share;
//   - the cycles the kernels spent inside the Boys call and inside the VRR
//     around it, in place, per arm - the span the swap actually moves,
//     accumulated by the kernels with clock64() and read back through
//     GpuJkFockBuilder::BoysProbeCounters.
//
// The Boys cycles are split by the evaluation region the argument fell in.
// Region A locates and sums a piece of the per-order Chebyshev table, region B
// sums the region-B seed fit, and region C is the F_0 asymptotic form with the
// upward recursion and no table read at all: a difference between the two
// lanes that lands in C is arithmetic, one that lands in A or B is a table
// access. That split is the only lever this instrument has on the question,
// and it is a split of the *whole* call, not a counter of memory events.
//
// Requires the A/B instrument: configure with -DQCX_BOYS_SWAP_INSTRUMENT=ON.
// Without it the probe builds and reports that it has nothing to measure.
//
// Usage: qcx-bench-boys-swap [--carbons <n>] [--basis <family>] [--rounds <n>]
//
// Both arms run in every round (library, then engine), so the two figures are
// separated by one build and one call, never by a rebuild: an uncontrolled
// second build measures the machine, not the kernel.

#include "alkane_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_cuda.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace {

using CpuTensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

/// The H core of the fixture (kinetic + nuclear attraction), the same helper
/// the CUDA benchmark builds against.
qcx::Result<CpuTensor> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
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
    return std::move(*kinetic);
}

/// The device's rated SM clock in kHz (cudaDevAttrClockRate): the figure the
/// kernel cycles are read against. It is the rated maximum, so a cycle count
/// divided by it is a lower bound on the elapsed time.
double DeviceClockKhz() {
    int khz = 0;

    if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0) != cudaSuccess || khz <= 0)
    {
        return 0.0;
    }

    return static_cast<double>(khz);
}

struct ArmRun {
    int arm = 0;
    long long round = 0;
    double wallMs = 0.0; ///< BuildFock, host wall.
    double deviceMs = 0.0; ///< the engine's device-timeline span of that call.
    double boysMs = 0.0; ///< in-situ Boys cycles over the rated SM clock.
    double vrrMs = 0.0; ///< in-situ VRR cycles over the rated SM clock.
    /// The Boys cycles by evaluation region: A and B read a table, C does not.
    /// \{
    double boysAMs = 0.0;
    double boysBMs = 0.0;
    double boysCMs = 0.0;
    /// \}
    unsigned long long calls = 0;
    unsigned long long refused = 0;
    double callsA = 0.0;
    double callsB = 0.0;
    double callsC = 0.0;
};

struct ArmMeans {
    std::size_t count = 0;
    double wallMs = 0.0;
    double deviceMs = 0.0;
    double boysMs = 0.0;
    double vrrMs = 0.0;
    double boysAMs = 0.0;
    double boysBMs = 0.0;
    double boysCMs = 0.0;
    double calls = 0.0;
    double callsA = 0.0;
    double callsB = 0.0;
    double callsC = 0.0;
};

const char* ArmName(int arm) {
    return arm == 0 ? "library" : "engine";
}

void PrintUsage(const char* program) {
    std::printf("usage: %s [--carbons <n>] [--basis <family>] [--rounds <n>]\n", program);
}

} // namespace

int main(int argc, char** argv) {
    std::size_t carbons = 6;
    std::size_t rounds = 3;
    std::string family = "def2-svp";

    for (int i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--carbons") == 0 && i + 1 < argc)
        {
            carbons = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(arg, "--basis") == 0 && i + 1 < argc)
        {
            family = argv[++i];
        } else if (std::strcmp(arg, "--rounds") == 0 && i + 1 < argc)
        {
            rounds = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else
        {
            PrintUsage(argv[0]);
            return 1;
        }
    }

#if !defined(QCX_BOYS_SWAP_INSTRUMENT)
    std::printf("built without QCX_BOYS_SWAP_INSTRUMENT: nothing to measure "
                "(configure with -DQCX_BOYS_SWAP_INSTRUMENT=ON)\n");
    return 77;
#else
    auto molecule = qcx::testing::MakeAlkaneSto3g(carbons);

    if (!molecule.has_value())
    {
        std::printf("molecule: %s\n", molecule.error().message.c_str());
        return 1;
    }

    const std::string basisDirectory = std::string(QcxBasisDataDir) + "/" + family;
    auto basis = qcx::basisset::ParseNwchemDirectory(basisDirectory);

    if (!basis.has_value())
    {
        std::printf("basis %s: %s\n", basisDirectory.c_str(), basis.error().message.c_str());
        return 1;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::printf("core Hamiltonian: %s\n", core.error().message.c_str());
        return 1;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        std::printf("pair list: %s\n", pairList.error().message.c_str());
        return 1;
    }

    const std::size_t n = pairList->functionCount;
    auto density = CpuTensor::Create({n, n});

    if (!density.has_value())
    {
        std::printf("density: %s\n", density.error().message.c_str());
        return 1;
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        (*density)(i, i) = 0.5;
    }

    density->MarkHostDirty();

    int deviceOrdinal = 0;
    int computeMajor = 0;
    int computeMinor = 0;
    int l2Bytes = 0;
    cudaDeviceProp properties{};
    (void)cudaGetDevice(&deviceOrdinal);
    (void)cudaGetDeviceProperties(&properties, deviceOrdinal);
    (void)cudaDeviceGetAttribute(&computeMajor, cudaDevAttrComputeCapabilityMajor, deviceOrdinal);
    (void)cudaDeviceGetAttribute(&computeMinor, cudaDevAttrComputeCapabilityMinor, deviceOrdinal);
    (void)cudaDeviceGetAttribute(&l2Bytes, cudaDevAttrL2CacheSize, deviceOrdinal);

    const double clockKhz = DeviceClockKhz();

    std::printf("fixture: c%zuH%zu / %s, %zu basis functions, %zu canonical pairs\n",
                carbons,
                2 * carbons + 2,
                family.c_str(),
                n,
                pairList->pairs.size());
    std::printf("card: %s (device %d), sm_%d%d, L2 %d MiB, rated SM clock %.0f MHz, rounds %zu\n",
                properties.name,
                deviceOrdinal,
                computeMajor,
                computeMinor,
                l2Bytes / (1024 * 1024),
                clockKhz / 1000.0,
                rounds);
    std::printf("arms: 0 = the Boys library's device lane, 1 = the engine's own ladder\n\n");

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;

    std::vector<ArmRun> runs;

    for (std::size_t round = 0; round < rounds; ++round)
    {
        for (int arm = 0; arm <= 1; ++arm)
        {
            auto builder =
                qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options, arm);

            if (!builder.has_value())
            {
                std::printf(
                    "builder (arm %s): %s\n", ArmName(arm), builder.error().message.c_str());
                return 1;
            }

            if (builder->BoysProbeArm() != arm)
            {
                std::printf("arm %s did not reach the engine\n", ArmName(arm));
                return 1;
            }

            // One untimed call: the first touch uploads the statics and warms
            // the caches, and its counters are drained so the timed call's
            // record is its own.
            auto warmup = builder->BuildFock(*density);

            if (!warmup.has_value())
            {
                std::printf("warmup (arm %s): %s\n", ArmName(arm), warmup.error().message.c_str());
                return 1;
            }

            (void)builder->BoysProbeCounters();

            const auto start = std::chrono::steady_clock::now();
            auto fock = builder->BuildFock(*density);
            const auto end = std::chrono::steady_clock::now();

            if (!fock.has_value())
            {
                std::printf("call (arm %s): %s\n", ArmName(arm), fock.error().message.c_str());
                return 1;
            }

            const auto counters = builder->BoysProbeCounters();

            ArmRun run;
            run.arm = arm;
            run.round = static_cast<long long>(round);
            run.wallMs = std::chrono::duration<double, std::milli>(end - start).count();
            run.deviceMs = builder->LastBuildDeviceSpanMs();
            run.calls = counters.Calls();
            run.refused = counters.refused;
            run.callsA = static_cast<double>(counters.callsA);
            run.callsB = static_cast<double>(counters.callsB);
            run.callsC = static_cast<double>(counters.callsC);

            if (clockKhz > 0.0)
            {
                const double toMs = 1.0 / clockKhz;
                run.boysMs = static_cast<double>(counters.BoysCycles()) * toMs;
                run.vrrMs = static_cast<double>(counters.vrrCycles) * toMs;
                run.boysAMs = static_cast<double>(counters.boysCyclesA) * toMs;
                run.boysBMs = static_cast<double>(counters.boysCyclesB) * toMs;
                run.boysCMs = static_cast<double>(counters.boysCyclesC) * toMs;
            }

            runs.push_back(run);

            // The spans are thread-cycle sums, not elapsed time: every thread's
            // own cycles are added, so they are comparable between the two arms
            // and their ratio is the answer, while their size against the wall
            // time is the concurrency of the call, not a duration.
            std::printf("round %zu arm %-7s wall %9.1f ms  device %9.1f ms  "
                        "VRR %8.3f Gcycles  Boys %8.3f Gcycles  calls %llu  refused %llu\n",
                        round,
                        ArmName(arm),
                        run.wallMs,
                        run.deviceMs,
                        run.vrrMs * clockKhz / 1e6,
                        run.boysMs * clockKhz / 1e6,
                        run.calls,
                        run.refused);

            if (run.refused != 0)
            {
                std::printf("  the library's entry refused %llu calls\n", run.refused);
            }
        }
    }

    ArmMeans means[2];

    for (const ArmRun& run : runs)
    {
        ArmMeans& mean = means[run.arm];
        ++mean.count;
        mean.wallMs += run.wallMs;
        mean.deviceMs += run.deviceMs;
        mean.boysMs += run.boysMs;
        mean.vrrMs += run.vrrMs;
        mean.boysAMs += run.boysAMs;
        mean.boysBMs += run.boysBMs;
        mean.boysCMs += run.boysCMs;
        mean.calls += static_cast<double>(run.calls);
        mean.callsA += run.callsA;
        mean.callsB += run.callsB;
        mean.callsC += run.callsC;
    }

    for (ArmMeans& mean : means)
    {
        if (mean.count == 0)
        {
            std::printf("no runs completed\n");
            return 1;
        }

        const double count = static_cast<double>(mean.count);
        mean.wallMs /= count;
        mean.deviceMs /= count;
        mean.boysMs /= count;
        mean.vrrMs /= count;
        mean.boysAMs /= count;
        mean.boysBMs /= count;
        mean.boysCMs /= count;
        mean.calls /= count;
        mean.callsA /= count;
        mean.callsB /= count;
        mean.callsC /= count;
    }

    std::printf("\nmeans over %zu rounds per arm (spans in Gcycles of thread time)\n",
                means[0].count);
    std::printf("%-8s %11s %11s %12s %12s %10s\n",
                "arm",
                "wall ms",
                "device ms",
                "VRR Gcyc",
                "Boys Gcyc",
                "calls");
    std::printf("%-8s %11.1f %11.1f %12.3f %12.3f %10.0f\n",
                ArmName(0),
                means[0].wallMs,
                means[0].deviceMs,
                means[0].vrrMs * clockKhz / 1e6,
                means[0].boysMs * clockKhz / 1e6,
                means[0].calls);
    std::printf("%-8s %11.1f %11.1f %12.3f %12.3f %10.0f\n",
                ArmName(1),
                means[1].wallMs,
                means[1].deviceMs,
                means[1].vrrMs * clockKhz / 1e6,
                means[1].boysMs * clockKhz / 1e6,
                means[1].calls);

    // The per-round paired ratios: one build between the two arms of a round,
    // so a round's ratio is the controlled comparison and the mean over rounds
    // is the headline.
    double boysRatioSum = 0.0;
    double vrrRatioSum = 0.0;
    double ratioMin = 0.0;
    double ratioMax = 0.0;
    std::size_t ratioCount = 0;

    for (std::size_t round = 0; round < rounds; ++round)
    {
        const ArmRun* libraryRun = nullptr;
        const ArmRun* engineRun = nullptr;

        for (const ArmRun& run : runs)
        {
            if (run.round != static_cast<long long>(round))
            {
                continue;
            }

            if (run.arm == 0)
            {
                libraryRun = &run;
            } else
            {
                engineRun = &run;
            }
        }

        if (libraryRun == nullptr || engineRun == nullptr || engineRun->boysMs <= 0.0 ||
            engineRun->vrrMs <= 0.0)
        {
            continue;
        }

        const double ratio = libraryRun->boysMs / engineRun->boysMs;
        boysRatioSum += ratio;
        vrrRatioSum += libraryRun->vrrMs / engineRun->vrrMs;
        ++ratioCount;

        if (ratioCount == 1 || ratio < ratioMin)
        {
            ratioMin = ratio;
        }

        if (ratioCount == 1 || ratio > ratioMax)
        {
            ratioMax = ratio;
        }
    }

    if (ratioCount > 0)
    {
        std::printf("\nlibrary / engine, paired per round: Boys %.3f [%.3f, %.3f], VRR %.3f "
                    "over %zu rounds\n",
                    boysRatioSum / static_cast<double>(ratioCount),
                    ratioMin,
                    ratioMax,
                    vrrRatioSum / static_cast<double>(ratioCount),
                    ratioCount);
    }

    // The Boys share of the VRR it sits inside: same kernels, same threads,
    // same units, so this one is a share in the ordinary sense. The device
    // timeline and the wall are elapsed time and the spans are thread-cycle
    // sums, so a ratio against them would be the call's concurrency wearing a
    // percentage sign - printed as a count instead, where it is honest.
    for (int arm = 0; arm <= 1; ++arm)
    {
        const ArmMeans& mean = means[arm];
        const double toCalls = mean.calls > 0.0 ? 1.0 / mean.calls : 0.0;

        std::printf("\narm %s\n", ArmName(arm));
        std::printf("  Boys of the VRR span (same kernels, same threads) %6.2f%%\n",
                    100.0 * mean.boysMs / mean.vrrMs);
        std::printf("  cycles per Boys call: A %8.1f  B %8.1f  C %8.1f  all %8.1f\n",
                    mean.boysAMs * clockKhz * toCalls,
                    mean.boysBMs * clockKhz * toCalls,
                    mean.boysCMs * clockKhz * toCalls,
                    mean.boysMs * clockKhz * toCalls);
        std::printf("  VRR cycles per Boys call: %8.1f\n", mean.vrrMs * clockKhz * toCalls);
        std::printf("  call mix: A %5.2f%%  B %5.2f%%  C %5.2f%%\n",
                    100.0 * mean.callsA / mean.calls,
                    100.0 * mean.callsB / mean.calls,
                    100.0 * mean.callsC / mean.calls);
    }

    const double libraryToCalls = means[0].calls > 0.0 ? 1.0 / means[0].calls : 0.0;
    const double engineToCalls = means[1].calls > 0.0 ? 1.0 / means[1].calls : 0.0;

    // The region split is where the two lanes differ most: A and B locate and
    // sum a table, C is the recursion and reads none.
    std::printf("\nregion split (cycles per call, library vs engine):\n");
    std::printf("  A (table)      %8.1f vs %8.1f\n",
                means[0].boysAMs * clockKhz * libraryToCalls,
                means[1].boysAMs * clockKhz * engineToCalls);
    std::printf("  B (table)      %8.1f vs %8.1f\n",
                means[0].boysBMs * clockKhz * libraryToCalls,
                means[1].boysBMs * clockKhz * engineToCalls);
    std::printf("  C (arithmetic) %8.1f vs %8.1f\n",
                means[0].boysCMs * clockKhz * libraryToCalls,
                means[1].boysCMs * clockKhz * engineToCalls);

    return 0;
#endif
}
