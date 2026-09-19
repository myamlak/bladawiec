// The profiling harness: a
// representative production-sized run to attach a real profiler (Windows:
// wpr CPU sampling, VS CPU profiler) to, before picking any
// candidate to redesign. Plain main() program - not a
// google-benchmark run (profiling wants one sustained, uninterrupted
// workload per phase, not benchmark interleaving).
//
// Phase A - the direct J/K Fock build at a production size: C80H162 at
// STO-3G (562 basis functions - past the 500-function floor the benchmark
// sizes note mandates), Create once + repeated BuildFock. This phase
// exercises the quartet-contraction candidate path: the MdQuartetTask lists
// (screening, class-batch assembly, contraction).
//
// Phase B - the RI-J engine at a moderate size (the RI path's realistic
// regime): C20H42 at STO-3G (142 functions) with def2-universal-jfit.
// Create exercises BuildRiTensor (the metric-build candidate: the n^2 x nAux
// riMatrix assembly traversal) and the metric factorization; the repeated
// BuildFock exercises the per-iteration RI contractions.
//
// Flags (the single entry point for every measurement made here):
//   --iterations N   BuildFock repetitions per phase (default 8)
//   --chunks N       FockBuildOptions::maxParallelChunks pass-through
//                    (0 = auto [the default], 1 = serial fallback,
//                    >= 2 = fixed split)
//   --serial         shorthand for --chunks 1
// Startup prints the machine-state banner (CPU topology, policy team size,
// OMP_NUM_THREADS override, host memory) and every phase reports wall AND
// process CPU time (the wall-vs-CPU attribution); the direct phases also
// print the FockBuildStats ERI/contract split per iteration.

#include "alkane_sto3g.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/backend/cpu_topology.hpp"
#include "qcx/backend/memory_topology.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string_view>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/resource.h>
#endif

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

double ElapsedMs(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

/// Process-wide CPU time (kernel + user), milliseconds - the process-wide
/// wall-vs-CPU attribution. Windows: GetProcessTimes (100 ns units; MSVC's
/// std::clock is not process-CPU-correct - the Eigen lapack precedent).
/// Linux/WSL: getrusage - the Linux path is GCC-only, and one
/// helper keeps both harness builds on a single code path.
double ProcessCpuTimeMs() {
#ifdef _WIN32
    FILETIME creationTime{};
    FILETIME exitTime{};
    FILETIME kernelTime{};
    FILETIME userTime{};

    if (GetProcessTimes(GetCurrentProcess(), &creationTime, &exitTime, &kernelTime, &userTime) == 0)
    {
        return 0.0;
    }

    ULARGE_INTEGER kernel{};
    kernel.HighPart = kernelTime.dwHighDateTime;
    kernel.LowPart = kernelTime.dwLowDateTime;
    ULARGE_INTEGER user{};
    user.HighPart = userTime.dwHighDateTime;
    user.LowPart = userTime.dwLowDateTime;

    return static_cast<double>(kernel.QuadPart + user.QuadPart) / 10000.0; // 100 ns -> ms
#else
    struct rusage usage = {};

    if (getrusage(RUSAGE_SELF, &usage) != 0)
    {
        return 0.0;
    }

    const double userMs = static_cast<double>(usage.ru_utime.tv_sec) * 1000.0 +
                          static_cast<double>(usage.ru_utime.tv_usec) / 1000.0;
    const double systemMs = static_cast<double>(usage.ru_stime.tv_sec) * 1000.0 +
                            static_cast<double>(usage.ru_stime.tv_usec) / 1000.0;

    return userMs + systemMs;
#endif
}

/// Per-run knobs.
struct RunOptions {
    int iterations = 8; ///< BuildFock repetitions per phase (--iterations).
    std::size_t maxParallelChunks = 0; ///< FockBuildOptions::maxParallelChunks
                                       ///< pass-through (--chunks / --serial);
                                       ///< 0 = auto, 1 = serial, >= 2 fixed.
};

bool ParseRunOptions(int argc, char** argv, RunOptions& options) {
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg = argv[i];

        if (arg == "--iterations" && i + 1 < argc)
        {
            const int value = std::atoi(argv[++i]);

            if (value < 1)
            {
                std::fprintf(stderr, "b2 profile harness: --iterations must be >= 1\n");
                return false;
            }

            options.iterations = value;
        } else if (arg == "--chunks" && i + 1 < argc)
        {
            const long long value = std::atoll(argv[++i]);

            if (value < 0)
            {
                std::fprintf(stderr, "b2 profile harness: --chunks must be >= 0\n");
                return false;
            }

            options.maxParallelChunks = static_cast<std::size_t>(value);
        } else if (arg == "--serial")
        {
            options.maxParallelChunks = 1;
        } else
        {
            return false;
        }
    }

    return true;
}

void PrintUsage(const char* program) {
    std::fprintf(stderr, "usage: %s [--iterations N] [--chunks N | --serial]\n", program);
    std::fprintf(stderr, "  --iterations N  BuildFock repetitions per phase (default 8)\n");
    std::fprintf(
        stderr,
        "  --chunks N      maxParallelChunks: 0 = auto (default), 1 = serial, >= 2 fixed\n");
    std::fprintf(stderr, "  --serial        shorthand for --chunks 1\n");
}

/// The machine-state banner: every timed number is
/// interpreted against this (the team-size policy). The
/// runner records the power plan and idle-state checks in the run log.
void PrintBanner(const RunOptions& options) {
    const qcx::backend::CpuTopology topology = qcx::backend::DetectCpuTopology();
    const qcx::backend::HostMemoryInfo memory = qcx::backend::DetectHostMemory();

    std::printf("b2 profile harness banner\n");
    std::printf(
        "  topology: %zu logical, %zu physical, %zu P-cores, %zu E-cores, hyperthreaded %s\n",
        topology.logicalProcessors,
        topology.physicalCores,
        topology.performanceCores,
        topology.efficiencyCores,
        topology.hyperthreaded ? "yes" : "no");
    std::printf("  policy team size (DefaultOmpTeamSize): %d\n",
                qcx::backend::DefaultOmpTeamSize());
    const char* ompThreads = std::getenv("OMP_NUM_THREADS");

    if (ompThreads != nullptr)
    {
        std::printf("  OMP_NUM_THREADS env: %s (overrides the policy team size)\n", ompThreads);
    } else
    {
        std::printf("  OMP_NUM_THREADS env: unset (policy team size in force)\n");
    }

    std::printf("  host memory: total %.2f GiB, available %.2f GiB\n",
                static_cast<double>(memory.totalBytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(memory.availableBytes) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  run: iterations %d, maxParallelChunks %zu\n",
                options.iterations,
                options.maxParallelChunks);
}

void PrintFockStats(int iteration, const qcx::integrals::FockBuildStats& stats) {
    std::printf("  stats (BuildFock %d): eri %.1f ms, contract %.1f ms, total %.1f ms, "
                "fp64 %zu, fp32 %zu\n",
                iteration,
                std::chrono::duration<double, std::milli>(stats.eriWallTime).count(),
                std::chrono::duration<double, std::milli>(stats.contractWallTime).count(),
                std::chrono::duration<double, std::milli>(stats.totalWallTime).count(),
                stats.fp64QuartetCount,
                stats.fp32QuartetCount);
}

qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
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

qcx::Result<CpuTensor2> MakeDensity(std::size_t n) {
    auto density = CpuTensor2::Create({n, n});

    if (!density.has_value())
    {
        return std::unexpected(density.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        (*density)(i, i) = 0.5;
    }

    density->MarkHostDirty();
    return std::move(*density);
}

void RunDirectPhase(const RunOptions& options) {
    constexpr std::size_t kCarbonCount = 80; // C80H162, 562 STO-3G functions.

    auto molecule = qcx::testing::MakeAlkaneSto3g(kCarbonCount);

    if (!molecule.has_value())
    {
        std::fprintf(stderr, "phase A: alkane molecule failed\n");
        return;
    }

    auto basis = qcx::testing::MakeAlkaneSto3gBasis();

    if (!basis.has_value())
    {
        std::fprintf(stderr, "phase A: alkane basis failed\n");
        return;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::fprintf(stderr, "phase A: core Hamiltonian failed\n");
        return;
    }

    auto density = MakeDensity(core->Shape()[0]);

    if (!density.has_value())
    {
        std::fprintf(stderr, "phase A: density failed\n");
        return;
    }

    std::printf("phase A: direct Create, n = %zu\n", core->Shape()[0]);
    qcx::integrals::FockBuildOptions fockOptions;
    fockOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    fockOptions.maxParallelChunks = options.maxParallelChunks;
    const auto createStart = std::chrono::steady_clock::now();
    const double createCpuStart = ProcessCpuTimeMs();
    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fockOptions);

    if (!builder.has_value())
    {
        std::fprintf(stderr, "phase A: builder Create failed\n");
        return;
    }

    const double createCpuMs = ProcessCpuTimeMs() - createCpuStart;
    std::printf("phase A: Create %.1f ms wall, %.1f ms cpu\n", ElapsedMs(createStart), createCpuMs);
    std::printf("phase A: builder ready, running BuildFock x%d\n", options.iterations);

    for (int i = 0; i < options.iterations; ++i)
    {
        const auto iterationStart = std::chrono::steady_clock::now();
        const double iterationCpuStart = ProcessCpuTimeMs();
        qcx::integrals::FockBuildStats stats;
        auto fock = builder->BuildFock(*density, nullptr, &stats);

        if (!fock.has_value())
        {
            std::fprintf(stderr, "phase A: BuildFock failed\n");
            return;
        }

        const double iterationCpuMs = ProcessCpuTimeMs() - iterationCpuStart;
        std::printf("phase A: BuildFock %d: %.1f ms wall, %.1f ms cpu\n",
                    i,
                    ElapsedMs(iterationStart),
                    iterationCpuMs);
        PrintFockStats(i, stats);
    }
}

void RunRiPhase(const RunOptions& options) {
    constexpr std::size_t kCarbonCount = 20; // C20H42, 142 STO-3G functions.

    auto molecule = qcx::testing::MakeAlkaneSto3g(kCarbonCount);

    if (!molecule.has_value())
    {
        std::fprintf(stderr, "phase B: alkane molecule failed\n");
        return;
    }

    auto basis = qcx::testing::MakeAlkaneSto3gBasis();

    if (!basis.has_value())
    {
        std::fprintf(stderr, "phase B: alkane basis failed\n");
        return;
    }

    const std::filesystem::path root(QcxBasisDataDir);
    // The per-element aux parse filter: the alkane phase needs only C and H,
    // so the parsed set is exactly the molecule's elements.
    const std::array<int, 2> auxElements{1, 6};
    auto aux = qcx::basisset::ParseNwchemDirectoryFiltered((root / "def2-universal-jfit").string(),
                                                           auxElements);

    if (!aux.has_value())
    {
        std::fprintf(stderr, "phase B: jfit aux basis failed\n");
        return;
    }

    if (!qcx::integrals::SupportsL(4))
    {
        std::fprintf(stderr, "phase B: kMaxEngineL below the jfit g shells\n");
        return;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::fprintf(stderr, "phase B: core Hamiltonian failed\n");
        return;
    }

    auto density = MakeDensity(core->Shape()[0]);

    if (!density.has_value())
    {
        std::fprintf(stderr, "phase B: density failed\n");
        return;
    }

    std::printf("phase B: RI Create (BuildRiTensor + metric solve), n = %zu\n", core->Shape()[0]);
    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    const auto createStart = std::chrono::steady_clock::now();
    const double createCpuStart = ProcessCpuTimeMs();
    auto builder =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, riOptions);

    if (!builder.has_value())
    {
        std::fprintf(stderr, "phase B: RI Create failed\n");
        return;
    }

    const double createCpuMs = ProcessCpuTimeMs() - createCpuStart;
    std::printf("phase B: Create %.1f ms wall, %.1f ms cpu\n", ElapsedMs(createStart), createCpuMs);
    std::printf("phase B: builder ready, running BuildFock x%d\n", options.iterations);

    for (int i = 0; i < options.iterations; ++i)
    {
        const auto iterationStart = std::chrono::steady_clock::now();
        const double iterationCpuStart = ProcessCpuTimeMs();
        auto fock = builder->BuildFock(*density);

        if (!fock.has_value())
        {
            std::fprintf(stderr, "phase B: BuildFock failed\n");
            return;
        }

        const double iterationCpuMs = ProcessCpuTimeMs() - iterationCpuStart;
        std::printf("phase B: BuildFock %d: %.1f ms wall, %.1f ms cpu\n",
                    i,
                    ElapsedMs(iterationStart),
                    iterationCpuMs);
    }
}

} // namespace

int main(int argc, char** argv) {
    RunOptions options;

    if (!ParseRunOptions(argc, argv, options))
    {
        PrintUsage(argv[0]);
        return 1;
    }

    PrintBanner(options);
    std::printf("b2 profile harness: phase A start\n");
    RunDirectPhase(options);
    std::printf("b2 profile harness: phase A end\n");
    std::printf("b2 profile harness: phase B start\n");
    RunRiPhase(options);
    std::printf("b2 profile harness: phase B end\n");
    return 0;
}
