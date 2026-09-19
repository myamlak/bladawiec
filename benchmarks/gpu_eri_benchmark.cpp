// The CUDA ERI throughput benchmarks:
// per-variant rates on representative classes (kV0: L <= 4, kV1: 5..8,
// kV2: L >= 9) plus the CPU-vs-GPU comparison on the (6,6) peak class.
// Record-only - no thresholds: the numbers feed the validation
// sweep and peak-efficiency tracking (the flop model).
// The suite skips without a CUDA device (the engine's Create probes the
// device count), so CUDA builds without a GPU stay green.
//
// Plus the host-vs-device split profile: the GpuJkFockBuilder::BuildFock
// wall vs the device-timeline
// span of the launches inside that call, per iteration - CUDA events on
// the default stream bracketing each device pass (the corrected-method
// instrument, replacing the invalid isolation-engine decomposition of
// 48ed9d0) plus the DirectJkFockBuilder host-only wall leg - the
// measurement that decides whether overlapping host-side screening over
// device compute is worth pursuing.

#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_cuda.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/gpu_fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

inline constexpr std::size_t kChainAtomCount = 6;
inline constexpr double kChainSpacing = 2.0;

// The s/p chain: classes (0,0), (0,1), (1,1) - the kV0 range (L <= 4).
inline constexpr std::string_view kSpBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    P
      7.0000000000E-01       1.0000000000E+00
END
)";

// The s/d chain: classes up to (4,4), L = 8 - the kV1 range.
inline constexpr std::string_view kDdBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    D
      5.0000000000E-01       1.0000000000E+00
END
)";

// The s/i chain: the (i, s) pairs form the homogeneous (6,6) class, L = 12 -
// the kV2 range and the peak-efficiency target class.
inline constexpr std::string_view kIChainBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    I
      5.0000000000E-01       1.0000000000E+00
END
)";

qcx::Result<qcx::molecule::Molecule> MakeChainMolecule() {
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({kChainAtomCount, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t i = 0; i < kChainAtomCount; ++i)
    {
        (*coordinates)(i, 0) = static_cast<double>(i) * kChainSpacing;
        (*coordinates)(i, 1) = 0.0;
        (*coordinates)(i, 2) = 0.0;
        atoms.push_back({"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

std::size_t CountIntegrals(const std::vector<qcx::integrals::ShellQuartet>& quartets,
                           const qcx::integrals::ShellPairList& pairList) {
    std::size_t total = 0;

    for (const qcx::integrals::ShellQuartet& quartet : quartets)
    {
        const auto count = [&pairList](std::size_t shell) {
            const qcx::integrals::ShellInfo& info = pairList.shells[shell];
            const std::size_t angular =
                info.isSpherical ? static_cast<std::size_t>(2 * info.angularMomentum + 1)
                                 : static_cast<std::size_t>((info.angularMomentum + 1) *
                                                            (info.angularMomentum + 2) / 2);
            return info.contractionCount * angular;
        };
        total += count(quartet.i) * count(quartet.j) * count(quartet.k) * count(quartet.l);
    }

    return total;
}

/// All canonical quartets over the given shell indices (the eri_batch
/// canonicalization contract: bra pair >= ket pair).
std::vector<qcx::integrals::ShellQuartet> AllCanonicalQuartets(
    const std::vector<std::size_t>& shells, const qcx::integrals::ShellPairList& pairList) {
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (const std::size_t a : shells)
    {
        for (const std::size_t b : shells)
        {
            if (b < a)
            {
                continue;
            }

            for (const std::size_t c : shells)
            {
                for (const std::size_t d : shells)
                {
                    if (d < c)
                    {
                        continue;
                    }

                    if (qcx::integrals::PairIndexOf(a, b, pairList) <
                        qcx::integrals::PairIndexOf(c, d, pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    return quartets;
}

/// The homogeneous (6,6) workload: (i_a, s_a) x (i_b, s_b) quartets.
std::vector<qcx::integrals::ShellQuartet> Class66Quartets() {
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (std::size_t atomA = 0; atomA < kChainAtomCount; ++atomA)
    {
        const std::size_t iShellA = 2 * atomA + 1;
        const std::size_t sShellA = 2 * atomA;

        for (std::size_t atomB = 0; atomB <= atomA; ++atomB)
        {
            const std::size_t iShellB = 2 * atomB + 1;
            const std::size_t sShellB = 2 * atomB;
            quartets.push_back({iShellA, sShellA, iShellB, sShellB});
        }
    }

    return quartets;
}

/// Runs one fixed engine + workload pair; returns false when the device is
/// absent (the suite then skips).
bool RunEngine(benchmark::State& state,
               qcx::integrals::EriCudaEngine& engine,
               const std::vector<qcx::integrals::ShellQuartet>& quartets,
               std::size_t integralsPerRun) {
    for (auto _ : state)
    {
        auto batch = engine.ComputeBatch(quartets);

        if (!batch.has_value())
        {
            state.SkipWithError(batch.error().message.c_str());
            return false;
        }
    }

    state.SetItemsProcessed(state.iterations() * integralsPerRun);
    return true;
}

void BenchGpuVariant(benchmark::State& state,
                     qcx::integrals::EriCudaVariant variant,
                     std::string_view basisText) {
    auto basis = qcx::basisset::ParseNwchemText(basisText);

    if (!basis.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto molecule = MakeChainMolecule();

    if (!molecule.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    qcx::integrals::EriCudaOptions options;
    options.variant = variant;
    auto engine = qcx::integrals::EriCudaEngine::Create(*molecule, *basis, options);

    if (!engine.has_value())
    {
        state.SkipWithError(engine.error().message.c_str());
        return;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        state.SkipWithError("pair construction failed");
        return;
    }

    const std::vector<std::size_t> shells = {0, 1};
    const std::vector<qcx::integrals::ShellQuartet> quartets =
        AllCanonicalQuartets(shells, *pairList);
    RunEngine(state, *engine, quartets, CountIntegrals(quartets, *pairList));
}

void BenchGpuV0(benchmark::State& state) {
    BenchGpuVariant(state, qcx::integrals::EriCudaVariant::kV0, kSpBasis);
}

void BenchGpuV1(benchmark::State& state) {
    BenchGpuVariant(state, qcx::integrals::EriCudaVariant::kV1, kDdBasis);
}

void BenchGpuV2(benchmark::State& state) {
    BenchGpuVariant(state, qcx::integrals::EriCudaVariant::kV2, kIChainBasis);
}

void BenchCpuVsGpuClass66(benchmark::State& state) {
    if (!qcx::integrals::SupportsL(6))
    {
        state.SkipWithError("this build has no l = 6 classes");
        return;
    }

    auto basis = qcx::basisset::ParseNwchemText(kIChainBasis);

    if (!basis.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto molecule = MakeChainMolecule();

    if (!molecule.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto engine = qcx::integrals::EriCudaEngine::Create(*molecule, *basis);

    if (!engine.has_value())
    {
        state.SkipWithError(engine.error().message.c_str());
        return;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        state.SkipWithError("pair construction failed");
        return;
    }

    const std::vector<qcx::integrals::ShellQuartet> quartets = Class66Quartets();
    const std::size_t integralsPerRun = CountIntegrals(quartets, *pairList);

    for (auto _ : state)
    {
        auto cpu = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);

        if (!cpu.has_value())
        {
            state.SkipWithError(cpu.error().message.c_str());
            return;
        }

        auto gpu = engine->ComputeBatch(quartets);

        if (!gpu.has_value())
        {
            state.SkipWithError(gpu.error().message.c_str());
            return;
        }
    }

    state.SetItemsProcessed(state.iterations() * integralsPerRun);
}

// The Fock-build comparison: the same screened
// workload - the s/d chain at kLoose with the density gate active - through
// DirectJkFockBuilder (CPU) and GpuJkFockBuilder (GPU). Record-only, the
// explicit acceptance ("speedups recorded, not asserted as a hard
// gate - hardware-dependent"): the two builds/sec numbers give the speedup
// by division; no threshold is asserted anywhere.
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoreHamiltonian(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
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

void BenchFockBuild(benchmark::State& state, bool gpu) {
    auto basis = qcx::basisset::ParseNwchemText(kDdBasis);

    if (!basis.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto molecule = MakeChainMolecule();

    if (!molecule.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        state.SkipWithError("core Hamiltonian construction failed");
        return;
    }

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        state.SkipWithError("pair construction failed");
        return;
    }

    // A modest symmetric density (0.5 * I in function space): the density
    // gate reads the max-|D| block weights, so the same screened list runs
    // on both sides.
    const std::size_t n = pairList->functionCount;
    auto density = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});

    if (!density.has_value())
    {
        state.SkipWithError("density construction failed");
        return;
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        (*density)(i, i) = 0.5;
    }

    density->MarkHostDirty();

    if (gpu)
    {
        auto builder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);

        if (!builder.has_value())
        {
            state.SkipWithError(builder.error().message.c_str());
            return;
        }

        for (auto _ : state)
        {
            auto fock = builder->BuildFock(*density);

            if (!fock.has_value())
            {
                state.SkipWithError(fock.error().message.c_str());
                return;
            }
        }
    } else
    {
        auto builder =
            qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);

        if (!builder.has_value())
        {
            state.SkipWithError(builder.error().message.c_str());
            return;
        }

        for (auto _ : state)
        {
            auto fock = builder->BuildFock(*density);

            if (!fock.has_value())
            {
                state.SkipWithError(fock.error().message.c_str());
                return;
            }
        }
    }
}

void BenchCpuFockBuild(benchmark::State& state) {
    BenchFockBuild(state, false);
}

void BenchGpuFockBuild(benchmark::State& state) {
    BenchFockBuild(state, true);
}

// The host-vs-device split, the CORRECTED-METHOD instrument
// (the 48ed9d0 protocol flaw - the isolated second engine's ComputeBatch
// over the FULL canonical quartet list measured ~39 ms against a ~25.5 ms
// concurrent BuildFock total, so host = total - device underflowed
// NEGATIVE: the isolation run was a different, fuller device workload,
// not a component of the concurrent path). Per iteration, the corrected
// decomposition measures ON the concurrent path itself: the
// GpuJkFockBuilder::BuildFock wall (chrono) and the device-timeline span
// of the device launches INSIDE that call (CUDA events on the default
// stream bracketing each device pass - the fp64 and certified fp32 lanes;
// GpuJkFockBuilder::LastBuildDeviceSpanMs). Host = wall - span is then
// the concurrent path's own host share (screening, the pre-pass assembly,
// the read-backs) and can never underflow. The host-only leg: the
// DirectJkFockBuilder wall on the same fixture (no device involvement at
// all) as the host-bound reference - what the host alone needs for the
// full build, the alternative candidate (b). The split prints
// once per invocation.
void BenchGpuHostDeviceSplit(benchmark::State& state) {
    auto basis = qcx::basisset::ParseNwchemText(kDdBasis);

    if (!basis.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto molecule = MakeChainMolecule();

    if (!molecule.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        state.SkipWithError("core Hamiltonian construction failed");
        return;
    }

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        state.SkipWithError("pair construction failed");
        return;
    }

    const std::size_t n = pairList->functionCount;
    auto density = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});

    if (!density.has_value())
    {
        state.SkipWithError("density construction failed");
        return;
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        (*density)(i, i) = 0.5;
    }

    density->MarkHostDirty();

    auto builder = qcx::integrals::GpuJkFockBuilder::Create(*molecule, *basis, *core, options);

    if (!builder.has_value())
    {
        state.SkipWithError(builder.error().message.c_str());
        return;
    }

    // The host-only leg: the CPU builder over the identical fixture - the
    // screening semantics are the SHARED path (internal/fock_screen.hpp),
    // so the wall is the host-alone reference for the same workload.
    auto hostBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);

    if (!hostBuilder.has_value())
    {
        state.SkipWithError(hostBuilder.error().message.c_str());
        return;
    }

    double totalMs = 0.0;
    double deviceSpanMs = 0.0;
    double hostOnlyMs = 0.0;
    double sink = 0.0;
    std::size_t iterations = 0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        const auto buildStart = std::chrono::steady_clock::now();
        auto fock = builder->BuildFock(*density);
        const auto buildStop = std::chrono::steady_clock::now();

        if (!fock.has_value())
        {
            state.SkipWithError(fock.error().message.c_str());
            return;
        }

        // The device-timeline span of the launches inside the concurrent
        // BuildFock (the CUDA events bracket the passes internally - see
        // LastBuildDeviceSpanMs); the span is read after the wall markers
        // so both clocks cover the same call.
        const double deviceSpan = builder->LastBuildDeviceSpanMs();

        const auto hostStart = std::chrono::steady_clock::now();
        auto hostFock = hostBuilder->BuildFock(*density);
        const auto hostStop = std::chrono::steady_clock::now();

        if (!hostFock.has_value())
        {
            state.SkipWithError(hostFock.error().message.c_str());
            return;
        }

        totalMs += std::chrono::duration<double, std::milli>(buildStop - buildStart).count();
        deviceSpanMs += deviceSpan;
        hostOnlyMs += std::chrono::duration<double, std::milli>(hostStop - hostStart).count();
        sink += (*fock)(0, 0) + (*hostFock)(0, 0);
        ++iterations;
    }

    benchmark::DoNotOptimize(sink);

    if (iterations != 0)
    {
        const double totalMean = totalMs / static_cast<double>(iterations);
        const double deviceMean = deviceSpanMs / static_cast<double>(iterations);
        const double hostMean = totalMean - deviceMean;
        const double hostOnlyMean = hostOnlyMs / static_cast<double>(iterations);

        if (deviceSpanMs == 0.0)
        {
            std::cout << "GPU host/device split: total " << totalMean << " ms | device span "
                      << deviceMean << " ms (INSTRUMENT UNAVAILABLE - the engine's event "
                      << "creation failed) | host-only wall (DirectJkFockBuilder) " << hostOnlyMean
                      << " ms\n";
        } else
        {
            std::cout << "GPU host/device split: total " << totalMean << " ms | device span "
                      << deviceMean << " ms (" << (100.0 * deviceMean / totalMean) << " %)"
                      << " | host " << hostMean << " ms (" << (100.0 * hostMean / totalMean)
                      << " %)"
                      << " | host-only wall (DirectJkFockBuilder) " << hostOnlyMean << " ms\n";
        }
    }
}

BENCHMARK(BenchGpuV0)->Unit(benchmark::kSecond);
BENCHMARK(BenchGpuV1)->Unit(benchmark::kSecond);
BENCHMARK(BenchGpuV2)->Unit(benchmark::kSecond);
BENCHMARK(BenchCpuVsGpuClass66)->Unit(benchmark::kSecond);
BENCHMARK(BenchCpuFockBuild)->Unit(benchmark::kSecond);
BENCHMARK(BenchGpuFockBuild)->Unit(benchmark::kSecond);
BENCHMARK(BenchGpuHostDeviceSplit)->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
