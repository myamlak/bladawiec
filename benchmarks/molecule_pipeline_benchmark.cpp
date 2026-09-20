// Scaling-benchmark pattern: the molecule pipeline (create + connectivity +
// nuclear repulsion) on the analytic C60 fixture. Benchmarks are local-only:
// QCX_BUILD_BENCHMARKS is ON in the windows-msvc preset and never in CI, whose
// jobs configure with bare cmake and build no benchmark target.

#include "large_molecules.hpp"
#include "qcx/molecule/connectivity.hpp"
#include "qcx/molecule/mass_properties.hpp"

#include <benchmark/benchmark.h>

namespace {

void BenchmarkMoleculePipeline(benchmark::State& state) {
    const auto molecule = qcx::molecule::testing::MakeBuckminsterfullerene();

    // Benchmarks have no ASSERT: fail loudly instead of silently skipping.
    if (!molecule.has_value())
    {
        state.SkipWithError(molecule.error().message.c_str());
        return;
    }

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        const auto connectivity = qcx::molecule::BuildConnectivity(*molecule);
        double repulsion = qcx::molecule::NuclearRepulsionEnergy(*molecule);
        benchmark::DoNotOptimize(connectivity.csr.neighbors.size());
        benchmark::DoNotOptimize(repulsion);
    }
}

BENCHMARK(BenchmarkMoleculePipeline)->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
