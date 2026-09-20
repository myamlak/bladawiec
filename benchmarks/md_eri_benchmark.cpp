// The MD engine throughput benchmarks: the pure (6,6)-class workload
// (i-shell pairs - the peak-efficiency target, >= 30% of the machine's FMA
// peak) and the all-class shellset spread. Reported as integrals/second;
// the (6,6) lane also prints the achieved fraction of the machine's FMA
// peak (the class's per-integral flop model is the constant below).
//
// The pair tables are per-molecule-basis constants: a Fock build constructs
// them once and reuses them. ComputeEriBatch rebuilds them on every call -
// correct for tests, but the (i,i) l = 12 pair-table fold dominates the
// pass time at class (6,6) (measured 2026-08-22: ~320 ms of the ~325 ms
// pass). This benchmark therefore drives the internal batch API directly:
// BuildPairData + AssembleClassBatches once in the fixture, RunBatches per
// iteration - the same split the Fock builder and the CUDA host setup
// already use (fock_build.cpp Create).

#include "internal/md_batch.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "shellset_fixture.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string_view>
#include <vector>

namespace {

// A synthetic two-center i-shell basis: each of the kChainAtomCount atoms
// carries one i shell (l = 6) and one s shell, so the (i, s) pairs form
// one homogeneous class-6 batch.
inline constexpr std::size_t kChainAtomCount = 6;
inline constexpr double kChainSpacing = 2.0;

// The class (6,6) flop model: per primitive quadruple - ket GEMM
// 2 * kHerm^2 * 13, bra GEMM 2 * 13 * kHerm * 13, VRR ~455 nodes, scatter
// ~14.7k iterations (kHerm = Hermite3DCount(6) = 84) - about 2.3e5 flops
// per quartet of 169 integrals. Corrected 2026-08-23: the earlier 2.7e5 /
// 1.6e3-per-integral record over-counted the same enumeration by ~19% (the
// corrected value propagates to the GFLOPS ratio only).
inline constexpr double kClass66FlopsPerIntegral = 1.34e3;

// The machine's measured FP64 peak: numpy/OpenBLAS dgemm, n = 2048,
// 2026-08-23 - 90.6 GFLOPS. The theoretical all-core number is ~394 GFLOPS
// (6 x 2 FMA x 256-bit x ~4.1 GHz); the measured number is the acceptance
// reference (the target is stated against the machine's measured FP64 peak,
// not against the theoretical one).
inline constexpr double kMachineFmaPeakGflops = 90.6;

inline constexpr std::string_view kIChainBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    I
      5.0000000000E-01       1.0000000000E+00
END
)";

qcx::Result<qcx::molecule::Molecule> MakeIChainMolecule() {
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

// The per-fixture engine state: the pair store and the class batches are
// constants of the molecule/basis/quartet triple; every timed iteration
// only re-runs the kernels over the same output buffer.
struct EngineFixture {
    EngineFixture(qcx::molecule::Molecule moleculeValue, qcx::basisset::BasisSet basisValue) :
        molecule(std::move(moleculeValue)), basis(std::move(basisValue)) {}

    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    qcx::integrals::ShellPairList pairList;
    std::vector<qcx::integrals::internal::MdPairData> pairStore;
    std::vector<qcx::integrals::internal::MdClassBatch> batches;
    std::vector<double> values;
    std::size_t integralsPerPass = 0;
};

// BuildPairData + AssembleClassBatches, once - the mirror of the public
// entry's setup (eri_batch.cpp ComputeEriBatch), with the batch cap from
// the default EriBatchOptions. The output pointers are wired into the
// fixture's values buffer and the batch pair-store pointers are re-bound
// to the fixture's own store (the assembler's pointers reference the
// original vector).
bool PrepareEngine(EngineFixture& fixture,
                   const std::vector<qcx::integrals::ShellQuartet>& quartets) {
    auto pairList = qcx::integrals::BuildShellPairs(fixture.molecule, fixture.basis);

    if (!pairList.has_value())
    {
        return false;
    }

    auto pairStore =
        qcx::integrals::internal::BuildPairData(fixture.molecule, fixture.basis, *pairList);

    if (!pairStore.has_value())
    {
        return false;
    }

    std::vector<qcx::integrals::ShellQuartet> computed;
    auto batches = qcx::integrals::internal::AssembleClassBatches(
        *pairStore, *pairList, quartets, qcx::integrals::EriBatchOptions{}.maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return false;
    }

    fixture.pairList = std::move(*pairList);
    fixture.pairStore = std::move(*pairStore);
    fixture.batches = std::move(*batches);

    std::size_t total = 0;

    for (const qcx::integrals::internal::MdClassBatch& batch : fixture.batches)
    {
        for (const qcx::integrals::internal::MdQuartetTask& task : batch.tasks)
        {
            total +=
                fixture.pairStore[task.braPair].nFuncs * fixture.pairStore[task.ketPair].nFuncs;
        }
    }

    fixture.values.assign(total, 0.0);
    std::size_t base = 0;

    for (qcx::integrals::internal::MdClassBatch& batch : fixture.batches)
    {
        batch.pairStore = &fixture.pairStore;
        batch.outF64 = fixture.values.data() + base;

        for (const qcx::integrals::internal::MdQuartetTask& task : batch.tasks)
        {
            base += fixture.pairStore[task.braPair].nFuncs * fixture.pairStore[task.ketPair].nFuncs;
        }
    }

    fixture.integralsPerPass = CountIntegrals(computed, fixture.pairList);
    return true;
}

std::unique_ptr<EngineFixture> gClass66Fixture;
std::unique_ptr<EngineFixture> gShellsetFixture;

// The (6,6) chain fixture: quartets {iShellA, sShellA, iShellB, sShellB}
// for atomB <= atomA - one homogeneous (6,6) class of 21 quartets.
bool PrepareClass66() {
    if (gClass66Fixture != nullptr)
    {
        return true;
    }

    auto basis = qcx::basisset::ParseNwchemText(kIChainBasis);
    auto molecule = MakeIChainMolecule();

    if (!basis.has_value() || !molecule.has_value())
    {
        return false;
    }

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

    gClass66Fixture = std::make_unique<EngineFixture>(std::move(*molecule), std::move(*basis));
    return PrepareEngine(*gClass66Fixture, quartets);
}

// The shellset fixture of eri_batch_test.cpp: s/p/d shells on two centers,
// all canonical quartets - the mixed-class spread.
bool PrepareShellset() {
    if (gShellsetFixture != nullptr)
    {
        return true;
    }

    auto basis = qcx::basisset::ParseNwchemText(R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
H    P
      1.0000000000E+00       1.0000000000E+00
      3.0000000000E-01       1.0000000000E+00
H    D
      8.0000000000E-01       1.0000000000E+00
He    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
He    P
      1.0000000000E+00       1.0000000000E+00
      3.0000000000E-01       1.0000000000E+00
END
)");
    auto molecule = qcx::testing::MakeShellsetMolecule();

    if (!basis.has_value() || !molecule.has_value())
    {
        return false;
    }

    std::vector<qcx::integrals::ShellQuartet> quartets;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        return false;
    }

    const std::size_t nShells = pairList->shells.size();

    for (std::size_t a = 0; a < nShells; ++a)
    {
        for (std::size_t b = a; b < nShells; ++b)
        {
            for (std::size_t c = 0; c < nShells; ++c)
            {
                for (std::size_t d = c; d < nShells; ++d)
                {
                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    gShellsetFixture = std::make_unique<EngineFixture>(std::move(*molecule), std::move(*basis));
    return PrepareEngine(*gShellsetFixture, quartets);
}

void BenchMdEriClass66(benchmark::State& state) {
    if (!qcx::integrals::SupportsL(6))
    {
        state.SkipWithError("this build has no l = 6 classes");
        return;
    }

    if (!PrepareClass66())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto run = qcx::integrals::internal::RunBatches(gClass66Fixture->batches);

        if (!run.has_value())
        {
            state.SkipWithError(run.error().message.c_str());
            return;
        }

        sink += gClass66Fixture->values.front();
    }

    benchmark::DoNotOptimize(sink);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // kIsRate divides by the run's total time, so the value must already
    // carry the iteration count to report integrals/s (2026-08-23).
    state.counters["integrals/s"] =
        benchmark::Counter(static_cast<double>(gClass66Fixture->integralsPerPass) *
                               static_cast<double>(state.iterations()),
                           benchmark::Counter::kIsRate);

    const double rate = static_cast<double>(gClass66Fixture->integralsPerPass) *
                        static_cast<double>(state.iterations()) / elapsed;
    const double gflops = rate * kClass66FlopsPerIntegral / 1.0e9;
    std::printf("[class (6,6)] %.3e integrals/s -> %.2f GFLOPS = %.2f%% of the %.0f GFLOPS "
                "FMA peak (modeled: %.0f flops/integral)\n",
                rate,
                gflops,
                100.0 * gflops / kMachineFmaPeakGflops,
                kMachineFmaPeakGflops,
                kClass66FlopsPerIntegral);
}

void BenchMdEriShellset(benchmark::State& state) {
    if (!PrepareShellset())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto run = qcx::integrals::internal::RunBatches(gShellsetFixture->batches);

        if (!run.has_value())
        {
            state.SkipWithError(run.error().message.c_str());
            return;
        }

        sink += gShellsetFixture->values.front();
    }

    benchmark::DoNotOptimize(sink);
    // kIsRate divides by the run's total time (see BenchMdEriClass66).
    state.counters["integrals/s"] =
        benchmark::Counter(static_cast<double>(gShellsetFixture->integralsPerPass) *
                               static_cast<double>(state.iterations()),
                           benchmark::Counter::kIsRate);
}

BENCHMARK(BenchMdEriClass66)->Unit(benchmark::kSecond);
BENCHMARK(BenchMdEriShellset)->Unit(benchmark::kSecond);

} // namespace

BENCHMARK_MAIN();
