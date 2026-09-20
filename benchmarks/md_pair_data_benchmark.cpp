// The pair-data construction benchmarks: the per-pair MD table path that used
// to heap-allocate per primitive pair and now runs on fixed-size std::array
// storage.
//
//   - BuildPairData: the whole per-pair pass - geometry, the per-axis E
//     tables (PerAxisETable: a std::vector before, std::array<double,
//     kMaxPerAxisTableValues> after), and the E-table folds of
//     BuildContractedPairTransform (FoldPairETable: two std::vector<double>
//     e3d/mid before, static thread_local std::array<double,
//     kMaxFoldElements> after). The sizes are compile-time-bounded by
//     L <= 6 (kMaxShellL, md_tables_gen.hpp; QcxIntegralsLMax).
//   - FoldPairETable on one (6,6) pair: the worst-case fold - la + lb = 12,
//     e3d/mid are nCartA*nCartB*nHerm = 28*28*455 = 356,720 doubles = 2.7 MB
//     per allocation, two per fold, on the allocating path.
//
// The i-chain fixture is the same synthetic two-center i-shell basis as
// md_eri_benchmark.cpp, with a three-primitive I shell so each (6,6) pair
// exercises nine (3 x 3) primitive-table builds and nine folds.
// The 2026-08-22 measurement ("the (i,i) l = 12 pair-table fold dominates
// the class (6,6) pass time") was made with the heap-allocating fold path;
// these benchmarks record the before/after of that path.

#include "internal/md_batch.hpp"
#include "internal/md_hermite.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <string_view>
#include <vector>

namespace {

// A synthetic two-center i-shell basis: each of the kChainAtomCount atoms
// carries one i shell (l = 6, three primitives) and one s shell, so the
// (i, s) pairs form one homogeneous class-6 chain.
inline constexpr std::size_t kChainAtomCount = 6;
inline constexpr double kChainSpacing = 2.0;

inline constexpr std::string_view kIChainBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    I
      1.0000000000E+00       5.0000000000E-01
      5.0000000000E-01       3.0000000000E-01
      2.0000000000E-01       2.0000000000E-01
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

// The whole per-pair pass, once per timed iteration: BuildPairData over the
// (6,6) i-chain pair list (BuildShellPairs + BuildPairData = the same setup
// every 1e/2e entry runs; the pair store itself is a per-iteration fresh
// allocation, identical before and after the fixed-storage change - only the
// per-primitive-pair table/fold storage changed).
void BenchBuildPairData(benchmark::State& state) {
    if (!qcx::integrals::SupportsL(6))
    {
        state.SkipWithError("this build has no l = 6 classes");
        return;
    }

    auto basis = qcx::basisset::ParseNwchemText(kIChainBasis);
    auto molecule = MakeIChainMolecule();

    if (!basis.has_value() || !molecule.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        state.SkipWithError("BuildShellPairs failed");
        return;
    }

    std::size_t primitivePairs = 0;

    for (const qcx::integrals::ShellPairIndex& pair : pairList->pairs)
    {
        primitivePairs +=
            pairList->shells[pair.i].contractionCount * pairList->shells[pair.j].contractionCount;
    }

    double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto store = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);

        if (!store.has_value())
        {
            state.SkipWithError(store.error().message.c_str());
            return;
        }

        sink += static_cast<double>(store->size());
    }

    benchmark::DoNotOptimize(sink);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::size_t pairs = pairList->pairs.size();
    // kIsRate divides by the run's total time, so the value must already
    // carry the iteration count to report X/s (2026-08-23).
    state.counters["pairs/s"] =
        benchmark::Counter(static_cast<double>(pairs) * static_cast<double>(state.iterations()),
                           benchmark::Counter::kIsRate);
    state.counters["prim-pairs/s"] = benchmark::Counter(static_cast<double>(primitivePairs) *
                                                            static_cast<double>(state.iterations()),
                                                        benchmark::Counter::kIsRate);
    const double rate =
        static_cast<double>(primitivePairs) * static_cast<double>(state.iterations()) / elapsed;
    std::printf("[BuildPairData %zu pairs, %zu primitive pairs] %.3e prim-pairs/s, %.3e pairs/s\n",
                pairs,
                primitivePairs,
                rate,
                static_cast<double>(pairs) * static_cast<double>(state.iterations()) / elapsed);
}

// The isolated fold path: one (6,6) pair from the built store, all its
// primitive pairs folded per iteration (9 x FoldPairETable at la = lb = 6,
// nHerm = Hermite3DCount(12) = 455). On the allocating path every fold
// allocated two 2.7 MB std::vectors (e3d, mid); on the fixed-storage path
// both are static thread_local fixed arrays. The output vector is reused
// across iterations (caller-side, identical in both states).
void BenchFoldPairETable(benchmark::State& state) {
    if (!qcx::integrals::SupportsL(6))
    {
        state.SkipWithError("this build has no l = 6 classes");
        return;
    }

    auto basis = qcx::basisset::ParseNwchemText(kIChainBasis);
    auto molecule = MakeIChainMolecule();

    if (!basis.has_value() || !molecule.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        state.SkipWithError("BuildShellPairs failed");
        return;
    }

    auto store = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);

    if (!store.has_value())
    {
        state.SkipWithError("BuildPairData failed");
        return;
    }

    const qcx::integrals::internal::MdPairData* pair = nullptr;

    for (const qcx::integrals::internal::MdPairData& candidate : *store)
    {
        if (candidate.la == 6 && candidate.lb == 6)
        {
            pair = &candidate;
            break;
        }
    }

    if (pair == nullptr)
    {
        state.SkipWithError("no (6,6) pair in the store");
        return;
    }

    const int nHerm = qcx::integrals::internal::Hermite3DCount(pair->la + pair->lb);
    std::vector<double> folded;
    double sink = 0.0;
    const auto t0 = std::chrono::steady_clock::now();

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        for (const qcx::integrals::internal::MdPrimPair& prim : pair->primPairs)
        {
            qcx::integrals::internal::FoldPairETable(pair->la,
                                                     pair->lb,
                                                     prim.perAxisTables,
                                                     pair->isSphericalA,
                                                     pair->isSphericalB,
                                                     nHerm,
                                                     folded);
            sink += folded.front();
        }
    }

    benchmark::DoNotOptimize(sink);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const std::size_t folds = pair->primPairs.size();
    state.counters["folds/s"] =
        benchmark::Counter(static_cast<double>(folds) * static_cast<double>(state.iterations()),
                           benchmark::Counter::kIsRate);
    const double rate =
        static_cast<double>(folds) * static_cast<double>(state.iterations()) / elapsed;
    std::printf(
        "[FoldPairETable (6,6), %zu prims, %d Hermite rows] %.3e folds/s\n", folds, nHerm, rate);
}

BENCHMARK(BenchBuildPairData)->Unit(benchmark::kSecond);
BENCHMARK(BenchFoldPairETable)->Unit(benchmark::kSecond);

} // namespace

BENCHMARK_MAIN();
