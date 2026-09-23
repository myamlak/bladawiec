// Boys-function kernel benchmarks, registered with Google Benchmark.
//
// Workloads mirror the design study: uniform (n, x) pairs with n uniform in
// [0, 32] as used by the literature benchmarks, plus a molecular
// x-distribution sampled from real benzene 6-31G(d) primitive pairs
// (NAI-style x = p*|P-C|^2 with the nuclear-attraction center C; the ERI-style
// second primitive pair of the design study is not recreated here - the
// NAI-style values dominate the x-range of interest). The grouped entry is
// measured on region-sorted arrays (the engine pattern); the unsorted
// penalty is measured by the mixed per-vector kernel in
// boys_unsorted_simd_benchmark.cpp.
#include "boys/boys_coefficients.hpp"
#include "qcx/integrals/boys.hpp"

#include <benchmark/benchmark.h>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kInputCount = 1u << 22;

struct Item {
    int n;
    double x;
};

double gSink = 0.0;

std::vector<Item> UniformInputs() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> xd(0.0, 40.0);
    std::uniform_int_distribution<int> nd(0, qcx::integrals::kMaxBoysOrder);
    std::vector<Item> items(kInputCount);

    for (auto& item : items)
    {
        item.n = nd(rng);
        item.x = xd(rng);
    }

    return items;
}

// Benzene at 6-31G(d): primitive exponents (Basis Set Exchange values) and
// geometry (C-C 1.39 A, C-H 1.09 A); x samples as in the design study.
std::vector<Item> MolecularInputs() {
    constexpr double kBohr = 1.8897261246257702;
    constexpr double kR = 1.39;
    constexpr double kCH = 1.09;
    constexpr double kCExp[] = {3047.52490,
                                457.369510,
                                103.948690,
                                29.2101550,
                                9.28666300,
                                3.16392700,
                                7.86827240,
                                1.88128850,
                                0.54424930,
                                0.16871440,
                                0.80000000};
    constexpr double kHExp[] = {18.7311370, 2.8253937, 0.6401217, 0.1612778};

    struct Primitive {
        double exponent;
        double x, y, z;
    };

    std::vector<Primitive> primitives;
    constexpr double kPi = 3.14159265358979323846;

    for (int k = 0; k < 6; ++k)
    {
        const double angle = k * kPi / 3.0;
        const double cx = kR * std::cos(angle);
        const double cy = kR * std::sin(angle);

        for (double exponent : kCExp)
        {
            primitives.push_back({exponent, cx * kBohr, cy * kBohr, 0.0});
        }
    }

    for (int k = 0; k < 6; ++k)
    {
        const double angle = k * kPi / 3.0;
        const double hx = (kR + kCH) * std::cos(angle);
        const double hy = (kR + kCH) * std::sin(angle);

        for (double exponent : kHExp)
        {
            primitives.push_back({exponent, hx * kBohr, hy * kBohr, 0.0});
        }
    }

    std::mt19937_64 rng(43);
    std::vector<Item> items;
    items.reserve(kInputCount);

    while (items.size() < kInputCount)
    {
        const Primitive& a = primitives[rng() % primitives.size()];
        const Primitive& b = primitives[rng() % primitives.size()];
        const double p = a.exponent + b.exponent;
        const double px = (a.exponent * a.x + b.exponent * b.x) / p;
        const double py = (a.exponent * a.y + b.exponent * b.y) / p;
        const double pz = (a.exponent * a.z + b.exponent * b.z) / p;
        const Primitive& c = primitives[rng() % primitives.size()];
        const double d2 =
            (px - c.x) * (px - c.x) + (py - c.y) * (py - c.y) + (pz - c.z) * (pz - c.z);
        // Geometric n: the low orders dominate real integral workloads.
        int n = 0;

        while (n < qcx::integrals::kMaxBoysOrder && (rng() & 1u) == 0)
        {
            ++n;
        }

        items.push_back({n, p * d2});
    }

    return items;
}

void RunSingle(const std::vector<Item>& items, bool f32) {
    if (f32)
    {
        for (const auto& item : items)
        {
            gSink += qcx::integrals::BoysSingleF32(item.n, static_cast<float>(item.x));
        }
    } else
    {
        for (const auto& item : items)
        {
            gSink += qcx::integrals::BoysSingle(item.n, item.x);
        }
    }
}

void RunBatch(const std::vector<Item>& items, bool f32) {
    double batchD[qcx::integrals::kMaxBoysOrder + 1];
    float batchF[qcx::integrals::kMaxBoysOrder + 1];

    if (f32)
    {
        for (const auto& item : items)
        {
            qcx::integrals::BoysAllOrdersF32(item.n, static_cast<float>(item.x), batchF);
            gSink += batchF[item.n];
        }
    } else
    {
        for (const auto& item : items)
        {
            qcx::integrals::BoysAllOrders(item.n, item.x, batchD);
            gSink += batchD[item.n];
        }
    }
}

std::vector<Item> gUniform = UniformInputs();
std::vector<Item> gMolecular = MolecularInputs();

} // namespace

static void BmBoysSingleUniform(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunSingle(gUniform, false);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gUniform.size()) * state.iterations());
}

BENCHMARK(BmBoysSingleUniform);

static void BmBoysBatchUniform(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunBatch(gUniform, false);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gUniform.size()) * state.iterations());
}

BENCHMARK(BmBoysBatchUniform);

static void BmBoysSingleMolecular(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunSingle(gMolecular, false);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gMolecular.size()) * state.iterations());
}

BENCHMARK(BmBoysSingleMolecular);

static void BmBoysBatchMolecular(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunBatch(gMolecular, false);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gMolecular.size()) * state.iterations());
}

BENCHMARK(BmBoysBatchMolecular);

static void BmBoysSingleF32Uniform(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunSingle(gUniform, true);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gUniform.size()) * state.iterations());
}

BENCHMARK(BmBoysSingleF32Uniform);

// The many-argument entry on a region-sorted stream (the engine pattern). The
// regions are ordered intervals, so A ++ B ++ C is non-decreasing and the
// sorted-arguments overload states that property and skips the sort the general
// call would pay for. The unsorted mixed variant measured a 2.3x divergence
// penalty in the design study.
namespace {

struct GroupedInputs {
    std::vector<double> sorted; // region A ++ region B ++ region C: non-decreasing
    std::vector<double> out;
    std::size_t count = 0;
};

GroupedInputs BuildGroupedInputs(int n) {
    GroupedInputs s;
    std::vector<double> xA, xB, xC;
    xA.reserve(kInputCount);
    xB.reserve(kInputCount);
    xC.reserve(kInputCount);
    std::mt19937_64 rng(44);
    std::uniform_real_distribution<double> xd(1e-4, 60.0);

    for (std::size_t i = 0; i < kInputCount; ++i)
    {
        const double x = xd(rng);

        if (x < qcx::integrals::detail::kX0)
        {
            xA.push_back(x);
        } else if (x < qcx::integrals::detail::kX1)
        {
            xB.push_back(x);
        } else
        {
            xC.push_back(x);
        }
    }

    s.sorted = xA;
    s.sorted.insert(s.sorted.end(), xB.begin(), xB.end());
    s.sorted.insert(s.sorted.end(), xC.begin(), xC.end());
    s.count = s.sorted.size();
    s.out.assign(s.count * static_cast<std::size_t>(n + 1), 0.0);
    return s;
}

GroupedInputs gGrouped = BuildGroupedInputs(8);

} // namespace

static void BmBoysAllNSortedN8(benchmark::State& state) {
    if (!qcx::integrals::BoysAvx2Available())
    {
        state.SkipWithError("AVX2 required for the grouped entry's vector lane");
        return;
    }

    constexpr int n = 8;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        qcx::integrals::BoysAllN(n,
                                 gGrouped.sorted.data(),
                                 gGrouped.out.data(),
                                 gGrouped.count,
                                 qcx::integrals::BoysSortedArgs{});
        benchmark::DoNotOptimize(gGrouped.out.data());
    }

    state.SetItemsProcessed(static_cast<int64_t>(gGrouped.count) * state.iterations());
}

BENCHMARK(BmBoysAllNSortedN8);

#if QcxIntegralsFp16
// The fp16 lane: F16/Bf16 I/O around the certified fp32 engine (the certified
// mixed-precision lane extended to the half types). Inputs round the double x
// grid to the half type - the same (n, x) pairs as the f32 lanes, so the half
// lanes report the I/O-conversion overhead on top of the same engine work.
namespace {

template <typename Half, Half (*SingleFn)(int, Half) noexcept>
void RunSingleHalf(const std::vector<Item>& items) {
    for (const auto& item : items)
    {
        gSink += static_cast<float>(SingleFn(item.n, static_cast<Half>(item.x)));
    }
}

template <typename Half, void (*BatchFn)(int, Half, Half*) noexcept>
void RunBatchHalf(const std::vector<Item>& items) {
    Half batch[qcx::integrals::kMaxBoysOrder + 1];

    for (const auto& item : items)
    {
        BatchFn(item.n, static_cast<Half>(item.x), batch);
        gSink += static_cast<float>(batch[item.n]);
    }
}

} // namespace

static void BmBoysSingleF16Uniform(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunSingleHalf<qcx::integrals::F16, qcx::integrals::BoysSingleF16>(gUniform);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gUniform.size()) * state.iterations());
}

BENCHMARK(BmBoysSingleF16Uniform);

static void BmBoysSingleF16Molecular(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunSingleHalf<qcx::integrals::F16, qcx::integrals::BoysSingleF16>(gMolecular);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gMolecular.size()) * state.iterations());
}

BENCHMARK(BmBoysSingleF16Molecular);

static void BmBoysBatchF16Uniform(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunBatchHalf<qcx::integrals::F16, qcx::integrals::BoysAllOrdersF16>(gUniform);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gUniform.size()) * state.iterations());
}

BENCHMARK(BmBoysBatchF16Uniform);

static void BmBoysBatchF16Molecular(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunBatchHalf<qcx::integrals::F16, qcx::integrals::BoysAllOrdersF16>(gMolecular);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gMolecular.size()) * state.iterations());
}

BENCHMARK(BmBoysBatchF16Molecular);

static void BmBoysSingleBf16Uniform(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunSingleHalf<qcx::integrals::Bf16, qcx::integrals::BoysSingleBf16>(gUniform);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gUniform.size()) * state.iterations());
}

BENCHMARK(BmBoysSingleBf16Uniform);

static void BmBoysBatchBf16Uniform(benchmark::State& state) {
    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        RunBatchHalf<qcx::integrals::Bf16, qcx::integrals::BoysAllOrdersBf16>(gUniform);
        benchmark::DoNotOptimize(gSink);
    }

    state.SetItemsProcessed(static_cast<int64_t>(gUniform.size()) * state.iterations());
}

BENCHMARK(BmBoysBatchBf16Uniform);
#endif // QcxIntegralsFp16

BENCHMARK_MAIN();
