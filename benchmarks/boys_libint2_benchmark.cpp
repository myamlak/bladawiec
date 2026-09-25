// The Boys-kernel paper's comparative benchmark: the matched-accuracy
// throughput comparator - the qcx BoysAllOrders ladder vs libint2's
// FmEval_Chebyshev7<double> ladder on the SAME (n, x) input streams at the
// SAME accuracy target. Custom main() rather than a Google Benchmark
// harness, following the runs-log protocol: --self-check first (the
// accuracy gate), then the timing mode on [uniform|molecular] - one warmup
// pass per side, then 3 recorded passes, min/median/max per side, median =
// the paper cell.
//
// The two streams move n and x together, so a ratio between them is not
// attributable to either one. Three further modes hold one axis still and
// sweep the other on the same protocol, one cell at a time:
// --order-sweep (n = 0..32 at each stream's own x marginal), --region-sweep
// (each x band at n = 0, 1, 2, 4, 8, 16, 24, 32) and --stratify [stream]
// (one stream cut into (n, x-band) cells of its own values, each timed, then
// summed back into the stream row the same run measured).
//
// Comparator pin (the audit's comparator choice): libint2 v2.13.1 ==
// 5fb07b4862f219749c51d59ee08073d20a56d506 (tag v2.13.1, 2026-02-03),
// consumed header-only from the canonical consumer artifact
// libint-2.13.1.tgz (https://github.com/evaleev/libint/releases/download/
// v2.13.1/libint-2.13.1.tgz). The Boys headers are LGPL-3.0-or-later
// (COPYING.LESSER); consumption is benchmark-time-only into a private,
// never-distributed benchmark exe.
//
// Convention match (both libraries' definitions align - values compare
// 1:1): the plain F_m(T) = int_0^1 u^(2m) exp(-T u^2) du with no prefactor.
// libint2 states it in-header (FmEval_Reference2's erf anchor F0 =
// (sqrt(pi)/2) erf(sqrt(T))/sqrt(T) and the (2m+1)/(2T) upward recursion,
// boys.h) and qcx's certified BoysSingle/BoysAllOrders carry the same plain
// definition (external/boys). Each side evaluates the identical per-input
// ladder F_0(x)..F_n(x) - BoysAllOrders fills out[0..n], libint2 eval fills
// Fm[0..m_max] - so a stream pass does the same work on both sides.
//
// The streams are the paper's benchmark streams, generated verbatim like
// boys_benchmark.cpp: uniform (n in [0, 32], x in [0, 40], mt19937_64(42),
// 4,194,304 inputs) and molecular (benzene 6-31G(d) NAI-style pairs,
// mt19937_64(43)). The uniform stream sits entirely inside libint2's table
// region (n <= 32 <= 40, x <= 40 <= 117); the molecular stream does not -
// its x = p*|P - C|^2 reaches 3.26e5 and 61.8% of its inputs exceed the
// table's tmax of 117, where libint2 runs its large-T recursion and qcx its
// own large-argument branch. Both sides evaluate the same F_0..F_n ladder
// over the same inputs on either stream; only the branch taken differs.
//
// Accuracy gate: both sides evaluate at their documented epsilon-level
// floors (qcx |F_hat - F| <= m*5.5e-14 per value in every region - the batch
// row of the contract table at m = 1.0, the tighter per-region column being
// the single-value entries' promise; libint2's table header documents its own
// relative floor as numeric_limits<double>::epsilon()). The
// self-check asserts max |qcx - libint2| over each stream against the shared
// budget 5.5e-14 (the qcx absolute contract is pinned by the certified
// reference suite; libint2's ~1e-16 floor leaves headroom). A convention
// mismatch would show as an O(1) difference and fail the gate.
//
// MSVC exposes the <cmath> constants (M_PI and peers) only under
// _USE_MATH_DEFINES - libint2's headers use them unguarded. The define
// arrives as a TU compile option from the CMake target below (libint2's
// own targets add it the same way); the source-level macro audit keeps
// in-source defines to the sanctioned set.
// AVX note: libint2's 4-wide Chebyshev interpolation path is compile-time
// gated on __AVX__ (boys.h), so this TU is built with /arch:AVX2 (see the
// CMake target); the banner below records which path was timed. qcx's SIMD
// BoysAllOrders lives in the linked library (boys_simd.cpp, /arch:AVX2).
// libint2's vector_x86.h pulls intrinsics via <intrin.h> on MSVC and
// <x86intrin.h> elsewhere, and its SIMD sections key on the GCC-style
// feature macros (__SSE2__, __SSE__, __AVX__) that MSVC never defines
// (__AVX__ excepted, under /arch:AVX2). The TU shims the missing MSVC
// surface in before the libint2 headers: <immintrin.h> for the AVX types
// and the feature macros so the vector classes the 4-wide path uses exist
// (the upstream include chain is GCC/Clang-only on MSVC). The fetched
// include tree is mapped to libint2's installed layout: its install rules
// place the root-level libint2_params.h / libint2_iface.h / libint2_types.h
// into libint2/.
#if defined(_MSC_VER) && defined(__AVX__)
#include <immintrin.h>
#ifndef __SSE2__
#define __SSE2__ 1
#endif
#ifndef __SSE__
#define __SSE__ 1
#endif
#endif
#include "qcx/integrals/boys.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <libint2/boys.h>
#include <random>
#include <vector>

// libint2's header-only surface carries a managed-singleton initializer
// whose hook functions (libint2_static_init / libint2_static_cleanup) are
// defined in the compiled library. A Boys-only consumer never builds the C
// API, so it supplies the trivial no-op pair - there is no C-API state to
// set up or tear down (no libint2 target is linked here).
extern "C" void libint2_static_init() {}

extern "C" void libint2_static_cleanup() {}

namespace {

constexpr std::size_t kInputCount = 1u << 22; // 4,194,304 values
constexpr int kPasses = 3;
constexpr double kSelfCheckBudget = 5.5e-14;
constexpr std::size_t kLadderSize = static_cast<std::size_t>(qcx::integrals::kMaxBoysOrder) + 1;

struct Item {
    int n;
    double x;
};

// The uniform stream. Generation verbatim from boys_benchmark.cpp
// (UniformInputs): one mt19937_64(42), n drawn first, then x, per input.
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

// The molecular stream. Generation verbatim from boys_benchmark.cpp
// (MolecularInputs): benzene 6-31G(d) primitive pairs sampled by
// mt19937_64(43), NAI-style x = p*|P - C|^2, geometric n distribution.
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

// One full-stream pass of the qcx ladder: BoysAllOrders evaluates F_0..F_n per
// input (the engine pattern); the input's own row feeds the sink.
void QcxLadderSweep(const std::vector<Item>& items, volatile double& sink) {
    std::array<double, kLadderSize> f{};

    for (const Item& item : items)
    {
        qcx::integrals::BoysAllOrders(item.n, item.x, f.data());
        sink += f[static_cast<std::size_t>(item.n)];
    }
}

// One full-stream pass of the libint2 ladder: eval fills F_0..F_m_max per
// input; the input's own row feeds the sink. The sink is observable so the
// inlined interpolation cannot be eliminated.
void Libint2LadderSweep(const libint2::FmEval_Chebyshev7<double>& fm,
                        const std::vector<Item>& items,
                        volatile double& sink) {
    std::array<double, kLadderSize> f{};

    for (const Item& item : items)
    {
        fm.eval(f.data(), item.x, item.n);
        sink += f[static_cast<std::size_t>(item.n)];
    }
}

// Max |qcx F_n(x) - libint2 F_n(x)| over one stream - the mutual
// matched-accuracy gate, which doubles as the convention-match check.
double MaxMutualDiff(const libint2::FmEval_Chebyshev7<double>& fm,
                     const std::vector<Item>& items,
                     Item& worstItem) {
    std::array<double, kLadderSize> qcxF{};
    std::array<double, kLadderSize> libint2F{};
    double worst = 0.0;
    worstItem = Item{-1, -1.0};

    for (const Item& item : items)
    {
        qcx::integrals::BoysAllOrders(item.n, item.x, qcxF.data());
        fm.eval(libint2F.data(), item.x, item.n);
        const std::size_t row = static_cast<std::size_t>(item.n);
        const double err = std::abs(qcxF[row] - libint2F[row]);

        if (err > worst)
        {
            worst = err;
            worstItem = item;
        }
    }

    return worst;
}

// The comparator's self-check row (one per stream), pinned by the same
// findstr anchors as the rest of the family: "self-check:", "| PASS".
bool PrintSelfCheckRow(const libint2::FmEval_Chebyshev7<double>& fm,
                       const char* streamName,
                       const std::vector<Item>& items) {
    Item worstItem{};
    const double err = MaxMutualDiff(fm, items, worstItem);
    const bool pass = err <= kSelfCheckBudget;
    std::printf("self-check: qcx-vs-libint2-%s | max_abs_err: %.3e | budget: 5.50e-14 | %s\n",
                streamName,
                err,
                pass ? "PASS" : "FAIL");
    std::printf("  worst at n = %d, x = %.9f\n", worstItem.n, worstItem.x);
    return pass;
}

void PrintKernelRow(const char* kernelName,
                    const char* workload,
                    const std::vector<double>& passes) {
    std::vector<double> sorted = passes;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    // count/median is items per millisecond = 1e3 items/s; the /1e3 below is
    // what makes the printed value the unit its field names (Mvals/s), at the
    // three decimals the runs log's rows carry.
    std::printf(
        "kernel: %s | workload: %s | count: %zu | passes: %d | min_ms: %.3f | median_ms: %.3f | "
        "max_ms: %.3f | median_Mvals_per_s: %.3f\n",
        kernelName,
        workload,
        kInputCount,
        kPasses,
        sorted.front(),
        median,
        sorted.back(),
        static_cast<double>(kInputCount) / median / 1e3);
}

// ---------------------------------------------------------------------------
// The sweeps: the order axis and the argument axis held apart.
//
// Both committed streams move n and x together - the uniform stream draws
// them independently over the whole ladder, the molecular stream ties n to
// the shell pair and x to the geometry - so a ratio between the two streams
// is a ratio of two joint distributions and attributes nothing to either
// variable. The modes below hold one axis fixed and sweep the other, and
// they cut the argument at the boundaries the two libraries' dispatch keys
// on rather than at round numbers.
// ---------------------------------------------------------------------------

/// The x bands: the three regions the Boys library names, split at the
/// sub-boundaries either library's dispatch has inside them.
enum class Band {
    kZero = 0, // x == 0: an explicit early branch in both libraries
    kA0, // (0, kExtendedBX0): one fit at nmax in the batch entry, then the downward recursion
    kA1, // [kExtendedBX0, kX0): the extended seed, the ladder, then the tail fits
    kB, // [kX0, kX1): the region-B seed and its ladder; libint2 still interpolating
    kC1, // [kX1, 117): the asymptotic branch against libint2's interpolation table
    kC2, // [117, inf): both asymptotic, on the same two formulas
    kCount,
};

constexpr std::size_t kBandCount = static_cast<std::size_t>(Band::kCount);

// libint2's cheb_table_tmax (boys_cheb7_v2.h): the right edge of its
// interpolation table, above which it recurses from the asymptotic F_0.
// The class keeps the constant private, so the value is restated here.
constexpr double kLibint2TableMax = 117.0;

// A ceiling over the stream x ranges: the molecular stream reaches 3.26e5,
// and the tail band's draw has to cover it.
constexpr double kTailXMax = 4.0e5;

const char* BandName(Band band) {
    switch (band)
    {
    case Band::kZero:
        return "x0";
    case Band::kA0:
        return "A0";
    case Band::kA1:
        return "A1";
    case Band::kB:
        return "B";
    case Band::kC1:
        return "C1";
    case Band::kC2:
        return "C2";
    default:
        return "all";
    }
}

// One argument's band, at the boundaries the two libraries dispatch on.
Band ClassifyX(double x) {
    if (x == 0.0)
    {
        return Band::kZero;
    }

    if (x < qcx::integrals::detail::kExtendedBX0)
    {
        return Band::kA0;
    }

    if (x < qcx::integrals::detail::kX0)
    {
        return Band::kA1;
    }

    if (x < qcx::integrals::detail::kX1)
    {
        return Band::kB;
    }

    return x < kLibint2TableMax ? Band::kC1 : Band::kC2;
}

double MedianOf(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// One cell's cost, per side, in nanoseconds per input: the per-pass readings
// and the per-pass ratios between them.
struct CellTimes {
    std::vector<double> qcxNs;
    std::vector<double> libint2Ns;
    std::vector<double> ratio;
};

// One cell's timing, in the committed protocol's shape: one warmup pass per
// side, then kPasses recorded rounds, the sides alternating inside every
// round so a ratio is read off rounds that shared the machine state.
// A pass runs the cell's items kReps times, which is how a cell too small to
// fill a pass on its own still gets a pass long enough to read.
CellTimes TimeCell(const libint2::FmEval_Chebyshev7<double>& fm,
                   const std::vector<Item>& items,
                   std::size_t reps,
                   volatile double& qcxSink,
                   volatile double& libint2Sink) {
    CellTimes times;
    times.qcxNs.resize(kPasses);
    times.libint2Ns.resize(kPasses);
    times.ratio.resize(kPasses);
    const double nsPerInput = 1e6 / static_cast<double>(items.size() * reps);

    for (std::size_t r = 0; r < reps; ++r)
    {
        QcxLadderSweep(items, qcxSink);
    }

    for (std::size_t r = 0; r < reps; ++r)
    {
        Libint2LadderSweep(fm, items, libint2Sink);
    }

    for (int p = 0; p < kPasses; ++p)
    {
        const std::size_t row = static_cast<std::size_t>(p);
        const auto qcxT0 = std::chrono::steady_clock::now();

        for (std::size_t r = 0; r < reps; ++r)
        {
            QcxLadderSweep(items, qcxSink);
        }

        const auto qcxT1 = std::chrono::steady_clock::now();
        times.qcxNs[row] =
            std::chrono::duration<double, std::milli>(qcxT1 - qcxT0).count() * nsPerInput;

        const auto libint2T0 = std::chrono::steady_clock::now();

        for (std::size_t r = 0; r < reps; ++r)
        {
            Libint2LadderSweep(fm, items, libint2Sink);
        }

        const auto libint2T1 = std::chrono::steady_clock::now();
        times.libint2Ns[row] =
            std::chrono::duration<double, std::milli>(libint2T1 - libint2T0).count() * nsPerInput;
        times.ratio[row] = times.qcxNs[row] / times.libint2Ns[row];
    }

    return times;
}

// One cell's row: both sides' minima and medians over the recorded passes,
// and the ratio as the median of the per-round ratios. The count and the
// repeat count are on the row because ns per input is what makes cells of
// different sizes comparable, and the reader has to be able to undo it.
void PrintCellRow(const char* axis,
                  const char* xLabel,
                  int n,
                  const char* band,
                  std::size_t count,
                  std::size_t reps,
                  const CellTimes& times) {
    const double qcxMin = *std::min_element(times.qcxNs.begin(), times.qcxNs.end());
    const double qcxMedian = MedianOf(times.qcxNs);
    const double libint2Min = *std::min_element(times.libint2Ns.begin(), times.libint2Ns.end());
    const double libint2Median = MedianOf(times.libint2Ns);
    const double ratioMedian = MedianOf(times.ratio);
    const double ratioMin = *std::min_element(times.ratio.begin(), times.ratio.end());
    std::printf(
        "cell: %s | x: %s | n: %d | band: %s | count: %zu | reps: %zu | qcx_min_ns: %.4f | "
        "qcx_median_ns: %.4f | libint2_min_ns: %.4f | libint2_median_ns: %.4f | ratio_median: "
        "%.4f | ratio_min: %.4f\n",
        axis,
        xLabel,
        n,
        band,
        count,
        reps,
        qcxMin,
        qcxMedian,
        libint2Min,
        libint2Median,
        ratioMedian,
        ratioMin);
}

// A cell's arguments: uniform inside the band, except the tail band, whose
// stream values span four decades and are drawn log-uniform there. Both are
// choices - they are the sweep's own input distribution, and the
// stream-weighted reading is the stratification instead.
std::vector<Item> BandInputs(int n, Band band, std::size_t count) {
    std::vector<Item> items(count);
    std::mt19937_64 rng(0x51ee7u + 131u * static_cast<std::size_t>(band) +
                        static_cast<std::size_t>(n));
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    double lo = 1e-6;
    double hi = qcx::integrals::detail::kExtendedBX0;

    if (band == Band::kA1)
    {
        lo = qcx::integrals::detail::kExtendedBX0;
        hi = qcx::integrals::detail::kX0;
    } else if (band == Band::kB)
    {
        lo = qcx::integrals::detail::kX0;
        hi = qcx::integrals::detail::kX1;
    } else if (band == Band::kC1)
    {
        lo = qcx::integrals::detail::kX1;
        hi = kLibint2TableMax;
    } else if (band == Band::kC2)
    {
        lo = kLibint2TableMax;
        hi = kTailXMax;
    }

    const bool logDrawn = band == Band::kC2;
    const double logLo = std::log(lo);

    for (Item& item : items)
    {
        item.n = n;
        item.x = band == Band::kZero
                     ? 0.0
                     : (logDrawn ? std::exp(logLo + unit(rng) * (std::log(hi) - logLo))
                                 : lo + unit(rng) * (hi - lo));
    }

    return items;
}

// The order axis at a fixed argument population: every input carries the
// same order and the population is the stream's own x marginal, so the sweep
// moves n alone.
void OrderSweepFor(const libint2::FmEval_Chebyshev7<double>& fm,
                   const std::vector<double>& xs,
                   const char* xLabel,
                   volatile double& qcxSink,
                   volatile double& libint2Sink) {
    std::vector<Item> items(xs.size());

    for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
    {
        for (std::size_t i = 0; i < xs.size(); ++i)
        {
            items[i] = Item{n, xs[i]};
        }

        const CellTimes times = TimeCell(fm, items, 1, qcxSink, libint2Sink);
        PrintCellRow("order-sweep", xLabel, n, "all", items.size(), 1, times);
    }
}

// The stream's own x values, strided down to the sweep's input count: the
// marginal the stream feeds, at a fraction of the run time.
std::vector<double> StridedMarginal(const std::vector<Item>& items, std::size_t count) {
    const std::size_t stride = std::max<std::size_t>(1, items.size() / count);
    std::vector<double> xs;
    xs.reserve(items.size() / stride + 1);

    for (std::size_t i = 0; i < items.size(); i += stride)
    {
        xs.push_back(items[i].x);
    }

    return xs;
}

// One stream's inputs, bucketed by (order, band). Each bucket keeps the
// stream's own arguments, so a bucket's x distribution is the stream's
// conditional distribution and no draw has to be assumed for it.
std::vector<std::vector<Item>> BucketStream(const std::vector<Item>& items) {
    const std::size_t orders = static_cast<std::size_t>(qcx::integrals::kMaxBoysOrder) + 1;
    std::vector<std::vector<Item>> buckets(orders * kBandCount);

    for (const Item& item : items)
    {
        const std::size_t cell = static_cast<std::size_t>(item.n) * kBandCount +
                                 static_cast<std::size_t>(ClassifyX(item.x));
        buckets[cell].push_back(item);
    }

    return buckets;
}

// The machine-state banner (the runs log's identity-block fields the exe can
// know itself): the CPU id, the logical-processor count, and the compile-time
// libint2 path actually timed.
void PrintBanner() {
    const char* cpu = std::getenv("PROCESSOR_IDENTIFIER");
    const char* nProc = std::getenv("NUMBER_OF_PROCESSORS");
    std::printf(
        "comparator: qcx BoysAllOrders vs libint2 FmEval_Chebyshev7<double> - matched accuracy\n");
    std::printf("libint2 pin: v2.13.1 = 5fb07b4862f219749c51d59ee08073d20a56d506, header-only\n");
#if defined(__AVX__)
    std::printf("libint2 path timed: 4-wide AVX Chebyshev interpolation (__AVX__ set)\n");
#else
    std::printf("libint2 path timed: scalar Chebyshev interpolation (no __AVX__)\n");
#endif

    if (cpu != nullptr && nProc != nullptr)
    {
        std::printf("machine: %s | %s logical processors\n", cpu, nProc);
    }
}

// The order sweep's input count per order: the marginal is strided down to
// it, which keeps 33 orders across 2 populations inside a few seconds.
constexpr std::size_t kOrderSweepInputs = 1u << 20;

// The region sweep's input count per cell. One order and one band is a
// homogeneous cell, so a shorter pass reads the same as a long one.
constexpr std::size_t kRegionSweepInputs = 1u << 20;

// The stratification's target pass length, its repeat cap (a big bucket is
// not run 64 times) and the size under which a bucket is left to the
// unmodelled remainder rather than timed.
constexpr std::size_t kStratifyPassInputs = 1u << 20;
constexpr std::size_t kStratifyMaxReps = 64;
constexpr std::size_t kStratifyMinInputs = 256;

// The two sub-bands of one region combined into the region's own reading,
// with the weight the region's stated draw gives the first of them: the
// per-input cost of a mixture is the arithmetic mean of its parts' per-input
// costs, so the mixture's per-pass readings mix the same way.
CellTimes MixCells(const CellTimes& first, double firstWeight, const CellTimes& second) {
    CellTimes mixed = first;

    for (std::size_t p = 0; p < mixed.qcxNs.size(); ++p)
    {
        mixed.qcxNs[p] = firstWeight * first.qcxNs[p] + (1.0 - firstWeight) * second.qcxNs[p];
        mixed.libint2Ns[p] =
            firstWeight * first.libint2Ns[p] + (1.0 - firstWeight) * second.libint2Ns[p];
        mixed.ratio[p] = mixed.qcxNs[p] / mixed.libint2Ns[p];
    }

    return mixed;
}

// The order axis at a fixed argument population: each stream's own x
// marginal, strided down, swept through the whole ladder.
void OrderSweepMode(const libint2::FmEval_Chebyshev7<double>& fm) {
    const std::vector<Item> uniform = UniformInputs();
    const std::vector<Item> molecular = MolecularInputs();
    volatile double qcxSink = 0.0;
    volatile double libint2Sink = 0.0;

    OrderSweepFor(fm,
                  StridedMarginal(uniform, kOrderSweepInputs),
                  "uniform-x-marginal",
                  qcxSink,
                  libint2Sink);
    OrderSweepFor(fm,
                  StridedMarginal(molecular, kOrderSweepInputs),
                  "molecular-x-marginal",
                  qcxSink,
                  libint2Sink);
    std::printf("sink: qcx %.17g | libint2 %.17g\n",
                static_cast<double>(qcxSink),
                static_cast<double>(libint2Sink));
}

// The argument axis at a fixed order: one cell per band, plus the two
// regions that own more than one band combined at the share the region's
// own draw gives each part.
void RegionSweepMode(const libint2::FmEval_Chebyshev7<double>& fm) {
    constexpr int kNSet[] = {0, 1, 2, 4, 8, 16, 24, 32};
    const double kX0 = qcx::integrals::detail::kX0;
    const double kX1 = qcx::integrals::detail::kX1;
    // Region A's uniform draw splits by width, region C's log-uniform one by
    // log-width; both are the shares of the draw the sweep states, not of
    // either stream.
    const double a0Share = qcx::integrals::detail::kExtendedBX0 / kX0;
    const double c1Share = std::log(kLibint2TableMax / kX1) / std::log(kTailXMax / kX1);
    volatile double qcxSink = 0.0;
    volatile double libint2Sink = 0.0;

    for (int n : kNSet)
    {
        const CellTimes zero =
            TimeCell(fm, BandInputs(n, Band::kZero, kRegionSweepInputs), 1, qcxSink, libint2Sink);
        const CellTimes a0 =
            TimeCell(fm, BandInputs(n, Band::kA0, kRegionSweepInputs), 1, qcxSink, libint2Sink);
        const CellTimes a1 =
            TimeCell(fm, BandInputs(n, Band::kA1, kRegionSweepInputs), 1, qcxSink, libint2Sink);
        const CellTimes b =
            TimeCell(fm, BandInputs(n, Band::kB, kRegionSweepInputs), 1, qcxSink, libint2Sink);
        const CellTimes c1 =
            TimeCell(fm, BandInputs(n, Band::kC1, kRegionSweepInputs), 1, qcxSink, libint2Sink);
        const CellTimes c2 =
            TimeCell(fm, BandInputs(n, Band::kC2, kRegionSweepInputs), 1, qcxSink, libint2Sink);

        PrintCellRow("region-sweep", "band", n, BandName(Band::kZero), kRegionSweepInputs, 1, zero);
        PrintCellRow("region-sweep", "band", n, BandName(Band::kA0), kRegionSweepInputs, 1, a0);
        PrintCellRow("region-sweep", "band", n, BandName(Band::kA1), kRegionSweepInputs, 1, a1);
        PrintCellRow(
            "region-sweep", "band", n, "A", kRegionSweepInputs, 1, MixCells(a0, a0Share, a1));
        PrintCellRow("region-sweep", "band", n, BandName(Band::kB), kRegionSweepInputs, 1, b);
        PrintCellRow("region-sweep", "band", n, BandName(Band::kC1), kRegionSweepInputs, 1, c1);
        PrintCellRow("region-sweep", "band", n, BandName(Band::kC2), kRegionSweepInputs, 1, c2);
        PrintCellRow(
            "region-sweep", "band", n, "C", kRegionSweepInputs, 1, MixCells(c1, c1Share, c2));
    }

    std::printf("sink: qcx %.17g | libint2 %.17g\n",
                static_cast<double>(qcxSink),
                static_cast<double>(libint2Sink));
}

// One stream cut into (order, band) cells of its own values, each timed on
// both sides, then summed back into the stream the cells came from. The
// reconstruction is a sum of parts: where it disagrees with the whole
// stream, the per-input cost is not separable over the cells.
void StratifyMode(const libint2::FmEval_Chebyshev7<double>& fm,
                  const std::vector<Item>& items,
                  const char* streamName) {
    volatile double qcxSink = 0.0;
    volatile double libint2Sink = 0.0;
    const std::size_t orderCount = static_cast<std::size_t>(qcx::integrals::kMaxBoysOrder) + 1;
    // The whole stream first, in this process and this machine state: the
    // reconstruction below is read against this row, not against a number
    // another run produced.
    const CellTimes whole = TimeCell(fm, items, 1, qcxSink, libint2Sink);
    PrintCellRow("stratify-whole", streamName, -1, "all", items.size(), 1, whole);

    const std::vector<std::vector<Item>> buckets = BucketStream(items);
    std::vector<std::size_t> bandCount(kBandCount, 0);
    std::vector<std::size_t> orderOccupancy(orderCount, 0);
    double reconQcxMs = 0.0;
    double reconLibint2Ms = 0.0;
    std::size_t cells = 0;
    std::size_t skipped = 0;

    for (std::size_t n = 0; n < orderCount; ++n)
    {
        for (std::size_t b = 0; b < kBandCount; ++b)
        {
            const std::vector<Item>& bucket = buckets[n * kBandCount + b];

            if (bucket.empty())
            {
                continue;
            }

            bandCount[b] += bucket.size();
            orderOccupancy[n] += bucket.size();

            if (bucket.size() < kStratifyMinInputs)
            {
                skipped += bucket.size();
                continue;
            }

            std::size_t reps = 1;

            while (bucket.size() * reps < kStratifyPassInputs && reps < kStratifyMaxReps)
            {
                ++reps;
            }

            const CellTimes times = TimeCell(fm, bucket, reps, qcxSink, libint2Sink);
            PrintCellRow("stratified",
                         streamName,
                         static_cast<int>(n),
                         BandName(static_cast<Band>(b)),
                         bucket.size(),
                         reps,
                         times);
            reconQcxMs += static_cast<double>(bucket.size()) * MedianOf(times.qcxNs) / 1e6;
            reconLibint2Ms += static_cast<double>(bucket.size()) * MedianOf(times.libint2Ns) / 1e6;
            ++cells;
        }
    }

    for (std::size_t b = 0; b < kBandCount; ++b)
    {
        if (bandCount[b] == 0)
        {
            continue;
        }

        std::printf("occupancy: stream %s | band: %s | count: %zu | share: %.6f\n",
                    streamName,
                    BandName(static_cast<Band>(b)),
                    bandCount[b],
                    static_cast<double>(bandCount[b]) / static_cast<double>(items.size()));
    }

    char histogram[512];
    int used = 0;

    for (std::size_t n = 0; n < orderCount && used < 400; ++n)
    {
        used += std::snprintf(histogram + used,
                              sizeof(histogram) - static_cast<std::size_t>(used),
                              "%s%zu:%zu",
                              n == 0 ? "" : ",",
                              n,
                              orderOccupancy[n]);
    }

    std::printf("occupancy: stream %s | order_histogram: %s\n", streamName, histogram);
    const double directQcxMs = MedianOf(whole.qcxNs) * static_cast<double>(items.size()) / 1e6;
    const double directLibint2Ms =
        MedianOf(whole.libint2Ns) * static_cast<double>(items.size()) / 1e6;
    std::printf("reconstruct: stream %s | inputs: %zu | cells: %zu | skipped_inputs: %zu | "
                "direct_qcx_ms: %.3f | direct_libint2_ms: %.3f | direct_ratio: %.4f | "
                "recon_qcx_ms: %.3f | recon_libint2_ms: %.3f | recon_ratio: %.4f | "
                "qcx_parts_pct: %.3f | libint2_parts_pct: %.3f\n",
                streamName,
                items.size(),
                cells,
                skipped,
                directQcxMs,
                directLibint2Ms,
                directQcxMs / directLibint2Ms,
                reconQcxMs,
                reconLibint2Ms,
                reconQcxMs / reconLibint2Ms,
                100.0 * reconQcxMs / directQcxMs,
                100.0 * reconLibint2Ms / directLibint2Ms);
}

} // namespace

int main(int argc, char** argv) {
    const bool selfCheck = argc > 1 && std::strcmp(argv[1], "--self-check") == 0;
    const char* caseArg = argc > 1 ? argv[1] : "uniform";
    const bool orderSweep = std::strcmp(caseArg, "--order-sweep") == 0;
    const bool regionSweep = std::strcmp(caseArg, "--region-sweep") == 0;
    const bool stratify = std::strcmp(caseArg, "--stratify") == 0;
    const bool stream =
        std::strcmp(caseArg, "uniform") == 0 || std::strcmp(caseArg, "molecular") == 0;

    if (!selfCheck && !orderSweep && !regionSweep && !stratify && !stream)
    {
        std::fprintf(
            stderr,
            "usage: %s [--self-check | uniform | molecular | --order-sweep | --region-sweep | "
            "--stratify [uniform|molecular]]\n",
            argv[0]);
        return 2;
    }

    PrintBanner();

    if (selfCheck)
    {
        const libint2::FmEval_Chebyshev7<double> fm(qcx::integrals::kMaxBoysOrder);
        const std::vector<Item> uniform = UniformInputs();
        const std::vector<Item> molecular = MolecularInputs();
        const bool passUniform = PrintSelfCheckRow(fm, "uniform-n32-x40", uniform);
        const bool passMolecular = PrintSelfCheckRow(fm, "molecular", molecular);
        return passUniform && passMolecular ? 0 : 1;
    }

    const libint2::FmEval_Chebyshev7<double> fm(qcx::integrals::kMaxBoysOrder);

    if (orderSweep)
    {
        OrderSweepMode(fm);
        return 0;
    }

    if (regionSweep)
    {
        RegionSweepMode(fm);
        return 0;
    }

    if (stratify)
    {
        const char* streamArg = argc > 2 ? argv[2] : "uniform";

        if (std::strcmp(streamArg, "uniform") != 0 && std::strcmp(streamArg, "molecular") != 0)
        {
            std::fprintf(stderr, "usage: %s --stratify [uniform|molecular]\n", argv[0]);
            return 2;
        }

        const bool molecular = std::strcmp(streamArg, "molecular") == 0;
        StratifyMode(fm,
                     molecular ? MolecularInputs() : UniformInputs(),
                     molecular ? "molecular" : "uniform-n32-x40");
        return 0;
    }

    const std::vector<Item> items =
        std::strcmp(caseArg, "molecular") == 0 ? MolecularInputs() : UniformInputs();
    const char* workload = std::strcmp(caseArg, "molecular") == 0 ? "molecular" : "uniform-n32-x40";

    // Runs-log protocol: one warmup pass per side, then 3 recorded passes,
    // the sides alternated within each round so both rows share the machine
    // state of every pass.
    volatile double qcxSink = 0.0;
    volatile double libint2Sink = 0.0;
    QcxLadderSweep(items, qcxSink);
    Libint2LadderSweep(fm, items, libint2Sink);
    std::vector<double> qcxPasses(kPasses);
    std::vector<double> libint2Passes(kPasses);

    for (int p = 0; p < kPasses; ++p)
    {
        const auto qcxT0 = std::chrono::steady_clock::now();
        QcxLadderSweep(items, qcxSink);
        const auto qcxT1 = std::chrono::steady_clock::now();
        qcxPasses[p] = std::chrono::duration<double, std::milli>(qcxT1 - qcxT0).count();
        const auto libint2T0 = std::chrono::steady_clock::now();
        Libint2LadderSweep(fm, items, libint2Sink);
        const auto libint2T1 = std::chrono::steady_clock::now();
        libint2Passes[p] = std::chrono::duration<double, std::milli>(libint2T1 - libint2T0).count();
    }

    PrintKernelRow("qcx-batch", workload, qcxPasses);
    PrintKernelRow("libint2-cheb7-ladder", workload, libint2Passes);
    std::printf("sink: qcx %.17g | libint2 %.17g\n",
                static_cast<double>(qcxSink),
                static_cast<double>(libint2Sink));
    return 0;
}
