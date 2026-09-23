// The Boys-kernel paper's comparative benchmark: the matched-accuracy
// throughput comparator - the qcx BoysAllOrders ladder vs libint2's
// FmEval_Chebyshev7<double> ladder on the SAME (n, x) input streams at the
// SAME accuracy target. Custom main() rather than a Google Benchmark
// harness, following the runs-log protocol: --self-check first (the
// accuracy gate), then the timing mode on [uniform|molecular] - one warmup
// pass per side, then 3 recorded passes, min/median/max per side, median =
// the paper cell.
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

} // namespace

int main(int argc, char** argv) {
    const bool selfCheck = argc > 1 && std::strcmp(argv[1], "--self-check") == 0;
    const char* caseArg = argc > 1 ? argv[1] : "uniform";

    if (!selfCheck && std::strcmp(caseArg, "uniform") != 0 &&
        std::strcmp(caseArg, "molecular") != 0)
    {
        std::fprintf(stderr, "usage: %s [--self-check | uniform | molecular]\n", argv[0]);
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

    const std::vector<Item> items =
        std::strcmp(caseArg, "molecular") == 0 ? MolecularInputs() : UniformInputs();
    const char* workload = std::strcmp(caseArg, "molecular") == 0 ? "molecular" : "uniform-n32-x40";
    const libint2::FmEval_Chebyshev7<double> fm(qcx::integrals::kMaxBoysOrder);

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
