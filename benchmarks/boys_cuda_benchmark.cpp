// The GPU throughput rows of the accompanying manuscript (tab:throughput),
// on the same uniform (n, x) workload as the CPU benchmark. Lanes:
//   cheb-f64   - BoysCuda::SingleF64 (the certified double lane)
//   cheb-f32   - BoysCuda::SingleF32 (the recommended GPU lane)
//   erf-f64    - erf-F0 + upward recursion (competitor scheme, kernels in
//                the companion kernels file)
//   lut-f64    - Tsuji-style gridded LUT + Taylor corrections (competitor)
//   fp16-single - BoysCuda::SingleF16 (I/O-only lane; device-pointer and
//                asynchronous like the other lanes, so the measured time is
//                the pure device-side cost on the same event protocol)
//
// Custom main(): --self-check runs the verifier against the CPU references
// and exits; the default mode runs the runs-log protocol (warmup + 3
// passes, min/median/max, median = paper cell).
//
// Self-check budgets (the lanes are compared against the CPU references):
//   cheb-f64      |out - BoysSingle|    <= 5.5e-14   (the GPU double row)
//   cheb-f32      |out - BoysSingleF32| <= 3.5e-7    (the GPU float row)
//   erf-f64       <= 1e-13 for x >= 10.0 (asserted device-lane domain;
//                    tab:boundaries' k_max = 32 boundary is 9.70 on the CPU,
//                    the device lane's turning-point rounding is ~2.3e-13 at
//                    x = 9.73, so the gate takes headroom; off-domain errors
//                    are recorded as the scheme's honest cost)
//   lut-f64       <= 1e-12               (LUT values <= 5.5e-14 plus
//                    degree-5 Taylor truncation ~1e-15)
//   fp16-single   GPU vs the shipped CPU fp16 lane, <= 3.5e-7 + 1 full ULP
//                    (the GPU-vs-CPU comparison contract; the absolute
//                    contract is pinned CPU-side by the accuracy record)
#include "boys/boys.hpp"
#include "boys/boys_cuda.hpp"
#include "boys_cuda_benchmark_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kInputCount = 1u << 22; // 4,194,304 values
constexpr int kThreads = 256;
constexpr int kPasses = 3;

struct Item {
    int n;
    double x;
};

std::vector<Item> UniformInputs() {
    // Identical workload to the CPU benchmark (boys_benchmark.cpp): n uniform
    // in [0, 32], x uniform in [0, 40], mt19937_64(42).
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> xd(0.0, 40.0);
    std::uniform_int_distribution<int> nd(0, boys::kMaxBoysOrder);
    std::vector<Item> items(kInputCount);

    for (auto& item : items)
    {
        item.n = nd(rng);
        item.x = xd(rng);
    }

    return items;
}

// The paper's reference series for the LUT rows beyond the shipped range:
// F_n(x) = 0.5 * e^{-x} * sum_l x^l / prod_{j=0}^{l} (n + j + 1/2)
// (all-positive terms, no cancellation; F_n(0) = 1/(2n+1) follows from the
// l = 0 term).
double StableSeriesF(int n, double x) {
    double sum = 1.0 / (n + 0.5);
    double term = sum;

    for (int l = 1; l <= 96; ++l)
    {
        term *= x / (n + l + 0.5);
        sum += term;
    }

    return 0.5 * std::exp(-x) * sum;
}

// Builds the 38 x 1025 LUT: rows 0..32 from BoysBatchF64 on the grid
// (certified <= 5.5e-14), rows 33..37 via the stable series above (the
// degree-5 corrections of order-32 inputs reach F_37). BatchF64 takes
// DEVICE pointers (boys_cuda.hpp), so the grid travels through device
// memory for the 1025-point batch (once per process).
bool BuildLutRows(std::vector<double>& rows) {
    rows.assign(38 * 1025, 0.0);
    std::vector<double> grid(1025);
    std::vector<int> nmax(1025, boys::kMaxBoysOrder);
    std::vector<double> batch(33 * 1025);

    for (int i = 0; i < 1025; ++i)
    {
        grid[i] = i * 0.03125;
    }

    int* dN = nullptr;
    double* dGrid = nullptr;
    double* dBatch = nullptr;
    cudaError_t e = cudaMalloc(&dN, 1025 * sizeof(int));

    if (e == cudaSuccess)
    {
        e = cudaMalloc(&dGrid, 1025 * sizeof(double));
    }

    if (e == cudaSuccess)
    {
        e = cudaMalloc(&dBatch, 33 * 1025 * sizeof(double));
    }

    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc (LUT rows): %s\n", cudaGetErrorString(e));
        return false;
    }

    e = cudaMemcpy(dN, nmax.data(), 1025 * sizeof(int), cudaMemcpyHostToDevice);

    if (e == cudaSuccess)
    {
        e = cudaMemcpy(dGrid, grid.data(), 1025 * sizeof(double), cudaMemcpyHostToDevice);
    }

    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMemcpy (LUT rows): %s\n", cudaGetErrorString(e));
        cudaFree(dN);
        cudaFree(dGrid);
        cudaFree(dBatch);
        return false;
    }

    const auto status = boys::BoysCuda::BatchF64(dN, dGrid, dBatch, 1025, nullptr);

    if (status != boys::BoysStatus::kSuccess)
    {
        std::fprintf(stderr, "BatchF64 (LUT rows): status %d\n", static_cast<int>(status));
        cudaFree(dN);
        cudaFree(dGrid);
        cudaFree(dBatch);
        return false;
    }

    // The entry is asynchronous: the readback below is stream-ordered after
    // the batch kernel on the default stream.
    e = cudaMemcpy(batch.data(), dBatch, 33 * 1025 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dN);
    cudaFree(dGrid);
    cudaFree(dBatch);

    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMemcpy (LUT rows, back): %s\n", cudaGetErrorString(e));
        return false;
    }

    for (int j = 0; j <= 32; ++j)
    {
        for (int i = 0; i < 1025; ++i)
        {
            rows[j * 1025 + i] = batch[j * 1025 + i];
        }
    }

    for (int j = 33; j <= 37; ++j)
    {
        for (int i = 0; i < 1025; ++i)
        {
            rows[j * 1025 + i] = StableSeriesF(j, grid[i]);
        }
    }

    return true;
}

struct Timing {
    double minMs;
    double medianMs;
    double maxMs;
};

Timing TimeLane(const char* name,
                int blocks,
                int* dN,
                double* dX,
                double* dOutF64,
                float* dOutF32,
                boys::F16* dF16In,
                boys::F16* dF16Out) {
    cudaEvent_t t0;
    cudaEvent_t t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    const auto runOnce = [&]() {
        if (std::strcmp(name, "cheb-f64") == 0)
        {
            const auto status = boys::BoysCuda::SingleF64(dN, dX, dOutF64, kInputCount, nullptr);

            if (status != boys::BoysStatus::kSuccess)
            {
                std::exit(2);
            }
        } else if (std::strcmp(name, "cheb-f32") == 0)
        {
            const auto status = boys::BoysCuda::SingleF32(dN, dX, dOutF32, kInputCount, nullptr);

            if (status != boys::BoysStatus::kSuccess)
            {
                std::exit(2);
            }
        } else if (std::strcmp(name, "erf-f64") == 0)
        {
            BoysBenchLaunchErfF64(dN, dX, dOutF64, kInputCount, blocks, kThreads, nullptr);
            cudaDeviceSynchronize();
        } else if (std::strcmp(name, "lut-f64") == 0)
        {
            BoysBenchLaunchLutF64(dN, dX, dOutF64, kInputCount, blocks, kThreads, nullptr);
            cudaDeviceSynchronize();
        } else if (std::strcmp(name, "fp16-single") == 0)
        {
#if BoysFp16
            const auto status =
                boys::BoysCuda::SingleF16(dN, dF16In, dF16Out, kInputCount, nullptr);

            if (status != boys::BoysStatus::kSuccess)
            {
                std::exit(2);
            }
#endif
        }
    };

    // Warmup pass (also uploads the Chebyshev tables on first use).
    runOnce();

    std::vector<double> passes(kPasses);

    for (int p = 0; p < kPasses; ++p)
    {
        cudaEventRecord(t0);
        runOnce();
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        passes[p] = ms;
    }

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    std::sort(passes.begin(), passes.end());
    Timing timing{passes.front(), passes[1], passes.back()};
    // count/median is items per millisecond = 1e3 items/s; the /1e3 below is
    // what makes the printed value the unit its field names (Mvals/s), at the
    // three decimals the runs log's rows carry.
    std::printf("kernel: %s | workload: uniform-n32-x40 | count: %zu | passes: %d | "
                "min_ms: %.3f | median_ms: %.3f | max_ms: %.3f | median_Mvals_per_s: %.3f\n",
                name,
                kInputCount,
                kPasses,
                timing.minMs,
                timing.medianMs,
                timing.maxMs,
                kInputCount / timing.medianMs / 1e3);
    return timing;
}

int SelfCheck(const std::vector<Item>& items,
              int* dN,
              double* dX,
              double* dOutF64,
              float* dOutF32,
              const std::vector<boys::F16>& f16In,
              boys::F16* dF16In,
              boys::F16* dF16Out,
              int blocks) {
    int failed = 0;

    // The device results are compared on the host, so each lane copies back
    // first (stream-ordered reads on the default stream).
    std::vector<double> outF64Host(kInputCount);
    std::vector<float> outF32Host(kInputCount);

    // cheb-f64
    {
        const auto status = boys::BoysCuda::SingleF64(dN, dX, dOutF64, kInputCount, nullptr);

        if (status != boys::BoysStatus::kSuccess)
        {
            std::fprintf(stderr, "SingleF64: status %d\n", static_cast<int>(status));
            return 1;
        }

        cudaMemcpy(
            outF64Host.data(), dOutF64, kInputCount * sizeof(double), cudaMemcpyDeviceToHost);
        double worst = 0.0;

        for (std::size_t i = 0; i < kInputCount; ++i)
        {
            const double err = std::abs(outF64Host[i] - boys::BoysSingle(items[i].n, items[i].x));
            worst = std::max(worst, err);
        }

        const bool pass = worst <= 5.5e-14;
        std::printf("self-check: cheb-f64 | max_abs_err: %.3e | budget: 5.50e-14 | %s\n",
                    worst,
                    pass ? "PASS" : "FAIL");
        failed += !pass;
    }

    // cheb-f32
    {
        const auto status = boys::BoysCuda::SingleF32(dN, dX, dOutF32, kInputCount, nullptr);

        if (status != boys::BoysStatus::kSuccess)
        {
            std::fprintf(stderr, "SingleF32: status %d\n", static_cast<int>(status));
            return 1;
        }

        cudaMemcpy(outF32Host.data(), dOutF32, kInputCount * sizeof(float), cudaMemcpyDeviceToHost);
        double worst = 0.0;

        for (std::size_t i = 0; i < kInputCount; ++i)
        {
            const double err =
                std::abs(static_cast<double>(outF32Host[i]) -
                         boys::BoysSingleF32(items[i].n, static_cast<float>(items[i].x)));
            worst = std::max(worst, err);
        }

        const bool pass = worst <= 3.5e-7;
        std::printf("self-check: cheb-f32 | max_abs_err: %.3e | budget: 3.50e-07 | %s\n",
                    worst,
                    pass ? "PASS" : "FAIL");
        failed += !pass;
    }

    // erf-f64 (competitor lane). The upward recursion from an F0 seed is the
    // paper's measured-boundary phenomenon (tab:boundaries): for k_max = 32
    // the reference recursion stays within 5e-14 for x >= 9.70 (measured on
    // the CPU). The device lane's own rounding near the turning point is a
    // little worse (2.3e-13 measured at x = 9.73, n = 32 on this T1000), so
    // the asserted domain takes headroom: x >= 10.0 at the 1e-13 budget. The
    // x >= 5 worst is recorded as the scheme's honest off-domain behavior.
    {
        constexpr double kErfBoundary32 = 10.0; // asserted domain (device lane)
        BoysBenchLaunchErfF64(dN, dX, dOutF64, kInputCount, blocks, kThreads, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(
            outF64Host.data(), dOutF64, kInputCount * sizeof(double), cudaMemcpyDeviceToHost);
        double worst = 0.0;
        double worstAll = 0.0;
        int worstN = 0;
        double worstX = 0.0;
        int worstAllN = 0;
        double worstAllX = 0.0;
        std::size_t checked = 0;

        for (std::size_t i = 0; i < kInputCount; ++i)
        {
            if (items[i].x < 5.0)
            {
                continue;
            }

            const double err = std::abs(outF64Host[i] - boys::BoysSingle(items[i].n, items[i].x));

            if (err > worstAll)
            {
                worstAll = err;
                worstAllN = items[i].n;
                worstAllX = items[i].x;
            }

            if (items[i].x < kErfBoundary32)
            {
                continue;
            }

            ++checked;

            if (err > worst)
            {
                worst = err;
                worstN = items[i].n;
                worstX = items[i].x;
            }
        }

        const bool pass = worst <= 1e-13;
        std::printf("self-check: erf-f64 (x >= 10.0, %zu pts) | max_abs_err: %.3e | budget: "
                    "1.00e-13 | %s\n",
                    checked,
                    worst,
                    pass ? "PASS" : "FAIL");
        std::printf("  erf x >= 5 worst: %.3e (n = %d, x = %.3f); asserted-domain worst at n = %d, "
                    "x = %.3f\n",
                    worstAll,
                    worstAllN,
                    worstAllX,
                    worstN,
                    worstX);
        failed += !pass;
    }

    // lut-f64 (Tsuji scheme; the gridded-Taylor region and the asymptotic
    // region are both exercised by the uniform workload)
    {
        BoysBenchLaunchLutF64(dN, dX, dOutF64, kInputCount, blocks, kThreads, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(
            outF64Host.data(), dOutF64, kInputCount * sizeof(double), cudaMemcpyDeviceToHost);
        double worst = 0.0;

        for (std::size_t i = 0; i < kInputCount; ++i)
        {
            const double err = std::abs(outF64Host[i] - boys::BoysSingle(items[i].n, items[i].x));
            worst = std::max(worst, err);
        }

        const bool pass = worst <= 1e-12;
        std::printf("self-check: lut-f64 | max_abs_err: %.3e | budget: 1.00e-12 | %s\n",
                    worst,
                    pass ? "PASS" : "FAIL");
        failed += !pass;
    }

    // fp16-single: GPU vs the shipped CPU fp16 lane on the same fp16 inputs
    // (BoysSingleF16, the CPU reference lane), budget 3.5e-7 + one full fp16
    // ULP of the CPU value — the GPU-vs-CPU comparison of the accompanying
    // test suite (the full-ULP term absorbs the fp16 rounding-boundary flips
    // between the two float engines). The absolute contract (m * 1e-7 + 1/2
    // ULP vs the exact value) is pinned CPU-side by the accuracy record.
#if BoysFp16
    {
        const auto status = boys::BoysCuda::SingleF16(dN, dF16In, dF16Out, kInputCount, nullptr);

        if (status != boys::BoysStatus::kSuccess)
        {
            std::fprintf(stderr, "SingleF16: status %d\n", static_cast<int>(status));
            return 1;
        }

        std::vector<boys::F16> f16OutHost(kInputCount);
        // Stream-ordered readback of the asynchronous kernel.
        cudaMemcpy(
            f16OutHost.data(), dF16Out, kInputCount * sizeof(boys::F16), cudaMemcpyDeviceToHost);
        double worst = 0.0;

        for (std::size_t i = 0; i < kInputCount; ++i)
        {
            const boys::F16 cpu = boys::BoysSingleF16(items[i].n, f16In[i]);
            // Full grid step of the GPU-vs-CPU comparison contract.
            const double gridStep =
                static_cast<double>(boys::NextUp(cpu)) - static_cast<double>(cpu);
            const double err =
                std::abs(static_cast<double>(f16OutHost[i]) - static_cast<double>(cpu));
            worst = std::max(worst, err - gridStep);
        }

        const bool pass = worst <= 3.5e-7;
        std::printf("self-check: fp16-single | max_abs_err_over_ulp: %.3e | budget: 3.50e-07 + 1 "
                    "ULP | %s\n",
                    worst,
                    pass ? "PASS" : "FAIL");
        failed += !pass;
    }
#endif

    return failed == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    const bool selfCheck = argc > 1 && std::strcmp(argv[1], "--self-check") == 0;

    const int deviceError = BoysBenchSetDevice(0);

    if (deviceError != 0)
    {
        std::fprintf(stderr,
                     "cudaSetDevice(0): %s\n",
                     cudaGetErrorString(static_cast<cudaError_t>(deviceError)));
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int clockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, 0);
    std::printf("GPU: %s | CC %d.%d | SMs %d | clock %.0f MHz\n",
                prop.name,
                prop.major,
                prop.minor,
                prop.multiProcessorCount,
                clockKHz * 1e-3);
    std::fflush(stdout);

    const auto items = UniformInputs();

    // LUT rows: rows 0..32 from the certified batch lane, rows 33..37 from
    // the paper's reference series.
    std::vector<double> lutRows;

    if (!BuildLutRows(lutRows))
    {
        return 1;
    }

    const int lutError = BoysBenchUploadLut(lutRows.data());

    if (lutError != 0)
    {
        std::fprintf(
            stderr, "LUT upload: %s\n", cudaGetErrorString(static_cast<cudaError_t>(lutError)));
        return 1;
    }

    const auto init = boys::BoysCuda::InitializeTables();

    if (init != boys::BoysStatus::kSuccess)
    {
        std::fprintf(stderr, "InitializeTables: status %d\n", static_cast<int>(init));
        return 1;
    }

    int* dN = nullptr;
    double* dX = nullptr;
    double* dOutF64 = nullptr;
    float* dOutF32 = nullptr;
    boys::F16* dF16In = nullptr;
    boys::F16* dF16Out = nullptr;
    cudaError_t e = cudaMalloc(&dN, kInputCount * sizeof(int));

    if (e == cudaSuccess)
    {
        e = cudaMalloc(&dX, kInputCount * sizeof(double));
    }

    if (e == cudaSuccess)
    {
        e = cudaMalloc(&dOutF64, kInputCount * sizeof(double));
    }

    if (e == cudaSuccess)
    {
        e = cudaMalloc(&dOutF32, kInputCount * sizeof(float));
    }

    if (e == cudaSuccess)
    {
        e = cudaMalloc(&dF16In, kInputCount * sizeof(boys::F16));
    }

    if (e == cudaSuccess)
    {
        e = cudaMalloc(&dF16Out, kInputCount * sizeof(boys::F16));
    }

    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(e));
        return 1;
    }

    std::vector<int> nHost(kInputCount);
    std::vector<double> xHost(kInputCount);
    std::vector<boys::F16> f16In(kInputCount);
    std::vector<boys::F16> f16Out(kInputCount);

    for (std::size_t i = 0; i < kInputCount; ++i)
    {
        nHost[i] = items[i].n;
        xHost[i] = items[i].x;
        f16In[i] = boys::F16(static_cast<float>(items[i].x));
    }

    e = cudaMemcpy(dN, nHost.data(), kInputCount * sizeof(int), cudaMemcpyHostToDevice);

    if (e == cudaSuccess)
    {
        e = cudaMemcpy(dX, xHost.data(), kInputCount * sizeof(double), cudaMemcpyHostToDevice);
    }

    if (e == cudaSuccess)
    {
        e = cudaMemcpy(
            dF16In, f16In.data(), kInputCount * sizeof(boys::F16), cudaMemcpyHostToDevice);
    }

    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "cudaMemcpy: %s\n", cudaGetErrorString(e));
        return 1;
    }

    const int blocks = static_cast<int>((kInputCount + kThreads - 1) / kThreads);

    if (selfCheck)
    {
        return SelfCheck(items, dN, dX, dOutF64, dOutF32, f16In, dF16In, dF16Out, blocks);
    }

    TimeLane("cheb-f64", blocks, dN, dX, dOutF64, dOutF32, dF16In, dF16Out);
    TimeLane("cheb-f32", blocks, dN, dX, dOutF64, dOutF32, dF16In, dF16Out);
    TimeLane("erf-f64", blocks, dN, dX, dOutF64, dOutF32, dF16In, dF16Out);
    TimeLane("lut-f64", blocks, dN, dX, dOutF64, dOutF32, dF16In, dF16Out);
#if BoysFp16
    TimeLane("fp16-single", blocks, dN, dX, dOutF64, dOutF32, dF16In, dF16Out);
#endif
    return 0;
}
