#include "boys/boys.hpp"
#include "boys/boys_cuda.hpp"
#include "boys/boys_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#if BoysFp16
#include "boys/f16.hpp"
#endif

namespace {

constexpr std::size_t kCount = 1u << 16;
// Cross-lane agreement bar for the double lanes: each of the two is within the lane's published
// figure of the truth - the larger of 1e-13 * F_n(x) and 1e-13 * F_0(x) (README.md, "What the bound
// is, and what it is not") - so a difference between them is within twice the wider arm, and that
// is at most 2e-13 over the whole domain. Flat, because each sweep below asserts its own worst.
constexpr double kDoubleTolerance = 2e-13;
// Cross-lane budget: the CPU lane is validated <= 1.5e-7 against the reference grid, and
// the default CUDA lane holds the same 1.5e-7, so GPU-vs-CPU agreement on the default path
// holds within ~3e-7, stated at 3.5e-7. It covers the default path only, not the single
// entry's fast region-B exponential (RegionBExp::kFast), which carries a larger bound.
constexpr float kFloatTolerance = 3.5e-7f;

struct DeviceSetup {
    DeviceSetup() {
        cudaError_t error = cudaMalloc(&n, kCount * sizeof(int));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&x, kCount * sizeof(double));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&outF32, kCount * (boys::kMaxBoysOrder + 1) * sizeof(float));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&outF64, kCount * (boys::kMaxBoysOrder + 1) * sizeof(double));
        EXPECT_EQ(error, cudaSuccess);

        std::mt19937_64 rng(20260817);
        std::uniform_real_distribution<double> xd(1e-4, 60.0);
        std::vector<int> hostN(kCount);
        std::vector<double> hostX(kCount);

        for (std::size_t i = 0; i < kCount; ++i)
        {
            hostN[i] = static_cast<int>(rng() % (boys::kMaxBoysOrder + 1));
            hostX[i] = xd(rng);
        }
        // The x == 0 device path is a dedicated branch in every kernel.
        hostN[0] = 3;
        hostX[0] = 0.0;
        hostN[1] = 17;
        hostX[1] = 0.0;
        cudaMemcpy(n, hostN.data(), kCount * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(x, hostX.data(), kCount * sizeof(double), cudaMemcpyHostToDevice);
    }

    ~DeviceSetup() {
        cudaFree(n);
        cudaFree(x);
        cudaFree(outF32);
        cudaFree(outF64);
    }

    int* n = nullptr;
    double* x = nullptr;
    float* outF32 = nullptr;
    double* outF64 = nullptr;
};

#if BoysFp16
// The fp16 lane compares against the CPU fp16 lane: both compute in float and
// round to fp16, so the cross-lane budget is the F32 one above plus one ULP of
// quantization (HalfUlp is the full grid step NextUp(x) - x, not half of it).
double HalfUlp(boys::F16 x) {
    return static_cast<double>(boys::NextUp(x)) - static_cast<double>(x);
}

double F16Tolerance(boys::F16 cpuValue) {
    return 3.5e-7 + HalfUlp(cpuValue);
}
#endif // BoysFp16

// The uniform-order entries' precondition, made concrete: non-decreasing
// arguments covering the zero path, a dense sweep of region A below kX0, both
// exact region boundaries and a sweep of region C above kX1.
std::vector<double> SortedArguments() {
    std::mt19937_64 rng(20260922);
    std::uniform_real_distribution<double> below(1e-4, boys::detail::kX0);
    std::uniform_real_distribution<double> above(boys::detail::kX1, 60.0);
    const std::size_t lower = kCount / 2;
    const std::size_t upper = kCount - lower - 3;
    std::vector<double> x;
    x.reserve(kCount);
    x.push_back(0.0);

    for (std::size_t i = 0; i < lower; ++i)
    {
        x.push_back(below(rng));
    }

    std::sort(x.begin() + 1, x.end());
    x.push_back(boys::detail::kX0);
    x.push_back(boys::detail::kX1);
    std::vector<double> tail(upper);

    for (double& value : tail)
    {
        value = above(rng);
    }

    std::sort(tail.begin(), tail.end());
    x.insert(x.end(), tail.begin(), tail.end());
    return x;
}

// Device buffers for the uniform-order entries: one top order for the whole
// batch, so the output is one (kMaxBoysOrder + 1) x count plane set.
struct AllNDeviceSetup {
    explicit AllNDeviceSetup(const std::vector<double>& hostX) : count(hostX.size()) {
        cudaError_t error = cudaMalloc(&x, count * sizeof(double));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&outF32, count * (boys::kMaxBoysOrder + 1) * sizeof(float));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&outF64, count * (boys::kMaxBoysOrder + 1) * sizeof(double));
        EXPECT_EQ(error, cudaSuccess);
        cudaMemcpy(x, hostX.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    }

    ~AllNDeviceSetup() {
        cudaFree(x);
        cudaFree(outF32);
        cudaFree(outF64);
    }

    std::size_t count = 0;
    double* x = nullptr;
    float* outF32 = nullptr;
    double* outF64 = nullptr;
};

} // namespace

TEST(BoysCudaTest, SingleF32MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(boys::BoysCuda::SingleF32(setup.n, setup.x, setup.outF32, kCount, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<float> hostOut(kCount);
    cudaMemcpy(hostOut.data(), setup.outF32, kCount * sizeof(float), cudaMemcpyDeviceToHost);
    float worst = 0.0f;

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const float cpu = boys::BoysSingleF32(hostN[i], static_cast<float>(hostX[i]));
        worst = std::max(worst, std::abs(hostOut[i] - cpu));
    }

    EXPECT_LE(worst, kFloatTolerance);
    std::printf("SingleF32 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, SingleF32ExpOptionIsCertifiedAtTheRegionBBoundary) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    // The first float of region B, where the two options separate and the recurrence's condition
    // number peaks (7.6e4 at this order): F_32(x) is itself at the level of a seed error there and
    // the ladder amplifies it by that factor. The double lane is the reference, its own error
    // within the lane's published figure, six orders under.
    const int n = boys::kMaxBoysOrder;
    const double x = static_cast<double>(static_cast<float>(boys::detail::kX0));
    std::vector<double> reference(static_cast<std::size_t>(n) + 1);
    boys::BoysAllOrders(n, x, reference.data());
    const double want = reference[static_cast<std::size_t>(n)];

    const auto evaluate = [&](boys::RegionBExp option) {
        int* deviceN = nullptr;
        double* deviceX = nullptr;
        float* deviceOut = nullptr;
        const int hostN = n;
        EXPECT_EQ(cudaMalloc(&deviceN, sizeof(int)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&deviceX, sizeof(double)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&deviceOut, sizeof(float)), cudaSuccess);
        cudaMemcpy(deviceN, &hostN, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceX, &x, sizeof(double), cudaMemcpyHostToDevice);

        const auto status =
            option == boys::RegionBExp::kFast
                ? boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>(
                      deviceN, deviceX, deviceOut, 1, nullptr)
                : boys::BoysCuda::SingleF32(deviceN, deviceX, deviceOut, 1, nullptr);
        EXPECT_EQ(status, boys::BoysStatus::kSuccess);
        cudaDeviceSynchronize();

        float got = 0.0f;
        cudaMemcpy(&got, deviceOut, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(deviceN);
        cudaFree(deviceX);
        cudaFree(deviceOut);
        return static_cast<double>(got);
    };

    const double accurate = evaluate(boys::RegionBExp::kAccurate);
    const double fast = evaluate(boys::RegionBExp::kFast);

    // Both options run, and they are two arithmetics, not one under two names.
    EXPECT_NE(accurate, fast);

    // Both hold their documented bounds: the lane's 1.5e-7 for the default, and the
    // lane's plus the corrected seed's own 8e-8 for the fast one.
    EXPECT_LE(std::abs(accurate - want), 1.5e-7);
    EXPECT_LE(std::abs(fast - want), 1.5e-7 + 8e-8);

    // This is the cell the bare approximation returned the wrong sign at, and why
    // the fast option carries a correction: with it the return has the value's sign.
    EXPECT_GT(fast * want, 0.0);
    EXPECT_LT(std::abs(fast - accurate), std::abs(want));

    std::printf("SingleF32 at the region-B boundary: accurate %.9g, fast %.9g, value %.9g\n",
                accurate,
                fast,
                want);
}

// Outside region A the single entry's seed and ladder are the all-orders body's.
// Region A is excluded because the seeds differ there by design: the batch seeds
// its downward recursion from the double piece table, the single entry from float.
TEST(BoysCudaTest, SingleF32DefaultIsTheBatchArithmeticOutsideRegionA) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = 512;
    // Arguments that are exactly floats at or above the first float of region B, so
    // both entries evaluate the same argument; regions B and C are both covered.
    const float firstB = static_cast<float>(boys::detail::kX0);
    std::vector<double> hostX(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        const float value =
            firstB + (60.0f - firstB) * static_cast<float>(i) / static_cast<float>(count - 1);
        hostX[i] = static_cast<double>(value);
    }

    const std::size_t cells = count * static_cast<std::size_t>(nmax + 1);
    std::vector<int> gridN(cells);
    std::vector<double> gridX(cells);
    std::vector<int> batchN(count, nmax);

    for (int order = 0; order <= nmax; ++order)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::size_t e = static_cast<std::size_t>(order) * count + i;
            gridN[e] = order;
            gridX[e] = hostX[i];
        }
    }

    int* deviceN = nullptr;
    int* deviceBatchN = nullptr;
    double* deviceX = nullptr;
    double* deviceXGrid = nullptr;
    float* deviceOut = nullptr;
    ASSERT_EQ(cudaMalloc(&deviceN, cells * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceBatchN, count * sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceX, count * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceXGrid, cells * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&deviceOut, cells * sizeof(float)), cudaSuccess);
    cudaMemcpy(deviceN, gridN.data(), cells * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBatchN, batchN.data(), count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceX, hostX.data(), count * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceXGrid, gridX.data(), cells * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<float> single(cells);
    ASSERT_EQ(boys::BoysCuda::SingleF32(deviceN, deviceXGrid, deviceOut, cells, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();
    cudaMemcpy(single.data(), deviceOut, cells * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> batch(cells);
    ASSERT_EQ(boys::BoysCuda::AllOrdersF32(deviceBatchN, deviceX, deviceOut, count, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();
    cudaMemcpy(batch.data(), deviceOut, cells * sizeof(float), cudaMemcpyDeviceToHost);

    std::size_t differing = 0;

    for (std::size_t e = 0; e < cells; ++e)
    {
        if (single[e] != batch[e])
        {
            ++differing;
        }
    }

    EXPECT_EQ(differing, std::size_t{0});
    std::printf(
        "SingleF32 vs AllOrdersF32 outside region A: %zu of %zu cells differ\n", differing, cells);

    cudaFree(deviceN);
    cudaFree(deviceBatchN);
    cudaFree(deviceX);
    cudaFree(deviceXGrid);
    cudaFree(deviceOut);
}

TEST(BoysCudaTest, SingleF64MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    ASSERT_EQ(boys::BoysCuda::SingleF64(setup.n, setup.x, setup.outF64, kCount, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> hostOut(kCount);
    cudaMemcpy(hostOut.data(), setup.outF64, kCount * sizeof(double), cudaMemcpyDeviceToHost);
    double worst = 0.0;

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const double cpu = boys::BoysSingle(hostN[i], hostX[i]);
        worst = std::max(worst, std::abs(hostOut[i] - cpu));
    }

    EXPECT_LE(worst, kDoubleTolerance);
    std::printf("SingleF64 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, AllOrdersF32MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    ASSERT_EQ(boys::BoysCuda::AllOrdersF32(setup.n, setup.x, setup.outF32, kCount, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<float> hostOut(kCount * (boys::kMaxBoysOrder + 1));
    cudaMemcpy(hostOut.data(),
               setup.outF32,
               kCount * (boys::kMaxBoysOrder + 1) * sizeof(float),
               cudaMemcpyDeviceToHost);
    float worst = 0.0f;
    std::vector<float> cpuBatch(boys::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boys::BoysAllOrdersF32(hostN[i], static_cast<float>(hostX[i]), cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            worst = std::max(worst, std::abs(hostOut[k * kCount + i] - cpuBatch[k]));
        }
    }

    EXPECT_LE(worst, kFloatTolerance);
    std::printf("AllOrdersF32 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, AllOrdersF64MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(boys::BoysCuda::AllOrdersF64(setup.n, setup.x, setup.outF64, kCount, stream),
              boys::BoysStatus::kSuccess);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaDeviceSynchronize();

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> hostOut(kCount * (boys::kMaxBoysOrder + 1));
    cudaMemcpy(hostOut.data(),
               setup.outF64,
               kCount * (boys::kMaxBoysOrder + 1) * sizeof(double),
               cudaMemcpyDeviceToHost);
    double worst = 0.0;
    std::vector<double> cpuBatch(boys::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boys::BoysAllOrders(hostN[i], hostX[i], cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            worst = std::max(worst, std::abs(hostOut[k * kCount + i] - cpuBatch[k]));
        }
    }

    EXPECT_LE(worst, kDoubleTolerance);
    std::printf("AllOrdersF64 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, AllNF64MatchesCpuAllN) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<double> hostX = SortedArguments();
    ASSERT_TRUE(std::is_sorted(hostX.begin(), hostX.end()));
    const std::size_t count = hostX.size();
    AllNDeviceSetup setup(hostX);
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    ASSERT_EQ(boys::BoysCuda::AllNF64(boys::kMaxBoysOrder, setup.x, setup.outF64, count, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<double> hostOut(count * (boys::kMaxBoysOrder + 1));
    cudaMemcpy(
        hostOut.data(), setup.outF64, hostOut.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // The CPU twin over the same arguments, compared in the shipped layout: every
    // plane of every argument, so an unwritten plane fails too.
    std::vector<double> cpuOut(hostOut.size());
    boys::BoysAllN(boys::kMaxBoysOrder, hostX.data(), cpuOut.data(), count, boys::BoysSortedArgs{});
    double worst = 0.0;
    std::size_t worstAt = 0;

    for (std::size_t j = 0; j < hostOut.size(); ++j)
    {
        const double error = std::abs(hostOut[j] - cpuOut[j]);

        if (error > worst)
        {
            worst = error;
            worstAt = j;
        }
    }

    EXPECT_LE(worst, kDoubleTolerance)
        << "i=" << worstAt % count << " k=" << worstAt / count << " x=" << hostX[worstAt % count];
    std::printf("AllNF64 GPU vs CPU BoysAllN: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, AllNF32MatchesCpuAllOrders) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    const std::vector<double> hostX = SortedArguments();
    ASSERT_TRUE(std::is_sorted(hostX.begin(), hostX.end()));
    const std::size_t count = hostX.size();
    AllNDeviceSetup setup(hostX);
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    ASSERT_EQ(boys::BoysCuda::AllNF32(boys::kMaxBoysOrder, setup.x, setup.outF32, count, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<float> hostOut(count * (boys::kMaxBoysOrder + 1));
    cudaMemcpy(
        hostOut.data(), setup.outF32, hostOut.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // The float lane's shape twin is the per-element entry at one order for
    // the batch (the CPU surface has no uniform-order float batch).
    std::vector<float> cpuBatch(boys::kMaxBoysOrder + 1);
    float worst = 0.0f;

    for (std::size_t i = 0; i < count; ++i)
    {
        boys::BoysAllOrdersF32(boys::kMaxBoysOrder, static_cast<float>(hostX[i]), cpuBatch.data());

        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            worst = std::max(worst, std::abs(hostOut[k * count + i] - cpuBatch[k]));
        }
    }

    EXPECT_LE(worst, kFloatTolerance);
    std::printf("AllNF32 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, AllNF32IgnoresTheArgumentOrder) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    // The non-decreasing precondition is a performance one (one classification path
    // per warp), not a correctness one: shuffled order must return the same planes.
    std::vector<double> hostX = SortedArguments();
    std::mt19937_64 rng(20260923);
    std::shuffle(hostX.begin(), hostX.end(), rng);
    ASSERT_FALSE(std::is_sorted(hostX.begin(), hostX.end()));
    const std::size_t count = hostX.size();
    AllNDeviceSetup setup(hostX);
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    ASSERT_EQ(boys::BoysCuda::AllNF32(boys::kMaxBoysOrder, setup.x, setup.outF32, count, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<float> hostOut(count * (boys::kMaxBoysOrder + 1));
    cudaMemcpy(
        hostOut.data(), setup.outF32, hostOut.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> cpuBatch(boys::kMaxBoysOrder + 1);
    float worst = 0.0f;
    std::size_t worstAt = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        boys::BoysAllOrdersF32(boys::kMaxBoysOrder, static_cast<float>(hostX[i]), cpuBatch.data());

        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            const float error = std::abs(hostOut[k * count + i] - cpuBatch[k]);

            if (error > worst)
            {
                worst = error;
                worstAt = i;
            }
        }
    }

    EXPECT_LE(worst, kFloatTolerance) << "i=" << worstAt << " x=" << hostX[worstAt];
    std::printf("AllNF32 shuffled vs CPU: worst |diff| = %.3e\n", worst);
}

#if BoysFp16
// Device buffers shared by the fp16 lane tests: every entry takes device
// pointers (cudaMalloc/cudaMemcpy here — the caller owns the memory) and is
// asynchronous, so each test synchronizes the stream before reading back.
struct F16DeviceSetup {
    F16DeviceSetup(std::size_t count) {
        cudaError_t error = cudaMalloc(&n, count * sizeof(int));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&x, count * sizeof(boys::F16));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&out, count * sizeof(boys::F16));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&batchOut, count * (boys::kMaxBoysOrder + 1) * sizeof(boys::F16));
        EXPECT_EQ(error, cudaSuccess);
    }

    ~F16DeviceSetup() {
        cudaFree(n);
        cudaFree(x);
        cudaFree(out);
        cudaFree(batchOut);
    }

    int* n = nullptr;
    boys::F16* x = nullptr;
    boys::F16* out = nullptr;
    boys::F16* batchOut = nullptr;
};

void FillF16Inputs(std::vector<int>& hostN, std::vector<boys::F16>& hostX, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> xd(1e-4f, 60.0f);

    for (std::size_t i = 0; i < hostN.size(); ++i)
    {
        hostN[i] = static_cast<int>(rng() % (boys::kMaxBoysOrder + 1));
        hostX[i] = static_cast<boys::F16>(xd(rng));
    }
    // The x == 0 device path is a dedicated branch in every kernel.
    hostN[0] = 3;
    hostX[0] = boys::F16{0.0f};
    hostN[1] = 17;
    hostX[1] = boys::F16{0.0f};
}

void UploadF16Inputs(F16DeviceSetup& setup,
                     const std::vector<int>& hostN,
                     const std::vector<boys::F16>& hostX) {
    cudaMemcpy(setup.n, hostN.data(), hostN.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(setup.x, hostX.data(), hostX.size() * sizeof(boys::F16), cudaMemcpyHostToDevice);
}

TEST(BoysCudaTest, SingleF16MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<boys::F16> hostX(kCount);
    std::vector<boys::F16> hostOut(kCount);
    FillF16Inputs(hostN, hostX, 20260823);
    F16DeviceSetup setup(kCount);
    UploadF16Inputs(setup, hostN, hostX);

    ASSERT_EQ(boys::BoysCuda::SingleF16(setup.n, setup.x, setup.out, kCount, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();
    cudaMemcpy(hostOut.data(), setup.out, kCount * sizeof(boys::F16), cudaMemcpyDeviceToHost);

    double worst = 0.0;

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const boys::F16 cpu = boys::BoysSingleF16(hostN[i], hostX[i]);
        const double error = std::abs(static_cast<double>(hostOut[i]) - static_cast<double>(cpu));
        EXPECT_LE(error, F16Tolerance(cpu))
            << "i=" << i << " n=" << hostN[i] << " x=" << static_cast<float>(hostX[i]);
        worst = std::max(worst, error);
    }

    std::printf("SingleF16 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, AllOrdersF16MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<boys::F16> hostX(kCount);
    std::vector<boys::F16> hostOut(kCount * (boys::kMaxBoysOrder + 1));
    FillF16Inputs(hostN, hostX, 20260824);
    F16DeviceSetup setup(kCount);
    UploadF16Inputs(setup, hostN, hostX);

    ASSERT_EQ(boys::BoysCuda::AllOrdersF16(setup.n, setup.x, setup.batchOut, kCount, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();
    cudaMemcpy(
        hostOut.data(), setup.batchOut, hostOut.size() * sizeof(boys::F16), cudaMemcpyDeviceToHost);

    double worst = 0.0;
    std::vector<boys::F16> cpuBatch(boys::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boys::BoysAllOrdersF16(hostN[i], hostX[i], cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            const double error = std::abs(static_cast<double>(hostOut[k * kCount + i]) -
                                          static_cast<double>(cpuBatch[k]));
            EXPECT_LE(error, F16Tolerance(cpuBatch[k]))
                << "i=" << i << " k=" << k << " x=" << static_cast<float>(hostX[i]);
            worst = std::max(worst, error);
        }
    }

    std::printf("AllOrdersF16 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, AllOrdersF16MatchesCpuOnAStream) {
    // The async contract: a non-default stream must produce the same values
    // as the default-stream path once the stream is synchronized.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<boys::F16> hostX(kCount);
    std::vector<boys::F16> hostOut(kCount * (boys::kMaxBoysOrder + 1));
    FillF16Inputs(hostN, hostX, 20260824);
    F16DeviceSetup setup(kCount);
    UploadF16Inputs(setup, hostN, hostX);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(boys::BoysCuda::AllOrdersF16(setup.n, setup.x, setup.batchOut, kCount, stream),
              boys::BoysStatus::kSuccess);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaDeviceSynchronize();
    cudaMemcpy(
        hostOut.data(), setup.batchOut, hostOut.size() * sizeof(boys::F16), cudaMemcpyDeviceToHost);

    std::vector<boys::F16> cpuBatch(boys::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boys::BoysAllOrdersF16(hostN[i], hostX[i], cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            EXPECT_LE(std::abs(static_cast<double>(hostOut[k * kCount + i]) -
                               static_cast<double>(cpuBatch[k])),
                      F16Tolerance(cpuBatch[k]))
                << "i=" << i << " k=" << k << " x=" << static_cast<float>(hostX[i]);
        }
    }
}

TEST(BoysCudaTest, AllNF16MatchesCpuAllOrders) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    std::vector<boys::F16> hostX;
    hostX.reserve(kCount);

    for (const double x : SortedArguments())
    {
        hostX.push_back(boys::F16{static_cast<float>(x)});
    }

    ASSERT_TRUE(std::is_sorted(hostX.begin(), hostX.end()));
    F16DeviceSetup setup(kCount);
    cudaMemcpy(setup.x, hostX.data(), hostX.size() * sizeof(boys::F16), cudaMemcpyHostToDevice);

    ASSERT_EQ(
        boys::BoysCuda::AllNF16(boys::kMaxBoysOrder, setup.x, setup.batchOut, kCount, nullptr),
        boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<boys::F16> hostOut(kCount * (boys::kMaxBoysOrder + 1));
    cudaMemcpy(
        hostOut.data(), setup.batchOut, hostOut.size() * sizeof(boys::F16), cudaMemcpyDeviceToHost);

    std::vector<boys::F16> cpuBatch(boys::kMaxBoysOrder + 1);
    double worst = 0.0;

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boys::BoysAllOrdersF16(boys::kMaxBoysOrder, hostX[i], cpuBatch.data());

        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            const double error = std::abs(static_cast<double>(hostOut[k * kCount + i]) -
                                          static_cast<double>(cpuBatch[k]));
            EXPECT_LE(error, F16Tolerance(cpuBatch[k]))
                << "i=" << i << " k=" << k << " x=" << static_cast<float>(hostX[i]);
            worst = std::max(worst, error);
        }
    }

    std::printf("AllNF16 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, CountZeroIsANoOpInEveryEntry) {
    // One behaviour for every family: count == 0 queues no kernel, writes nothing,
    // and returns kSuccess. Buffers are poisoned first so a write would be seen; a
    // zero-block launch is a CUDA error, which is why the host side short-circuits.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    constexpr std::size_t kProbe = 64;
    constexpr int kPoison = 0xAB;
    const std::size_t bytes[3] = {
        kProbe * sizeof(double), kProbe * sizeof(float), kProbe * sizeof(boys::F16)};
    void* deviceOut[3] = {nullptr, nullptr, nullptr};

    for (int family = 0; family < 3; ++family)
    {
        ASSERT_EQ(cudaMalloc(&deviceOut[family], bytes[family]), cudaSuccess);
        ASSERT_EQ(cudaMemset(deviceOut[family], kPoison, bytes[family]), cudaSuccess);
    }

    auto* out64 = static_cast<double*>(deviceOut[0]);
    auto* out32 = static_cast<float*>(deviceOut[1]);
    auto* out16 = static_cast<boys::F16*>(deviceOut[2]);
    int n = 1;
    double x = 1.0;
    boys::F16 y = boys::F16{1.0f};

    EXPECT_EQ(boys::BoysCuda::SingleF64(&n, &x, out64, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllOrdersF64(&n, &x, out64, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllNF64(1, &x, out64, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::SingleF32(&n, &x, out32, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllOrdersF32(&n, &x, out32, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllNF32(1, &x, out32, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::SingleF16(&n, &y, out16, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllOrdersF16(&n, &y, out16, 0, nullptr), boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllNF16(1, &y, out16, 0, nullptr), boys::BoysStatus::kSuccess);

    for (int family = 0; family < 3; ++family)
    {
        std::vector<unsigned char> hostOut(bytes[family], 0);
        ASSERT_EQ(
            cudaMemcpy(hostOut.data(), deviceOut[family], bytes[family], cudaMemcpyDeviceToHost),
            cudaSuccess);

        for (std::size_t i = 0; i < bytes[family]; ++i)
        {
            ASSERT_EQ(hostOut[i], static_cast<unsigned char>(kPoison))
                << "family " << family << " byte " << i;
        }

        ASSERT_EQ(cudaFree(deviceOut[family]), cudaSuccess);
    }
}

TEST(BoysCudaTest, AllNChecksTheOrder) {
    // The uniform-order entries' order is a host scalar, so it is checked
    // before any device pointer is touched — these calls need no device memory.
    // The last three pin the order of the checks: an out-of-range nmax is
    // kInvalidArgument at count == 0 too, where a valid nmax is a no-op.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    double x = 1.0;
    double out64 = 0.0;
    float out32 = 0.0f;
    ASSERT_EQ(boys::BoysCuda::AllNF64(-1, &x, &out64, 1, nullptr),
              boys::BoysStatus::kInvalidArgument);
    ASSERT_EQ(boys::BoysCuda::AllNF64(boys::kMaxBoysOrder + 1, &x, &out64, 1, nullptr),
              boys::BoysStatus::kInvalidArgument);
    ASSERT_EQ(boys::BoysCuda::AllNF32(-1, &x, &out32, 1, nullptr),
              boys::BoysStatus::kInvalidArgument);
    ASSERT_EQ(boys::BoysCuda::AllNF32(boys::kMaxBoysOrder + 1, &x, &out32, 1, nullptr),
              boys::BoysStatus::kInvalidArgument);
    boys::F16 y = boys::F16{1.0f};
    ASSERT_EQ(boys::BoysCuda::AllNF16(-1, &y, &y, 1, nullptr), boys::BoysStatus::kInvalidArgument);
    ASSERT_EQ(boys::BoysCuda::AllNF16(boys::kMaxBoysOrder + 1, &y, &y, 1, nullptr),
              boys::BoysStatus::kInvalidArgument);

    ASSERT_EQ(boys::BoysCuda::AllNF64(-1, &x, &out64, 0, nullptr),
              boys::BoysStatus::kInvalidArgument);
    ASSERT_EQ(boys::BoysCuda::AllNF32(boys::kMaxBoysOrder + 1, &x, &out32, 0, nullptr),
              boys::BoysStatus::kInvalidArgument);
    ASSERT_EQ(boys::BoysCuda::AllNF16(-1, &y, &y, 0, nullptr), boys::BoysStatus::kInvalidArgument);
}
#endif // BoysFp16

// The axis states that every entry of the space runs every form. accuracy.hpp says the refined form
// is bit-identical to exact division and the plain one rounds twice where the exact form rounds
// once - a difference no published bound can see - so the check crosses entries with the three
// forms and counts the bits that disagree. The negative control (BOYS_CUDA_TEST_DROP_DIVISION_FORM)
// hands every launch kExactDivision.

namespace {

// The forms, in the axis's own order, read off the library's enumeration rather
// than listed from memory.
constexpr std::array<boys::DivisionForm, 3> kDivisionForms = {
    boys::DivisionForm::kExactDivision,
    boys::DivisionForm::kPlainReciprocal,
    boys::DivisionForm::kRefinedReciprocal,
};

/// The form a launch actually runs. The negative control named above makes this
/// answer kExactDivision for every form asked for: the axis is still crossed,
/// every call still succeeds and every count below is still taken, and only the
/// arithmetic the kernel runs is the same three times.
boys::DivisionForm LaunchedForm(boys::DivisionForm asked) noexcept {
#if defined(BOYS_CUDA_TEST_DROP_DIVISION_FORM)
    static_cast<void>(asked);
    return boys::DivisionForm::kExactDivision;
#else
    return asked;
#endif
}

/// A value's bits - the comparison this check makes. A float's four bytes are read
/// into the low half of the word.
template <typename T> std::uint64_t BitsOf(T value) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(T));
    return bits;
}

/// The distance between two values counted in representable steps: 0 when they are
/// the same value, 1 when they are adjacent. An IEEE value's bit pattern read as a
/// sign-magnitude integer is monotone in the value, and the map below turns it into
/// a plain ordinal, so the distance is a subtraction. It is what turns "the plain
/// form differs" into "the plain form differs by at most this much".
template <typename T> std::uint64_t UlpDistance(T a, T b) noexcept {
    if constexpr (std::is_same_v<T, float>)
    {
        std::uint32_t ua = 0;
        std::uint32_t ub = 0;
        std::memcpy(&ua, &a, sizeof(float));
        std::memcpy(&ub, &b, sizeof(float));
        const auto ordinal = [](std::uint32_t u) -> std::uint32_t {
            return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
        };
        const std::uint32_t first = ordinal(ua);
        const std::uint32_t second = ordinal(ub);
        return first > second ? first - second : second - first;
    } else
    {
        std::uint64_t ua = 0;
        std::uint64_t ub = 0;
        std::memcpy(&ua, &a, sizeof(double));
        std::memcpy(&ub, &b, sizeof(double));
        const auto ordinal = [](std::uint64_t u) -> std::uint64_t {
            return (u & 0x8000000000000000ull) != 0ull ? ~u : (u | 0x8000000000000000ull);
        };
        const std::uint64_t first = ordinal(ua);
        const std::uint64_t second = ordinal(ub);
        return first > second ? first - second : second - first;
    }
}

/// How many values two forms' outputs disagree on, how far apart the widest of them
/// is, and the first few of them, so a disagreement is reportable and not only
/// countable.
struct FormPair {
    std::size_t values = 0;
    std::size_t differ = 0;
    std::uint64_t widest = 0;
    std::string widestWhere;
    /// The values the two forms place more than a thousandth of the value apart:
    /// the ulp distance alone is dynamic-range sensitive, since two values near zero
    /// are far apart in representable steps while agreeing to the last bit of their
    /// magnitude, and this count is what separates "an ulp here and there" from a
    /// value one of the two forms got wrong outright.
    std::size_t apartByAThousandth = 0;
    double widestRelative = 0.0;
    std::string widestRelativeWhere;
    /// The largest absolute gap between the two forms, and where. The ulp count above
    /// is a distance in representable steps and the relative figure beside it is
    /// undefined where the value crosses zero, so the error itself is stated here:
    /// it is the quantity a per-form bound is written in.
    double widestAbsolute = 0.0;
    std::string widestAbsoluteWhere;
    std::vector<std::string> examples;
};

constexpr std::size_t kFormExamples = 4;

/// One disagreement, with the argument and the order that produced it and both
/// values as a decimal and as their bits. A finding, if there is one, has to name
/// the x at which it happened.
template <typename T>
std::string FormatDifference(std::size_t element, int order, double arg, T first, T second) {
    char values[192];

    if constexpr (std::is_same_v<T, float>)
    {
        std::snprintf(values,
                      sizeof(values),
                      "a=%.9g [%08x]  b=%.9g [%08x]",
                      static_cast<double>(first),
                      static_cast<unsigned>(BitsOf(first)),
                      static_cast<double>(second),
                      static_cast<unsigned>(BitsOf(second)));
    } else
    {
        std::snprintf(values,
                      sizeof(values),
                      "a=%.17g [%016llx]  b=%.17g [%016llx]",
                      static_cast<double>(first),
                      static_cast<unsigned long long>(BitsOf(first)),
                      static_cast<double>(second),
                      static_cast<unsigned long long>(BitsOf(second)));
    }

    char line[288];
    std::snprintf(
        line, sizeof(line), "    i=%zu order=%d x=%.17g  %s", element, order, arg, values);
    return std::string(line);
}

/// The counts for one pair, over the whole output. The order-major entries write
/// `out[order * count + i]`; the per-argument ones write `out[i]` and their order
/// is the one the caller asked for at `i`.
template <typename T>
FormPair CompareBits(const std::vector<T>& a,
                     const std::vector<T>& b,
                     const std::vector<int>& n,
                     const std::vector<double>& x,
                     std::size_t count,
                     bool planes) {
    FormPair pair;
    pair.values = a.size();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (BitsOf(a[i]) == BitsOf(b[i]))
        {
            continue;
        }

        ++pair.differ;

        const std::size_t element = planes ? i % count : i;
        const int order = planes ? static_cast<int>(i / count) : n[element];

        if (const std::uint64_t distance = UlpDistance(a[i], b[i]); distance > pair.widest)
        {
            pair.widest = distance;
            pair.widestWhere = FormatDifference(element, order, x[element], a[i], b[i]);
        }

        const double magnitude =
            std::max(std::abs(static_cast<double>(a[i])), std::abs(static_cast<double>(b[i])));
        const double relative =
            magnitude > 0.0
                ? std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])) / magnitude
                : 0.0;

        if (const double gap = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            gap > pair.widestAbsolute)
        {
            pair.widestAbsolute = gap;
            pair.widestAbsoluteWhere = FormatDifference(element, order, x[element], a[i], b[i]);
        }

        if (relative > 1e-3)
        {
            ++pair.apartByAThousandth;

            if (relative > pair.widestRelative)
            {
                pair.widestRelative = relative;
                pair.widestRelativeWhere = FormatDifference(element, order, x[element], a[i], b[i]);
            }
        }

        if (pair.examples.size() < kFormExamples)
        {
            pair.examples.push_back(FormatDifference(element, order, x[element], a[i], b[i]));
        }
    }

    return pair;
}

/// Runs one entry at one form over the batch and returns what it wrote. Every
/// launch goes through LaunchedForm, so the negative control reaches the entries
/// below without any of them knowing it.
template <typename T, typename Launch>
std::vector<T> RunEntryAtForm(const char* entry,
                              Launch launch,
                              const int* deviceN,
                              const double* deviceX,
                              T* deviceOut,
                              std::size_t values,
                              std::size_t count,
                              boys::DivisionForm form) {
    const boys::BoysStatus status =
        launch(deviceN, deviceX, deviceOut, count, nullptr, LaunchedForm(form));
    EXPECT_EQ(status, boys::BoysStatus::kSuccess)
        << entry << " at " << boys::DivisionFormName(form);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess) << entry;

    std::vector<T> host(values);
    EXPECT_EQ(cudaMemcpy(host.data(), deviceOut, values * sizeof(T), cudaMemcpyDeviceToHost),
              cudaSuccess)
        << entry;
    return host;
}

/// The counts over every entry crossed, plus the examples of the first pair.
struct FormTally {
    std::size_t entries = 0;
    std::size_t values = 0;
    std::size_t refinedVsExact = 0;
    std::size_t plainVsExact = 0;
    std::size_t plainVsRefined = 0;
    std::uint64_t widestRefinedVsExact = 0;
    std::uint64_t widestPlainVsExact = 0;
    std::uint64_t widestPlainVsRefined = 0;
    std::size_t plainVsExactApart = 0;
    double plainVsExactWidestRelative = 0.0;
    double plainVsExactWidestAbsolute = 0.0;
    std::string widestPlainVsExactWhere;
    std::string widestRelativePlainVsExactWhere;
    std::string widestAbsolutePlainVsExactWhere;
    std::vector<std::string> examples;
};

/// The whole check for one entry: three launches, three pairwise counts, one line
/// of the report. The totals are asserted on by the test.
template <typename T, typename Launch>
void CrossEntryWithForms(FormTally& tally,
                         const char* entry,
                         Launch launch,
                         const int* deviceN,
                         const double* deviceX,
                         T* deviceOut,
                         const std::vector<int>& n,
                         const std::vector<double>& x,
                         std::size_t count,
                         bool planes) {
    const std::size_t values = planes ? count * (boys::kMaxBoysOrder + 1) : count;
    std::array<std::vector<T>, 3> output;

    for (std::size_t f = 0; f < kDivisionForms.size(); ++f)
    {
        output[f] = RunEntryAtForm(
            entry, launch, deviceN, deviceX, deviceOut, values, count, kDivisionForms[f]);
    }

    const FormPair refinedVsExact = CompareBits(output[0], output[2], n, x, count, planes);
    const FormPair plainVsExact = CompareBits(output[0], output[1], n, x, count, planes);
    const FormPair plainVsRefined = CompareBits(output[1], output[2], n, x, count, planes);

    ++tally.entries;
    tally.values += values;
    tally.refinedVsExact += refinedVsExact.differ;
    tally.plainVsExact += plainVsExact.differ;
    tally.plainVsRefined += plainVsRefined.differ;
    tally.widestRefinedVsExact = std::max(tally.widestRefinedVsExact, refinedVsExact.widest);
    tally.widestPlainVsRefined = std::max(tally.widestPlainVsRefined, plainVsRefined.widest);

    if (plainVsExact.widest > tally.widestPlainVsExact)
    {
        tally.widestPlainVsExact = plainVsExact.widest;
        tally.widestPlainVsExactWhere = std::string(entry) + ":\n" + plainVsExact.widestWhere;
    }

    tally.plainVsExactApart += plainVsExact.apartByAThousandth;

    if (plainVsExact.widestRelative > tally.plainVsExactWidestRelative)
    {
        tally.plainVsExactWidestRelative = plainVsExact.widestRelative;
        tally.widestRelativePlainVsExactWhere =
            std::string(entry) + ":\n" + plainVsExact.widestRelativeWhere;
    }

    if (plainVsExact.widestAbsolute > tally.plainVsExactWidestAbsolute)
    {
        tally.plainVsExactWidestAbsolute = plainVsExact.widestAbsolute;
        tally.widestAbsolutePlainVsExactWhere =
            std::string(entry) + ":\n" + plainVsExact.widestAbsoluteWhere;
    }

    for (const std::string& line : refinedVsExact.examples)
    {
        tally.examples.push_back(std::string(entry) + " exact vs refined:\n" + line);
    }

    std::printf("  %-16s %9zu values   exact/refined %8zu (max %llu ulp)   exact/plain %8zu (max "
                "%llu ulp)   plain/refined %8zu (max %llu ulp)\n",
                entry,
                values,
                refinedVsExact.differ,
                static_cast<unsigned long long>(refinedVsExact.widest),
                plainVsExact.differ,
                static_cast<unsigned long long>(plainVsExact.widest),
                plainVsRefined.differ,
                static_cast<unsigned long long>(plainVsRefined.widest));
}

/// The uniform-order entries take their order as a host scalar - that is what
/// makes them uniform, and what the launched surface exposes as `AllNF*` rather
/// than as an order array - so the two adapters below supply it and every other
/// entry is passed through as it stands.
struct UniformOrderF64 {
    int nmax;

    boys::BoysStatus operator()(const int*,
                                const double* x,
                                double* out,
                                std::size_t count,
                                void* stream,
                                boys::DivisionForm form) const {
        return boys::BoysCuda::AllNF64(nmax, x, out, count, stream, form);
    }
};

struct UniformOrderF32 {
    int nmax;

    boys::BoysStatus operator()(const int*,
                                const double* x,
                                float* out,
                                std::size_t count,
                                void* stream,
                                boys::DivisionForm form) const {
        return boys::BoysCuda::AllNF32(nmax, x, out, count, stream, form);
    }
};

} // namespace

TEST(BoysCudaTest, DivisionFormsDifferByBits) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    ASSERT_EQ(cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost),
              cudaSuccess);
    ASSERT_EQ(cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost),
              cudaSuccess);

    const int nmax = boys::kMaxBoysOrder;
    FormTally tally;

    std::printf("division forms: %zu forms, entry by entry\n", kDivisionForms.size());
    CrossEntryWithForms(tally,
                        "SingleF64",
                        &boys::BoysCuda::SingleF64,
                        setup.n,
                        setup.x,
                        setup.outF64,
                        hostN,
                        hostX,
                        kCount,
                        false);
    CrossEntryWithForms(tally,
                        "AllOrdersF64",
                        &boys::BoysCuda::AllOrdersF64,
                        setup.n,
                        setup.x,
                        setup.outF64,
                        hostN,
                        hostX,
                        kCount,
                        true);
    CrossEntryWithForms(tally,
                        "AllNF64",
                        UniformOrderF64{nmax},
                        setup.n,
                        setup.x,
                        setup.outF64,
                        hostN,
                        hostX,
                        kCount,
                        true);
    CrossEntryWithForms(tally,
                        "SingleF32",
                        &boys::BoysCuda::SingleF32<>,
                        setup.n,
                        setup.x,
                        setup.outF32,
                        hostN,
                        hostX,
                        kCount,
                        false);
    CrossEntryWithForms(tally,
                        "AllOrdersF32",
                        &boys::BoysCuda::AllOrdersF32,
                        setup.n,
                        setup.x,
                        setup.outF32,
                        hostN,
                        hostX,
                        kCount,
                        true);
    CrossEntryWithForms(tally,
                        "AllNF32",
                        UniformOrderF32{nmax},
                        setup.n,
                        setup.x,
                        setup.outF32,
                        hostN,
                        hostX,
                        kCount,
                        true);

    std::printf("division forms: %zu entries, %zu values per form\n", tally.entries, tally.values);
    std::printf("division form check: exact vs refined %zu of %zu (max %llu ulp); exact vs plain "
                "%zu of %zu (max %llu ulp); plain vs refined %zu of %zu (max %llu ulp)\n",
                tally.refinedVsExact,
                tally.values,
                static_cast<unsigned long long>(tally.widestRefinedVsExact),
                tally.plainVsExact,
                tally.values,
                static_cast<unsigned long long>(tally.widestPlainVsExact),
                tally.plainVsRefined,
                tally.values,
                static_cast<unsigned long long>(tally.widestPlainVsRefined));

    std::printf("exact vs plain: %zu values of %zu are more than a thousandth of the value apart; "
                "the widest of those is %.3e of the value\n",
                tally.plainVsExactApart,
                tally.values,
                tally.plainVsExactWidestRelative);

    if (!tally.widestPlainVsExactWhere.empty())
    {
        std::printf("the widest exact-vs-plain difference, %llu representable steps apart:\n%s\n",
                    static_cast<unsigned long long>(tally.widestPlainVsExact),
                    tally.widestPlainVsExactWhere.c_str());
    }

    if (!tally.widestRelativePlainVsExactWhere.empty())
    {
        std::printf("the widest exact-vs-plain RELATIVE difference:\n%s\n",
                    tally.widestRelativePlainVsExactWhere.c_str());
    }

    // The gap in the units a bound is written in. A per-form device figure is not published at this
    // revision (BoysLaneContracts(), the fp32-device row), so this is a measurement and not a
    // comparison against a bar; it is printed so that the figure the lane owes is a number a reader
    // can see rather than one to be measured again to find out.
    std::printf("exact vs plain: the widest ABSOLUTE gap between the two forms, over the values "
                "compared, is %.6g\n",
                tally.plainVsExactWidestAbsolute);

    if (!tally.widestAbsolutePlainVsExactWhere.empty())
    {
        std::printf("the widest exact-vs-plain absolute gap:\n%s\n",
                    tally.widestAbsolutePlainVsExactWhere.c_str());
    }

    for (const std::string& line : tally.examples)
    {
        std::printf("%s\n", line.c_str());
    }

    // The claim the refined form is carried for, in accuracy.hpp's own words ("this form is
    // bit-identical to exact division"), stated over every value of every entry crossed here, so
    // one differing bit violates it: not a tolerance to widen, since a value the two forms place in
    // different bins is a defect in the refinement or in what "exact" compiles to, and both are
    // findings.
    EXPECT_EQ(tally.refinedVsExact, 0u)
        << "the refined form is not bit-identical to exact division over the values compared; "
           "the differing values and their x are printed above";

    // What makes the first assertion mean something: if the plain form agreed with
    // exact division on every value too, then no form would be reaching any kernel
    // and the first count would be zero for the same reason. The negative control
    // build makes this line fail.
    EXPECT_GT(tally.plainVsExact, 0u)
        << "the plain reciprocal agrees with exact division on every value compared, so the forms "
           "reaching the kernels are not the forms the calls named";
}

TEST(BoysCudaTest, AnUnknownDivisionFormIsRefused) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    // Refused and not substituted: a form the library does not carry is an
    // argument error. A call that quietly ran the default instead would be exactly
    // the substitution the axis exists to make visible, and the refinement's
    // bit-identity would hide it.
    const auto unknown = static_cast<boys::DivisionForm>(200);
    EXPECT_EQ(boys::BoysCuda::SingleF64(setup.n, setup.x, setup.outF64, 1, nullptr, unknown),
              boys::BoysStatus::kInvalidArgument);
    EXPECT_EQ(boys::BoysCuda::AllOrdersF32(setup.n, setup.x, setup.outF32, 1, nullptr, unknown),
              boys::BoysStatus::kInvalidArgument);
    EXPECT_EQ(
        boys::BoysCuda::AllNF64(boys::kMaxBoysOrder, setup.x, setup.outF64, 1, nullptr, unknown),
        boys::BoysStatus::kInvalidArgument);
}
