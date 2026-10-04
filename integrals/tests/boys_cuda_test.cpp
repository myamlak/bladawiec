#include "boys/boys.hpp"
#include "boys/boys_cuda.hpp"
#include "boys/boys_impl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

#if BoysFp16
#include "boys/f16.hpp"
#endif

// The lane's residency comparison (src/boys_cuda.cu), reached here to assert its
// rows directly: the arguments are the device in hand and the multiplier asked
// for, then the record of what was last uploaded and where. Both arrive as
// arguments, so the assertion needs no second card and no CUDA call.
extern "C" int BoysCudaEffTablesResidentOn(int device,
                                           double m,
                                           int recordedDevice,
                                           double recordedM);

namespace {

constexpr std::size_t kCount = 1u << 16;
constexpr double kDoubleTolerance = 5.5e-14;
// Cross-lane budget: the CPU lane is validated <= 1.5e-7 against the
// reference grid, and the default CUDA lane holds the same 1.5e-7, so
// GPU-vs-CPU agreement on the default path holds within ~3e-7; the budget is
// stated at 3.5e-7. It is a budget on the default path only: the single
// entry's fast region-B exponential (RegionBExp::kFast) carries a larger
// bound of its own, which no cross-lane figure covers.
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
// The fp16 lane compares against the CPU fp16 lane (both compute in float
// and round to fp16): the cross-lane float budget is the F32 one above, and
// one ULP of quantization covers the fp16 rounding (HalfUlp here is the
// full grid step, NextUp(x) - x, not half of it).
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

    // The first float of region B, where the two options separate and where the
    // recurrence's condition number is at its largest (7.6e4 at this order): the
    // value F_32(x) is itself at the level of a seed error there, and the ladder
    // amplifies it by that factor. The double lane is the reference; its own
    // error at this argument is below 5.5e-14, six orders under the figures
    // this test is about.
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
                ? boys::BoysCuda::SingleF32<1.0, boys::RegionBExp::kFast>(
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

    // Both options run, and they are two arithmetics rather than one with a
    // name for the other.
    EXPECT_NE(accurate, fast);

    // Both hold their documented bounds here: the lane's 1.5e-7 for the
    // default, and the lane's plus the corrected seed's own contribution
    // (8e-8) for the fast one.
    EXPECT_LE(std::abs(accurate - want), 1.5e-7);
    EXPECT_LE(std::abs(fast - want), 1.5e-7 + 8e-8);

    // This is the cell the bare approximation returned the wrong sign at, and
    // the reason the fast option carries a correction: with it, the return has
    // the value's sign, and the option's own contribution here is a fraction of
    // the value rather than larger than it. Pinned so that a regression to the
    // uncorrected seed is a failure and not a footnote.
    EXPECT_GT(fast * want, 0.0);
    EXPECT_LT(std::abs(fast - accurate), std::abs(want));

    std::printf("SingleF32 at the region-B boundary: accurate %.9g, fast %.9g, value %.9g\n",
                accurate,
                fast,
                want);
}

// The default option is the batch entries' arithmetic, and the contract says
// so: outside region A the single entry's seed and ladder are the all-orders
// body's, so a consumer that reads one and the other of the same (n, x) is
// reading one arithmetic. Region A is excluded because the two seeds differ
// there by design - the batch seeds its downward recursion from the double
// piece table, the single entry from the float one.
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
    // Arguments that are exactly floats at or above the first float of region
    // B, so both entries evaluate the same argument and classify it the same
    // way, and region B and region C are both covered.
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

    // The CPU twin over the same arguments, compared in the shipped layout:
    // every plane of every argument, so a plane the device left unwritten is a
    // failure too.
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

TEST(BoysCudaTest, AllNF64RelaxedTierKeepsTheBound) {
    // The multiplier is named at the call rather than passed to it, so the tier
    // is an instantiation: m = 10 relaxes the budget to m * 5.5e-14 and uploads
    // its own degree tables on first use. The reference is the full-accuracy CPU
    // path, whose own error is a tenth of this budget.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    constexpr double kRelaxed = 10.0;
    const std::vector<double> hostX = SortedArguments();
    const std::size_t count = hostX.size();
    AllNDeviceSetup setup(hostX);
    ASSERT_EQ(boys::BoysCuda::InitializeTables(), boys::BoysStatus::kSuccess);

    ASSERT_EQ(boys::BoysCuda::AllNF64<kRelaxed>(
                  boys::kMaxBoysOrder, setup.x, setup.outF64, count, nullptr),
              boys::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<double> hostOut(count * (boys::kMaxBoysOrder + 1));
    cudaMemcpy(
        hostOut.data(), setup.outF64, hostOut.size() * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> reference(hostOut.size());
    boys::BoysAllN(
        boys::kMaxBoysOrder, hostX.data(), reference.data(), count, boys::BoysSortedArgs{});
    double worst = 0.0;
    std::size_t worstAt = 0;

    for (std::size_t j = 0; j < hostOut.size(); ++j)
    {
        const double error = std::abs(hostOut[j] - reference[j]);

        if (error > worst)
        {
            worst = error;
            worstAt = j;
        }
    }

    EXPECT_LE(worst, kRelaxed * kDoubleTolerance)
        << "i=" << worstAt % count << " k=" << worstAt / count << " x=" << hostX[worstAt % count];
    std::printf("AllNF64 m=10 vs CPU full accuracy: worst |diff| = %.3e (budget %.3e)\n",
                worst,
                kRelaxed * kDoubleTolerance);
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

    // The entry's non-decreasing precondition is a performance one (one
    // classification path per warp), not a correctness one: the same batch in
    // arbitrary order must return the same planes.
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
    // One behaviour for every family: count == 0 queues no kernel, writes
    // nothing, and returns kSuccess. The output buffers are poisoned first, so
    // the second half of that sentence is measured, not assumed — a zero-block
    // launch is a CUDA error, which is why the host side has to short-circuit.
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

    // The relaxed multipliers reach the no-op through their own instantiation
    // and their own launch symbol, so each family is pinned there too.
    EXPECT_EQ(boys::BoysCuda::SingleF64<10.0>(&n, &x, out64, 0, nullptr),
              boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllOrdersF32<10.0>(&n, &x, out32, 0, nullptr),
              boys::BoysStatus::kSuccess);
    EXPECT_EQ(boys::BoysCuda::AllNF16<10.0>(1, &y, out16, 0, nullptr), boys::BoysStatus::kSuccess);

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

TEST(BoysCudaTest, EffTableResidencyNamesTheDevice) {
    // The effective-degree tables are per-device copies of __constant__
    // symbols, so a record compared on the multiplier alone would answer for a
    // device that has never held them, and a device no upload has reached reads
    // zero-initialized tables. The row that decides it is the second: the same
    // multiplier on another device, which must not be answered as resident.
    EXPECT_EQ(BoysCudaEffTablesResidentOn(0, 2.0, 0, 2.0), 1);
    EXPECT_EQ(BoysCudaEffTablesResidentOn(1, 2.0, 0, 2.0), 0);
    EXPECT_EQ(BoysCudaEffTablesResidentOn(0, 2.0, 1, 2.0), 0);
    EXPECT_EQ(BoysCudaEffTablesResidentOn(0, 10.0, 0, 2.0), 0);

    // Before any upload the record names no device and no multiplier.
    EXPECT_EQ(BoysCudaEffTablesResidentOn(0, 2.0, -1, -1.0), 0);
}
