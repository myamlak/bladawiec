// The CUDA compute-profile probes (C++20-safe, no qcx headers - the
// C++/CUDA bridge contract): the tensor-core capability query and the
// fp32/fp64 FMA-throughput micro-benchmark the precision ladder prices
// its bands with (gpu_compute_profile.hpp). All error reporting goes
// through the driver's own error text (empty on success); the throughput
// measurement is deliberately coarse - a register-bound FMA chain
// saturated across the device - because the ladder consumes an
// order-of-magnitude lane-price ratio, not a tuning number.

#include "gpu_compute_profile_detail.hpp"

#include <cuda_runtime.h>
#include <string>

namespace {

// Restores the saved device on scope exit (best-effort - a destructor
// cannot report a failure, so the restore result is dropped; the saved
// device is queried before any set, so the caller's context survives
// every probe failure path).
struct DeviceRestore {
    int saved;

    ~DeviceRestore() {
        cudaSetDevice(saved);
    }
};

__device__ float FmaLane(float a, float b, float c) {
    return fmaf(a, b, c);
}

__device__ double FmaLane(double a, double b, double c) {
    return fma(a, b, c);
}

// The register-bound FMA chain: each thread runs `reps` iterations of two
// independent FMA chains (the 4-FLOP-per-iteration counting unit) and
// writes once behind a guard that is never true for these seeds (all
// non-negative, so -1.0 is unreachable) - the compiler cannot prove it,
// so the chains survive. One lane per launch: precision is a launch
// parameter, never a branch inside the kernel (the ladder's own rule).
template <typename T> __global__ void FmaChainKernel(T* output, int reps) {
    const int lane = threadIdx.x & 31;
    T a = static_cast<T>(lane) * static_cast<T>(0.25);
    T b = a + static_cast<T>(1.0);
    const T c = a + static_cast<T>(0.5);

    for (int i = 0; i < reps; ++i)
    {
        a = FmaLane(a, b, c);
        b = FmaLane(b, a, c);
    }

    if (a == static_cast<T>(-1.0) && b == static_cast<T>(-1.0))
    {
        output[threadIdx.x] = a + b;
    }
}

// One timed lane: warmup launch, then the event-timed run. The caller
// checks cudaGetLastError after; the events and the launch must all
// succeed or the measurement reports failure via the driver text.
template <typename T>
std::string MeasureLaneGflops(int blocks, int threads, int reps, T* output, double* gflops) {
    const double flops = static_cast<double>(blocks) * static_cast<double>(threads) *
                         static_cast<double>(reps) * 4.0;

    FmaChainKernel<T><<<blocks, threads>>>(output, reps);

    if (cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    if (cudaError_t err = cudaEventCreate(&start); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    if (cudaError_t err = cudaEventCreate(&stop); err != cudaSuccess)
    {
        cudaEventDestroy(start);
        return cudaGetErrorString(err);
    }

    if (cudaError_t err = cudaEventRecord(start); err != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return cudaGetErrorString(err);
    }

    FmaChainKernel<T><<<blocks, threads>>>(output, reps);

    if (cudaError_t err = cudaEventRecord(stop); err != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return cudaGetErrorString(err);
    }

    if (cudaError_t err = cudaEventSynchronize(stop); err != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return cudaGetErrorString(err);
    }

    float elapsedMs = 0.0F;

    if (cudaError_t err = cudaEventElapsedTime(&elapsedMs, start, stop); err != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return cudaGetErrorString(err);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (cudaError_t err = cudaDeviceSynchronize(); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    if (elapsedMs <= 0.0F)
    {
        return "throughput measurement: zero elapsed time";
    }

    *gflops = flops / (static_cast<double>(elapsedMs) * 1.0e6);
    return {};
}

} // namespace

namespace qcx::backend {

std::string CudaComputeCapabilities(int deviceId, bool* tensorCores) {
    int currentDevice = 0;

    if (cudaError_t err = cudaGetDevice(&currentDevice); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    const DeviceRestore restore{currentDevice};

    if (cudaError_t err = cudaSetDevice(deviceId); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    cudaDeviceProp prop{};

    if (cudaError_t err = cudaGetDeviceProperties(&prop, deviceId); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    *tensorCores = prop.major >= 7;
    return {};
}

std::string CudaMeasureThroughput(int deviceId, double* fp32Gflops, double* fp64Gflops) {
    int currentDevice = 0;

    if (cudaError_t err = cudaGetDevice(&currentDevice); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    const DeviceRestore restore{currentDevice};

    if (cudaError_t err = cudaSetDevice(deviceId); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    // The launch shape: enough resident warps to saturate any single
    // device while keeping the measured window short (the fp64 consumer
    // lane dominates: ~4.3 GFLOP at a 1/32 rate is tens of milliseconds).
    constexpr int kThreadsPerBlock = 256;
    constexpr int kBlocks = 2048;
    constexpr int kReps = 2048;

    float* f32Out = nullptr;
    double* f64Out = nullptr;

    if (cudaError_t err = cudaMalloc(&f32Out, kThreadsPerBlock * sizeof(float)); err != cudaSuccess)
    {
        return cudaGetErrorString(err);
    }

    if (cudaError_t err = cudaMalloc(&f64Out, kThreadsPerBlock * sizeof(double));
        err != cudaSuccess)
    {
        cudaFree(f32Out);
        return cudaGetErrorString(err);
    }

    std::string failure = MeasureLaneGflops(kBlocks, kThreadsPerBlock, kReps, f32Out, fp32Gflops);

    if (failure.empty())
    {
        failure = MeasureLaneGflops(kBlocks, kThreadsPerBlock, kReps, f64Out, fp64Gflops);
    }

    if (cudaError_t err = cudaFree(f32Out); err != cudaSuccess && failure.empty())
    {
        failure = cudaGetErrorString(err);
    }

    if (cudaError_t err = cudaFree(f64Out); err != cudaSuccess && failure.empty())
    {
        failure = cudaGetErrorString(err);
    }

    return failure;
}

} // namespace qcx::backend
