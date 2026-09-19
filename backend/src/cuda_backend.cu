#include "cuda_backend_detail.hpp"

#include <cuda_runtime.h>
#include <string>

namespace qcx::backend {

namespace {

// __restrict__ on all three: the buffers are distinct device allocations (the
// public API's precondition, see cuda_backend.hpp). Without it nvcc must
// assume every out store can alias a/b and reloads them per iteration.
__global__ void VectorAddKernel(const float* __restrict__ a,
                                const float* __restrict__ b,
                                float* __restrict__ out,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
    {
        out[i] = a[i] + b[i];
    }
}

// The driver's description of err, or an empty string on success.
std::string Describe(cudaError_t err) {
    return err == cudaSuccess ? std::string{} : cudaGetErrorString(err);
}

} // namespace

std::string CudaGetDevice(int* deviceId) {
    if (std::string err = Describe(cudaGetDevice(deviceId)); !err.empty())
    {
        return "cudaGetDevice failed: " + err;
    }

    return {};
}

std::string CudaGetDeviceCount(int* deviceCount) {
    if (std::string err = Describe(cudaGetDeviceCount(deviceCount)); !err.empty())
    {
        return "cudaGetDeviceCount failed: " + err;
    }

    return {};
}

std::string CudaDeviceMemInfo(int deviceId, std::size_t* freeBytes, std::size_t* totalBytes) {
    int currentDevice = 0;

    if (std::string err = Describe(cudaGetDevice(&currentDevice)); !err.empty())
    {
        return "cudaGetDevice failed: " + err;
    }

    // cudaMemGetInfo reports the CURRENT device only; the sweep moves to
    // the target id for the query and restores the original device
    // afterwards - the probe must never leave the caller's device context
    // changed (the GPU builder later creates on the current device).
    const cudaError_t switchError = cudaSetDevice(deviceId);
    const cudaError_t memError =
        switchError == cudaSuccess ? cudaMemGetInfo(freeBytes, totalBytes) : cudaSuccess;
    const cudaError_t restoreError = cudaSetDevice(currentDevice);

    if (switchError != cudaSuccess)
    {
        return "cudaSetDevice(" + std::to_string(deviceId) +
               ") failed: " + cudaGetErrorString(switchError);
    }

    if (memError != cudaSuccess)
    {
        return "cudaMemGetInfo failed: " + Describe(memError);
    }

    if (restoreError != cudaSuccess)
    {
        return "cudaSetDevice restore failed: " + Describe(restoreError);
    }

    return {};
}

std::string CudaGetMemInfo(std::size_t* freeBytes, std::size_t* totalBytes) {
    if (std::string err = Describe(cudaMemGetInfo(freeBytes, totalBytes)); !err.empty())
    {
        return "cudaMemGetInfo failed: " + err;
    }

    return {};
}

std::string CudaVectorAdd(const float* a, const float* b, float* out, int n) {
    // n <= 0 (the public API's precondition never excludes n == 0): the
    // grid expression below would yield an invalid <<<0, 256>>> launch
    // config for an empty add - a no-op, not an error.
    if (n <= 0)
    {
        return {};
    }

    constexpr int kThreadsPerBlock = 256;
    // size_t arithmetic: n + kThreadsPerBlock - 1 overflows signed int
    // (UB) for n > INT_MAX - 255.
    const std::size_t grid =
        (static_cast<std::size_t>(n) + kThreadsPerBlock - 1) / kThreadsPerBlock;
    VectorAddKernel<<<grid, kThreadsPerBlock>>>(a, b, out, n);

    if (std::string err = Describe(cudaGetLastError()); !err.empty())
    {
        return "kernel launch failed: " + err;
    }

    if (std::string err = Describe(cudaDeviceSynchronize()); !err.empty())
    {
        return "cudaDeviceSynchronize failed: " + err;
    }

    return {};
}

} // namespace qcx::backend
