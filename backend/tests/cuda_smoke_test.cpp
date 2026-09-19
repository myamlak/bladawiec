#include "qcx/backend/cuda_backend.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

// Host-side C++23 test: the kernel itself lives in backend/src/cuda_backend.cu;
// this file only exercises the host CUDA API and Backend<CudaTag>.

namespace {

// RAII owner for raw cudaMalloc memory; frees on scope exit.
struct CudaFreeDeleter {
    void operator()(float* p) const noexcept {
        cudaFree(p);
    }
};

using CudaFloatPtr = std::unique_ptr<float, CudaFreeDeleter>;

// Allocates count floats of device memory, or returns an empty handle on failure.
CudaFloatPtr AllocateDevice(int count) {
    void* raw = nullptr;

    if (cudaMalloc(&raw, count * sizeof(float)) != cudaSuccess)
    {
        return nullptr;
    }

    return CudaFloatPtr(static_cast<float*>(raw));
}

} // namespace

TEST(CudaBackendSmoke, VectorAddOnDevice) {
    constexpr int kCount = 1024;
    std::vector<float> a(kCount, 1.0f);
    std::vector<float> b(kCount, 2.0f);
    std::vector<float> out(kCount, 0.0f);

    CudaFloatPtr dA = AllocateDevice(kCount);
    CudaFloatPtr dB = AllocateDevice(kCount);
    CudaFloatPtr dOut = AllocateDevice(kCount);
    ASSERT_TRUE(dA);
    ASSERT_TRUE(dB);
    ASSERT_TRUE(dOut);
    EXPECT_EQ(cudaMemcpy(dA.get(), a.data(), kCount * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaMemcpy(dB.get(), b.data(), kCount * sizeof(float), cudaMemcpyHostToDevice),
              cudaSuccess);

    auto backend = qcx::backend::Backend<qcx::backend::CudaTag>::Create();
    ASSERT_TRUE(backend.has_value());
    ASSERT_TRUE(backend->VectorAdd(dA.get(), dB.get(), dOut.get(), kCount).has_value());

    EXPECT_EQ(cudaMemcpy(out.data(), dOut.get(), kCount * sizeof(float), cudaMemcpyDeviceToHost),
              cudaSuccess);

    for (float v : out)
    {
        EXPECT_FLOAT_EQ(v, 3.0f);
    }
}

TEST(CudaBackendSmoke, VectorAddWithZeroCountIsANoOp) {
    // n == 0 must not produce an invalid <<<0, 256>>> launch config (the
    // pre-fix grid expression (0 + 255)/256 == 0 and the kernel launch
    // failed with kDeviceError despite the precondition never excluding
    // n == 0): an empty add is a no-op - no kernel runs, no memory is
    // touched, so the pointers may be null.
    auto backend = qcx::backend::Backend<qcx::backend::CudaTag>::Create();
    ASSERT_TRUE(backend.has_value());
    ASSERT_TRUE(backend->VectorAdd(nullptr, nullptr, nullptr, 0).has_value());
}
