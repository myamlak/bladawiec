#include "qcx/backend/cuda_backend.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/memory/cuda_allocator_traits.hpp"
#include "qcx/memory/device_buffer.hpp"

#include <cstddef>
#include <gtest/gtest.h>

// Host-side C++23 test: the CUDA kernels themselves live in backend's
// cuda_backend.cu (compiled by nvcc as C++20); everything here is host API
// (cudaMalloc/cudaMemcpy through the traits, kernel launches via
// Backend<CudaTag>::VectorAdd), so cl.exe compiles this file as C++23.

using CudaBuffer = qcx::memory::DeviceBuffer<float, qcx::backend::CudaTag>;

TEST(DeviceBufferCudaTest, HostToDeviceToHostRoundTrip) {
    auto buf = CudaBuffer::Create(256);
    ASSERT_TRUE(buf.has_value());
    auto& host = buf->HostView();

    for (std::size_t i = 0; i < host.size(); ++i)
    {
        host[i] = static_cast<float>(i);
    }

    buf->MarkHostDirty();
    ASSERT_TRUE(buf->SyncToDevice().has_value()); // host -> device

    auto dptr = buf->DeviceHandle();
    EXPECT_TRUE(static_cast<bool>(dptr));

    buf->MarkDeviceDirty();
    ASSERT_TRUE(buf->SyncToHost().has_value()); // device -> host

    auto& back = buf->HostView();

    for (std::size_t i = 0; i < back.size(); ++i)
    {
        EXPECT_FLOAT_EQ(back[i], static_cast<float>(i));
    }
}

TEST(DeviceBufferCudaTest, VectorAddOnBufferMemory) {
    auto a = CudaBuffer::Create(1024);
    auto b = CudaBuffer::Create(1024);
    auto out = CudaBuffer::Create(1024);
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    ASSERT_TRUE(out.has_value());

    auto& hostA = a->HostView();
    auto& hostB = b->HostView();

    for (std::size_t i = 0; i < hostA.size(); ++i)
    {
        hostA[i] = 1.0f;
        hostB[i] = 2.0f;
    }

    a->MarkHostDirty();
    b->MarkHostDirty();
    ASSERT_TRUE(a->SyncToDevice().has_value());
    ASSERT_TRUE(b->SyncToDevice().has_value());

    auto dA = a->DeviceHandle();
    auto dB = b->DeviceHandle();
    auto dOut = out->DeviceHandle();

    auto backend = qcx::backend::Backend<qcx::backend::CudaTag>::Create();
    ASSERT_TRUE(backend.has_value());
    ASSERT_TRUE(backend
                    ->VectorAdd(static_cast<const float*>(dA.Raw()),
                                static_cast<const float*>(dB.Raw()),
                                static_cast<float*>(dOut.Raw()),
                                1024)
                    .has_value());

    out->MarkDeviceDirty();
    ASSERT_TRUE(out->SyncToHost().has_value());
    auto& back = out->HostView();

    for (std::size_t i = 0; i < back.size(); ++i)
    {
        EXPECT_FLOAT_EQ(back[i], 3.0f);
    }
}
