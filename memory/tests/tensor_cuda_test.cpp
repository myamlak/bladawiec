#include "qcx/backend/tags.hpp"
#include "qcx/memory/cuda_allocator_traits.hpp"
#include "qcx/memory/tensor.hpp"

#include <cstddef>
#include <gtest/gtest.h>

// Host-side C++23 test (see device_buffer_cuda_test.cpp): the migration
// plumbing below launches no kernels, so cl.exe compiles this file.

using CudaTensor1D = qcx::memory::Tensor<float, 1, qcx::backend::CudaTag>;

TEST(TensorCudaTest, WithDeviceMigratesAndRoundTrips) {
    auto t = CudaTensor1D::Create({256});
    ASSERT_TRUE(t.has_value());
    // CudaTag tensors have no operator() - seed via the host copy.
    auto& host = t->HostView();

    for (std::size_t i = 0; i < host.size(); ++i)
    {
        host[i] = static_cast<float>(i);
    }

    t->MarkHostDirty();

    auto migrated = qcx::memory::WithDevice<qcx::backend::CudaTag>(
        *t, [](auto dptr) { EXPECT_TRUE(static_cast<bool>(dptr)); });
    ASSERT_TRUE(migrated.has_value());

    ASSERT_TRUE(t->ToHost().has_value());
    auto& back = t->HostView();

    for (std::size_t i = 0; i < back.size(); ++i)
    {
        EXPECT_FLOAT_EQ(back[i], static_cast<float>(i));
    }
}
