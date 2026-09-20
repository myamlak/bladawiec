#pragma once

#include "qcx/backend/tags.hpp"
#include "qcx/error.hpp"
#include "qcx/logging.hpp"
#include "qcx/memory/allocator_traits.hpp"

#include <cstddef>
#include <cuda_runtime.h>
#include <string>

namespace qcx::memory {

/// Allocation traits for the CUDA backend: cudaMalloc/cudaFree/cudaMemcpy.
///
/// Compiled only where the CUDA toolkit is available (QCX_ENABLE_CUDA).
/// Every failure is reported as a qcx::Error with the CUDA driver's own
/// description embedded in the message.
/// \ingroup qcx-memory
template <> struct AllocatorTraits<qcx::backend::CudaTag> {
    /// Allocates \p bytes of device memory with cudaMalloc.
    /// \param bytes Number of bytes to allocate; never zero.
    /// \returns The allocated block, or an Error carrying the cudaMalloc
    /// failure description.
    static qcx::Result<void*> Allocate(std::size_t bytes) {
        void* p = nullptr;

        if (cudaError_t err = cudaMalloc(&p, bytes); err != cudaSuccess)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kDeviceError,
                           std::string{"cudaMalloc failed: "} + cudaGetErrorString(err)});
        }

        return p;
    }

    /// Releases memory previously returned by Allocate.
    ///
    /// Failures cannot be reported (destructor path), so they are logged.
    /// \param p Device allocation to release (nullptr is a no-op).
    static void Deallocate(void* p) noexcept {
        if (p == nullptr)
        {
            return;
        }

        if (cudaError_t err = cudaFree(p); err != cudaSuccess)
        {
            qcx::log::Error("cudaFree failed: {}", cudaGetErrorString(err));
        }
    }

    /// Copies \p bytes from device memory \p src into host memory \p dst.
    /// \param dst Destination host buffer.
    /// \param src Source device buffer.
    /// \param bytes Number of bytes to copy.
    /// \returns An Error (kDeviceError) when the copy fails.
    static qcx::Result<void> CopyToHost(void* dst, const void* src, std::size_t bytes) {
        if (cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
            err != cudaSuccess)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kDeviceError,
                                              std::string{"cudaMemcpy device-to-host failed: "} +
                                                  cudaGetErrorString(err)});
        }

        return {};
    }

    /// Copies \p bytes from host memory \p src into device memory \p dst.
    /// \param dst Destination device buffer.
    /// \param src Source host buffer.
    /// \param bytes Number of bytes to copy.
    /// \returns An Error (kDeviceError) when the copy fails.
    static qcx::Result<void> CopyFromHost(void* dst, const void* src, std::size_t bytes) {
        if (cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
            err != cudaSuccess)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kDeviceError,
                                              std::string{"cudaMemcpy host-to-device failed: "} +
                                                  cudaGetErrorString(err)});
        }

        return {};
    }
};

} // namespace qcx::memory
