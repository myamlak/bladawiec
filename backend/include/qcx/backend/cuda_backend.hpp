#pragma once

#include "qcx/backend/backend.hpp"
#include "qcx/backend/concepts.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/error.hpp"

namespace qcx::backend {

/// CUDA backend: device selection plus a real device kernel.
///
/// Kept free of CUDA runtime headers so plain C++ translation units can
/// include it; the implementation lives in cuda_backend.cu (kernel, C++20)
/// and cuda_backend.cpp (Result layer, C++23). Construction goes through
/// Create() because querying the device can fail.
/// \ingroup qcx-backend
template <> class Backend<CudaTag> {
public:
    using StreamType = void*; ///< Opaque here; cudaStream_t is used inside the .cu file.
    using DeviceIdType = int; ///< CUDA device identifier.

    /// Constructs the backend on the current CUDA device.
    /// \returns The backend, or an Error (kDeviceError) when the device
    /// query fails.
    static qcx::Result<Backend> Create();

    /// Id of the device this backend was constructed on.
    /// \returns The CUDA device id.
    DeviceIdType DeviceId() const {
        return _deviceId;
    }

    /// Computes out[i] = a[i] + b[i] for i in [0, n) on the device.
    /// \param a Device pointer to the first input array.
    /// \param b Device pointer to the second input array.
    /// \param out Device pointer to the output array.
    /// \param n Number of elements; all three pointers must be valid for n
    /// elements (may be null when n == 0 - an empty add is a no-op: no
    /// kernel is launched and nothing is read or written).
    /// \pre a, b and out are distinct (non-aliasing) device allocations.
    /// \returns An Error (kDeviceError) when the kernel launch or the device
    /// synchronization fails.
    qcx::Result<void> VectorAdd(const float* a, const float* b, float* out, int n) const;

private:
    /// Constructs from a known device id (used by Create).
    explicit Backend(DeviceIdType deviceId) : _deviceId(deviceId) {}

    DeviceIdType _deviceId{0};
};

static_assert(ExecutionBackend<Backend<CudaTag>>);

} // namespace qcx::backend
