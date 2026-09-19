// Result-typed (C++23) layer of the CUDA backend. All CUDA runtime calls
// live in cuda_backend.cu behind the C++20-safe bridge in
// cuda_backend_detail.hpp; this TU converts the driver's error text into
// qcx errors.
#include "qcx/backend/cuda_backend.hpp"

#include "cuda_backend_detail.hpp"
#include "qcx/backend/memory_topology.hpp"
#include "qcx/backend/topology_profile.hpp"

#include <string>
#include <utility>
#include <vector>

namespace qcx::backend {

qcx::Result<Backend<CudaTag>> Backend<CudaTag>::Create() {
    int deviceId = 0;

    if (std::string err = CudaGetDevice(&deviceId); !err.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kDeviceError, std::move(err)});
    }

    return Backend{deviceId};
}

std::vector<DeviceProfile> DetectDevices() noexcept {
    int deviceCount = 0;

    if (std::string err = CudaGetDeviceCount(&deviceCount); !err.empty())
    {
        // Probe failure - "no GPU tier" (the header contract).
        return {};
    }

    std::vector<DeviceProfile> devices;
    devices.reserve(static_cast<std::size_t>(deviceCount));

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId)
    {
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;

        if (std::string err = CudaDeviceMemInfo(deviceId, &freeBytes, &totalBytes); !err.empty())
        {
            // A device whose own query fails is skipped, not fatal (the
            // header contract): the remaining devices still describe the
            // tier. The bridge already restored the caller's current
            // device.
            continue;
        }

        devices.push_back(DeviceProfile{deviceId, totalBytes, freeBytes});
    }

    return devices;
}

GpuMemoryInfo DetectGpuMemory() noexcept {
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;

    if (std::string err = CudaGetMemInfo(&freeBytes, &totalBytes); !err.empty())
    {
        // Probe failure - "no GPU tier" (the header contract).
        return GpuMemoryInfo{0, 0};
    }

    return GpuMemoryInfo{totalBytes, freeBytes};
}

qcx::Result<void> Backend<CudaTag>::VectorAdd(const float* a,
                                              const float* b,
                                              float* out,
                                              int n) const {
    if (std::string err = CudaVectorAdd(a, b, out, n); !err.empty())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kDeviceError, std::move(err)});
    }

    return {};
}

} // namespace qcx::backend
