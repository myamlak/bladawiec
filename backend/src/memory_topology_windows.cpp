// The Windows host-RAM probe: GlobalMemoryStatusEx. The GPU probe's real
// implementation lives in cuda_backend.cpp (the CUDA-gated TU of
// qcx-backend-cuda); this TU carries the no-CUDA fallback so the symbol is
// always defined - in a CUDA build QcxHasCuda is defined here (PUBLIC
// compile definition on qcx-backend, mirrors integrals' QcxHasCuda=1) and
// the fallback compiles out.

#include "qcx/backend/memory_topology.hpp"

#include <windows.h>

namespace qcx::backend {

HostMemoryInfo DetectHostMemory() noexcept {
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);

    if (!GlobalMemoryStatusEx(&status))
    {
        // Probe failure - callers fall back conservatively (zero =
        // "unknown", per the header contract).
        return HostMemoryInfo{0, 0};
    }

    return HostMemoryInfo{static_cast<std::size_t>(status.ullTotalPhys),
                          static_cast<std::size_t>(status.ullAvailPhys)};
}

#ifndef QcxHasCuda

GpuMemoryInfo DetectGpuMemory() noexcept {
    // No CUDA in this build: there is no device tier, report "no GPU
    // memory" (the header contract).
    return GpuMemoryInfo{0, 0};
}

#endif

} // namespace qcx::backend
