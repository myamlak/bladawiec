// The POSIX host-RAM probe: sysconf page counts on Linux, the Mach VM
// statistics on macOS (which has no _SC_AVPHYS_PAGES). The GPU probe's real
// implementation lives in cuda_backend.cpp (the CUDA-gated TU of
// qcx-backend-cuda); this TU carries the no-CUDA fallback so the symbol is
// always defined - in a CUDA build QcxHasCuda is defined here (PUBLIC
// compile definition on qcx-backend, mirrors integrals' QcxHasCuda=1) and
// the fallback compiles out.

#include "qcx/backend/memory_topology.hpp"

#include <cstdint>
#include <unistd.h>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#endif

namespace qcx::backend {

#if defined(__APPLE__)

namespace {

/// The macOS half of the host probe. sysconf reaches only the TOTAL physical
/// page count there, so the available figure comes from the Mach VM
/// statistics instead. "Available" is counted the way the system's own
/// memory-pressure view counts it: free, inactive and speculative pages can
/// all be handed out without writing to swap first.
HostMemoryInfo AppleHostMemory() noexcept {
    std::uint64_t totalBytes = 0;
    std::size_t totalSize = sizeof(totalBytes);

    if (sysctlbyname("hw.memsize", &totalBytes, &totalSize, nullptr, 0) != 0 || totalBytes == 0)
    {
        return HostMemoryInfo{0, 0};
    }

    const long pageSize = sysconf(_SC_PAGESIZE);

    if (pageSize <= 0)
    {
        return HostMemoryInfo{0, 0};
    }

    vm_statistics64_data_t vmStats{};
    mach_msg_type_number_t statsCount = HOST_VM_INFO64_COUNT;

    if (host_statistics64(mach_host_self(),
                          HOST_VM_INFO64,
                          reinterpret_cast<host_info64_t>(&vmStats),
                          &statsCount) != KERN_SUCCESS)
    {
        return HostMemoryInfo{0, 0};
    }

    const std::size_t total = static_cast<std::size_t>(totalBytes);
    const std::uint64_t availablePages = static_cast<std::uint64_t>(vmStats.free_count) +
                                         static_cast<std::uint64_t>(vmStats.inactive_count) +
                                         static_cast<std::uint64_t>(vmStats.speculative_count);
    std::size_t available =
        static_cast<std::size_t>(availablePages) * static_cast<std::size_t>(pageSize);

    if (available > total)
    {
        // The VM counters are sampled after the total and a machine under
        // pressure can momentarily report more reclaimable pages than it has
        // installed. availableBytes <= totalBytes is part of the contract.
        available = total;
    }

    return HostMemoryInfo{total, available};
}

} // namespace

#endif // __APPLE__

HostMemoryInfo DetectHostMemory() noexcept {
#if defined(__APPLE__)
    return AppleHostMemory();
#else
    const long pageSize = sysconf(_SC_PAGESIZE);

    if (pageSize <= 0)
    {
        return HostMemoryInfo{0, 0};
    }

    const long physicalPages = sysconf(_SC_PHYS_PAGES);
    const long availablePages = sysconf(_SC_AVPHYS_PAGES);

    if (physicalPages <= 0 || availablePages <= 0)
    {
        // Probe failure - callers fall back conservatively (zero =
        // "unknown", per the header contract).
        return HostMemoryInfo{0, 0};
    }

    return HostMemoryInfo{
        static_cast<std::size_t>(physicalPages) * static_cast<std::size_t>(pageSize),
        static_cast<std::size_t>(availablePages) * static_cast<std::size_t>(pageSize)};
#endif
}

#ifndef QcxHasCuda

GpuMemoryInfo DetectGpuMemory() noexcept {
    // No CUDA in this build: there is no device tier, report "no GPU
    // memory" (the header contract).
    return GpuMemoryInfo{0, 0};
}

#endif

} // namespace qcx::backend
