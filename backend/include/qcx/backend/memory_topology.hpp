#pragma once

/// \file
/// Host RAM and GPU device-memory probes: the shared sizing primitive of the
/// caching tiers (an in-memory ERI-value cache when the direct Fock path
/// justifies one, and the disk-integral store's buffer/chunk sizing).
///
/// Both probes follow the DetectCpuTopology precedent (cpu_topology.hpp)
/// in shape - noexcept probe functions returning plain structs - with one
/// deliberate difference: the results are NOT cached process-wide the way
/// DefaultOmpTeamSize caches the CPU team size. CPU core count is static
/// for the process lifetime; available memory is not - the process itself
/// consumes RAM as it allocates (density/Fock matrices, the RI tensor, a
/// growing cache), so a startup-time snapshot is a sizing hint, not a
/// budget that stays accurate mid-run. Each consumer decides explicitly
/// whether a fresh probe per sizing decision is needed or whether a single
/// startup-time probe with a safety margin is acceptable (the latter is
/// fine for this project's actual use pattern - one single-user
/// workstation running one QC calculation at a time - but it is a stated
/// assumption, not a silent one).

#include <cstddef>

namespace qcx::backend {

/// Host RAM of this node, probed once.
/// \ingroup qcx-backend
struct HostMemoryInfo {
    std::size_t totalBytes; ///< Total installed physical RAM.
    std::size_t availableBytes; ///< Currently available - a point-in-time snapshot, NOT a
                                ///< reservation: it changes as this process and others
                                ///< allocate, so treat it as a sizing hint at probe time,
                                ///< not a guarantee that stays valid for the process
                                ///< lifetime.
};

/// Probes the host RAM.
/// \returns The totals; both zero on probe failure (callers must treat
/// zero availableBytes as "unknown, be conservative" - size to a small
/// fraction of totalBytes or a documented floor, never to a zero
/// "available" budget).
/// \ingroup qcx-backend
HostMemoryInfo DetectHostMemory() noexcept;

/// GPU device memory of the current CUDA device, probed once.
/// \ingroup qcx-backend
struct GpuMemoryInfo {
    std::size_t totalBytes; ///< Total device memory.
    std::size_t freeBytes; ///< Free right now - same "hint, not guarantee" caveat as
                           ///< HostMemoryInfo::availableBytes.
};

/// Probes the current CUDA device's memory via cudaMemGetInfo.
///
/// Only meaningful when CUDA is enabled and a device is present; returns a
/// zeroed struct otherwise (checked the same way EriCudaEngine::Create()
/// gates on device presence - a no-device build or machine is not an
/// error, it is "no GPU tier").
///
/// \warning cudaMemGetInfo reports free memory across the WHOLE device,
/// not what this process is entitled to. On a single GPU used exclusively
/// by this project (the stated development setup) the distinction does not
/// matter; on a shared/multi-tenant GPU (a cluster node) a naive "size to
/// freeBytes" policy can over-commit when another process uses the device.
/// Callers must not silently assume single-tenant; state the assumption.
/// \returns The device totals; both zero when CUDA is off, no device is
/// present, or the probe fails.
/// \ingroup qcx-backend
GpuMemoryInfo DetectGpuMemory() noexcept;

} // namespace qcx::backend
