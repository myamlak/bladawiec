#pragma once

/// \file
/// The node's topology profile: the single input the
/// algorithm-selection heuristic's cost table, the device budgets, and the
/// precision-ladder referee cadence consume. One bundle probed once at driver
/// start - nodes, cores per node (the P/E split of DetectCpuTopology), devices
/// per node with VRAM each, and the interconnect class.
///
/// Staged reality (explicit, per the design): only the single-node class is
/// implemented. Nothing distributes a Fock build across nodes today and
/// multi-GPU has no plan yet, so nodeCount is always 1 and interconnect is
/// always kUnknown; the
/// struct lands early so the cost table never needs reshaping when the
/// paths arrive.

#include "qcx/backend/cpu_topology.hpp"
#include "qcx/backend/memory_topology.hpp"

#include <cstddef>
#include <vector>

namespace qcx::backend {

/// The interconnect class between the profile's compute nodes. kUnknown
/// until a distributed build lands; every non-single-node candidate
/// reports "not selectable" through the cost table's stub columns.
/// \ingroup qcx-backend
enum class InterconnectClass {
    kUnknown, ///< No interconnect probe exists yet (MPI is not linked).
};

/// One CUDA device of this node: its id and its device memory, probed
/// once. The free-bytes value is a point-in-time snapshot, not a
/// reservation - the same caveat as GpuMemoryInfo::freeBytes.
/// \ingroup qcx-backend
struct DeviceProfile {
    int deviceId = 0; ///< The CUDA device id (cudaGetDeviceCount order).
    std::size_t totalBytes = 0; ///< Total device memory.
    std::size_t freeBytes = 0; ///< Free right now - "hint, not guarantee".
};

/// The node's topology bundle.
/// \ingroup qcx-backend
struct TopologyProfile {
    /// The node count: the MPI world size when a distributed build is
    /// linked, else 1. Always 1 today - nothing distributes a Fock build yet.
    std::size_t nodeCount = 1;
    /// This node's CPU topology (the P/E split of DetectCpuTopology).
    CpuTopology cpu;
    /// This node's host RAM (a point-in-time snapshot).
    HostMemoryInfo hostMemory;
    /// This node's CUDA devices with VRAM each; empty = no device tier.
    std::vector<DeviceProfile> devices;
    /// The interconnect class between nodes; kUnknown until MPI lands.
    InterconnectClass interconnect = InterconnectClass::kUnknown;
};

/// The CUDA device sweep of this node: the device count and each device's
/// total/free memory. The free-VRAM half of the device-profile probe the
/// precision-ladder lane (Package B) shares.
///
/// Returns an empty vector when CUDA is off, no device is present, or a
/// device query fails - the same "no GPU tier" semantics as
/// DetectGpuMemory. A device whose own query fails is skipped, not fatal.
/// \returns The per-device profiles in cudaGetDeviceCount order.
/// \ingroup qcx-backend
std::vector<DeviceProfile> DetectDevices() noexcept;

/// Probes this node's topology once at driver start: DetectCpuTopology +
/// DetectHostMemory + DetectDevices, all probe-failure-safe (the CPU and
/// host probes fall back to their documented conservative estimates, the
/// device sweep to empty). Pure probe - no state, no caching; the caller
/// decides when to call it.
///
/// \param includeDevices True to sweep the CUDA devices (default). False
/// skips the device half only - the CPU and host halves still probe - for
/// a caller whose memory cap cannot hold the CUDA build's host footprint
/// (the probe gate; the runtime DLLs are load-time, so a CUDA-linked
/// process already holds the footprint against the cap): the sweep
/// initializes the runtime, and under a sub-footprint cap that init is a
/// hard access violation, never a clean CUDA error. The caller loses
/// nothing: below that floor the GPU tier is admission-floored off by the
/// cap, so the device half could not change any selection.
/// \returns The topology profile.
/// \ingroup qcx-backend
TopologyProfile DetectTopologyProfile(bool includeDevices = true) noexcept;

} // namespace qcx::backend
