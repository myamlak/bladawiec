// The topology-profile probe (topology_profile.hpp): composes the CPU, host
// and device probes into the one bundle the algorithm-selection heuristic
// consumes. The device sweep's real implementation lives in the CUDA-gated
// TU of qcx-backend-cuda (cuda_backend.cpp); this TU carries the no-CUDA
// fallback under #ifndef QcxHasCuda, the same pattern as the
// DetectGpuMemory fallback in the memory-topology TUs.

#include "qcx/backend/topology_profile.hpp"

namespace qcx::backend {

#ifndef QcxHasCuda

std::vector<DeviceProfile> DetectDevices() noexcept {
    // No CUDA tier in a non-CUDA build: the empty sweep is "no GPU tier"
    // (the header contract), never an error.
    return {};
}

#endif

TopologyProfile DetectTopologyProfile(bool includeDevices) noexcept {
    TopologyProfile profile;
    profile.cpu = DetectCpuTopology();
    profile.hostMemory = DetectHostMemory();

    if (includeDevices)
    {
        profile.devices = DetectDevices();
    }

    // nodeCount and interconnect keep their single-node defaults - MPI is
    // not implemented and nothing links it today.
    return profile;
}

} // namespace qcx::backend
