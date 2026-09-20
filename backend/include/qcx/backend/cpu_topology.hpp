#pragma once

/// \file
/// CPU topology probe and the default team-size policy.
///
/// Cluster nodes are homogeneous (no hyperthreading, all cores equal);
/// consumer hardware carries P-cores, E-cores, and hyperthreaded siblings -
/// treating every logical CPU as an equal OpenMP thread misloads both. The
/// probe reports the topology; DefaultTeamSize implements the policy:
/// OMP_NUM_THREADS wins when a scheduler set it (clusters), otherwise the
/// physical cores (hyperthreaded siblings excluded; E-cores excluded where
/// the probe can distinguish the classes - see the per-platform notes).
/// Nothing here changes any default by itself - the ExecutionContext
/// consumes it when it lands.

#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::backend {

/// The CPU topology of this node.
/// \ingroup qcx-backend
struct CpuTopology {
    std::size_t logicalProcessors; ///< OS-visible logical CPUs (groups summed).
    std::size_t physicalCores; ///< Physical cores (packages x cores).
    std::size_t performanceCores; ///< P-cores; equals physicalCores on uniform hardware.
    std::size_t efficiencyCores; ///< E-cores; 0 on uniform hardware.
    bool hyperthreaded; ///< True when logical > physical.
};

/// Counts the distinct physical cores described by (package, core-id)
/// pairs - the sysfs probe's core-counting primitive. core_id values are
/// per-package on sysfs (every socket numbers its cores from 0), so
/// deduping core ids alone undercounts multi-socket hosts by the socket
/// count; the distinct (package, core) pairs are the physical cores.
/// \param packageCoreIds One (physical_package_id, core_id) pair per
/// logical CPU, in any order.
/// \returns The number of distinct (package, core) combinations.
/// \ingroup qcx-backend
std::size_t CountDistinctPhysicalCores(
    const std::vector<std::pair<std::size_t, std::size_t>>& packageCoreIds) noexcept;

/// Probes this node's CPU topology.
/// \returns The topology; on probe failure a conservative flat estimate
/// (logical count from the C++ runtime, no hybrid split).
/// \ingroup qcx-backend
CpuTopology DetectCpuTopology() noexcept;

/// The default OpenMP team size for a topology: OMP_NUM_THREADS when set
/// and positive (the scheduler's pinning is authoritative on clusters),
/// else the P-cores when the probe reports a hybrid split, else the
/// physical cores.
/// \param topology The probed topology.
/// \returns The team size (>= 1).
/// \ingroup qcx-backend
std::size_t DefaultTeamSize(const CpuTopology& topology) noexcept;

} // namespace qcx::backend
