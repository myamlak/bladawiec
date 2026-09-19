// The POSIX CPU topology probe: sysfs (distinct (package, core-id) pairs
// per physical core; cpu_capacity below 80% of the node maximum marks an
// E-core) and the default team-size policy.

#include "qcx/backend/cpu_topology.hpp"

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace qcx::backend {

std::size_t CountDistinctPhysicalCores(
    const std::vector<std::pair<std::size_t, std::size_t>>& packageCoreIds) noexcept {
    std::set<std::pair<std::size_t, std::size_t>> distinct;

    for (const auto& entry : packageCoreIds)
    {
        distinct.insert(entry);
    }

    return distinct.size();
}

CpuTopology DetectCpuTopology() noexcept {
    CpuTopology topology{};
    topology.logicalProcessors = static_cast<std::size_t>(std::thread::hardware_concurrency());
    topology.physicalCores = 0;
    topology.performanceCores = 0;
    topology.efficiencyCores = 0;
    topology.hyperthreaded = false;

    // The per-logical-CPU (package, core) pairs and capacities from sysfs.
    // core_id alone is per-package - every socket numbers its cores from
    // 0 - so the distinct pairs, not the distinct core ids, are the
    // physical cores (CountDistinctPhysicalCores).
    std::vector<std::pair<std::size_t, std::size_t>> packageCoreIds;
    std::vector<std::pair<std::pair<std::size_t, std::size_t>, long>> capacities;
    std::size_t cpu = 0;

    for (;; ++cpu)
    {
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
        std::ifstream coreFile(base + "/topology/core_id");
        std::size_t coreId = 0;

        if (!(coreFile >> coreId))
        {
            break; // no more CPUs
        }

        // physical_package_id lives in the same topology dir; kernels
        // without it behave as a single package (package 0).
        std::size_t packageId = 0;
        std::ifstream packageFile(base + "/topology/physical_package_id");
        static_cast<void>(packageFile >> packageId);
        packageCoreIds.push_back({packageId, coreId});
        std::ifstream capacityFile(base + "/cpu_capacity");
        long capacity = 0;

        if (capacityFile >> capacity)
        {
            capacities.push_back({{packageId, coreId}, capacity});
        }
    }

    topology.physicalCores = CountDistinctPhysicalCores(packageCoreIds);

    if (topology.physicalCores == 0)
    {
        // Probe failure: a conservative flat estimate.
        topology.physicalCores = topology.logicalProcessors;
        topology.performanceCores = topology.logicalProcessors;
        return topology;
    }

    long maxCapacity = 0;

    for (const auto& entry : capacities)
    {
        if (entry.second > maxCapacity)
        {
            maxCapacity = entry.second;
        }
    }

    if (maxCapacity > 0 && capacities.size() == topology.physicalCores)
    {
        std::set<std::pair<std::size_t, std::size_t>> efficiencyCores;

        for (const auto& entry : capacities)
        {
            if (entry.second * 5 < maxCapacity * 4)
            {
                efficiencyCores.insert(entry.first);
            }
        }

        topology.efficiencyCores = efficiencyCores.size();
        topology.performanceCores = topology.physicalCores - topology.efficiencyCores;
    } else
    {
        topology.performanceCores = topology.physicalCores;
    }

    topology.hyperthreaded = topology.logicalProcessors > topology.physicalCores;
    return topology;
}

std::size_t DefaultTeamSize(const CpuTopology& topology) noexcept {
    // OMP_NUM_THREADS wins: a scheduler's pinning is authoritative.
    if (const char* requested = std::getenv("OMP_NUM_THREADS"))
    {
        const int parsed = std::atoi(requested);

        if (parsed > 0)
        {
            return static_cast<std::size_t>(parsed);
        }
    }

    if (topology.efficiencyCores > 0 && topology.performanceCores > 0)
    {
        // Hybrid parts: the P-cores, hyperthreaded siblings excluded.
        return topology.performanceCores;
    }

    return topology.physicalCores > 0 ? topology.physicalCores : 1;
}

} // namespace qcx::backend
