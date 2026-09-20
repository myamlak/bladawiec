// The Windows CPU topology probe: GetLogicalProcessorInformationEx over
// RelationProcessorCore (per-core groups carry the EfficiencyClass on
// Windows 11 hybrid parts; class 0 = pre-hybrid, treat as P), and the
// default team-size policy.

#include "qcx/backend/cpu_topology.hpp"

#include <cstdlib>
#include <set>
#include <thread>
#include <utility>
#include <vector>
#include <windows.h>

namespace qcx::backend {

std::size_t CountDistinctPhysicalCores(
    const std::vector<std::pair<std::size_t, std::size_t>>& packageCoreIds) noexcept {
    // The shared declaration lives in the header; the Windows probe does
    // not use it (GetLogicalProcessorInformationEx returns one entry per
    // physical core already), but the sysfs-based POSIX probe counts
    // through it - one definition per platform file.
    std::set<std::pair<std::size_t, std::size_t>> distinct;

    for (const auto& entry : packageCoreIds)
    {
        distinct.insert(entry);
    }

    return distinct.size();
}

CpuTopology DetectCpuTopology() noexcept {
    CpuTopology topology{};
    topology.logicalProcessors =
        static_cast<std::size_t>(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS));
    topology.physicalCores = 0;
    topology.performanceCores = 0;
    topology.efficiencyCores = 0;
    topology.hyperthreaded = false;

    // The first pass sizes the buffer; every processor appears in one
    // group of one RelationProcessorCore entry.
    DWORD bytes = 0;
    (void)GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes);

    if (bytes == 0)
    {
        // Probe failure: a conservative flat estimate.
        topology.physicalCores = topology.logicalProcessors;
        topology.performanceCores = topology.logicalProcessors;
        return topology;
    }

    std::vector<BYTE> buffer(bytes);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data());

    if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &bytes) == FALSE)
    {
        topology.physicalCores = topology.logicalProcessors;
        topology.performanceCores = topology.logicalProcessors;
        return topology;
    }

    for (DWORD offset = 0; offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bytes;)
    {
        info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + offset);

        if (info->Relationship == RelationProcessorCore)
        {
            // On hybrid parts each entry carries one EfficiencyClass (the
            // relationship-level field) with GroupCount cores of that
            // class; pre-hybrid parts report class 0.
            const BYTE efficiencyClass = info->Processor.EfficiencyClass;

            for (WORD group = 0; group < info->Processor.GroupCount; ++group)
            {
                topology.physicalCores += 1;

                // The efficiency-class convention is vendor-dependent:
                // Intel hybrid parts report P = 0, E = 1, AMD hybrid parts
                // the inverted scheme (P = 1, E = 0). No single threshold
                // can exclude the E-cores on both vendors, and counting an
                // E-core as a P-core only costs a little speed while
                // counting a P-core as an E-core would halve the team - so
                // the <= 1 split keeps every physical core on the
                // performance side and the E-core exclusion is left to the
                // POSIX probe (sysfs cpu_capacity), where the classes are
                // actually distinguishable.

                if (efficiencyClass <= 1)
                {
                    topology.performanceCores += 1;
                } else
                {
                    topology.efficiencyCores += 1;
                }
            }
        }

        if (info->Size == 0)
        {
            break;
        }

        offset += info->Size;
    }

    if (topology.performanceCores == 0)
    {
        // Pre-hybrid parts report EfficiencyClass 0; the fallback above
        // already handled the probe-failure case.
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
