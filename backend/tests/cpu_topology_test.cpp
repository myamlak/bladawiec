// CPU topology probe and the default team-size policy: the probe reports a
// consistent topology, and the policy prefers OMP_NUM_THREADS, then the
// P-cores on hybrid parts, then the physical cores.

#include "qcx/backend/cpu_topology.hpp"

#include <cstdlib>
#include <gtest/gtest.h>

namespace {

TEST(CpuTopologyTest, DistinctPhysicalCoreCountHandlesMultiSocket) {
    // sysfs core_id values are per-package: two sockets both number their
    // cores 0..15, so deduping core ids alone would report 16 physical
    // cores where there are 32 - the distinct (package, core) pairs are
    // the physical cores.
    std::vector<std::pair<std::size_t, std::size_t>> twoSockets;

    for (std::size_t package = 0; package < 2; ++package)
    {
        for (std::size_t core = 0; core < 16; ++core)
        {
            twoSockets.push_back({package, core});
        }
    }

    EXPECT_EQ(qcx::backend::CountDistinctPhysicalCores(twoSockets), 32u);

    // A hyperthreaded single socket: both logical CPUs of a core share
    // the (package, core) pair.
    const std::vector<std::pair<std::size_t, std::size_t>> htSingleSocket = {
        {0, 0}, {0, 0}, {0, 1}, {0, 1}, {0, 2}, {0, 2}};
    EXPECT_EQ(qcx::backend::CountDistinctPhysicalCores(htSingleSocket), 3u);

    // A probe failure (no sysfs entries) still counts zero, keeping the
    // conservative fallback path live.
    EXPECT_EQ(qcx::backend::CountDistinctPhysicalCores({}), 0u);
}

TEST(CpuTopologyTest, ProbeIsConsistent) {
    const qcx::backend::CpuTopology topology = qcx::backend::DetectCpuTopology();
    EXPECT_GE(topology.logicalProcessors, 1u);
    EXPECT_GE(topology.physicalCores, 1u);
    EXPECT_LE(topology.physicalCores, topology.logicalProcessors);
    EXPECT_EQ(topology.performanceCores + topology.efficiencyCores, topology.physicalCores);
    EXPECT_EQ(topology.hyperthreaded, topology.logicalProcessors > topology.physicalCores);
}

TEST(CpuTopologyTest, PolicyPrefersOmpNumThreads) {
    const qcx::backend::CpuTopology flat{12, 6, 6, 0, true};

    const std::string previous = [] {
        const char* value = std::getenv("OMP_NUM_THREADS");
        return value == nullptr ? std::string() : std::string(value);
    }();
    const int setResult = [] {
#ifdef _WIN32
        return _putenv_s("OMP_NUM_THREADS", "3");
#else
        return setenv("OMP_NUM_THREADS", "3", 1);
#endif
    }();

    if (setResult == 0)
    {
        EXPECT_EQ(qcx::backend::DefaultTeamSize(flat), 3u);
    }

    // Restore the environment.
#ifdef _WIN32
    if (previous.empty())
    {
        _putenv_s("OMP_NUM_THREADS", "");
    } else
    {
        _putenv_s("OMP_NUM_THREADS", previous.c_str());
    }
#else
    if (previous.empty())
    {
        unsetenv("OMP_NUM_THREADS");
    } else
    {
        setenv("OMP_NUM_THREADS", previous.c_str(), 1);
    }
#endif
}

TEST(CpuTopologyTest, PolicyChoosesTheFastCores) {
    // A hybrid part: 16 logical (8 P x 2 HT + 8 E), the team is the 8
    // physical P-cores.
    const qcx::backend::CpuTopology hybrid{16, 16, 8, 8, false};
    EXPECT_EQ(qcx::backend::DefaultTeamSize(hybrid), 8u);

    // A uniform cluster node: every physical core.
    const qcx::backend::CpuTopology uniform{64, 64, 64, 0, false};
    EXPECT_EQ(qcx::backend::DefaultTeamSize(uniform), 64u);

    // A consumer part with HT: physical cores, siblings excluded.
    const qcx::backend::CpuTopology hyperthreaded{12, 6, 6, 0, true};
    EXPECT_EQ(qcx::backend::DefaultTeamSize(hyperthreaded), 6u);

    // A degenerate probe result still yields a usable team.
    const qcx::backend::CpuTopology degenerate{0, 0, 0, 0, false};
    EXPECT_EQ(qcx::backend::DefaultTeamSize(degenerate), 1u);
}

} // namespace
