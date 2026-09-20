#include "qcx/backend/cpu_backend.hpp"
#include "qcx/backend/cpu_scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

namespace {

// Tiny accumulator type for the reduction test - the real consumers
// (molecule's inertia tensor etc.) use their own accumulator types.
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 Add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

} // namespace

TEST(CpuBackendTest, ParallelForTouchesAllElements) {
    qcx::backend::Backend<qcx::backend::CpuTag> backend;
    std::vector<int> data(1000, 0);
    backend.ParallelFor(data.size(), [&](std::size_t i) { data[i] = 1; });
    EXPECT_EQ(std::count(data.begin(), data.end(), 1), 1000);
}

TEST(CpuBackendTest, ParallelForAccumulatesAllIndices) {
    qcx::backend::Backend<qcx::backend::CpuTag> backend;
    constexpr std::size_t kCount = 10000;
    std::atomic<std::size_t> sum{0};
    backend.ParallelFor(kCount,
                        [&](std::size_t i) { sum.fetch_add(i, std::memory_order_relaxed); });
    EXPECT_EQ(sum.load(), kCount * (kCount - 1) / 2);
}

TEST(CpuSchedulerTest, SchedulesWorkOnThreadPool) {
    std::atomic<bool> ran = false;
    qcx::backend::ScheduleAndRun([&] { ran = true; });
    EXPECT_TRUE(ran.load());
}

TEST(CpuBackendTest, ParallelReduceSumsIntegers) {
    constexpr std::size_t kCount = 10000;
    const std::size_t total = qcx::backend::ParallelReduce<std::size_t>(
        kCount,
        0,
        [](const std::size_t& acc, std::size_t i) { return acc + i; },
        [](const std::size_t& a, const std::size_t& b) { return a + b; });
    EXPECT_EQ(total, kCount * (kCount - 1) / 2);
}

TEST(CpuBackendTest, ParallelReduceAccumulatesStruct) {
    constexpr std::size_t kCount = 5000;
    const Vec3 total = qcx::backend::ParallelReduce<Vec3>(
        kCount,
        Vec3{},
        [](const Vec3& acc, std::size_t i) {
            const double v = static_cast<double>(i);
            return Vec3{acc.x + v, acc.y + v, acc.z + v};
        },
        Add);
    const double expected = static_cast<double>(kCount * (kCount - 1)) / 2.0;
    EXPECT_DOUBLE_EQ(total.x, expected);
    EXPECT_DOUBLE_EQ(total.y, expected);
    EXPECT_DOUBLE_EQ(total.z, expected);
}

// The dynamic-schedule variant must sum identically and cover
// degenerate counts like the static default.
TEST(CpuBackendTest, ParallelReduceDynamicMatchesStatic) {
    constexpr std::size_t kCount = 10000;
    const std::size_t total = qcx::backend::ParallelReduce<std::size_t>(
        kCount,
        0,
        [](const std::size_t& acc, std::size_t i) { return acc + i; },
        [](const std::size_t& a, const std::size_t& b) { return a + b; },
        true);
    EXPECT_EQ(total, kCount * (kCount - 1) / 2);

    const std::size_t zero = qcx::backend::ParallelReduce<std::size_t>(
        0,
        0,
        [](const std::size_t& acc, std::size_t) { return acc + 1; },
        [](const std::size_t& a, const std::size_t& b) { return a + b; },
        true);
    EXPECT_EQ(zero, 0u);
}

TEST(CpuBackendTest, ParallelReduceHandlesDegenerateCounts) {
    const std::size_t zero = qcx::backend::ParallelReduce<std::size_t>(
        0,
        0,
        [](const std::size_t& acc, std::size_t) { return acc + 1; },
        [](const std::size_t& a, const std::size_t& b) { return a + b; });
    EXPECT_EQ(zero, 0u);

    const std::size_t one = qcx::backend::ParallelReduce<std::size_t>(
        1,
        0,
        [](const std::size_t& acc, std::size_t i) { return acc + i + 1; },
        [](const std::size_t& a, const std::size_t& b) { return a + b; });
    EXPECT_EQ(one, 1u);
}

TEST(CpuBackendTest, ParallelForDynamicMatchesParallelFor) {
    qcx::backend::Backend<qcx::backend::CpuTag> backend;
    std::vector<int> data(2000, 0);
    backend.ParallelForDynamic(data.size(), [&](std::size_t i) { data[i] = 1; });
    EXPECT_EQ(std::count(data.begin(), data.end(), 1), 2000);
}

TEST(CpuBackendTest, DefaultOmpTeamSizeIsPositiveAndBounded) {
    // The policy-filtered team never exceeds the logical processor count,
    // whatever the topology probe finds on this machine. Note the value is
    // the process-wide cache (warmed by the ParallelFor tests above), so
    // this asserts consistency with the policy, not first-call semantics -
    // those live in the dedicated team_size_test binary, which must be the
    // first thing to touch DefaultOmpTeamSize in its own process.
    const int team = qcx::backend::DefaultOmpTeamSize();
    EXPECT_GE(team, 1);
    // A degenerate probe (hardware_concurrency() == 0) pins the team to 1,
    // so the upper bound cannot be the raw logical count.
    const unsigned int logicalCount = std::thread::hardware_concurrency();
    EXPECT_LE(team, static_cast<int>(std::max(1u, logicalCount)));
}

// The thread ceiling clamps the team after the cache -
// set late, it still clamps; 1 yields exactly the one-thread team (the
// serial path at the backend level); clearing it restores the policy
// value. The ceiling is process-wide and sticky, so this test restores it
// before asserting: a failure can never leave the rest of the suite
// running clamped (and a clamped run is within the pins' tolerance anyway
// - the driver-level equivalence test pins that).
TEST(CpuBackendTest, ThreadCeilingClampsTheCachedTeam) {
    const int policyTeam = qcx::backend::DefaultOmpTeamSize();

    // The clamp is only observable with a team of at least two - a
    // one-core machine (CI runners included) cannot form one, so skip
    // rather than fail on the environment (the SimdMicroGemmTest pattern).
    if (policyTeam < 2)
    {
        GTEST_SKIP() << "this test needs a multi-thread team to clamp";
    }

    qcx::backend::SetOmpThreadCeiling(1);
    EXPECT_EQ(qcx::backend::DefaultOmpTeamSize(), 1);
    EXPECT_NE(qcx::backend::DefaultOmpTeamSize(), policyTeam);

    qcx::backend::SetOmpThreadCeiling(policyTeam + 1);
    EXPECT_EQ(qcx::backend::DefaultOmpTeamSize(), policyTeam)
        << "a ceiling above the policy team is not a floor";

    // Restore before asserting: the process-wide ceiling must never leak
    // into the remaining tests.
    qcx::backend::SetOmpThreadCeiling(0);
    EXPECT_EQ(qcx::backend::DefaultOmpTeamSize(), policyTeam);
}

// The runtime-pool bound (2026-09-18): the runtime sizes its pool by its own
// thread maximum, not by the num_threads() clause a region asks for, so the
// maximum is what a hard memory cap has to hold - the ri_jk fixture peaks
// 19.6 MiB higher at the logical-processor maximum than at the team, and
// under a 0.06 GiB cap the first configuration hangs in the runtime's pool
// allocation while the second completes. The ceiling setter is the run-start
// seam that knows the team, so it pins the maximum too. Asserted through
// OmpRuntimeMaxThreads rather than inferred: the pin's whole point is the
// value the runtime will use, which is not observable from the team alone.
TEST(CpuBackendTest, ThreadCeilingPinsTheOmpRuntimeMaximum) {
    const int policyTeam = qcx::backend::DefaultOmpTeamSize();

    qcx::backend::SetOmpThreadCeiling(1);
    EXPECT_EQ(qcx::backend::OmpRuntimeMaxThreads(), 1)
        << "the runtime's own maximum must follow the ceiling down";

    qcx::backend::SetOmpThreadCeiling(policyTeam + 1);
    EXPECT_EQ(qcx::backend::OmpRuntimeMaxThreads(), policyTeam)
        << "and must come back up to the team, not to the runtime default";

    qcx::backend::SetOmpThreadCeiling(0);
    EXPECT_EQ(qcx::backend::OmpRuntimeMaxThreads(), policyTeam);
}

// The invariant a run relies on: after the run-start ceiling call (the driver
// calls it unconditionally, thread_cap 0 included), the runtime can never
// create more threads than the engine's team. Without the pin the maximum is
// the logical processor count, which on a hyperthreaded host is up to twice
// the team.
TEST(CpuBackendTest, OmpRuntimeMaximumNeverExceedsTheTeam) {
    qcx::backend::SetOmpThreadCeiling(0);
    EXPECT_LE(qcx::backend::OmpRuntimeMaxThreads(), qcx::backend::DefaultOmpTeamSize());
}
