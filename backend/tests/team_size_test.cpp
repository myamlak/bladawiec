// The first-call semantics of DefaultOmpTeamSize: OMP_NUM_THREADS set in
// the process environment before the first call wins the process-wide
// cache. This lives in its OWN test binary because the cache warms on the
// first call anywhere in the process - sharing qcx-backend-tests (which
// calls ParallelFor early, warming the cache) would silently test the
// cached value, not the env-honoring policy, and would look broken when it
// is only mis-sequenced. Do not merge this file into backend_test.cpp.

#include "qcx/backend/cpu_backend.hpp"

#include <cstdlib>
#include <gtest/gtest.h>
#include <string>

namespace {

TEST(TeamSizeTest, OmpNumThreadsWinsOnFirstCall) {
    // Only honored on the first call (the function-local cache). This test
    // binary has exactly one test, so this IS the first call in the process
    // - keep it that way; adding a sibling test here breaks the assumption.
    constexpr int kRequestedTeam = 3;

    const std::string previous = [] {
        const char* value = std::getenv("OMP_NUM_THREADS");
        return value == nullptr ? std::string() : std::string(value);
    }();
    const int setResult = [] {
#ifdef _WIN32
        return _putenv_s("OMP_NUM_THREADS", std::to_string(kRequestedTeam).c_str());
#else
        return setenv("OMP_NUM_THREADS", std::to_string(kRequestedTeam).c_str(), 1);
#endif
    }();
    ASSERT_EQ(setResult, 0) << "the environment must allow setting OMP_NUM_THREADS";

    EXPECT_EQ(qcx::backend::DefaultOmpTeamSize(), kRequestedTeam);

    // Restore the environment (the save/restore pattern of cpu_topology_test).
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

} // namespace
