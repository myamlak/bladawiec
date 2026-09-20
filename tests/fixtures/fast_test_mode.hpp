#pragma once

// The Debug fast-smoke gate. MSVC Debug runs heavy numerical tests
// 10-50x slower than Release (no optimization + STL iterator checking on
// dense MD loops), so heavy tests - full reference grids, molecule SCF pins
// - run in Release only. The root CMake helper qcx_apply_fast_test_mode
// defines QcxTestFastOnly per configuration (a generator expression: the
// local preset is multi-config MSBuild, CMAKE_BUILD_TYPE is empty there);
// heavy tests gate themselves with GTEST_SKIP. Debug keeps its unique value
// - the STL iterator checking that catches memory errors Release masks -
// but runs the fast smoke subset only. Test-only helper - not part of the
// public API.

namespace qcx::testing {

/// True when this build configuration runs the fast smoke subset only
/// (QcxTestFastOnly is defined in Debug builds of the gated test targets).
/// Heavy numerical tests skip themselves through this check.
inline bool IsFastOnlyMode() noexcept {
#ifdef QcxTestFastOnly
    return true;
#else
    return false;
#endif
}

} // namespace qcx::testing
