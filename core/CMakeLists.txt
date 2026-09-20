add_library(qcx-core
  src/error.cpp
  src/logging.cpp
)
target_include_directories(qcx-core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-core PUBLIC cxx_std_23)
target_link_libraries(qcx-core PUBLIC spdlog::spdlog)

# spdlog + std::string dominate every core TU; one PCH per target (PRIVATE -
# PCH never propagates through linking; test targets carry their own lists
# because REUSE_FROM cannot combine with header lists).
# PCH entries in angle brackets pass through verbatim and resolve via the
# target's include dirs; bare relative paths would be source-dir-prefixed.
target_precompile_headers(qcx-core PRIVATE
    <spdlog/spdlog.h>
    <qcx/error.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-core-tests tests/error_test.cpp)
    target_link_libraries(qcx-core-tests PRIVATE qcx-core gtest gtest_main)
    target_precompile_headers(qcx-core-tests PRIVATE
        <gtest/gtest.h>
        <qcx/error.hpp>)

    qcx_apply_asan_runtime(qcx-core-tests)

    include(GoogleTest)
    gtest_discover_tests(qcx-core-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
endif()
