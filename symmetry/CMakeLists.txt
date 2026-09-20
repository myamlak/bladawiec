add_library(qcx-symmetry
  src/detection.cpp
  src/salc.cpp
)
target_include_directories(qcx-symmetry PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-symmetry PUBLIC cxx_std_23)
target_link_libraries(qcx-symmetry PUBLIC qcx-molecule eigen)

# PCH (angle-bracket entries resolve via the target's include dirs).
target_precompile_headers(qcx-symmetry PRIVATE
    <qcx/symmetry/detection.hpp>
    <Eigen/Dense>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-symmetry-tests
      tests/detection_test.cpp
      tests/character_tables_test.cpp
      tests/salc_test.cpp
      tests/full_group_tables_test.cpp
    )
    target_link_libraries(qcx-symmetry-tests PRIVATE qcx-symmetry qcx-test-fixtures qcx-linalg gtest gtest_main)

    qcx_apply_asan_runtime(qcx-symmetry-tests)
    qcx_apply_blas_runtime(qcx-symmetry-tests)

    target_precompile_headers(qcx-symmetry-tests PRIVATE
        <gtest/gtest.h>
        <qcx/symmetry/detection.hpp>
        <qcx/symmetry/character_tables.hpp>
        <qcx/symmetry/full_group_tables.hpp>)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # Same MKL discovery rules as linalg: DLLs must exist before
        # the exe runs, and sequential threading avoids OpenMP conflicts.
        gtest_discover_tests(qcx-symmetry-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-symmetry-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
