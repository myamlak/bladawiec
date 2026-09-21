add_library(qcx-response
  src/response_operator.cpp
  src/response_solver.cpp
)
target_include_directories(qcx-response PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-response PUBLIC cxx_std_23)
# qcx-core is the ONLY dependency of the library: the headers expose qcx::Result
# and std::span/std::vector, and no molecule, basis, integral or SCF type appears
# in a signature. Eigen is deliberately absent too - the operator seam is written
# against spans, so a consumer that has never heard of Eigen can use it. The
# physical operator a caller wants (OrbitalHessianOperator) consumes the MO
# two-electron tensor and the orbital energies as plain spans for that reason.
target_link_libraries(qcx-response PUBLIC qcx-core)

# PCH is mandatory on every target; response_solver.hpp carries the operator
# header, so this one entry covers both translation units.
target_precompile_headers(qcx-response PRIVATE
    <qcx/response/response_solver.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-response-tests
      tests/response_solver_test.cpp
    )
    # The C1 anchor runs the real H2/STO-3G SCF at three field strengths, so the
    # test target pulls scf, integrals, basisset and the shared H2/STO-3G
    # fixtures in - qcx-test-fixtures is defined in tests/, which the root
    # CMakeLists adds after this directory (CMake resolves the name at generate
    # time, the same edge qcx-scf-tests uses). The library's own dependencies
    # are unchanged by this: a test-only edge, and one the DAG permits (scf is
    # rank 9, this module sits after it).
    target_link_libraries(qcx-response-tests PRIVATE
        qcx-response qcx-scf qcx-integrals gtest gtest_main qcx-test-fixtures)
    target_precompile_headers(qcx-response-tests PRIVATE
        <gtest/gtest.h>
        <qcx/response/response_solver.hpp>)

    qcx_apply_asan_runtime(qcx-response-tests)
    qcx_apply_blas_runtime(qcx-response-tests)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # PRE_TEST discovery: the discovery process does not inherit the test's
        # environment, so the MKL DLLs copied above must be present before the
        # exe runs. Sequential threading avoids Intel-OpenMP/vcomp conflicts -
        # same wiring as qcx-linalg-tests and qcx-scf-tests.
        gtest_discover_tests(qcx-response-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-response-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
