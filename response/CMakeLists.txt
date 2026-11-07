add_library(qcx-response ${QCX_MODULE_LIBRARY_TYPE}
  src/response_operator.cpp
  src/response_solver.cpp
  src/response_gradient.cpp
  src/response_kernel.cpp
  src/response_band.cpp
  src/response_state.cpp
  src/matrix_free_eri_derivative.cpp
)
target_include_directories(qcx-response PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-response PUBLIC cxx_std_23)
# The operator seam itself is spans and qcx::Result. The matrix-free derivative
# is the one place a physical type appears: it consumes the molecule and the
# basis set and runs the direct Fock engine, so qcx-integrals is a public
# dependency (its AccuracyPreset is in the options struct the header exposes)
# and molecule/basis enter through forward declarations in the header and the
# engine's headers in the implementation. The DAG permits all three - they sit
# to the left of response. Eigen is still absent: the seam's signatures are
# spans, so a consumer that has never heard of Eigen can use the operator.
target_link_libraries(qcx-response PUBLIC qcx-core qcx-linalg qcx-integrals)

# PCH is mandatory on every target; response_solver.hpp carries the operator
# header, so this one entry covers both translation units.
target_precompile_headers(qcx-response PRIVATE
    <qcx/response/response_solver.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-response-tests
      tests/response_solver_test.cpp
      tests/response_gradient_test.cpp
      tests/response_kernel_test.cpp
      tests/response_band_test.cpp
      tests/response_state_test.cpp
    )
    # The C1 anchor runs the real H2/STO-3G SCF at three field strengths, so the
    # test target pulls scf, integrals, basisset and the shared H2/STO-3G
    # fixtures in - qcx-test-fixtures is defined in tests/, which the root
    # CMakeLists adds after this directory (CMake resolves the name at generate
    # time, the same edge qcx-scf-tests uses). The Kohn-Sham kernel test adds
    # the grid: its finite-difference reference integrates the XC energy of a
    # converged RKS reference. The library's own dependencies are unchanged by
    # this: test-only edges, and ones the DAG permits (response follows grid and
    # scf in the module order).
    target_link_libraries(qcx-response-tests PRIVATE
        ${QCX_LINK_response} ${QCX_LINK_scf} ${QCX_LINK_integrals} ${QCX_LINK_grid} gtest gtest_main qcx-test-fixtures)
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
        gtest_discover_tests(qcx-response-tests DISCOVERY_TIMEOUT ${QCX_TEST_DISCOVERY_TIMEOUT_SECONDS} PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
