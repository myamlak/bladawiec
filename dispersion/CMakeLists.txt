# The qcx-dispersion module (geometry-only London dispersion corrections).
#
# Provides the D3 zero-damping energy and its coordinate gradient and the D4
# two-body energy the run driver adds to the total: the request vocabulary,
# the parameter resolution and the evaluation seam onto excgrid. Sits after
# grid in the module DAG
# (check_dag.py): the geometry translation and the excgrid error translation
# are grid's, and io (which carries the parsed request in RunInput) and the
# driver come after it. The module depends on neither io nor scf - an input
# parser fills the request type and the driver resolves and evaluates it.
add_library(qcx-dispersion
  src/dispersion.cpp
)
target_include_directories(qcx-dispersion PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-dispersion PUBLIC cxx_std_23)
# The public options header is plain std types; the implementation consumes
# the excgrid kernels and grid's translation seam, so those travel with the
# link. molecule is public because EvaluateDispersion takes a Molecule.
target_link_libraries(qcx-dispersion
  PUBLIC qcx-core qcx-molecule qcx-grid excgrid::excgrid)

# PCH is mandatory on every first-party target.
target_precompile_headers(qcx-dispersion PRIVATE
    <qcx/dispersion/dispersion.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-dispersion-tests
        tests/dispersion_test.cpp
        tests/dispersion_gradient_test.cpp)
    target_link_libraries(qcx-dispersion-tests PRIVATE
        qcx-dispersion gtest gtest_main)

    qcx_apply_asan_runtime(qcx-dispersion-tests)
    qcx_apply_fast_test_mode(qcx-dispersion-tests)

    target_precompile_headers(qcx-dispersion-tests PRIVATE
        <gtest/gtest.h>
        <qcx/dispersion/dispersion.hpp>)

    include(GoogleTest)
    gtest_discover_tests(qcx-dispersion-tests DISCOVERY_TIMEOUT ${QCX_TEST_DISCOVERY_TIMEOUT_SECONDS} PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
endif()
