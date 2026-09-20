# The properties module (populations, bond orders, multipole
# moments, condensed indices, density partitioning charges, ESP fits,
# EDDB delocalization, Nalewajski-Mrozek bond orders, ETS-NOCV, density
# at the nuclei).
# Consumes the scf results' densities through pure matrix algebra and
# grid quadrature; links integrals for the overlap/dipole/quadrupole
# matrices and the shell list. Sits after scf in the module DAG
# (check_dag.py).
add_library(qcx-properties
  src/populations.cpp
  src/multipoles.cpp
  src/fukui.cpp
  src/charges.cpp
  src/esp.cpp
  src/eddb.cpp
  src/nalewajski.cpp
  src/nocv.cpp
  src/density_at_nuclei.cpp
  src/qtaim.cpp
)
target_include_directories(qcx-properties PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(qcx-properties PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(qcx-properties PUBLIC cxx_std_23)
# The public headers expose Molecule, BasisSet, MolecularGrid, and Eigen
# matrix types, so the consumers need those include paths and link
# targets too; integrals stays implementation-only (the overlap/moment
# builders and the shell list are called from the .cpp files).
target_link_libraries(qcx-properties
  PUBLIC qcx-core qcx-molecule qcx-basisset qcx-grid eigen
  PRIVATE qcx-integrals qcx-scf)

# PCH is mandatory on every target.
target_precompile_headers(qcx-properties PRIVATE
    <qcx/properties/populations.hpp>
    <qcx/properties/multipoles.hpp>
    <qcx/properties/fukui.hpp>
    <qcx/properties/charges.hpp>
    <qcx/properties/esp.hpp>
    <qcx/properties/eddb.hpp>
    <qcx/properties/nalewajski.hpp>
    <qcx/properties/nocv.hpp>
    <qcx/properties/density_at_nuclei.hpp>
    <qcx/properties/qtaim.hpp>
    <qcx/grid/molecular_grid.hpp>
    <Eigen/Dense>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-properties-tests
        tests/populations_test.cpp
        tests/multipoles_test.cpp
        tests/fukui_test.cpp
        tests/charges_test.cpp
        tests/esp_test.cpp
        tests/eddb_test.cpp
        tests/nalewajski_test.cpp
        tests/nocv_test.cpp
        tests/density_at_nuclei_test.cpp
        tests/qtaim_test.cpp)
    target_link_libraries(qcx-properties-tests PRIVATE qcx-properties qcx-integrals qcx-scf qcx-grid qcx-test-fixtures gtest gtest_main)

    qcx_apply_asan_runtime(qcx-properties-tests)
    qcx_apply_blas_runtime(qcx-properties-tests)
    qcx_apply_fast_test_mode(qcx-properties-tests)

    target_precompile_headers(qcx-properties-tests PRIVATE
        <gtest/gtest.h>
        <qcx/properties/populations.hpp>
        <qcx/properties/multipoles.hpp>
        <qcx/properties/fukui.hpp>
        <qcx/properties/charges.hpp>
        <qcx/properties/esp.hpp>
        <qcx/properties/eddb.hpp>
        <qcx/properties/nalewajski.hpp>
        <qcx/properties/nocv.hpp>
        <qcx/properties/density_at_nuclei.hpp>
        <qcx/properties/qtaim.hpp>)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # PRE_TEST discovery: the discovery process does not inherit the
        # test's environment, so the MKL DLLs copied above must be present
        # before the exe runs - same wiring as qcx-scf-tests.
        gtest_discover_tests(qcx-properties-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-properties-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
