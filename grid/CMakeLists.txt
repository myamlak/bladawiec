# The qcx-grid module (molecular integration grids).
#
# Provides the radial (Euler-Maclaurin) and angular (Lebedev, recovered
# from first principles by tools/grid/gen_lebedev.py) quadratures, the
# per-atom product grid, the Becke/SSF atomic partition, the molecular
# grid that combines them, and the AO-at-point evaluator.  The grid
# consumers (Hirshfeld/Voronoi charges, ESP fits) live in the properties
# module.  Sits between basisset and integrals in the module DAG
# (check_dag.py).
add_library(qcx-grid
  src/angular_grid.cpp
  src/radial_grid.cpp
  src/atomic_grid.cpp
  src/partition_function.cpp
  src/molecular_grid.cpp
  src/ao_evaluator.cpp
  src/geometry_translation.cpp
  src/xc_grid_engine.cpp
)
target_include_directories(qcx-grid PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-grid PUBLIC cxx_std_23)
# The public headers expose core, molecule, and basisset types (the AO
# evaluator consumes BasisSet). The XC engine's public headers additionally
# expose Eigen matrices and the excgrid geometry/block/kernel types, so those
# include paths travel with the link - PRIVATE would
# only be correct if the engine header hid every excgrid type.
target_link_libraries(qcx-grid
  PUBLIC qcx-core qcx-molecule qcx-basisset eigen excgrid::excgrid)

# PCH is mandatory on every first-party target.
target_precompile_headers(qcx-grid PRIVATE
    <qcx/grid/angular_grid.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-grid-tests
        tests/angular_grid_test.cpp
        tests/radial_grid_test.cpp
        tests/atomic_grid_test.cpp
        tests/partition_function_test.cpp
        tests/molecular_grid_test.cpp
        tests/ao_evaluator_test.cpp
        tests/xc_assembly_test.cpp
        tests/quadrature_cross_validation_test.cpp
        tests/synthetic_fixture.cpp
        tests/synthetic_fixture_test.cpp
        tests/xc_screening_test.cpp)
    target_link_libraries(qcx-grid-tests PRIVATE qcx-grid qcx-test-fixtures gtest gtest_main)

    qcx_apply_asan_runtime(qcx-grid-tests)
    qcx_apply_fast_test_mode(qcx-grid-tests)

    target_precompile_headers(qcx-grid-tests PRIVATE
        <gtest/gtest.h>
        <qcx/grid/angular_grid.hpp>)

    include(GoogleTest)
    gtest_discover_tests(qcx-grid-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})

    # THE LARGE CATEGORY (the root CMakeLists carries the mechanism). These
    # two leave the routine selection and run with
    # `ctest -L large -j`.
    #
    # THE ASYMMETRY THAT PUT A DEBUG-ONLY COST IN A RELEASE-NAMED CATEGORY,
    # stated because it is the reason the membership rule is "long in ANY
    # regular configuration" rather than "long in Release": measured on this
    # tree (2026-09-18), ScreenedPairTermsScale costs 363 s in Debug against
    # ~14 s in Release, and ZeroToleranceReproducesTheDensePath 303 s against
    # ~14 s. Against a 60 s bar neither registers in Release at all and both
    # register hugely in Debug, and it was exactly this file - plus
    # footprint_test.cpp - that carried ZERO qcx::testing::IsFastOnlyMode()
    # calls, so the per-test fast-mode gate was wired (grid/CMakeLists.txt calls
    # qcx_apply_fast_test_mode below) and inert in the one file that needed
    # it. A per-test opt-in cannot be made to fail safe; a manifest can, and
    # that is what this is.
    #
    # This is a manifest rather than a directory move because
    # xc_screening_test.cpp carries 9 cases of which only these 2 are long;
    # splitting the file is the follow-up, naming them here is what takes the
    # Debug leg down today.
    qcx_mark_large_cases(grid
        XcScreeningTest.ScreenedPairTermsScaleWithTheKeptCountNotTheBasisSize
        XcScreeningTest.ZeroToleranceReproducesTheDensePathForEveryAngularMomentum)
endif()
