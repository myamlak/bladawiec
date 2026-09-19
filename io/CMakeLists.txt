add_library(qcx-io
  src/parse_input.cpp
  src/result_json.cpp
  src/validate_input.cpp
  src/molden_export.cpp
  src/selection_coefficients.cpp
)
target_include_directories(qcx-io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-io PUBLIC cxx_std_23)
# The public headers expose the AccuracyPreset (accuracy.hpp), the
# Result/Error types and the Eigen matrix types (the MoldenMolecularOrbitals
# struct of molden_export.hpp, the Molden writer - which forward-declares
# Molecule/BasisSet, so those stay PRIVATE implementation deps), so
# qcx-integrals, qcx-core and eigen are PUBLIC; tomlplusplus and
# nlohmann/json stay PRIVATE (implementation-only, no public-header
# exposure - the schema structs use std types only).
# qcx-grid is PRIVATE and implementation-only for the same reason: the
# [grid] block's parser refuses an angular size that is not a shipped
# Lebedev one BY NAME, so it reads the size list from the module that owns
# it (AngularGrid::kAvailableSizes, parse_input.cpp) instead of copying it.
# No io header names a grid type.
target_link_libraries(qcx-io
  PUBLIC qcx-core qcx-integrals eigen
  PRIVATE qcx-molecule qcx-basisset qcx-grid tomlplusplus json)

# The bundled basis directory is baked into the library (the same macro the
# driver and the basisset tests define for themselves); validate_input.cpp
# resolves basis names against it - callers never name the path.
target_compile_definitions(qcx-io PRIVATE
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")

# PCH is mandatory on every target.
target_precompile_headers(qcx-io PRIVATE
    <toml++/toml.hpp>
    <nlohmann/json.hpp>
    <qcx/io/run_input.hpp>
    <qcx/io/molden_export.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-io-tests
        tests/parse_input_test.cpp
        tests/validate_input_test.cpp
        tests/result_json_test.cpp
        tests/approximation_notice_test.cpp
        tests/exchange_error_test.cpp
        tests/molden_export_test.cpp
        tests/selection_coefficients_test.cpp)
    target_link_libraries(qcx-io-tests PRIVATE qcx-io qcx-molecule qcx-basisset gtest gtest_main)

    qcx_apply_asan_runtime(qcx-io-tests)
    qcx_apply_blas_runtime(qcx-io-tests)
    qcx_apply_fast_test_mode(qcx-io-tests)

    target_precompile_headers(qcx-io-tests PRIVATE
        <gtest/gtest.h>
        <qcx/io/parse_input.hpp>
        <qcx/io/result_json.hpp>
        <qcx/io/molden_export.hpp>)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # Same MKL DLL/threading wiring as qcx-scf-tests.
        gtest_discover_tests(qcx-io-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-io-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
