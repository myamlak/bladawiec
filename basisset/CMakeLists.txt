add_library(qcx-basisset
  src/basis_set.cpp
)
target_include_directories(qcx-basisset PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-basisset PUBLIC cxx_std_23)
target_link_libraries(qcx-basisset PUBLIC qcx-molecule)

# PCH is mandatory on every first-party target.
target_precompile_headers(qcx-basisset PRIVATE
    <qcx/basisset/basis_set.hpp>
    <qcx/molecule/elements.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-basisset-tests tests/basis_set_test.cpp)
    target_link_libraries(qcx-basisset-tests PRIVATE qcx-basisset gtest gtest_main)
    # The corpus walk reads the vendored data directly from the source tree.
    target_compile_definitions(qcx-basisset-tests PRIVATE
        QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")

    qcx_apply_asan_runtime(qcx-basisset-tests)

    target_precompile_headers(qcx-basisset-tests PRIVATE
        <gtest/gtest.h>
        <qcx/basisset/basis_set.hpp>)

    include(GoogleTest)
    gtest_discover_tests(qcx-basisset-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})

    # The vendored corpus must stay byte-identical to its committed manifest
    # (data/basis/manifest.json); regeneration is a network operation.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_FOUND)
        add_test(NAME basis-corpus-integrity
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/fetch_basis_data.py --check
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
endif()
