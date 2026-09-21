# Boost is the one vcpkg/apt system-style dependency. CMake 4.x removed
# FindBoost (CMP0167): the CONFIG-mode package from vcpkg (local) or apt (CI)
# replaces it. Locally boost-graph is classic-installed via vcpkg; CI Linux
# jobs install libboost-graph-dev, the CI Windows job installs boost-graph
# with the vcpkg toolchain.
cmake_policy(SET CMP0167 NEW)
find_package(Boost CONFIG REQUIRED COMPONENTS graph)

add_library(qcx-molecule
  src/molecule.cpp
  src/mass_properties.cpp
  src/connectivity.cpp
)
target_include_directories(qcx-molecule PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-molecule PUBLIC cxx_std_23)
target_link_libraries(qcx-molecule PUBLIC qcx-core qcx-memory eigen Boost::graph)

# Eigen/Dense + Boost.Graph dominate this module's frontend time (trace-
# measured); one PCH per target (angle-bracket entries resolve via the
# target's include dirs, so vcpkg/apt Boost needs no repo-relative path).
target_precompile_headers(qcx-molecule PRIVATE
    <qcx/molecule/molecule.hpp>
    <qcx/molecule/elements.hpp>
    <qcx/molecule/harmonic.hpp>
    <Eigen/Dense>
    <boost/graph/adjacency_list.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-molecule-tests
      tests/elements_test.cpp
      tests/molecule_test.cpp
      tests/connectivity_test.cpp
      tests/mass_properties_test.cpp
      tests/large_molecule_test.cpp
      tests/harmonic_test.cpp
    )
    # C60/nanotube fixtures live in qcx-test-fixtures (tests/), shared with
    # the symmetry corpus; the target is defined later but plain-name links
    # resolve lazily.
    target_link_libraries(qcx-molecule-tests PRIVATE qcx-molecule qcx-test-fixtures gtest gtest_main)
    target_precompile_headers(qcx-molecule-tests PRIVATE
        <gtest/gtest.h>
        <qcx/molecule/molecule.hpp>
        <qcx/molecule/elements.hpp>
        <qcx/molecule/harmonic.hpp>
        <Eigen/Dense>
        <boost/graph/adjacency_list.hpp>)

    qcx_apply_asan_runtime(qcx-molecule-tests)

    include(GoogleTest)
    gtest_discover_tests(qcx-molecule-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})

    # The generated header must stay byte-identical to its committed source of
    # truth (tools/periodic/elements_data.json) - regeneration is offline.
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_FOUND)
        add_test(NAME elements-regeneration-deterministic
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/periodic/generate_element_table.py
                    --check ${CMAKE_CURRENT_SOURCE_DIR}/include/qcx/molecule/elements.hpp)
    endif()
endif()
