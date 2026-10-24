add_library(qcx-basisset
  src/basis_set.cpp
)
target_include_directories(qcx-basisset PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-basisset PUBLIC cxx_std_23)
target_link_libraries(qcx-basisset PUBLIC qcx-molecule)

# PCH is mandatory on every target.
target_precompile_headers(qcx-basisset PRIVATE
    <qcx/basisset/basis_set.hpp>
    <qcx/molecule/elements.hpp>)

# Apple's libc++ provides no usable floating-point std::from_chars: the
# integer overloads are deleted for floating types (Xcode 16.4). The parser
# probes its own call instead of reading a feature-test macro, because a
# library that declares the floating overloads but deletes them still ships
# <charconv>, so only the call itself answers the question. Where it does not
# compile, basis_set.cpp parses with std::strtod under the same strictness.
# -DQCX_HAS_FLOATING_FROM_CHARS=ON|OFF overrides the probe; OFF is how the
# fallback is exercised on a host whose standard library has the function.
include(CheckCXXSourceCompiles)

if (NOT DEFINED QCX_HAS_FLOATING_FROM_CHARS)
    check_cxx_source_compiles("
#include <charconv>
#include <system_error>
int main()
{
    double value = 0.0;
    const auto result = std::from_chars(\"0.5\", \"0.5\" + 3, value);
    return result.ec == std::errc{} ? 0 : 1;
}
" _qcxBasissetFloatingFromChars)

    set(QCX_HAS_FLOATING_FROM_CHARS ${_qcxBasissetFloatingFromChars} CACHE INTERNAL
        "True when the standard library provides a usable floating-point std::from_chars")
endif()

if (QCX_HAS_FLOATING_FROM_CHARS)
    target_compile_definitions(qcx-basisset PRIVATE QCX_HAS_FLOATING_FROM_CHARS=1)
else()
    target_compile_definitions(qcx-basisset PRIVATE QCX_HAS_FLOATING_FROM_CHARS=0)
endif()

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
