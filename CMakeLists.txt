cmake_minimum_required(VERSION 3.27)
project(qcx VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(QCX_BUILD_TESTS "Build tests" ON)
option(QCX_ASAN "Enable AddressSanitizer (MSVC only; Clang/GCC builds use their own flags)" OFF)

# Added as a compile option rather than via CMAKE_CXX_FLAGS so CMake's default
# MSVC flags (/EHsc, /W3, ...) survive alongside the sanitizer flag.
if (QCX_ASAN)
    add_compile_options(/fsanitize=address)
    # ASAN needs debug-info emission: MSVC warns C5072 ("ASAN enabled
    # without debug information emission"), which /WX in third-party
    # targets (gtest, benchmark) promotes to C2220. /Zi at compile and
    # /DEBUG at link is the warning's own prescription. /FS above keeps
    # the PDB access serialized for the parallel build.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zi>)
    add_link_options(/DEBUG)
endif()

if (MSVC)
    # The vcpkg toolchain enables /MP for parallel builds; /FS serializes PDB
    # access so multiple cl.exe in one project cannot collide (C1041).
    # Scoped to CXX: nvcc chokes on a bare /FS in CUDA compilation, and
    # MSBuild passes its own /FS to the CUDA host compiler anyway.
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/FS>)
endif()

# Backend options. CUDA is a local-only feature: CI never
# enables it, so its builds stay CPU-only and need no GPU toolchain.
option(QCX_ENABLE_CUDA "Build the CUDA backend (requires the CUDA toolkit)" OFF)

if (QCX_ENABLE_CUDA)
    enable_language(CUDA)
    # CUDAToolkit provides the CUDA::cudart* imported targets for host-only
    # C++ consumers (e.g. the memory CUDA tests compiled by cl.exe).
    find_package(CUDAToolkit REQUIRED)
    # Device code tops out at C++20 in nvcc 13.x even though the project is C++23.
    set(CMAKE_CUDA_STANDARD 20)
    # Auto-detect the installed GPU, but honor an explicit -D override: CMake's
    # "native" detection needs NVML visibility and fails in non-interactive
    # sessions (disconnected RDP, agents) even when the GPU is present.
    if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)  # auto-detect the installed GPU
    endif()
endif()

# Dependency resolution. HDF5 is a compiled
# dependency, so it sits behind an option: the CI Windows job configures
# without it (no vcpkg support there, and hdf5 in CI would burn 2x-billed
# minutes). Benchmarks are local-only.
option(QCX_ENABLE_IO "Enable HDF5 I/O via HighFive" ON)
option(QCX_BUILD_BENCHMARKS "Build benchmarks (local-only; CI never enables it)" OFF)

# The architecture probe, the same shape the vendored boys tree uses for the
# same question. This tree's vector tiers are x86-64 only: the AVX2 micro-GEMMs,
# the CPUID feature probe, and the AVX2 measurement lane of the host
# compute-profile probe. The probe is asked ONCE here, and the modules that own
# those translation units use the answer to decide which per-file target
# features are legal to set. The sources answer the same question from the
# compiler's own architecture macros (QcxArchX86_64, defined by
# qcx/backend/cpu_features.hpp), so a TU reached with different flags still
# compiles the same code.
#
# -DQCX_ARCH_X86_64=ON|OFF overrides the probe. That exists to compile the
# non-x86-64 branches on an x86-64 host, which is how they are checked without
# an arm64 machine, and it is the only reason to set it.
include(CheckCXXSourceCompiles)

if (NOT DEFINED QCX_ARCH_X86_64)
    check_cxx_source_compiles("
#if !defined(__x86_64__) && !defined(_M_X64)
#  error not an x86-64 target
#endif
int main() { return 0; }
" _qcxArchX86_64Probe)

    if (_qcxArchX86_64Probe)
        set(_qcxArchX86_64 ON)
    else()
        set(_qcxArchX86_64 OFF)
    endif()

    set(QCX_ARCH_X86_64 ${_qcxArchX86_64} CACHE INTERNAL
        "True when the target architecture is x86-64 (probed from the compiler)")
endif()

# Floating-point contraction is pinned OFF on every target that is not x86-64.
#
# FMA is baseline on arm64, so the compiler is free to fuse `a * b + c` into an
# FMA in every scalar translation unit. On x86-64 the baseline has no FMA, so
# the same source cannot fuse, and the two architectures compute different
# numbers from identical code. The per-file pins in the module CMakeLists
# cover the SIMD translation units only, whose claim is bit-identity with
# their scalar twin; the scalar units carry no flag because on x86-64 there is
# nothing there to fuse. That reasoning is x86-specific and does not hold on
# the targets this block covers, so here the guarantee is stated once for the
# tree instead of per file.
#
# Scoped rather than unconditional: on x86-64 the flag would be inert for the
# scalar units and would contradict the deliberate per-file pins on the SIMD
# ones, hiding the regression those pins exist to catch.
#
# MSVC needs no counterpart: /fp:precise is the default and does not contract.
if (NOT QCX_ARCH_X86_64 AND NOT MSVC)
    add_compile_options(-ffp-contract=off)
endif()

# MSVC's ASan runtime is a DLL that must sit next to test executables when
# running outside a Developer prompt (plain ctest). Applies only when ASan
# is actually enabled in this build.
function(qcx_apply_asan_runtime target)
    if (QCX_ASAN AND MSVC)
        get_filename_component(_clDir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        set(_asanDll "${_clDir}/clang_rt.asan_dynamic-x86_64.dll")
        if (EXISTS "${_asanDll}")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_asanDll}" "$<TARGET_FILE_DIR:${target}>")
        endif()
    endif()
endfunction()

# Debug builds of the heavy numerical test suites run a fast smoke
# subset only (QcxTestFastOnly - the fixture helper IsFastOnlyMode() gates
# the heavy tests with GTEST_SKIP); Release runs the full suite. A
# generator expression because the local preset is multi-config MSBuild
# (CMAKE_BUILD_TYPE is empty there).
#
# THE GATE IS CONFIG-BASED, NOT PLATFORM-BASED. This comment said otherwise
# until 2026-09-16: it read "Linux CI Debug is ~2x Release, not 10-50x like
# MSVC Debug, and stays on the full suite". The Linux leg does NOT stay on
# the full suite - `$<CONFIG:Debug>` is true there too, so EVERY Debug
# configuration takes the subset. The cost asymmetry that sentence recorded
# is real, and it is why the mistake was easy to make; what it was not is a
# mechanism. Making Linux Debug run the full suite is a change to this
# expression, not to this comment.
function(qcx_apply_fast_test_mode target)
    target_compile_definitions(${target} PRIVATE
        $<$<CONFIG:Debug>:QcxTestFastOnly=1>)
endfunction()

# ===========================================================================
# THE LARGE TEST CATEGORY (owner ruling 2026-09-18)
# ===========================================================================
#
# Long tests are MOVED out of the routine selection, never shortened. They are
# performance/scale tests, not unit tests, so they get their own home, their
# own registration gate, and their own runs.
#
# WHY THIS REPLACED THE PER-TEST CONVENTION. Until 2026-09-18 the only thing
# keeping a long test out of a routine run was a per-test call to
# qcx::testing::IsFastOnlyMode() inside the test body. That is an
# opt-in made in the TEST SOURCE, so a test that forgets the call runs full
# strength and nothing says so: integrals/tests/ri_orbit_test.cpp carried
# zero such calls, and one of its cases turned a 73 s Release test into a
# 47-minute Debug test that dominated a whole leg (measured 2026-09-18).
#
# MEMBERSHIP IS NOW BY TARGET AND BY DIRECTORY, and that is the fail-safe:
# a long test's source lives in <module>/tests/large/ and is compiled into
# qcx-<module>-large-tests, which this option gates. A routine `ctest`
# cannot see a test that was never registered, so forgetting a flag cannot
# pull one back in. The gate defaults OFF, the same shape as
# QCX_REFERENCE_REGENERATION_CHECK.
#
# The `large` label is the single exclusion mechanism. Tests registered from
# a large target carry it structurally; cases that must stay in a normal
# target (a file another change is editing, a file whose long and short
# cases cannot be separated) are named in an explicit per-module manifest
# through qcx_mark_large_cases() below. The routine selection excludes both with
# `ctest -LE large`. The two QFMM chain pins' former `heavy` label folded
# into this one on 2026-09-18: one exclusion mechanism, not two.
#
# RUN IT: configure with -DQCX_ENABLE_LARGE_TESTS=ON, then
#   ctest --test-dir <build> -C Release -L large -j <N>
# CI runs it in the release lane only (.github/workflows/ci-windows-release.yml);
# never Debug, never on the per-push lane.
option(QCX_ENABLE_LARGE_TESTS
    "Register the long-running test targets qcx-<module>-large-tests (owner ruling 2026-09-18)"
    OFF)

# qcx_add_large_tests(<module> <source>... [LINK <lib>...] [INCLUDES <dir>...]
#                     [DEFINES <def>...] [PCH <header>...])
#
# Creates qcx-<module>-large-tests, or creates NOTHING when the gate is OFF -
# that absence is the fail-safe, so the early return is the load-bearing line
# and comes first.
#
# No qcx_apply_fast_test_mode() here, deliberately: the whole point of the
# category is that it runs full strength. Large tests are excluded from the
# routine selection by the gate and the label, never by skipping themselves.
function(qcx_add_large_tests module)
    if (NOT QCX_ENABLE_LARGE_TESTS OR NOT QCX_BUILD_TESTS OR NOT TARGET gtest)
        return()
    endif()

    cmake_parse_arguments(ARG "" "" "LINK;INCLUDES;DEFINES;PCH" ${ARGN})
    if (NOT ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "qcx_add_large_tests(${module}): no sources given")
    endif()

    add_executable(qcx-${module}-large-tests ${ARG_UNPARSED_ARGUMENTS})
    target_compile_features(qcx-${module}-large-tests PRIVATE cxx_std_23)
    target_link_libraries(qcx-${module}-large-tests PRIVATE
        qcx-${module} ${ARG_LINK} gtest gtest_main qcx-test-fixtures)

    if (ARG_INCLUDES)
        target_include_directories(qcx-${module}-large-tests PRIVATE ${ARG_INCLUDES})
    endif()
    if (ARG_DEFINES)
        target_compile_definitions(qcx-${module}-large-tests PRIVATE ${ARG_DEFINES})
    endif()

    # PCH is mandatory on every first-party target. <gtest/gtest.h> is
    # the floor; a caller adds its own the same way its normal target does.
    target_precompile_headers(qcx-${module}-large-tests PRIVATE
        <gtest/gtest.h> ${ARG_PCH})

    # The same runtime deployment every sibling test target applies (the
    # MKL runtime DLLs, the ASan DLL); a large test that dies at load would read
    # as a real defect.
    qcx_apply_asan_runtime(qcx-${module}-large-tests)
    qcx_apply_blas_runtime(qcx-${module}-large-tests)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxLargeMkl)
    if (_qcxLargeMkl)
        gtest_discover_tests(qcx-${module}-large-tests
            PROPERTIES LABELS large
                       ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
                       TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-${module}-large-tests
            PROPERTIES LABELS large TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endfunction()

# qcx_mark_large_cases(<module> <case>...)
#
# Applies LABELS large to cases that stay registered by a NORMAL test target.
# This is the escape hatch for a long test whose source file cannot leave the
# normal target in the change that introduces the category, so it is always
# the second-best answer: prefer moving the source into <module>/tests/large/.
# Every case named here must be reached by the routine `ctest -LE large`.
#
# The label cannot be attached at configure time: gtest_discover_tests
# registers the cases from the built binary, so on a PRE_TEST target the names
# do not exist yet and set_tests_properties is a hard configure error
# ("Can not find test to add properties to to" - measured, CMake 4.4.2, by
# the chain-pin lane). TEST_INCLUDE_FILES is the hook that runs at ctest time,
# after the discovery include, which is when the names exist. APPEND, not set:
# the generator keeps its own gtest include in the same property.
#
# A STALE NAME MUST BE LOUD HERE. It is not enough to hand the names to
# set_tests_properties and let a bad one fall out: measured on this tree
# (CMake 4.4.2, ctest script context, 2026-09-18) set_tests_properties on a
# name NO test registers is a SILENT NO-OP - it neither errors nor warns. So
# the manifest would quietly stop excluding a renamed long test, which is
# precisely the fail-open the owner's ruling forbids, and it is why this
# assertion exists rather than being an optional extra.
#
# HOW THE CHECK IS MADE, because the obvious form does not work: the
# `if (NOT TEST "<name>")` operator was tried first and MISFIRES here - it
# reported every name missing while the tests were in fact registered (proved
# by deleting the assertion and watching set_tests_properties apply the label
# cleanly). What does work is the discovered list the GoogleTest module itself
# writes: each test target's discovery file defines `<target>_TESTS`, the exact
# names it registered (POST_TEST in its <hash>_<config>_tests.cmake, PRE_TEST in
# its discover_tests(TEST_LIST ...) call). Checking membership in that list is
# deterministic and needs no operator.
#
# The target name is the module convention qcx-<module>-tests, so the list
# variable is qcx-<module>-tests_TESTS. If a module ever registers its tests
# under another target name this assertion says so by name rather than
# silently passing on an empty list.
function(qcx_mark_large_cases module)
    if (NOT ARGN)
        return()
    endif()

    set(_qcxTarget "qcx-${module}-tests")
    set(_qcxLargeManifest "${CMAKE_CURRENT_BINARY_DIR}/qcx-large-cases-${module}.cmake")
    set(_qcxLargeBody "# Generated by ${module}/CMakeLists.txt - edit that file, not this one.\n")
    string(APPEND _qcxLargeBody
        "# Included by CTestTestfile.cmake after the gtest discovery include.\n")
    # CMP0057 IS SET BY THIS FILE, and without it the check below is not merely
    # weak - it is UNPARSEABLE. The manifest runs in ctest's SCRIPT context
    # (CTestTestfile.cmake includes it), where cmake_minimum_required() was never
    # called, so the policy is unset: IN_LIST is then not an operator, the
    # `if (NOT <case> IN_LIST <list>)` below fails with "Unknown arguments
    # specified", and ctest exits 8 BEFORE RUNNING A SINGLE TEST. Observed in CI
    # on 2026-09-18 on every ctest leg, under the runners' CMake 3.x. A local
    # CMake 4.x tree neither shows it nor can: 4.x refuses CMP0057 OLD outright
    # ("may not be set to OLD behavior ... NEW behavior is now required"), so
    # IN_LIST is an operator there whatever this file says. The policy cannot be
    # set from the root CMakeLists instead: that file's policies do not carry
    # into the script context this one runs in.
    string(APPEND _qcxLargeBody
        "# CMP0057 is set here because this file runs in ctest's script context,\n"
        "# where cmake_minimum_required() was never called and the policy would\n"
        "# otherwise be unset - with OLD behaviour IN_LIST is not an operator and\n"
        "# the membership check below cannot be parsed at all.\n"
        "cmake_policy(SET CMP0057 NEW)\n")
    string(APPEND _qcxLargeBody "set(_qcxLargeCases\n")
    foreach(_qcxCase IN LISTS ARGN)
        string(APPEND _qcxLargeBody "    \"${_qcxCase}\"\n")
    endforeach()
    string(APPEND _qcxLargeBody ")\n")
    # Guard the guard: an undefined list variable would make every case below
    # read as missing, which is loud but wrong about why. Name the real cause.
    string(APPEND _qcxLargeBody
        "if (NOT DEFINED ${_qcxTarget}_TESTS)\n"
        "    message(FATAL_ERROR\n"
        "        \"qcx: '${_qcxTarget}_TESTS' is not defined, so ${module}'s large-case \"\n"
        "        \"manifest cannot be checked. That list is written by \"\n"
        "        \"gtest_discover_tests for the target '${_qcxTarget}'; if this module's \"\n"
        "        \"test target was renamed, update qcx_mark_large_cases in \"\n"
        "        \"${module}/CMakeLists.txt to match. Until then the large category is \"\n"
        "        \"not being excluded correctly.\")\n"
        "endif()\n")
    string(APPEND _qcxLargeBody
        "foreach (_qcxCase IN LISTS _qcxLargeCases)\n"
        "    if (NOT _qcxCase IN_LIST ${_qcxTarget}_TESTS)\n"
        "        message(FATAL_ERROR\n"
        "            \"qcx: ${module}'s large-case manifest names '\${_qcxCase}', but \"\n"
        "            \"'${_qcxTarget}' registers no such test. TWO CAUSES, most likely \"\n"
        "            \"first: (1) this tree has not been BUILT since the manifest changed - \"\n"
        "            \"gtest discovery runs at build time, so a new TEST_INCLUDE_FILES \"\n"
        "            \"entry is not visible to ctest until 'cmake --build <build dir> \"\n"
        "            \"--target ${_qcxTarget}' has run once; (2) the case was renamed or \"\n"
        "            \"removed - then fix the manifest in ${module}/CMakeLists.txt. Do NOT \"\n"
        "            \"silence this, and do NOT 'fix' it by deleting the name: while it \"\n"
        "            \"stands, the large category is not being excluded correctly and the \"\n"
        "            \"long test it names is back in the routine selection.\")\n"
        "    endif()\n"
        "endforeach()\n"
        "set_tests_properties(\${_qcxLargeCases} PROPERTIES LABELS large)\n")
    file(WRITE "${_qcxLargeManifest}" "${_qcxLargeBody}")
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_qcxLargeManifest}")
endfunction()

# The job-object memory caps (process_caps.cpp / the workspace budget) are
# process-global: a test that caps its own process makes every OTHER test
# running beside it fail for a reason that has nothing to do with it. Under
# the `ctest -j` the large category now uses, that is a flake source, and a
# flake reads as a real defect. RUN_SERIAL - not RESOURCE_LOCK - because a
# lock only serializes the cap holders against each other while every other
# test keeps running alongside them.
#
# A helper rather than a bare set_tests_properties for the same reason as
# above: the cases must be named after discovery, and a case given here that
# no longer exists must fail loudly instead of dropping the serialization.
function(qcx_mark_serial_cases module)
    if (NOT ARGN)
        return()
    endif()

    set(_qcxTarget "qcx-${module}-tests")
    set(_qcxSerialManifest "${CMAKE_CURRENT_BINARY_DIR}/qcx-serial-cases-${module}.cmake")
    set(_qcxSerialBody "# Generated by ${module}/CMakeLists.txt - edit that file, not this one.\n")
    # Same CMP0057 reason as qcx_mark_large_cases above: the membership check
    # below reads IN_LIST, and in ctest's script context the policy is unset,
    # so the check cannot be parsed at all.
    string(APPEND _qcxSerialBody
        "# CMP0057 for the IN_LIST check below - see qcx_mark_serial_cases in the\n"
        "# root CMakeLists.txt for why it must be set in this file.\n"
        "cmake_policy(SET CMP0057 NEW)\n")
    string(APPEND _qcxSerialBody "set(_qcxSerialCases\n")
    foreach(_qcxCase IN LISTS ARGN)
        string(APPEND _qcxSerialBody "    \"${_qcxCase}\"\n")
    endforeach()
    string(APPEND _qcxSerialBody ")\n")
    string(APPEND _qcxSerialBody
        "if (NOT DEFINED ${_qcxTarget}_TESTS)\n"
        "    message(FATAL_ERROR\n"
        "        \"qcx: '${_qcxTarget}_TESTS' is not defined, so ${module}'s serial-case \"\n"
        "        \"list cannot be checked - see qcx_mark_serial_cases in the root \"\n"
        "        \"CMakeLists.txt.\")\n"
        "endif()\n")
    string(APPEND _qcxSerialBody
        "foreach (_qcxCase IN LISTS _qcxSerialCases)\n"
        "    if (NOT _qcxCase IN_LIST ${_qcxTarget}_TESTS)\n"
        "        message(FATAL_ERROR\n"
        "            \"qcx: ${module}'s serial-case list names '\${_qcxCase}', which \"\n"
        "            \"'${_qcxTarget}' does not register - a renamed cap test would run \"\n"
        "            \"concurrently with the rest of the suite, which is what RUN_SERIAL \"\n"
        "            \"exists to prevent. Most likely this tree has not been built since \"\n"
        "            \"the list changed (gtest discovery runs at build time); otherwise \"\n"
        "            \"the case was renamed - fix the list in ${module}/CMakeLists.txt.\")\n"
        "    endif()\n"
        "endforeach()\n"
        "set_tests_properties(\${_qcxSerialCases} PROPERTIES RUN_SERIAL TRUE)\n")
    file(WRITE "${_qcxSerialManifest}" "${_qcxSerialBody}")
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES "${_qcxSerialManifest}")
endfunction()

# Deploys the MKL runtime DLLs next to a test executable when the MKL vendor
# is the configured BLAS: MKL loads its runtime (mkl_core, the CPU-dispatch
# libs) dynamically, which vcpkg's app-local deployment does not
# cover. Shared by every test target that links qcx-linalg.
function(qcx_apply_blas_runtime target)
    get_property(_qcxMkl GLOBAL PROPERTY qcxVendorBlasIsMkl)
    get_property(_qcxMklRoot GLOBAL PROPERTY qcxMklRoot)
    if (_qcxMkl AND _qcxMklRoot)
        # Only the runtime set MKL actually loads is deployed (the full glob
        # copied 41 DLLs incl. never-loaded sycl/blacs/scalapack/cdft/tbb
        # variants, ~1 GB per target dir): dumpbin /dependents on every qcx
        # binary shows the single import mkl_sequential.2.dll (linalg configures
        # MKL_THREADING sequential), which imports mkl_core; mkl_core loads
        # exactly one CPU-dispatch lib at runtime (CPUID), so all four must
        # ride along.
        file(GLOB _qcxMklDlls
            "${_qcxMklRoot}/bin/mkl_core*.dll"
            "${_qcxMklRoot}/bin/mkl_sequential*.dll"
            "${_qcxMklRoot}/bin/mkl_def*.dll"
            "${_qcxMklRoot}/bin/mkl_mc3*.dll"
            "${_qcxMklRoot}/bin/mkl_avx2*.dll"
            "${_qcxMklRoot}/bin/mkl_avx512*.dll")
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_qcxMklDlls}
                    "$<TARGET_FILE_DIR:${target}>")
    endif()
endfunction()

# Whether the MKL vendor was selected as the BLAS backend (set by linalg
# during vendor selection; consumed by test targets that link it).
function(qcx_blas_is_mkl outVar)
    get_property(_qcxMkl GLOBAL PROPERTY qcxVendorBlasIsMkl)
    set(${outVar} ${_qcxMkl} PARENT_SCOPE)
endfunction()

enable_testing()

# The ctest timeout every discovered test carries (applied at the
# gtest_discover_tests call sites below). ctest's own default is unbounded, so
# a hung or spinning test blocks the rest of the leg until someone kills it by
# hand: the 2026-09-18 Clang Debug leg lost 30 minutes to exactly that
# (RiOrbitExpansionTest.RefusesABasisWithAnOverWideShell, 1828.57 sec,
# "Subprocess killed") and had to be interrupted to release the remaining 813
# tests. A timeout makes the leg continue by itself.
#
# 2400 s is derived from the measured distribution, not picked round. ctest
# reports per-test time, and on the GCC Release limb (2026-09-18) the slowest
# test that PASSES is FourCellAttributionTest.ConvergedEnergiesAttributeThe
# Divergence at 1814.8 s, with the C12 chain pin next at 952.7 s - so the guard
# must sit above 1814.8 s or it turns a green test red. 2400 s clears that tail
# while still bounding both known runaways (the C24 chain pin at 5955.5 s, and
# any unbounded spin).
#
# Re-derive it as the slow tests are trimmed. This is a RUNAWAY GUARD, never a
# performance gate: it must stay above every test the suite is expected to keep.
set(QCX_TEST_TIMEOUT_SECONDS 2400 CACHE STRING
    "Per-test ctest timeout in seconds (a runaway guard, not a performance gate)")

# Third-party libraries: pinned git submodules under third_party/.

# Eigen is header-only - an INTERFACE target exposing the include path.
# SYSTEM like every other vendored lib: clang-tidy must not analyze Eigen's
# internals (its stack-memory-destructor pattern trips the analyzer).
add_library(eigen INTERFACE)
target_include_directories(eigen SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)

# stdexec is the header-only reference implementation of std::execution (P2300),
# used by the backend's scheduler prototype. SYSTEM so third-party warnings
# stay suppressed in our warning-as-error world.
add_library(stdexec INTERFACE)
target_include_directories(stdexec SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stdexec/include)
if (MSVC)
    # stdexec reads __cplusplus (pinned to 199711L without the first switch)
    # and requires the conforming preprocessor (the second). Scoped to CXX:
    # nvcc chokes when these flags leak into CUDA compilation.
    target_compile_options(stdexec INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>
        $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)
endif()

# Header-only pinned submodules: INTERFACE
# targets with SYSTEM includes so third-party warnings stay suppressed.

add_library(json INTERFACE)
target_include_directories(json SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include)

add_library(tomlplusplus INTERFACE)
target_include_directories(tomlplusplus SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tomlplusplus/include)

# Spectra is Eigen-native; the linalg seams are its only consumers.
add_library(spectra INTERFACE)
target_include_directories(spectra SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/spectra/include)
target_link_libraries(spectra INTERFACE eigen)

# amgcl runs in its boost-free mode: params become typed structs and
# no Boost headers are pulled into any lane - the CI Windows job stays on
# boost-graph alone (the property_tree-including path is what failed there).
add_library(amgcl INTERFACE)
target_include_directories(amgcl SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/amgcl)
target_compile_definitions(amgcl INTERFACE AMGCL_NO_BOOST)

if (QCX_ENABLE_IO)
    find_package(HDF5 COMPONENTS C REQUIRED)
    add_library(highfive INTERFACE)
    target_include_directories(highfive SYSTEM INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/highfive/include)

    if (TARGET HDF5::HDF5)
        target_link_libraries(highfive INTERFACE HDF5::HDF5)
    else()
        target_include_directories(highfive INTERFACE ${HDF5_INCLUDE_DIRS})
        target_link_libraries(highfive INTERFACE ${HDF5_LIBRARIES})
    endif()
endif()

# spdlog builds as a static library from source.
set(SPDLOG_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/spdlog)

# GoogleTest comes before the modules so their CMakeLists can reference the
# gtest targets. Missing submodule degrades to a warning, not an error.
if (QCX_BUILD_TESTS)
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
        # GoogleTest defaults to the static CRT (/MT); match our /MD on MSVC.
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        add_subdirectory(third_party/googletest)
    else()
        message(WARNING "QCX_BUILD_TESTS is ON but third_party/googletest is missing - run: git submodule update --init")
    endif()
endif()

# Google Benchmark is a vendored library, so it joins the trees above rather
# than sitting next to the benchmarks/ targets that consume it.
if (QCX_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/benchmark)
endif()

# excgrid: the standalone grid + XC-kernel + DFT-D3 library. In-tree
# excgrid/ is the dev home until the
# user-go cut; external/excgrid is the pinned gitlink afterwards; the explicit
# binary dir keeps artifacts in place across that flip. excgrid is not a
# module-DAG entry (tools/check_dag.py reads the module CMakeLists only) - the
# boys precedent. qcx-grid's XC engine links it, so the subproject is added
# here, above the module list.
#
# Tests are forced OFF: excgrid's suite fetches googletest from the network and
# would define gtest/gtest_main a second time next to third_party/googletest.
#
# CUT COMMIT: the in-tree excgrid/ directory must be DELETED in the same commit
# that adds external/excgrid, because the branch order below prefers the in-tree
# path - a leftover directory would silently shadow the pinned gitlink.
set(EXCGRID_BUILD_TESTS OFF CACHE BOOL "" FORCE)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/excgrid/CMakeLists.txt")
    set(EXCGRID_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/excgrid")
elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/external/excgrid/CMakeLists.txt")
    set(EXCGRID_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/excgrid")
endif()
if (NOT EXCGRID_ROOT)
    # The tree is committed in-repo, so absence means a broken checkout. A
    # warning would only defer the failure to qcx-grid's link line, where CMake
    # reports a missing target instead of the fix.
    message(FATAL_ERROR
        "excgrid tree missing: neither excgrid/ nor external/excgrid/ has a "
        "CMakeLists.txt, and qcx-grid links excgrid::excgrid. Restore "
        "the in-tree tree or run: git submodule update --init external/excgrid")
endif()
add_subdirectory(${EXCGRID_ROOT} external/excgrid)

# THE VENDORED TREES END HERE: every add_subdirectory() below this line is
# first-party, and that ordering IS the mechanism of the block below.

# First-party warning level (2026-09-17). Every qcx target - the modules, their
# tests, the driver and the benchmarks - compiles at /W4. This is a MEASUREMENT
# INSTRUMENT, not a gate: there is deliberately no /WX, so a first-party warning
# can never fail a build.
#
# The option goes on the DIRECTORY, once, here, so every subdirectory added
# below inherits it - including a module appended to the module list later. A
# first-party target therefore cannot silently miss it. The other candidate
# shape, a per-target helper, can: it reaches only the targets whose CMakeLists
# remember to call it.
#
# Vendored trees never inherit it: they are all added above - the header-only
# interface targets at the top of this file, spdlog, googletest, plus Google
# Benchmark and excgrid, which moved up here on 2026-09-17. Those last two used
# to sit below the module list, where the level was first cancelled with a /W3
# around their add_subdirectory(). That cancel was measured and rejected: it
# does not stay local. Re-appending /W4 is deduplicated, so /W3 stayed last and
# every first-party target AFTER the call silently dropped to /W3, and the /W3
# itself overrode the level a vendored tree sets on its own targets (excgrid's
# own /W4 PRIVATE came back Level3). Ordering has nothing to cancel.
if (MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
endif()

add_subdirectory(core)
add_subdirectory(backend)
add_subdirectory(memory)
add_subdirectory(linalg)
add_subdirectory(molecule)
add_subdirectory(symmetry)
add_subdirectory(basisset)

add_subdirectory(grid)
add_subdirectory(integrals)
add_subdirectory(scf)
if (QCX_ENABLE_IO)
    add_subdirectory(storage)
endif()
add_subdirectory(io)
# Kept in `tools/check_dag.py` ORDER (io -> properties -> driver). CMake resolves
# a target named before the directory that defines it is added - it reads target
# names at generate time and orders the build from the target graph, so this list
# cannot break the build either way - but a sequence that disagrees with ORDER
# still reads as a claim about the DAG, and the claim would be false.
add_subdirectory(properties)
add_subdirectory(driver)

# tests/ also hosts the shared test fixtures; benchmarks consume those, so
# the directory is added when either consumer exists.
if ((QCX_BUILD_TESTS OR QCX_BUILD_BENCHMARKS) AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
    add_subdirectory(tests)
endif()

# Benchmarks: the first-party targets only. Google Benchmark itself is a
# vendored library and is added with the other third-party trees above, gated
# by the same option (ON in the windows-msvc preset, never in CI).
if (QCX_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# API documentation - optional: Doxygen must be on PATH (C:\Tools on this
# machine; CI installs it via apt). Build with
# `cmake --build build/windows-msvc --target qcx-docs`; output lands in
# build/doxygen/html. A plain custom target runs the checked-in Doxyfile
# directly - doxygen_add_docs would generate its own config and drop the
# Doxyfile settings (EXTRACT_ALL, README mainpage, treeview, ...).
# Doxygen 1.18.0 lives at C:\Tools (portable zip); background shells may
# lack it on PATH, so hint the finder explicitly.
if (WIN32)
    list(APPEND CMAKE_PROGRAM_PATH "C:/Tools")
endif()
find_package(Doxygen QUIET)
if (DOXYGEN_FOUND)
    add_custom_target(qcx-docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generate qcx API documentation (HTML)")
endif()
