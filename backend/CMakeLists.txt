# macOS ships neither an OpenMP runtime nor <omp.h>, and Homebrew's libomp
# lands outside every path the compiler searches by default. The CI legs
# install it and name the RUNTIME to CMake (OpenMP_CXX_FLAGS,
# OpenMP_CXX_LIB_NAMES, OpenMP_omp_LIBRARY) but never the HEADER directory -
# and that half cannot be left to FindOpenMP here. The module skips its entire
# search when those flags arrive pre-set (its own
# `if(NOT DEFINED OpenMP_<LANG>_FLAGS ...)` guard at the top-level loop), so
# OpenMP_CXX_INCLUDE_DIR is never discovered, OpenMP::OpenMP_CXX is left with
# no INTERFACE_INCLUDE_DIRECTORIES, find_package(OpenMP REQUIRED) reports
# SUCCESS, and the build dies much later on `#include <omp.h>` - which is
# exactly what the 2026-09-18 macOS legs show: the failing compile line carries
# every other module's include directory and not OpenMP's. The control flow
# above is read from CMake 4.4.2's FindOpenMP.cmake (the flags guard at
# lines 640-657, and INCLUDE_DIR -> INCLUDE_DIRS -> the imported target at
# lines 752-773).
#
# Resolving the header directory HERE closes the gap on the CMake side, so it
# does not depend on the leg passing one more flag. An explicitly supplied
# OpenMP_CXX_INCLUDE_DIR - the module's documented input - is honoured and
# this block stands aside.
if (APPLE AND NOT OpenMP_CXX_INCLUDE_DIR)
    find_program(_qcxBrewExecutable brew)
    if (_qcxBrewExecutable)
        execute_process(
            COMMAND "${_qcxBrewExecutable}" --prefix libomp
            OUTPUT_VARIABLE _qcxLibompPrefix
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _qcxLibompResult)
        if (_qcxLibompResult EQUAL 0 AND EXISTS "${_qcxLibompPrefix}/include/omp.h")
            set(OpenMP_CXX_INCLUDE_DIR "${_qcxLibompPrefix}/include")
            message(STATUS "qcx: OpenMP headers taken from Homebrew libomp at ${_qcxLibompPrefix}/include")
        endif()
    endif()
endif()

find_package(OpenMP REQUIRED)

# The CPU topology probe has per-OS backends behind one interface.
if (WIN32)
    set(_qcxCpuTopologySource src/cpu_topology_windows.cpp)
else()
    set(_qcxCpuTopologySource src/cpu_topology_posix.cpp)
endif()

# The host-RAM probe follows the same per-OS split; the GPU
# probe's real implementation lives in the CUDA-gated TU of
# qcx-backend-cuda, so these TUs carry the no-CUDA fallback under
# #ifndef QcxHasCuda (defined on qcx-backend in CUDA builds below).
if (WIN32)
    set(_qcxMemoryTopologySource src/memory_topology_windows.cpp)
else()
    set(_qcxMemoryTopologySource src/memory_topology_posix.cpp)
endif()

add_library(qcx-backend
  src/cpu_backend.cpp
  src/gpu_compute_profile.cpp
  src/host_compute_profile.cpp
  src/host_compute_profile_simd.cpp
  src/topology_profile.cpp
  ${_qcxCpuTopologySource}
  ${_qcxMemoryTopologySource}
)
target_include_directories(qcx-backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-backend PUBLIC cxx_std_23)
target_link_libraries(qcx-backend PUBLIC qcx-core OpenMP::OpenMP_CXX stdexec)
target_precompile_headers(qcx-backend PRIVATE
    <qcx/backend/cpu_backend.hpp>)

# The host compute-profile probe's AVX2 lane carries the /arch:AVX2
# flag on its own TU only, and that TU skips the module PCH - the same pair of
# source properties the integrals module's SIMD TUs carry (clang rejects a PCH
# built without the target feature the TU compiles with). The probe's scalar
# reference lane stays in the portable TU precisely so the flag cannot reach
# it: an auto-vectorized "scalar" lane would let the compiler's vector width
# into the measurement.
if (QCX_ARCH_X86_64)
    if (MSVC)
        set_source_files_properties(src/host_compute_profile_simd.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(src/host_compute_profile_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
    set_source_files_properties(src/host_compute_profile_simd.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
else()
    # No AVX2 target feature exists here to set, and the TU compiles none of
    # its intrinsics (its guards read the architecture test from
    # qcx/backend/cpu_features.hpp): MeasureFmaLaneAvx2() returns the scalar
    # lane's reading instead. The PCH skip is not needed either - it exists so
    # clang does not reject a PCH built without the target feature this TU
    # compiles with.
    message(STATUS "qcx-backend: non-x86-64 target - the host compute-profile probe has no AVX2 lane")
endif()

if (QCX_ENABLE_CUDA)
    # QcxHasCuda marks the CUDA-capable TUs and their consumers (the same
    # PUBLIC compile-definition pattern the integrals module uses): the
    # memory-topology fallback TUs of qcx-backend compile out their
    # DetectGpuMemory fallback, so only the real implementation in
    # cuda_backend.cpp defines the symbol.
    target_compile_definitions(qcx-backend PUBLIC QcxHasCuda=1)

    add_library(qcx-backend-cuda
      src/cuda_backend.cpp
      src/cuda_backend.cu
      src/gpu_compute_profile.cu
    )
    target_include_directories(qcx-backend-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(qcx-backend-cuda PUBLIC qcx-backend)
    target_compile_definitions(qcx-backend-cuda PUBLIC QcxHasCuda=1)
    # CUDA-safe list only (C++20): qcx C++23 headers would poison the nvcc TUs.
    target_precompile_headers(qcx-backend-cuda PRIVATE
        <cuda_runtime.h>)
endif()

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-backend-tests
      tests/backend_test.cpp
      tests/cpu_topology_test.cpp
      tests/gpu_compute_profile_test.cpp
      tests/host_compute_profile_test.cpp
      tests/memory_topology_test.cpp
      tests/topology_profile_test.cpp)
    target_link_libraries(qcx-backend-tests PRIVATE qcx-backend gtest gtest_main)
    target_precompile_headers(qcx-backend-tests PRIVATE
        <gtest/gtest.h>
        <qcx/backend/cpu_backend.hpp>)

    qcx_apply_asan_runtime(qcx-backend-tests)

    include(GoogleTest)
    gtest_discover_tests(qcx-backend-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})

    # DefaultOmpTeamSize's env-honoring first call must be the first thing in
    # its process (the function-local cache warms on first call), so the
    # OMP_NUM_THREADS test gets its own binary instead of sharing
    # qcx-backend-tests' already-warm cache.
    add_executable(qcx-backend-team-size-tests tests/team_size_test.cpp)
    target_link_libraries(qcx-backend-team-size-tests PRIVATE qcx-backend gtest gtest_main)
    target_precompile_headers(qcx-backend-team-size-tests PRIVATE
        <gtest/gtest.h>
        <qcx/backend/cpu_backend.hpp>)

    qcx_apply_asan_runtime(qcx-backend-team-size-tests)

    gtest_discover_tests(qcx-backend-team-size-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})

    if (QCX_ENABLE_CUDA)
        # Host-side C++23 test (kernels live in qcx-backend-cuda's .cu); needs
        # the CUDA toolkit headers and cudart, which CUDA targets get implicitly.
        add_executable(qcx-backend-cuda-tests
          tests/cuda_smoke_test.cpp
          tests/gpu_device_profile_test.cpp
          tests/gpu_memory_test.cpp
          tests/gpu_topology_test.cpp)
        target_include_directories(qcx-backend-cuda-tests PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIR})
        target_link_libraries(qcx-backend-cuda-tests PRIVATE qcx-backend-cuda CUDA::cudart_static gtest gtest_main)
        target_precompile_headers(qcx-backend-cuda-tests PRIVATE
            <gtest/gtest.h>
            <cuda_runtime.h>)
        qcx_apply_asan_runtime(qcx-backend-cuda-tests)
        gtest_discover_tests(qcx-backend-cuda-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
