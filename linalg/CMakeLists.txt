add_library(qcx-linalg
  src/dense_ops.cpp
  src/iterative_eigensolver.cpp
  src/sparse_solver.cpp
  src/batched_ops.cpp
)
target_include_directories(qcx-linalg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-linalg PUBLIC cxx_std_23)
target_link_libraries(qcx-linalg PUBLIC qcx-core eigen)
# The Spectra/amgcl seams are implementation-only consumers: the public
# headers expose qcx/Eigen types only, so the vendored libs stay PRIVATE.
target_link_libraries(qcx-linalg PRIVATE spectra amgcl)

# Eigen/Dense is the dominant frontend cost in this module (trace-measured).
target_precompile_headers(qcx-linalg PRIVATE
    <qcx/linalg/dense_ops.hpp>)

# Vendor BLAS via the uniform CBLAS interface: every
# vendor exports the same cblas_* symbols, so this block is the only
# vendor-specific part of the codebase. Auto-detect MKL -> AOCL -> OpenBLAS;
# without a vendor, DenseMultiply stays on the Eigen path and BLAS-less
# builds (CI, WSL) stay green. An explicit QCX_BLAS_VENDOR choice
# fails hard if that vendor is missing.
set(QCX_BLAS_VENDOR "" CACHE STRING
    "Vendor BLAS: MKL, AOCL, or OpenBLAS (empty = auto-detect, None = Eigen only)")
set(_qcxBlasVendors MKL AOCL OpenBLAS None)
if (QCX_BLAS_VENDOR AND NOT QCX_BLAS_VENDOR IN_LIST _qcxBlasVendors)
    message(FATAL_ERROR "Unknown QCX_BLAS_VENDOR '${QCX_BLAS_VENDOR}' - "
                        "allowed: MKL, AOCL, OpenBLAS, None, or empty for auto-detect")
endif()
# MKL must use its sequential threading layer: the intel_thread layer loads
# Intel's OpenMP runtime (libiomp5md), whose Microsoft-compat exports
# (_vcomp_fork) win the linker over vcomp in any binary that also carries
# the backend's #pragma omp code, while the for/critical helpers still bind
# to vcomp - the split-brain construct then runs every loop iteration on
# every thread (measured 12x on H2/STO-3G, 2026-08-17).
# Parallelism is the backend's OpenMP loops; the BLAS kernels run serial per
# thread by design.
set(MKL_THREADING sequential)
if (QCX_BLAS_VENDOR STREQUAL "None")
    message(STATUS "BLAS disabled - DenseMultiply uses Eigen for all sizes")
elseif (QCX_BLAS_VENDOR)
    find_package(${QCX_BLAS_VENDOR} CONFIG REQUIRED)
    target_link_libraries(qcx-linalg PUBLIC ${QCX_BLAS_VENDOR}::${QCX_BLAS_VENDOR})
    target_compile_definitions(qcx-linalg PUBLIC QcxHasVendorBlas=1)
    if (QCX_BLAS_VENDOR STREQUAL "MKL")
        # MKL ships mkl_cblas.h instead of cblas.h - the only per-vendor
        # header difference in the whole BLAS seam.
        target_compile_definitions(qcx-linalg PUBLIC QcxVendorBlasIsMkl=1)
        set(_qcxBlasVendorMkl TRUE)
        set_property(GLOBAL PROPERTY qcxVendorBlasIsMkl TRUE)
        set_property(GLOBAL PROPERTY qcxMklRoot "${MKL_ROOT}")
    elseif (QCX_BLAS_VENDOR STREQUAL "AOCL")
        # AOCL-BLAS threads by default - pin it to one thread via the
        # BLIS_NUM_THREADS process environment (no link-level sequential
        # variant).
        set(_qcxBlasNonMklThreadPin "BLIS_NUM_THREADS=1")
    elseif (QCX_BLAS_VENDOR STREQUAL "OpenBLAS")
        # OpenBLAS threads by default - OPENBLAS_NUM_THREADS=1 for the
        # same one-thread pin.
        set(_qcxBlasNonMklThreadPin "OPENBLAS_NUM_THREADS=1")
    endif()
else()
    find_package(MKL QUIET CONFIG)
    if (MKL_FOUND)
        target_link_libraries(qcx-linalg PUBLIC MKL::MKL)
        target_compile_definitions(qcx-linalg PUBLIC QcxHasVendorBlas=1 QcxVendorBlasIsMkl=1)
        set(_qcxBlasVendorMkl TRUE)
        set_property(GLOBAL PROPERTY qcxVendorBlasIsMkl TRUE)
        set_property(GLOBAL PROPERTY qcxMklRoot "${MKL_ROOT}")
    else()
        find_package(AOCL QUIET CONFIG)
        if (AOCL_FOUND)
            target_link_libraries(qcx-linalg PUBLIC AOCL::AOCL)
            target_compile_definitions(qcx-linalg PUBLIC QcxHasVendorBlas=1)
            # AOCL-BLAS threads by default - the BLIS_NUM_THREADS=1
            # process-environment pin is the only single-thread control
            # (parity with the MKL link-level pin above; no link-level
            # sequential variant exists).
            set(_qcxBlasNonMklThreadPin "BLIS_NUM_THREADS=1")
        else()
            find_package(OpenBLAS QUIET CONFIG)
            if (OpenBLAS_FOUND)
                target_link_libraries(qcx-linalg PUBLIC OpenBLAS::OpenBLAS)
                target_compile_definitions(qcx-linalg PUBLIC QcxHasVendorBlas=1)
                # OpenBLAS threads by default - OPENBLAS_NUM_THREADS=1
                # delivers the same one-thread pin.
                set(_qcxBlasNonMklThreadPin "OPENBLAS_NUM_THREADS=1")
            else()
                message(STATUS "No vendor BLAS found - DenseMultiply uses Eigen for all sizes")
            endif()
        endif()
    endif()
endif()

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-linalg-tests
      tests/dense_ops_test.cpp
      tests/iterative_eigensolver_test.cpp
      tests/sparse_solver_test.cpp
      tests/batched_ops_test.cpp
    )
    target_link_libraries(qcx-linalg-tests PRIVATE qcx-linalg gtest gtest_main)
    target_precompile_headers(qcx-linalg-tests PRIVATE
        <gtest/gtest.h>
        <qcx/linalg/dense_ops.hpp>)

    qcx_apply_asan_runtime(qcx-linalg-tests)
    qcx_apply_blas_runtime(qcx-linalg-tests)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # PRE_TEST discovery: the discovery process does not inherit the
        # test's environment, so the MKL DLLs copied above must be present
        # before the exe runs. Sequential threading avoids Intel-OpenMP/
        # vcomp conflicts.
        gtest_discover_tests(qcx-linalg-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    elseif (DEFINED _qcxBlasNonMklThreadPin)
        # The OpenBLAS/AOCL fallback vendors thread by default, so every
        # ctest-launched test process receives the one-thread pin (asserted by
        # DenseOpsTest.FallbackVendorBlasThreadingPinIsApplied). Drivers and
        # benchmarks on the fallback vendors must export the same variable at
        # launch - CMake can only enforce it for ctest targets.
        gtest_discover_tests(qcx-linalg-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "${_qcxBlasNonMklThreadPin}"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-linalg-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
