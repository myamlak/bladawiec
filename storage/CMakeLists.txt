add_library(qcx-storage
    src/fingerprint.cpp
    src/hdf5_util.cpp
    src/eri_store.cpp
    src/cached_eri_batch_engine.cpp
    src/scf_checkpoint.cpp
    src/ri_tensor_store.cpp
    src/ri_tensor_chunk_store.cpp
    src/disk_ri_fock_build.cpp
    src/screened_eri_store.cpp
)
target_include_directories(qcx-storage PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-storage PUBLIC cxx_std_23)
# Public headers expose qcx-integrals (EriBatch, CanonicalQuartetInfo) and
# qcx-scf (ScfRestartState) types, so both are PUBLIC; highfive and linalg
# (the disk builder's products go through the DenseMultiply seam of plan
# Section 6.4) are implementation-only and stay PRIVATE (no HighFive or
# linalg type reaches a header; linalg sits upstream of scf, so the DAG is
# unaffected).
target_link_libraries(qcx-storage
    PUBLIC qcx-core qcx-integrals qcx-scf
    PRIVATE highfive qcx-linalg)
target_precompile_headers(qcx-storage PRIVATE
    <highfive/H5File.hpp>
    <Eigen/Dense>
    <qcx/integrals/eri_batch.hpp>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-storage-tests
        tests/fingerprint_test.cpp
        tests/eri_store_test.cpp
        tests/cached_eri_engine_test.cpp
        tests/engine_decorator_seam_test.cpp
        tests/disk_tier_four_config_test.cpp
        tests/ri_tensor_test.cpp
        tests/ri_tensor_chunk_store_test.cpp
        tests/disk_ri_fock_build_test.cpp
        tests/screened_eri_store_test.cpp
        tests/scf_checkpoint_test.cpp
        tests/pyscf_cross_check_test.cpp)
    target_link_libraries(qcx-storage-tests PRIVATE
        qcx-storage qcx-test-fixtures gtest gtest_main highfive)
    # The tests reach the module's internal test hook (src/test_hooks.hpp)
    # for the append failure-injection seam.
    target_include_directories(qcx-storage-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(qcx-storage-tests PRIVATE
        QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis"
        QcxStorageDataDir="${CMAKE_SOURCE_DIR}/storage/tests/data")
    qcx_apply_asan_runtime(qcx-storage-tests)
    qcx_apply_blas_runtime(qcx-storage-tests)
    qcx_apply_fast_test_mode(qcx-storage-tests)
    target_precompile_headers(qcx-storage-tests PRIVATE
        <gtest/gtest.h>
        <highfive/H5File.hpp>)
    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        gtest_discover_tests(qcx-storage-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-storage-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()

# The pyscf reference grids' byte-identity check (the regeneration-check
# convention): the committed CSVs must match a fresh generation.
# Local-only (QCX_REFERENCE_REGENERATION_CHECK, default OFF): no CI
# configuration ever registers it (pyscf lives in WSL, not Windows); the
# script exits 77 (SKIP) when pyscf is missing.
if (QCX_REFERENCE_REGENERATION_CHECK)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME stage8-pyscf-reference-regeneration
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/gen_stage8_reference.py --check)
endif()
