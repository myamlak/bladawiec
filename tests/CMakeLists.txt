# Shared test fixtures (C60, nanotubes, the symmetry corpus, H2/STO-3G)
# consumed by the module test suites; test-only, not part of the public API.
add_library(qcx-test-fixtures STATIC
    fixtures/large_molecules.cpp
    fixtures/corpus.cpp
    fixtures/h2_sto3g.cpp
    fixtures/hf_sto3g.cpp
    fixtures/h2o_sto3g.cpp
    fixtures/benzene_sto3g.cpp
    fixtures/o2_sto3g.cpp
    fixtures/alkane_sto3g.cpp
    fixtures/shellset_fixture.cpp
    fixtures/h2o_ccpvdz.cpp
)
target_include_directories(qcx-test-fixtures PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/fixtures)
target_compile_features(qcx-test-fixtures PUBLIC cxx_std_23)
target_link_libraries(qcx-test-fixtures PUBLIC qcx-molecule qcx-basisset)
# The corpus-parsing fixtures (h2o_ccpvdz) read data/basis through this
# define; the storage tests' QcxStorageDataDir analog lives on their own
# target (storage/CMakeLists.txt).
target_compile_definitions(qcx-test-fixtures PRIVATE
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
# The IsFastOnlyMode() helper must see the same per-config definition
# as the consuming test targets.
qcx_apply_fast_test_mode(qcx-test-fixtures)
target_precompile_headers(qcx-test-fixtures PRIVATE
    <qcx/molecule/molecule.hpp>
    <qcx/basisset/basis_set.hpp>
    <Eigen/Dense>)

# The test executables below need gtest; the fixtures library above does
# not (benchmarks consume it with QCX_BUILD_TESTS off).
if (QCX_BUILD_TESTS AND TARGET gtest)
add_executable(qcx-smoke-test smoke_test.cpp)
target_link_libraries(qcx-smoke-test PRIVATE GTest::gtest_main)
target_precompile_headers(qcx-smoke-test PRIVATE
    <gtest/gtest.h>)

qcx_apply_asan_runtime(qcx-smoke-test)

add_test(NAME smoke COMMAND qcx-smoke-test)

# CRTP + variant dispatch pressure test.
add_executable(qcx-dispatch-prototype-test dispatch_prototype_test.cpp)
target_link_libraries(qcx-dispatch-prototype-test PRIVATE GTest::gtest_main qcx-backend)
target_precompile_headers(qcx-dispatch-prototype-test PRIVATE
    <gtest/gtest.h>
    <qcx/backend/cpu_backend.hpp>)

qcx_apply_asan_runtime(qcx-dispatch-prototype-test)

add_test(NAME dispatch-prototype COMMAND qcx-dispatch-prototype-test)

# SparsityPolicy prototype: the CSR primitive serving a LinK-style
# neighbor list and a QFMM-style tree.
add_executable(qcx-sparsity-pattern-test sparsity_pattern_test.cpp)
target_link_libraries(qcx-sparsity-pattern-test PRIVATE GTest::gtest_main qcx-memory qcx-test-fixtures)
target_precompile_headers(qcx-sparsity-pattern-test PRIVATE
    <gtest/gtest.h>
    <qcx/memory/sparsity_pattern.hpp>)

qcx_apply_asan_runtime(qcx-sparsity-pattern-test)

add_test(NAME sparsity-pattern COMMAND qcx-sparsity-pattern-test)

# The alkane fixture's non-C12 contract: every QFMM call site uses
# carbonCount = 12, so the error path and the methane/ethane branches are
# pinned here.
add_executable(qcx-alkane-fixture-test alkane_fixture_test.cpp)
target_link_libraries(qcx-alkane-fixture-test PRIVATE GTest::gtest_main qcx-test-fixtures)
target_precompile_headers(qcx-alkane-fixture-test PRIVATE
    <gtest/gtest.h>
    <qcx/molecule/molecule.hpp>)

qcx_apply_asan_runtime(qcx-alkane-fixture-test)

add_test(NAME alkane-fixture COMMAND qcx-alkane-fixture-test)

# One compile-check + round-trip per vendored Phase 0 dependency;
# the HDF5 half is gated by the same option that gates the dependency.
add_executable(qcx-vendor-smoke-test vendor_smoke_test.cpp)
target_link_libraries(qcx-vendor-smoke-test PRIVATE GTest::gtest_main json tomlplusplus)

if (QCX_ENABLE_IO)
    target_link_libraries(qcx-vendor-smoke-test PRIVATE highfive)
    target_compile_definitions(qcx-vendor-smoke-test PRIVATE QcxHasHdf5=1)
endif()

target_precompile_headers(qcx-vendor-smoke-test PRIVATE
    <gtest/gtest.h>
    <nlohmann/json.hpp>
    <toml++/toml.hpp>)

qcx_apply_asan_runtime(qcx-vendor-smoke-test)

add_test(NAME vendor-smoke COMMAND qcx-vendor-smoke-test)

# The CUDA MD tests. The exe lives here,
# not in integrals/: it links qcx-scf (the H2 RHF pin) - the module DAG
# (check_dag.py) forbids integrals from linking downstream modules. The suite
# self-skips without a CUDA device (the engine's Create probes the device
# count), so CUDA builds without a GPU stay green.
#
# DECLARED LIMITATION - NO CI JOB COMPILES THIS BLOCK. QCX_ENABLE_CUDA
# defaults OFF (CMakeLists.txt:34) and the only tree that turns it ON is the local
# windows-msvc-cuda preset (CMakePresets.json:30); the hosted runners ship no CUDA
# toolkit, so no job in .github/workflows reaches this guard. These five TUs, and the
# qcx-integrals-cuda-md sources they link, are therefore compiled by a local
# QCX_ENABLE_CUDA=ON tree only - a drift that breaks one of them fails no gate. The same hole
# affects integrals/src/eri_cuda.cpp too. Closing it needs a
# GPU-backed runner or an explicit toolkit install, neither of which fits the minutes
# budget.
if (QCX_ENABLE_CUDA AND TARGET qcx-integrals-cuda-md)
    add_executable(qcx-cuda-md-tests ${CMAKE_SOURCE_DIR}/integrals/tests/eri_cuda_test.cpp
        ${CMAKE_SOURCE_DIR}/integrals/tests/gpu_fock_build_test.cpp
        ${CMAKE_SOURCE_DIR}/integrals/tests/gpu_precision_ladder_test.cpp
        ${CMAKE_SOURCE_DIR}/integrals/tests/gpu_device_budget_test.cpp
        ${CMAKE_SOURCE_DIR}/integrals/tests/gpu_mode_selection_test.cpp)
    target_link_libraries(qcx-cuda-md-tests PRIVATE
        qcx-integrals-cuda-md qcx-scf qcx-test-fixtures gtest gtest_main)
    target_compile_definitions(qcx-cuda-md-tests PRIVATE
        QcxIntegralsDataDir="${CMAKE_SOURCE_DIR}/integrals/tests/data"
        QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")

    qcx_apply_asan_runtime(qcx-cuda-md-tests)
    qcx_apply_fast_test_mode(qcx-cuda-md-tests)
    # The CPU reference lane in the suite hits the MKL batch BLAS for larger
    # shellsets (the MicroGemm path covers only small batches); without the
    # full MKL runtime the delay load dies with exit 127 - the
    # app-local-deployment gap (the MKL incident class).
    qcx_apply_blas_runtime(qcx-cuda-md-tests)

    target_precompile_headers(qcx-cuda-md-tests PRIVATE
        <gtest/gtest.h>
        <qcx/integrals/eri_cuda.hpp>)

    include(GoogleTest)
    gtest_discover_tests(qcx-cuda-md-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
endif()
endif()
