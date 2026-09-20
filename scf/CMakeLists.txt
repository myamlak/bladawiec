add_library(qcx-scf
  src/rhf.cpp
  src/density_energy.cpp
  src/uhf.cpp
  src/diis.cpp
  src/symmetry_reduction.cpp
  src/internal/scf_common.cpp
  src/internal/symmetry_blocks.cpp
)
target_include_directories(qcx-scf PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-scf PUBLIC cxx_std_23)
# The public headers expose Molecule, Tensor, and Eigen matrix types, so the
# consumers need those include paths and link targets too; linalg stays
# implementation-only (DenseMultiply is called from rhf.cpp). The symmetry
# module feeds the blocked diagonalization and - since the
# full-group labeling header (qcx/scf/symmetry_labels.hpp) exposes
# qcx::symmetry::PointGroup and PointGroupName in its public structs -
# must reach the consumers as well; integrals provides the flattened shell
# list of the AO basis and - since the symmetry-reduction seam
# (qcx/scf/symmetry_reduction.hpp) exposes qcx::integrals::SymmetryReduction
# and qcx::basisset::BasisSet in its signatures - their include paths and
# links must reach the consumers (the consumers of scf sit downstream; both
# modules are upstream of scf, so the DAG is unaffected).
target_link_libraries(qcx-scf
  PUBLIC qcx-core qcx-molecule qcx-memory qcx-integrals qcx-basisset qcx-symmetry eigen
  PRIVATE qcx-linalg)
# symmetry_blocks.cpp hosts the full-group labeling stage and
# outgrows the default COFF section budget (C1128) - same pattern as the
# vendored stdexec/amgcl.
target_compile_options(qcx-scf PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

# PCH is mandatory on every target.
target_precompile_headers(qcx-scf PRIVATE
    <qcx/scf/rhf.hpp>
    <qcx/scf/uhf.hpp>
    <qcx/scf/density_energy.hpp>
    <Eigen/Dense>)

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-scf-tests
        tests/rhf_test.cpp
        tests/rhf_convergence_test.cpp
        tests/energy_leg_test.cpp
        tests/diis_test.cpp
        tests/dense_rhf_test.cpp
        tests/density_energy_test.cpp
        tests/seam_test.cpp
        tests/direct_rhf_test.cpp
        tests/ri_rhf_test.cpp
        tests/uhf_test.cpp
        tests/direct_uhf_test.cpp
        tests/symmetry_block_test.cpp
        tests/symmetry_reduction_test.cpp
        tests/robustness_uhf_test.cpp
        tests/uhf_symmetry_constraint_test.cpp
        tests/rhf_symmetry_constraint_test.cpp
        tests/rhf_2cycle_fallback_test.cpp
        tests/full_group_labels_test.cpp
        tests/rks_convergence_probe_test.cpp)
    # qcx-grid is here for the RKS convergence probe ONLY: that test
    # composes a Kohn-Sham Fock (XcGridEngine's Vxc and Exc) to measure which
    # convergence-gate leg binds under DFT's numerical profile. It is NOT
    # drift - qcx-scf itself does not link grid, and the module DAG permits
    # this edge (grid is rank 7, scf is rank 9, so scf may link anything
    # earlier). A test target only; the library's dependencies are unchanged.
    target_link_libraries(qcx-scf-tests PRIVATE qcx-scf qcx-integrals qcx-symmetry qcx-grid gtest gtest_main qcx-test-fixtures)
    # The internal step functions (scf_common.hpp, symmetry_blocks.hpp) are
    # exercised by the tests directly.
    target_include_directories(qcx-scf-tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/internal)

    qcx_apply_asan_runtime(qcx-scf-tests)
    qcx_apply_blas_runtime(qcx-scf-tests)
    qcx_apply_fast_test_mode(qcx-scf-tests)

    # The RI-J pin reads the vendored def2-universal-jfit from the source tree.
    target_compile_definitions(qcx-scf-tests PRIVATE
        QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")

    target_precompile_headers(qcx-scf-tests PRIVATE
        <gtest/gtest.h>
        <qcx/scf/rhf.hpp>
        <qcx/scf/uhf.hpp>
        <qcx/scf/density_energy.hpp>)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # PRE_TEST discovery: the discovery process does not inherit the
        # test's environment, so the MKL DLLs copied above must be present
        # before the exe runs. Sequential threading avoids Intel-OpenMP/
        # vcomp conflicts - same wiring as qcx-linalg-tests.
        gtest_discover_tests(qcx-scf-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-scf-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()
endif()
