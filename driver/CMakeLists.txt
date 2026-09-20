# The qcx run driver (the `qcx run input.toml` entry point):
# the composition layer that turns the parsed RunInput schema into the SCF
# run, the properties block, and the serialized JSON
# result. Module-DAG position: io -> properties -> driver (check_dag.py) -
# driver is the last link, the one module allowed to compose everything.

add_library(qcx-driver
  src/run_driver.cpp
  src/process_caps.cpp
  src/accuracy_class_report.cpp
  src/selection_heuristic.cpp
  src/selection_resolution.cpp
)
target_include_directories(qcx-driver PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qcx-driver PUBLIC cxx_std_23)
# io is PUBLIC because run_driver.hpp exposes RunInput in the API; the seam
# modules (scf, the builders, the molecule/basis layers) are the body the
# driver composes; properties, memory and grid stay PRIVATE
# (implementation details: properties feeds the result block, memory the
# attribution opt-in and the workspace-budget routes, grid the Kohn-Sham
# composition's XC engine - the Kohn-Sham consumer wired in
# run_driver.cpp, whose engine never leaves the driver's own callbacks).
# storage is PRIVATE as well, and CONDITIONAL: the module is optional (it
# carries the disk rung's driver route - the ri_tensor_mode disk knob -
# which constructs DiskRiFockBuilder inside run_driver.cpp), so it is linked
# in the QCX_ENABLE_IO block below rather than here. Linked here it was a bare
# library name on a build that never added the module, and CMake resolves that
# without error: no usage requirements travel, the include directory never
# reaches the compile, and the driver fails to compile instead.
target_link_libraries(qcx-driver
  PUBLIC qcx-io qcx-scf qcx-integrals qcx-basisset qcx-molecule
  PRIVATE qcx-properties qcx-memory qcx-grid)
# The bundled basis directory is baked into the library (the same macro the
# basisset tests define for themselves); callers never name the path.
target_compile_definitions(qcx-driver PRIVATE
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")

# Precompiled headers.
target_precompile_headers(qcx-driver PRIVATE
    <Eigen/Dense>
    <qcx/driver/run_driver.hpp>
    <qcx/properties/populations.hpp>
    <qcx/properties/multipoles.hpp>)

# The qcx CLI: `qcx run <input.toml>` (OUTPUT_NAME "qcx", set below).
add_executable(qcx-driver-run
  src/main.cpp
)
target_compile_features(qcx-driver-run PRIVATE cxx_std_23)
target_link_libraries(qcx-driver-run PRIVATE qcx-driver qcx-io)
target_precompile_headers(qcx-driver-run PRIVATE
    <qcx/driver/run_driver.hpp>
    <qcx/io/parse_input.hpp>)
set_target_properties(qcx-driver-run PROPERTIES OUTPUT_NAME "qcx")
# Every target that links the BLAS seam applies the MKL runtime deployment;
# without it the exe dies with the 0xC06D007E raise (2026-08-18).
qcx_apply_blas_runtime(qcx-driver-run)

if (QCX_ENABLE_CUDA)
    # GpuJkFockBuilder's definitions live in the CUDA MD lane library
    # (integrals/CMakeLists.txt, eri_cuda.cpp), not in qcx-integrals; the
    # kGpu wire in WireRhfFockBuilder references it under QcxHasCuda, so
    # every driver target must link the lane library in CUDA builds.
    target_link_libraries(qcx-driver PRIVATE qcx-integrals-cuda-md)
    target_link_libraries(qcx-driver-run PRIVATE qcx-integrals-cuda-md)
endif()

# The storage module is optional (the root CMakeLists adds the subdirectory
# under the same option), and the driver's four storage-backed sites - the ERI
# store decorator, the disk rung, the checkpoint read and the checkpoint write
# - are guarded by QcxHasStorage, the QcxHasCuda pattern (see
# backend/CMakeLists.txt). PUBLIC, for the QcxHasCuda reason: the test targets
# and the CLI link qcx-driver and inherit the definition, so the guards in
# their sources and the module they link cannot disagree. Without the module
# the driver still builds and every storage-backed route refuses or discloses
# by name - the cause is disclosed, never the request silently dropped.
if (QCX_ENABLE_IO)
    target_compile_definitions(qcx-driver PUBLIC QcxHasStorage=1)
    target_link_libraries(qcx-driver PRIVATE qcx-storage)
endif()

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-driver-tests
        tests/run_driver_test.cpp
        tests/process_caps_test.cpp
        tests/accuracy_class_report_test.cpp
        tests/selection_heuristic_test.cpp
        tests/selection_resolution_test.cpp
        tests/selection_reporting_test.cpp
        tests/ks_composition_test.cpp
        tests/ri_chunk_bytes_record_test.cpp
        tests/eri_store_record_test.cpp
        tests/run_input_help_test.cpp)
    # src PRIVATE: ks_composition_test.cpp includes the driver-internal
    # composition header (src/internal/ks_composition.hpp) to pin its
    # arithmetic against halves whose numbers are written down in the test.
    # Same spelling as qcx-integrals-tests (integrals/CMakeLists.txt).
    target_include_directories(qcx-driver-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    # The schema-help case reads the help text this tree prints
    # (src/internal/cli_help_text.hpp, so PRIVATE src above is what brings it
    # in) and holds it to the PARSER: the key set is scanned out of
    # io/src/parse_input.cpp, so the test needs the tree root. The same device
    # the driver library uses for data/basis, one level up.
    target_compile_definitions(qcx-driver-tests PRIVATE
        QcxSourceDir="${CMAKE_SOURCE_DIR}")
    target_link_libraries(qcx-driver-tests PRIVATE
        qcx-driver qcx-grid qcx-test-fixtures gtest gtest_main)

    if (QCX_ENABLE_IO)
        # QcxHasStorage comes in through qcx-driver (PUBLIC), but the
        # definition alone links nothing: the cases that call the checkpoint
        # reader and writer directly need the module's own target, whose
        # PUBLIC include directory is what the guarded include resolves to.
        target_link_libraries(qcx-driver-tests PRIVATE qcx-storage)
    endif()

    if (QCX_ENABLE_CUDA)
        target_link_libraries(qcx-driver-tests PRIVATE qcx-integrals-cuda-md)
    endif()

    # The small-cap test child (team_size_test pattern): a
    # dedicated binary, because a memory cap applies to the whole process -
    # the parent test capping its own host binary would kill the suite.
    # process_caps_test.cpp spawns it; both exes build into driver/<cfg>,
    # so the test finds it next to itself (no path plumbing).
    add_executable(qcx-driver-cap-child
        tests/cap_child.cpp)
    target_compile_features(qcx-driver-cap-child PRIVATE cxx_std_23)
    target_link_libraries(qcx-driver-cap-child PRIVATE qcx-driver)
    qcx_apply_blas_runtime(qcx-driver-cap-child)
    target_precompile_headers(qcx-driver-cap-child PRIVATE
        <qcx/driver/run_driver.hpp>)

    if (QCX_ENABLE_CUDA)
        target_link_libraries(qcx-driver-cap-child PRIVATE qcx-integrals-cuda-md)
    endif()

    qcx_apply_asan_runtime(qcx-driver-tests)
    qcx_apply_blas_runtime(qcx-driver-tests)
    qcx_apply_fast_test_mode(qcx-driver-tests)

    target_precompile_headers(qcx-driver-tests PRIVATE
        <gtest/gtest.h>
        <qcx/driver/run_driver.hpp>)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # Same MKL DLL/threading wiring as qcx-scf-tests.
        gtest_discover_tests(qcx-driver-tests
            DISCOVERY_MODE PRE_TEST
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-driver-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()

    # THE LARGE CATEGORY (owner ruling 2026-09-18; see the root CMakeLists for
    # the mechanism). These cases leave the routine selection and run with
    # `ctest -L large` instead.
    #
    # The `heavy` label these pins used to carry FOLDED INTO `large` here:
    # one exclusion mechanism, not two. The pins keep their old cost story
    # (measured 2026-09-13: C12 1040.63 s, C24 killed at 1996.45 s; measured
    # again 2026-09-18 in the GCC Release limb: C12 952.66 s, C24 5955.53 s)
    # and go on asserting everything they did. The `heavy` name is RETIRED -
    # nothing else references it.
    #
    # DriverErrorTest.RefusesNocvDenseTensorWhenTheCapCannotFitIt is here at
    # 76.08 s, the only driver error case over the 60 s bar.
    #
    # TRIMMED 2026-09-18 (the fixture-size trim; owner ruling 2026-09-18). The
    # kTight row of both pins moved to
    # C6ChainQfmmRhfMatchesDirectInsideThePresetBudgets, so that claim now sits
    # on the smallest fixture that carries it - and that case is IN the large
    # list below, with the other two pins. The note used to call it "left
    # UNLABELLED on purpose", a sentence written before this category existed
    # and left behind when it was created: the list below is what the file does,
    # and the case carries ~9 s + ~55 s of real SCF work.
    # AlkaneChainToml's SCF gate came down from 1e-10/1e-10 to the
    # defaults 1e-8/1e-6; no band, reference or expected value moved with it.
    #
    # THE CAP FAMILY LEFT THE ROUTINE SELECTION (2026-09-19) and joined this
    # category: every case that spawns a capped child process, i.e. the whole
    # RUN_SERIAL set below. They are INTEGRATION cases - real SCFs inside a
    # job-object-capped child, 1210 basis functions in the largest - not unit
    # tests, and a case whose subject is the ADMISSION DECISION at a given cap
    # is the kind that breaks whenever the admission is retuned; the routine
    # selection carries the algorithm's behaviour (selection_resolution_test
    # and selection_reporting_test hold the ladder's own edges). Their measured
    # cost, after the bounded wait in run_driver_test.cpp turned a 2400 s stall
    # into a 0.31 s refusal (2026-09-19, CI run 35451596080, Release legs):
    # 21.4 s for the whole family on linux-x86 gcc (plus 0.42 s for the
    # disk-rung case, which joined the list afterwards) and 5.8 s on windows-msvc -
    # RiJkGrantSelectsTheBlockedRungWhenTheFastRungDoesNotFit 17.70 s (Linux
    # only; skipped on Windows), ClassTableAdmissionGateEngagesOnThe
    # MaterializedShape 2.45 s there and 5.01 s on Windows, every other case
    # under a second. DriverErrorTest.LeanWordAdmitsTheLeanMemberAboveThe
    # Ceiling is live in THIS tree (the public cut deleted it) and is labelled
    # with the family like the rest: its own cost is 8.84 s on the GCC Release
    # limb (one run, this machine's CTestCostData), well under the 60 s bar.
    qcx_mark_large_cases(driver
        DriverPinTest.C6ChainQfmmRhfMatchesDirectInsideThePresetBudgets
        DriverPinTest.C12ChainQfmmRhfMatchesDirectInsideThePresetBudgets
        DriverPinTest.C24ChainQfmmRhfMatchesDirectInsideThePresetBudgets
        DriverErrorTest.RefusesNocvDenseTensorWhenTheCapCannotFitIt
        DriverErrorTest.LeanWordAdmitsTheLeanMemberAboveTheCeiling
        DriverErrorTest.TightCapRefusesBeforeThePreGateSetupRamp
        DriverErrorTest.OverCapDirectRunProceedsUnderTheRealCap
        DriverErrorTest.RiJProceedsWhenTheCapDoesNotClearTheBase
        DriverErrorTest.RiJkProceedsWhenTheCapDoesNotClearTheBase
        DriverErrorTest.RiJkGrantSelectsTheBlockedRungWhenTheFastRungDoesNotFit
        DriverErrorTest.ClassTableAdmissionGateEngagesOnTheMaterializedShape
        DriverRunTest.RiTensorModeDiskEngagesTheDiskRungAfterTheLaddersRefusal
        ProcessCapsTest.SmallCapDoesNotDisturbASmallRun
        ProcessCapsTest.TinyCapKillsTheChildAtTheCap
        ProcessCapsTest.EqualCapAppliesReuseTheJobInForce
        ProcessCapsTest.JobObjectCeilingRefusalNamesTheCeiling
        ProcessCapsTest.CappedProcessReadsItsCapAsEnforced)

    # THE CAP FAMILY: every case that spawns a capped child process
    # (qcx::testing::RunCapChild, i.e. qcx-driver-cap-child under a job-object
    # memory ceiling). A cap is PROCESS-GLOBAL, so a cap test running beside
    # anything else fails the neighbour for a reason that has nothing to do
    # with it - and under the `ctest -j` this category introduces that reads
    # as a real defect. RUN_SERIAL, not RESOURCE_LOCK: a lock serializes only
    # the cap holders against each other and lets every other test run
    # alongside them.
    #
    # Derived from the RunCapChild call sites in run_driver_test.cpp and from
    # process_caps_test.cpp's whole suite (its cases cap by construction; the
    # file's _WIN32/#else pairs register one name per platform, so the same
    # five names hold on both).
    #
    # Serialization still matters now that the family is large-labelled:
    # the large lane runs `ctest -L large -j <N>` and a cap is process-global
    # there too.
    qcx_mark_serial_cases(driver
        ProcessCapsTest.SmallCapDoesNotDisturbASmallRun
        ProcessCapsTest.TinyCapKillsTheChildAtTheCap
        ProcessCapsTest.EqualCapAppliesReuseTheJobInForce
        ProcessCapsTest.JobObjectCeilingRefusalNamesTheCeiling
        ProcessCapsTest.CappedProcessReadsItsCapAsEnforced
        DriverErrorTest.LeanWordAdmitsTheLeanMemberAboveTheCeiling
        DriverErrorTest.TightCapRefusesBeforeThePreGateSetupRamp
        DriverErrorTest.OverCapDirectRunProceedsUnderTheRealCap
        DriverErrorTest.RiJProceedsWhenTheCapDoesNotClearTheBase
        DriverErrorTest.RiJkProceedsWhenTheCapDoesNotClearTheBase
        DriverErrorTest.RiJkGrantSelectsTheBlockedRungWhenTheFastRungDoesNotFit
        DriverErrorTest.ClassTableAdmissionGateEngagesOnTheMaterializedShape
        DriverErrorTest.RefusesNocvDenseTensorWhenTheCapCannotFitIt
        DriverRunTest.RiTensorModeDiskEngagesTheDiskRungAfterTheLaddersRefusal)
    # NOT in this list, and the omissions are deliberate - the set is derived
    # from the sources (every case that calls RunCapChild), not from a
    # recollection of the family:
    #   * DriverRunTest.RiTensorModeDiskEngagesTheDiskRungAfterTheLaddersRefusal
    #     IS in both lists, and a note here once recorded the opposite ("DOES
    #     NOT EXIST - removed 2026-09-18"): the case is live in the sources
    #     (grepped 2026-09-19, driver/tests/run_driver_test.cpp) and had been
    #     the one RunCapChild call site carrying no label at all. Re-derive
    #     this set from the sources whenever it moves - a name that does not
    #     exist makes every ctest run in the tree fail, and a live case left
    #     out runs beside the other cap holders, which is what RUN_SERIAL is
    #     here to prevent.
    #   * DriverRunTest.RiTensorModeDiskNeverPreemptsAFittingMemoryRung and
    #     RiTensorModeForcedDiskBypassesAFittingLadder are NOT serial members.
    #     They were proposed for this set on the reasoning that the family is
    #     cap-bound, but neither body contains RunCapChild or any job-object
    #     call (checked by extracting both bodies, 2026-09-18) - they are
    #     ordinary in-process tests, and RUN_SERIAL on them would cost wall
    #     clock for no isolation.
    # The cap family is exactly the 14 above: 5 ProcessCapsTest cases (each
    # registered twice in process_caps_test.cpp, in its _WIN32 and #else arms,
    # under the same name - so 5 names hold on both platforms, not 10), 8
    # DriverErrorTest cases, and the DriverRunTest disk-rung case.
    # DriverErrorTest.RefusesNocvDenseTensorWhenThe
    # CapCannotFitIt is in BOTH lists on purpose: RUN_SERIAL and LABELS large
    # are independent properties and both apply.
endif()
