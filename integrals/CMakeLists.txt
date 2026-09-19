# Instantiation cap for the per-class MD integral engines.
# The local build compiles the full l <= 6 class matrix; CI configures with 2 so
# per-push jobs stay within the minutes budget - no CI configuration ever builds
# the full matrix.
set(QCX_INTEGRALS_LMAX "6" CACHE STRING
    "Max shell l instantiated by the MD 2e/3c engines (0..6; CI builds use 2)")
if (NOT QCX_INTEGRALS_LMAX MATCHES "^[0-6]$")
    message(FATAL_ERROR "QCX_INTEGRALS_LMAX must be an integer in 0..6, got '${QCX_INTEGRALS_LMAX}'")
endif()

set(QCX_INTEGRALS_CUDA_LMAX "" CACHE STRING
    "CUDA instantiation max l (empty = QCX_INTEGRALS_LMAX)")
set(QCX_INTEGRALS_CHUNKED_VRR OFF CACHE BOOL
    "Runtime-loop VRR fallback for classes L >= 8 (MSVC codegen gate)")
set(QCX_INTEGRALS_F32 ON CACHE BOOL
    "Instantiate the certified fp32 pipeline alongside fp64 (doubles the class matrix - the gate measures both)")
set(QCX_REFERENCE_REGENERATION_CHECK OFF CACHE BOOL
    "Register the table/CSV byte-identity regeneration ctests (local-only, never in CI; mpmath, minutes per run)")
# The fp16/bf16 Boys lane (the certified fp32 boundary extension): fp16 I/O
# around the certified fp32 engine, a build-time seam in the QCX_ENABLE_CUDA /
# QCX_ASAN shape, default ON. F16/Bf16 alias std::float16_t / std::bfloat16_t
# where the toolchain ships them (GCC 13+, Clang 17+); the MSVC STL declares
# no extended types (its <stdfloat> is empty), so f16.hpp supplies the
# self-contained wrappers there - the lane is available on every toolchain.
set(QCX_INTEGRALS_FP16 ON CACHE BOOL
    "Build the certified fp16/bf16 Boys lane (F16/Bf16 I/O over the fp32 engine)")

# The Boys-function kernel is consumed from the boys submodule
# (external/boys, gitlink pinned at v1.0.0 - the submodule's own repository is
# the source of truth, so this repository bumps the gitlink per release rather
# than editing the submodule). The kernel compiles from
# the submodule's sources into qcx-integrals; the qcx/integrals/boys.hpp and
# f16.hpp headers are forwarding shims. The generated kernel files
# (src/boys_coefficients.hpp, tests/data/boys_reference.csv) live in the
# submodule and are pinned by boys-reference-regeneration below - this
# repository never regenerates or edits them.
set(_qcxBoysExt "${CMAKE_SOURCE_DIR}/external/boys")

add_library(qcx-integrals
  ${_qcxBoysExt}/src/boys.cpp
  ${_qcxBoysExt}/src/boys_simd.cpp
  src/one_electron.cpp
  src/two_electron.cpp
  src/eri_batch.cpp
  src/eri_cache.cpp
  src/eri_dense.cpp
  src/fock_build.cpp
  src/fock_contract_simd.cpp
  src/lean_fock_build.cpp
  src/incremental_fock.cpp
  src/precision_policy.cpp
  src/ri_engine.cpp
  src/ri_engine_chunk.cpp
  src/ri_occ_k.cpp
  src/ri_full_fock.cpp
  src/screening.cpp
  src/shell_pairs.cpp
  src/symmetry_reduction.cpp
  src/internal/md_batch.cpp
  src/internal/md_bra_simd.cpp
  src/internal/qfmm_geometry.cpp
  src/internal/qfmm_tree.cpp
  src/internal/qfmm_multipole.cpp
  src/qfmm_fock_build.cpp
  src/qfmm_hf_build.cpp
)
target_include_directories(qcx-integrals PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${_qcxBoysExt}/include)
target_include_directories(qcx-integrals PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${_qcxBoysExt}/src)
target_compile_features(qcx-integrals PUBLIC cxx_std_23)
# The public headers expose Molecule/BasisSet/Tensor/Result types, so the
# consumers need those include paths too; Eigen stays implementation-only
# (internal/shells_flat.hpp). The linalg batched seam and the CPU
# backend are implementation-only consumers (DAG-legal: both precede
# integrals). ${_qcxBoysExt}/include (PUBLIC) lets the forwarding shims'
# <boys/...> includes resolve in every consumer TU.
target_link_libraries(qcx-integrals
  PUBLIC qcx-core qcx-molecule qcx-basisset qcx-memory
  PRIVATE eigen qcx-linalg qcx-backend)
# The engine caps consumed by the public limits.hpp and the internal
# dispatch table (QcxIntegralsF32=0 is defined too so #if guards are
# uniform). BoysFp16 mirrors QCX_INTEGRALS_FP16 onto the submodule's seam
# (its headers gate the fp16/bf16 entries on BoysFp16); QcxIntegralsFp16
# stays defined for the retained files gated on the old name.
target_compile_definitions(qcx-integrals PUBLIC
    QcxIntegralsLMax=${QCX_INTEGRALS_LMAX}
    QcxIntegralsF32=$<BOOL:${QCX_INTEGRALS_F32}>
    QcxIntegralsFp16=$<BOOL:${QCX_INTEGRALS_FP16}>
    BoysFp16=$<BOOL:${QCX_INTEGRALS_FP16}>)
if (QCX_INTEGRALS_CHUNKED_VRR)
    target_compile_definitions(qcx-integrals PUBLIC QcxIntegralsChunkedVrr=1)
endif()

# AVX2 intrinsics only in the SIMD translation unit; the entry points are
# precondition-checked at runtime via BoysAvx2Available(). The TU skips the
# module PCH: clang rejects a PCH built without the target feature the TU
# compiles with.
if (QCX_ARCH_X86_64)
    if (MSVC)
        # /arch:AVX2 also enables the F16C intrinsics the fp16 lane uses.
        set_source_files_properties(${_qcxBoysExt}/src/boys_simd.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
        set_source_files_properties(src/internal/md_bra_simd.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        # -mf16c for the fp16 lane's F16C load/store (GCC does not imply it via
        # -mavx2, unlike MSVC's /arch:AVX2).
        set_source_files_properties(${_qcxBoysExt}/src/boys_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
        # -ffp-contract=off is LOAD-BEARING here, not hygiene (2026-09-13). The
        # TU's own BIT-IDENTITY section contracts this tier to the SCALAR kernels
        # the class TUs instantiate (no /arch flag), and simd_micro_gemm_test.cpp
        # enforces it with memcmp. MulAdd keeps the hand-written vectors
        # contraction-neutral, but that only covers what the SOURCE can control:
        # the scalar tails (SmallnScalarTailAdd and friends, plain `a * b + c`)
        # are fused by the COMPILER under this TU's -mfma, and gcc re-fuses the
        # _mm256_mul_pd + _mm256_add_pd pairs inside MulAdd itself. Measured, the
        # scalar TUs unmoved at 0 both ways: clang 21 at the CI Debug geometry 7
        # fused -> 0, g++ 15.2 -O3 54 (32 packed) -> 0.
        set_source_files_properties(src/internal/md_bra_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
    endif()
    set_source_files_properties(${_qcxBoysExt}/src/boys_simd.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    set_source_files_properties(src/internal/md_bra_simd.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
else()
    # Non-x86-64: no AVX2/FMA/F16C option exists to set, and neither TU defines
    # the tier. md_bra_simd.cpp routes its entry points to the portable kernels.
    # boys' SIMD TU is told its answer explicitly, because its own guard is
    # written for a consumer that compiles that source into a target of its own
    # and refuses to guess where the compiler is not x86-64. The PCH skip goes
    # with the flag it exists for.
    set_source_files_properties(${_qcxBoysExt}/src/boys_simd.cpp PROPERTIES COMPILE_DEFINITIONS "BOYS_SIMD_X86=0")
    message(STATUS "qcx-integrals: non-x86-64 target - the AVX2 micro-GEMM tier is excluded")
endif()

# The Fock-contraction kernel's AVX2 copy. The kernel body is
# ONE template (internal/fock_contract_kernel.hpp); the scalar copy
# instantiates inside fock_build.cpp (no /arch flag on that TU) and this TU
# compiles the AVX2 copy, entered only after the runtime cpuid check
# (FockAvx2Available - the boys_simd.cpp dispatch shape, auto-vectorization
# only, no hand-written intrinsics at this stage). Same PCH skip as the
# other SIMD TUs.
#
# -ffp-contract=off is LOAD-BEARING here, not hygiene (2026-09-13). The
# header states "both copies must deliver bit-identical output"; the scalar
# TUs carry no target flag, so there is no FMA instruction for them to
# contract into, while this TU's -mfma does - and clang defaults to
# -ffp-contract=on and gcc to -ffp-contract=fast. Measured at the CI Debug
# geometry: 10 vfmadd213sd in this TU against 0 in lean_fock_build.cpp and
# fock_build.cpp, which broke
# LeanDirectFockBuilderTest.SerialPathPinsItsOwnBytesAcrossCallsAndBuilders
# on the clang leg (5 of the 49 H2O/STO-3G elements, max 4.44e-16).
if (QCX_ARCH_X86_64)
    if (MSVC)
        # No counterpart owed: /fp:precise (the default) does not contract -
        # measured 0 fused in this tree's own Avx2 objects, Debug and Release,
        # with /fp:fast at /O2 as the 2-fused positive control. A future
        # /fp:contract would have to come back here.
        set_source_files_properties(src/fock_contract_simd.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    else()
        # The flag belongs on the SIMD TUs ONLY. On a scalar TU it is inert
        # (measured 0 fused either way, there is no FMA to fuse) and it would
        # hide the regression it exists to catch: a scalar TU silently gaining a
        # target feature and contracting away from the copy it must match.
        set_source_files_properties(src/fock_contract_simd.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
    endif()
    set_source_files_properties(src/fock_contract_simd.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
else()
    # Same as the micro-GEMM block above: no target feature exists here to set,
    # and this TU's copy is entered only after FockAvx2Available(), which reads
    # the x86-64-only CPU feature probe and is false on this target. No flag
    # means no contraction, which is exactly what the scalar copies carry, so
    # the bit-identity the two are compared under still holds.
    message(STATUS "qcx-integrals: non-x86-64 target - the Fock contraction AVX2 copy is excluded")
endif()

# PCH.
target_precompile_headers(qcx-integrals PRIVATE
    <qcx/integrals/boys.hpp>)

# The per-class instantiation matrix: one OBJECT library per {L_bra, L_ket}
# class - the 2e triangle (LBra <= LKet <= 2*QCX_INTEGRALS_LMAX) UNION the
# RI rectangle (LKet <= QCX_INTEGRALS_LMAX, LBra <= 2*QCX_INTEGRALS_LMAX);
# 154 classes at Lmax=6, 24 at Lmax=2 (the counts match the CUDA matrix
# below). Parallel compilation (MSBuild multi-project), incremental builds
# recompile only touched classes, and the hot template surface stays out of
# the public headers - dispatch runs through the generated function-pointer
# table (md_dispatch_gen.hpp).
function(qcx_add_md_class_object lBra lKet)
    set(_genDir ${CMAKE_CURRENT_BINARY_DIR}/md_instantiate)
    set(_tu ${_genDir}/md_instantiate_${lBra}_${lKet}.cpp)
    configure_file(src/md_instantiate.cpp.in ${_tu} @ONLY)
    add_library(qcx-md-class-${lBra}-${lKet} OBJECT ${_tu})
    target_include_directories(qcx-md-class-${lBra}-${lKet} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${_qcxBoysExt}/include)
    target_compile_features(qcx-md-class-${lBra}-${lKet} PRIVATE cxx_std_23)
    target_compile_definitions(qcx-md-class-${lBra}-${lKet} PRIVATE
        QcxIntegralsLMax=${QCX_INTEGRALS_LMAX}
        QcxIntegralsF32=$<BOOL:${QCX_INTEGRALS_F32}>)
    if (QCX_INTEGRALS_CHUNKED_VRR)
        target_compile_definitions(qcx-md-class-${lBra}-${lKet} PRIVATE
            QcxIntegralsChunkedVrr=1)
    endif()
    # Usage requirements of the linked targets reach the object TU's compile
    # (include dirs, definitions); nothing links from the object lib itself.
    target_link_libraries(qcx-md-class-${lBra}-${lKet} PRIVATE
        qcx-core qcx-molecule qcx-basisset qcx-memory qcx-linalg eigen)
    target_precompile_headers(qcx-md-class-${lBra}-${lKet} PRIVATE
        <internal/md_vrr.hpp>)
endfunction()

math(EXPR _qcxLmax2 "2 * ${QCX_INTEGRALS_LMAX}")
foreach(_lket RANGE 0 ${_qcxLmax2})
    # The 2e triangle (lBra <= lKet) UNION the RI rectangle
    # (lKet <= Lmax, lBra <= 2*Lmax): the 3c kernels use the orbital pair
    # as the bra and the aux shell as the ket, so lBra can exceed lKet.
    if (_lket LESS_EQUAL ${QCX_INTEGRALS_LMAX})
        set(_lbraMax ${_qcxLmax2})
    else()
        set(_lbraMax ${_lket})
    endif()
    foreach(_lbra RANGE 0 ${_lbraMax})
        qcx_add_md_class_object(${_lbra} ${_lket})
        list(APPEND _qcxMdClassObjects $<TARGET_OBJECTS:qcx-md-class-${_lbra}-${_lket}>)
    endforeach()
endforeach()
target_sources(qcx-integrals PRIVATE ${_qcxMdClassObjects})

if (QCX_BUILD_TESTS AND TARGET gtest)
    add_executable(qcx-integrals-tests
        tests/batch_partition_test.cpp
        tests/boys_test.cpp
        tests/boys_accuracy_test.cpp
        tests/boys_boundary_test.cpp
        tests/f16_test.cpp
        tests/aux_basis_test.cpp
        tests/one_electron_test.cpp
        tests/two_electron_test.cpp
        tests/md_tables_test.cpp
        tests/md_scatter_table_test.cpp
        tests/shell_pairs_test.cpp
        tests/symmetry_reduction_test.cpp
        tests/symmetry_parity_refusal_test.cpp
        tests/screening_test.cpp
        tests/eri_batch_test.cpp
        tests/eri_cache_test.cpp
        tests/fock_build_test.cpp
        tests/lean_direct_fock_builder_test.cpp
        tests/lean_point_group_test.cpp
        tests/four_cell_attribution_test.cpp
        tests/footprint_test.cpp
        tests/device_footprint_test.cpp
        tests/incremental_fock_test.cpp
        tests/ri_engine_test.cpp
        tests/ri_engine_chunk_test.cpp
        tests/ri_orbit_test.cpp
        tests/ri_occ_k_test.cpp
        tests/ri_full_fock_test.cpp
        tests/ri_aux_screen_test.cpp
        tests/ri_aux_screen_benzene_test.cpp
        tests/ri_jk_validation_test.cpp
        tests/certified_bound_sweep_test.cpp
        tests/certified_budget_test.cpp
        tests/certified_lane_device_default_test.cpp
        tests/precision_ladder_test.cpp
        tests/precision_policy_test.cpp
        tests/simd_micro_gemm_test.cpp
        tests/qfmm_geometry_test.cpp
        tests/qfmm_tree_test.cpp
        tests/qfmm_interaction_lists_test.cpp
        tests/qfmm_monopole_test.cpp
        tests/qfmm_lmult_test.cpp
        tests/qfmm_fock_build_test.cpp
        tests/qfmm_hf_build_test.cpp)
    target_link_libraries(qcx-integrals-tests PRIVATE qcx-integrals qcx-linalg gtest gtest_main qcx-test-fixtures)
    # qcx-linalg is PRIVATE on qcx-integrals but its headers are required
    # at compile time here: screening_test.cpp reaches md_transform.hpp's
    # qcx/linalg/batched_ops.hpp through internal/fock_screen.hpp (the same
    # seam the CUDA-md target carries; the same PRIVATE link).
    # The unfolded reference path (tests/unfolded_reference.hpp) consumes
    # the internal headers - test-only access. ${_qcxBoysExt}/src resolves
    # the retained boys tests' direct includes of the submodule's internal
    # kernel headers (boys_coefficients.hpp / boys_impl.hpp /
    # boys_effective_degrees.hpp), and QcxBoysDataDir points at the
    # submodule's committed reference grid (the CSV lives in the submodule now).
    target_include_directories(qcx-integrals-tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${_qcxBoysExt}/src)
    target_compile_definitions(qcx-integrals-tests PRIVATE
        QcxBoysDataDir="${_qcxBoysExt}/tests/data"
        QcxIntegralsDataDir="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")

    qcx_apply_asan_runtime(qcx-integrals-tests)
    qcx_apply_blas_runtime(qcx-integrals-tests)
    qcx_apply_fast_test_mode(qcx-integrals-tests)

    target_precompile_headers(qcx-integrals-tests PRIVATE
        <gtest/gtest.h>
        <qcx/integrals/boys.hpp>)

    include(GoogleTest)
    qcx_blas_is_mkl(_qcxMkl)
    if (_qcxMkl)
        # The MKL threading layer must be sequential when the backend team
        # threads call the BLAS seam (the default Intel layer deadlocks the
        # RI metric runs). The default (POST_TEST) discovery mode is kept:
        # the PRE_TEST mode re-runs each test inside the discovery context,
        # which hung the RI tests (2026-08-18).
        gtest_discover_tests(qcx-integrals-tests
            PROPERTIES ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL"
            TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    else()
        gtest_discover_tests(qcx-integrals-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()

    # THE LARGE CATEGORY (the root CMakeLists carries
    # the mechanism and the reasoning). These cases leave the routine
    # selection; the routine run is `ctest -LE large`, and they run with
    # `ctest -L large -j`.
    #
    # THIS IS A MANIFEST, NOT A DIRECTORY MOVE, and that is the second-best
    # answer the mechanism allows, so it is stated plainly rather than implied.
    # Every case below sits in a file that also carries fast cases, so a case
    # cannot leave without splitting its file first:
    #
    #   file (lines)                tests   long cases
    #   qfmm_fock_build_test.cpp     18        5
    #   qfmm_hf_build_test.cpp        6        1
    #   qfmm_lmult_test.cpp          14        1
    #   footprint_test.cpp           55        3
    #   ri_occ_k_test.cpp            11        1
    #   four_cell_attribution_test   2        1
    #
    # The split is the follow-up; naming them here is what takes the Release
    # leg from 3.59 h to minutes today, and it costs no test source a line.
    # Three of those files are under concurrent edit (ri_occ_k_test.cpp,
    # four_cell_attribution_test.cpp, and the driver pin sources), so a
    # manifest is the only route that does not collide.
    #
    # Every name is ASSERTED to exist at ctest time (qcx_mark_large_cases), so
    # a rename fails the run loudly instead of silently returning a long test
    # to the routine selection - the failure mode this whole category exists
    # to remove.
    #
    # Times are the GCC Release limb of build/gcc-release-verify.log
    # (2026-09-18, one machine, agent-run): C24-equivalent rows below are the
    # ones that made this necessary.
    qcx_mark_large_cases(integrals
        FourCellAttributionTest.ConvergedEnergiesAttributeTheDivergence        # 1814.77 s
        QfmmLMultTest.FarFieldErrorDecreasesWithLMultAndTighterTheta           #  567.47 s
        RiOccKTest.BatchedTraversalCostAgainstTheChainOnACacheRelevantSlab     #  549.93 s
        QfmmFockBuildTest.LeafDrivenNearFieldIsBitIdenticalToTheGatedSweep     #  448.26 s
        QfmmFockBuildTest.ChunkedFarFieldBuildMatchesTheSerialPinWithinBudget  #  410.43 s
        QfmmFockBuildTest.FarFieldErrorDecreasesMonotonicallyInThetaAndLMult   #  283.47 s
        QfmmHfBuildTest.PresetLadderStaysInsideTheCommittedBudgetsAtEqualPreset #  152.66 s
        QfmmFockBuildTest.PresetLadderMatchesTheRecordedP3Rungs                #  114.44 s
        QfmmFockBuildTest.TheCorrectedDefaultStaysInsideThePresetBudgetsOnC12  #   67.06 s
        QfmmFockBuildTest.ChunkedBuildsAreDeterministicWithinTheMergeScatterBand # 62.90 s
        Footprint.LightPathCountedAdmissionAtScaleSelectsTheFastRung           #  Debug-slow
        Footprint.ExchangePerCallTermRidesTheBatchClamp                        #  Debug-slow
        Footprint.RiFirstIterationFiresTheLightRung)                           #  Debug-slow

    # The byte-identity regeneration checks (the elements-regeneration
    # precedent): the committed tables/CSVs must match a fresh
    # generation. Local-only (QCX_REFERENCE_REGENERATION_CHECK, default OFF):
    # no CI configuration ever registers them (mpmath + minutes per job).
    # Both scripts exit 77 (SKIP) when mpmath is missing. The check must run
    # at the full l (no --max-l): the committed CSVs carry the high-l rows.
    if (QCX_REFERENCE_REGENERATION_CHECK)
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        add_test(NAME md-tables-regeneration
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/gen_md_tables.py --check)
        add_test(NAME md-reference-regeneration
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/gen_md_reference.py --check)
        # The Boys kernel's own generator: the committed
        # coefficients header + reference CSV must match a fresh generation.
        # The fp16 lane reuses the same tables (fp32 engine), so this check
        # covers the half lanes as well. Both files live in the boys
        # submodule now (the submodule's committed files are pinned and never
        # regenerated into this tree), so the check drives the submodule's generator with explicit
        # paths.
        add_test(NAME boys-reference-regeneration
                 COMMAND ${Python3_EXECUTABLE}
                     ${_qcxBoysExt}/tools/gen_boys_coefficients.py --check
                     --header ${_qcxBoysExt}/src/boys_coefficients.hpp
                     --reference ${_qcxBoysExt}/tests/data/boys_reference.csv)
        # The H2/STO-3G s-only grid has its own generator (the older
        # gen_integrals_reference.py) - without its own check it silently
        # went stale against the unit-norm contraction renormalization
        # (2026-08-22: the 4 borderline 1e/2e failures at ~1.1e-10).
        add_test(NAME h2-reference-regeneration
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/gen_integrals_reference.py --check)
        # The QFMM translation tables' own generator (following the
        # regeneration-check precedent, e8fdd81): the committed qfmm_tables_gen.hpp must match a fresh
        # generation. The script's verification checks (V1-V7; 40556 at the
        # committed L=8 cap) run inside --check before the byte compare; a
        # verification failure
        # exits 1 (the script never emits on a failed verify).
        add_test(NAME qfmm-tables-regeneration
                 COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/gen_qfmm_translation_tables.py --check)
        set_tests_properties(md-tables-regeneration md-reference-regeneration
            boys-reference-regeneration h2-reference-regeneration
            qfmm-tables-regeneration PROPERTIES
            SKIP_RETURN_CODE 77)
    endif()
endif()

if (QCX_ENABLE_CUDA)
    # The CUDA lane is the submodule's own (boys_cuda.cpp + boys_cuda.cu,
    # boys::BoysStatus semantics - the retired qcx::Result /
    # DeviceBuffer host layer is gone). It is self-contained: no qcx-core /
    # qcx-backend / qcx-memory links, raw device pointers + void* streams.
    add_library(qcx-integrals-cuda
        ${_qcxBoysExt}/src/boys_cuda.cpp
        ${_qcxBoysExt}/src/boys_cuda.cu)
    target_include_directories(qcx-integrals-cuda PUBLIC
        ${_qcxBoysExt}/include)
    target_include_directories(qcx-integrals-cuda PRIVATE
        ${_qcxBoysExt}/src)
    target_link_libraries(qcx-integrals-cuda
        PUBLIC qcx-integrals)
    # CUDA-safe list only: qcx C++23 headers would poison the nvcc TUs.
    target_precompile_headers(qcx-integrals-cuda PRIVATE
        <cuda_runtime.h>)

    if (QCX_BUILD_TESTS AND TARGET gtest)
        # tests/boys_cuda_test.cpp is the submodule's canonical test verbatim
        # (boys namespace, BoysStatus) - single source of truth.
        add_executable(qcx-integrals-cuda-tests tests/boys_cuda_test.cpp)
        target_link_libraries(qcx-integrals-cuda-tests PRIVATE qcx-integrals-cuda gtest gtest_main)

        qcx_apply_asan_runtime(qcx-integrals-cuda-tests)

        target_precompile_headers(qcx-integrals-cuda-tests PRIVATE
            <gtest/gtest.h>
            <qcx/integrals/boys.hpp>)

        include(GoogleTest)
        gtest_discover_tests(qcx-integrals-cuda-tests PROPERTIES TIMEOUT ${QCX_TEST_TIMEOUT_SECONDS})
    endif()

    # ----------------------------------------------------------------------
    # The CUDA MD lane: eri_cuda.cpp (host
    # orchestration) + eri_cuda.cu (the registry TU) + one OBJECT library
    # per {L_bra, L_ket} class from eri_cuda_instantiate.cu.in. The
    # instantiation matrix is the 2e triangle (lBra <= lKet <= 2*Lmax) UNION
    # the RI rectangle (lKet <= Lmax, lBra <= 2*Lmax) - 154 classes at
    # Lmax=6, 24 at Lmax=2 (CI is kept at Lmax=2).
    # ----------------------------------------------------------------------
    set(_qcxCudaLmax "${QCX_INTEGRALS_CUDA_LMAX}")
    if (_qcxCudaLmax STREQUAL "")
        set(_qcxCudaLmax "${QCX_INTEGRALS_LMAX}")
    endif()
    if (NOT _qcxCudaLmax MATCHES "^[0-6]$")
        message(FATAL_ERROR "QCX_INTEGRALS_CUDA_LMAX must be an integer in 0..6, got '${_qcxCudaLmax}'")
    endif()
    math(EXPR _qcxCudaLkMax "2 * ${_qcxCudaLmax}")

    # The generated dispatch table (the CPU lane's md_dispatch_gen.hpp
    # pattern): extern "C" launcher declarations plus the class table whose
    # entries reference the launchers directly, so the per-class OBJECT
    # libraries stay in the link - a runtime registrar would be dropped by
    # the static-archive linker (nothing references the launchers).
    set(_qcxCudaDispatchDir "${CMAKE_CURRENT_BINARY_DIR}/md_cuda_instantiate")
    file(MAKE_DIRECTORY "${_qcxCudaDispatchDir}")
    set(_qcxCudaDispatch "${_qcxCudaDispatchDir}/eri_cuda_dispatch_gen.hpp")
    file(WRITE "${_qcxCudaDispatch}"
        "// Generated by integrals/CMakeLists.txt - do not edit.\n"
        "#pragma once\n"
        "// Part 1 of eri_cuda.cu supplies the types; the guard keeps this\n"
        "// header includable from the registry TU (eri_cuda.cu Part 4).\n"
        "#ifndef QCX_ERI_CUDA_HEADER_ONLY\n"
        "#define QCX_ERI_CUDA_HEADER_ONLY\n"
        "#endif\n"
        "#include \"eri_cuda.cu\"\n\n")
    foreach(_qcxLk RANGE 0 ${_qcxCudaLkMax})
        if (_qcxLk GREATER ${_qcxCudaLmax})
            set(_qcxLbMax "${_qcxLk}")
        else()
            set(_qcxLbMax "${_qcxCudaLkMax}")
        endif()
        foreach(_qcxLb RANGE 0 ${_qcxLbMax})
            math(EXPR _qcxL "${_qcxLb} + ${_qcxLk}")
            if (_qcxL LESS 5)
                set(_qcxFamily 0)
            elseif (_qcxL LESS 9)
                set(_qcxFamily 1)
            else()
                set(_qcxFamily 2)
            endif()
            if (_qcxFamily EQUAL 2)
                file(APPEND "${_qcxCudaDispatch}"
                    "extern \"C\" int QcxEriCudaLaunchVrr_${_qcxLb}_${_qcxLk}(qcx::integrals::internal::cuda::EriCudaClassArgs& args);\n"
                    "extern \"C\" int QcxEriCudaLaunchBound_${_qcxLb}_${_qcxLk}(qcx::integrals::internal::cuda::EriCudaClassArgs& args);\n")
            else()
                file(APPEND "${_qcxCudaDispatch}"
                    "extern \"C\" int QcxEriCudaLaunchFused_${_qcxLb}_${_qcxLk}(qcx::integrals::internal::cuda::EriCudaClassArgs& args);\n")
            endif()
        endforeach()
    endforeach()
    file(APPEND "${_qcxCudaDispatch}"
        "\nnamespace qcx::integrals::internal::cuda {\n"
        "inline const EriCudaClassDispatchEntry gClassTable[kEriCudaMaxClassL + 1][kEriCudaMaxClassL + 1] = {\n")
    # Row-major [lBra][lKet]: the outer loop is lBra, the inner lKet - the
    # layout QcxEriCudaFindClass indexes (gClassTable[lBra][lKet]). The old
    # lKet-outer order filled a transposed table, so every off-diagonal
    # class dispatched its (lKet, lBra) kernel: class (0,1) ran
    # EriFusedKernel<1,0>, which reads the ket transform with kHermKet = 1
    # (all-zero rows for the p-aux) - the p-aux zeros root cause (2026-08-22).
    foreach(_qcxLb RANGE 0 12)
        file(APPEND "${_qcxCudaDispatch}" "    {\n")
        foreach(_qcxLk RANGE 0 12)
            set(_qcxCudaInst FALSE)
            if (_qcxLk LESS ${_qcxCudaLkMax} OR _qcxLk EQUAL ${_qcxCudaLkMax})
                if (_qcxLk GREATER ${_qcxCudaLmax})
                    if (_qcxLb LESS ${_qcxLk} OR _qcxLb EQUAL ${_qcxLk})
                        set(_qcxCudaInst TRUE)
                    endif()
                elseif (_qcxLb LESS ${_qcxCudaLkMax} OR _qcxLb EQUAL ${_qcxCudaLkMax})
                    set(_qcxCudaInst TRUE)
                endif()
            endif()
            if (_qcxCudaInst)
                math(EXPR _qcxL "${_qcxLb} + ${_qcxLk}")
                if (_qcxL LESS 5)
                    file(APPEND "${_qcxCudaDispatch}" "        {0, &QcxEriCudaLaunchFused_${_qcxLb}_${_qcxLk}, nullptr, nullptr}")
                elseif (_qcxL LESS 9)
                    file(APPEND "${_qcxCudaDispatch}" "        {1, &QcxEriCudaLaunchFused_${_qcxLb}_${_qcxLk}, nullptr, nullptr}")
                else()
                    file(APPEND "${_qcxCudaDispatch}" "        {2, nullptr, &QcxEriCudaLaunchVrr_${_qcxLb}_${_qcxLk}, &QcxEriCudaLaunchBound_${_qcxLb}_${_qcxLk}}")
                endif()
            else()
                file(APPEND "${_qcxCudaDispatch}" "        {0, nullptr, nullptr, nullptr}")
            endif()
            if (_qcxLk LESS 12)
                file(APPEND "${_qcxCudaDispatch}" ",\n")
            else()
                file(APPEND "${_qcxCudaDispatch}" "\n")
            endif()
        endforeach()
        file(APPEND "${_qcxCudaDispatch}" "    },\n")
    endforeach()
    file(APPEND "${_qcxCudaDispatch}" "};\n\n} // namespace qcx::integrals::internal::cuda\n")

    # One OBJECT library per class: parallel + incremental nvcc compilation
    # (the CPU lane's qcx_add_md_class_object pattern).
    function(qcx_add_md_cuda_class_object _qcxLb _qcxLk _qcxFamily)
        set(_qcxGenDir ${CMAKE_CURRENT_BINARY_DIR}/md_cuda_instantiate)
        set(_qcxTu ${_qcxGenDir}/eri_cuda_instantiate_${_qcxLb}_${_qcxLk}.cu)
        # @ONLY substitutes CMake variables: the template's @lBra@/@lKet@/
        # @family@ placeholders must match these exact variable names.
        set(lBra ${_qcxLb})
        set(lKet ${_qcxLk})
        set(family ${_qcxFamily})
        configure_file(src/eri_cuda_instantiate.cu.in ${_qcxTu} @ONLY)
        add_library(qcx-md-cuda-class-${_qcxLb}-${_qcxLk} OBJECT ${_qcxTu})
        target_include_directories(qcx-md-cuda-class-${_qcxLb}-${_qcxLk} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_BINARY_DIR})
        target_compile_definitions(qcx-md-cuda-class-${_qcxLb}-${_qcxLk} PRIVATE
            QcxIntegralsF32=$<BOOL:${QCX_INTEGRALS_F32}>
            QcxIntegralsCudaLMax=${_qcxCudaLmax})
        target_compile_features(qcx-md-cuda-class-${_qcxLb}-${_qcxLk} PRIVATE cxx_std_20)
        # CUDA-safe PCH only.
        target_precompile_headers(qcx-md-cuda-class-${_qcxLb}-${_qcxLk} PRIVATE
            <cuda_runtime.h>)
    endfunction()

    set(_qcxCudaMdObjects "")
    foreach(_qcxLk RANGE 0 ${_qcxCudaLmax})
        foreach(_qcxLb RANGE 0 ${_qcxCudaLkMax})
            math(EXPR _qcxL "${_qcxLb} + ${_qcxLk}")
            if (_qcxL LESS 5)
                set(_qcxFamily 0)
            elseif (_qcxL LESS 9)
                set(_qcxFamily 1)
            else()
                set(_qcxFamily 2)
            endif()
            qcx_add_md_cuda_class_object(${_qcxLb} ${_qcxLk} ${_qcxFamily})
            list(APPEND _qcxCudaMdObjects $<TARGET_OBJECTS:qcx-md-cuda-class-${_qcxLb}-${_qcxLk}>)
        endforeach()
    endforeach()
    math(EXPR _qcxCudaLkStart "${_qcxCudaLmax} + 1")
    if (_qcxCudaLkStart LESS ${_qcxCudaLkMax} OR _qcxCudaLkStart EQUAL ${_qcxCudaLkMax})
        foreach(_qcxLk RANGE ${_qcxCudaLkStart} ${_qcxCudaLkMax})
            foreach(_qcxLb RANGE 0 ${_qcxLk})
                math(EXPR _qcxL "${_qcxLb} + ${_qcxLk}")
                if (_qcxL LESS 5)
                    set(_qcxFamily 0)
                elseif (_qcxL LESS 9)
                    set(_qcxFamily 1)
                else()
                    set(_qcxFamily 2)
                endif()
                qcx_add_md_cuda_class_object(${_qcxLb} ${_qcxLk} ${_qcxFamily})
                list(APPEND _qcxCudaMdObjects $<TARGET_OBJECTS:qcx-md-cuda-class-${_qcxLb}-${_qcxLk}>)
            endforeach()
        endforeach()
    endif()

    # The RI-J cuBLAS dispatch: ri_engine.cpp routes the two
    # n2 x nAux RI contractions through cuBLAS when
    # RiEngineOptions::useCudaGemm is set. QcxHasCuda is the
    # integrals-side mirror of linalg's QcxHasVendorBlas seam - PUBLIC so
    # the CPU test targets inherit both the define and the cublas link.
    target_link_libraries(qcx-integrals PUBLIC CUDA::cublas)
    target_compile_definitions(qcx-integrals PUBLIC QcxHasCuda=1)

    add_library(qcx-integrals-cuda-md src/eri_cuda.cpp src/eri_cuda.cu
        src/eri_cuda_fock.cu)
    target_sources(qcx-integrals-cuda-md PRIVATE ${_qcxCudaMdObjects})
    target_include_directories(qcx-integrals-cuda-md PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_include_directories(qcx-integrals-cuda-md PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${_qcxBoysExt}/src
        ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_features(qcx-integrals-cuda-md PUBLIC cxx_std_23)
    target_link_libraries(qcx-integrals-cuda-md
        PUBLIC qcx-integrals qcx-core qcx-molecule qcx-basisset qcx-memory
        PRIVATE qcx-linalg qcx-backend-cuda CUDA::cublas)
    # qcx-linalg is PRIVATE but required at compile time: the eri_cuda.cpp
    # TU reaches md_transform.hpp's qcx/linalg/batched_ops.hpp through the
    # md_vrr_3c.hpp / md_engine.hpp chain (the class-object libraries at
    # line 117 carry the same PRIVATE linalg link - the same implementation seam).
    target_compile_definitions(qcx-integrals-cuda-md PUBLIC
        QcxIntegralsF32=$<BOOL:${QCX_INTEGRALS_F32}>
        QcxIntegralsCudaLMax=${_qcxCudaLmax})
    # CUDA-safe PCH only.
    target_precompile_headers(qcx-integrals-cuda-md PRIVATE
        <cuda_runtime.h>)
endif()
