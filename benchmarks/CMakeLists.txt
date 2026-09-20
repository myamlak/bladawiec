# Scaling benchmarks ("benchmarkable from day one").
# Local-only: QCX_BUILD_BENCHMARKS is ON in the windows-msvc preset and
# never in CI - the free-tier minutes budget is for correctness, not perf
# runs. Every target that links the BLAS seam applies the MKL runtime
# deployment (qcx_apply_blas_runtime): without it the benchmark exe
# resolves MKL's dynamically-loaded DLLs from the system and dies with an
# unhandled 0xC06D007E raise (2026-08-18 incident).

add_executable(qcx-bench-molecule molecule_pipeline_benchmark.cpp)
target_include_directories(qcx-bench-molecule PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures)
target_compile_features(qcx-bench-molecule PRIVATE cxx_std_23)
target_link_libraries(qcx-bench-molecule PRIVATE qcx-test-fixtures benchmark::benchmark)
target_precompile_headers(qcx-bench-molecule PRIVATE
    <benchmark/benchmark.h>
    <qcx/molecule/molecule.hpp>)
qcx_apply_blas_runtime(qcx-bench-molecule)

# Boys kernel: throughput + the molecular x-distribution workload.
# The benchmark partitions inputs with the
# exact region boundaries, so it includes the generated detail header.
add_executable(qcx-bench-boys boys_benchmark.cpp)
# The Boys kernel internals live in the boys submodule now (the
# table constants the driver reads come from its src/).
target_include_directories(qcx-bench-boys PRIVATE ${CMAKE_SOURCE_DIR}/external/boys/src)
target_compile_features(qcx-bench-boys PRIVATE cxx_std_23)
target_link_libraries(qcx-bench-boys PRIVATE qcx-integrals benchmark::benchmark)
target_precompile_headers(qcx-bench-boys PRIVATE
    <benchmark/benchmark.h>
    <qcx/integrals/boys.hpp>)
qcx_apply_blas_runtime(qcx-bench-boys)

# The unsorted-SIMD lane of the Boys kernel
# - the mixed per-vector kernel over all three region paths, measured on the
# unsorted stream the engine actually faces (the sorted lanes live in
# qcx-bench-boys). Plain main() with --self-check; the
# AVX2 intrinsic TU mirrors the boys_simd.cpp compile pattern (PCH skipped).
if (QCX_ARCH_X86_64)
    add_executable(qcx-bench-boys-unsorted-simd boys_unsorted_simd_benchmark.cpp)
    target_include_directories(qcx-bench-boys-unsorted-simd PRIVATE
        ${CMAKE_SOURCE_DIR}/external/boys/src)
    target_compile_features(qcx-bench-boys-unsorted-simd PRIVATE cxx_std_23)
    target_link_libraries(qcx-bench-boys-unsorted-simd PRIVATE qcx-integrals)
    target_precompile_headers(qcx-bench-boys-unsorted-simd PRIVATE
        <qcx/integrals/boys.hpp>)
    if (MSVC)
        set_source_files_properties(boys_unsorted_simd_benchmark.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2")
    else()
        set_source_files_properties(boys_unsorted_simd_benchmark.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
    set_source_files_properties(boys_unsorted_simd_benchmark.cpp PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON)
    qcx_apply_blas_runtime(qcx-bench-boys-unsorted-simd)
else()
    # The lane measures the mixed per-vector kernel, so there is no portable
    # version of it to build: the target is reported and skipped rather than
    # compiled into something that would time a different kernel.
    message(STATUS "qcx-bench-boys-unsorted-simd: non-x86-64 target - its AVX2 lane is not built")
endif()

# The matched-accuracy throughput comparator: the qcx BoysBatch ladder vs
# libint2's FmEval_Chebyshev7<double> ladder on the identical benchmark input
# streams. The comparator set was resolved to libint2
# because libcint v6.1.3 exposes no public F_m
# surface, while libint2 v2.13.1 = 5fb07b48 does - header-only, in-header
# ~epsilon accuracy contract). LOCAL-ONLY, gated on QCX_BENCH_LIBINT2_DIR (a
# cache variable pointing at the fetched libint-2.13.1 consumer-artifact
# include tree) so CI never sees libint2 and the timed runs stay local. The
# libint2 Boys headers are LGPL-3.0-or-later: benchmark-time-only
# consumption, nothing redistributed. The TU compiles with /arch:AVX2 to
# activate libint2's 4-wide Chebyshev path (__AVX__ gate in boys.h; the
# scalar fallback would time a different kernel) - the same PCH skip as
# the AVX2 TU above, and the same single-TU pattern.
# QCX_ARCH_X86_64 joins the cache-variable gate: the flag is what activates
# libint2's vector path, so a target without the feature has none to compare.
if (QCX_BENCH_LIBINT2_DIR AND QCX_ARCH_X86_64)
    add_executable(qcx-bench-boys-libint2 boys_libint2_benchmark.cpp)
    target_include_directories(qcx-bench-boys-libint2 PRIVATE
        ${QCX_BENCH_LIBINT2_DIR}/include)
    target_compile_features(qcx-bench-boys-libint2 PRIVATE cxx_std_23)
    target_link_libraries(qcx-bench-boys-libint2 PRIVATE qcx-integrals)
    if (MSVC)
        set_source_files_properties(boys_libint2_benchmark.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2;/D_USE_MATH_DEFINES")
    else()
        set_source_files_properties(boys_libint2_benchmark.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mfma;-D_USE_MATH_DEFINES")
    endif()
    set_source_files_properties(boys_libint2_benchmark.cpp PROPERTIES
        SKIP_PRECOMPILE_HEADERS ON)
    qcx_apply_blas_runtime(qcx-bench-boys-libint2)
endif()

# MD engine scaling benchmarks: the
# (6,6)-class ERI throughput, the direct-vs-dense Fock build, and the RI-J
# vs direct Fock build.
add_executable(qcx-bench-md-eri md_eri_benchmark.cpp)
target_include_directories(qcx-bench-md-eri PRIVATE
    ${CMAKE_SOURCE_DIR}/integrals/src
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-md-eri PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-md-eri PRIVATE NOMINMAX)
target_link_libraries(qcx-bench-md-eri PRIVATE
    qcx-integrals qcx-test-fixtures benchmark::benchmark)
target_precompile_headers(qcx-bench-md-eri PRIVATE
    <benchmark/benchmark.h>
    <qcx/integrals/eri_batch.hpp>)
qcx_apply_blas_runtime(qcx-bench-md-eri)

# The pair-data path: the
# before/after record of the per-pair table construction and the (6,6)
# E-table fold that used to heap-allocate per primitive pair (PerAxisETable
# vectors, FoldPairETable's e3d/mid) and now runs on fixed-size std::array
# storage.
add_executable(qcx-bench-md-pair-data md_pair_data_benchmark.cpp)
target_include_directories(qcx-bench-md-pair-data PRIVATE
    ${CMAKE_SOURCE_DIR}/integrals/src)
target_compile_features(qcx-bench-md-pair-data PRIVATE cxx_std_23)
target_link_libraries(qcx-bench-md-pair-data PRIVATE
    qcx-integrals benchmark::benchmark)
target_precompile_headers(qcx-bench-md-pair-data PRIVATE
    <benchmark/benchmark.h>
    <qcx/integrals/shell_pairs.hpp>)
qcx_apply_blas_runtime(qcx-bench-md-pair-data)

add_executable(qcx-bench-md-fock md_fock_benchmark.cpp)
target_include_directories(qcx-bench-md-fock PRIVATE
    ${CMAKE_SOURCE_DIR}/integrals/src
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-md-fock PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-md-fock PRIVATE
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-md-fock PRIVATE qcx-integrals qcx-test-fixtures benchmark::benchmark)
target_precompile_headers(qcx-bench-md-fock PRIVATE
    <benchmark/benchmark.h>
    <qcx/integrals/fock_build.hpp>)
qcx_apply_blas_runtime(qcx-bench-md-fock)

# The survived-quartet recurrence of the direct Fock path across a real SCF
# run - per-iteration screened counts, cumulative-union hit rate, the
# ERI/contract wall-time split (the FockBuildStats seam), and the exact
# value-store footprint of the union, read against the host-RAM probe.
# Plain main() program, not a google-benchmark run.
add_executable(qcx-bench-direct-recurrence direct_recurrence_harness.cpp)
target_include_directories(qcx-bench-direct-recurrence PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-direct-recurrence PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-direct-recurrence PRIVATE NOMINMAX)
target_link_libraries(qcx-bench-direct-recurrence PRIVATE
    qcx-integrals qcx-scf qcx-test-fixtures)
target_precompile_headers(qcx-bench-direct-recurrence PRIVATE
    <qcx/integrals/fock_build.hpp>
    <qcx/scf/rhf.hpp>
    <Eigen/Dense>)
qcx_apply_blas_runtime(qcx-bench-direct-recurrence)

# The profiler-attachment harness: a
# representative production-sized run (C80H162 direct Fock, C20H42 RI-J)
# to attach a real profiler to before picking any optimization candidate.
# Plain main() program, not a google-benchmark run.
add_executable(qcx-bench-profile profile_harness.cpp)
target_include_directories(qcx-bench-profile PRIVATE
    ${CMAKE_SOURCE_DIR}/integrals/src
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-profile PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-profile PRIVATE
    NOMINMAX
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-profile PRIVATE
    qcx-integrals qcx-test-fixtures)
target_precompile_headers(qcx-bench-profile PRIVATE
    <qcx/integrals/fock_build.hpp>
    <Eigen/Dense>)
qcx_apply_blas_runtime(qcx-bench-profile)

# The accuracy-preset sweep: records max
# |J - J_exact| per accuracy preset - the tuning data behind the preset
# mapper constants - plus the acceptance half: the converged SCF total
# energy per preset vs the
# kTight reference (links qcx-scf for the FockBuilderFn seam). Plain main()
# program, not a google-benchmark run.
add_executable(qcx-bench-preset-sweep preset_sweep.cpp)
target_include_directories(qcx-bench-preset-sweep PRIVATE
    ${CMAKE_SOURCE_DIR}/integrals/src
    ${CMAKE_SOURCE_DIR}/linalg/include
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-preset-sweep PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-preset-sweep PRIVATE
    NOMINMAX
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-preset-sweep PRIVATE qcx-integrals qcx-scf qcx-test-fixtures)
target_precompile_headers(qcx-bench-preset-sweep PRIVATE
    <qcx/integrals/fock_build.hpp>
    <Eigen/Dense>)
qcx_apply_blas_runtime(qcx-bench-preset-sweep)

# The preset-ladder recalibration sweep: the QFMM J-energy error of the
# far-field multipole
# truncation across the (tau, theta, L) grid on the C12H26/C24H50
# fixtures (C24H50/def2-SVP behind --def2-svp; C60 is OUT of the pack -
# the 16 GiB cap and the compact-3D far-dead degeneracy), with the
# far-pair liveness printed per row (the
# vacuous-ladder alarm). Measure-only: the recalibrated
# ThetaForPreset/LMultForPreset/QfmmExtentForPreset constants land in
# the follow-up calibration commit. Plain main() program, local-only.
add_executable(qcx-bench-qfmm-ladder-sweep qfmm_ladder_sweep.cpp)
target_include_directories(qcx-bench-qfmm-ladder-sweep PRIVATE
    ${CMAKE_SOURCE_DIR}/integrals/src
    ${CMAKE_SOURCE_DIR}/linalg/include
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-qfmm-ladder-sweep PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-qfmm-ladder-sweep PRIVATE
    NOMINMAX
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-qfmm-ladder-sweep PRIVATE qcx-integrals qcx-test-fixtures)
target_precompile_headers(qcx-bench-qfmm-ladder-sweep PRIVATE
    <qcx/integrals/qfmm_fock_build.hpp>
    <Eigen/Dense>)
qcx_apply_blas_runtime(qcx-bench-qfmm-ladder-sweep)

# The Fock-parallel acceptance benchmark:
# wall-time of the direct Fock build on the certified mixed-precision lane
# (kLoose), with an exact %.17g Fock-matrix dump for the bit-identity
# checks across the two-pass -> merged -> aligned chunking states. Plain
# main() program, not a google-benchmark run.
add_executable(qcx-bench-fock-parallel fock_parallel_benchmark.cpp)
target_include_directories(qcx-bench-fock-parallel PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-fock-parallel PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-fock-parallel PRIVATE
    NOMINMAX
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-fock-parallel PRIVATE qcx-integrals qcx-test-fixtures)
target_precompile_headers(qcx-bench-fock-parallel PRIVATE
    <qcx/integrals/fock_build.hpp>
    <Eigen/Dense>)
qcx_apply_blas_runtime(qcx-bench-fock-parallel)

# The disk-store read-vs-recompute crossover measurement:
# per class of the disk-integral surface, wall time of serving the
# whole class from the store vs recomputing it through the underlying
# engine. Gated on QCX_ENABLE_IO (the storage module it measures) like the
# storage targets themselves.
if (QCX_ENABLE_IO)
    add_executable(qcx-bench-io-crossover io_crossover_benchmark.cpp)
    target_include_directories(qcx-bench-io-crossover PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)
    target_compile_features(qcx-bench-io-crossover PRIVATE cxx_std_23)
    target_compile_definitions(qcx-bench-io-crossover PRIVATE
        QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
    target_link_libraries(qcx-bench-io-crossover PRIVATE
        qcx-storage qcx-test-fixtures)
    target_precompile_headers(qcx-bench-io-crossover PRIVATE
        <qcx/integrals/eri_batch.hpp>
        <qcx/storage/cached_eri_batch_engine.hpp>)
    qcx_apply_blas_runtime(qcx-bench-io-crossover)
endif()

# The comparative crossover benchmark - per-iteration BuildFock wall time of
# the direct / RI-J / QFMM builders over ~7 -> ~840 basis functions at every
# preset, the direct-vs-RI-J crossover and the QFMM-vs-RI-J
# crossover tables. Plain main(), not a google-benchmark run; the physics
# guard (H2O/STO-3G kTight pin through the seam) refuses to produce tables
# from a regressed fixture construction. --no-timing runs the guard and
# fixture statistics only (verification mode - no timing
# claims from an unguarded run).
add_executable(qcx-bench-crossover crossover_benchmark.cpp)
target_include_directories(qcx-bench-crossover PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-crossover PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-crossover PRIVATE
    NOMINMAX
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-crossover PRIVATE
    qcx-integrals qcx-scf qcx-test-fixtures)
target_precompile_headers(qcx-bench-crossover PRIVATE
    <qcx/integrals/fock_build.hpp>
    <qcx/integrals/qfmm_fock_build.hpp>
    <qcx/integrals/ri_engine.hpp>
    <Eigen/Dense>)
qcx_apply_blas_runtime(qcx-bench-crossover)

add_executable(qcx-bench-ri-j ri_j_benchmark.cpp)
target_include_directories(qcx-bench-ri-j PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-ri-j PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-ri-j PRIVATE
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-ri-j PRIVATE qcx-integrals qcx-test-fixtures benchmark::benchmark)
target_precompile_headers(qcx-bench-ri-j PRIVATE
    <benchmark/benchmark.h>
    <qcx/integrals/ri_engine.hpp>)
qcx_apply_blas_runtime(qcx-bench-ri-j)

# Wall time of the plain DiagonalizeFock vs the
# symmetry-blocked path at a real-symmetry mid-size molecule (benzene,
# D6h -> computational D2h), plus the one-time block build the
# per-iteration saving amortizes against.
add_executable(qcx-bench-sym-blocked symmetry_blocked_benchmark.cpp)
target_include_directories(qcx-bench-sym-blocked PRIVATE
    ${CMAKE_SOURCE_DIR}/scf/src/internal
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-sym-blocked PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-sym-blocked PRIVATE
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-sym-blocked PRIVATE
    qcx-scf qcx-integrals qcx-symmetry qcx-test-fixtures benchmark::benchmark)
target_precompile_headers(qcx-bench-sym-blocked PRIVATE
    <benchmark/benchmark.h>
    <qcx/molecule/molecule.hpp>)
qcx_apply_blas_runtime(qcx-bench-sym-blocked)

# The SCF robustness harness: the O2/STO-3G seed x accelerator x gate matrix
# that records the
# two-phase DIIS, virtual-space level shift, robust-gate, and
# joint-system DIIS evidence. Plain main() program, not a google-benchmark
# run; rows
# are recorded, not asserted (the gtest pins live in
# scf/tests/robustness_uhf_test.cpp).
add_executable(qcx-bench-scf-robustness scf_robustness_benchmark.cpp)
target_include_directories(qcx-bench-scf-robustness PRIVATE ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-scf-robustness PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-scf-robustness PRIVATE NOMINMAX)
target_link_libraries(qcx-bench-scf-robustness PRIVATE
    qcx-integrals qcx-scf qcx-test-fixtures)
target_precompile_headers(qcx-bench-scf-robustness PRIVATE
    <qcx/integrals/fock_build.hpp>
    <qcx/scf/uhf.hpp>
    <Eigen/Dense>)
qcx_apply_blas_runtime(qcx-bench-scf-robustness)

# CUDA MD lane throughput: per-variant rates
# and the CPU-vs-GPU (6,6) comparison. Local-only like the rest (the Quadro);
# skips without a device.
if (QCX_ENABLE_CUDA AND TARGET qcx-integrals-cuda-md)
    add_executable(qcx-bench-gpu-eri gpu_eri_benchmark.cpp)
    target_compile_features(qcx-bench-gpu-eri PRIVATE cxx_std_23)
    target_compile_definitions(qcx-bench-gpu-eri PRIVATE NOMINMAX)
    target_link_libraries(qcx-bench-gpu-eri PRIVATE
        qcx-integrals-cuda-md qcx-integrals benchmark::benchmark)
    target_precompile_headers(qcx-bench-gpu-eri PRIVATE
        <benchmark/benchmark.h>
        <qcx/integrals/eri_cuda.hpp>)
    qcx_apply_blas_runtime(qcx-bench-gpu-eri)
endif()

# The Boys GPU throughput
# lanes - the certified cheb-f64/cheb-f32/fp16 entries plus the erf-f64 and
# Tsuji-style lut-f64 competitor kernels (boys_cuda_benchmark_kernels.cu,
# nvcc). Plain main() with --self-check; PCH
# <cuda_runtime.h> only - CUDA-safe for the nvcc TU.
if (QCX_ENABLE_CUDA AND TARGET qcx-integrals-cuda)
    add_executable(qcx-bench-boys-cuda
        boys_cuda_benchmark.cpp
        boys_cuda_benchmark_kernels.cu)
    # The driver is the submodule's canonical copy (boys namespace,
    # BoysStatus semantics); the kernels TU is CUDA-clean and unchanged.
    target_include_directories(qcx-bench-boys-cuda PRIVATE
        ${CMAKE_SOURCE_DIR}/external/boys/src)
    target_compile_features(qcx-bench-boys-cuda PRIVATE cxx_std_23)
    target_compile_definitions(qcx-bench-boys-cuda PRIVATE NOMINMAX)
    target_link_libraries(qcx-bench-boys-cuda PRIVATE qcx-integrals-cuda)
    target_precompile_headers(qcx-bench-boys-cuda PRIVATE
        <cuda_runtime.h>)
    qcx_apply_blas_runtime(qcx-bench-boys-cuda)
endif()

# The memory-accounting delta probe:
# no-chemistry legs (plain/base/threads/synthetic/selftest) that isolate the
# components behind each real run's realized aggregate delta - the request-
# to-commit curve over the model's size classes, measured under the per-term
# attribution instrument. Plain main(); never a timed pass (byte counters
# only); the Python runner launches the legs under its cap job.
add_executable(qcx-bench-delta-probe delta_probe.cpp)
target_compile_features(qcx-bench-delta-probe PRIVATE cxx_std_23)
target_link_libraries(qcx-bench-delta-probe PRIVATE qcx-memory)
target_precompile_headers(qcx-bench-delta-probe PRIVATE
    <qcx/memory/allocation_instrument.hpp>)

# The overlap conditioning probe (linear-dependence lane, 2026-09-14): the
# spectral ratio that gates the SCF's linear-dependence removal
# (scf_common.hpp OrthogonalizeOverlap), per case file and basis - s_min,
# s_max, s_min/s_max and the count below the relative floor. A measurement
# instrument, not a timed pass: it exists so the threshold stays checkable.
add_executable(qcx-bench-overlap-condition overlap_condition_probe.cpp)
target_compile_features(qcx-bench-overlap-condition PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-overlap-condition PRIVATE
    NOMINMAX
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-overlap-condition PRIVATE qcx-integrals)
target_precompile_headers(qcx-bench-overlap-condition PRIVATE
    <qcx/integrals/one_electron.hpp>)
qcx_apply_blas_runtime(qcx-bench-overlap-condition)

# The QFMM separation-counts instrument (2026-09-14): tree shape, leaf-size
# and interaction-list distributions and
# the near-field share of the direct CSR, swept over theta and the leaf-size
# cap. Counts ONLY - no wall clock, no SCF.
add_executable(qcx-bench-qfmm-separation-counts qfmm_separation_counts.cpp)
target_include_directories(qcx-bench-qfmm-separation-counts PRIVATE
    ${CMAKE_SOURCE_DIR}/integrals/src
    ${CMAKE_SOURCE_DIR}/linalg/include
    ${CMAKE_SOURCE_DIR}/tests/fixtures)
target_compile_features(qcx-bench-qfmm-separation-counts PRIVATE cxx_std_23)
target_compile_definitions(qcx-bench-qfmm-separation-counts PRIVATE
    NOMINMAX
    QcxBasisDataDir="${CMAKE_SOURCE_DIR}/data/basis")
target_link_libraries(qcx-bench-qfmm-separation-counts PRIVATE qcx-integrals qcx-test-fixtures)
target_precompile_headers(qcx-bench-qfmm-separation-counts PRIVATE
    <qcx/integrals/fock_build.hpp>)
qcx_apply_blas_runtime(qcx-bench-qfmm-separation-counts)
