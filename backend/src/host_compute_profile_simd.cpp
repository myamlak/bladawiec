// The AVX2 measurement lane of the host compute-profile probe:
// the design's register-bound SIMD FMA chains, explicit intrinsics only.
//
// The TU is compiled with /arch:AVX2 and skips the module PCH (clang rejects
// a PCH built without the target feature the TU compiles with) - the same
// pair of CMake properties the integrals module's SIMD TUs carry. It is
// entered only after the runtime CpuHasAvx2Fma() check in
// host_compute_profile.cpp, so no AVX2 instruction here can execute on a
// machine that lacks it.
//
// WHY INTRINSICS AND NEVER `a * b + c`. Two reasons, both load-bearing.
// MSVC at its default /fp:precise does not contract, so a written
// multiply-add would compile to two operations and measure an FMA-less
// machine's arithmetic; and an auto-vectorized loop lets the COMPILER's
// choice of vector width into the measurement, which is the one hazard that
// can push the ratio the wrong way (a scalar fp32 lane against a vectorized
// fp64 one would read a ratio of 4 - a false ON for the certified lane).
// Explicit intrinsics pin both lanes to 8 and 4 lanes.
//
// ON A NON-x86-64 TARGET THIS TU COMPILES NO INTRINSICS AT ALL. The guards
// below read the tree's one architecture test (QcxArchX86_64, defined by
// qcx/backend/cpu_features.hpp), the intrinsics header is not included, and
// MeasureFmaLaneAvx2() returns the scalar lane's reading. The caller's gate -
// CpuHasAvx2Fma(), false on such a target - means the profile never selects
// this lane there, but the function has to exist, and a caller that reaches it
// anyway (a test, say) gets a measurement rather than a fabricated ratio.

#include "host_compute_profile_detail.hpp"
// The architecture test is included before <immintrin.h> because the guard on
// that include reads it.
#include <qcx/backend/cpu_features.hpp>

#if QcxArchX86_64
#include <immintrin.h>
#endif

namespace qcx::backend::internal {

#if QcxArchX86_64

namespace {

// The keep-alive sinks: one guarded store the compiler cannot prove dead
// (the seeds are positive, so the accumulated value never reaches -1.0). The
// device probe keeps its chains alive the same way
// (gpu_compute_profile.cu:55-58); without it the loop is dead code and the
// "measurement" times an empty loop.
volatile float gAvx2Fp32Sink = 0.0F;
volatile double gAvx2Fp64Sink = 0.0;

/// The AVX2 fp32 body: kProbeChains independent 8-lane chains, one
/// _mm256_fmadd_ps each per iteration. Eight chains is not a tuning knob: an
/// FMA latency of 4 cycles against 2 FMA ports is 8 operations in flight, and
/// a single host thread has no warps to cover the gap.
void RunAvx2Fp32(int reps) noexcept {
    __m256 acc0 = _mm256_set1_ps(0.25F);
    __m256 acc1 = _mm256_set1_ps(0.50F);
    __m256 acc2 = _mm256_set1_ps(0.75F);
    __m256 acc3 = _mm256_set1_ps(1.00F);
    __m256 acc4 = _mm256_set1_ps(1.25F);
    __m256 acc5 = _mm256_set1_ps(1.50F);
    __m256 acc6 = _mm256_set1_ps(1.75F);
    __m256 acc7 = _mm256_set1_ps(2.00F);
    const __m256 factor = _mm256_set1_ps(1.000004F);
    const __m256 addend = _mm256_set1_ps(0.5F);

    for (int i = 0; i < reps; ++i)
    {
        acc0 = _mm256_fmadd_ps(acc0, factor, addend);
        acc1 = _mm256_fmadd_ps(acc1, factor, addend);
        acc2 = _mm256_fmadd_ps(acc2, factor, addend);
        acc3 = _mm256_fmadd_ps(acc3, factor, addend);
        acc4 = _mm256_fmadd_ps(acc4, factor, addend);
        acc5 = _mm256_fmadd_ps(acc5, factor, addend);
        acc6 = _mm256_fmadd_ps(acc6, factor, addend);
        acc7 = _mm256_fmadd_ps(acc7, factor, addend);
    }

    const __m256 total =
        _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)),
                      _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7)));

    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, total);
    const float sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                      ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));

    if (sum == -1.0F)
    {
        gAvx2Fp32Sink = sum;
    }
}

/// The AVX2 fp64 body: the same shape on 4-lane doubles. The two lanes issue
/// the SAME number of FMAs per iteration, so the ratio between them is the
/// lane-count ratio (8 against 4) multiplied by the two lanes' issue-rate
/// ratio - which is what makes a register-bound probe read a machine's
/// precision width rather than its clock.
void RunAvx2Fp64(int reps) noexcept {
    __m256d acc0 = _mm256_set1_pd(0.25);
    __m256d acc1 = _mm256_set1_pd(0.50);
    __m256d acc2 = _mm256_set1_pd(0.75);
    __m256d acc3 = _mm256_set1_pd(1.00);
    __m256d acc4 = _mm256_set1_pd(1.25);
    __m256d acc5 = _mm256_set1_pd(1.50);
    __m256d acc6 = _mm256_set1_pd(1.75);
    __m256d acc7 = _mm256_set1_pd(2.00);
    const __m256d factor = _mm256_set1_pd(1.000004);
    const __m256d addend = _mm256_set1_pd(0.5);

    for (int i = 0; i < reps; ++i)
    {
        acc0 = _mm256_fmadd_pd(acc0, factor, addend);
        acc1 = _mm256_fmadd_pd(acc1, factor, addend);
        acc2 = _mm256_fmadd_pd(acc2, factor, addend);
        acc3 = _mm256_fmadd_pd(acc3, factor, addend);
        acc4 = _mm256_fmadd_pd(acc4, factor, addend);
        acc5 = _mm256_fmadd_pd(acc5, factor, addend);
        acc6 = _mm256_fmadd_pd(acc6, factor, addend);
        acc7 = _mm256_fmadd_pd(acc7, factor, addend);
    }

    const __m256d total =
        _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(acc0, acc1), _mm256_add_pd(acc2, acc3)),
                      _mm256_add_pd(_mm256_add_pd(acc4, acc5), _mm256_add_pd(acc6, acc7)));

    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, total);
    const double sum = (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);

    if (sum == -1.0)
    {
        gAvx2Fp64Sink = sum;
    }
}

} // namespace

#endif // QcxArchX86_64

FmaLaneReading MeasureFmaLaneAvx2() noexcept {
#if QcxArchX86_64
    return EstimateFmaLane(8, 4, RunAvx2Fp32, RunAvx2Fp64);
#else
    // No AVX2 lane exists on this target, so there is no 8-lane against 4-lane
    // pair to time. The scalar lane's own reading is the honest answer: it is a
    // measurement of this machine, made by the lane this target does have.
    return MeasureFmaLaneScalar();
#endif
}

} // namespace qcx::backend::internal
