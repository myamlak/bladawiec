// The portable layer of the host compute-profile probe: the public
// API, the lane selection, and the scalar reference lane. The 256-bit FMA
// lane lives in its own /arch:AVX2 TU (host_compute_profile_simd.cpp) and
// this TU never names an intrinsic - the split and its reason are documented
// in host_compute_profile_detail.hpp.
//
// Probe-failure-safe, mirroring the device probe (gpu_compute_profile.cpp:15-22):
// a failed lane selection or measurement returns the header's documented
// "unknown" profile and is NEVER an error. The unknown ratio is 1.0, below
// the certified lane's threshold by construction, so a failed probe leaves
// that lane OFF with no second branch to get wrong.

#include "qcx/backend/host_compute_profile.hpp"

#include "host_compute_profile_detail.hpp"
#include "qcx/backend/cpu_features.hpp"

namespace qcx::backend {

namespace {

// The keep-alive sinks: a store behind a guard the compiler cannot prove
// dead, because it cannot know the accumulated value never reaches -1.0 (the
// seeds are positive and the recurrence grows). The device probe keeps its
// chains alive the same way (gpu_compute_profile.cu:55-58); without it the
// whole loop is dead code and the "measurement" times an empty loop.
volatile float gScalarFp32Sink = 0.0F;
volatile double gScalarFp64Sink = 0.0;

/// The scalar fp32 reference body: kProbeChains independent chains, one
/// scalar FMA each per iteration - the same iteration shape as the AVX2 lane,
/// with a lane width of one.
void RunScalarFp32(int reps) noexcept {
    float value0 = 0.25F;
    float value1 = 0.50F;
    float value2 = 0.75F;
    float value3 = 1.00F;
    float value4 = 1.25F;
    float value5 = 1.50F;
    float value6 = 1.75F;
    float value7 = 2.00F;
    const float factor = 1.000004F;
    const float addend = 0.5F;

    for (int i = 0; i < reps; ++i)
    {
        value0 = value0 * factor + addend;
        value1 = value1 * factor + addend;
        value2 = value2 * factor + addend;
        value3 = value3 * factor + addend;
        value4 = value4 * factor + addend;
        value5 = value5 * factor + addend;
        value6 = value6 * factor + addend;
        value7 = value7 * factor + addend;
    }

    const float sum =
        ((value0 + value1) + (value2 + value3)) + ((value4 + value5) + (value6 + value7));

    if (sum == -1.0F)
    {
        gScalarFp32Sink = sum;
    }
}

/// The scalar fp64 reference body: the same shape, one double per chain.
void RunScalarFp64(int reps) noexcept {
    double value0 = 0.25;
    double value1 = 0.50;
    double value2 = 0.75;
    double value3 = 1.00;
    double value4 = 1.25;
    double value5 = 1.50;
    double value6 = 1.75;
    double value7 = 2.00;
    const double factor = 1.000004;
    const double addend = 0.5;

    for (int i = 0; i < reps; ++i)
    {
        value0 = value0 * factor + addend;
        value1 = value1 * factor + addend;
        value2 = value2 * factor + addend;
        value3 = value3 * factor + addend;
        value4 = value4 * factor + addend;
        value5 = value5 * factor + addend;
        value6 = value6 * factor + addend;
        value7 = value7 * factor + addend;
    }

    const double sum =
        ((value0 + value1) + (value2 + value3)) + ((value4 + value5) + (value6 + value7));

    if (sum == -1.0)
    {
        gScalarFp64Sink = sum;
    }
}

} // namespace

namespace internal {

// The scalar reference lane: both precisions one lane wide, so the ratio is
// the two lanes' issue-rate ratio alone. A compiler that auto-vectorizes this
// TU still leaves the reading near 1 (it moves both precisions together), and
// the profile's simdLane flag records that this was the fallback lane rather
// than the SIMD one - so a scalar machine's ~1.0 can never be read as a
// measured machine width.
FmaLaneReading MeasureFmaLaneScalar() noexcept {
    return EstimateFmaLane(1, 1, RunScalarFp32, RunScalarFp64);
}

} // namespace internal

HostComputeProfile DetectHostComputeProfile() noexcept {
    const bool simd = CpuHasAvx2Fma();
    const internal::FmaLaneReading reading =
        simd ? internal::MeasureFmaLaneAvx2() : internal::MeasureFmaLaneScalar();

    HostComputeProfile profile;
    profile.simdLane = simd;

    if (!reading.ok)
    {
        // The documented "unknown" profile - never an error, and below the
        // certified lane's threshold by construction.
        return profile;
    }

    profile.fp32ToFp64Ratio = reading.ratio;
    profile.fp32Gflops = reading.fp32Gflops;
    profile.fp64Gflops = reading.fp64Gflops;
    profile.pairs = reading.pairs;
    profile.ratioSpread = reading.ratioSpread;
    profile.measured = true;
    return profile;
}

} // namespace qcx::backend
