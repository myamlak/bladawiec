#pragma once

// The internal bridge between the host compute-profile probe's two TUs: the
// portable one (host_compute_profile.cpp - the public API, the lane
// selection, and the scalar reference lane) and the AVX2 one
// (host_compute_profile_simd.cpp - the 256-bit FMA lane, compiled with
// /arch:AVX2 and skipping the module PCH, the SIMD-TU pattern the integrals
// module uses).
//
// The split exists for ONE reason. /arch:AVX2 is a whole-translation-unit
// flag, and a scalar lane compiled under it would be auto-vectorized into
// the very thing it is the reference for. The compiler's choice of vector
// width must not become a term in the measurement: an auto-vectorized loop
// can put 4-lane fp32 against 2-lane fp64 (ratio 2, harmless here) or a
// scalar lane against a vectorized one (ratio 4 - a false ON, the one
// wrong-direction hazard of this instrument).

#include <algorithm>
#include <chrono>
#include <cmath>

namespace qcx::backend::internal {

/// One host FMA-lane measurement: both precisions' throughput, the ratio
/// between them, and the noise of that ratio. `ok = false` means no usable
/// reading (the caller falls back to the documented unknown profile); every
/// other member is then meaningless.
struct FmaLaneReading {
    double fp32Gflops = 0.0; ///< fp32 FMA throughput (scalar FLOP/s, 1e9).
    double fp64Gflops = 0.0; ///< fp64 FMA throughput (scalar FLOP/s, 1e9).
    double ratio = 1.0; ///< fp32Gflops / fp64Gflops.
    double ratioSpread = 0.0; ///< The per-pair ratios' spread (max - min).
    int pairs = 0; ///< The timed pairs the estimate came from.
    bool ok = false; ///< False when no usable reading was produced.
};

/// The estimator's shape, shared by both lanes so that the two differ in
/// nothing but their instruction set. Eight independent accumulator chains:
/// the device probe needs two because a GPU hides FMA latency behind
/// thousands of warps, while one host thread has none, and an FMA latency of
/// 4 cycles against 2 FMA ports needs 8 chains to become throughput-bound
/// rather than latency-bound. The FLOP per iteration is 2 (one FMA) times
/// kProbeChains times the LANE WIDTH, and it is counted that way - the same
/// scalar-FLOP convention the device probe's kernel uses - so the ratio is
/// the lane-width ratio and nothing else.
inline constexpr int kProbeChains = 8;
/// The timed lane pairs, interleaved fp32/fp64 so symmetric contention and
/// symmetric clock drift cancel in the ratio.
inline constexpr int kProbePairs = 5;
/// The per-lane span the rep budget is sized to.
inline constexpr double kProbeTargetMs = 5.0;
/// The rep count the calibration probes run at, and the number of probes it
/// takes the minimum of.
inline constexpr int kProbeCalibrationReps = 200000;
inline constexpr int kProbeCalibrationProbes = 3;
/// The rep budget's floor and ceiling: the floor keeps the loop overhead
/// from dominating a fast lane, the ceiling bounds a pathological calibration
/// rather than letting it run away.
inline constexpr int kProbeRepsFloor = 4096;
inline constexpr int kProbeRepsCeiling = 400000000;

/// The estimator itself, shared by both lanes: size each precision's rep
/// budget onto the target span, take one untimed warmup pair, then
/// kProbePairs interleaved timed pairs, and report the ratio of the two
/// lanes' best spans. The minimum is the peak estimator - contention and
/// clock drift can only ever slow a lane down - and the per-pair spread is
/// carried out so a reading's noise is visible rather than implied.
///
/// The core is warmed BEFORE the budgets are sized: a cold calibration reads
/// the ramp-up clock, which under-sizes one lane and lands the two on
/// different spans (measured 2.2 ms against 6.5 ms for one target before this
/// was ordered, on the reference box) - and drift cancels between the two
/// lanes only while they occupy equal spans.
///
/// All work is on the stack: no allocation, no OpenMP team, no memory traffic
/// beyond the lanes' own accumulator reduction.
/// \param fp32Lanes The fp32 lane width (8 under AVX2, 1 scalar).
/// \param fp64Lanes The fp64 lane width (4 under AVX2, 1 scalar).
/// \param runFp32 The fp32 lane body, taking a rep count.
/// \param runFp64 The fp64 lane body, taking a rep count.
/// \returns The reading; `ok = false` when no usable measurement was made.
template <typename RunFp32, typename RunFp64>
FmaLaneReading EstimateFmaLane(int fp32Lanes,
                               int fp64Lanes,
                               RunFp32 runFp32,
                               RunFp64 runFp64) noexcept {
    using Clock = std::chrono::steady_clock;

    const auto timeLane = [](const auto& run, int reps) {
        const auto start = Clock::now();
        run(reps);
        const auto stop = Clock::now();
        return std::chrono::duration<double, std::milli>(stop - start).count();
    };

    const auto calibrate = [&timeLane](const auto& run) {
        double best = 0.0;

        for (int probe = 0; probe < kProbeCalibrationProbes; ++probe)
        {
            const double ms = timeLane(run, kProbeCalibrationReps);

            if (probe == 0 || ms < best)
            {
                best = ms;
            }
        }

        if (!(best > 0.0))
        {
            return 0;
        }

        const double scaled = static_cast<double>(kProbeCalibrationReps) * (kProbeTargetMs / best);

        if (!std::isfinite(scaled))
        {
            return 0;
        }

        // The clamp, not a verdict: a lane whose scaled budget is empty gets
        // the measurement refused rather than a fabricated one.
        int reps = 0;

        if (scaled >= static_cast<double>(kProbeRepsCeiling))
        {
            reps = kProbeRepsCeiling;
        } else if (scaled <= static_cast<double>(kProbeRepsFloor))
        {
            reps = kProbeRepsFloor;
        } else
        {
            reps = static_cast<int>(scaled);
        }

        return reps;
    };

    // Warm every lane body once before any of them is measured: the first
    // call of a freshly entered loop pays for the ramp, and the calibration
    // below must not read it.
    runFp32(kProbeRepsFloor);
    runFp64(kProbeRepsFloor);

    const int reps32 = calibrate(runFp32);
    const int reps64 = calibrate(runFp64);

    FmaLaneReading reading;

    if (reps32 <= 0 || reps64 <= 0)
    {
        return reading;
    }

    // The untimed warmup pair at the final budgets.
    runFp32(reps32);
    runFp64(reps64);

    double min32 = 0.0;
    double min64 = 0.0;
    double minPairRatio = 0.0;
    double maxPairRatio = 0.0;

    for (int pair = 0; pair < kProbePairs; ++pair)
    {
        // Interleaved, fp32 first in every pair: the two lanes share one
        // machine state, so the ordering is fixed rather than alternated.
        const double t32 = timeLane(runFp32, reps32);
        const double t64 = timeLane(runFp64, reps64);

        if (!(t32 > 0.0) || !(t64 > 0.0) || !std::isfinite(t32) || !std::isfinite(t64))
        {
            return reading;
        }

        if (pair == 0 || t32 < min32)
        {
            min32 = t32;
        }

        if (pair == 0 || t64 < min64)
        {
            min64 = t64;
        }

        const double pairRatio =
            (static_cast<double>(fp32Lanes) * static_cast<double>(reps32) * t64) /
            (static_cast<double>(fp64Lanes) * static_cast<double>(reps64) * t32);

        if (pair == 0 || pairRatio < minPairRatio)
        {
            minPairRatio = pairRatio;
        }

        if (pair == 0 || pairRatio > maxPairRatio)
        {
            maxPairRatio = pairRatio;
        }
    }

    // Scalar FLOP per lane: 2 FLOP per FMA, kProbeChains FMAs per iteration,
    // times the lane width, times the reps.
    const double flop32 =
        2.0 * static_cast<double>(kProbeChains) * static_cast<double>(fp32Lanes) * reps32;
    const double flop64 =
        2.0 * static_cast<double>(kProbeChains) * static_cast<double>(fp64Lanes) * reps64;

    reading.fp32Gflops = flop32 / (min32 * 1.0e6);
    reading.fp64Gflops = flop64 / (min64 * 1.0e6);
    reading.ratio = reading.fp32Gflops / reading.fp64Gflops;
    reading.ratioSpread = maxPairRatio - minPairRatio;
    reading.pairs = kProbePairs;

    if (!std::isfinite(reading.fp32Gflops) || !std::isfinite(reading.fp64Gflops) ||
        !std::isfinite(reading.ratio) || !std::isfinite(reading.ratioSpread) ||
        !(reading.fp32Gflops > 0.0) || !(reading.fp64Gflops > 0.0) || !(reading.ratio > 0.0))
    {
        return FmaLaneReading{};
    }

    reading.ok = true;
    return reading;
}

/// The 256-bit FMA lane (AVX2: 8 fp32 lanes, 4 fp64 lanes), compiled in its
/// own /arch:AVX2 TU. Called only when CpuHasAvx2Fma() holds.
FmaLaneReading MeasureFmaLaneAvx2() noexcept;

/// The scalar reference lane: one fp32 and one fp64 FMA chain per step of the
/// same shape. Its ratio reads ~1 on a machine with no wide fp64 path; the
/// `simdLane` flag of the resulting profile is what keeps that ~1 from being
/// mistaken for a measured machine width.
FmaLaneReading MeasureFmaLaneScalar() noexcept;

} // namespace qcx::backend::internal
