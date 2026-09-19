#pragma once

/// \file
/// The HOST compute-profile probe: the CPU-side counterpart of the device
/// probe (gpu_compute_profile.hpp), and the producer that keeps the certified
/// fp32 lane's default HARDWARE-AWARE - the lane's default follows the
/// measured ratio, never a label.
///
/// WHY A MEASUREMENT AND NOT A LABEL. The discriminator is the measured
/// fp32/fp64 FMA-throughput ratio, never the presence of an "AVX" or "GPU"
/// label: a consumer AVX2 host measures near 2x fp32-over-fp64, where the
/// certified lane's conversion and routing overhead dominates and the lane
/// measured 13-16% SLOWER, while the local Quadro T1000 measures 31.2, where
/// it should pay. Labels have already failed in this tree in exactly this
/// place - the FMA flag was once read from the wrong CPUID leaf, and the whole
/// AVX2 micro-GEMM tier was dead code on a machine that has FMA
/// (cpu_features.hpp:8-19). A measured ratio cannot be wrong that way; it can
/// only be noisy, which is why the reading carries its pair count and spread.

namespace qcx::backend {

/// One host's compute profile. The default is the documented "unknown"
/// profile - a 1:1 throughput ratio, nothing measured - and it is the value
/// every probe failure falls back to; probing is never an error (the
/// GpuComputeProfile contract, mirrored).
///
/// `fp32ToFp64Ratio` is what the precision policy consumes: below
/// `kCertifiedLaneMinRatio` the certified lane's routing and conversion
/// overhead is not covered by fp32's throughput premium, and the lane's
/// default resolves OFF.
///
/// The struct records WHICH LANE produced the reading, not just its value.
/// On a mainstream x86 host the peak FMA ratio is a SIMD lane-width fact (8
/// fp32 lanes against 4 fp64 lanes under AVX2), so the reading is ~2 - and a
/// scalar host, whose two lanes are one lane each, reads ~1. Those two
/// numbers are numerically close but mean different things, and a *measured*
/// 1.0 (a scalar machine, `measured = true`, `simdLane = false`) must never
/// be read as the *unknown* 1.0 of a probe that did not run
/// (`measured = false`). The verdict is OFF either way; the record is not.
/// \ingroup qcx-backend
struct HostComputeProfile {
    /// The measured fp32/fp64 FMA-throughput ratio (fp32Gflops /
    /// fp64Gflops). 1.0 is the "unknown" value: the probe did not run or a
    /// measurement failed - which is also, by construction, below the
    /// certified lane's threshold, so a failed probe leaves the lane OFF
    /// with no second branch to get wrong.
    double fp32ToFp64Ratio = 1.0;
    /// The measured fp32 FMA throughput in GFLOP/s (0.0 = unmeasured). Scalar
    /// FLOP, the SIMD lane width included - the same counting convention the
    /// device probe's kernel uses, so the two arms' ratios are comparable.
    /// NEVER compare this number across machines: it is a clock- and
    /// core-dependent throughput, while the ratio it feeds is a lane-width
    /// property that survives turbo, SMT and hybrid P/E cores.
    double fp32Gflops = 0.0;
    /// The measured fp64 FMA throughput in GFLOP/s (0.0 = unmeasured).
    double fp64Gflops = 0.0;
    /// True when the two GFLOP/s figures and the ratio came out of an actual
    /// measurement (as opposed to the failed-probe fallback).
    bool measured = false;
    /// True when the measurement lane was the SIMD one (256-bit AVX2 FMA:
    /// 8 fp32 lanes against 4 fp64 lanes); false on the scalar reference
    /// lane, whose ~1.0 ratio says "this machine has no fp32 premium", not
    /// "nothing was measured".
    bool simdLane = false;
    /// The timed lane pairs the ratio was estimated from (0 = unmeasured).
    /// The estimator is a min-of-pairs peak, so more pairs means a tighter
    /// reading, and 0 means there is no reading to judge.
    int pairs = 0;
    /// The spread of the per-pair ratios (max - min, in ratio units) - the
    /// instrument's own noise, recorded so a reading can be judged rather
    /// than trusted. 0.0 with `measured = true` and `pairs = 1` is a single
    /// reading, not a perfect one.
    double ratioSpread = 0.0;
};

/// Probes the host's compute profile: a register-bound SIMD FMA
/// micro-benchmark, one lane per precision, run once per process (~50-100 ms
/// of single-threaded CPU time, so callers run it once at run start, never
/// per build). The measurement is a ratio of two interleaved lanes, which
/// makes it invariant to timer calibration and to symmetric DVFS drift; the
/// absolute GFLOP/s are the raw readings and are not portable.
///
/// Probe-failure-safe: a failed measurement (a degenerate span, a
/// non-positive time, a machine whose lanes cannot be timed) leaves the
/// documented "unknown" defaults. Never an error. No allocation, no OpenMP
/// team, no memory traffic beyond the stack, so it is safe to run inside a
/// memory-cap-gated run - unlike the device probe, whose CUDA runtime init is
/// what that gate exists for (run_driver.cpp).
/// \returns The host's compute profile.
/// \ingroup qcx-backend
HostComputeProfile DetectHostComputeProfile() noexcept;

} // namespace qcx::backend
