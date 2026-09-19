#pragma once

/// \file
/// The device compute-profile probe: fp16 tensor-core availability, the
/// measured fp32/fp64 throughput ratio (a local-only micro-benchmark), and
/// the capability flags the precision ladder gates its fp16 band on and
/// prices its band boundaries with. The free-VRAM half of the device-profile
/// probe lives in DeviceProfile (topology_profile.hpp) - the precision policy
/// consumes both profiles together, and all band boundaries and cost weights
/// live in the one policy-side profile struct they feed.

#include <cstddef>

namespace qcx::backend {

/// One device's compute profile. All defaults are the documented "unknown"
/// profile of a device-less host - no tensor cores, a 1:1 throughput
/// ratio, nothing measured - the value every probe failure falls back to;
/// probing is never an error.
/// \ingroup qcx-backend
struct GpuComputeProfile {
    /// Tensor cores present (compute capability >= 7.0, Volta or newer).
    bool tensorCores = false;
    /// fp16 tensor-core operations available: every Volta+ tensor core has
    /// the fp16 path, so this mirrors tensorCores until a finer probe
    /// lands. The precision ladder's fp16 band requires this flag.
    bool fp16TensorCores = false;
    /// The measured fp32/fp64 FMA-throughput ratio (fp32Gflops /
    /// fp64Gflops). 1.0 is the "unknown" value: the micro-benchmark did
    /// not run (no device, no CUDA build, or a failed measurement) - the
    /// ladder then prices both lanes equally and keeps the fp16 band off
    /// the tensor-core flag alone. On the local Turing Quadro the
    /// measurement lands near the consumer 1/32 rate; the datacenter
    /// reference class measures near 1/2.
    double fp32ToFp64Ratio = 1.0;
    /// The measured fp32 FMA throughput in GFLOP/s (0.0 = unmeasured).
    double fp32Gflops = 0.0;
    /// The measured fp64 FMA throughput in GFLOP/s (0.0 = unmeasured).
    double fp64Gflops = 0.0;
};

/// Probes one device's compute profile: the capability flags from the
/// device properties plus the measured fp32/fp64 throughput ratio (a
/// short local-only micro-benchmark of register-bound FMA kernels - a
/// few hundred milliseconds per device, so callers run it once at driver
/// start, not per build). Probe-failure-safe: any failed query or
/// measurement leaves the "unknown" defaults for that field. The probe
/// restores the caller's current device, the same convention as
/// DetectDevices (topology_profile.hpp).
/// \param deviceId The CUDA device id (cudaGetDeviceCount order).
/// \returns The device's compute profile.
/// \ingroup qcx-backend
GpuComputeProfile DetectGpuComputeProfile(int deviceId) noexcept;

} // namespace qcx::backend
