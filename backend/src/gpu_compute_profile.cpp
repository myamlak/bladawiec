// Result-typed (C++23) layer of the device compute-profile probe. All
// CUDA runtime calls live in gpu_compute_profile.cu behind the C++20-safe
// bridge in gpu_compute_profile_detail.hpp; this TU composes the
// capability flags and the measured throughput ratio into the one profile
// struct the precision ladder consumes. Probe-failure-safe: a failed
// query or measurement leaves the documented "unknown" defaults for that
// field - never an error.

#include "qcx/backend/gpu_compute_profile.hpp"

#include "gpu_compute_profile_detail.hpp"

namespace qcx::backend {

#ifndef QcxHasCuda

GpuComputeProfile DetectGpuComputeProfile(int deviceId) noexcept {
    (void)deviceId;
    // No CUDA tier in a non-CUDA build: the documented "unknown" profile
    // (the header contract), never an error.
    return GpuComputeProfile{};
}

#else

GpuComputeProfile DetectGpuComputeProfile(int deviceId) noexcept {
    GpuComputeProfile profile;
    bool tensorCores = false;

    if (std::string err = CudaComputeCapabilities(deviceId, &tensorCores); err.empty())
    {
        profile.tensorCores = tensorCores;
        // Every Volta+ tensor core has the fp16 path (the header note);
        // a finer capability probe would split this flag, none exists yet.
        profile.fp16TensorCores = tensorCores;
    }

    double fp32Gflops = 0.0;
    double fp64Gflops = 0.0;

    if (std::string err = CudaMeasureThroughput(deviceId, &fp32Gflops, &fp64Gflops);
        err.empty() && fp32Gflops > 0.0 && fp64Gflops > 0.0)
    {
        profile.fp32Gflops = fp32Gflops;
        profile.fp64Gflops = fp64Gflops;
        profile.fp32ToFp64Ratio = fp32Gflops / fp64Gflops;
    }

    return profile;
}

#endif

} // namespace qcx::backend
