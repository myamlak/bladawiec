// Private C++20-safe bridge between the C++23 host TU (gpu_compute_profile.cpp)
// and the nvcc-compiled probe TU (gpu_compute_profile.cu). The .cu TU must
// stay free of qcx C++23 headers (the bridge contract), so it
// exposes plain functions returning the driver's own error text (empty on
// success) and this header declares them for the C++23 side.
#pragma once

#include <string>

namespace qcx::backend {

// Reads the tensor-core capability flags of one device from the device
// properties (compute capability >= 7.0 = Volta or newer); returns the
// error description (empty on success).
std::string CudaComputeCapabilities(int deviceId, bool* tensorCores);

// Runs the fp32/fp64 FMA-throughput micro-benchmark on one device and
// reports both lanes in GFLOP/s; returns the error description (empty on
// success). The probe restores the previously current device even on
// failure, so the caller's device context is never left changed.
std::string CudaMeasureThroughput(int deviceId, double* fp32Gflops, double* fp64Gflops);

} // namespace qcx::backend
