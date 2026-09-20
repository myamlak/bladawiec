#pragma once

// The extern "C" seam to boys_cuda_benchmark_kernels.cu (nvcc). Keep this
// header free of qcx C++23 headers — the .cu translation unit is compiled at
// CMAKE_CUDA_STANDARD 20.

#include <cstddef>

extern "C" {

/// Selects the CUDA device (cudaSetDevice). Returns a cudaError_t (0 = ok).
int BoysBenchSetDevice(int device);

/// Uploads the 38 x 1025 double LUT rows (orders 0..37 over x in [0, 32]).
/// Returns a cudaError_t (0 = ok).
int BoysBenchUploadLut(const double* rows);

/// Launches the erf-F0 + upward-recursion lane; the caller synchronizes.
int BoysBenchLaunchErfF64(const int* n,
                          const double* x,
                          double* out,
                          std::size_t count,
                          int blocks,
                          int threads,
                          void* stream);

/// Launches the Tsuji-style gridded-LUT lane; the caller synchronizes.
int BoysBenchLaunchLutF64(const int* n,
                          const double* x,
                          double* out,
                          std::size_t count,
                          int blocks,
                          int threads,
                          void* stream);

} // extern "C"
