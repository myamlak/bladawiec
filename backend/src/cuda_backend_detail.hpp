// Private C++20-safe bridge between the public (C++23, Result-typed) CUDA
// backend and the nvcc-compiled kernel TU. cuda_backend.cu must stay free
// of qcx C++23 headers, so it exposes plain functions returning
// the driver's own error text (empty on success) and this header declares
// them for the C++23 side.
#pragma once

#include <cstddef>
#include <string>

namespace qcx::backend {

// Returns the driver's error description (empty on success).
std::string CudaGetDevice(int* deviceId);

// Returns the number of CUDA-capable devices (cudaGetDeviceCount); returns
// the error description (empty on success).
std::string CudaGetDeviceCount(int* deviceCount);

// Reads the device memory of one device by id (cudaDeviceGetAttribute for
// the total, a temporary cudaSetDevice + cudaMemGetInfo for the free);
// returns the error description (empty on success). The probe restores the
// previously current device even on failure, so the caller's device context
// is never left changed.
std::string CudaDeviceMemInfo(int deviceId, std::size_t* freeBytes, std::size_t* totalBytes);

// Reads the device's free/total memory (cudaMemGetInfo); returns the error
// description (empty on success).
std::string CudaGetMemInfo(std::size_t* freeBytes, std::size_t* totalBytes);

// Runs VectorAddKernel and synchronizes; returns the error description (empty on success).
std::string CudaVectorAdd(const float* a, const float* b, float* out, int n);

} // namespace qcx::backend
