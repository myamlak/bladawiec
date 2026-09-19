#pragma once

// The shared host/device contract of the GPU Fock build:
// the per-quartet contraction metadata and the launch arguments of the
// generic EriFockContractKernel (eri_cuda_fock.cu, nvcc), consumed by the
// GpuJkFockBuilder host side (eri_cuda.cpp, MSVC). CUDA-safe C++20 only -
// this header is compiled by nvcc, so no qcx host headers.
//
// The kernel is GENERIC (one instantiation for every class): the whole
// per-quartet geometry arrives per task in EriCudaFockTaskMeta, and the
// integral block is indexed with the packed layout of eri_batch.hpp -
// EriBlockIndex(fi, fj, fk, fl, nI, nJ, nK, nL) = (fj * nI + fi) *
// (nK * nL) + (fl * nK + fk), with (fi, fj, fk, fl) = (fa, fb, fc, fd) and
// (nI, nJ, nK, nL) = (nA, nB, nC, nD). A skeleton formula
// ((fa * nB + fb) * nC + fc) * nD + fd was verified wrong against the live
// EriBlockIndex implementation and is NOT used (the corrected formula
// below is the single layout every consumer must use - eri_batch.hpp's
// documented contract).

#include <cstddef>

namespace qcx::integrals::internal::cuda {

/// The integral-precision selector of the contraction kernel.
/// \{
inline constexpr int kEriCudaFockPrecisionFp64 = 0;
inline constexpr int kEriCudaFockPrecisionFp32 = 1;
/// \}

/// The launcher success code (the eri_cuda.cu kEriCudaOk convention, kept
/// local so this header stays self-contained).
inline constexpr int kEriCudaFockOk = 0;

/// The per-task contraction metadata of one GPU Fock batch (FockTaskMeta,
/// renamed to the cuda-namespace prefix convention). Every
/// field the generic kernel needs from the quartet's shell geometry,
/// computed host-side once per batch task (in batch.tasks order).
struct EriCudaFockTaskMeta {
    unsigned int outputOffset; ///< Block start into this batch's integral data
                               ///< (the batch task's outputOffset).
    unsigned int offsetA; ///< Function offset of shell a (row-major base).
    unsigned int offsetB; ///< Function offset of shell b.
    unsigned int offsetC; ///< Function offset of shell c.
    unsigned int offsetD; ///< Function offset of shell d.
    unsigned int nA; ///< Function count of shell a.
    unsigned int nB; ///< Function count of shell b.
    unsigned int nC; ///< Function count of shell c.
    unsigned int nD; ///< Function count of shell d.
    /// The K-target orbit divisor: ((a == b) ? 2 : 1) * ((c == d) ? 2 : 1) *
    /// ((a == c && b == d) ? 2 : 1) - the contraction DIVIDES by it, the
    /// CPU AccumulateBlock's kScaled = k / kMultiplicity.
    double kMultiplicity;
    bool sameBraPair; ///< a == b (the J-bra transpose-write guard).
    bool sameKetPair; ///< c == d (the J-ket transpose-write guard).
    bool isDiagonal; ///< pairBra == pairKet (the J-ket skip).
};

/// The launch arguments of the GPU Fock contraction. Every pointer refers
/// to a DeviceBuffer allocation (memory module); stream is a
/// cudaStream_t passed as void*.
struct EriCudaFockArgs {
    const EriCudaFockTaskMeta* tasks; ///< Per-task geometry (this batch, task order).
    const double* integralsF64; ///< This batch's integral block data (fp64 lane).
    const float* integralsF32; ///< This batch's integral block data (fp32 lane).
    const double* density; ///< The spatial density, row-major n x n (the row-major
                           ///< u*n+v flattening - NOT the Eigen column-major order).
    double* fock; ///< The Fock matrix, row-major n x n, PRE-INITIALIZED with the
                  ///< flat core Hamiltonian; the kernel accumulates into it.
    std::size_t nTasks; ///< Number of tasks (= grid blocks).
    int n; ///< Total function count (the density/fock row stride).
    int precision; ///< kEriCudaFockPrecisionFp64 / kEriCudaFockPrecisionFp32.
    bool buildExchangeOnly; ///< Skip the J phases (the RI-provides-J mode).
    bool buildCoulombOnly; ///< Skip the K phases (the UHF J-only mode).
    int threads; ///< Block size (256; one block per quartet).
    void* stream; ///< The engine's stream (cudaStream_t).
};

} // namespace qcx::integrals::internal::cuda

extern "C" {
/// Launches the generic Fock contraction for one batch: one block per
/// quartet task, 256 threads, every J/K accumulation an atomicAdd (the
/// phases of one quartet can collide on the same Fock element - e.g. the
/// K1 target of a diagonal quartet - and the 8-fold symmetry maps
/// different quartets to shared elements, so plain writes would race).
/// \param args The launch arguments (EriCudaFockArgs above).
/// \returns 0 on success, or the positive cudaError_t of the launch (the
/// CheckLaunchCode convention of eri_cuda.cpp).
int QcxEriCudaFockContraction(qcx::integrals::internal::cuda::EriCudaFockArgs& args);
} // extern "C"
