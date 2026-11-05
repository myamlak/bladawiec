#pragma once

// The AVX2 vectorized micro-GEMMs of the MD transform path: the fp64 inner
// loops of MicroGemmAdd/MicroGemm (md_gemm.hpp) vectorized 4-wide over the
// n columns - the broadcast of A(i,l) makes every operand access contiguous.
// Defined in md_bra_simd.cpp, compiled at the program's tier (QCX_ISA=avx2 in
// the shipped configuration; the portable kernels at QCX_ISA=baseline, where
// it carries no target feature of its own); md_transform.hpp dispatches
// through HasAvx2Fma() at runtime, the scalar kernels remain the fallback.

#include <cstddef>

namespace qcx::integrals::internal {

/// Whether this build carries the AVX2 + FMA micro kernels AND the CPU
/// supports them. False at QCX_ISA=baseline, where the entry points below are
/// the portable md_gemm.hpp kernels themselves. The machine half is
/// CpuHasAvx2Fma() (cpu_features.hpp), shared with the module's other SIMD
/// translation unit; the build half is QcxIsaAvx2Fma, the tier the flags
/// selected.
bool HasAvx2Fma() noexcept;

/// C (m x n) += A (m x k) x B (k x n), row-major A/B/C - the vectorized
/// MicroGemmAdd.
void MicroGemmAddSimd(
    std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c);

/// C (m x n) = A (m x k) x B (k x n), row-major A/B/C - the vectorized
/// MicroGemm.
void MicroGemmSimd(
    std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c);

} // namespace qcx::integrals::internal
