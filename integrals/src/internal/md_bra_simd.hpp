#pragma once

// The AVX2 vectorized micro-GEMMs of the MD transform path: the fp64 inner
// loops of MicroGemmAdd/MicroGemm (md_gemm.hpp) vectorized 4-wide over the
// n columns - the broadcast of A(i,l) makes every operand access contiguous.
// Defined in md_bra_simd.cpp,
// compiled with /arch:AVX2 via per-file compile options (the class TUs and
// their PCH stay portable); md_transform.hpp dispatches through
// HasAvx2Fma() at runtime, the scalar kernels remain the fallback.

#include <cstddef>

namespace qcx::integrals::internal {

/// Whether the CPU supports the AVX2 + FMA instruction set of the SIMD
/// micro kernels (a runtime check - the kernels are compiled with
/// /arch:AVX2 and must not run elsewhere). The detection itself is
/// CpuHasAvx2Fma() (cpu_features.hpp), shared with the module's other AVX2
/// translation unit.
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
