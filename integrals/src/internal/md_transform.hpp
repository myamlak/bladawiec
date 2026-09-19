#pragma once

// The GEMM dispatch of the MD transforms: the internal register-tiled micro
// kernel for tiny shapes, the linalg batched seam for the fp64 lane, Eigen
// products for the certified fp32 lane (the seam's float path is an
// addition). The ket transform batches
// over tasks sharing one primitive pair's transform matrix (shared B); the
// bra transform is per task with the shared contracted T_ab^ctd (shared A).
//
// NOTE (a possible optimization): the transform GEMM dimensions here are the
// full 3D-Hermite counts (md_defs.hpp). The z-folded "2D-Hermite"
// formulation (Hermite2DCount dims) shrinks the ket GEMM k-dim
// by the z-fold; the batch/seam architecture is unchanged when that lands.

#include "md_bra_simd.hpp"
#include "md_gemm.hpp"
#include "qcx/linalg/batched_ops.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace qcx::integrals::internal {

/// Shapes at or below this in every dimension stay in the micro kernel.
inline constexpr std::size_t kMicroGemmShape = 16;

/// The runtime AVX2+FMA gate, cached (the micro path dispatches per batch
/// task; CPUID once per process is the right cadence).
inline bool Avx2BraEnabled() {
    static const bool enabled = HasAvx2Fma();
    return enabled;
}

/// The micro-gate shape test of both transforms: a shape at or below
/// kMicroGemmShape in every dimension stays in the in-tree micro kernel, any
/// other shape reaches the linalg batched seam. The transforms and the
/// kernel-span probe's seam counter both read this one predicate, so the
/// counter can never drift from the branch that actually decides.
///
/// The AVX2 check is deliberately NOT part of the gate: a disabled AVX2 unit
/// selects MicroGemm over MicroGemmSimd, both in-tree - it never leaves for
/// the seam.
/// \param m Rows of the transform's left operand.
/// \param k Shared (contracted) dimension.
/// \param n Columns of the transform's right operand.
/// \returns True when the shape stays in the in-tree micro kernel.
inline constexpr bool MicroGemmEligible(std::size_t m, std::size_t k, std::size_t n) noexcept {
    return m <= kMicroGemmShape && k <= kMicroGemmShape && n <= kMicroGemmShape;
}

/// C_t (m x n) += A_t (m x k) x B (k x n), B shared, A_t/C_t contiguous per
/// task (stride m*k / m*n). The fp32 lane converts B once into \p bCopy
/// (the conversion is covered by the certified bound).
/// \returns An Error from the batched-GEMM seam (kInvalidArgument); the
/// arguments here are valid by construction, so a failure is a programming
/// or vendor-state bug.
template <typename T>
qcx::Result<void> KetTransformBatched(std::size_t batch,
                                      std::size_t m,
                                      std::size_t k,
                                      std::size_t n,
                                      const T* a,
                                      const double* b,
                                      T* c,
                                      T* bCopy) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

    if constexpr (std::is_same_v<T, double>)
    {
        if (MicroGemmEligible(m, k, n))
        {
            if (Avx2BraEnabled())
            {
                for (std::size_t t = 0; t < batch; ++t)
                {
                    MicroGemmAddSimd(m, n, k, a + t * m * k, b, c + t * m * n);
                }

                return {};
            }

            for (std::size_t t = 0; t < batch; ++t)
            {
                MicroGemmAdd(m, n, k, a + t * m * k, b, c + t * m * n);
            }

            return {};
        }

        // The vendor seam handles the rest; without a vendor it falls back
        // to serial Eigen. Row-major leading dimensions:
        // lda = k, ldb = n, ldc = n.
        return qcx::linalg::MultiplyBatched(batch,
                                            m,
                                            k,
                                            n,
                                            1.0,
                                            a,
                                            k,
                                            static_cast<std::ptrdiff_t>(m * k),
                                            b,
                                            n,
                                            0,
                                            1.0,
                                            c,
                                            n,
                                            static_cast<std::ptrdiff_t>(m * n));
    } else
    {
        for (std::size_t i = 0; i < k * n; ++i)
        {
            bCopy[i] = static_cast<T>(b[i]);
        }

        using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        for (std::size_t t = 0; t < batch; ++t)
        {
            Eigen::Map<const MatrixT> aMap(
                a + t * m * k, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
            Eigen::Map<const MatrixT> bMap(
                bCopy, static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
            Eigen::Map<MatrixT> cMap(
                c + t * m * n, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
            cMap.noalias() += aMap * bMap;
        }

        return {};
    }
}

/// C (m x n) = A (m x k) x B (k x n), A shared. The fp32 lane converts A
/// once into \p aCopy.
/// \returns An Error from the batched-GEMM seam (kInvalidArgument); the
/// arguments here are valid by construction, so a failure is a programming
/// or vendor-state bug.
template <typename T>
qcx::Result<void> BraTransformSingle(
    std::size_t m, std::size_t k, std::size_t n, const double* a, const T* b, T* c, T* aCopy) {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);

    if constexpr (std::is_same_v<T, double>)
    {
        if (MicroGemmEligible(m, k, n))
        {
            if (Avx2BraEnabled())
            {
                MicroGemmSimd(m, n, k, a, b, c);
            } else
            {
                MicroGemm(m, n, k, a, b, c);
            }

            return {};
        }

        return qcx::linalg::MultiplyBatched(1,
                                            m,
                                            k,
                                            n,
                                            1.0,
                                            a,
                                            k,
                                            0,
                                            b,
                                            n,
                                            static_cast<std::ptrdiff_t>(k * n),
                                            0.0,
                                            c,
                                            n,
                                            static_cast<std::ptrdiff_t>(m * n));
    } else
    {
        for (std::size_t i = 0; i < m * k; ++i)
        {
            aCopy[i] = static_cast<T>(a[i]);
        }

        using MatrixT = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const MatrixT> aMap(
            aCopy, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(k));
        Eigen::Map<const MatrixT> bMap(
            b, static_cast<Eigen::Index>(k), static_cast<Eigen::Index>(n));
        Eigen::Map<MatrixT> cMap(c, static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n));
        cMap.noalias() = aMap * bMap;

        return {};
    }
}

} // namespace qcx::integrals::internal
