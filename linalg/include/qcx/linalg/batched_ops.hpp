#pragma once

/// \file
/// The batched-GEMM seam of the MD engine:
/// a JIT-compiled fixed-shape GEMM for repeated same-shape transform calls
/// (MKL JIT when MKL is the vendor; cblas for the other vendors; Eigen
/// without one), and a strided-batched product over contiguous arrays
/// (stride 0 = shared matrix). All matrices row-major, matching the rest of
/// the linalg module.

#include "qcx/error.hpp"

#include <cstddef>
#include <memory>

namespace qcx::linalg {

/// \ingroup qcx-linalg
/// A GEMM kernel compiled once for one fixed shape, served repeatedly.
///
/// With MKL as the vendor the kernel is JIT-compiled (mkl_jit_create_dgemm);
/// other vendors dispatch to cblas_dgemm per call; without a vendor the
/// kernel falls back to an Eigen product (CI stays green).
class GemmKernel {
public:
    /// Compiles (or prepares) the kernel for C = alpha A B + beta C with
    /// A (m x k), B (k x n), C (m x n), row-major.
    /// \param m Rows of A/C.
    /// \param n Columns of B/C.
    /// \param k Inner dimension.
    /// \param alpha Scalar of A B.
    /// \param beta Scalar of C.
    /// \returns The kernel, or an Error (kInvalidArgument for a zero
    /// dimension, kInternalError when the MKL JIT compilation fails).
    static qcx::Result<GemmKernel> Create(
        std::size_t m, std::size_t n, std::size_t k, double alpha = 1.0, double beta = 0.0);

    /// Applies the kernel: C = alpha A B + beta C.
    /// \param a Left factor (m x k), row-major, leading dimension lda.
    /// \param lda Leading dimension of A: the canonical row-major minimum
    /// k (the kernel is compiled for the fixed shape, so the leading
    /// dimensions are exact, not minimums).
    /// \param b Right factor (k x n), row-major, leading dimension ldb.
    /// \param ldb Leading dimension of B: the canonical minimum n.
    /// \param c Accumulator (m x n), row-major, leading dimension ldc.
    /// \param ldc Leading dimension of C: the canonical minimum n.
    /// \returns An Error (kInvalidArgument) on a null operand or a
    /// non-canonical leading dimension.
    qcx::Result<void> Apply(const double* a,
                            std::size_t lda,
                            const double* b,
                            std::size_t ldb,
                            double* c,
                            std::size_t ldc) const;

private:
    /// The fixed shape, the scalars, and the vendor handle (shared_ptr with
    /// the JIT destroyer when MKL is the vendor; empty otherwise). No
    /// vendor headers needed here - the .cpp owns the vendor surface.
    struct State {
        std::size_t m;
        std::size_t n;
        std::size_t k;
        double alpha;
        double beta;
        std::shared_ptr<void> jitHandle;
    };

    qcx::Result<void> Validate(const double* a,
                               std::size_t lda,
                               const double* b,
                               std::size_t ldb,
                               double* c,
                               std::size_t ldc) const;
    explicit GemmKernel(State state);
    State _state;
};

/// \ingroup qcx-linalg
/// Strided-batched matrix product over contiguous arrays:
/// C_i (m x n) = alpha A_i (m x k) B_i (k x n) + beta C_i, i = 0..batch-1,
/// with A_i at a + i*strideA (row-major, leading dimension lda), likewise
/// B_i/C_i. stride 0 means the matrix is SHARED across the batch (same
/// pointer every element) - the layout the MD ket transform uses for its
/// shared per-primitive-pair transform matrix.
///
/// MKL serves the batch through the cblas_dgemm_batch group API; the other
/// vendors run a per-element cblas_dgemm loop; without a vendor the
/// elements run serially through Eigen (CI stays green).
/// \param batch Number of products.
/// \param m Rows of A_i/C_i.
/// \param k Inner dimension.
/// \param n Columns of B_i/C_i.
/// \param alpha Scalar of A B.
/// \param a First element of the A block.
/// \param lda Leading dimension of A (>= k).
/// \param strideA Element stride between A_i (0 = shared).
/// \param b First element of the B block.
/// \param ldb Leading dimension of B (>= n).
/// \param strideB Element stride between B_i (0 = shared).
/// \param beta Scalar of C.
/// \param c First element of the C block.
/// \param ldc Leading dimension of C (>= n).
/// \param strideC Element stride between C_i (0 = shared).
/// \returns An Error (kInvalidArgument) on a zero batch or dimension, a
/// leading dimension below its minimum, a negative stride, or a null
/// operand.
qcx::Result<void> MultiplyBatched(std::size_t batch,
                                  std::size_t m,
                                  std::size_t k,
                                  std::size_t n,
                                  double alpha,
                                  const double* a,
                                  std::size_t lda,
                                  std::ptrdiff_t strideA,
                                  const double* b,
                                  std::size_t ldb,
                                  std::ptrdiff_t strideB,
                                  double beta,
                                  double* c,
                                  std::size_t ldc,
                                  std::ptrdiff_t strideC);

} // namespace qcx::linalg
