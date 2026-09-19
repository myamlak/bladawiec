// The batched-GEMM seam: GemmKernel - MKL JIT for the MKL vendor,
// cblas_dgemm for the other vendors, an Eigen product without one - and
// MultiplyBatched over contiguous arrays through the MKL group API
// (cblas_dgemm_batch), a per-element cblas_dgemm loop for the other
// vendors, or serial Eigen without one (no vendor, CI). All matrices
// row-major.

#include "qcx/linalg/batched_ops.hpp"

#include <Eigen/Core>
#include <array>
#include <climits>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#ifdef QcxHasVendorBlas
#ifdef QcxVendorBlasIsMkl
#include <mkl.h>
#include <mkl_cblas.h> // MKL ships mkl_cblas.h instead of cblas.h
#else
#include <cblas.h>
#endif
#endif

namespace qcx::linalg {
namespace {

// The JIT destroyer: mkl_jit_destroy releases the generated kernel.
#ifdef QcxVendorBlasIsMkl
void DestroyJitHandle(void* handle) noexcept {
    mkl_jit_destroy(handle);
}
#endif

// The MKL group-API pointer arrays fit on the stack up to this batch size.
inline constexpr std::size_t kInlineBatch = 16;

// The BLAS-less lane (CI): a strided Eigen product honoring the
// leading dimensions. Eigen 5.0 accepts the (ptr, rows, cols, stride) Map
// constructor only when the Map declares the Stride template parameter -
// hence the explicit StridedMap aliases.
void EigenStridedProduct(const double* a,
                         std::size_t lda,
                         const double* b,
                         std::size_t ldb,
                         double* c,
                         std::size_t ldc,
                         std::size_t m,
                         std::size_t k,
                         std::size_t n,
                         double alpha,
                         double beta) {
    using StridedMap =
        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
                   0,
                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    using MutableStridedMap =
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
                   0,
                   Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
    MutableStridedMap cMap(
        c,
        static_cast<Eigen::Index>(m),
        static_cast<Eigen::Index>(n),
        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(static_cast<Eigen::Index>(ldc), 1));
    cMap.noalias() = alpha * (StridedMap(a,
                                         static_cast<Eigen::Index>(m),
                                         static_cast<Eigen::Index>(k),
                                         Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                             static_cast<Eigen::Index>(lda), 1)) *
                              StridedMap(b,
                                         static_cast<Eigen::Index>(k),
                                         static_cast<Eigen::Index>(n),
                                         Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>(
                                             static_cast<Eigen::Index>(ldb), 1))) +
                     beta * cMap;
}

} // namespace

GemmKernel::GemmKernel(State state) : _state(std::move(state)) {}

qcx::Result<GemmKernel> GemmKernel::Create(
    std::size_t m, std::size_t n, std::size_t k, double alpha, double beta) {
    if (m == 0 || n == 0 || k == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "GEMM dimensions must be positive"});
    }

    // Every vendor lane passes the dimensions to a 32-bit BLAS parameter
    // (MKL_INT under the LP64 interface this build uses, int elsewhere);
    // a size_t dimension above INT_MAX would truncate silently (lin-2).
    if (m > static_cast<std::size_t>(INT_MAX) || n > static_cast<std::size_t>(INT_MAX) ||
        k > static_cast<std::size_t>(INT_MAX))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "GEMM dimensions exceed the 32-bit BLAS range"});
    }

    State state{m, n, k, alpha, beta, nullptr};
#if defined(QcxHasVendorBlas) && defined(QcxVendorBlasIsMkl)
    void* jitter = nullptr;
    // The fixed-shape kernel: one JIT compilation serves every batch - the
    // MD transform GEMMs are fixed-shape per class. The JIT interface
    // computes column-major GEMMs, so the row-major request C = A B maps to
    // the column-major C^T = B^T A^T with the pointers unchanged: the
    // row-major A (m x k, ld lda) IS the column-major A^T (k x m, ld lda),
    // and likewise B (n x k, ld ldb). The JIT operands are therefore
    // (m', n', k') = (n, m, k) with lda' = ldb = n, ldb' = lda = k,
    // ldc' = ldc = n (the canonical row-major minimums), and the kernel
    // receives the operands swapped (Apply passes b, a, c): the compiled
    // C^T = B^T A^T written column-major at c is exactly the row-major
    // A B. Validated against cblas on random shapes up to 97 x 71 x 80
    // (2026-08-17).
    const mkl_jit_status_t status = mkl_jit_create_dgemm(&jitter,
                                                         MKL_COL_MAJOR,
                                                         MKL_NOTRANS,
                                                         MKL_NOTRANS,
                                                         static_cast<MKL_INT>(n),
                                                         static_cast<MKL_INT>(m),
                                                         static_cast<MKL_INT>(k),
                                                         alpha,
                                                         static_cast<MKL_INT>(n),
                                                         static_cast<MKL_INT>(k),
                                                         beta,
                                                         static_cast<MKL_INT>(n));

    if (status != MKL_JIT_SUCCESS || jitter == nullptr)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInternalError,
                                          "MKL JIT failed to compile the fixed-shape GEMM"});
    }

    state.jitHandle = std::shared_ptr<void>(jitter, DestroyJitHandle);
#endif
    return GemmKernel{std::move(state)};
}

qcx::Result<void> GemmKernel::Validate(const double* a,
                                       std::size_t lda,
                                       const double* b,
                                       std::size_t ldb,
                                       double* c,
                                       std::size_t ldc) const {
    if (a == nullptr || b == nullptr || c == nullptr)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "null matrix pointer"});
    }

    // The canonical row-major contract: the kernel is compiled for the
    // fixed shape, so the leading dimensions are the exact minimums
    // (A (m x k) lda = k, B (k x n) ldb = n, C (m x n) ldc = n) - the
    // JIT bakes them in and the other lanes match.
    if (lda != _state.k || ldb != _state.n || ldc != _state.n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "leading dimension not canonical"});
    }

    return {};
}

qcx::Result<void> GemmKernel::Apply(const double* a,
                                    std::size_t lda,
                                    const double* b,
                                    std::size_t ldb,
                                    double* c,
                                    std::size_t ldc) const {
    auto validation = Validate(a, lda, b, ldb, c, ldc);

    if (!validation.has_value())
    {
        return std::unexpected(validation.error());
    }

    const State& state = _state;

#if defined(QcxHasVendorBlas) && defined(QcxVendorBlasIsMkl)
    // The JIT execute path: the handle resolves to a fixed-signature
    // kernel pointer (mkl_jit_get_dgemm_ptr) with the leading dimensions
    // baked in at creation; the kernel never writes to its operands. The
    // compiled kernel computes C^T = B^T A^T (see Create), so the operand
    // pointers swap roles: the kernel's first matrix is B^T (our b) and
    // its second is A^T (our a) - the same memory, no copies.
    const auto kernel = mkl_jit_get_dgemm_ptr(state.jitHandle.get());
    // The MKL kernel signature takes non-const operands (mkl_types.h);
    // the kernel never writes to them, so the casts adapt our const
    // operands without mutation.
    kernel(state.jitHandle.get(), const_cast<double*>(b), const_cast<double*>(a), c);
#elif defined(QcxHasVendorBlas)
    cblas_dgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                static_cast<int>(state.m),
                static_cast<int>(state.n),
                static_cast<int>(state.k),
                state.alpha,
                a,
                static_cast<int>(lda),
                b,
                static_cast<int>(ldb),
                state.beta,
                c,
                static_cast<int>(ldc));
#else
    // The BLAS-less fallback (CI).
    EigenStridedProduct(a, lda, b, ldb, c, ldc, state.m, state.k, state.n, state.alpha, state.beta);
#endif
    return {};
}

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
                                  std::ptrdiff_t strideC) {
    if (batch == 0 || m == 0 || k == 0 || n == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "batch and dimensions must be positive"});
    }

    if (lda < k || ldb < n || ldc < n || strideA < 0 || strideB < 0 || strideC < 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "invalid leading dimension or stride"});
    }

    // Every vendor lane passes batch and the dimensions/leading dimensions
    // to 32-bit BLAS parameters (MKL_INT under the LP64 interface this
    // build uses, int elsewhere); a size_t value above INT_MAX would
    // truncate silently (lin-2). Rejected up front, before any pointer
    // arithmetic.
    if (batch > static_cast<std::size_t>(INT_MAX) || m > static_cast<std::size_t>(INT_MAX) ||
        k > static_cast<std::size_t>(INT_MAX) || n > static_cast<std::size_t>(INT_MAX) ||
        lda > static_cast<std::size_t>(INT_MAX) || ldb > static_cast<std::size_t>(INT_MAX) ||
        ldc > static_cast<std::size_t>(INT_MAX))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "batch, dimensions and leading dimensions must fit the 32-bit BLAS "
                       "range"});
    }

    if (a == nullptr || b == nullptr || c == nullptr)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "null matrix pointer"});
    }

#ifdef QcxHasVendorBlas
#ifdef QcxVendorBlasIsMkl
    // cblas_dgemm_batch is the group API: one group of size batch over the
    // per-element pointers (stride 0 = the same pointer every element).
    // The MKL batch signatures take MKL_INT: int under the LP64 interface
    // this build uses (no MKL_ILP64 define), 64-bit under ILP64; the
    // INT_MAX guard above keeps the casts below within the 32-bit range in
    // either interface. Small batches (the MD transform's typical case,
    // see the OpenBLAS comment below) use stack pointer arrays; larger
    // batches fall back to the heap.
    using BatchInt = MKL_INT;
    std::array<const double*, kInlineBatch> aInline{};
    std::array<const double*, kInlineBatch> bInline{};
    std::array<double*, kInlineBatch> cInline{};
    std::vector<const double*> aHeap;
    std::vector<const double*> bHeap;
    std::vector<double*> cHeap;
    const double** aPtrs = aInline.data();
    const double** bPtrs = bInline.data();
    double** cPtrs = cInline.data();

    if (batch > kInlineBatch)
    {
        aHeap.resize(batch);
        bHeap.resize(batch);
        cHeap.resize(batch);
        aPtrs = aHeap.data();
        bPtrs = bHeap.data();
        cPtrs = cHeap.data();
    }

    for (std::size_t i = 0; i < batch; ++i)
    {
        aPtrs[i] = a + static_cast<std::ptrdiff_t>(i) * strideA;
        bPtrs[i] = b + static_cast<std::ptrdiff_t>(i) * strideB;
        cPtrs[i] = c + static_cast<std::ptrdiff_t>(i) * strideC;
    }

    const CBLAS_TRANSPOSE noTrans[1] = {CblasNoTrans};
    const BatchInt mInt = static_cast<BatchInt>(m);
    const BatchInt kInt = static_cast<BatchInt>(k);
    const BatchInt nInt = static_cast<BatchInt>(n);
    const BatchInt ldaInt = static_cast<BatchInt>(lda);
    const BatchInt ldbInt = static_cast<BatchInt>(ldb);
    const BatchInt ldcInt = static_cast<BatchInt>(ldc);
    const BatchInt groupSize = static_cast<BatchInt>(batch);
    cblas_dgemm_batch(CblasRowMajor,
                      noTrans,
                      noTrans,
                      &mInt,
                      &nInt,
                      &kInt,
                      &alpha,
                      aPtrs,
                      &ldaInt,
                      bPtrs,
                      &ldbInt,
                      &beta,
                      cPtrs,
                      &ldcInt,
                      1,
                      &groupSize);
#else
    // OpenBLAS: a per-element cblas_dgemm loop. The group API has no
    // coverage on any build lane that uses it (this machine's vendor is
    // MKL), the batch counts the engine produces are small (typically 1),
    // and plain dgemm is the most battle-tested path.
    for (std::size_t i = 0; i < batch; ++i)
    {
        cblas_dgemm(CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    static_cast<int>(m),
                    static_cast<int>(n),
                    static_cast<int>(k),
                    alpha,
                    a + static_cast<std::ptrdiff_t>(i) * strideA,
                    static_cast<int>(lda),
                    b + static_cast<std::ptrdiff_t>(i) * strideB,
                    static_cast<int>(ldb),
                    beta,
                    c + static_cast<std::ptrdiff_t>(i) * strideC,
                    static_cast<int>(ldc));
    }
#endif
#else
    // The BLAS-less lane (CI): the strided product honors the leading
    // dimensions like the vendor lanes (the padded-ld regression guard lives
    // in batched_ops_test.cpp - plain contiguous maps used to ignore them).
    for (std::size_t i = 0; i < batch; ++i)
    {
        EigenStridedProduct(a + static_cast<std::ptrdiff_t>(i) * strideA,
                            lda,
                            b + static_cast<std::ptrdiff_t>(i) * strideB,
                            ldb,
                            c + static_cast<std::ptrdiff_t>(i) * strideC,
                            ldc,
                            m,
                            k,
                            n,
                            alpha,
                            beta);
    }
#endif
    return {};
}

} // namespace qcx::linalg
