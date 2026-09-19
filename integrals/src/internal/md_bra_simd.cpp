// The AVX2 micro-GEMMs (md_bra_simd.hpp): the fp64 inner loops vectorized
// 4-wide over n - for every (l, i) one broadcast of A(i,l) against the
// contiguous B(l, j..j+3) and C(i, j..j+3). This TU is compiled with
// /arch:AVX2 (per-file compile options, SKIP_PRECOMPILE_HEADERS - the
// boys_simd pattern); the callers stay portable and gate on HasAvx2Fma().
//
// THE SMALL-n BRANCHES, and why they exist. An AVX2 vector holds four
// doubles, so a kernel that vectorizes over n does NOTHING for a shape with
// fewer than four columns: n < 4 lands in the scalar tail unvectorized. The
// MD transforms meet that shape constantly - the ket transform's n is the
// ket pair's function count nFunc(lc) * nFunc(ld), so n = 1 is every s-s ket
// pair and n = 3 every s-p one - and a measured census of the
// C4H10/def2-SVP k = 1 leg found the in-tree transform kernels running at
// 4.4 ns per multiply-add, several times the vendor seam's own rate on the
// shapes the seam does serve.
//
// Three changes, all structural rather than arithmetic:
//   * the four-row panel (n in 2..3): one register accumulator per column,
//     four rows at a time, with the column broadcasts of B hoisted out of
//     the row loop;
//   * the single-column form (n = 1): the output is one column, so the rows
//     are an independent scaled accumulate and vectorize directly;
//   * the accumulator of the n-vectorized branch is held in a register
//     across the l loop instead of being re-loaded and re-stored for every
//     (column panel, l, row) triple - the add form used to round-trip C
//     through memory k times per cell.
//
// BIT-IDENTITY. Every branch keeps the per-cell term order of the kernel it
// replaces: start from the stored C value (the add forms) or from zero (the
// assigning forms), then take l = 0, 1, 2, ... in order. Only the order in
// which INDEPENDENT cells are visited changes, and the order in which terms
// are summed into one cell never does. The CONTRACTION is the subtle half -
// see MulAdd below; getting it wrong is a one-ulp difference that a
// checksum never sees.
//
// THE GATE. Nothing here runs unless HasAvx2Fma() holds - md_transform.hpp
// dispatches on it, and it is qcx::backend::CpuHasAvx2Fma()
// (qcx/backend/cpu_features.hpp), which reads
// the FMA flag from the ARCHITECTURAL bit CPUID.01H:ECX bit 12. It used to
// read CPUID.(EAX=7,ECX=0):EBX bit 12, RDT-M/PQM, which no consumer CPU
// reports: the answer was "no FMA" on a machine that has FMA, every in-tree
// transform shape went to the scalar md_gemm.hpp kernels instead, and this
// whole tier was dead code (found and measured 2026-09-12; the detection now
// lives in one header shared with fock_contract_simd.cpp and with the
// backend module's host compute-profile probe - so no two consumers can
// drift apart again).

#include "md_bra_simd.hpp"

#include <cstddef>
// The architecture test is included before the intrinsics header because the
// guard on that include reads it.
#include <qcx/backend/cpu_features.hpp>
#if QcxArchX86_64
#include <immintrin.h>
#else
// The portable kernels this file's two entry points fall back to below.
#include "md_gemm.hpp"
#endif

namespace qcx::integrals::internal {

#if QcxArchX86_64

namespace {

/// The widest column count the small-n kernels take: below four columns an
/// AVX2 vector is not filled, which is exactly the case they exist for.
inline constexpr std::size_t kSmallNMax = 3;

/// \returns a * b + c, contracted exactly as the SCALAR kernels this tier
/// replaces contract it - which is to say, not at all. `MicroGemmAdd` /
/// `MicroGemm` (md_gemm.hpp) are instantiated inside the class TUs, which
/// carry no /arch flag, so `c[i * n + j] += a[i * k + l] * bValue` is a
/// separate multiply and add with no FMA to fuse into. A `_mm256_fmadd_pd`
/// anywhere in this file is therefore one ulp away from the kernel it
/// replaces - on some operands, silently - so EVERY branch uses this form,
/// including the two n-vectorized ones that were the last fused sites.
///
/// MEASURED, not assumed. The bit probes memcmp the tier against the
/// scalar kernels cell for cell, raw bytes, over several accumulating calls
/// of pseudo-random operands, on all 78 (m, k, n) shapes the ket census
/// reported: the FMA form agrees on 17 of 78, this one on 78 of 78
/// (against the no-/arch reference and against the HEAD kernels).
///
/// The price of the form is stated rather than hidden: it is two operations
/// per term where the FMA is one. Bit-identity is what the k = 1 pins
/// require of the transform path, and the tier is measured with and without
/// the fusion so the cost is a number, not a belief.
/// \param a The broadcast A element.
/// \param b The broadcast B element.
/// \param c The running accumulator, the addend.
__m256d MulAdd(__m256d a, __m256d b, __m256d c) noexcept {
    return _mm256_add_pd(_mm256_mul_pd(a, b), c);
}

/// The deepest shared dimension the small-n kernels' hoisted broadcast table
/// takes. The transforms gate every shape they dispatch on
/// kHermKet / kBra <= kMicroGemmShape, so 16 covers them all; a deeper k
/// falls back to the general kernels.
inline constexpr std::size_t kSmallNKMax = 16;

/// C (m x n) += A (m x k) x B (k x n) as the general kernel's scalar tail
/// writes it - the same loop nest, so a shape too short for one four-row
/// panel is bit-for-bit the work the n-vectorized kernel would have done.
/// The (m, n, k) and (a, b, c) orders are BLAS's.
void SmallnScalarTailAdd(std::size_t m, // NOLINT(bugprone-easily-swappable-parameters)
                         std::size_t n,
                         std::size_t k,
                         const double* a, // NOLINT(bugprone-easily-swappable-parameters)
                         const double* b,
                         double* c) {
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t l = 0; l < k; ++l)
        {
            const double bValue = b[l * n + j];

            for (std::size_t i = 0; i < m; ++i)
            {
                c[i * n + j] += a[i * k + l] * bValue;
            }
        }
    }
}

/// C (m x n) = A (m x k) x B (k x n) as the general kernel's scalar tail
/// writes it (the assigning twin of SmallnScalarTailAdd).
void SmallnScalarTail(
    // (m, n, k) is the BLAS-style micro-kernel order, fixed by every call site.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::size_t m,
    std::size_t n,
    std::size_t k,
    const double* a,
    const double* b,
    double* c) {
    for (std::size_t j = 0; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            double value = 0.0;

            for (std::size_t l = 0; l < k; ++l)
            {
                value += a[i * k + l] * b[l * n + j];
            }

            c[i * n + j] = value;
        }
    }
}

/// C (m x 1) += A (m x k) x B (k x 1): n = 1, where the output is a single
/// column and the rows are independent dot products of length k whose terms
/// are taken in l order after the stored value - the general kernel's order.
// (m, k) is the BLAS-style micro-kernel order, fixed by every call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void SmallnColumnAdd(std::size_t m, std::size_t k, const double* a, const double* b, double* c) {
    if (k == 1)
    {
        // The ket transform's most numerous shape: an s-s ket pair, whose
        // contract is one column of one primitive, so the whole GEMM is
        // c[i] += a[i] * b[0] over contiguous rows.
        const __m256d bValue = _mm256_set1_pd(b[0]);
        std::size_t i = 0;

        for (; i + 4 <= m; i += 4)
        {
            _mm256_storeu_pd(c + i, MulAdd(_mm256_loadu_pd(a + i), bValue, _mm256_loadu_pd(c + i)));
        }

        for (; i < m; ++i)
        {
            c[i] += a[i] * b[0];
        }

        return;
    }

    for (std::size_t l = 0; l < k; ++l)
    {
        const __m256d bValue = _mm256_set1_pd(b[l]);
        std::size_t i = 0;

        for (; i + 4 <= m; i += 4)
        {
            // A's column l is strided by k, so the four lanes are set one by
            // one; C is contiguous, so its vector is a plain load.
            const __m256d aValue = _mm256_set_pd(
                a[(i + 3) * k + l], a[(i + 2) * k + l], a[(i + 1) * k + l], a[i * k + l]);
            _mm256_storeu_pd(c + i, MulAdd(aValue, bValue, _mm256_loadu_pd(c + i)));
        }

        for (; i < m; ++i)
        {
            c[i] += a[i * k + l] * b[l];
        }
    }
}

/// C (m x 1) = A (m x k) x B (k x 1), the assigning twin of SmallnColumnAdd.
///
/// THE ONE PLACE THE TIER IS NOT BYTE-IDENTICAL, and why it stays that way.
/// The scalar kernel it replaces starts each cell from `T value = 0` and then
/// adds the first term, so a product of exactly -0.0 lands as +0.0 there; the
/// l = 0 store below writes the product directly, so it lands as -0.0 here.
/// Adding the zero back does NOT work: MSVC folds `_mm256_add_pd(x,
/// _mm256_setzero_pd())` into `x` (measured — the scalar `0.0 + x` is
/// preserved and gives +0.0, the vector form still gives -0.0, see
/// tools/bench/probes/zeroprobe.cpp, where the measurement was taken),
/// so the vector path cannot
/// express `0 + x` at all. What the difference costs is nothing: -0.0 and
/// +0.0 are the same value, every later term washes the sign out of any
/// non-zero cell, and no consumer of a transform divides by one.
/// SimdMicroGemmTest.SimdDiffersFromScalarOnlyInTheSignOfZero pins exactly
/// this much: values agree everywhere, and every differing byte is a cell
/// where both are zero.
// (m, k) is the BLAS-style micro-kernel order, fixed by every call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void SmallnColumn(std::size_t m, std::size_t k, const double* a, const double* b, double* c) {
    if (k == 1)
    {
        const __m256d bValue = _mm256_set1_pd(b[0]);
        std::size_t i = 0;

        for (; i + 4 <= m; i += 4)
        {
            _mm256_storeu_pd(c + i, _mm256_mul_pd(_mm256_loadu_pd(a + i), bValue));
        }

        for (; i < m; ++i)
        {
            c[i] = a[i] * b[0];
        }

        return;
    }

    for (std::size_t l = 0; l < k; ++l)
    {
        const __m256d bValue = _mm256_set1_pd(b[l]);
        std::size_t i = 0;

        for (; i + 4 <= m; i += 4)
        {
            const __m256d aValue = _mm256_set_pd(
                a[(i + 3) * k + l], a[(i + 2) * k + l], a[(i + 1) * k + l], a[i * k + l]);

            if (l == 0)
            {
                _mm256_storeu_pd(c + i, _mm256_mul_pd(aValue, bValue));
            } else
            {
                _mm256_storeu_pd(c + i, MulAdd(aValue, bValue, _mm256_loadu_pd(c + i)));
            }
        }

        for (; i < m; ++i)
        {
            if (l == 0)
            {
                c[i] = a[i * k + l] * b[l];
            } else
            {
                c[i] += a[i * k + l] * b[l];
            }
        }
    }
}

/// C (m x n) += A (m x k) x B (k x n) for n in 2..kSmallNMax: four rows at a
/// time, one register accumulator vector per column, the column broadcasts
/// of B hoisted out of the row loop. The rows are independent, so the panel
/// is only a different visit order for independent cells; per cell the
/// accumulation starts from the stored value and takes l in order.
void SmallnPanelAdd(
    std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c) {
    if (m < 4)
    {
        // Too short for one panel: the general kernel's own scalar tail,
        // verbatim, so no shape regresses on the panel's setup cost.
        SmallnScalarTailAdd(m, n, k, a, b, c);
        return;
    }

    __m256d bVec[kSmallNKMax][kSmallNMax] = {};

    for (std::size_t l = 0; l < k; ++l)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            bVec[l][j] = _mm256_set1_pd(b[l * n + j]);
        }
    }

    const std::size_t nPanel = m & ~std::size_t{3};

    for (std::size_t i = 0; i < nPanel; i += 4)
    {
        __m256d acc[kSmallNMax];

        for (std::size_t j = 0; j < n; ++j)
        {
            acc[j] = _mm256_set_pd(
                c[(i + 3) * n + j], c[(i + 2) * n + j], c[(i + 1) * n + j], c[i * n + j]);
        }

        for (std::size_t l = 0; l < k; ++l)
        {
            const __m256d aValue = _mm256_set_pd(
                a[(i + 3) * k + l], a[(i + 2) * k + l], a[(i + 1) * k + l], a[i * k + l]);

            for (std::size_t j = 0; j < n; ++j)
            {
                acc[j] = MulAdd(aValue, bVec[l][j], acc[j]);
            }
        }

        for (std::size_t j = 0; j < n; ++j)
        {
            alignas(32) double lanes[4] = {};
            _mm256_storeu_pd(lanes, acc[j]);
            c[(i + 3) * n + j] = lanes[3];
            c[(i + 2) * n + j] = lanes[2];
            c[(i + 1) * n + j] = lanes[1];
            c[i * n + j] = lanes[0];
        }
    }

    if (nPanel < m)
    {
        SmallnScalarTailAdd(m - nPanel, n, k, a + nPanel * k, b, c + nPanel * n);
    }
}

/// C (m x n) = A (m x k) x B (k x n) for n in 2..kSmallNMax - the assigning
/// twin of SmallnPanelAdd. Per cell the accumulation starts from zero and
/// takes l in order, which is the general assigning kernel's order.
void SmallnPanel(
    std::size_t m, std::size_t n, std::size_t k, const double* a, const double* b, double* c) {
    if (m < 4)
    {
        SmallnScalarTail(m, n, k, a, b, c);
        return;
    }

    __m256d bVec[kSmallNKMax][kSmallNMax] = {};

    for (std::size_t l = 0; l < k; ++l)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            bVec[l][j] = _mm256_set1_pd(b[l * n + j]);
        }
    }

    const std::size_t nPanel = m & ~std::size_t{3};

    for (std::size_t i = 0; i < nPanel; i += 4)
    {
        __m256d acc[kSmallNMax] = {};

        for (std::size_t l = 0; l < k; ++l)
        {
            const __m256d aValue = _mm256_set_pd(
                a[(i + 3) * k + l], a[(i + 2) * k + l], a[(i + 1) * k + l], a[i * k + l]);

            for (std::size_t j = 0; j < n; ++j)
            {
                acc[j] = MulAdd(aValue, bVec[l][j], acc[j]);
            }
        }

        for (std::size_t j = 0; j < n; ++j)
        {
            alignas(32) double lanes[4] = {};
            _mm256_storeu_pd(lanes, acc[j]);
            c[(i + 3) * n + j] = lanes[3];
            c[(i + 2) * n + j] = lanes[2];
            c[(i + 1) * n + j] = lanes[1];
            c[i * n + j] = lanes[0];
        }
    }

    if (nPanel < m)
    {
        SmallnScalarTail(m - nPanel, n, k, a + nPanel * k, b, c + nPanel * n);
    }
}

/// Whether a shape takes one of the small-n kernels: fewer than four columns
/// (so the n-vectorized branch cannot fill a vector) and a shared dimension
/// the hoisted broadcast table covers.
inline bool SmallnShape(std::size_t n, std::size_t k) noexcept {
    return n <= kSmallNMax && k <= kSmallNKMax;
}

} // namespace

#endif // QcxArchX86_64

bool HasAvx2Fma() noexcept {
    return qcx::backend::CpuHasAvx2Fma();
}

void MicroGemmAddSimd(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (m, n, k) micro-gemm order.
    std::size_t m,
    std::size_t n,
    std::size_t k,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (a, b, c) buffers, same order.
    const double* a,
    const double* b,
    double* c) {
#if QcxArchX86_64
    const std::size_t nFull = n & ~std::size_t{3};

    if (nFull == 0)
    {
        // Fewer than four columns: the n-vectorized loops below would leave
        // the whole shape to their scalar tails. Take the branch built for
        // it instead (the same per-cell order, so the same result).
        if (n == 1)
        {
            SmallnColumnAdd(m, k, a, b, c);
        } else if (SmallnShape(n, k))
        {
            SmallnPanelAdd(m, n, k, a, b, c);
        } else
        {
            SmallnScalarTailAdd(m, n, k, a, b, c);
        }

        return;
    }

    for (std::size_t j = 0; j < nFull; j += 4)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            // The accumulator is held across the l loop: the cell used to be
            // re-loaded and re-stored for every l, k round trips through
            // memory per cell.
            __m256d cValue = _mm256_loadu_pd(c + i * n + j);

            for (std::size_t l = 0; l < k; ++l)
            {
                cValue = MulAdd(
                    _mm256_broadcast_sd(a + i * k + l), _mm256_loadu_pd(b + l * n + j), cValue);
            }

            _mm256_storeu_pd(c + i * n + j, cValue);
        }
    }

    for (std::size_t j = nFull; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            double cValue = c[i * n + j];

            for (std::size_t l = 0; l < k; ++l)
            {
                cValue += a[i * k + l] * b[l * n + j];
            }

            c[i * n + j] = cValue;
        }
    }
#else
    // Not an x86-64 target: there is no vector tier here for the code above to
    // be the fast path of, so the portable kernel is the whole implementation.
    // It is the same kernel the runtime check selects on an x86-64 machine
    // without FMA, so a caller's arithmetic is unchanged from that path.
    MicroGemmAdd<double>(m, n, k, a, b, c);
#endif
}

void MicroGemmSimd(
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (m, n, k) micro-gemm order.
    std::size_t m,
    std::size_t n,
    std::size_t k,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (a, b, c) buffers, same order.
    const double* a,
    const double* b,
    double* c) {
#if QcxArchX86_64
    const std::size_t nFull = n & ~std::size_t{3};

    if (nFull == 0)
    {
        // Fewer than four columns: the n-vectorized loops below would leave
        // the whole shape to their scalar tails. Take the branch built for
        // it instead (the same per-cell order, so the same result).
        if (n == 1)
        {
            SmallnColumn(m, k, a, b, c);
        } else if (SmallnShape(n, k))
        {
            SmallnPanel(m, n, k, a, b, c);
        } else
        {
            SmallnScalarTail(m, n, k, a, b, c);
        }

        return;
    }

    for (std::size_t j = 0; j < nFull; j += 4)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            __m256d acc = _mm256_setzero_pd();

            for (std::size_t l = 0; l < k; ++l)
            {
                const __m256d aValue = _mm256_broadcast_sd(a + i * k + l);
                acc = MulAdd(aValue, _mm256_loadu_pd(b + l * n + j), acc);
            }

            _mm256_storeu_pd(c + i * n + j, acc);
        }
    }

    for (std::size_t j = nFull; j < n; ++j)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            double value = 0.0;

            for (std::size_t l = 0; l < k; ++l)
            {
                value += a[i * k + l] * b[l * n + j];
            }

            c[i * n + j] = value;
        }
    }
#else
    // Same as MicroGemmAddSimd above: on a target without the vector tier the
    // portable kernel is the whole implementation, not a fallback from one.
    MicroGemm<double>(m, n, k, a, b, c);
#endif
}

} // namespace qcx::integrals::internal
