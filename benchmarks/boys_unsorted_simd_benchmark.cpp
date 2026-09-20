// The unsorted-SIMD lane of the Boys benchmark suite. The engine-realistic
// sorted lanes live in boys_benchmark.cpp; this file measures the mixed
// per-vector kernel that evaluates all three region paths (A piecewise
// Chebyshev, B F0-Chebyshev + upward with a gathered e^{-x} table, C pure
// asymptotic) and blends per lane — the cost of the unsorted input stream
// the engine actually faces.
//
// Ported from a separate read-only design study onto the shipped qcx tables
// (qcx::integrals::detail:: kPieces / kPieceStart / kCoeffs / kBcoeffs /
// kBDeg / kX0 / kX1). Two corrections against that study:
//   * the e^{-x} gather follows the SHIPPED boys_simd.cpp ExpTable
//     convention (rows padded to 8 doubles, grid index pre-shifted by 3
//     before the scale-8 gather); the design-study copy skipped the shift
//     and read the wrong rows, so its unsorted values were garbage — the
//     self-check here pins the values against BoysSingle.
//   * the scalar tail uses the certified BoysSingle<double> reference.
//
// Custom main() rather than a Google Benchmark harness: --self-check runs the
// verifier (max |out - BoysSingle(8, x)| over the benchmark inputs, budget
// 5.5e-14) and exits; the default mode runs the runs-log protocol
// (warmup + 3 passes, min/median/max, median = paper cell).
#include "boys_coefficients.hpp"
#include "qcx/integrals/boys.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kInputCount = 1u << 22; // 4,194,304 values
constexpr int kOrder = 8; // the design-study unsorted workload order
constexpr int kPasses = 3;
constexpr double kHalfSqrtPi = 0.886226925452758014;

// e^{-x} on [0, 30]: degree-4 Taylor table, rows padded to
// 8 doubles so the gathers can use the legal scale 8. Mirrors the shipped
// construction in external/boys/src/boys_simd.cpp (the ~192 KB gather table the
// paper's footprint cells cite).
class ExpTable {
public:
    static constexpr double kStep = 0.01;
    static constexpr int kNumPoints = 3000;

    ExpTable() noexcept {
        for (int i = 0; i <= kNumPoints; ++i)
        {
            const double z = -(i * kStep);
            const double z2 = z * z / 2;
            const double z3 = z2 * z / 3;
            const double z4 = z3 * z / 4;
            const double e = std::exp(z);
            _coefficients[i][0] = e * (1 - z + z2 - z3 + z4);
            _coefficients[i][1] = e * (1 - z + z2 - z3);
            _coefficients[i][2] = e * (1 - z + z2) / 2;
            _coefficients[i][3] = e * (1 - z) / 6;
            _coefficients[i][4] = e / 24;
        }
    }

    // The stored coefficients are the ALTERNATING-sign monomial form of the
    // Taylor sum sum_k (z - x)^k / k!: evaluate
    // c4*x^4 - c3*x^3 + c2*x^2 - c1*x + c0.
    __m256d Eval4(__m256d x) const noexcept {
        __m128i index = _mm256_cvtpd_epi32(_mm256_mul_pd(x, _mm256_set1_pd(1.0 / kStep)));
        index = _mm_min_epi32(index, _mm_set1_epi32(kNumPoints));
        // Rows are 8 doubles apart; the grid index pre-shifts by 3 so the
        // scale-8 gather lands on the row address.
        index = _mm_slli_epi32(index, 3);
        __m256d c0 = _mm256_i32gather_pd(&_coefficients[0][0], index, 8);
        __m256d c1 = _mm256_i32gather_pd(&_coefficients[0][1], index, 8);
        __m256d c2 = _mm256_i32gather_pd(&_coefficients[0][2], index, 8);
        __m256d c3 = _mm256_i32gather_pd(&_coefficients[0][3], index, 8);
        __m256d c4 = _mm256_i32gather_pd(&_coefficients[0][4], index, 8);
        __m256d result = _mm256_fmsub_pd(c4, x, c3);
        result = _mm256_fmadd_pd(result, x, c2);
        result = _mm256_fmsub_pd(result, x, c1);
        result = _mm256_fmadd_pd(result, x, c0);
        return result;
    }

private:
    double _coefficients[kNumPoints + 1][8]{};
};

// Split Clenshaw (even/odd), 4-wide, half-depth FMA chains — the shipped
// kernel shape (boys_simd.cpp).
__m256d Clenshaw4Split(const qcx::integrals::detail::OrderPiece& piece, __m256d xv) {
    const double* c = qcx::integrals::detail::kCoeffs.data() + piece.offset;
    const int deg = piece.deg;

    if (deg == 0)
    {
        return _mm256_set1_pd(c[0]);
    }

    __m256d t = _mm256_sub_pd(xv, _mm256_set1_pd(piece.a));
    t = _mm256_fmadd_pd(t, _mm256_set1_pd(2.0 / (piece.b - piece.a)), _mm256_set1_pd(-1.0));

    if (deg == 1)
    {
        return _mm256_fmadd_pd(t, _mm256_set1_pd(c[1]), _mm256_set1_pd(c[0]));
    }

    const __m256d v =
        _mm256_fmsub_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(t, t), _mm256_set1_pd(1.0));
    const __m256d twoV = _mm256_add_pd(v, v);
    // even part: e_k = c[2k]
    const int m = deg / 2;
    __m256d b1 = _mm256_set1_pd(c[2 * m]);
    __m256d b2 = _mm256_setzero_pd();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256d b0 = _mm256_fmadd_pd(twoV, b1, _mm256_sub_pd(_mm256_set1_pd(c[2 * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256d even = _mm256_fmadd_pd(v, b1, _mm256_sub_pd(_mm256_set1_pd(c[0]), b2));
    // odd part: o_k = c[2k+1], D recurrence (D_1 = 2v-1)
    __m256d o1 = _mm256_set1_pd(c[2 * m - 1]);
    __m256d o2 = _mm256_setzero_pd();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256d o0 =
            _mm256_fmadd_pd(twoV, o1, _mm256_sub_pd(_mm256_set1_pd(c[2 * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256d odd = _mm256_fmadd_pd(
        _mm256_sub_pd(twoV, _mm256_set1_pd(1.0)), o1, _mm256_sub_pd(_mm256_set1_pd(c[1]), o2));
    return _mm256_fmadd_pd(t, odd, even);
}

// The unsorted variant: all three paths computed per vector, blended per
// lane. Measures the divergence penalty vs. the sorted region-specialized
// lanes of boys_benchmark.cpp.
void ChebSimdMixed(
    int n, const double* x, double* out, std::size_t count, const ExpTable& expTable) {
    using namespace qcx::integrals::detail;

    for (std::size_t i = 0; i + 3 < count; i += 4)
    {
        const __m256d xv = _mm256_loadu_pd(x + i);
        const __m256d mA = _mm256_cmp_pd(xv, _mm256_set1_pd(kX0), _CMP_LT_OQ);
        const __m256d mB = _mm256_cmp_pd(xv, _mm256_set1_pd(kX1), _CMP_LT_OQ);
        const __m256d mBonly = _mm256_andnot_pd(mA, mB);

        __m256d fA = _mm256_setzero_pd();
        {
            const int s = kPieceStart[n];
            const int e = kPieceStart[n + 1];

            for (int p = s; p < e; ++p)
            {
                const OrderPiece& piece = kPieces[p];
                const __m256d mask = _mm256_and_pd(
                    _mm256_and_pd(mA, _mm256_cmp_pd(xv, _mm256_set1_pd(piece.a), _CMP_GE_OQ)),
                    _mm256_cmp_pd(xv, _mm256_set1_pd(piece.b), _CMP_LT_OQ));
                fA = _mm256_blendv_pd(fA, Clenshaw4Split(piece, xv), mask);
            }
        }

        __m256d fB = _mm256_setzero_pd();
        {
            // Split Clenshaw (even/odd) on kBcoeffs — the shipped region-B
            // F0 seed shape (boys_simd.cpp ClenshawB4K).
            const double* c = kBcoeffs.data();
            const int m = kBDeg / 2;
            __m256d t = _mm256_sub_pd(xv, _mm256_set1_pd(kX0));
            t = _mm256_fmadd_pd(t, _mm256_set1_pd(2.0 / (kX1 - kX0)), _mm256_set1_pd(-1.0));
            const __m256d v =
                _mm256_fmsub_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(t, t), _mm256_set1_pd(1.0));
            const __m256d twoV = _mm256_add_pd(v, v);
            __m256d b1 = _mm256_set1_pd(c[2 * m]);
            __m256d b2 = _mm256_setzero_pd();

            for (int k = m - 1; k >= 1; --k)
            {
                const __m256d b0 =
                    _mm256_fmadd_pd(twoV, b1, _mm256_sub_pd(_mm256_set1_pd(c[2 * k]), b2));
                b2 = b1;
                b1 = b0;
            }

            const __m256d even = _mm256_fmadd_pd(v, b1, _mm256_sub_pd(_mm256_set1_pd(c[0]), b2));
            __m256d o1 = _mm256_set1_pd(c[2 * m - 1]);
            __m256d o2 = _mm256_setzero_pd();

            for (int k = m - 2; k >= 1; --k)
            {
                const __m256d o0 =
                    _mm256_fmadd_pd(twoV, o1, _mm256_sub_pd(_mm256_set1_pd(c[2 * k + 1]), o2));
                o2 = o1;
                o1 = o0;
            }

            const __m256d odd = _mm256_fmadd_pd(_mm256_sub_pd(twoV, _mm256_set1_pd(1.0)),
                                                o1,
                                                _mm256_sub_pd(_mm256_set1_pd(c[1]), o2));
            __m256d f = _mm256_fmadd_pd(t, odd, even);
            const __m256d expx = expTable.Eval4(xv);
            const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);

            for (int l = 0; l < n; ++l)
            {
                f = _mm256_fmadd_pd(
                    _mm256_set1_pd(l + 0.5), f, _mm256_mul_pd(_mm256_set1_pd(-0.5), expx));
                f = _mm256_mul_pd(f, invx);
            }

            fB = f;
        }

        __m256d fC = _mm256_setzero_pd();
        {
            const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);
            __m256d f = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_sqrt_pd(invx));

            for (int l = 0; l < n; ++l)
            {
                f = _mm256_mul_pd(_mm256_mul_pd(f, _mm256_set1_pd(l + 0.5)), invx);
            }

            fC = f;
        }

        __m256d res = _mm256_blendv_pd(fC, fB, mBonly);
        res = _mm256_blendv_pd(res, fA, mA);
        _mm256_storeu_pd(out + i, res);
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        out[i] = qcx::integrals::BoysSingle(n, x[i]);
    }
}

double MaxAbsError(const double* out, const double* x, std::size_t count, double* worstX) {
    double worst = 0.0;
    double xAtWorst = 0.0;

    for (std::size_t i = 0; i < count; ++i)
    {
        const double err = std::abs(out[i] - qcx::integrals::BoysSingle(kOrder, x[i]));

        if (err > worst)
        {
            worst = err;
            xAtWorst = x[i];
        }
    }

    *worstX = xAtWorst;
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    const bool selfCheck = argc > 1 && std::strcmp(argv[1], "--self-check") == 0;

    // The design-study unsorted workload: x uniform in [0, 40], rng(46),
    // order 8 for the whole array.
    std::mt19937_64 rng(46);
    std::uniform_real_distribution<double> xd(0.0, 40.0);
    std::vector<double> x(kInputCount);

    for (auto& value : x)
    {
        value = xd(rng);
    }

    std::vector<double> out(kInputCount);
    const ExpTable expTable;

    if (selfCheck)
    {
        ChebSimdMixed(kOrder, x.data(), out.data(), kInputCount, expTable);
        double xAtWorst = 0.0;
        const double maxErr = MaxAbsError(out.data(), x.data(), kInputCount, &xAtWorst);
        const bool pass = maxErr <= 5.5e-14;
        std::printf(
            "self-check: cheb-simd-unsorted-n8 | max_abs_err: %.3e | budget: 5.50e-14 | %s\n",
            maxErr,
            pass ? "PASS" : "FAIL");
        std::printf("  worst at x = %.9f\n", xAtWorst);
        return pass ? 0 : 1;
    }

    // Runs-log protocol: warmup + 3 passes, min/median/max.
    ChebSimdMixed(kOrder, x.data(), out.data(), kInputCount, expTable);
    std::vector<double> passes(kPasses);

    for (int p = 0; p < kPasses; ++p)
    {
        const auto t0 = std::chrono::steady_clock::now();
        ChebSimdMixed(kOrder, x.data(), out.data(), kInputCount, expTable);
        const auto t1 = std::chrono::steady_clock::now();
        passes[p] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::sort(passes.begin(), passes.end());
    const double median = passes[1];
    // count/median is items per millisecond = 1e3 items/s; the /1e3 below is
    // what makes the printed value the unit its field names (Mvals/s), at the
    // three decimals the runs log's rows carry.
    std::printf(
        "kernel: cheb-simd-unsorted-n8 | workload: uniform-x40-n8 | count: %zu | passes: %d | "
        "min_ms: %.3f | median_ms: %.3f | max_ms: %.3f | median_Mvals_per_s: %.3f\n",
        kInputCount,
        kPasses,
        passes.front(),
        median,
        passes.back(),
        kInputCount / median / 1e3);
    return 0;
}
