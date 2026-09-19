#include "boys_coefficients.hpp"
#include "qcx/integrals/boys.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct ReferenceRow {
    int n;
    double x;
    double value;
};

// The committed reference grid (tools/gen_boys_coefficients.py, 30-digit
// mpmath values rounded to double). Definitive accuracy gate for the kernel.
std::vector<ReferenceRow> LoadReference() {
    const std::string path = std::string(QcxBoysDataDir) + "/boys_reference.csv";
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<ReferenceRow> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        ReferenceRow row{};
        std::getline(ss, cell, ',');
        row.n = std::stoi(cell);
        std::getline(ss, cell, ',');
        row.x = std::strtod(cell.c_str(), nullptr);
        std::getline(ss, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);
        // The committed grid ends at x = 100: beyond it every F_n is covered
        // by region C's asymptotic form (verified against mpmath up to x =
        // 100). The filter guards against a future grid extension past the
        // series' convergence limit (~x = 250 in the generator).
        if (row.x <= 100.0)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

// Tolerance with a small margin over the design target: the reference values
// carry ~1e-17 double-rounding, and the fits are validated at 4.5e-14.
constexpr double kDoubleTolerance = 5.5e-14;
constexpr float kFloatTolerance = 1.5e-7f;

// Per-region worst bounds for the double single lane (paper tab:accuracy
// cells; design-study measured worsts in parentheses):
// region A <= 1e-15 (6.7e-16), region B <= 3e-14 (2.9e-14), region C shares
// the overall 5.5e-14 (5.0e-14 at (32, x1)). Each bound matches the paper
// cell within one significant digit.
constexpr double kRegionATolerance = 1e-15;
constexpr double kRegionBTolerance = 3e-14;

// Region bucketing per the paper's tab:accuracy caption: region A x < kX0,
// region B kX0 <= x < kX1, region C x >= kX1 (the x = kX1 and x = 100 grid
// rows land in C, so the paper's "(32, x1)" worst sits in region C, the
// shared asymptotic cutoff). The extended band [kExtendedBX0, kX0) is its
// own region (E): the per-range F0 seed + upward recursion serves it per
// kmax tier, carrying the region-B-style budgets. The boundaries are the
// shipped kernel's own (boys_coefficients.hpp kX0/kX1/kExtendedBX0).
enum class BoysRegion : std::uint8_t { A, B, C, E };

BoysRegion RegionOf(double x) {
    using qcx::integrals::detail::kExtendedBX0;
    using qcx::integrals::detail::kX0;
    using qcx::integrals::detail::kX1;

    if (x < kX0)
    {
        return x >= kExtendedBX0 ? BoysRegion::E : BoysRegion::A;
    }

    if (x < kX1)
    {
        return BoysRegion::B;
    }

    return BoysRegion::C;
}

// Per-region worst-error accumulator over the committed reference grid.
struct RegionWorsts {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double e = 0.0;

    // (error, x) are the candidate error and its x - the per-region max
    // accumulator's pair.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void Update(double error, double x) {
        switch (RegionOf(x))
        {
        case BoysRegion::A:
            a = std::max(a, error);
            break;

        case BoysRegion::B:
            b = std::max(b, error);
            break;

        case BoysRegion::C:
            c = std::max(c, error);
            break;

        case BoysRegion::E:
            e = std::max(e, error);
            break;
        }
    }
};

std::vector<ReferenceRow> gReference = LoadReference();

#if QcxIntegralsFp16
// The fp16 lane tests reuse the certified accuracy targets in their
// absolute sense: the reference is the certified double lane (5e-14)
// evaluated at the fp16-rounded argument, and the tolerance is the fp16
// lanes' asserted bound — the 1e-7 base of the fp16 bound formula plus one
// half-ULP of the fp16-rounded reference, strictly stronger than the
// error-bounded 1.5e-7 + ½ULP contract (the fp16 output quantizes at
// ~1e-3 near x = 0, far above any 1e-7 absolute assertion).
double HalfUlp(qcx::integrals::F16 x) {
    return 0.5 * (static_cast<double>(qcx::integrals::NextUp(x)) - static_cast<double>(x));
}

double HalfUlp(qcx::integrals::Bf16 x) {
    return 0.5 * (static_cast<double>(qcx::integrals::NextUp(x)) - static_cast<double>(x));
}

// Reference-grid sweep shared by the F16/Bf16 single and batch tests. The
// half type is the API, so the engine receives the fp16-rounded argument;
// the reference is therefore the certified double lane (5e-14)
// evaluated at that same rounded argument, and the tolerance is the fp16
// lanes' asserted bound — the 1e-7 base of the fp16 bound formula plus one
// half-ULP of the fp16-rounded reference (strictly stronger than the
// error-bounded 1.5e-7 + ½ULP contract).
template <typename Half,
          Half (*SingleFn)(int, Half) noexcept,
          void (*BatchFn)(int, Half, Half*) noexcept>
void RunReferenceChecks(const char* label) {
    double worstSingle = 0.0;
    double worstBatch = 0.0;
    RegionWorsts singleWorst;
    RegionWorsts batchWorst;
    RegionWorsts singleWorstTolerance;
    RegionWorsts batchWorstTolerance;
    std::vector<Half> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const auto& row : gReference)
    {
        const Half x = static_cast<Half>(row.x);
        const double reference = qcx::integrals::BoysSingle(row.n, static_cast<double>(x));
        const Half half = static_cast<Half>(reference);
        const double tolerance = 1e-7 + HalfUlp(half);
        const double single = static_cast<double>(SingleFn(row.n, x));
        const double singleError = std::abs(single - reference);
        EXPECT_LE(singleError, tolerance)
            << "n=" << row.n << " x=" << row.x << " got=" << single << " want=" << reference;
        worstSingle = std::max(worstSingle, singleError);
        singleWorst.Update(singleError, row.x);
        singleWorstTolerance.Update(tolerance, row.x);
        BatchFn(row.n, x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            // The row tolerance is tied to F_{row.n}(x), which underflows to
            // fp16 zero at large x for high orders; each batch element is
            // quantized at its own F_k(x), so it needs its own half-ULP.
            const double batchReference = qcx::integrals::BoysSingle(k, static_cast<double>(x));
            const double batchTolerance = 1e-7 + HalfUlp(static_cast<Half>(batchReference));
            const double batchError = std::abs(static_cast<double>(batch[k]) - batchReference);
            EXPECT_LE(batchError, batchTolerance)
                << "batch F" << k << " at x=" << row.x << " got=" << static_cast<double>(batch[k])
                << " want=" << batchReference;
            worstBatch = std::max(worstBatch, batchError);
            batchWorst.Update(batchError, row.x);
            batchWorstTolerance.Update(batchTolerance, row.x);
        }
    }

    // fp16 tab:accuracy cells (full fp16 treatment): per-region worst
    // <= 1e-7 + one half-ULP of representation (the certified
    // mixed-precision contract, budget-style cells). Each
    // region's bound is its max per-value budget; the per-value asserts above
    // imply it, and the printed worsts go to the runs record.
    EXPECT_LE(singleWorst.a, singleWorstTolerance.a) << "fp16 single region A (x < kX0)";
    EXPECT_LE(singleWorst.b, singleWorstTolerance.b) << "fp16 single region B (kX0 <= x < kX1)";
    EXPECT_LE(singleWorst.c, singleWorstTolerance.c) << "fp16 single region C (x >= kX1)";
    EXPECT_LE(singleWorst.e, singleWorstTolerance.e)
        << "fp16 single extended band (kExtendedBX0 <= x < kX0)";
    EXPECT_LE(batchWorst.a, batchWorstTolerance.a) << "fp16 batch region A (x < kX0)";
    EXPECT_LE(batchWorst.b, batchWorstTolerance.b) << "fp16 batch region B (kX0 <= x < kX1)";
    EXPECT_LE(batchWorst.c, batchWorstTolerance.c) << "fp16 batch region C (x >= kX1)";
    EXPECT_LE(batchWorst.e, batchWorstTolerance.e)
        << "fp16 batch extended band (kExtendedBX0 <= x < kX0)";

    std::printf(
        "%s: worst single |error| = %.3e (region A %.3e, region B %.3e, region C %.3e, "
        "extended band %.3e), "
        "worst batch |error| = %.3e (region A %.3e, region B %.3e, region C %.3e, "
        "extended band %.3e); "
        "single region budgets %.3e/%.3e/%.3e/%.3e, batch region budgets %.3e/%.3e/%.3e/%.3e\n",
        label,
        worstSingle,
        singleWorst.a,
        singleWorst.b,
        singleWorst.c,
        singleWorst.e,
        worstBatch,
        batchWorst.a,
        batchWorst.b,
        batchWorst.c,
        batchWorst.e,
        singleWorstTolerance.a,
        singleWorstTolerance.b,
        singleWorstTolerance.c,
        singleWorstTolerance.e,
        batchWorstTolerance.a,
        batchWorstTolerance.b,
        batchWorstTolerance.c,
        batchWorstTolerance.e);
}

// SIMD-lane sweep shared by the F16/Bf16 tests: each region lane must agree
// with the scalar fp16 lane within one half-ULP (both compute in the F32
// engine and round to the same half type). The draw ranges keep the
// fp16-rounded arguments inside their region bands (fp16 spacing near x0 is
// 2^-7, near x1 2^-6 - the double ranges used above would round across the
// region boundaries).
template <typename Half,
          Half (*SingleFn)(int, Half) noexcept,
          void (*BatchFn)(int, Half, Half*) noexcept,
          void (*SimdAFn)(int, const Half*, Half*, std::size_t) noexcept,
          void (*SimdBFn)(int, const Half*, Half*, std::size_t) noexcept,
          void (*SimdCFn)(int, const Half*, Half*, std::size_t) noexcept>
void RunSimdLaneChecks() {
    std::mt19937_64 rng(97531);
    constexpr std::size_t kCount = 4096;
    const int n = 8;
    std::vector<Half> x(kCount);
    std::vector<Half> out(kCount);
    std::vector<Half> batchOut(kCount * (qcx::integrals::kMaxBoysOrder + 1));

    // Region A (x < x0): same-n array. The tolerance is one full ULP (and
    // the same in regions B and C below): the SIMD kernel maps x to the
    // Chebyshev argument with one fused multiply-add while the scalar lane
    // rounds each step separately, so the two F32 values can sit on opposite
    // sides of a half grid step when the fit value lands within ~1e-10 of an
    // fp16 rounding boundary (the F16 grid spans many binades across region
    // A, from 1/(2n+1) at x = 0 down to subnormals at x0).
    std::uniform_real_distribution<float> xdA(1e-4f, 11.85f);

    for (auto& v : x)
    {
        v = static_cast<Half>(xdA(rng));
    }

    SimdAFn(n, x.data(), out.data(), kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const Half scalar = SingleFn(n, x[i]);
        EXPECT_LE(std::abs(static_cast<float>(out[i]) - static_cast<float>(scalar)),
                  2.0 * HalfUlp(scalar))
            << "region A i=" << i << " x=" << static_cast<float>(x[i]);
    }

    // Region B (x0 <= x < x1): full batch layout.
    std::uniform_real_distribution<float> xdB(11.95f, 28.95f);

    for (auto& v : x)
    {
        v = static_cast<Half>(xdB(rng));
    }

    SimdBFn(n, x.data(), batchOut.data(), kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        std::array<Half, qcx::integrals::kMaxBoysOrder + 1> scalar{};
        BatchFn(n, x[i], scalar.data());

        for (int k = 0; k <= n; ++k)
        {
            EXPECT_LE(std::abs(static_cast<float>(batchOut[k * kCount + i]) -
                               static_cast<float>(scalar[k])),
                      2.0 * HalfUlp(scalar[k]))
                << "region B i=" << i << " k=" << k << " x=" << static_cast<float>(x[i]);
        }
    }

    // Region C (x >= x1).
    std::uniform_real_distribution<float> xdC(29.1f, 60.0f);

    for (auto& v : x)
    {
        v = static_cast<Half>(xdC(rng));
    }

    SimdCFn(n, x.data(), out.data(), kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const Half scalar = SingleFn(n, x[i]);
        EXPECT_LE(std::abs(static_cast<float>(out[i]) - static_cast<float>(scalar)),
                  2.0 * HalfUlp(scalar))
            << "region C i=" << i << " x=" << static_cast<float>(x[i]);
    }

    // Tail fallbacks: counts that are not multiples of eight exercise the
    // scalar tail; count = 0 must be a no-op. Each region function receives
    // arguments from its own domain (the documented precondition).
    const std::array<Half, 5> tailA = {static_cast<Half>(1.0f),
                                       static_cast<Half>(2.0f),
                                       static_cast<Half>(5.0f),
                                       static_cast<Half>(8.0f),
                                       static_cast<Half>(11.0f)};
    const std::array<Half, 5> tailC = {static_cast<Half>(30.0f),
                                       static_cast<Half>(35.0f),
                                       static_cast<Half>(40.0f),
                                       static_cast<Half>(45.0f),
                                       static_cast<Half>(50.0f)};
    std::array<Half, 5> tailOut{};
    SimdAFn(n, tailA.data(), tailOut.data(), 5);

    for (std::size_t i = 0; i < 5; ++i)
    {
        EXPECT_LE(
            std::abs(static_cast<float>(tailOut[i]) - static_cast<float>(SingleFn(n, tailA[i]))),
            HalfUlp(SingleFn(n, tailA[i])));
    }

    SimdCFn(n, tailC.data(), tailOut.data(), 5);

    for (std::size_t i = 0; i < 5; ++i)
    {
        EXPECT_LE(
            std::abs(static_cast<float>(tailOut[i]) - static_cast<float>(SingleFn(n, tailC[i]))),
            HalfUlp(SingleFn(n, tailC[i])));
    }

    SimdAFn(n, tailA.data(), tailOut.data(), 0);
    SimdBFn(n, tailA.data(), batchOut.data(), 0);
    SimdCFn(n, tailC.data(), tailOut.data(), 0);
}
#endif // QcxIntegralsFp16

} // namespace

TEST(BoysTest, ReferenceDataLoaded) {
    EXPECT_GT(gReference.size(), 500u);
}

TEST(BoysTest, SingleMatchesReferenceDouble) {
    double worst = 0.0;
    RegionWorsts regionWorst;

    for (const auto& row : gReference)
    {
        const double value = qcx::integrals::BoysSingle(row.n, row.x);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, kDoubleTolerance)
            << "n=" << row.n << " x=" << row.x << " got=" << value << " want=" << row.value;
        worst = std::max(worst, error);
        regionWorst.Update(error, row.x);
    }

    // Per-region worsts over the committed grid against the paper's
    // tab:accuracy cells (see kRegionATolerance/kRegionBTolerance above).
    EXPECT_LE(regionWorst.a, kRegionATolerance) << "region A (x < kX0)";
    EXPECT_LE(regionWorst.b, kRegionBTolerance) << "region B (kX0 <= x < kX1)";
    EXPECT_LE(regionWorst.c, kDoubleTolerance) << "region C (x >= kX1)";
    EXPECT_LE(regionWorst.e, kRegionBTolerance) << "extended band (kExtendedBX0 <= x < kX0)";

    std::printf("BoysSingle: worst |error| = %.3e (region A %.3e, region B %.3e, region C %.3e, "
                "extended band %.3e)\n",
                worst,
                regionWorst.a,
                regionWorst.b,
                regionWorst.c,
                regionWorst.e);
}

TEST(BoysTest, BatchMatchesReferenceDouble) {
    // The batch downward recursion is the path the weighted region-A fits
    // exist for; it must hold the same tolerance as the single evaluations.
    double worst = 0.0;
    RegionWorsts regionWorst;
    std::vector<double> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const auto& row : gReference)
    {
        qcx::integrals::BoysBatch(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            // Compare against the single evaluation of the same order from the
            // reference grid: fetch the reference value for (k, row.x).
            double reference = 0.0;

            for (const auto& other : gReference)
            {
                if (other.n == k && other.x == row.x)
                {
                    reference = other.value;
                    break;
                }
            }

            const double error = std::abs(batch[k] - reference);
            EXPECT_LE(error, kDoubleTolerance) << "batch F" << k << " at x=" << row.x
                                               << " got=" << batch[k] << " want=" << reference;
            worst = std::max(worst, error);
            regionWorst.Update(error, row.x);
        }
    }

    // The batch lane asserts its own merged 5.5e-14 cell per region: the
    // weighted region-A worst was measured at 1.6e-14, so a 1e-15
    // single-lane bound would not hold for the batch.
    EXPECT_LE(regionWorst.a, kDoubleTolerance) << "batch region A (x < kX0)";
    EXPECT_LE(regionWorst.b, kDoubleTolerance) << "batch region B (kX0 <= x < kX1)";
    EXPECT_LE(regionWorst.c, kDoubleTolerance) << "batch region C (x >= kX1)";
    EXPECT_LE(regionWorst.e, kDoubleTolerance) << "batch extended band (kExtendedBX0 <= x < kX0)";

    std::printf("BoysBatch: worst |error| = %.3e (region A %.3e, region B %.3e, region C %.3e, "
                "extended band %.3e)\n",
                worst,
                regionWorst.a,
                regionWorst.b,
                regionWorst.c,
                regionWorst.e);
}

TEST(BoysTest, SingleMatchesReferenceFloat) {
    float worst = 0.0f;
    RegionWorsts regionWorst;

    for (const auto& row : gReference)
    {
        if (row.x > 100.0)
        {
            continue;
        }

        const float value = qcx::integrals::BoysSingleF32(row.n, static_cast<float>(row.x));
        const float error = std::abs(value - static_cast<float>(row.value));
        EXPECT_LE(error, kFloatTolerance)
            << "n=" << row.n << " x=" << row.x << " got=" << value << " want=" << row.value;
        worst = std::max(worst, error);
        regionWorst.Update(static_cast<double>(error), row.x);
    }

    // Float tab:accuracy rows are budget-style cells: <= 1.5e-7 per region.
    EXPECT_LE(regionWorst.a, static_cast<double>(kFloatTolerance)) << "region A (x < kX0)";
    EXPECT_LE(regionWorst.b, static_cast<double>(kFloatTolerance)) << "region B (kX0 <= x < kX1)";
    EXPECT_LE(regionWorst.c, static_cast<double>(kFloatTolerance)) << "region C (x >= kX1)";
    EXPECT_LE(regionWorst.e, static_cast<double>(kFloatTolerance))
        << "extended band (kExtendedBX0 <= x < kX0)";

    std::printf("BoysSingleF32: worst |error| = %.3e (region A %.3e, region B %.3e, region C %.3e, "
                "extended band %.3e)\n",
                worst,
                regionWorst.a,
                regionWorst.b,
                regionWorst.c,
                regionWorst.e);
}

TEST(BoysTest, BatchMatchesReferenceFloat) {
    float worst = 0.0f;
    RegionWorsts regionWorst;
    std::vector<float> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const auto& row : gReference)
    {
        if (row.x > 100.0)
        {
            continue;
        }

        qcx::integrals::BoysBatchF32(row.n, static_cast<float>(row.x), batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            double reference = 0.0;

            for (const auto& other : gReference)
            {
                if (other.n == k && other.x == row.x)
                {
                    reference = other.value;
                    break;
                }
            }

            const float error = std::abs(batch[k] - static_cast<float>(reference));
            EXPECT_LE(error, kFloatTolerance) << "batch F" << k << " at x=" << row.x
                                              << " got=" << batch[k] << " want=" << reference;
            worst = std::max(worst, error);
            regionWorst.Update(static_cast<double>(error), row.x);
        }
    }

    // Float tab:accuracy rows are budget-style cells: <= 1.5e-7 per region.
    EXPECT_LE(regionWorst.a, static_cast<double>(kFloatTolerance)) << "batch region A (x < kX0)";
    EXPECT_LE(regionWorst.b, static_cast<double>(kFloatTolerance))
        << "batch region B (kX0 <= x < kX1)";
    EXPECT_LE(regionWorst.c, static_cast<double>(kFloatTolerance)) << "batch region C (x >= kX1)";
    EXPECT_LE(regionWorst.e, static_cast<double>(kFloatTolerance))
        << "batch extended band (kExtendedBX0 <= x < kX0)";

    std::printf("BoysBatchF32: worst |error| = %.3e (region A %.3e, region B %.3e, region C %.3e, "
                "extended band %.3e)\n",
                worst,
                regionWorst.a,
                regionWorst.b,
                regionWorst.c,
                regionWorst.e);
}

TEST(BoysTest, ZeroArgumentIsExact) {
    for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
    {
        EXPECT_DOUBLE_EQ(qcx::integrals::BoysSingle(n, 0.0), 1.0 / (2.0 * n + 1.0));
        EXPECT_FLOAT_EQ(qcx::integrals::BoysSingleF32(n, 0.0f),
                        1.0f / (2.0f * static_cast<float>(n) + 1.0f));
    }

    double batch[qcx::integrals::kMaxBoysOrder + 1];
    qcx::integrals::BoysBatch(8, 0.0, batch);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_DOUBLE_EQ(batch[k], 1.0 / (2.0 * k + 1.0));
    }

    float batchF32[qcx::integrals::kMaxBoysOrder + 1];
    qcx::integrals::BoysBatchF32(8, 0.0f, batchF32);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_FLOAT_EQ(batchF32[k], 1.0f / (2.0f * static_cast<float>(k) + 1.0f));
    }
}

TEST(BoysTest, BatchConsistentWithSingleDouble) {
    // Different arithmetic paths (seed + recursion vs. per-order fits) must
    // agree within the combined error bound.
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> xd(1e-4, 40.0);
    std::vector<double> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const double x = xd(rng);
        const int nmax = static_cast<int>(rng() % (qcx::integrals::kMaxBoysOrder + 1));
        qcx::integrals::BoysBatch(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const double single = qcx::integrals::BoysSingle(k, x);
            EXPECT_LE(std::abs(batch[k] - single), 1e-13) << "n=" << k << " x=" << x;
        }
    }
}

TEST(BoysTest, AsymptoticBehavior) {
    // F_0(x) ~ 1/2 sqrt(pi/x) for large x; values decay monotonically.
    for (double x : {30.0, 40.0, 50.0, 100.0})
    {
        const double f0 = qcx::integrals::BoysSingle(0, x);
        EXPECT_NEAR(f0, 0.886226925452758014 / std::sqrt(x), 5e-14);
    }

    double previous = 2.0;

    for (int i = 0; i <= 100; ++i)
    {
        const double x = 0.1 * i;
        const double f0 = qcx::integrals::BoysSingle(0, x);
        EXPECT_LE(f0, previous);
        previous = f0;
    }
}

// ---------------------------------------------------------------------------
// Region-B exp-Taylor gather-table pin (the paper: "on [x0, x1) its measured
// worst absolute error is 1.8e-17"; the SIMD lane measured 1.83e-17 at
// x ~ 11.99). The shipped ExpTable is private to its TU; this replica
// is deliberately independent, replicating the
// construction exactly: 3001 rows at step 0.01, row i centered on
// z = -(i * 0.01), degree-4 Taylor coefficients of e^{-x} built with
// std::exp. Evaluation mirrors the SIMD chain's Horner form
// (alternating-sign convention):
// c4*x^4 - c3*x^3 + c2*x^2 - c1*x + c0, with row index
// i = min(3000, trunc(x / 0.01)) (the _mm256_cvtpd_epi32 semantics; the
// clamp is inert on [kX0, kX1)).
// ---------------------------------------------------------------------------
struct ExpTaylorReplica {
    static constexpr double kStep = 0.01;
    static constexpr int kNumPoints = 3000;
    std::array<std::array<double, 5>, kNumPoints + 1> c{};

    ExpTaylorReplica() {
        for (int i = 0; i <= kNumPoints; ++i)
        {
            const double z = -(i * kStep);
            const double z2 = z * z / 2;
            const double z3 = z2 * z / 3;
            const double z4 = z3 * z / 4;
            const double e = std::exp(z);
            c[i][0] = e * (1 - z + z2 - z3 + z4);
            c[i][1] = e * (1 - z + z2 - z3);
            c[i][2] = e * (1 - z + z2) / 2;
            c[i][3] = e * (1 - z) / 6;
            c[i][4] = e / 24;
        }
    }

    double Eval(double x) const {
        int i = std::min(static_cast<int>(x / kStep), kNumPoints);
        const double c0 = c[i][0];
        const double c1 = c[i][1];
        const double c2 = c[i][2];
        const double c3 = c[i][3];
        const double c4 = c[i][4];
        double result = c4 * x - c3;
        result = result * x + c2;
        result = result * x - c1;
        result = result * x + c0;
        return result;
    }
};

TEST(BoysTest, ExpTaylorGatherTableRegionB) {
    using qcx::integrals::detail::kX0;
    using qcx::integrals::detail::kX1;

    const ExpTaylorReplica table;
    double worst = 0.0;
    double worstX = 0.0;

    // Dense deterministic sweep of [kX0, kX1) at 1e-4 (the error is smooth,
    // so the row-midpoint peaks are caught far inside the pin's headroom).
    constexpr double kSweepStep = 1e-4;

    for (int j = 0;; ++j)
    {
        const double x = kX0 + j * kSweepStep;

        if (x >= kX1)
        {
            break;
        }

        const double error = std::abs(table.Eval(x) - std::exp(-x));

        if (error > worst)
        {
            worst = error;
            worstX = x;
        }
    }

    // One order above the measured 1.83e-17 (still ~500x below the 5e-14
    // target). If the printed worst drifts materially above 1.83e-17 (libm
    // differences in std::exp feed the table construction), the paper cell
    // is updated to the measured value — the paper follows the measurements.
    EXPECT_LE(worst, 1e-16) << "exp-Taylor worst at x=" << worstX;
    std::printf("ExpTaylor[%g, %g): worst |error| = %.3e at x = %.6f\n", kX0, kX1, worst, worstX);
}

TEST(BoysTest, FootprintSizes) {
    namespace detail = qcx::integrals::detail;

    // tab:throughput/Discussion footprint cells (paper: ~18 KB Chebyshev,
    // i.e. 17,904 B incl. metadata — ~192 KB region-B gather table, ~5 MB
    // flat Taylor table). The first two
    // are computed from the committed tables; the flat-table cell follows
    // the design study's comparator geometry (maxn = 24, step 0.01,
    // limit = 50 -> nx = 5000):
    //   _b: (maxn + 1) x (nx + 1) x 5 doubles per order
    //   _c: (nx + 1) x 6 doubles (the e^{-x} degree-5 companion table)
    const std::size_t chebyshevBytes =
        detail::kCoeffs.size() * sizeof(double) + detail::kBcoeffs.size() * sizeof(double) +
        detail::f32::kCoeffs.size() * sizeof(float) + detail::f32::kBcoeffs.size() * sizeof(float);
    const std::size_t pieceMetadataBytes =
        detail::kPieces.size() * sizeof(detail::OrderPiece) +
        detail::kPieceStart.size() * sizeof(int) +
        detail::f32::kPieces.size() * sizeof(detail::f32::OrderPiece) +
        detail::f32::kPieceStart.size() * sizeof(int);
    const std::size_t gatherBytes = std::size_t{3001} * 8u * sizeof(double);
    const std::size_t flatTaylorBytes = std::size_t{25} * 5001u * 5u * sizeof(double);
    const std::size_t flatExpBytes = std::size_t{5001} * 6u * sizeof(double);

    std::printf("footprint: Chebyshev coefficients %zu B (%.1f KB) + piece metadata %zu B "
                "(%.1f KB); region-B gather table %zu B (%.1f KB); flat Taylor table %zu B "
                "(%.1f MB) + e^-x table %zu B (%.1f KB)\n",
                chebyshevBytes,
                chebyshevBytes / 1000.0,
                pieceMetadataBytes,
                pieceMetadataBytes / 1000.0,
                gatherBytes,
                gatherBytes / 1000.0,
                flatTaylorBytes,
                flatTaylorBytes / 1.0e6,
                flatExpBytes,
                flatExpBytes / 1000.0);
}

TEST(BoysTest, SimdMatchesScalarWhenAvailable) {
    if (!qcx::integrals::BoysAvx2Available())
    {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }

    std::mt19937_64 rng(67890);
    constexpr std::size_t kCount = 4096;
    std::vector<double> x(kCount);
    std::vector<double> out(kCount);
    std::vector<double> batchOut(kCount * (qcx::integrals::kMaxBoysOrder + 1));

    // Region A (x < x0): same-n array. The SIMD lane serves the extended
    // band with the region-A per-order fits (its untouched path); the
    // scalar single serves it with the per-range extended seed, so the
    // band draws agree within the two paths' combined budgets (3e-14 +
    // 1e-15) while the below-band draws keep the old bit-close 1e-15.
    std::uniform_real_distribution<double> xdA(1e-4, 11.89);

    for (auto& v : x)
    {
        v = xdA(rng);
    }

    const int n = 8;
    qcx::integrals::BoysRegionASimd(n, x.data(), out.data(), kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const double tolerance =
            x[i] >= qcx::integrals::detail::kExtendedBX0 ? 3e-14 + 1e-15 : 1e-15;
        EXPECT_NEAR(out[i], qcx::integrals::BoysSingle(n, x[i]), tolerance);
    }

    // Region B (x0 <= x < x1): full batch layout.
    std::uniform_real_distribution<double> xdB(11.90, 28.98);

    for (auto& v : x)
    {
        v = xdB(rng);
    }

    qcx::integrals::BoysRegionBSimd(n, x.data(), batchOut.data(), kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        double scalar[qcx::integrals::kMaxBoysOrder + 1];
        qcx::integrals::BoysBatch(n, x[i], scalar);

        for (int k = 0; k <= n; ++k)
        {
            EXPECT_NEAR(batchOut[k * kCount + i], scalar[k], 1e-15);
        }
    }

    // Region C (x >= x1).
    std::uniform_real_distribution<double> xdC(28.99, 60.0);

    for (auto& v : x)
    {
        v = xdC(rng);
    }

    qcx::integrals::BoysRegionCSimd(n, x.data(), out.data(), kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        EXPECT_NEAR(out[i], qcx::integrals::BoysSingle(n, x[i]), 1e-15);
    }

    // Tail fallbacks: counts that are not multiples of four (count = 5
    // exercises the scalar tail; count = 0 must be a no-op). Each region
    // function receives arguments from its own domain (the documented
    // precondition). The band draws of the A tail use the combined
    // SIMD-region-A/scalar-extended budgets, as in the main loop above.
    double tailA[5] = {1.0, 2.0, 5.0, 8.0, 11.0};
    double tailC[5] = {30.0, 35.0, 40.0, 45.0, 50.0};
    double tailOut[5] = {};
    qcx::integrals::BoysRegionASimd(n, tailA, tailOut, 5);

    for (std::size_t i = 0; i < 5; ++i)
    {
        const double tolerance =
            tailA[i] >= qcx::integrals::detail::kExtendedBX0 ? 3e-14 + 1e-15 : 1e-15;
        EXPECT_NEAR(tailOut[i], qcx::integrals::BoysSingle(n, tailA[i]), tolerance);
    }

    qcx::integrals::BoysRegionCSimd(n, tailC, tailOut, 5);

    for (std::size_t i = 0; i < 5; ++i)
    {
        EXPECT_NEAR(tailOut[i], qcx::integrals::BoysSingle(n, tailC[i]), 1e-15);
    }

    qcx::integrals::BoysRegionASimd(n, tailA, tailOut, 0);
    qcx::integrals::BoysRegionBSimd(n, tailA, batchOut.data(), 0);
    qcx::integrals::BoysRegionCSimd(n, tailC, tailOut, 0);
}

#if QcxIntegralsFp16
TEST(BoysTest, SingleMatchesReferenceF16) {
    RunReferenceChecks<qcx::integrals::F16,
                       qcx::integrals::BoysSingleF16,
                       qcx::integrals::BoysBatchF16>("BoysF16");
}

TEST(BoysTest, BatchMatchesReferenceF16) {
    // Covered by the single sweep's batch half; this test name documents the
    // batch gate explicitly for the fp16 lane.
    RunReferenceChecks<qcx::integrals::F16,
                       qcx::integrals::BoysSingleF16,
                       qcx::integrals::BoysBatchF16>("BoysF16(batch)");
}

TEST(BoysTest, SingleMatchesReferenceBf16) {
    RunReferenceChecks<qcx::integrals::Bf16,
                       qcx::integrals::BoysSingleBf16,
                       qcx::integrals::BoysBatchBf16>("BoysBf16");
}

TEST(BoysTest, BatchMatchesReferenceBf16) {
    RunReferenceChecks<qcx::integrals::Bf16,
                       qcx::integrals::BoysSingleBf16,
                       qcx::integrals::BoysBatchBf16>("BoysBf16(batch)");
}

TEST(BoysTest, ZeroArgumentIsExactF16) {
    // The engine computes in float: the exact value 1/(2n+1) must survive to
    // the fp16 output up to one half-ULP of quantization.
    for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
    {
        const qcx::integrals::F16 got = qcx::integrals::BoysSingleF16(n, qcx::integrals::F16{0.0f});
        const qcx::integrals::F16 want =
            static_cast<qcx::integrals::F16>(1.0f / (2.0f * static_cast<float>(n) + 1.0f));
        EXPECT_LE(std::abs(static_cast<float>(got) - static_cast<float>(want)), HalfUlp(want))
            << "n=" << n;
    }

    std::array<qcx::integrals::F16, qcx::integrals::kMaxBoysOrder + 1> batchF16{};
    qcx::integrals::BoysBatchF16(8, qcx::integrals::F16{0.0f}, batchF16.data());

    for (int k = 0; k <= 8; ++k)
    {
        const qcx::integrals::F16 want =
            static_cast<qcx::integrals::F16>(1.0f / (2.0f * static_cast<float>(k) + 1.0f));
        EXPECT_LE(std::abs(static_cast<float>(batchF16[k]) - static_cast<float>(want)),
                  HalfUlp(want))
            << "k=" << k;
    }
}

TEST(BoysTest, ZeroArgumentIsExactBf16) {
    for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
    {
        const qcx::integrals::Bf16 got =
            qcx::integrals::BoysSingleBf16(n, qcx::integrals::Bf16{0.0f});
        const qcx::integrals::Bf16 want =
            static_cast<qcx::integrals::Bf16>(1.0f / (2.0f * static_cast<float>(n) + 1.0f));
        EXPECT_LE(std::abs(static_cast<float>(got) - static_cast<float>(want)), HalfUlp(want))
            << "n=" << n;
    }

    std::array<qcx::integrals::Bf16, qcx::integrals::kMaxBoysOrder + 1> batchBf16{};
    qcx::integrals::BoysBatchBf16(8, qcx::integrals::Bf16{0.0f}, batchBf16.data());

    for (int k = 0; k <= 8; ++k)
    {
        const qcx::integrals::Bf16 want =
            static_cast<qcx::integrals::Bf16>(1.0f / (2.0f * static_cast<float>(k) + 1.0f));
        EXPECT_LE(std::abs(static_cast<float>(batchBf16[k]) - static_cast<float>(want)),
                  HalfUlp(want))
            << "k=" << k;
    }
}

TEST(BoysTest, BatchConsistentWithSingleF16) {
    // Seed + downward recursion (batch) and per-order fits (single) are
    // independent float paths, each within its own 1e-7 absolute budget of
    // the certified value; where F_n is tiny their sum can straddle an fp16
    // rounding boundary, so the agreement bound is two budgets plus one ULP
    // of the fp16 output (the mixed-precision contract applied to a
    // lane-vs-lane check).
    std::mt19937_64 rng(24680);
    std::uniform_real_distribution<float> xd(1e-4f, 100.0f);
    std::vector<qcx::integrals::F16> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const qcx::integrals::F16 x = static_cast<qcx::integrals::F16>(xd(rng));
        const int nmax = static_cast<int>(rng() % (qcx::integrals::kMaxBoysOrder + 1));
        qcx::integrals::BoysBatchF16(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const qcx::integrals::F16 single = qcx::integrals::BoysSingleF16(k, x);
            EXPECT_LE(std::abs(static_cast<float>(batch[k]) - static_cast<float>(single)),
                      2.0 * (1e-7 + HalfUlp(single)))
                << "n=" << k << " x=" << static_cast<float>(x);
        }
    }
}

TEST(BoysTest, BatchConsistentWithSingleBf16) {
    // Same two-budgets-plus-one-ULP bound as the F16 variant: the batch
    // recursion and the per-order fits each hold the 1e-7 absolute budget,
    // and their sum can straddle a half grid step where F_n is tiny.
    std::mt19937_64 rng(24681);
    std::uniform_real_distribution<float> xd(1e-4f, 100.0f);
    std::vector<qcx::integrals::Bf16> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const qcx::integrals::Bf16 x = static_cast<qcx::integrals::Bf16>(xd(rng));
        const int nmax = static_cast<int>(rng() % (qcx::integrals::kMaxBoysOrder + 1));
        qcx::integrals::BoysBatchBf16(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const qcx::integrals::Bf16 single = qcx::integrals::BoysSingleBf16(k, x);
            EXPECT_LE(std::abs(static_cast<float>(batch[k]) - static_cast<float>(single)),
                      2.0 * (1e-7 + HalfUlp(single)))
                << "n=" << k << " x=" << static_cast<float>(x);
        }
    }
}

TEST(BoysTest, SimdMatchesScalarF16) {
    if (!qcx::integrals::BoysAvx2Available())
    {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }

    RunSimdLaneChecks<qcx::integrals::F16,
                      qcx::integrals::BoysSingleF16,
                      qcx::integrals::BoysBatchF16,
                      qcx::integrals::BoysRegionASimdF16,
                      qcx::integrals::BoysRegionBSimdF16,
                      qcx::integrals::BoysRegionCSimdF16>();
}

TEST(BoysTest, SimdMatchesScalarBf16) {
    if (!qcx::integrals::BoysAvx2Available())
    {
        GTEST_SKIP() << "AVX2 not available on this CPU";
    }

    RunSimdLaneChecks<qcx::integrals::Bf16,
                      qcx::integrals::BoysSingleBf16,
                      qcx::integrals::BoysBatchBf16,
                      qcx::integrals::BoysRegionASimdBf16,
                      qcx::integrals::BoysRegionBSimdBf16,
                      qcx::integrals::BoysRegionCSimdBf16>();
}
#endif // QcxIntegralsFp16
