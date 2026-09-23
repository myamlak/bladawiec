// The accuracy-parametrization contract tests.
//
// Sampled-m per-region contract: for m in {1, 2, 10, 1e2, 1e4, 1e8} x
// {double single, double batch, float single, float batch}, assert
// |F_hat_m - ref| <= m * B_region per region on the committed reference
// grid (region bucketing: A x < kX0, B kX0 <= x < kX1, C x >= kX1), with
// the asserted bounds B: double single 1e-15/3e-14/5.5e-14, double batch
// 5.5e-14 per region, float 1.5e-7 per region. The m = 1 rows re-assert
// the certified pins.
//
// The effective-degree tables are non-increasing in m (a larger
// budget may only truncate further) over all six lane roles and both
// regions, and the m = 1 tables are the full degrees (the m = 1 path is
// today's path by construction).
//
// The fp16/Bf16 lanes forward the multiplier to the F32 engine
// (I/O-only wrappers, no fp16-specific degree tables): assert
// |F_hat - F(x16)| <= m * 1e-7 + 1/2 ULP per value on the reference grid at
// sampled m, F(x16) being the certified double lane evaluated at the
// fp16-rounded argument (the reference lane).
//
// The sampled-m instantiations are compiled from the internal headers
// (boys_impl.hpp, boys_effective_degrees.hpp) — the library exports only
// the m = 1 instantiations; the m = 1 call sites below still route to the
// library's certified instantiations. The relaxed SIMD lanes consume the
// same constexpr degree tables as their scalar twins (region A:
// kDoubleSingle; region B: kDoubleBatch) and the same Clenshaw recursions
// with runtime degrees (the full-accuracy shape), so the grid contract
// below pins the mechanism the SIMD lanes share.
//
// The file lives alongside boys_test.cpp rather than inside it to keep that
// file's existing tests untouched (the split is editorial only — same test
// binary, same contract).

#include "boys_effective_degrees.hpp"
#include "boys_impl.hpp"
#include "qcx/integrals/boys.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using qcx::integrals::BoysAllOrders;
using qcx::integrals::BoysAllOrdersF32;
using qcx::integrals::BoysSingle;
using qcx::integrals::BoysSingleF32;
using qcx::integrals::detail::BoysRole;
using qcx::integrals::detail::kExtendedBX0;
using qcx::integrals::detail::kX0;
using qcx::integrals::detail::kX1;
using qcx::integrals::detail::RegionADegrees;
using qcx::integrals::detail::RegionBDegrees;

struct ReferenceRow {
    int n;
    double x;
    double value;
};

// The committed reference grid (tools/gen_boys_coefficients.py, 30-digit
// mpmath values rounded to double) — the same loader as boys_test.cpp.
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
        rows.push_back(row);
    }

    return rows;
}

// The grid holds every (n, x) pair of its x set (the generator emits all
// orders per x; the batch lanes look up F_k(x) for k <= nmax), so a
// per-order binary search is exact. Indexed once at startup.
struct ReferenceGrid {
    std::array<std::vector<ReferenceRow>, qcx::integrals::kMaxBoysOrder + 1> byN{};

    explicit ReferenceGrid(const std::vector<ReferenceRow>& rows) {
        for (const ReferenceRow& row : rows)
        {
            byN[static_cast<std::size_t>(row.n)].push_back(row);
        }

        for (auto& v : byN)
        {
            std::sort(v.begin(), v.end(), [](const ReferenceRow& a, const ReferenceRow& b) {
                return a.x < b.x;
            });
        }
    }

    // The committed value F_n(x); the pair is guaranteed present (the grid
    // is complete in n per x — see above).
    double Value(int n, double x) const {
        const std::vector<ReferenceRow>& v = byN[static_cast<std::size_t>(n)];
        const auto it = std::lower_bound(
            v.begin(), v.end(), x, [](const ReferenceRow& row, double xv) { return row.x < xv; });
        EXPECT_TRUE(it != v.end() && it->x == x) << "missing grid pair n=" << n << " x=" << x;
        return it != v.end() ? it->value : 0.0;
    }
};

enum class BoysRegion : std::uint8_t { A, B, C, E };

BoysRegion RegionOf(double x) {
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

enum class LaneKind : std::uint8_t { kDoubleSingle, kDoubleBatch, kFloatSingle, kFloatBatch };

// The asserted per-region bounds of the m = 1 contract. The extended band
// carries the region-B budget: at m = 1 its per-range F0 seed serves the
// certified upward recursion (the band is the m = 1 lane's; the m > 1
// branch keeps the region-A treatment there, whose tighter budget implies
// this cell at every m).
double RegionBound(BoysRegion region, LaneKind lane) {
    switch (lane)
    {
    case LaneKind::kDoubleSingle:

        switch (region)
        {
        case BoysRegion::A:
            return 1e-15;

        case BoysRegion::B:
        case BoysRegion::E:
            return 3e-14;

        case BoysRegion::C:
            return 5.5e-14;
        }

        break;

    case LaneKind::kDoubleBatch:
        return 5.5e-14;

    case LaneKind::kFloatSingle:
    case LaneKind::kFloatBatch:
        return 1.5e-7;
    }

    return 0.0; // unreachable
}

// Runs the callable once per sampled multiplier, the
// multiplier passed as a compile-time constant — the NTTP surface is the
// contract under test.
template <typename Fn> void ForEachSampledMultiplier(Fn&& fn) {
    fn.template operator()<1.0>();
    fn.template operator()<2.0>();
    fn.template operator()<10.0>();
    fn.template operator()<100.0>();
    fn.template operator()<1e4>();
    fn.template operator()<1e8>();
}

const std::vector<ReferenceRow> gReference = LoadReference();
const ReferenceGrid gGrid(gReference);

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

void PrintWorsts(const char* lane, double m, const RegionWorsts& worst) {
    std::printf("%s m=%.0e: worst region A %.3e, B %.3e, C %.3e, extended band %.3e\n",
                lane,
                m,
                worst.a,
                worst.b,
                worst.c,
                worst.e);
}

// ---------------------------------------------------------------------------
// The sampled-m grid contract, one templated sweep per lane
// ---------------------------------------------------------------------------

template <double kM> void SweepDoubleSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = BoysSingle<kM>(row.n, row.x);
        const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleSingle);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "m=" << kM << " n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("double single", kM, worst);
}

template <double kM> void SweepDoubleBatch() {
    RegionWorsts worst;
    std::vector<double> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrders<kM>(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleBatch);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "m=" << kM << " batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("double batch", kM, worst);
}

template <double kM> void SweepFloatSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const float value = BoysSingleF32<kM>(row.n, static_cast<float>(row.x));
        const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kFloatSingle);
        const double error = std::abs(static_cast<double>(value) - row.value);
        EXPECT_LE(error, bound) << "m=" << kM << " n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("float single", kM, worst);
}

template <double kM> void SweepFloatBatch() {
    RegionWorsts worst;
    std::vector<float> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrdersF32<kM>(row.n, static_cast<float>(row.x), batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kFloatBatch);
            const double error =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - reference);
            EXPECT_LE(error, bound)
                << "m=" << kM << " batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("float batch", kM, worst);
}

TEST(BoysAccuracyTest, DoubleSingleSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepDoubleSingle<kM>(); });
}

TEST(BoysAccuracyTest, DoubleBatchSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepDoubleBatch<kM>(); });
}

TEST(BoysAccuracyTest, FloatSingleSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepFloatSingle<kM>(); });
}

TEST(BoysAccuracyTest, FloatBatchSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepFloatBatch<kM>(); });
}

// ---------------------------------------------------------------------------
// The effective degrees: non-increasing in m, full at m = 1, inside
// the evaluator domain {0, 1, 2, 4, 6, ...} at every (m, role, point)
// ---------------------------------------------------------------------------

constexpr bool InEvaluatorDomain(int d) {
    return d == 0 || d == 1 || d == 2 || (d >= 4 && d % 2 == 0);
}

template <BoysRole kRole> void AssertRoleTables() {
    constexpr auto a1 = RegionADegrees<1.0, kRole>();
    constexpr auto b1 = RegionBDegrees<1.0, kRole>();

    for (std::size_t i = 0; i < a1.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(a1[i])) << "region-A degree " << a1[i] << " at index " << i;
    }

    for (std::size_t i = 0; i < b1.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(b1[i])) << "region-B degree " << b1[i] << " at index " << i;
    }

    // The m = 1 tables are the full degrees: the criterion's budget
    // (m - 1)*B is zero, so only the full degree has an empty tail.
    if constexpr (qcx::integrals::detail::RoleUsesDoubleTables(kRole))
    {
        for (std::size_t p = 0; p < qcx::integrals::detail::kPieces.size(); ++p)
        {
            EXPECT_EQ(a1[p], qcx::integrals::detail::kPieces[p].deg)
                << "region-A piece " << p << " must keep its full degree at m = 1";
        }
    } else
    {
        for (std::size_t p = 0; p < qcx::integrals::detail::f32::kPieces.size(); ++p)
        {
            EXPECT_EQ(a1[p], qcx::integrals::detail::f32::kPieces[p].deg)
                << "region-A piece " << p << " must keep its full degree at m = 1";
        }
    }

    if constexpr (kRole == BoysRole::kDoubleSingle || kRole == BoysRole::kDoubleBatch)
    {
        for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
        {
            EXPECT_EQ(b1[static_cast<std::size_t>(n)], qcx::integrals::detail::kBDeg)
                << "region-B order " << n << " must keep its full degree at m = 1";
        }
    } else
    {
        for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
        {
            EXPECT_EQ(b1[static_cast<std::size_t>(n)], qcx::integrals::detail::f32::kBDeg)
                << "region-B order " << n << " must keep its full degree at m = 1";
        }
    }
}

TEST(BoysAccuracyTest, EffectiveDegreesAtM1AreFullAndInDomain) {
    AssertRoleTables<BoysRole::kDoubleSingle>();
    AssertRoleTables<BoysRole::kDoubleBatch>();
    AssertRoleTables<BoysRole::kF32Single>();
    AssertRoleTables<BoysRole::kF32Batch>();
    AssertRoleTables<BoysRole::kF32Fp16Single>();
    AssertRoleTables<BoysRole::kF32Fp16Batch>();
}

template <double kMLo, double kMHi, BoysRole kRole> void AssertRoleNonIncreasing() {
    static_assert(kMLo < kMHi);
    constexpr auto loA = RegionADegrees<kMLo, kRole>();
    constexpr auto hiA = RegionADegrees<kMHi, kRole>();
    static_assert(loA.size() == hiA.size());

    for (std::size_t i = 0; i < loA.size(); ++i)
    {
        EXPECT_GE(loA[i], hiA[i]) << "region-A index " << i << " degree must not grow with m ("
                                  << kMLo << " -> " << kMHi << ")";
    }

    constexpr auto loB = RegionBDegrees<kMLo, kRole>();
    constexpr auto hiB = RegionBDegrees<kMHi, kRole>();

    for (std::size_t i = 0; i < loB.size(); ++i)
    {
        EXPECT_GE(loB[i], hiB[i]) << "region-B index " << i << " degree must not grow with m ("
                                  << kMLo << " -> " << kMHi << ")";
    }
}

TEST(BoysAccuracyTest, EffectiveDegreesNonIncreasingInMultiplier) {
    // Non-increasing in m over every pair of sampled multipliers and every role.
    ForEachSampledMultiplier([]<double kMLo>() {
        ForEachSampledMultiplier([]<double kMHi>() {
            if constexpr (kMLo < kMHi)
            {
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kDoubleSingle>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kDoubleBatch>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Single>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Batch>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Fp16Single>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Fp16Batch>();
            }
        });
    });
}

// ---------------------------------------------------------------------------
// Delivered-path consistency: the relaxed single and batch lanes (separate
// arithmetic paths — per-order fits vs. seed + recursion) must agree within
// the sum of their m-scaled region bounds, and the exact x = 0 values must
// survive at every sampled m.
// ---------------------------------------------------------------------------

template <double kM> void CheckSingleBatchAgree() {
    std::mt19937_64 rng(424242);
    std::uniform_real_distribution<double> xd(1e-4, 40.0);
    std::vector<double> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const double x = xd(rng);
        const BoysRegion region = RegionOf(x);
        const int nmax = static_cast<int>(rng() % (qcx::integrals::kMaxBoysOrder + 1));
        BoysAllOrders<kM>(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const double single = BoysSingle<kM>(k, x);
            const double bound = kM * (RegionBound(region, LaneKind::kDoubleSingle) +
                                       RegionBound(region, LaneKind::kDoubleBatch));
            EXPECT_LE(std::abs(batch[static_cast<std::size_t>(k)] - single), bound)
                << "m=" << kM << " n=" << k << " x=" << x;
        }
    }
}

template <double kM> void CheckSingleBatchAgreeF32() {
    std::mt19937_64 rng(424243);
    std::uniform_real_distribution<float> xd(1e-4f, 40.0f);
    std::vector<float> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const float x = xd(rng);
        const BoysRegion region = RegionOf(static_cast<double>(x));
        const int nmax = static_cast<int>(rng() % (qcx::integrals::kMaxBoysOrder + 1));
        BoysAllOrdersF32<kM>(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const float single = BoysSingleF32<kM>(k, x);
            const double bound = kM * (RegionBound(region, LaneKind::kFloatSingle) +
                                       RegionBound(region, LaneKind::kFloatBatch));
            EXPECT_LE(std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) -
                               static_cast<double>(single)),
                      bound)
                << "m=" << kM << " n=" << k << " x=" << x;
        }
    }
}

template <double kM> void CheckZeroArgument() {
    for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
    {
        EXPECT_DOUBLE_EQ(BoysSingle<kM>(n, 0.0), 1.0 / (2.0 * n + 1.0));
        EXPECT_FLOAT_EQ(BoysSingleF32<kM>(n, 0.0f), 1.0f / (2.0f * static_cast<float>(n) + 1.0f));
    }

    double batch[qcx::integrals::kMaxBoysOrder + 1];
    BoysAllOrders<kM>(8, 0.0, batch);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_DOUBLE_EQ(batch[k], 1.0 / (2.0 * k + 1.0));
    }

    float batchF32[qcx::integrals::kMaxBoysOrder + 1];
    BoysAllOrdersF32<kM>(8, 0.0f, batchF32);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_FLOAT_EQ(batchF32[k], 1.0f / (2.0f * static_cast<float>(k) + 1.0f));
    }
}

TEST(BoysAccuracyTest, SingleBatchAgreeAtSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        CheckSingleBatchAgree<kM>();
        CheckSingleBatchAgreeF32<kM>();
    });
}

TEST(BoysAccuracyTest, ZeroArgumentExactAtSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { CheckZeroArgument<kM>(); });
}

// ---------------------------------------------------------------------------
// The fp16/Bf16 lanes at sampled m: |F_hat - F(x16)| <= m*1e-7 + 1/2
// ULP per value, F(x16) the certified double lane at the fp16-rounded
// argument (the reference lane). The multiplier forwards to the F32
// engine's kFp16-budget roles (no fp16-specific degree tables).
// ---------------------------------------------------------------------------
#if QcxIntegralsFp16

double HalfUlp(qcx::integrals::F16 x) {
    return 0.5 * (static_cast<double>(qcx::integrals::NextUp(x)) - static_cast<double>(x));
}

double HalfUlp(qcx::integrals::Bf16 x) {
    return 0.5 * (static_cast<double>(qcx::integrals::NextUp(x)) - static_cast<double>(x));
}

template <typename Half,
          double kM,
          Half (*SingleFn)(int, Half) noexcept,
          void (*BatchFn)(int, Half, Half*) noexcept>
void RunHalfSampledCheck(const char* label) {
    double worstSingle = 0.0;
    double worstBatch = 0.0;
    std::vector<Half> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        const Half x = static_cast<Half>(row.x);
        const double reference = BoysSingle<1.0>(row.n, static_cast<double>(x));
        const Half half = static_cast<Half>(reference);
        const double tolerance = kM * 1e-7 + HalfUlp(half);
        const double single = static_cast<double>(SingleFn(row.n, x));
        const double singleError = std::abs(single - reference);
        EXPECT_LE(singleError, tolerance) << "m=" << kM << " n=" << row.n << " x=" << row.x
                                          << " got=" << single << " want=" << reference;
        worstSingle = std::max(worstSingle, singleError);
        BatchFn(row.n, x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double batchReference = BoysSingle<1.0>(k, static_cast<double>(x));
            const double batchTolerance = kM * 1e-7 + HalfUlp(static_cast<Half>(batchReference));
            const double batchError =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - batchReference);
            EXPECT_LE(batchError, batchTolerance)
                << "m=" << kM << " batch F" << k << " at x=" << row.x
                << " got=" << static_cast<double>(batch[static_cast<std::size_t>(k)])
                << " want=" << batchReference;
            worstBatch = std::max(worstBatch, batchError);
        }
    }

    std::printf("%s m=%.0e: worst single |error| %.3e, worst batch |error| %.3e "
                "(bound m*1e-7 + 1/2 ULP per value)\n",
                label,
                kM,
                worstSingle,
                worstBatch);
}

TEST(BoysAccuracyTest, F16SampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        RunHalfSampledCheck<qcx::integrals::F16,
                            kM,
                            qcx::integrals::BoysSingleF16<kM>,
                            qcx::integrals::BoysAllOrdersF16<kM>>("BoysF16");
    });
}

TEST(BoysAccuracyTest, Bf16SampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        RunHalfSampledCheck<qcx::integrals::Bf16,
                            kM,
                            qcx::integrals::BoysSingleBf16<kM>,
                            qcx::integrals::BoysAllOrdersBf16<kM>>("BoysBf16");
    });
}

#endif // QcxIntegralsFp16

} // namespace
