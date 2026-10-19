// The Boys accuracy-contract tests at the pinned library's entries.
//
// Certified-lane grid contract: on the committed reference grid
// (tools/gen_boys_coefficients.py, 30-digit mpmath values rounded to double),
// the double lanes hold the larger of 1e-13 * F_n(x) and 1e-13 * F_0(x) per
// value, and the float lanes the larger of 2^-22 * F_n(x) and 1.5e-7, plus
// 1e-7 under the plain reciprocal. The bound is formed from the grid's own
// F_0(x) (F_n <= F_0, so the larger arm is the F_0 one), which is the shape
// the library publishes; the previous revision stated one absolute figure per
// region, and no region bucketing survives in the contract.
//
// The effective-degree tables are the evaluator's own: every degree is inside
// the evaluator domain {0, 1, 2, 4, 6, ...} for all six lane roles.
//
// The fp16/Bf16 lanes forward to the F32 engine (I/O-only wrappers, no
// fp16-specific degree tables): assert |F_hat - F(x16)| <= 1.5e-7 + 1/2 ULP
// per value on the reference grid, F(x16) being the certified double lane
// evaluated at the fp16-rounded argument (the reference lane).
//
// The file lives alongside boys_test.cpp rather than inside it to keep that
// file's existing tests untouched (the split is editorial only — same test
// binary, same contract).

// The split's group headers: the coarse bases this file's degree tables read are defined there,
// and a group header precedes the front door below.
#include "boys/boys_coefficients_d_coarse_cheb.hpp"
#include "boys/boys_coefficients_f32_coarse_cheb.hpp"

#include "boys/boys.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/boys_impl.hpp"
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

const std::vector<ReferenceRow> gReference = LoadReference();
const ReferenceGrid gGrid(gReference);

// The published bound of one double-lane value: the larger of
// 1e-13 * F_n(x) and 1e-13 * F_0(x).
double DoubleBound(int n, double x) {
    return 1e-13 * std::max(gGrid.Value(n, x), gGrid.Value(0, x));
}

// The published bound of one float-lane value: the larger of 2^-22 * F_n(x)
// and 1.5e-7, plus 1e-7 under the plain reciprocal. The entry names no
// division form, so the test carries the plain-reciprocal term the published
// contract adds.
double FloatBound(int n, double x) {
    return std::max(std::ldexp(gGrid.Value(n, x), -22), 1.5e-7) + 1e-7;
}

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

void PrintWorsts(const char* lane, const RegionWorsts& worst) {
    std::printf("%s: worst region A %.3e, B %.3e, C %.3e, extended band %.3e\n",
                lane,
                worst.a,
                worst.b,
                worst.c,
                worst.e);
}

// ---------------------------------------------------------------------------
// The certified-lane grid contract, one sweep per lane
// ---------------------------------------------------------------------------

void SweepDoubleSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = BoysSingle(row.n, row.x);
        const double bound = DoubleBound(row.n, row.x);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "n=" << row.n << " x=" << row.x << " got=" << value
                                << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("double single", worst);
}

void SweepDoubleBatch() {
    RegionWorsts worst;
    std::vector<double> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrders(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = DoubleBound(k, row.x);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("double batch", worst);
}

void SweepFloatSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const float value = BoysSingleF32(row.n, static_cast<float>(row.x));
        const double bound = FloatBound(row.n, row.x);
        const double error = std::abs(static_cast<double>(value) - row.value);
        EXPECT_LE(error, bound) << "n=" << row.n << " x=" << row.x << " got=" << value
                                << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("float single", worst);
}

void SweepFloatBatch() {
    RegionWorsts worst;
    std::vector<float> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrdersF32(row.n, static_cast<float>(row.x), batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = FloatBound(k, row.x);
            const double error =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - reference);
            EXPECT_LE(error, bound)
                << "batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("float batch", worst);
}

TEST(BoysAccuracyTest, DoubleSingleHoldsThePublishedBound) {
    SweepDoubleSingle();
}

TEST(BoysAccuracyTest, DoubleBatchHoldsThePublishedBound) {
    SweepDoubleBatch();
}

TEST(BoysAccuracyTest, FloatSingleHoldsThePublishedBound) {
    SweepFloatSingle();
}

TEST(BoysAccuracyTest, FloatBatchHoldsThePublishedBound) {
    SweepFloatBatch();
}

// ---------------------------------------------------------------------------
// The effective degrees: inside the evaluator domain
// {0, 1, 2, 4, 6, ...} at every (role, point)
// ---------------------------------------------------------------------------

constexpr bool InEvaluatorDomain(int d) {
    return d == 0 || d == 1 || d == 2 || (d >= 4 && d % 2 == 0);
}

template <BoysRole kRole> void AssertRoleTables() {
    constexpr auto a = RegionADegrees<kRole>();
    constexpr auto b = RegionBDegrees<kRole>();

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(a[i])) << "region-A degree " << a[i] << " at index " << i;
    }

    for (std::size_t i = 0; i < b.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(b[i])) << "region-B degree " << b[i] << " at index " << i;
    }
}

TEST(BoysAccuracyTest, EffectiveDegreesAreInTheEvaluatorDomain) {
    AssertRoleTables<BoysRole::kDoubleSingle>();
    AssertRoleTables<BoysRole::kDoubleBatch>();
    AssertRoleTables<BoysRole::kF32Single>();
    AssertRoleTables<BoysRole::kF32Batch>();
    AssertRoleTables<BoysRole::kF32Fp16Single>();
    AssertRoleTables<BoysRole::kF32Fp16Batch>();
}

// ---------------------------------------------------------------------------
// Delivered-path consistency: the single and batch lanes (separate arithmetic
// paths — per-order fits vs. seed + recursion) must agree within the sum of
// their published bounds, and the exact x = 0 values must survive.
// ---------------------------------------------------------------------------

// These samples take arguments off the reference grid, so the bound is the
// published one evaluated at its largest: F_n <= F_0 <= 1 for every x >= 0,
// so the double lane's larger arm is at most 1e-13 and the float lane's at
// most 2^-22 + 1e-7.
constexpr double kDoubleBoundSupremum = 1e-13;
constexpr double kFloatBoundSupremum = 0x1p-22 + 1e-7;

void CheckSingleBatchAgree() {
    std::mt19937_64 rng(424242);
    std::uniform_real_distribution<double> xd(1e-4, 40.0);
    std::vector<double> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const double x = xd(rng);
        const int nmax = static_cast<int>(rng() % (qcx::integrals::kMaxBoysOrder + 1));
        BoysAllOrders(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const double single = BoysSingle(k, x);
            const double bound = 2.0 * kDoubleBoundSupremum;
            EXPECT_LE(std::abs(batch[static_cast<std::size_t>(k)] - single), bound)
                << "n=" << k << " x=" << x;
        }
    }
}

void CheckSingleBatchAgreeF32() {
    std::mt19937_64 rng(424243);
    std::uniform_real_distribution<float> xd(1e-4f, 40.0f);
    std::vector<float> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const float x = xd(rng);
        const int nmax = static_cast<int>(rng() % (qcx::integrals::kMaxBoysOrder + 1));
        BoysAllOrdersF32(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const float single = BoysSingleF32(k, x);
            const double bound = 2.0 * kFloatBoundSupremum;
            EXPECT_LE(std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) -
                               static_cast<double>(single)),
                      bound)
                << "n=" << k << " x=" << x;
        }
    }
}

void CheckZeroArgument() {
    for (int n = 0; n <= qcx::integrals::kMaxBoysOrder; ++n)
    {
        EXPECT_DOUBLE_EQ(BoysSingle(n, 0.0), 1.0 / (2.0 * n + 1.0));
        EXPECT_FLOAT_EQ(BoysSingleF32(n, 0.0f), 1.0f / (2.0f * static_cast<float>(n) + 1.0f));
    }

    double batch[qcx::integrals::kMaxBoysOrder + 1];
    BoysAllOrders(8, 0.0, batch);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_DOUBLE_EQ(batch[k], 1.0 / (2.0 * k + 1.0));
    }

    float batchF32[qcx::integrals::kMaxBoysOrder + 1];
    BoysAllOrdersF32(8, 0.0f, batchF32);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_FLOAT_EQ(batchF32[k], 1.0f / (2.0f * static_cast<float>(k) + 1.0f));
    }
}

TEST(BoysAccuracyTest, SingleBatchAgreeWithinTheSumOfBounds) {
    CheckSingleBatchAgree();
    CheckSingleBatchAgreeF32();
}

TEST(BoysAccuracyTest, ZeroArgumentIsExact) {
    CheckZeroArgument();
}

// ---------------------------------------------------------------------------
// The fp16/Bf16 lanes: |F_hat - F(x16)| <= 1.5e-7 + 1/2 ULP per value,
// F(x16) the certified double lane at the fp16-rounded argument (the
// reference lane). The wrappers forward to the F32 engine's kFp16-budget
// roles (no fp16-specific degree tables).
// ---------------------------------------------------------------------------
#if QcxIntegralsFp16

double HalfUlp(qcx::integrals::F16 x) {
    return 0.5 * (static_cast<double>(qcx::integrals::NextUp(x)) - static_cast<double>(x));
}

double HalfUlp(qcx::integrals::Bf16 x) {
    return 0.5 * (static_cast<double>(qcx::integrals::NextUp(x)) - static_cast<double>(x));
}

template <typename Half,
          Half (*SingleFn)(int, Half) noexcept,
          void (*BatchFn)(int, Half, Half*) noexcept>
void RunHalfCheck(const char* label) {
    double worstSingle = 0.0;
    double worstBatch = 0.0;
    std::vector<Half> batch(qcx::integrals::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        const Half x = static_cast<Half>(row.x);
        const double reference = BoysSingle(row.n, static_cast<double>(x));
        const Half half = static_cast<Half>(reference);
        const double tolerance = 1.5e-7 + HalfUlp(half);
        const double single = static_cast<double>(SingleFn(row.n, x));
        const double singleError = std::abs(single - reference);
        EXPECT_LE(singleError, tolerance)
            << "n=" << row.n << " x=" << row.x << " got=" << single << " want=" << reference;
        worstSingle = std::max(worstSingle, singleError);
        BatchFn(row.n, x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double batchReference = BoysSingle(k, static_cast<double>(x));
            const double batchTolerance = 1.5e-7 + HalfUlp(static_cast<Half>(batchReference));
            const double batchError =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - batchReference);
            EXPECT_LE(batchError, batchTolerance)
                << "batch F" << k << " at x=" << row.x
                << " got=" << static_cast<double>(batch[static_cast<std::size_t>(k)])
                << " want=" << batchReference;
            worstBatch = std::max(worstBatch, batchError);
        }
    }

    std::printf("%s: worst single |error| %.3e, worst batch |error| %.3e "
                "(bound 1.5e-7 + 1/2 ULP per value)\n",
                label,
                worstSingle,
                worstBatch);
}

TEST(BoysAccuracyTest, F16HoldsThePublishedBound) {
    RunHalfCheck<qcx::integrals::F16,
                 qcx::integrals::BoysSingleF16,
                 qcx::integrals::BoysAllOrdersF16>("BoysF16");
}

TEST(BoysAccuracyTest, Bf16HoldsThePublishedBound) {
    RunHalfCheck<qcx::integrals::Bf16,
                 qcx::integrals::BoysSingleBf16,
                 qcx::integrals::BoysAllOrdersBf16>("BoysBf16");
}

#endif // QcxIntegralsFp16

} // namespace
