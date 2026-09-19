// Recursion-boundaries harness (tab:boundaries of the accompanying paper):
// measures where the erf-seeded upward recursion leaves the
// 5e-14 absolute window of the V&S eq. 26 series reference, per kmax in
// {4, 8, 16, 32}, and pins the four recorded cells within
// kMeasuredTolerance (the paper's cells adopted the
// first-failure-at-0.0001 record - see the reconciliation note on
// kThresholdRows). Both halves of the recursion are implemented here: no
// std::erf exists in the library's sources (region B seeds from the
// Chebyshev fit), so the harness carries the erf seed itself and mirrors
// the shipped upward step exactly.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The boundary threshold: the recursion stays within this absolute window of
// the reference for all n <= kmax (the paper's 5e-14, shared with the
// tab:accuracy cells).
constexpr double kBoundaryThreshold = 5e-14;

// Asserted agreement of the measured cells with the pins recorded below (the
// recording machine's first-failure-at-0.0001 cells, see kThresholdRows; the
// paper's tab:boundaries cells carry the same record as x0Paper). The cells
// sit inside an oscillating error envelope, so a ~1-ulp std::erf/std::exp
// spread across libms - and the resolution difference between this machine's
// 0.001-grid two-phase measurement and the 0.0001 record - can move the
// measured value by up to ~15-20% (measured spread of the kmax=4 crossing
// across seed roundings: 0.388..0.44 at 0.001 resolution, vs the 0.4625
// recorded at 0.0001). 20% per cell is a loose, cross-libm- and
// cross-resolution-stable gate that still fails on a broken seed/step/
// reference.
constexpr double kMeasuredTolerance = 0.2;

// Series term cap and truncation floor, mirroring the committed generator
// (tools/gen_boys_coefficients.py, boys_ref: range(300), break at
// term < 1e-26 after l > 20).
constexpr int kMaxSeriesTerms = 300;
constexpr long double kSeriesTailFloor = 1e-26L;

// Inline-series vs committed-grid agreement (relative; see the cross-check
// test): ~1.6x the measured worst on MSVC's 64-bit long double
// (3.200e-15 at (n=31, x=38.3119) - measured on this machine).
constexpr double kCsvAgreementTolerance = 5e-15;

// Toolchain premise pin (the "(80-bit on MSVC)" premise is false): MSVC's long
// double is 64-bit - a synonym for double - while GCC/Clang carry the x87
// 80-bit type. The reference series degrades to double precision on MSVC
// (~1e-16 relative instead of ~1e-19); F_n values are O(1) or smaller in
// the sweep, so the reference error stays 2-3 orders below kBoundaryThreshold
// and the boundary measurement is unaffected (verified by the CSV
// cross-check, whose tolerance reflects this).
static_assert(sizeof(long double) == 8 || sizeof(long double) == 16);

// F0(x) = 0.5 sqrt(pi/x) erf(sqrt(x)) - the erf seed, implemented in-test
// (the measurement definition; std::erf is deliberately not added to
// the production library).
double ErfSeedF0(double x) {
    return 0.5 * std::sqrt(std::numbers::pi / x) * std::erf(std::sqrt(x));
}

// F_n(x) via the provably stable series (V&S eq. 26): all terms positive,
// F_n(x) = 0.5 e^-x sum_l x^l / prod_{j=0..l} (n + j + 0.5). The same series
// the committed reference grid is generated from (gen_boys_coefficients.py
// boys_ref), evaluated here in long double (see the static_assert above).
// (n, x) are the order and the argument - convertible, never swapped.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
long double SeriesReference(int n, double x) {
    if (x == 0.0)
    {
        return 1.0L / (2.0L * n + 1.0L);
    }

    const long double xL = static_cast<long double>(x);
    long double sum = 0.0L;
    long double term = 1.0L / (static_cast<long double>(n) + 0.5L);

    for (int l = 0; l < kMaxSeriesTerms; ++l)
    {
        sum += term;

        if (l > 20 && term < kSeriesTailFloor)
        {
            break;
        }

        term *= xL / (static_cast<long double>(n + l) + 1.5L);
    }

    // std::exp, not std::expl: C99 math names are not required in namespace std
    // (libstdc++ has no std::expl); the long double overload
    // is selected by the argument.
    return std::exp(-xL) * 0.5L * sum;
}

// The shipped region-B upward step (expx = 0.5 e^-x hoisted once, then
// f = ((l + 0.5) f - expx) / x), seeded from the erf definition instead of
// the Chebyshev fit. Runs in double: the boundary is a property of fp64
// arithmetic.
// (n, x) are the order and the argument - convertible, never swapped.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ErfSeededUpward(int n, double x) {
    double f = ErfSeedF0(x);
    const double expx = 0.5 * std::exp(-x);

    for (int l = 0; l < n; ++l)
    {
        f = ((l + 0.5) * f - expx) / x;
    }

    return f;
}

// Pass criterion: |recursion(n, x) - reference(n, x)| <= 5e-14 for all
// n in 0..kmax. (kmax, x) are the order cap and the argument - convertible,
// never swapped.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool PassesThreshold(int kmax, double x) {
    for (int n = 0; n <= kmax; ++n)
    {
        const double recursion = ErfSeededUpward(n, x);
        const double reference = static_cast<double>(SeriesReference(n, x));

        if (std::abs(recursion - reference) > kBoundaryThreshold)
        {
            return false;
        }
    }

    return true;
}

// Two-phase deterministic descending sweep: phase 1 walks
// x from 12.0 down in 0.01 steps (the "down to 0.001" floor is the
// sweep's low end - 12.0 -> 0.001 at step 0.01 does not land on 0.001, and
// the smallest recorded cell, kmax = 4's 0.4625, sits far above it); phase 2
// refines the pass/fail transition band in 0.001 steps (three decimal places
// of resolution; the current cells record the 0.0001-resolution first
// failure - see the reconciliation note on kThresholdRows). No random draws
// anywhere.
//
// The "error is monotone over the transition" premise is FALSE: the
// amplified seed rounding error oscillates through the 5e-14 line hundreds
// of times near the transition (verified on a 1e-4 grid), so "the" boundary
// is a grid-dependent measurement, not a monotone threshold. Stopping both
// phases at the first fail is therefore the deterministic definition
// of the measured cell, not a monotonicity shortcut.
double MeasureBoundary(int kmax) {
    constexpr double kSweepStart = 12.0;
    constexpr double kSweepStep = 0.01;
    constexpr double kSweepFloor = 0.001;
    constexpr double kRefineStep = 0.001;

    // The pass region must contain the sweep start: at x = 12.0 the upward
    // amplification is O(1) for all n <= 32 (seed error ~1e-16 -> ~2e-16),
    // far below the threshold.
    if (!PassesThreshold(kmax, kSweepStart))
    {
        ADD_FAILURE() << "kmax=" << kmax << ": no pass at the sweep start x = 12.0";
        return -1.0;
    }

    double xPass = kSweepStart;

    for (int i = 1;; ++i)
    {
        const double x = kSweepStart - static_cast<double>(i) * kSweepStep;

        if (x < kSweepFloor)
        {
            break;
        }

        if (PassesThreshold(kmax, x))
        {
            xPass = x;
        } else
        {
            break;
        }
    }

    const double xFail = xPass - kSweepStep;
    double x0 = xPass;

    for (int i = 1;; ++i)
    {
        const double x = xPass - static_cast<double>(i) * kRefineStep;

        if (x <= xFail)
        {
            break;
        }

        if (PassesThreshold(kmax, x))
        {
            x0 = x;
        } else
        {
            break;
        }
    }

    return x0;
}

// tab:boundaries rows: per kmax, x0Measured (the asserted pin - the
// recording machine's first failing sample of the 0.0001 descending sweep,
// within kMeasuredTolerance), x0Paper (the paper's cell - the same record),
// and the published formula (V&S eq. 25/13, arXiv:2512.10059 - literature
// input, hardcoded, not measured). Margins = formula / x0 are computed
// ratios in the test, not asserted independently.
//
// RECONCILIATION RECORD - THREE GENERATIONS (the record-and-investigate
// protocol: "if a machine deviates beyond it, record and investigate before
// editing the paper"): (1) the paper's ORIGINAL cells (0.244/0.754/3.82/9.70)
// were
// measurements of the design-study harness's coarse x*1.02 geometric grid
// (first passing grid point), reproduced
// bit-for-bit by simulating that exact grid (0.244/0.754/2.069/3.823/5.910/
// 9.697). The recursion error is NOT monotone over the transition (the
// "empirically monotone" premise is false): the amplified seed
// rounding error oscillates through the 5e-14 line hundreds of times, so
// the coarse grid skipped over failure bands and its cells are grid
// artifacts. (2) The fine two-phase sweep (0.01 then 0.001
// steps) then measured 0.393/1.484/4.150/9.866 on this machine's pinned
// builds (fresh builds draw 0.401/1.435 for kmax = 4/8), and the writing
// pass adopted those cells - but the 0.0001-resolution post-check refuted
// them too: failing 1e-4 samples sit above the
// cells (+17.7%/+10.3%/+2.1%/+1.9% for kmax = 4/8/16/32), with 402-3351
// pass/fail alternations per band, so the 0.001 cells are themselves lattice
// artifacts of the two-phase sweep. (3) CURRENT: the paper presents the cells
// as first-failure-at-resolution values
// - the first failing samples of the 0.0001 descending sweep (0.4625/1.6373/
// 4.2367/10.0492 - the x0Paper column below, which the paper adopted), with
// the passing sample one 1e-4 step above each (0.4626/1.6374/
// 4.2368/10.0493); the record is explicitly resolution-limited, not a
// stability threshold. The recording machine's 1e-5 scans over the
// envelope-top neighborhoods extend the largest failing samples to
// ~0.4625/1.64959/4.29768/10.05917 (the supplementary record). The pins
// below guard these cells with the same ±20% band: this machine's own
// two-phase measurement is a coarser (0.001) lattice draw of the same
// oscillating envelope, and the band absorbs both the libm roundings and
// the resolution difference while still failing on a broken seed/step/
// reference.
struct ThresholdRow {
    int kmax;
    double x0Measured;
    double x0Paper;
    double formula;
};

const std::array<ThresholdRow, 4> kThresholdRows = {
    ThresholdRow{4, 0.4625, 0.4625, 1.60},
    ThresholdRow{8, 1.6373, 1.6373, 3.07},
    ThresholdRow{16, 4.2367, 4.2367, 6.01},
    ThresholdRow{32, 10.0492, 10.0492, 11.9},
};

// The committed reference grid (tools/gen_boys_coefficients.py, 30-digit
// mpmath values rounded to double) - cross-checked against the inline
// series as a self-test.
struct CsvRow {
    int n;
    double x;
    double value;
};

std::vector<CsvRow> LoadCsvRows() {
    const std::string path = std::string(QcxBoysDataDir) + "/boys_reference.csv";
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<CsvRow> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream stream(line);
        std::string cell;
        CsvRow row{};
        std::getline(stream, cell, ',');
        row.n = std::stoi(cell);
        std::getline(stream, cell, ',');
        row.x = std::strtod(cell.c_str(), nullptr);
        std::getline(stream, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);
        // The committed grid ends at x = 100: beyond it every F_n is covered
        // by region C's asymptotic form (see boys_test.cpp; the filter guards
        // against a future grid extension past the series' convergence
        // limit, ~x = 250 in the generator).
        if (row.x <= 100.0)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

} // namespace

TEST(BoysBoundaryTest, ErfSeededUpwardRecursionThresholds) {
    for (const ThresholdRow& row : kThresholdRows)
    {
        const double x0 = MeasureBoundary(row.kmax);
        EXPECT_GT(x0, 0.0) << "kmax=" << row.kmax;

        if (x0 <= 0.0)
        {
            continue;
        }

        // Pin this machine's measured cell; print the paper cell and the
        // deviation as the runs record (reconciliation note on kThresholdRows).
        const double relativeDeviation = std::abs(x0 - row.x0Measured) / row.x0Measured;
        EXPECT_LE(relativeDeviation, kMeasuredTolerance)
            << "kmax=" << row.kmax << ": measured x0 = " << x0 << " vs pinned " << row.x0Measured;
        const double paperDeviation = std::abs(x0 - row.x0Paper) / row.x0Paper;
        std::printf("BoysBoundary kmax=%2d: measured x0 = %.6f, paper %.4f, formula %.2f "
                    "(V&S eq. 25/13), margin %.1fx, deviation vs paper %.2e\n",
                    row.kmax,
                    x0,
                    row.x0Paper,
                    row.formula,
                    row.formula / x0,
                    paperDeviation);
    }
}

TEST(BoysBoundaryTest, InlineReferenceMatchesCommittedGrid) {
    // Self-test of the inline series: both it and the CSV evaluate the same
    // V&S eq. 26 series (the CSV at 30 mpmath digits, rounded to double), so
    // their agreement is bounded by the two roundings. The "agree to
    // ~1e-17" assumed the 80-bit long-double premise; with MSVC's 64-bit
    // long double the inline error is ~1e-16 relative, so the asserted
    // agreement (kCsvAgreementTolerance, ~5x the measured worst on MSVC)
    // reflects that - the gate's purpose is only to catch a transcription
    // error in the series, 1-2 orders below the 5e-14 boundary threshold.
    const std::vector<CsvRow> rows = LoadCsvRows();
    ASSERT_GT(rows.size(), 500u);

    double worstRelative = 0.0;
    int worstN = -1;
    double worstX = 0.0;

    for (const CsvRow& row : rows)
    {
        const double inlineValue = static_cast<double>(SeriesReference(row.n, row.x));
        const double relativeError = std::abs(inlineValue - row.value) / row.value;
        EXPECT_LE(relativeError, kCsvAgreementTolerance) << "n=" << row.n << " x=" << row.x;

        if (relativeError > worstRelative)
        {
            worstRelative = relativeError;
            worstN = row.n;
            worstX = row.x;
        }
    }

    std::printf("BoysBoundary: inline long-double series vs committed grid: "
                "worst relative |error| = %.3e at (n=%d, x=%g)\n",
                worstRelative,
                worstN,
                worstX);
}
