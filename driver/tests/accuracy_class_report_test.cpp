// The size-class accuracy report of the composed QFMM runs: the
// pre-registered bin assignment of a
// basis count, the per-run band line (bin, fixture, N_basis, preset,
// reference, the composed deviation and the halves when measured), and
// the aggregate band table (per (preset, reference) group and bin:
// n_fixtures, the N_basis range, max/median/p95 absolute deviations, the
// worst fixture, the fitted scaling law deltaE = C * threshold * N_bf^p,
// and the n < 3 underpopulation marker). Pure formatting - no I/O, no
// JSON - so these tests run in every configuration.

#include "alkane_sto3g.hpp"
#include "qcx/driver/accuracy_class_report.hpp"
#include "qcx/driver/basis_counts.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <utility>
#include <vector>

namespace {

using qcx::driver::FormatSizeClassBandLine;
using qcx::driver::FormatSizeClassBandTable;
using qcx::driver::QfmmAccuracyRunRow;
using qcx::driver::SizeClassBin;
using qcx::driver::SizeClassBinForBasisCount;
using qcx::driver::SizeClassBinLabel;
using qcx::integrals::AccuracyPreset;

// One measured row: a composed QFMM run of the named fixture at the
// preset, measured against the reference. The deviations are the caller's
// measured numbers (Eh).
QfmmAccuracyRunRow MakeRow(std::string fixture,
                           int basisCount,
                           AccuracyPreset preset,
                           std::string reference,
                           double composedDeviation) {
    QfmmAccuracyRunRow row;
    row.fixture = std::move(fixture);
    row.basisCount = basisCount;
    row.preset = preset;
    row.reference = std::move(reference);
    row.composedDeviation = composedDeviation;
    return row;
}

} // namespace

TEST(AccuracyClassReportTest, BinAssignmentUsesThePreRegisteredBoundaries) {
    EXPECT_EQ(SizeClassBinForBasisCount(0), SizeClassBin::kBasis0To99);
    EXPECT_EQ(SizeClassBinForBasisCount(1), SizeClassBin::kBasis0To99);
    EXPECT_EQ(SizeClassBinForBasisCount(86), SizeClassBin::kBasis0To99);
    EXPECT_EQ(SizeClassBinForBasisCount(99), SizeClassBin::kBasis0To99);
    EXPECT_EQ(SizeClassBinForBasisCount(100), SizeClassBin::kBasis100To199);
    EXPECT_EQ(SizeClassBinForBasisCount(170), SizeClassBin::kBasis100To199);
    EXPECT_EQ(SizeClassBinForBasisCount(199), SizeClassBin::kBasis100To199);
    EXPECT_EQ(SizeClassBinForBasisCount(200), SizeClassBin::kBasis200To399);
    EXPECT_EQ(SizeClassBinForBasisCount(399), SizeClassBin::kBasis200To399);
    EXPECT_EQ(SizeClassBinForBasisCount(400), SizeClassBin::kBasis400To799);
    EXPECT_EQ(SizeClassBinForBasisCount(586), SizeClassBin::kBasis400To799);
    EXPECT_EQ(SizeClassBinForBasisCount(799), SizeClassBin::kBasis400To799);
    EXPECT_EQ(SizeClassBinForBasisCount(800), SizeClassBin::kBasis800To1599);
    EXPECT_EQ(SizeClassBinForBasisCount(1599), SizeClassBin::kBasis800To1599);

    // Beyond the pre-registered range - and below zero - there is no bin:
    // nothing is claimed for them.
    EXPECT_EQ(SizeClassBinForBasisCount(1600), SizeClassBin::kNoPreRegisteredBin);
    EXPECT_EQ(SizeClassBinForBasisCount(5000), SizeClassBin::kNoPreRegisteredBin);
    EXPECT_EQ(SizeClassBinForBasisCount(-1), SizeClassBin::kNoPreRegisteredBin);
}

TEST(AccuracyClassReportTest, AlkaneChainFixtureCountsAreMachineCounted) {
    // The C12/C24 chain fixtures' basis totals, counted through the
    // framework's own parser - the same basis text the driver runs parse,
    // so these are the runs' n_basis (the report's bin assignment input).
    // C has S + SP = 1 + (1 + 3) spherical functions, H has S = 1: 86 for
    // C12H26 and 170 for C24H50 (the 194 that is quoted for it is
    // arithmetic, not a parse; the memory-model convention counts O the
    // same way - the driver's basis-function count).
    const auto c12 = qcx::testing::MakeAlkaneSto3g(12);
    ASSERT_TRUE(c12.has_value());
    const auto c24 = qcx::testing::MakeAlkaneSto3g(24);
    ASSERT_TRUE(c24.has_value());
    const auto basis = qcx::testing::MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value());

    EXPECT_EQ(qcx::driver::CountBasisFunctions(*c12, *basis), 86u);
    EXPECT_EQ(qcx::driver::CountBasisFunctions(*c24, *basis), 170u);
}

TEST(AccuracyClassReportTest, BinLabelsReadLikeTheSchema) {
    EXPECT_EQ(SizeClassBinLabel(SizeClassBin::kBasis0To99), "0-99");
    EXPECT_EQ(SizeClassBinLabel(SizeClassBin::kBasis100To199), "100-199");
    EXPECT_EQ(SizeClassBinLabel(SizeClassBin::kBasis200To399), "200-399");
    EXPECT_EQ(SizeClassBinLabel(SizeClassBin::kBasis400To799), "400-799");
    EXPECT_EQ(SizeClassBinLabel(SizeClassBin::kBasis800To1599), "800-1599");
    EXPECT_EQ(SizeClassBinLabel(SizeClassBin::kNoPreRegisteredBin), "no pre-registered bin");
}

TEST(AccuracyClassReportTest, BandLinePrintsTheBinAndComposedDeviation) {
    const auto row =
        MakeRow("C12H26/STO-3G", 86, AccuracyPreset::kLoose, "kTight direct", 2.2894e-5);
    const auto line = FormatSizeClassBandLine(row);

    EXPECT_NE(line.find("size-class band [0-99]"), std::string::npos);
    EXPECT_NE(line.find("fixture C12H26/STO-3G"), std::string::npos);
    EXPECT_NE(line.find("N_basis 86"), std::string::npos);
    EXPECT_NE(line.find("preset kLoose"), std::string::npos);
    EXPECT_NE(line.find("reference kTight direct"), std::string::npos);
    EXPECT_NE(line.find("composed |dE| 2.289400e-05 Eh"), std::string::npos);

    // A run without the halves' decomposition prints its bin and the
    // composed deviation only - no half claims are invented.
    EXPECT_EQ(line.find("coulomb-half"), std::string::npos);
    EXPECT_EQ(line.find("exchange-half"), std::string::npos);
}

TEST(AccuracyClassReportTest, BandLinePrintsTheHalvesWhenTheContextMeasuredThem) {
    auto row =
        MakeRow("C24H50/STO-3G", 170, AccuracyPreset::kNormal, "kNormal direct", 6.51653e-10);
    row.coulombHalfDeviation = 1.9e-9;
    row.exchangeHalfDeviation = 6.5e-10;
    const auto line = FormatSizeClassBandLine(row);

    EXPECT_NE(line.find("size-class band [100-199]"), std::string::npos);
    EXPECT_NE(line.find("composed |dE| 6.516530e-10 Eh"), std::string::npos);
    EXPECT_NE(line.find("coulomb-half |dE| 1.900000e-09 Eh"), std::string::npos);
    EXPECT_NE(line.find("exchange-half |dE| 6.500000e-10 Eh"), std::string::npos);
}

TEST(AccuracyClassReportTest, BandLineHandlesBasisCountsWithoutABin) {
    const auto row = MakeRow("big/STO-3G", 1600, AccuracyPreset::kLoose, "kTight direct", 1e-4);
    const auto line = FormatSizeClassBandLine(row);

    EXPECT_NE(line.find("size-class band [no pre-registered bin]"), std::string::npos);
    EXPECT_NE(line.find("N_basis 1600"), std::string::npos);
}

TEST(AccuracyClassReportTest, EmptyTableIsEmpty) {
    EXPECT_EQ(FormatSizeClassBandTable({}), "");
}

TEST(AccuracyClassReportTest, UnderpopulatedBinsCarryTheMarkerAndNoFit) {
    // One fixture in the bin: n < 3, marked, no fit.
    const auto single =
        MakeRow("C12H26/STO-3G", 86, AccuracyPreset::kLoose, "kTight direct", 2.2894e-5);
    const auto singleTable = FormatSizeClassBandTable({single});

    EXPECT_NE(singleTable.find("size-class band table"), std::string::npos);
    EXPECT_NE(singleTable.find("preset kLoose, reference kTight direct"), std::string::npos);
    EXPECT_NE(singleTable.find("bin 0-99: n_fixtures 1, N_basis 86"), std::string::npos);
    EXPECT_NE(singleTable.find("|dE| max 2.289400e-05, median 2.289400e-05, p95 "
                               "2.289400e-05 Eh, worst fixture C12H26/STO-3G"),
              std::string::npos);
    EXPECT_NE(singleTable.find("[UNDERPOPULATED: n_fixtures < 3 - no fit claimed]"),
              std::string::npos);
    EXPECT_EQ(singleTable.find("fit deltaE"), std::string::npos);

    // Two fixtures in the bin: still underpopulated by the n < 3 rule.
    const auto second =
        MakeRow("C12H26/STO-3G", 86, AccuracyPreset::kLoose, "kTight direct", 2.28e-5);
    const auto twoTable = FormatSizeClassBandTable({single, second});
    EXPECT_NE(twoTable.find("n_fixtures 2"), std::string::npos);
    EXPECT_NE(twoTable.find("[UNDERPOPULATED: n_fixtures < 3 - no fit claimed]"),
              std::string::npos);
    EXPECT_EQ(twoTable.find("fit deltaE"), std::string::npos);
}

TEST(AccuracyClassReportTest, TableFitsTheScalingLawOnThreeOrMoreFixtures) {
    // Three fixtures on the exact scaling law deltaE = 0.4 * 1e-8 * N^2.5
    // (the consult's C ~ 0.4 for n-alkanes at the kLoose density gate): the
    // table reports the fitted C and p and drops the underpopulation
    // marker.
    std::vector<QfmmAccuracyRunRow> rows;
    constexpr double kGate = 1e-8;
    constexpr double kPrefactor = 0.4;
    constexpr double kExponent = 2.5;
    const int basisCounts[] = {120, 150, 190};

    for (int count : basisCounts)
    {
        const double deviation = kPrefactor * kGate * std::pow(count, kExponent);
        rows.push_back(
            MakeRow("synthetic/STO-3G", count, AccuracyPreset::kLoose, "kTight direct", deviation));
    }

    const auto table = FormatSizeClassBandTable(rows);

    EXPECT_NE(table.find("preset kLoose, reference kTight direct (density gate 1e-08):"),
              std::string::npos);
    EXPECT_NE(table.find("bin 100-199: n_fixtures 3, N_basis 120-190"), std::string::npos);
    EXPECT_EQ(table.find("[UNDERPOPULATED"), std::string::npos);
    EXPECT_NE(table.find("fit deltaE = C * 1e-08 * N^p: C 0.4, p 2.5"), std::string::npos);
    EXPECT_NE(table.find("worst fixture synthetic/STO-3G"), std::string::npos);
}

TEST(AccuracyClassReportTest, TableGroupsByPresetAndReferenceAndSortsBins) {
    // Two groups (kLoose vs kTight direct, kNormal vs kNormal direct) with
    // rows in different bins, fed out of order: the table sorts groups by
    // preset strictness and prints the bins in ascending pre-registered
    // order.
    std::vector<QfmmAccuracyRunRow> rows;
    rows.push_back(MakeRow("C24H50/STO-3G", 170, AccuracyPreset::kNormal, "kNormal direct", 1e-9));
    rows.push_back(MakeRow("mid/STO-3G", 300, AccuracyPreset::kNormal, "kNormal direct", 2e-9));
    rows.push_back(MakeRow("C12H26/STO-3G", 86, AccuracyPreset::kLoose, "kTight direct", 3e-5));
    rows.push_back(MakeRow("large/STO-3G", 1000, AccuracyPreset::kLoose, "kTight direct", 4e-5));
    rows.push_back(MakeRow("outside/STO-3G", 1700, AccuracyPreset::kLoose, "kTight direct", 5e-5));

    const auto table = FormatSizeClassBandTable(rows);

    const auto kLoosePos = table.find("preset kLoose, reference kTight direct");
    const auto kNormalPos = table.find("preset kNormal, reference kNormal direct");
    EXPECT_NE(kLoosePos, std::string::npos);
    EXPECT_NE(kNormalPos, std::string::npos);
    EXPECT_LT(kLoosePos, kNormalPos);

    // The kLoose group's bins in ascending order: 0-99, then 800-1599, then
    // the out-of-range rows reported last.
    const auto bin0Pos = table.find("bin 0-99:", kLoosePos);
    const auto binBigPos = table.find("bin 800-1599:", kLoosePos);
    const auto outsidePos = table.find("bin no pre-registered bin:", kLoosePos);
    EXPECT_NE(bin0Pos, std::string::npos);
    EXPECT_NE(binBigPos, std::string::npos);
    EXPECT_NE(outsidePos, std::string::npos);
    EXPECT_LT(bin0Pos, binBigPos);
    EXPECT_LT(binBigPos, outsidePos);
}
