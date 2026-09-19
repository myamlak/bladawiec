#include "qcx/driver/accuracy_class_report.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string_view>

namespace qcx::driver {
namespace {

// The preset name as the schema writes it.
std::string_view PresetLabel(qcx::integrals::AccuracyPreset preset) {
    switch (preset)
    {
    case qcx::integrals::AccuracyPreset::kLoose:
        return "kLoose";
    case qcx::integrals::AccuracyPreset::kNormal:
        return "kNormal";
    case qcx::integrals::AccuracyPreset::kTight:
        return "kTight";
    }

    return "unknown";
}

// A measured deviation in scientific notation (the report's magnitudes run
// 1e-12 through 1e-3 Eh, so scientific is the readable form).
std::string FormatDeviation(double deviation) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(6) << deviation;
    return text.str();
}

// A fitted coefficient (the prefactor C or the exponent p) in plain form.
std::string FormatCoefficient(double value) {
    std::ostringstream text;
    text << std::setprecision(4) << value;
    return text.str();
}

// The density gate of a preset in plain form (the threshold the scaling
// law's C is anchored to).
std::string FormatThreshold(double threshold) {
    std::ostringstream text;
    text << std::setprecision(2) << threshold;
    return text.str();
}

// The rows of one (preset, reference) group.
struct GroupedRunRows {
    qcx::integrals::AccuracyPreset preset = qcx::integrals::AccuracyPreset::kNormal;
    std::string reference;
    std::vector<const QfmmAccuracyRunRow*> rows;
};

// Partitions the measured rows into their (preset, reference) groups. The
// rows of one group are all measured against the same reference at the
// same preset, so the group's per-bin aggregates are the comparison the
// table reports. The groups sort by preset strictness then
// reference, so two callers feeding the same data in different orders
// print the same table.
std::vector<GroupedRunRows> GroupByPresetAndReference(const std::vector<QfmmAccuracyRunRow>& rows) {
    std::vector<GroupedRunRows> groups;

    for (const auto& row : rows)
    {
        auto found =
            std::find_if(groups.begin(), groups.end(), [&row](const GroupedRunRows& group) {
                return group.preset == row.preset && group.reference == row.reference;
            });

        if (found == groups.end())
        {
            GroupedRunRows group;
            group.preset = row.preset;
            group.reference = row.reference;
            group.rows.push_back(&row);
            groups.push_back(std::move(group));
        } else
        {
            found->rows.push_back(&row);
        }
    }

    std::stable_sort(
        groups.begin(), groups.end(), [](const GroupedRunRows& a, const GroupedRunRows& b) {
            if (a.preset != b.preset)
            {
                return a.preset < b.preset;
            }

            return a.reference < b.reference;
        });
    return groups;
}

// The rows of one bin of one group (only the rows whose basis count falls
// in the bin).
std::vector<const QfmmAccuracyRunRow*> RowsInBin(const GroupedRunRows& group, SizeClassBin bin) {
    std::vector<const QfmmAccuracyRunRow*> binRows;

    for (const auto* row : group.rows)
    {
        if (SizeClassBinForBasisCount(row->basisCount) == bin)
        {
            binRows.push_back(row);
        }
    }

    return binRows;
}

// The composed deviations of the bin's rows, ascending.
std::vector<double> SortedDeviations(const std::vector<const QfmmAccuracyRunRow*>& rows) {
    std::vector<double> deviations;
    deviations.reserve(rows.size());

    for (const auto* row : rows)
    {
        deviations.push_back(row->composedDeviation);
    }

    std::sort(deviations.begin(), deviations.end());
    return deviations;
}

// The nearest-rank percentile of an ascending deviation list.
double Percentile(const std::vector<double>& sortedDeviations, double fraction) {
    const std::size_t rank =
        std::max<std::size_t>(1,
                              static_cast<std::size_t>(std::ceil(
                                  fraction * static_cast<double>(sortedDeviations.size()))));
    return sortedDeviations[rank - 1];
}

// The log-space least-squares fit of the bin's composed deviations to the
// scaling law deltaE = C * threshold * N_bf^p: the slope is p and the
// intercept fixes C * threshold. Only rows with a positive deviation and a
// positive basis count enter the log fit; the fit needs at least three
// such rows spanning at least two distinct basis counts (a degenerate
// column cannot anchor a slope).
SizeClassScalingFit FitScalingLaw(const std::vector<const QfmmAccuracyRunRow*>& rows,
                                  double threshold) {
    SizeClassScalingFit fit;
    fit.threshold = threshold;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumXx = 0.0;
    double sumXy = 0.0;
    std::size_t count = 0;

    for (const auto* row : rows)
    {
        if (row->basisCount <= 0 || row->composedDeviation <= 0.0)
        {
            continue;
        }

        const double x = std::log(static_cast<double>(row->basisCount));
        const double y = std::log(row->composedDeviation);
        sumX += x;
        sumY += y;
        sumXx += x * x;
        sumXy += x * y;
        ++count;
    }

    // The n < 3 underpopulation rule: no fit from fewer than
    // three fixtures. A zero denominator means every fitted row shares one
    // basis count - nothing to anchor the slope to.
    const double countD = static_cast<double>(count);

    if (count < 3 || countD * sumXx - sumX * sumX <= 0.0)
    {
        return fit;
    }

    const double slope = (countD * sumXy - sumX * sumY) / (countD * sumXx - sumX * sumX);
    const double intercept = (sumY - slope * sumX) / countD;

    fit.valid = true;
    fit.exponent = slope;
    fit.prefactor = std::exp(intercept) / threshold;
    return fit;
}

// The basis-count range of the bin's rows.
struct BasisCountRange {
    int min = 0;
    int max = 0;
};

BasisCountRange CountRange(const std::vector<const QfmmAccuracyRunRow*>& rows) {
    BasisCountRange range{rows.front()->basisCount, rows.front()->basisCount};

    for (const auto* row : rows)
    {
        range.min = std::min(range.min, row->basisCount);
        range.max = std::max(range.max, row->basisCount);
    }

    return range;
}

// The fixture whose composed deviation is the bin's maximum.
std::string_view WorstFixture(const std::vector<const QfmmAccuracyRunRow*>& rows) {
    const QfmmAccuracyRunRow* worst = rows.front();

    for (const auto* row : rows)
    {
        if (row->composedDeviation > worst->composedDeviation)
        {
            worst = row;
        }
    }

    return worst->fixture;
}

// One bin's table line: n_fixtures, the N_basis range, the max/median/p95
// absolute composed deviation, the worst fixture, and - only when the bin
// holds three or more fixtures - the fitted scaling law. Bins below the
// n < 3 threshold are marked UNDERPOPULATED and carry no fit.
std::string FormatBinLine(const std::vector<const QfmmAccuracyRunRow*>& rows,
                          SizeClassBin bin,
                          double threshold) {
    std::ostringstream line;
    const auto deviations = SortedDeviations(rows);
    const auto range = CountRange(rows);
    line << "  bin " << SizeClassBinLabel(bin) << ": n_fixtures " << rows.size() << ", N_basis "
         << range.min;

    if (range.max != range.min)
    {
        line << "-" << range.max;
    }

    line << ", |dE| max " << FormatDeviation(deviations.back()) << ", median "
         << FormatDeviation(Percentile(deviations, 0.5)) << ", p95 "
         << FormatDeviation(Percentile(deviations, 0.95)) << " Eh, worst fixture "
         << WorstFixture(rows);

    const auto fit = FitScalingLaw(rows, threshold);

    if (rows.size() < 3)
    {
        line << " [UNDERPOPULATED: n_fixtures < 3 - no fit claimed]";
    } else if (fit.valid)
    {
        line << ", fit deltaE = C * " << FormatThreshold(threshold) << " * N^p: C "
             << FormatCoefficient(fit.prefactor) << ", p " << FormatCoefficient(fit.exponent);
    } else
    {
        line << " [no fit: fewer than three fitted rows or one shared basis count]";
    }

    return line.str();
}

} // namespace

SizeClassBin SizeClassBinForBasisCount(int basisCount) noexcept {
    if (basisCount < 0)
    {
        return SizeClassBin::kNoPreRegisteredBin;
    }

    for (std::size_t index = 0; index < kSizeClassBinUpperBounds.size(); ++index)
    {
        if (basisCount <= kSizeClassBinUpperBounds[index])
        {
            return static_cast<SizeClassBin>(index);
        }
    }

    return SizeClassBin::kNoPreRegisteredBin;
}

std::string_view SizeClassBinLabel(SizeClassBin bin) noexcept {
    switch (bin)
    {
    case SizeClassBin::kBasis0To99:
        return "0-99";
    case SizeClassBin::kBasis100To199:
        return "100-199";
    case SizeClassBin::kBasis200To399:
        return "200-399";
    case SizeClassBin::kBasis400To799:
        return "400-799";
    case SizeClassBin::kBasis800To1599:
        return "800-1599";
    case SizeClassBin::kNoPreRegisteredBin:
        return "no pre-registered bin";
    }

    return "unknown";
}

std::string FormatSizeClassBandLine(const QfmmAccuracyRunRow& row) {
    std::ostringstream line;
    line << "size-class band [" << SizeClassBinLabel(SizeClassBinForBasisCount(row.basisCount))
         << "]: fixture " << row.fixture << ", N_basis " << row.basisCount << ", preset "
         << PresetLabel(row.preset) << ", reference " << row.reference << ", composed |dE| "
         << FormatDeviation(row.composedDeviation) << " Eh";

    if (row.coulombHalfDeviation.has_value())
    {
        line << ", coulomb-half |dE| " << FormatDeviation(*row.coulombHalfDeviation) << " Eh";
    }

    if (row.exchangeHalfDeviation.has_value())
    {
        line << ", exchange-half |dE| " << FormatDeviation(*row.exchangeHalfDeviation) << " Eh";
    }

    return line.str();
}

std::string FormatSizeClassBandTable(const std::vector<QfmmAccuracyRunRow>& rows) {
    if (rows.empty())
    {
        return "";
    }

    std::ostringstream table;
    table << "size-class band table (pre-registered bins, fixed before the measurements; "
             "per-run REPORTING, never a budget):";

    for (const auto& group : GroupByPresetAndReference(rows))
    {
        const double threshold = qcx::integrals::DensityThreshold(group.preset);
        table << "\npreset " << PresetLabel(group.preset) << ", reference " << group.reference
              << " (density gate " << FormatThreshold(threshold) << "):";

        // The pre-registered bins in ascending order, then the outside-range
        // rows (reported, but the table's claim stops at the pre-registered
        // range).
        for (int binIndex = 0; binIndex <= static_cast<int>(SizeClassBin::kNoPreRegisteredBin);
             ++binIndex)
        {
            const auto bin = static_cast<SizeClassBin>(binIndex);
            const auto binRows = RowsInBin(group, bin);

            if (!binRows.empty())
            {
                table << "\n" << FormatBinLine(binRows, bin, threshold);
            }
        }
    }

    return table.str();
}

} // namespace qcx::driver
