// The human-readable run report. See result_report.hpp for the contract:
// what the report states, what it omits, and where each line's facts come
// from.

#include "qcx/io/result_report.hpp"

#include "qcx/molecule/elements.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <utility>

namespace qcx::io {
namespace {

// The report's columns. Every field is padded to one of these widths, so a
// value's column is fixed whatever its digits are.
constexpr std::size_t kHeaderLabelWidth = 14; ///< "molecule      "
constexpr std::size_t kHeaderValueWidth = 11; ///< "H2O        "
constexpr std::size_t kHeaderFieldWidth = 13; ///< "charge 0     "
constexpr std::size_t kBuilderValueWidth = 23; ///< "direct / lean / cpu    "
constexpr std::size_t kScfLabelWidth = 19; ///< "|E_n - E_(n-1)|    "
constexpr std::size_t kScfVerdictWidth = 13; ///< "yes          "
constexpr std::size_t kAtomLabelWidth = 3; ///< "O  "
constexpr std::size_t kAtomValueWidth = 5; ///< "0.196" - a wider value shifts its entry.
constexpr std::size_t kEntryGapWidth = 5; ///< The spaces between two entries on a line.
constexpr std::size_t kEntriesPerLine = 4; ///< Entries per Mulliken-section line.
constexpr std::size_t kRuleWidth = 66; ///< The rule under the title.
constexpr int kEnergyDecimals = 4; ///< Hartree, read to the 0.1 mHartree.
constexpr int kResidualDecimals = 1; ///< Residuals, read for their leading digits.
constexpr int kEntryDecimals = 3; ///< Electrons.

// Pads a field to a fixed width, the caller having given it the spaces that
// separate it from what follows: a field that outgrows its width is returned
// as it stands, never truncated - the column it starts is fixed either way,
// and a clipped number is a wrong one.
std::string PadTo(std::string text, std::size_t width) {
    if (text.size() < width)
    {
        text.append(width - text.size(), ' ');
    }

    return text;
}

// Trims the trailing spaces a fixed-width field leaves at the end of a line:
// padded columns, never padded line ends.
std::string TrimTrailing(const std::string& text) {
    const std::size_t end = text.find_last_not_of(' ');

    if (end == std::string::npos)
    {
        return std::string();
    }

    return text.substr(0, end + 1);
}

// One number in fixed-point notation; the word beside it carries the unit.
std::string Fixed(double value, int decimals) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(decimals) << value;
    return stream.str();
}

// One number in scientific notation: a convergence residual is read for its
// leading digits, whatever its exponent.
std::string Scientific(double value, int decimals) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(decimals) << value;
    return stream.str();
}

// Whole milliseconds: a wall time is read to the millisecond, and the
// remainder of a reading rounded to one is noise.
std::string Milliseconds(double wallMs) {
    return std::to_string(std::llround(wallMs));
}

// "<n> atoms" / "1 atom": a count with its unit word, singular at one.
std::string CountedText(std::size_t count, std::string_view singular, std::string_view plural) {
    return std::to_string(count) + " " + std::string(count == 1 ? singular : plural);
}

// One header row: the label column, then the row's own fields.
std::string HeaderRow(std::string_view label, const std::string& fields) {
    return PadTo(std::string(label), kHeaderLabelWidth) + TrimTrailing(fields) + "\n";
}

// One SCF row: the section indent, the label column, then the value.
std::string ScfRow(std::string_view label, const std::string& value) {
    return "  " + PadTo(std::string(label), kScfLabelWidth) + value + "\n";
}

// The molecular formula in the Hill order a chemist reads: carbon, hydrogen,
// then the remaining elements alphabetically, each followed by its count
// unless it is one. The formula does not depend on the order the file wrote
// its rows in.
std::string MolecularFormula(const std::vector<RunAtom>& atoms) {
    std::vector<std::string> symbols;

    for (const RunAtom& atom : atoms)
    {
        if (std::find(symbols.begin(), symbols.end(), atom.symbol) == symbols.end())
        {
            symbols.push_back(atom.symbol);
        }
    }

    const auto hillRank = [](const std::string& symbol) {
        if (symbol == "C")
        {
            return 0;
        }

        if (symbol == "H")
        {
            return 1;
        }

        return 2;
    };

    std::sort(symbols.begin(),
              symbols.end(),
              [&hillRank](const std::string& left, const std::string& right) {
                  if (hillRank(left) != hillRank(right))
                  {
                      return hillRank(left) < hillRank(right);
                  }

                  return left < right;
              });

    std::string formula;

    for (const std::string& symbol : symbols)
    {
        formula += symbol;

        const auto count = static_cast<std::size_t>(
            std::count_if(atoms.begin(), atoms.end(), [&symbol](const RunAtom& atom) {
                return atom.symbol == symbol;
            }));

        if (count > 1)
        {
            formula += std::to_string(count);
        }
    }

    return formula;
}

// The vocabulary the builder selection came from, spelled out for a reader:
// the axis keys, the deprecated conflated key, or the size ladder that
// decided when the file named no builder key at all. A word outside that
// vocabulary is printed as it came rather than replaced by a guess.
std::string ChosenByText(const std::string& requestedBy) {
    if (requestedBy == "axes")
    {
        return "axes";
    }

    if (requestedBy == "fock_builder")
    {
        return "deprecated fock_builder key";
    }

    if (requestedBy == "ladder")
    {
        return "size ladder";
    }

    return requestedBy;
}

// The header block: what the file asked for, and the builder the run resolved.
void AppendHeader(std::string& report,
                  const RunInput& input,
                  const RunResult& result,
                  const RunReportFacts& facts) {
    report += "qcx run report\n";
    report += std::string(kRuleWidth, '-') + "\n";

    report += HeaderRow(
        "molecule",
        PadTo(MolecularFormula(input.molecule.atoms) + "  ", kHeaderValueWidth) +
            PadTo("charge " + std::to_string(input.molecule.charge) + "  ", kHeaderFieldWidth) +
            "multiplicity " + std::to_string(input.molecule.multiplicity));

    std::string basisFields =
        PadTo(input.basis.orbital + "  ", kHeaderValueWidth) +
        PadTo(CountedText(input.molecule.atoms.size(), "atom", "atoms") + "  ", kHeaderFieldWidth);

    if (facts.basisFunctionCount.has_value())
    {
        basisFields += CountedText(*facts.basisFunctionCount, "basis function", "basis functions");
    }

    report += HeaderRow("basis", basisFields);

    report +=
        HeaderRow("method",
                  PadTo(std::string(ToString(input.method.method)) + "  ", kHeaderValueWidth) +
                      "accuracy " + std::string(ToString(input.method.accuracy)));

    // The axes are the record's own resolved words - the vocabulary the
    // resolution produced, not a second spelling of it.
    const std::string axes = result.builderAxes.integralFamily + " / " +
                             result.builderAxes.storageTier + " / " +
                             result.builderAxes.executionBackend;

    report += HeaderRow("builder",
                        PadTo(axes + "  ", kBuilderValueWidth) + "(" +
                            ChosenByText(result.builderAxes.requestedBy) + ")");
}

// The SCF section: the convergence verdict and the residuals the gate
// compared. The two residual rows appear only when the result carries them -
// a result no SCF loop filled has neither, and the section then states the
// verdict alone rather than a zero no run measured.
void AppendScfSection(std::string& report, const RunResult& result) {
    report += "\nSCF\n";
    report += ScfRow(
        "converged",
        PadTo(std::string(result.converged ? "yes" : "no") + "  ", kScfVerdictWidth) +
            CountedText(static_cast<std::size_t>(result.iterations), "iteration", "iterations"));
    report +=
        ScfRow("total energy", Fixed(result.totalEnergyHartree, kEnergyDecimals) + " Hartree");

    if (result.energyDeltaHartree.has_value())
    {
        report +=
            ScfRow("|E_n - E_(n-1)|", Scientific(*result.energyDeltaHartree, kResidualDecimals));
    }

    if (result.rmsDensityDelta.has_value())
    {
        report +=
            ScfRow("RMS density delta", Scientific(*result.rmsDensityDelta, kResidualDecimals));
    }
}

// The nuclear charge a symbol stands for: the element table's Z, or 0 when the
// symbol is no element's. The parser refuses an unknown symbol before any run
// reaches here, so a zero is the formatter stating the absence rather than
// assuming a charge it cannot form.
int NuclearCharge(std::string_view symbol) {
    const qcx::molecule::ElementData* element = qcx::molecule::FindElement(symbol);

    return element == nullptr ? 0 : element->atomicNumber;
}

// One (label, value) pair per atom, and the heading that names what the value
// is. The population array is indexed in the molecule's canonical order, so
// pairing a number with the atom it belongs to takes the run's permutation -
// and turning a population into a charge takes the atom's symbol as well,
// because a Mulliken partial charge is the nuclear charge less the gross
// population, q_A = Z_A - P_A ([Mulliken1955]). The section reports charges
// under their own name where both are available; where a symbol or a whole
// permutation is missing it reports the populations themselves, each labelled
// by its 1-based index in the result's own order rather than borrowing a
// symbol that may belong to another atom.
struct MullikenSection {
    const char* heading;
    std::vector<std::pair<std::string, double>> rows;
};

MullikenSection MullikenRows(const RunInput& input,
                             const RunReportFacts& facts,
                             const std::vector<double>& populations) {
    const std::vector<std::size_t>& order = facts.canonicalAtomOrder;
    const bool pairsRows = order.size() == input.molecule.atoms.size() &&
                           std::all_of(order.begin(), order.end(), [&populations](std::size_t row) {
                               return row < populations.size();
                           });

    if (pairsRows)
    {
        std::vector<std::pair<std::string, double>> rows;
        rows.reserve(populations.size());
        bool charged = true;

        for (std::size_t atom = 0; atom < input.molecule.atoms.size(); ++atom)
        {
            const std::string& symbol = input.molecule.atoms[atom].symbol;
            const int z = NuclearCharge(symbol);

            if (z == 0)
            {
                charged = false;
                break;
            }

            rows.emplace_back(symbol, static_cast<double>(z) - populations[order[atom]]);
        }

        if (charged)
        {
            return MullikenSection{"\nMulliken charges\n", std::move(rows)};
        }
    }

    std::vector<std::pair<std::string, double>> rows;
    rows.reserve(populations.size());

    for (std::size_t atom = 0; atom < populations.size(); ++atom)
    {
        rows.emplace_back("#" + std::to_string(atom + 1), populations[atom]);
    }

    return MullikenSection{"\nMulliken populations\n", std::move(rows)};
}

// The Mulliken section of the density the run produced, one entry per atom,
// a fixed number to a line, the entries column-aligned. The section is
// written only when the result carries the populations: a run that produced
// no population block gets no section at all, rather than a row of zeros.
void AppendMullikenSection(std::string& report,
                           const RunInput& input,
                           const RunResult& result,
                           const RunReportFacts& facts) {
    if (!result.properties.has_value() || result.properties->populations.mullikenTotal.empty())
    {
        return;
    }

    const MullikenSection section =
        MullikenRows(input, facts, result.properties->populations.mullikenTotal);
    const std::vector<std::pair<std::string, double>>& rows = section.rows;

    report += section.heading;

    for (std::size_t index = 0; index < rows.size(); ++index)
    {
        const std::string entry = PadTo(rows[index].first, kAtomLabelWidth) +
                                  PadTo(Fixed(rows[index].second, kEntryDecimals), kAtomValueWidth);
        const bool endsLine = (index + 1) % kEntriesPerLine == 0 || index + 1 == rows.size();

        if (index % kEntriesPerLine == 0)
        {
            report += "  ";
        }

        report += endsLine ? entry : entry + std::string(kEntryGapWidth, ' ');

        if (endsLine)
        {
            report += "\n";
        }
    }
}

// The wall-time row: the whole run and the SCF loop inside it. The result
// carries no finer breakdown, so none is stated.
void AppendTimings(std::string& report, const RunResult& result) {
    report += "\n";
    report += HeaderRow("timings",
                        "total " + Milliseconds(result.timingsMs.totalMs) + " ms   (scf loop " +
                            Milliseconds(result.timingsMs.scfLoopMs) + " ms)");
}

} // namespace

std::string FormatRunReport(const RunInput& input,
                            const RunResult& result,
                            const RunReportFacts& facts) {
    std::string report;
    AppendHeader(report, input, result, facts);
    AppendScfSection(report, result);
    AppendMullikenSection(report, input, result, facts);
    AppendTimings(report, result);

    return report;
}

} // namespace qcx::io
