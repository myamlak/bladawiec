// The report acceptance: the document a full result produces, the absence
// policy (a block the result does not carry prints no section at all - never
// a zero-filled row), the atom labelling through the run's canonicalization
// permutation, and the vocabulary the builder resolution came from.

#include "qcx/io/result_report.hpp"

#include <gtest/gtest.h>
#include <string>

namespace {

// The water fixture, written the way a chemist writes it: the oxygen row
// first. The molecule renumbers its atoms canonically on construction (by
// element, then by position), so this file's permutation names input row 2
// (the -y hydrogen) as canonical atom 0, row 1 as canonical atom 1, and row 0
// (the oxygen) as canonical atom 2 - which is what separates "the order the
// file wrote" from "the order the result's per-atom vectors use".
qcx::io::RunInput WaterInput() {
    qcx::io::RunInput input;
    input.molecule.charge = 0;
    input.molecule.multiplicity = 1;
    input.molecule.atoms = {qcx::io::RunAtom{"O", 0.0, 0.0, 0.0},
                            qcx::io::RunAtom{"H", 0.0, 1.4304, 1.1089},
                            qcx::io::RunAtom{"H", 0.0, -1.4304, 1.1089}};
    input.basis.orbital = "sto-3g";
    input.method.method = qcx::io::MethodType::kRhf;
    input.method.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    return input;
}

// The result that run produced: converged in 12 iterations, with the
// population block and both convergence residuals. The fixture's numbers are
// the Mulliken GROSS POPULATIONS the record carries, in the molecule's
// canonical order - the run does not carry charges, a Mulliken charge being
// the nuclear charge less the population, q_A = Z_A - P_A - and they are
// DISTINCT per atom (the oxygen's 8.397 population beside the hydrogens'
// 0.799 and 0.804, which print as the charges -0.397, 0.201 and 0.196), so an
// index-permutation mistake reads as a named difference instead of passing
// equal to equal.
qcx::io::RunResult WaterResult() {
    qcx::io::RunResult result;
    result.converged = true;
    result.iterations = 12;
    result.energyDeltaHartree = 3.2e-09;
    result.rmsDensityDelta = 8.1e-07;
    result.totalEnergyHartree = -74.942;
    result.builderAxes.integralFamily = "direct";
    result.builderAxes.storageTier = "lean";
    result.builderAxes.executionBackend = "cpu";
    result.builderAxes.requestedBy = "ladder";
    result.properties.emplace();
    result.properties->populations.mullikenTotal = {0.799, 0.804, 8.397};
    result.timingsMs.totalMs = 412.0;
    result.timingsMs.scfLoopMs = 88.0;
    return result;
}

qcx::io::RunReportFacts WaterFacts() {
    qcx::io::RunReportFacts facts;
    facts.basisFunctionCount = 7;
    facts.canonicalAtomOrder = {2, 1, 0};
    return facts;
}

TEST(ResultReportTest, FullResultPinsEverySection) {
    // The whole document, character for character: the columns are what a
    // reader's eye follows, so a width that drifts is a change to the report
    // and not a cosmetic one.
    const std::string expected =
        "qcx run report\n"
        "------------------------------------------------------------------\n"
        "molecule      H2O        charge 0     multiplicity 1\n"
        "basis         sto-3g     3 atoms      7 basis functions\n"
        "method        rhf        accuracy kNormal\n"
        "builder       direct / lean / cpu    (size ladder)\n"
        "\n"
        "SCF\n"
        "  converged          yes          12 iterations\n"
        "  total energy       -74.9420 Hartree\n"
        "  |E_n - E_(n-1)|    3.2e-09\n"
        "  RMS density delta  8.1e-07\n"
        "\n"
        "Mulliken charges\n"
        "  O  -0.397     H  0.196     H  0.201\n"
        "\n"
        "timings       total 412 ms   (scf loop 88 ms)\n";

    EXPECT_EQ(qcx::io::FormatRunReport(WaterInput(), WaterResult(), WaterFacts()), expected);
}

TEST(ResultReportTest, AbsentBlocksPrintNoSection) {
    // A single-atom run that did not converge: the SCF loop filled neither
    // residual, and no population block was computed. The report states the
    // convergence it reached and stops - no printed zero, no dash and no
    // placeholder stands in for a number the run never produced - and the
    // singular unit words ride along.
    qcx::io::RunInput input;
    input.molecule.atoms = {qcx::io::RunAtom{"He", 0.0, 0.0, 0.0}};
    input.basis.orbital = "sto-3g";
    input.method.method = qcx::io::MethodType::kUhf;
    input.method.accuracy = qcx::integrals::AccuracyPreset::kTight;

    qcx::io::RunReportFacts facts;
    facts.basisFunctionCount = 1;
    facts.canonicalAtomOrder = {0};

    qcx::io::RunResult result;
    result.converged = false;
    result.iterations = 100;
    result.totalEnergyHartree = -2.8551604;
    result.builderAxes.integralFamily = "direct";
    result.builderAxes.storageTier = "in_memory";
    result.builderAxes.executionBackend = "cpu";
    result.builderAxes.requestedBy = "axes";
    result.timingsMs.totalMs = 1530.0;
    result.timingsMs.scfLoopMs = 1520.0;

    const std::string report = qcx::io::FormatRunReport(input, result, facts);
    const std::string expected =
        "qcx run report\n"
        "------------------------------------------------------------------\n"
        "molecule      He         charge 0     multiplicity 1\n"
        "basis         sto-3g     1 atom       1 basis function\n"
        "method        uhf        accuracy kTight\n"
        "builder       direct / in_memory / cpu  (axes)\n"
        "\n"
        "SCF\n"
        "  converged          no           100 iterations\n"
        "  total energy       -2.8552 Hartree\n"
        "\n"
        "timings       total 1530 ms   (scf loop 1520 ms)\n";

    EXPECT_EQ(report, expected);
    EXPECT_EQ(report.find("Mulliken"), std::string::npos);
    EXPECT_EQ(report.find("RMS density delta"), std::string::npos);
    EXPECT_EQ(report.find("|E_n - E_(n-1)|"), std::string::npos);
}

TEST(ResultReportTest, AtomsPrintInTheOrderTheInputWroteThem) {
    // The populations arrive in the molecule's canonical order while the file
    // wrote its rows in another one; the run's permutation pairs them, so
    // every charge is printed beside the symbol of the atom it belongs to.
    // The same fixture read WITHOUT a permutation cannot make that pairing,
    // and a charge cannot be formed without knowing which atom the population
    // belongs to: the section then reports the populations, each atom's by its
    // index in the result's own order, rather than borrowing a symbol that may
    // belong to another atom.
    qcx::io::RunInput input = WaterInput();
    qcx::io::RunResult result = WaterResult();

    const std::string paired = qcx::io::FormatRunReport(input, result, WaterFacts());
    EXPECT_NE(paired.find("  O  -0.397     H  0.196     H  0.201\n"), std::string::npos);
    EXPECT_NE(paired.find("Mulliken charges\n"), std::string::npos);

    const std::string unpaired = qcx::io::FormatRunReport(input, result, qcx::io::RunReportFacts{});
    EXPECT_NE(unpaired.find("  #1 0.799     #2 0.804     #3 8.397\n"), std::string::npos);
    EXPECT_NE(unpaired.find("Mulliken populations\n"), std::string::npos);
    EXPECT_EQ(unpaired.find("O  -0.397"), std::string::npos);
}

TEST(ResultReportTest, BasisCountAndTheBuilderVocabulary) {
    // The basis-function count is the run's own resolved number: when the
    // caller has none the phrase is dropped and the rest of the line stays.
    const std::string withoutCount =
        qcx::io::FormatRunReport(WaterInput(), WaterResult(), qcx::io::RunReportFacts{});
    EXPECT_NE(withoutCount.find("basis         sto-3g     3 atoms\n"), std::string::npos);

    // The vocabulary that chose the builder is spelled out, one word per
    // source: the axis keys, the deprecated conflated key, or the size ladder.
    const std::pair<const char*, const char*> vocabulary[] = {
        {"axes", "(axes)"},
        {"fock_builder", "(deprecated fock_builder key)"},
        {"ladder", "(size ladder)"}};

    for (const auto& [word, expected] : vocabulary)
    {
        qcx::io::RunResult result = WaterResult();
        result.builderAxes.requestedBy = word;

        EXPECT_NE(qcx::io::FormatRunReport(WaterInput(), result, WaterFacts()).find(expected),
                  std::string::npos);
    }
}

} // namespace
