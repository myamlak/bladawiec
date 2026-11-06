// The XC grid's screening counters, per SCF iteration, in the run record.
//
// The grid engine counts the work its screened integration does - live
// points, shell tests and keeps, AO fetches, the evaluated and dense AO slots
// and contraction pairs - beside the energy and potentials it returns, but
// those counters used to be reachable only from the in-process evaluation.
// This file owns the driver-level gate that the record carries them: one
// .toml on disk per functional, the way `qcx run <file.toml>` starts a run,
// and the returned record's xc_screening block read for its CONTENT rather
// than its presence. The dense denominators are identities of the grid
// (points x AOCount and 2 x points x AOCount^2), the screened counters sit
// under them, the rows line up with the iterations, and each row says how
// many integrations it summed.
//
// Counts, never times: this run on a loaded machine produces the same
// counters, so the gate is load-independent by construction.
//
// A capped build (CI's kMaxEngineL below 4) cannot resolve the universal-J
// fitting set's oxygen f and g shells the Kohn-Sham lane resolves at run
// time, so the density-functional rows are skipped there - the same guard
// the neighbouring Kohn-Sham input tests carry.

#include "qcx/driver/run_driver.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_json.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/io/validate_input.hpp"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

using qcx::io::ParseRunInputFile;
using qcx::io::RunInput;
using qcx::io::ValidateInput;

// One unique .toml per call: a reused path could read a previous row's bytes
// if a write failed halfway, and the run under test would be a run of the
// wrong file.
std::filesystem::path TempInputPath(std::string_view tag) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("qcx-xc-screening-" + std::string(tag) + "-" + std::to_string(counter++) + ".toml");
}

// Writes \p toml to a fresh file and parses THAT, so the run's input is the
// bytes on disk. The stream is binary so the parser sees the newlines written.
qcx::Result<RunInput> ParseFile(const std::filesystem::path& path, std::string_view toml) {
    {
        std::ofstream stream(path, std::ios::out | std::ios::binary | std::ios::trunc);

        if (!stream)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kIOError, "cannot write " + path.string()});
        }

        stream << toml;

        if (!stream)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kIOError, "cannot write " + path.string()});
        }
    }

    return ParseRunInputFile(path);
}

// The fixture: the repository's water/STO-3G geometry in Angstrom (no units
// key, which is the parser's default), one [method] body, everything else
// identical between the runs this file compares.
std::string WaterToml(std::string_view methodBody) {
    return std::string(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 0.7569503270127429, 0.58588227657553, 0.0],
    ["H", -0.7569503270127429, 0.58588227657553, 0.0],
]

[basis]
orbital = "sto-3g"

[method]
)") + std::string(methodBody) +
           R"(accuracy = "kNormal"
)";
}

// One run from a written file, validated and driven exactly as the command
// line drives it (main.cpp: ParseRunInputFile, ValidateInput, RunDriverOutcome).
qcx::Result<qcx::driver::RunOutcome> RunFile(const std::filesystem::path& path,
                                             std::string_view toml) {
    auto input = ParseFile(path, toml);
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    if (!input.has_value())
    {
        return std::unexpected(input.error());
    }

    const auto validation = ValidateInput(*input);

    if (!validation.IsValid())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "validation: " + validation.issues.front()});
    }

    return qcx::driver::RunDriverOutcome(*input);
}

} // namespace

// The primary gate: a pure GGA (pbe), a meta-GGA (tau_x) and a hybrid (pbe0)
// on the same water/STO-3G fixture each record their grid's counters per SCF
// iteration, and every row's content answers to the grid the run built.
TEST(XcScreeningTest, PureGgaMetaGgaAndHybridRecordTheirCountsPerIteration) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "the universal-J aux f and g shells exceed this build's kMaxEngineL "
                        "(a reduced-size build; CI builds 6)";
    }

    // Water/STO-3G spans seven AO functions: O 1s, 2s, 2p x, 2p y, 2p z and
    // one 1s per hydrogen.
    constexpr std::size_t kWaterNbasis = 7;

    for (const std::string functional : {"pbe", "tau_x", "pbe0"})
    {
        auto outcome = RunFile(TempInputPath(functional),
                               WaterToml("type = \"rks\"\nfunctional = \"" + functional + "\"\n"));
        ASSERT_TRUE(outcome.has_value()) << functional << ": " << outcome.error().message;

        const qcx::io::RunResult& result = outcome->result;
        ASSERT_TRUE(result.converged) << functional;
        ASSERT_TRUE(result.xcScreening.has_value())
            << functional << ": a Kohn-Sham run must record its grid counters";
        const auto& rows = result.xcScreening->perIteration;
        ASSERT_GT(rows.size(), 0u) << functional;

        // One row per iteration: the row boundary is the Fock build, and the
        // SCF loop builds one Fock per iteration.
        EXPECT_EQ(rows.size(), static_cast<std::size_t>(result.iterations)) << functional;

        std::size_t evaluationTotal = 0;
        // The first row sums the seed density's integration and the first
        // iterate's, so the per-integration grid is read off its row rather
        // than assumed: every later row is a single integration.
        const std::size_t leadEvaluations = static_cast<std::size_t>(rows.front().evaluations);
        ASSERT_EQ(leadEvaluations, 2u)
            << functional << ": the first row carries the seed and the first iterate";
        ASSERT_EQ(rows.front().counts.points % leadEvaluations, 0u) << functional;
        const std::size_t pointsPerEvaluation = rows.front().counts.points / leadEvaluations;
        const std::size_t batchesPerEvaluation = rows.front().counts.batches / leadEvaluations;
        const std::size_t shellTestsPerEvaluation =
            rows.front().counts.shellTests / leadEvaluations;

        for (std::size_t row = 0; row < rows.size(); ++row)
        {
            const qcx::io::RunXcScreeningIteration& iteration = rows[row];
            EXPECT_EQ(iteration.iteration, static_cast<int>(row) + 1) << functional;
            EXPECT_GE(iteration.evaluations, 1) << functional;

            if (row > 0)
            {
                EXPECT_EQ(iteration.evaluations, 1)
                    << functional << ": every later row is one iterate's integration";
            }

            const std::size_t evaluations = static_cast<std::size_t>(iteration.evaluations);
            evaluationTotal += evaluations;
            const qcx::io::RunXcScreeningCounts& counts = iteration.counts;

            // The dense denominators are the schema's own identities, and both
            // sides add linearly over a row's integrations.
            EXPECT_EQ(counts.aoSlotsDense, counts.points * kWaterNbasis) << functional;
            EXPECT_EQ(counts.contractionPairsDense, 2 * counts.points * kWaterNbasis * kWaterNbasis)
                << functional;
            // The closed-shell lane evaluates the pair (rho, rho), one shared
            // density-weight pass: AOCount^2 terms per integration, summed
            // over the row's integrations.
            EXPECT_EQ(counts.densityWeightTerms, evaluations * kWaterNbasis * kWaterNbasis)
                << functional;

            // The screened counters are bounded by the dense ones, and the
            // grid did work on every row.
            EXPECT_GT(counts.points, 0u) << functional;
            EXPECT_GT(counts.batches, 0u) << functional;
            EXPECT_GT(counts.aoFetches, 0u) << functional;
            EXPECT_LE(counts.aoFetches, counts.points) << functional;
            EXPECT_LE(counts.shellKeeps, counts.shellTests) << functional;
            EXPECT_LE(counts.aoSlotsEvaluated, counts.aoSlotsDense) << functional;
            EXPECT_LE(counts.contractionPairs, counts.contractionPairsDense) << functional;

            // Points, batches and shell tests are properties of the grid
            // alone, so per integration they repeat exactly across one run's
            // rows.
            EXPECT_EQ(counts.points, evaluations * pointsPerEvaluation) << functional;
            EXPECT_EQ(counts.batches, evaluations * batchesPerEvaluation) << functional;
            EXPECT_EQ(counts.shellTests, evaluations * shellTestsPerEvaluation) << functional;
        }

        // The row boundary's arithmetic: the first row carries the Fock build's
        // seed evaluation plus the first iterate's energy-path evaluation,
        // each later row the next iterate's. A converging run therefore makes
        // one integration per iteration plus the seed.
        EXPECT_EQ(evaluationTotal, rows.size() + 1) << functional;
    }
}

// The direct-builder route records the same block: the timeline is attached
// by EVERY Kohn-Sham wiring arm, not only the ladder's (the direct family
// builds its own Coulomb and exchange halves and had to carry the collector
// explicitly). A route that dropped it would leave a run whose grid work is
// invisible in the record while the same run on another builder reports it.
TEST(XcScreeningTest, DirectBuilderRouteAlsoRecordsItsCounts) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "the universal-J aux f and g shells exceed this build's kMaxEngineL "
                        "(a reduced-size build; CI builds 6)";
    }

    constexpr std::size_t kWaterNbasis = 7;
    auto outcome =
        RunFile(TempInputPath("direct"),
                WaterToml("type = \"rks\"\nfunctional = \"pbe\"\nfock_builder = \"direct\"\n"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;
    ASSERT_TRUE(result.converged);
    ASSERT_TRUE(result.xcScreening.has_value()) << "the direct route must carry the block";

    const auto& rows = result.xcScreening->perIteration;
    ASSERT_GT(rows.size(), 0u);
    EXPECT_EQ(rows.size(), static_cast<std::size_t>(result.iterations));
    EXPECT_EQ(rows.front().evaluations, 2);

    for (const auto& iteration : rows)
    {
        const auto& counts = iteration.counts;
        EXPECT_EQ(counts.aoSlotsDense, counts.points * kWaterNbasis);
        EXPECT_EQ(counts.contractionPairsDense, 2 * counts.points * kWaterNbasis * kWaterNbasis);
        EXPECT_EQ(counts.densityWeightTerms,
                  static_cast<std::size_t>(iteration.evaluations) * kWaterNbasis * kWaterNbasis);
    }
}

// The presence rule's other end: a run that never integrated a density
// functional has no grid work to account for, so the key is absent rather
// than an empty block.
TEST(XcScreeningTest, HartreeFockRunCarriesNoScreeningBlock) {
    auto outcome = RunFile(TempInputPath("rhf"), WaterToml("type = \"rhf\"\n"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    EXPECT_TRUE(outcome->result.converged);
    EXPECT_FALSE(outcome->result.xcScreening.has_value());
}
