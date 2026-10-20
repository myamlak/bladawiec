// The eri_cache request record, end to end through the driver: the
// requested-vs-ran pairing of the in-memory ERI cache budget
// (`method.eri_cache_bytes`, schema 43).
//
// The key's consumer is the engine's own Create-time cache budget
// (FockBuildOptions::maxCacheBytes), and the budget that FIRES is the
// engine's decision, not the input's: the Create-time sure-fit clamp verifies
// the grant against the workspace budget and every path that cannot serve the
// cache fires zero. So the defect this suite exists to prevent is a document
// that shows the requested number as though it were the one that ran.
//
// Each row drives the real pipeline (ParseRunInput -> RunDriver), so the key
// is reached the way a user's TOML reaches it:
//   - HONOURED: the grant fires whole on both legs, and the run's energy is
//     unmoved beside the identical input without the key (a cache cannot move
//     the answer).
//   - CLAMPED: a request the workspace cannot hold fires below the grant, and
//     the record says so instead of letting the request stand as the ran
//     budget.
//   - REFUSED BY NAME: a route that owns no cache budget (the lean member,
//     every family but the direct one) and every run at memory_cap_gib = 0,
//     where no workspace budget exists to clamp the grant against.
//   - NEVER ASKED: no block at all, so an inherited default can never read as
//     a request.
//
// This is the block's own suite rather than a section of run_driver_test.cpp:
// the record spans two modules' contracts (io's shape, the engine's own mode
// record) and its rows are the ones that must move together with its schema
// number.

#include "qcx/driver/run_driver.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_json.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <string_view>

namespace {

using qcx::driver::RunDriver;
using qcx::io::ParseRunInput;

qcx::Result<std::string> RunInputText(const std::string& toml) {
    auto input = ParseRunInput(toml);

    if (!input.has_value())
    {
        return std::unexpected(input.error());
    }

    return RunDriver(*input);
}

// The eri_cache block's OWN text, braces included, or "" when the document
// carries no such block. Matched by brace depth rather than scanned to the
// first closing one, so a member that nests later cannot make an assertion
// pass against a truncated block.
std::string EriCacheBlock(const std::string& json) {
    const std::string key = "\"eri_cache\":";
    const std::size_t at = json.find(key);

    if (at == std::string::npos)
    {
        return {};
    }

    const std::size_t open = json.find('{', at);

    if (open == std::string::npos)
    {
        return {};
    }

    int depth = 0;

    for (std::size_t i = open; i < json.size(); ++i)
    {
        if (json[i] == '{')
        {
            ++depth;
        } else if (json[i] == '}')
        {
            --depth;

            if (depth == 0)
            {
                return json.substr(open, i - open + 1);
            }
        }
    }

    return {};
}

// One number out of a document (or out of the block's own text) - the
// energy comparisons below need the number, not a JSON parser.
std::optional<double> JsonNumber(const std::string& json, std::string_view key) {
    const auto keyPos = json.find(key);

    if (keyPos == std::string::npos)
    {
        return std::nullopt;
    }

    const auto valuePos = json.find(':', keyPos);

    if (valuePos == std::string::npos)
    {
        return std::nullopt;
    }

    try
    { return std::stod(json.substr(valuePos + 1)); } catch (const std::exception&)
    {
        // The null policy: an unset optional member serializes as null, and
        // std::stod throws on it - absent, not zero.
        return std::nullopt;
    }
}

// The restricted fixture: a deliberately asymmetric water, whose point group
// is C1 and which therefore takes the direct family's PLAIN path. That is a
// property the honoured rows need - the engine disengages its whole cache
// tier on the class-aware path, so a symmetric fixture would fire zero and
// could not measure a grant that fired whole.
constexpr const char* kAsymmetricWater = R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.62, 0.55, 0.0], ["H", -0.71, 0.41, 0.55]]
[basis]
orbital = "sto-3g"
)";

// The unrestricted fixture: the same asymmetric water as the H2O+ cation
// doublet, so the per-spin coulomb/exchange pair is the wiring under test.
constexpr const char* kUnrestrictedWater = R"(
[molecule]
charge = 1
multiplicity = 2
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.62, 0.55, 0.0], ["H", -0.71, 0.41, 0.55]]
[basis]
orbital = "sto-3g"
)";

// One thread on the rows that compare two energies: the SCF's parallel
// reduction order moves the last digits between runs, and a bit-parity claim
// made over that would be measuring the scheduler (the eri_store suite's own
// measured note, same fixture family).
constexpr const char* kOneThreadTeam = R"(
[resources]
thread_cap = 1
)";

// 1 MiB: a grant small enough that the engine's sure-fit clamp fires it whole
// against the default 16 GiB cap, and distinct from the ~5 GiB grant the
// unrestricted legs derive from that cap - so `effective_bytes` equalling the
// request is evidence the key OVERRODE the inherited grant rather than a
// coincidence of two equal numbers.
constexpr std::size_t kOneMebibyte = 1048576;

// kNormal's energy convergence leg (the 1e-8/1e-6 operating default): the
// SCF stops when an iteration's energy change falls below it. Two converged
// runs of the same system are equal only to this leg, never to the last bit -
// which is the instrument the unrestricted row below needs.
constexpr double kNormalEnergyLegHartree = 1e-8;

std::string RunWithCache(const char* fixture,
                         const char* methodWord,
                         std::size_t bytes,
                         const std::string& extra = {}) {
    const std::string method =
        std::string("[method]\ntype = \"") + methodWord +
        "\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\neri_cache_bytes = " +
        std::to_string(bytes) + "\n";

    return std::string(fixture) + method + extra;
}

const std::string kRhfWithoutKey = std::string(kAsymmetricWater) + R"(
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)" + kOneThreadTeam;

const std::string kUhfWithoutKey = std::string(kUnrestrictedWater) + R"(
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
)" + kOneThreadTeam;

} // namespace

TEST(EriCacheRecordTest, RestrictedRunFiresTheGrantWholeAndLandsOnTheUnmovedEnergy) {
    const auto run =
        RunInputText(RunWithCache(kAsymmetricWater, "rhf", kOneMebibyte, kOneThreadTeam));

    ASSERT_TRUE(run.has_value()) << "the cache request failed the run: " << run.error().message;
    EXPECT_NE(run->find("\"converged\": true"), std::string::npos);

    const std::string block = EriCacheBlock(*run);
    ASSERT_FALSE(block.empty()) << "the honoured request left no record: " << *run;
    EXPECT_NE(block.find("\"requested_bytes\": " + std::to_string(kOneMebibyte)), std::string::npos)
        << block;
    EXPECT_NE(block.find("\"granted_bytes\": " + std::to_string(kOneMebibyte)), std::string::npos)
        << block;
    // The engine FIRED the grant whole: the mode record's cache term divided
    // by three is the request, and nothing was clamped. A record that merely
    // echoed the request would pass the two lines above but not this one.
    EXPECT_NE(block.find("\"effective_bytes\": " + std::to_string(kOneMebibyte)), std::string::npos)
        << block;
    EXPECT_NE(block.find("\"clamped\": false"), std::string::npos) << block;

    // The correctness instrument: the identical input WITHOUT the key reaches
    // the same energy. The cache returns the engine's own bytes, so the tier
    // cannot move the answer.
    const auto control = RunInputText(kRhfWithoutKey);
    ASSERT_TRUE(control.has_value()) << control.error().message;

    const auto runEnergy = JsonNumber(*run, "\"total_energy_hartree\"");
    const auto controlEnergy = JsonNumber(*control, "\"total_energy_hartree\"");
    ASSERT_TRUE(runEnergy.has_value()) << *run;
    ASSERT_TRUE(controlEnergy.has_value()) << *control;
    EXPECT_DOUBLE_EQ(*runEnergy, *controlEnergy);
}

TEST(EriCacheRecordTest, UnrestrictedRunOverridesTheCapDerivedGrantOnBothHalves) {
    const auto run =
        RunInputText(RunWithCache(kUnrestrictedWater, "uhf", kOneMebibyte, kOneThreadTeam));

    ASSERT_TRUE(run.has_value()) << "the cache request failed the run: " << run.error().message;
    EXPECT_NE(run->find("\"converged\": true"), std::string::npos);

    const std::string block = EriCacheBlock(*run);
    ASSERT_FALSE(block.empty()) << "the honoured request left no record: " << *run;
    EXPECT_NE(block.find("\"requested_bytes\": " + std::to_string(kOneMebibyte)), std::string::npos)
        << block;
    // The request OVERRODE the cap-derived grant (about 5 GiB at the default
    // 16 GiB cap), on both halves: the record's effective budget is the
    // minimum across the half-builds that made a mode decision, and it equals
    // the request - a half left on the inherited grant could not produce this
    // number.
    EXPECT_NE(block.find("\"effective_bytes\": " + std::to_string(kOneMebibyte)), std::string::npos)
        << block;
    EXPECT_NE(block.find("\"clamped\": false"), std::string::npos) << block;

    const auto control = RunInputText(kUhfWithoutKey);
    ASSERT_TRUE(control.has_value()) << control.error().message;
    EXPECT_NE(control->find("\"converged\": true"), std::string::npos);

    const auto runEnergy = JsonNumber(*run, "\"total_energy_hartree\"");
    const auto controlEnergy = JsonNumber(*control, "\"total_energy_hartree\"");
    ASSERT_TRUE(runEnergy.has_value()) << *run;
    ASSERT_TRUE(controlEnergy.has_value()) << *control;
    // The cache cannot move the answer beyond the SCF's own convergence leg.
    // It does move the engine's MODE on this fixture: without the key the
    // unrestricted leg derives its ~5 GiB grant, whose 3x footprint charge
    // (about 15 GiB) leaves the exchange half too little of the default
    // 16 GiB budget and it decides the LightPath; the 1 MiB request charges
    // 3 MiB and both halves keep the FastPath. The restricted row above,
    // whose no-key leg has no grant at all, keeps its bit-identity pin - that
    // is the pair with identical arithmetic. These two runs are converged
    // answers from two engine modes, and the light mode is equivalent to the
    // bound its certified lane proves, not bit for bit.
    EXPECT_NEAR(*runEnergy, *controlEnergy, kNormalEnergyLegHartree)
        << "run " << *runEnergy << " vs control " << *controlEnergy;
}

TEST(EriCacheRecordTest, AGrantTheWorkspaceCannotHoldFiresBelowTheRequest) {
    // A 2 GiB request under a 1 GiB cap: the engine's Create-time sure-fit
    // clamp cannot fire it whole, so the record must show the request and the
    // fired budget as two different numbers - the clamped flag is the
    // statement, and the effective budget (0 or the clamped grant) is below
    // the request in either arm.
    std::string toml = RunWithCache(kAsymmetricWater, "rhf", 2147483648);
    toml += R"(
[resources]
thread_cap = 1
memory_cap_gib = 1
)";

    const auto run = RunInputText(toml);
    ASSERT_TRUE(run.has_value()) << "the cache request failed the run: " << run.error().message;

    const std::string block = EriCacheBlock(*run);
    ASSERT_FALSE(block.empty()) << *run;
    EXPECT_NE(block.find("\"requested_bytes\": 2147483648"), std::string::npos) << block;
    EXPECT_NE(block.find("\"clamped\": true"), std::string::npos) << block;

    const auto requested = JsonNumber(block, "\"requested_bytes\"");
    const auto effective = JsonNumber(block, "\"effective_bytes\"");
    ASSERT_TRUE(requested.has_value()) << block;
    ASSERT_TRUE(effective.has_value()) << "the cache did not report a fired budget: " << block;
    EXPECT_LT(*effective, *requested) << block;
}

TEST(EriCacheRecordTest, AKeyOnARouteWithoutTheKnobIsRefusedByName) {
    // The RI-J link family assembles its own options struct, which carries no
    // in-memory cache budget: the request is answered, not dropped.
    const std::string toml = std::string(kAsymmetricWater) + R"(
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
eri_cache_bytes = 1048576
)";

    const auto run = RunInputText(toml);
    ASSERT_FALSE(run.has_value());
    EXPECT_EQ(run.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(run.error().message.find("method.eri_cache_bytes resolves to fock_builder = "
                                       "\"ri_j_link\""),
              std::string::npos)
        << "message: " << run.error().message;
    EXPECT_NE(run.error().message.find("maxCacheBytes"), std::string::npos)
        << "message: " << run.error().message;

    // The lean member is the ladder's default at this size: it builds
    // LeanFockBuildOptions, which has no cache field at all.
    const auto lean = RunInputText(std::string(kAsymmetricWater) + R"(
[method]
type = "rhf"
accuracy = "kNormal"
eri_cache_bytes = 1048576
)");
    ASSERT_FALSE(lean.has_value());
    EXPECT_EQ(lean.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(lean.error().message.find("the direct family's LEAN member"), std::string::npos)
        << "message: " << lean.error().message;
}

TEST(EriCacheRecordTest, AKeyAtTheNullBudgetPathIsRefusedWithTheGateNamed) {
    // The engine's sure-fit clamp reads the workspace budget's remaining
    // bytes; memory_cap_gib = 0 selects the legacy null-budget path, where no
    // such gate exists. The refusal names the key that would re-engage it.
    std::string toml = RunWithCache(kAsymmetricWater, "rhf", kOneMebibyte);
    toml += R"(
[resources]
memory_cap_gib = 0
)";

    const auto run = RunInputText(toml);
    ASSERT_FALSE(run.has_value());
    EXPECT_EQ(run.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(run.error().message.find("method.eri_cache_bytes requires a positive "
                                       "memory_cap_gib"),
              std::string::npos)
        << "message: " << run.error().message;
}

TEST(EriCacheRecordTest, AnOmittedKeyLeavesNoBlockEvenOnTheGrantLegs) {
    // The unrestricted legs compute a cap-derived grant with no key written:
    // that grant is the system's judgement, not a request, and the block is
    // absent - so "never asked" can never be read as "asked and clamped".
    const auto uhf = RunInputText(kUhfWithoutKey);
    ASSERT_TRUE(uhf.has_value()) << uhf.error().message;
    EXPECT_EQ(uhf->find("eri_cache"), std::string::npos) << *uhf;

    const auto rhf = RunInputText(kRhfWithoutKey);
    ASSERT_TRUE(rhf.has_value()) << rhf.error().message;
    EXPECT_EQ(rhf->find("eri_cache"), std::string::npos) << *rhf;
}
