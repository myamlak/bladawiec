// The eri_store request record, end to end through the driver: the
// disclosure half of the disk-tier ERI store key (`method.eri_cache_store`, the
// engine-decorator seam). The key's consumption site is a decorator the
// ENGINE cannot see - the driver constructs `storage`'s CachedEriBatchEngine,
// hands `integrals` an opaque engine pair (the module boundary) and owns the
// decorator for the run - so without these rows a run that asked for a store and
// a run that never asked serialize identically, and a request the run quietly
// dropped reads as an ordinary run.
//
// These rows pin the DOCUMENT a consumer reads, because that is where the
// defect would live and where the fix has to be visible, and they pin it beside
// the run's own arithmetic: a record that claimed a store ran while the energy
// moved would fail here rather than merely reading oddly. Every row drives the
// real pipeline (ParseRunInput -> RunDriver), so the key is reached the way a
// user's TOML reaches it - not by calling the wiring helper directly.
//
// What the rows are chosen to separate:
//   - HONOURED: a plain-path (no symmetry reduction) direct run opens the store
//     at the named path, serves the later SCF iterations from it, and lands on
//     the same energy as the identical input without the key.
//   - PERSISTED: the same input run a second time over the same file misses on
//     nothing, which is the cache's whole reason for existing and is only
//     observable through the record's own counters.
//   - DEMOTED, three ways that are three different sentences: a store the
//     process cannot open carries the store's own error; a family with no batch
//     engine pair names that family; and a molecule whose symmetry reduction
//     engages the class-aware path names the disengagement, because there the
//     decorator is installed and the engine never calls it.
//   - REFUSED BY NAME: the unrestricted legs, which the key's own increment
//     boundary leaves unwired.
//   - NEVER ASKED: no block at all, so "dropped" can never be read as
//     "never asked".
//
// This is the block's own suite rather than a section of run_driver_test.cpp:
// the record spans two modules' contracts (io's shape, the driver's decision)
// and its rows are the ones that must move together with its schema number.

#include "qcx/driver/run_driver.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_json.hpp"

#include <atomic>
#include <cstddef>
#include <filesystem>
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

// The eri_store block's OWN text, braces included, or "" when the document
// carries no such block. The braces are MATCHED rather than scanned to the
// first closing one: this block nests (hit_quartets_by_class is an array of
// objects), so a flat scan would cut it short the day a class row appears, and
// an assertion made against a truncated block could pass while the block it
// describes is wrong. Scoping the absence assertions to the block is the point:
// "this document contains no demoted_reason" would be satisfied by a reason
// belonging to some other block, which is not what an honoured store claims.
std::string EriStoreBlock(const std::string& json) {
    const std::string key = "\"eri_store\":";
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

// One number out of the document text (or, given the block's own text, out of
// the block) - the neighbouring-run comparisons below need the total energy and
// the store's counters, not a JSON parser.
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

// The TOML spelling of a path: the key's value is a basic string, so a Windows
// separator reaching it is read as an escape sequence (`\U...` is unicode
// escape syntax, and the parse fails) - the same reason every path in the
// repository's own input files is written with forward slashes.
std::string TomlPath(std::string path) {
    for (char& c : path)
    {
        if (c == '\\')
        {
            c = '/';
        }
    }

    return path;
}

// One unique scratch store path per call. The key takes the path as its whole
// request, and a store is append-only over its chunks, so a row that reused a
// path would measure the previous row's content (a warm store, all hits) rather
// than the fresh-open behaviour it was written for.
std::string TempStorePath(std::string_view tag) {
    static std::atomic<int> counter{0};
    const std::string path = TomlPath(std::filesystem::temp_directory_path().string()) +
                             "/qcx-r75-" + std::string(tag) + "-" +
                             std::to_string(counter.fetch_add(1)) + ".h5";

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    return path;
}

// The row's fixture, in one place so every leg of a comparison is the same
// molecule: a deliberately asymmetric water, whose point group is C1 and which
// therefore takes the direct family's PLAIN path. That is a property the cell
// needs, not a convenience - the engine disengages its whole engine tier
// (decorator included) on the class-aware path, so a symmetric fixture cannot
// measure an engaged store at all (the row below pins exactly that).
constexpr const char* kAsymmetricWater = R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.62, 0.55, 0.0], ["H", -0.71, 0.41, 0.55]]
[basis]
orbital = "sto-3g"
)";

// The thread cap on every row that compares two energies, and it is part of the
// INSTRUMENT rather than a convenience. MEASURED (2026-09-17, this fixture,
// this machine): with the default team, two consecutive runs of the same input
// in one process disagree in the last one or two digits of the total energy
// (~7e-14 to 1.3e-13 Ha over six pairs) - and the row whose store could not be
// OPENED shows the same spread, so the disagreement belongs to the SCF's
// parallel reduction order and not to anything the store does. A bit-parity
// assertion made against that would be measuring the scheduler. One thread
// fixes the reduction order, which is what makes the store's own claim - the
// bytes it serves are the engine's verbatim, so an honoured run and the same
// run without the key agree BIT FOR BIT - assertable at all.
constexpr const char* kOneThreadTeam = R"(
[resources]
thread_cap = 1
)";

std::string DirectWaterWithStore(const std::string& storePath) {
    return std::string(kAsymmetricWater) + R"(
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
eri_cache_store = ")" +
           storePath + "\"\n" + kOneThreadTeam;
}

const std::string kDirectWaterWithoutStore = std::string(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.62, 0.55, 0.0], ["H", -0.71, 0.41, 0.55]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)") + kOneThreadTeam;

} // namespace

TEST(EriStoreRecordTest, PlainPathRunOpensTheStoreAndLandsOnTheUnmovedEnergy) {
#if !defined(QcxHasStorage)
    // This row is about a store that OPENED, and a build configured with
    // QCX_ENABLE_IO=OFF compiles no storage module: the driver demotes the
    // request with the build's own cause (the disclosure half of the key,
    // unchanged), nothing is written at the path, and every assertion below
    // about an honoured run is unreachable by construction.
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): this row asserts an opened "
                    "store";
#else
    const std::string path = TempStorePath("plain");
    const auto served = RunInputText(DirectWaterWithStore(path));

    ASSERT_TRUE(served.has_value())
        << "the store request failed the run: " << served.error().message;
    EXPECT_NE(served->find("\"converged\": true"), std::string::npos);

    // The run reached the store: the file the input named exists and carries
    // bytes. The record below is a statement ABOUT this file, and a record that
    // claimed a disk tier while the path held nothing would be the fabricated
    // reading this block exists to prevent.
    std::error_code sizeError;
    const auto bytes = std::filesystem::file_size(path, sizeError);
    EXPECT_FALSE(sizeError) << "the honoured request named a store that was never written: "
                            << path;
    EXPECT_GT(bytes, 0U) << path;

    const std::string block = EriStoreBlock(*served);
    ASSERT_FALSE(block.empty()) << "the honoured request left no record: " << *served;
    EXPECT_NE(block.find("\"engaged\": \"disk\""), std::string::npos) << block;
    EXPECT_NE(block.find("\"demoted\": false"), std::string::npos) << block;
    EXPECT_NE(block.find("\"path\": \"" + path + "\""), std::string::npos) << block;
    // Nothing to explain on the honoured arm: an absent reason is never a
    // fabricated "fine" (the null honesty policy).
    EXPECT_EQ(block.find("demoted_reason"), std::string::npos) << block;

    // The counters are the decorator's own traffic, and a fresh store cannot
    // have served anything: every quartet the run asked for was a miss. This is
    // what separates "a decorator was handed to the builder" from "the
    // decorator ran", which is the distinction the whole instrument is built on
    // - a store that is opened and never called would otherwise read as a
    // served run.
    const auto misses = JsonNumber(block, "\"miss_quartets\"");
    ASSERT_TRUE(misses.has_value()) << block;
    EXPECT_GT(*misses, 0.0) << block;

    // The correctness instrument beside the enforcement one: the identical
    // input WITHOUT the key reaches the same energy - the store returns the
    // engine's own bytes, so the tier cannot move the answer.
    const auto control = RunInputText(kDirectWaterWithoutStore);
    ASSERT_TRUE(control.has_value()) << control.error().message;

    const auto servedEnergy = JsonNumber(*served, "\"total_energy_hartree\"");
    const auto controlEnergy = JsonNumber(*control, "\"total_energy_hartree\"");
    ASSERT_TRUE(servedEnergy.has_value()) << *served;
    ASSERT_TRUE(controlEnergy.has_value()) << *control;
    EXPECT_DOUBLE_EQ(*servedEnergy, *controlEnergy);

    // And the third reader state: a run that never named the key carries no
    // block, so "served" can never be confused with "never asked" - the
    // distinction the optional block exists for.
    EXPECT_EQ(EriStoreBlock(*control), std::string{}) << *control;

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
#endif
}

TEST(EriStoreRecordTest, SecondRunOverTheSameStoreMissesOnNothing) {
#if !defined(QcxHasStorage)
    // The persistence row: a second run over the same FILE. The file is the
    // storage module's, so a build configured with QCX_ENABLE_IO=OFF has no
    // store to re-open and no hit path to count.
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): this row re-opens a written "
                    "store";
#else
    // The persistence row, and the one that is only observable through the
    // record: the store is a FILE, so a second run over it recomputes nothing.
    // A decorator that engaged but wrote nothing usable - or one whose stats
    // were fabricated from the install rather than read from the run - cannot
    // produce a zero here, because the count comes from the store's own hit
    // path having served every request.
    const std::string path = TempStorePath("warm");

    const auto cold = RunInputText(DirectWaterWithStore(path));
    ASSERT_TRUE(cold.has_value()) << cold.error().message;

    const auto warm = RunInputText(DirectWaterWithStore(path));
    ASSERT_TRUE(warm.has_value()) << warm.error().message;

    const std::string block = EriStoreBlock(*warm);
    ASSERT_FALSE(block.empty()) << *warm;
    EXPECT_NE(block.find("\"engaged\": \"disk\""), std::string::npos) << block;
    EXPECT_NE(block.find("\"demoted\": false"), std::string::npos) << block;

    const auto hits = JsonNumber(block, "\"hit_quartets\"");
    const auto misses = JsonNumber(block, "\"miss_quartets\"");
    ASSERT_TRUE(hits.has_value()) << block;
    ASSERT_TRUE(misses.has_value()) << block;
    EXPECT_GT(*hits, 0.0) << block;
    EXPECT_DOUBLE_EQ(*misses, 0.0) << block;

    // The arithmetic is unmoved across the two runs, which is what the stored
    // bytes being the engine's verbatim means: a served quartet equals a
    // recomputed one.
    const auto coldEnergy = JsonNumber(*cold, "\"total_energy_hartree\"");
    const auto warmEnergy = JsonNumber(*warm, "\"total_energy_hartree\"");
    ASSERT_TRUE(coldEnergy.has_value()) << *cold;
    ASSERT_TRUE(warmEnergy.has_value()) << *warm;
    EXPECT_DOUBLE_EQ(*coldEnergy, *warmEnergy);

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
#endif
}

TEST(EriStoreRecordTest, AStoreThatCannotOpenDemotesWithTheStoresOwnError) {
#if !defined(QcxHasStorage)
    // The row's subject is the STORE's own refusal (the temp directory cannot
    // be opened as a store), and a build configured with QCX_ENABLE_IO=OFF has
    // no store to refuse: its demotion carries the build's cause instead, so
    // the assertions below would hold for a reason the row does not test.
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): this row needs a store that "
                    "could have opened";
#else
    // The demotion's first cause: the path exists and cannot be opened as a
    // store (the temp DIRECTORY is the fixture - an open that must fail on
    // every platform). The owner's 2026-09-12 ruling is what this row pins: a
    // store that cannot open costs the run a cache, not the run, so the run
    // COMPLETES on the raw engine pair - and the record names both what was
    // asked for and why it was not honoured.
    const std::string dir = TomlPath(std::filesystem::temp_directory_path().string());
    const auto run = RunInputText(DirectWaterWithStore(dir));

    ASSERT_TRUE(run.has_value()) << "a store that cannot open refused the run instead of "
                                    "demoting it: "
                                 << run.error().message;
    EXPECT_NE(run->find("\"converged\": true"), std::string::npos);

    const std::string block = EriStoreBlock(*run);
    ASSERT_FALSE(block.empty()) << "the demoted request left no record: " << *run;
    EXPECT_NE(block.find("\"demoted\": true"), std::string::npos) << block;
    EXPECT_EQ(block.find("\"engaged\": \"disk\""), std::string::npos) << block;
    // The path that was asked for is written on the demoted arm too: a
    // disclosure that did not say what was requested could not be read as one.
    EXPECT_NE(block.find("\"path\": \""), std::string::npos) << block;
    EXPECT_NE(block.find("demoted_reason"), std::string::npos) << block;
    // WHICH demotion it was, and the distinction is the row's own point: this
    // run reached the decorator and the STORE refused it, so the reason cannot
    // be the family text. A run that never installed a factory at all would
    // produce exactly that other sentence, and a record that answered "this
    // family has no seam" on a family that has one would send a reader to
    // delete a key that works (the store's own words are pinned by the CLI
    // invocation recorded with this change; the exclusion is what makes this
    // row die when the install is removed).
    EXPECT_EQ(block.find("has no such seam"), std::string::npos) << block;
    // The counters are the decorator's and no decorator served: zero, never a
    // fabricated reading, and never a substitute for the `engaged` word.
    const auto hits = JsonNumber(block, "\"hit_quartets\"");
    ASSERT_TRUE(hits.has_value()) << block;
    EXPECT_DOUBLE_EQ(*hits, 0.0) << block;

    // The demoted run is the unmoved run: the demotion is an arrangement, not
    // an approximation, so the energy is the control's exactly.
    const auto control = RunInputText(kDirectWaterWithoutStore);
    ASSERT_TRUE(control.has_value()) << control.error().message;
    const auto runEnergy = JsonNumber(*run, "\"total_energy_hartree\"");
    const auto controlEnergy = JsonNumber(*control, "\"total_energy_hartree\"");
    ASSERT_TRUE(runEnergy.has_value()) << *run;
    ASSERT_TRUE(controlEnergy.has_value()) << *control;
    EXPECT_DOUBLE_EQ(*runEnergy, *controlEnergy);
#endif
}

TEST(EriStoreRecordTest, AFamilyWithNoSeamDemotesWithTheFamilyNamed) {
    // The demotion's second cause: the run resolved to a family whose options
    // carry no batch engine pair, so the store was never opened. The lean member
    // is the ladder's default at this size - reached here by dropping the family
    // word entirely - and it is the case a reader most needs told apart from a
    // served run, because it is what a user gets by forgetting the family word.
    const std::string path = TempStorePath("lean");
    const std::string toml = std::string(kAsymmetricWater) + R"(
[method]
type = "rhf"
accuracy = "kNormal"
eri_cache_store = ")" + path +
                             "\"\n" + kOneThreadTeam;

    const auto run = RunInputText(toml);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    const std::string block = EriStoreBlock(*run);
    ASSERT_FALSE(block.empty()) << *run;
    EXPECT_NE(block.find("\"demoted\": true"), std::string::npos) << block;
    EXPECT_EQ(block.find("\"engaged\": \"disk\""), std::string::npos) << block;
    EXPECT_NE(block.find("LEAN member"), std::string::npos) << block;
    // The store it refused to open was never created: the demotion is not a
    // quiet "opened it anyway".
    std::error_code ignored;
    EXPECT_FALSE(std::filesystem::exists(path, ignored)) << path;
}

TEST(EriStoreRecordTest, AClassAwareRunDisclosesTheDisengagedEngineTier) {
    // The demotion a record that read "a decorator was installed" would report
    // as SUCCESS. H2 is symmetric, the driver's own reduction engages the
    // class-aware path, and the builder disengages the whole engine tier -
    // decorator with it - inside its Create, long after the driver
    // handed its factory over: the factory is never called, no store is ever
    // opened, and the run's batches go nowhere near the decorator. What this
    // row pins is that the block says so instead of claiming a disk tier that
    // served nothing. Its precondition is the measured `class_table_bytes` on
    // this fixture's own record, asserted below rather than assumed: without
    // it the row would pass on a fixture that no longer exercises the
    // disengagement.
    //
    // The traffic test inside the record maker (a store that exists and served
    // nothing) is NOT what this row reaches - it is unreachable by both of
    // today's disengagements, which are caught one step earlier by the absent
    // store - and this row does not claim to pin it.
    const std::string path = TempStorePath("symmetric");
    const std::string toml = R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
eri_cache_store = ")" + path +
                             "\"\n" + kOneThreadTeam;

    const auto run = RunInputText(toml);
    ASSERT_TRUE(run.has_value()) << run.error().message;

    // The precondition, read off the record rather than asserted from theory:
    // this run really did build a class table, so it really is the case the row
    // claims to pin. Without it the row would pass on a fixture that no longer
    // exercises the disengagement.
    const auto classBytes = JsonNumber(*run, "\"class_table_bytes\"");
    ASSERT_TRUE(classBytes.has_value()) << *run;
    ASSERT_GT(*classBytes, 0.0) << "the fixture no longer engages the class-aware path: " << *run;

    const std::string block = EriStoreBlock(*run);
    ASSERT_FALSE(block.empty()) << *run;
    EXPECT_NE(block.find("\"demoted\": true"), std::string::npos) << block;
    EXPECT_EQ(block.find("\"engaged\": \"disk\""), std::string::npos) << block;
    EXPECT_NE(block.find("never called its factory"), std::string::npos) << block;

    const auto hits = JsonNumber(block, "\"hit_quartets\"");
    ASSERT_TRUE(hits.has_value()) << block;
    EXPECT_DOUBLE_EQ(*hits, 0.0) << block;

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

TEST(EriStoreRecordTest, TheUnrestrictedLegsRefuseTheKeyByName) {
    // The increment boundary, pinned as a REFUSAL rather than a silence: an
    // explicit request the run drops is a disclosure-rule violation whether or not a
    // follow-on is planned, so the key is answered on the unrestricted legs
    // instead of ignored there. The refusal names the state and the remedy,
    // which is what makes a half-applied contract conformant.
    const std::string path = TempStorePath("uhf");
    const std::string toml = R"(
[molecule]
charge = 0
multiplicity = 3
atoms = [["O", 0.0, 0.0, 0.0], ["H", 0.62, 0.55, 0.0], ["H", -0.71, 0.41, 0.55]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
eri_cache_store = ")" + path +
                             "\"\n";

    const auto run = RunInputText(toml);
    ASSERT_FALSE(run.has_value()) << "the unrestricted leg ran with the key silently dropped: "
                                  << *run;
    EXPECT_NE(run.error().message.find("not yet wired on the UHF path"), std::string::npos)
        << run.error().message;
    // The refusal names the key it is about, so a user reading it can find the
    // line to delete.
    EXPECT_NE(run.error().message.find("method.eri_cache_store"), std::string::npos)
        << run.error().message;
}
