// The ri_chunk_bytes request record (schema 23), end to end through the driver:
// the disclosure arm of the size-hint key split. The
// key is a pure SIZE HINT that exactly one code path reads - the ri_j_link
// family's disk-rung construction, where it lands in DiskRiFockOptions - so on
// every other family, and on ri_j_link when a fitting in-memory rung rides, the
// run proceeds and the hint is never read. Dropping it is deliberate
// (refusing a size hint on a family with no chunking to size is pedantic);
// dropping it SILENTLY was the defect, and this record is what closes it.
//
// These rows pin the DOCUMENT a consumer reads, because that is where the
// defect lived and where the fix has to be visible: the outcome word, the size
// that was dropped, and the reason - told apart from an input that never named
// the key at all. The energy appears in the same rows in its own right, the
// correctness instrument beside the enforcement instrument, so a record that
// claimed a drop while the run behaved differently fails here rather than
// merely reading oddly. A row that only compared energies could not see the
// disclosure at all; a row that only read the block could not see a run that
// moved.
//
// This is the block's own suite rather than a section of run_driver_test.cpp:
// the record spans two modules' contracts (io's shape, the driver's decision)
// and its rows are the ones that must move together with its schema number.

#include "qcx/driver/run_driver.hpp"
#include "qcx/integrals/limits.hpp"
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

qcx::Result<std::string> RunInputText(const char* toml) {
    auto input = ParseRunInput(toml);

    if (!input.has_value())
    {
        return std::unexpected(input.error());
    }

    return RunDriver(*input);
}

// The ri_chunk_bytes block's OWN text, braces included, or "" when the document
// carries no such block. Scoping the absence assertions to the block is the
// point: "this document contains no reason" would be satisfied by a reason
// belonging to some other block, which is not what the honoured outcome claims.
std::string RiChunkBytesBlock(const std::string& json) {
    const std::string key = "\"ri_chunk_bytes\":";
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

    // The block is one flat object of scalars, so the first closing brace after
    // it ends it - no nesting to match.
    const std::size_t close = json.find('}', open);

    if (close == std::string::npos)
    {
        return {};
    }

    return json.substr(open, close - open + 1);
}

// One number out of the document text (the neighbouring-run comparisons below
// need the total energy, not a JSON parser).
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

// The two H2 inputs differ by ONE line: the chunk-size hint. It is a hint the
// direct family never reads, so the run must be unmoved by it - and the record
// must say that the key was dropped rather than leaving the reader to wonder
// whether an unmentioned key was honoured.
constexpr const char* kDirectWithHint = R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
ri_chunk_bytes = 134217728
)";

constexpr const char* kDirectWithoutHint = R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)";

} // namespace

TEST(RiChunkBytesRecordTest, DirectFamilyRunDisclosesTheDropAndTheEnergyIsUnmoved) {
    const auto dropped = RunInputText(kDirectWithHint);
    ASSERT_TRUE(dropped.has_value())
        << "the size hint was refused on a family with no chunking to size: "
        << dropped.error().message;
    EXPECT_NE(dropped->find("\"converged\": true"), std::string::npos);

    // The record: a reader of the document can see that the key was named, what
    // it asked for, that it was dropped, and which family had no use for it -
    // without the command line or the input file.
    const std::string block = RiChunkBytesBlock(*dropped);
    ASSERT_FALSE(block.empty()) << "the dropped key left no record anywhere: " << *dropped;
    EXPECT_NE(block.find("\"outcome\": \"dropped\""), std::string::npos) << block;
    EXPECT_NE(block.find("\"requested_bytes\": 134217728"), std::string::npos) << block;
    EXPECT_NE(block.find("has no disk rung"), std::string::npos) << block;
    // The word that must not stand beside it: a drop never also claims the hint
    // was used.
    EXPECT_EQ(block.find("honoured"), std::string::npos) << block;

    // The correctness instrument beside the enforcement one: the identical input
    // WITHOUT the key reaches the same energy, which is what makes the record
    // and the run two statements about one fact rather than two guesses.
    const auto withoutKey = RunInputText(kDirectWithoutHint);
    ASSERT_TRUE(withoutKey.has_value()) << withoutKey.error().message;

    const auto droppedEnergy = JsonNumber(*dropped, "\"total_energy_hartree\"");
    const auto controlEnergy = JsonNumber(*withoutKey, "\"total_energy_hartree\"");
    ASSERT_TRUE(droppedEnergy.has_value()) << *dropped;
    ASSERT_TRUE(controlEnergy.has_value()) << *withoutKey;
    EXPECT_DOUBLE_EQ(*droppedEnergy, *controlEnergy);

    // And the third reader state: a run that never named the key carries no
    // block, so "dropped" can never be confused with "never asked" - the
    // distinction the optional block exists for.
    EXPECT_EQ(RiChunkBytesBlock(*withoutKey), std::string{}) << *withoutKey;
}

TEST(RiChunkBytesRecordTest, ForcedDiskRungRunDisclosesTheHonour) {
#if !defined(QcxHasStorage)
    GTEST_SKIP() << "storage module not built (QCX_ENABLE_IO=OFF): the forced disk rung IS the "
                    "storage module's disk-backed builder, so the hint has no reader here";
#else
    // The fixture's explicit universal-J aux carries f shells (L = 3), which is
    // the whole point of the row - the hint is read by the DISK rung's own
    // construction - so a build whose 2e engines stop below L = 3 has no run to
    // record. Skip there (CI lmax=2), as the driver suite's ri_j_link cells do.
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the universal-J aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    // The honoured case has to be reachable from an input file to be pinned at
    // all: the ladder engages the disk rung only after an estimate-time refusal
    // under a budget, so the forced diagnostic route is the one admission that
    // reaches the store at this size.
    const auto honoured = RunInputText(R"(
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
aux = "def2-universal-jfit"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_chunk_bytes = 134217728

[diagnostics]
force_disk_ri = true
)");
    ASSERT_TRUE(honoured.has_value()) << honoured.error().message;
    EXPECT_NE(honoured->find("\"converged\": true"), std::string::npos);

    const std::string block = RiChunkBytesBlock(*honoured);
    ASSERT_FALSE(block.empty()) << *honoured;
    EXPECT_NE(block.find("\"outcome\": \"honoured\""), std::string::npos) << block;
    EXPECT_NE(block.find("\"requested_bytes\": 134217728"), std::string::npos) << block;
    EXPECT_EQ(block.find("\"dropped\""), std::string::npos) << block;
    // No reason on an honoured run: an absent reason is never a fabricated
    // "fine" (the null honesty policy the sibling block follows).
    EXPECT_EQ(block.find("reason"), std::string::npos) << block;

    // The claim the outcome word makes, checked against the run rather than
    // taken on trust: the store ran. The driver-synthesized kDisk mode record
    // has no other producer than the disk rung's own construction.
    EXPECT_NE(honoured->find("\"mode\": \"kDisk\""), std::string::npos) << *honoured;
#endif
}

TEST(RiChunkBytesRecordTest, ARiJLinkLadderFitDropsTheHintTheSiblingBlockCallsFit) {
    // Same fixture, same f shells, same reason as the sibling above: at lmax=2
    // the ri_j_link run cannot be built, so neither of the two drop states this
    // block distinguishes is reachable.
    if (!qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "the universal-J aux f shells exceed this build's kMaxEngineL (CI lmax=2)";
    }

    // The second drop state, and the one that justifies this block NOT reusing
    // the sibling vocabulary: on ri_j_link the `disk` RUNG REQUEST is not lost
    // (a fitting memory rung is exactly what it prescribes - the sibling says
    // ladder_fit, which is not a demotion), while the chunk-size hint on the
    // same run IS dropped, because the rung that would read it never engaged.
    // One word with opposite polarity in two sibling blocks would have to be
    // decoded; these two records each state their own key.
    const auto ladder = RunInputText(R"(
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
aux = "def2-universal-jfit"

[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_tensor_mode = "disk"
ri_chunk_bytes = 134217728
)");
    ASSERT_TRUE(ladder.has_value()) << ladder.error().message;
    EXPECT_NE(ladder->find("\"converged\": true"), std::string::npos);

    // The sibling block on this very run: the rung request got what it
    // prescribed, and the disk store did not run.
    EXPECT_NE(ladder->find("\"outcome\": \"ladder_fit\""), std::string::npos) << *ladder;

    const std::string block = RiChunkBytesBlock(*ladder);
    ASSERT_FALSE(block.empty()) << *ladder;
    EXPECT_NE(block.find("\"outcome\": \"dropped\""), std::string::npos) << block;
    EXPECT_NE(block.find("did not engage"), std::string::npos) << block;
    // The other drop text, not this one: the family HAS a disk rung here, it
    // simply did not engage, and the record must not blame the family for it.
    EXPECT_EQ(block.find("has no disk rung"), std::string::npos) << block;
    EXPECT_EQ(block.find("honoured"), std::string::npos) << block;
}
