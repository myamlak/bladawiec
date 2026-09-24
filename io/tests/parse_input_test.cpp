// The run-input parser and the
// result-JSON serializer, exercised against the schema vocabulary and the
// null-vs-fabricated policy. The parser tests use the exact H2 example of
// the stage plan; the serializer tests pin the JSON shape (keys, RHF
// spin_squared null, the optional properties block: null when unset,
// populated when set).

#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_json.hpp"

#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>

namespace {

using qcx::io::BuilderKind;
using qcx::io::CoordinateUnit;
using qcx::io::DeviceSelectorText;
using qcx::io::DeviceTarget;
using qcx::io::GuessKind;
using qcx::io::MethodType;
using qcx::io::ParseRunInput;
using qcx::io::ParseRunInputFile;
using qcx::io::RiTensorMode;
using qcx::io::RunInput;
using qcx::io::RunProperties;
using qcx::io::RunResult;
using qcx::io::SerializeRunResultJson;

const char* kExampleH2Toml = R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.0, 0.0, 0.74],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
max_iterations = 100
energy_tolerance = 1e-8
density_tolerance = 1e-6
use_diis = true
trace_file = "scf_trace.log"
density_dump = "scf_density.dmp"

[guess]
type = "core"
)";

} // namespace

TEST(RunInputParseTest, ParsesThePlanExampleH2File) {
    const auto input = ParseRunInput(kExampleH2Toml);
    ASSERT_TRUE(input.has_value());

    EXPECT_EQ(input->molecule.charge, 0);
    EXPECT_EQ(input->molecule.multiplicity, 1);
    ASSERT_EQ(input->molecule.atoms.size(), 2u);
    EXPECT_EQ(input->molecule.atoms[0].symbol, "H");
    EXPECT_EQ(input->molecule.atoms[0].x, 0.0);
    // The parser converts the file's unit ONCE (schema 28): a file that names
    // no unit is Angstrom, and what lands in the struct is Bohr - 0.74 A is
    // 1.3983973322230698 a0, the exact double 0.74 * kAngstromToBohr produces.
    EXPECT_EQ(input->molecule.atoms[1].z, 1.3983973322230698);
    EXPECT_EQ(input->molecule.coordinateUnit, CoordinateUnit::kAngstrom);

    EXPECT_EQ(input->basis.orbital, "sto-3g");
    EXPECT_FALSE(input->basis.aux.has_value());

    EXPECT_EQ(input->method.method, MethodType::kRhf);
    EXPECT_EQ(input->method.builder, BuilderKind::kDirect);
    EXPECT_EQ(input->method.accuracy, qcx::integrals::AccuracyPreset::kNormal);

    EXPECT_EQ(input->scf.maxIterations, 100);
    EXPECT_EQ(input->scf.energyTolerance, 1e-8);
    EXPECT_EQ(input->scf.densityTolerance, 1e-6);
    EXPECT_TRUE(input->scf.useDiis);
    EXPECT_EQ(input->scf.traceFile, "scf_trace.log");
    EXPECT_EQ(input->scf.densityDumpFile, "scf_density.dmp");

    EXPECT_EQ(input->guess, GuessKind::kCore);
}

TEST(RunInputParseTest, ParsesTheGuessRestartAndCheckpointKeys) {
    // The run-restart schema: guess.type = "restart" names the
    // checkpoint file to seed from (guess.restart_path), and the [scf]
    // block can request the complementary last-iterate write
    // (checkpoint_file; checkpoint_converged widens it to the converged
    // exit too).
    const auto input = ParseRunInput(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.0, 0.0, 0.74],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[scf]
checkpoint_file = "partial.h5"
checkpoint_converged = true

[guess]
type = "restart"
restart_path = "partial.h5"
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_EQ(input->guess, GuessKind::kRestart);
    EXPECT_EQ(input->guessRestartPath, "partial.h5");
    EXPECT_EQ(input->scf.checkpointFile, "partial.h5");
    EXPECT_TRUE(input->scf.checkpointConverged);
}

TEST(RunInputParseTest, RestartTypeRequiresARestartPath) {
    const auto input = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
        "[guess]\ntype = \"restart\"\n");
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(input.error().message.find("guess.restart_path"), std::string::npos);
}

TEST(RunInputParseTest, NonRestartGuessToleratesAnInertRestartPath) {
    // The path is inert on the other guess types (the driver never reads
    // it): io stores what the file says, so a stray key is neither
    // a schema error nor a silent restart.
    const auto input = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
        "[guess]\ntype = \"gwh\"\nrestart_path = \"stray.h5\"\n");
    ASSERT_TRUE(input.has_value());
    EXPECT_EQ(input->guess, GuessKind::kGwh);
    EXPECT_EQ(input->guessRestartPath, "stray.h5");
}

TEST(RunInputParseTest, AppliesDefaultsWhenBlocksAreOmitted) {
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2]]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kTight"
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_EQ(input->molecule.charge, 0);
    EXPECT_EQ(input->molecule.multiplicity, 1);
    EXPECT_EQ(input->method.method, MethodType::kUhf);
    EXPECT_EQ(input->method.accuracy, qcx::integrals::AccuracyPreset::kTight);
    EXPECT_EQ(input->scf.maxIterations, 100);
    EXPECT_TRUE(input->scf.useDiis);
    EXPECT_TRUE(input->scf.traceFile.empty());
    // The restart keys default to the zero-cost path: no file to
    // write, no converged-exit write, no restart seed.
    EXPECT_TRUE(input->scf.checkpointFile.empty());
    EXPECT_FALSE(input->scf.checkpointConverged);
    EXPECT_TRUE(input->guessRestartPath.empty());
    EXPECT_EQ(input->resources.memoryCapGiB, 16.0);
    EXPECT_EQ(input->resources.threadCap, 0);
    EXPECT_EQ(input->guess, GuessKind::kCore);
    // The opt-in analyses all default off (the density_at_nuclei and
    // qtaim keys included).
    EXPECT_FALSE(input->properties.qtaim);
}

TEST(RunInputParseTest, ParsesTheKohnShamMethodWords) {
    // The schema accepts what its vocabulary documents: both Kohn-Sham
    // spellings reach the enum here. Whether a run path exists behind them is
    // the driver's up-front refusal, which is a different question and a
    // different layer - the parser must not be the thing that decides it.
    const auto rks = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
)");
    ASSERT_TRUE(rks.has_value());
    EXPECT_EQ(rks->method.method, MethodType::kRks);

    const auto uks = ParseRunInput(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2075]]
[basis]
orbital = "sto-3g"
[method]
type = "uks"
accuracy = "kNormal"
)");
    ASSERT_TRUE(uks.has_value());
    EXPECT_EQ(uks->method.method, MethodType::kUks);

    // A word outside the vocabulary still names the whole set, so the
    // rejection tells a user what the schema does accept.
    const auto typo = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rkss"
accuracy = "kNormal"
)");
    ASSERT_FALSE(typo.has_value());
    EXPECT_EQ(typo.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(typo.error().message.find("(rhf | uhf | rks | uks)"), std::string::npos);
}

TEST(RunInputParseTest, ScfToleranceDefaultsApplyWhenTheBlockOmitsTheKeys) {
    // A [scf] block that omits the tolerance keys falls
    // back to the struct defaults (1e-8 / 1e-6), never the earlier
    // 1e-10 pair - the parse fallback mirrors the struct defaults.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.7]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kNormal"

[scf]
use_diis = false
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_EQ(input->scf.energyTolerance, 1e-8);
    EXPECT_EQ(input->scf.densityTolerance, 1e-6);
    EXPECT_FALSE(input->scf.useDiis);
}

TEST(RunInputParseTest, AbsentFockBuilderMeansAutoSelection) {
    // The auto-selection contract: an absent
    // [method].fock_builder leaves the member empty - "auto" is the absence
    // of an explicit word, never a builder of its own. The driver resolves
    // the effective builder through the heuristic (selection_resolution).
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(input.has_value());
    EXPECT_FALSE(input->method.builder.has_value());
}

TEST(RunInputParseTest, RejectsAnEmptyFockBuilder) {
    // Type-strict when present: an empty string is not a builder. Omitting
    // the key means auto; spelling it empty is a malformed request, never
    // a silent "direct".
    const auto input = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"\"\naccuracy = \"kNormal\"\n");
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(input.error().message.find("method.fock_builder"), std::string::npos);
}

TEST(RunInputParseTest, LeanBuilderWordSetsTheLeanDirectFlag) {
    // The explicit fock_builder = "lean" is the direct family's
    // within-family member, so it never becomes a BuilderKind: the word is
    // intercepted at the parse site into leanDirect and the builder slot
    // stays ABSENT (no family word was given - the kGpuSplit no-word
    // precedent). An absent key leaves both empty/false, and the family
    // word "direct" still lands in the slot with the flag clear.
    const auto lean = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "lean"
accuracy = "kNormal"
)");
    ASSERT_TRUE(lean.has_value()) << lean.error().message;
    EXPECT_FALSE(lean->method.builder.has_value());
    EXPECT_TRUE(lean->method.leanDirect);

    const auto absent = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_FALSE(absent->method.builder.has_value());
    EXPECT_FALSE(absent->method.leanDirect);

    const auto direct = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    ASSERT_TRUE(direct->method.builder.has_value());
    EXPECT_EQ(*direct->method.builder, BuilderKind::kDirect);
    EXPECT_FALSE(direct->method.leanDirect);
}

TEST(RunInputParseTest, InMemoryBuilderWordIsTheInMemoryTierAlias) {
    // The builder vocabulary's TWO AXES: the
    // FAMILY axis (where the integrals come from - the BuilderKind
    // enumerators) and the TIER axis (where the working set lives - lean,
    // in_memory, blocked, disk). `direct` names the direct FAMILY, which runs
    // its in-memory tier; `in_memory` names that tier. One request, two
    // spellings: the two words must land on ONE state - the same builder slot
    // value and the same clear leanDirect flag - not on two states that happen
    // to agree today. `lean`, the same axis's other direct-family word, keeps
    // its own meaning (no slot word, the flag set).
    const auto parseWithWord = [](const std::string& word) {
        const std::string toml =
            "[molecule]\n"
            "atoms = [[\"H\", 0.0, 0.0, 0.0], [\"H\", 0.74084809526419992, 0.0, 0.0]]\n"
            "[basis]\n"
            "orbital = \"sto-3g\"\n"
            "[method]\n"
            "type = \"rhf\"\n"
            "fock_builder = \"" +
            word + "\"\naccuracy = \"kNormal\"\n";
        return ParseRunInput(toml);
    };

    const auto inMemory = parseWithWord("in_memory");
    ASSERT_TRUE(inMemory.has_value()) << inMemory.error().message;
    ASSERT_TRUE(inMemory->method.builder.has_value());
    EXPECT_EQ(*inMemory->method.builder, BuilderKind::kDirect);
    EXPECT_FALSE(inMemory->method.leanDirect);

    const auto direct = parseWithWord("direct");
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    ASSERT_TRUE(direct->method.builder.has_value());
    EXPECT_EQ(*direct->method.builder, BuilderKind::kDirect);
    EXPECT_FALSE(direct->method.leanDirect);
    EXPECT_EQ(inMemory->method.builder, direct->method.builder);
    EXPECT_EQ(inMemory->method.leanDirect, direct->method.leanDirect);

    const auto lean = parseWithWord("lean");
    ASSERT_TRUE(lean.has_value()) << lean.error().message;
    EXPECT_FALSE(lean->method.builder.has_value());
    EXPECT_TRUE(lean->method.leanDirect);
}

TEST(RunInputParseTest, UnknownFockBuilderRefusesWithTheAcceptedListThatNamesBothTierWords) {
    // An unknown word is refused by name with the accepted list, and the list
    // names the TIER words beside the family words: an author who typed
    // "inmemory" is reaching for the tier word, and a message that omitted it
    // would send them looking among the family words instead.
    const auto input = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"inmemory\"\naccuracy = \"kNormal\"\n");
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(input.error().message.find("unknown method.fock_builder \"inmemory\""),
              std::string::npos);
    EXPECT_NE(input.error().message.find("in_memory"), std::string::npos);
    EXPECT_NE(input.error().message.find("lean"), std::string::npos);
}

TEST(RunInputParseTest, EnforceCertifiedBoundIsOffUnlessAskedFor) {
    // The budget enforcement switch (schema 16): absent = off, present
    // must be a TOML boolean (a string is a type error, not a truthy
    // coercion), and the flag is independent of the builder word - the
    // enforcement is a policy the run asks for, never a property of a
    // builder spelling.
    const auto absent = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_FALSE(absent->method.enforceCertifiedBound);

    const auto enabled = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
enforce_certified_bound = true
)");
    ASSERT_TRUE(enabled.has_value()) << enabled.error().message;
    EXPECT_TRUE(enabled->method.enforceCertifiedBound);

    const auto wrongType = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
enforce_certified_bound = "true"
)");
    EXPECT_FALSE(wrongType.has_value());
}

TEST(RunInputParseTest, ParsesTheForceCertifiedLaneKey) {
    // The certified fp32 lane's force-on request (schema 21): optional and
    // type-strict, absent = false. The io layer stores what the file says -
    // no route judgement lives here, and the key is INDEPENDENT of the
    // enforcement switch above: one
    // asks for the lane, the other for the budget comparison over it.
    const auto absent = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_FALSE(absent->method.forceCertifiedLane);
    EXPECT_FALSE(absent->method.enforceCertifiedBound);

    const auto enabled = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
force_certified_lane = true
)");
    ASSERT_TRUE(enabled.has_value()) << enabled.error().message;
    EXPECT_TRUE(enabled->method.forceCertifiedLane);
    // The two keys are separate switches on the same lane, and asking for one
    // must never set the other: the enforcement moves the lane's quartets to
    // fp64, which a user who only asked for the lane did not request.
    EXPECT_FALSE(enabled->method.enforceCertifiedBound);

    const auto wrongType = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74084809526419992, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
force_certified_lane = "true"
)");
    EXPECT_FALSE(wrongType.has_value());
}

TEST(RunInputParseTest, ParsesThePropertiesBlockKeys) {
    // The opt-in properties analyses all default off; the bool keys parse
    // type-strictly and land on their fields (the density_at_nuclei
    // key rides the same path).
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[properties]
density_at_nuclei = true
qtaim = true
xc_gradient = true
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_TRUE(input->properties.densityAtNuclei);
    EXPECT_TRUE(input->properties.qtaim);
    EXPECT_TRUE(input->properties.xcGradient);
    EXPECT_FALSE(input->properties.hirshfeld);
    EXPECT_FALSE(input->properties.voronoi);
    EXPECT_FALSE(input->properties.eddb);
    EXPECT_FALSE(input->properties.fukui);
    EXPECT_FALSE(input->properties.nalewajski);
}

TEST(RunInputParseTest, ParsesTheResourcesBlockKeys) {
    // The [resources] block: both keys parse type-strictly and
    // land on their fields.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[resources]
memory_cap_gib = 4.0
thread_cap = 2
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_EQ(input->resources.memoryCapGiB, 4.0);
    EXPECT_EQ(input->resources.threadCap, 2);
}

TEST(RunInputParseTest, ResourcesZeroValuesAreEscapeHatches) {
    // memory_cap_gib = 0 (no input cap) and thread_cap = 0 (all detected
    // hardware threads) are the documented escape hatches: explicit zeros
    // parse, they are not range errors.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[resources]
memory_cap_gib = 0.0
thread_cap = 0
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_EQ(input->resources.memoryCapGiB, 0.0);
    EXPECT_EQ(input->resources.threadCap, 0);
}

TEST(RunInputParseTest, ParsesTheSymmetryBlockKeys) {
    // The [symmetry] block: full_group parses type-strictly; false
    // restores the pre-labeling behavior bit-identically (the output loses
    // the symmetry block, nothing else changes).
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[symmetry]
full_group = false
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_FALSE(input->symmetry.fullGroup);
}

TEST(RunInputParseTest, ParsesTheMemoryInstrumentBlockKeys) {
    // The [memory_instrument] block: all three keys parse
    // type-strictly and land on their fields.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[memory_instrument]
enabled = true
snapshot_interval_ms = 100
trace_file = "C:/runs/h2.attribution.csv"
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_TRUE(input->memoryInstrument.enabled);
    EXPECT_EQ(input->memoryInstrument.snapshotIntervalMs, 100);
    EXPECT_EQ(input->memoryInstrument.traceFile, "C:/runs/h2.attribution.csv");
}

TEST(RunInputParseTest, MemoryInstrumentBlockAbsentKeepsTheZeroCostPath) {
    // Absent block = the documented defaults: disabled, the 250 ms cadence,
    // no trace file - the zero-cost path the driver checks before it starts
    // any watchdog.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_FALSE(input->memoryInstrument.enabled);
    EXPECT_EQ(input->memoryInstrument.snapshotIntervalMs, 250);
    EXPECT_TRUE(input->memoryInstrument.traceFile.empty());
}

TEST(RunInputParseTest, SymmetryBlockAbsentDefaultsToTrue) {
    // The labeling stage is a pure a-posteriori classification and runs by
    // default: an absent block leaves fullGroup true.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_TRUE(input->symmetry.fullGroup);
}

TEST(RunInputParseTest, RejectsEmptyEspScheme) {
    // Coverage: an
    // empty properties.esp string passed the type-strict ReadString and
    // was silently treated as unset - indistinguishable from the omitted
    // key in the JSON output. It is a typo for the enum, not a request to
    // skip the block, so it is rejected up front.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[properties]
esp = ""
)");
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(input.error().message.find("must not be empty"), std::string::npos);
}

TEST(RunInputParseTest, ParsesTheMoldenExportKey) {
    // The molden-export key: a non-empty string path lands on the member.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[properties]
molden = "C:/runs/h2.molden"
)");
    ASSERT_TRUE(input.has_value());
    EXPECT_EQ(input->properties.molden, "C:/runs/h2.molden");
}

TEST(RunInputParseTest, MoldenKeyAbsentOrEmptyLeavesTheMemberEmpty) {
    // Absent = off (the default); the path-key contract also maps an empty
    // string to unset - unlike properties.esp, where the empty value is a
    // rejected typo, a molden path cannot be mistyped into another valid
    // value, so empty means "not requested".
    const auto absent = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_TRUE(absent.has_value());
    EXPECT_EQ(absent->properties.molden, "");

    const auto empty = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[properties]
molden = ""
)");
    ASSERT_TRUE(empty.has_value());
    EXPECT_EQ(empty->properties.molden, "");
}

TEST(RunInputParseTest, RejectsNonStringMoldenValue) {
    // Type-strict ReadString: a non-string molden value is a typo, not a
    // silent default.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"

[properties]
molden = 42
)");
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(input.error().message.find("\"properties.molden\" must be a string"),
              std::string::npos);
}

TEST(RunInputParseTest, RejectsUnknownVocabularyWords) {
    const auto unknownMethod = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rohf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n");
    ASSERT_FALSE(unknownMethod.has_value());
    EXPECT_EQ(unknownMethod.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto unknownBuilder = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"nonsense\"\naccuracy = \"kNormal\"\n");
    ASSERT_FALSE(unknownBuilder.has_value());
    EXPECT_EQ(unknownBuilder.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto unknownAccuracy = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kFast\"\n");
    ASSERT_FALSE(unknownAccuracy.has_value());
    EXPECT_EQ(unknownAccuracy.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RunInputParseTest, RejectsMalformedStructures) {
    const auto noMolecule = ParseRunInput("[method]\ntype = \"rhf\"\n");
    ASSERT_FALSE(noMolecule.has_value());
    EXPECT_EQ(noMolecule.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto emptyAtoms = ParseRunInput(
        "[molecule]\natoms = []\n[basis]\norbital = \"sto-3g\"\n[method]\ntype = \"rhf\"\n"
        "fock_builder = \"direct\"\naccuracy = \"kNormal\"\n");
    ASSERT_FALSE(emptyAtoms.has_value());
    EXPECT_EQ(emptyAtoms.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto badAtomRow = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0]]\n[basis]\norbital = \"sto-3g\"\n[method]\n"
        "type = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n");
    ASSERT_FALSE(badAtomRow.has_value());
    EXPECT_EQ(badAtomRow.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto badToml = ParseRunInput("this is [not toml");
    ASSERT_FALSE(badToml.has_value());
    EXPECT_EQ(badToml.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto zeroTolerance = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
        "[scf]\nenergy_tolerance = 0.0\n");
    ASSERT_FALSE(zeroTolerance.has_value());
    EXPECT_EQ(zeroTolerance.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RunInputParseTest, RejectsTheRemainingMalformedInputs) {
    // The value-range and vocabulary edges the parser pins explicitly.
    struct Case {
        const char* text;
        const char* messageFragment;
    };

    const Case cases[] = {
        // multiplicity = 0 is below the schema minimum of 1.
        {"[molecule]\ncharge = 0\nmultiplicity = 0\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n"
         "[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "multiplicity"},
        // A zero iteration budget cannot converge; rejected like the zero
        // tolerance.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[scf]\nmax_iterations = 0\n",
         "scf.max_iterations"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[scf]\ndensity_tolerance = 0.0\n",
         "scf.max_iterations"},
        // The required tables and keys are missing.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[method]\ntype = \"rhf\"\n"
         "fock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "basis"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n", "method"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "basis.orbital"},
        // Unknown vocabulary words are rejected with the accepted list.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[guess]\ntype = \"nonsense\"\n",
         "guess.type"},
        // guess.type = "restart" without a path has nothing to seed from;
        // the checkpoint keys are type-strict like every other scalar.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[guess]\ntype = \"restart\"\n",
         "guess.restart_path"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[guess]\ntype = \"restart\"\nrestart_path = 42\n",
         "guess.restart_path"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[scf]\ncheckpoint_file = 42\n",
         "scf.checkpoint_file"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[scf]\ncheckpoint_converged = \"yes\"\n",
         "scf.checkpoint_converged"},
        // A malformed atom row: non-string symbol, non-numeric coordinate.
        {"[molecule]\natoms = [[42, 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "symbol"},
        {"[molecule]\natoms = [[\"H\", \"x\", 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "coordinates"},
        // An empty fragment group is not a valid partition member.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[properties]\nnocv_fragments = [[]]\n",
         "nocv_fragments"},
        // The [resources] ceilings are non-negative; a negative memory cap
        // would invert the job-object limit and a negative thread cap is
        // meaningless.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[resources]\nmemory_cap_gib = -1.0\n",
         "resources.memory_cap_gib"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[resources]\nthread_cap = -1\n",
         "resources.thread_cap"},
        // A zero or negative watchdog cadence is meaningless: the row
        // cadence floor is 1 ms (the interval only bounds the commit-
        // sampling staleness; the final row lands on close regardless).
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[memory_instrument]\nenabled = true\nsnapshot_interval_ms = 0\n",
         "memory_instrument.snapshot_interval_ms"},
    };

    for (const auto& test : cases)
    {
        const auto input = ParseRunInput(test.text);
        ASSERT_FALSE(input.has_value()) << "case: " << test.text;
        EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(input.error().message.find(test.messageFragment), std::string::npos)
            << "message: " << input.error().message;
    }
}

TEST(RunInputParseTest, ParsesTheQfmmSchemaKnobs) {
    // The four optional QFMM knobs land
    // in RunMethodInput; the engine-default spellings (-1, 0) parse like
    // any explicit value and resolve in the driver/engine, not here.
    const auto knobbed = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
        "theta = 0.45\nl_mult = 3\nmax_leaf_size = 16\n"
        "crossover_basis_function_count = 600\n");
    ASSERT_TRUE(knobbed.has_value()) << knobbed.error().message;
    ASSERT_TRUE(knobbed->method.theta.has_value());
    EXPECT_EQ(*knobbed->method.theta, 0.45);
    ASSERT_TRUE(knobbed->method.lMult.has_value());
    EXPECT_EQ(*knobbed->method.lMult, 3);
    ASSERT_TRUE(knobbed->method.maxLeafSize.has_value());
    EXPECT_EQ(*knobbed->method.maxLeafSize, 16u);
    ASSERT_TRUE(knobbed->method.crossoverBasisFunctionCount.has_value());
    EXPECT_EQ(*knobbed->method.crossoverBasisFunctionCount, 600);

    // The contract edges that stay valid: theta 0 (= the preset's theta,
    // the engine's == 0 branch), the -1 l_mult sentinel (the preset's
    // order), leaf size 1 and crossover 0 (>= 0).
    const auto edges = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kLoose\"\n"
        "theta = 0.0\nl_mult = -1\nmax_leaf_size = 1\n"
        "crossover_basis_function_count = 0\n");
    ASSERT_TRUE(edges.has_value()) << edges.error().message;
    ASSERT_TRUE(edges->method.theta.has_value());
    EXPECT_EQ(*edges->method.theta, 0.0);
    ASSERT_TRUE(edges->method.lMult.has_value());
    EXPECT_EQ(*edges->method.lMult, -1);
    ASSERT_TRUE(edges->method.maxLeafSize.has_value());
    EXPECT_EQ(*edges->method.maxLeafSize, 1u);
    ASSERT_TRUE(edges->method.crossoverBasisFunctionCount.has_value());
    EXPECT_EQ(*edges->method.crossoverBasisFunctionCount, 0);

    // Absent knobs stay unset (nullopt = the engine default at wiring) -
    // on a qfmm run and on any other run alike (the io layer stores what
    // the file says; the driver's qfmm path is the only consumer).
    const auto plain = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n");
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_FALSE(plain->method.theta.has_value());
    EXPECT_FALSE(plain->method.lMult.has_value());
    EXPECT_FALSE(plain->method.maxLeafSize.has_value());
    EXPECT_FALSE(plain->method.crossoverBasisFunctionCount.has_value());
}

TEST(RunInputParseTest, RejectsOutOfContractQfmmKnobValues) {
    // The strict side of the QFMM knob contract: a knob outside the engine's
    // Create() range is kInvalidArgument naming the schema key - never a
    // silent park of the value or a deferral to a builder error that names
    // no TOML key.
    struct Case {
        const char* text;
        const char* messageFragment;
    };

    const Case cases[] = {
        // theta: non-finite values would silently bias the
        // well-separatedness comparisons (nan/-inf: nothing well separated;
        // +inf: everything) - the finiteness check is explicit (the
        // resources.memory_cap_gib precedent).
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "theta = nan\n",
         "method.theta"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "theta = inf\n",
         "method.theta"},
        // l_mult: one past the kQfmmMaxLMult = 8 cap, and below the -1
        // sentinel, both reject by name.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "l_mult = 9\n",
         "method.l_mult"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "l_mult = -2\n",
         "method.l_mult"},
        // max_leaf_size: the Create() contract is >= 1.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "max_leaf_size = 0\n",
         "method.max_leaf_size"},
        // crossover: the override vocabulary is >= 0 (absent = the engine
        // default; the -1 sentinel is internal to QfmmOptions).
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "crossover_basis_function_count = -1\n",
         "method.crossover_basis_function_count"},
        // The knob types are strict like every schema scalar.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "theta = \"loose\"\n",
         "method.theta"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"qfmm\"\naccuracy = \"kNormal\"\n"
         "l_mult = 2.5\n",
         "method.l_mult"},
    };

    for (const auto& test : cases)
    {
        const auto input = ParseRunInput(test.text);
        ASSERT_FALSE(input.has_value()) << "case: " << test.text;
        EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(input.error().message.find(test.messageFragment), std::string::npos)
            << "message: " << input.error().message;
    }
}

TEST(RunInputParseTest, ParsesTheRiOrbitExpansionOptIn) {
    // The orbit-expansion key (schema 26): optional and type-strict, and
    // the only driver-side switch for the increment. Absent is the plain walk,
    // which is what every run did before the key existed - so the parser must
    // leave the optional EMPTY rather than default it, because the record's
    // `requested` member reads exactly that difference: a false request and an
    // omitted key are different facts about the input.
    const auto optedIn = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_orbit_expansion = true
)");
    ASSERT_TRUE(optedIn.has_value()) << optedIn.error().message;
    ASSERT_TRUE(optedIn->method.riOrbitExpansion.has_value());
    EXPECT_TRUE(*optedIn->method.riOrbitExpansion);

    const auto optedOut = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_orbit_expansion = false
)");
    ASSERT_TRUE(optedOut.has_value()) << optedOut.error().message;
    ASSERT_TRUE(optedOut->method.riOrbitExpansion.has_value());
    EXPECT_FALSE(*optedOut->method.riOrbitExpansion);

    const auto omitted = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    ASSERT_TRUE(omitted.has_value()) << omitted.error().message;
    EXPECT_FALSE(omitted->method.riOrbitExpansion.has_value());

    // Type-strict, refused by its schema name rather than deferred to a builder
    // error that names no TOML key.
    const auto wrongType = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
ri_orbit_expansion = "yes"
)");
    ASSERT_FALSE(wrongType.has_value());
    EXPECT_NE(wrongType.error().message.find("ri_orbit_expansion"), std::string::npos);
}

TEST(RunInputParseTest, ParsesTheRiTensorSchemaKnobs) {
    // The two optional RI-J knobs (the disk rung) land in RunMethodInput:
    // ri_tensor_mode is the io enum's own
    // vocabulary ("auto" | "disk" - a rung selector selects a rung), and
    // ri_chunk_bytes the DiskRiFockOptions passthrough.
    const auto knobbed = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "ri_tensor_mode = \"disk\"\nri_chunk_bytes = 134217728\n");
    ASSERT_TRUE(knobbed.has_value()) << knobbed.error().message;
    ASSERT_TRUE(knobbed->method.riTensorMode.has_value());
    EXPECT_EQ(*knobbed->method.riTensorMode, RiTensorMode::kDisk);
    ASSERT_TRUE(knobbed->method.riChunkBytes.has_value());
    EXPECT_EQ(*knobbed->method.riChunkBytes, 134217728u);
    EXPECT_EQ(qcx::io::ToString(RiTensorMode::kAuto), std::string_view{"auto"});
    EXPECT_EQ(qcx::io::ToString(RiTensorMode::kDisk), std::string_view{"disk"});

    // The disk-tier ERI store request (the engine-decorator seam's request
    // surface, schema 33): a PATH is the whole request, the
    // scf.checkpoint_file / properties.molden spelling - there is deliberately
    // no separate boolean, because a second key could disagree with the first.
    const auto stored = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "eri_cache_store = \"C:/runs/h2-eri.h5\"\n");
    ASSERT_TRUE(stored.has_value()) << stored.error().message;
    EXPECT_EQ(stored->method.eriCacheStore, "C:/runs/h2-eri.h5");

    // Absent and empty are one not-requested state: `eri_store` is the same
    // string either way, so no consumer can read "asked for nothing" off the
    // difference between them.
    const auto absent = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n");
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_EQ(absent->method.eriCacheStore, "");

    const auto empty = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "eri_cache_store = \"\"\n");
    ASSERT_TRUE(empty.has_value()) << empty.error().message;
    EXPECT_EQ(empty->method.eriCacheStore, "");

    // Type-strict when present: a non-string store is a typo, not a path.
    const auto mistyped = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "eri_cache_store = 42\n");
    ASSERT_FALSE(mistyped.has_value());
    EXPECT_NE(mistyped.error().message.find("\"method.eri_cache_store\" must be a string"),
              std::string::npos);

    // The FORCE is a separate key in its own table, not a third word on the
    // rung selector: the selector names a rung, the force names a mechanism,
    // and one key carrying both is what the key split separates.
    const auto forced = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "[diagnostics]\nforce_disk_ri = true\n");
    ASSERT_TRUE(forced.has_value()) << forced.error().message;
    EXPECT_TRUE(forced->diagnostics.forceDiskRi);
    EXPECT_FALSE(forced->method.riTensorMode.has_value());

    // The force words are type-strict and default OFF: an explicit false,
    // and an absent block, both leave every run's rung decision where it
    // was - `false` is not a force-off.
    const auto declined = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "[diagnostics]\nforce_disk_ri = false\n");
    ASSERT_TRUE(declined.has_value()) << declined.error().message;
    EXPECT_FALSE(declined->diagnostics.forceDiskRi);

    // The auto word (the default when the key is absent) and the chunk-size
    // floor edge 1 both parse like any explicit value; the resolution to a
    // rung is the driver's, not the parser's.
    const auto autoMode = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kTight\"\n"
        "ri_tensor_mode = \"auto\"\nri_chunk_bytes = 1\n");
    ASSERT_TRUE(autoMode.has_value()) << autoMode.error().message;
    ASSERT_TRUE(autoMode->method.riTensorMode.has_value());
    EXPECT_EQ(*autoMode->method.riTensorMode, RiTensorMode::kAuto);
    ASSERT_TRUE(autoMode->method.riChunkBytes.has_value());
    EXPECT_EQ(*autoMode->method.riChunkBytes, 1u);

    // Absent knobs stay unset (nullopt = the engine default at wiring) -
    // on any run alike (the io layer stores what the file says; the
    // driver's ri_j_link disk route is the only consumer).
    const auto plain = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n");
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_FALSE(plain->method.riTensorMode.has_value());
    EXPECT_FALSE(plain->method.riChunkBytes.has_value());
    EXPECT_FALSE(plain->diagnostics.forceDiskRi);
}

TEST(RunInputParseTest, RetiredForcedDiskWordNamesTheKeyThatReplacedIt) {
    // The word that moved (schema 22). A
    // reader who wrote the old spelling is not making a typo: the force
    // exists and it is spelled elsewhere, so the refusal must name the key
    // that carries it. An error listing only the two surviving rung words
    // would read as "this capability is gone", which is the opposite of
    // true - and it would send the reader looking for a feature that only
    // moved.
    const auto retired = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "ri_tensor_mode = \"forced_disk\"\n");
    ASSERT_FALSE(retired.has_value());
    EXPECT_EQ(retired.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(retired.error().message.find("method.ri_tensor_mode does not take \"forced_disk\""),
              std::string::npos)
        << "message: " << retired.error().message;
    EXPECT_NE(retired.error().message.find("[diagnostics] force_disk_ri = true"), std::string::npos)
        << "message: " << retired.error().message;
    EXPECT_NE(retired.error().message.find("auto | disk"), std::string::npos)
        << "message: " << retired.error().message;
}

TEST(RunInputParseTest, ForceDiskKeyIsRefusedByNameInEveryOtherTable) {
    // The force key is read from [diagnostics] and from nowhere else. Written
    // in another table it is not one of the keys the schema does not name (a
    // file may carry those, and they are ignored): the name is this schema's
    // own and it names a mechanism, so a run that read past it would compute
    // unforced while the file's author believed the route was forced. Each
    // table therefore gets the same refusal, naming the path the key was
    // written at and the one that is read - the retired rung word's
    // treatment, in the same voice.
    struct Case {
        const char* text;
        const char* writtenAt;
    };

    const Case cases[] = {
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "force_disk_ri = true\n",
         "method.force_disk_ri"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "[scf]\nforce_disk_ri = true\n",
         "scf.force_disk_ri"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"pbe\"\naccuracy = \"kNormal\"\n"
         "[grid]\nforce_disk_ri = true\n",
         "grid.force_disk_ri"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "[builder]\nforce_disk_ri = true\n",
         "builder.force_disk_ri"},
        // Dotted and inline spellings are one table in the parsed document, so
        // they are refused with the path that table really has. At the
        // document's own level that is the plain path; written under a table
        // header, TOML makes the dotted key relative to that table, and the
        // refusal names the qualified path rather than the one it looked like.
        {"method.type = \"rhf\"\nmethod.accuracy = \"kNormal\"\nmethod.force_disk_ri = true\n"
         "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n",
         "method.force_disk_ri"},
        {"method = { type = \"rhf\", accuracy = \"kNormal\", force_disk_ri = true }\n"
         "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n",
         "method.force_disk_ri"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "method.force_disk_ri = true\n",
         "method.method.force_disk_ri"},
        // A key beside no table at all carries the same claim as the one under
        // [method], and is refused with the path it was written at.
        {"force_disk_ri = true\n"
         "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n",
         "force_disk_ri"},
        // Nested under a table of its own making: still not the home.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "[method.notes]\nforce_disk_ri = true\n",
         "method.notes.force_disk_ri"},
    };

    const std::string tail =
        " is not read: the RI disk force is spelled in the [diagnostics] table alone, and a copy "
        "of the key elsewhere forces nothing. Set [diagnostics] force_disk_ri = true instead";

    for (const Case& testCase : cases)
    {
        const auto parsed = ParseRunInput(testCase.text);
        const std::string writtenAt(testCase.writtenAt);
        ASSERT_FALSE(parsed.has_value()) << "written at: " << writtenAt;
        EXPECT_EQ(parsed.error().code, qcx::ErrorCode::kInvalidArgument);

        // The whole message, path included: the path is the message's head, so
        // a refusal naming a longer path than the file really has - the
        // `basis.`-qualified one a reader's guess is not - fails here, where a
        // substring test on the path alone would pass it.
        EXPECT_EQ(parsed.error().message, writtenAt + tail) << parsed.error().message;
    }

    // The key's own table is unaffected, and so is the tolerance for the keys
    // the schema does not name: only the exact name is refused, in or out of
    // place. The home itself is read whichever way it is spelled.
    const auto home = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "[diagnostics]\nforce_disk_ri = true\n");
    ASSERT_TRUE(home.has_value()) << home.error().message;
    EXPECT_TRUE(home->diagnostics.forceDiskRi);

    const auto homeInline = ParseRunInput(
        "diagnostics = { force_disk_ri = true }\n"
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n");
    ASSERT_TRUE(homeInline.has_value()) << homeInline.error().message;
    EXPECT_TRUE(homeInline->diagnostics.forceDiskRi);

    const auto otherKey = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
        "force_disk_ri_notes = \"kept for another tool\"\n");
    ASSERT_TRUE(otherKey.has_value()) << otherKey.error().message;
    EXPECT_FALSE(otherKey->diagnostics.forceDiskRi);
}

TEST(RunInputParseTest, RejectsOutOfContractRiTensorKnobValues) {
    // The strict side of the knob-vocabulary contract: a knob outside its
    // vocabulary is kInvalidArgument naming the schema key with the
    // accepted words - never a silent park of the value or a deferral to a
    // builder error that names no TOML key.
    struct Case {
        const char* text;
        const char* messageFragment;
    };

    const Case cases[] = {
        // ri_tensor_mode: an unknown word names the key and lists the
        // accepted vocabulary (the RETIRED force word has its own case
        // above: it names the key that replaced it).
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "ri_tensor_mode = \"fast\"\n",
         "unknown method.ri_tensor_mode \"fast\" (auto | disk)"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "ri_tensor_mode = \"Disk\"\n",
         "method.ri_tensor_mode"},
        // ri_chunk_bytes: the Create() contract is >= 1.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "ri_chunk_bytes = 0\n",
         "method.ri_chunk_bytes must be >= 1"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "ri_chunk_bytes = -1\n",
         "method.ri_chunk_bytes"},
        // The knob types are strict like every schema scalar.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "ri_tensor_mode = 2\n",
         "method.ri_tensor_mode"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"ri_j_link\"\naccuracy = \"kNormal\"\n"
         "ri_chunk_bytes = \"disk\"\n",
         "method.ri_chunk_bytes"},
    };

    for (const auto& test : cases)
    {
        const auto input = ParseRunInput(test.text);
        ASSERT_FALSE(input.has_value()) << "case: " << test.text;
        EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(input.error().message.find(test.messageFragment), std::string::npos)
            << "message: " << input.error().message;
    }
}

TEST(RunInputParseTest, EveryBuilderKindStatesWhetherItConsumesAnAux) {
    // The aux answer is a compatibility surface, not a preference: the aux name
    // enters ComputeFingerprint (storage/fingerprint.hpp) through the driver's
    // checkpoint binding, so a value that moved would make a checkpoint written
    // before the change fail to load. The comparison below is against the
    // PRE-FIX derivation verbatim - `kind == BuilderKind::kRiJLink` - value by
    // value over the enum, which is why this is a loop and not an argument.
    //
    // What this proves: every enumerator EXCEPT kRiJk derives the value it
    // derived before, and the row is mutation-sensitive (flipping one arm of
    // BuilderConsumesAux fails it, measured). What it cannot prove: that a
    // SEVENTH enumerator cannot pass silently - no runtime row can observe an
    // enumerator that does not exist yet. That half is the guarded switch's
    // build failure, which this row can only keep honest by listing the enum
    // it knows.
    //
    // THE ONE EXCEPTION, and why it is stated rather than looped over: kRiJk
    // moved to true when the composed full-RI builder was wired and the kind
    // became an aux consumer in fact. The compatibility argument above does
    // not reach it - the driver refused every ri_jk request by name until that
    // builder existed, so no run could write a checkpoint carrying this kind
    // and there is no pre-change value for a load to disagree with. Excluding
    // it from the loop is therefore the honest statement, not a relaxation:
    // leaving it in would assert a compatibility that cannot exist, and
    // dropping the value entirely would stop pinning it (the row below still
    // does).
    const BuilderKind kinds[] = {BuilderKind::kDirect,
                                 BuilderKind::kRiJLink,
                                 BuilderKind::kQfmm,
                                 BuilderKind::kGpu,
                                 BuilderKind::kGpuSplit};

    for (const auto kind : kinds)
    {
        const bool before = (kind == BuilderKind::kRiJLink);
        EXPECT_EQ(qcx::io::BuilderConsumesAux(kind), before) << "kind: " << qcx::io::ToString(kind);
    }

    EXPECT_TRUE(qcx::io::BuilderConsumesAux(BuilderKind::kRiJk));
}

TEST(RunInputParseTest, MissingMethodTypeHintNamesTheWholeAcceptedVocabulary) {
    // The key-presence hint must not under-report the vocabulary: ParseMethod
    // accepts four words and its own unknown-name error names all four, so a
    // missing key has to name all four too. This said "rhf | uhf" until
    // 2026-09-12, which means a reader who omitted the key while wanting rks
    // was told by the parser's own hint that the word does not exist - while
    // the parser accepted it. The expectation is verbatim on purpose: a future
    // narrowing of the hint (or a widening of ParseMethod) fails here rather
    // than reaching a user as a contradiction. The other seven hint lists in
    // this file were checked against their Parse* functions in the same pass
    // (fock_builder with its intercepted lean word, ri_tensor_mode, accuracy,
    // its presence hint included, properties.esp, guess.type, and
    // basis.orbital, which makes no claim) and were already complete.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(input.error().message, "\"method.type\" is required (rhf | uhf | rks | uks)");
}

TEST(RunInputParseTest, RejectsWrongTypedScalarsInsteadOfSilentDefaults) {
    // A present-but-wrong-typed key is an error (kInvalidArgument naming
    // the schema key), never a silent fallback to the schema default.
    struct Case {
        const char* text;
        const char* messageFragment;
    };

    const Case cases[] = {
        // molecule.charge and multiplicity are integers only.
        {"[molecule]\ncharge = \"x\"\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n"
         "[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "molecule.charge"},
        {"[molecule]\nmultiplicity = 1.5\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n"
         "[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "molecule.multiplicity"},
        // The method vocabulary words are strings.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = 42\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n",
         "method.type"},
        // The SCF tolerances are numbers, use_diis is a bool.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[scf]\nenergy_tolerance = \"1e-10\"\n",
         "scf.energy_tolerance"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[scf]\nuse_diis = 1\n",
         "scf.use_diis"},
        // trace_file is a path string.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[scf]\ntrace_file = 1\n",
         "scf.trace_file"},
        // The properties booleans are bools.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[properties]\nhirshfeld = 1\n",
         "properties.hirshfeld"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[properties]\ndensity_at_nuclei = \"yes\"\n",
         "properties.density_at_nuclei"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[properties]\nqtaim = \"yes\"\n",
         "properties.qtaim"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[properties]\nxc_gradient = \"yes\"\n",
         "properties.xc_gradient"},
        // guess.type is a string; the wrong type must not fall back to core.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[guess]\ntype = true\n",
         "guess.type"},
        // The [resources] scalars are a number and an integer; a wrong
        // type must not fall back to the 16 GiB / 0 defaults.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[resources]\nmemory_cap_gib = \"16\"\n",
         "resources.memory_cap_gib"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[resources]\nthread_cap = 2.5\n",
         "resources.thread_cap"},
        // A non-finite memory cap (TOML 1.0 nan) cannot bound a process;
        // the range check alone would admit it (NaN comparisons are
        // false), so the finiteness check must fire.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[resources]\nmemory_cap_gib = nan\n",
         "resources.memory_cap_gib"},
        // The [symmetry] switch is a bool; a wrong type must not fall back
        // to the run-labeling default.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
         "[symmetry]\nfull_group = \"no\"\n",
         "symmetry.full_group"},
    };

    for (const auto& test : cases)
    {
        const auto input = ParseRunInput(test.text);
        ASSERT_FALSE(input.has_value()) << "case: " << test.text;
        EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(input.error().message.find(test.messageFragment), std::string::npos)
            << "message: " << input.error().message;
    }

    // An integer where a double is expected is unambiguous and accepted.
    const auto intTolerance = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rhf\"\nfock_builder = \"direct\"\naccuracy = \"kNormal\"\n"
        "[scf]\nenergy_tolerance = 1\n");
    ASSERT_TRUE(intTolerance.has_value());
    EXPECT_EQ(intTolerance->scf.energyTolerance, 1.0);
}

TEST(RunInputParseTest, ParseRunInputFileReadsAndPropagatesErrors) {
    const auto path = std::filesystem::temp_directory_path() / "qcx_parse_input_test.toml";

    {
        std::ofstream stream(path);

        ASSERT_TRUE(stream.is_open());
        stream << kExampleH2Toml;
    }

    const auto input = ParseRunInputFile(path);
    ASSERT_TRUE(input.has_value());
    EXPECT_EQ(input->method.method, MethodType::kRhf);
    EXPECT_EQ(input->molecule.atoms.size(), 2u);

    const auto missing = ParseRunInputFile(path.parent_path() / "qcx_no_such_input.toml");
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RunResultJsonTest, UnsetPropertiesEmitNullSpinSquaredAndNullBlocks) {
    RunResult result;
    result.converged = true;
    result.iterations = 12;
    result.totalEnergyHartree = -1.1167143252;
    result.electronicEnergyHartree = -1.8394;
    result.timingsMs.totalMs = 143.2;
    result.timingsMs.scfLoopMs = 138.7;

    const auto json = SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"converged\": true"), std::string::npos);
    EXPECT_NE(json.find("\"iterations\": 12"), std::string::npos);
    EXPECT_NE(json.find("\"total_energy_hartree\": -1.1167143252"), std::string::npos);
    EXPECT_NE(json.find("\"spin_squared\": null"), std::string::npos);
    EXPECT_NE(json.find("\"populations\": null"), std::string::npos);
    EXPECT_NE(json.find("\"moments\": null"), std::string::npos);
    EXPECT_NE(json.find("\"scf_loop\": 138.7"), std::string::npos);
}

// A non-finite number must not silently serialize
// as null (null means "not computed" in this schema) - it gets an explicit
// string marker, at the top level and inside nested blocks alike, while
// finite neighbors stay numbers.
TEST(RunResultJsonTest, NonFiniteNumbersSerializeAsExplicitMarkers) {
    RunResult result;
    result.converged = true;
    result.totalEnergyHartree = std::numeric_limits<double>::quiet_NaN();
    result.electronicEnergyHartree = std::numeric_limits<double>::infinity();

    RunProperties properties;
    properties.populations.mullikenAlpha = {std::numeric_limits<double>::quiet_NaN(), 1.0};
    result.properties = properties;

    const auto json = SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"total_energy_hartree\": \"nan\""), std::string::npos);
    EXPECT_NE(json.find("\"electronic_energy_hartree\": \"inf\""), std::string::npos);
    EXPECT_NE(json.find("\"spin_squared\": null"), std::string::npos);
    EXPECT_NE(json.find("\"alpha\": [\n        \"nan\",\n        1.0\n      ]"), std::string::npos);
}

TEST(RunResultJsonTest, SchemaVersionAppearsInEveryDocument) {
    RunResult result;
    const auto json = SerializeRunResultJson(result);
    const std::string marker = "\"schema_version\": " + std::to_string(RunResult::kSchemaVersion);

    EXPECT_NE(json.find(marker), std::string::npos);
}

// resources_resolved is part of the schema - every
// document carries the block, defaults included, so consumers always find
// the caps a run was actually executed under.
TEST(RunResultJsonTest, ResourcesResolvedBlockAlwaysPresent) {
    RunResult result;
    result.converged = true;
    result.resourcesResolved.memoryCapGiB = 16.0;
    result.resourcesResolved.threadCap = 0;
    result.resourcesResolved.inProcessCapApplied = true;

    const auto json = SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"resources_resolved\""), std::string::npos);
    EXPECT_NE(json.find("\"memory_cap_gib\": 16.0"), std::string::npos);
    EXPECT_NE(json.find("\"thread_cap\": 0"), std::string::npos);
    EXPECT_NE(json.find("\"in_process_cap_applied\": true"), std::string::npos);
    EXPECT_NE(json.find("\"timings_ms\""), std::string::npos);
}

// The audit note (cap not applied: already in an external job, or a
// platform without job objects) is emitted only when present - a silent
// absence would look like "applied" to consumers of the block.
TEST(RunResultJsonTest, CapNoteEmittedWhenNotApplied) {
    RunResult result;
    result.converged = true;
    result.resourcesResolved.inProcessCapApplied = false;
    result.resourcesResolved.capNote = "the process is already in an external job";

    const auto json = SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"in_process_cap_applied\": false"), std::string::npos);
    EXPECT_NE(json.find("\"cap_note\": \"the process is already in an external job\""),
              std::string::npos);
}

TEST(RunResultJsonTest, UhfEmitsTheSpinSquaredValue) {
    RunResult result;
    result.converged = true;
    result.spinSquared = 2.0034108576810308;

    const auto json = SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"spin_squared\": 2.0034108576810308"), std::string::npos);
    EXPECT_NE(json.find("\"populations\": null"), std::string::npos);
}

TEST(RunResultJsonTest, PopulatedPropertiesSerializeTheStage71Block) {
    RunResult result;
    result.converged = true;
    qcx::io::RunProperties properties;
    properties.populations.mullikenTotal = {0.8168220177, 0.8168220177, 8.3663559645};
    properties.populations.mullikenSpin = {0.0, 0.0, 0.0};
    properties.populations.lowdinTotal = {0.8733083820, 0.8733083820, 8.2533832360};
    properties.populations.mayerBondOrders = {{0.0, 0.9539547924}, {0.9539547924, 0.0}};
    properties.populations.gopinathanJugBondOrders = {{0.3813337650, 0.4916422210},
                                                      {0.4916422210, 0.3813337650}};
    properties.moments.dipole = {0.0, 0.67898079210, 0.0};
    properties.moments.quadrupole[0] = {1.7609360829, 0.0, 0.0};
    properties.moments.quadrupole[1] = {0.0, 0.35190091054, 0.0};
    properties.moments.quadrupole[2] = {0.0, 0.0, -2.1128369935};
    result.properties = properties;

    const auto json = SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"populations\""), std::string::npos);
    EXPECT_NE(json.find("\"mulliken\""), std::string::npos);
    EXPECT_NE(json.find("\"lowdin\""), std::string::npos);
    EXPECT_NE(json.find("\"mayer\""), std::string::npos);
    EXPECT_NE(json.find("\"gopinathan_jug\""), std::string::npos);
    EXPECT_NE(json.find("\"dipole\""), std::string::npos);
    EXPECT_NE(json.find("\"quadrupole\""), std::string::npos);
    // The values survive the round trip (nlohmann's shortest round-trip
    // double serialization reproduces the literals).
    EXPECT_NE(json.find("0.6789807921"), std::string::npos);
    EXPECT_NE(json.find("8.3663559645"), std::string::npos);
    EXPECT_NE(json.find("2.1128369935"), std::string::npos);
}

TEST(RunInputParseTest, ParsesTheFunctionalAndScreeningKeys) {
    // The two method-scoped Kohn-Sham keys. io records what
    // the file says and resolves NOTHING: the functional is a string whose
    // vocabulary this layer does not know (the driver resolves it against the
    // registry), and the tolerance is a bare number. The presence assertions
    // are the point of the test: a `double` member with a default would pass
    // every value check below and still be unable to tell a written key from
    // an omitted one - which is exactly the distinction the rhf/uhf refusal
    // needs, so it is pinned here rather than assumed.
    const auto written = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
functional = "pbe"
screening_tolerance = 1e-9
)");
    ASSERT_TRUE(written.has_value()) << written.error().message;
    ASSERT_TRUE(written->method.functional.has_value());
    EXPECT_EQ(*written->method.functional, "pbe");
    ASSERT_TRUE(written->method.screeningTolerance.has_value());
    EXPECT_DOUBLE_EQ(*written->method.screeningTolerance, 1e-9);

    // An explicit 0.0 is the dense path and is still PRESENT - the value the
    // default would never produce, so presence and value are independent.
    const auto dense = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
screening_tolerance = 0.0
)");
    ASSERT_TRUE(dense.has_value()) << dense.error().message;
    ASSERT_TRUE(dense->method.screeningTolerance.has_value());
    EXPECT_DOUBLE_EQ(*dense->method.screeningTolerance, 0.0);
    EXPECT_FALSE(dense->method.functional.has_value());

    // Omitted means absent on a Hartree-Fock word too: the parser does not
    // refuse the keys here (that is the validator's rule, one layer up), and
    // it does not invent them either.
    const auto omitted = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(omitted.has_value()) << omitted.error().message;
    EXPECT_FALSE(omitted->method.functional.has_value());
    EXPECT_FALSE(omitted->method.screeningTolerance.has_value());
}

TEST(RunInputParseTest, RejectsAnUnusableScreeningTolerance) {
    // The tolerance is type-checked and range-checked the way the QFMM knobs
    // are. A nan would silently select a screening behaviour the file does not
    // appear to ask for (every comparison false, so nothing screens); +inf
    // would screen everything away; a negative value has no meaning at all,
    // since 0.0 is already the dense path and IS expressible.
    const auto nanTolerance = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
screening_tolerance = nan
)");
    EXPECT_FALSE(nanTolerance.has_value());
    EXPECT_EQ(nanTolerance.error().code, qcx::ErrorCode::kInvalidArgument);

    const auto negative = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
screening_tolerance = -1.0
)");
    EXPECT_FALSE(negative.has_value());
    EXPECT_EQ(negative.error().code, qcx::ErrorCode::kInvalidArgument);

    // A wrongly-typed key is a parse error rather than a silent coercion (the
    // key is type-strict when present, like every other [method] key).
    const auto wrongType = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
screening_tolerance = "tight"
)");
    EXPECT_FALSE(wrongType.has_value());
    EXPECT_EQ(wrongType.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The [molecule] units key (schema 28). Its acceptance criterion is EXACT, not
// approximate: the same physical molecule written in Angstrom and in Bohr must
// parse to the BIT-IDENTICAL internal geometry. That is the assertion which
// would have caught the 2026-09-15 hunt, where Bohr numbers were fed to the
// Angstrom-default key and a whole defect hunt chased a stretched molecule (and
// its mirror, 2026-08-26, pyscf's Angstrom default against qcx's Bohr
// convention).
//
// The Bohr file's literals are the exact shortest spellings of
// `x_angstrom * kAngstromToBohr` (1.430428808, 1.107157044), so a differing
// double here is a differing double in the parser, not a rounding artifact of
// the fixture. The comparison is on the BIT PATTERN: `==` would already be
// exact for these values, but a bit comparison is what "identical geometry"
// has to mean - a sign-of-zero or one-ulp difference is exactly the error class
// this test exists to refuse.
TEST(RunInputParseTest, BothUnitSpellingsGiveTheBitIdenticalGeometry) {
    const auto angstrom = ParseRunInput(R"(
[molecule]
units = "angstrom"
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
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(angstrom.has_value());

    const auto bohr = ParseRunInput(R"(
[molecule]
units = "bohr"
charge = 0
multiplicity = 1
atoms = [
    ["O", 0.0, 0.0, 0.0],
    ["H", 1.430428808, 1.107157044, 0.0],
    ["H", -1.430428808, 1.107157044, 0.0],
]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(bohr.has_value());

    // The two files really did differ in the key, so the identity below is the
    // conversion's work and not a fixture that never exercised it.
    EXPECT_EQ(angstrom->molecule.coordinateUnit, CoordinateUnit::kAngstrom);
    EXPECT_EQ(bohr->molecule.coordinateUnit, CoordinateUnit::kBohr);

    ASSERT_EQ(angstrom->molecule.atoms.size(), bohr->molecule.atoms.size());

    for (std::size_t i = 0; i < angstrom->molecule.atoms.size(); ++i)
    {
        EXPECT_EQ(angstrom->molecule.atoms[i].symbol, bohr->molecule.atoms[i].symbol);
        EXPECT_EQ(std::bit_cast<std::uint64_t>(angstrom->molecule.atoms[i].x),
                  std::bit_cast<std::uint64_t>(bohr->molecule.atoms[i].x));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(angstrom->molecule.atoms[i].y),
                  std::bit_cast<std::uint64_t>(bohr->molecule.atoms[i].y));
        EXPECT_EQ(std::bit_cast<std::uint64_t>(angstrom->molecule.atoms[i].z),
                  std::bit_cast<std::uint64_t>(bohr->molecule.atoms[i].z));
    }
}

// The absent key is NOT a third state: it means Angstrom, which is what every
// input file written before this key existed means. Pinned here so the default
// cannot drift into "unset" or into Bohr.
TEST(RunInputParseTest, AbsentUnitsKeyMeansAngstrom) {
    const auto input = ParseRunInput(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.0, 0.0, 0.74],
]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(input.has_value());

    EXPECT_EQ(input->molecule.coordinateUnit, CoordinateUnit::kAngstrom);
    // 0.74 A -> 1.3983973322230698 a0 (the literal, not the code's own
    // expression): the absent key converts, and it converts to the same double
    // an explicit `units = "angstrom"` file does (asserted bit-exactly against
    // the Bohr spelling in the sibling test above).
    EXPECT_EQ(std::bit_cast<std::uint64_t>(input->molecule.atoms[1].z),
              std::bit_cast<std::uint64_t>(1.3983973322230698));
}

// A word outside the vocabulary is refused BY NAME with the accepted list, in
// the same commit as the enum that adds it. The refusal is the point of the
// key: `units = "bohrs"` must not run as Angstrom (a freely-interpreted typo is
// the silent default this packet exists to remove), and neither must an empty
// string, which is a malformed request rather than an omission.
TEST(RunInputParseTest, UnknownMoleculeUnitsWordIsRefusedByName) {
    constexpr const char* kBody = R"(
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)";

    const auto typo = ParseRunInput(std::string(R"(
[molecule]
units = "bohrs"
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
)") + kBody);
    ASSERT_FALSE(typo.has_value());
    EXPECT_EQ(typo.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(typo.error().message.find("molecule.units"), std::string::npos);
    EXPECT_NE(typo.error().message.find("\"bohrs\""), std::string::npos);
    EXPECT_NE(typo.error().message.find("(angstrom | bohr)"), std::string::npos);

    const auto empty = ParseRunInput(std::string(R"(
[molecule]
units = ""
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
)") + kBody);
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(empty.error().message.find("molecule.units"), std::string::npos);

    // Type-strict when present, like every other key: a unit is a word, and a
    // number here is malformed rather than a silently-ignored extra.
    const auto wrongType = ParseRunInput(std::string(R"(
[molecule]
units = 1
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
)") + kBody);
    ASSERT_FALSE(wrongType.has_value());
    EXPECT_EQ(wrongType.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(wrongType.error().message.find("\"molecule.units\" must be a string"),
              std::string::npos);
}

// The record half of the same packet, end to end inside io: the unit the
// parser RESOLVED is the word the serializer publishes, so a record and the
// parse it came from cannot disagree. (The driver copies the resolved member
// straight into RunResult, which the end-to-end run exercises.)
TEST(RunInputParseTest, TheParsedUnitReachesTheRunRecord) {
    const auto input = ParseRunInput(R"(
[molecule]
units = "bohr"
charge = 0
multiplicity = 1
atoms = [["H", 0.0, 0.0, 0.0], ["H", 1.4, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(input.has_value());
    EXPECT_EQ(input->molecule.coordinateUnit, CoordinateUnit::kBohr);
    // A bohr file's 1.4 is 1.4 a0, untouched: the conversion is the
    // angstrom-only branch, so the value is the file's own double.
    EXPECT_EQ(std::bit_cast<std::uint64_t>(input->molecule.atoms[1].x),
              std::bit_cast<std::uint64_t>(1.4));

    RunResult result;
    result.moleculeUnits = input->molecule.coordinateUnit;
    EXPECT_NE(SerializeRunResultJson(result).find("\"molecule_units\": \"bohr\""),
              std::string::npos);
}

// The accuracy preset's word in BOTH directions, one vocabulary (schema 31):
// the parser accepts the three words and ToString prints the same three, so a
// record that names a preset - `exchange_error.bar_preset`, the approximated-
// exchange disclosure's tie between a bar and the preset it belongs to - names
// the word the input could have written. The ROUND TRIP is the point: a
// print-side word list would drift from the parser's silently, and the whole
// reason that key exists is that a record be checkable against what ran.
TEST(RunInputParseTest, AccuracyPresetWordRoundTripsThroughTheParser) {
    const auto roundTrips = [](const std::string& word) {
        const std::string text =
            "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
            "[method]\ntype = \"rhf\"\naccuracy = \"" +
            word + "\"\n";
        const auto input = ParseRunInput(text);
        ASSERT_TRUE(input.has_value()) << input.error().message;
        EXPECT_EQ(qcx::io::ToString(input->method.accuracy), word);
    };

    roundTrips("kLoose");
    roundTrips("kNormal");
    roundTrips("kTight");
}

TEST(RunInputParseTest, MethodTypeWordRoundTripsThroughTheParser) {
    // The vocabulary pin: the four [method] type words the parser accepts
    // are the four words ToString writes into the record's `method` key, so
    // the input side and the artifact side are one vocabulary - a consumer
    // that reads "rks" out of a record can write "rks" back into a file.
    const auto roundTrips = [](const std::string& word) {
        const std::string text =
            "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
            "[method]\ntype = \"" +
            word + "\"\naccuracy = \"kNormal\"\n";
        const auto input = ParseRunInput(text);
        ASSERT_TRUE(input.has_value()) << input.error().message;
        EXPECT_EQ(qcx::io::ToString(input->method.method), word);
    };

    roundTrips("rhf");
    roundTrips("uhf");
    roundTrips("rks");
    roundTrips("uks");
}

TEST(RunInputParseTest, ParsesTheGridBlockKeys) {
    // The [grid] block: all six keys are optional, type-strict, and land
    // on the resolved members the driver carries into XcGridSettings and the
    // record discloses as xc_grid. Values chosen so each one differs from the
    // engine default in the same direction a real override would.
    const auto input = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.74, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rks"
functional = "slater"
accuracy = "kNormal"

[grid]
radial_points = 40
angular_points = 50
alpha = 0.6
radial_exponent = 3
trim_weight = 1e-12
block_target = 64
)");
    ASSERT_TRUE(input.has_value()) << input.error().message;
    ASSERT_TRUE(input->grid.has_value());

    EXPECT_EQ(input->grid->radialPoints, 40U);
    EXPECT_EQ(input->grid->angularPoints, 50U);
    EXPECT_EQ(input->grid->alpha, 0.6);
    EXPECT_EQ(input->grid->radialExponent, 3U);
    EXPECT_EQ(input->grid->trimWeight, 1e-12);
    EXPECT_EQ(input->grid->blockTarget, 64U);
}

TEST(RunInputParseTest, GridBlockAbsentKeepsTheEngineDefaults) {
    // An absent block (and an absent key inside a present one) resolves to the
    // engine's own compile-time values, which is what makes the block additive:
    // a file written before it existed builds the same grid it always did. The
    // numbers are pinned here as literals because THIS is the surface a user
    // reads them from - a default that moved would be a silently different
    // quadrature for every run that never wrote a [grid] block at all.
    const auto absent = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kNormal"
)");
    ASSERT_TRUE(absent.has_value()) << absent.error().message;

    // No block: the member is ABSENT, and the defaults the driver resolves in
    // its place are this struct's own - pinned as literals, because they are
    // the quadrature every file that never wrote a [grid] block has always run.
    EXPECT_FALSE(absent->grid.has_value());

    const qcx::io::RunGridInput engineDefaults;

    EXPECT_EQ(engineDefaults.radialPoints, 75U);
    EXPECT_EQ(engineDefaults.angularPoints, 302U);
    EXPECT_EQ(engineDefaults.alpha, 0.5);
    EXPECT_EQ(engineDefaults.radialExponent, 2U);
    EXPECT_EQ(engineDefaults.trimWeight, 1e-15);
    EXPECT_EQ(engineDefaults.blockTarget, 1024U);

    // A present block omitting five of the six keys keeps the other five
    // defaults - the per-key rule, not a block-level all-or-nothing.
    const auto partial = ParseRunInput(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kNormal"

[grid]
radial_points = 40
)");
    ASSERT_TRUE(partial.has_value()) << partial.error().message;
    ASSERT_TRUE(partial->grid.has_value());

    EXPECT_EQ(partial->grid->radialPoints, 40U);
    EXPECT_EQ(partial->grid->angularPoints, 302U);
    EXPECT_EQ(partial->grid->alpha, 0.5);
    EXPECT_EQ(partial->grid->radialExponent, 2U);
    EXPECT_EQ(partial->grid->trimWeight, 1e-15);
    EXPECT_EQ(partial->grid->blockTarget, 1024U);
}

TEST(RunInputParseTest, RejectsOutOfContractGridValuesByName) {
    // Every [grid] refusal names the key it refused. Nothing is clamped and
    // nothing is snapped to the nearest legal value: a substituted quadrature
    // is a run on a grid the file did not ask for, and (since the record now
    // discloses the resolved settings) a silently-fixed key would be disclosed
    // as though the author had written it. The floors mirror the grid build's
    // own admission, so a bad number fails here with a key to blame instead of
    // surfacing as a bare kInvalidArgument from inside excgrid.
    struct Case {
        const char* text;
        const char* messageFragment;
    };

    const Case cases[] = {
        // Zero radial points is what RadialGrid::Create refuses; the parser
        // says which key.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nradial_points = 0\n",
         "grid.radial_points must be >= 1"},
        // A negative integer of the same shape.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nradial_points = -1\n",
         "grid.radial_points must be >= 1"},
        // 7 is the Lebedev size everyone reaches for first and this build does
        // not carry: the refusal names the shipped set rather than leaving the
        // author to guess which quadratures exist.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nangular_points = 7\n",
         "unknown grid.angular_points \"7\""},
        // ... and a dozen points is not a vocabulary word either (the angular
        // size is a SET, not a range).
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nangular_points = 12\n",
         "unknown grid.angular_points \"12\""},
        // alpha = 0 collapses every radial point onto the nucleus and excgrid
        // accepts it unchecked - the silent nonsense this check exists for.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nalpha = 0.0\n",
         "grid.alpha must be a finite number > 0"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nalpha = -0.5\n",
         "grid.alpha must be a finite number > 0"},
        // A non-finite alpha (TOML 1.0 nan) passes any sign test written the
        // wrong way round; the finiteness check is explicit (the
        // resources.memory_cap_gib spelling).
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nalpha = nan\n",
         "grid.alpha must be a finite number > 0"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nradial_exponent = 0\n",
         "grid.radial_exponent must be >= 1"},
        // A non-finite trim_weight is not a loose threshold: every comparison
        // against it is false, so EVERY grid point would be dropped and the run
        // would integrate nothing.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\ntrim_weight = nan\n",
         "grid.trim_weight must be a finite number >= 0"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\ntrim_weight = -1e-15\n",
         "grid.trim_weight must be a finite number >= 0"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nblock_target = 0\n",
         "grid.block_target must be >= 1"},
        // Wrong types never fall back to the defaults.
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nradial_points = \"75\"\n",
         "\"grid.radial_points\" must be an integer"},
        {"[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
         "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
         "[grid]\nalpha = \"0.5\"\n",
         "\"grid.alpha\" must be a number"},
    };

    for (const auto& test : cases)
    {
        const auto input = ParseRunInput(test.text);
        ASSERT_FALSE(input.has_value()) << "case: " << test.text;
        EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(input.error().message.find(test.messageFragment), std::string::npos)
            << "message: " << input.error().message;
    }

    // The angular refusal names the SHIPPED set, not just the offending number:
    // an author who wrote a size this build does not carry needs the ones it
    // does. Both ends of the list are checked, so a truncated list fails here.
    const auto angular = ParseRunInput(
        "[molecule]\natoms = [[\"H\", 0.0, 0.0, 0.0]]\n[basis]\norbital = \"sto-3g\"\n"
        "[method]\ntype = \"rks\"\nfunctional = \"slater\"\naccuracy = \"kNormal\"\n"
        "[grid]\nangular_points = 7\n");
    ASSERT_FALSE(angular.has_value());
    EXPECT_NE(angular.error().message.find("6 | 14"), std::string::npos)
        << "message: " << angular.error().message;
    EXPECT_NE(angular.error().message.find("302 | 350 | 434"), std::string::npos)
        << "message: " << angular.error().message;
}

// ---------------------------------------------------------------------------
// THE ORTHOGONAL BUILDER AXES (schema 35).
//
// The axis table is the contract, so it is checked as one: the MIGRATION check
// below holds every deprecated spelling against the axis spelling of the same
// selection, which is what makes this a migration rather than a rewrite - if the
// two vocabularies ever disagree about one combination, this test names it.

// One run input whose builder selection is spelled either way: `deprecatedLine`
// is a whole `[method]` key line or empty, and `builderBlock` is a whole
// `[builder]` table or empty. The rest of the file is fixed, minimal and
// orthogonal to the selection.
std::string BuilderSelectionToml(const std::string& deprecatedLine,
                                 const std::string& builderBlock) {
    return std::string(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [
    ["H", 0.0, 0.0, 0.0],
    ["H", 0.0, 0.0, 0.74],
]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
accuracy = "kNormal"
)") + deprecatedLine +
           "\n" + builderBlock;
}

// The selection a parsed input carries, as the pair the wiring reads - the one
// value both vocabularies must resolve to.
struct Selection {
    std::optional<BuilderKind> kind;
    bool lean = false;
};

Selection SelectionOf(const RunInput& input) {
    return {input.method.builder, input.method.leanDirect};
}

// THE MIGRATION CHECK, and the reason this file's axis tests exist: every
// spelling the deprecated key accepted resolves to exactly the selection its
// axis spelling resolves to. The table below is the axis table's own reading of
// each word, so a word that moved axes without the other side following fails
// here by name.
TEST(ParseInputTest, DeprecatedSpellingsResolveExactlyAsTheirAxisSpellingDoes) {
    struct Case {
        const char* legacyWord; ///< the deprecated [method] fock_builder word.
        const char* family; ///< The axis spelling of the same selection.
        const char* tier;
        const char* backend;
    };

    const Case cases[] = {
        {"direct", "direct", "in_memory", "cpu"},
        {"in_memory", "direct", "in_memory", "cpu"},
        {"lean", "direct", "lean", "cpu"},
        {"ri_j_link", "ri_j_link", "in_memory", "cpu"},
        {"ri_jk", "ri_jk", "in_memory", "cpu"},
        {"qfmm", "qfmm", "in_memory", "cpu"},
        {"gpu", "direct", "in_memory", "gpu"},
    };

    for (const auto& test : cases)
    {
        const auto legacy = ParseRunInput(
            BuilderSelectionToml(std::string("fock_builder = \"") + test.legacyWord + "\"", ""));
        ASSERT_TRUE(legacy.has_value())
            << "legacy word: " << test.legacyWord << " -> " << legacy.error().message;

        const auto axes = ParseRunInput(BuilderSelectionToml(
            "",
            std::string("[builder]\nintegral_family = \"") + test.family + "\"\nstorage_tier = \"" +
                test.tier + "\"\nexecution_backend = \"" + test.backend + "\"\n"));
        ASSERT_TRUE(axes.has_value())
            << "axis spelling of: " << test.legacyWord << " -> " << axes.error().message;

        EXPECT_EQ(SelectionOf(*legacy).kind, SelectionOf(*axes).kind)
            << "word: " << test.legacyWord;
        EXPECT_EQ(SelectionOf(*legacy).lean, SelectionOf(*axes).lean)
            << "word: " << test.legacyWord;
    }
}

// The tier word "in_memory" is an ALIAS of "direct" and not a second state: one
// BuilderKind value comes out of the two spellings, so no record and no refusal
// ever has to decide between them.
TEST(ParseInputTest, TheInMemoryAliasIsOneValueAndNotASecondState) {
    const auto alias = ParseRunInput(BuilderSelectionToml("fock_builder = \"in_memory\"", ""));
    ASSERT_TRUE(alias.has_value()) << alias.error().message;
    ASSERT_TRUE(alias->method.builder.has_value());
    EXPECT_EQ(*alias->method.builder, BuilderKind::kDirect);
    EXPECT_FALSE(alias->method.leanDirect);
}

// The lean member is the direct family's WITHIN-FAMILY tier, so it fills no
// family slot: the deprecated word leaves the builder slot empty and sets the
// flag, and its axis spelling does exactly the same.
TEST(ParseInputTest, TheLeanTierFillsNoFamilySlotInEitherVocabulary) {
    const auto legacy = ParseRunInput(BuilderSelectionToml("fock_builder = \"lean\"", ""));
    ASSERT_TRUE(legacy.has_value()) << legacy.error().message;
    EXPECT_FALSE(legacy->method.builder.has_value());
    EXPECT_TRUE(legacy->method.leanDirect);

    const auto axes =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nstorage_tier = \"lean\"\n"));
    ASSERT_TRUE(axes.has_value()) << axes.error().message;
    EXPECT_FALSE(axes->method.builder.has_value());
    EXPECT_TRUE(axes->method.leanDirect);
}

// An axis key alone is a complete request: the other two axes take the defaults
// the table names, so `integral_family` alone is that family with its own tier
// and the CPU backend.
TEST(ParseInputTest, AnAbsentAxisResolvesToTheTablesDefault) {
    const auto familyOnly =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nintegral_family = \"qfmm\"\n"));
    ASSERT_TRUE(familyOnly.has_value()) << familyOnly.error().message;
    ASSERT_TRUE(familyOnly->method.builder.has_value());
    EXPECT_EQ(*familyOnly->method.builder, BuilderKind::kQfmm);

    const auto backendOnly =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nexecution_backend = \"gpu\"\n"));
    ASSERT_TRUE(backendOnly.has_value()) << backendOnly.error().message;
    ASSERT_TRUE(backendOnly->method.builder.has_value());
    EXPECT_EQ(*backendOnly->method.builder, BuilderKind::kGpu);
}

// The axis -> kind table, checked against the enum itself: every kind the enum
// declares is placed on the three axes, and the GPU kind is a BACKEND on the
// direct family rather than a family of its own. That is the leak in its
// sharpest form, so it is pinned rather than described.
TEST(ParseInputTest, TheAxisTableIsTotalAndTheGpuKindIsABackend) {
    using qcx::io::AxesOfBuilder;
    using qcx::io::ExecutionBackend;
    using qcx::io::IntegralFamily;
    using qcx::io::StorageTier;

    const auto direct = AxesOfBuilder(BuilderKind::kDirect, false);
    EXPECT_EQ(direct.integralFamily, IntegralFamily::kDirect);
    EXPECT_EQ(direct.storageTier, StorageTier::kInMemory);
    EXPECT_EQ(direct.executionBackend, ExecutionBackend::kCpu);

    const auto lean = AxesOfBuilder(BuilderKind::kDirect, true);
    EXPECT_EQ(lean.integralFamily, IntegralFamily::kDirect);
    EXPECT_EQ(lean.storageTier, StorageTier::kLean);

    const auto gpu = AxesOfBuilder(BuilderKind::kGpu, false);
    EXPECT_EQ(gpu.integralFamily, IntegralFamily::kDirect)
        << "the GPU builder runs the DIRECT family's algorithm on a device";
    EXPECT_EQ(gpu.executionBackend, ExecutionBackend::kGpu);

    EXPECT_EQ(AxesOfBuilder(BuilderKind::kGpuSplit, false).executionBackend,
              ExecutionBackend::kGpuSplit);
    EXPECT_EQ(AxesOfBuilder(BuilderKind::kRiJLink, false).integralFamily, IntegralFamily::kRiJLink);
    EXPECT_EQ(AxesOfBuilder(BuilderKind::kRiJk, false).integralFamily, IntegralFamily::kRiJk);
    EXPECT_EQ(AxesOfBuilder(BuilderKind::kQfmm, false).integralFamily, IntegralFamily::kQfmm);

    // The words are the input's own vocabulary on both sides of the axis table.
    EXPECT_EQ(qcx::io::ToString(IntegralFamily::kRiJLink), "ri_j_link");
    EXPECT_EQ(qcx::io::ToString(StorageTier::kInMemory), "in_memory");
    EXPECT_EQ(qcx::io::ToString(ExecutionBackend::kGpu), "gpu");
}

// ONE REQUEST, ONE KEY: the deprecated key and the axes are two spellings of one
// selection, and a file that writes both is refused by name rather than having
// one silently elected.
TEST(ParseInputTest, TheTwoSpellingsOfOneSelectionAreRefusedTogether) {
    const auto input = ParseRunInput(BuilderSelectionToml(
        "fock_builder = \"qfmm\"", "[builder]\nintegral_family = \"direct\"\n"));
    ASSERT_FALSE(input.has_value());
    EXPECT_EQ(input.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(input.error().message.find("fock_builder"), std::string::npos)
        << "message: " << input.error().message;
    EXPECT_NE(input.error().message.find("One request, one key"), std::string::npos)
        << "message: " << input.error().message;
}

// The two tiers a RUNG key reaches are refused at the axis key, each by name and
// the disk one with the key that owns it as the remedy - the `forced_disk`
// precedent (schema 22), not a silent no-op and not a second spelling of a rung
// the key split deliberately put elsewhere.
TEST(ParseInputTest, TheRungTiersAreRefusedAtTheAxisKeyWithTheirOwner) {
    const auto disk =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nstorage_tier = \"disk\"\n"));
    ASSERT_FALSE(disk.has_value());
    EXPECT_EQ(disk.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(disk.error().message.find("ri_tensor_mode"), std::string::npos)
        << "message: " << disk.error().message;
    EXPECT_NE(disk.error().message.find("force_disk_ri"), std::string::npos)
        << "message: " << disk.error().message;

    const auto blocked =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nstorage_tier = \"blocked\"\n"));
    ASSERT_FALSE(blocked.has_value());
    EXPECT_EQ(blocked.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(blocked.error().message.find("no request surface"), std::string::npos)
        << "message: " << blocked.error().message;
}

// The vocabulary-only backend value stays vocabulary-only: "gpu_split" is named
// on the axis and refused as a request, so the axis carries the candidate's name
// while execution stays on the wired backends (the kGpuSplit precedent).
TEST(ParseInputTest, TheSplitBackendIsNamedOnTheAxisAndRefusedAsARequest) {
    const auto split =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nexecution_backend = \"gpu_split\"\n"));
    ASSERT_FALSE(split.has_value());
    EXPECT_EQ(split.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(split.error().message.find("no run path"), std::string::npos)
        << "message: " << split.error().message;
}

// An un-runnable COMBINATION is refused with both sides named - the rule (a
// request that names a mechanism on a family that cannot honour it) read across
// axes instead of across keys.
TEST(ParseInputTest, UnRunnableAxisCombinationsAreRefusedByName) {
    const auto leanOnRiJ = ParseRunInput(BuilderSelectionToml(
        "", "[builder]\nintegral_family = \"ri_j_link\"\nstorage_tier = \"lean\"\n"));
    ASSERT_FALSE(leanOnRiJ.has_value());
    EXPECT_EQ(leanOnRiJ.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(leanOnRiJ.error().message.find("ri_j_link"), std::string::npos)
        << "message: " << leanOnRiJ.error().message;
    EXPECT_NE(leanOnRiJ.error().message.find("lean"), std::string::npos)
        << "message: " << leanOnRiJ.error().message;

    const auto gpuOnQfmm = ParseRunInput(BuilderSelectionToml(
        "", "[builder]\nintegral_family = \"qfmm\"\nexecution_backend = \"gpu\"\n"));
    ASSERT_FALSE(gpuOnQfmm.has_value());
    EXPECT_EQ(gpuOnQfmm.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(gpuOnQfmm.error().message.find("qfmm"), std::string::npos)
        << "message: " << gpuOnQfmm.error().message;

    const auto leanGpu = ParseRunInput(BuilderSelectionToml(
        "", "[builder]\nstorage_tier = \"lean\"\nexecution_backend = \"gpu\"\n"));
    ASSERT_FALSE(leanGpu.has_value());
    EXPECT_EQ(leanGpu.error().code, qcx::ErrorCode::kInvalidArgument);
}

// An unknown word on any axis is refused by name WITH the accepted list, so the
// axis keys are not a hole a typo can fall through.
TEST(ParseInputTest, AnUnknownAxisWordIsRefusedWithTheAcceptedList) {
    const auto family =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nintegral_family = \"ri_j\"\n"));
    ASSERT_FALSE(family.has_value());
    EXPECT_NE(family.error().message.find("direct | ri_j_link | ri_jk | qfmm"), std::string::npos)
        << "message: " << family.error().message;

    const auto tier =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nstorage_tier = \"memory\"\n"));
    ASSERT_FALSE(tier.has_value());
    EXPECT_NE(tier.error().message.find("lean | in_memory"), std::string::npos)
        << "message: " << tier.error().message;

    const auto backend =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nexecution_backend = \"cuda\"\n"));
    ASSERT_FALSE(backend.has_value());
    EXPECT_NE(backend.error().message.find("cpu | gpu"), std::string::npos)
        << "message: " << backend.error().message;
}

// THE DEVICE AXIS. The key states WHICH device the run requires its kernels to
// execute on - the question `execution_backend` cannot answer, since `gpu` names
// the device CLASS and nothing could name the particular device. Two properties
// are pinned here, and the first is the whole reason the key is not a fourth
// word on an existing axis key.
TEST(ParseInputTest, TheDeviceAxisStatesARequirementWithoutSelectingABuilder) {
    // A requirement is not a selection: `[builder] device` alone must leave the
    // family, the tier and the backend at their own defaults, because filling
    // them would make stating where a run executes silently decide what it runs
    // - and it would move an absent-key run off the size ladder onto the direct
    // family's in-memory tier.
    const auto hostAlone =
        ParseRunInput(BuilderSelectionToml("", "[builder]\ndevice = \"host\"\n"));
    ASSERT_TRUE(hostAlone.has_value()) << hostAlone.error().message;
    ASSERT_TRUE(hostAlone->builder.device.has_value());
    EXPECT_EQ(hostAlone->builder.device->target, DeviceTarget::kHost);
    EXPECT_EQ(hostAlone->builder.device->index, -1);
    EXPECT_FALSE(hostAlone->builder.integralFamily.has_value());
    EXPECT_FALSE(hostAlone->builder.storageTier.has_value());
    EXPECT_FALSE(hostAlone->builder.executionBackend.has_value());
    EXPECT_FALSE(hostAlone->method.builder.has_value());

    // The device this key exists for: a PARTICULAR device, by index, beside the
    // backend word that names the class. Parsed, not stored as text - the index
    // is the requirement, and a record that echoed a string could not be held to
    // one.
    const auto cuda = ParseRunInput(
        BuilderSelectionToml("", "[builder]\nexecution_backend = \"gpu\"\ndevice = \"cuda:1\"\n"));
    ASSERT_TRUE(cuda.has_value()) << cuda.error().message;
    ASSERT_TRUE(cuda->builder.device.has_value());
    EXPECT_EQ(cuda->builder.device->target, DeviceTarget::kCuda);
    EXPECT_EQ(cuda->builder.device->index, 1);
    EXPECT_EQ(DeviceSelectorText(*cuda->builder.device), "cuda:1");

    // The agree arm, one word each side: host beside cpu.
    const auto hostOnCpu = ParseRunInput(
        BuilderSelectionToml("", "[builder]\nexecution_backend = \"cpu\"\ndevice = \"host\"\n"));
    ASSERT_TRUE(hostOnCpu.has_value()) << hostOnCpu.error().message;
    EXPECT_EQ(DeviceSelectorText(*hostOnCpu->builder.device), "host");

    // And an absent key states nothing: no requirement, so the field is absent
    // rather than defaulted to the host - the null-honesty rule, because a
    // defaulted "host" would read as a requirement the file never wrote.
    const auto absent = ParseRunInput(BuilderSelectionToml("", ""));
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_FALSE(absent->builder.device.has_value());
}

// THE DEVICE REQUIREMENT AGAINST THE BACKEND: the pair must AGREE, and a pair
// that does not is REFUSED by name with both sides stated rather than resolved.
TEST(ParseInputTest, ADeviceRequirementTheBackendCannotHonourIsRefusedByName) {
    // A CUDA device beside the host backend, and the same request beside NO
    // backend word at all - whose resolution is cpu, so it is the same refusal.
    for (const std::string& block :
         {std::string{"[builder]\nexecution_backend = \"cpu\"\ndevice = \"cuda:0\"\n"},
          std::string{"[builder]\ndevice = \"cuda:0\"\n"}})
    {
        const auto run = ParseRunInput(BuilderSelectionToml("", block));
        ASSERT_FALSE(run.has_value()) << "a device requirement was resolved rather than "
                                         "answered: "
                                      << block;
        EXPECT_EQ(run.error().code, qcx::ErrorCode::kInvalidArgument);
        // The refusal names BOTH sides, so an author knows which key to change.
        EXPECT_NE(run.error().message.find("builder.device = \"cuda:0\""), std::string::npos)
            << run.error().message;
        EXPECT_NE(run.error().message.find("builder.execution_backend"), std::string::npos)
            << run.error().message;
    }

    // The other direction: the host required beside the device backend. One run
    // cannot require both, and the refusal says so rather than electing one.
    const auto hostOnGpu = ParseRunInput(
        BuilderSelectionToml("", "[builder]\nexecution_backend = \"gpu\"\ndevice = \"host\"\n"));
    ASSERT_FALSE(hostOnGpu.has_value());
    EXPECT_EQ(hostOnGpu.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(hostOnGpu.error().message.find("builder.device = \"host\""), std::string::npos)
        << hostOnGpu.error().message;
}

// The device SELECTOR is parsed, not stored as opaque text: a selector that names
// no device is refused by name at the key, with the shape it wanted, rather than
// accepted and compared as a string by every later reader.
TEST(ParseInputTest, AnUnusableDeviceSelectorIsRefusedByName) {
    const auto tpu = ParseRunInput(BuilderSelectionToml("", "[builder]\ndevice = \"tpu\"\n"));
    ASSERT_FALSE(tpu.has_value());
    EXPECT_EQ(tpu.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(tpu.error().message.find("unknown builder.device"), std::string::npos)
        << "message: " << tpu.error().message;
    EXPECT_NE(tpu.error().message.find("(host | cuda)"), std::string::npos)
        << "message: " << tpu.error().message;

    // The index is the requirement, so a `cuda` selection without a usable one
    // is refused rather than stored as text no later reader can be held to. The
    // WORD is accepted - it is the index that is missing - so the two sentences
    // name different things, and a user who wrote the word correctly is not sent
    // looking for the accepted set.
    for (const char* selector : {"cuda:", "cuda:x", "cuda:-1"})
    {
        const auto badIndex = ParseRunInput(
            BuilderSelectionToml("", std::string{"[builder]\ndevice = \""} + selector + "\"\n"));
        ASSERT_FALSE(badIndex.has_value()) << selector;
        EXPECT_EQ(badIndex.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_NE(badIndex.error().message.find("names no CUDA device"), std::string::npos)
            << "message: " << badIndex.error().message;
        EXPECT_NE(badIndex.error().message.find("cuda:0"), std::string::npos)
            << "message: " << badIndex.error().message;
    }
}

// THE DEPRECATION RECORD. The old key is ACCEPTED - never refused - and the
// input states which vocabulary it was written in, so the run
// record can carry that fact. An absent key records nothing.
TEST(ParseInputTest, TheDeprecatedKeyIsAcceptedAndRecordsItsOwnWord) {
    const auto legacy = ParseRunInput(BuilderSelectionToml("fock_builder = \"qfmm\"", ""));
    ASSERT_TRUE(legacy.has_value()) << legacy.error().message;
    ASSERT_TRUE(legacy->builder.legacyFockBuilderWord.has_value());
    EXPECT_EQ(*legacy->builder.legacyFockBuilderWord, "qfmm");

    const auto axes =
        ParseRunInput(BuilderSelectionToml("", "[builder]\nintegral_family = \"qfmm\"\n"));
    ASSERT_TRUE(axes.has_value()) << axes.error().message;
    EXPECT_FALSE(axes->builder.legacyFockBuilderWord.has_value());
    ASSERT_TRUE(axes->builder.integralFamily.has_value());

    // No builder key at all: nothing claimed, nothing recorded - the ladder's run.
    const auto absent = ParseRunInput(BuilderSelectionToml("", ""));
    ASSERT_TRUE(absent.has_value()) << absent.error().message;
    EXPECT_FALSE(absent->builder.legacyFockBuilderWord.has_value());
    EXPECT_FALSE(absent->builder.integralFamily.has_value());
}

// THE RECORD SIDE. The resolved selection reaches the record in the new
// vocabulary, with the deprecation recorded when the input used the old key. A
// record that could only spell the legacy word is the defect the block exists to
// remove, so its presence and its contents are both pinned.
TEST(ParseInputTest, TheRecordCarriesTheResolvedAxesAndTheDeprecation) {
    const auto input = ParseRunInput(BuilderSelectionToml("fock_builder = \"lean\"", ""));
    ASSERT_TRUE(input.has_value()) << input.error().message;
    ASSERT_TRUE(input->method.leanDirect);

    const auto resolved = qcx::io::AxesOfBuilder(BuilderKind::kDirect, true);

    RunResult result;
    result.builderAxes = qcx::io::RunBuilderAxes{
        std::string(qcx::io::ToString(resolved.integralFamily)),
        std::string(qcx::io::ToString(resolved.storageTier)),
        std::string(qcx::io::ToString(resolved.executionBackend)),
        input->builder.legacyFockBuilderWord,
        input->builder.legacyFockBuilderWord.has_value() ? "fock_builder" : "ladder"};

    const std::string json = SerializeRunResultJson(result);
    EXPECT_NE(json.find("\"builder_axes\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"integral_family\": \"direct\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"storage_tier\": \"lean\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"execution_backend\": \"cpu\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"requested_by\": \"fock_builder\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"legacy_spelling\": \"lean\""), std::string::npos) << json;
    EXPECT_NE(json.find("\"deprecated_key\": \"method.fock_builder\""), std::string::npos) << json;
}
