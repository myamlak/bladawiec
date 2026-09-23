// The Kohn-Sham path started from a user's INPUT FILE: one .toml on disk, the
// way `qcx run <file.toml>` starts a run.
//
// Every other Kohn-Sham driver test drives ParseRunInput on a string literal
// (run_driver_test.cpp's DriverPinTest rows). Those rows pin the physics; none
// of them reads a file, so nothing in the tree asserted the leg a user
// actually takes - the path through ParseRunInputFile, a path on disk, and the
// record that comes back. That leg is what this file owns, and it is the leg a
// report of "a density-functional run cannot be started from an input file"
// is about.
//
// The fixture is the repository's water/STO-3G geometry (run_driver_test.cpp's
// kH2oToml, Angstrom: no [molecule] units key, which is what makes this the
// unit the parser resolves by default) and B3LYP, a shipped hybrid. The
// assertions are chosen so that a run which ignored the schema and computed
// Hartree-Fock under the "rks" label fails here: the same file one method word
// apart is the Hartree-Fock run this one must NOT reproduce, and the same file
// one functional name apart is a different energy again, which is what shows
// the name reached the physics rather than only the record.

#include "qcx/driver/run_driver.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/io/validate_input.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using qcx::io::MethodType;
using qcx::io::ParseRunInputFile;
using qcx::io::RunInput;
using qcx::io::ValidateInput;

// One unique .toml per call: a row that reused a path would read the previous
// row's bytes if a write ever failed halfway, and the run under test would be
// a run of the wrong file.
std::filesystem::path TempInputPath(std::string_view tag) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("qcx-ks-input-" + std::string(tag) + "-" + std::to_string(counter++) + ".toml");
}

// Writes \p toml to a fresh file and parses THAT, so the run's input is the
// bytes on disk and not the string this file holds. The stream is binary so
// the newlines written are the bytes written (the text mode would turn each
// one into a carriage-return pair, and the parser's line handling is part of
// what a file-driven test is checking).
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

// The fixture: water/STO-3G in Angstrom, one [method] body from the two this
// file compares, plus an optional trailing block. Everything except the [method]
// block is identical between any two runs here, so whatever separates their
// energies is the method block.
std::string WaterToml(std::string_view methodBody, std::string_view trailing = {}) {
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
)" + std::string(trailing);
}

// One run from a written file, validated and driven exactly as the command
// line drives it (main.cpp: ParseRunInputFile, ValidateInput, RunDriverOutcome).
// Fails the calling test on any refusal, so each caller reads only numbers.
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

// The whole user path in one row: a file with a method, a functional, a basis
// and a geometry; the run started from that file; the result read back.
TEST(KsInputFileTest, KohnShamRunStartedFromAnInputFileReachesTheFunctional) {
    const auto path = TempInputPath("b3lyp");

    auto outcome = RunFile(path, WaterToml(R"(type = "rks"
functional = "b3lyp"
)"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;

    // The record names the physics it ran: the method word the file wrote and
    // the functional the registry resolved it to.
    EXPECT_EQ(result.method, MethodType::kRks);
    ASSERT_TRUE(result.functional.has_value());
    EXPECT_EQ(*result.functional, "b3lyp");

    // The grid the run built: this file wrote no [grid] block, so the six
    // numbers are the engine's own defaults, disclosed rather than implied.
    ASSERT_TRUE(result.xcGrid.has_value());
    EXPECT_EQ(result.xcGrid->radialPoints, 75u);
    EXPECT_EQ(result.xcGrid->angularPoints, 302u);
    EXPECT_EQ(result.xcGrid->alpha, 0.5);
    EXPECT_EQ(result.xcGrid->radialExponent, 2u);
    EXPECT_EQ(result.xcGrid->trimWeight, 1e-15);
    EXPECT_EQ(result.xcGrid->blockTarget, 1024u);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0);
}

// The run the file above must NOT be: the same molecule, basis and tolerances
// one method word away, with the Kohn-Sham keys removed (the validator refuses
// a functional on an rhf run, which is itself pinned next door). A DFT label
// over a Hartree-Fock Fock would land on the second number, so the two are
// compared rather than the Kohn-Sham energy alone.
TEST(KsInputFileTest, TheSameFileOnTheHartreeFockWordIsADifferentEnergy) {
    auto ks = RunFile(TempInputPath("rks-slater"), WaterToml(R"(type = "rks"
functional = "slater"
)"));
    ASSERT_TRUE(ks.has_value()) << ks.error().message;

    auto hf = RunFile(TempInputPath("rhf"), WaterToml(R"(type = "rhf"
)"));
    ASSERT_TRUE(hf.has_value()) << hf.error().message;

    // The Hartree-Fock arm names no functional and builds no grid - the two
    // Kohn-Sham payloads are absent rather than null or zero-filled.
    EXPECT_EQ(hf->result.method, MethodType::kRhf);
    EXPECT_FALSE(hf->result.functional.has_value());
    EXPECT_FALSE(hf->result.xcGrid.has_value());

    // The Hartree-Fock number is the repository's pinned water/STO-3G energy,
    // so this fixture is the pinned one and the difference below is the
    // method's rather than the geometry's.
    EXPECT_NEAR(hf->result.totalEnergyHartree, -74.96292827, 1e-5);

    // Slater exchange moves this fixture by a fraction of a Hartree, orders of
    // magnitude above the pin's tolerance: a run that had dropped the
    // functional and built the Hartree-Fock Fock under the "rks" label would
    // land on the same number as the Hartree-Fock leg.
    EXPECT_GT(std::abs(ks->result.totalEnergyHartree - hf->result.totalEnergyHartree), 1e-3);
}

// The functional NAME reaches the physics and not only the record: two files
// whose [method] blocks differ by that one word come back with different
// energies and different functional words. A driver that resolved the name for
// the record and integrated something else - or the same thing for every name
// - passes every other assertion in this file.
TEST(KsInputFileTest, ChangingTheFunctionalNameMovesTheEnergy) {
    auto b3lyp = RunFile(TempInputPath("byp"), WaterToml(R"(type = "rks"
functional = "b3lyp"
)"));
    ASSERT_TRUE(b3lyp.has_value()) << b3lyp.error().message;

    auto slater = RunFile(TempInputPath("sl"), WaterToml(R"(type = "rks"
functional = "slater"
)"));
    ASSERT_TRUE(slater.has_value()) << slater.error().message;

    ASSERT_TRUE(b3lyp->result.functional.has_value());
    ASSERT_TRUE(slater->result.functional.has_value());
    EXPECT_EQ(*b3lyp->result.functional, "b3lyp");
    EXPECT_EQ(*slater->result.functional, "slater");

    // B3LYP and Slater differ by the gradient corrections and the exact-
    // exchange fraction; no quadrature error comes near this gap.
    EXPECT_GT(std::abs(b3lyp->result.totalEnergyHartree - slater->result.totalEnergyHartree), 1e-3);
}

// The exchange-correlation gradient, asked for BY THE FILE. This is the only
// input key that reaches the fixed-density walk
// (driver/internal/ks_grid.hpp's AddXcGradientContribution), which until this
// key existed had no producer on a real run: the walk was reachable from a test
// and from nothing a user could write.
//
// Which entry belongs to which atom. The record's per-atom blocks are indexed
// in the molecule's canonical atom order - the renumbering by element and then
// by position that a molecule applies on construction - and this file writes
// its rows as O, H, H, so the two orders differ here and a block read in file
// order is a permuted vector that still looks like a gradient. The assertions
// below therefore do not assume an index: the density-at-nuclei block beside
// the gradient names the oxygen (rho at an oxygen nucleus is orders of magnitude
// above rho at a hydrogen's), and the symmetry the gradient must obey is then
// stated about THAT entry and the two the witness leaves. A block indexed in
// another order than the block beside it fails here.
//
// What the symmetry is. Water in this geometry has the mirror plane x = 0 and
// lies in the plane z = 0, so: the oxygen carries no x component, the two
// hydrogens carry mirrored x components and equal y components, every z
// component vanishes, and the walk carries no net force at all (a property of
// the construction rather than of this fixture: the weights see the geometry
// only through differences).
TEST(KsInputFileTest, TheGradientKeyAsksForTheExchangeCorrelationWalk) {
    const std::string ksBody = R"(type = "rks"
functional = "b3lyp"
)";

    // Absent key, no block: the opt-in rule, and the statement that a record
    // without the block means "no walk ran" rather than "the contribution was
    // zero".
    auto plain = RunFile(TempInputPath("no-gradient"), WaterToml(ksBody));
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    ASSERT_TRUE(plain->result.properties.has_value());
    EXPECT_FALSE(plain->result.properties->xcGradient.has_value());

    // The same file with the key.
    auto asked = RunFile(TempInputPath("gradient"), WaterToml(ksBody, R"([properties]
xc_gradient = true
density_at_nuclei = true
)"));
    ASSERT_TRUE(asked.has_value()) << asked.error().message;

    // Asking for the derivative must not move the run it differentiates: same
    // energy, same stop, same grid. A walk that shared mutable state with the
    // energy path would show up here.
    EXPECT_EQ(asked->result.totalEnergyHartree, plain->result.totalEnergyHartree);
    EXPECT_EQ(asked->result.converged, plain->result.converged);
    EXPECT_EQ(asked->result.iterations, plain->result.iterations);
    ASSERT_TRUE(asked->result.xcGrid.has_value());
    ASSERT_TRUE(plain->result.xcGrid.has_value());
    EXPECT_EQ(asked->result.xcGrid->radialPoints, plain->result.xcGrid->radialPoints);

    ASSERT_TRUE(asked->result.properties.has_value());
    ASSERT_TRUE(asked->result.properties->xcGradient.has_value());
    const auto& block = *asked->result.properties->xcGradient;

    // Three atoms, three directions, atom-major.
    ASSERT_EQ(block.gradient.size(), 9U);
    EXPECT_NE(block.energyHartree, 0.0);

    // The witness: the entry the record's density-at-nuclei block puts the
    // oxygen on. rho(O) and rho(H) differ by three orders of magnitude in this
    // basis, so which entry is the oxygen is not a close call.
    ASSERT_TRUE(asked->result.properties->densityAtNuclei.has_value());
    const std::vector<double>& rho = asked->result.properties->densityAtNuclei->values;
    ASSERT_EQ(rho.size(), 3U);

    std::size_t oxygen = 0;

    for (std::size_t atom = 0; atom < rho.size(); ++atom)
    {
        if (rho[atom] > rho[oxygen])
        {
            oxygen = atom;
        }
    }

    const std::size_t firstHydrogen = (oxygen + 1) % rho.size();
    const std::size_t secondHydrogen = (oxygen + 2) % rho.size();
    EXPECT_GT(rho[oxygen], 100.0 * rho[firstHydrogen])
        << "the witness did not resolve an oxygen; rho = " << rho[0] << ", " << rho[1] << ", "
        << rho[2];

    // One component of one atom: direction d of atom i at 3*i + d.
    const auto component = [&block](std::size_t atom, std::size_t direction) {
        return block.gradient[3U * atom + direction];
    };

    for (std::size_t atom = 0; atom < rho.size(); ++atom)
    {
        for (std::size_t direction = 0; direction < 3U; ++direction)
        {
            EXPECT_TRUE(std::isfinite(component(atom, direction)));
        }

        // The molecule is planar in z.
        EXPECT_NEAR(component(atom, 2), 0.0, 1e-10) << "atom " << atom;
    }

    // The oxygen sits on the mirror plane; the hydrogens mirror each other
    // through it and share the y component a mirror plane leaves alone.
    EXPECT_NEAR(component(oxygen, 0), 0.0, 1e-10);
    EXPECT_NEAR(component(firstHydrogen, 0), -component(secondHydrogen, 0), 1e-10);
    EXPECT_NEAR(component(firstHydrogen, 1), component(secondHydrogen, 1), 1e-10);

    double netForce = 0.0;

    for (const double value : block.gradient)
    {
        netForce += value;
    }

    RecordProperty("xc gradient energy (Hartree)", block.energyHartree);
    RecordProperty("xc gradient net force (Hartree/Bohr)", netForce);
    EXPECT_NEAR(netForce, 0.0, 1e-10);

    // The name still reaches the physics: a second functional's gradient is a
    // different vector, so a block filled from a default functional - or from a
    // walk that ignored the name - is caught.
    auto other = RunFile(TempInputPath("gradient-slater"),
                         WaterToml(R"(type = "rks"
functional = "slater"
)",
                                   R"([properties]
xc_gradient = true
)"));
    ASSERT_TRUE(other.has_value()) << other.error().message;
    ASSERT_TRUE(other->result.properties->xcGradient.has_value());
    const auto& otherBlock = *other->result.properties->xcGradient;
    ASSERT_EQ(otherBlock.gradient.size(), 9U);

    double maxDifference = 0.0;

    for (std::size_t i = 0; i < block.gradient.size(); ++i)
    {
        maxDifference =
            std::max(maxDifference, std::abs(block.gradient[i] - otherBlock.gradient[i]));
    }

    EXPECT_GT(maxDifference, 1e-6);
}

// The gradient request is refused by NAME on a method that has no functional to
// differentiate, through the file path a user takes rather than only in the
// validator's own unit test.
TEST(KsInputFileTest, TheGradientKeyOnAHartreeFockFileIsRefusedByName) {
    auto refused = RunFile(TempInputPath("gradient-rhf"),
                           WaterToml(R"(type = "rhf"
)",
                                     R"([properties]
xc_gradient = true
)"));
    ASSERT_FALSE(refused.has_value());
    EXPECT_NE(refused.error().message.find("properties.xc_gradient"), std::string::npos)
        << refused.error().message;
}
