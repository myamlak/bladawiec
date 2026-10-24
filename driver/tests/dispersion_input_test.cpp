// The dispersion path started from a user's INPUT FILE: one .toml on disk, the
// way `qcx run <file.toml>` starts a run, and the correction read back from the
// record.
//
// The acceptance is two claims. First, the input-file block and the total
// energy must agree with the LIBRARY call on the same geometry: the record's
// `dispersion.energy_hartree` is compared with excgrid's own D3/D4 entry point
// on the molecule the file built, and the total-energy difference against the
// same file without the block must be that number - so a seam that evaluated
// something else, or added it twice, fails here. Second, the parameter source
// the driver resolves is recorded: the method family (`hf` on the Hartree-Fock
// lanes, the functional name on the Kohn-Sham lanes), a named preset, and the
// refusals the contract states by name.
//
// The D4 caffeine reference ratio (0.997913) lives in the module's own suite
// (`dispersion/tests/dispersion_test.cpp`), where the published geometry and
// number are compared without an SCF in the way; this file drives the same D4
// arm through the input-file leg on the water fixture.

#include "qcx/driver/run_driver.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/io/validate_input.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <excgrid/d4.hpp>
#include <excgrid/dispersion.hpp>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using qcx::io::ParseRunInputFile;
using qcx::io::RunInput;
using qcx::io::ValidateInput;

// One unique .toml per call: a row that reused a path would read the previous
// row's bytes if a write ever failed halfway (the ks_input_file_test rule).
std::filesystem::path TempInputPath(std::string_view tag) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("qcx-dispersion-input-" + std::string(tag) + "-" + std::to_string(counter++) + ".toml");
}

// Writes \p toml to a fresh file and parses THAT, so the run's input is the
// bytes on disk, not the string this file holds.
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

// The fixture: water/STO-3G in Angstrom (the ks_input_file_test geometry), one
// [method] body, plus an optional trailing block. Everything except the
// [method] body and the trailing block is identical between two runs, so
// whatever separates their records is those two blocks.
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

struct FixtureAtom {
    const char* symbol;
    int atomicNumber;
    double x;
    double y;
    double z;
};

// The same water geometry the file writes, converted by the tree's one factor
// (the parser's own conversion), so the direct call and the input-file run
// stand on bit-identical coordinates.
constexpr FixtureAtom kWater[] = {
    {"O", 8, 0.0, 0.0, 0.0},
    {"H", 1, 0.7569503270127429, 0.58588227657553, 0.0},
    {"H", 1, -0.7569503270127429, 0.58588227657553, 0.0},
};

qcx::Result<qcx::molecule::Molecule> MakeMolecule(std::span<const FixtureAtom> atoms) {
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atoms.size(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> elements;
    elements.reserve(atoms.size());

    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        elements.push_back({atoms[i].symbol, atoms[i].atomicNumber, 0.0});
        (*coordinates)(i, 0) = atoms[i].x * qcx::molecule::kAngstromToBohr;
        (*coordinates)(i, 1) = atoms[i].y * qcx::molecule::kAngstromToBohr;
        (*coordinates)(i, 2) = atoms[i].z * qcx::molecule::kAngstromToBohr;
    }

    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(std::move(elements), std::move(*coordinates), 0, 1);
}

} // namespace

TEST(DispersionInputTest, D3InputFileRunMatchesTheLibraryCall) {
    // The method-default row: `form = "d3"` and nothing else, on rhf. The
    // family is "hf", the record says so, and the values are that preset's.
    const auto baselinePath = TempInputPath("d3-baseline");
    auto baseline = RunFile(baselinePath, WaterToml(R"(type = "rhf"
)"));
    ASSERT_TRUE(baseline.has_value()) << baseline.error().message;

    const auto path = TempInputPath("d3-default");
    auto outcome = RunFile(path,
                           WaterToml(R"(type = "rhf"
)",
                                     "[dispersion]\nform = \"d3\"\n"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;
    ASSERT_TRUE(result.dispersion.has_value());
    EXPECT_EQ(result.dispersion->form, "d3");
    EXPECT_EQ(result.dispersion->parameterSource, "method");
    ASSERT_TRUE(result.dispersion->parameterSet.has_value());
    EXPECT_EQ(*result.dispersion->parameterSet, "hf");

    const auto preset = excgrid::D3Preset("hf");
    ASSERT_TRUE(preset.has_value());
    EXPECT_EQ(result.dispersion->parameters.size(), 6u);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("s6"), preset->s6);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("s8"), preset->s8);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("rs6"), preset->rs6);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("alpha6"), preset->alpha6);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("rs8"), preset->rs8);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("alpha8"), preset->alpha8);

    // The library's own call on the same geometry: the record's number is the
    // kernel's number, not a second evaluation with other parameters.
    auto molecule = MakeMolecule(kWater);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto geometry = qcx::grid::ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    auto direct = excgrid::GrimmeD3Energy(*geometry, *preset);
    ASSERT_TRUE(direct.has_value());
    EXPECT_NEAR(result.dispersion->energyHartree, *direct, 1e-12);
    EXPECT_LT(*direct, 0.0);

    // The correction reaches the TOTAL exactly once: the same file without the
    // block is the same SCF run, and the total-energy difference is the
    // block's own energy. The electronic energy is untouched, which is the
    // statement that the SCF loop stayed dispersion-free.
    EXPECT_NEAR(result.totalEnergyHartree - baseline->result.totalEnergyHartree, *direct, 1e-12);
    EXPECT_NEAR(result.electronicEnergyHartree, baseline->result.electronicEnergyHartree, 1e-12);
}

TEST(DispersionInputTest, D4InputFileRunMatchesTheLibraryCall) {
    // The D4 arm through the input-file leg: a named preset, the two-body
    // kernel's values, and the record naming the preset it consumed.
    const auto path = TempInputPath("d4-preset");
    auto outcome = RunFile(path,
                           WaterToml(R"(type = "rhf"
)",
                                     "[dispersion]\nform = \"d4\"\npreset = \"b3lyp\"\n"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;
    ASSERT_TRUE(result.dispersion.has_value());
    EXPECT_EQ(result.dispersion->form, "d4");
    EXPECT_EQ(result.dispersion->parameterSource, "preset");
    ASSERT_TRUE(result.dispersion->parameterSet.has_value());
    EXPECT_EQ(*result.dispersion->parameterSet, "b3lyp");

    const auto preset = excgrid::D4Preset("b3lyp");
    ASSERT_TRUE(preset.has_value());
    EXPECT_EQ(result.dispersion->parameters.size(), 4u);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("s6"), preset->s6);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("s8"), preset->s8);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("a1"), preset->a1);
    EXPECT_DOUBLE_EQ(result.dispersion->parameters.at("a2"), preset->a2);

    auto molecule = MakeMolecule(kWater);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto geometry = qcx::grid::ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    auto direct = excgrid::D4TwoBodyEnergy(*geometry, *preset);
    ASSERT_TRUE(direct.has_value());
    EXPECT_NEAR(result.dispersion->energyHartree, *direct, 1e-12);
}

TEST(DispersionInputTest, TheKohnShamFamilyIsTheFunctionalName) {
    // On the Kohn-Sham lanes the no-parameter default reads the functional
    // the file named, not "hf": the record's parameter set is "b3lyp", and
    // the number is that preset's - a driver that hardcoded "hf" here would
    // record a different set and a different energy.
    const auto path = TempInputPath("rks-family");
    auto outcome = RunFile(path,
                           WaterToml(R"(type = "rks"
functional = "b3lyp"
)",
                                     "[dispersion]\nform = \"d3\"\n"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;
    ASSERT_TRUE(result.dispersion.has_value());
    EXPECT_EQ(result.dispersion->form, "d3");
    EXPECT_EQ(result.dispersion->parameterSource, "method");
    ASSERT_TRUE(result.dispersion->parameterSet.has_value());
    EXPECT_EQ(*result.dispersion->parameterSet, "b3lyp");

    const auto preset = excgrid::D3Preset("b3lyp");
    ASSERT_TRUE(preset.has_value());
    auto molecule = MakeMolecule(kWater);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto geometry = qcx::grid::ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    auto direct = excgrid::GrimmeD3Energy(*geometry, *preset);
    ASSERT_TRUE(direct.has_value());
    EXPECT_NEAR(result.dispersion->energyHartree, *direct, 1e-12);
}

TEST(DispersionInputTest, ARequestTheContractRefusesStopsTheRunByName) {
    // A key of the other form: the validator's cross-key refusal, before any
    // run starts.
    auto crossForm = RunFile(TempInputPath("cross-form"),
                             WaterToml(R"(type = "rhf"
)",
                                       "[dispersion]\nform = \"d3\"\na1 = 0.4\n"));
    ASSERT_FALSE(crossForm.has_value());
    EXPECT_EQ(crossForm.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(crossForm.error().message.find("does not take a1"), std::string::npos)
        << "message: " << crossForm.error().message;

    // D4 on a charged molecule: refused by name at validation (the shipped
    // EEQ charges are evaluated at total charge 0).
    const char* const chargedToml = R"(
[molecule]
charge = 1
multiplicity = 2
atoms = [["He", 0.0, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
accuracy = "kNormal"

[dispersion]
form = "d4"
)";
    auto charged = RunFile(TempInputPath("charged"), chargedToml);
    ASSERT_FALSE(charged.has_value());
    EXPECT_EQ(charged.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(charged.error().message.find("molecule.charge = 1"), std::string::npos)
        << "message: " << charged.error().message;

    // D4 above its table (the parameterization ends at radon, Z = 86): the
    // module refuses francium (Z = 87), and the refusal reaches the caller
    // instead of a number from a partial table.
    const char* const franciumToml = R"(
[molecule]
multiplicity = 2
atoms = [["Fr", 0.0, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
accuracy = "kNormal"

[dispersion]
form = "d4"
preset = "b3lyp"
)";
    auto francium = RunFile(TempInputPath("francium"), franciumToml);
    ASSERT_FALSE(francium.has_value());
    EXPECT_EQ(francium.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(francium.error().message.find("87"), std::string::npos)
        << "message: " << francium.error().message;

    // D3 above ITS table: excgrid's kernel refuses, translated with the
    // dispersion context rather than surfacing as a bare code.
    const char* const americiumToml = R"(
[molecule]
multiplicity = 2
atoms = [["Am", 0.0, 0.0, 0.0]]

[basis]
orbital = "sto-3g"

[method]
type = "uhf"
accuracy = "kNormal"

[dispersion]
form = "d3"
preset = "b3lyp"
)";
    auto americium = RunFile(TempInputPath("americium"), americiumToml);
    ASSERT_FALSE(americium.has_value());
    EXPECT_EQ(americium.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(americium.error().message.find("d3"), std::string::npos)
        << "message: " << americium.error().message;
}
