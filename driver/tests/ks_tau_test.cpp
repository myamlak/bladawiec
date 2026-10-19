// B3's end-to-end tau gate, with B4's input-file leg folded in: one .toml on
// disk asks for tau_x on water/STO-3G, the way `qcx run <file.toml>` starts a
// run, and the record that comes back must say the run converged, resolved the
// functional it was given, and carried the xc grid block.
//
// Why the file and not ParseRunInput on a literal: the user's leg is
// ParseRunInputFile, a path on disk, and the record that comes back. A run
// wired only on the string path is a run no input file can start.
//
// Dependency: the driver reaches tau_x through XcGridEngine, so this gate is
// red until B1's tau plumbing lands. The failure it must not be is a silent
// one: with the five-argument adapter still in place the engine reads tau = 0,
// converges on the 13/12 defect and records "tau_x" anyway. The second case is
// what makes the name's reach into the physics visible: a tau-reading
// functional and its tau-free parent cannot land on the same number unless tau
// was dropped. The exact-identity pins on tau itself live in
// grid/tests/tau_reference_test.cpp.
//
// A capped build (CI's kMaxEngineL below 4) cannot resolve the universal-J
// fitting set's oxygen f and g shells, so the runs are skipped there rather
// than reported as failures - the same guard the neighbouring Kohn-Sham input
// tests carry.

#include "qcx/driver/run_driver.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/io/validate_input.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

namespace {

using qcx::io::MethodType;
using qcx::io::ParseRunInputFile;
using qcx::io::RunInput;
using qcx::io::ValidateInput;

// One unique .toml per call: a reused path could read a previous row's bytes if
// a write failed halfway, and the run under test would be a run of the wrong
// file.
std::filesystem::path TempInputPath(std::string_view tag) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("qcx-ks-tau-" + std::string(tag) + "-" + std::to_string(counter++) + ".toml");
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

// The primary gate: an RKS/STO-3G water input requesting tau_x converges, and
// the record carries the xc block beside the resolved functional.
TEST(KsTauTest, WaterSto3gRksTauXConvergesAndCarriesTheXcBlock) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "the universal-J aux f and g shells exceed this build's kMaxEngineL "
                        "(CI lmax=2)";
    }

    auto outcome = RunFile(TempInputPath("tau-x"), WaterToml(R"(type = "rks"
functional = "tau_x"
)"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;

    // The physics the file asked for reached the run, by the record's own
    // words: the method word and the registry's spelling of the functional.
    EXPECT_EQ(result.method, MethodType::kRks);
    ASSERT_TRUE(result.functional.has_value());
    EXPECT_EQ(*result.functional, "tau_x");

    // The grid the run built. The file wrote no [grid] block, so these are the
    // engine's defaults, disclosed rather than implied.
    ASSERT_TRUE(result.xcGrid.has_value());
    EXPECT_EQ(result.xcGrid->radialPoints, 75u);
    EXPECT_EQ(result.xcGrid->angularPoints, 302u);
    EXPECT_DOUBLE_EQ(result.xcGrid->alpha, 0.5);
    EXPECT_EQ(result.xcGrid->radialExponent, 2u);
    EXPECT_DOUBLE_EQ(result.xcGrid->trimWeight, 1e-15);
    EXPECT_EQ(result.xcGrid->blockTarget, 1024u);

    EXPECT_TRUE(result.converged);
    EXPECT_GT(result.iterations, 0);
    EXPECT_TRUE(std::isfinite(result.totalEnergyHartree));
}

// The name reaches the physics, not only the record. tau_x is the same
// spin-scaled Slater exchange multiplied by an enhancement that is 1 at the
// uniform gas and 13/12 when tau is zero, so a run whose engine dropped tau
// lands on the tau-free functional's regime; the two numbers below separate
// whenever tau arrived. The bar is far above SCF noise: the enhancement's
// defect at the uniform-gas limit is 1/12 of exchange, orders larger than
// 1e-6 Hartree on this fixture.
TEST(KsTauTest, TheFunctionalNameReachesThePhysics) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "the universal-J aux f and g shells exceed this build's kMaxEngineL "
                        "(CI lmax=2)";
    }

    auto tau = RunFile(TempInputPath("tau-x-physics"), WaterToml(R"(type = "rks"
functional = "tau_x"
)"));
    ASSERT_TRUE(tau.has_value()) << tau.error().message;

    auto slater = RunFile(TempInputPath("slater-physics"), WaterToml(R"(type = "rks"
functional = "slater"
)"));
    ASSERT_TRUE(slater.has_value()) << slater.error().message;

    ASSERT_TRUE(tau->result.converged);
    ASSERT_TRUE(slater->result.converged);
    ASSERT_TRUE(tau->result.functional.has_value());
    ASSERT_TRUE(slater->result.functional.has_value());
    EXPECT_EQ(*tau->result.functional, "tau_x");
    EXPECT_EQ(*slater->result.functional, "slater");

    EXPECT_GT(std::abs(tau->result.totalEnergyHartree - slater->result.totalEnergyHartree), 1e-6)
        << "tau_x and its tau-free parent landed on the same energy: the tau never arrived";
}
