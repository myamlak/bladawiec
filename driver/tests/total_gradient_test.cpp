// The total analytic gradient (schema 46) started from a user's INPUT FILE:
// one .toml on disk, the way `qcx run <file.toml>` starts a run, and the block
// read back both from the structured record and from its JSON text.
//
// The acceptance is six claims. First, an rhf run that asks for the gradient
// records every term of the decomposition, and the terms sum to the total -
// the block cannot disagree with itself. Second, the block is ABSENT when the
// key was not written (the null-honesty rule: absent, never a zero-filled
// block). Third, a [dispersion] block adds the dispersion term and its
// energy, matching the record's own `dispersion` energy - the same
// EvaluateDispersionGradient call - and a run without one carries neither
// key inside the block. Fourth, every refusal the properties entry states is
// refused by name through the driver: a non-converged SCF, a non-stationary
// reference, and a D4 dispersion gradient (while a D4 run that asks for the
// energy alone still runs). Fifth, the run's [method] accuracy reaches the
// assembly: the two-electron derivative walk's screening counts in the block
// move with the preset, so a kLoose run's block carries the kLoose-derived
// figure rather than the kNormal default. Sixth, a restricted Kohn-Sham run
// records the exchange-correlation term beside the Hartree-Fock ones, its
// response names the xc contribution, and its analytic gradient agrees with a
// central difference of the run's own total energy - the gradient
// differentiates the energy the run reports.

#include "qcx/driver/run_driver.hpp"
#include "qcx/io/parse_input.hpp"
#include "qcx/io/result_json.hpp"
#include "qcx/io/run_input.hpp"
#include "qcx/io/validate_input.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

// One unique .toml per call: a row that reused a path would read the previous
// row's bytes if a write ever failed halfway (the ks_input_file_test rule).
std::filesystem::path TempInputPath(std::string_view tag) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("qcx-total-gradient-" + std::string(tag) + "-" + std::to_string(counter++) + ".toml");
}

// Writes \p toml to a fresh file and parses THAT, so the run's input is the
// bytes on disk, not the string this file holds.
qcx::Result<qcx::io::RunInput> ParseFile(const std::filesystem::path& path, std::string_view toml) {
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

    return qcx::io::ParseRunInputFile(path);
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

    const auto validation = qcx::io::ValidateInput(*input);

    if (!validation.IsValid())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "validation: " + validation.issues.front()});
    }

    return qcx::driver::RunDriverOutcome(*input);
}

// The fixture: hydrogen/STO-3G in Angstrom, with the [method] body a row
// chooses (so the refusal rows can name another method) and an optional
// trailing block. Everything except those two pieces is identical between two
// runs, so whatever separates their records is those two blocks.
std::string HydrogenToml(std::string_view methodBody, std::string_view trailing = {}) {
    return std::string(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]

[basis]
orbital = "sto-3g"

[method]
)") + std::string(methodBody) +
           R"(accuracy = "kNormal"
)" + std::string(trailing);
}

// Water/STO-3G in Angstrom, for the row that needs a reference the exact Fock
// rebuild puts outside the stationarity gate (see that row's comment).
std::string WaterToml(std::string_view methodBody, std::string_view trailing = {}) {
    return std::string(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [["O", 0.0, 0.0, 0.117], ["H", 0.0, 0.757, -0.469], ["H", 0.0, -0.757, -0.469]]

[basis]
orbital = "sto-3g"

[method]
)") + std::string(methodBody) +
           R"(accuracy = "kNormal"
)" + std::string(trailing);
}

// The same water in CANONICAL molecule order, at a caller-chosen geometry.
// Molecule::Create renumbers atoms by (Z, x, y, z), so the record's gradient
// comes back hydrogens first (y decides between them) and oxygen last - not
// necessarily in the order `atoms` was written. A finite-difference row that
// displaces a written coordinate and compares it with `gradient[3 * atom +
// direction]` needs the two orders to coincide, and this fixture is the file
// order that is already canonical. Symbols follow the order: H, H, O.
using WaterAngstromPositions = std::array<std::array<double, 3>, 3>;

std::string CanonicalWaterToml(const WaterAngstromPositions& positions,
                               std::string_view methodBody,
                               std::string_view trailing = {}) {
    static constexpr std::array<std::string_view, 3> kSymbols{"H", "H", "O"};

    std::ostringstream text;
    text << "[molecule]\ncharge = 0\nmultiplicity = 1\natoms = [";

    for (std::size_t atom = 0; atom < kSymbols.size(); ++atom)
    {
        // Full round-trip precision, so the coordinates on disk are the ones
        // the caller displaced.
        text << (atom == 0 ? "[\"" : ", [\"") << kSymbols[atom] << "\", " << std::setprecision(17)
             << positions[atom][0] << ", " << positions[atom][1] << ", " << positions[atom][2]
             << "]";
    }

    text << "]\n\n[basis]\norbital = \"sto-3g\"\n\n[method]\n"
         << methodBody << "accuracy = \"kNormal\"\n"
         << trailing;

    return text.str();
}

// Stretched H2 at 5.75 Angstrom: the one off-diagonal canonical quartet's
// density-weighted Schwarz bound sits between the kLoose and kNormal screen
// thresholds, so the two presets' derivative walks count it differently (the
// measured crossings are ~5.2 and ~6.3 Angstrom). The accuracy word is the
// only thing that differs between the two runs this fixture is used for.
std::string StretchedHydrogenToml(std::string_view accuracy) {
    return std::string(R"(
[molecule]
charge = 0
multiplicity = 1
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 5.75]]

[basis]
orbital = "sto-3g"

[method]
type = "rhf"
fock_builder = "direct"
)") + "accuracy = \"" +
           std::string(accuracy) + "\"\n\n" + R"([properties]
total_gradient = true
)";
}

constexpr std::string_view kRhfBody = R"(type = "rhf"
fock_builder = "direct"
)";

constexpr std::string_view kRequest = "[properties]\ntotal_gradient = true\n";

// The Kohn-Sham row's method: PBE on the direct builder. The builder is
// stated rather than left to the auto ladder for the reason the finite-
// difference row gives - a pure functional's ladder resolves to the RI-J
// approximation, whose fitted Coulomb energy is not the operator the gradient
// assembly differentiates.
constexpr std::string_view kKohnShamBody = R"(type = "rks"
functional = "pbe"
fock_builder = "direct"
)";

// The SCF the finite-difference row needs: the difference divides an SCF
// energy error by 2h, so the operating 1e-8 energy gate would put a ~2.6e-6
// floor under the comparison (h = 1e-3 Angstrom on the first side). At 1e-10
// the floor is 2.6e-8, orders below the bar. The pair is the kTight rung
// (1e-10/1e-8), not a private gate.
constexpr std::string_view kTightScf =
    "[scf]\nenergy_tolerance = 1e-10\ndensity_tolerance = 1e-8\n";

// The Kohn-Sham row's reference geometry: WaterToml's water, written in the
// canonical atom order (H at y = -0.757, H at y = +0.757, then O).
constexpr WaterAngstromPositions kCanonicalWaterAngstrom{
    {{0.0, -0.757, -0.469}, {0.0, 0.757, -0.469}, {0.0, 0.0, 0.117}}};

} // namespace

TEST(TotalGradientTest, InputFileRunRecordsEveryTermAndTheTotal) {
    // The plain operating defaults of the [scf] block: the emission rows are
    // what a user's file does, and measurement says a converged reference
    // from this SCF sits far inside the assembly's stationarity gate (H2 at
    // the default gates: max|2 F_ai| = 5.3e-16 against the 1e-6 gate).
    const auto path = TempInputPath("rhf");
    auto outcome = RunFile(path, HydrogenToml(kRhfBody, kRequest));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;
    ASSERT_TRUE(result.properties.has_value());
    ASSERT_TRUE(result.properties->totalGradient.has_value())
        << "the requested total_gradient block is absent from the record";

    const qcx::io::RunTotalGradient& total = *result.properties->totalGradient;
    const std::size_t expected = std::size_t{3} * 2;

    EXPECT_EQ(total.gradient.size(), expected);

    for (const std::vector<double>* term : {&total.terms.oneElectron,
                                            &total.terms.nuclearRepulsion,
                                            &total.terms.coulomb,
                                            &total.terms.exchange,
                                            &total.terms.response})
    {
        EXPECT_EQ(term->size(), expected);
    }

    // The dispersion term is absent, not zero-filled, and the energy follows
    // the same rule.
    EXPECT_FALSE(total.terms.dispersion.has_value());
    EXPECT_FALSE(total.energies.dispersion.has_value());

    // The total is the terms summed in the documented order, entry for entry.
    for (std::size_t i = 0; i < expected; ++i)
    {
        EXPECT_DOUBLE_EQ(total.gradient[i],
                         total.terms.oneElectron[i] + total.terms.nuclearRepulsion[i] +
                             total.terms.coulomb[i] + total.terms.exchange[i] +
                             total.terms.response[i]);
    }

    // The block's energies are the record's own: the reference SCF energy and
    // the run's total (no correction on this run).
    EXPECT_NEAR(total.energies.electronic, result.electronicEnergyHartree, 1e-12);
    EXPECT_NEAR(total.energies.total, result.totalEnergyHartree, 1e-12);
    EXPECT_NEAR(
        total.energies.total, total.energies.electronic + total.energies.nuclearRepulsion, 1e-12);
    EXPECT_LE(total.stationarityResidual, 1e-6);
    EXPECT_LE(total.energyResidualHartree, 1e-8);

    // The dimensions the assembly reports, and the response it solved.
    EXPECT_EQ(total.sizes.atomCount, 2u);
    EXPECT_EQ(total.sizes.numBasisFunctions, 2u);
    EXPECT_EQ(total.sizes.numOccupied, 1u);
    EXPECT_EQ(total.sizes.numVirtual, 1u);
    EXPECT_EQ(total.response.solves, expected);
    EXPECT_GT(total.response.iterations, 0u);

    // The honesty key: the response's wired contributions, in order.
    EXPECT_EQ(total.rightHandSideTerms,
              (std::vector<std::string>{"one-electron", "coulomb", "exchange"}));

    // Every walk that must visit something did. `quartets` counts the
    // quartets differentiated and `screened_quartets` the ones discarded, so
    // the two are disjoint visits and no order between them holds.
    EXPECT_GT(total.counts.oneElectron.overlapTuples, 0u);
    EXPECT_GT(total.counts.oneElectron.kineticTuples, 0u);
    EXPECT_GT(total.counts.twoElectron.quartets, 0u);

    // Translation invariance: the gradient's Cartesian components sum to zero
    // over the atoms (a rigid translation moves no energy). H2 lies on z, so
    // every x and y component is zero by symmetry as well.
    for (std::size_t d = 0; d < 3; ++d)
    {
        EXPECT_NEAR(total.gradient[d] + total.gradient[3 + d], 0.0, 1e-6) << "direction " << d;
    }

    EXPECT_NEAR(total.gradient[0], 0.0, 1e-10);
    EXPECT_NEAR(total.gradient[1], 0.0, 1e-10);
    EXPECT_NEAR(total.gradient[3], 0.0, 1e-10);
    EXPECT_NEAR(total.gradient[4], 0.0, 1e-10);

    // The JSON round-trip: the document carries the same block, with the same
    // numbers, and the absent dispersion pair stays absent.
    const nlohmann::json document = nlohmann::json::parse(outcome->json);
    ASSERT_TRUE(document.contains("total_gradient"));
    const nlohmann::json& block = document.at("total_gradient");

    EXPECT_EQ(block.at("gradient").size(), expected);
    EXPECT_FALSE(block.at("terms").contains("dispersion"));
    EXPECT_FALSE(block.at("energies").contains("dispersion_hartree"));
    EXPECT_EQ(block.at("sizes").at("atom_count").get<std::size_t>(), 2u);
    EXPECT_EQ(block.at("sizes").at("num_basis_functions").get<std::size_t>(), 2u);
    EXPECT_EQ(block.at("sizes").at("num_occupied").get<std::size_t>(), 1u);
    EXPECT_EQ(block.at("sizes").at("num_virtual").get<std::size_t>(), 1u);
    EXPECT_EQ(block.at("response").at("solves").get<std::size_t>(), expected);
    EXPECT_EQ(block.at("right_hand_side_terms").get<std::vector<std::string>>(),
              total.rightHandSideTerms);
    EXPECT_DOUBLE_EQ(block.at("gradient").at(0).get<double>(), total.gradient[0]);
    EXPECT_DOUBLE_EQ(block.at("stationarity_residual").get<double>(), total.stationarityResidual);
    EXPECT_DOUBLE_EQ(block.at("energy_residual_hartree").get<double>(),
                     total.energyResidualHartree);

    for (const char* key : {"one_electron", "nuclear_repulsion", "coulomb", "exchange", "response"})
    {
        EXPECT_TRUE(block.at("terms").contains(key)) << key;
        EXPECT_EQ(block.at("terms").at(key).size(), expected) << key;
    }
}

TEST(TotalGradientTest, TheBlockIsAbsentWhenTheKeyWasNotWritten) {
    // The null-honesty rule: an unrequested analysis is absent from the
    // document, never a zero-filled block a consumer could read as a computed
    // zero gradient.
    const auto path = TempInputPath("no-request");
    auto outcome = RunFile(path, HydrogenToml(kRhfBody));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    ASSERT_TRUE(outcome->result.properties.has_value());
    EXPECT_FALSE(outcome->result.properties->totalGradient.has_value());
    EXPECT_EQ(outcome->json.find("total_gradient"), std::string::npos);
}

TEST(TotalGradientTest, TheMethodAccuracyPresetScreensTheGradientWalk) {
    // The [method] accuracy is the assembly's ONE screening preset, and the
    // driver must hand it to TotalGradientOptions::eri.accuracy, which the
    // two-electron derivative walk reads. The walk's own screened count is
    // therefore the block's evidence of the preset the gradient ran at: the
    // same fixture at the same geometry must report a kLoose-derived split on
    // the kLoose run and a kNormal-derived split on the kNormal one. If the
    // driver dropped the preset (the kNormal default), the kLoose block would
    // carry kNormal's counts and this row would fail.
    const auto loosePath = TempInputPath("accuracy-kloose");
    auto loose = RunFile(loosePath, StretchedHydrogenToml("kLoose"));
    ASSERT_TRUE(loose.has_value()) << loose.error().message;

    const auto normalPath = TempInputPath("accuracy-knormal");
    auto normal = RunFile(normalPath, StretchedHydrogenToml("kNormal"));
    ASSERT_TRUE(normal.has_value()) << normal.error().message;

    ASSERT_TRUE(loose->result.properties.has_value());
    ASSERT_TRUE(normal->result.properties.has_value());
    ASSERT_TRUE(loose->result.properties->totalGradient.has_value());
    ASSERT_TRUE(normal->result.properties->totalGradient.has_value());

    const qcx::io::RunTotalGradientTwoElectronCounts& looseCounts =
        loose->result.properties->totalGradient->counts.twoElectron;
    const qcx::io::RunTotalGradientTwoElectronCounts& normalCounts =
        normal->result.properties->totalGradient->counts.twoElectron;

    // Same walk, different screen: both runs visit the same canonical
    // quartets, and the preset decides which side of the screen each falls on.
    EXPECT_EQ(looseCounts.quartets + looseCounts.screenedQuartets,
              normalCounts.quartets + normalCounts.screenedQuartets);

    // The kLoose block carries the kLoose-derived figure - one quartet
    // screened - and not the kNormal one (none screened).
    EXPECT_EQ(looseCounts.quartets, 5u);
    EXPECT_EQ(looseCounts.screenedQuartets, 1u);
    EXPECT_EQ(normalCounts.quartets, 6u);
    EXPECT_EQ(normalCounts.screenedQuartets, 0u);
    EXPECT_GT(looseCounts.screenedQuartets, normalCounts.screenedQuartets);
    EXPECT_LT(looseCounts.quartets, normalCounts.quartets);

    // The document carries the same kLoose-derived figures.
    const nlohmann::json document = nlohmann::json::parse(loose->json);
    const nlohmann::json& counted = document.at("total_gradient").at("counts").at("two_electron");

    EXPECT_EQ(counted.at("quartets").get<std::size_t>(), 5u);
    EXPECT_EQ(counted.at("screened_quartets").get<std::size_t>(), 1u);
}

TEST(TotalGradientTest, AD3RequestAddsTheTermAndTheEnergyTheRecordReports) {
    const auto path = TempInputPath("d3");
    auto outcome = RunFile(path,
                           HydrogenToml(kRhfBody,
                                        "[dispersion]\nform = \"d3\"\n"
                                        "[properties]\ntotal_gradient = true\n"));
    ASSERT_TRUE(outcome.has_value()) << outcome.error().message;

    const qcx::io::RunResult& result = outcome->result;
    ASSERT_TRUE(result.properties.has_value());
    ASSERT_TRUE(result.properties->totalGradient.has_value());
    ASSERT_TRUE(result.dispersion.has_value());

    const qcx::io::RunTotalGradient& total = *result.properties->totalGradient;
    const std::size_t expected = std::size_t{3} * 2;

    ASSERT_TRUE(total.terms.dispersion.has_value());
    ASSERT_TRUE(total.energies.dispersion.has_value());
    EXPECT_EQ(total.terms.dispersion->size(), expected);

    // The same correction energy the top-level `dispersion` block reports.
    // The two come from different kernels' calls - the block from
    // EvaluateDispersion, the gradient term from EvaluateDispersionGradient -
    // so they agree to the 1e-14 the properties suite measured, not bit for
    // bit.
    EXPECT_NEAR(*total.energies.dispersion, result.dispersion->energyHartree, 1e-14);
    EXPECT_NEAR(total.energies.total,
                total.energies.electronic + total.energies.nuclearRepulsion +
                    *total.energies.dispersion,
                1e-12);

    // The term is a real contribution, not a zero vector.
    double magnitude = 0.0;

    for (const double value : *total.terms.dispersion)
    {
        magnitude += std::abs(value);
    }

    EXPECT_GT(magnitude, 0.0);
    EXPECT_DOUBLE_EQ(total.gradient[0],
                     total.terms.oneElectron[0] + total.terms.nuclearRepulsion[0] +
                         total.terms.coulomb[0] + total.terms.exchange[0] +
                         total.terms.response[0] + (*total.terms.dispersion)[0]);

    // The JSON carries both members in this shape.
    const nlohmann::json document = nlohmann::json::parse(outcome->json);
    const nlohmann::json& block = document.at("total_gradient");
    EXPECT_TRUE(block.at("terms").contains("dispersion"));
    EXPECT_TRUE(block.at("energies").contains("dispersion_hartree"));
    EXPECT_DOUBLE_EQ(block.at("energies").at("dispersion_hartree").get<double>(),
                     *total.energies.dispersion);
}

TEST(TotalGradientTest, ANonConvergedScfIsRefusedByName) {
    // A last iterate is not the energy's stationary point, so its gradient
    // would differentiate an energy the density is not the minimum of. The
    // refusal names the key and the SCF's own report.
    // One iteration cannot meet the operating default gate (the kNormal
    // rung), so the budget alone produces the non-converged report; the
    // tolerances stay unset and no gate literal is written here.
    const auto path = TempInputPath("non-converged");
    auto outcome = RunFile(path,
                           HydrogenToml(kRhfBody,
                                        "[properties]\ntotal_gradient = true\n"
                                        "[scf]\nmax_iterations = 1\n"));
    ASSERT_FALSE(outcome.has_value()) << "a non-converged reference was differentiated";
    EXPECT_EQ(outcome.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(outcome.error().message.find("properties.total_gradient:"), std::string::npos)
        << outcome.error().message;
    EXPECT_NE(outcome.error().message.find("converged = false"), std::string::npos)
        << outcome.error().message;
}

TEST(TotalGradientTest, ANonStationaryReferenceIsRefusedByName) {
    // The stationarity gate: the assembly rebuilds the exact Roothaan Fock and
    // refuses a reference whose occupied-virtual block is not zero. Reaching
    // the gate through the direct builder is not possible on these systems -
    // the SCF's own density gate bounds the rebuilt residual orders below 1e-6
    // (measured: 5.3e-16 on H2 at the operating defaults; the kLoose rung, the
    // loosest, stayed below the gate on water, benzene, C2, F2, N2 and
    // stretched H2). The route that does reach it is an approximate Fock
    // builder: the
    // RI-J lane converges the density against a FITTED Fock, so the exact
    // rebuilt matrix is not the one the orbitals diagonalized and the gate sees
    // a real occupied-virtual block (measured: 3.1e-4 on water at the operating
    // defaults). The refusal names the residual and the request.
    const auto path = TempInputPath("non-stationary");
    auto outcome = RunFile(path,
                           WaterToml(R"(type = "rhf"
fock_builder = "ri_j_link"
)",
                                     kRequest));
    ASSERT_FALSE(outcome.has_value()) << "a non-stationary reference was differentiated";
    EXPECT_EQ(outcome.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(outcome.error().message.find("properties.total_gradient:"), std::string::npos)
        << outcome.error().message;
    EXPECT_NE(outcome.error().message.find("not stationary"), std::string::npos)
        << outcome.error().message;
    EXPECT_NE(outcome.error().message.find("max|2 F_ai|"), std::string::npos)
        << outcome.error().message;
}

TEST(TotalGradientTest, AD4GradientIsRefusedWhileTheEnergyAloneStillRuns) {
    // The dispersion module's gradient entry has no D4 tier; the refusal is
    // that entry's own, propagated with the request's key in front of it. A
    // D4 run that asks for the energy alone is schema 41's supported request
    // and must still run.
    const auto refusedPath = TempInputPath("d4-gradient");
    auto refused = RunFile(refusedPath,
                           HydrogenToml(kRhfBody,
                                        "[dispersion]\nform = \"d4\"\n"
                                        "[properties]\ntotal_gradient = true\n"));
    ASSERT_FALSE(refused.has_value()) << "a D4 coordinate gradient was emitted";
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(refused.error().message.find("properties.total_gradient:"), std::string::npos)
        << refused.error().message;
    EXPECT_NE(refused.error().message.find("d4"), std::string::npos) << refused.error().message;
    EXPECT_NE(refused.error().message.find("coordinate-gradient"), std::string::npos)
        << refused.error().message;

    const auto energyPath = TempInputPath("d4-energy");
    auto energy = RunFile(energyPath, HydrogenToml(kRhfBody, "[dispersion]\nform = \"d4\"\n"));
    ASSERT_TRUE(energy.has_value()) << energy.error().message;
    ASSERT_TRUE(energy->result.dispersion.has_value());
    EXPECT_EQ(energy->result.dispersion->form, "d4");
    EXPECT_FALSE(energy->result.properties->totalGradient.has_value());
}

TEST(TotalGradientTest, TheKeyIsRefusedOnTheUnrestrictedMethods) {
    // The validator's rule, through the input file the way a user writes it:
    // the closed-shell assembly's key is legal on the two closed-shell lanes
    // (rhf, and rks through the density functional's kernel term) and refused
    // by name on the unrestricted ones, whose references the assembly does
    // not describe. The rks arm that runs is the Kohn-Sham row below.
    const auto uhfPath = TempInputPath("uhf");
    auto uhf = RunFile(uhfPath,
                       HydrogenToml(R"(type = "uhf"
fock_builder = "direct"
)",
                                    kRequest));
    ASSERT_FALSE(uhf.has_value());
    EXPECT_EQ(uhf.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(uhf.error().message.find("properties.total_gradient: uhf"), std::string::npos)
        << uhf.error().message;

    const auto uksPath = TempInputPath("uks");
    auto uks = RunFile(uksPath,
                       HydrogenToml(R"(type = "uks"
functional = "pbe"
)",
                                    kRequest));
    ASSERT_FALSE(uks.has_value());
    EXPECT_EQ(uks.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(uks.error().message.find("properties.total_gradient: uks"), std::string::npos)
        << uks.error().message;
}

TEST(TotalGradientTest, TheKohnShamGradientMatchesTheEnergyItDifferentiates) {
    // The restricted Kohn-Sham lane through a file: the record carries the
    // exchange-correlation term beside the Hartree-Fock ones, the response's
    // right-hand side names the xc contribution, and the analytic gradient
    // agrees with a central difference of the run's own total energy. The
    // difference is the claim no term list can make on its own - one missing
    // or mis-scaled term leaves every individual number plausible and only
    // the sum wrong.
    //
    // Two settings are load-bearing. `fock_builder = "direct"`: a pure
    // functional's auto ladder resolves to the RI-J builder, and the fitted
    // Coulomb energy is not the exact operator the assembly differentiates -
    // the energy gate refuses such a reference (measured 6.9e-5 on this
    // system, the same refusal the rhf lane has always produced). The
    // tightened [scf] tolerances: see kTightScf's comment.
    const auto gradientPath = TempInputPath("rks-gradient");
    auto gradientRun = RunFile(gradientPath,
                               CanonicalWaterToml(kCanonicalWaterAngstrom,
                                                  kKohnShamBody,
                                                  std::string(kRequest) + std::string(kTightScf)));
    ASSERT_TRUE(gradientRun.has_value()) << gradientRun.error().message;

    const qcx::io::RunResult& result = gradientRun->result;
    ASSERT_TRUE(result.properties.has_value());
    ASSERT_TRUE(result.properties->totalGradient.has_value());
    const qcx::io::RunTotalGradient& total = *result.properties->totalGradient;
    const std::size_t expected = std::size_t{3} * 3;

    ASSERT_EQ(total.gradient.size(), expected);
    ASSERT_TRUE(total.terms.xc.has_value()) << "the rks record carries no xc term";
    EXPECT_EQ(total.terms.xc->size(), expected);

    for (const std::vector<double>* term : {&total.terms.oneElectron,
                                            &total.terms.nuclearRepulsion,
                                            &total.terms.coulomb,
                                            &total.terms.exchange,
                                            &*total.terms.xc,
                                            &total.terms.response})
    {
        EXPECT_EQ(term->size(), expected);
    }

    EXPECT_FALSE(total.terms.dispersion.has_value());
    EXPECT_FALSE(total.energies.dispersion.has_value());

    // The total is the terms summed in the documented order, the xc term
    // between exchange and response.
    for (std::size_t i = 0; i < expected; ++i)
    {
        EXPECT_DOUBLE_EQ(total.gradient[i],
                         total.terms.oneElectron[i] + total.terms.nuclearRepulsion[i] +
                             total.terms.coulomb[i] + total.terms.exchange[i] +
                             (*total.terms.xc)[i] + total.terms.response[i]);
    }

    // The xc skeleton is a real contribution, not a zero-filled vector. PBE
    // carries no exact exchange (c_HF = 0), so the exchange term is present
    // and exactly zero - present because the term list is the assembly's
    // shape, zero because the functional scales the exchange by nothing.
    double xcMagnitude = 0.0;

    for (const double value : *total.terms.xc)
    {
        xcMagnitude += std::abs(value);
    }

    EXPECT_GT(xcMagnitude, 1e-3);

    for (const double value : total.terms.exchange)
    {
        EXPECT_DOUBLE_EQ(value, 0.0);
    }

    // The block's energies are the record's own: the reconstruction adds E_xc
    // to the derivative walks' terms and reproduces the reference energy at
    // the reference geometry.
    EXPECT_NEAR(total.energies.electronic, result.electronicEnergyHartree, 1e-12);
    EXPECT_NEAR(total.energies.total, result.totalEnergyHartree, 1e-12);
    EXPECT_NEAR(
        total.energies.total, total.energies.electronic + total.energies.nuclearRepulsion, 1e-12);
    EXPECT_LE(total.stationarityResidual, 1e-6);
    EXPECT_LE(total.energyResidualHartree, 1e-8);

    EXPECT_EQ(total.sizes.atomCount, 3u);
    EXPECT_EQ(total.sizes.numBasisFunctions, 7u);
    EXPECT_EQ(total.sizes.numOccupied, 5u);
    EXPECT_EQ(total.sizes.numVirtual, 2u);
    EXPECT_EQ(total.response.solves, expected);
    EXPECT_GT(total.response.iterations, 0u);

    // The honesty key: the response read the xc contribution as well.
    EXPECT_EQ(total.rightHandSideTerms,
              (std::vector<std::string>{"one-electron", "coulomb", "exchange", "xc"}));

    EXPECT_GT(total.counts.oneElectron.overlapTuples, 0u);
    EXPECT_GT(total.counts.twoElectron.quartets, 0u);

    // The JSON carries the same block, the xc term included.
    const nlohmann::json document = nlohmann::json::parse(gradientRun->json);
    const nlohmann::json& block = document.at("total_gradient");

    EXPECT_TRUE(block.at("terms").contains("xc"));
    EXPECT_EQ(block.at("terms").at("xc").size(), expected);
    EXPECT_FALSE(block.at("terms").contains("dispersion"));
    EXPECT_EQ(block.at("right_hand_side_terms").get<std::vector<std::string>>(),
              total.rightHandSideTerms);
    EXPECT_DOUBLE_EQ(block.at("energies").at("total_hartree").get<double>(), total.energies.total);
    EXPECT_DOUBLE_EQ(block.at("stationarity_residual").get<double>(), total.stationarityResidual);

    // The finite difference. First the base geometry's energy alone, which
    // must be the gradient's own: the two runs then read one potential-energy
    // surface, so the comparison below differentiates the energy the record
    // reports.
    const auto basePath = TempInputPath("rks-energy-base");
    auto base =
        RunFile(basePath, CanonicalWaterToml(kCanonicalWaterAngstrom, kKohnShamBody, kTightScf));
    ASSERT_TRUE(base.has_value()) << base.error().message;
    EXPECT_NEAR(base->result.totalEnergyHartree, total.energies.total, 1e-12);

    // One step, and a bar four times the measured truncation error: the
    // observed max |analytic - central difference| is 5.2e-7 Hartree/Bohr at
    // 1e-3 Angstrom, and 2.1e-6 at 2e-3 - the h^2 scaling of a central
    // difference, not SCF noise (whose floor at the tightened tolerances is
    // ~2.6e-8).
    constexpr double kStep = 1e-3;
    constexpr double kAgreement = 2e-6;
    double worst = 0.0;

    for (std::size_t atom = 0; atom < kCanonicalWaterAngstrom.size(); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            double energies[2] = {};

            for (std::size_t side = 0; side < 2; ++side)
            {
                WaterAngstromPositions displaced = kCanonicalWaterAngstrom;
                displaced[atom][direction] += side == 0 ? kStep : -kStep;

                const auto path = TempInputPath("rks-energy-displaced");
                auto energy =
                    RunFile(path, CanonicalWaterToml(displaced, kKohnShamBody, kTightScf));
                ASSERT_TRUE(energy.has_value()) << energy.error().message;
                energies[side] = energy->result.totalEnergyHartree;
            }

            const double difference =
                (energies[0] - energies[1]) / (2.0 * kStep * qcx::molecule::kAngstromToBohr);
            const std::size_t entry = 3 * atom + direction;
            worst = std::max(worst, std::abs(total.gradient[entry] - difference));

            EXPECT_NEAR(total.gradient[entry], difference, kAgreement)
                << "atom " << atom << ", direction " << direction;
        }
    }

    // The measured agreement is the number whoever reports this run quotes.
    std::cout << "[kohn-sham gradient] h = " << kStep
              << " Angstrom worst |analytic - finite| = " << worst << " Hartree/Bohr\n";
}
