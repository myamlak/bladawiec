// The dispersion module's acceptance: the request resolution's refusals and
// the two evaluation arms against excgrid's own numbers.
//
// The D3 arm is checked against the published dftd3 V3.2 Rev 1 reference
// energy for the water fixture (-6.34e-6 Ha, transcribed with its sources in
// excgrid's D3_DATA_PROVENANCE.md) as well as against equality with the
// library: the reference pins the physics, the equality pins the wiring. The
// D4 arm carries excgrid's published-reference ratio for caffeine - the
// model's two-body energy is 0.997913 of the published number with the
// shipped neutral baseline - so a run that reached the kernel with different
// parameters or a different geometry fails here.

#include "qcx/dispersion/dispersion.hpp"
#include "qcx/dispersion/options.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <excgrid/d4.hpp>
#include <excgrid/dispersion.hpp>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

struct FixtureAtom {
    const char* symbol;
    int atomicNumber;
    double x;
    double y;
    double z;
};

qcx::Result<qcx::molecule::Molecule> MakeMolecule(std::span<const FixtureAtom> atoms,
                                                  int charge,
                                                  int multiplicity) {
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
        (*coordinates)(i, 0) = atoms[i].x;
        (*coordinates)(i, 1) = atoms[i].y;
        (*coordinates)(i, 2) = atoms[i].z;
    }

    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::move(elements), std::move(*coordinates), charge, multiplicity);
}

// excgrid's D3 fixture (external/excgrid/tests/d3_test.cpp): water, in Bohr.
constexpr FixtureAtom kWater[] = {
    {"O", 8, 0.0, 0.0, 0.0},
    {"H", 1, 1.4, 0.0, 0.9},
    {"H", 1, -1.4, 0.0, 0.9},
};

// excgrid's D4 fixture (external/excgrid/tests/d4_test.cpp), in Angstrom:
// the caffeine geometry the published two-body energy is quoted for.
struct CaffeineAtom {
    const char* symbol;
    int atomicNumber;
    double x;
    double y;
    double z;
};

constexpr CaffeineAtom kCaffeine[] = {
    {"C", 6, -3.2668063145, -1.0612383311, 0.0019792586},
    {"N", 7, -2.2766003898, -0.0195172936, -0.0013208218},
    {"C", 6, -0.9134590858, -0.2019878522, 0.0036747135},
    {"C", 6, -0.3724701006, 1.0762202467, 0.0017569102},
    {"N", 7, -1.3526586265, 2.0066755645, -0.0051933750},
    {"C", 6, -2.4700895686, 1.3111433173, -0.0065911261},
    {"H", 1, -3.4434528448, 1.7598589152, -0.0115392456},
    {"N", 7, 0.9701620055, 1.2833365802, -0.0008793828},
    {"C", 6, 1.8257144377, 0.2028321683, -0.0007168653},
    {"O", 8, 3.0278876281, 0.3474912012, -0.0051812078},
    {"N", 7, 1.2515027114, -1.0636950051, -0.0008922263},
    {"C", 6, -0.1065018202, -1.3842039491, 0.0029991689},
    {"O", 8, -0.5015980556, -2.5366534217, 0.0044099955},
    {"C", 6, 2.2018357820, -2.1638123093, -0.0037185988},
    {"H", 1, 1.6323675079, -3.0900622488, 0.0097444448},
    {"H", 1, 2.8244315108, -2.1119787186, -0.8964193499},
    {"H", 1, 2.8450781508, -2.0977859157, 0.8731480371},
    {"C", 6, 1.5216365707, 2.6211569486, 0.0093869454},
    {"H", 1, 0.7306184352, 3.3088549484, -0.2771144995},
    {"H", 1, 1.8886888508, 2.8714188674, 1.0060887389},
    {"H", 1, 2.3551115434, 2.6687817985, -0.6902763920},
    {"H", 1, -2.7510404767, -2.0202634640, -0.0176576390},
    {"H", 1, -3.9109098848, -0.9753935259, -0.8744768432},
    {"H", 1, -3.8816811438, -0.9984445054, 0.9013764258},
};

constexpr double kCaffeinePublishedEnergy = -0.04677824450285543;

qcx::Result<qcx::molecule::Molecule> MakeCaffeine() {
    std::vector<FixtureAtom> atoms;
    atoms.reserve(std::size(kCaffeine));

    for (const CaffeineAtom& atom : kCaffeine)
    {
        atoms.push_back({atom.symbol,
                         atom.atomicNumber,
                         atom.x * qcx::molecule::kAngstromToBohr,
                         atom.y * qcx::molecule::kAngstromToBohr,
                         atom.z * qcx::molecule::kAngstromToBohr});
    }

    return MakeMolecule(atoms, 0, 1);
}

qcx::dispersion::DispersionOptions PresetOptions(qcx::dispersion::DispersionForm form,
                                                 std::string name) {
    qcx::dispersion::DispersionOptions options;
    options.form = form;
    options.preset = std::move(name);
    return options;
}

qcx::dispersion::DispersionOptions MethodDefaultOptions(qcx::dispersion::DispersionForm form) {
    qcx::dispersion::DispersionOptions options;
    options.form = form;
    return options;
}

} // namespace

TEST(DispersionTest, AD3PresetResolvesToTheLibraryValues) {
    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    EXPECT_EQ(resolved->source, qcx::dispersion::DispersionParameterSource::kPreset);
    ASSERT_TRUE(resolved->parameterSet.has_value());
    EXPECT_EQ(*resolved->parameterSet, "b3lyp");

    const auto expected = excgrid::D3Preset("b3lyp");
    ASSERT_TRUE(expected.has_value());
    const auto* values = std::get_if<qcx::dispersion::D3ParameterValues>(&resolved->parameters);
    ASSERT_NE(values, nullptr);
    EXPECT_DOUBLE_EQ(values->s6, expected->s6);
    EXPECT_DOUBLE_EQ(values->s8, expected->s8);
    EXPECT_DOUBLE_EQ(values->rs6, expected->rs6);
    EXPECT_DOUBLE_EQ(values->alpha6, expected->alpha6);
    EXPECT_DOUBLE_EQ(values->rs8, expected->rs8);
    EXPECT_DOUBLE_EQ(values->alpha8, expected->alpha8);
}

TEST(DispersionTest, TheMethodFamilyIsTheNoParameterDefault) {
    auto resolved = qcx::dispersion::ResolveDispersion(
        MethodDefaultOptions(qcx::dispersion::DispersionForm::kD3), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    EXPECT_EQ(resolved->source, qcx::dispersion::DispersionParameterSource::kMethod);
    ASSERT_TRUE(resolved->parameterSet.has_value());
    EXPECT_EQ(*resolved->parameterSet, "hf");

    const auto expected = excgrid::D3Preset("hf");
    ASSERT_TRUE(expected.has_value());
    const auto* values = std::get_if<qcx::dispersion::D3ParameterValues>(&resolved->parameters);
    ASSERT_NE(values, nullptr);
    EXPECT_DOUBLE_EQ(values->s6, expected->s6);
    EXPECT_DOUBLE_EQ(values->s8, expected->s8);
}

TEST(DispersionTest, AnUnknownMethodFamilyIsRefusedWithThePresetRemedy) {
    auto resolved = qcx::dispersion::ResolveDispersion(
        MethodDefaultOptions(qcx::dispersion::DispersionForm::kD3), "svwn");
    ASSERT_FALSE(resolved.has_value());
    const std::string message = resolved.error().message;
    EXPECT_NE(message.find("svwn"), std::string::npos);
    EXPECT_NE(message.find("d3"), std::string::npos);
    EXPECT_NE(message.find("dispersion.preset"), std::string::npos);
}

TEST(DispersionTest, AnExplicitSetIsCarriedVerbatim) {
    qcx::dispersion::DispersionOptions options;
    options.form = qcx::dispersion::DispersionForm::kD3;
    options.d3.s6 = 1.0;
    options.d3.s8 = 2.0;
    options.d3.rs6 = 3.0;
    options.d3.alpha6 = 4.0;
    options.d3.rs8 = 5.0;
    options.d3.alpha8 = 6.0;

    auto resolved = qcx::dispersion::ResolveDispersion(options, "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    EXPECT_EQ(resolved->source, qcx::dispersion::DispersionParameterSource::kExplicit);
    EXPECT_FALSE(resolved->parameterSet.has_value());

    const auto* values = std::get_if<qcx::dispersion::D3ParameterValues>(&resolved->parameters);
    ASSERT_NE(values, nullptr);
    EXPECT_DOUBLE_EQ(values->s6, 1.0);
    EXPECT_DOUBLE_EQ(values->s8, 2.0);
    EXPECT_DOUBLE_EQ(values->rs6, 3.0);
    EXPECT_DOUBLE_EQ(values->alpha6, 4.0);
    EXPECT_DOUBLE_EQ(values->rs8, 5.0);
    EXPECT_DOUBLE_EQ(values->alpha8, 6.0);

    const auto map = qcx::dispersion::ParameterMap(*resolved);
    EXPECT_EQ(map.size(), 6u);
    EXPECT_DOUBLE_EQ(map.at("alpha8"), 6.0);
}

TEST(DispersionTest, AnIncompleteExplicitSetNamesEveryMissingKey) {
    qcx::dispersion::DispersionOptions options;
    options.form = qcx::dispersion::DispersionForm::kD3;
    options.d3.s6 = 0.5;

    auto resolved = qcx::dispersion::ResolveDispersion(options, "hf");
    ASSERT_FALSE(resolved.has_value());
    const std::string message = resolved.error().message;
    EXPECT_NE(message.find("dispersion.rs8"), std::string::npos);
    EXPECT_NE(message.find("dispersion.alpha8"), std::string::npos);
}

TEST(DispersionTest, APresetAndANumericKeyAreOneSource) {
    auto options = PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp");
    options.d3.s6 = 0.5;

    auto resolved = qcx::dispersion::ResolveDispersion(options, "hf");
    ASSERT_FALSE(resolved.has_value());
    const std::string message = resolved.error().message;
    EXPECT_NE(message.find("dispersion.preset"), std::string::npos);
    EXPECT_NE(message.find("dispersion.s6"), std::string::npos);
}

TEST(DispersionTest, AKeyOfTheOtherFormIsRefusedByName) {
    auto options = PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp");
    options.d4.a1 = 0.4;

    auto resolved = qcx::dispersion::ResolveDispersion(options, "hf");
    ASSERT_FALSE(resolved.has_value());
    const std::string message = resolved.error().message;
    EXPECT_NE(message.find("dispersion.a1"), std::string::npos);
    EXPECT_NE(message.find("d3"), std::string::npos);
}

TEST(DispersionTest, AKeyWithoutAFormIsRefused) {
    qcx::dispersion::DispersionOptions options;
    options.preset = "b3lyp";

    auto resolved = qcx::dispersion::ResolveDispersion(options, "hf");
    ASSERT_FALSE(resolved.has_value());
    EXPECT_NE(resolved.error().message.find("dispersion.form"), std::string::npos);
}

TEST(DispersionTest, D3EnergyEqualsTheLibraryOnTheWaterFixture) {
    auto molecule = MakeMolecule(kWater, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_TRUE(energy.has_value()) << energy.error().message;

    auto geometry = qcx::grid::ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    const auto parameters = excgrid::D3Preset("b3lyp");
    ASSERT_TRUE(parameters.has_value());
    auto direct = excgrid::GrimmeD3Energy(*geometry, *parameters);
    ASSERT_TRUE(direct.has_value());

    EXPECT_NEAR(*energy, *direct, 1e-12);
    // dftd3 V3.2 Rev 1 on the same geometry prints -0.00000634 au.
    EXPECT_NEAR(*energy, -6.34e-6, 5e-8);
    EXPECT_LT(*energy, 0.0);
}

TEST(DispersionTest, D4CaffeineMatchesThePublishedReferenceRatio) {
    auto molecule = MakeCaffeine();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_TRUE(energy.has_value()) << energy.error().message;

    // The kernel's own bar (external/excgrid/tests/d4_test.cpp): the shipped
    // two-body energy is 0.997913 of the published number.
    EXPECT_NEAR(*energy / kCaffeinePublishedEnergy, 0.997913, 1e-6);
}

TEST(DispersionTest, D4RefusesAChargedMolecule) {
    auto molecule = MakeMolecule(kWater, 1, 2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_FALSE(energy.has_value());
    EXPECT_EQ(energy.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(energy.error().message.find("charge"), std::string::npos);
}

TEST(DispersionTest, D4RefusesAnElementAboveItsTable) {
    constexpr FixtureAtom kFrancium[] = {{"Fr", 87, 0.0, 0.0, 0.0}};
    auto molecule = MakeMolecule(kFrancium, 0, 2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_FALSE(energy.has_value());
    EXPECT_EQ(energy.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(energy.error().message.find("87"), std::string::npos);
}

TEST(DispersionTest, D3RefusesAnElementAboveItsTable) {
    constexpr FixtureAtom kAmericium[] = {{"Am", 95, 0.0, 0.0, 0.0}};
    auto molecule = MakeMolecule(kAmericium, 0, 2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_FALSE(energy.has_value());
    EXPECT_EQ(energy.error().code, qcx::ErrorCode::kUnimplemented);
}
