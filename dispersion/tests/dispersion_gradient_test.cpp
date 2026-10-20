// The dispersion module's coordinate-gradient acceptance: the D3 arm's
// analytic gradient against central finite differences of the energy entry,
// the translation sum, and the D4 arm's refusal by name.
//
// The finite-difference reference is EvaluateDispersion itself, not an
// external number: the energy and gradient entries must be the same model at
// the same parameters, so a divergence between them is exactly what this gate
// catches. The kernel's own suite carries the gradient-versus-displacement
// and external-reference claims; this one carries the consumer's seam.

#include "qcx/dispersion/dispersion.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace {

struct FixtureAtom {
    const char* symbol;
    int atomicNumber;
    double x;
    double y;
    double z;
};

// excgrid's D3 fixture (external/excgrid/tests/d3_test.cpp): water, in Bohr.
constexpr FixtureAtom kWater[] = {
    {"O", 8, 0.0, 0.0, 0.0},
    {"H", 1, 1.4, 0.0, 0.9},
    {"H", 1, -1.4, 0.0, 0.9},
};

Eigen::VectorXd Flatten(std::span<const FixtureAtom> atoms) {
    Eigen::VectorXd coordinates(static_cast<Eigen::Index>(atoms.size() * 3));

    for (std::size_t atom = 0; atom < atoms.size(); ++atom)
    {
        coordinates(static_cast<Eigen::Index>(3 * atom)) = atoms[atom].x;
        coordinates(static_cast<Eigen::Index>(3 * atom + 1)) = atoms[atom].y;
        coordinates(static_cast<Eigen::Index>(3 * atom + 2)) = atoms[atom].z;
    }

    return coordinates;
}

qcx::Result<qcx::molecule::Molecule> MakeMolecule(std::span<const FixtureAtom> atoms,
                                                  const Eigen::VectorXd& coordinates,
                                                  int charge,
                                                  int multiplicity) {
    auto positions =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atoms.size(), 3});

    if (!positions.has_value())
    {
        return std::unexpected(positions.error());
    }

    std::vector<qcx::molecule::Atom> elements;
    elements.reserve(atoms.size());

    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        elements.push_back({atoms[i].symbol, atoms[i].atomicNumber, 0.0});
        (*positions)(i, 0) = coordinates(static_cast<Eigen::Index>(3 * i));
        (*positions)(i, 1) = coordinates(static_cast<Eigen::Index>(3 * i + 1));
        (*positions)(i, 2) = coordinates(static_cast<Eigen::Index>(3 * i + 2));
    }

    positions->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::move(elements), std::move(*positions), charge, multiplicity);
}

// The molecule's atoms and coordinates in the order it reports them.
// Molecule::Create renumbers atoms canonically (by Z, then x, y, z), so the
// fixture's input order is not the order a gradient is indexed in.
std::vector<FixtureAtom> CanonicalAtoms(const qcx::molecule::Molecule& molecule) {
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinates =
        molecule.CoordinatesBohr();
    std::vector<FixtureAtom> atoms;
    atoms.reserve(molecule.AtomCount());

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        atoms.push_back({molecule.Atoms()[atom].symbol.c_str(),
                         molecule.Atoms()[atom].atomicNumber,
                         coordinates(atom, 0),
                         coordinates(atom, 1),
                         coordinates(atom, 2)});
    }

    return atoms;
}

Eigen::VectorXd CanonicalCoordinates(const qcx::molecule::Molecule& molecule) {
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinates =
        molecule.CoordinatesBohr();
    Eigen::VectorXd flattened(static_cast<Eigen::Index>(molecule.AtomCount() * 3));

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            flattened(static_cast<Eigen::Index>(3 * atom + direction)) =
                coordinates(atom, direction);
        }
    }

    return flattened;
}

qcx::dispersion::DispersionOptions PresetOptions(qcx::dispersion::DispersionForm form,
                                                 std::string name) {
    qcx::dispersion::DispersionOptions options;
    options.form = form;
    options.preset = std::move(name);
    return options;
}

} // namespace

TEST(DispersionGradientTest, D3GradientMatchesFiniteDifferenceOfTheEnergyEntry) {
    auto molecule = MakeMolecule(kWater, Flatten(kWater), 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto analytic = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    ASSERT_EQ(analytic->gradient.size(), static_cast<Eigen::Index>(3 * molecule->AtomCount()));
    EXPECT_LT(analytic->energy, 0.0);

    // The gradient is indexed by the molecule's atoms, so the finite
    // difference runs over the molecule's own coordinates; the fixture's
    // input order was renumbered by Create.
    const std::vector<FixtureAtom> canonical = CanonicalAtoms(*molecule);
    const Eigen::VectorXd coordinates = CanonicalCoordinates(*molecule);

    // h = 1e-4 Bohr: the central difference's truncation error at h = 1e-3
    // reaches 5.6e-5 on this geometry, above the gate; it is quadratic in h,
    // while the roundoff stays far below 1e-6.
    constexpr double kStep = 1e-4;

    for (Eigen::Index coordinate = 0; coordinate < coordinates.size(); ++coordinate)
    {
        Eigen::VectorXd plus = coordinates;
        Eigen::VectorXd minus = coordinates;
        plus(coordinate) += kStep;
        minus(coordinate) -= kStep;

        auto plusMolecule = MakeMolecule(canonical, plus, 0, 1);
        ASSERT_TRUE(plusMolecule.has_value()) << plusMolecule.error().message;
        auto minusMolecule = MakeMolecule(canonical, minus, 0, 1);
        ASSERT_TRUE(minusMolecule.has_value()) << minusMolecule.error().message;

        // The displaced frame must still be canonical: if Create renumbered
        // it, the analytic slot and the displaced row would be different
        // atoms and the comparison would be meaningless.
        const auto row = static_cast<std::size_t>(coordinate) / 3;
        const auto column = static_cast<std::size_t>(coordinate) % 3;
        ASSERT_EQ(plusMolecule->CoordinatesBohr()(row, column), plus(coordinate));
        ASSERT_EQ(minusMolecule->CoordinatesBohr()(row, column), minus(coordinate));

        auto plusEnergy = qcx::dispersion::EvaluateDispersion(*plusMolecule, *resolved);
        ASSERT_TRUE(plusEnergy.has_value()) << plusEnergy.error().message;
        auto minusEnergy = qcx::dispersion::EvaluateDispersion(*minusMolecule, *resolved);
        ASSERT_TRUE(minusEnergy.has_value()) << minusEnergy.error().message;

        const double finiteDifference = (*plusEnergy - *minusEnergy) / (2.0 * kStep);
        EXPECT_NEAR(analytic->gradient(coordinate), finiteDifference, 1e-6)
            << "coordinate " << coordinate;
    }
}

TEST(DispersionGradientTest, TheGradientEntryReportsTheEnergyEntrysOwnNumber) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_TRUE(energy.has_value()) << energy.error().message;
    auto gradient = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    EXPECT_DOUBLE_EQ(gradient->energy, *energy);
}

TEST(DispersionGradientTest, TheAnalyticGradientSumsToZeroOverTheAtoms) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto gradient = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    // The correction depends on interatomic distances alone, so the sum of
    // the per-atom derivatives is the derivative of a translation and
    // vanishes.
    Eigen::Vector3d total = Eigen::Vector3d::Zero();

    for (std::size_t atom = 0; atom < std::size(kWater); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            total(static_cast<Eigen::Index>(direction)) +=
                gradient->gradient(static_cast<Eigen::Index>(3 * atom + direction));
        }
    }

    EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-10) << total.transpose();
}

TEST(DispersionGradientTest, D4IsRefusedByNameAsAMissingGradientTier) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto gradient = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_FALSE(gradient.has_value());
    EXPECT_EQ(gradient.error().code, qcx::ErrorCode::kUnimplemented);
    const std::string message = gradient.error().message;
    EXPECT_NE(message.find("d4"), std::string::npos);
    EXPECT_NE(message.find("gradient"), std::string::npos);
    EXPECT_NE(message.find("d3"), std::string::npos);
}

TEST(DispersionGradientTest, D3RefusesAnElementAboveItsTableOnTheGradientArm) {
    constexpr FixtureAtom kAmericium[] = {{"Am", 95, 0.0, 0.0, 0.0}};
    const Eigen::VectorXd coordinates = Flatten(kAmericium);
    auto molecule = MakeMolecule(kAmericium, coordinates, 0, 2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD3, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto gradient = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_FALSE(gradient.has_value());
    EXPECT_EQ(gradient.error().code, qcx::ErrorCode::kUnimplemented);
}
