// The dispersion module's coordinate-gradient acceptance: the D3 arm's
// analytic gradient against central finite differences of the energy entry,
// the translation sum, and the D4 arm's frozen-charge gradient against central
// finite differences of the same energy expression with its pair coefficients
// held fixed.
//
// The D3 finite-difference reference is EvaluateDispersion itself, not an
// external number: the energy and gradient entries must be the same model at
// the same parameters, so a divergence between them is exactly what this gate
// catches. The D4 reference is the pair list the module differentiates, summed
// with the reference geometry's coefficients - the tier's own definition. A
// difference of the full D4 energy would measure the charge-relaxed derivative
// instead, which is a different derivative and is refused by name. The
// kernel's own suite carries the gradient-versus-displacement and
// external-reference claims; this one carries the consumer's seam.

#include "qcx/dispersion/dispersion.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <excgrid/d4.hpp>
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

// The D4 arm is the frozen-charge derivative: the pair coefficients and the
// damping radius are held at the geometry's own values and only the
// interatomic distances move. The reference energy below is that expression
// reconstructed from the module's own pair list, so the finite difference
// differentiates exactly the formula the analytic gradient differentiates.
TEST(DispersionGradientTest, D4FrozenChargeGradientMatchesFiniteDifferenceOfTheFrozenEnergy) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto analytic = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_TRUE(analytic.has_value()) << analytic.error().message;
    ASSERT_EQ(analytic->gradient.size(), static_cast<Eigen::Index>(3 * molecule->AtomCount()));
    EXPECT_LT(analytic->energy, 0.0);

    // The gradient entry reports the energy entry's own number: one model.
    auto energy = qcx::dispersion::EvaluateDispersion(*molecule, *resolved);
    ASSERT_TRUE(energy.has_value()) << energy.error().message;
    EXPECT_EQ(analytic->energy, *energy);

    auto geometry = qcx::grid::ToExcgridGeometry(*molecule);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;

    const auto parameters = qcx::dispersion::ParameterMap(*resolved);
    const double s6 = parameters.at("s6");
    const double s8 = parameters.at("s8");
    const excgrid::D4Parameters kernel{s6, s8, parameters.at("a1"), parameters.at("a2")};

    // The module's own pairs, in the module's own order: the D4 kernel fills
    // the same list its energy sums.
    auto pairs = excgrid::D4TwoBodyPairs(*geometry, kernel);
    ASSERT_TRUE(pairs.has_value()) << "the D4 pair list was refused";

    const Eigen::VectorXd reference = CanonicalCoordinates(*molecule);

    const auto frozenEnergy = [&pairs, s6, s8](const Eigen::VectorXd& positions) {
        double sum = 0.0;

        for (const excgrid::D4TwoBodyPair& pair : *pairs)
        {
            const double dx = positions(static_cast<Eigen::Index>(3 * pair.first)) -
                              positions(static_cast<Eigen::Index>(3 * pair.second));
            const double dy = positions(static_cast<Eigen::Index>(3 * pair.first + 1)) -
                              positions(static_cast<Eigen::Index>(3 * pair.second + 1));
            const double dz = positions(static_cast<Eigen::Index>(3 * pair.first + 2)) -
                              positions(static_cast<Eigen::Index>(3 * pair.second + 2));
            const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
            const double r6 = std::pow(r, 6.0);
            const double r8 = r6 * r * r;
            const double radius6 = std::pow(pair.radius, 6.0);
            const double radius8 = radius6 * pair.radius * pair.radius;
            sum += s6 * pair.c6 / (r6 + radius6) + s8 * pair.c8 / (r8 + radius8);
        }

        return -sum;
    };

    // At the reference geometry the reconstructed expression is the energy
    // entry's number, bit for bit: the same operands in the same order.
    EXPECT_DOUBLE_EQ(frozenEnergy(reference), *energy);

    // h = 1e-6 Bohr: the expression is smooth at the separation scale here, so
    // the central difference's truncation error is far below the 1e-9 gate,
    // and the roundoff (|E| ~ 1e-3, h = 1e-6) stays below it too.
    constexpr double kStep = 1e-6;

    for (Eigen::Index coordinate = 0; coordinate < reference.size(); ++coordinate)
    {
        Eigen::VectorXd plus = reference;
        Eigen::VectorXd minus = reference;
        plus(coordinate) += kStep;
        minus(coordinate) -= kStep;

        const double finiteDifference = (frozenEnergy(plus) - frozenEnergy(minus)) / (2.0 * kStep);
        EXPECT_NEAR(analytic->gradient(coordinate), finiteDifference, 1e-9)
            << "coordinate " << coordinate;
    }
}

TEST(DispersionGradientTest, TheD4FrozenChargeGradientSumsToZeroOverTheAtoms) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto gradient = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_TRUE(gradient.has_value()) << gradient.error().message;

    // Every pair contributes equal and opposite vectors, so the sum over the
    // atoms is the derivative of a rigid translation and vanishes.
    Eigen::Vector3d total = Eigen::Vector3d::Zero();

    for (std::size_t atom = 0; atom < molecule->AtomCount(); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            total(static_cast<Eigen::Index>(direction)) +=
                gradient->gradient(static_cast<Eigen::Index>(3 * atom + direction));
        }
    }

    EXPECT_LT(total.cwiseAbs().maxCoeff(), 1e-10) << total.transpose();
}

TEST(DispersionGradientTest, TheD4FrozenChargeGradientIsUnchangedByRigidTranslation) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto reference = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    // A rigid translation moves every coordinate by the same vector: the
    // coefficients and every interatomic distance are unchanged, so the
    // gradient must be unchanged too.
    constexpr std::array<double, 3> kShift{1.7, -2.3, 4.1};
    Eigen::VectorXd shifted = coordinates;

    for (std::size_t atom = 0; atom < std::size(kWater); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            shifted(static_cast<Eigen::Index>(3 * atom + direction)) += kShift[direction];
        }
    }

    auto shiftedMolecule = MakeMolecule(kWater, shifted, 0, 1);
    ASSERT_TRUE(shiftedMolecule.has_value()) << shiftedMolecule.error().message;

    // The canonical order must be the same on both sides, or the two vectors
    // would be indexed by different atoms.
    ASSERT_EQ(shiftedMolecule->AtomCount(), molecule->AtomCount());

    for (std::size_t atom = 0; atom < molecule->AtomCount(); ++atom)
    {
        ASSERT_EQ(shiftedMolecule->Atoms()[atom].atomicNumber,
                  molecule->Atoms()[atom].atomicNumber);
    }

    auto translated = qcx::dispersion::EvaluateDispersionGradient(*shiftedMolecule, *resolved);
    ASSERT_TRUE(translated.has_value()) << translated.error().message;

    for (Eigen::Index entry = 0; entry < reference->gradient.size(); ++entry)
    {
        EXPECT_NEAR(reference->gradient(entry), translated->gradient(entry), 1e-12)
            << "entry " << entry;
    }
}

TEST(DispersionGradientTest, TheChargeRelaxedD4GradientIsRefusedByName) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto gradient = qcx::dispersion::EvaluateDispersionGradient(
        *molecule, *resolved, qcx::dispersion::DispersionGradientTier::kChargeRelaxed);
    ASSERT_FALSE(gradient.has_value());
    EXPECT_EQ(gradient.error().code, qcx::ErrorCode::kUnimplemented);
    const std::string message = gradient.error().message;
    EXPECT_NE(message.find("charge-relaxed"), std::string::npos) << message;
    EXPECT_NE(message.find("coordination numbers"), std::string::npos) << message;
    EXPECT_NE(message.find("EEQ"), std::string::npos) << message;
}

TEST(DispersionGradientTest, D4RefusesAChargedMoleculeOnTheGradientArm) {
    const Eigen::VectorXd coordinates = Flatten(kWater);
    auto molecule = MakeMolecule(kWater, coordinates, 1, 2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto resolved = qcx::dispersion::ResolveDispersion(
        PresetOptions(qcx::dispersion::DispersionForm::kD4, "b3lyp"), "hf");
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    auto gradient = qcx::dispersion::EvaluateDispersionGradient(*molecule, *resolved);
    ASSERT_FALSE(gradient.has_value());
    EXPECT_EQ(gradient.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(gradient.error().message.find("charge 0"), std::string::npos)
        << gradient.error().message;
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
