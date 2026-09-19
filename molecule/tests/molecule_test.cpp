// Molecule construction: validation, canonical ordering, and deep copy.
#include "qcx/molecule/molecule.hpp"

#include <gtest/gtest.h>
#include <type_traits>
#include <vector>

namespace qcx::molecule {
namespace {

qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> MakeCoords(
    std::vector<std::vector<double>> rows) {
    auto coords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({rows.size(), 3});
    EXPECT_TRUE(coords.has_value());

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coords)(i, d) = rows[i][d];
        }
    }

    return std::move(*coords);
}

std::vector<Atom> MakeWaterAtoms() {
    return {Atom{"O", 8, 0.0}, Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}};
}

// Water in the xy plane: O at the origin, H at (+-1.43, 1.108, 0) Bohr.
qcx::Result<Molecule> MakeWater() {
    return Molecule::Create(MakeWaterAtoms(),
                            MakeCoords({{0.0, 0.0, 0.0}, {1.43, 1.108, 0.0}, {-1.43, 1.108, 0.0}}),
                            0,
                            1);
}

TEST(MoleculeTest, WaterCreatesAndFillsDefaultMasses) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value()) << water.error().message;
    EXPECT_EQ(water->AtomCount(), 3u);
    EXPECT_EQ(water->Charge(), 0);
    EXPECT_EQ(water->Multiplicity(), 1);
    EXPECT_EQ(water->ElectronCount(), 10);
    // Canonical order is Z-ascending: H, H, O.
    EXPECT_EQ(water->Atoms()[0].atomicNumber, 1);
    EXPECT_EQ(water->Atoms()[2].atomicNumber, 8);
    // Missing isotopic masses fall back to the most-abundant isotope.
    EXPECT_DOUBLE_EQ(water->Atoms()[0].isotopicMass, 1.0078250321);
    EXPECT_NEAR(water->Atoms()[2].isotopicMass, 15.9949146, 1e-6);
}

TEST(MoleculeTest, CanonicalOrderingByZThenPosition) {
    // Input order scrambles Z and position; canonical order is H(x=0), H(x=2), O.
    auto mol = Molecule::Create({Atom{"O", 8, 0.0}, Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}},
                                MakeCoords({{1.0, 0.0, 0.0}, {2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}),
                                0,
                                1);
    ASSERT_TRUE(mol.has_value()) << mol.error().message;
    ASSERT_EQ(mol->AtomCount(), 3u);
    EXPECT_EQ(mol->Atoms()[0].atomicNumber, 1);
    EXPECT_DOUBLE_EQ(mol->CoordinatesBohr()(0, 0), 0.0); // H at x=0 moved with its row
    EXPECT_EQ(mol->Atoms()[1].atomicNumber, 1);
    EXPECT_DOUBLE_EQ(mol->CoordinatesBohr()(1, 0), 2.0);
    EXPECT_EQ(mol->Atoms()[2].atomicNumber, 8);
    EXPECT_DOUBLE_EQ(mol->CoordinatesBohr()(2, 0), 1.0);
}

TEST(MoleculeTest, ChargeBudgetAllowsRemovingAllElectrons) {
    auto mol = Molecule::Create(
        MakeWaterAtoms(), MakeCoords({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}}), 10, 1);
    ASSERT_TRUE(mol.has_value()) << mol.error().message;
    EXPECT_EQ(mol->ElectronCount(), 0);
}

TEST(MoleculeTest, RejectsEmptyAtomList) {
    // Any coordinate tensor works: the empty-atom check fires first.
    auto mol = Molecule::Create({}, MakeCoords({{0.0, 0.0, 0.0}}), 0, 1);
    ASSERT_FALSE(mol.has_value());
    EXPECT_EQ(mol.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(MoleculeTest, RejectsWrongCoordinateShape) {
    auto mol = Molecule::Create(MakeWaterAtoms(), MakeCoords({{0.0, 0.0, 0.0}}), 0, 1);
    ASSERT_FALSE(mol.has_value());
    EXPECT_EQ(mol.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(MoleculeTest, RejectsUnknownAtomicNumber) {
    auto mol = Molecule::Create({Atom{"Xx", 200, 0.0}}, MakeCoords({{0.0, 0.0, 0.0}}), 0, 1);
    ASSERT_FALSE(mol.has_value());
    EXPECT_EQ(mol.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(MoleculeTest, RejectsSymbolMismatch) {
    auto mol = Molecule::Create({Atom{"H", 8, 0.0}}, MakeCoords({{0.0, 0.0, 0.0}}), 0, 1);
    ASSERT_FALSE(mol.has_value());
    EXPECT_EQ(mol.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(MoleculeTest, RejectsInvalidMultiplicityAndCharge) {
    auto multiplicity = Molecule::Create(
        MakeWaterAtoms(), MakeCoords({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}}), 0, 0);
    ASSERT_FALSE(multiplicity.has_value());
    EXPECT_EQ(multiplicity.error().code, qcx::ErrorCode::kInvalidArgument);

    auto charge = Molecule::Create(MakeWaterAtoms(),
                                   MakeCoords({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {-1.0, 0.0, 0.0}}),
                                   11,
                                   1); // sum Z is 10
    ASSERT_FALSE(charge.has_value());
    EXPECT_EQ(charge.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(MoleculeTest, RejectsCoincidentAtoms) {
    auto mol = Molecule::Create({Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}},
                                MakeCoords({{0.5, 0.0, 0.0}, {0.5, 0.0, 0.0}}),
                                0,
                                1);
    ASSERT_FALSE(mol.has_value());
    EXPECT_EQ(mol.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(MoleculeTest, CloneProducesEqualIndependentCopy) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value());
    auto clone = water->Clone();
    ASSERT_TRUE(clone.has_value()) << clone.error().message;
    EXPECT_EQ(clone->AtomCount(), water->AtomCount());
    EXPECT_EQ(clone->Charge(), water->Charge());
    EXPECT_EQ(clone->Multiplicity(), water->Multiplicity());

    for (std::size_t i = 0; i < water->AtomCount(); ++i)
    {
        EXPECT_EQ(clone->Atoms()[i].symbol, water->Atoms()[i].symbol);
        EXPECT_DOUBLE_EQ(clone->Atoms()[i].isotopicMass, water->Atoms()[i].isotopicMass);

        for (int d = 0; d < 3; ++d)
        {
            EXPECT_DOUBLE_EQ(clone->CoordinatesBohr()(i, d), water->CoordinatesBohr()(i, d));
        }
    }

    auto cloneOfClone = clone->Clone();
    ASSERT_TRUE(cloneOfClone.has_value());
    EXPECT_DOUBLE_EQ(cloneOfClone->CoordinatesBohr()(0, 0), water->CoordinatesBohr()(0, 0));
}

TEST(MoleculeTest, MoleculeIsMoveOnly) {
    static_assert(!std::is_copy_constructible_v<Molecule>);
    static_assert(!std::is_copy_assignable_v<Molecule>);
    static_assert(std::is_move_constructible_v<Molecule>);
    static_assert(std::is_move_assignable_v<Molecule>);
}

} // namespace
} // namespace qcx::molecule
