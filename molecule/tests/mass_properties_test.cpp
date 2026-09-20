// Mass properties: center of mass, inertia, and the parallel nuclear repulsion.
#include "qcx/molecule/elements.hpp"
#include "qcx/molecule/mass_properties.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace qcx::molecule {
namespace {

qcx::Result<Molecule> MakeWater() {
    // O at the origin, H at (+-1.43, 1.108, 0) Bohr.
    std::vector<std::vector<double>> rows = {
        {0.0, 0.0, 0.0}, {1.43, 1.108, 0.0}, {-1.43, 1.108, 0.0}};
    auto coords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coords.has_value())
    {
        return std::unexpected(coords.error());
    }

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coords)(i, d) = rows[i][d];
        }
    }

    return Molecule::Create(
        {Atom{"O", 8, 0.0}, Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}}, std::move(*coords), 0, 1);
}

// Independent serial reference for the parallel reduction.
double SerialRepulsion(const Molecule& molecule) {
    double energy = 0.0;
    const auto& coords = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();

    for (std::size_t i = 0; i < molecule.AtomCount(); ++i)
    {
        for (std::size_t j = i + 1; j < molecule.AtomCount(); ++j)
        {
            const double dx = coords(i, 0) - coords(j, 0);
            const double dy = coords(i, 1) - coords(j, 1);
            const double dz = coords(i, 2) - coords(j, 2);
            energy += static_cast<double>(atoms[i].atomicNumber * atoms[j].atomicNumber) /
                      std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    return energy;
}

TEST(MassPropertiesTest, WaterCenterOfMassAndTotalMass) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value());
    const auto properties = ComputeMassProperties(*water);
    const double mH = FindElement(1)->mostAbundantIsotopeMass;
    const double mO = FindElement(8)->mostAbundantIsotopeMass;
    EXPECT_NEAR(properties.totalMass, mO + 2.0 * mH, 1e-12);
    EXPECT_NEAR(properties.centerOfMass[0], 0.0, 1e-12);
    EXPECT_NEAR(properties.centerOfMass[1], 2.0 * mH * 1.108 / (mO + 2.0 * mH), 1e-9);
    EXPECT_NEAR(properties.centerOfMass[2], 0.0, 1e-12);
    EXPECT_FALSE(properties.isLinear);
    // Planar molecule: the out-of-plane axis is the unique largest moment.
    EXPECT_NEAR(std::abs(properties.principalAxes(2, 0)), 1.0, 1e-9);
}

TEST(MassPropertiesTest, PlanarMoleculeInertiaTraceRelation) {
    // For a planar molecule (xy plane): Izz = Ixx + Iyy, exactly.
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value());
    const auto properties = ComputeMassProperties(*water);
    EXPECT_NEAR(properties.inertiaTensor(2, 2),
                properties.inertiaTensor(0, 0) + properties.inertiaTensor(1, 1),
                1e-9);
    EXPECT_NEAR(properties.inertiaTensor(0, 2), 0.0, 1e-9);
    EXPECT_NEAR(properties.inertiaTensor(1, 2), 0.0, 1e-9);
}

TEST(MassPropertiesTest, LinearMoleculesDetected) {
    // CO2 along x: O-C-O at +-2.2 Bohr.
    std::vector<std::vector<double>> rows = {{0.0, 0.0, 0.0}, {2.2, 0.0, 0.0}, {-2.2, 0.0, 0.0}};
    auto coords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});
    ASSERT_TRUE(coords.has_value());

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coords)(i, d) = rows[i][d];
        }
    }

    auto co2 = Molecule::Create(
        {Atom{"C", 6, 0.0}, Atom{"O", 8, 0.0}, Atom{"O", 8, 0.0}}, std::move(*coords), 0, 1);
    ASSERT_TRUE(co2.has_value());
    const auto properties = ComputeMassProperties(*co2);
    EXPECT_TRUE(properties.isLinear);
    EXPECT_NEAR(properties.principalMoments[2], 0.0, 1e-10);
    // The two non-zero moments are degenerate (symmetric top), so their
    // eigenvectors are basis-ambiguous - only the moment values are asserted.
}

TEST(MassPropertiesTest, SingleAtomIsNotLinear) {
    // A single atom has no axis: all moments vanish, which would trivially
    // satisfy the linearity threshold without the n > 1 guard.
    auto coords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});
    ASSERT_TRUE(coords.has_value());
    auto atom = Molecule::Create({Atom{"He", 2, 0.0}}, std::move(*coords), 0, 1);
    ASSERT_TRUE(atom.has_value());
    const auto properties = ComputeMassProperties(*atom);
    EXPECT_FALSE(properties.isLinear);
    EXPECT_DOUBLE_EQ(properties.totalMass, FindElement(2)->mostAbundantIsotopeMass);
    EXPECT_NEAR(properties.principalMoments[0], 0.0, 1e-12);
}

TEST(MassPropertiesTest, NuclearRepulsionHydrogenDimer) {
    // Two H at +-0.7 Bohr: E = 1 / 1.4.
    std::vector<std::vector<double>> rows = {{0.7, 0.0, 0.0}, {-0.7, 0.0, 0.0}};
    auto coords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});
    ASSERT_TRUE(coords.has_value());

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coords)(i, d) = rows[i][d];
        }
    }

    auto h2 = Molecule::Create({Atom{"H", 1, 0.0}, Atom{"H", 1, 0.0}}, std::move(*coords), 0, 1);
    ASSERT_TRUE(h2.has_value());
    EXPECT_NEAR(NuclearRepulsionEnergy(*h2), 1.0 / 1.4, 1e-12);
}

TEST(MassPropertiesTest, NuclearRepulsionMatchesSerialReference) {
    auto water = MakeWater();
    ASSERT_TRUE(water.has_value());
    const double parallel = NuclearRepulsionEnergy(*water);
    const double serial = SerialRepulsion(*water);
    EXPECT_NEAR(parallel, serial, 1e-12 * std::abs(serial));
}

} // namespace
} // namespace qcx::molecule
