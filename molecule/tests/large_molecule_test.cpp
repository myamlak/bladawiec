// Large-molecule scaling and symmetry: C60 (Ih) and a 2400-atom zigzag
// nanotube exercise the O(N^2) kernels at a scale where the parallel
// structure is measurable.
#include "large_molecules.hpp"
#include "qcx/molecule/connectivity.hpp"
#include "qcx/molecule/mass_properties.hpp"

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

namespace qcx::molecule {
namespace {

// Independent serial reference for the parallel nuclear repulsion.
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

// Independent serial reference for the parallel mass-property reductions.
struct SerialMassProperties {
    double totalMass;
    Eigen::Vector3d centerOfMass;
    Eigen::Matrix3d inertiaTensor;
};

SerialMassProperties SerialMassPropertiesReference(const Molecule& molecule) {
    const auto& coords = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();
    double totalMass = 0.0;
    Eigen::Vector3d massWeighted = Eigen::Vector3d::Zero();

    for (std::size_t i = 0; i < molecule.AtomCount(); ++i)
    {
        totalMass += atoms[i].isotopicMass;
        massWeighted +=
            atoms[i].isotopicMass * Eigen::Vector3d(coords(i, 0), coords(i, 1), coords(i, 2));
    }

    const Eigen::Vector3d centerOfMass = massWeighted / totalMass;
    Eigen::Matrix3d inertia = Eigen::Matrix3d::Zero();

    for (std::size_t i = 0; i < molecule.AtomCount(); ++i)
    {
        const double m = atoms[i].isotopicMass;
        const double x = coords(i, 0) - centerOfMass[0];
        const double y = coords(i, 1) - centerOfMass[1];
        const double z = coords(i, 2) - centerOfMass[2];
        inertia(0, 0) += m * (y * y + z * z);
        inertia(1, 1) += m * (x * x + z * z);
        inertia(2, 2) += m * (x * x + y * y);
        inertia(0, 1) += -m * x * y;
        inertia(0, 2) += -m * x * z;
        inertia(1, 2) += -m * y * z;
    }

    inertia(1, 0) = inertia(0, 1);
    inertia(2, 0) = inertia(0, 2);
    inertia(2, 1) = inertia(1, 2);
    return {totalMass, centerOfMass, inertia};
}

double MillisecondsSince(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}

TEST(LargeMoleculeTest, BuckminsterfullereneIsSphericalTop) {
    auto c60 = testing::MakeBuckminsterfullerene();
    ASSERT_TRUE(c60.has_value()) << c60.error().message;
    ASSERT_EQ(c60->AtomCount(), 60u);

    // Ih symmetry -> the inertia tensor is isotropic (degenerate top); the
    // Day 7 detector must survive exactly this degenerate case.
    const auto properties = ComputeMassProperties(*c60);
    EXPECT_FALSE(properties.isLinear);
    EXPECT_NEAR(properties.principalMoments[0],
                properties.principalMoments[1],
                1e-3 * properties.principalMoments[0]);
    EXPECT_NEAR(properties.principalMoments[1],
                properties.principalMoments[2],
                1e-3 * properties.principalMoments[0]);
}

TEST(LargeMoleculeTest, ZigzagNanotubeScalingAndSymmetry) {
    // (10,0) tube, 60 translational cells: 2400 atoms, ~2.9M atom pairs.
    auto tube = testing::MakeZigzagNanotube(10, 60);
    ASSERT_TRUE(tube.has_value()) << tube.error().message;
    EXPECT_EQ(tube->AtomCount(), 2400u);

    auto start = std::chrono::steady_clock::now();
    const auto connectivity = BuildConnectivity(*tube);
    const double connectivityMs = MillisecondsSince(start);
    EXPECT_EQ(boost::num_edges(connectivity.graph), 3590u);
    // Interior atoms have degree 3; the 2n = 20 atoms of the two open end
    // rings have degree 2 (one missing axial bond each).
    std::size_t degree2 = 0;
    std::size_t degree3 = 0;

    for (std::size_t i = 0; i < tube->AtomCount(); ++i)
    {
        const std::size_t degree = connectivity.csr.offsets[i + 1] - connectivity.csr.offsets[i];
        EXPECT_LE(degree, 3u);

        if (degree == 2)
        {
            ++degree2;
        } else
        {
            ++degree3;
        }
    }

    EXPECT_EQ(degree2, 20u);
    EXPECT_EQ(degree3, 2380u);

    start = std::chrono::steady_clock::now();
    const double repulsion = NuclearRepulsionEnergy(*tube);
    const double repulsionMs = MillisecondsSince(start);

    start = std::chrono::steady_clock::now();
    const auto properties = ComputeMassProperties(*tube);
    const double massPropsMs = MillisecondsSince(start);

    // Dnh tube: symmetric top with two equal principal moments, axis along z.
    EXPECT_FALSE(properties.isLinear);
    EXPECT_NEAR(properties.principalMoments[0],
                properties.principalMoments[1],
                1e-3 * properties.principalMoments[0]);
    EXPECT_NEAR(std::abs(properties.principalAxes(2, 2)), 1.0, 1e-6);

    // Parallel reductions equal the independent serial references.
    const auto serialProps = SerialMassPropertiesReference(*tube);
    EXPECT_NEAR(properties.totalMass, serialProps.totalMass, 1e-12 * serialProps.totalMass);
    EXPECT_NEAR((properties.centerOfMass - serialProps.centerOfMass).norm(), 0.0, 1e-9);
    EXPECT_NEAR((properties.inertiaTensor - serialProps.inertiaTensor).norm(),
                0.0,
                1e-9 * serialProps.inertiaTensor.norm());
    EXPECT_NEAR(repulsion, SerialRepulsion(*tube), 1e-9 * std::abs(SerialRepulsion(*tube)));

    std::cout << "[large-molecule] 2400 atoms: connectivity " << connectivityMs << " ms, repulsion "
              << repulsionMs << " ms, mass properties " << massPropsMs << " ms\n";
    RecordProperty("connectivity_ms", connectivityMs);
    RecordProperty("repulsion_ms", repulsionMs);
    RecordProperty("mass_properties_ms", massPropsMs);
}

TEST(LargeMoleculeTest, SmallNanotubeRejectedBoundaryCheck) {
    // A tiny tube keeps the boundary accounting honest: (8,0) x 2 cells = 64 atoms.
    auto tube = testing::MakeZigzagNanotube(8, 2);
    ASSERT_TRUE(tube.has_value()) << tube.error().message;
    const auto connectivity = BuildConnectivity(*tube);
    // 2n = 16 boundary atoms with degree 2; the rest 3; edges = 3*64/2 - 8 = 88.
    EXPECT_EQ(boost::num_edges(connectivity.graph), 88u);
}

} // namespace
} // namespace qcx::molecule
