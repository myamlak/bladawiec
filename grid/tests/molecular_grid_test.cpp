// Molecular grid end to end: with the Becke partition the grid must
// integrate a sum of atomic densities over all of R^3, e.g. the
// prototypical H2 promolecular density (one frozen 1s Slater orbital
// per hydrogen, rho(r) = sum_a e^{-2 |r - R_a|}/pi - the normalized
// hydrogen 1s density, integrating to two electrons).

#include "h2_sto3g.hpp"
#include "qcx/grid/molecular_grid.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace qcx::grid {
namespace {

constexpr double kPi = 3.14159265358979323846;

TEST(MolecularGridTest, IntegratesPromolecularDensity) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());
    const auto& coordinates = molecule->CoordinatesBohr();

    const auto grid = MolecularGrid::Create(*molecule, 40, 50);
    ASSERT_TRUE(grid.has_value());

    double electrons = 0.0;

    for (std::size_t i = 0; i < grid->Size(); ++i)
    {
        const std::array<double, 3>& p = grid->Point(i);
        double density = 0.0;

        for (std::size_t a = 0; a < 2; ++a)
        {
            const double dx = p[0] - coordinates(a, 0);
            const double dy = p[1] - coordinates(a, 1);
            const double dz = p[2] - coordinates(a, 2);
            density += std::exp(-2.0 * std::sqrt(dx * dx + dy * dy + dz * dz)) / kPi;
        }

        electrons += grid->Weight(i) * density;
    }

    EXPECT_NEAR(electrons, 2.0, 1e-3);
}

TEST(MolecularGridTest, SingleAtomGridMatchesAtomicIntegral) {
    // For one atom the molecular grid is the atomic grid with partition
    // weight 1, so int e^{-|r - c|^2} d^3r = pi^{3/2} must come out
    // exactly as in the atomic-grid test.
    const auto molecule = qcx::testing::MakeHeAtom();
    ASSERT_TRUE(molecule.has_value());

    const auto grid = MolecularGrid::Create(*molecule, 40, 50);
    ASSERT_TRUE(grid.has_value());

    double sum = 0.0;

    for (std::size_t i = 0; i < grid->Size(); ++i)
    {
        const std::array<double, 3>& p = grid->Point(i);
        sum += grid->Weight(i) * std::exp(-(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]));
    }

    EXPECT_NEAR(sum, std::pow(kPi, 1.5), 1e-4);
}

TEST(MolecularGridTest, LayoutIsAtomMajorThenAngularFast) {
    const auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value());

    const auto grid = MolecularGrid::Create(*molecule, 10, 26);
    ASSERT_TRUE(grid.has_value());

    EXPECT_EQ(grid->AtomCount(), 2u);
    EXPECT_EQ(grid->Size(), 2u * 10u * 26u);
    // All points of atom 0 come before any of atom 1.
    for (std::size_t i = 0; i < std::size_t{10} * 26; ++i)
    {
        EXPECT_EQ(grid->AtomIndex(i), 0u);
    }

    for (std::size_t i = std::size_t{10} * 26; i < grid->Size(); ++i)
    {
        EXPECT_EQ(grid->AtomIndex(i), 1u);
    }
}

} // namespace
} // namespace qcx::grid
