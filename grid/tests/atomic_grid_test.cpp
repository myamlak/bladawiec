// Atomic product grid: radial x angular about a center must integrate
// radially symmetric functions over R^3:
//     int e^{-|r - c|^2} d^3r = pi^{3/2} ~ 5.5683
// with the error shrinking as either quadrature refines.

#include "qcx/grid/atomic_grid.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace qcx::grid {
namespace {

constexpr double kPi = 3.14159265358979323846;

double IntegrateGaussian(const AtomicGrid& grid) {
    const std::array<double, 3>& c = grid.Center();
    double sum = 0.0;

    for (std::size_t i = 0; i < grid.Size(); ++i)
    {
        const std::array<double, 3>& p = grid.Point(i);
        const double dx = p[0] - c[0];
        const double dy = p[1] - c[1];
        const double dz = p[2] - c[2];
        sum += grid.Weight(i) * std::exp(-(dx * dx + dy * dy + dz * dz));
    }

    return sum;
}

TEST(AtomicGridTest, IntegratesUnitGaussian) {
    const auto grid = AtomicGrid::Create(40, 50, 1.0, {0.3, -0.2, 0.5});
    ASSERT_TRUE(grid.has_value());
    EXPECT_EQ(grid->Size(), 40u * 50u);
    EXPECT_NEAR(IntegrateGaussian(*grid), std::pow(kPi, 1.5), 1e-4);
}

TEST(AtomicGridTest, ErrorShrinksWithAngularPoints) {
    const auto coarse = AtomicGrid::Create(40, 26, 1.0, {0.0, 0.0, 0.0});
    const auto fine = AtomicGrid::Create(40, 110, 1.0, {0.0, 0.0, 0.0});
    ASSERT_TRUE(coarse.has_value());
    ASSERT_TRUE(fine.has_value());

    const double target = std::pow(kPi, 1.5);
    const double errorCoarse = std::abs(IntegrateGaussian(*coarse) - target);
    const double errorFine = std::abs(IntegrateGaussian(*fine) - target);
    EXPECT_GT(errorCoarse, errorFine) << "expected angular refinement to help";
}

TEST(AtomicGridTest, PointsLieWithinRMaxOfCenter) {
    const auto grid = AtomicGrid::Create(30, 38, 1.0, {1.0, 1.0, 1.0});
    ASSERT_TRUE(grid.has_value());
    // r_max = alpha * pointCount^2 = 1.0 * 30^2.
    const double rMax = 900.0;

    for (std::size_t i = 0; i < grid->Size(); ++i)
    {
        const std::array<double, 3>& p = grid->Point(i);
        const double dx = p[0] - 1.0;
        const double dy = p[1] - 1.0;
        const double dz = p[2] - 1.0;
        EXPECT_LE(std::sqrt(dx * dx + dy * dy + dz * dz), rMax * 1.000001);
    }
}

} // namespace
} // namespace qcx::grid
