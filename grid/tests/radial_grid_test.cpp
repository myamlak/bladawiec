// Radial quadrature: the Euler-Maclaurin mapping must reproduce the
// classic radial integrals
//     int_0^inf r^2 e^{-r} dr = 2,        int_0^inf r^2 e^{-r^2} dr = sqrt(pi)/4,
// with the error shrinking as the point count grows (monotone in the
// approximation parameter - the test shape used throughout qcx for
// resolution-dependent quantities).

#include "qcx/grid/radial_grid.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace qcx::grid {
namespace {

constexpr double kPi = 3.14159265358979323846;

double Integrate(const RadialGrid& grid, double (*f)(double)) {
    double sum = 0.0;

    for (std::size_t i = 0; i < grid.Size(); ++i)
    {
        const double r = grid.Point(i);
        sum += grid.Weight(i) * r * r * f(r);
    }

    return sum;
}

TEST(RadialGridTest, IntegratesExpMinusR) {
    for (const std::size_t points : {30u, 60u, 120u})
    {
        const auto grid = RadialGrid::Create(points, 1.0);
        ASSERT_TRUE(grid.has_value());
        // Reference: int r^2 e^{-r} dr = 2.
        EXPECT_NEAR(Integrate(*grid, [](double r) { return std::exp(-r); }), 2.0, 1e-3)
            << "points " << points;
    }
}

TEST(RadialGridTest, ErrorShrinksWithPointCount) {
    const auto coarse = RadialGrid::Create(20, 1.0);
    const auto fine = RadialGrid::Create(200, 1.0);
    ASSERT_TRUE(coarse.has_value());
    ASSERT_TRUE(fine.has_value());

    const auto f = [](double r) { return std::exp(-r) * r * r * r; };
    // Reference: int r^5 e^{-r} dr = Gamma(6) = 120.
    const double errorCoarse = std::abs(Integrate(*coarse, f) - 120.0);
    const double errorFine = std::abs(Integrate(*fine, f) - 120.0);
    EXPECT_GT(errorCoarse, errorFine * 10) << "expected monotone refinement";
}

TEST(RadialGridTest, IntegratesExpMinusRSquared) {
    const auto grid = RadialGrid::Create(200, 1.0);
    ASSERT_TRUE(grid.has_value());
    // Reference: int r^2 e^{-r^2} dr = sqrt(pi)/4.
    const double expected = std::sqrt(kPi) / 4.0;
    EXPECT_NEAR(Integrate(*grid, [](double r) { return std::exp(-r * r); }), expected, 1e-6);
}

TEST(RadialGridTest, OuterPointMatchesRMax) {
    const auto grid = RadialGrid::Create(50, 2.0);
    ASSERT_TRUE(grid.has_value());
    // The last point is alpha * (N/(N+1))^m mapped, i.e. r_max up to
    // pow() rounding.
    EXPECT_NEAR(grid->Point(grid->Size() - 1), grid->RMax(), 1e-9);
    EXPECT_NEAR(grid->RMax(), 2.0 * 50.0 * 50.0, 1e-9);
}

TEST(RadialGridTest, CreateRejectsBadParameters) {
    EXPECT_FALSE(RadialGrid::Create(0, 1.0).has_value());
    EXPECT_FALSE(RadialGrid::Create(10, 0.0).has_value());
    EXPECT_FALSE(RadialGrid::Create(10, -1.0).has_value());
    // A zero mapping exponent would collapse every point onto r = 1
    // (r(q) = alpha (q/(1-q))^0); rejected like the other degenerate
    // parameters.
    EXPECT_FALSE(RadialGrid::Create(10, 1.0, 0).has_value());
}

} // namespace
} // namespace qcx::grid
