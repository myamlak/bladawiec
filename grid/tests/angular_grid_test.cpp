// The certified Lebedev tables are the foundation of the whole grid
// module, so the first test re-derives their defining property at
// double precision: sum_i w_i m(p_i) == int_{S^2} m dOmega for every
// even x^A y^B monomial up to the grid's degree of exactness, for every
// provided size.  The tables carry a 1e-60 mpmath certificate; here the
// emitted doubles must reproduce the moments to 1e-12.

#include "qcx/grid/angular_grid.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace qcx::grid {
namespace {

// int_{S^2} x^A y^B dOmega for even A, B (z = 0 exponent):
// 2 * Gamma((A+1)/2) Gamma((B+1)/2) Gamma(1/2) / Gamma((A+B+3)/2).
// (a, b) are the two exponents - the moment's roles are symmetric.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double SphereMoment(std::size_t a, std::size_t b) {
    const double ha = static_cast<double>(a + 1) / 2.0;
    const double hb = static_cast<double>(b + 1) / 2.0;
    return 2.0 * std::tgamma(ha) * std::tgamma(hb) * std::tgamma(0.5) / std::tgamma(ha + hb + 0.5);
}

// Every even (A, B) pair with A >= B and A + B <= degree.
std::vector<std::array<std::size_t, 2>> MomentBasis(std::size_t degree) {
    std::vector<std::array<std::size_t, 2>> basis;

    for (std::size_t a = 0; a <= degree; a += 2)
    {
        for (std::size_t b = 0; b <= a && a + b <= degree; b += 2)
        {
            basis.push_back({a, b});
        }
    }

    return basis;
}

TEST(AngularGridTest, WeightsSumTo4PiForEverySize) {
    for (const std::size_t size : AngularGrid::kAvailableSizes)
    {
        const auto grid = AngularGrid::Create(size);
        ASSERT_TRUE(grid.has_value()) << "size " << size;

        double sum = 0.0;

        for (std::size_t i = 0; i < grid->Size(); ++i)
        {
            sum += grid->Weight(i);
        }

        EXPECT_NEAR(sum, 4.0 * std::acos(-1.0), 1e-12) << "size " << size;
    }
}

TEST(AngularGridTest, NodesLieOnTheUnitSphere) {
    for (const std::size_t size : AngularGrid::kAvailableSizes)
    {
        const auto grid = AngularGrid::Create(size);
        ASSERT_TRUE(grid.has_value()) << "size " << size;

        for (std::size_t i = 0; i < grid->Size(); ++i)
        {
            const std::array<double, 3> p = grid->Point(i);
            const double norm = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            EXPECT_NEAR(norm, 1.0, 1e-12) << "size " << size << " point " << i;
        }
    }
}

// The defining property: exactness for every even monomial up to degree.
TEST(AngularGridTest, MomentsAreExactUpToTheDegree) {
    for (const std::size_t size : AngularGrid::kAvailableSizes)
    {
        const auto grid = AngularGrid::Create(size);
        ASSERT_TRUE(grid.has_value()) << "size " << size;

        for (const auto [a, b] : MomentBasis(grid->Degree()))
        {
            double sum = 0.0;

            for (std::size_t i = 0; i < grid->Size(); ++i)
            {
                const std::array<double, 3> p = grid->Point(i);
                sum += grid->Weight(i) * std::pow(p[0], static_cast<double>(a)) *
                       std::pow(p[1], static_cast<double>(b));
            }

            const double expected = SphereMoment(a, b);
            EXPECT_NEAR(sum, expected, 1e-12) << "size " << size << " degree " << grid->Degree()
                                              << " monomial x^" << a << " y^" << b;
        }
    }
}

TEST(AngularGridTest, DegreeGrowsWithSize) {
    const auto six = AngularGrid::Create(6);
    const auto largest = AngularGrid::Create(AngularGrid::kAvailableSizes.back());
    ASSERT_TRUE(six.has_value());
    ASSERT_TRUE(largest.has_value());
    EXPECT_EQ(six->Degree(), 3);
    EXPECT_EQ(largest->Degree(), 35);
}

TEST(AngularGridTest, CreateRejectsNonLebedevCounts) {
    const auto grid = AngularGrid::Create(100);
    ASSERT_FALSE(grid.has_value());
    EXPECT_EQ(grid.error().code, ErrorCode::kInvalidArgument);
}

} // namespace
} // namespace qcx::grid
