#include "qcx/grid/atomic_grid.hpp"

#include <cmath>

namespace qcx::grid {

AtomicGrid::AtomicGrid(std::vector<std::array<double, 3>> points,
                       std::vector<double> weights,
                       std::array<double, 3> center) noexcept :
    _points(std::move(points)), _weights(std::move(weights)), _center(center) {}

// (radialPoints, angularPoints) are the radial and angular quadrature
// point counts - each feeds its own builder.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Result<AtomicGrid> AtomicGrid::Create(std::size_t radialPoints,
                                      std::size_t angularPoints,
                                      double alpha,
                                      std::array<double, 3> centerBohr) {
    auto radial = RadialGrid::Create(radialPoints, alpha);

    if (!radial.has_value())
    {
        return std::unexpected(radial.error());
    }

    auto angular = AngularGrid::Create(angularPoints);

    if (!angular.has_value())
    {
        return std::unexpected(angular.error());
    }

    std::vector<std::array<double, 3>> points;
    std::vector<double> weights;
    points.reserve(radial->Size() * angular->Size());
    weights.reserve(radial->Size() * angular->Size());

    for (std::size_t i = 0; i < radial->Size(); ++i)
    {
        const double r = radial->Point(i);
        // The volume element contributes r^2 dr dOmega.
        const double radialWeight = radial->Weight(i) * r * r;

        for (std::size_t j = 0; j < angular->Size(); ++j)
        {
            const std::array<double, 3> p = angular->Point(j);
            points.push_back(
                {centerBohr[0] + r * p[0], centerBohr[1] + r * p[1], centerBohr[2] + r * p[2]});
            weights.push_back(radialWeight * angular->Weight(j));
        }
    }

    return AtomicGrid(std::move(points), std::move(weights), centerBohr);
}

} // namespace qcx::grid
