#include "qcx/grid/radial_grid.hpp"

#include <cmath>
#include <string>

namespace qcx::grid {

RadialGrid::RadialGrid(std::vector<double> points,
                       std::vector<double> weights,
                       double rMax) noexcept :
    _points(std::move(points)), _weights(std::move(weights)), _rMax(rMax) {}

Result<RadialGrid> RadialGrid::Create(std::size_t pointCount, double alpha, std::size_t exponent) {
    if (pointCount == 0)
    {
        return std::unexpected(
            Error{ErrorCode::kInvalidArgument, "radial grid needs at least one point"});
    }

    if (alpha <= 0.0 || exponent == 0)
    {
        return std::unexpected(Error{ErrorCode::kInvalidArgument,
                                     "radial grid needs alpha > 0 and a nonzero exponent"});
    }

    const double n = static_cast<double>(pointCount + 1);
    const double m = static_cast<double>(exponent);
    std::vector<double> points;
    std::vector<double> weights;
    points.reserve(pointCount);
    weights.reserve(pointCount);

    for (std::size_t i = 1; i <= pointCount; ++i)
    {
        const double q = static_cast<double>(i) / n;
        const double oneMinusQ = 1.0 - q;
        // r(q) = alpha * (q / (1 - q))^m
        const double ratio = q / oneMinusQ;
        points.push_back(alpha * std::pow(ratio, m));
        // dr/dq = alpha * m * q^(m-1) / (1 - q)^(m+1);  dq = 1 / N.
        weights.push_back(alpha * m * std::pow(q, m - 1.0) / std::pow(oneMinusQ, m + 1.0) / n);
    }

    const double rMax = alpha * std::pow(static_cast<double>(pointCount), m);
    return RadialGrid(std::move(points), std::move(weights), rMax);
}

} // namespace qcx::grid
