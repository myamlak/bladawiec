#pragma once

#include "qcx/error.hpp"

#include <cstddef>
#include <vector>

namespace qcx::grid {

/// A radial quadrature for the atomic radial coordinate.
///
/// Uses the Euler-Maclaurin-style substitution of Murray, Handy, and
/// Laming (Mol. Phys. 78 (1993) 997):
///     r(q) = alpha * (q / (1 - q))^m
/// which maps q in [0, 1) onto r in [0, inf).  Points sit at q_i = i / N
/// with N = pointCount + 1, so the outermost finite point is r_max =
/// alpha * pointCount^m.  Weights carry the substitution Jacobian
/// dr/dq * (1/N) but NOT the r^2 of the volume element - the atomic
/// product grid folds r^2 in when it combines radial and angular
/// quadratures.
/// \ingroup qcx-grid
class RadialGrid {
public:
    /// Creates the quadrature.
    /// \param pointCount Radial points; the outermost lies at r_max.
    /// \param alpha Length scale of the mapping (Bohr).
    /// \param exponent The mapping exponent m; 2 is the standard choice.
    /// \returns The grid, or kInvalidArgument for a zero point count or a
    /// zero exponent.
    /// \ingroup qcx-grid
    static Result<RadialGrid> Create(std::size_t pointCount,
                                     double alpha,
                                     std::size_t exponent = 2);

    /// The number of radial points.
    /// \returns The number of radial points.
    [[nodiscard]] std::size_t Size() const noexcept {
        return _points.size();
    }

    /// The radial coordinate of point \p index, in Bohr.
    /// \param index The point index in [0, Size()).
    /// \returns The radial coordinate, in Bohr.
    [[nodiscard]] double Point(std::size_t index) const noexcept {
        const double point = _points[index];

        return point;
    }

    /// The quadrature weight of point \p index: dr/dq * (1/N).  Multiply
    /// by r^2 for the 3D volume element.
    /// \param index The point index in [0, Size()).
    /// \returns The quadrature weight (dr/dq * (1/N)).
    [[nodiscard]] double Weight(std::size_t index) const noexcept {
        const double weight = _weights[index];

        return weight;
    }

    /// The outermost radial point, alpha * pointCount^exponent.
    /// \returns The outermost radial point, in Bohr.
    [[nodiscard]] double RMax() const noexcept {
        return _rMax;
    }

private:
    RadialGrid(std::vector<double> points, std::vector<double> weights, double rMax) noexcept;

    std::vector<double> _points;
    std::vector<double> _weights;
    double _rMax;
};

} // namespace qcx::grid
