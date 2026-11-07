#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/angular_grid.hpp"
#include "qcx/grid/radial_grid.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::grid {

/// A single-atom product grid: radial x angular quadrature about a center.
///
/// Point (i, j) is center + r_i * p_j with p_j on the unit sphere; the
/// weight is w_radial(r_i) * r_i^2 * w_angular(p_j), the correct weight
/// for integrals of the form
///     int f(r) d^3r = sum_i sum_j f(center + r_i p_j) w_i w_j.
/// Points are laid out angular-fastest (all angular points of radial
/// point 0, then radial point 1, ...).
/// \ingroup qcx-grid
class AtomicGrid {
public:
    /// Creates the product grid.
    /// \param radialPoints Radial point count (RadialGrid).
    /// \param angularPoints Angular point count (AngularGrid::kAvailableSizes).
    /// \param alpha Radial mapping length scale in Bohr.
    /// \param centerBohr The atom position in Bohr.
    /// \returns The grid, or kInvalidArgument for a non-Lebedev angular
    /// count or an invalid radial parameter.
    /// \ingroup qcx-grid
    static Result<AtomicGrid> Create(std::size_t radialPoints,
                                     std::size_t angularPoints,
                                     double alpha,
                                     std::array<double, 3> centerBohr);

    /// Creates the product grid on a symmetry-adapted angular rule.
    ///
    /// The radial quadrature, the r^2 volume element and the product
    /// layout are those of Create: only the angular source changes, so the
    /// angular adaptation cannot move a radial node or a radial weight.
    /// \param radialPoints Radial point count (RadialGrid).
    /// \param angularPoints Base Lebedev point count
    ///        (AngularGrid::kAvailableSizes).
    /// \param group The point group the angular rule is adapted to.
    /// \param alpha Radial mapping length scale in Bohr.
    /// \param centerBohr The atom position in Bohr.
    /// \returns The grid, or kInvalidArgument for a non-Lebedev angular
    /// count, an invalid radial parameter, or a group the rule cannot build.
    /// \ingroup qcx-grid
    static Result<AtomicGrid> CreateSymmetryAdapted(std::size_t radialPoints,
                                                    std::size_t angularPoints,
                                                    symmetry::PointGroupName group,
                                                    double alpha,
                                                    std::array<double, 3> centerBohr);

    /// The total number of points, radial x angular.
    /// \returns The total point count, radial x angular.
    [[nodiscard]] std::size_t Size() const noexcept {
        return _points.size();
    }

    /// The absolute position of point \p index, in Bohr.
    /// \param index The point index in [0, Size()).
    /// \returns The point position, in Bohr.
    [[nodiscard]] const std::array<double, 3>& Point(std::size_t index) const noexcept {
        const std::array<double, 3>& point = _points[index];

        return point;
    }

    /// The quadrature weight of point \p index (includes r^2).
    /// \param index The point index in [0, Size()).
    /// \returns The quadrature weight (includes r^2).
    [[nodiscard]] double Weight(std::size_t index) const noexcept {
        const double weight = _weights[index];

        return weight;
    }

    /// The grid center (the atom position), in Bohr.
    /// \returns The grid center, in Bohr.
    [[nodiscard]] const std::array<double, 3>& Center() const noexcept {
        return _center;
    }

private:
    AtomicGrid(std::vector<std::array<double, 3>> points,
               std::vector<double> weights,
               std::array<double, 3> center) noexcept;

    // The product loop, shared by every angular source so that the radial
    // quadrature and the r^2 volume element cannot drift between them.
    // Defined in the source; only the factories there instantiate it.
    template <typename Angular>
    static Result<AtomicGrid> BuildProduct(const RadialGrid& radial,
                                           const Angular& angular,
                                           std::array<double, 3> centerBohr);

    std::vector<std::array<double, 3>> _points;
    std::vector<double> _weights;
    std::array<double, 3> _center;
};

} // namespace qcx::grid
