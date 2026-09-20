#pragma once

#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::grid {

/// The molecular integration grid: per-atom radial x angular product
/// grids combined into one point set, each point weighted by its Becke
/// partition weight.
///
/// Point layout: all of atom 0's points, then atom 1's, ...; within an
/// atom, angular-fastest (see AtomicGrid).  The weight of a point is
/// w_atomic * W_atom(point) with W_atom the normalized partition weight,
/// so that sums over the grid approximate integrals over all of R^3:
///     sum_i w_i f(point_i) ~ int f(r) d^3r.
/// \ingroup qcx-grid
class MolecularGrid {
public:
    /// Creates the grid.
    /// \param molecule The molecule whose atoms center the sub-grids.
    /// \param radialPoints Radial points per atom (RadialGrid).
    /// \param angularPoints Angular points per atom
    /// (AngularGrid::kAvailableSizes).
    /// \param alpha Radial mapping length scale in Bohr.
    /// \returns The grid, or an Error from the underlying quadratures.
    /// \ingroup qcx-grid
    static Result<MolecularGrid> Create(const qcx::molecule::Molecule& molecule,
                                        std::size_t radialPoints,
                                        std::size_t angularPoints,
                                        double alpha = 0.5);

    /// The total number of points across all atoms.
    /// \returns The total point count across all atoms.
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

    /// The quadrature weight of point \p index, including the partition
    /// weight.
    /// \param index The point index in [0, Size()).
    /// \returns The quadrature weight (includes the partition weight).
    [[nodiscard]] double Weight(std::size_t index) const noexcept {
        const double weight = _weights[index];

        return weight;
    }

    /// The atom whose sub-grid point \p index belongs to.
    /// \param index The point index in [0, Size()).
    /// \returns The atom index owning the point.
    [[nodiscard]] std::size_t AtomIndex(std::size_t index) const noexcept {
        const std::size_t atomIndex = _atomIndex[index];

        return atomIndex;
    }

    /// The number of atoms (sub-grids).
    /// \returns The number of atoms (sub-grids).
    [[nodiscard]] std::size_t AtomCount() const noexcept {
        return _atomCount;
    }

private:
    MolecularGrid(std::vector<std::array<double, 3>> points,
                  std::vector<double> weights,
                  std::vector<std::size_t> atomIndex,
                  std::size_t atomCount) noexcept;

    std::vector<std::array<double, 3>> _points;
    std::vector<double> _weights;
    std::vector<std::size_t> _atomIndex;
    std::size_t _atomCount;
};

} // namespace qcx::grid
