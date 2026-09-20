#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/internal/lebedev_tables.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace qcx::grid {

/// \defgroup qcx-grid Grid module
/// Numerical integration grids: radial and angular quadratures, the
/// atomic and molecular product grids, the Becke fuzzy-cell partition,
/// and the AO-at-point evaluator.
/// \{

/// A Lebedev-Laikov angular quadrature on the unit sphere.
///
/// The point set is closed under the octahedral group and the weights
/// integrate every polynomial of total degree <= Degree() exactly.  The
/// tables are SOLVED for from these two defining conditions by
/// tools/grid/gen_lebedev.py (no copied tables) and certified per size to
/// max |moment residual| < 1e-60.  Weights are normalized so that the
/// weights sum to 4 pi, the area of the unit sphere - the angular
/// convention of the molecular product grid.
/// \ingroup qcx-grid
class AngularGrid {
public:
    /// The available point counts, smallest first.
    /// \ingroup qcx-grid
    static constexpr std::span<const std::size_t, internal::kLebedevSizeCount> kAvailableSizes =
        internal::kLebedevSizes;

    /// Creates the quadrature with the given point count.
    /// \param pointCount A size from kAvailableSizes.
    /// \returns The grid, or kInvalidArgument for a non-Lebedev count.
    /// \ingroup qcx-grid
    static Result<AngularGrid> Create(std::size_t pointCount);

    /// The number of quadrature nodes.
    /// \returns The node count.
    [[nodiscard]] std::size_t Size() const noexcept {
        return _size;
    }

    /// The algebraic degree of exactness: every polynomial of total degree
    /// <= Degree() integrates exactly.
    /// \returns The algebraic degree of exactness.
    [[nodiscard]] std::size_t Degree() const noexcept {
        return _degree;
    }

    /// The \p index-th node, a unit vector on the sphere.
    /// \param index The node index in [0, Size()).
    /// \returns The node as a unit vector on the sphere.
    [[nodiscard]] std::array<double, 3> Point(std::size_t index) const noexcept;

    /// The \p index-th quadrature weight; the weights sum to 4 pi.
    /// \param index The node index in [0, Size()).
    /// \returns The quadrature weight (the weights sum to 4 pi).
    [[nodiscard]] double Weight(std::size_t index) const noexcept;

private:
    AngularGrid(std::size_t offset, std::size_t size, std::size_t degree) noexcept;

    // Start of this grid's nodes in the flat internal tables.
    std::size_t _offset;
    std::size_t _size;
    std::size_t _degree;
};

/// \}

} // namespace qcx::grid
