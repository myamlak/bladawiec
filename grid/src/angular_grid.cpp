#include "qcx/grid/angular_grid.hpp"

#include <string>

namespace qcx::grid {

// (offset, size, degree) are the Lebedev table lookup triple: the rule
// row offset, the point count, and the polynomial degree.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
AngularGrid::AngularGrid(std::size_t offset, std::size_t size, std::size_t degree) noexcept :
    _offset(offset), _size(size), _degree(degree) {}

Result<AngularGrid> AngularGrid::Create(std::size_t pointCount) {
    for (std::size_t i = 0; i < internal::kLebedevSizeCount; ++i)
    {
        if (internal::kLebedevSizes[i] == pointCount)
        {
            return AngularGrid(
                internal::kLebedevOffsets[i], pointCount, internal::kLebedevDegrees[i]);
        }
    }

    return std::unexpected(Error{ErrorCode::kInvalidArgument,
                                 "not a Lebedev point count: " + std::to_string(pointCount)});
}

std::array<double, 3> AngularGrid::Point(std::size_t index) const noexcept {
    // The tables are flat x,y,z triples; node k lives at 3 * (offset + k).
    const std::size_t base = 3 * (_offset + index);
    return std::array<double, 3>{internal::kLebedevPoints[base],
                                 internal::kLebedevPoints[base + 1],
                                 internal::kLebedevPoints[base + 2]};
}

double AngularGrid::Weight(std::size_t index) const noexcept {
    return internal::kLebedevWeights[_offset + index];
}

} // namespace qcx::grid
