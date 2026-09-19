#include "qcx/grid/geometry_translation.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace qcx::grid {

qcx::Error TranslateExcgridError(excgrid::ErrorCode code, std::string_view context) {
    switch (code)
    {
    case excgrid::ErrorCode::kUnsupported:
        return qcx::Error{qcx::ErrorCode::kUnimplemented,
                          std::string(context) + ": unsupported by excgrid"};

    case excgrid::ErrorCode::kInvalidArgument:
        return qcx::Error{qcx::ErrorCode::kInvalidArgument,
                          std::string(context) + ": invalid argument"};
    }

    // Unreachable for a shipped code: the switch above is exhaustive, and GCC
    // and Clang report a missing enumerator the moment excgrid grows a third
    // one. This tail exists only because MSVC cannot rule out an out-of-range
    // value at compile time and warns about the missing return otherwise.
    return qcx::Error{qcx::ErrorCode::kInvalidArgument,
                      std::string(context) + ": unrecognized excgrid error code"};
}

qcx::Result<excgrid::Geometry> ToExcgridGeometry(const qcx::molecule::Molecule& molecule) {
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinates =
        molecule.CoordinatesBohr();
    const std::size_t atomCount = molecule.AtomCount();

    if (coordinates.Shape()[0] != atomCount || coordinates.Shape()[1] != 3)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "ToExcgridGeometry: coordinates are not {atomCount, 3}"});
    }

    const std::vector<qcx::molecule::Atom>& atoms = molecule.Atoms();
    excgrid::Geometry geometry;
    geometry.atoms.reserve(atomCount);

    for (std::size_t atom = 0; atom < atomCount; ++atom)
    {
        geometry.atoms.push_back(excgrid::Atom{atoms[atom].atomicNumber,
                                               std::array<double, 3>{coordinates(atom, 0),
                                                                     coordinates(atom, 1),
                                                                     coordinates(atom, 2)}});
    }

    return geometry;
}

} // namespace qcx::grid
