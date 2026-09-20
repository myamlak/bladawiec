#include "qcx/grid/molecular_grid.hpp"

#include "qcx/grid/atomic_grid.hpp"
#include "qcx/grid/partition_function.hpp"

#include <cstddef>

namespace qcx::grid {

MolecularGrid::MolecularGrid(std::vector<std::array<double, 3>> points,
                             std::vector<double> weights,
                             std::vector<std::size_t> atomIndex,
                             std::size_t atomCount) noexcept :
    _points(std::move(points)), _weights(std::move(weights)), _atomIndex(std::move(atomIndex)),
    _atomCount(atomCount) {}

Result<MolecularGrid> MolecularGrid::Create(const qcx::molecule::Molecule& molecule,
                                            std::size_t radialPoints,
                                            std::size_t angularPoints,
                                            double alpha) {
    const std::size_t atomCount = molecule.Atoms().size();
    const auto& coordinates = molecule.CoordinatesBohr();

    std::vector<std::array<double, 3>> points;
    std::vector<double> weights;
    std::vector<std::size_t> atomIndex;

    for (std::size_t a = 0; a < atomCount; ++a)
    {
        const std::array<double, 3> center = {
            coordinates(a, 0), coordinates(a, 1), coordinates(a, 2)};
        auto atomic = AtomicGrid::Create(radialPoints, angularPoints, alpha, center);

        if (!atomic.has_value())
        {
            return std::unexpected(atomic.error());
        }

        for (std::size_t i = 0; i < atomic->Size(); ++i)
        {
            const std::array<double, 3> point = atomic->Point(i);
            points.push_back(point);
            const std::vector<double> partition = BeckePartitionWeights(point, molecule);
            weights.push_back(atomic->Weight(i) * partition[a]);
            atomIndex.push_back(a);
        }
    }

    return MolecularGrid(std::move(points), std::move(weights), std::move(atomIndex), atomCount);
}

} // namespace qcx::grid
