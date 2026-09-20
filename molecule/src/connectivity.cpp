// Connectivity via the covalent-radius-sum heuristic: a parallel pair loop
// with one neighbor list per atom (each atom's row is written by exactly
// one iteration - no merge step), producing both the Boost.Graph and the
// CSR index representation.
#include "qcx/molecule/connectivity.hpp"

#include "qcx/backend/cpu_backend.hpp"
#include "qcx/molecule/elements.hpp"

#include <algorithm>
#include <cmath>

namespace qcx::molecule {

namespace {

// Deterministic finalization: mirror each directed neighbor entry, sort the
// per-atom lists, pack the CSR (reserved up front), and add one Boost edge
// per unordered pair.
void FinalizeConnectivity(const std::vector<std::vector<std::size_t>>& neighbors,
                          Connectivity& result) {
    const std::size_t n = neighbors.size();
    std::vector<std::vector<std::size_t>> symmetric(n);
    std::size_t symmetricSize = 0;

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j : neighbors[i])
        {
            symmetric[i].push_back(j);
            symmetric[j].push_back(i);
            symmetricSize += 2;
        }
    }

    result.csr.offsets.resize(n + 1);
    result.csr.neighbors.reserve(symmetricSize);
    std::size_t total = 0;

    for (std::size_t i = 0; i < n; ++i)
    {
        std::sort(symmetric[i].begin(), symmetric[i].end());
        symmetric[i].erase(std::unique(symmetric[i].begin(), symmetric[i].end()),
                           symmetric[i].end());
        result.csr.offsets[i] = total;
        total += symmetric[i].size();
        result.csr.neighbors.insert(
            result.csr.neighbors.end(), symmetric[i].begin(), symmetric[i].end());
    }

    result.csr.offsets[n] = total;
    result.graph = ConnectivityGraph(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j : symmetric[i])
        {
            if (i < j)
            {
                boost::add_edge(i, j, result.graph);
            }
        }
    }
}

} // namespace

Connectivity BuildConnectivity(const Molecule& molecule, double toleranceBohr) {
    const auto& coords = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();
    const std::size_t n = atoms.size();

    // Hoist the per-element covalent radii out of the O(n^2) pair loop: the
    // table lookup is the same for every pair involving an atom. A zero
    // radius means "no published value / does not bond".
    std::vector<double> radiusBohr(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        radiusBohr[i] = FindElement(atoms[i].atomicNumber)->covalentRadiusBohr;
    }

    // Parallel pair loop: iteration i writes only neighbors[i], so the
    // disjoint per-atom lists merge without any synchronization.
    std::vector<std::vector<std::size_t>> neighbors(n);
    // Dynamic scheduling: the triangular j = i+1..n scan gives iteration
    // i=0 O(n) inner work and iteration i=n-1 O(1) - a static contiguous
    // split is structurally imbalanced regardless of the data.
    qcx::backend::Backend<qcx::backend::CpuTag>{}.ParallelForDynamic(n, [&](std::size_t i) {
        for (std::size_t j = i + 1; j < n; ++j)
        {
            if (radiusBohr[i] == 0.0 || radiusBohr[j] == 0.0)
            {
                continue;
            }

            const double dx = coords(i, 0) - coords(j, 0);
            const double dy = coords(i, 1) - coords(j, 1);
            const double dz = coords(i, 2) - coords(j, 2);
            const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (distance <= radiusBohr[i] + radiusBohr[j] + toleranceBohr)
            {
                neighbors[i].push_back(j);
            }
        }
    });

    Connectivity result;
    FinalizeConnectivity(neighbors, result);
    return result;
}

} // namespace qcx::molecule
