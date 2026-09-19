// Molecule construction: validation against the element table, canonical
// atom renumbering, and deep copy.
#include "qcx/molecule/molecule.hpp"

#include "qcx/backend/cpu_backend.hpp"
#include "qcx/molecule/elements.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <numeric>
#include <vector>

namespace qcx::molecule {

namespace {

// Distance check between two coordinate rows (Bohr).
double DistanceSquared(const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coords,
                       std::size_t i,
                       std::size_t j) {
    const double dx = coords(i, 0) - coords(j, 0);
    const double dy = coords(i, 1) - coords(j, 1);
    const double dz = coords(i, 2) - coords(j, 2);
    return dx * dx + dy * dy + dz * dz;
}

// Squared-distance threshold for coincident-atom rejection: 1e-7 Bohr, about
// 5e-8 Angstrom - far below any physically meaningful interatomic distance.
constexpr double kCoincidentDistanceSquaredThresholdBohr2 = 1e-14;

// Validates atoms against the element table (unknown Z, symbol/Z mismatch),
// fills default isotope masses, checks the charge budget, and rejects
// coincident atoms. On success the atom list carries normalized masses.
qcx::Result<void> ValidateAndNormalize(
    std::vector<Atom>& atoms,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinatesBohr,
    int charge) {
    int chargeBudget = 0;

    for (auto& atom : atoms)
    {
        const ElementData* element = FindElement(atom.atomicNumber);

        if (element == nullptr)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "unknown atomic number: " + std::to_string(atom.atomicNumber)});
        }

        if (atom.symbol != element->symbol)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "symbol '" + atom.symbol + "' does not match Z=" +
                                                  std::to_string(atom.atomicNumber)});
        }

        if (atom.isotopicMass <= 0.0)
        {
            atom.isotopicMass = element->mostAbundantIsotopeMass; // default isotope
        }

        chargeBudget += atom.atomicNumber;
    }

    if (charge > chargeBudget)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "charge cannot exceed the sum of atomic numbers"});
    }

    // Coincident atoms: a parallel pair scan. Row i checks the pairs
    // (i, i+1..n-1); dynamic scheduling balances the shrinking row lengths.
    // A relaxed atomic flag short-circuits the scan once any pair matches.
    const std::size_t n = atoms.size();
    std::atomic<bool> hasCoincident = false;
    qcx::backend::Backend<qcx::backend::CpuTag>{}.ParallelForDynamic(n, [&](std::size_t i) {
        if (hasCoincident.load(std::memory_order_relaxed))
        {
            return;
        }

        for (std::size_t j = i + 1; j < n; ++j)
        {
            if (DistanceSquared(coordinatesBohr, i, j) < kCoincidentDistanceSquaredThresholdBohr2)
            {
                hasCoincident.store(true, std::memory_order_relaxed);
                return;
            }
        }
    });

    if (hasCoincident.load())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "molecule has coincident atoms"});
    }

    return {};
}

} // namespace

qcx::Result<Molecule> Molecule::Create(
    std::vector<Atom> atoms,
    qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> coordinatesBohr,
    int charge,
    int multiplicity) {
    if (atoms.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "molecule must have at least one atom"});
    }

    if (coordinatesBohr.Shape()[0] != atoms.size() || coordinatesBohr.Shape()[1] != 3)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "coordinatesBohr shape must be {atomCount, 3}"});
    }

    if (multiplicity < 1)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "multiplicity must be at least 1"});
    }

    auto validation = ValidateAndNormalize(atoms, coordinatesBohr, charge);

    if (!validation.has_value())
    {
        return std::unexpected(validation.error());
    }

    // Canonical renumbering: sort by (Z, x, y, z); permute coordinates to match.
    const std::size_t n = atoms.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (atoms[a].atomicNumber != atoms[b].atomicNumber)
        {
            return atoms[a].atomicNumber < atoms[b].atomicNumber;
        }

        for (int d = 0; d < 3; ++d)
        {
            if (coordinatesBohr(a, d) != coordinatesBohr(b, d))
            {
                return coordinatesBohr(a, d) < coordinatesBohr(b, d);
            }
        }

        return false;
    });

    const bool renumbered = !std::is_sorted(order.begin(), order.end());

    if (renumbered)
    {
        auto reorderedCoords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, 3});

        if (!reorderedCoords.has_value())
        {
            return std::unexpected(reorderedCoords.error());
        }

        for (std::size_t newIndex = 0; newIndex < n; ++newIndex)
        {
            const std::size_t oldIndex = order[newIndex];

            for (int d = 0; d < 3; ++d)
            {
                (*reorderedCoords)(newIndex, d) = coordinatesBohr(oldIndex, d);
            }
        }

        std::vector<Atom> reorderedAtoms;
        reorderedAtoms.reserve(n);

        for (std::size_t newIndex = 0; newIndex < n; ++newIndex)
        {
            reorderedAtoms.push_back(std::move(atoms[order[newIndex]]));
        }

        atoms = std::move(reorderedAtoms);
        coordinatesBohr = std::move(*reorderedCoords);
    }

    return Molecule(std::move(atoms), std::move(coordinatesBohr), charge, multiplicity);
}

qcx::Result<Molecule> Molecule::Clone() {
    auto coordinatesCopy = _coordinates.Clone();

    if (!coordinatesCopy.has_value())
    {
        return std::unexpected(coordinatesCopy.error());
    }

    return Molecule(_atoms, std::move(*coordinatesCopy), _charge, _multiplicity);
}

} // namespace qcx::molecule
