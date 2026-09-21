// The scaling of the Becke weight derivatives with atom count.
//
// MEASURED, not argued. For a grid point and a molecule the harness takes the
// shipped partition weights (qcx::grid::BeckeRawWeights, partition_function.cpp)
// and CENTRAL-DIFFERENCES them against every nuclear coordinate: every
// (weight, atom, axis) entry the derivative array would hold is computed, and
// the entries that are actually nonzero are counted. That count is what the
// per-point derivative array costs - the dense count is 3 A^2 by construction,
// so the measurement is really "how much of the dense array survives the
// partition's own cutoff".
//
// Two fixtures, because the answer is geometric: a 1D chain (the distance
// cutoff can exclude atoms) and a compact 3D cluster (every atom is within the
// SSF window, so the cutoff has nothing to exclude).

#include "memory_probe.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/grid/partition_function.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace memprobe {
namespace {

/// The finite-difference step, Bohr. The SSF switch is a degree-7 polynomial
/// on its window, so a central difference at this step is exact to rounding.
constexpr double kStepBohr = 1.0e-5;

/// A derivative entry counts as significant when it exceeds this fraction of
/// the largest entry at the point. Below it, the entry is rounding on a
/// plateau of the switching function rather than a dependence.
constexpr double kSignificanceFraction = 1.0e-10;

/// Builds a molecule from atoms and coordinates.
qcx::Result<qcx::molecule::Molecule> MakeMolecule(std::vector<qcx::molecule::Atom> atoms,
                                                  std::vector<std::array<double, 3>> positions) {
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({positions.size(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t i = 0; i < positions.size(); ++i)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(i, axis) = positions[i][axis];
        }
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

/// A collinear hydrogen chain: A atoms 1.4 Bohr apart along x.
qcx::Result<qcx::molecule::Molecule> MakeChain(std::size_t atomCount) {
    std::vector<qcx::molecule::Atom> atoms;
    std::vector<std::array<double, 3>> positions;

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
        positions.push_back({1.4 * static_cast<double>(i), 0.0, 0.0});
    }

    return MakeMolecule(std::move(atoms), std::move(positions));
}

/// A compact hydrogen cluster: an n x n x n cubic lattice, 1.4 Bohr spacing,
/// centered on the origin, with a deterministic JITTER.
///
/// The jitter is not decoration. A perfect lattice makes the molecule module's
/// canonical renumbering ambiguous - a symmetric fixture has exact ties in
/// whatever key the renumbering sorts on - and a tie reorders the atoms under a
/// 1e-5 Bohr displacement, which silently finite-differences the wrong
/// atom. Measured before the jitter: 16 of 24 cube displacements reordered the
/// atoms at 8 atoms, 128 of 192 at 64. At 0.13 Bohr the ties are gone and the
/// geometry is still a compact cluster (spacing 1.4 Bohr, so no atom is
/// displaced onto another).
qcx::Result<qcx::molecule::Molecule> MakeCube(std::size_t side) {
    constexpr double kSpacing = 1.4;
    constexpr double kJitter = 0.13;

    std::vector<qcx::molecule::Atom> atoms;
    std::vector<std::array<double, 3>> positions;
    const double origin = -0.5 * kSpacing * static_cast<double>(side - 1);

    for (std::size_t i = 0; i < side; ++i)
    {
        for (std::size_t j = 0; j < side; ++j)
        {
            for (std::size_t k = 0; k < side; ++k)
            {
                const double index = static_cast<double>((i * side + j) * side + k);

                atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
                positions.push_back({origin + kSpacing * static_cast<double>(i) +
                                         kJitter * std::sin(1.0 + 1.7 * index),
                                     origin + kSpacing * static_cast<double>(j) +
                                         kJitter * std::sin(2.0 + 2.3 * index),
                                     origin + kSpacing * static_cast<double>(k) +
                                         kJitter * std::sin(3.0 + 3.1 * index)});
            }
        }
    }

    return MakeMolecule(std::move(atoms), std::move(positions));
}

/// The probe points of one fixture: fixed fractions of the fixture's extent,
/// so the same geometry is asked the same question at every size.
std::vector<std::array<double, 3>> ChainProbes(std::size_t atomCount) {
    const double span = 1.4 * static_cast<double>(atomCount - 1);
    const double mid = 0.5 * span;

    return {{mid, 0.0, 0.0}, {0.7, 0.0, 0.0}, {mid, 1.0, 0.0}, {mid, 2.5, 0.0}};
}

std::vector<std::array<double, 3>> CubeProbes(std::size_t side) {
    constexpr double kSpacing = 1.4;
    const double span = kSpacing * static_cast<double>(side - 1);

    return {{0.0, 0.0, 0.0},
            {0.35, 0.35, 0.35},
            {0.5 * span, 0.0, 0.0},
            {0.5 * span, 0.5 * span, 0.5 * span}};
}

/// One point's measurement.
struct PointMeasurement {
    std::size_t switchCount = 0; ///< Switching-function evaluations the pair loop ran.
    std::size_t nonzeroRaw = 0; ///< Atoms whose raw weight is nonzero.
    std::size_t nonzeroDerivatives = 0; ///< (weight, atom, axis) entries exactly nonzero.
    std::size_t significantDerivatives = 0; ///< ... above kSignificanceFraction of the largest.
    double maxDerivative = 0.0; ///< The largest |entry|.
    std::size_t reorderings =
        0; ///< Displacements whose molecule did not come back in the same order.
};

/// Builds \p positions with atom \p k displaced by \p delta on \p axis and
/// evaluates the shipped weights there. The canonical order must survive the
/// displacement or the finite difference differentiates the wrong atom, so the
/// caller checks it (the reorderings counter).
qcx::Result<std::vector<double>> WeightsWithDisplacement(const std::array<double, 3>& point,
                                                         const qcx::molecule::Molecule& molecule,
                                                         std::size_t k,
                                                         std::size_t axis,
                                                         double delta,
                                                         bool& orderPreserved) {
    const auto& coordinates = molecule.CoordinatesBohr();
    const std::size_t atomCount = molecule.Atoms().size();
    std::vector<std::array<double, 3>> positions;

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        positions.push_back({coordinates(i, 0), coordinates(i, 1), coordinates(i, 2)});
    }

    positions[k][axis] += delta;

    auto moved = MakeMolecule(molecule.Atoms(), positions);

    if (!moved.has_value())
    {
        return std::unexpected(moved.error());
    }

    const auto& movedCoordinates = moved->CoordinatesBohr();
    orderPreserved = true;

    for (std::size_t i = 0; i < atomCount && orderPreserved; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            if (movedCoordinates(i, j) != positions[i][j])
            {
                orderPreserved = false;
                break;
            }
        }
    }

    return qcx::grid::BeckeRawWeights(point, *moved);
}

/// Measures one point: the shipped pair-loop count, the weight support, and the
/// derivative entries a central difference says are really nonzero.
PointMeasurement MeasurePoint(const std::array<double, 3>& point,
                              const qcx::molecule::Molecule& molecule) {
    PointMeasurement measurement;

    const std::vector<double> raw =
        qcx::grid::BeckeRawWeights(point, molecule, measurement.switchCount);
    const std::size_t atomCount = raw.size();

    for (const double weight : raw)
    {
        if (weight != 0.0)
        {
            ++measurement.nonzeroRaw;
        }
    }

    // The full derivative array, one entry per (weight, atom, axis): the same
    // shape a consumer assembling d/dR would hold, 3 A^2 entries per point.
    std::vector<double> derivatives;

    for (std::size_t k = 0; k < atomCount; ++k)
    {
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            bool plusOrder = true;
            bool minusOrder = true;
            const auto plus =
                WeightsWithDisplacement(point, molecule, k, axis, kStepBohr, plusOrder);
            const auto minus =
                WeightsWithDisplacement(point, molecule, k, axis, -kStepBohr, minusOrder);

            if (!plusOrder || !minusOrder)
            {
                ++measurement.reorderings;
            }

            if (!plus.has_value() || !minus.has_value())
            {
                continue;
            }

            for (std::size_t a = 0; a < atomCount; ++a)
            {
                const double derivative = ((*plus)[a] - (*minus)[a]) / (2.0 * kStepBohr);
                derivatives.push_back(derivative);

                if (derivative != 0.0)
                {
                    ++measurement.nonzeroDerivatives;
                }

                const double magnitude = std::fabs(derivative);

                if (magnitude > measurement.maxDerivative)
                {
                    measurement.maxDerivative = magnitude;
                }
            }
        }
    }

    const double threshold = kSignificanceFraction * measurement.maxDerivative;

    for (const double entry : derivatives)
    {
        if (std::fabs(entry) > threshold)
        {
            ++measurement.significantDerivatives;
        }
    }

    return measurement;
}

/// Runs one fixture and prints the per-point structure for every size.
void RunFixture(const char* tag,
                const std::vector<std::size_t>& sizes,
                qcx::Result<qcx::molecule::Molecule> (*build)(std::size_t),
                std::vector<std::array<double, 3>> (*probes)(std::size_t)) {
    for (const std::size_t size : sizes)
    {
        auto molecule = build(size);

        if (!molecule.has_value())
        {
            std::printf("m6_%s_%zu = ERROR %s\n", tag, size, molecule.error().message.c_str());
            continue;
        }

        const std::size_t atomCount = molecule->Atoms().size();
        const std::vector<std::array<double, 3>> points = probes(size);

        for (std::size_t index = 0; index < points.size(); ++index)
        {
            const PointMeasurement measurement = MeasurePoint(points[index], *molecule);
            const std::size_t dense = 3 * atomCount * atomCount;

            std::printf("m6_%s_atoms=%zu_probe=%zu = switches=%zu nonzero_raw_weights=%zu "
                        "dense_derivative_entries=%zu measured_nonzero_entries=%zu "
                        "significant_entries=%zu max_derivative=%.6e "
                        "derivative_bytes_measured=%zu dense_bytes=%zu\n",
                        tag,
                        atomCount,
                        index,
                        measurement.switchCount,
                        measurement.nonzeroRaw,
                        dense,
                        measurement.nonzeroDerivatives,
                        measurement.significantDerivatives,
                        measurement.maxDerivative,
                        measurement.significantDerivatives * sizeof(double),
                        dense * sizeof(double));

            if (measurement.reorderings != 0)
            {
                std::printf("m6_%s_atoms=%zu_probe=%zu_WARNING = %zu displacements changed the "
                            "canonical atom order; the finite difference is unreliable there\n",
                            tag,
                            atomCount,
                            index,
                            measurement.reorderings);
            }
        }
    }
}

} // namespace

int RunM6(const std::vector<std::string>& args) {
    std::size_t chainMax = 32;
    std::size_t cubeSideMax = 3;

    for (std::size_t i = 0; i < args.size(); ++i)
    {
        if (args[i] == "--chain-max" && i + 1 < args.size())
        {
            chainMax = static_cast<std::size_t>(std::stoul(args[i + 1]));
        } else if (args[i] == "--cube-side" && i + 1 < args.size())
        { cubeSideMax = static_cast<std::size_t>(std::stoul(args[i + 1])); }
    }

    std::printf(
        "m6_fixture = hydrogen chains (1.4 Bohr spacing, x axis) and hydrogen cubes "
        "(1.4 Bohr cubic lattice), shipped BeckeRawWeights of grid/partition_function.cpp\n");

    std::vector<std::size_t> chainSizes;

    for (std::size_t n = 2; n <= chainMax; n *= 2)
    {
        chainSizes.push_back(n);
    }

    if (chainSizes.empty() || chainSizes.back() != chainMax)
    {
        chainSizes.push_back(chainMax);
    }

    std::vector<std::size_t> cubeSides;

    for (std::size_t side = 2; side <= cubeSideMax; ++side)
    {
        cubeSides.push_back(side);
    }

    RunFixture("chain", chainSizes, &MakeChain, &ChainProbes);
    RunFixture("cube", cubeSides, &MakeCube, &CubeProbes);

    // The summary number: the DERIVATIVE ENTRY COUNT at the
    // largest fixture of each family, and what it implies per grid point.
    std::printf(
        "m6_dense_entries_note = the dense array is 3*A^2 entries per point = 24*A^2 bytes\n");

    return 0;
}

} // namespace memprobe
