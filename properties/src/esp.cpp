// ESP-derived charges (CHELPG/MK): point-charge fits to the
// molecular electrostatic potential under the total-charge constraint.
// The electron potential comes from the integrals module's point kernel
// (BuildElectronPotentialAtPoints), the nuclear part is analytic here,
// and the constrained least-squares solution runs through the augmented
// normal equations.
#include "qcx/properties/esp.hpp"

#include "qcx/grid/angular_grid.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/molecule/elements.hpp"

#include <Eigen/Cholesky>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace qcx::properties {
namespace {

// The published MK shell radii in units of the van der Waals radius
// (SinghKollman1984).
constexpr std::array<double, 4> kMerzKollmanRadiusFactors = {1.4, 1.6, 1.8, 2.0};

// Eigen matrix -> rank-2 CPU tensor through explicit element copies (the
// production twin of the test fixture's square-only ToTensor; the
// row-major to column-major ordering must not alias).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> ToCpuTensor(
    const Eigen::MatrixXd& matrix) {
    const std::size_t rows = static_cast<std::size_t>(matrix.rows());
    const std::size_t cols = static_cast<std::size_t>(matrix.cols());
    auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({rows, cols});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t i = 0; i < rows; ++i)
    {
        for (std::size_t j = 0; j < cols; ++j)
        {
            (*tensor)(i, j) = matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

// The CHELPG lattice: the axis-aligned cube spanning the molecule plus the
// outer radius, keeping the nodes whose distance to the nearest nucleus
// falls between the inner and outer radii (BrenemanWiberg1990).  Nodes are
// enumerated from the box corner so the lattice positions are exactly
// spacing * k offsets - no drift accumulation.
std::vector<Eigen::Vector3d> ChelpgPoints(const qcx::molecule::Molecule& molecule,
                                          const EspFitOptions& options) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::array<double, 3> minimum = {std::numeric_limits<double>::max(),
                                     std::numeric_limits<double>::max(),
                                     std::numeric_limits<double>::max()};
    std::array<double, 3> maximum = {std::numeric_limits<double>::lowest(),
                                     std::numeric_limits<double>::lowest(),
                                     std::numeric_limits<double>::lowest()};

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t k = 0; k < 3; ++k)
        {
            minimum[k] = std::min(minimum[k], coordinates(atom, k));
            maximum[k] = std::max(maximum[k], coordinates(atom, k));
        }
    }

    std::vector<Eigen::Vector3d> points;
    std::array<double, 3> span{};
    std::array<std::size_t, 3> steps{};

    for (std::size_t k = 0; k < 3; ++k)
    {
        minimum[k] -= options.chelpgOuterRadiusBohr;
        maximum[k] += options.chelpgOuterRadiusBohr;
        span[k] = maximum[k] - minimum[k];
        steps[k] =
            static_cast<std::size_t>(std::floor(span[k] / options.chelpgSpacingBohr + 1e-10));
    }

    for (std::size_t i = 0; i <= steps[0]; ++i)
    {
        for (std::size_t j = 0; j <= steps[1]; ++j)
        {
            for (std::size_t k = 0; k <= steps[2]; ++k)
            {
                const Eigen::Vector3d point = {
                    minimum[0] + static_cast<double>(i) * options.chelpgSpacingBohr,
                    minimum[1] + static_cast<double>(j) * options.chelpgSpacingBohr,
                    minimum[2] + static_cast<double>(k) * options.chelpgSpacingBohr};
                double nearest2 = std::numeric_limits<double>::max();

                for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
                {
                    double distance2 = 0.0;

                    for (std::size_t c = 0; c < 3; ++c)
                    {
                        const double delta =
                            point[static_cast<Eigen::Index>(c)] - coordinates(atom, c);
                        distance2 += delta * delta;
                    }

                    nearest2 = std::min(nearest2, distance2);
                }

                const double nearest = std::sqrt(nearest2);

                if (nearest >= options.chelpgInnerRadiusBohr &&
                    nearest <= options.chelpgOuterRadiusBohr)
                {
                    points.push_back(point);
                }
            }
        }
    }

    return points;
}

// The MK shells: around every atom, four shells at the published radius
// factors times the element's van der Waals radius, each shell sampled on
// one Lebedev orientation set (SinghKollman1984).
qcx::Result<std::vector<Eigen::Vector3d>> MerzKollmanPoints(const qcx::molecule::Molecule& molecule,
                                                            const EspFitOptions& options) {
    auto angular = qcx::grid::AngularGrid::Create(options.mkLebedevPoints);

    if (!angular.has_value())
    {
        return std::unexpected(angular.error());
    }

    const auto& coordinates = molecule.CoordinatesBohr();
    const auto& atoms = molecule.Atoms();
    std::vector<Eigen::Vector3d> points;
    points.reserve(molecule.AtomCount() * kMerzKollmanRadiusFactors.size() * angular->Size());

    for (std::size_t atom = 0; atom < atoms.size(); ++atom)
    {
        const qcx::molecule::ElementData* element =
            qcx::molecule::FindElement(atoms[atom].atomicNumber);

        if (element == nullptr || element->vanDerWaalsRadiusBohr == 0.0)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "MK shells: no van der Waals radius for element " +
                                                  std::to_string(atoms[atom].atomicNumber)});
        }

        for (const double factor : kMerzKollmanRadiusFactors)
        {
            const double radius = factor * element->vanDerWaalsRadiusBohr;

            for (std::size_t i = 0; i < angular->Size(); ++i)
            {
                const std::array<double, 3> direction = angular->Point(i);
                points.emplace_back(coordinates(atom, 0) + radius * direction[0],
                                    coordinates(atom, 1) + radius * direction[1],
                                    coordinates(atom, 2) + radius * direction[2]);
            }
        }
    }

    return points;
}

// The analytic nuclear potential of the molecule at p.
double NuclearPotential(const qcx::molecule::Molecule& molecule, const Eigen::Vector3d& point) {
    const auto& coordinates = molecule.CoordinatesBohr();
    double value = 0.0;

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        const double distance =
            (point -
             Eigen::Vector3d(coordinates(atom, 0), coordinates(atom, 1), coordinates(atom, 2)))
                .norm();
        value += static_cast<double>(molecule.Atoms()[atom].atomicNumber) / distance;
    }

    return value;
}

} // namespace

qcx::Result<EspFitResult> AnalyzeEspCharges(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const Eigen::MatrixXd& density,
                                            EspFitScheme scheme,
                                            const EspFitOptions& options) {
    // The electron count N_e = tr(D S) constrains the total fitted charge
    // (sum_A q_A = sum_A Z_A - N_e); the overlap is needed here anyway for
    // the AO count check.
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const std::size_t n = overlap->Shape()[0];

    if (density.rows() != static_cast<Eigen::Index>(n) ||
        density.cols() != static_cast<Eigen::Index>(n))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "density must be n x n with n the AO count"});
    }

    double electronCount = 0.0;

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            electronCount += density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) *
                             (*overlap)(mu, nu);
        }
    }

    double totalCharge = 0.0;

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        totalCharge += static_cast<double>(atom.atomicNumber);
    }

    totalCharge -= electronCount;

    std::vector<Eigen::Vector3d> points;

    if (scheme == EspFitScheme::kMerzKollman)
    {
        auto mk = MerzKollmanPoints(molecule, options);

        if (!mk.has_value())
        {
            return std::unexpected(mk.error());
        }

        points = std::move(*mk);
    } else
    {
        points = ChelpgPoints(molecule, options);
    }

    // The fit resolves one charge per atom, so fewer points than atoms
    // leaves the normal equations rank-deficient: a
    // partial point set must not silently produce garbage charges.
    if (points.size() < molecule.AtomCount())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "ESP fit: at least one fit point per atom is required"});
    }

    // The electron potential of the density at the points (the integrals
    // module owns the <u|1/|r - p||v> kernel).
    auto densityTensor = ToCpuTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    Eigen::MatrixXd pointMatrix(static_cast<Eigen::Index>(points.size()), 3);

    for (std::size_t p = 0; p < points.size(); ++p)
    {
        pointMatrix(static_cast<Eigen::Index>(p), 0) = points[p][0];
        pointMatrix(static_cast<Eigen::Index>(p), 1) = points[p][1];
        pointMatrix(static_cast<Eigen::Index>(p), 2) = points[p][2];
    }

    auto pointTensor = ToCpuTensor(pointMatrix);

    if (!pointTensor.has_value())
    {
        return std::unexpected(pointTensor.error());
    }

    auto electron = qcx::integrals::BuildElectronPotentialAtPoints(
        molecule, basisSet, *densityTensor, *pointTensor);

    if (!electron.has_value())
    {
        return std::unexpected(electron.error());
    }

    // The augmented normal equations [A e; e^T 0] [q; lambda] = [b; Q]:
    // A_AB = sum_p 1/(|p - R_A| |p - R_B|), b_A = sum_p V_QM(p)/|p - R_A|,
    // the multiplier row enforces sum_A q_A = Q_mol exactly.
    const std::size_t nAtoms = molecule.AtomCount();
    Eigen::MatrixXd system(nAtoms + 1, nAtoms + 1);
    system.setZero();
    Eigen::VectorXd rhs(nAtoms + 1);
    rhs.setZero();
    const auto& coordinates = molecule.CoordinatesBohr();

    for (std::size_t p = 0; p < points.size(); ++p)
    {
        const double vQuantum = (*electron)[p] + NuclearPotential(molecule, points[p]);
        std::vector<double> inverseDistance(nAtoms);

        for (std::size_t a = 0; a < nAtoms; ++a)
        {
            const double distance =
                (points[p] -
                 Eigen::Vector3d(coordinates(a, 0), coordinates(a, 1), coordinates(a, 2)))
                    .norm();
            inverseDistance[a] = 1.0 / distance;
        }

        for (std::size_t a = 0; a < nAtoms; ++a)
        {
            system(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(a)) +=
                inverseDistance[a] * inverseDistance[a];
            rhs(static_cast<Eigen::Index>(a)) += vQuantum * inverseDistance[a];

            for (std::size_t b = a + 1; b < nAtoms; ++b)
            {
                const double entry = inverseDistance[a] * inverseDistance[b];
                system(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) += entry;
                system(static_cast<Eigen::Index>(b), static_cast<Eigen::Index>(a)) += entry;
            }
        }
    }

    for (std::size_t a = 0; a < nAtoms; ++a)
    {
        system(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(nAtoms)) = 1.0;
        system(static_cast<Eigen::Index>(nAtoms), static_cast<Eigen::Index>(a)) = 1.0;
    }

    rhs(static_cast<Eigen::Index>(nAtoms)) = totalCharge;

    Eigen::LDLT<Eigen::MatrixXd> solver(system);

    if (solver.info() != Eigen::Success)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "ESP fit: the normal equations did not solve"});
    }

    // Condition gate: near-coincident atoms (down to
    // 1e-7 Bohr) make the A block nearly rank-1 with entries ~1/r^2 ~
    // 1e14 - LDLT reports Success (no exactly-zero pivot) and the solve
    // returns silently absurd charges. The pivot ratio bounds the
    // condition number from below, so a ratio beyond the threshold
    // PROVES an ill-conditioned system. Well-conditioned fits sit at
    // ratios ~1e3-1e6.
    const Eigen::VectorXd& pivots = solver.vectorD();
    const double maxPivot = pivots.cwiseAbs().maxCoeff();
    const double minPivot = pivots.cwiseAbs().minCoeff();

    if (maxPivot / minPivot > 1e12)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "ESP fit: the normal equations are ill-conditioned "
                                          "(near-coincident atoms?)"});
    }

    const Eigen::VectorXd solution = solver.solve(rhs);

    EspFitResult result;
    result.charges = solution.head(static_cast<Eigen::Index>(nAtoms));
    result.pointCount = points.size();

    // Sanity cap (prop-1): no physically meaningful ESP fit exceeds a
    // few electrons per atom (the constraint row fixes only the SUM),
    // so charges beyond 100 e are a numerical blowup the condition gate
    // missed - reject rather than emit garbage.
    if (result.charges.cwiseAbs().maxCoeff() > 100.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "ESP fit: fitted charges exceed the sanity bound"});
    }

    // The RMS deviation of the fitted potential over the fit points.
    double error2 = 0.0;

    for (std::size_t p = 0; p < points.size(); ++p)
    {
        double vFit = 0.0;

        for (std::size_t a = 0; a < nAtoms; ++a)
        {
            const double distance =
                (points[p] -
                 Eigen::Vector3d(coordinates(a, 0), coordinates(a, 1), coordinates(a, 2)))
                    .norm();
            vFit += result.charges(static_cast<Eigen::Index>(a)) / distance;
        }

        const double deviation = vFit - ((*electron)[p] + NuclearPotential(molecule, points[p]));
        error2 += deviation * deviation;
    }

    result.rmsError = points.empty() ? 0.0 : std::sqrt(error2 / static_cast<double>(points.size()));
    return result;
}

} // namespace qcx::properties
