#include "qcx/properties/charges.hpp"

#include "qcx/grid/ao_evaluator.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace qcx::properties {
namespace {

// The value of rho(r) = sum_mu nu D_mu nu phi_mu(r) phi_nu(r) at every
// grid point, in grid order.  The evaluator fills the AOs in the basis
// ordering (ao_evaluator.hpp), so the density matrix contracts directly.
// Validates D against the evaluator's AO count; the per-point AO values
// are also handed back when the caller needs them (Hirshfeld fragment
// densities reuse the same evaluation).
struct DensityOnGrid {
    std::vector<double> molecular; ///< rho(r_i), one value per grid point.
    std::vector<std::vector<double>> atomic; ///< [atom][point] fragment rho_A(r_i).
    std::vector<double> promolecular; ///< [point] sum_A rho_A(r_i).
};

qcx::Result<DensityOnGrid> DensitiesOnGrid(const qcx::grid::MolecularGrid& grid,
                                           const qcx::grid::AoEvaluator& evaluator,
                                           const Eigen::MatrixXd& density,
                                           const Eigen::MatrixXd& fragment,
                                           const std::vector<AoRange>& aoRanges) {
    const std::size_t n = evaluator.AOCount();

    if (density.rows() != n || density.cols() != n)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "density must be n x n with n the AO count"});
    }

    // A zero-row fragment matrix skips the per-atom pass (the Voronoi
    // partition needs no promolecular densities); otherwise the fragment
    // must be n x n assembled block-diagonal over aoRanges.
    if (fragment.rows() != 0 && (fragment.rows() != n || fragment.cols() != n))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "fragment densities must be n x n with n the AO count"});
    }

    // The caller-supplied ranges index the size-n AO vector directly, so a
    // range leaking past n is an out-of-bounds read, not a graceful error.
    for (const AoRange& range : aoRanges)
    {
        if (range.firstFunction + range.functionCount > n)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "aoRanges must stay within the AO count"});
        }
    }

    DensityOnGrid result;
    result.molecular.assign(grid.Size(), 0.0);
    result.atomic.assign(aoRanges.size(), std::vector<double>(grid.Size(), 0.0));
    result.promolecular.assign(grid.Size(), 0.0);

    std::vector<double> ao(n);

    for (std::size_t i = 0; i < grid.Size(); ++i)
    {
        evaluator.Evaluate(grid.Point(i), std::span<double>(ao));

        double rho = 0.0;

        for (std::size_t mu = 0; mu < n; ++mu)
        {
            for (std::size_t nu = 0; nu < n; ++nu)
            {
                rho += density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) *
                       ao[mu] * ao[nu];
            }
        }

        result.molecular[i] = rho;

        for (std::size_t atom = 0; atom < aoRanges.size(); ++atom)
        {
            double rhoA = 0.0;

            const std::size_t rangeStart = aoRanges[atom].firstFunction;
            const std::size_t rangeEnd = rangeStart + aoRanges[atom].functionCount;

            for (std::size_t mu = rangeStart; mu < rangeEnd; ++mu)
            {
                for (std::size_t nu = rangeStart; nu < rangeEnd; ++nu)
                {
                    rhoA += fragment(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu)) *
                            ao[mu] * ao[nu];
                }
            }

            result.atomic[atom][i] = rhoA;
            result.promolecular[i] += rhoA;
        }
    }

    return result;
}

} // namespace

qcx::Result<Eigen::VectorXd> AnalyzeHirshfeld(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::grid::MolecularGrid& grid,
    // (density, fragmentAlpha) are the total and per-species densities -
    // distinct quantities, fixed call order.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& fragmentAlpha,
    const Eigen::MatrixXd& fragmentBeta,
    const std::vector<AoRange>& aoRanges) {
    if (aoRanges.size() != molecule.AtomCount())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "aoRanges must hold one range per atom"});
    }

    auto evaluator = qcx::grid::AoEvaluator::Create(molecule, basisSet);

    if (!evaluator.has_value())
    {
        return std::unexpected(evaluator.error());
    }

    // Both fragment channels must be n x n (their blocks index aoRanges in
    // the per-atom pass below) BEFORE the sum: an Eigen add of mismatched
    // matrices is undefined behavior, so the shape check cannot wait for
    // DensitiesOnGrid.
    const std::size_t n = evaluator->AOCount();

    if (fragmentAlpha.rows() != n || fragmentAlpha.cols() != n || fragmentBeta.rows() != n ||
        fragmentBeta.cols() != n)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "fragment densities must be n x n with n the AO count"});
    }

    const Eigen::MatrixXd fragment = fragmentAlpha + fragmentBeta;
    auto densities = DensitiesOnGrid(grid, *evaluator, density, fragment, aoRanges);

    if (!densities.has_value())
    {
        return std::unexpected(densities.error());
    }

    // Q_A = Z_A - sum_i w_i (rho_A(r_i) / rho_0(r_i)) rho(r_i); a zero
    // promolecular density (exponentially far regions) contributes no
    // weight.  The grid weights already include the Becke partition, so
    // the sum is a plain quadrature of the ratio-weighted integral.
    Eigen::VectorXd charges =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(molecule.AtomCount()));

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        double population = 0.0;

        for (std::size_t i = 0; i < grid.Size(); ++i)
        {
            const double promolecular = densities->promolecular[i];

            if (promolecular <= 0.0)
            {
                continue;
            }

            const double weight = densities->atomic[atom][i] / promolecular;
            population += grid.Weight(i) * weight * densities->molecular[i];
        }

        charges(static_cast<Eigen::Index>(atom)) = molecule.Atoms()[atom].atomicNumber - population;
    }

    return charges;
}

qcx::Result<Eigen::VectorXd> AnalyzeVoronoi(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet,
                                            const qcx::grid::MolecularGrid& grid,
                                            const Eigen::MatrixXd& density) {
    auto evaluator = qcx::grid::AoEvaluator::Create(molecule, basisSet);

    if (!evaluator.has_value())
    {
        return std::unexpected(evaluator.error());
    }

    // The nearest-nucleus cell partition needs no fragment densities;
    // reuse the shared evaluator pass with an empty fragment matrix.
    const std::vector<AoRange> emptyRanges;
    auto densities = DensitiesOnGrid(grid, *evaluator, density, Eigen::MatrixXd(), emptyRanges);

    if (!densities.has_value())
    {
        return std::unexpected(densities.error());
    }

    Eigen::VectorXd charges =
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(molecule.AtomCount()));

    for (std::size_t i = 0; i < grid.Size(); ++i)
    {
        const std::array<double, 3>& point = grid.Point(i);
        std::size_t nearest = 0;
        double nearestSquared = std::numeric_limits<double>::infinity();

        for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
        {
            double distanceSquared = 0.0;

            for (std::size_t c = 0; c < 3; ++c)
            {
                const double delta = point[c] - molecule.CoordinatesBohr()(atom, c);

                distanceSquared += delta * delta;
            }

            if (distanceSquared < nearestSquared)
            {
                nearest = atom;
                nearestSquared = distanceSquared;
            }
        }

        charges(static_cast<Eigen::Index>(nearest)) -= grid.Weight(i) * densities->molecular[i];
    }

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        charges(static_cast<Eigen::Index>(atom)) += molecule.Atoms()[atom].atomicNumber;
    }

    return charges;
}

} // namespace qcx::properties
