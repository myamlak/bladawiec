#pragma once

// The per-point quantities both XC walks assemble, in one place because the
// gradient's own consistency check is exact only when its density is
// bit-identical to the energy path's: a second copy of this arithmetic would
// put a rounding between the two numbers that check compares, and a check with
// a tolerance of its own cannot tell a rounding from a defect.

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <span>
#include <vector>

namespace qcx::grid::internal {

/// One spin's density and density gradient at a grid point.
struct PointDensity {
    double rho = 0.0;
    std::array<double, 3> gradient{};
};

/// A row-major copy of a density matrix.
///
/// ShellDensityWeights walks rows and Eigen's default storage is column-major,
/// so the screening weights are taken from a copy laid out the way the walk
/// reads it.
/// \param density The density matrix.
/// \returns AOCount()^2 entries in row-major order.
inline std::vector<double> FlattenRowMajor(const Eigen::MatrixXd& density) {
    const Eigen::Index n = density.rows();
    std::vector<double> flat(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
        flat.data(), n, n);
    map = density;

    return flat;
}

/// Contracts one spin's density matrix with a point's AO values and gradients,
/// over the selected shells' AO slots only.
///
///     rho      = sum_mu nu D_mu nu phi_mu phi_nu
///     grad rho = sum_mu nu D_mu nu (grad phi_mu phi_nu + phi_mu grad phi_nu)
///
/// Both sums factor through (D phi)_mu and (D grad phi)_mu, so one point costs
/// O(nAO^2) rather than O(nAO^3). The double-sum form is written out, so the
/// result is exact for a non-symmetric D as well. Dropping a shell drops its
/// density-matrix row and column along with its AO values, which is what the
/// significance test argued for.
///
/// Only the selected slots are read, as the slot contract requires: the fetch
/// writes the selected range of a point's slice and leaves the rest of it
/// holding whatever the slice had.
/// \param density The spin density matrix, AOCount() x AOCount().
/// \param values The point's AO values, size AOCount().
/// \param gradients The point's AO gradients, size 3 * AOCount(), mu-major.
/// \param selectedAos The AO slots the selection kept.
/// \param valueContraction Scratch of size AOCount(); the selected slots are
/// written with (D phi)_mu and read back by callers that need them.
/// \param gradientContraction Scratch of size 3 * AOCount(); the selected slots
/// are written with (D grad phi)_mu in the same layout as \p gradients.
/// \returns The point's density and density gradient.
inline PointDensity ContractSelectedSpinDensity(const Eigen::MatrixXd& density,
                                                std::span<const double> values,
                                                std::span<const double> gradients,
                                                std::span<const std::size_t> selectedAos,
                                                std::vector<double>& valueContraction,
                                                std::vector<double>& gradientContraction) {
    // (D phi)_mu and (D grad phi)_mu over the selected block.
    for (const std::size_t mu : selectedAos)
    {
        double contracted = 0.0;
        std::array<double, 3> contractedGradient{};

        for (const std::size_t nu : selectedAos)
        {
            const double element =
                density(static_cast<Eigen::Index>(mu), static_cast<Eigen::Index>(nu));
            contracted += element * values[nu];

            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                contractedGradient[axis] += element * gradients[3 * nu + axis];
            }
        }

        valueContraction[mu] = contracted;

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            gradientContraction[3 * mu + axis] = contractedGradient[axis];
        }
    }

    PointDensity result;

    for (const std::size_t mu : selectedAos)
    {
        result.rho += values[mu] * valueContraction[mu];

        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            result.gradient[axis] += gradients[3 * mu + axis] * valueContraction[mu] +
                                     values[mu] * gradientContraction[3 * mu + axis];
        }
    }

    return result;
}

} // namespace qcx::grid::internal
