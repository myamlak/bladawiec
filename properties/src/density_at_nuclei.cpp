// Electron density at the nuclear positions:
// rho(R_A) = sum_mu,nu D_mu,nu phi_mu(R_A) phi_nu(R_A) via the grid
// module's point evaluator.  The evaluator's AO ordering is the integrals
// module's basis ordering (its doc comment is the convention authority),
// so the density contracts directly.  No quadrature is involved - the
// value is a point evaluation, which is exactly why it pins the deferred
// Bader increment's r -> 0 seeds.

#include "qcx/properties/density_at_nuclei.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::properties {

qcx::Result<Eigen::VectorXd> AnalyzeDensityAtNuclei(const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet,
                                                    const Eigen::MatrixXd& density) {
    auto evaluator = qcx::grid::AoEvaluator::Create(molecule, basisSet);

    if (!evaluator.has_value())
    {
        return std::unexpected(evaluator.error());
    }

    const Eigen::Index n = static_cast<Eigen::Index>(evaluator->AOCount());

    if (density.rows() != n || density.cols() != n)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the density must be n x n with n the AO count"});
    }

    const auto& coordinates = molecule.CoordinatesBohr();
    Eigen::VectorXd values(static_cast<Eigen::Index>(molecule.AtomCount()));
    std::vector<double> phi(static_cast<std::size_t>(n));

    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        const std::array<double, 3> point = {
            coordinates(atomIndex, 0), coordinates(atomIndex, 1), coordinates(atomIndex, 2)};
        evaluator->Evaluate(point, phi);

        double rho = 0.0;

        for (Eigen::Index mu = 0; mu < n; ++mu)
        {
            for (Eigen::Index nu = 0; nu < n; ++nu)
            {
                rho += density(mu, nu) * phi[static_cast<std::size_t>(mu)] *
                       phi[static_cast<std::size_t>(nu)];
            }
        }

        values(static_cast<Eigen::Index>(atomIndex)) = rho;
    }

    return values;
}

} // namespace qcx::properties
