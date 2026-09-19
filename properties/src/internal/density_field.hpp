#pragma once

// The spin-summed electron density field of the Bader QTAIM increment:
// rho, grad rho, and the symmetric Hessian at a point, from the grid
// module's analytic AO derivatives contracted with the spin-summed AO
// density D = P_alpha + P_beta (Bohr units).  The AO
// ordering is the integrals module's basis ordering (the AoEvaluator doc
// comment is the convention authority), so the density contracts directly.
// Module-private helper (no Doxygen obligation); the
// basin-volume increment reuses this interface.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace qcx::properties::internal {

// The electron density field of one spin-summed AO density.
class DensityField {
public:
    // The field values at one point: rho (e/bohr^3), grad rho (e/bohr^4),
    // and the symmetric Hessian (e/bohr^5) packed (xx, xy, xz, yy, yz, zz).
    struct Values {
        double rho;
        std::array<double, 3> gradient;
        std::array<double, 6> hessian;
    };

    // Creates the field.  The density must be n x n with n the AO count
    // (the caller, AnalyzeQtaim, verifies the shape).
    static Result<DensityField> Create(const qcx::molecule::Molecule& molecule,
                                       const qcx::basisset::BasisSet& basisSet,
                                       const Eigen::MatrixXd& density) {
        auto evaluator = qcx::grid::AoEvaluator::Create(molecule, basisSet);

        if (!evaluator.has_value())
        {
            return std::unexpected(evaluator.error());
        }

        return DensityField(std::move(*evaluator), density);
    }

    // The AO count of the density.
    [[nodiscard]] std::size_t AOCount() const noexcept {
        return _aoCount;
    }

    // The field values at one point (Bohr):
    //     g_j = sum_mu,nu D_mu,nu (phi_nu dphi_mu/dx_j + phi_mu dphi_nu/dx_j)
    //     H_jk = sum_mu,nu D_mu,nu (phi_nu d2phi_mu/dxjdxk
    //            + dphi_mu/dxj dphi_nu/dxk + dphi_mu/dxk dphi_nu/dxj
    //            + phi_mu d2phi_nu/dxjdxk)
    // O(n^2) per point with the AO ordering the evaluator writes (the
    // spin-summed density contract).  The per-point scratch buffers are
    // call-local, so the method is re-entrant and the const contract is
    // honest: concurrent EvaluateAll calls on one field do not share state.
    Values EvaluateAll(const std::array<double, 3>& pointBohr) const {
        // Scratch: the evaluator writes values (n), gradients (3 per AO,
        // mu-major) and Hessians (6 per AO, mu-major) at the point.
        std::vector<double> phi(_aoCount);
        std::vector<double> gradients(3 * _aoCount);
        std::vector<double> hessians(6 * _aoCount);

        _evaluator->EvaluateDerivatives(pointBohr, phi, gradients, hessians);

        const Eigen::Index n = static_cast<Eigen::Index>(_aoCount);
        Values values{};

        for (Eigen::Index mu = 0; mu < n; ++mu)
        {
            for (Eigen::Index nu = 0; nu < n; ++nu)
            {
                const double dMuNu = _density(mu, nu);
                const double phiMu = phi[static_cast<std::size_t>(mu)];
                const double phiNu = phi[static_cast<std::size_t>(nu)];
                values.rho += dMuNu * phiMu * phiNu;

                for (int j = 0; j < 3; ++j)
                {
                    const std::size_t muJ =
                        3 * static_cast<std::size_t>(mu) + static_cast<std::size_t>(j);
                    const std::size_t nuJ =
                        3 * static_cast<std::size_t>(nu) + static_cast<std::size_t>(j);
                    values.gradient[j] += dMuNu * (phiNu * gradients[muJ] + phiMu * gradients[nuJ]);
                }

                for (int j = 0; j < 3; ++j)
                {
                    for (int k = j; k < 3; ++k)
                    {
                        // The packed (xx, xy, xz, yy, yz, zz) index of (j, k).
                        const int pack = j == 0 ? k : (j == 1 ? k + 2 : 5);
                        const std::size_t muPack =
                            6 * static_cast<std::size_t>(mu) + static_cast<std::size_t>(pack);
                        const std::size_t nuPack =
                            6 * static_cast<std::size_t>(nu) + static_cast<std::size_t>(pack);
                        const std::size_t muJ =
                            3 * static_cast<std::size_t>(mu) + static_cast<std::size_t>(j);
                        const std::size_t muK =
                            3 * static_cast<std::size_t>(mu) + static_cast<std::size_t>(k);
                        const std::size_t nuJ =
                            3 * static_cast<std::size_t>(nu) + static_cast<std::size_t>(j);
                        const std::size_t nuK =
                            3 * static_cast<std::size_t>(nu) + static_cast<std::size_t>(k);
                        values.hessian[pack] +=
                            dMuNu * (phiNu * hessians[muPack] + gradients[muJ] * gradients[nuK] +
                                     gradients[muK] * gradients[nuJ] + phiMu * hessians[nuPack]);
                    }
                }
            }
        }

        return values;
    }

private:
    DensityField(qcx::grid::AoEvaluator evaluator, Eigen::MatrixXd density) :
        _evaluator(std::move(evaluator)), _density(std::move(density)),
        _aoCount(_evaluator->AOCount()) {}

    std::optional<qcx::grid::AoEvaluator> _evaluator;
    Eigen::MatrixXd _density;
    std::size_t _aoCount;
};

} // namespace qcx::properties::internal
