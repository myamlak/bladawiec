#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/xc_kernel_action.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <excgrid/grid_derivatives.hpp>
#include <vector>

namespace qcx::grid {

class XcGridEngine;

/// The derivative of the exchange-correlation potential with respect to the
/// nuclear coordinates, in the occupied-virtual MO block.
///
/// The potential is differentiated at fixed density: the density matrix does
/// not move with the nuclei, so every term comes from an explicit coordinate
/// channel - the quadrature weight, the point position, the AO values and their
/// first derivatives, and through them the density gradient invariants and the
/// kinetic-energy density. This is the right-hand side of the coupled-perturbed
/// response equations, not a force.
///
/// The block is the restricted potential V = (V_alpha + V_beta) / 2 in the
/// basis of the reference MOs, so entry (coordinate q, occupied i, virtual a)
/// is sum_mu nu C_mu a (dV_mu nu / dR_q) C_nu i. The nuclear coordinate
/// ordering is the caller's: q = 3 * atom + axis.
///
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct XcKernelDerivativeBlock {
    std::size_t numOccupied = 0; ///< The occupied count o.
    std::size_t numVirtual = 0; ///< The virtual count v.
    std::size_t nuclearCoordinateCount = 0; ///< 3N.
    /// The block, flattened [q * o * v + i * v + a]; dV/dR at fixed density.
    std::vector<double> values;

    /// One entry of the block.
    /// \param coordinate The nuclear coordinate q = 3 * atom + axis.
    /// \param occupied The occupied orbital index.
    /// \param virtualOrbital The virtual orbital index.
    /// \returns The entry.
    /// \ingroup qcx-grid
    [[nodiscard]] double At(std::size_t coordinate,
                            std::size_t occupied,
                            std::size_t virtualOrbital) const {
        return values[coordinate * numOccupied * numVirtual + occupied * numVirtual +
                      virtualOrbital];
    }
};

/// Evaluates the fixed-density derivative of the XC potential in the
/// occupied-virtual MO block.
///
/// The walk mirrors the reference-potential assembly: the closed-shell density
/// splits into D/2 per spin, the screening is frozen at that density, the
/// second-derivative tier is contracted point by point, and every coordinate
/// channel is carried explicitly. The kernel's second derivatives give the
/// response of the first derivatives to the components' change along the
/// displacement, and the potential's own AO-motion terms come from the AO
/// gradient and Hessian.
///
/// The provider must be built for the same geometry and grid parameters as the
/// engine (excgrid::CreateGridDerivativeProvider is the only way to obtain
/// one, so the pairing cannot go wrong by accident). The functional must carry
/// a second-derivative tier spanning everything it reads; a functional without
/// one refuses here rather than returning a zero block.
/// \param engine The engine whose grid, evaluator, functional and envelopes
/// the walk uses.
/// \param provider The grid's geometric derivative provider.
/// \param moCoefficients The MO coefficients, AOCount() x (o + v), occupied
/// columns first.
/// \param density The closed-shell spin-summed density, AOCount() x AOCount().
/// \param numOccupied The occupied count o.
/// \param numVirtual The virtual count v.
/// \param settings The screening settings; the grid's own trim settings live in
/// the engine.
/// \returns The block, or kInvalidArgument for a shape mismatch or a provider
/// refusal, and kUnimplemented for a functional without a usable tier.
/// \ingroup qcx-grid
qcx::Result<XcKernelDerivativeBlock> EvaluateXcKernelDerivativeBlock(
    const XcGridEngine& engine,
    const excgrid::GridDerivativeProvider& provider,
    const Eigen::MatrixXd& moCoefficients,
    const Eigen::MatrixXd& density,
    std::size_t numOccupied,
    std::size_t numVirtual,
    const XcKernelSettings& settings = {});

} // namespace qcx::grid
