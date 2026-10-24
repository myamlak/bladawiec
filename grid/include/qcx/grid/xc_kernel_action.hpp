#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/grid/xc_grid_engine.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <excgrid/grid.hpp>
#include <excgrid/kernel.hpp>
#include <span>
#include <vector>

namespace qcx::grid {

/// The settings a kernel action integrates under.
///
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct XcKernelSettings {
    /// The shell significance threshold the per-point selection uses.
    ///
    /// The selection is frozen at the reference density when the action is
    /// created, so the action is the second derivative of the REFERENCE-SCREENED
    /// energy, not of the dense one. A positive tolerance is the derivative of
    /// what the matching screened energy path computed; zero screens nothing
    /// whose weight and decay are both nonzero, which reproduces the dense
    /// entry points to rounding; a negative tolerance keeps every shell and is
    /// the setting the finite-difference gate uses.
    double screeningTolerance = 0.0;
};

/// The exchange-correlation kernel's action on the closed-shell
/// occupied-virtual block.
///
/// The XC part of the orbital Hessian is the second functional derivative of
/// E_xc contracted twice with the density response to an orbital rotation. For
/// a closed-shell reference with spin densities D_s = D/2, one unit of the
/// amplitude x_ai changes the components at a point by
///     d rho_s = 2 phi_a phi_i,
///     d sigma   = 4 grad(rho_s) . grad(u_ai)   (all three invariants),
///     d tau_s   = grad(phi_a) . grad(phi_i),
/// with u_ai = phi_a phi_i. The kernel returns Q delta q, and the action is
///     y_ai = 1/4 sum_p w_p sum_k delta q[k, ai] (Q delta q)[k].
///
/// The 1/4 is the closed-shell convention: the true energy second derivative
/// with respect to the amplitudes is four times this action, which is the
/// relation the orbital-Hessian finite-difference gates pin.
///
/// The sigma components are quadratic in the density, so the complete second
/// derivative also carries the functional's first derivative contracted with
/// the curvature of sigma:
///     sum_p w_p 2 (d e / d sigma_aa + d e / d sigma_ab + d e / d sigma_bb)
///          (delta g . delta g),
/// with delta g = 2 grad(u_ai). The kernel's matrix cannot supply it - it is
/// differentiated with respect to the components, not with respect to the
/// density - so the action adds it explicitly. It vanishes for an LDA and is
/// the GGA term that a finite-difference gate fails without.
///
/// The shell screening is frozen at the reference density and at the settings
/// tolerance. Recomputing the selection from the trial vector would make the
/// operator depend on the vector it acts on; freezing it makes the operator the
/// exact second derivative of the energy the reference screening defined.
/// \\ingroup qcx-grid
class XcKernelAction {
public:
    /// Creates the action for a grid engine.
    /// \param engine The engine whose grid, evaluator, functional and envelopes
    /// the action walks. It must outlive the action.
    /// \param moCoefficients The MO coefficients, AOCount() x (o + v), in the
    /// evaluator's AO order; the occupied columns come first.
    /// \param density The closed-shell spin-summed density, AOCount() x
    /// AOCount(), in the same order.
    /// \param numOccupied The number of occupied orbitals o.
    /// \param numVirtual The number of virtual orbitals v.
    /// \param settings The screening settings.
    /// \returns The action, or kInvalidArgument for a shape mismatch and
    /// kUnimplemented for a functional without a second-derivative tier or one
    /// whose tier does not span the components it reads (the message names
    /// them).
    /// \\ingroup qcx-grid
    static qcx::Result<XcKernelAction> Create(const XcGridEngine& engine,
                                              const Eigen::MatrixXd& moCoefficients,
                                              const Eigen::MatrixXd& density,
                                              std::size_t numOccupied,
                                              std::size_t numVirtual,
                                              const XcKernelSettings& settings = {});

    /// Creates the action from the pieces a grid engine walks.
    ///
    /// The engine form delegates here. This form exists so a caller can pair
    /// the action with its own grid and functional - a test functional, or a
    /// grid built elsewhere - without constructing an engine.
    /// \param blocks The grid blocks. They must outlive the action.
    /// \param evaluator The AO evaluator the blocks were built for. It must
    /// outlive the action.
    /// \param functional The functional kernel. It must outlive the action.
    /// \param envelopes The screening envelopes, in the evaluator's shell
    /// order. They must outlive the action.
    /// \param moCoefficients The MO coefficients, AOCount() x (o + v).
    /// \param density The closed-shell density, AOCount() x AOCount().
    /// \param numOccupied The number of occupied orbitals.
    /// \param numVirtual The number of virtual orbitals.
    /// \param settings The screening settings.
    /// \returns The action, or the same refusals as the engine form.
    /// \\ingroup qcx-grid
    static qcx::Result<XcKernelAction> Create(std::span<const excgrid::Block> blocks,
                                              const AoEvaluator& evaluator,
                                              const excgrid::XcFunctional& functional,
                                              std::span<const ShellEnvelope> envelopes,
                                              const Eigen::MatrixXd& moCoefficients,
                                              const Eigen::MatrixXd& density,
                                              std::size_t numOccupied,
                                              std::size_t numVirtual,
                                              const XcKernelSettings& settings = {});

    /// Move construction.
    XcKernelAction(XcKernelAction&&) = default;
    /// Move assignment.
    /// \returns This action.
    XcKernelAction& operator=(XcKernelAction&&) = default;
    XcKernelAction(const XcKernelAction&) = delete;
    /// Copying is deleted: the action owns its frozen selection.
    XcKernelAction& operator=(const XcKernelAction&) = delete;
    ~XcKernelAction() = default;

    /// Applies the action to one amplitude vector.
    /// \param amplitudes The trial vector, Dimension() entries, occupied-major
    /// (i * numVirtual + a).
    /// \param result Receives the action, Dimension() entries, same order;
    /// overwritten.
    /// \returns kOk, or kInvalidArgument for a size mismatch.
    /// \\ingroup qcx-grid
    qcx::Result<void> Apply(std::span<const double> amplitudes, std::span<double> result) const;

    /// The operator dimension, numOccupied() * numVirtual().
    /// \returns The dimension.
    /// \\ingroup qcx-grid
    [[nodiscard]] std::size_t Dimension() const noexcept {
        return _numOccupied * _numVirtual;
    }

    /// The number of occupied orbitals.
    /// \returns The occupied count.
    /// \\ingroup qcx-grid
    [[nodiscard]] std::size_t NumOccupied() const noexcept {
        return _numOccupied;
    }

    /// The number of virtual orbitals.
    /// \returns The virtual count.
    /// \\ingroup qcx-grid
    [[nodiscard]] std::size_t NumVirtual() const noexcept {
        return _numVirtual;
    }

private:
    XcKernelAction(std::span<const excgrid::Block> blocks,
                   const AoEvaluator& evaluator,
                   const excgrid::XcFunctional& functional,
                   std::span<const ShellEnvelope> envelopes,
                   Eigen::MatrixXd moCoefficients,
                   Eigen::MatrixXd spinDensity,
                   std::vector<double> shellWeights,
                   std::vector<excgrid::Component> active,
                   std::size_t numOccupied,
                   std::size_t numVirtual,
                   std::size_t aoCount,
                   double screeningTolerance,
                   bool needsGradient) noexcept;

    std::span<const excgrid::Block> _blocks;
    const AoEvaluator* _evaluator;
    const excgrid::XcFunctional* _functional;
    std::span<const ShellEnvelope> _envelopes;
    Eigen::MatrixXd _moCoefficients; ///< nAO x (o + v), row-major access by (mu, orbital).
    Eigen::MatrixXd _spinDensity; ///< D/2.
    std::vector<double> _shellWeights;
    std::vector<excgrid::Component> _active; ///< Identifier order.
    std::size_t _numOccupied;
    std::size_t _numVirtual;
    std::size_t _numOrbitals;
    std::size_t _aoCount;
    double _screeningTolerance;
    bool _needsGradient;
};

} // namespace qcx::grid
