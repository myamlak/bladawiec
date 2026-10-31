#pragma once

#include "qcx/error.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/grid/xc_grid_engine.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <excgrid/grid.hpp>
#include <excgrid/grid_derivatives.hpp>
#include <excgrid/kernel.hpp>
#include <span>

namespace qcx::grid {

/// What one gradient assembly counted.
///
/// The accounting the energy path already keeps, in the same columns, so the
/// two walks are comparable column for column: a per-point claim has to pay for
/// the setup columns first. Zero-weight points are skipped, are not counted, and
/// are not visited. The dense columns are the same quantities computed as if no
/// shell had ever been dropped, which is what turns the counters into a ratio.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct XcGradientCounts {
    std::size_t points = 0; ///< Points visited (zero-weight points are skipped).
    std::size_t batches = 0; ///< Batch passes the grid was walked in.
    std::size_t densityWeightTerms = 0; ///< |D| entries summed for the weights.
    std::size_t shellTests = 0; ///< Significance tests run: points x shells considered.
    std::size_t shellKeeps = 0; ///< Those tests that kept their shell.
    std::size_t aoFetches = 0; ///< AO evaluations issued: one per live point.
    std::size_t aoSlotsEvaluated = 0; ///< AO slots evaluated, selected shells only.
    std::size_t aoSlotsDense = 0; ///< points x AOCount(), the cost with no screening.
    std::size_t contractionPairs = 0; ///< Terms the density contraction read.
    std::size_t contractionPairsDense = 0; ///< points x 2 x AOCount()^2, with no screening.
    // The AO-derivative terms the gradient accumulated: one per kept AO slot per
    // spin, since each kept slot's motion reaches exactly one atom's row.
    std::size_t aoDerivativeTerms = 0; ///< Kept slots whose motion entered the gradient.
    std::size_t aoDerivativeTermsDense = 0; ///< points x AOCount() x 2, with no screening.
    // The grid's own motion, as the derivative provider hands it over: the whole
    // (point, nuclear coordinate) weight and coordinate arrays are read, so these
    // columns are dense by construction and count what the walk paid for the
    // right to read them.
    std::size_t weightDerivativeTerms = 0; ///< (point, nuclear coordinate) weight entries read.
    std::size_t coordinateDerivativeTerms = 0; ///< (point, direction, coordinate) entries read.
    // The second-order counterpart of the two columns above. The provider fills
    // each block's curvature arrays whole, so the second-order walk reads them
    // whole and these columns are its bill rather than a screened fraction.
    std::size_t weightCurvatureTerms = 0; ///< (point, q, r) weight entries read.
    std::size_t coordinateCurvatureTerms = 0; ///< (point, direction, q, r) entries read.
    // The density channel's second response: one entry per atom pair per
    // direction pair at each point, and the kernel contractions that supply its
    // curvature. The dense column is the same as if no shell had ever been
    // dropped, which is what turns these two into a ratio. The call column is
    // the channel's visible cost against the first-order walk: a functional that
    // reads a density now fetches the AO Hessian tier it did not fetch before.
    std::size_t electronicCurvatureTerms = 0; ///< Atom-pair second-response entries filled.
    std::size_t electronicCurvatureTermsDense = 0; ///< points x atomCount^2 x 9, no screening.
    std::size_t kernelSecondDerivativeCalls = 0; ///< Second-derivative contractions issued.
};

/// One grid integration's exchange-correlation gradient, at fixed density.
///
/// The density matrices are held fixed: `gradient` is the derivative of
/// E_xc(R; D) with respect to the nuclear positions, so the density matrix's own
/// response to the displacement - the SCF's business, not this walk's - is not
/// in it. It is the contribution the gradient assembly owes the total nuclear
/// gradient, and the whole of what the grid and its functional contribute to it.
///
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct XcGradientEvaluation {
    /// dE_xc/dR, 3N entries in Hartree/Bohr, atom-major: entry 3*i + d is
    /// direction d of atom i, the same ordering the nuclear coordinates use.
    Eigen::VectorXd gradient;
    /// The energy this walk integrated, Hartree: accumulated point by point from
    /// the same slice the gradient read.
    ///
    /// Reported on its own because it is the one number that separates the two
    /// ways a gradient can be wrong: a gradient that truncates a different point
    /// set or screens by a different rule differs from the energy path here,
    /// whereas a gradient that agrees with finite differences while the energy
    /// does not move with it differs nowhere else.
    double energy = 0.0;
    /// The thresholds this walk ran under, echoed back rather than assumed: a
    /// caller comparing them against the energy path's own is asserting the
    /// reuse instead of reading two call sites and trusting them to agree.
    XcIntegrationThresholds thresholds;
    /// The counted cost of this assembly.
    XcGradientCounts counts;
};

/// Assembles the exchange-correlation contribution to the nuclear gradient.
///
/// One walk of the same blocks the energy path walks, under the same thresholds,
/// that accumulates three contributions at each point:
///
///   - the quadrature weight's motion, `e * dw/dR`. The Becke partition weighs
///     each point by its distance to every atom, so this term is the reason a
///     gradient cannot reuse the energy's point set unchanged.
///   - the point's motion, `w * (grad e) . dx/dR`, where grad e is the spatial
///     gradient of the composite integrand. For a functional that consumes the
///     density gradients this needs the density field's second derivative, and
///     therefore the AO tier's second derivative; for one that consumes the
///     kinetic-energy density it also carries that field's own derivative, the
///     same contracted gradient against the AO tier's second derivative.
///   - the atomic orbitals' motion: the density's own chain rule through the AO
///     functions, which move with the atoms they sit on. Each atom's row takes
///     the density's contraction with the AO gradients and with their curvature,
///     over the shells that point kept, and the functional's partials weight
///     them [Helgaker2000]. The kinetic-energy density's motion is the same
///     contraction weighted by vtau; the half in its definition cancels the two
///     density index slots, so its row carries coefficient one rather than two.
///
/// And nothing else: the density matrix is fixed, so its response to the
/// displacement is the SCF's business rather than this walk's.
///
/// Every point's data comes from one fetch, and the shells the significance test
/// keeps are the only ones any of the four terms reads, so the gradient's
/// neglected terms are the ones the energy path already neglected.
///
/// \param blocks The grid blocks, in the order the energy path walks them.
/// \param provider The grid's geometric derivatives for these blocks. It must be
/// the provider built for the geometry and the build parameters these blocks came
/// from, and its derivative arrays are indexed by nuclear coordinate in the
/// caller's own 3N order.
/// \param evaluator The AO evaluator the blocks' points are evaluated with.
/// \param functional The functional kernel; its sigma and kinetic-energy-density
/// conventions are the energy path's, and every seven collinear components are
/// supplied to it at each point.
/// \param envelopes The screening envelopes, in the evaluator's shell order.
/// \param thresholds The energy path's truncation and screening thresholds.
/// \param batch The batch granularity; it changes no number, only how the grid is
/// walked.
/// \param densityAlpha The alpha spin density, AOCount() x AOCount(), in the
/// evaluator's AO order.
/// \param densityBeta The beta spin density, same shape, order, and contract.
/// \returns The gradient, the energy this walk integrated, the thresholds it ran
/// under and the counted cost, or an Error: kInvalidArgument for a shape
/// mismatch, and the provider's own refusal, by name, when it declines a block
/// rather than answering it with zeros.
/// \ingroup qcx-grid
qcx::Result<XcGradientEvaluation> EvaluateXcGradient(
    std::span<const excgrid::Block> blocks,
    const excgrid::GridDerivativeProvider& provider,
    const AoEvaluator& evaluator,
    const excgrid::XcFunctional& functional,
    std::span<const ShellEnvelope> envelopes,
    const XcIntegrationThresholds& thresholds,
    const XcBatchSettings& batch,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta);

/// Assembles the gradient on an engine's own grid, evaluator and functional.
///
/// The form a run uses: the engine already holds the blocks, the AO tier, the
/// envelopes and the functional, so the walk takes them from one place and the
/// gradient cannot be assembled on a grid the energy was not integrated over.
/// \param engine The engine whose grid and functional the gradient belongs to.
/// \param densityAlpha The alpha spin density, AOCount() x AOCount(), in the
/// evaluator's AO order.
/// \param densityBeta The beta spin density, same shape, order, and contract.
/// \param provider The grid's geometric derivatives for this engine's blocks.
/// \param thresholds The thresholds the energy path ran under; the gradient
/// takes them from the same object rather than deriving its own.
/// \param batch The batch granularity, which must be the one the energy path was
/// given: it changes no number, but it does change the order the energy is summed
/// in, and the walk's energy is compared against the energy path's.
/// \returns The gradient, the energy this walk integrated, the thresholds it ran
/// under and the counted cost, or an Error as the free function of the same name
/// returns it.
/// \ingroup qcx-grid
qcx::Result<XcGradientEvaluation> EvaluateXcGradient(
    const XcGridEngine& engine,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    const excgrid::GridDerivativeProvider& provider,
    const XcIntegrationThresholds& thresholds,
    const XcBatchSettings& batch = {});

/// One grid integration's second-order contribution to the fixed-density XC
/// Hessian.
///
/// At fixed density the exchange-correlation energy is
/// `E_xc(R) = sum_p W_p(R) e(X_p(R), R)`, and its second nuclear derivative
/// separates by what carries the displacement:
///
///     d2E/dR_q dR_r = sum_p [ W_,qr e + W_,q T_r + W_,r T_q + W d2e/dR_q dR_r ],
///
/// with `T_r` the integrand's total first derivative - the point's own motion
/// and the orbitals', the two channels EvaluateXcGradient accumulates
/// separately - and `d2e/dR_q dR_r` its total second. This object is the
/// derivative of what that walk accumulates.
///
/// That derivative is complete for a functional that reads only the two spin
/// densities. The rho fields' second total response is expressible with the
/// value, gradient and Hessian tiers the evaluator already supplies, so for such
/// a functional this object is the whole fixed-density Hessian. A functional
/// that reads a sigma invariant or a kinetic-energy density is refused by name
/// instead: that field's second response differentiates the AO Hessian once
/// more and needs a third-derivative tier the evaluator does not supply yet, so
/// answering it with the density channel alone would be a plausible wrong
/// number.
///
/// The grid's own contribution at that order is accumulated in full: the
/// quadrature weight's curvature, the two cross terms the weight's first
/// derivative carries, and the point's own position curvature. The default grid
/// rides each point on its owning atom, so the point's position curvature
/// vanishes and that channel's carrier is the weight's own derivatives.
///
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct XcGridMotionHessian {
    /// d2E_xc/dR2, (3N) x (3N) in Hartree/Bohr^2, atom-major like the gradient:
    /// entry (3*i + d, 3*j + e) is the second derivative in direction d of atom
    /// i and direction e of atom j. Symmetric to the last bit, because one
    /// triangle is accumulated and then mirrored rather than both being summed.
    Eigen::MatrixXd hessian;
    /// dE_xc/dR, the same gradient EvaluateXcGradient returns. This walk
    /// computes it on the way, and it is reported because a second-order row
    /// that does not differentiate the first-order result beside it is the
    /// defect this adjacency makes visible.
    Eigen::VectorXd gradient;
    /// The energy this walk integrated, Hartree, summed point by point in block
    /// order.
    double energy = 0.0;
    /// The thresholds this walk ran under, echoed back rather than assumed.
    XcIntegrationThresholds thresholds;
    /// The counted cost of this assembly.
    XcGradientCounts counts;
};

/// Assembles the fixed-density exchange-correlation Hessian.
///
/// The same walk as EvaluateXcGradient - the same blocks, the same thresholds,
/// the same per-point selection, kernel call and orbital-motion sums - with the
/// derivative provider asked for second order, accumulating at each visited
/// point
///
///     H_qr += W_,qr e + W_,q T_r + W_,r T_q
///           + W sum_c (de/dx^c) X^c_,qr + W M_qr ,
///
/// where `M` is the integrand's second total response at fixed density: the
/// functional's partials against the fields' second responses and its second
/// partials against their first. Sharing the first-order pieces is what makes
/// the two walks one derivative apart: a point set or a screening rule that
/// differed between them would make this the curvature of a different gradient.
///
/// \param blocks The grid blocks, in the order the energy path walks them.
/// \param provider The grid's geometric derivatives for these blocks. It must
/// report DerivativeOrder::kSecond, and it must be the provider built for the
/// geometry and the parameters these blocks came from; its arrays are indexed by
/// nuclear coordinate in the caller's own 3N order.
/// \param evaluator The AO evaluator the blocks' points are evaluated with.
/// \param functional The functional kernel, read exactly as the gradient walk
/// reads it.
/// \param envelopes The screening envelopes, in the evaluator's shell order.
/// \param thresholds The energy path's truncation and screening thresholds.
/// \param densityAlpha The alpha spin density, AOCount() x AOCount(), in the
/// evaluator's AO order.
/// \param densityBeta The beta spin density, same shape, order, and contract.
/// \returns The Hessian, the gradient this walk accumulated, the energy, the
/// thresholds and the counted cost, or an Error: kInvalidArgument for a shape
/// mismatch or a provider that does not supply second order; kUnimplemented, by
/// name, for a functional that reads a sigma invariant or a kinetic-energy
/// density, whose second response needs the third-derivative AO tier, or that
/// declares no second-derivative tier, or one that does not span the components
/// it reads; and the provider's own refusal, by name, when it declines a block
/// rather than answering it with zeros.
/// \ingroup qcx-grid
qcx::Result<XcGridMotionHessian> EvaluateXcGridMotionHessian(
    std::span<const excgrid::Block> blocks,
    const excgrid::GridDerivativeProvider& provider,
    const AoEvaluator& evaluator,
    const excgrid::XcFunctional& functional,
    std::span<const ShellEnvelope> envelopes,
    const XcIntegrationThresholds& thresholds,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta);

/// Assembles the fixed-density exchange-correlation Hessian on an engine's own
/// grid.
///
/// The form a consumer uses: the engine already holds the blocks, the AO tier,
/// the envelopes and the functional, so the second derivative cannot be taken
/// on a grid or a functional the energy was not integrated over.
/// \param engine The engine whose grid, evaluator and functional the Hessian
/// belongs to.
/// \param densityAlpha The alpha spin density, AOCount() x AOCount(), in the
/// evaluator's AO order.
/// \param densityBeta The beta spin density, same shape, order, and contract.
/// \param provider The grid's geometric derivatives for this engine's blocks.
/// \param thresholds The thresholds the energy path ran under.
/// \returns The Hessian, the gradient, the energy, the thresholds and the
/// counted cost, or an Error as the free function of the same name returns it.
/// \ingroup qcx-grid
qcx::Result<XcGridMotionHessian> EvaluateXcGridMotionHessian(
    const XcGridEngine& engine,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    const excgrid::GridDerivativeProvider& provider,
    const XcIntegrationThresholds& thresholds);

} // namespace qcx::grid
