#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <excgrid/grid.hpp>
#include <excgrid/kernel.hpp>
#include <span>
#include <string_view>
#include <vector>

namespace qcx::grid {

/// The molecular XC grid settings.
///
/// A qcx-side mirror of excgrid's grid parameters, defaults included, so the
/// common case does not make callers name an excgrid type. Every field is
/// forwarded to the grid build unchanged.
/// \ingroup qcx-grid
struct XcGridSettings {
    std::size_t radialPoints = 75; ///< Radial points per atom.
    std::size_t angularPoints = 302; ///< A Lebedev size (excgrid::AngularGrid::kAvailableSizes).
    double alpha = 0.5; ///< Radial mapping length scale, Bohr.
    std::size_t radialExponent = 2; ///< The Murray-Handy-Laming mapping exponent m.
    double trimWeight = 1e-15; ///< Points with |weight| below this are dropped.
    std::size_t blockTarget = 1024; ///< Spatial re-batching target (points per block).
};

/// What one integration actually did, in counted operations.
///
/// A wall clock cannot separate a screening win from a noise band, and cannot
/// say whether the setup ate it. These counters are the accounting: the SETUP
/// columns (weight terms and significance tests) are what a claim has to pay
/// for before the per-point ratio means anything end to end. The dense entry
/// points fill them too - zero setup, the dense costs - so the two paths are
/// comparable column for column.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct XcScreeningCounts {
    std::size_t points = 0; ///< Points visited (zero-weight points are skipped).
    std::size_t batches = 0; ///< Batch passes the grid was walked in.
    std::size_t densityWeightTerms = 0; ///< |D| entries summed for the weights.
    std::size_t shellTests = 0; ///< Significance tests run: points x shells considered.
    std::size_t shellKeeps = 0; ///< Those tests that kept their shell.
    // The fetch count is the batch API's own gate: a point's AOs are evaluated
    // once, in the fetch phase, and every consumer then reads the fetched
    // slice. A consumer that fetched again would show up here as more fetches
    // than live points, which is the failure this column exists to catch.
    std::size_t aoFetches = 0; ///< AO evaluations issued: one per live point.
    std::size_t aoSlotsEvaluated = 0; ///< AO values evaluated, selected shells only.
    std::size_t aoSlotsDense = 0; ///< points x AOCount(), the cost with no screening.
    // The density contraction's (mu, nu) reads, one pass per spin. The
    // potential accumulation touches the same pairs in the same shrinking
    // block, so the two move together and this one count speaks for both.
    std::size_t contractionPairs = 0; ///< Terms the contraction read.
    std::size_t contractionPairsDense = 0; ///< points x 2 x AOCount()^2, with no screening.
};

/// The batch granularity of the screened integration.
///
/// The screened pass walks each grid block in batches of this many points: it
/// selects and FETCHES the AO data for the whole batch first, then assembles
/// every point of the batch from the fetched slices. The split is what makes
/// the "fetched once" contract structural rather than incidental - a second
/// quantity computed from the same slice (the density gradient today, the
/// kinetic-energy density tomorrow) reuses it instead of re-entering the AO
/// tier, and XcScreeningCounts::aoFetches makes a violation visible.
///
/// The size is a knob, not a tuning: the numbers are identical at every
/// setting because each point's arithmetic is unchanged. It trades buffer
/// memory - per batched point, AOCount() values, 3 x AOCount() gradients, and
/// that point's selection lists - against the length of the two phases, and it
/// is clamped to the largest block, so no setting can allocate past a block's
/// worth of points.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct XcBatchSettings {
    std::size_t pointBatch = 64; ///< Points per batch; zero means one batch per block.
};

/// One grid integration of an XC functional: the energy, the spin potentials it
/// induces, and what the integration cost.
///
/// Public aggregate: the fields are the API and need no accessors.
/// \ingroup qcx-grid
struct XcEvaluation {
    double energy = 0.0; ///< The XC energy, Hartree.
    Eigen::MatrixXd potentialAlpha; ///< dE/dD_alpha, AOCount() x AOCount(), symmetric.
    Eigen::MatrixXd potentialBeta; ///< dE/dD_beta, AOCount() x AOCount(), symmetric.
    XcScreeningCounts counts; ///< The counted cost of this integration.
};

/// The molecular exchange-correlation engine: the qcx consumer of excgrid.
///
/// Wires excgrid's molecular block grid and frozen per-point kernel API to
/// qcx's AO tier. It builds the grid for a molecule once, then integrates a
/// named functional over it for a pair of spin densities,
///     E_xc = sum_i w_i e_xc(rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb)(r_i),
///     V_s  = dE_xc/dD_s,
/// assembling each point's densities and density gradients from the AO values
/// and first derivatives (AoEvaluator::EvaluateDerivatives) contracted with
/// the density matrices.
///
/// The density matrices must address the AOs in the evaluator's own order (the
/// integrals module's basis ordering). A mismatch is not detectable here and
/// yields a silently wrong energy, so the caller owns the convention.
///
/// The functional is a kernel-registry entry resolved by name
/// (excgrid::FindFunctional), and a name may denote a single component: "pbe"
/// is PBE exchange without correlation, "pbe_c" its correlation. UsesGradient()
/// and ExchangeFraction() are exposed because a hybrid's exact-exchange
/// fraction is the SCF's business, not the engine's.
///
/// Per-point cost is O(nAO^2) for the density contraction and O(nAO^2) for
/// the potential accumulation; the dense entry points make no screening or
/// batching assumption beyond excgrid's own spatial blocks. The screened entry
/// points add the density-weighted shell screening of shell_screening.hpp: at
/// each point only the significant shells are evaluated and only their block is
/// contracted and accumulated, with the setup counted alongside the per-point
/// work in XcScreeningCounts. A batched density tier is a deliberate follow-up,
/// not pre-empted here.
/// \ingroup qcx-grid
class XcGridEngine {
public:
    /// Creates the engine for one molecule, basis set, and functional.
    /// \param molecule The molecule that centers the grid.
    /// \param basis The basis set; must contain every element of the molecule.
    /// \param functionalName A shipped functional name (excgrid::FunctionalNames()).
    /// \param settings The grid build parameters.
    /// \returns The engine, or an Error: kInvalidArgument for an unknown
    /// functional name, a molecule whose coordinates are unusable, or grid
    /// parameters excgrid rejects; kUnimplemented for a basis function whose
    /// angular momentum the AO tier does not support.
    /// \ingroup qcx-grid
    static qcx::Result<XcGridEngine> Create(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basis,
                                            std::string_view functionalName,
                                            const XcGridSettings& settings = {});

    /// Move construction.
    XcGridEngine(XcGridEngine&&) = default;
    /// Move assignment.
    /// \returns This engine.
    XcGridEngine& operator=(XcGridEngine&&) = default;
    XcGridEngine(const XcGridEngine&) = delete;
    /// Copying is deleted: the engine owns its grid.
    XcGridEngine& operator=(const XcGridEngine&) = delete;
    ~XcGridEngine() = default;

    /// Integrates the functional for an unrestricted spin-density pair.
    /// \param densityAlpha The alpha spin density, AOCount() x AOCount(), in
    /// the evaluator's AO order; it must be a valid spin density (positive
    /// semidefinite, so that rho >= 0 on the grid).
    /// \param densityBeta The beta spin density, same shape, order, and
    /// contract.
    /// \returns The energy and both spin potentials, or kInvalidArgument for
    /// a shape mismatch.
    /// \ingroup qcx-grid
    qcx::Result<XcEvaluation> Evaluate(const Eigen::MatrixXd& densityAlpha,
                                       const Eigen::MatrixXd& densityBeta) const;

    /// Integrates the functional with density-weighted shell screening.
    ///
    /// At each point the engine tests every shell against the density - a shell
    /// is kept when `max(weight_alpha, weight_beta) * decay > tolerance`, with
    /// the weights of ShellDensityWeights - evaluates only the shells that
    /// survive, and contracts and accumulates only over their AO blocks.
    /// Dropping a shell drops its AO values AND its density-matrix rows and
    /// columns, so the neglected energy and potential terms are the ones whose
    /// density weight the test bounded. That bound is practical, not rigorous
    /// (the weight assumes unit-scale AO magnitudes), so a consumer owns the
    /// choice of tolerance and the screening gate measures the realized error
    /// against the dense path rather than trusting the constant.
    ///
    /// The tolerance has no default: which terms may be neglected is a physics
    /// choice, and it belongs at the call site rather than in a value nobody
    /// revisits. For scale, 1e-10 is what the fixtures here back - it keeps
    /// about 85% of the AO slots on a two-atom grid and moves the energy by
    /// ~1e-12 Hartree - and 1e-8, the threshold family this project runs at
    /// elsewhere, keeps about 81% at ~5e-10 Hartree. A diffuse basis is the
    /// case to tighten for, and the counted cost is in XcEvaluation::counts.
    ///
    /// A tolerance of zero keeps every shell whose weight and decay are both
    /// nonzero, which is the equivalence setting: the result then reproduces
    /// the dense entry points to rounding. Only AO magnitudes that have
    /// underflowed to zero are still dropped, and their contribution is zero in
    /// double precision. A tolerance below zero keeps every shell at every
    /// point unconditionally, which is what pins the assembly itself apart from
    /// the neglect rule.
    /// \param densityAlpha The alpha spin density, AOCount() x AOCount(), in
    /// the evaluator's AO order.
    /// \param densityBeta The beta spin density, same shape, order, and contract.
    /// \param tolerance The neglect threshold; zero screens nothing.
    /// \param batch The batch granularity; it changes no number, only how the
    /// grid is walked.
    /// \returns The energy, both spin potentials, and the counted cost of this
    /// integration, or kInvalidArgument for a shape mismatch.
    /// \ingroup qcx-grid
    qcx::Result<XcEvaluation> EvaluateScreened(const Eigen::MatrixXd& densityAlpha,
                                               const Eigen::MatrixXd& densityBeta,
                                               double tolerance,
                                               const XcBatchSettings& batch = {}) const;

    /// Integrates the functional for a closed-shell (spin-summed) density.
    ///
    /// The engine splits D into D_alpha = D_beta = D/2, the restricted
    /// convention. The returned potentials are still dE/dD_s, so the chain
    /// rule through that split gives the restricted potential
    ///     V = (V_alpha + V_beta) / 2:
    /// a restricted consumer HALVES the sum. Using the sum as it stands
    /// double-counts the spin split and is wrong by a factor of two.
    /// \param density The spin-summed density, AOCount() x AOCount().
    /// \returns The energy and both spin potentials, or kInvalidArgument for
    /// a shape mismatch.
    /// \ingroup qcx-grid
    qcx::Result<XcEvaluation> EvaluateClosedShell(const Eigen::MatrixXd& density) const;

    /// Integrates the functional for a closed-shell density, with screening.
    ///
    /// The screened form of EvaluateClosedShell: the same D/2 spin split and
    /// the same `V = (V_alpha + V_beta) / 2` chain rule, with both spins sharing
    /// one density weight pass because they are the same matrix.
    /// \param density The spin-summed density, AOCount() x AOCount().
    /// \param tolerance The neglect threshold; zero screens nothing.
    /// \param batch The batch granularity; it changes no number, only how the
    /// grid is walked.
    /// \returns The energy, both spin potentials, and the counted cost, or
    /// kInvalidArgument for a shape mismatch.
    /// \ingroup qcx-grid
    qcx::Result<XcEvaluation> EvaluateClosedShellScreened(const Eigen::MatrixXd& density,
                                                          double tolerance,
                                                          const XcBatchSettings& batch = {}) const;

    /// Whether the functional consumes the density gradients (GGA or beyond).
    /// \returns True when sigma terms are present.
    [[nodiscard]] bool UsesGradient() const noexcept {
        return _functional->UsesGradient();
    }

    /// The exact-exchange fraction a consumer routes through its HF path.
    /// \returns The HF exchange fraction in [0, 1].
    [[nodiscard]] double ExchangeFraction() const noexcept {
        return _functional->ExchangeFraction();
    }

    /// The number of AO basis functions.
    /// \returns The AO count.
    [[nodiscard]] std::size_t AOCount() const noexcept {
        return _aoCount;
    }

    /// The total number of grid points across all blocks.
    /// \returns The point count.
    [[nodiscard]] std::size_t PointCount() const noexcept {
        return _grid.TotalPointCount();
    }

    /// The number of grid blocks.
    /// \returns The block count.
    [[nodiscard]] std::size_t BlockCount() const noexcept {
        return _grid.BlockCount();
    }

    /// The block grid the engine walks, for consumers integrating further
    /// quantities on the same point set.
    /// \returns The grid.
    [[nodiscard]] const excgrid::BlockGrid& Grid() const noexcept {
        return _grid;
    }

    /// The screening envelopes, in the evaluator's shell order.
    ///
    /// Exposed so a consumer or a gate can reproduce the selection the engine
    /// makes - the envelopes plus ShellDensityWeights and ShellIsSignificant
    /// are the whole rule - and so the shell count behind
    /// XcScreeningCounts::shellTests is checkable rather than implicit.
    /// \returns The envelopes, one per evaluator shell.
    [[nodiscard]] std::span<const ShellEnvelope> ShellEnvelopes() const noexcept {
        return _envelopes;
    }

private:
    XcGridEngine(excgrid::BlockGrid grid,
                 const excgrid::XcFunctional* functional,
                 AoEvaluator evaluator,
                 std::vector<ShellEnvelope> envelopes) noexcept;

    excgrid::BlockGrid _grid;
    // Borrowed from excgrid's kernel registry, which outlives every engine.
    const excgrid::XcFunctional* _functional;
    AoEvaluator _evaluator;
    std::vector<ShellEnvelope> _envelopes;
    std::size_t _aoCount;
};

} // namespace qcx::grid
