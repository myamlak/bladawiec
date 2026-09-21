#pragma once

// — the driver-side Kohn-Sham grid.
//
// A Kohn-Sham run integrates the functional over a grid, and a run asked for a
// gradient differentiates the same grid. Both walks have to see ONE grid: a
// provider answers for a block only when it was built for the geometry and the
// build parameters that produced the block, and the truncation and screening
// thresholds decide which points and which shells each walk reads.
//
// So the engine, the provider and the thresholds are built together here, from
// one settings object and one tolerance:
//
//   - the engine and the settings are the pair the grid was built from, so the
//     provider cannot answer for a different point set;
//   - the thresholds are ONE object, built from those settings and that
//     tolerance, read by the energy path and handed to the gradient walk. A
//     gradient that derived its own would be the derivative of another energy,
//     and both numbers would still look reasonable;
//   - the batch is one value for both walks: it moves no number, but it fixes
//     the order the energy is summed in, and the gradient walk's energy is
//     compared against the energy path's.
//
// The provider is built when a gradient asks for one, so a run that never asks
// pays nothing for it.
//
// Driver-internal (src/internal/, the ks_composition.hpp footprint): tested
// against a real molecule and a real grid without a whole run.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/geometry_translation.hpp"
#include "qcx/grid/xc_gradient.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <excgrid/geometry.hpp>
#include <excgrid/grid.hpp>
#include <excgrid/grid_derivatives.hpp>
#include <memory>
#include <string_view>
#include <utility>

namespace qcx::driver::internal {

/// One Kohn-Sham run's grid, as the energy path and the gradient path see it.
///
/// Public aggregate: the fields are the API (the aggregate-struct exemption).
/// \ingroup qcx-driver
struct KsGrid {
    std::shared_ptr<qcx::grid::XcGridEngine> engine; ///< The run's grid engine.
    excgrid::Geometry geometry; ///< The geometry the engine's grid was built for.
    excgrid::GridParams params; ///< The parameters the engine's grid was built with.
    /// The truncation and screening both walks run under: the energy path is
    /// handed this object's own screening value, the gradient walk the object.
    qcx::grid::XcIntegrationThresholds thresholds;
    /// The batch granularity both walks are given.
    qcx::grid::XcBatchSettings batch;

    /// Builds this grid's derivative provider.
    ///
    /// The geometry and the parameters above are the pair the engine's grid was
    /// built from, so the provider answers for exactly the blocks that engine
    /// walks.
    /// \returns The provider, or the library's refusal, translated.
    /// \ingroup qcx-driver
    [[nodiscard]] qcx::Result<std::unique_ptr<excgrid::GridDerivativeProvider>> Derivatives()
        const {
        auto provider = excgrid::CreateGridDerivativeProvider(geometry, params);

        if (!provider.has_value())
        {
            return std::unexpected(qcx::grid::TranslateExcgridError(
                provider.error(), "the grid's geometric derivative provider could not be built"));
        }

        return std::move(*provider);
    }
};

/// Builds a run's grid from ONE settings object and ONE tolerance.
///
/// The engine and the thresholds are built from the same two values, which is
/// the whole point of the function: a call site cannot hand the grid one
/// truncation and the walk another.
/// \param molecule The molecule the grid is centered on.
/// \param basis The orbital basis the AO tier evaluates.
/// \param functionalName A shipped functional name (excgrid::FunctionalNames()).
/// \param settings The resolved `[grid]` settings the engine is created with.
/// \param tolerance The screening tolerance the energy path passes.
/// \returns The grid, or an Error: whatever XcGridEngine::Create refuses, or the
/// geometry translation's own refusal.
/// \ingroup qcx-driver
inline qcx::Result<KsGrid> CreateKsGrid(const qcx::molecule::Molecule& molecule,
                                        const qcx::basisset::BasisSet& basis,
                                        std::string_view functionalName,
                                        const qcx::grid::XcGridSettings& settings,
                                        double tolerance) {
    auto engine = qcx::grid::XcGridEngine::Create(molecule, basis, functionalName, settings);

    if (!engine.has_value())
    {
        return std::unexpected(engine.error());
    }

    auto geometry = qcx::grid::ToExcgridGeometry(molecule);

    if (!geometry.has_value())
    {
        return std::unexpected(geometry.error());
    }

    return KsGrid{std::make_shared<qcx::grid::XcGridEngine>(std::move(*engine)),
                  std::move(*geometry),
                  qcx::grid::ToExcgridParams(settings),
                  qcx::grid::XcIntegrationThresholds::FromEnergyPath(settings, tolerance),
                  {}};
}

/// Adds a run's fixed-density exchange-correlation gradient into a total.
///
/// The walk differentiates the energy path's own blocks under the grid's own
/// thresholds, at the density handed in. What it leaves out is the density's
/// RESPONSE to the displacement, which belongs to the self-consistent-field and
/// response layers - so this is a contribution to a total and never the whole
/// of one.
/// \param grid The run's grid.
/// \param densityAlpha The alpha spin density, AOCount() x AOCount(), in the
/// AO tier's order.
/// \param densityBeta The beta spin density, same shape, order and contract.
/// \param totalGradient The run's total gradient, atom-major 3N in
/// Hartree/Bohr: the walk's own vector is ADDED into it, so it must already
/// carry every coordinate and be exactly the walk's length.
/// \returns The walk's accounting - its gradient (the same vector that was
/// added), the energy it integrated, the thresholds it ran under and the
/// counted cost - or an Error: kInvalidArgument when the accumulator is not the
/// walk's own length, or whatever the provider and the walk refuse.
/// \ingroup qcx-driver
inline qcx::Result<qcx::grid::XcGradientEvaluation> AddXcGradientContribution(
    const KsGrid& grid,
    const Eigen::MatrixXd& densityAlpha,
    const Eigen::MatrixXd& densityBeta,
    Eigen::VectorXd& totalGradient) {
    auto provider = grid.Derivatives();

    if (!provider.has_value())
    {
        return std::unexpected(provider.error());
    }

    const excgrid::GridDerivativeProvider& derivatives = **provider;
    auto walk = qcx::grid::EvaluateXcGradient(
        *grid.engine, densityAlpha, densityBeta, derivatives, grid.thresholds, grid.batch);

    if (!walk.has_value())
    {
        return std::unexpected(walk.error());
    }

    if (totalGradient.size() != walk->gradient.size())
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "AddXcGradientContribution: the accumulator is not the walk's own 3N vector; adding a "
            "gradient into another length would place every term on the wrong coordinate"});
    }

    // The fixed-density exchange-correlation term, added into the run's total:
    // the derivative of E_xc(R; D) with D held at the matrices handed in. The
    // density's own response to the displacement is deliberately NOT here - it
    // is the self-consistent-field's and the response layer's term - so a
    // reader who came looking for the rest of the gradient will find it there
    // and not in this walk.
    totalGradient += walk->gradient;

    return walk;
}

} // namespace qcx::driver::internal
