#pragma once

/// \file
/// The representational band selector beside the response solve: which of the
/// three bands a problem is solved in, and the solve that runs in the band it
/// selects.
///
/// The choice is representational, not iterative: it depends on the problem's
/// dimension and on how many entries an explicit matrix would hold, and never
/// on how a previous solve converged. A small system is assembled densely and
/// solved directly; a medium one is assembled in compressed sparse row form and
/// solved by AMG-preconditioned conjugate gradient; a large one is solved
/// matrix-free by preconditioned conjugate gradient against the operator's
/// action.
///
/// Warm starts are not a property of the selector. The explicit bands are cold
/// by construction - they assemble the matrix once and solve it - while the
/// matrix-free band reuses whatever the caller's ResponseSolver has stored. The
/// solve therefore takes the caller's matrix-free solver rather than owning
/// one, and the report says how many right-hand sides it warmed.

#include "qcx/error.hpp"
#include "qcx/linalg/sparse_solver.hpp"
#include "qcx/response/response_matrix.hpp"
#include "qcx/response/response_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace qcx::response {

/// \ingroup qcx-response
/// Which representation a response system is solved in.
enum class SolverBand : std::uint8_t {
    kDense, ///< An explicit dense matrix, solved directly.
    kExplicitCsr, ///< An explicit compressed-sparse-row matrix, AMG-preconditioned CG.
    kMatrixFree, ///< The operator's action, preconditioned CG.
};

/// \ingroup qcx-response
/// Names a band, for reporting and for a message that names what it selected
/// rather than leaving the choice implicit.
/// \param band The band to name.
/// \returns The band's name ("dense", "explicit-csr", "matrix-free").
std::string_view ToString(SolverBand band);

/// \ingroup qcx-response
/// The ceilings that decide the band. They are a policy, not a measurement:
/// a caller may lower them to force a larger problem onto a smaller band, and
/// the selector refuses nothing for a self-consistent set.
struct ResponseBandLimits {
    /// Dimensions at or below this are assembled densely.
    std::size_t denseMaxDimension = 128;

    /// Dimensions at or below this (and within the nonzero cap) are assembled
    /// in CSR form.
    std::size_t csrMaxDimension = 5000;

    /// The most nonzeros the CSR assembly may be asked to hold for. The
    /// estimate is the guard because the assembly's own temporary is dense: at
    /// the 5000-unknown ceiling it is Dimension()^2 doubles.
    std::size_t csrMaxNonzeros = 25'000'000;
};

/// \ingroup qcx-response
/// What the selector reads: the problem's dimension and an upper bound on the
/// nonzeros an explicit assembly would hold.
struct ResponseBandProblem {
    /// Flattened response dimension (occupied times virtual orbitals); must be
    /// positive.
    std::size_t dimension = 0;

    /// An upper bound on the assembled nonzeros, used only for the CSR band's
    /// memory guard. A caller that knows no sparsity passes
    /// dimension * dimension, which is the dense upper bound.
    std::size_t estimatedNonzeros = 0;
};

/// \ingroup qcx-response
/// Selects the band a problem is solved in.
///
/// The order is fixed: dense up to its ceiling; explicit CSR up to its own
/// dimension ceiling while the estimated nonzeros stay within the cap;
/// matrix-free otherwise. The nonzero estimate is not consulted for a problem
/// the dense band takes, because the dense band does not allocate by it.
/// \param problem The problem's dimension and nonzero estimate.
/// \param limits The band ceilings.
/// \returns The band, or an Error (kInvalidArgument) when the problem has no
/// dimension.
qcx::Result<SolverBand> SelectResponseBand(const ResponseBandProblem& problem,
                                           const ResponseBandLimits& limits = {});

/// \ingroup qcx-response
/// What a band solve cost and which representation it used.
struct ResponseBandSolveReport {
    SolverBand band = SolverBand::kMatrixFree; ///< The band that ran.
    std::size_t nonzeros = 0; ///< Entries the band stored: Dimension()^2 densely,
                              ///< the CSR value count explicitly, 0 matrix-free.
    std::size_t iterations = 0; ///< Iterative iterations in total; 0 for the direct band.
    std::size_t warmStarted = 0; ///< Right-hand sides started from a stored solution.
    std::size_t regularised = 0; ///< Right-hand sides whose preconditioner was clamped.
    double gapRatio = 0.0; ///< min|d| / max|d| of the operator's preconditioner.
    double worstResidual = 0.0; ///< Worst true relative residual over the right-hand sides.
};

namespace detail {

/// Solves a dense row-major system by full-pivoted LU.
///
/// Defined in the module rather than in this header so that Eigen stays out of
/// the response module's public surface.
/// \param dimension Rows and columns; must be positive.
/// \param matrix Row-major entries, size dimension^2.
/// \param rhsCount Number of right-hand sides; must be positive.
/// \param rhs Concatenated right-hand sides, right-hand-side-major.
/// \returns The concatenated solution, or an Error (kInvalidArgument for a
/// shape mismatch or a singular matrix).
qcx::Result<std::vector<double>> SolveDenseSystem(std::size_t dimension,
                                                  std::span<const double> matrix,
                                                  std::size_t rhsCount,
                                                  std::span<const double> rhs);

} // namespace detail

/// \ingroup qcx-response
/// Solves a response system in the band the selector chooses for it.
///
/// The dense band assembles the matrix and solves it directly; the explicit-CSR
/// band assembles it and hands it to the AMGCL seam with one hierarchy for all
/// right-hand sides; the matrix-free band runs the caller's preconditioned
/// conjugate-gradient solver against the operator's action and keeps its warm
/// start. The explicit bands ignore the warm-start option, because they hold no
/// state to reuse - that is what "cold by design" means for them.
///
/// The report's residual is the TRUE relative residual recomputed from the
/// operator after the solve, for every band, so the three bands' answers are
/// compared by one definition rather than by each solver's own claim.
/// \tparam Operator A type with `std::size_t Dimension() const`,
/// `qcx::Result<void> Apply(std::span<const double>, std::span<double>) const`
/// and `qcx::Result<void> Preconditioner(std::span<double>) const` - satisfied
/// by OrbitalHessianOperator and by DenseResponseOperator.
/// \param op The operator.
/// \param problem The problem's dimension and nonzero estimate; the dimension
/// must equal the operator's.
/// \param rhsCount Number of right-hand sides; must be positive.
/// \param rhs Right-hand sides, contiguous and right-hand-side-major.
/// \param solution Output, the same shape as \p rhs.
/// \param matrixFreeSolver The stateful solver the matrix-free band runs; its
/// options also carry the tolerance the explicit-CSR band requests. The dense
/// band needs neither.
/// \param limits The band ceilings.
/// \param assemblyOptions What the explicit-CSR assembly keeps.
/// \returns The report, or an Error (kInvalidArgument for a zero count, a
/// shape mismatch or an inconsistent problem; a band's own refusal is
/// propagated unchanged).
template <typename Operator>
qcx::Result<ResponseBandSolveReport> SolveResponseByBand(
    const Operator& op,
    const ResponseBandProblem& problem,
    std::size_t rhsCount,
    std::span<const double> rhs,
    std::span<double> solution,
    ResponseSolver<Operator>& matrixFreeSolver,
    const ResponseBandLimits& limits = {},
    ResponseMatrixOptions assemblyOptions = {}) {
    if (rhsCount == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "a solve needs at least one right-hand side"});
    }

    const std::size_t dimension = op.Dimension();

    if (dimension == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the response operator is empty"});
    }

    if (problem.dimension != dimension)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the response problem's dimension does not match the operator's"});
    }

    if (rhsCount > std::numeric_limits<std::size_t>::max() / dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "rhsCount times the dimension overflows the size type"});
    }

    if (rhs.size() != rhsCount * dimension || solution.size() != rhsCount * dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "rhs and solution must each hold rhsCount * Dimension() "
                                          "values"});
    }

    auto selected = SelectResponseBand(problem, limits);

    if (!selected.has_value())
    {
        return std::unexpected(selected.error());
    }

    ResponseBandSolveReport report;
    report.band = *selected;

    // The preconditioner's gap is a property of the problem, not of the band
    // that solves it: measuring it for every band is what lets the same
    // small-gap physics be reported whichever representation ran. The
    // matrix-free band overwrites it with its own prepared value, which is the
    // one its fallback decision was made from.
    std::vector<double> diagonal(dimension);
    auto diagonalStatus = op.Preconditioner(diagonal);

    if (!diagonalStatus.has_value())
    {
        return std::unexpected(diagonalStatus.error());
    }

    double largest = 0.0;

    for (const double value : diagonal)
    {
        largest = std::max(largest, std::fabs(value));
    }

    if (largest > 0.0)
    {
        double smallest = largest;

        for (const double value : diagonal)
        {
            smallest = std::min(smallest, std::fabs(value));
        }

        report.gapRatio = smallest / largest;
    }

    switch (*selected)
    {
    case SolverBand::kDense: {
        auto assembled = AssembleDenseResponseMatrix(op);

        if (!assembled.has_value())
        {
            return std::unexpected(assembled.error());
        }

        auto solved = detail::SolveDenseSystem(dimension, assembled->values, rhsCount, rhs);

        if (!solved.has_value())
        {
            return std::unexpected(solved.error());
        }

        std::copy(solved->begin(), solved->end(), solution.begin());
        report.nonzeros = dimension * dimension;
        break;
    }

    case SolverBand::kExplicitCsr: {
        auto assembled = AssembleSparseResponseMatrix(op, assemblyOptions);

        if (!assembled.has_value())
        {
            return std::unexpected(assembled.error());
        }

        auto solved =
            qcx::linalg::SolveSparseSystemMultipleRhs(assembled->rowOffsets,
                                                      assembled->columnIndices,
                                                      assembled->values,
                                                      rhs,
                                                      rhsCount,
                                                      matrixFreeSolver.Options().tolerance);

        if (!solved.has_value())
        {
            return std::unexpected(solved.error());
        }

        std::copy(solved->solution.begin(), solved->solution.end(), solution.begin());
        report.nonzeros = assembled->values.size();
        report.iterations = solved->iterations;
        break;
    }

    case SolverBand::kMatrixFree: {
        auto solved = matrixFreeSolver.Solve(op, rhsCount, rhs, solution);

        if (!solved.has_value())
        {
            return std::unexpected(solved.error());
        }

        report.iterations = solved->iterations;
        report.warmStarted = solved->warmStarted;
        report.regularised = solved->regularised;
        report.gapRatio = solved->gapRatio;
        break;
    }
    }

    std::vector<double> product(dimension);
    std::vector<double> residual(dimension);

    for (std::size_t k = 0; k < rhsCount; ++k)
    {
        const std::span<const double> side = rhs.subspan(k * dimension, dimension);
        const std::span<const double> answer = solution.subspan(k * dimension, dimension);

        auto status = op.Apply(answer, product);

        if (!status.has_value())
        {
            return std::unexpected(status.error());
        }

        double residualSquared = 0.0;
        double rhsSquared = 0.0;

        for (std::size_t i = 0; i < dimension; ++i)
        {
            residual[i] = side[i] - product[i];
            residualSquared += residual[i] * residual[i];
            rhsSquared += side[i] * side[i];
        }

        const double rhsNorm = std::sqrt(rhsSquared);
        const double relative =
            rhsNorm > 0.0 ? std::sqrt(residualSquared) / rhsNorm : std::sqrt(residualSquared);

        report.worstResidual = std::max(report.worstResidual, relative);
    }

    return report;
}

} // namespace qcx::response
