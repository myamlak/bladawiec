#pragma once

/// \file
/// The stateful response solver: a preconditioned conjugate-gradient solve of
/// the orbital response to one or more perturbations, against a matrix-free
/// operator, keeping what a displacement sweep needs to keep.
///
/// The solver is written to be called MANY times against a slowly changing
/// operator, so its state is the point of it: the previous solution vectors,
/// the convergence history and the per-right-hand-side counters survive a
/// solve and are reused by the next one. A stateless implementation would
/// return the same numbers and make a sweep unusable, which is why the reuse
/// is reported as a cost and a counter rather than as a value.

#include "qcx/error.hpp"
#include "qcx/response/response_operator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace qcx::response {

/// \ingroup qcx-response
/// Which diagonal preconditioner a solve ran with.
enum class PreconditionerKind : std::uint8_t {
    kNone, ///< No preconditioner: M is the identity.
    kDiagonal, ///< The operator's diagonal preconditioner, used as given.
    kRegularised, ///< The diagonal with its small entries clamped - the small-gap fallback.
};

/// \ingroup qcx-response
/// Names a preconditioner kind, for reporting and for a refusal that names what
/// it refused rather than falling back quietly.
/// \param kind The kind to name.
/// \returns The kind's name ("none", "diagonal", "regularised").
std::string_view ToString(PreconditionerKind kind);

/// \ingroup qcx-response
/// Options for a response solve.
struct ResponseSolveOptions {
    /// Relative residual tolerance, measured against the right-hand side's
    /// norm. The default is the project's energy rung.
    double tolerance = 1e-8;

    /// Iteration budget per right-hand side.
    std::size_t maxIterations = 1000;

    /// The preconditioner to start from. kRegularised is what the small-gap
    /// fallback selects; asking for it directly is refused when the diagonal
    /// has no small entries rather than quietly ignored.
    PreconditionerKind preconditioner = PreconditionerKind::kDiagonal;

    /// Whether a diagonal whose smallest entry is far below its largest may be
    /// clamped before it is inverted. When false, such a diagonal makes the
    /// solve REFUSE BY NAME instead of falling back.
    bool allowSmallGapFallback = true;

    /// The ratio min|d| / max|d| below which the diagonal counts as small-gapped.
    double smallGapRatio = 1e-3;

    /// The floor the clamp holds the diagonal to, as a fraction of max|d|.
    /// Above smallGapRatio this is never applied.
    double regularisationRatio = 1e-3;

    /// Whether the next solve starts from the solution this solver last stored
    /// for the same right-hand-side ordinal. Turning it off is how a caller
    /// asks for a cold solve against the same operator.
    bool reusePrevious = true;
};

/// \ingroup qcx-response
/// What a solve cost and which state it used - the evidence the reuse test
/// reads, in place of a returned number that a stateless solver would also get
/// right.
struct ResponseSolveReport {
    std::size_t rhsCount = 0; ///< Right-hand sides solved.
    std::size_t iterations = 0; ///< Conjugate-gradient iterations in total.
    std::size_t matvecs = 0; ///< Operator applications in total.
    std::size_t warmStarted = 0; ///< Right-hand sides started from a stored solution.
    std::size_t regularised = 0; ///< Right-hand sides whose preconditioner was clamped.
    PreconditionerKind preconditioner = PreconditionerKind::kNone; ///< The kind the solve ran with.
    double gapRatio = 0.0; ///< min|d| / max|d| of the operator's preconditioner.
    double worstResidual = 0.0; ///< Worst true relative residual over the right-hand sides.
};

/// \ingroup qcx-response
/// A preconditioner in the form the solve applies it: the inverse diagonal.
struct PreparedPreconditioner {
    std::vector<double> inverse; ///< The inverse diagonal, size Dimension().
    PreconditionerKind kind = PreconditionerKind::kNone; ///< Which kind was prepared.
    double gapRatio = 0.0; ///< min|d| / max|d| of the diagonal as supplied.
};

/// \ingroup qcx-response
/// Prepares the preconditioner a solve will apply, applying the small-gap rule.
///
/// This is the one place the project's refusal rule bites: a diagonal whose
/// smallest entry is far below its largest is clamped only if the caller allows
/// it, and otherwise the solve refuses by name rather than falling back to
/// another algorithm without saying so.
/// \param diagonal The operator's preconditioner diagonal, size Dimension().
/// \param options The solve's options.
/// \returns The prepared preconditioner, or an Error (kInvalidArgument when a
/// diagonal is empty or identically zero, when a requested kind cannot be
/// prepared, or when the gap is small and the fallback is disallowed - the
/// message names the ratio, the threshold and the fallback).
qcx::Result<PreparedPreconditioner> PreparePreconditioner(std::span<const double> diagonal,
                                                          const ResponseSolveOptions& options);

namespace detail {

/// Euclidean dot product of two equal-length spans.
/// \param a First span.
/// \param b Second span.
/// \returns The dot product.
inline double Dot(std::span<const double> a, std::span<const double> b) {
    double sum = 0.0;

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * b[i];
    }

    return sum;
}

/// Euclidean norm of a span.
/// \param a The span.
/// \returns Its norm.
inline double Norm(std::span<const double> a) {
    return std::sqrt(Dot(a, a));
}

} // namespace detail

/// \ingroup qcx-response
/// Solves A x = b for the orbital response to one or more perturbations,
/// matrix-free and preconditioned, keeping its previous answers.
///
/// The solve is preconditioned conjugate gradient on the preconditioned
/// residual. CG is the right member of the solver ladder here because a
/// response operator at a stable reference state is symmetric positive
/// definite; a band that meets a non-definite operator is a named refusal, not
/// a silent substitution.
///
/// The convergence test is run on the TRUE residual, recomputed from the
/// operator every iteration rather than carried by the CG recurrence. That
/// costs one extra operator application per iteration, and it is worth it: a
/// recurrence residual drifts, and this solver's report is what a caller reads
/// to decide whether to trust the answer.
///
/// \tparam Operator A type with `std::size_t Dimension() const`,
/// `qcx::Result<void> Apply(std::span<const double>, std::span<double>) const`
/// and `qcx::Result<void> Preconditioner(std::span<double>) const` - satisfied
/// by OrbitalHessianOperator and by DenseResponseOperator.
template <typename Operator> class ResponseSolver {
public:
    /// Builds a solver with no stored state.
    /// \param options The solve options.
    explicit ResponseSolver(ResponseSolveOptions options = {}) : _options(options) {}

    /// Solves every right-hand side against one operator.
    ///
    /// The right-hand-side ordinal is the index within \p rhs, and the ordinal
    /// is what the stored state is keyed on - so a caller running a sweep calls
    /// this once per geometry with the same right-hand sides, and each solve
    /// starts from the previous geometry's answer.
    /// \param op The operator, by const reference for the duration of the call.
    /// \param rhsCount Number of right-hand sides.
    /// \param rhs Right-hand sides, contiguous and right-hand-side-major: side
    /// k occupies [k * Dimension(), (k + 1) * Dimension()).
    /// \param solution Output, the same shape as \p rhs.
    /// \returns The report, or an Error (kInvalidArgument for a zero count, a
    /// length mismatch or a refused preconditioner; kConvergenceFailure when a
    /// right-hand side exhausts the iteration budget, with the count in the
    /// message).
    qcx::Result<ResponseSolveReport> Solve(const Operator& op,
                                           std::size_t rhsCount,
                                           std::span<const double> rhs,
                                           std::span<double> solution);

    /// The solution this solver stored for one right-hand-side ordinal.
    /// \param ordinal The right-hand-side ordinal.
    /// \returns The stored vector, empty when nothing has been stored for it.
    const std::vector<double>& PreviousSolution(std::size_t ordinal) const {
        static const std::vector<double> kEmpty;
        return ordinal < _previousSolutions.size() ? _previousSolutions[ordinal] : kEmpty;
    }

    /// The iteration count of every right-hand side solved so far, oldest
    /// first. It is retained across solves on purpose: a sweep that restarts
    /// its history every geometry cannot be told from one that restarts its
    /// work.
    /// \returns The recorded iteration counts.
    const std::vector<std::size_t>& ConvergenceHistory() const {
        return _history;
    }

    /// \returns How many right-hand sides this solver has solved.
    std::size_t SolveCount() const {
        return _history.size();
    }

    /// \returns The options in force.
    const ResponseSolveOptions& Options() const {
        return _options;
    }

    /// Drops every stored solution and the convergence history.
    void Reset() {
        _previousSolutions.clear();
        _history.clear();
        _dimension = 0;
    }

private:
    ResponseSolveOptions _options;
    std::vector<std::vector<double>> _previousSolutions;
    std::vector<std::size_t> _history;
    std::size_t _dimension = 0;
};

template <typename Operator>
qcx::Result<ResponseSolveReport> ResponseSolver<Operator>::Solve(const Operator& op,
                                                                 std::size_t rhsCount,
                                                                 std::span<const double> rhs,
                                                                 std::span<double> solution) {
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

    if (rhs.size() != rhsCount * dimension || solution.size() != rhsCount * dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "rhs and solution must each hold rhsCount * Dimension() "
                                          "values"});
    }

    // A change of shape invalidates every stored vector: keeping them would be
    // reusing state that no longer describes this problem.
    if (_dimension != dimension || _previousSolutions.size() != rhsCount)
    {
        Reset();
        _dimension = dimension;
        _previousSolutions.resize(rhsCount);
    }

    std::vector<double> diagonal(dimension);
    auto diagonalStatus = op.Preconditioner(diagonal);

    if (!diagonalStatus.has_value())
    {
        return std::unexpected(diagonalStatus.error());
    }

    auto prepared = PreparePreconditioner(diagonal, _options);

    if (!prepared.has_value())
    {
        return std::unexpected(prepared.error());
    }

    ResponseSolveReport report;
    report.rhsCount = rhsCount;
    report.preconditioner = prepared->kind;
    report.gapRatio = prepared->gapRatio;

    std::vector<double> residual(dimension);
    std::vector<double> direction(dimension);
    std::vector<double> preconditioned(dimension);
    std::vector<double> product(dimension);

    for (std::size_t k = 0; k < rhsCount; ++k)
    {
        const std::span<const double> b = rhs.subspan(k * dimension, dimension);
        const std::span<double> x = solution.subspan(k * dimension, dimension);

        const bool warm = _options.reusePrevious && _previousSolutions[k].size() == dimension;

        if (warm)
        {
            std::copy(_previousSolutions[k].begin(), _previousSolutions[k].end(), x.begin());
            ++report.warmStarted;
        } else
        {
            std::fill(x.begin(), x.end(), 0.0);
        }

        const double bnorm = detail::Norm(b);

        if (bnorm == 0.0)
        {
            std::fill(x.begin(), x.end(), 0.0);
            _previousSolutions[k].assign(dimension, 0.0);
            _history.push_back(0);
            continue;
        }

        // r = b - A x, recomputed from the operator. The recurrence is never
        // trusted for the convergence decision.
        auto applyStatus = op.Apply(x, product);

        if (!applyStatus.has_value())
        {
            return std::unexpected(applyStatus.error());
        }

        for (std::size_t i = 0; i < dimension; ++i)
        {
            residual[i] = b[i] - product[i];
        }

        ++report.matvecs;
        double residualNorm = detail::Norm(residual);
        std::size_t iterations = 0;
        bool converged = residualNorm <= _options.tolerance * bnorm;

        if (!converged)
        {
            for (std::size_t i = 0; i < dimension; ++i)
            {
                preconditioned[i] = prepared->inverse[i] * residual[i];
            }

            double rz = detail::Dot(residual, preconditioned);
            std::copy(preconditioned.begin(), preconditioned.end(), direction.begin());

            for (std::size_t iteration = 1; iteration <= _options.maxIterations; ++iteration)
            {
                auto stepStatus = op.Apply(direction, product);

                if (!stepStatus.has_value())
                {
                    return std::unexpected(stepStatus.error());
                }

                ++report.matvecs;

                const double curvature = detail::Dot(direction, product);

                if (curvature <= 0.0)
                {
                    // A non-positive curvature means the operator is not
                    // positive definite on this direction, which the CG band
                    // does not serve. Say so rather than taking the step.
                    return std::unexpected(qcx::Error{
                        qcx::ErrorCode::kConvergenceFailure,
                        "the response operator has non-positive curvature at iteration " +
                            std::to_string(iteration) +
                            ", so the conjugate-gradient band does not apply to it"});
                }

                const double alpha = rz / curvature;

                for (std::size_t i = 0; i < dimension; ++i)
                {
                    x[i] += alpha * direction[i];
                }

                ++iterations;

                auto trueStatus = op.Apply(x, product);

                if (!trueStatus.has_value())
                {
                    return std::unexpected(trueStatus.error());
                }

                ++report.matvecs;

                for (std::size_t i = 0; i < dimension; ++i)
                {
                    residual[i] = b[i] - product[i];
                }

                residualNorm = detail::Norm(residual);

                if (residualNorm <= _options.tolerance * bnorm)
                {
                    converged = true;
                    report.iterations += iterations;
                    break;
                }

                for (std::size_t i = 0; i < dimension; ++i)
                {
                    preconditioned[i] = prepared->inverse[i] * residual[i];
                }

                const double rzNext = detail::Dot(residual, preconditioned);

                if (rzNext <= 0.0)
                {
                    return std::unexpected(
                        qcx::Error{qcx::ErrorCode::kConvergenceFailure,
                                   "the preconditioner is not positive definite at iteration " +
                                       std::to_string(iteration)});
                }

                const double beta = rzNext / rz;

                for (std::size_t i = 0; i < dimension; ++i)
                {
                    direction[i] = preconditioned[i] + beta * direction[i];
                }

                rz = rzNext;
            }
        }

        if (!converged)
        {
            return std::unexpected(qcx::Error{
                qcx::ErrorCode::kConvergenceFailure,
                "the response solve for right-hand side " + std::to_string(k) +
                    " did not reach the tolerance in " + std::to_string(_options.maxIterations) +
                    " iterations (final relative residual " + std::to_string(residualNorm) + ")"});
        }

        report.worstResidual = std::max(report.worstResidual, residualNorm / bnorm);

        if (prepared->kind == PreconditionerKind::kRegularised)
        {
            ++report.regularised;
        }

        _previousSolutions[k].assign(x.begin(), x.end());
        _history.push_back(iterations);
    }

    return report;
}

} // namespace qcx::response
