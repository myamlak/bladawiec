#include "qcx/linalg/sparse_solver.hpp"

#include <algorithm>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/backend/interface.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <span>
#include <string>
#include <tuple>
#include <utility>

namespace qcx::linalg {

namespace {

using Backend = amgcl::backend::builtin<double>;
using AmgclSolver = amgcl::make_solver<
    amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>,
    amgcl::solver::cg<Backend>>;

/// amgcl's CRS stores indices as ptrdiff_t; the public API stays size_t and
/// converts here.
constexpr std::ptrdiff_t kIndexLimit = std::numeric_limits<std::ptrdiff_t>::max();

/// Fewer unknowns than this stay on the coarse level's direct solve.
constexpr int kCoarseEnough = 64;

} // namespace

qcx::Result<SparseSolveReport> SolveSparseSystemMultipleRhs(
    const std::vector<std::size_t>& rowOffsets,
    const std::vector<std::size_t>& columnIndices,
    const std::vector<double>& values,
    RightHandSides rhs,
    double tolerance) {
    const std::span<const double> rightHandSides = rhs.values;
    const std::size_t rhsCount = rhs.count;

    if (rhsCount == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "a solve needs at least one right-hand side"});
    }

    if (rowOffsets.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "rowOffsets must have size n + 1"});
    }

    const std::size_t n = rowOffsets.size() - 1;

    if (n == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument, "the system is empty"});
    }

    if (rhsCount > std::numeric_limits<std::size_t>::max() / n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "rhsCount times the system size overflows the size type"});
    }

    if (rightHandSides.size() != rhsCount * n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "rhs must hold rhsCount * n values"});
    }

    if (rowOffsets.front() != 0 || rowOffsets.back() != values.size() ||
        !std::is_sorted(rowOffsets.begin(), rowOffsets.end()))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "CSR offsets must start at 0, end at the value "
                                          "count, and be monotonic"});
    }

    if (columnIndices.size() != values.size())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "columnIndices and values must have the same size"});
    }

    if (values.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the matrix has no entries"});
    }

    if (tolerance <= 0.0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "tolerance must be positive"});
    }

    if (std::any_of(columnIndices.begin(), columnIndices.end(), [n](std::size_t column) {
            return column >= n;
        }))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "column index out of range"});
    }

    const auto indicesFit = [](const std::vector<std::size_t>& indices) {
        return std::all_of(indices.begin(), indices.end(), [](std::size_t index) {
            return index <= static_cast<std::size_t>(kIndexLimit);
        });
    };

    if (!indicesFit(rowOffsets) || !indicesFit(columnIndices))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "indices exceed amgcl's ptrdiff_t range"});
    }

    std::vector<std::ptrdiff_t> ptr(rowOffsets.size());
    std::vector<std::ptrdiff_t> col(columnIndices.size());

    std::transform(rowOffsets.begin(), rowOffsets.end(), ptr.begin(), [](std::size_t index) {
        return static_cast<std::ptrdiff_t>(index);
    });
    std::transform(columnIndices.begin(), columnIndices.end(), col.begin(), [](std::size_t index) {
        return static_cast<std::ptrdiff_t>(index);
    });

    // Typed parameter structs: amgcl is built with AMGCL_NO_BOOST, so
    // the stringly-typed property_tree path is replaced by named members.
    AmgclSolver::params params;
    params.solver.tol = tolerance;
    params.precond.coarse_enough = kCoarseEnough;

    try
    {
        // ONE hierarchy for every right-hand side: the construction is the
        // expensive half, and the loop below reuses it.
        AmgclSolver solver(std::tie(n, ptr, col, values), params);

        SparseSolveReport report;
        report.solution.assign(rhsCount * n, 0.0);
        std::vector<double> side(n, 0.0);
        std::vector<double> answer(n, 0.0);

        for (std::size_t k = 0; k < rhsCount; ++k)
        {
            std::copy(rightHandSides.begin() + static_cast<std::ptrdiff_t>(k * n),
                      rightHandSides.begin() + static_cast<std::ptrdiff_t>((k + 1) * n),
                      side.begin());
            std::fill(answer.begin(), answer.end(), 0.0);

            const auto solveResult = solver(side, answer);
            const auto error = std::get<1>(solveResult);
            report.iterations += std::get<0>(solveResult);

            // NaN is NOT convergence: amgcl reports the relative residual, and
            // "NaN >= tolerance" is false, so a NaN residual used to sail
            // through as success with a garbage solution (NaN input or a CG
            // breakdown). Inf is caught by the comparison; NaN needs the
            // explicit finiteness check (the documented kConvergenceFailure
            // contract, sparse_solver.hpp).
            if (!std::isfinite(error) || error >= tolerance)
            {
                std::string message = "AMGCL did not reach the requested tolerance";

                if (rhsCount > 1)
                {
                    message += " for right-hand side " + std::to_string(k);
                }

                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kConvergenceFailure, std::move(message)});
            }

            report.worstResidual = std::max(report.worstResidual, static_cast<double>(error));
            std::copy(answer.begin(),
                      answer.end(),
                      report.solution.begin() + static_cast<std::ptrdiff_t>(k * n));
        }

        return report;
    } catch (const std::exception& exception)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError,
                       std::string("AMGCL setup or solve failed: ") + exception.what()});
    }
}

qcx::Result<std::vector<double>> SolveSparseSystem(const std::vector<std::size_t>& rowOffsets,
                                                   const std::vector<std::size_t>& columnIndices,
                                                   const std::vector<double>& values,
                                                   const std::vector<double>& rhs,
                                                   double tolerance) {
    auto solved = SolveSparseSystemMultipleRhs(
        rowOffsets, columnIndices, values, RightHandSides{rhs, 1}, tolerance);

    if (!solved.has_value())
    {
        return std::unexpected(solved.error());
    }

    return std::move(solved->solution);
}

} // namespace qcx::linalg
