#pragma once

#include "qcx/error.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace qcx::linalg {

/// What a multi-right-hand-side sparse solve returns: the concatenated
/// solution and the cost the solve took.
struct SparseSolveReport {
    /// Concatenated solutions, right-hand-side-major: side k occupies
    /// [k * n, (k + 1) * n).
    std::vector<double> solution;

    /// Conjugate-gradient iterations in total, summed over the right-hand
    /// sides.
    std::size_t iterations = 0;

    /// Worst relative residual AMGCL reported across the right-hand sides.
    double worstResidual = 0.0;
};

/// Solves A x = b for one or more real square sparse systems given in CSR
/// form, via AMGCL's smoothed-aggregation AMG preconditioner with a
/// conjugate-gradient solver [Demidov2019], building ONE hierarchy for every
/// right-hand side and applying it once per side.
///
/// The CSR triple is the same index-array shape as
/// memory::SparsityPattern - the sparse index primitive - so LinK/QFMM/domain-list
/// index structures feed this solver without conversion.
///
/// The hierarchy build is the expensive half of a sparse solve, so a caller
/// with several right-hand sides against one matrix should pass them together
/// rather than call the single-right-hand-side entry once per side.
/// \param rowOffsets CSR offsets, size n + 1, first element 0, monotonic,
/// last element equal to the packed size.
/// \param columnIndices Packed column indices, same size as \p values.
/// \param values Packed values, same size as \p columnIndices.
/// \param rhs Concatenated right-hand sides, right-hand-side-major: side k
/// occupies [k * n, (k + 1) * n).
/// \param rhsCount Number of right-hand sides; must be positive.
/// \param tolerance Relative residual tolerance for convergence; must be
/// positive.
/// \returns The concatenated solution and the solve's cost, or an Error
/// (kInvalidArgument for shape mismatches or a zero count, kConvergenceFailure
/// when a right-hand side does not meet the tolerance - the message names the
/// side when there is more than one - and kInternalError when the solver setup
/// or a solve throws).
// 1e-8: the project's own energy rung, and the closest analogue a
// relative-residual tolerance has to it. The default was 1e-10, which sits
// below anything a caller of this seam asserts, and it is retired: no
// convergence gate at or below 1e-10 is offered. A caller that genuinely
// needs a tighter residual passes it explicitly.
qcx::Result<SparseSolveReport> SolveSparseSystemMultipleRhs(
    const std::vector<std::size_t>& rowOffsets,
    const std::vector<std::size_t>& columnIndices,
    const std::vector<double>& values,
    std::span<const double> rhs,
    std::size_t rhsCount,
    double tolerance = 1e-8);

/// Solves A x = b for one real square sparse system given in CSR form, via
/// the same AMGCL hierarchy the multi-right-hand-side entry builds [Demidov2019].
///
/// This entry delegates to SolveSparseSystemMultipleRhs with one right-hand
/// side, so the validation, the hierarchy parameters and the convergence
/// decision are shared rather than duplicated.
/// \param rowOffsets CSR offsets, size n + 1, first element 0, monotonic,
/// last element equal to the packed size.
/// \param columnIndices Packed column indices, same size as \p values.
/// \param values Packed values, same size as \p columnIndices.
/// \param rhs Right-hand side, size n.
/// \param tolerance Relative residual tolerance for convergence; must be
/// positive.
/// \returns The solution vector, or an Error (kInvalidArgument for shape
/// mismatches, kConvergenceFailure when the tolerance is not met,
/// kInternalError when the solver setup throws).
qcx::Result<std::vector<double>> SolveSparseSystem(const std::vector<std::size_t>& rowOffsets,
                                                   const std::vector<std::size_t>& columnIndices,
                                                   const std::vector<double>& values,
                                                   const std::vector<double>& rhs,
                                                   double tolerance = 1e-8);

} // namespace qcx::linalg
