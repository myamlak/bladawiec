#pragma once

#include "qcx/error.hpp"

#include <cstddef>
#include <vector>

namespace qcx::linalg {

/// Solves A x = b for a real square sparse system given in CSR form, via
/// AMGCL's smoothed-aggregation AMG preconditioner with a conjugate-gradient
/// solver [Demidov2019].
///
/// The CSR triple is the same index-array shape as
/// memory::SparsityPattern - the same sparse-index primitive - so LinK/QFMM/domain-list
/// index structures feed this solver without conversion.
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
// 1e-8: the project's own energy rung, and the closest analogue a
// relative-residual tolerance has to it. The default was 1e-10 - below
// anything a caller of this seam asserts, and a gate at or below 1e-10 is
// not accepted. A caller that genuinely needs a tighter residual passes it
// explicitly.
qcx::Result<std::vector<double>> SolveSparseSystem(const std::vector<std::size_t>& rowOffsets,
                                                   const std::vector<std::size_t>& columnIndices,
                                                   const std::vector<double>& values,
                                                   const std::vector<double>& rhs,
                                                   double tolerance = 1e-8);

} // namespace qcx::linalg
