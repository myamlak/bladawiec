#pragma once

/// \file
/// Explicit assembly of a response operator from its action: what the dense
/// and explicit-CSR bands of the solver ladder consume, and the one place the
/// matrix-free seam is turned into a matrix.
///
/// The assembly applies the operator to every unit vector, so it costs
/// Dimension() matvecs and never asks the operator to expose its matrix - the
/// operator seam stays action-only. No symmetry is assumed: every entry is
/// filled from the action, including the entries on either side of the
/// diagonal. A symmetric operator assembles symmetric because its action says
/// so, not because the assembler copies one triangle.
///
/// Memory is the reason these functions exist beside the selector rather than
/// under it: a dense matrix is Dimension()^2 doubles, and the sparse form
/// holds a dense Dimension() x Dimension() temporary while it compresses. At
/// the explicit-CSR band's 5000-unknown ceiling the temporary alone is 200 MB,
/// which is the cost the band's nonzero cap guards.

#include "qcx/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace qcx::response {

/// \ingroup qcx-response
/// A response operator assembled as a dense row-major matrix.
struct DenseResponseMatrix {
    std::size_t dimension = 0; ///< Rows and columns.
    std::vector<double> values; ///< Row-major entries, size dimension^2.
};

/// \ingroup qcx-response
/// A response operator assembled in compressed sparse row form.
///
/// The index arrays have the shape qcx::linalg::SolveSparseSystem takes, so a
/// matrix assembled here feeds the sparse solver without conversion.
struct SparseResponseMatrix {
    std::size_t dimension = 0; ///< Rows and columns.
    std::vector<std::size_t> rowOffsets; ///< CSR offsets, size dimension + 1, first element 0.
    std::vector<std::size_t> columnIndices; ///< Packed column indices.
    std::vector<double> values; ///< Packed values, same size as \p columnIndices.
};

/// \ingroup qcx-response
/// What the sparse assembly keeps.
struct ResponseMatrixOptions {
    /// Entries with |value| <= dropTolerance are omitted from the sparse form.
    /// The default, 0.0, omits exact zeros only; a positive tolerance also
    /// omits entries below it. The dense form stores every entry regardless.
    /// A NaN entry is never dropped: no comparison holds for it.
    double dropTolerance = 0.0;
};

/// \ingroup qcx-response
/// Assembles the operator's matrix in dense row-major form by applying the
/// operator to every unit vector.
/// \tparam Operator A type with `std::size_t Dimension() const` and
/// `qcx::Result<void> Apply(std::span<const double>, std::span<double>) const`
/// - satisfied by OrbitalHessianOperator and by DenseResponseOperator.
/// \param op The operator.
/// \returns The assembled matrix, or an Error (kInvalidArgument when the
/// operator is empty or its Dimension()^2 does not fit the size type; an
/// operator's own error is propagated unchanged).
template <typename Operator>
qcx::Result<DenseResponseMatrix> AssembleDenseResponseMatrix(const Operator& op) {
    const std::size_t dimension = op.Dimension();

    if (dimension == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the response operator is empty"});
    }

    if (dimension > std::numeric_limits<std::size_t>::max() / dimension)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the dense matrix would be larger than the addressable size"});
    }

    DenseResponseMatrix matrix;
    matrix.dimension = dimension;
    matrix.values.assign(dimension * dimension, 0.0);

    std::vector<double> trial(dimension, 0.0);
    std::vector<double> product(dimension, 0.0);

    for (std::size_t column = 0; column < dimension; ++column)
    {
        std::fill(trial.begin(), trial.end(), 0.0);
        trial[column] = 1.0;

        auto status = op.Apply(trial, product);

        if (!status.has_value())
        {
            return std::unexpected(status.error());
        }

        for (std::size_t row = 0; row < dimension; ++row)
        {
            matrix.values[row * dimension + column] = product[row];
        }
    }

    return matrix;
}

/// \ingroup qcx-response
/// Assembles the operator's matrix in compressed sparse row form by applying
/// the operator to every unit vector and compressing the result.
///
/// The compression runs over a dense Dimension() x Dimension() temporary: the
/// action produces a column at a time, and the CSR form is written a row at a
/// time.
/// \tparam Operator A type with `std::size_t Dimension() const` and
/// `qcx::Result<void> Apply(std::span<const double>, std::span<double>) const`
/// - satisfied by OrbitalHessianOperator and by DenseResponseOperator.
/// \param op The operator.
/// \param options What the assembly keeps.
/// \returns The assembled matrix, or an Error (kInvalidArgument for a negative
/// or NaN tolerance, an empty operator, or an unaddressable temporary; an
/// operator's own error is propagated unchanged).
template <typename Operator>
qcx::Result<SparseResponseMatrix> AssembleSparseResponseMatrix(const Operator& op,
                                                               ResponseMatrixOptions options = {}) {
    if (!(options.dropTolerance >= 0.0))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the drop tolerance must be a non-negative number"});
    }

    auto dense = AssembleDenseResponseMatrix(op);

    if (!dense.has_value())
    {
        return std::unexpected(dense.error());
    }

    const std::size_t dimension = dense->dimension;
    SparseResponseMatrix matrix;
    matrix.dimension = dimension;
    matrix.rowOffsets.reserve(dimension + 1);
    matrix.rowOffsets.push_back(0);

    for (std::size_t row = 0; row < dimension; ++row)
    {
        for (std::size_t column = 0; column < dimension; ++column)
        {
            const double value = dense->values[row * dimension + column];

            // A NaN is kept: the comparison is false for it, and dropping a
            // value the operator produced would hide the operator's fault.
            if (!(std::fabs(value) <= options.dropTolerance))
            {
                matrix.columnIndices.push_back(column);
                matrix.values.push_back(value);
            }
        }

        matrix.rowOffsets.push_back(matrix.values.size());
    }

    return matrix;
}

} // namespace qcx::response
