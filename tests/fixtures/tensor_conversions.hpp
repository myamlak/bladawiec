#pragma once

// Test-only conversions between rank-2 CPU tensors and Eigen matrices.
// Explicit element copies pin the row-major to column-major ordering
// instead of aliasing Eigen storage conventions over tensor data. Not part
// of the public API.

#include "qcx/memory/tensor.hpp"

#include <Eigen/Dense>
#include <cstddef>

namespace qcx::testing {

/// Copies a rank-2 CPU tensor into an Eigen matrix element by element.
/// \param tensor Square rank-2 tensor, host-canonical.
/// \returns The square matrix copy.
inline Eigen::MatrixXd ToMatrix(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& tensor) {
    const std::size_t n = tensor.Shape()[0];
    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = tensor(i, j);
        }
    }

    return matrix;
}

/// Copies an Eigen matrix into a rank-2 CPU tensor element by element (the
/// inverse of ToMatrix; the FockBuilderFn adapter glue uses it).
/// \param matrix Square Eigen matrix.
/// \returns The tensor copy, host-canonical, or an Error when the
/// allocation fails.
inline qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> ToTensor(
    const Eigen::MatrixXd& matrix) {
    const std::size_t n = static_cast<std::size_t>(matrix.rows());
    auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*tensor)(i, j) = matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

} // namespace qcx::testing
