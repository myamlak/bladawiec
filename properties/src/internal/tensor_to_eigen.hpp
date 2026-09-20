#pragma once

// Module-private conversion of a rank-2 CPU tensor into an Eigen matrix:
// explicit element copies pin the row-major to column-major ordering
// instead of aliasing Eigen storage conventions over tensor data. The
// properties module consumes the integrals builders' tensors but must not
// depend on the test-only qcx::testing::ToMatrix (tests/fixtures). Not
// part of the public API.

#include "qcx/memory/tensor.hpp"

#include <Eigen/Core>
#include <cstddef>

namespace qcx::properties::internal {

/// Copies a rank-2 CPU tensor into an Eigen matrix element by element.
/// \param tensor Square rank-2 tensor, host-canonical.
/// \returns The square matrix copy.
inline Eigen::MatrixXd TensorToEigen(
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

} // namespace qcx::properties::internal
