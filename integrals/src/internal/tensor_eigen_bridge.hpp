#pragma once

// The Tensor <-> Eigen boundary: Tensor is
// row-major, Eigen::MatrixXd is column-major (Eigen's default - no
// RowMajor template argument is used for the plain MatrixXd alias anywhere
// in this codebase), so no element-copy loop order can be cache-friendly on
// both sides at once. These two helpers are the single shared
// implementation, replacing the pattern previously duplicated (and
// independently drifting) in fock_build.cpp (x3: the core-Hamiltonian,
// density, and Fock-return copies), incremental_fock.cpp (ToEigen and
// EigenToTensor), ri_engine.cpp (ToMatrix), qfmm_fock_build.cpp (x3), and
// eri_cuda.cpp (x2). Every real call site is square (n == m - densities,
// Fock matrices, and the core Hamiltonian are all n x n), so the general
// rectangular handling below is unexercised in production; it is kept
// because nothing in these helpers requires the square case.
//
// The per-direction rule is "make the DESTINATION side's writes contiguous,
// since writes benefit more from good locality than reads do", applied
// separately to each direction - deliberately NOT the same loop order in
// both functions:
//   * TensorToEigen: j outer / i inner, so the Eigen destination is written
//     contiguously (column-major: one whole column per outer step) at the
//     cost of strided reads from the row-major Tensor source. Every
//     consumer of the result (the Fock-build/RI arithmetic) immediately
//     does further column-major-friendly work on the matrix, so a
//     contiguously-written matrix pays off downstream too.
//   * EigenToTensor: i outer / j inner, so the Tensor destination is
//     written contiguously (row-major: one whole row per outer step) at the
//     cost of strided reads from the column-major Eigen source. This is the
//     ORIGINAL loop order the pre-consolidation copies already used for
//     this direction - kept, not flipped.

#include "qcx/memory/tensor.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <utility>

namespace qcx::integrals::internal {

/// Copies a rank-2 CPU tensor into an Eigen matrix element by element.
///
/// The destination-oriented loop order: j (column) outer, i (row) inner
/// (see the file comment for the per-direction rationale).
/// \param tensor The host-canonical tensor to copy.
/// \returns The matrix copy.
inline Eigen::MatrixXd TensorToEigen(
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& tensor) {
    const std::size_t n = tensor.Shape()[0];
    const std::size_t m = tensor.Shape()[1];
    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(m));

    // j (column) outer, i (row) inner: contiguous writes into `matrix`
    // (column-major - a whole column is contiguous), strided reads from
    // `tensor` (row-major - a whole row is contiguous, so this reads one
    // element per row per outer step). Writes benefit more from being
    // contiguous than reads do, and every consumer of the result
    // (fock_build.cpp/ri_engine.cpp's actual arithmetic) immediately does
    // further column-major-friendly work on `matrix`, so a contiguously
    // written matrix is the one that pays off downstream too, not just
    // during this copy.
    for (std::size_t j = 0; j < m; ++j)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = tensor(i, j);
        }
    }

    return matrix;
}

/// Copies an Eigen matrix into a rank-2 CPU tensor element by element.
///
/// The destination-oriented loop order: i (row) outer, j (column) inner -
/// the mirror-image conclusion of TensorToEigen, NOT the same loop order
/// (see the file comment for the per-direction rationale).
/// \param matrix The matrix to copy.
/// \returns The host-canonical tensor copy, or an Error when the
/// allocation fails.
inline qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> EigenToTensor(
    const Eigen::MatrixXd& matrix) {
    const std::size_t n = static_cast<std::size_t>(matrix.rows());
    const std::size_t m = static_cast<std::size_t>(matrix.cols());
    auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, m});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    // Mirrored reasoning, opposite conclusion: here Tensor is the
    // destination, so i outer / j inner keeps the TENSOR side's writes
    // contiguous (row-major: a whole row per outer step) at the cost of
    // strided reads from `matrix`. This is the ORIGINAL loop order already
    // used at the pre-consolidation fock_build.cpp/incremental_fock.cpp
    // return paths for exactly this direction - keep it, don't flip it;
    // only the Tensor -> Eigen direction (above) needed the fix.
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < m; ++j)
        {
            (*tensor)(i, j) = matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

} // namespace qcx::integrals::internal
