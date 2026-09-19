#pragma once

#include "qcx/error.hpp"

#include <Eigen/Dense>
#include <cstddef>

namespace qcx::linalg {

/// \defgroup qcx-linalg Linalg module
/// Dense linear algebra: Eigen for small fixed-size blocks, an optional
/// vendor BLAS (MKL/AOCL/OpenBLAS) for larger ones.
/// \{

/// Matrices at or below this size (rows and columns) stay on Eigen's path.
constexpr std::size_t kSmallBlockThreshold = 64;

#ifdef QcxHasVendorBlas

/// Matrix product via the configured vendor BLAS (uniform CBLAS interface).
///
/// Exposed separately so tests can cross-check it against Eigen's own
/// product on the same inputs. Declared only when a vendor BLAS was found
/// at configure time.
/// \param a Left factor (m x k).
/// \param b Right factor (k x n).
/// \returns The product (m x n), or an Error on dimension mismatch.
qcx::Result<Eigen::MatrixXd> BlasMultiply(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

#endif

/// Matrix product: Eigen for small blocks, vendor BLAS above
/// kSmallBlockThreshold; with no vendor configured, Eigen handles all sizes.
/// \param a Left factor (m x k).
/// \param b Right factor (k x n).
/// \returns The product (m x n), or an Error on dimension mismatch.
inline qcx::Result<Eigen::MatrixXd> DenseMultiply(const Eigen::MatrixXd& a,
                                                  const Eigen::MatrixXd& b) {
    if (a.cols() != b.rows())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "matrix dimension mismatch"});
    }

#ifdef QcxHasVendorBlas

    if (static_cast<std::size_t>(a.rows()) > kSmallBlockThreshold ||
        static_cast<std::size_t>(b.cols()) > kSmallBlockThreshold)
    {
        return BlasMultiply(a, b);
    }

#endif
    return a * b;
}

/// \}
} // namespace qcx::linalg
