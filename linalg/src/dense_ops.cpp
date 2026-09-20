#include "qcx/linalg/dense_ops.hpp"

#ifdef QcxHasVendorBlas

extern "C" {
#ifdef QcxVendorBlasIsMkl
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
}

namespace qcx::linalg {

qcx::Result<Eigen::MatrixXd> BlasMultiply(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.cols() != b.rows())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "matrix dimension mismatch"});
    }

    const int m = static_cast<int>(a.rows());
    const int k = static_cast<int>(a.cols());
    const int n = static_cast<int>(b.cols());

    Eigen::MatrixXd result(m, n);
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                m,
                n,
                k,
                1.0,
                a.data(),
                m,
                b.data(),
                k,
                0.0,
                result.data(),
                m);
    return result;
}

} // namespace qcx::linalg

#endif
