#include "qcx/integrals/ri_occ_k.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <utility>

namespace qcx::integrals {
namespace {

// The row-major n x m view of the contraction layouts (ri_occ_k.hpp): a
// tensor column's rows u*n + v (or u*nOcc + i) are exactly the row-major
// order of its n x m block, so every block below is read and written in
// place - no packing copy and no layout transpose anywhere in this file.
using RowMajorView =
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
using ConstRowMajorView =
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

} // namespace

qcx::Result<Eigen::MatrixXd> MetricInverseRoot(const Eigen::MatrixXd& auxMetric,
                                               double metricFloorEpsilon) {
    if (metricFloorEpsilon < 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "metricFloorEpsilon must not be negative"});
    }

    if (auxMetric.rows() == 0 || auxMetric.rows() != auxMetric.cols())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the auxiliary-basis metric must be a non-empty square matrix"});
    }

    // The floored eigen-inverse, verbatim from the shipped path
    // (ri_engine.cpp: the (P|Q) metric is SPD and extremely ill-conditioned,
    // so the near-null directions are zeroed rather than inverted - the BUG-2
    // fix whose unregularized predecessor amplified them into O(1e9)
    // components that cancel in J).
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(auxMetric);

    if (solver.info() != Eigen::Success)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the auxiliary-basis metric failed its eigendecomposition"});
    }

    const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
    const double lambdaMax = eigenvalues.maxCoeff();
    const double floor = metricFloorEpsilon * lambdaMax;
    Eigen::VectorXd inverseEigenvalues(eigenvalues.size());

    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i)
    {
        inverseEigenvalues(i) = eigenvalues(i) >= floor ? 1.0 / eigenvalues(i) : 0.0;
    }

    if (lambdaMax <= 0.0 || !(inverseEigenvalues.array() > 0.0).any())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the auxiliary-basis metric is degenerate below the RI-J floor"});
    }

    // s = V diag(sqrt(invLambda)) V^T: s s = V diag(invLambda) V^T exactly
    // (V orthogonal), and the floor's zeroed entries contribute exactly zero.
    const Eigen::VectorXd inverseRoots = inverseEigenvalues.cwiseSqrt();
    const auto& eigenvectors = solver.eigenvectors();
    return Eigen::MatrixXd(eigenvectors * inverseRoots.asDiagonal() * eigenvectors.transpose());
}

qcx::Result<Eigen::MatrixXd> BuildMetricTransformedTensor(const Eigen::MatrixXd& riTensor,
                                                          const Eigen::MatrixXd& auxMetric,
                                                          double metricFloorEpsilon) {
    if (metricFloorEpsilon < 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "metricFloorEpsilon must not be negative"});
    }

    if (auxMetric.rows() == 0 || auxMetric.rows() != auxMetric.cols())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the auxiliary-basis metric must be a non-empty square matrix"});
    }

    if (riTensor.rows() == 0 || riTensor.cols() != auxMetric.rows())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the 3-center tensor's columns must match the auxiliary metric's "
                       "dimension"});
    }

    auto root = MetricInverseRoot(auxMetric, metricFloorEpsilon);

    if (!root.has_value())
    {
        return std::unexpected(root.error());
    }

    // The same product the pre-refactor body ran, against the same root: the
    // tensor shape checks above stay ahead of it, so every error this entry
    // point raised before the split it still raises, in the same order.
    return Eigen::MatrixXd(riTensor * *root);
}

qcx::Result<Eigen::MatrixXd> TransformToOccupiedOrbitals(const Eigen::MatrixXd& transformed,
                                                         const Eigen::MatrixXd& occupiedOrbitals) {
    const Eigen::Index n = occupiedOrbitals.rows();
    const Eigen::Index nOcc = occupiedOrbitals.cols();

    if (n == 0 || nOcc == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the occupied orbital block must be a non-empty matrix"});
    }

    if (transformed.rows() != n * n || transformed.cols() == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the transformed tensor's rows must be n*n for n = the occupied "
                       "orbital block's row count, with at least one auxiliary function"});
    }

    Eigen::MatrixXd occTransformed(n * nOcc, transformed.cols());

    for (Eigen::Index p = 0; p < transformed.cols(); ++p)
    {
        const ConstRowMajorView block(transformed.col(p).data(), n, n);
        RowMajorView out(occTransformed.col(p).data(), n, nOcc);
        out.noalias() = block * occupiedOrbitals;
    }

    return occTransformed;
}

qcx::Result<Eigen::MatrixXd> BuildRiExchangeMatrix(const Eigen::MatrixXd& occTransformed,
                                                   std::size_t occupiedCount) {
    if (occupiedCount == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "occupiedCount must be positive"});
    }

    const Eigen::Index nOcc = static_cast<Eigen::Index>(occupiedCount);
    const Eigen::Index rows = occTransformed.rows();

    if (rows == 0 || occTransformed.cols() == 0 || rows % nOcc != 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the occ-transformed tensor's rows must be n*occupiedCount for a "
                       "positive n, with at least one auxiliary function"});
    }

    const Eigen::Index n = rows / nOcc;
    Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index p = 0; p < occTransformed.cols(); ++p)
    {
        const ConstRowMajorView block(occTransformed.col(p).data(), n, nOcc);
        exchange.noalias() += block * block.transpose();
    }

    return exchange;
}

std::size_t RiContractionBlockWidth(std::size_t nSquared,
                                    // (columnCount, maxBatchBytes, blockTargetBytes) is how-many,
                                    // how-wide, how-big - named at the how-many, how-wide, how-big
                                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                    std::size_t columnCount,
                                    std::size_t maxBatchBytes,
                                    std::size_t blockTargetBytes) {
    if (nSquared == 0 || columnCount == 0)
    {
        return 0;
    }

    // One column of the contraction layout: the slab's unit, and the floor both
    // byte arguments are clamped to. A cap or a target smaller than this cannot
    // split a column, so one column per block is the strongest partition the
    // layout admits and the value-neutral answer to both.
    const std::size_t bytesPerColumn = 8 * nSquared;
    const std::size_t cacheColumns = std::max<std::size_t>(1, blockTargetBytes / bytesPerColumn);
    const std::size_t seamColumns = std::max<std::size_t>(1, maxBatchBytes / bytesPerColumn);
    return std::min(columnCount, std::min(cacheColumns, seamColumns));
}

qcx::Result<RiBatchedContractions> BuildBatchedRiContractions(
    const Eigen::MatrixXd& transformed,
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& occupiedOrbitals,
    std::size_t maxBatchBytes,
    std::size_t blockTargetBytes) {
    const Eigen::Index n = occupiedOrbitals.rows();
    const Eigen::Index nOcc = occupiedOrbitals.cols();

    // The chain's own guards, in the chain's own order and wording: a caller
    // that swaps three calls for this one must not see a different error for
    // the same wrong input.
    if (n == 0 || nOcc == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the occupied orbital block must be a non-empty matrix"});
    }

    if (transformed.rows() != n * n || transformed.cols() == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the transformed tensor's rows must be n*n for n = the occupied "
                       "orbital block's row count, with at least one auxiliary function"});
    }

    // The density guard the Coulomb half owns: the chain reached it through the
    // n x n working matrices of the composition, so the shape is stated here
    // rather than assumed from the occupied block.
    if (density.rows() != n || density.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the density's shape must be n x n for n = the occupied orbital "
                       "block's row count"});
    }

    if (maxBatchBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxBatchBytes must be positive"});
    }

    if (blockTargetBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "blockTargetBytes must be positive"});
    }

    const Eigen::Index nAux = transformed.cols();
    const std::size_t width = RiContractionBlockWidth(static_cast<std::size_t>(n * n),
                                                      static_cast<std::size_t>(nAux),
                                                      maxBatchBytes,
                                                      blockTargetBytes);
    // The density in the contraction layout (u*n + v), the form the Coulomb
    // weights contract: the same vector the chain's Coulomb half builds, by the
    // same loop.
    Eigen::VectorXd densityVector(n * n);

    for (Eigen::Index u = 0; u < n; ++u)
    {
        for (Eigen::Index v = 0; v < n; ++v)
        {
            densityVector(u * n + v) = density(u, v);
        }
    }

    RiBatchedContractions products;
    products.blockWidth = width;
    products.exchange = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd coulombVector = Eigen::VectorXd::Zero(n * n);
    // The occ block of one auxiliary block: n*nOcc x width, the buffer whose
    // residency is the second half of the cache argument (it is tiny against
    // the slab, and it is sized by the same width, so it can never be the term
    // the block size was chosen by).
    Eigen::MatrixXd occBlock(n * nOcc, static_cast<Eigen::Index>(width));

    for (Eigen::Index first = 0; first < nAux; first += static_cast<Eigen::Index>(width))
    {
        const Eigen::Index blockWidth = std::min(static_cast<Eigen::Index>(width), nAux - first);
        ++products.blockCount;
        const auto slab = transformed.middleCols(first, blockWidth);

        // The Coulomb half, block-local: w_P = sum_uv B_uv^P d_uv over this
        // block's columns, then J += B_block w_block. Both read the slab, and
        // the second read is the one the block width is chosen to keep in
        // cache.
        const Eigen::VectorXd weights = slab.transpose() * densityVector;
        coulombVector.noalias() += slab * weights;

        // The exchange half, column by column: the occ transform of one column
        // is one n x nOcc product per auxiliary function - the chain's own
        // expression on the chain's own block view, so the occ values are the
        // chain's bit for bit - and its rank-nOcc update lands in K in the same
        // ascending auxiliary order the chain accumulates them in.
        for (Eigen::Index column = 0; column < blockWidth; ++column)
        {
            const ConstRowMajorView block(transformed.col(first + column).data(), n, n);
            RowMajorView out(occBlock.col(column).data(), n, nOcc);
            out.noalias() = block * occupiedOrbitals;
            products.exchange.noalias() += out * out.transpose();
        }
    }

    products.coulomb.resize(n, n);

    for (Eigen::Index u = 0; u < n; ++u)
    {
        for (Eigen::Index v = 0; v < n; ++v)
        {
            products.coulomb(u, v) = coulombVector(u * n + v);
        }
    }

    return products;
}

} // namespace qcx::integrals
