#include "qcx/linalg/iterative_eigensolver.hpp"

#include <Eigen/Core>
#include <Spectra/SymEigsSolver.h>
#include <algorithm>
#include <utility>

namespace qcx::linalg {

namespace {

/// Smallest subspace multiple: Spectra needs ncv strictly between
/// numEigenpairs and the matrix dimension.
constexpr std::size_t kMinSubspaceMultiple = 2;
constexpr std::size_t kMinSubspaceExtra = 1;

/// Dense symmetric matvec operator for Spectra: y = A x with symmetric A.
///
/// Spectra's own DenseSymMatProd routes the product through Eigen's
/// selfadjointView kernel, whose internals the static analyzer's Malloc
/// check false-positives on when instantiated from a PCH'd TU (verified
/// 2026-08-16: four "potential leak" findings in Eigen's
/// SelfadjointMatrixVector.h; clean without the PCH). The two explicit
/// triangular products below are numerically identical for symmetric A,
/// read only the lower triangle (the same half-the-matrix cost), and use
/// Eigen's triangular-product kernel, which is analyzer-clean.
class DenseSymmetricOperator {
public:
    /// The operator's scalar and index types (Spectra's OpType contract).
    using Scalar = double;
    using Index = Eigen::Index;

    explicit DenseSymmetricOperator(const Eigen::MatrixXd& matrix) : _matrix(matrix) {}

    Eigen::Index rows() const noexcept {
        return _matrix.rows();
    }

    Eigen::Index cols() const noexcept {
        return _matrix.cols();
    }

    void perform_op(const double* xIn, double* yOut) const {
        Eigen::Map<const Eigen::VectorXd> x(xIn, _matrix.cols());
        Eigen::Map<Eigen::VectorXd> y(yOut, _matrix.rows());
        y.noalias() = _matrix.template triangularView<Eigen::Lower>() * x +
                      _matrix.template triangularView<Eigen::StrictlyLower>().transpose() * x;
    }

private:
    const Eigen::MatrixXd& _matrix;
};

} // namespace

qcx::Result<std::vector<Eigenpair>> ComputeLowestEigenpairs(
    const Eigen::MatrixXd& matrix,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): count, subspace,
    // iterations is the standard solver-parameter order (see the declaration).
    std::size_t numEigenpairs,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): see above.
    std::size_t numLanczosVectors,
    std::size_t maxIterations,
    double tolerance) {
    const auto dimension = static_cast<std::size_t>(matrix.rows());

    if (matrix.rows() != matrix.cols())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "eigenpairs need a square matrix"});
    }

    if (!matrix.isApprox(matrix.transpose()))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the matrix must be symmetric"});
    }

    if (numEigenpairs == 0 || numEigenpairs >= dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "numEigenpairs must be between 1 and dimension - 1"});
    }

    if (maxIterations == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxIterations must be positive"});
    }

    if (tolerance <= 0.0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "tolerance must be positive"});
    }

    const auto subspace =
        numLanczosVectors == 0
            ? std::min(dimension, kMinSubspaceMultiple * numEigenpairs + kMinSubspaceExtra)
            : numLanczosVectors;

    if (subspace <= numEigenpairs || subspace > dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the Lanczos subspace must be larger than "
                                          "numEigenpairs and at most the matrix dimension"});
    }

    DenseSymmetricOperator op(matrix);
    Spectra::SymEigsSolver<DenseSymmetricOperator> solver(
        op, static_cast<int>(numEigenpairs), static_cast<int>(subspace));
    solver.init();
    const auto converged = solver.compute(Spectra::SortRule::SmallestAlge,
                                          static_cast<int>(maxIterations),
                                          tolerance,
                                          Spectra::SortRule::SmallestAlge);

    if (solver.info() != Spectra::CompInfo::Successful ||
        converged < static_cast<int>(numEigenpairs))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kConvergenceFailure,
                                          "lowest eigenpairs did not converge within the "
                                          "iteration budget"});
    }

    std::vector<Eigenpair> pairs;
    pairs.reserve(static_cast<std::size_t>(converged));

    for (int i = 0; i < converged; ++i)
    {
        pairs.push_back(Eigenpair{solver.eigenvalues()[i], solver.eigenvectors().col(i)});
    }

    std::sort(pairs.begin(), pairs.end(), [](const Eigenpair& a, const Eigenpair& b) {
        return a.eigenvalue < b.eigenvalue;
    });

    return pairs;
}

} // namespace qcx::linalg
