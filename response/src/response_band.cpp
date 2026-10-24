#include "qcx/response/response_band.hpp"

#include <Eigen/Core>
#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

namespace qcx::response {

std::string_view ToString(SolverBand band) {
    switch (band)
    {
    case SolverBand::kDense:
        return "dense";
    case SolverBand::kExplicitCsr:
        return "explicit-csr";
    case SolverBand::kMatrixFree:
        return "matrix-free";
    }

    return "unknown";
}

qcx::Result<SolverBand> SelectResponseBand(const ResponseBandProblem& problem,
                                           const ResponseBandLimits& limits) {
    if (problem.dimension == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the response problem is empty"});
    }

    if (problem.dimension <= limits.denseMaxDimension)
    {
        return SolverBand::kDense;
    }

    if (problem.dimension <= limits.csrMaxDimension &&
        problem.estimatedNonzeros <= limits.csrMaxNonzeros)
    {
        return SolverBand::kExplicitCsr;
    }

    return SolverBand::kMatrixFree;
}

namespace detail {

qcx::Result<std::vector<double>> SolveDenseSystem(std::size_t dimension,
                                                  std::span<const double> matrix,
                                                  std::size_t rhsCount,
                                                  std::span<const double> rhs) {
    if (dimension == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the dense system is empty"});
    }

    if (dimension > static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the dense system is larger than Eigen's index range"});
    }

    if (dimension > std::numeric_limits<std::size_t>::max() / dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the dense matrix would be larger than the addressable "
                                          "size"});
    }

    if (matrix.size() != dimension * dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the dense matrix must hold dimension^2 values"});
    }

    if (rhsCount == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "a solve needs at least one right-hand side"});
    }

    if (rhsCount > std::numeric_limits<std::size_t>::max() / dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "rhsCount times the dimension overflows the size type"});
    }

    if (rhs.size() != rhsCount * dimension)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "rhs must hold rhsCount * dimension values"});
    }

    const auto size = static_cast<Eigen::Index>(dimension);
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        assembled(matrix.data(), size, size);
    const Eigen::MatrixXd dense = assembled;

    // Full pivoting, not partial: the band refuses a singular operator by
    // name, and FullPivLU is the decomposition that reports invertibility.
    // The dense band's ceiling is 128, so the extra pivoting cost is bounded.
    const Eigen::FullPivLU<Eigen::MatrixXd> factorisation(dense);

    // A direct band that returned a garbage answer for a singular operator
    // would hide the operator's fault behind the solver's; it refuses instead.
    if (!factorisation.isInvertible())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the assembled response matrix is singular"});
    }

    std::vector<double> solution(rhsCount * dimension);

    for (std::size_t k = 0; k < rhsCount; ++k)
    {
        const Eigen::Map<const Eigen::VectorXd> side(rhs.data() + k * dimension, size);
        const Eigen::VectorXd answer = factorisation.solve(side);
        std::copy(answer.data(),
                  answer.data() + static_cast<Eigen::Index>(dimension),
                  solution.begin() + static_cast<std::ptrdiff_t>(k * dimension));
    }

    return solution;
}

} // namespace detail

} // namespace qcx::response
