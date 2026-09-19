#pragma once

#include "qcx/error.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <vector>

namespace qcx::linalg {

/// One eigenpair of a real symmetric matrix: eigenvalue plus eigenvector.
/// \ingroup qcx-linalg
struct Eigenpair {
    double eigenvalue; ///< Eigenvalue (ascending order in ComputeLowestEigenpairs).
    Eigen::VectorXd eigenvector; ///< Corresponding eigenvector, Euclidean norm 1.
};

/// Computes the \p numEigenpairs algebraically smallest eigenpairs of a real
/// symmetric matrix via Spectra's implicitly restarted Lanczos method
/// [Qiu2015] - the iterative eigenproblem seam for systems where a full
/// dense diagonalization is out of reach (SCF of large bases, later CI).
///
/// The matrix is passed as a dense Eigen type for now: Spectra needs only
/// operator* and the dimensions, so a sparse/operator-valued overload plugs
/// into the same seam when one is needed.
/// \param matrix Real symmetric matrix (symmetry is checked).
/// \param numEigenpairs Number of lowest eigenpairs; between 1 and dimension - 1.
/// \param numLanczosVectors Subspace dimension; 0 picks 2 * numEigenpairs + 1,
/// capped at the matrix dimension.
/// \param maxIterations Iteration budget; must be positive.
/// \param tolerance Residual tolerance; must be positive.
/// \returns Ascending eigenpairs, or an Error (kInvalidArgument for bad
/// shapes or budgets, kConvergenceFailure when the budget runs out).
qcx::Result<std::vector<Eigenpair>> ComputeLowestEigenpairs(
    const Eigen::MatrixXd& matrix,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): count, subspace,
    // iterations is the standard solver-parameter order (kept named at the
    // call site by design; see the nanotube fixture precedent in
    // tests/fixtures).
    std::size_t numEigenpairs,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters): see above.
    std::size_t numLanczosVectors = 0,
    std::size_t maxIterations = 1000,
    // 1e-8: the project's own energy rung, and the closest analogue a
    // residual tolerance has to it. The default was 1e-12 - four orders
    // below the rung and below anything a caller of this seam asserts -
    // and no convergence gate at or below 1e-10 is accepted. A caller that
    // genuinely needs a tighter residual passes it explicitly.
    double tolerance = 1e-8);

} // namespace qcx::linalg
