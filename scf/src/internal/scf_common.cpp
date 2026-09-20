// Shared step-function implementations: moved verbatim out of rhf.cpp's
// anonymous namespace so UHF can call them
// across translation units. Pure code motion - no behavior change.
#include "scf_common.hpp"

#include "qcx/linalg/dense_ops.hpp"

#include <utility>

namespace qcx::scf::internal {

JkSupermatrices BuildJkSupermatrices(
    const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri, std::size_t n) {
    // Eigen indices are signed (ptrdiff_t): the supermatrix dimension is
    // cast once here, and every Eigen call below indexes with Eigen::Index.
    const Eigen::Index eigenN2 = static_cast<Eigen::Index>(n * n);
    Eigen::MatrixXd coulomb = Eigen::MatrixXd::Zero(eigenN2, eigenN2);
    Eigen::MatrixXd exchange = Eigen::MatrixXd::Zero(eigenN2, eigenN2);

    for (std::size_t mu = 0; mu < n; ++mu)
    {
        for (std::size_t nu = 0; nu < n; ++nu)
        {
            const Eigen::Index row = static_cast<Eigen::Index>(mu * n + nu);

            for (std::size_t l = 0; l < n; ++l)
            {
                for (std::size_t s = 0; s < n; ++s)
                {
                    const Eigen::Index col = static_cast<Eigen::Index>(l * n + s);
                    coulomb(row, col) = eri(mu, nu, l, s);
                    exchange(row, col) = eri(mu, l, s, nu);
                }
            }
        }
    }

    return {std::move(coulomb), std::move(exchange)};
}

// Loewdin symmetric orthogonalization [Lowdin1950] with linear-dependence
// removal: X = U_kept diag(1/sqrt(s_kept)).
//
// SelfAdjointEigenSolver covers the full dense spectrum - the Spectra seam
// (linalg::ComputeLowestEigenpairs) is deliberately not used here: the SCF
// needs every eigenvector, and Spectra targets the lowest few of large
// matrices.
//
// The removal replaces the former abort. 1/sqrt(s) is applied
// unconditionally, so a
// near-singular overlap produces an enormous or infinite orthogonalizer and
// the transformed Fock carries NaN or absurd energies with no signal. The
// old response surfaced that as an error; the response is to DROP the
// offending directions instead, because in a large diffuse basis they are
// expected rather than exceptional. Measured 2026-09-14: aug-cc-pVTZ on
// C24H50 sits at s_min/s_max = 4.92e-09 with one direction below the floor,
// so the old behaviour was that a 74-atom alkane in a diffuse basis simply
// would not run (the threshold is a SYSTEM property - basis and molecule -
// and is not a per-method switch; see the constant's comment).
qcx::Result<Orthogonalization> OrthogonalizeOverlap(const Eigen::MatrixXd& overlap) {
    if (overlap.rows() == 0 || overlap.rows() != overlap.cols())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the overlap matrix must be square and non-empty"});
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap);

    if (solver.info() != Eigen::Success)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the overlap matrix could not be diagonalized (not symmetric)"});
    }

    const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
    const Eigen::Index n = eigenvalues.size();
    const double largestEigenvalue = eigenvalues.maxCoeff();
    const double floor = kOverlapEigenvalueFloorTolerance * largestEigenvalue;

    // Eigenvalues ascend, so every direction at or below the floor is a
    // PREFIX of the spectrum and the kept ones are the trailing columns.
    Eigen::Index numKept = 0;

    for (Eigen::Index i = 0; i < n; ++i)
    {
        if (eigenvalues(i) >= floor)
        {
            ++numKept;
        }
    }

    if (numKept == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "every overlap eigenvector is below the relative floor; no orthonormal "
                       "space remains"});
    }

    const Eigen::MatrixXd uKept = solver.eigenvectors().rightCols(numKept);
    const Eigen::VectorXd sKept = eigenvalues.tail(numKept);

    Orthogonalization result;
    result.largestEigenvalue = largestEigenvalue;
    result.smallestKeptEigenvalue = sKept.minCoeff();
    result.numRemoved = static_cast<std::size_t>(n - numKept);

    if (result.numRemoved == 0)
    {
        // Nothing to remove: keep the classic SYMMETRIC Loewdin form
        // X = U diag(1/sqrt(s)) U^T exactly as it was. This matters beyond
        // bit-compatibility - X = S^{-1/2} commutes with the point group, so
        // U_b^T X U_b is block-diagonal, and the per-irrep blocked
        // diagonalization is built on that. The rectangular form
        // below is a different orthogonalizer (it differs by a right
        // multiplication by U) and does NOT have that property, so it is
        // used only when directions were actually removed - where the
        // blocked path falls back to the plain solve anyway.
        result.x = uKept * sKept.array().rsqrt().matrix().asDiagonal() * uKept.transpose();
        return result;
    }

    // Directions were removed: X = U_kept diag(1/sqrt(s_kept)) is n x numKept
    // and X^T S X = I_numKept, so the SCF runs in the numKept-dimensional
    // orthonormal space. Callers map back with C_AO = X C_orth and
    // D_AO = X D_orth X^T (n x n, full AO dimension).
    result.x = uKept * sKept.array().rsqrt().matrix().asDiagonal();
    return result;
}

// Diagonalizes the Fock matrix in the orthogonal basis and transforms the
// eigenvectors back to the AO basis: C = X * V, columns ascending by energy.
Eigen::MatrixXd DiagonalizeFock(const Eigen::MatrixXd& fock, const Eigen::MatrixXd& x) {
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(x.transpose() * fock * x);
    return x * solver.eigenvectors();
}

} // namespace qcx::scf::internal
