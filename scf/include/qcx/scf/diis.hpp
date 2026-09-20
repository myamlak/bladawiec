#pragma once

/// \file
/// CDIIS convergence acceleration for the SCF loop.
///
/// The direct inversion of the iterative subspace [Pulay1980]: a short
/// history of Fock matrices and their error vectors is combined linearly to
/// extrapolate the next Fock matrix, minimizing the error-vector norm under
/// the unit-sum constraint.
///
/// The error history is stored in the exact factored form (DiisErrorFactors)
/// wherever the caller holds the density's own factor: one error vector costs
/// 2 n n_occ doubles plus an n_occ x n_occ block instead of the n^2 doubles of
/// the full square it represents. That history is the DIIS contribution to a
/// large-basis run's footprint, and it is the one term a run cannot trade away
/// by choosing a cheaper algorithm (the Fock and density terms move to lighter
/// rungs of the memory ladder; the DIIS state is the same size on every rung).

#include "qcx/error.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::scf {

/// A serializable snapshot of the extrapolator's history (checkpointing):
/// value semantics for storage.
///
/// The error entries are the full n x n squares, expanded from the stored
/// factors (DiisErrorFactors) when the snapshot is taken: the serialized
/// form and the checkpoint schema are unchanged, and the expansion costs
/// O(n^2 n_occ) per entry once per write, not per iteration.
/// \ingroup qcx-scf
struct DiisState {
    std::vector<Eigen::MatrixXd> fockHistory; ///< Stored Fock matrices, oldest first.
    std::vector<Eigen::MatrixXd> errorHistory; ///< Matching error vectors.
};

/// The stable factored form of one CDIIS error vector.
///
/// The error vector is e = X^T (F D S - S D F) X with X = S^{-1/2}. With
/// C_s = sqrt(2) C_occ the density's own factor (BuildDensity forms
/// D = 2 C_occ C_occ^T, i.e. D = C_s C_s^T up to the product's rounding),
/// the commutator is an exact algebraic rewrite of
///
///     e = A B^T - B A^T,   A = X^T F C_s,   B = X^T S C_s
///
/// with no approximation: the identity is a regrouping of the same products,
/// and it holds on real iterates to 1-2e-15 relative (the rounding of those
/// products, the same order as the antisymmetry residual of the square it
/// replaces). One error vector therefore costs 2 n n_occ doubles instead of
/// n^2 - a 14.7x reduction at the flagship size (n = 4974, n_occ = 169:
/// 12.83 MiB against 188.76 MiB per entry).
///
/// The factors are STORED PROJECTED, and the projection is what makes the
/// inner products of two history entries stable. Splitting A at the part that
/// commutes with B,
///
///     M = (1/2) B^T A,   R = A - B M,   s = M - M^T,
///
/// is exact (B^T B = 2 I because C_occ is S-orthonormal, so A = B M + R by
/// construction), and gives the equally exact identity
///
///     e = B s B^T + R B^T - B R^T.
///
/// Both small pieces are genuinely small where the error is: R carries the
/// part of A outside range(B) and vanishes as the density converges, s is the
/// antisymmetric part of the occupied-basis Fock C_occ^T F C_occ and vanishes
/// for any symmetric F, while the one part of A that does NOT vanish at
/// convergence is carried by B alone and stays out of the subtraction.
/// Forming <e_i, e_j> from A and B directly instead makes it a difference of
/// two O(1) trace contractions whose value is O(||e||^2): its relative error
/// grows without bound as the errors shrink (measured on the water/STO-3G
/// production-gate run: the self-inner-product read 7.7 relative away from the
/// value taken on the expanded square at the seventh iterate, and an
/// off-diagonal entry 5.3e4 at the ninth). An ill-conditioned least-squares
/// system does not absorb that - it amplifies it by its condition number,
/// which is why the B-system's operand has to be formed at the error's own
/// scale and not at the factors'.
///
/// The storage is 2 n n_occ doubles plus the n_occ x n_occ s block - the same
/// order as the bare (A, B) pair, and the memory win is the whole reason for
/// the form.
/// \ingroup qcx-scf
struct DiisErrorFactors {
    Eigen::MatrixXd b; ///< X^T S C_s (n x n_occ); B^T B = 2 I.
    Eigen::MatrixXd r; ///< A - B M (n x n_occ): the part of A outside range(B).
    Eigen::MatrixXd s; ///< M - M^T (n_occ x n_occ), M = (1/2) B^T A.

    /// The full n x n error the factors represent, e = B s B^T + R B^T -
    /// B R^T. Materializing it costs O(n^2 n_occ) - the norm and the reported
    /// diiserr= are taken on this, never on a trace formula over the factors,
    /// so both gates keep the operand they have always had.
    /// \returns The n x n square.
    Eigen::MatrixXd Expand() const;
};

/// Pulay DIIS extrapolation of Fock matrices.
///
/// Callers append (fock, error) pairs - the error vector is the
/// orthogonal-basis commutator X^T (F D S - S D F) X, so vectors from
/// different iterations live in one comparable basis - and extrapolate once
/// the history holds at least two pairs. The B-system B_ij = <e_i, e_j>
/// with the unit-sum constraint is solved as a minimum-norm least-squares
/// problem (complete orthogonal decomposition). The bordered system
/// [[B, -1], [-1^T, 0]] is well posed even where the Gram block is singular
/// (the constraint removes the null direction of the coefficient vector), but
/// it is ILL-CONDITIONED as an SCF converges: the error vectors become nearly
/// linearly dependent, so minimizing a nearly zero residual is nearly flat
/// and the coefficients lose accuracy with it. That loss is inherent to DIIS
/// in floating point, not an artifact of how the system is assembled. The
/// extrapolated matrix is the same linear combination of the stored Fock
/// matrices.
///
/// Two append forms, one history: the full-square overload stores the error
/// vector as given (k n^2 doubles), the DiisErrorFactors overload stores the
/// projected factors (2 k n n_occ doubles plus k n_occ^2) and forms the
/// B-matrix entries from them. A factored history's B entries are NOT bitwise
/// the ones a full-square history produces - the two forms factor the same
/// error vector differently and the extrapolation is knife-edge near
/// convergence - so the form is a storage and arithmetic decision, and the
/// factored form is the stable one (DiisErrorFactors' note). Entries of both
/// forms may coexist in one history (a checkpoint restore materializes full
/// squares; the loop then appends factors), and the inner products of a mixed
/// pair are computed by expanding the factored side and taking the elementwise
/// product on the two squares.
///
/// Diagnostics: with the environment variable QCX_DIIS_DIAGNOSTICS set, every
/// B-matrix entry is printed to stderr against the value taken on the expanded
/// squares (the check that catches a form losing accuracy as the error
/// shrinks - it is invisible in the extrapolated Fock and in the converged
/// energy, and shows up only as a different convergence path), and every
/// extrapolation reports the bordered system's spectrum. Nothing is printed
/// and nothing extra is computed when the variable is unset.
/// \ingroup qcx-scf
class DiisExtrapolator {
public:
    /// Creates an extrapolator keeping the most recent \p historyLimit pairs.
    /// \param historyLimit Pairs kept; must be at least 2.
    explicit DiisExtrapolator(std::size_t historyLimit = 8);

    /// Appends one (fock, error) pair, evicting the oldest beyond the limit.
    /// The full-square form: the error is stored as given.
    /// \param fock Fock matrix of the current iteration.
    /// \param error Error vector of the current iteration (same shape as fock).
    void Append(const Eigen::MatrixXd& fock, const Eigen::MatrixXd& error);

    /// Appends one (fock, factors) pair, evicting the oldest beyond the limit.
    /// The factored form: the same error vector at 2 n n_occ doubles plus the
    /// n_occ x n_occ s block.
    /// \param fock Fock matrix of the current iteration.
    /// \param factors The exact factorization of the current iteration's error.
    void Append(const Eigen::MatrixXd& fock, const DiisErrorFactors& factors);

    /// True once the history holds at least two pairs.
    /// \returns The readiness flag.
    bool Ready() const noexcept;

    /// Extrapolates the Fock matrix from the stored history.
    /// \returns The extrapolated matrix, or an Error (kInvalidArgument with
    /// fewer than two stored pairs).
    qcx::Result<Eigen::MatrixXd> Extrapolate() const;

    /// Extrapolates like Extrapolate(), also returning the combination
    /// weights (the per-iteration DIIS trace status: the SCF trace
    /// records which history weights an extrapolation used, so a run's DIIS
    /// activity is observable per iteration instead of silently inert).
    /// \returns The extrapolated matrix and the weights as a pair - the
    /// weights are the COD solution's first history entries (the Fock
    /// combination weights, which sum to ~1 under the unit-sum constraint;
    /// the trailing Lagrange-multiplier slot is not included), or an Error
    /// (kInvalidArgument with fewer than two stored pairs).
    qcx::Result<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> ExtrapolateWithCoefficients() const;

    /// Drops the whole history.
    void Reset();

    /// The history as a value (checkpointing): the error entries are
    /// the full squares the factors represent (DiisState's note).
    /// \returns The (fock, error) pairs, oldest first.
    DiisState Snapshot() const;

    /// Replaces the history. An empty state is a reset (no-op), so
    /// default-constructed options behave exactly as before. The restored
    /// errors are full squares - the factors are not recoverable from a
    /// square without a rank-revealing tolerance this type does not take -
    /// so they enter the history in the full form and age out of the window
    /// as factored pairs are appended behind them.
    /// \param state The pairs to restore; the history must not exceed the
    /// extrapolator's limit and every pair must be square and
    /// shape-consistent (kInvalidArgument otherwise).
    /// \returns An Error (kInvalidArgument on a too-large or shape-mismatched
    /// state).
    qcx::Result<void> Restore(const DiisState& state);

private:
    /// One stored error vector: the projected factors, or the full square the
    /// restore path and the full-square Append overload hand over.
    struct ErrorEntry {
        bool factored = false; ///< Whether the projected fields or `full` carry the entry.
        Eigen::MatrixXd b; ///< X^T S C_s (n x n_occ), factored entries only.
        Eigen::MatrixXd r; ///< A - B M (n x n_occ), factored entries only.
        Eigen::MatrixXd s; ///< M - M^T (n_occ x n_occ), factored entries only.
        Eigen::MatrixXd full; ///< The n x n square, full-form entries only.
    };

    /// The Frobenius inner product <lhs, rhs> of two stored errors, from
    /// whichever form each carries.
    static double InnerProduct(const ErrorEntry& lhs, const ErrorEntry& rhs);

    /// <lhs, rhs> of two FACTORED errors from the projected triples
    /// (DiisErrorFactors' note): e = P W P^T with P = [B R] and
    /// W = [[s, -I], [I, 0]], so the value is tr(W_i G W_j^T G^T) with
    /// G = P_i^T P_j. Four n_occ x n_occ contractions of the n x n_occ
    /// factors - the O(n n_occ^2) cost class - and no n x n matrix is formed.
    static double ProjectedInnerProduct(const ErrorEntry& lhs, const ErrorEntry& rhs);

    /// <lhs, rhs> taken on the two n x n squares themselves: the elementwise
    /// product sum at the error's own scale. O(n^2) on the expanded squares.
    static double ExpandedInnerProduct(const ErrorEntry& lhs, const ErrorEntry& rhs);

    /// Prints one B-matrix entry against its expanded-square value
    /// (QCX_DIIS_DIAGNOSTICS). \p lhs and \p rhs are the same object for the
    /// diagonal entry.
    static void ReportPair(std::size_t position, const ErrorEntry& lhs, const ErrorEntry& rhs);

    /// The full n x n square one stored error represents: the stored matrix,
    /// or the projected triple expanded as B s B^T + R B^T - B R^T.
    static Eigen::MatrixXd ExpandError(const ErrorEntry& entry);

    /// Appends an entry and its inner products against the stored ones,
    /// evicting the oldest beyond the limit. The cross-Gram values are kept
    /// rather than recomputed: an extrapolation reads all m^2 of them, and
    /// only the m new ones are computable incrementally.
    void AppendEntry(const Eigen::MatrixXd& fock, ErrorEntry entry);

    std::size_t _historyLimit; ///< Pairs kept.
    std::vector<Eigen::MatrixXd> _fockHistory; ///< Most recent first-evicted order.
    std::vector<ErrorEntry> _errorHistory; ///< Error vectors, matching _fockHistory.
    std::vector<std::vector<double>> _crossGram; ///< _crossGram[i][j] = <e_i, e_j>, j <= i.
};

} // namespace qcx::scf
