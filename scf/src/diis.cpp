// CDIIS: the Pulay extrapolation of Fock matrices from a short history of
// (fock, error) pairs [Pulay1980].
#include "qcx/scf/diis.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

namespace qcx::scf {

namespace {

// The full n x n square a factored entry represents (DiisErrorFactors'
// note): e = B s B^T + R B^T - B R^T. The three products are formed and added
// at the element level of n x n matrices - the scale at which
// X^T (F D S - S D F) X performs its own subtraction - so a norm taken on
// this square carries the square's accuracy rather than a difference of the
// factors' O(1) contractions.
Eigen::MatrixXd ExpandProjected(const Eigen::MatrixXd& b,
                                const Eigen::MatrixXd& r,
                                const Eigen::MatrixXd& s) {
    return b * s * b.transpose() + r * b.transpose() - b * r.transpose();
}

// The B-system diagnostics, armed by QCX_DIIS_DIAGNOSTICS (any value):
// DiisExtrapolator::ReportPair prints every B-matrix entry against the value
// taken on the expanded squares, and every extrapolation reports the bordered
// system's spectrum. A form that loses accuracy as the error shrinks is
// invisible in the extrapolated Fock and in the converged energy - it shows up
// only as a different convergence path - so the check is what catches it. The
// cost of arming it is one expansion per pair (the DIIS step then carries the
// full-square arithmetic it is being checked against); nothing is computed
// when the variable is unset.
bool DiagnosticsEnabled() {
    static const bool enabled = std::getenv("QCX_DIIS_DIAGNOSTICS") != nullptr;

    return enabled;
}

} // namespace

Eigen::MatrixXd DiisErrorFactors::Expand() const {
    return ExpandProjected(b, r, s);
}

DiisExtrapolator::DiisExtrapolator(std::size_t historyLimit) : _historyLimit(historyLimit) {
    _fockHistory.reserve(historyLimit);
    _errorHistory.reserve(historyLimit);
    _crossGram.reserve(historyLimit);
}

Eigen::MatrixXd DiisExtrapolator::ExpandError(const ErrorEntry& entry) {
    if (!entry.factored)
    {
        return entry.full;
    }

    return ExpandProjected(entry.b, entry.r, entry.s);
}

double DiisExtrapolator::ExpandedInnerProduct(const ErrorEntry& lhs, const ErrorEntry& rhs) {
    const Eigen::MatrixXd lhsSquare = ExpandError(lhs);
    const Eigen::MatrixXd rhsSquare = ExpandError(rhs);

    return (lhsSquare.array() * rhsSquare.array()).sum();
}

double DiisExtrapolator::ProjectedInnerProduct(const ErrorEntry& lhs, const ErrorEntry& rhs) {
    // e = P W P^T with P = [B R] (n x 2 n_occ) and W = [[s, -I], [I, 0]], so
    //
    //     <e_i, e_j> = tr(W_i G W_j^T G^T),   G = P_i^T P_j,
    //
    // and every block of that product is n_occ x n_occ: the four contractions
    // of the factors against each other, then small dense matrix products. No
    // n x n matrix is formed and no difference of O(1) quantities is taken
    // (DiisErrorFactors' note).
    const Eigen::Index m = lhs.b.cols();
    const Eigen::Index width = 2 * m;
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(m, m);

    Eigen::MatrixXd g(width, width);
    g.topLeftCorner(m, m) = lhs.b.transpose() * rhs.b;
    g.topRightCorner(m, m) = lhs.b.transpose() * rhs.r;
    g.bottomLeftCorner(m, m) = lhs.r.transpose() * rhs.b;
    g.bottomRightCorner(m, m) = lhs.r.transpose() * rhs.r;

    Eigen::MatrixXd lhsW = Eigen::MatrixXd::Zero(width, width);
    lhsW.topLeftCorner(m, m) = lhs.s;
    lhsW.topRightCorner(m, m) = -identity;
    lhsW.bottomLeftCorner(m, m) = identity;

    Eigen::MatrixXd rhsWT = Eigen::MatrixXd::Zero(width, width);
    rhsWT.topLeftCorner(m, m) = -rhs.s;
    rhsWT.topRightCorner(m, m) = identity;
    rhsWT.bottomLeftCorner(m, m) = -identity;

    return (lhsW * g * rhsWT * g.transpose()).trace();
}

double DiisExtrapolator::InnerProduct(const ErrorEntry& lhs, const ErrorEntry& rhs) {
    if (!lhs.factored && !rhs.factored)
    {
        // The full-square pair: the elementwise product the history has
        // always used, on the stored matrices themselves (a restored history
        // extrapolates bit-identically to the pre-factorization code).
        return (lhs.full.array() * rhs.full.array()).sum();
    }

    if (lhs.factored && rhs.factored)
    {
        return ProjectedInnerProduct(lhs, rhs);
    }

    // A mixed pair - a restored square against an appended factor - expands
    // the factored side and takes the elementwise product on the two squares,
    // which is the operand the full-square form would have handed over. Only a
    // checkpoint restart reaches this.
    return ExpandedInnerProduct(lhs, rhs);
}

void DiisExtrapolator::ReportPair(std::size_t position,
                                  const ErrorEntry& lhs,
                                  const ErrorEntry& rhs) {
    const double value = ProjectedInnerProduct(lhs, rhs);
    const double expanded = ExpandedInnerProduct(lhs, rhs);
    const double magnitude = std::fabs(expanded);

    std::fprintf(stderr,
                 "diis-diag: position=%zu diagonal=%d value=%.17g expanded=%.17g reldiff=%.3g\n",
                 position,
                 &lhs == &rhs ? 1 : 0,
                 value,
                 expanded,
                 magnitude > 0.0 ? std::fabs(value - expanded) / magnitude : -1.0);
}

void DiisExtrapolator::AppendEntry(const Eigen::MatrixXd& fock, ErrorEntry entry) {
    // The new entry's inner products against the stored ones, taken while
    // the indices still match (the eviction below shifts them). The cache is
    // a full symmetric square: one pair is computed once and written to both
    // of its slots, so the B-system sees exactly consistent values.
    const std::size_t stored = _errorHistory.size();
    const bool diagnostics = DiagnosticsEnabled();
    std::vector<double> row;
    row.reserve(stored + 1);

    for (std::size_t j = 0; j < stored; ++j)
    {
        const double value = InnerProduct(entry, _errorHistory[j]);
        row.push_back(value);
        _crossGram[j].push_back(value);

        if (diagnostics && entry.factored && _errorHistory[j].factored)
        {
            ReportPair(j, entry, _errorHistory[j]);
        }
    }

    row.push_back(InnerProduct(entry, entry));

    if (diagnostics && entry.factored)
    {
        ReportPair(stored, entry, entry);
    }

    _fockHistory.push_back(fock);
    _errorHistory.push_back(std::move(entry));
    _crossGram.push_back(std::move(row));

    if (_fockHistory.size() > _historyLimit)
    {
        _fockHistory.erase(_fockHistory.begin());
        _errorHistory.erase(_errorHistory.begin());
        _crossGram.erase(_crossGram.begin());

        for (std::vector<double>& cached : _crossGram)
        {
            cached.erase(cached.begin());
        }
    }
}

// (fock, error) is the DIIS vector order, fixed by the lone call site.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void DiisExtrapolator::Append(const Eigen::MatrixXd& fock, const Eigen::MatrixXd& error) {
    ErrorEntry entry;
    entry.full = error;
    AppendEntry(fock, std::move(entry));
}

void DiisExtrapolator::Append(const Eigen::MatrixXd& fock, const DiisErrorFactors& factors) {
    ErrorEntry entry;
    entry.factored = true;
    entry.b = factors.b;
    entry.r = factors.r;
    entry.s = factors.s;
    AppendEntry(fock, std::move(entry));
}

bool DiisExtrapolator::Ready() const noexcept {
    return _fockHistory.size() >= 2;
}

qcx::Result<Eigen::MatrixXd> DiisExtrapolator::Extrapolate() const {
    auto result = ExtrapolateWithCoefficients();

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    return std::move(result->first);
}

qcx::Result<std::pair<Eigen::MatrixXd, Eigen::VectorXd>>
DiisExtrapolator::ExtrapolateWithCoefficients() const {
    if (!Ready())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "DIIS extrapolation needs at least two stored pairs"});
    }

    const Eigen::Index m = static_cast<Eigen::Index>(_fockHistory.size());

    // B = [[<e_i, e_j>], -1; [-1^T, 0]], rhs = [0, ..., 0, -1]: the
    // Lagrange multiplier in the last slot enforces sum c_i = 1 while the
    // quadratic form minimizes the error-vector norm. The inner products are
    // the append path's own values (the cross-Gram cache), not a second
    // pass over the history.
    Eigen::MatrixXd b = Eigen::MatrixXd::Zero(m + 1, m + 1);

    for (Eigen::Index i = 0; i < m; ++i)
    {
        for (Eigen::Index j = 0; j < m; ++j)
        {
            b(i, j) = _crossGram[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        }

        b(i, m) = -1.0;
        b(m, i) = -1.0;
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(m + 1);
    rhs(m) = -1.0;

    // Complete orthogonal decomposition: a minimum-norm least-squares solve.
    // The bordered system is well posed even where the Gram block is singular
    // - the unit-sum constraint removes the coefficient vector's null
    // direction - but it is ILL-CONDITIONED as the SCF converges, because the
    // error vectors become nearly linearly dependent and minimizing a nearly
    // zero residual is nearly flat. The minimum-norm solution is the treatment
    // every production code effectively computes here; no regularization is
    // applied (a coefficient-accuracy loss in the last iterations is inherent
    // to DIIS in floating point, and QCX_DIIS_DIAGNOSTICS reports the
    // spectrum that measures it).
    const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> decomposition(b);
    const Eigen::VectorXd fullSolution = decomposition.solve(rhs);

    if (DiagnosticsEnabled())
    {
        // The bordered system is symmetric, so the ratio of its extreme
        // |eigenvalue| is its condition number in the 2-norm.
        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> spectrum(b);
        const Eigen::VectorXd& eigenvalues = spectrum.eigenvalues();
        const double largest = eigenvalues.cwiseAbs().maxCoeff();
        const double smallest = eigenvalues.cwiseAbs().minCoeff();

        std::fprintf(
            stderr,
            "diis-diag: bsystem m=%td condition=%.6g largest=%.6g smallest=%.6g rank=%td\n",
            m,
            smallest > 0.0 ? largest / smallest : std::numeric_limits<double>::infinity(),
            largest,
            smallest,
            decomposition.rank());
    }

    // The first m slots are the actual Fock combination weights (they sum to
    // ~1 under the unit-sum constraint); the trailing slot is the Lagrange
    // multiplier of the constraint and is not a Fock weight.
    Eigen::VectorXd coefficients = fullSolution.head(m);

    Eigen::MatrixXd extrapolated =
        Eigen::MatrixXd::Zero(_fockHistory.front().rows(), _fockHistory.front().cols());

    for (Eigen::Index i = 0; i < m; ++i)
    {
        extrapolated += coefficients(i) * _fockHistory[static_cast<std::size_t>(i)];
    }

    return std::make_pair(std::move(extrapolated), std::move(coefficients));
}

void DiisExtrapolator::Reset() {
    _fockHistory.clear();
    _errorHistory.clear();
    _crossGram.clear();
}

DiisState DiisExtrapolator::Snapshot() const {
    std::vector<Eigen::MatrixXd> errors;
    errors.reserve(_errorHistory.size());

    for (const ErrorEntry& entry : _errorHistory)
    {
        errors.push_back(ExpandError(entry));
    }

    return DiisState{_fockHistory, std::move(errors)};
}

qcx::Result<void> DiisExtrapolator::Restore(const DiisState& state) {
    if (state.fockHistory.size() != state.errorHistory.size())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the DIIS fock and error histories must match"});
    }

    if (state.fockHistory.size() > _historyLimit)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the DIIS history exceeds the extrapolator's limit"});
    }

    for (std::size_t i = 0; i < state.fockHistory.size(); ++i)
    {
        const Eigen::MatrixXd& fock = state.fockHistory[i];
        const Eigen::MatrixXd& error = state.errorHistory[i];

        if (fock.rows() != fock.cols() || fock.rows() != error.rows() ||
            fock.cols() != error.cols())
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the DIIS history matrices must be square and "
                                              "shape-consistent"});
        }
    }

    _fockHistory.clear();
    _errorHistory.clear();
    _crossGram.clear();

    for (std::size_t i = 0; i < state.fockHistory.size(); ++i)
    {
        ErrorEntry entry;
        entry.full = state.errorHistory[i];
        AppendEntry(state.fockHistory[i], std::move(entry));
    }

    return {};
}

} // namespace qcx::scf
