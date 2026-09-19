// UHF SCF loop: independent alpha/beta spin channels over the shared
// step-functions of internal/scf_common.hpp, with per-spin CDIIS (plus
// the opt-in two-phase variant), the two-way convergence gate (plus the
// opt-in three-way robust gate), the optional virtual-space level shift,
// the <S^2> diagnostic, and the Step-C initial guess tiers (GWH, SAD).
// The robustness heuristics land behind UhfOptions flags
// defaulting to the current behavior, so every default trajectory is
// bit-identical. The supermatrix Fock build contracts the REAL (unhalved)
// densities: F_sigma = H + J(D_alpha + D_beta) - K(D_sigma) - contrast
// rhf.cpp's half-density 2J - K/2 convention.
#include "qcx/scf/uhf.hpp"

#include "internal/scf_common.hpp"
#include "internal/symmetry_blocks.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/linalg/dense_ops.hpp"
#include "qcx/memory/first_touch.hpp"
#include "qcx/molecule/elements.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "qcx/scf/diis.hpp"
#include "qcx/symmetry/detection.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace qcx::scf {

namespace {

// Derives (nAlpha, nBeta) from the molecule's electron count and spin
// multiplicity. nAlpha - nBeta = multiplicity - 1 (2S = multiplicity - 1);
// nAlpha + nBeta = electronCount. Rejects an inconsistent pair
// (Molecule::Create does not check this today - only multiplicity >= 1).
qcx::Result<std::pair<int, int>> DeriveSpinOccupations(const qcx::molecule::Molecule& molecule) {
    const int electronCount = molecule.ElectronCount();
    const int multiplicity = molecule.Multiplicity();
    const int twoSPlusOneMinusOne = multiplicity - 1; // nAlpha - nBeta

    if ((electronCount - twoSPlusOneMinusOne) % 2 != 0 || electronCount < twoSPlusOneMinusOne)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "electron count and multiplicity are inconsistent"});
    }

    const int nBeta = (electronCount - twoSPlusOneMinusOne) / 2;
    const int nAlpha = electronCount - nBeta;
    return std::make_pair(nAlpha, nBeta);
}

// The integer occupation vector: 1.0 on the numOccupied lowest orbitals,
// 0.0 elsewhere. n then numOccupied: the vector size precedes the occupancy
// count (head(numOccupied) consumes them in this order).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::VectorXd IntegerOccupations(Eigen::Index n, int numOccupied) {
    Eigen::VectorXd occupations = Eigen::VectorXd::Zero(n);
    occupations.head(numOccupied).setOnes();
    return occupations;
}

// UHF per-spin density: C diag(occupations) C^T - NO factor of 2 (contrast
// rhf.cpp's BuildDensity, which doubles for the spin-summed closed-shell
// convention). Copying that factor in here is the single most likely
// transcription error in this file - it would silently double every energy
// contribution derived from this density. Fractional occupations (the SAD
// atomic fragments' spherical averaging) flow through the same path.
// The product rides the linalg seam (Eigen below the threshold, vendor
// BLAS above): the density build is BLAS-shaped (the
// recorded follow-up of rhf.cpp's routing). Below the threshold the
// seam path is bit-identical to the direct product (the per-column
// occupation scale and the materialized transpose are exact data movement
// - 0/1 scales, one rounding per element for the fractional SAD
// occupations, and the transpose is a permutation).
qcx::Result<Eigen::MatrixXd> BuildSpinDensity(const Eigen::MatrixXd& coefficients,
                                              const Eigen::VectorXd& occupations) {
    // Allocate uninitialized and touch before the copy fills the block: an
    // anonymous page is placed on the NUMA node of the thread that first
    // accesses it, so the pre-fill pass across the OpenMP team keeps the
    // Fock-build team's density reads local on multi-socket machines. The
    // touch is a non-destructive write-back and the copy is unchanged, so
    // the values are exactly the pre-pass values.
    Eigen::MatrixXd density(coefficients.rows(), coefficients.rows());
    qcx::memory::TouchPagesAcrossTeam(density.data(),
                                      static_cast<std::size_t>(density.size()) * sizeof(double));
    const Eigen::MatrixXd scaled = coefficients * occupations.asDiagonal();
    const Eigen::MatrixXd coefficientsTranspose = coefficients.transpose();
    auto product = qcx::linalg::DenseMultiply(scaled, coefficientsTranspose);

    if (!product.has_value())
    {
        return std::unexpected(product.error());
    }

    density = *product;
    return density;
}

// The occupied orbital indices of one spin channel: occupations above 1/2.
std::vector<int> OccupiedIndices(const Eigen::VectorXd& occupations) {
    std::vector<int> indices;

    for (Eigen::Index i = 0; i < occupations.size(); ++i)
    {
        if (occupations(i) > 0.5)
        {
            indices.push_back(static_cast<int>(i));
        }
    }

    return indices;
}

// One spin channel's PROJECTOR residual (the converged-state contract): the Frobenius norm
// ||D_sigma - C_sigma,occ C_sigma,occ^T|| between the returned density and the
// density the returned occupied MOs define. The occupied set is the run's own
// occupation rule's support (aufbau, or the MOM mask - both are 0/1), so the
// identity holds exactly on any integer-occupation run and a fractional one
// (the SAD atomic fragments' averaged shells) is outside it by construction -
// its density is a weighted sum of orbitals, not a projector.
// UhfResult::stateConsistencyResidualAlpha/Beta carry what the number means.
// (density, coefficients) is the density-then-orbitals order of the residual.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ReturnedStateResidual(const Eigen::MatrixXd& density,
                             const Eigen::MatrixXd& coefficients,
                             const Eigen::VectorXd& occupations) {
    const std::vector<int> occupied = OccupiedIndices(occupations);
    Eigen::MatrixXd occupiedColumns(coefficients.rows(),
                                    static_cast<Eigen::Index>(occupied.size()));

    for (std::size_t i = 0; i < occupied.size(); ++i)
    {
        occupiedColumns.col(static_cast<Eigen::Index>(i)) = coefficients.col(occupied[i]);
    }

    return (density - occupiedColumns * occupiedColumns.transpose()).norm();
}

// UHF electronic energy: 1/2 sum_sigma Tr[D_sigma (H + F_sigma)].
// Contrast RHF's 1/2 Tr[D (H+F)] with the spin-summed D - that formula is
// NOT equivalent to summing this UHF form with D_total in place of
// D_sigma, because F_alpha != F_beta in general (different K terms).
double ComputeUhfElectronicEnergy(const Eigen::MatrixXd& coreHamiltonian,
                                  const Eigen::MatrixXd& fockAlpha,
                                  const Eigen::MatrixXd& fockBeta,
                                  const Eigen::MatrixXd& dAlpha,
                                  const Eigen::MatrixXd& dBeta) {
    return 0.5 * ((dAlpha.cwiseProduct(coreHamiltonian + fockAlpha)).sum() +
                  (dBeta.cwiseProduct(coreHamiltonian + fockBeta)).sum());
}

// The trace half of the two-callback UHF path: Tr[D_total H] + 1/2 Tr[D_total J].
//
// Derived, not copied from rhf.cpp's ComputeCoulombTraceEnergy - the UHF trace
// identity carries its own overall 1/2, and the two only LOOK like the same
// expression once that factor is distributed:
//   1/2 sum_sigma Tr[D_sigma (H + F_sigma)]
//     = 1/2 sum_sigma Tr[D_sigma (2H + J(D_total) - K(D_sigma))]   [F_sigma = H + J - K_sigma]
//     = Tr[D_total H] + 1/2 Tr[D_total J] - 1/2 sum_sigma Tr[D_sigma K(D_sigma)].
// The first two terms are the Coulomb trace; the third is the caller's
// contribution, because K is not recoverable from D and an opaque Fock builder.
// Halving the H term as well (a plausible transcription slip, since the UHF
// identity is written with a leading 1/2) drops half of the one-electron energy:
// the HF-shaped regression test pins the total against the existing overload, and
// in the closed-shell limit dAlpha = dBeta = D/2 this reduces exactly to rhf.cpp's
// Tr[D H] + 1/2 Tr[D J] with D = D_total.
double ComputeUhfCoulombTraceEnergy(const Eigen::MatrixXd& coreHamiltonian,
                                    const Eigen::MatrixXd& coulomb,
                                    const Eigen::MatrixXd& totalDensity) {
    return (totalDensity.cwiseProduct(coreHamiltonian)).sum() +
           0.5 * (totalDensity.cwiseProduct(coulomb)).sum();
}

// The UHF dense Fock build: F_sigma = H + J(D_total) - K(D_sigma).
// Contrast rhf.cpp's BuildFock: that one scales as 2*J(rho) - 0.5*K(rho)
// because rho is the RHF HALF density. Here dAlpha/dBeta are REAL
// (unhalved) densities, so J and K both apply with NO scaling prefactor
// beyond what the supermatrices already encode.
qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> BuildUhfFock(
    const internal::JkSupermatrices& super,
    const Eigen::MatrixXd& dAlpha,
    // dAlpha/dBeta are the per-spin densities (J and K consume them in
    // order), coreHamiltonian last - the standard Fock-build argument order.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& dBeta,
    const Eigen::MatrixXd& coreHamiltonian) {
    const Eigen::Index n = dAlpha.rows();
    const Eigen::MatrixXd dTotal = dAlpha + dBeta;

    // Flatten the three densities in the tensor's row-major order so their
    // entries line up with the supermatrix columns (rhf.cpp's dVec).
    Eigen::MatrixXd dTotalVec(n * n, 1);
    Eigen::MatrixXd dAlphaVec(n * n, 1);
    Eigen::MatrixXd dBetaVec(n * n, 1);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            const Eigen::Index flat = mu * n + nu;
            dTotalVec(flat, 0) = dTotal(mu, nu);
            dAlphaVec(flat, 0) = dAlpha(mu, nu);
            dBetaVec(flat, 0) = dBeta(mu, nu);
        }
    }

    const auto jTotal = qcx::linalg::DenseMultiply(super.coulomb, dTotalVec);

    if (!jTotal.has_value())
    {
        return std::unexpected(jTotal.error());
    }

    const auto kAlpha = qcx::linalg::DenseMultiply(super.exchange, dAlphaVec);

    if (!kAlpha.has_value())
    {
        return std::unexpected(kAlpha.error());
    }

    const auto kBeta = qcx::linalg::DenseMultiply(super.exchange, dBetaVec);

    if (!kBeta.has_value())
    {
        return std::unexpected(kBeta.error());
    }

    Eigen::MatrixXd fockAlpha = coreHamiltonian;
    Eigen::MatrixXd fockBeta = coreHamiltonian;

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            const Eigen::Index flat = mu * n + nu;
            fockAlpha(mu, nu) += (*jTotal)(flat, 0) - (*kAlpha)(flat, 0);
            fockBeta(mu, nu) += (*jTotal)(flat, 0) - (*kBeta)(flat, 0);
        }
    }

    return std::make_pair(std::move(fockAlpha), std::move(fockBeta));
}

// The CDIIS error vector [Pulay1980]: X^T (F D S - S D F) X in the
// orthogonal basis, so vectors from different iterations are comparable.
// Per-spin: called once per spin channel with the sigma-specific Fock and
// density (two independent DiisExtrapolator instances). The F D S
// contraction rides the linalg seam (BLAS-shaped, the recorded
// follow-up of rhf.cpp's routing): Eigen below the threshold - bit-
// identical to the direct product - vendor BLAS above. The X^T
// (...) X pair stays Eigen: it runs in the serial section and is the
// blocked diagonalization's concern.
qcx::Result<Eigen::MatrixXd> DiisError(const Eigen::MatrixXd& fock,
                                       const Eigen::MatrixXd& density,
                                       // fock/density/overlap/x: the CDIIS contraction
                                       // order, FDS - SDF then X^T (...) X.
                                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                       const Eigen::MatrixXd& overlap,
                                       const Eigen::MatrixXd& x) {
    auto fd = qcx::linalg::DenseMultiply(fock, density);

    if (!fd.has_value())
    {
        return std::unexpected(fd.error());
    }

    auto fds = qcx::linalg::DenseMultiply(*fd, overlap);

    if (!fds.has_value())
    {
        return std::unexpected(fds.error());
    }

    return x.transpose() * ((*fds) - fds->transpose()) * x;
}

// CDIIS history length: the standard 8-vector window.
constexpr std::size_t kDiisHistoryLimit = 8;

// Two-phase DIIS phase-1 history length [niedoida diis.cpp]: a short
// 2-pair window until the per-spin error norm drops below the phase
// threshold, then the full kDiisHistoryLimit window. This threshold is
// distinct from RHF's kDiisStartThreshold ENGAGEMENT threshold (kept only
// at the rhf.cpp call site; the recorded re-expression) - UHF
// deliberately runs no engagement threshold.
constexpr std::size_t kDiisPhaseHistoryLimit = 2;

// The two-phase handoff threshold [niedoida diis.cpp]: the per-spin CDIIS
// error norm in the orthogonal basis - the same quantity the floor guard
// measures - must drop below 0.5 for the phase-1 window to hand off. The
// flip is one-way, no hysteresis (niedoida's real_diis_enabled_ is one-way
// too).
constexpr double kDiisPhaseThreshold = 0.5;

// The two-phase DIIS wrapper of one spin channel [niedoida diis.cpp]: a
// phase-1 extrapolator with a short history (kDiisPhaseHistoryLimit) until
// the spin's error norm drops below kDiisPhaseThreshold, then a one-way
// handoff to the full kDiisHistoryLimit window. The handoff snapshots the
// phase-1 extrapolator and restores the state into a full-window one (the
// Snapshot/Restore round-trip is bit-identical,
// DiisTest.SnapshotRestoreRoundTrips), so no history entry is lost or
// duplicated. No public DiisExtrapolator surface is added: the
// phase logic lives entirely here.
struct TwoPhaseDiis {
    // historyLimit: kDiisPhaseHistoryLimit on fresh two-phase runs, the
    // full kDiisHistoryLimit otherwise; twoPhase: whether the phase
    // handoff is engaged (options.useTwoPhaseDiis).
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    TwoPhaseDiis(std::size_t historyLimit, bool twoPhase) :
        _extrapolator(historyLimit), _twoPhase(twoPhase), _useFullWindow(!twoPhase) {}

    qcx::Result<void> Append(const Eigen::MatrixXd& fock, const Eigen::MatrixXd& error) {
        if (_twoPhase && !_useFullWindow && error.norm() < kDiisPhaseThreshold)
        {
            DiisExtrapolator fullWindow(kDiisHistoryLimit);
            const auto restore = fullWindow.Restore(_extrapolator.Snapshot());

            if (!restore.has_value())
            {
                return std::unexpected(restore.error());
            }

            _extrapolator = std::move(fullWindow);
            _useFullWindow = true;
        }

        _extrapolator.Append(fock, error);
        return {};
    }

    // True once the history holds at least two pairs.
    bool Ready() const noexcept {
        return _extrapolator.Ready();
    }

    // Extrapolates the Fock matrix from the stored history.
    qcx::Result<Eigen::MatrixXd> Extrapolate() const {
        return _extrapolator.Extrapolate();
    }

    // The history as a value (checkpointing).
    DiisState Snapshot() const {
        return _extrapolator.Snapshot();
    }

    // Replaces the history; an empty state is a reset (no-op). A restored
    // NON-empty history implies a run mid-acceleration: the full window
    // applies from here (a restored 1-2-vector history extrapolates
    // identically in either window, and future eviction follows the
    // phase-2 discipline). An empty restore leaves a fresh run fresh:
    // phase 1 starts from scratch.
    qcx::Result<void> Restore(const DiisState& state) {
        const auto result = _extrapolator.Restore(state);

        if (!result.has_value())
        {
            return std::unexpected(result.error());
        }

        if (!state.fockHistory.empty())
        {
            _useFullWindow = true;
        }

        return {};
    }

private:
    DiisExtrapolator _extrapolator;
    bool _twoPhase;
    bool _useFullWindow;
};

// The Wolfsberg-Helmholtz coupling constant of the GWH initial guess
// [Wolfsberg1952]: the standard 1.75.
constexpr double kGwhCoefficient = 1.75;

// Runs the CDIIS extrapolation when engaged (rhf.cpp's MaybeExtrapolateFock
// per-spin copy): the (fock, error) pair is appended to the spin's own
// two-phase wrapper and fock is replaced with the extrapolated matrix once
// the history holds at least two pairs. Returns fock unchanged when DIIS is
// off or the error is at the floor.
//
// A floor-degenerate error vector (e.g. the P=0 start's exactly-zero
// commutator at iteration 0) is never appended: its zero row/column in the
// DIIS B-matrix makes the constraint system degenerate and the minimum-norm
// solve can then pick the wild first Fock (niedoida's diis.cpp removes such
// vectors at its singular solve; dropping them here is the equivalent). The
// error norm is hoisted to the call site (computed once per spin per
// iteration): the robust gate's DIIS-error leg consumes exactly the
// quantity the floor guard and the extrapolation used. The two-phase path
// re-measures the norm inside Append for its phase handoff only.
qcx::Result<Eigen::MatrixXd> MaybeExtrapolateFock(const UhfOptions& options,
                                                  TwoPhaseDiis& diis,
                                                  Eigen::MatrixXd fock,
                                                  const Eigen::MatrixXd& error,
                                                  double diisErrorNorm) {
    if (!options.useDiis)
    {
        return fock;
    }

    if (diisErrorNorm <= internal::kDiisFloorThreshold)
    {
        return fock;
    }

    const auto appended = diis.Append(fock, error);

    if (!appended.has_value())
    {
        return std::unexpected(appended.error());
    }

    if (!diis.Ready())
    {
        return fock;
    }

    const auto extrapolated = diis.Extrapolate();

    if (!extrapolated.has_value())
    {
        return std::unexpected(extrapolated.error());
    }

    return *extrapolated;
}

// Runs the engaged extrapolation path for one iteration (Step 5): the
// joint system extrapolates BOTH spins from one coefficient vector over
// the combined error subspace (niedoida's joint form), where the per-spin
// path lets each channel's extrapolator minimize its own error blind to
// the other - the P=0 locking is a cross-spin effect, and the coupling is
// the Step-5 hypothesis. The two-phase handoff is not ported (the
// combination is rejected up front). Returns the (possibly extrapolated)
// fock pair.
qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> MaybeExtrapolateFockPair(
    const UhfOptions& options,
    std::optional<internal::JointSystemDiis>& jointDiis,
    TwoPhaseDiis& diisAlpha,
    TwoPhaseDiis& diisBeta,
    Eigen::MatrixXd fockAlpha,
    const Eigen::MatrixXd& errorAlpha,
    double diisErrorNormAlpha,
    // fockBeta/errorBeta/diisErrorNormBeta: the beta half of the pair,
    // consumed after the alpha half.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    Eigen::MatrixXd fockBeta,
    const Eigen::MatrixXd& errorBeta,
    double diisErrorNormBeta) {
    if (jointDiis.has_value())
    {
        return internal::MaybeExtrapolateJointFock(options,
                                                   *jointDiis,
                                                   std::move(fockAlpha),
                                                   errorAlpha,
                                                   diisErrorNormAlpha,
                                                   std::move(fockBeta),
                                                   errorBeta,
                                                   diisErrorNormBeta);
    }

    auto extrapolatedAlpha = MaybeExtrapolateFock(
        options, diisAlpha, std::move(fockAlpha), errorAlpha, diisErrorNormAlpha);

    if (!extrapolatedAlpha.has_value())
    {
        return std::unexpected(extrapolatedAlpha.error());
    }

    auto extrapolatedBeta =
        MaybeExtrapolateFock(options, diisBeta, std::move(fockBeta), errorBeta, diisErrorNormBeta);

    if (!extrapolatedBeta.has_value())
    {
        return std::unexpected(extrapolatedBeta.error());
    }

    return std::make_pair(std::move(*extrapolatedAlpha), std::move(*extrapolatedBeta));
}

// The maximum-overlap occupation mask [Besley2009]: O = C0_occ^T S C (the
// reference occupied orbitals against the current MOs), each current MO's
// projection the squared column norm of O, and the numOccupied orbitals
// with the largest projections occupied (niedoida's
// MaximumOverlapOccupations selection rule, core_kit/src/occupations.cpp).
// coefficients/reference/overlap then numOccupied: the projection
// O = C0_occ^T S C consumes them in this order.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::VectorXd MomOccupationMask(const Eigen::MatrixXd& coefficients,
                                  const Eigen::MatrixXd& reference,
                                  const Eigen::MatrixXd& overlap,
                                  int numOccupied) {
    const Eigen::MatrixXd referenceOccupied = reference.leftCols(numOccupied);
    const Eigen::MatrixXd o = referenceOccupied.transpose() * overlap * coefficients;

    std::vector<std::pair<double, Eigen::Index>> projections;
    projections.reserve(static_cast<std::size_t>(coefficients.cols()));

    for (Eigen::Index j = 0; j < coefficients.cols(); ++j)
    {
        projections.emplace_back(o.col(j).squaredNorm(), j);
    }

    std::sort(projections.begin(), projections.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });

    Eigen::VectorXd mask = Eigen::VectorXd::Zero(coefficients.cols());

    for (int i = 0; i < numOccupied; ++i)
    {
        mask(projections[static_cast<std::size_t>(i)].second) = 1.0;
    }

    return mask;
}

// The <S^2> spin-contamination diagnostic of the final occupied orbitals:
// S^2 = S(S+1) + nBeta - sum_ij |<psi_alpha,i|psi_beta,j>|^2 with the
// overlap sum over the occupied alpha/beta MO pairs (S = (nAlpha-nBeta)/2).
// A sign or transpose error in the overlap contraction usually produces a
// wildly wrong <S^2> even when the energy still looks close. overlap,
// coefficientsAlpha, coefficientsBeta then the occupation lists: the <S^2>
// formula reads them in this order.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ComputeSpinSquared(const Eigen::MatrixXd& overlap,
                          const Eigen::MatrixXd& coefficientsAlpha,
                          const Eigen::MatrixXd& coefficientsBeta,
                          const std::vector<int>& occupiedAlpha,
                          const std::vector<int>& occupiedBeta) {
    const double s = 0.5 * static_cast<double>(static_cast<int>(occupiedAlpha.size()) -
                                               static_cast<int>(occupiedBeta.size()));

    Eigen::MatrixXd occAlpha(coefficientsAlpha.rows(), occupiedAlpha.size());
    Eigen::MatrixXd occBeta(coefficientsBeta.rows(), occupiedBeta.size());

    for (std::size_t i = 0; i < occupiedAlpha.size(); ++i)
    {
        occAlpha.col(static_cast<Eigen::Index>(i)) = coefficientsAlpha.col(occupiedAlpha[i]);
    }

    for (std::size_t i = 0; i < occupiedBeta.size(); ++i)
    {
        occBeta.col(static_cast<Eigen::Index>(i)) = coefficientsBeta.col(occupiedBeta[i]);
    }

    const Eigen::MatrixXd overlapOcc = occAlpha.transpose() * overlap * occBeta;
    return s * (s + 1.0) + static_cast<double>(occupiedBeta.size()) - overlapOcc.squaredNorm();
}

// The internal loop result. UhfResult itself now carries the full converged
// state (densities, coefficients, orbital energies), so the loop
// result is just the summary; the SAD fragments read the coefficients from
// UhfResult::coefficientsAlpha/Beta (their MOM-reference handoff).
struct UhfLoopResult {
    UhfResult summary;
};

// Attaches the final SCF state to the summary: the per-spin
// densities, MO coefficients, and the orbital energies - the diagonal of the
// final MO-basis Fock C_sigma^T F_sigma C_sigma, computed from the inputs
// before the moves consume them.
UhfResult FinishSummary(UhfResult summary,
                        Eigen::MatrixXd dAlpha,
                        Eigen::MatrixXd dBeta,
                        Eigen::MatrixXd coefficientsAlpha,
                        // coefficientsAlpha/Beta mirror the result member order.
                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                        Eigen::MatrixXd coefficientsBeta,
                        const Eigen::MatrixXd& finalFockAlpha,
                        const Eigen::MatrixXd& finalFockBeta) {
    summary.orbitalEnergiesAlpha =
        (coefficientsAlpha.transpose() * finalFockAlpha * coefficientsAlpha).diagonal();
    summary.orbitalEnergiesBeta =
        (coefficientsBeta.transpose() * finalFockBeta * coefficientsBeta).diagonal();
    summary.densityAlpha = std::move(dAlpha);
    summary.densityBeta = std::move(dBeta);
    summary.coefficientsAlpha = std::move(coefficientsAlpha);
    summary.coefficientsBeta = std::move(coefficientsBeta);
    return summary;
}

// Runs the UHF Roothaan-Hall iteration with a caller-provided Fock-builder
// pair. The occupation vectors fix the per-spin density build each
// iteration (integer for the public runs, fractional for the SAD atomic
// fragments, MOM-selected when options.useMom).
//
// coulomb/energyContribution are the energy seam (the two-spin analog of
// rhf.cpp's): null selects the Hartree-Fock trace identity every existing
// caller uses, non-null selects Tr[D_total H] + 1/2 Tr[D_total J] plus the
// caller's contribution. Both are pointers rather than references so the
// null state is expressible; the public overload rejects one without the other.
qcx::Result<UhfLoopResult> RunUhfLoop(const qcx::molecule::Molecule& molecule,
                                      const Eigen::MatrixXd& overlap,
                                      const Eigen::MatrixXd& coreHamiltonian,
                                      const UhfOptions& options,
                                      const UhfFockBuilderFn& buildFock,
                                      const CoulombFn* coulomb,
                                      const UhfEnergyContributionFn* energyContribution,
                                      const Eigen::VectorXd& fullOccupationsAlpha,
                                      const Eigen::VectorXd& fullOccupationsBeta,
                                      const qcx::basisset::BasisSet* basisSet) {
    if (options.maxIterations <= 0 || options.energyTolerance <= 0.0 ||
        options.densityTolerance <= 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "maxIterations and the tolerances must be positive"});
    }

    if (options.levelShiftAlpha < 0.0 || options.levelShiftBeta < 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the level-shift values must be non-negative"});
    }

    if (options.useLevelShift && options.useMom)
    {
        // The shift's zero block is defined against the aufbau occupied
        // set; MOM may occupy a different set - the interaction is out of
        // scope (recorded), so the combination is refused.
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the level shift and MOM cannot be combined"});
    }

    if (options.useJointDiis && options.useTwoPhaseDiis)
    {
        // The two-phase cap is per-spin (each channel's own phase handoff
        // window); the joint form couples the channels into one subspace
        // with one window (the two-phase cap is not
        // ported to the joint form), so the combination is refused.
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the joint-system and two-phase DIIS cannot be "
                                          "combined"});
    }

    if (options.useJointDiis && !options.useDiis)
    {
        // The joint extrapolator engages through the per-spin DIIS switch;
        // without useDiis the flag would be a silent no-op.
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the joint-system DIIS requires useDiis"});
    }

    if (options.useTwoPhaseDiis && !options.useDiis)
    {
        // Same gating for the two-phase per-spin variant: its handoff
        // logic lives on the DIIS path, and without useDiis the flag
        // would be a silent no-op.
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the two-phase DIIS requires useDiis"});
    }

    // The per-iteration trace side-channel (diagnostics): opened once
    // at loop start, one line per iteration after its convergence decision
    // (scf_common.hpp ScfTraceWriter). Only the main run's options carry a
    // traceFile (the SAD atomic fragments pass fragment-local options), so
    // exactly one RunUhfLoop call writes the trace.
    internal::ScfTraceWriter trace(options.traceFile);

    const std::size_t n = static_cast<std::size_t>(overlap.rows());

    if (static_cast<std::size_t>(overlap.cols()) != n ||
        static_cast<std::size_t>(coreHamiltonian.rows()) != n ||
        static_cast<std::size_t>(coreHamiltonian.cols()) != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "overlap and core Hamiltonian must share the function count"});
    }

    // The occupation vectors arrive at the AO function count n - the dimension
    // their callers can know. The loop works in the numKept-dimensional
    // orthonormal space once the removal has fired, so it truncates them to
    // that space below (see the derivation after the removal guard).
    if (fullOccupationsAlpha.size() != static_cast<Eigen::Index>(n) ||
        fullOccupationsBeta.size() != static_cast<Eigen::Index>(n))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "occupation vectors must match the function count"});
    }

    const int numOccupiedAlpha = static_cast<int>(fullOccupationsAlpha.sum());
    const int numOccupiedBeta = static_cast<int>(fullOccupationsBeta.sum());

    if (numOccupiedAlpha > static_cast<int>(n) || numOccupiedBeta > static_cast<int>(n))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "more occupied orbitals than basis functions in the requested spin state"});
    }

    if (options.useMom && (options.momReferenceAlpha.rows() != static_cast<Eigen::Index>(n) ||
                           options.momReferenceBeta.rows() != static_cast<Eigen::Index>(n) ||
                           options.momReferenceAlpha.cols() < numOccupiedAlpha ||
                           options.momReferenceBeta.cols() < numOccupiedBeta))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "MOM references must match the function count and carry the occupied set"});
    }

    if ((options.initialDensityAlpha.size() != 0 &&
         (options.initialDensityAlpha.rows() != static_cast<Eigen::Index>(n) ||
          options.initialDensityAlpha.cols() != static_cast<Eigen::Index>(n))) ||
        (options.initialDensityBeta.size() != 0 &&
         (options.initialDensityBeta.rows() != static_cast<Eigen::Index>(n) ||
          options.initialDensityBeta.cols() != static_cast<Eigen::Index>(n))))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "initial densities must match the function count"});
    }

    const Eigen::Index eigenN = static_cast<Eigen::Index>(n);
    Eigen::MatrixXd dAlpha = options.initialDensityAlpha.size() == 0
                                 ? Eigen::MatrixXd::Zero(eigenN, eigenN)
                                 : options.initialDensityAlpha;
    Eigen::MatrixXd dBeta = options.initialDensityBeta.size() == 0
                                ? Eigen::MatrixXd::Zero(eigenN, eigenN)
                                : options.initialDensityBeta;

    // Restart seeding: the checkpoint's per-spin densities
    // replace the initial guess, the DIIS histories are restored, and the
    // convergence gate's previous energies are seeded. Empty fields are
    // no-ops, so default-constructed options behave exactly as before.
    if ((options.initialScfState.densityAlpha.size() != 0 &&
         (options.initialScfState.densityAlpha.rows() != static_cast<Eigen::Index>(n) ||
          options.initialScfState.densityAlpha.cols() != static_cast<Eigen::Index>(n))) ||
        (options.initialScfState.densityBeta.size() != 0 &&
         (options.initialScfState.densityBeta.rows() != static_cast<Eigen::Index>(n) ||
          options.initialScfState.densityBeta.cols() != static_cast<Eigen::Index>(n))))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the restart densities must match the function count"});
    }

    if (options.initialScfState.densityAlpha.size() != 0)
    {
        dAlpha = options.initialScfState.densityAlpha;
    }

    if (options.initialScfState.densityBeta.size() != 0)
    {
        dBeta = options.initialScfState.densityBeta;
    }

    auto orthogonalized = internal::OrthogonalizeOverlap(overlap);

    if (!orthogonalized.has_value())
    {
        return std::unexpected(orthogonalized.error());
    }

    // The linear-dependence removal (scf_common.hpp OrthogonalizeOverlap) is
    // DISCLOSED, never silent, and it is a SYSTEM property: the count
    // here equals the RHF count for the same molecule and basis, because the
    // threshold lives in the orthogonalizer and nothing branches on the
    // method. A system that trips the floor runs in the numKept-dimensional
    // orthonormal space and maps back - D_AO = X D_orth X^T is n x n, the
    // full AO dimension. The one genuine error is needing more occupied
    // orbitals than the removal left directions for.
    if (static_cast<int>(orthogonalized->KeptDimension()) <
        std::max(numOccupiedAlpha, numOccupiedBeta))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the overlap carries fewer usable directions (" +
                           std::to_string(orthogonalized->KeptDimension()) + " after removing " +
                           std::to_string(orthogonalized->numRemoved) +
                           " near-dependent ones) than the occupied orbitals this system needs"});
    }

    const Eigen::MatrixXd x = orthogonalized->x;
    const std::size_t numRemovedOverlap = orthogonalized->numRemoved;

    // The removal's other half on the unrestricted path: the working space is
    // numKept-dimensional, and BOTH occupation vectors are indexed by the MO
    // index of THAT space. DiagonalizeFock returns C_AO = X C_orth with
    // numKept columns and BuildSpinDensity contracts the mask against those
    // columns, so the n-sized vectors the callers can build are truncated
    // here, where the kept dimension is first known. Nothing moves: MO index j
    // of the reduced space is column j, and the removed directions are the
    // near-null subspace of S, which no aufbau or SAD vector occupies.
    //
    // Weight at an index the removal dropped is therefore a property of the
    // CALLER's vector, not of the reduction - it would be silently lost by the
    // truncation, so it is refused by name instead (honoured, refused by
    // name, or demoted with the disclosure). The test is exact rather than
    // tolerant because every producer of these vectors builds them with exact
    // zeros outside the occupied range (IntegerOccupations and the SAD
    // fragment's Zero(nZ)), so a nonzero tail is a real error and not
    // round-off.
    const Eigen::Index numKept = static_cast<Eigen::Index>(orthogonalized->KeptDimension());

    if (fullOccupationsAlpha.tail(fullOccupationsAlpha.size() - numKept).sum() != 0.0 ||
        fullOccupationsBeta.tail(fullOccupationsBeta.size() - numKept).sum() != 0.0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the occupation vectors carry weight on overlap directions the "
                       "linear-dependence removal dropped"});
    }

    const Eigen::VectorXd occupationsAlpha = fullOccupationsAlpha.head(numKept);
    const Eigen::VectorXd occupationsBeta = fullOccupationsBeta.head(numKept);

    // Symmetry-blocked diagonalization: when the caller supplies the
    // AO basis, detect the point group once and build the AO-space
    // decomposition once; every iteration then diagonalizes both spin
    // Focks' irrep blocks separately (equivalent to the plain n x n solve
    // up to rounding). C1 molecules and unrealizable groups fall back to
    // the plain path; an inconsistent detection is a real error and
    // propagates. The GWH guess diagonalization (BuildGwhGuess) stays on
    // the plain path - it is a one-shot, negligible cost.
    std::optional<internal::SymmetryBlocks> symmetryBlocks;
    std::optional<internal::BlockedDiagonalizeData> blockedDiagonalizeData;
    std::optional<qcx::symmetry::SymmetryAnalysis> analysis;

    if (basisSet != nullptr)
    {
        analysis = qcx::symmetry::DetectPointGroup(molecule);
        auto built = internal::BuildSymmetryBlocks(molecule, *basisSet, *analysis);

        if (!built.has_value() && built.error().code != qcx::ErrorCode::kUnimplemented)
        {
            return std::unexpected(built.error());
        }

        if (built.has_value() && !built->isTrivial)
        {
            symmetryBlocks = std::move(*built);

            // The per-irrep transform data: U^T X U is
            // block-diagonal up to noise (X = S^{-1/2} commutes with the
            // group), so its diagonal blocks are built once per run and the
            // per-iteration diagonalization transforms per irrep.
            auto data = internal::BuildBlockedDiagonalizeData(x, *symmetryBlocks);

            if (!data.has_value())
            {
                return std::unexpected(data.error());
            }

            blockedDiagonalizeData = std::move(*data);
        }
    }

    // The guard's per-spin disclosure state, declared beside the
    // diagonalizer because that is where the decision is made. Each
    // spin's OWN Fock decides its own path: the alpha and beta Focks are
    // different matrices, and on a symmetry-broken unrestricted solution one
    // may be symmetry-adapted while the other is not - so the choice is
    // per-spin and per-diagonalization, never one run-level flag.
    SymmetryBlockingReport symmetryBlockingAlpha;
    SymmetryBlockingReport symmetryBlockingBeta;

    // Both spins share the symmetry blocks; the lambda keeps the
    // blocked/plain choice at one place. Hoisted out of the
    // iteration so the finalizer re-diagonalizes the returned
    // densities' Focks with EXACTLY the transform the walk used: two call
    // sites, one rule. Columns come back ascending by orbital energy
    // (SelfAdjointEigenSolver orders its eigenvalues; the blocked path
    // preserves that order).
    //
    // the EQUIVALENCE IS CONDITIONAL, so the choice is guarded (this is the
    // guard's home on the unrestricted path). The argument "U^T (X^T F X) U
    // is block-diagonal up to integral noise, so the blocked solve equals the
    // plain one" is exactly as good as "F commutes with the group", and that
    // is a property of the SOLUTION, not of the molecule. The point group is
    // a property of the NUCLEAR FRAMEWORK; the group that may legitimately
    // constrain a solution is the invariance group OF THAT SOLUTION. An
    // unrestricted solution is free to have a smaller one - that is what an
    // RHF -> UHF instability IS - and blocking by the full point group does
    // not then accelerate it: it projects the broken solution out and returns
    // the symmetry-adapted one as if it were the answer, converged and
    // variational-looking, at the HIGHER energy (measured on stretched
    // H2/STO-3G, scf/tests/uhf_symmetry_constraint_test.cpp).
    //
    // So each iteration measures the precondition on that spin's own Fock
    // (internal::MeasureSymmetryAdaptation) and refuses the blocks when it
    // fails. The refusal is the SAFE direction: a blocked solve of a
    // non-adapted Fock silently CONSTRAINS the variational search, while a
    // plain solve of an adapted Fock only loses the speedup. The test is
    // necessary and not sufficient - a small commutator says the matrix about
    // to be blocked is adapted, never that no broken solution exists - which
    // is why refusing does not change what a plain run would have found.
    const auto diagonalizeFock =
        [&symmetryBlocks, &blockedDiagonalizeData, &x](
            const Eigen::MatrixXd& fock,
            SymmetryBlockingReport& report) -> qcx::Result<Eigen::MatrixXd> {
        // The blocking needs both the decomposition and data that can serve
        // it: the linear-dependence removal leaves a non-square orthogonalizer
        // and DiagonalizeFockBlocked disengages from it by design (its own
        // reasoning), so the run must not report having blocked anything here.
        if (!symmetryBlocks.has_value() || blockedDiagonalizeData->isTrivial)
        {
            if (symmetryBlocks.has_value())
            {
                report.action = SymmetryBlockingAction::kUnavailable;
            }

            return qcx::Result<Eigen::MatrixXd>(internal::DiagonalizeFock(fock, x));
        }

        const auto adaptation =
            internal::MeasureSymmetryAdaptation(fock, *symmetryBlocks, *blockedDiagonalizeData);
        report.tolerance = internal::kSymmetryAdaptationTolerance;

        // The per-generator norms travel with the WORST measurement of the
        // channel, so the recorded vector always explains the recorded
        // maximum - a reader can check the decision rather than take it.
        if (report.generatorCommutatorNorms.empty() ||
            adaptation.maxCommutatorNorm > report.maxGeneratorCommutatorNorm)
        {
            report.generatorCommutatorNorms = adaptation.generatorCommutatorNorms;
            report.maxGeneratorCommutatorNorm = adaptation.maxCommutatorNorm;
        }

        if (adaptation.maxCommutatorNorm <= internal::kSymmetryAdaptationTolerance)
        {
            ++report.blockedSolveCount;
            report.action = report.plainSolveCount == 0 ? SymmetryBlockingAction::kUsed
                                                        : SymmetryBlockingAction::kDemoted;
            return internal::DiagonalizeFockBlocked(
                fock, x, *symmetryBlocks, *blockedDiagonalizeData);
        }

        ++report.plainSolveCount;
        report.action = report.blockedSolveCount == 0 ? SymmetryBlockingAction::kRefused
                                                      : SymmetryBlockingAction::kDemoted;
        return qcx::Result<Eigen::MatrixXd>(internal::DiagonalizeFock(fock, x));
    };

    // The full-group labeling stage, applied per spin to the returned
    // state (labels and the symmetrized densities; nothing computed in the
    // loop changes). C1 molecules, the missing-basis runs, the off toggle,
    // and unrealizable groups pass through unlabeled; a realized
    // inconsistency is a real error and propagates.
    const auto applyLabeling = [&](UhfResult& result,
                                   const Eigen::MatrixXd& fockAlpha,
                                   const Eigen::MatrixXd& fockBeta,
                                   std::size_t occupiedCountAlpha,
                                   std::size_t occupiedCountBeta) -> qcx::Result<void> {
        if (basisSet == nullptr || !options.fullGroupLabeling || !analysis.has_value() ||
            analysis->group == qcx::symmetry::PointGroupName::kC1)
        {
            return {};
        }

        // The linear-dependence removal DISENGAGES this stage, exactly
        // as it does on the closed-shell path and for the same reason: the
        // stage labels and symmetrizes the n-dimensional MO set, while the
        // removal leaves coefficients n x numKept. See rhf.cpp's applyLabeling
        // for the full reasoning and for which record key discloses it.
        if (numRemovedOverlap != 0)
        {
            return {};
        }

        const auto labelSpin =
            [&](const Eigen::MatrixXd& fock,
                const Eigen::MatrixXd& coefficients,
                const Eigen::MatrixXd& density,
                std::size_t occupiedCount) -> qcx::Result<std::optional<qcx::scf::SymmetryLabels>> {
            auto labeled = internal::SymmetryLabelAndSymmetrize(fock,
                                                                coefficients,
                                                                density,
                                                                static_cast<int>(occupiedCount),
                                                                molecule,
                                                                *basisSet,
                                                                *analysis);

            if (!labeled.has_value())
            {
                if (labeled.error().code == qcx::ErrorCode::kUnimplemented)
                {
                    return std::nullopt; // Unrealizable group: pass through unlabeled.
                }

                return std::unexpected(labeled.error());
            }

            return std::optional<qcx::scf::SymmetryLabels>{std::move(*labeled)};
        };

        auto labeledAlpha =
            labelSpin(fockAlpha, result.coefficientsAlpha, result.densityAlpha, occupiedCountAlpha);

        if (!labeledAlpha.has_value())
        {
            return std::unexpected(labeledAlpha.error());
        }

        auto labeledBeta =
            labelSpin(fockBeta, result.coefficientsBeta, result.densityBeta, occupiedCountBeta);

        if (!labeledBeta.has_value())
        {
            return std::unexpected(labeledBeta.error());
        }

        result.symmetryLabelsAlpha = std::move(*labeledAlpha);
        result.symmetryLabelsBeta = std::move(*labeledBeta);
        return {};
    };

    // The density-consistent reporting contract - the per-spin
    // analog of rhf.cpp's reportReturnedDensityEnergy, which carries the
    // derivation and the measured route dependence. The loop's energy is the
    // trace of the DIIS-EXTRAPOLATED Focks at the new spin densities, while
    // the Focks of the RETURNED densities are what the returned state's
    // energy belongs to; the gap is first-order in the extrapolation mismatch
    // (delta = 1/2 sum_sigma Tr[D_sigma (G_sigma[D-bar] - G_sigma[D])]).
    //
    // Corrected ONCE, at the exit, on the result the run returns: rebuild both
    // spin Focks from the returned spin densities and re-form the trace - one
    // extra Fock build per run, never per iteration. The trajectory, the
    // iteration counts, the gate legs and the gate's recorded operands
    // (achievedEnergyDelta / achievedRmsDensityChange) are untouched, and the
    // returned orbital spectra follow the returned densities (the CONTRACT
    // change, rhf.cpp's twin block carries the derivation). The
    // converged-state contract then closes the second half: the returned
    // densities are the returned coefficients' OWN densities (the exit builds
    // them from those coefficients), so the result is one state rather than a
    // density beside another state's orbitals, and the two residuals it
    // reports say which legs hold exactly and which sit at the polish residual.
    // The two-callback (seam/KS) path is EXCLUDED - it already evaluates at the
    // returned density, and its remaining gap (its returned C and eps are the
    // DIIS-extrapolated Focks' eigenpairs) needs the iterate-until-consistent
    // form the nonlinear Vxc leg requires.
    //
    // The restart state is refreshed with the same pair: ScfRestartState
    // documents those fields as the GATE's state (scf_state.hpp), the next run
    // reads them back as its previous-iterate energy, and the checkpoint's
    // densities are the returned ones - a trace energy beside them would
    // re-inject this defect at the restart seam.
    // The occupied counts are PASSED IN rather than recomputed here: they are
    // the labeling stage's occupied counts. The DENSITY side does not reuse
    // them - the returned occupation masks are re-derived from the returned
    // coefficients inside (see the comment there), because the trajectory's
    // mask indexes the extrapolated Focks' orbital ordering.
    // The occupation mask of one returned spin channel: the run's own rule (the
    // aufbau vector, or the MOM selection) applied to THOSE coefficients.
    // Derived from the coefficients it will accompany rather than carried over
    // from the trajectory, whose mask indexes the extrapolated Focks' orbital
    // ordering. One home for the rule - the entry measurement below and the
    // polish both use it.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const auto returnedOccupationMask = [&](const Eigen::MatrixXd& coefficients,
                                            const Eigen::MatrixXd& reference,
                                            const Eigen::VectorXd& aufbauOccupations,
                                            int numOccupied) -> Eigen::VectorXd {
        if (!options.useMom)
        {
            return aufbauOccupations;
        }

        return MomOccupationMask(coefficients, reference, overlap, numOccupied);
    };

    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const auto reportReturnedDensityEnergy =
        [&](UhfResult& result,
            std::size_t occupiedCountAlpha,
            std::size_t occupiedCountBeta) -> qcx::Result<void> {
        // The projector leg is measured on EVERY path, the seam/KS carve-out
        // below included: a field left at its default would report a
        // consistency the run never checked (a run record that can
        // silently differ from what ran), and the carve-out returns the loop's
        // own per-spin pair, whose projector residual is a real measurement.
        // The polish below overwrites these with the returned pair's own.
        result.stateConsistencyResidualAlpha =
            ReturnedStateResidual(result.densityAlpha,
                                  result.coefficientsAlpha,
                                  returnedOccupationMask(result.coefficientsAlpha,
                                                         options.momReferenceAlpha,
                                                         occupationsAlpha,
                                                         numOccupiedAlpha));
        result.stateConsistencyResidualBeta =
            ReturnedStateResidual(result.densityBeta,
                                  result.coefficientsBeta,
                                  returnedOccupationMask(result.coefficientsBeta,
                                                         options.momReferenceBeta,
                                                         occupationsBeta,
                                                         numOccupiedBeta));

        if (coulomb != nullptr)
        {
            return {};
        }

        const auto returnedFocks = buildFock(result.densityAlpha, result.densityBeta);

        if (!returnedFocks.has_value())
        {
            return std::unexpected(returnedFocks.error());
        }

        // The per-spin orbital spectra move with the returned densities: the
        // returned MO sets are the returned
        // densities' Fock eigenpairs, not the extrapolated Focks'.
        // Re-diagonalizing (rather than taking diag(C^T F[D] C) in the old
        // basis) is what keeps the result self-describing: the solver
        // returns ASCENDING eigenvalues with matching eigenvectors, so the
        // documented "columns ascending by orbital energy" holds by
        // construction and no re-sort is needed.
        const auto finalAlpha = diagonalizeFock(returnedFocks->first, symmetryBlockingAlpha);

        if (!finalAlpha.has_value())
        {
            return std::unexpected(finalAlpha.error());
        }

        const auto finalBeta = diagonalizeFock(returnedFocks->second, symmetryBlockingBeta);

        if (!finalBeta.has_value())
        {
            return std::unexpected(finalBeta.error());
        }

        // The converged-state contract: each returned density is the
        // returned MOs' OWN density, D_sigma = C_sigma,occ C_sigma,occ^T - not
        // the trajectory density the Focks above were built from. The two are
        // one SCF step apart (the trajectory's densities came out of the
        // DIIS-EXTRAPOLATED Focks' eigenvectors; these are the physical ones),
        // which is exactly the gap the O2 triplet pin caught: 3.4e-9 (alpha)
        // and 1.3e-6 (beta) on a run that reported converged. No convergence
        // criterion bounds it - the density gate compares two TRAJECTORY
        // iterates, and the extrapolation is not a physical tolerance.
        //
        // The returned occupation masks follow the returned coefficients
        // (returnedOccupationMask, the loop's own rule applied to them) - the
        // trajectory's mask indexes the extrapolated Focks' orbital ordering,
        // so reusing it here would occupy different orbitals of this set.
        const Eigen::VectorXd returnedOccAlpha = returnedOccupationMask(
            *finalAlpha, options.momReferenceAlpha, occupationsAlpha, numOccupiedAlpha);
        const Eigen::VectorXd returnedOccBeta = returnedOccupationMask(
            *finalBeta, options.momReferenceBeta, occupationsBeta, numOccupiedBeta);

        const auto finalDensityAlpha = BuildSpinDensity(*finalAlpha, returnedOccAlpha);

        if (!finalDensityAlpha.has_value())
        {
            return std::unexpected(finalDensityAlpha.error());
        }

        const auto finalDensityBeta = BuildSpinDensity(*finalBeta, returnedOccBeta);

        if (!finalDensityBeta.has_value())
        {
            return std::unexpected(finalDensityBeta.error());
        }

        // One more per-spin Fock build, on the RETURNED densities: the pair the
        // returned energy belongs to. Deliberately NOT the source of the
        // returned orbital energies - those stay the eigenpairs of the density
        // estimate's Focks above, which the returned coefficients diagonalize
        // exactly, while they are not an eigenbasis of these matrices. That is
        // the one leg of the contract left at the fixed-point residual
        // (F_sigma[D_returned] - F_sigma[D_seed] is linear in it - F is affine
        // in D - and bounded by it: the contract's HF one-pass exactness).
        const auto finalFocks = buildFock(*finalDensityAlpha, *finalDensityBeta);

        if (!finalFocks.has_value())
        {
            return std::unexpected(finalFocks.error());
        }

        result.stateConsistencyResidualAlpha =
            ReturnedStateResidual(*finalDensityAlpha, *finalAlpha, returnedOccAlpha);
        result.stateConsistencyResidualBeta =
            ReturnedStateResidual(*finalDensityBeta, *finalBeta, returnedOccBeta);
        result.stateFixedPointResidualAlpha = (*finalDensityAlpha - result.densityAlpha).norm();
        result.stateFixedPointResidualBeta = (*finalDensityBeta - result.densityBeta).norm();
        result.densityAlpha = *finalDensityAlpha;
        result.densityBeta = *finalDensityBeta;
        result.coefficientsAlpha = *finalAlpha;
        result.coefficientsBeta = *finalBeta;
        result.orbitalEnergiesAlpha =
            (result.coefficientsAlpha.transpose() * returnedFocks->first * result.coefficientsAlpha)
                .diagonal();
        result.orbitalEnergiesBeta =
            (result.coefficientsBeta.transpose() * returnedFocks->second * result.coefficientsBeta)
                .diagonal();
        result.electronicEnergy = ComputeUhfElectronicEnergy(coreHamiltonian,
                                                             finalFocks->first,
                                                             finalFocks->second,
                                                             result.densityAlpha,
                                                             result.densityBeta);
        result.totalEnergy =
            result.electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);

        // The labels move with the coefficients: SymmetryLabels snapshots the
        // MO set and the density it was handed, so leaving the in-loop pass in
        // place would describe a different set than the one this result
        // returns. Re-derived against the returned densities' Focks - the
        // matrices the returned coefficients are eigenvectors of, with the
        // returned densities beside them. On the seam/KS path the finalizer
        // returns before this point and the in-loop pass stands.
        if (const auto labeled = applyLabeling(result,
                                               returnedFocks->first,
                                               returnedFocks->second,
                                               occupiedCountAlpha,
                                               occupiedCountBeta);
            !labeled.has_value())
        {
            return std::unexpected(labeled.error());
        }

        // The checkpoint carries the returned state: ScfRestartState documents
        // these as the densities the run stands on and the gate's own energies,
        // and a restart a density apart from the result would hand the next run
        // a first energy leg comparing unlike objects.
        result.restart.densityAlpha = result.densityAlpha;
        result.restart.densityBeta = result.densityBeta;
        result.restart.previousTotalEnergy = result.totalEnergy;
        result.restart.previousElectronicEnergy = result.electronicEnergy;

        if (internal::ReturnDiagnosticsEnabled())
        {
            std::fprintf(stderr,
                         "scf-return: e=%.17g iters=%d naset=%zu consistencyA=%.6g "
                         "consistencyB=%.6g fixedpointA=%.6g fixedpointB=%.6g\n",
                         result.totalEnergy,
                         result.iterations,
                         result.numRemovedOverlapDirections,
                         result.stateConsistencyResidualAlpha,
                         result.stateConsistencyResidualBeta,
                         result.stateFixedPointResidualAlpha,
                         result.stateFixedPointResidualBeta);
        }

        return {};
    };

    // Per-spin CDIIS: the two-phase wrapper on fresh runs when
    // options.useTwoPhaseDiis (phase-1 window kDiisPhaseHistoryLimit), the
    // plain 8-window extrapolator otherwise - the default trajectory is
    // bit-identical to the pre-stage form.
    TwoPhaseDiis diisAlpha(options.useTwoPhaseDiis ? kDiisPhaseHistoryLimit : kDiisHistoryLimit,
                           options.useTwoPhaseDiis);
    TwoPhaseDiis diisBeta(options.useTwoPhaseDiis ? kDiisPhaseHistoryLimit : kDiisHistoryLimit,
                          options.useTwoPhaseDiis);

    if (const auto restoreAlpha = diisAlpha.Restore(options.initialScfState.diisAlpha);
        !restoreAlpha.has_value())
    {
        return std::unexpected(restoreAlpha.error());
    }

    if (const auto restoreBeta = diisBeta.Restore(options.initialScfState.diisBeta);
        !restoreBeta.has_value())
    {
        return std::unexpected(restoreBeta.error());
    }

    // Joint-system DIIS (Step 5): when options.useJointDiis, ONE
    // extrapolator over the combined alpha+beta error vectors replaces the
    // two per-spin ones (niedoida's joint form). The restart state's two
    // per-spin DiisState fields carry the joint history (alpha pair in
    // diisAlpha, beta pair in diisBeta - the same two-field shape, no new
    // serialization surface). The per-spin extrapolators stay constructed
    // (their snapshots back the joint path's restart handoff only when
    // joint is off).
    std::optional<internal::JointSystemDiis> jointDiis;

    if (options.useJointDiis)
    {
        jointDiis.emplace(kDiisHistoryLimit);

        if (const auto restore = jointDiis->Restore(options.initialScfState.diisAlpha,
                                                    options.initialScfState.diisBeta);
            !restore.has_value())
        {
            return std::unexpected(restore.error());
        }
    }

    // The restart state's DIIS fields: the per-spin histories, or the joint
    // histories (alpha pair in diisAlpha, beta pair in diisBeta - the same
    // two-field shape) when the joint extrapolator is active.
    const auto restartDiisStates = [&]() -> std::pair<DiisState, DiisState> {
        if (jointDiis.has_value())
        {
            return jointDiis->Snapshot();
        }

        return {diisAlpha.Snapshot(), diisBeta.Snapshot()};
    };

    double previousTotalEnergy = options.initialScfState.previousTotalEnergy;
    double previousElectronicEnergy = options.initialScfState.previousElectronicEnergy;
    Eigen::MatrixXd coefficientsAlpha;
    Eigen::MatrixXd coefficientsBeta;
    // The last-iterate Focks captured into the result: the orbital
    // energies are the diagonal of the final MO-basis Fock, and the
    // non-converged path must return the FINAL iterate (uhf.hpp's UhfResult
    // comment).
    Eigen::MatrixXd finalFockAlpha;
    Eigen::MatrixXd finalFockBeta;
    std::vector<int> occupiedAlpha;
    std::vector<int> occupiedBeta;

    // The gate's own operands of the LAST executed iteration, hoisted so the
    // budget exit below records them too (UhfResult::achievedEnergyDelta /
    // achievedRmsDensityChange - the HfResult pair's meaning).
    double achievedEnergyDelta = 0.0;
    double achievedRmsDensityChange = 0.0;

    for (int iteration = 0; iteration < options.maxIterations; ++iteration)
    {
        const auto fockResult = buildFock(dAlpha, dBeta);

        if (!fockResult.has_value())
        {
            return std::unexpected(fockResult.error());
        }

        Eigen::MatrixXd fockAlpha = fockResult->first;
        Eigen::MatrixXd fockBeta = fockResult->second;

        // The per-spin error channels, computed once per spin per iteration
        // (the norm is hoisted to the call site: the robust gate's
        // DIIS-error leg sees exactly the quantity the extrapolation used).
        const auto errorAlpha = DiisError(fockAlpha, dAlpha, overlap, x);

        if (!errorAlpha.has_value())
        {
            return std::unexpected(errorAlpha.error());
        }

        const double diisErrorNormAlpha = errorAlpha->norm();
        const auto errorBeta = DiisError(fockBeta, dBeta, overlap, x);

        if (!errorBeta.has_value())
        {
            return std::unexpected(errorBeta.error());
        }

        const double diisErrorNormBeta = errorBeta->norm();

        // The per-spin CDIIS path (default; each channel's error against
        // its own density goes into its own extrapolator - niedoida's
        // uhf.cpp uses a joint system instead, and the per-spin form is
        // kept deliberately) or the Step-5 joint
        // path, dispatched in MaybeExtrapolateFockPair.
        const auto extrapolation = MaybeExtrapolateFockPair(options,
                                                            jointDiis,
                                                            diisAlpha,
                                                            diisBeta,
                                                            fockAlpha,
                                                            *errorAlpha,
                                                            diisErrorNormAlpha,
                                                            fockBeta,
                                                            *errorBeta,
                                                            diisErrorNormBeta);

        if (!extrapolation.has_value())
        {
            return std::unexpected(extrapolation.error());
        }

        fockAlpha = extrapolation->first;
        fockBeta = extrapolation->second;

        finalFockAlpha = fockAlpha;
        finalFockBeta = fockBeta;

        // The virtual-space level shift: per spin, the
        // extrapolated Fock gains (S·C)·B·(S·C)^T before the
        // diagonalization - AFTER the DIIS extrapolation and OUTSIDE the
        // energy computation and the DIIS error (finalFockAlpha/Beta still
        // hold the unshifted extrapolated Focks; the next iteration's error
        // is computed from the freshly built Fock). The shifted matrix
        // never enters the energy path, so the 0.5·Delta·Tr[D(S−D)] bias of
        // the record is structurally absent; shifting after the
        // extrapolation moves no fixed points (measured), so the
        // converged density/energy is the unshifted solution's and no
        // correction is needed. The shift uses the PREVIOUS iteration's
        // coefficients and is skipped on iteration 0 (no previous C yet -
        // niedoida's m_first_iteration behavior).
        if (options.useLevelShift && iteration > 0)
        {
            internal::ApplyVirtualSpaceLevelShift(
                fockAlpha, overlap, coefficientsAlpha, options.levelShiftAlpha, numOccupiedAlpha);
            internal::ApplyVirtualSpaceLevelShift(
                fockBeta, overlap, coefficientsBeta, options.levelShiftBeta, numOccupiedBeta);
        }

        // Both spins share the symmetry blocks (the hoisted diagonalizeFock
        // above keeps the blocked/plain choice, and the guard, at one
        // place); each spin's own Fock decides its own path.
        const auto coefficientsAlphaResult = diagonalizeFock(fockAlpha, symmetryBlockingAlpha);

        if (!coefficientsAlphaResult.has_value())
        {
            return std::unexpected(coefficientsAlphaResult.error());
        }

        coefficientsAlpha = *coefficientsAlphaResult;
        const auto coefficientsBetaResult = diagonalizeFock(fockBeta, symmetryBlockingBeta);

        if (!coefficientsBetaResult.has_value())
        {
            return std::unexpected(coefficientsBetaResult.error());
        }

        coefficientsBeta = *coefficientsBetaResult;

        const Eigen::VectorXd occAlphaVector =
            options.useMom
                ? MomOccupationMask(
                      coefficientsAlpha, options.momReferenceAlpha, overlap, numOccupiedAlpha)
                : occupationsAlpha;
        const Eigen::VectorXd occBetaVector =
            options.useMom
                ? MomOccupationMask(
                      coefficientsBeta, options.momReferenceBeta, overlap, numOccupiedBeta)
                : occupationsBeta;

        auto newDAlphaResult = BuildSpinDensity(coefficientsAlpha, occAlphaVector);

        if (!newDAlphaResult.has_value())
        {
            return std::unexpected(newDAlphaResult.error());
        }

        const Eigen::MatrixXd& newDAlpha = *newDAlphaResult;
        auto newDBetaResult = BuildSpinDensity(coefficientsBeta, occBetaVector);

        if (!newDBetaResult.has_value())
        {
            return std::unexpected(newDBetaResult.error());
        }

        const Eigen::MatrixXd& newDBeta = *newDBetaResult;

        // The energy path reads the UNSHIFTED extrapolated Focks
        // (finalFockAlpha/Beta), never the shifted diagonalization inputs -
        // the bias accounting of the level shift. Two paths, selected by
        // whether the caller supplied the energy seam: the Hartree-Fock trace
        // identity every existing caller uses, or the Coulomb trace plus the
        // caller's contribution, which is what a Kohn-Sham run needs because no
        // trace of the Fock can recover Exc.
        double electronicEnergy = 0.0;

        if (coulomb == nullptr)
        {
            electronicEnergy = ComputeUhfElectronicEnergy(
                coreHamiltonian, finalFockAlpha, finalFockBeta, newDAlpha, newDBeta);
        } else
        {
            // J is built from the TOTAL density - the same spin-summed object
            // the Fock builder's Coulomb half contracts - while the
            // contribution sees both spin densities separately. The total is
            // formed once: the provider and the trace below must see the same
            // matrix, not two independently rounded sums of it.
            const Eigen::MatrixXd totalDensity = newDAlpha + newDBeta;
            auto coulombMatrix = (*coulomb)(totalDensity);

            if (!coulombMatrix.has_value())
            {
                return std::unexpected(coulombMatrix.error());
            }

            auto contributed = (*energyContribution)(newDAlpha, newDBeta);

            if (!contributed.has_value())
            {
                return std::unexpected(contributed.error());
            }

            electronicEnergy =
                ComputeUhfCoulombTraceEnergy(coreHamiltonian, *coulombMatrix, totalDensity) +
                *contributed;
        }

        const double totalEnergy =
            electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);

        // The RMS density change summed over both spins, / n (not / n^2 -
        // the n^2 divisor made the RHF gate ~n x too lenient; rhf.cpp's
        // BUG-1 lesson, don't reintroduce it here).
        const double rmsDensityChange =
            ((newDAlpha - dAlpha).norm() + (newDBeta - dBeta).norm()) / static_cast<double>(n);

        // The robust gate's max-density leg: the largest element of
        // |newD - D| over both spins (the RMS leg keeps the /n divisor; the
        // max leg carries the full densityTolerance).
        const double maxDensityChange = std::max((newDAlpha - dAlpha).cwiseAbs().maxCoeff(),
                                                 (newDBeta - dBeta).cwiseAbs().maxCoeff());

        const bool converged = internal::IsUhfConverged(iteration,
                                                        totalEnergy,
                                                        previousTotalEnergy,
                                                        rmsDensityChange,
                                                        maxDensityChange,
                                                        diisErrorNormAlpha,
                                                        diisErrorNormBeta,
                                                        options);

        // The trace's energy delta: the same |E_n - E_{n-1}| the gate
        // compared (the UHF gate runs before the bookkeeping below, so
        // previousTotalEnergy still holds the previous iteration's energy).
        const double energyDelta = std::fabs(totalEnergy - previousTotalEnergy);

        achievedEnergyDelta = energyDelta;
        achievedRmsDensityChange = rmsDensityChange;

        dAlpha = newDAlpha;
        dBeta = newDBeta;
        previousTotalEnergy = totalEnergy;
        previousElectronicEnergy = electronicEnergy;
        occupiedAlpha = OccupiedIndices(occAlphaVector);
        occupiedBeta = OccupiedIndices(occBetaVector);

        // The trace line lands AFTER the convergence decision with the
        // exact numbers that decision used (write-only: nothing reads it
        // back, so the trajectory is untouched).
        trace.Write(iteration + 1, totalEnergy, energyDelta, rmsDensityChange, converged);

        if (converged)
        {
            UhfResult summary;
            summary.totalEnergy = totalEnergy;
            summary.electronicEnergy = electronicEnergy;
            summary.converged = true;
            summary.iterations = iteration + 1;
            summary.spinSquared = ComputeSpinSquared(
                overlap, coefficientsAlpha, coefficientsBeta, occupiedAlpha, occupiedBeta);
            summary.achievedEnergyDelta = achievedEnergyDelta;
            summary.achievedRmsDensityChange = achievedRmsDensityChange;
            summary.numRemovedOverlapDirections = numRemovedOverlap;
            summary.symmetryBlockingAlpha = symmetryBlockingAlpha;
            summary.symmetryBlockingBeta = symmetryBlockingBeta;
            UhfResult finished = FinishSummary(std::move(summary),
                                               std::move(dAlpha),
                                               std::move(dBeta),
                                               std::move(coefficientsAlpha),
                                               std::move(coefficientsBeta),
                                               finalFockAlpha,
                                               finalFockBeta);
            const auto diisStates = restartDiisStates();
            finished.restart = ScfRestartState{{},
                                               finished.densityAlpha,
                                               finished.densityBeta,
                                               {},
                                               diisStates.first,
                                               diisStates.second,
                                               previousTotalEnergy,
                                               previousElectronicEnergy};

            if (const auto labeled = applyLabeling(finished,
                                                   finalFockAlpha,
                                                   finalFockBeta,
                                                   occupiedAlpha.size(),
                                                   occupiedBeta.size());
                !labeled.has_value())
            {
                return std::unexpected(labeled.error());
            }

            // The density-consistent rule: the reported energy is re-formed from the returned
            // densities before the result leaves the loop (one extra Fock
            // build for the run; the walk above is untouched).
            if (const auto corrected = reportReturnedDensityEnergy(
                    finished, occupiedAlpha.size(), occupiedBeta.size());
                !corrected.has_value())
            {
                return std::unexpected(corrected.error());
            }

            return UhfLoopResult{std::move(finished)};
        }
    }

    UhfResult summary;
    summary.totalEnergy = previousTotalEnergy;
    summary.electronicEnergy = previousElectronicEnergy;
    summary.converged = false;
    summary.iterations = options.maxIterations;
    summary.spinSquared = ComputeSpinSquared(
        overlap, coefficientsAlpha, coefficientsBeta, occupiedAlpha, occupiedBeta);
    summary.achievedEnergyDelta = achievedEnergyDelta;
    summary.achievedRmsDensityChange = achievedRmsDensityChange;
    summary.numRemovedOverlapDirections = numRemovedOverlap;
    summary.symmetryBlockingAlpha = symmetryBlockingAlpha;
    summary.symmetryBlockingBeta = symmetryBlockingBeta;
    UhfResult finished = FinishSummary(std::move(summary),
                                       std::move(dAlpha),
                                       std::move(dBeta),
                                       std::move(coefficientsAlpha),
                                       std::move(coefficientsBeta),
                                       finalFockAlpha,
                                       finalFockBeta);
    const auto diisStates = restartDiisStates();
    finished.restart = ScfRestartState{{},
                                       finished.densityAlpha,
                                       finished.densityBeta,
                                       {},
                                       diisStates.first,
                                       diisStates.second,
                                       previousTotalEnergy,
                                       previousElectronicEnergy};

    if (const auto labeled = applyLabeling(
            finished, finalFockAlpha, finalFockBeta, occupiedAlpha.size(), occupiedBeta.size());
        !labeled.has_value())
    {
        return std::unexpected(labeled.error());
    }

    // The density-consistent rule: the budget exit reports the returned density's energy too
    // (the last iterate, matching the density the result carries).
    if (const auto corrected =
            reportReturnedDensityEnergy(finished, occupiedAlpha.size(), occupiedBeta.size());
        !corrected.has_value())
    {
        return std::unexpected(corrected.error());
    }

    return UhfLoopResult{std::move(finished)};
}

// The aufbau shell degeneracies of a neutral atom (niedoida's 19-shell
// table, core_kit/src/atomic_fragment.cpp - the shells in aufbau order
// with their angular degeneracies).
constexpr int kAufbauShellStates[19] = {1, 1, 3, 1, 3, 1, 5, 3, 1, 5, 3, 1, 7, 5, 3, 1, 7, 5, 3};

// The highest partially filled shell of a neutral atom, and the
// AverageOccupations [niedoida] counts over it: the core states are doubly
// occupied, the degenerate shell carries avr_alpha/avr_beta fractional
// occupations (the spherical-averaging procedure: a fractionally occupied
// degenerate shell yields the rotation-invariant density).
struct DegenerateShell {
    int noCoreStates; // Doubly occupied orbitals below the shell.
    int noDegStates; // Shell degeneracy.
    int noDegElectrons; // Electrons in the shell.
    double averageOccupancyAlpha; // avr_occ_alpha = min(deg_e, deg_s) / deg_s.
    double averageOccupancyBeta; // avr_occ_beta = max(deg_e - deg_s, 0) / deg_s.
};

qcx::Result<DegenerateShell> FindDegenerateShell(int electronCount) {
    int remaining = electronCount;

    for (int shell = 0; shell < 19; ++shell)
    {
        const int states = kAufbauShellStates[shell];

        if (remaining > 2 * states)
        {
            remaining -= 2 * states;
            continue;
        }

        return DegenerateShell{(electronCount - remaining) / 2,
                               states,
                               remaining,
                               static_cast<double>(std::min(remaining, states)) / states,
                               static_cast<double>(std::max(remaining - states, 0)) / states};
    }

    return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                      "atoms beyond the 19-shell aufbau table are not supported"});
}

// The converged atomic fragment of one element: its average-occupation
// densities and its orbital coefficients (embedded later; the C0 MOM
// handoff). The atomic run uses the caller-provided dense integrals of the
// single atom - scf cannot build them itself (module DAG).
struct AtomicFragmentResult {
    Eigen::MatrixXd densityAlpha;
    Eigen::MatrixXd densityBeta;
    Eigen::MatrixXd coefficientsAlpha;
    Eigen::MatrixXd coefficientsBeta;
    double nAlpha = 0.0; // Fractional electron counts of the fragment.
    double nBeta = 0.0;
};

qcx::Result<AtomicFragmentResult> RunAtomicFragment(int atomicNumber,
                                                    const AtomicUhfInputs& inputs,
                                                    const UhfOptions& atomicOptions) {
    const int electronCount = atomicNumber; // The neutral atom.
    const auto shell = FindDegenerateShell(electronCount);

    if (!shell.has_value())
    {
        return std::unexpected(shell.error());
    }

    // The fragment multiplicity from the degenerate-shell occupancy
    // (niedoida's rule in atomic_fragment.cpp).
    const int multiplicity = (shell->noDegElectrons < shell->noDegStates)
                                 ? shell->noDegElectrons + 1
                                 : 2 * shell->noDegStates - shell->noDegElectrons + 1;

    const qcx::molecule::ElementData* element = qcx::molecule::FindElement(atomicNumber);

    if (element == nullptr)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "unknown atomic number in the SAD guess"});
    }

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();

    auto molecule = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{std::string(element->symbol), atomicNumber, 0.0}},
        std::move(*coordinates),
        0,
        multiplicity);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    const Eigen::Index nZ = static_cast<Eigen::Index>(inputs.overlap.rows());

    if (static_cast<Eigen::Index>(inputs.overlap.cols()) != nZ ||
        static_cast<Eigen::Index>(inputs.coreHamiltonian.rows()) != nZ ||
        static_cast<Eigen::Index>(inputs.coreHamiltonian.cols()) != nZ ||
        static_cast<Eigen::Index>(inputs.eri.Shape()[0]) != nZ ||
        static_cast<Eigen::Index>(inputs.eri.Shape()[1]) != nZ ||
        static_cast<Eigen::Index>(inputs.eri.Shape()[2]) != nZ ||
        static_cast<Eigen::Index>(inputs.eri.Shape()[3]) != nZ)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the atomic integrals must share the fragment's function count"});
    }

    // The AverageOccupations occupation vectors: 1.0 on the core states,
    // avr_alpha/avr_beta on the degenerate shell.
    Eigen::VectorXd occupationsAlpha = Eigen::VectorXd::Zero(nZ);
    Eigen::VectorXd occupationsBeta = Eigen::VectorXd::Zero(nZ);

    for (int i = 0; i < shell->noCoreStates; ++i)
    {
        occupationsAlpha(i) = 1.0;
        occupationsBeta(i) = 1.0;
    }

    for (int i = 0; i < shell->noDegStates; ++i)
    {
        occupationsAlpha(shell->noCoreStates + i) = shell->averageOccupancyAlpha;
        occupationsBeta(shell->noCoreStates + i) = shell->averageOccupancyBeta;
    }

    const internal::JkSupermatrices super = internal::BuildJkSupermatrices(inputs.eri, nZ);
    const Eigen::MatrixXd coreHamiltonian = inputs.coreHamiltonian;

    const UhfFockBuilderFn buildFock = [super, coreHamiltonian](const Eigen::MatrixXd& dAlpha,
                                                                const Eigen::MatrixXd& dBeta) {
        return BuildUhfFock(super, dAlpha, dBeta, coreHamiltonian);
    };

    auto result = RunUhfLoop(*molecule,
                             inputs.overlap,
                             inputs.coreHamiltonian,
                             atomicOptions,
                             buildFock,
                             nullptr,
                             nullptr,
                             occupationsAlpha,
                             occupationsBeta,
                             nullptr);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    AtomicFragmentResult fragment;
    auto densityAlphaResult = BuildSpinDensity(result->summary.coefficientsAlpha, occupationsAlpha);

    if (!densityAlphaResult.has_value())
    {
        return std::unexpected(densityAlphaResult.error());
    }

    fragment.densityAlpha = *densityAlphaResult;
    auto densityBetaResult = BuildSpinDensity(result->summary.coefficientsBeta, occupationsBeta);

    if (!densityBetaResult.has_value())
    {
        return std::unexpected(densityBetaResult.error());
    }

    fragment.densityBeta = *densityBetaResult;
    fragment.coefficientsAlpha = std::move(result->summary.coefficientsAlpha);
    fragment.coefficientsBeta = std::move(result->summary.coefficientsBeta);
    fragment.nAlpha = occupationsAlpha.sum();
    fragment.nBeta = occupationsBeta.sum();
    return fragment;
}

// The number of basis functions of one element's shells, in the integrals
// module's layout (atoms in molecule order, element shells in file order,
// function count per shell = contraction rows x angular components).
std::size_t ElementFunctionCount(const qcx::basisset::ElementBasis& element) {
    std::size_t count = 0;

    for (const qcx::basisset::Shell& shell : element.shells)
    {
        const std::size_t angular = shell.isSpherical
                                        ? static_cast<std::size_t>(2 * shell.angularMomentum + 1)
                                        : static_cast<std::size_t>((shell.angularMomentum + 1) *
                                                                   (shell.angularMomentum + 2) / 2);
        count += shell.coefficients.size() * angular;
    }

    return count;
}

} // namespace

qcx::Result<UhfResult> RunUhfScf(const qcx::molecule::Molecule& molecule,
                                 const Eigen::MatrixXd& overlap,
                                 const Eigen::MatrixXd& coreHamiltonian,
                                 const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
                                 const UhfOptions& options) {
    const std::size_t n = static_cast<std::size_t>(overlap.rows());

    if (eri.Shape()[0] != n || eri.Shape()[1] != n || eri.Shape()[2] != n || eri.Shape()[3] != n)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the ERI tensor must share the function count with the one-electron matrices"});
    }

    const auto occupations = DeriveSpinOccupations(molecule);

    if (!occupations.has_value())
    {
        return std::unexpected(occupations.error());
    }

    const internal::JkSupermatrices super = internal::BuildJkSupermatrices(eri, n);

    // The lambda captures the core Hamiltonian by copy (a reference capture
    // here would dangle once this function's parameters go out of scope).
    const UhfFockBuilderFn buildFock = [super, coreHamiltonian](const Eigen::MatrixXd& dAlpha,
                                                                const Eigen::MatrixXd& dBeta) {
        return BuildUhfFock(super, dAlpha, dBeta, coreHamiltonian);
    };

    const Eigen::VectorXd occupationsAlpha =
        IntegerOccupations(static_cast<Eigen::Index>(n), occupations->first);
    const Eigen::VectorXd occupationsBeta =
        IntegerOccupations(static_cast<Eigen::Index>(n), occupations->second);

    auto result = RunUhfLoop(molecule,
                             overlap,
                             coreHamiltonian,
                             options,
                             buildFock,
                             nullptr,
                             nullptr,
                             occupationsAlpha,
                             occupationsBeta,
                             nullptr);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    return result->summary;
}

qcx::Result<UhfResult> RunUhfScf(const qcx::molecule::Molecule& molecule,
                                 const Eigen::MatrixXd& overlap,
                                 const Eigen::MatrixXd& coreHamiltonian,
                                 const UhfOptions& options,
                                 const UhfFockBuilderFn& fockBuilder,
                                 const qcx::basisset::BasisSet* basisSet) {
    if (!fockBuilder)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the UhfFockBuilderFn must be callable"});
    }

    const auto occupations = DeriveSpinOccupations(molecule);

    if (!occupations.has_value())
    {
        return std::unexpected(occupations.error());
    }

    const std::size_t n = static_cast<std::size_t>(overlap.rows());
    const Eigen::VectorXd occupationsAlpha =
        IntegerOccupations(static_cast<Eigen::Index>(n), occupations->first);
    const Eigen::VectorXd occupationsBeta =
        IntegerOccupations(static_cast<Eigen::Index>(n), occupations->second);

    auto result = RunUhfLoop(molecule,
                             overlap,
                             coreHamiltonian,
                             options,
                             fockBuilder,
                             nullptr,
                             nullptr,
                             occupationsAlpha,
                             occupationsBeta,
                             basisSet);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    return result->summary;
}

qcx::Result<UhfResult> RunUhfScf(const qcx::molecule::Molecule& molecule,
                                 const Eigen::MatrixXd& overlap,
                                 const Eigen::MatrixXd& coreHamiltonian,
                                 const UhfOptions& options,
                                 const UhfFockBuilderFn& fockBuilder,
                                 const CoulombFn& coulomb,
                                 const UhfEnergyContributionFn& energyContribution,
                                 const qcx::basisset::BasisSet* basisSet) {
    if (!fockBuilder)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the UhfFockBuilderFn must be callable"});
    }

    if (!coulomb || !energyContribution)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the CoulombFn and the UhfEnergyContributionFn must both "
                                          "be callable: the two-callback seam replaces the trace "
                                          "identity with Tr[D_total H] + 1/2 Tr[D_total J] plus "
                                          "the caller's contribution, so neither half alone "
                                          "defines an energy"});
    }

    const auto occupations = DeriveSpinOccupations(molecule);

    if (!occupations.has_value())
    {
        return std::unexpected(occupations.error());
    }

    const std::size_t n = static_cast<std::size_t>(overlap.rows());
    const Eigen::VectorXd occupationsAlpha =
        IntegerOccupations(static_cast<Eigen::Index>(n), occupations->first);
    const Eigen::VectorXd occupationsBeta =
        IntegerOccupations(static_cast<Eigen::Index>(n), occupations->second);

    auto result = RunUhfLoop(molecule,
                             overlap,
                             coreHamiltonian,
                             options,
                             fockBuilder,
                             &coulomb,
                             &energyContribution,
                             occupationsAlpha,
                             occupationsBeta,
                             basisSet);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    return result->summary;
}

qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> BuildGwhGuess(
    const Eigen::MatrixXd& overlap, const Eigen::MatrixXd& coreHamiltonian, int nAlpha, int nBeta) {
    const Eigen::Index n = overlap.rows();

    if (overlap.cols() != n || coreHamiltonian.rows() != n || coreHamiltonian.cols() != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "overlap and core Hamiltonian must share the function count"});
    }

    if (nAlpha < 0 || nBeta < 0 || nAlpha > n || nBeta > n)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "occupation counts must be within the function count"});
    }

    // The GWH guess Fock [Wolfsberg1952]: H on the diagonal, the
    // Wolfsberg-Helmholz off-diagonal 0.5 * K * (H_ii + H_jj) * S_ij with
    // the standard K = 1.75, diagonalized against the overlap.
    Eigen::MatrixXd guessFock = Eigen::MatrixXd::Zero(n, n);

    for (Eigen::Index i = 0; i < n; ++i)
    {
        for (Eigen::Index j = 0; j < n; ++j)
        {
            guessFock(i, j) = (i == j) ? coreHamiltonian(i, i)
                                       : 0.5 * kGwhCoefficient *
                                             (coreHamiltonian(i, i) + coreHamiltonian(j, j)) *
                                             overlap(i, j);
        }
    }

    auto orthogonalized = internal::OrthogonalizeOverlap(overlap);

    if (!orthogonalized.has_value())
    {
        return std::unexpected(orthogonalized.error());
    }

    const Eigen::MatrixXd coefficients = internal::DiagonalizeFock(guessFock, orthogonalized->x);
    // The occupation vector lives in the ORTHONORMAL dimension, which the
    // linear-dependence removal makes smaller than n (the guess diagonalizes
    // in the same reduced space the SCF runs in).
    const Eigen::Index guessDimension = static_cast<Eigen::Index>(orthogonalized->KeptDimension());
    auto densityAlpha = BuildSpinDensity(coefficients, IntegerOccupations(guessDimension, nAlpha));

    if (!densityAlpha.has_value())
    {
        return std::unexpected(densityAlpha.error());
    }

    auto densityBeta = BuildSpinDensity(coefficients, IntegerOccupations(guessDimension, nBeta));

    if (!densityBeta.has_value())
    {
        return std::unexpected(densityBeta.error());
    }

    return std::make_pair(std::move(*densityAlpha), std::move(*densityBeta));
}

qcx::Result<SadGuess> BuildSadGuess(const qcx::molecule::Molecule& molecule,
                                    const qcx::basisset::BasisSet& basisSet,
                                    const std::map<int, AtomicUhfInputs>& atomicInputs,
                                    const UhfOptions& atomicOptions) {
    const auto occupations = DeriveSpinOccupations(molecule);

    if (!occupations.has_value())
    {
        return std::unexpected(occupations.error());
    }

    // Per-atom function offsets in the integrals layout (atoms in molecule
    // order, each atom's element shells in file order).
    std::vector<std::size_t> offsets;
    offsets.reserve(molecule.AtomCount() + 1);
    offsets.push_back(0);

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        const qcx::basisset::ElementBasis* element = basisSet.Find(atom.atomicNumber);

        if (element == nullptr)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "basis set has no entry for element " +
                                                  std::to_string(atom.atomicNumber)});
        }

        offsets.push_back(offsets.back() + ElementFunctionCount(*element));
    }

    // One atomic fragment per DISTINCT element (never per atom instance -
    // a repeated element runs its fragment once), but the electron-count
    // totals sum every atom INSTANCE (each atom embeds the same fragment).
    std::map<int, AtomicFragmentResult> fragments;
    double nAlphaAtoms = 0.0;
    double nBetaAtoms = 0.0;

    for (const qcx::molecule::Atom& atom : molecule.Atoms())
    {
        if (fragments.find(atom.atomicNumber) == fragments.end())
        {
            const auto inputs = atomicInputs.find(atom.atomicNumber);

            if (inputs == atomicInputs.end())
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                                  "missing atomic integrals for element " +
                                                      std::to_string(atom.atomicNumber)});
            }

            auto fragment = RunAtomicFragment(atom.atomicNumber, inputs->second, atomicOptions);

            if (!fragment.has_value())
            {
                return std::unexpected(fragment.error());
            }

            fragments.emplace(atom.atomicNumber, std::move(*fragment));
        }

        nAlphaAtoms += fragments.at(atom.atomicNumber).nAlpha;
        nBetaAtoms += fragments.at(atom.atomicNumber).nBeta;
    }

    // The deliberate deviation from niedoida's InitialGuessFragments:
    // niedoida spin-UNpolarizes (both spins get the same spatial density,
    // scaled by the alpha/beta electron fractions); here the atomic spin
    // polarization is PRESERVED - each fragment's P_alpha/P_beta is scaled
    // by the molecular N_alpha/N_beta over the atomic totals, so open
    // shells start polarized.
    const double scaleAlpha = nAlphaAtoms > 0.0 ? occupations->first / nAlphaAtoms : 1.0;
    const double scaleBeta = nBetaAtoms > 0.0 ? occupations->second / nBetaAtoms : 1.0;

    const Eigen::Index eigenN = static_cast<Eigen::Index>(offsets.back());
    // The guess densities are touched before the zero fill so their pages
    // spread over the Fock-build team (the same pass as BuildSpinDensity);
    // the explicit setZero stays - the off-diagonal blocks must be zero and
    // the touch writes raw bytes, not values.
    Eigen::MatrixXd densityAlpha(eigenN, eigenN);
    Eigen::MatrixXd densityBeta(eigenN, eigenN);
    qcx::memory::TouchPagesAcrossTeam(
        densityAlpha.data(), static_cast<std::size_t>(densityAlpha.size()) * sizeof(double));
    qcx::memory::TouchPagesAcrossTeam(
        densityBeta.data(), static_cast<std::size_t>(densityBeta.size()) * sizeof(double));
    densityAlpha.setZero();
    densityBeta.setZero();
    Eigen::MatrixXd coefficientsAlpha = Eigen::MatrixXd::Zero(eigenN, eigenN);
    Eigen::MatrixXd coefficientsBeta = Eigen::MatrixXd::Zero(eigenN, eigenN);

    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        const qcx::molecule::Atom& atom = molecule.Atoms()[atomIndex];
        const AtomicFragmentResult& fragment = fragments.at(atom.atomicNumber);
        const Eigen::Index offset = static_cast<Eigen::Index>(offsets[atomIndex]);
        const Eigen::Index nZ =
            static_cast<Eigen::Index>(offsets[atomIndex + 1] - offsets[atomIndex]);

        densityAlpha.block(offset, offset, nZ, nZ) = scaleAlpha * fragment.densityAlpha;
        densityBeta.block(offset, offset, nZ, nZ) = scaleBeta * fragment.densityBeta;
        coefficientsAlpha.block(offset, offset, nZ, nZ) = fragment.coefficientsAlpha;
        coefficientsBeta.block(offset, offset, nZ, nZ) = fragment.coefficientsBeta;
    }

    return SadGuess{std::move(densityAlpha),
                    std::move(densityBeta),
                    std::move(coefficientsAlpha),
                    std::move(coefficientsBeta)};
}

} // namespace qcx::scf
