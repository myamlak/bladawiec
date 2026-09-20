// RHF SCF loop: supermatrix J/K through the linalg seam, Loewdin
// orthogonalization, and the Roothaan-Hall iteration with optional CDIIS.
// The named step functions below are the future FockBuilder<Derived> hooks
// of the loop's builder seam - see rhf.hpp.
#include "qcx/scf/rhf.hpp"

#include "internal/scf_common.hpp"
#include "internal/symmetry_blocks.hpp"
#include "qcx/linalg/dense_ops.hpp"
#include "qcx/memory/first_touch.hpp"
#include "qcx/molecule/mass_properties.hpp"
#include "qcx/scf/diis.hpp"
#include "qcx/scf/uhf.hpp"
#include "qcx/symmetry/detection.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <utility>

namespace qcx::scf {

namespace {

// Contracts the density with the supermatrices: J = E d and K = E_x d
// through the linalg seam (Eigen for small blocks, vendor BLAS above the
// threshold). dVec unpacks the density in the tensor's row-major
// order so its entries line up with the supermatrix columns.
qcx::Result<Eigen::MatrixXd> BuildFock(const internal::JkSupermatrices& super,
                                       // density then core Hamiltonian is the standard Fock-build
                                       // argument order (the linalg precedent in
                                       // iterative_eigensolver.hpp).
                                       // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                       const Eigen::MatrixXd& density,
                                       const Eigen::MatrixXd& coreHamiltonian) {
    const Eigen::Index n = density.rows();
    Eigen::MatrixXd dVec(n * n, 1);

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            dVec(mu * n + nu, 0) = density(mu, nu);
        }
    }

    const auto j = qcx::linalg::DenseMultiply(super.coulomb, dVec);

    if (!j.has_value())
    {
        return std::unexpected(j.error());
    }

    const auto k = qcx::linalg::DenseMultiply(super.exchange, dVec);

    if (!k.has_value())
    {
        return std::unexpected(k.error());
    }

    Eigen::MatrixXd fock = coreHamiltonian;

    for (Eigen::Index mu = 0; mu < n; ++mu)
    {
        for (Eigen::Index nu = 0; nu < n; ++nu)
        {
            // F = H + J - K/2: the 1/2 is the closed-shell exchange
            // prefactor of the Roothaan Fock matrix.
            const Eigen::Index flat = mu * n + nu;
            fock(mu, nu) += (*j)(flat, 0) - 0.5 * (*k)(flat, 0);
        }
    }

    return fock;
}

// The closed-shell density: 2 * C_occ C_occ^T over the numOccupied lowest
// orbitals. The product rides the linalg seam (Eigen below the threshold,
// vendor BLAS above): the density build is BLAS-shaped. Below
// the threshold the seam path is bit-identical to the direct product (the
// materialized transpose is a permutation and the scale-by-2 is a power of
// two, exact in IEEE).
qcx::Result<Eigen::MatrixXd> BuildDensity(const Eigen::MatrixXd& coefficients, int numOccupied) {
    // Allocate uninitialized and touch before the copy fills the block: an
    // anonymous page is placed on the NUMA node of the thread that first
    // accesses it, so the pre-fill pass across the OpenMP team keeps the
    // Fock-build team's density reads local on multi-socket machines. The
    // touch is a non-destructive write-back and the copy is unchanged, so
    // the values are exactly the pre-pass values.
    Eigen::MatrixXd density(coefficients.rows(), coefficients.rows());
    qcx::memory::TouchPagesAcrossTeam(density.data(),
                                      static_cast<std::size_t>(density.size()) * sizeof(double));
    const Eigen::MatrixXd occupied = coefficients.leftCols(numOccupied);
    const Eigen::MatrixXd occupiedTranspose = occupied.transpose();
    auto product = qcx::linalg::DenseMultiply(occupied, occupiedTranspose);

    if (!product.has_value())
    {
        return std::unexpected(product.error());
    }

    density = 2.0 * (*product);
    return density;
}

// The electronic energy of the current iterate: 1/2 Tr[D (H + F)].
double ComputeElectronicEnergy(const Eigen::MatrixXd& coreHamiltonian,
                               const Eigen::MatrixXd& fock,
                               const Eigen::MatrixXd& density) {
    return 0.5 * (density.cwiseProduct(coreHamiltonian + fock)).sum();
}

// The returned state's PROJECTOR residual: the Frobenius norm
// ||D - 2 C_occ C_occ^T|| between the returned density and the density the
// returned occupied MOs define. The check the converged-state contract needed
// and did not have; HfResult::stateConsistencyResidual carries what it means
// and why it is a defect signal rather than a tolerance.
// (density, coefficients) is the density-then-orbitals order of the residual.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ReturnedStateResidual(const Eigen::MatrixXd& density,
                             const Eigen::MatrixXd& coefficients,
                             int numOccupied) {
    const Eigen::MatrixXd occupied = coefficients.leftCols(numOccupied);
    return (density - 2.0 * occupied * occupied.transpose()).norm();
}

// The trace half of the two-callback path: Tr[D H] + 1/2 Tr[D J].
//
// The same identity as ComputeElectronicEnergy with the Fock split into the two
// operators the caller supplies. It is deliberately NOT the Hartree-Fock energy:
// the exchange belongs to the caller's contribution, because the trace identity
// that would recover it holds only for F = H + J - K/2 and a Kohn-Sham Fock has
// no such bilinear energy for its Vxc (Exc is a functional of D, not a form in
// it). Summed as two scalar traces rather than one matrix expression, which is
// what makes the totals agree with the Hartree-Fock path to rounding instead of
// bit for bit.
double ComputeCoulombTraceEnergy(const Eigen::MatrixXd& coreHamiltonian,
                                 const Eigen::MatrixXd& coulomb,
                                 const Eigen::MatrixXd& density) {
    return (density.cwiseProduct(coreHamiltonian)).sum() +
           0.5 * (density.cwiseProduct(coulomb)).sum();
}

// The CDIIS error vector [Pulay1980]: X^T (F D S - S D F) X in the
// orthogonal basis, so vectors from different iterations are comparable.
// The F D S contraction rides the linalg seam (BLAS-shaped): Eigen
// below the threshold - bit-identical to the direct product - vendor BLAS
// above. The X^T (...) X pair stays Eigen: it runs in the serial
// section and is the blocked diagonalization's concern.
qcx::Result<Eigen::MatrixXd> DiisError(const Eigen::MatrixXd& fock,
                                       const Eigen::MatrixXd& density,
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

// The exact factored form of the same commutator error (DiisErrorFactors'
// note): e = A B^T - B A^T, A = X^T F C_s, B = X^T S C_s, C_s = sqrt(2)
// C_occ, stored projected as the triple (B, R, s) with
//   M = (1/2) B^T A,  R = A - B M,  s = M - M^T,
// so that e = B s B^T + R B^T - B R^T as well and the extrapolator's inner
// products are formed at the error's own scale, never as a difference of the
// factors' O(1) contractions (the cancellation that made the direct factor
// traces worthless as the errors shrank - the 2026-09-15 landing). C_s is the
// density's own factor (BuildDensity forms D = 2 C_occ C_occ^T), so this
// takes the occupied block the loop already holds rather than the assembled
// density, and it builds two n^2 n_occ products where DiisError builds two n^3
// ones plus the same X^T (...) X pair. The projection is two n x n_occ by
// n_occ x n_occ products, and A is released once R is formed - the stored
// entry is 2 n n_occ doubles plus the n_occ x n_occ s block, against n^2 for
// the full square. The square they represent - the norm the floor gate and
// the trace read - comes back from DiisErrorFactors::Expand.
qcx::Result<DiisErrorFactors> DiisErrorFactorsOf(
    // (fock, coefficients) is the Fock-then-orbitals order.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& fock,
    const Eigen::MatrixXd& coefficients,
    int numOccupied,
    // fock/coefficients/overlap/x: the factored
    // contraction order, the C_s column count then
    // the X^T (...) pair.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& x) {
    const Eigen::MatrixXd cS = std::sqrt(2.0) * coefficients.leftCols(numOccupied);
    auto fc = qcx::linalg::DenseMultiply(fock, cS);

    if (!fc.has_value())
    {
        return std::unexpected(fc.error());
    }

    auto sc = qcx::linalg::DenseMultiply(overlap, cS);

    if (!sc.has_value())
    {
        return std::unexpected(sc.error());
    }

    const Eigen::MatrixXd a = x.transpose() * (*fc);
    const Eigen::MatrixXd b = x.transpose() * (*sc);

    // The projection, exact: A = B M + R because B^T B = 2 I (C_occ is
    // S-orthonormal).
    const Eigen::MatrixXd m = 0.5 * (b.transpose() * a);

    return DiisErrorFactors{b, a - b * m, m - m.transpose()};
}

// CDIIS history length: the standard 8-vector window.
constexpr std::size_t kDiisHistoryLimit = 8;

// The escape rung's virtual level shift: 1.0 Ha - the escape ladder's
// recorded middle value (uhf.hpp kLevelShiftDefault), provisional-measured
// like every other rung. No recorded case exercises the RHF rung yet
// (the C12H26 fix's primary mechanism is the DIIS engagement).
constexpr double kEscapeLevelShift = 1.0;

// The engagement fix in one sentence: the former CDIIS start gate
// (kDiisStartThreshold = 1e-1 on the commutator-error norm, "extrapolate
// only below it") is REMOVED. The measured C12H26 trajectory's norms sat
// at 33-38.5 - three orders of magnitude above the gate - so CDIIS never
// engaged on exactly the near-degenerate systems that need it (the
// measured verdict; pyscf's CDIIS extrapolates unconditionally from its second
// cycle). RHF now aligns with UHF, which deliberately runs no engagement
// threshold (uhf.cpp). The machine-precision FLOOR stays - the shared
// internal::kDiisFloorThreshold (scf_common.hpp): below it the error
// vectors are numerically degenerate (for H2/STO-3G the commutator
// F D S - S D F vanishes to rounding from the first iteration, because the
// symmetric-orbital density makes F D S exactly symmetric), and
// extrapolating the degenerate subspace only perturbs the converged
// density.
//
// The error enters the history in the FACTORED form whenever the caller
// holds the factor of the density that built this fock (`coefficients`) -
// every iteration but the seeded first one. The seeded iterate's density
// comes from the GWH guess or a checkpoint restart, neither of which the
// loop formed from its own coefficients, so that pair carries the square.
// The floor gate and the trace read the NORM OF THE SQUARE either way
// (the factored form expands to it), so kDiisFloorThreshold and the reported
// diiserr= keep the operand they have always had; the square's own arithmetic
// is the projected expansion, which carries the error's scale and not the
// factors' (DiisErrorFactors' note).

// Runs the CDIIS extrapolation when engaged: appends the current
// (fock, commutator-error) pair to the history and replaces fock with the
// extrapolated matrix once the two-pair minimum is reached - the
// engagement gate on the error norm is gone (see the engagement
// comment block above), so the extrapolation starts at iteration 2 of
// every DIIS run whose errors stay above the floor. The pair result
// carries the per-iteration DIIS status for the trace (ScfDiisTraceStatus:
// floor skipped / appended / extrapolated / error norm / coefficients) -
// the C12H26 defect was SILENT DIIS (the trace's diis= tokens make any
// inertness observable per iteration). Returns fock unchanged when DIIS is
// off, the floor skipped the pair, or the two-pair minimum is not reached;
// the extrapolator's own failures propagate through qcx::Result. The floor
// check runs BEFORE the append - the discipline the UHF path already
// has: a numerically degenerate error vector must never enter the history,
// because the minimum-norm least-squares solve can then bias the
// extrapolation toward that stale fock (the P = 0 start's error is exactly
// zero, so an unguarded append plants a zero pair that the next
// extrapolations lean on until it ages out of the window).
qcx::Result<std::pair<Eigen::MatrixXd, internal::ScfDiisTraceStatus>> MaybeExtrapolateFock(
    const RhfOptions& options,
    DiisExtrapolator& diis,
    Eigen::MatrixXd fock,
    // (density, coefficients) is the density-then-orbitals order.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& coefficients,
    int numOccupied,
    bool densityFactorInHand,
    // density/coefficients/overlap/x: the error builders' own argument order,
    // carried here unchanged (the factored builder takes the occupied block
    // and the square builder the assembled density).
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const Eigen::MatrixXd& overlap,
    const Eigen::MatrixXd& x) {
    internal::ScfDiisTraceStatus status;

    if (!options.useDiis)
    {
        // status.useDiis stays false: DIIS-off runs keep the six-field
        // trace line shape (no diis= suffix, ScfTraceWriter::Write).
        return std::make_pair(std::move(fock), std::move(status));
    }

    status.useDiis = true;

    // The iterate's error in whichever exact form the loop can build, and the
    // square the gate and the trace read - the factored form's Expand, or the
    // commutator itself. Both carry the same ||e||_F up to the arithmetic of
    // the two expansions (the identity regroups the same products; the forms
    // differ only in their order), and both are taken at the error's own
    // scale, which is what the floor gate needs.
    std::optional<DiisErrorFactors> factors;
    Eigen::MatrixXd error;

    if (densityFactorInHand)
    {
        const auto built = DiisErrorFactorsOf(fock, coefficients, numOccupied, overlap, x);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        factors = *built;
        error = factors->Expand();

    } else
    {
        const auto built = DiisError(fock, density, overlap, x);

        if (!built.has_value())
        {
            return std::unexpected(built.error());
        }

        error = *built;
    }

    const double diisErrorNorm = error.norm();
    status.errorNorm = diisErrorNorm;

    if (diisErrorNorm <= internal::kDiisFloorThreshold)
    {
        status.floorSkipped = true;
        return std::make_pair(std::move(fock), std::move(status));
    }

    if (factors.has_value())
    {
        diis.Append(fock, *factors);
    } else
    {
        diis.Append(fock, error);
    }

    status.appended = true;

    if (!diis.Ready())
    {
        // One pair in the history (iteration 1): the extrapolation needs
        // the two-pair minimum; the next append enables it.
        return std::make_pair(std::move(fock), std::move(status));
    }

    const auto extrapolated = diis.ExtrapolateWithCoefficients();

    if (!extrapolated.has_value())
    {
        return std::unexpected(extrapolated.error());
    }

    status.extrapolated = true;
    const Eigen::VectorXd& weights = extrapolated->second;
    status.coefficients.assign(weights.data(), weights.data() + weights.size());

    return std::make_pair(extrapolated->first, std::move(status));
}

// The Roothaan-Hall convergence gate: both the energy and the density must
// settle - an energy-only check can declare convergence while the wave
// function is still drifting (the classic false-convergence of accelerated
// SCF). CDIIS is an accelerator, not a convergence criterion - the density
// change gates both paths (the commutator-error criterion is degenerate
// exactly where it would matter: for H2/STO-3G the commutator
// F D S - S D F vanishes to rounding from the first iteration, see the
// floor discussion in the MaybeExtrapolateFock comment). The RMS is the
// Frobenius norm over the
// n x n density matrix, / n (not / n^2 - the n^2 divisor made the gate
// ~n x too lenient and the dense HF/H2O pins converged ~1e-5 off; those
// pins are the regression). The energy leg is LIVE: the call site holds
// the bookkeeping back until after this call (see RunRhfScf), so the
// comparison reads the previous iterate's energy as intended. It was
// vacuous from the loop's first version until the deferral was closed
// - previousTotalEnergy was assigned before the gate ran, so the leg read
// the fresh energy against itself and energy_tolerance was inert at every
// value. The UHF path always had the correct ordering (uhf.cpp's
// IsUhfConverged call site).
bool IsConverged(int iteration,
                 double totalEnergy,
                 double previousTotalEnergy,
                 double rmsDensityChange,
                 const RhfOptions& options) {
    return iteration > 0 &&
           std::fabs(totalEnergy - previousTotalEnergy) < options.energyTolerance &&
           rmsDensityChange < options.densityTolerance;
}

// Runs the Roothaan-Hall iteration with a caller-provided Fock builder (the
// FockBuilderFn seam of rhf.hpp). The supermatrix overload passes a lambda
// over the prebuilt J/K supermatrices; the direct/RI Fock builders of
// qcx-integrals enter through the same path, since scf cannot link integrals
// (module DAG). Everything except the Fock build - orthogonalization,
// diagonalization, density, DIIS extrapolation, convergence gating - is the
// named-step pipeline, so both overloads share it bit for bit.
qcx::Result<HfResult> RunScfLoop(const qcx::molecule::Molecule& molecule,
                                 const Eigen::MatrixXd& overlap,
                                 const Eigen::MatrixXd& coreHamiltonian,
                                 const RhfOptions& options,
                                 const FockBuilderFn& buildFock,
                                 const CoulombFn* coulomb,
                                 const EnergyContributionFn* energyContribution,
                                 const qcx::basisset::BasisSet* basisSet) {
    if (options.maxIterations <= 0 || options.energyTolerance <= 0.0 ||
        options.densityTolerance <= 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "maxIterations and the tolerances must be positive"});
    }

    // The per-iteration trace side-channel (the diagnostics writer): opened once
    // at loop start, one line per iteration after its convergence decision
    // (scf_common.hpp ScfTraceWriter). The restart re-seeds from the
    // average of the last two densities (the 2-cycle members, see the
    // restart below) - the writer stays open across it, so the primary's
    // lines survive and the wall clock is continuous.
    internal::ScfTraceWriter trace(options.traceFile);
    internal::ScfDensityDumpWriter densityDump(options.densityDumpFile);

    const int electronCount = molecule.ElectronCount();

    if (electronCount % 2 != 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented, "only closed-shell RHF is supported today"});
    }

    const std::size_t n = static_cast<std::size_t>(overlap.rows());

    if (static_cast<std::size_t>(overlap.cols()) != n ||
        static_cast<std::size_t>(coreHamiltonian.rows()) != n ||
        static_cast<std::size_t>(coreHamiltonian.cols()) != n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "overlap and core Hamiltonian must share the function count"});
    }

    const int numOccupied = electronCount / 2;
    auto orthogonalized = internal::OrthogonalizeOverlap(overlap);

    if (!orthogonalized.has_value())
    {
        return std::unexpected(orthogonalized.error());
    }

    // The linear-dependence removal (scf_common.hpp OrthogonalizeOverlap) is
    // DISCLOSED, never silent: a system whose diffuse basis carries
    // near-dependent directions runs in the numKept-dimensional orthonormal
    // space and maps back - D_AO = X D_orth X^T is n x n, the full AO
    // dimension, so nothing is neglected. The one genuine error is a system
    // that needs more occupied orbitals than the removal left directions
    // for; the threshold itself is a system property, not an error.
    if (static_cast<int>(orthogonalized->KeptDimension()) < numOccupied)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the overlap carries fewer usable directions (" +
                std::to_string(orthogonalized->KeptDimension()) + " after removing " +
                std::to_string(orthogonalized->numRemoved) + " near-dependent ones) than the " +
                std::to_string(numOccupied) + " occupied orbitals this system needs"});
    }

    const Eigen::MatrixXd x = orthogonalized->x;
    const std::size_t numRemovedOverlap = orthogonalized->numRemoved;

    // Symmetry-blocked diagonalization: when the caller supplies the
    // AO basis, detect the point group once and build the AO-space
    // decomposition once; every iteration then diagonalizes the irrep
    // blocks separately (equivalent to the plain n x n solve up to
    // rounding). C1 molecules and unrealizable groups fall back to the
    // plain path; an inconsistent detection is a real error and propagates.
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

    // The blocked/plain diagonalization choice, hoisted so the
    // finalizer re-diagonalizes the returned density's Fock with EXACTLY the
    // transform the walk used: two call sites, one rule. The
    // columns come back ascending by orbital energy - SelfAdjointEigenSolver
    // orders its eigenvalues, and the blocked path preserves that order.
    const auto diagonalizeFock = [&](const Eigen::MatrixXd& fock) {
        if (symmetryBlocks.has_value())
        {
            return internal::DiagonalizeFockBlocked(
                fock, x, *symmetryBlocks, *blockedDiagonalizeData);
        }

        return qcx::Result<Eigen::MatrixXd>(internal::DiagonalizeFock(fock, x));
    };

    // The full-group labeling stage (labels and the symmetrized density;
    // nothing computed in the loop changes). C1 molecules, the missing-basis
    // runs, the off toggle, and unrealizable groups pass through unlabeled; a
    // realized inconsistency is a real error and propagates.
    //
    // Hoisted out of the trajectory and parameterized by the Fock because the
    // finalizer must re-derive the labels AFTER it rebinds the returned
    // coefficients: SymmetryLabels snapshots the coefficients it was given
    // (symmetry_blocks.cpp), so labels computed against the trajectory's
    // coefficients would describe a different MO set than the one the result
    // returns. The trajectory still labels (the seam/KS path returns through
    // it unchanged and never reaches the finalizer); the finalizer overwrites
    // the labels with the returned-density Fock's own.
    const auto applyLabeling = [&](HfResult& result,
                                   const Eigen::MatrixXd& fock) -> qcx::Result<void> {
        if (basisSet == nullptr || !options.fullGroupLabeling || !analysis.has_value() ||
            analysis->group == qcx::symmetry::PointGroupName::kC1)
        {
            return {};
        }

        // The linear-dependence removal DISENGAGES this stage, and it
        // disengages it by name rather than by failure: the stage's contract
        // is over the n-dimensional MO set (one label per coefficient column,
        // the full-group average of the n x n density), while the removal
        // leaves the run in a numKept-dimensional orthonormal space whose
        // coefficients are n x numKept. The blocked diagonalization beside it
        // falls back to the plain n-dimensional solve for the same reason
        // (X is rectangular, so U_b^T X U_b has no square per-irrep block);
        // this is that fallback's counterpart on the labeling side.
        //
        // The disclosure is the removal's own count in the run record
        // (numRemovedOverlapDirections -> num_removed_overlap_directions),
        // which is nonzero exactly when this branch is taken - so the run
        // record says the stage did not run instead of leaving a reader to
        // read its absence as one of the three named non-causes.
        if (numRemovedOverlap != 0)
        {
            return {};
        }

        const auto labeled = internal::SymmetryLabelAndSymmetrize(
            fock, result.coefficients, result.density, numOccupied, molecule, *basisSet, *analysis);

        if (!labeled.has_value())
        {
            if (labeled.error().code == qcx::ErrorCode::kUnimplemented)
            {
                return {};
            }

            return std::unexpected(labeled.error());
        }

        result.symmetryLabels = *labeled;
        return {};
    };

    const Eigen::Index eigenN = static_cast<Eigen::Index>(n);

    // Starting guess: the loop starts from the GWH guess (uhf.hpp
    // BuildGwhGuess) instead of the zero matrix. The
    // zero start is a known-bad seed on diffuse bases - the Roothaan map
    // from P = 0 locks a non-converging 2-cycle (H2O/aug-cc-pVDZ oscillates
    // between -93.23 and -58.38 Eh against the true -76.041844; O2 records
    // the same lesson), and it feeds the DIIS history an exactly-zero
    // error vector at iteration 0 that biases the first extrapolations (see
    // the floor guard in MaybeExtrapolateFock). From the GWH seed the same
    // iteration converges to the true fixed point (H2O/aug-cc-pVDZ: 16
    // iterations with CDIIS).
    // Sized but uninitialized, touched before the seed assignment below so
    // the iteration-0 pages spread over the Fock-build team (see
    // BuildDensity); both assignments are same-size and fill the existing
    // buffer.
    Eigen::MatrixXd density(eigenN, eigenN);
    qcx::memory::TouchPagesAcrossTeam(density.data(),
                                      static_cast<std::size_t>(density.size()) * sizeof(double));
    Eigen::MatrixXd previousDensity;

    // The two older densities of the failed-run lag: the
    // generalized period-k restart seed needs one full period of member
    // densities, so lags 2 and 3 (periods 3 and 4) live in this two-slot
    // move-ring beside previousDensity (lag 1); period 2 uses only
    // previousDensity alone (see the seeding
    // lambda below). The ring shifts with previousDensity every iteration.
    std::array<Eigen::MatrixXd, 2> olderDensity;

    // Restart seeding: a non-empty initial state replaces the
    // default GWH start with the checkpoint's density, restores the DIIS
    // history, and seeds the convergence gate's previous energies. Empty
    // fields are no-ops, so default-constructed options take the GWH start.
    const Eigen::MatrixXd& seed = options.initialScfState.density;

    if (seed.size() != 0 && (seed.rows() != eigenN || seed.cols() != eigenN))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the initial SCF density must match the function count"});
    }

    if (seed.size() != 0)
    {
        density = seed;
    } else
    {
        auto gwh = BuildGwhGuess(overlap, coreHamiltonian, numOccupied, numOccupied);

        if (!gwh.has_value())
        {
            return std::unexpected(gwh.error());
        }

        // Closed shell: D = dAlpha + dBeta, the two Aufbau occupations of
        // the same GWH Fock (BuildGwhGuess returns the pair of spin
        // densities).
        density = gwh->first + gwh->second;
    }

    // The last-iterate state captured into the result: the loop
    // computes these on every pass and previously dropped them at the
    // return - hoisted so the non-converged path returns the final iterate
    // too (rhf.hpp's HfResult comment).
    Eigen::MatrixXd coefficients;
    Eigen::MatrixXd finalFock;
    DiisExtrapolator diis(kDiisHistoryLimit);

    if (const auto restore = diis.Restore(options.initialScfState.diis); !restore.has_value())
    {
        return std::unexpected(restore.error());
    }

    double previousTotalEnergy = options.initialScfState.previousTotalEnergy;
    double previousElectronicEnergy = options.initialScfState.previousElectronicEnergy;

    // The Roothaan trajectory as a callable: the stationarity
    // detector lives inside the loop as a read-only energy observer - it
    // only sets the detectedPeriod out-parameter (0 = nothing fired), so a
    // trajectory runs unperturbed to its natural exit (the no-rescue output
    // is bit-identical to the no-restart behavior). applyLevelShift turns the
    // escape rung on: the virtual level shift is applied to the fock
    // after the DIIS extrapolation and before the diagonalization (the
    // shared placement - the shifted matrix never enters the energy path, so the
    // converged state is the unshifted fixed point). The seed-fallback
    // orchestration below re-invokes this same trajectory at the failed
    // exits.
    const auto runTrajectory = [&](bool applyLevelShift,
                                   int& detectedPeriod) -> qcx::Result<HfResult> {
        // The detector's rolling window of the most recent total energies,
        // oldest at index 0: one slot is shifted out per iteration, so once
        // the window is full every slot holds a real energy.
        std::array<double, internal::kTwoCycleWindow> recentTotalEnergies{};
        std::size_t recentCount = 0;

        // The full-group labeling stage (the hoisted applyLabeling above; the
        // finalizer re-derives it from the returned-density Fock after
        // the rebind, so this pass stands for the seam/KS path).

        // The gate's own operands of the LAST executed iteration, hoisted so
        // the non-converged return (the budget exit below) can record them
        // too - HfResult::achievedEnergyDelta / achievedRmsDensityChange, the
        // disclosure that qualifies a bare `converged` (the rhf.hpp note).
        double achievedEnergyDelta = 0.0;
        double achievedRmsDensityChange = 0.0;

        for (int iteration = 0; iteration < options.maxIterations; ++iteration)
        {
            const auto fockResult = buildFock(density);

            if (!fockResult.has_value())
            {
                return std::unexpected(fockResult.error());
            }

            Eigen::MatrixXd fock = *fockResult;

            // The C12H26 discriminating-experiment side-channel: record
            // the (density, physical-Fock) pair in hand - the density
            // this Fock was built from, BEFORE any DIIS extrapolation -
            // plus the overlap and core Hamiltonian once. The offline
            // invariant and variational-energy analysis lives in
            // tools/amf_density_invariants.py; the dump is write-only and
            // the empty path never opens (the bit-parity pins are
            // absolute).
            if (iteration == 0)
            {
                densityDump.WriteOverlap(overlap);
                densityDump.WriteCoreHamiltonian(coreHamiltonian);
            }

            densityDump.WritePair(static_cast<std::uint64_t>(iteration), density, *fockResult);

            // From iteration 1 on, `density` is BuildDensity(coefficients) of
            // the previous pass and the occupied block is the density's own
            // factor; iteration 0's density is the seed (the GWH guess or a
            // checkpoint restart), which only the square can describe. The
            // `iteration > 0` guard is the loop's own structure, not the
            // state of `coefficients`: a re-invoked trajectory starts again
            // at iteration 0 over a fresh seed, and the previous pass's
            // coefficients must not be read as that seed's factor.
            const auto extrapolation = MaybeExtrapolateFock(
                options, diis, fock, density, coefficients, numOccupied, iteration > 0, overlap, x);

            if (!extrapolation.has_value())
            {
                return std::unexpected(extrapolation.error());
            }

            fock = extrapolation->first;
            finalFock = fock;

            // The escape rung's level shift, applied only on the
            // escape trajectory (applyLevelShift; the orchestration below
            // launches it after a detected cycle AND a failed restart):
            // F += (S·C)·B·(S·C)^T over the virtuals, after the DIIS
            // extrapolation, before the diagonalization, skipped at
            // iteration 0, using the previous iteration's coefficients -
            // the shared placement (uhf.cpp's comment, shared seam). finalFock
            // stays the UNSHIFTED extrapolated matrix and the energy below
            // is computed from it, so the shift never enters the energy
            // path: the converged state is the unshifted fixed point.
            if (applyLevelShift && iteration > 0)
            {
                internal::ApplyVirtualSpaceLevelShift(
                    fock, overlap, coefficients, kEscapeLevelShift, numOccupied);
            }

            const auto diagonalized = diagonalizeFock(fock);

            if (!diagonalized.has_value())
            {
                return std::unexpected(diagonalized.error());
            }

            coefficients = *diagonalized;

            auto newDensityResult = BuildDensity(coefficients, numOccupied);

            if (!newDensityResult.has_value())
            {
                return std::unexpected(newDensityResult.error());
            }

            Eigen::MatrixXd& newDensity = *newDensityResult;

            // The iterate's electronic energy. Two paths, selected by whether
            // the caller supplied the energy seam: the Hartree-Fock trace
            // identity every existing caller uses, or the Coulomb trace plus
            // the caller's contribution, which is what a Kohn-Sham run needs
            // because no trace of the Fock can recover Exc.
            double electronicEnergy = 0.0;

            if (coulomb == nullptr)
            {
                electronicEnergy = ComputeElectronicEnergy(coreHamiltonian, finalFock, newDensity);
            } else
            {
                auto coulombMatrix = (*coulomb)(newDensity);

                if (!coulombMatrix.has_value())
                {
                    return std::unexpected(coulombMatrix.error());
                }

                auto contributed = (*energyContribution)(newDensity);

                if (!contributed.has_value())
                {
                    return std::unexpected(contributed.error());
                }

                electronicEnergy =
                    ComputeCoulombTraceEnergy(coreHamiltonian, *coulombMatrix, newDensity) +
                    *contributed;
            }

            const double totalEnergy =
                electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);

            const double rmsDensityChange = (newDensity - density).norm() / static_cast<double>(n);

            // The trace's energy delta: the TRUE iteration-to-iteration
            // change, and the same expression the gate's energy leg below
            // evaluates (captured before the bookkeeping overwrites
            // previousTotalEnergy, so both read the previous iterate).
            const double energyDelta = std::fabs(totalEnergy - previousTotalEnergy);

            // The previous iterates' densities: the fallback
            // seed needs one full period of a failed run's member densities
            // - period 2 uses previousDensity only, periods 3 and 4 the
            // two-slot ring (lags 2 and 3) - and the exit moves the last
            // one into the result. The ring shifts one slot per iteration.
            olderDensity[1] = std::move(olderDensity[0]);
            olderDensity[0] = std::move(previousDensity);
            previousDensity = std::move(density);
            density = std::move(newDensity);

            // The stationarity detector hook: a read-only energy
            // observer. Shift the window, then check once the transients
            // are past (kTwoCycleMinIterations) and only until a period has
            // fired (detectedPeriod 2-4; the period-2 test is the
            // detector's k = 2 branch, bit-identical, higher lags behind
            // the same flag).
            for (std::size_t j = 0; j + 1 < internal::kTwoCycleWindow; ++j)
            {
                recentTotalEnergies[j] = recentTotalEnergies[j + 1];
            }

            recentTotalEnergies[internal::kTwoCycleWindow - 1] = totalEnergy;
            ++recentCount;

            if (detectedPeriod == 0 && recentCount >= internal::kTwoCycleWindow &&
                iteration >= internal::kTwoCycleMinIterations)
            {
                internal::DetectStationaryCycle(recentTotalEnergies,
                                                internal::kTwoCycleParityTolerance,
                                                internal::kTwoCycleAlternationFloor,
                                                detectedPeriod);
            }

            // Evaluated into a named vector BEFORE the moves below: an Eigen
            // expression passed straight into the aggregate-init would be
            // evaluated lazily during construction, after std::move(coefficients)
            // has emptied it (the expression-template trap of the aggregate
            // move; UHF's FinishSummary has the same discipline).
            const Eigen::VectorXd orbitalEnergies =
                (coefficients.transpose() * finalFock * coefficients).diagonal();

            const bool converged =
                IsConverged(iteration, totalEnergy, previousTotalEnergy, rmsDensityChange, options);

            achievedEnergyDelta = energyDelta;
            achievedRmsDensityChange = rmsDensityChange;

            // The energy bookkeeping, AFTER the gate. The comparison above
            // needs the PREVIOUS iterate's energy for its |E_n - E_{n-1}|
            // leg; with the assignment before the call it read the fresh
            // energy against itself, so every positive energyTolerance
            // passed and density_tolerance alone decided an RHF stop
            // (energy_tolerance was inert - documented, measured, and
            // deferred). Moving it here closes that deferral.
            // Nothing below reads either variable as an INPUT: the
            // converged branch and the budget exit both report
            // totalEnergy / electronicEnergy and seed the restart from the
            // same pair, so only the GATE's operands change.
            previousTotalEnergy = totalEnergy;
            previousElectronicEnergy = electronicEnergy;

            // The trace line lands AFTER the convergence decision with the
            // exact numbers that decision used (write-only: nothing reads
            // it back, so the trajectory is untouched). The DIIS status of
            // the same iteration rides along (diis=floor|ext|wait
            // plus the error norm, the coefficients, and the
            // monitor-only negMass/parityImb measures when extrapolated).
            trace.Write(iteration + 1,
                        totalEnergy,
                        energyDelta,
                        rmsDensityChange,
                        converged,
                        extrapolation->second);

            if (converged)
            {
                HfResult result{totalEnergy,
                                electronicEnergy,
                                true,
                                iteration + 1,
                                std::move(density),
                                std::move(coefficients),
                                orbitalEnergies};
                result.achievedEnergyDelta = achievedEnergyDelta;
                result.achievedRmsDensityChange = achievedRmsDensityChange;
                result.numRemovedOverlapDirections = numRemovedOverlap;
                result.restart = ScfRestartState{result.density,
                                                 {},
                                                 {},
                                                 diis.Snapshot(),
                                                 {},
                                                 {},
                                                 previousTotalEnergy,
                                                 previousElectronicEnergy};

                if (const auto labeled = applyLabeling(result, finalFock); !labeled.has_value())
                {
                    return std::unexpected(labeled.error());
                }

                return result;
            }
        }

        const Eigen::VectorXd orbitalEnergies =
            (coefficients.transpose() * finalFock * coefficients).diagonal();

        HfResult result{previousTotalEnergy,
                        previousElectronicEnergy,
                        false,
                        options.maxIterations,
                        std::move(density),
                        std::move(coefficients),
                        orbitalEnergies};
        result.achievedEnergyDelta = achievedEnergyDelta;
        result.achievedRmsDensityChange = achievedRmsDensityChange;
        result.numRemovedOverlapDirections = numRemovedOverlap;
        result.restart = ScfRestartState{result.density,
                                         {},
                                         {},
                                         diis.Snapshot(),
                                         {},
                                         {},
                                         previousTotalEnergy,
                                         previousElectronicEnergy};

        if (const auto labeled = applyLabeling(result, finalFock); !labeled.has_value())
        {
            return std::unexpected(labeled.error());
        }

        return result;
    };

    // The density-consistent reporting contract. The energy the
    // loop forms each iteration is the trace of the DIIS-EXTRAPOLATED Fock at
    // the new density, 1/2 Tr[D_n (H + F[D-bar])], and the extrapolated
    // density combination D-bar is not D_n; the reported total energy
    // therefore belongs to a density the run does not return. The mismatch is
    // a FIRST-ORDER quantity (delta = 1/2 Tr[D (G[D-bar] - G[D])]), not a
    // convergence residual, and it is route-dependent - two equivalent routes
    // agreeing to 1e-9 on the converged energy stop at different iteration
    // counts and report energies apart by ~1e-5, which makes it a
    // silent-wrong-answer defect rather than a convention difference.
    //
    // The correction is applied ONCE, here, to the FINAL result of the run:
    // rebuild the physical Fock of the returned density and re-form the trace
    // from that pair - one extra Fock build per run, never per iteration. The
    // trajectory, the iteration counts, the gate legs and the gate's recorded
    // operands (achievedEnergyDelta / achievedRmsDensityChange) are untouched:
    // the reported numbers change, the walk does not. The escape rungs are
    // finalized through the orchestration below on the result that STANDS, so
    // a restarted run pays one build for that rung, not one per rung
    // attempted.
    //
    // The two-callback (seam/KS) path is EXCLUDED: its per-iteration energy
    // already reads newDensity through the ComputeCoulombTraceEnergy branch
    // (the seam twin that density_energy_test and seam_test assert), so there
    // is nothing to correct there - and its remaining gap (its returned C and
    // eps are the DIIS-extrapolated Fock's eigenpairs, not the returned
    // density's Fock's) needs the iterate-until-consistent form the nonlinear
    // Vxc leg requires; scoped out.
    //
    // The restart state is refreshed with the same pair. ScfRestartState
    // documents those two fields as the GATE's state (scf_state.hpp): the next
    // run reads them back as its previous-iterate energy, and the density the
    // checkpoint carries is the returned one - leaving the trajectory's trace
    // energy beside that density would re-inject this very defect at the
    // restart seam, as a spurious |E_1 - E_0| on the restarted run's first
    // energy leg.
    //
    // The orbital spectrum follows the energy (the owner's ruling of
    // 2026-09-13): the returned MO coefficients and orbital energies are the
    // returned-density Fock's eigenpairs. The trajectory's coefficients are
    // eigenvectors of the EXTRAPOLATED Fock, so reporting them beside a
    // density-consistent total energy left the two routes that now agree on
    // the energy still printing different HOMO/LUMO orderings - the same
    // defect class, one field over. This is a CONTRACT change, not a re-pin:
    // the field's documented meaning moves with the code (rhf.hpp's
    // orbitalEnergies brief).
    //
    // The converged-state contract closes the other half of the same class:
    // was the trajectory's - one SCF step away from the returned coefficients -
    // so the result handed out an idempotent-by-construction density only when
    // the trajectory happened to stop at its own fixed point (it does not: the
    // DIIS step moves it), and a linear combination of idempotent densities is
    // not idempotent. The exit now builds the density FROM the coefficients it
    // returns. Cost: one extra Fock build per run, never per iteration, and
    // only on the paths that reach this finalizer.
    const auto reportReturnedDensityEnergy = [&](HfResult& result) -> qcx::Result<void> {
        // The projector leg is measured on EVERY path, the seam/KS carve-out
        // below included: a field left at its default would report a
        // consistency the run never checked (a run record that can
        // silently differ from what ran), and the carve-out returns the loop's
        // own pair, whose projector residual is a real (if zero) measurement.
        result.stateConsistencyResidual =
            ReturnedStateResidual(result.density, result.coefficients, numOccupied);

        if (coulomb != nullptr)
        {
            return {};
        }

        const auto returnedFock = buildFock(result.density);

        if (!returnedFock.has_value())
        {
            return std::unexpected(returnedFock.error());
        }

        // The orbital spectrum is RE-DERIVED rather than carried over (the
        // owner's ruling of 2026-09-13): the returned MO set is the density
        // estimate's Fock's eigenpairs, not the extrapolated Fock's the loop
        // diagonalized per iterate. Re-diagonalizing - rather than taking
        // diag(C^T F[D] C) in the old basis - is what keeps the result
        // self-describing: SelfAdjointEigenSolver returns ASCENDING eigenvalues
        // with matching eigenvectors, so the documented "columns ascending by
        // orbital energy" holds by construction and no re-sort is needed.
        const auto finalCoefficients = diagonalizeFock(*returnedFock);

        if (!finalCoefficients.has_value())
        {
            return std::unexpected(finalCoefficients.error());
        }

        // The converged-state contract: the returned density is the
        // returned occupied MOs' OWN density, D = 2 C_occ C_occ^T - not the
        // trajectory density the Fock above was built from. The two are one SCF
        // step apart (the trajectory's density came out of the DIIS-EXTRAPOLATED
        // Fock's eigenvectors; this Fock is the physical one), which is exactly
        // the gap the O2 triplet pin caught: ||D - 2 C_occ C_occ^T|| = 3.4e-9
        // (alpha) and 1.3e-6 (beta) on a run that reported converged. No
        // convergence criterion bounds it - the density gate compares two
        // TRAJECTORY iterates - and a linear combination of idempotent
        // densities is generally not idempotent, so the extrapolated density is
        // not a valid one to hand out.
        const auto finalDensity = BuildDensity(*finalCoefficients, numOccupied);

        if (!finalDensity.has_value())
        {
            return std::unexpected(finalDensity.error());
        }

        // One more Fock build, on the RETURNED density: the pair the returned
        // energy belongs to. It is deliberately NOT the source of the returned
        // orbital energies - those stay the eigenpairs of the density
        // estimate's Fock above, which C diagonalizes exactly, while C is not
        // an eigenbasis of this matrix. That is the one leg of the contract
        // left at the fixed-point residual (F[D_returned] - F[D_seed] =
        // G[D_returned - D_seed]: F is affine in D, so the mismatch is linear
        // in the residual and bounded by it - T2's HF one-pass exactness).
        const auto returnedDensityFock = buildFock(*finalDensity);

        if (!returnedDensityFock.has_value())
        {
            return std::unexpected(returnedDensityFock.error());
        }

        result.stateConsistencyResidual =
            ReturnedStateResidual(*finalDensity, *finalCoefficients, numOccupied);
        result.stateFixedPointResidual = (*finalDensity - result.density).norm();
        result.density = *finalDensity;
        result.coefficients = *finalCoefficients;
        result.orbitalEnergies =
            (result.coefficients.transpose() * (*returnedFock) * result.coefficients).diagonal();
        result.electronicEnergy =
            ComputeElectronicEnergy(coreHamiltonian, *returnedDensityFock, result.density);
        result.totalEnergy =
            result.electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);

        // The labels move with the coefficients: SymmetryLabels snapshots the
        // MO set and the density it was handed, so leaving the trajectory pass
        // in place would describe a different set than the one this result
        // returns. Re-derived against F[D_seed] - the matrix the returned
        // coefficients are eigenvectors of, with the returned density beside
        // them - rather than the extrapolated Fock. The trajectory pass already
        // ran on the seam/KS path, which never reaches here.
        if (const auto labeled = applyLabeling(result, *returnedFock); !labeled.has_value())
        {
            return std::unexpected(labeled.error());
        }

        // The checkpoint carries the returned state: ScfRestartState documents
        // these as the density the run stands on and the gate's own energies,
        // and a restart a density apart from the result would hand the next run
        // a first energy leg comparing unlike objects.
        result.restart.density = result.density;
        result.restart.previousTotalEnergy = result.totalEnergy;
        result.restart.previousElectronicEnergy = result.electronicEnergy;

        if (internal::ReturnDiagnosticsEnabled())
        {
            std::fprintf(
                stderr,
                "scf-return: e=%.17g iters=%d naset=%zu consistency=%.6g fixedpoint=%.6g\n",
                result.totalEnergy,
                result.iterations,
                result.numRemovedOverlapDirections,
                result.stateConsistencyResidual,
                result.stateFixedPointResidual);
        }

        return {};
    };

    // The cycle-members seed: the average of one full period of
    // the failed trajectory's final member densities (the density ring as
    // left by that run's last shift; lastDensity = the run's final density,
    // moved into its result). Period 2 uses previousDensity alone - the
    // measured basin escape (the cycle's energies average to within ~0.6 Eh
    // of the psi4 value and the restart converges in 22-42 iterations
    // on the four measured records); periods 3 and 4 generalize the average
    // over the ring's lags 2 and 3 (1/3 and 1/4 of the member sum).
    const auto cycleMembersSeed = [&](const Eigen::MatrixXd& lastDensity,
                                      int period) -> Eigen::MatrixXd {
        if (period == 3)
        {
            return (1.0 / 3.0) * (lastDensity + previousDensity + olderDensity[0]);
        }

        if (period >= 4)
        {
            return 0.25 * (lastDensity + previousDensity + olderDensity[0] + olderDensity[1]);
        }

        return 0.5 * (lastDensity + previousDensity);
    };

    // The escape ladder. The stationarity detector fired during
    // the primary run, which exited NOT converged. Rung 2 - exactly one
    // restart from the detected cycle's member densities (the measured
    // basin escape) follows; rung 3 - if that restart fails too AND the run
    // uses DIIS (a DIIS-steered trajectory that still cycles = DIIS failed
    // to escape; useDiis=false runs keep the exactly-one-restart
    // semantics - the synthetic re-entry test's contract), one level-shifted
    // escape trajectory (kEscapeLevelShift) from the restart's own cycle
    // members. The caller's initialScfState is NOT re-applied (already
    // consumed); the DIIS history and the previous energies are reset
    // before each rung so its extrapolations and gate start clean; the
    // orthogonalization X and the symmetry blocks are density-
    // independent and reused.
    int detectedPeriod = 0;
    auto primary = runTrajectory(false, detectedPeriod);

    if (!primary.has_value())
    {
        return std::unexpected(primary.error());
    }

    if (primary->converged || detectedPeriod == 0)
    {
        // Rung 1 stands: no cycle was detected (the plain-map
        // result, bit-identical), or the primary converged.
        if (const auto corrected = reportReturnedDensityEnergy(*primary); !corrected.has_value())
        {
            return std::unexpected(corrected.error());
        }

        return primary;
    }

    // The restart seed: the average of the last detectedPeriod primary
    // densities, i.e. of the cycle's members. Measured: the plain
    // Roothaan map LOCKS the same 2-cycle on these records (the
    // "RHF-from-core converges" inference was drawn from the UHF-from-zero
    // run; the RHF single-chain CDIIS and the UHF per-spin two-phase
    // extrapolate different subspaces, and the plain map 2-cycles from both
    // zero and GWH here), while the members average sits in the fixed
    // point's basin: the restart converges in 22-42 iterations on all
    // four records.
    density = cycleMembersSeed(primary->density, detectedPeriod);
    diis.Reset();
    previousTotalEnergy = 0.0;
    previousElectronicEnergy = 0.0;
    int restartPeriod = 0;
    auto restart = runTrajectory(false, restartPeriod);

    if (!restart.has_value())
    {
        return std::unexpected(restart.error());
    }

    if (restart->converged)
    {
        // The rescue is observable only as converged=true; the iteration
        // count is primary spent + restart spent (the primary reports the
        // burned budget - the restart-only count under-reports).
        restart->iterations += primary->iterations;

        if (const auto corrected = reportReturnedDensityEnergy(*restart); !corrected.has_value())
        {
            return std::unexpected(corrected.error());
        }

        return restart;
    }

    if (options.useDiis)
    {
        // Rung 3: the level-shifted escape. Seed from the restart's
        // own final members (its detected period, else the generic period-2
        // pair); the trajectory applies kEscapeLevelShift on the virtuals
        // from its iteration 1 on.
        diis.Reset();
        previousTotalEnergy = 0.0;
        previousElectronicEnergy = 0.0;
        const int escapeSeedPeriod = (restartPeriod != 0) ? restartPeriod : 2;
        density = cycleMembersSeed(restart->density, escapeSeedPeriod);
        int escapePeriod = 0;
        auto escape = runTrajectory(true, escapePeriod);

        if (!escape.has_value())
        {
            return std::unexpected(escape.error());
        }

        if (escape->converged)
        {
            // The full burned budget: primary + restart + escape (each
            // trajectory reports its own spent iterations).
            escape->iterations += restart->iterations + primary->iterations;

            if (const auto corrected = reportReturnedDensityEnergy(*escape); !corrected.has_value())
            {
                return std::unexpected(corrected.error());
            }

            return escape;
        }
    }

    // Every rung failed (seeds are basin-dependent; e.g. P = 0 is a
    // known-bad seed on other records): report the ORIGINAL
    // failure result, so the no-rescue output stays bit-identical to the
    // the pre-restart behavior.
    if (const auto corrected = reportReturnedDensityEnergy(*primary); !corrected.has_value())
    {
        return std::unexpected(corrected.error());
    }

    return primary;
}

} // namespace

qcx::Result<HfResult> RunRhfScf(const qcx::molecule::Molecule& molecule,
                                const Eigen::MatrixXd& overlap,
                                const Eigen::MatrixXd& coreHamiltonian,
                                const qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>& eri,
                                const RhfOptions& options) {
    const std::size_t n = static_cast<std::size_t>(overlap.rows());

    if (eri.Shape()[0] != n || eri.Shape()[1] != n || eri.Shape()[2] != n || eri.Shape()[3] != n)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the ERI tensor must share the function count with the one-electron matrices"});
    }

    const internal::JkSupermatrices super = internal::BuildJkSupermatrices(eri, n);

    // The supermatrix Fock build as a FockBuilderFn: the J/K supermatrices
    // are density-independent and built once, so the callback only contracts
    // the current density (BuildFock keeps the standard Fock-build argument
    // order, density then core Hamiltonian).
    const FockBuilderFn buildFock = [super, coreHamiltonian](const Eigen::MatrixXd& density) {
        return BuildFock(super, density, coreHamiltonian);
    };

    return RunScfLoop(
        molecule, overlap, coreHamiltonian, options, buildFock, nullptr, nullptr, nullptr);
}

qcx::Result<HfResult> RunRhfScf(const qcx::molecule::Molecule& molecule,
                                const Eigen::MatrixXd& overlap,
                                const Eigen::MatrixXd& coreHamiltonian,
                                const RhfOptions& options,
                                const FockBuilderFn& fockBuilder,
                                const qcx::basisset::BasisSet* basisSet) {
    if (!fockBuilder)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the FockBuilderFn must be callable"});
    }

    return RunScfLoop(
        molecule, overlap, coreHamiltonian, options, fockBuilder, nullptr, nullptr, basisSet);
}

qcx::Result<HfResult> RunRhfScf(const qcx::molecule::Molecule& molecule,
                                const Eigen::MatrixXd& overlap,
                                const Eigen::MatrixXd& coreHamiltonian,
                                const RhfOptions& options,
                                const FockBuilderFn& fockBuilder,
                                const CoulombFn& coulomb,
                                const EnergyContributionFn& energyContribution,
                                const qcx::basisset::BasisSet* basisSet) {
    if (!fockBuilder)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the FockBuilderFn must be callable"});
    }

    if (!coulomb || !energyContribution)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the CoulombFn and the EnergyContributionFn must both be "
                                          "callable: the two-callback seam replaces the trace "
                                          "identity with Tr[D H] + 1/2 Tr[D J] plus the caller's "
                                          "contribution, so neither half alone defines an energy"});
    }

    return RunScfLoop(molecule,
                      overlap,
                      coreHamiltonian,
                      options,
                      fockBuilder,
                      &coulomb,
                      &energyContribution,
                      basisSet);
}

} // namespace qcx::scf
