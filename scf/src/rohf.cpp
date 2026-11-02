// Restricted open-shell SCF loop (ROHF/ROKS): one spatial orbital set with a
// closed block and an open block, solved through the Roothaan effective Fock
// over the shared step-functions of internal/scf_common.hpp.
//
// The loop consumes the per-spin seam unchanged (UhfFockBuilderFn): the pair
// it hands the builder is the UHF pair (D_alpha, D_beta), and the dictionary
// that makes the pair restricted is D_c = 2 D_beta, D_o = D_alpha - D_beta.
// The two physical spin Focks do not share an eigenbasis, so neither one is
// diagonalized: the effective Fock is assembled from the pair and the
// closed/open projectors, diagonalized, and filled by the fixed Roothaan
// occupation rule (lowest nC orbitals closed, next nO open). The energy is
// 1/2 sum_sigma Tr[D_sigma (H + F_sigma)] - algebraically the ROHF energy in
// the closed/open decomposition - or, on the two-callback path, the
// Kohn-Sham form Tr[D H] + 1/2 Tr[D J[D]] + contribution.
#include "qcx/scf/rohf.hpp"

#include "internal/rohf_fock.hpp"
#include "internal/scf_common.hpp"
#include "qcx/linalg/dense_ops.hpp"
#include "qcx/memory/first_touch.hpp"
#include "qcx/molecule/mass_properties.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

namespace qcx::scf {

namespace {

// The CDIIS window of the loop's single channel: the UHF loop's 8-pair
// bound (uhf.cpp's kDiisHistoryLimit), for its reason - one coefficient per
// stored pair, and the stored errors approach linear dependence as the run
// converges, so the window is bounded rather than extended.
constexpr std::size_t kRohfDiisHistoryLimit = 8;

// Derives the restricted open-shell occupations from the electron count N
// and the multiplicity M: nO = M - 1 open orbitals (one unpaired electron
// each), nC = (N - nO)/2 closed ones (two each), nAlpha = nC + nO and
// nBeta = nC. Rejects an inconsistent pair the way UHF's DeriveSpinOccupations
// does (molecule creation checks only multiplicity >= 1).
qcx::Result<std::pair<int, int>> DeriveRestrictedOccupations(
    const qcx::molecule::Molecule& molecule) {
    const int electronCount = molecule.ElectronCount();
    const int numOpen = molecule.Multiplicity() - 1;

    if (electronCount < numOpen || (electronCount - numOpen) % 2 != 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "electron count and multiplicity are inconsistent"});
    }

    const int numClosed = (electronCount - numOpen) / 2;
    return std::make_pair(numClosed + numOpen, numClosed); // (nAlpha, nBeta)
}

// The per-spin density of one set of orbitals: C diag(occupations) C^T, NO
// factor of 2 (the UhfFockBuilderFn convention). The product rides the
// linalg seam exactly as uhf.cpp's BuildSpinDensity does; the touch keeps
// the Fock-build team's reads local on multi-socket machines.
qcx::Result<Eigen::MatrixXd> BuildSpinDensity(const Eigen::MatrixXd& coefficients,
                                              const Eigen::VectorXd& occupations) {
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

// The integer occupation vector of one channel: 1 on the lowest numOccupied
// orbitals of the kept space, 0 elsewhere.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::VectorXd IntegerOccupations(Eigen::Index dimension, int numOccupied) {
    Eigen::VectorXd occupations = Eigen::VectorXd::Zero(dimension);
    occupations.head(numOccupied).setOnes();
    return occupations;
}

// The solver-space image X^T S D S X of one AO density: the density's
// representation in the orthonormal basis, and the projector of a channel
// when the channel's density is its own projector.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd SolverSpaceProjector(const Eigen::MatrixXd& density,
                                     const Eigen::MatrixXd& overlap,
                                     const Eigen::MatrixXd& x) {
    return x.transpose() * (overlap * density * overlap) * x;
}

// The closed/open projectors of the loop's own orbital set: the solver-space
// coefficient matrix's closed and open column blocks are orthonormal, so
// their Gram products ARE the projectors (and cost n n_occ^2 instead of the
// n^3 image of the densities). Empty blocks are handled explicitly - the
// closed-shell and high-spin limits set one block to zero columns.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd BlockProjector(const Eigen::MatrixXd& solverCoefficients,
                               Eigen::Index firstColumn,
                               Eigen::Index numColumns) {
    Eigen::MatrixXd projector(solverCoefficients.rows(), solverCoefficients.rows());
    projector.setZero();

    if (numColumns > 0)
    {
        const Eigen::MatrixXd block = solverCoefficients.middleCols(firstColumn, numColumns);
        projector.noalias() = block * block.transpose();
    }

    return projector;
}

// The effective Fock's eigenpairs: eigenvectors in the SOLVER space (the
// columns of the returned matrix), eigenvalues ascending. The AO
// coefficients are X times them; keeping the solver-space set lets the
// error factors and the projectors use the orthonormal objects directly.
Eigen::MatrixXd SolveEffectiveFock(const Eigen::MatrixXd& effectiveFock) {
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(effectiveFock);
    return solver.eigenvectors();
}

// The exact factored CDIIS error of the effective Fock, in the solver space:
// e = A B^T - B A^T with A = F_eff C_s and B = C_s, where the scaled factor
// C_s = [sqrt(2) C_closed, C_open] is the density's own factor (D_orth =
// C_s C_s^T with occupations 2 and 1). The per-column weights enter the
// projection (internal::ProjectDiisErrorFactors); with a uniform sqrt(2)
// this is the RHF loop's factored form.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
qcx::Result<DiisErrorFactors> BuildEffectiveErrorFactors(const Eigen::MatrixXd& effectiveFock,
                                                         const Eigen::MatrixXd& solverCoefficients,
                                                         int numClosed,
                                                         int numOpen) {
    const int numOccupied = numClosed + numOpen;
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(numOccupied);
    weights.head(numClosed).setConstant(std::sqrt(2.0));
    const Eigen::MatrixXd scaled = solverCoefficients.leftCols(numOccupied) * weights.asDiagonal();
    const Eigen::MatrixXd a = effectiveFock * scaled;
    return internal::ProjectDiisErrorFactors(a, scaled, weights);
}

// The electronic energy of the iterate: 1/2 sum_sigma Tr[D_sigma (H + F_sigma)],
// the UhfResult::electronicEnergy expression at the physical spin Focks. It is
// algebraically the ROHF energy in the closed/open decomposition; the
// derivation is on ComputeUhfElectronicEnergy in uhf.cpp.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ComputeRohfElectronicEnergy(const Eigen::MatrixXd& coreHamiltonian,
                                   const Eigen::MatrixXd& fockAlpha,
                                   const Eigen::MatrixXd& fockBeta,
                                   const Eigen::MatrixXd& dAlpha,
                                   const Eigen::MatrixXd& dBeta) {
    return 0.5 * ((dAlpha.cwiseProduct(coreHamiltonian + fockAlpha)).sum() +
                  (dBeta.cwiseProduct(coreHamiltonian + fockBeta)).sum());
}

// The trace half of the two-callback (Kohn-Sham) path:
// Tr[D_total H] + 1/2 Tr[D_total J], the same expression the UHF loop forms
// (ComputeUhfCoulombTraceEnergy's derivation): the exchange, or Exc, is the
// caller's contribution.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ComputeRohfCoulombTraceEnergy(const Eigen::MatrixXd& coreHamiltonian,
                                     const Eigen::MatrixXd& coulomb,
                                     const Eigen::MatrixXd& totalDensity) {
    return (totalDensity.cwiseProduct(coreHamiltonian)).sum() +
           0.5 * (totalDensity.cwiseProduct(coulomb)).sum();
}

// The returned state's PROJECTOR residual of one channel: the Frobenius norm
// between the returned density and the density its own occupied orbitals
// define. Zero by construction on the restricted pair (both channels share
// one coefficient set), so a nonzero value is a defect signal.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ReturnedStateResidual(const Eigen::MatrixXd& density,
                             const Eigen::MatrixXd& coefficients,
                             const Eigen::VectorXd& occupations) {
    Eigen::Index numOccupied = 0;

    for (Eigen::Index i = 0; i < occupations.size(); ++i)
    {
        if (occupations(i) > 0.5)
        {
            ++numOccupied;
        }
    }

    const Eigen::MatrixXd occupied = coefficients.leftCols(numOccupied);
    return (density - occupied * occupied.transpose()).norm();
}

// The <S^2> diagnostic of a restricted determinant: S(S+1) + nBeta -
// sum_ij |<psi_a,i|psi_b,j>|^2 with the beta occupied set a subset of the
// alpha one, so the overlap sum is nBeta and the value is S(S+1) with
// S = nO/2 = (nAlpha - nBeta)/2. Computed from the returned orbitals rather
// than asserted from the occupation counts, so an inconsistent returned set
// shows up in the number.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ComputeRestrictedSpinSquared(const Eigen::MatrixXd& overlap,
                                    const Eigen::MatrixXd& coefficients,
                                    int nAlpha,
                                    int nBeta) {
    const double s = 0.5 * static_cast<double>(nAlpha - nBeta);
    const Eigen::MatrixXd occupiedAlpha = coefficients.leftCols(nAlpha);
    const Eigen::MatrixXd occupiedBeta = coefficients.leftCols(nBeta);
    const Eigen::MatrixXd overlapBlock = occupiedAlpha.transpose() * overlap * occupiedBeta;
    return s * (s + 1.0) + static_cast<double>(nBeta) - overlapBlock.squaredNorm();
}

// Runs the restricted open-shell iteration. The public overloads derive the
// occupations and delegate; coulomb/energyContribution are the optional
// Kohn-Sham energy seam (both null selects the Hartree-Fock trace identity).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
qcx::Result<RohfResult> RunRohfLoop(const qcx::molecule::Molecule& molecule,
                                    const Eigen::MatrixXd& overlap,
                                    const Eigen::MatrixXd& coreHamiltonian,
                                    const RohfOptions& options,
                                    const UhfFockBuilderFn& buildFock,
                                    const CoulombFn* coulomb,
                                    const UhfEnergyContributionFn* energyContribution,
                                    int nAlpha,
                                    int nBeta) {
    if (options.maxIterations <= 0 || options.energyTolerance <= 0.0 ||
        options.densityTolerance <= 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "maxIterations and the tolerances must be positive"});
    }

    if (options.levelShiftOpen < 0.0 || options.levelShiftVirtual < 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the level-shift values must be non-negative"});
    }

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

    const Eigen::Index eigenN = static_cast<Eigen::Index>(n);

    if (nAlpha > static_cast<int>(n) || nBeta < 0 || nAlpha < nBeta)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "more occupied orbitals than basis functions in the requested spin state"});
    }

    if ((options.initialDensityAlpha.size() != 0 &&
         (options.initialDensityAlpha.rows() != eigenN ||
          options.initialDensityAlpha.cols() != eigenN)) ||
        (options.initialDensityBeta.size() != 0 && (options.initialDensityBeta.rows() != eigenN ||
                                                    options.initialDensityBeta.cols() != eigenN)))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "initial densities must match the function count"});
    }

    Eigen::MatrixXd dAlpha = options.initialDensityAlpha.size() == 0
                                 ? Eigen::MatrixXd::Zero(eigenN, eigenN)
                                 : options.initialDensityAlpha;
    Eigen::MatrixXd dBeta = options.initialDensityBeta.size() == 0
                                ? Eigen::MatrixXd::Zero(eigenN, eigenN)
                                : options.initialDensityBeta;

    // Restart seeding: the checkpoint's pair replaces the initial guess and
    // the gate's previous energies are seeded. Empty fields are no-ops.
    if ((options.initialScfState.densityAlpha.size() != 0 &&
         (options.initialScfState.densityAlpha.rows() != eigenN ||
          options.initialScfState.densityAlpha.cols() != eigenN)) ||
        (options.initialScfState.densityBeta.size() != 0 &&
         (options.initialScfState.densityBeta.rows() != eigenN ||
          options.initialScfState.densityBeta.cols() != eigenN)))
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

    // The linear-dependence removal is a SYSTEM property and is disclosed,
    // never silent (scf_common.hpp OrthogonalizeOverlap): the run happens in
    // the numKept-dimensional orthonormal space and maps back. The one
    // genuine error is needing more occupied orbitals than it left
    // directions for - the alpha block is the larger one.
    if (static_cast<int>(orthogonalized->KeptDimension()) < nAlpha)
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
    const Eigen::Index numKept = static_cast<Eigen::Index>(orthogonalized->KeptDimension());

    // The occupation vectors of the two channels, in the kept space.
    const Eigen::VectorXd occupationsAlpha = IntegerOccupations(numKept, nAlpha);
    const Eigen::VectorXd occupationsBeta = IntegerOccupations(numKept, nBeta);
    const int numOpen = nAlpha - nBeta;

    // The single CDIIS channel on the effective Fock, with the restart state's
    // single history (ScfRestartState::diis - this loop has one channel; the
    // per-spin fields are unused).
    DiisExtrapolator diis(kRohfDiisHistoryLimit);

    if (const auto restored = diis.Restore(options.initialScfState.diis); !restored.has_value())
    {
        return std::unexpected(restored.error());
    }

    // The last diagonalization's eigenvectors (solver space) and the AO
    // coefficients, kept because the next iteration's projectors and DIIS
    // error factors are read from the state that produced the current
    // densities.
    Eigen::MatrixXd solverCoefficients;
    Eigen::MatrixXd coefficients;
    Eigen::MatrixXd finalEffectiveFock;
    bool haveCoefficients = false;

    double previousTotalEnergy = options.initialScfState.previousTotalEnergy;
    double previousElectronicEnergy = options.initialScfState.previousElectronicEnergy;
    double achievedEnergyDelta = 0.0;
    double achievedRmsDensityChange = 0.0;

    // The density-consistent reporting contract, the restricted open-shell
    // form of uhf.cpp's reportReturnedDensityEnergy. The loop's energy is the
    // trace of the physical Focks built from the density the diagonalization
    // started from, while the returned state's energy belongs to the RETURNED
    // density; the exit rebuilds the pair's Focks once and re-forms the trace,
    // and rebuilds the returned state the ROHF way - assemble the effective
    // Fock from the returned pair, diagonalize it, and rebuild the pair from
    // the ONE coefficient set (never per spin: that would leave the restricted
    // manifold). The trajectory, the iteration count and the gate's recorded
    // operands are untouched. The two-callback (Kohn-Sham) path is excluded,
    // exactly as on the UHF path: it already evaluates at the returned
    // density, and its remaining gap needs the iterate-until-consistent form
    // the nonlinear Vxc leg requires.
    const auto reportReturnedDensityEnergy = [&](RohfResult& result) -> qcx::Result<void> {
        result.stateConsistencyResidualAlpha =
            ReturnedStateResidual(result.densityAlpha, result.coefficientsAlpha, occupationsAlpha);
        result.stateConsistencyResidualBeta =
            ReturnedStateResidual(result.densityBeta, result.coefficientsBeta, occupationsBeta);

        if (coulomb != nullptr)
        {
            return {};
        }

        const auto returnedFocks = buildFock(result.densityAlpha, result.densityBeta);

        if (!returnedFocks.has_value())
        {
            return std::unexpected(returnedFocks.error());
        }

        // The projectors from the returned PAIR (not from the coefficients):
        // the exit measures the state it was handed, and the pair is the
        // object the assembler consumes.
        const Eigen::MatrixXd closedProjector =
            SolverSpaceProjector(result.densityBeta, overlap, x);
        const Eigen::MatrixXd openProjector =
            SolverSpaceProjector(result.densityAlpha - result.densityBeta, overlap, x);
        const Eigen::MatrixXd effective =
            internal::AssembleRoothaanFock(x.transpose() * returnedFocks->first * x,
                                           x.transpose() * returnedFocks->second * x,
                                           closedProjector,
                                           openProjector);

        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(effective);

        if (solver.info() != Eigen::Success)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the effective Fock could not be diagonalized"});
        }

        const Eigen::MatrixXd returnedSolverCoefficients = solver.eigenvectors();
        const Eigen::MatrixXd returnedCoefficients = x * returnedSolverCoefficients;

        const auto newDensityAlpha = BuildSpinDensity(returnedCoefficients, occupationsAlpha);

        if (!newDensityAlpha.has_value())
        {
            return std::unexpected(newDensityAlpha.error());
        }

        const auto newDensityBeta = BuildSpinDensity(returnedCoefficients, occupationsBeta);

        if (!newDensityBeta.has_value())
        {
            return std::unexpected(newDensityBeta.error());
        }

        // One more Fock build, on the RETURNED densities: the pair the
        // returned energy belongs to. The returned coefficients remain the
        // eigenvectors of F_eff[D_seed], one SCF step from these matrices -
        // the fixed-point residual below measures that gap.
        const auto finalFocks = buildFock(*newDensityAlpha, *newDensityBeta);

        if (!finalFocks.has_value())
        {
            return std::unexpected(finalFocks.error());
        }

        result.stateConsistencyResidualAlpha =
            ReturnedStateResidual(*newDensityAlpha, returnedCoefficients, occupationsAlpha);
        result.stateConsistencyResidualBeta =
            ReturnedStateResidual(*newDensityBeta, returnedCoefficients, occupationsBeta);
        result.stateFixedPointResidualAlpha = (*newDensityAlpha - result.densityAlpha).norm();
        result.stateFixedPointResidualBeta = (*newDensityBeta - result.densityBeta).norm();
        result.densityAlpha = *newDensityAlpha;
        result.densityBeta = *newDensityBeta;
        result.coefficientsAlpha = returnedCoefficients;
        result.coefficientsBeta = returnedCoefficients;
        // The effective Fock's spectrum, both channels: the loop diagonalized
        // ONE matrix, and its eigenvalues ascend by construction.
        result.orbitalEnergiesAlpha = solver.eigenvalues();
        result.orbitalEnergiesBeta = solver.eigenvalues();
        result.electronicEnergy = ComputeRohfElectronicEnergy(coreHamiltonian,
                                                              finalFocks->first,
                                                              finalFocks->second,
                                                              result.densityAlpha,
                                                              result.densityBeta);
        result.totalEnergy =
            result.electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);

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

    // The shared summary tail: the returned state and the restart record.
    const auto finish = [&](bool converged, int iterations) -> qcx::Result<RohfResult> {
        RohfResult summary;
        summary.totalEnergy = previousTotalEnergy;
        summary.electronicEnergy = previousElectronicEnergy;
        summary.converged = converged;
        summary.iterations = iterations;
        summary.spinSquared = ComputeRestrictedSpinSquared(overlap, coefficients, nAlpha, nBeta);
        summary.achievedEnergyDelta = achievedEnergyDelta;
        summary.achievedRmsDensityChange = achievedRmsDensityChange;
        summary.numRemovedOverlapDirections = numRemovedOverlap;
        summary.densityAlpha = dAlpha;
        summary.densityBeta = dBeta;
        summary.coefficientsAlpha = coefficients;
        summary.coefficientsBeta = coefficients;
        // The effective Fock's diagonal in the returned orbital basis: a
        // spectrum on every run without a level shift, and the documented
        // carve-out of a shifted run (the same one the UHF result's orbital
        // energies carry).
        const Eigen::VectorXd orbitalEnergies =
            (solverCoefficients.transpose() * finalEffectiveFock * solverCoefficients).diagonal();
        summary.orbitalEnergiesAlpha = orbitalEnergies;
        summary.orbitalEnergiesBeta = orbitalEnergies;
        summary.restart.density = dAlpha + dBeta;
        summary.restart.densityAlpha = dAlpha;
        summary.restart.densityBeta = dBeta;
        summary.restart.diis = diis.Snapshot();
        summary.restart.previousTotalEnergy = previousTotalEnergy;
        summary.restart.previousElectronicEnergy = previousElectronicEnergy;

        if (const auto corrected = reportReturnedDensityEnergy(summary); !corrected.has_value())
        {
            return std::unexpected(corrected.error());
        }

        // The checkpoint carries the returned state and the gate's own
        // energies, exactly as the UHF exit refreshes them.
        summary.restart.density = summary.densityAlpha + summary.densityBeta;
        summary.restart.densityAlpha = summary.densityAlpha;
        summary.restart.densityBeta = summary.densityBeta;
        summary.restart.previousTotalEnergy = summary.totalEnergy;
        summary.restart.previousElectronicEnergy = summary.electronicEnergy;
        return summary;
    };

    for (int iteration = 0; iteration < options.maxIterations; ++iteration)
    {
        const auto fockResult = buildFock(dAlpha, dBeta);

        if (!fockResult.has_value())
        {
            return std::unexpected(fockResult.error());
        }

        // The physical spin Focks of the current pair, in the solver space.
        const Eigen::MatrixXd fockAlphaOrth = x.transpose() * fockResult->first * x;
        const Eigen::MatrixXd fockBetaOrth = x.transpose() * fockResult->second * x;

        // The closed/open projectors: from the diagonalization that produced
        // the current densities when there is one (the coefficient blocks are
        // orthonormal, so their Grams ARE the projectors), and from the seed
        // pair itself on the first iterate, where the seed may be a guess
        // density rather than a restricted determinant.
        Eigen::MatrixXd closedProjector;
        Eigen::MatrixXd openProjector;

        if (haveCoefficients)
        {
            closedProjector = BlockProjector(solverCoefficients, 0, nBeta);
            openProjector = BlockProjector(solverCoefficients, nBeta, numOpen);
        } else
        {
            closedProjector = SolverSpaceProjector(dBeta, overlap, x);
            openProjector = SolverSpaceProjector(dAlpha - dBeta, overlap, x);
        }

        const Eigen::MatrixXd effective = internal::AssembleRoothaanFock(
            fockAlphaOrth, fockBetaOrth, closedProjector, openProjector);

        // One DIIS channel: the error of the effective Fock against the
        // density's own factor. The factored form is used whenever the
        // coefficients that built the current densities are held (every
        // iterate but a seeded first one); the seeded iterate carries the
        // square, whose factor the loop cannot reconstruct exactly.
        double diisErrorNorm = 0.0;
        bool appended = false;

        if (haveCoefficients)
        {
            const auto factors =
                BuildEffectiveErrorFactors(effective, solverCoefficients, nBeta, numOpen);

            if (!factors.has_value())
            {
                return std::unexpected(factors.error());
            }

            diisErrorNorm = factors->Expand().norm();

            if (options.useDiis && diisErrorNorm > internal::kDiisFloorThreshold)
            {
                diis.Append(effective, *factors);
                appended = true;
            }
        } else
        {
            const Eigen::MatrixXd densityOrth = SolverSpaceProjector(dAlpha + dBeta, overlap, x);
            const Eigen::MatrixXd error = effective * densityOrth - densityOrth * effective;
            diisErrorNorm = error.norm();

            if (options.useDiis && diisErrorNorm > internal::kDiisFloorThreshold)
            {
                diis.Append(effective, error);
                appended = true;
            }
        }

        // The extrapolated effective Fock is the one diagonalized, once the
        // history holds at least two pairs.
        Eigen::MatrixXd diagonalizationFock = effective;

        if (appended && diis.Ready())
        {
            const auto extrapolated = diis.Extrapolate();

            if (!extrapolated.has_value())
            {
                return std::unexpected(extrapolated.error());
            }

            diagonalizationFock = *extrapolated;
        }

        // The unshifted effective Fock of this iterate: the result's orbital
        // spectrum and the DIIS history's matrix. Captured BEFORE the level
        // shift, which is outside the energy path.
        finalEffectiveFock = diagonalizationFock;

        if (options.useLevelShift && haveCoefficients)
        {
            internal::ApplyOpenShellLevelShift(diagonalizationFock,
                                               closedProjector,
                                               openProjector,
                                               options.levelShiftOpen,
                                               options.levelShiftVirtual);
        }

        const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(diagonalizationFock);

        if (solver.info() != Eigen::Success)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the effective Fock could not be diagonalized"});
        }

        solverCoefficients = solver.eigenvectors();
        coefficients = x * solverCoefficients;
        haveCoefficients = true;

        const auto newDensityAlpha = BuildSpinDensity(coefficients, occupationsAlpha);

        if (!newDensityAlpha.has_value())
        {
            return std::unexpected(newDensityAlpha.error());
        }

        const auto newDensityBeta = BuildSpinDensity(coefficients, occupationsBeta);

        if (!newDensityBeta.has_value())
        {
            return std::unexpected(newDensityBeta.error());
        }

        // The energy reads the physical spin Focks built from the density the
        // diagonalization started from (the UHF loop's convention: its
        // extrapolated Focks are the same one-step objects). The exit re-forms
        // it at the returned densities.
        double electronicEnergy = 0.0;

        if (coulomb == nullptr)
        {
            electronicEnergy = ComputeRohfElectronicEnergy(coreHamiltonian,
                                                           fockResult->first,
                                                           fockResult->second,
                                                           *newDensityAlpha,
                                                           *newDensityBeta);
        } else
        {
            const Eigen::MatrixXd totalDensity = *newDensityAlpha + *newDensityBeta;
            auto coulombMatrix = (*coulomb)(totalDensity);

            if (!coulombMatrix.has_value())
            {
                return std::unexpected(coulombMatrix.error());
            }

            auto contributed = (*energyContribution)(*newDensityAlpha, *newDensityBeta);

            if (!contributed.has_value())
            {
                return std::unexpected(contributed.error());
            }

            electronicEnergy =
                ComputeRohfCoulombTraceEnergy(coreHamiltonian, *coulombMatrix, totalDensity) +
                *contributed;
        }

        const double totalEnergy =
            electronicEnergy + qcx::molecule::NuclearRepulsionEnergy(molecule);

        // The convergence measure is the ALPHA density, not the per-spin
        // pair. The closed/open split is not a unique object when the
        // effective Fock's last closed and first open orbital are degenerate
        // in the energy: a linear radical's pi shell is the standard case,
        // where the open electron may sit in either of the two pi orbitals
        // and both choices are the same ROHF state at the same energy. The
        // per-spin pair then alternates between two equivalent descriptions
        // forever (the UHF loop's criterion would measure that two-cycle),
        // while the ALPHA density does not move at all: D_alpha = D_c/2 +
        // D_o is the projector onto the OCCUPIED SPACE (closed and open
        // together), which the split rotation leaves invariant - the beta
        // density is the one that carries the split. The energy leg keeps
        // the split honest in the non-degenerate case: moving a boundary
        // orbital between the blocks then moves the energy by far more than
        // the tolerance, so a genuine change of state is refused rather
        // than smoothed over.
        const Eigen::MatrixXd densityChange = *newDensityAlpha - dAlpha;
        const double rmsDensityChange = densityChange.norm() / static_cast<double>(n);
        const double maxDensityChange = densityChange.cwiseAbs().maxCoeff();

        const bool converged = internal::IsScfConverged(iteration,
                                                        totalEnergy,
                                                        previousTotalEnergy,
                                                        rmsDensityChange,
                                                        maxDensityChange,
                                                        diisErrorNorm,
                                                        options.energyTolerance,
                                                        options.densityTolerance,
                                                        options.useRobustGate);
        const double energyDelta = std::fabs(totalEnergy - previousTotalEnergy);

        achievedEnergyDelta = energyDelta;
        achievedRmsDensityChange = rmsDensityChange;

        dAlpha = *newDensityAlpha;
        dBeta = *newDensityBeta;
        previousTotalEnergy = totalEnergy;
        previousElectronicEnergy = electronicEnergy;

        trace.Write(iteration + 1, totalEnergy, energyDelta, rmsDensityChange, converged);

        if (converged)
        {
            return finish(true, iteration + 1);
        }
    }

    return finish(false, options.maxIterations);
}

} // namespace

qcx::Result<RohfResult> RunRohfScf(const qcx::molecule::Molecule& molecule,
                                   const Eigen::MatrixXd& overlap,
                                   const Eigen::MatrixXd& coreHamiltonian,
                                   const RohfOptions& options,
                                   const UhfFockBuilderFn& fockBuilder) {
    if (!fockBuilder)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the UhfFockBuilderFn must be callable"});
    }

    const auto occupations = DeriveRestrictedOccupations(molecule);

    if (!occupations.has_value())
    {
        return std::unexpected(occupations.error());
    }

    return RunRohfLoop(molecule,
                       overlap,
                       coreHamiltonian,
                       options,
                       fockBuilder,
                       nullptr,
                       nullptr,
                       occupations->first,
                       occupations->second);
}

qcx::Result<RohfResult> RunRohfScf(const qcx::molecule::Molecule& molecule,
                                   const Eigen::MatrixXd& overlap,
                                   const Eigen::MatrixXd& coreHamiltonian,
                                   const RohfOptions& options,
                                   const UhfFockBuilderFn& fockBuilder,
                                   const CoulombFn& coulomb,
                                   const UhfEnergyContributionFn& energyContribution) {
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

    const auto occupations = DeriveRestrictedOccupations(molecule);

    if (!occupations.has_value())
    {
        return std::unexpected(occupations.error());
    }

    return RunRohfLoop(molecule,
                       overlap,
                       coreHamiltonian,
                       options,
                       fockBuilder,
                       &coulomb,
                       &energyContribution,
                       occupations->first,
                       occupations->second);
}

} // namespace qcx::scf
