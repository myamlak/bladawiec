// THE GUARD'S PIN, on the canonical broken-symmetry system: the
// irrep-blocked diagonalization must not CONSTRAIN an unrestricted
// solution - it may only speed it up.
//
// AO-space blocking is justified by "U^T (X^T F X) U is block-diagonal up
// to integral noise, so each block is diagonalized separately ... equivalent
// to the plain n x n solve". That equivalence REQUIRES F to commute with the
// point group - a property of the SOLUTION. The point group is a property of
// the NUCLEAR FRAMEWORK, and an unrestricted solution is allowed to have a
// smaller invariance group; that is exactly what an RHF -> UHF instability
// IS. Blocking such a solution does not accelerate its diagonalization, it
// PROJECTS IT OUT.
//
// This file first measured that as a defect (2026-09-16): on stretched
// H2/STO-3G the blocked path,
// guarded only by the C1 test, returned the closed-shell solution from a
// broken start - the HIGHER one, +123.7 mHa above the plain path's answer and
// equal to the RHF energy to 1e-12, with per-spin polarization exactly zero.
// The fix is the per-spin, per-iteration commutator guard in uhf.cpp's
// diagonalizeFock (internal::MeasureSymmetryAdaptation): a Fock whose
// generators' relative commutator norms exceed
// internal::kSymmetryAdaptationTolerance is not blocked, and the decision
// (with the norms it was taken on) is disclosed in
// UhfResult::symmetryBlockingAlpha/Beta rather than left implicit.
//
// The measurements this file asserts are therefore of two kinds, and the
// second kind is why the first is believed:
//   - the OUTCOME: blocked and plain now land on ONE solution from the same
//     broken start, and a plain run from a symmetry-adapted start is still
//     solved on the irrep blocks (the guard did not disable the feature).
//   - the DISCLOSURE: each run's guard decision, so "the two paths agree" can
//     never be satisfied by two runs that both silently fell back.
//
// The seam is the existing RunUhfScf basisSet argument (nullptr = plain) and
// the public EvaluateUhfDensityEnergy Fock builder; nothing is monkey-patched.

#include "h2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/density_energy.hpp"
#include "qcx/scf/uhf.hpp"
#include "qcx/symmetry/detection.hpp"
#include "scf_common.hpp"
#include "symmetry_blocks.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// H with the STO-3G 1s contraction PLUS the repo's minimal p shell
// (h2_sto3g.hpp's kPOrbitalBasis text, inline so the merge is explicit).
// Four functions per atom, so the AO-space irreps carry >= 2 functions and
// the blocked solve is a genuine per-irrep matrix diagonalization rather
// than a set of 1 x 1 solves.
inline constexpr std::string_view kSto3gHydrogenWithP = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
H    P
      1.0000000000E+00       1.0000000000E+00
END
)";

qcx::Result<qcx::basisset::BasisSet> MakeHydrogenWithPBasis() {
    return qcx::basisset::ParseNwchemText(kSto3gHydrogenWithP);
}

// H2 in STO-3G on the x axis at an arbitrary separation (the h2_sto3g.hpp
// fixture shape, stretched on demand).
qcx::Result<qcx::molecule::Molecule> MakeH2AtDistance(double r) {
    auto coordinates = CpuTensor2::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = -0.5 * r;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.5 * r;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

struct OneElectron {
    Eigen::MatrixXd overlap;
    Eigen::MatrixXd coreHamiltonian;
};

qcx::Result<OneElectron> BuildOneElectron(const qcx::molecule::Molecule& molecule,
                                          const qcx::basisset::BasisSet& basisSet) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    return OneElectron{ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear)};
}

// One UHF run's reportable outcome.
struct RunOutcome {
    double totalEnergy = 0.0;
    double spinSquared = 0.0;
    int iterations = 0;
    bool converged = false;
    Eigen::MatrixXd densityAlpha;
    Eigen::MatrixXd densityBeta;
    Eigen::MatrixXd fockAlpha;
    Eigen::MatrixXd fockBeta;
    double wallSeconds = 0.0;
    // The run's own guard disclosure (what it requested, what the guard
    // decided, and the norms it decided on) - asserted below, because
    // "blocked == plain" is only evidence of the equivalence when the run
    // actually blocked.
    qcx::scf::SymmetryBlockingReport blockingAlpha;
    qcx::scf::SymmetryBlockingReport blockingBeta;
};

// Runs UHF through the ONE Fock builder, choosing the diagonalization path
// with the basisSet argument alone (nullptr = the plain n x n solve).
qcx::Result<RunOutcome> RunUhf(const qcx::molecule::Molecule& molecule,
                               const OneElectron& oneElectron,
                               const CpuTensor4& eri,
                               const qcx::scf::UhfOptions& options,
                               const qcx::basisset::BasisSet* basisSet) {
    const qcx::scf::UhfFockBuilderFn builder =
        [&molecule, &oneElectron, &eri](const Eigen::MatrixXd& dAlpha, const Eigen::MatrixXd& dBeta)
        -> qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
        auto evaluated = qcx::scf::EvaluateUhfDensityEnergy(
            molecule, oneElectron.coreHamiltonian, eri, dAlpha, dBeta);

        if (!evaluated.has_value())
        {
            return std::unexpected(evaluated.error());
        }

        return std::make_pair(evaluated->fockAlpha, evaluated->fockBeta);
    };

    const auto started = std::chrono::steady_clock::now();
    auto result = qcx::scf::RunUhfScf(
        molecule, oneElectron.overlap, oneElectron.coreHamiltonian, options, builder, basisSet);
    const auto finished = std::chrono::steady_clock::now();

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    RunOutcome outcome;
    outcome.totalEnergy = result->totalEnergy;
    outcome.spinSquared = result->spinSquared;
    outcome.iterations = result->iterations;
    outcome.converged = result->converged;
    outcome.densityAlpha = result->densityAlpha;
    outcome.densityBeta = result->densityBeta;
    outcome.blockingAlpha = result->symmetryBlockingAlpha;
    outcome.blockingBeta = result->symmetryBlockingBeta;
    outcome.wallSeconds = std::chrono::duration<double>(finished - started).count();

    auto reEvaluated = qcx::scf::EvaluateUhfDensityEnergy(
        molecule, oneElectron.coreHamiltonian, eri, outcome.densityAlpha, outcome.densityBeta);

    if (!reEvaluated.has_value())
    {
        return std::unexpected(reEvaluated.error());
    }

    outcome.fockAlpha = reEvaluated->fockAlpha;
    outcome.fockBeta = reEvaluated->fockBeta;
    return outcome;
}

// One full case: geometry, start densities, and the four runs that pin down
// whether the blocked path constrains the search.
struct CaseMeasurement {
    double distanceBohr = 0.0;
    bool groupRealized = false;
    std::size_t highestBlockSize = 0;
    double offBlockFockAlphaPlain = 0.0;
    double offBlockFockBetaPlain = 0.0;
    double rhfTotalEnergy = 0.0;
    RunOutcome plainFromBrokenStart;
    RunOutcome blockedFromBrokenStart;
    RunOutcome plainFromSymmetricStart;
    RunOutcome blockedFromSymmetricStart;
    RunOutcome plainFromBrokenSolution;
    RunOutcome blockedFromBrokenSolution;
};

const char* ActionName(qcx::scf::SymmetryBlockingAction action) {
    switch (action)
    {
    case qcx::scf::SymmetryBlockingAction::kNotRequested:
        return "not-requested";
    case qcx::scf::SymmetryBlockingAction::kUsed:
        return "used";
    case qcx::scf::SymmetryBlockingAction::kDemoted:
        return "demoted";
    case qcx::scf::SymmetryBlockingAction::kRefused:
        return "refused";
    case qcx::scf::SymmetryBlockingAction::kUnavailable:
        return "unavailable";
    }

    return "?";
}

void ReportRun(const char* label, const RunOutcome& outcome) {
    std::printf("      %-22s E=%.12f <S^2>=%.9f it=%3d conv=%d  %.4f s\n",
                label,
                outcome.totalEnergy,
                outcome.spinSquared,
                outcome.iterations,
                outcome.converged ? 1 : 0,
                outcome.wallSeconds);
    std::printf("        blocking alpha=%-13s %d blocked / %d plain  worst=%.3e (tol %.0e)",
                ActionName(outcome.blockingAlpha.action),
                outcome.blockingAlpha.blockedSolveCount,
                outcome.blockingAlpha.plainSolveCount,
                outcome.blockingAlpha.maxGeneratorCommutatorNorm,
                outcome.blockingAlpha.tolerance);

    for (const double norm : outcome.blockingAlpha.generatorCommutatorNorms)
    {
        std::printf(" %.3e", norm);
    }

    std::printf("\n        blocking beta =%-13s %d blocked / %d plain  worst=%.3e (tol %.0e)",
                ActionName(outcome.blockingBeta.action),
                outcome.blockingBeta.blockedSolveCount,
                outcome.blockingBeta.plainSolveCount,
                outcome.blockingBeta.maxGeneratorCommutatorNorm,
                outcome.blockingBeta.tolerance);

    for (const double norm : outcome.blockingBeta.generatorCommutatorNorms)
    {
        std::printf(" %.3e", norm);
    }

    std::printf("\n");
}

void ReportCase(const CaseMeasurement& measured) {
    std::printf("[uhf-symmetry-probe] R=%.3f  groupRealized=%d  maxBlock=%zu  RHF(plain) E=%.12f\n",
                measured.distanceBohr,
                measured.groupRealized ? 1 : 0,
                measured.highestBlockSize,
                measured.rhfTotalEnergy);
    std::printf("  L1 OffBlockNorm(F_a,F_b) at the converged PLAIN solution: %.3e  %.3e"
                "   (the blocked path needs the ~1e-12 noise level)\n",
                measured.offBlockFockAlphaPlain,
                measured.offBlockFockBetaPlain);
    std::printf("  L2 from the broken start:\n");
    ReportRun("plain", measured.plainFromBrokenStart);
    ReportRun("blocked", measured.blockedFromBrokenStart);
    std::printf("  L2b from a symmetry-adapted start:\n");
    ReportRun("plain", measured.plainFromSymmetricStart);
    ReportRun("blocked", measured.blockedFromSymmetricStart);
    std::printf("  L3 from the PLAIN broken solution:\n");
    ReportRun("plain", measured.plainFromBrokenSolution);
    ReportRun("blocked", measured.blockedFromBrokenSolution);
    const double alphaDensityGap =
        (measured.plainFromBrokenStart.densityAlpha - measured.blockedFromBrokenStart.densityAlpha)
            .cwiseAbs()
            .maxCoeff();
    const double betaDensityGap =
        (measured.plainFromBrokenStart.densityBeta - measured.blockedFromBrokenStart.densityBeta)
            .cwiseAbs()
            .maxCoeff();
    const double spinDensityGap =
        ((measured.plainFromBrokenStart.densityAlpha - measured.plainFromBrokenStart.densityBeta) -
         (measured.blockedFromBrokenStart.densityAlpha -
          measured.blockedFromBrokenStart.densityBeta))
            .cwiseAbs()
            .maxCoeff();
    std::printf("  density gap plain vs blocked (broken start): |dDa|=%.6e |dDb|=%.6e "
                "|d(Da-Db)|=%.6e\n",
                alphaDensityGap,
                betaDensityGap,
                spinDensityGap);
    // Each run's OWN spin polarization. <S^2> = 0 alone does not settle
    // whether a run is the closed-shell solution or a broken-symmetry
    // singlet (the O2 BS singlet is the standing counter-example), so the
    // polarization is reported separately.
    const double plainPolarization =
        (measured.plainFromBrokenStart.densityAlpha - measured.plainFromBrokenStart.densityBeta)
            .cwiseAbs()
            .maxCoeff();
    const double blockedPolarization =
        (measured.blockedFromBrokenStart.densityAlpha - measured.blockedFromBrokenStart.densityBeta)
            .cwiseAbs()
            .maxCoeff();
    std::printf("  own spin polarization |Da-Db|_max: plain=%.3e  blocked=%.3e\n",
                plainPolarization,
                blockedPolarization);
    std::fflush(stdout);
}

// Builds the case and runs all four legs. Returns nullopt-free Result so the
// caller can ASSERT on the machinery rather than on the physics.
qcx::Result<CaseMeasurement> MeasureCaseWithBasis(double r,
                                                  const qcx::basisset::BasisSet& basis,
                                                  const Eigen::MatrixXd& brokenStartAlpha,
                                                  const Eigen::MatrixXd& brokenStartBeta) {
    auto molecule = MakeH2AtDistance(r);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto oneElectron = BuildOneElectron(*molecule, basis);

    if (!oneElectron.has_value())
    {
        return std::unexpected(oneElectron.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    CaseMeasurement measured;
    measured.distanceBohr = r;

    // L1's scaffolding: the group must actually be realized, or the whole
    // measurement is void (a C1 or unrealizable group takes the plain path
    // in both runs and the two are trivially equal).
    const auto analysis = qcx::symmetry::DetectPointGroup(*molecule);
    auto blocks = qcx::scf::internal::BuildSymmetryBlocks(*molecule, basis, analysis);

    if (!blocks.has_value() && blocks.error().code != qcx::ErrorCode::kUnimplemented)
    {
        return std::unexpected(blocks.error());
    }

    const bool groupRealized = blocks.has_value() && !blocks->isTrivial;
    measured.groupRealized = groupRealized;

    if (groupRealized)
    {
        for (const Eigen::Index size : blocks->blockSizes)
        {
            measured.highestBlockSize =
                std::max(measured.highestBlockSize, static_cast<std::size_t>(size));
        }
    }

    qcx::scf::UhfOptions options;
    // Isolate the diagonalization path: the full-group labeling stage would
    // SYMMETRIZE the returned densities and confound the comparison.
    options.fullGroupLabeling = false;

    qcx::scf::UhfOptions brokenStartOptions = options;
    brokenStartOptions.initialDensityAlpha = brokenStartAlpha;
    brokenStartOptions.initialDensityBeta = brokenStartBeta;

    auto plainFromBroken = RunUhf(*molecule, *oneElectron, *eri, brokenStartOptions, nullptr);

    if (!plainFromBroken.has_value())
    {
        return std::unexpected(plainFromBroken.error());
    }

    auto blockedFromBroken =
        RunUhf(*molecule, *oneElectron, *eri, brokenStartOptions, groupRealized ? &basis : nullptr);

    if (!blockedFromBroken.has_value())
    {
        return std::unexpected(blockedFromBroken.error());
    }

    measured.plainFromBrokenStart = std::move(*plainFromBroken);
    measured.blockedFromBrokenStart = std::move(*blockedFromBroken);

    // L2b: the same two paths from a symmetry-adapted start (both spins on
    // the same unpolarized density). The plain path's whole SCF map preserves
    // symmetry, so this leg shows what each path does when the question of a
    // broken branch never arises.
    const Eigen::MatrixXd symmetricStart = 0.5 * (brokenStartAlpha + brokenStartBeta);
    qcx::scf::UhfOptions symmetricOptions = options;
    symmetricOptions.initialDensityAlpha = symmetricStart;
    symmetricOptions.initialDensityBeta = symmetricStart;

    auto plainFromSymmetric = RunUhf(*molecule, *oneElectron, *eri, symmetricOptions, nullptr);

    if (!plainFromSymmetric.has_value())
    {
        return std::unexpected(plainFromSymmetric.error());
    }

    auto blockedFromSymmetric =
        RunUhf(*molecule, *oneElectron, *eri, symmetricOptions, groupRealized ? &basis : nullptr);

    if (!blockedFromSymmetric.has_value())
    {
        return std::unexpected(blockedFromSymmetric.error());
    }

    measured.plainFromSymmetricStart = std::move(*plainFromSymmetric);
    measured.blockedFromSymmetricStart = std::move(*blockedFromSymmetric);

    // The closed-shell reference: at H2 the symmetry-adapted unrestricted
    // solution IS the RHF solution, so the RHF total energy names whatever
    // the constrained path lands on.
    auto rhf = qcx::scf::RunRhfScf(*molecule,
                                   oneElectron->overlap,
                                   oneElectron->coreHamiltonian,
                                   *eri,
                                   qcx::scf::RhfOptions{});

    if (!rhf.has_value())
    {
        return std::unexpected(rhf.error());
    }

    measured.rhfTotalEnergy = rhf->totalEnergy;

    if (groupRealized)
    {
        measured.offBlockFockAlphaPlain =
            qcx::scf::internal::OffBlockNorm(measured.plainFromBrokenStart.fockAlpha, *blocks);
        measured.offBlockFockBetaPlain =
            qcx::scf::internal::OffBlockNorm(measured.plainFromBrokenStart.fockBeta, *blocks);
    }

    // L3: restart BOTH paths from the plain path's converged broken
    // solution. The plain path must return its own fixed point; a blocked
    // path that walks away from it is constraining the search.
    qcx::scf::UhfOptions restartOptions = options;
    restartOptions.initialDensityAlpha = measured.plainFromBrokenStart.densityAlpha;
    restartOptions.initialDensityBeta = measured.plainFromBrokenStart.densityBeta;

    auto plainRestart = RunUhf(*molecule, *oneElectron, *eri, restartOptions, nullptr);

    if (!plainRestart.has_value())
    {
        return std::unexpected(plainRestart.error());
    }

    auto blockedRestart =
        RunUhf(*molecule, *oneElectron, *eri, restartOptions, groupRealized ? &basis : nullptr);

    if (!blockedRestart.has_value())
    {
        return std::unexpected(blockedRestart.error());
    }

    measured.plainFromBrokenSolution = std::move(*plainRestart);
    measured.blockedFromBrokenSolution = std::move(*blockedRestart);
    return measured;
}

qcx::Result<CaseMeasurement> MeasureCase(double r,
                                         const Eigen::MatrixXd& brokenStartAlpha,
                                         const Eigen::MatrixXd& brokenStartBeta) {
    auto basis = MakeSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    return MeasureCaseWithBasis(r, *basis, brokenStartAlpha, brokenStartBeta);
}

} // namespace

// The canonical RHF -> UHF instability: an H2 bond stretched well past the
// instability onset, started from a density that localizes alpha on one atom
// and beta on the other (the broken-symmetry solution's own character). This
// start is genuinely symmetry-broken: its spin density is antisymmetric under
// the exchange of the two atoms, so no symmetry operation of the molecule
// leaves the alpha Fock invariant.
TEST(UhfSymmetryConstraintTest, StretchedH2BrokenStartBlockedVersusPlain) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    // AO 0 is H at -R/2 and AO 1 is H at +R/2 (atom-major function order,
    // atoms renumbered by position along x). Tr[D_sigma S] = S_00 = 1 = the
    // per-spin electron count of H2.
    Eigen::MatrixXd brokenAlpha = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd brokenBeta = Eigen::MatrixXd::Zero(2, 2);
    brokenAlpha(0, 0) = 1.0;
    brokenBeta(1, 1) = 1.0;

    // The start is a legitimate density pair for H2: one electron per spin.
    auto probeMolecule = MakeH2AtDistance(3.5);
    ASSERT_TRUE(probeMolecule.has_value()) << probeMolecule.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*probeMolecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    const Eigen::MatrixXd s = ToMatrix(*overlap);
    EXPECT_NEAR((brokenAlpha * s).trace(), 1.0, 1e-12);
    EXPECT_NEAR((brokenBeta * s).trace(), 1.0, 1e-12);

    auto measured = MeasureCase(3.5, brokenAlpha, brokenBeta);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ReportCase(*measured);

    // The measurement is void unless the blocked path was actually taken.
    ASSERT_TRUE(measured->groupRealized)
        << "H2/STO-3G did not realize a non-trivial point group: the blocked path is not engaged";

    // Machinery: every run is a genuine converged SCF solution.
    ASSERT_TRUE(measured->plainFromBrokenStart.converged)
        << "iterations: " << measured->plainFromBrokenStart.iterations;
    ASSERT_TRUE(measured->blockedFromBrokenStart.converged)
        << "iterations: " << measured->blockedFromBrokenStart.iterations;
    ASSERT_TRUE(measured->plainFromSymmetricStart.converged);

    // The two paths from the same symmetry-adapted start agree: the own
    // case, and the one the seam was designed for. This is the other half of
    // the guard - it must not disable blocking where blocking is VALID - so
    // the agreement is asserted together with the evidence that the blocked
    // run actually blocked: an agreement between two runs that both fell back
    // to the plain solve would prove nothing.
    EXPECT_EQ(measured->blockedFromSymmetricStart.blockingAlpha.action,
              qcx::scf::SymmetryBlockingAction::kUsed);
    EXPECT_GT(measured->blockedFromSymmetricStart.blockingAlpha.blockedSolveCount, 0);
    EXPECT_EQ(measured->blockedFromSymmetricStart.blockingAlpha.plainSolveCount, 0);
    EXPECT_LT(measured->blockedFromSymmetricStart.blockingAlpha.maxGeneratorCommutatorNorm, 1e-11);
    EXPECT_NEAR(measured->blockedFromSymmetricStart.totalEnergy,
                measured->plainFromSymmetricStart.totalEnergy,
                1e-11);

    // The plain path, restarted from its own solution, is a fixed point.
    EXPECT_NEAR(measured->plainFromBrokenSolution.totalEnergy,
                measured->plainFromBrokenStart.totalEnergy,
                1e-9);

    // ---- the fix's contract, pinned --------------------------------------
    //
    // Before the guard (uhf.cpp's diagonalizeFock) this case measured the
    // defect: the blocked path returned the CLOSED-SHELL solution from a
    // broken start - E = -0.816344159870, <S^2> = 0.0000, per-spin
    // polarization exactly zero, +123.7 mHa ABOVE the plain path's answer and
    // equal to the RHF energy to 1e-12 - while the plain path found the broken
    // solution, E = -0.940030738235, <S^2> = 0.9095. The blocked path was not
    // slower, it was WRONG: blocking by the full point group projects out the
    // solution whose invariance group is smaller.

    // (1) the precondition is violated on this solution: the converged
    // plain Fock's off-block norm is ~1, not the ~1e-12 noise level the
    // block-diagonal argument needs. A property of the solution, so unchanged
    // by the guard - and the reason the guard has to exist.
    EXPECT_GT(measured->offBlockFockAlphaPlain, 1e-6);

    // (2) The guard's own reading of the same violation, and the refusal it
    // produced: the blocked run never blocked a single diagonalization, and
    // recorded the norms it refused on rather than a bare flag.
    EXPECT_EQ(measured->blockedFromBrokenStart.blockingAlpha.action,
              qcx::scf::SymmetryBlockingAction::kRefused);
    EXPECT_EQ(measured->blockedFromBrokenStart.blockingAlpha.blockedSolveCount, 0);
    EXPECT_GT(measured->blockedFromBrokenStart.blockingAlpha.plainSolveCount, 0);
    EXPECT_GT(measured->blockedFromBrokenStart.blockingAlpha.maxGeneratorCommutatorNorm,
              measured->blockedFromBrokenStart.blockingAlpha.tolerance);
    // The recorded vector is the measurement the maximum came from, so the
    // disclosure explains itself instead of asking for trust.
    const auto& refusedNorms =
        measured->blockedFromBrokenStart.blockingAlpha.generatorCommutatorNorms;
    ASSERT_FALSE(refusedNorms.empty())
        << "a refusal with no generator norms beside it is an unexplained refusal";
    const Eigen::Map<const Eigen::VectorXd> refusedNormsAsVector(
        refusedNorms.data(), static_cast<Eigen::Index>(refusedNorms.size()));
    EXPECT_NEAR(measured->blockedFromBrokenStart.blockingAlpha.maxGeneratorCommutatorNorm,
                refusedNormsAsVector.maxCoeff(),
                1e-15);

    // (3) A plain run asked for nothing (no basis set was supplied) and the
    // record says so, rather than leaving the fields at a default that reads
    // like a passed check.
    EXPECT_EQ(measured->plainFromBrokenStart.blockingAlpha.action,
              qcx::scf::SymmetryBlockingAction::kNotRequested);

    // (4) THE FIX: from the same broken start the blocked path now finds the
    // same solution the plain path finds - the broken one - and no longer the
    // closed-shell solution it used to return.
    EXPECT_NEAR(measured->blockedFromBrokenStart.totalEnergy,
                measured->plainFromBrokenStart.totalEnergy,
                1e-9);
    EXPECT_NEAR(measured->blockedFromBrokenStart.spinSquared,
                measured->plainFromBrokenStart.spinSquared,
                1e-9);
    EXPECT_GT(measured->plainFromBrokenStart.spinSquared, 0.5);
    EXPECT_GT(measured->blockedFromBrokenStart.spinSquared, 0.5);
    EXPECT_GT(std::abs(measured->blockedFromBrokenStart.totalEnergy - measured->rhfTotalEnergy),
              1e-3)
        << "the blocked path is back on the closed-shell solution: the guard did not fire";

    // (5) Fixed points: the plain path stays put when restarted from its own
    // solution, the blocked path does too, and both land on ONE solution.
    // (The guard refuses on every Fock of this run, so the two walks are the
    // same walk - asserted, not assumed, by the refusal above.)
    EXPECT_NEAR(measured->plainFromBrokenSolution.totalEnergy,
                measured->plainFromBrokenStart.totalEnergy,
                1e-9);
    EXPECT_NEAR(measured->blockedFromBrokenSolution.totalEnergy,
                measured->blockedFromBrokenStart.totalEnergy,
                1e-9);
    EXPECT_NEAR(measured->blockedFromBrokenSolution.totalEnergy,
                measured->plainFromBrokenSolution.totalEnergy,
                1e-9);
}

// Control: the SAME broken start on H2 at its equilibrium separation, where
// the symmetry-pure solution is the stable one. Whatever the blocked path
// does here, it is not evidence of a constraint - both paths should relax to
// the same symmetry-adapted solution, which is the case covers.
TEST(UhfSymmetryConstraintTest, EquilibriumH2BrokenStartRelaxesToTheSameSolution) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    Eigen::MatrixXd brokenAlpha = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd brokenBeta = Eigen::MatrixXd::Zero(2, 2);
    brokenAlpha(0, 0) = 1.0;
    brokenBeta(1, 1) = 1.0;

    auto measured = MeasureCase(1.4, brokenAlpha, brokenBeta);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ReportCase(*measured);

    ASSERT_TRUE(measured->groupRealized);
    ASSERT_TRUE(measured->plainFromBrokenStart.converged);
    ASSERT_TRUE(measured->blockedFromBrokenStart.converged);

    EXPECT_NEAR(measured->blockedFromBrokenStart.totalEnergy,
                measured->plainFromBrokenStart.totalEnergy,
                1e-11)
        << "at the equilibrium separation the symmetry-pure solution is stable, so the two paths "
           "must agree";

    // The guard's decision on this separation, and the control the fix's
    // acceptance turns on: the SAME broken start on a system whose solution
    // is symmetry-pure must still reach the irrep blocks on the way. A guard
    // that refused here would be inert (both runs would be the plain run, and
    // the agreement above would prove nothing about), and one that refused
    // ONLY here would be backwards.
    EXPECT_EQ(measured->blockedFromBrokenStart.blockingAlpha.action,
              qcx::scf::SymmetryBlockingAction::kDemoted)
        << "the broken start must still be refused on the first iteration, then unblocked once "
           "the walk is symmetry-adapted again";
    EXPECT_GT(measured->blockedFromBrokenStart.blockingAlpha.blockedSolveCount, 0);
    EXPECT_GT(measured->blockedFromBrokenStart.blockingAlpha.plainSolveCount, 0);

    // The symmetry-adapted start on the same separation: blocked throughout,
    // and the path the equivalence claim is actually about.
    EXPECT_EQ(measured->blockedFromSymmetricStart.blockingAlpha.action,
              qcx::scf::SymmetryBlockingAction::kUsed);
    EXPECT_EQ(measured->blockedFromSymmetricStart.blockingAlpha.plainSolveCount, 0);
    EXPECT_LT(measured->blockedFromSymmetricStart.blockingAlpha.maxGeneratorCommutatorNorm, 1e-11);
    EXPECT_NEAR(measured->blockedFromSymmetricStart.totalEnergy,
                measured->plainFromSymmetricStart.totalEnergy,
                1e-11);
}

// The same measurement on a basis whose AO-space irreps carry MORE THAN ONE
// function, so the blocked solve is a genuine per-irrep matrix
// diagonalization and not a set of 1 x 1 solves. H gets the STO-3G 1s
// contraction plus a p shell (4 functions per atom, 8 in total); the two
// atoms still sit on x, so the group and the broken start's construction are
// the same. AO 0 is the 1s of the atom at -R/2 and AO 4 the 1s of the atom
// at +R/2 (atom-major order, S shell before P shell within an atom).
TEST(UhfSymmetryConstraintTest, StretchedH2WithPBasisMultiFunctionBlocks) {
    auto basis = MakeHydrogenWithPBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    Eigen::MatrixXd brokenAlpha = Eigen::MatrixXd::Zero(8, 8);
    Eigen::MatrixXd brokenBeta = Eigen::MatrixXd::Zero(8, 8);
    brokenAlpha(0, 0) = 1.0;
    brokenBeta(4, 4) = 1.0;

    auto probeMolecule = MakeH2AtDistance(3.5);
    ASSERT_TRUE(probeMolecule.has_value()) << probeMolecule.error().message;
    auto overlap = qcx::integrals::BuildOverlapMatrix(*probeMolecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    const Eigen::MatrixXd s = ToMatrix(*overlap);
    EXPECT_NEAR((brokenAlpha * s).trace(), 1.0, 1e-12);
    EXPECT_NEAR((brokenBeta * s).trace(), 1.0, 1e-12);

    auto measured = MeasureCaseWithBasis(3.5, *basis, brokenAlpha, brokenBeta);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ReportCase(*measured);

    ASSERT_TRUE(measured->groupRealized);
    ASSERT_GT(measured->highestBlockSize, 1)
        << "this case exists to exercise multi-function irreps; a 1 x 1 decomposition makes it a "
           "duplicate of the STO-3G case";
    ASSERT_TRUE(measured->plainFromBrokenStart.converged)
        << "iterations: " << measured->plainFromBrokenStart.iterations;
    ASSERT_TRUE(measured->blockedFromBrokenStart.converged)
        << "iterations: " << measured->blockedFromBrokenStart.iterations;

    // The same two facts as the STO-3G case, on a decomposition whose blocks
    // hold more than one function: the guard refuses here too, and the two
    // paths land on one solution. See that test for what they mean.
    EXPECT_EQ(measured->blockedFromBrokenStart.blockingAlpha.action,
              qcx::scf::SymmetryBlockingAction::kRefused);
    EXPECT_EQ(measured->blockedFromBrokenStart.blockingAlpha.blockedSolveCount, 0);
    EXPECT_GT(measured->plainFromBrokenStart.spinSquared, 0.5);
    EXPECT_NEAR(measured->blockedFromBrokenStart.spinSquared,
                measured->plainFromBrokenStart.spinSquared,
                1e-9);
    EXPECT_NEAR(measured->blockedFromBrokenStart.totalEnergy,
                measured->plainFromBrokenStart.totalEnergy,
                1e-9);
    EXPECT_GT(std::abs(measured->blockedFromBrokenStart.totalEnergy - measured->rhfTotalEnergy),
              1e-3);
}
