// THE BLOCKING GUARD ON THE RESTRICTED PATH: PINNED, INCLUDING ITS PRICE.
//
// The unrestricted path carries a per-spin guard (uhf.cpp's diagonalizeFock):
// each iteration measures whether its Fock still commutes with the point
// group and refuses the irrep blocking when it does not. The restricted path
// had the same blocked/plain site (rhf.cpp's diagonalizeFock) and no guard.
// This file first measured that gap (2026-09-16) and its answer was to leave
// it open, in two parts.
//
// ONE - a restricted run CAN reach a non-symmetry-adapted Fock, by two routes,
// both measured below:
//   (a) RhfOptions::initialScfState, the named one: a caller-supplied density
//       breaks the group at the first Fock.
//   (b) NOT named before: the DEFAULT GWH start, on a framework whose HOMO is
//       degenerate by symmetry. The guess diagonalizes an adapted H_core, but
//       inside the degenerate set the eigensolver's columns are arbitrary and
//       their mixture need not be adapted: on square H4 (D4h, computational
//       group D2h) F[D_gwh] measures 9.157e-02 against the guard's 1e-8
//       tolerance, with no caller-supplied state anywhere.
//
// TWO - on THIS path the blocked solve is not always the worse one. Where the
// two paths disagreed on square H4, the blocked path returned the LOWER
// solution (-1.719789653762, adapted, reachable by restart) and the plain
// path a symmetry-BROKEN fixed point 86.27 mHa above it (-1.633519911629),
// reachable from the default start.
//
// The owner's ruling of 2026-10-25 ports the guard anyway, because basin
// selection is not the criterion the guard exists for. The blocked solve of a
// non-adapted Fock CONSTRAINS the variational search: it returns the answer
// restricted to the group's irreps, converged and variational-looking, and
// the run has no way to say so. On the response path that was a live WRONG
// answer rather than a higher one: an applied field couples sigma_g and
// sigma_u in H2/STO-3G, the blocked solve kept only the diagonal irrep
// blocks, and the finite-field polarizability came back as zero on a density
// that never saw the field (response/tests/response_kernel_test.cpp's
// BuildKsReference). The refusal is the safe direction - a plain solve of an
// adapted Fock only loses the speedup - and the price is the square-H4 basin
// recorded below.
//
// This file now pins the ported guard on both of its halves:
//   - the OUTCOME: a symmetry-breaking one-electron term is not projected out
//     (the applied-field cell), a refused walk returns exactly what the plain
//     path returns, and an adapted Fock still takes the irrep blocks;
//   - the DISCLOSURE: each run's HfResult::symmetryBlocking - the action, the
//     tolerance and the per-generator norms - so "the two paths agree" can
//     never be satisfied by a blocked run that silently fell back, and a
//     refusal can be checked instead of trusted.
//
// The seam is the existing RunRhfScf basisSet argument (nullptr = plain) and
// the public EvaluateRhfDensityEnergy Fock builder; nothing is monkey-patched.
// The guard's own reading of a Fock is taken through the internal seam
// (internal::MeasureSymmetryAdaptation, internal::OffBlockNorm) so this file
// reuses that implementation instead of stating a second version of it: one
// fact, one home.

#include "h2_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/scf/density_energy.hpp"
#include "qcx/scf/rhf.hpp"
#include "qcx/scf/uhf.hpp"
#include "qcx/symmetry/detection.hpp"
#include "scf_common.hpp"
#include "symmetry_blocks.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using CpuTensor4 = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// A hydrogen framework at arbitrary positions (bohr), neutral, singlet.
qcx::Result<qcx::molecule::Molecule> MakeHydrogenMolecule(
    const std::vector<std::array<double, 3>>& positions) {
    auto coordinates = CpuTensor2::Create({positions.size(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t index = 0; index < positions.size(); ++index)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            (*coordinates)(static_cast<Eigen::Index>(index), axis) = positions[index][axis];
        }

        atoms.push_back({"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

// Four hydrogens in a line on x, centred on the origin, equally spaced.
qcx::Result<qcx::molecule::Molecule> MakeH4Chain(double spacing) {
    const std::vector<std::array<double, 3>> positions = {{-1.5 * spacing, 0.0, 0.0},
                                                          {-0.5 * spacing, 0.0, 0.0},
                                                          {0.5 * spacing, 0.0, 0.0},
                                                          {1.5 * spacing, 0.0, 0.0}};
    return MakeHydrogenMolecule(positions);
}

// Four hydrogens in a rectangle (D2h for a != b), centred on the origin:
// atoms 0/1 are the left column (x = -a), atoms 2/3 the right one. a = b is
// the square - D4h, whose computational group is D2h and whose HOMO pair is
// degenerate by the parent group's symmetry.
qcx::Result<qcx::molecule::Molecule> MakeH4Rectangle(double a, double b) {
    const std::vector<std::array<double, 3>> positions = {
        {-a, -b, 0.0}, {-a, b, 0.0}, {a, -b, 0.0}, {a, b, 0.0}};
    return MakeHydrogenMolecule(positions);
}

// Six hydrogens in a line on x, centred on the origin, equally spaced.
qcx::Result<qcx::molecule::Molecule> MakeH6Chain(double spacing) {
    const std::vector<std::array<double, 3>> positions = {{-2.5 * spacing, 0.0, 0.0},
                                                          {-1.5 * spacing, 0.0, 0.0},
                                                          {-0.5 * spacing, 0.0, 0.0},
                                                          {0.5 * spacing, 0.0, 0.0},
                                                          {1.5 * spacing, 0.0, 0.0},
                                                          {2.5 * spacing, 0.0, 0.0}};
    return MakeHydrogenMolecule(positions);
}

// H2 on x at an arbitrary separation (the h2_sto3g.hpp fixture shape,
// stretched on demand): the unrestricted path's canonical broken case, and
// here the control showing the closed-shell path has no such case at all.
qcx::Result<qcx::molecule::Molecule> MakeH2AtDistance(double r) {
    const std::vector<std::array<double, 3>> positions = {{-0.5 * r, 0.0, 0.0},
                                                          {0.5 * r, 0.0, 0.0}};
    return MakeHydrogenMolecule(positions);
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

// One framework, opened once: the integrals, the detected group, and the
// once-per-run symmetry decomposition every leg below measures against. The
// molecule is a pointer because Molecule is movable-only; the caller owns it
// and outlives the System.
struct System {
    const qcx::molecule::Molecule* molecule = nullptr;
    const qcx::basisset::BasisSet* basis = nullptr;
    OneElectron oneElectron;
    std::optional<CpuTensor4> eri;
    bool groupRealized = false;
    qcx::scf::internal::SymmetryBlocks blocks;
    qcx::scf::internal::BlockedDiagonalizeData data;
};

qcx::Result<System> OpenSystem(const qcx::molecule::Molecule& molecule,
                               const qcx::basisset::BasisSet& basis) {
    System system;
    system.molecule = &molecule;
    system.basis = &basis;

    auto oneElectron = BuildOneElectron(molecule, basis);

    if (!oneElectron.has_value())
    {
        return std::unexpected(oneElectron.error());
    }

    system.oneElectron = std::move(*oneElectron);

    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basis);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    system.eri = std::move(*eri);

    const auto analysis = qcx::symmetry::DetectPointGroup(molecule);
    auto built = qcx::scf::internal::BuildSymmetryBlocks(molecule, basis, analysis);

    if (!built.has_value() && built.error().code != qcx::ErrorCode::kUnimplemented)
    {
        return std::unexpected(built.error());
    }

    system.groupRealized = built.has_value() && !built->isTrivial;

    if (system.groupRealized)
    {
        system.blocks = std::move(*built);

        auto data = qcx::scf::internal::BuildBlockedDiagonalizeData(system.oneElectron.overlap,
                                                                    system.blocks);

        if (!data.has_value())
        {
            return std::unexpected(data.error());
        }

        system.data = std::move(*data);
    }

    return system;
}

// One RHF run's reportable outcome.
struct RunOutcome {
    double totalEnergy = 0.0;
    int iterations = 0;
    bool converged = false;
    Eigen::MatrixXd density;
    /// The guard's own reading of the CONVERGED density's Fock: the largest
    /// relative generator commutator, and the off-block norm of the same
    /// matrix. Both are 0 when the run realized no group to measure against.
    double adaptationNorm = 0.0;
    double offBlockFock = 0.0;
    /// The run's own blocking-guard disclosure (what it requested, what the
    /// guard decided, and the norms it decided on).
    qcx::scf::SymmetryBlockingReport blocking;
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

void ReportOutcome(const char* label, const RunOutcome& outcome) {
    std::printf("      %-22s E=%+.12f it=%3d conv=%d attn=%.3e  blocking=%-13s "
                "%d blocked / %d plain  worst=%.3e (tol %.0e)\n",
                label,
                outcome.totalEnergy,
                outcome.iterations,
                outcome.converged ? 1 : 0,
                outcome.adaptationNorm,
                ActionName(outcome.blocking.action),
                outcome.blocking.blockedSolveCount,
                outcome.blocking.plainSolveCount,
                outcome.blocking.maxGeneratorCommutatorNorm,
                outcome.blocking.tolerance);
    std::fflush(stdout);
}

// Runs RHF through the one Fock builder with an EXPLICIT core Hamiltonian -
// the seam an applied field enters through (core + field * dipole) - choosing
// the diagonalization path with the basisSet argument alone (nullptr = the
// plain n x n solve).
qcx::Result<RunOutcome> RunRhfWithCore(const System& system,
                                       const Eigen::MatrixXd& coreHamiltonian,
                                       const qcx::scf::RhfOptions& options,
                                       const qcx::basisset::BasisSet* basisSet) {
    const qcx::scf::FockBuilderFn builder =
        [&system,
         &coreHamiltonian](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto evaluated = qcx::scf::EvaluateRhfDensityEnergy(
            *system.molecule, coreHamiltonian, *system.eri, density);

        if (!evaluated.has_value())
        {
            return std::unexpected(evaluated.error());
        }

        return evaluated->fock;
    };

    auto result = qcx::scf::RunRhfScf(
        *system.molecule, system.oneElectron.overlap, coreHamiltonian, options, builder, basisSet);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    RunOutcome outcome;
    outcome.totalEnergy = result->totalEnergy;
    outcome.iterations = result->iterations;
    outcome.converged = result->converged;
    outcome.density = result->density;
    outcome.blocking = result->symmetryBlocking;

    auto reEvaluated = qcx::scf::EvaluateRhfDensityEnergy(
        *system.molecule, coreHamiltonian, *system.eri, outcome.density);

    if (!reEvaluated.has_value())
    {
        return std::unexpected(reEvaluated.error());
    }

    if (system.groupRealized)
    {
        outcome.adaptationNorm = qcx::scf::internal::MeasureSymmetryAdaptation(
                                     reEvaluated->fock, system.blocks, system.data)
                                     .maxCommutatorNorm;
        outcome.offBlockFock = qcx::scf::internal::OffBlockNorm(reEvaluated->fock, system.blocks);
    }

    return outcome;
}

// The same run on the system's own core Hamiltonian.
qcx::Result<RunOutcome> RunRhf(const System& system,
                               const qcx::scf::RhfOptions& options,
                               const qcx::basisset::BasisSet* basisSet) {
    return RunRhfWithCore(system, system.oneElectron.coreHamiltonian, options, basisSet);
}

// The guard's own measure applied to the Fock built from an ARBITRARY density:
// what the unrestricted path's guard would read on a walk whose current
// density is `density`. Used to diagnose every START this file hands the loop,
// before any iteration runs.
qcx::Result<double> AdaptationOfDensity(const System& system, const Eigen::MatrixXd& density) {
    auto evaluated = qcx::scf::EvaluateRhfDensityEnergy(
        *system.molecule, system.oneElectron.coreHamiltonian, *system.eri, density);

    if (!evaluated.has_value())
    {
        return std::unexpected(evaluated.error());
    }

    return qcx::scf::internal::MeasureSymmetryAdaptation(
               evaluated->fock, system.blocks, system.data)
        .maxCommutatorNorm;
}

// One AO-diagonal start density: `perAtom[i]` electrons on the (single) s
// function of atom i, in the atom-major STO-3G function order. Charged like
// this it breaks the point group's atom permutation, which is the point.
Eigen::MatrixXd MakeChargeSeed(const std::vector<double>& perAtom) {
    Eigen::MatrixXd seed = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(perAtom.size()),
                                                 static_cast<Eigen::Index>(perAtom.size()));

    for (std::size_t atom = 0; atom < perAtom.size(); ++atom)
    {
        seed(static_cast<Eigen::Index>(atom), static_cast<Eigen::Index>(atom)) = perAtom[atom];
    }

    return seed;
}

// The default GWH start the loop uses when no restart state is supplied, as a
// density: what the restricted path's own interior begins from.
qcx::Result<Eigen::MatrixXd> MakeGwhDensity(const System& system) {
    const int numOccupied = system.molecule->ElectronCount() / 2;
    auto gwh = qcx::scf::BuildGwhGuess(
        system.oneElectron.overlap, system.oneElectron.coreHamiltonian, numOccupied, numOccupied);

    if (!gwh.has_value())
    {
        return std::unexpected(gwh.error());
    }

    return gwh->first + gwh->second;
}

qcx::Result<RunOutcome> RunRhfFromDensity(const System& system,
                                          const Eigen::MatrixXd* seed,
                                          bool useDiis,
                                          bool blocked) {
    qcx::scf::RhfOptions options;
    options.useDiis = useDiis;
    options.maxIterations = 400;
    // Isolate the diagonalization path: the full-group labeling stage would
    // SYMMETRIZE the returned density and confound the comparison.
    options.fullGroupLabeling = false;

    if (seed != nullptr)
    {
        options.initialScfState.density = *seed;
    }

    return RunRhf(system, options, blocked && system.groupRealized ? system.basis : nullptr);
}

struct CaseMeasurement {
    RunOutcome plain;
    RunOutcome blocked;
};

qcx::Result<CaseMeasurement> MeasureCase(const System& system,
                                         const Eigen::MatrixXd* seed,
                                         bool useDiis) {
    CaseMeasurement measured;
    auto plain = RunRhfFromDensity(system, seed, useDiis, false);

    if (!plain.has_value())
    {
        return std::unexpected(plain.error());
    }

    measured.plain = std::move(*plain);

    auto blocked = RunRhfFromDensity(system, seed, useDiis, true);

    if (!blocked.has_value())
    {
        return std::unexpected(blocked.error());
    }

    measured.blocked = std::move(*blocked);
    return measured;
}

struct SeedSpec {
    const char* name;
    std::vector<double> perAtom;
};

// The scan's running tally. The defect signature looked for here is a
// case where the BLOCKED path is HIGHER than the plain one - the unrestricted
// defect's shape. The tally is printed, and the cell that runs the scan
// asserts the count is zero over every case it measured.
struct ScanTally {
    int runs = 0;
    int disagreements = 0;
    int blockedHigher = 0;
    double worstBlockedHigher = 0.0;
};

void ReportStartReadings(const std::string& label,
                         const System& system,
                         const std::vector<SeedSpec>& seeds) {
    if (!system.groupRealized)
    {
        return;
    }

    auto gwh = MakeGwhDensity(system);

    if (gwh.has_value())
    {
        const auto reading = AdaptationOfDensity(system, *gwh);

        if (reading.has_value())
        {
            std::printf("[rhf-probe] %-22s START gwh    attn=%.3e\n", label.c_str(), *reading);
        }
    }

    for (const auto& seed : seeds)
    {
        const auto reading = AdaptationOfDensity(system, MakeChargeSeed(seed.perAtom));

        if (reading.has_value() && *reading > qcx::scf::internal::kSymmetryAdaptationTolerance)
        {
            std::printf("[rhf-probe] %-22s START %-11s attn=%.3e  (breaks the group)\n",
                        label.c_str(),
                        seed.name,
                        *reading);
        }
    }
}

void RecordGap(const std::string& label,
               const char* seedName,
               bool useDiis,
               const CaseMeasurement& measured,
               ScanTally& tally) {
    const double gap = measured.blocked.totalEnergy - measured.plain.totalEnergy;
    ++tally.runs;

    if (gap > 1e-9)
    {
        ++tally.blockedHigher;
        tally.worstBlockedHigher = std::max(tally.worstBlockedHigher, gap);
        std::printf("[rhf-probe] %s seed=%s diis=%d: BLOCKED IS HIGHER by %.6e\n",
                    label.c_str(),
                    seedName,
                    useDiis ? 1 : 0,
                    gap);
    }

    if (std::abs(gap) > 1e-9)
    {
        ++tally.disagreements;
        std::printf("[rhf-probe] %s seed=%s diis=%d: plain %+.12f (attn %.3e, it %d) vs blocked "
                    "%+.12f (attn %.3e, it %d)  dE=%+.6e\n",
                    label.c_str(),
                    seedName,
                    useDiis ? 1 : 0,
                    measured.plain.totalEnergy,
                    measured.plain.adaptationNorm,
                    measured.plain.iterations,
                    measured.blocked.totalEnergy,
                    measured.blocked.adaptationNorm,
                    measured.blocked.iterations,
                    gap);
    }
}

void ScanSystem(const std::string& label,
                const System& system,
                const std::vector<SeedSpec>& seeds,
                ScanTally& tally) {
    ReportStartReadings(label, system, seeds);

    for (const auto& seed : seeds)
    {
        const Eigen::MatrixXd start = MakeChargeSeed(seed.perAtom);

        for (const bool useDiis : {false, true})
        {
            auto measured = MeasureCase(system, &start, useDiis);

            if (measured.has_value())
            {
                RecordGap(label, seed.name, useDiis, *measured, tally);
            }
        }
    }

    for (const bool useDiis : {false, true})
    {
        auto measured = MeasureCase(system, nullptr, useDiis);

        if (measured.has_value())
        {
            RecordGap(label, "<gwh>", useDiis, *measured, tally);
        }
    }

    std::fflush(stdout);
}

} // namespace

// The scan: the evidence table the pinned cells below were written from. It
// asserts machinery only, so its output stays readable, plus the one claim of
// this file that is a count rather than a pin - over every case it measured,
// the guarded blocked path never DISAGREED with the plain one. That is the
// ported guard's whole outcome claim: a non-adapted walk is refused, so it IS
// the plain walk, and an adapted walk is blocked, which is the plain answer to
// rounding. A disagreement would mean a Fock slipped past the guard and the
// blocks changed the answer.
TEST(RhfSymmetryConstraintTest, ScanBrokenStartsAcrossFrameworks) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const std::vector<SeedSpec> h4Seeds = {
        {"uniform", {1.0, 1.0, 1.0, 1.0}},
        {"left-pair", {2.0, 2.0, 0.0, 0.0}},
        {"left-atom", {4.0, 0.0, 0.0, 0.0}},
        {"split", {2.0, 0.0, 0.0, 2.0}},
        {"alternating", {2.0, 0.0, 2.0, 0.0}},
    };
    const std::vector<SeedSpec> h6Seeds = {
        {"uniform", {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
        {"left-trio", {2.0, 2.0, 2.0, 0.0, 0.0, 0.0}},
        {"left-atom", {6.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
    };

    ScanTally tally;

    for (const double spacing : {1.4, 2.6, 3.4})
    {
        auto molecule = MakeH4Chain(spacing);

        if (!molecule.has_value())
        {
            continue;
        }

        auto system = OpenSystem(*molecule, *basis);

        if (system.has_value())
        {
            ScanSystem("H4 chain d=" + std::to_string(spacing), *system, h4Seeds, tally);
        }
    }

    for (const double b : {1.4, 2.2})
    {
        auto molecule = MakeH4Rectangle(1.4, b);

        if (!molecule.has_value())
        {
            continue;
        }

        auto system = OpenSystem(*molecule, *basis);

        if (system.has_value())
        {
            ScanSystem("H4 rect 1.4x" + std::to_string(b), *system, h4Seeds, tally);
        }
    }

    for (const double spacing : {2.0, 2.6})
    {
        auto molecule = MakeH6Chain(spacing);

        if (!molecule.has_value())
        {
            continue;
        }

        auto system = OpenSystem(*molecule, *basis);

        if (system.has_value())
        {
            ScanSystem("H6 chain d=" + std::to_string(spacing), *system, h6Seeds, tally);
        }
    }

    auto h2 = MakeH2AtDistance(3.5);

    if (h2.has_value())
    {
        auto system = OpenSystem(*h2, *basis);

        if (system.has_value())
        {
            ScanSystem("H2 R=3.5", *system, {{"left-atom", {2.0, 0.0}}}, tally);
        }
    }

    std::printf("[rhf-probe] TALLY: %d runs, %d disagreements, %d with the blocked path HIGHER "
                "(worst %.3e)\n",
                tally.runs,
                tally.disagreements,
                tally.blockedHigher,
                tally.worstBlockedHigher);
    std::fflush(stdout);

    EXPECT_EQ(tally.blockedHigher, 0)
        << "the blocked path returned a HIGHER energy than the plain one: that is the unrestricted "
           "path's defect shape, and the reason a guard exists there";
    EXPECT_GT(tally.runs, 40);
    EXPECT_EQ(tally.disagreements, 0)
        << "the guarded blocked path returned a different answer from the plain one: a non-adapted "
           "Fock was blocked (a silent constraint) or an adapted one was not (the speedup lost "
           "without cause) - see the reported norms";
}

// THE PIN. Square H4 reaches a non-adapted Fock from the DEFAULT start (the
// start reading below is 9.157e-02 against the 1e-8 tolerance), and the
// ported guard refuses the irrep blocks on it: the blocked run IS the plain
// run, and it returns the plain path's symmetry-BROKEN fixed point at
// -1.633519911629.
//
// The price is pinned as a number rather than argued away. The unguarded
// blocked run of 2026-09-16 returned the ADAPTED solution at -1.719789653762,
// 86.27 mHa below this one. That solution is still REACHABLE - the plain path
// finds it from an adapted start, and the guarded path BLOCKS on that walk -
// so the ruling costs this basin on this start, not the feature.
TEST(RhfSymmetryConstraintTest, DefaultStartGuardRefusesAndReturnsThePlainAnswer) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH4Rectangle(1.4, 1.4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto system = OpenSystem(*molecule, *basis);
    ASSERT_TRUE(system.has_value()) << system.error().message;
    ASSERT_TRUE(system->groupRealized)
        << "square H4 must realize a non-trivial computational group, or the blocked path is not "
           "engaged and this cell proves nothing";

    // Route (b), the one the amendment did not name: the DEFAULT start is
    // already outside the adapted manifold, with nothing caller-supplied.
    auto gwh = MakeGwhDensity(*system);
    ASSERT_TRUE(gwh.has_value()) << gwh.error().message;
    const auto startReading = AdaptationOfDensity(*system, *gwh);
    ASSERT_TRUE(startReading.has_value()) << startReading.error().message;
    EXPECT_GT(*startReading, qcx::scf::internal::kSymmetryAdaptationTolerance * 1e3)
        << "the default GWH start is adapted here: route (b) has closed and this cell's premise "
           "with it";

    auto measured = MeasureCase(*system, nullptr, true);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ReportOutcome("plain", measured->plain);
    ReportOutcome("blocked", measured->blocked);
    ASSERT_TRUE(measured->plain.converged) << measured->plain.iterations;
    ASSERT_TRUE(measured->blocked.converged) << measured->blocked.iterations;

    // The plain walk converges to a symmetry-BROKEN solution: the guard's own
    // measure reads ~1e-1 on its converged Fock, seven orders over tolerance.
    EXPECT_GT(measured->plain.adaptationNorm, 1e-2)
        << "the plain path is expected to converge to a broken fixed point here";
    EXPECT_GT(measured->plain.offBlockFock, 1e-3);

    // The guarded blocked walk refuses on the non-adapted Fock and returns the
    // plain answer - not the adapted one the unguarded run used to find.
    EXPECT_EQ(measured->blocked.blocking.action, qcx::scf::SymmetryBlockingAction::kRefused);
    EXPECT_EQ(measured->blocked.blocking.blockedSolveCount, 0);
    EXPECT_GT(measured->blocked.blocking.plainSolveCount, 0);
    EXPECT_GT(measured->blocked.blocking.maxGeneratorCommutatorNorm,
              measured->blocked.blocking.tolerance);
    EXPECT_NEAR(measured->blocked.totalEnergy, measured->plain.totalEnergy, 1e-9);
    EXPECT_NEAR(measured->blocked.adaptationNorm, measured->plain.adaptationNorm, 1e-12);

    // The numbers themselves, so a change of framework or gate shows up as a
    // number rather than as a direction.
    EXPECT_NEAR(measured->plain.totalEnergy, -1.633519911629, 1e-8);
    EXPECT_NEAR(measured->blocked.totalEnergy, -1.633519911629, 1e-8);

    // The adapted solution is still reachable and still blocked: the SAME
    // framework from an adapted start converges to -1.719789653762 on both
    // paths, and the guarded blocked run uses the irrep blocks on it. This is
    // the control that keeps the refusal from reading as "blocking is dead".
    const Eigen::MatrixXd adaptedStart = MakeChargeSeed({1.0, 1.0, 1.0, 1.0});
    auto plainFromAdapted = RunRhfFromDensity(*system, &adaptedStart, true, false);
    auto blockedFromAdapted = RunRhfFromDensity(*system, &adaptedStart, true, true);
    ASSERT_TRUE(plainFromAdapted.has_value()) << plainFromAdapted.error().message;
    ASSERT_TRUE(blockedFromAdapted.has_value()) << blockedFromAdapted.error().message;
    ReportOutcome("plain  (adapted start)", *plainFromAdapted);
    ReportOutcome("blocked (adapted start)", *blockedFromAdapted);
    ASSERT_TRUE(plainFromAdapted->converged);
    ASSERT_TRUE(blockedFromAdapted->converged);
    EXPECT_LT(plainFromAdapted->adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);
    EXPECT_NEAR(plainFromAdapted->totalEnergy, -1.719789653762, 1e-8);
    EXPECT_EQ(blockedFromAdapted->blocking.action, qcx::scf::SymmetryBlockingAction::kUsed);
    EXPECT_GT(blockedFromAdapted->blocking.blockedSolveCount, 0);
    EXPECT_EQ(blockedFromAdapted->blocking.plainSolveCount, 0);
    EXPECT_NEAR(blockedFromAdapted->totalEnergy, plainFromAdapted->totalEnergy, 1e-11);
}

// Each basin is a fixed point of the map, and with the guard each path KEEPS
// the basin it is started in: the plain path stays where it is; the blocked
// path refuses a broken Fock (so it IS the plain walk) and blocks an adapted
// one (so it is the plain answer to rounding). The unguarded blocked path used
// to walk from the broken solution to the adapted one, which is exactly the
// constraint the guard removes.
TEST(RhfSymmetryConstraintTest, RestartLegsKeepTheirStartingBasin) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH4Rectangle(1.4, 1.4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto system = OpenSystem(*molecule, *basis);
    ASSERT_TRUE(system.has_value()) << system.error().message;
    ASSERT_TRUE(system->groupRealized);

    // The two solutions on this framework: the broken one the default start
    // finds, and the adapted one an adapted start finds.
    auto measured = MeasureCase(*system, nullptr, true);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ASSERT_TRUE(measured->plain.converged);
    ASSERT_TRUE(measured->blocked.converged);

    const Eigen::MatrixXd adaptedStart = MakeChargeSeed({1.0, 1.0, 1.0, 1.0});
    auto adapted = RunRhfFromDensity(*system, &adaptedStart, true, false);
    ASSERT_TRUE(adapted.has_value()) << adapted.error().message;
    ASSERT_TRUE(adapted->converged);
    ASSERT_LT(adapted->adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance)
        << "the adapted start did not reach an adapted solution, so this cell has one basin "
           "only";

    struct Leg {
        const char* name;
        const Eigen::MatrixXd* density;
        double energy;
        bool adapted;
    };

    for (const auto& leg :
         {Leg{"broken", &measured->plain.density, measured->plain.totalEnergy, false},
          Leg{"adapted", &adapted->density, adapted->totalEnergy, true}})
    {
        auto plainAgain = RunRhfFromDensity(*system, leg.density, true, false);
        auto blockedAgain = RunRhfFromDensity(*system, leg.density, true, true);
        ASSERT_TRUE(plainAgain.has_value()) << plainAgain.error().message;
        ASSERT_TRUE(blockedAgain.has_value()) << blockedAgain.error().message;
        ReportOutcome(leg.name, *plainAgain);
        ReportOutcome(leg.name, *blockedAgain);

        ASSERT_TRUE(plainAgain->converged);
        ASSERT_TRUE(blockedAgain->converged);
        EXPECT_NEAR(plainAgain->totalEnergy, leg.energy, 1e-9)
            << "the solution is not a fixed point of the plain map";
        EXPECT_NEAR(blockedAgain->totalEnergy, plainAgain->totalEnergy, 1e-9)
            << "the guarded blocked path left the basin the plain path stayed in";

        if (leg.adapted)
        {
            EXPECT_EQ(blockedAgain->blocking.action, qcx::scf::SymmetryBlockingAction::kUsed);
            EXPECT_GT(blockedAgain->blocking.blockedSolveCount, 0);
            EXPECT_EQ(blockedAgain->blocking.plainSolveCount, 0);
        } else
        {
            EXPECT_EQ(blockedAgain->blocking.action, qcx::scf::SymmetryBlockingAction::kRefused);
            EXPECT_EQ(blockedAgain->blocking.blockedSolveCount, 0);
            EXPECT_GT(blockedAgain->blocking.plainSolveCount, 0);
        }
    }
}

// The other half, and the control that keeps the finding from being read as
// "the restricted path is broken": on a framework whose restricted solution IS
// symmetry-adapted, the default start is adapted, both paths agree, the
// converged Fock is adapted, AND the guard says so - the blocked run reports
// kUsed with every solve blocked. Engagement is the report's business now: the
// two paths agree by construction under the guard, so a silent fallback can
// only be caught by reading the disclosure, which is what this cell does.
TEST(RhfSymmetryConstraintTest, HealthyRectangularFrameworkAgreesAndStaysAdapted) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH4Rectangle(1.4, 2.2);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto system = OpenSystem(*molecule, *basis);
    ASSERT_TRUE(system.has_value()) << system.error().message;
    ASSERT_TRUE(system->groupRealized);

    auto gwh = MakeGwhDensity(*system);
    ASSERT_TRUE(gwh.has_value()) << gwh.error().message;
    const auto startReading = AdaptationOfDensity(*system, *gwh);
    ASSERT_TRUE(startReading.has_value()) << startReading.error().message;
    EXPECT_LT(*startReading, qcx::scf::internal::kSymmetryAdaptationTolerance);

    auto measured = MeasureCase(*system, nullptr, true);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ReportOutcome("plain", measured->plain);
    ReportOutcome("blocked", measured->blocked);
    ASSERT_TRUE(measured->plain.converged);
    ASSERT_TRUE(measured->blocked.converged);
    EXPECT_NEAR(measured->blocked.totalEnergy, measured->plain.totalEnergy, 1e-11);
    EXPECT_LT(measured->blocked.adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);

    // The default walk stays adapted, so every solve takes the irrep blocks and
    // the guard never falls back.
    EXPECT_EQ(measured->blocked.blocking.action, qcx::scf::SymmetryBlockingAction::kUsed);
    EXPECT_GT(measured->blocked.blocking.blockedSolveCount, 0);
    EXPECT_EQ(measured->blocked.blocking.plainSolveCount, 0);
    EXPECT_EQ(measured->blocked.blocking.tolerance,
              qcx::scf::internal::kSymmetryAdaptationTolerance);
    EXPECT_LT(measured->blocked.blocking.maxGeneratorCommutatorNorm,
              qcx::scf::internal::kSymmetryAdaptationTolerance);
    // The plain run requested nothing, and its disclosure says so.
    EXPECT_EQ(measured->plain.blocking.action, qcx::scf::SymmetryBlockingAction::kNotRequested);
    EXPECT_EQ(measured->plain.blocking.blockedSolveCount, 0);

    // A broken caller-supplied start on the same framework: the run relaxes to
    // that same adapted solution on both paths, so the exposed route
    // (initialScfState) is harmless wherever the solution is adapted - which is
    // why the finding above needed a framework with a degenerate HOMO to
    // appear. The guard DEMOTES on the way: it refuses the broken opening Fock,
    // then blocks once the walk enters the adapted manifold.
    const Eigen::MatrixXd broken = MakeChargeSeed({4.0, 0.0, 0.0, 0.0});
    auto fromBroken = MeasureCase(*system, &broken, true);
    ASSERT_TRUE(fromBroken.has_value()) << fromBroken.error().message;
    ReportOutcome("plain  (broken start)", fromBroken->plain);
    ReportOutcome("blocked (broken start)", fromBroken->blocked);
    ASSERT_TRUE(fromBroken->plain.converged) << fromBroken->plain.iterations;
    ASSERT_TRUE(fromBroken->blocked.converged) << fromBroken->blocked.iterations;
    EXPECT_NEAR(fromBroken->blocked.totalEnergy, measured->plain.totalEnergy, 1e-9);
    EXPECT_LT(fromBroken->blocked.adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);
    EXPECT_EQ(fromBroken->blocked.blocking.action, qcx::scf::SymmetryBlockingAction::kDemoted);
    EXPECT_GT(fromBroken->blocked.blocking.plainSolveCount, 0);
    EXPECT_GT(fromBroken->blocked.blocking.blockedSolveCount, 0);
    EXPECT_GT(fromBroken->blocked.blocking.maxGeneratorCommutatorNorm,
              fromBroken->blocked.blocking.tolerance)
        << "the demotion must report the broken opening Fock's norm, not a later adapted one";
}

// THE APPLIED-FIELD CELL: the defect that triggered the port, reproduced on
// the restricted path. H2/STO-3G realises a non-trivial computational group,
// and a uniform field along the bond puts a sigma_u term in the core: the
// dipole operator couples sigma_g and sigma_u, so every Fock in the field
// carries an off-block part of order the field. The unguarded blocked solve
// kept only the diagonal irrep blocks, i.e. it projected the perturbation
// straight back out and converged to the FIELD-FREE density on a run that
// never saw the field - the finite-field polarizability came back zero in
// response/tests/response_kernel_test.cpp.
//
// With the guard the run refuses and returns the polarized density. The two
// halves of "not projected out" are asserted separately: the converged
// DENSITY moves with the field, and the DISCLOSURE says the fallback was
// taken. The field-free control shows the same run still takes the blocks when
// nothing breaks the symmetry.
TEST(RhfSymmetryConstraintTest, AppliedFieldIsNotProjectedOut) {
    constexpr double kField = 1e-2;

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2AtDistance(1.4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto system = OpenSystem(*molecule, *basis);
    ASSERT_TRUE(system.has_value()) << system.error().message;
    ASSERT_TRUE(system->groupRealized)
        << "H2/STO-3G must realize a non-trivial computational group, or the blocked path is not "
           "engaged and this cell proves nothing";

    auto dipole = qcx::integrals::BuildDipoleMatrix(*molecule, *basis);
    ASSERT_TRUE(dipole.has_value()) << dipole.error().message;
    const Eigen::MatrixXd fieldCore =
        system->oneElectron.coreHamiltonian + kField * ToMatrix((*dipole)[0]);

    qcx::scf::RhfOptions options;
    options.useDiis = true;
    options.maxIterations = 400;
    options.fullGroupLabeling = false;

    auto fieldFree = RunRhf(*system, options, system->basis);
    auto fieldPlain = RunRhfWithCore(*system, fieldCore, options, nullptr);
    auto fieldBlocked = RunRhfWithCore(*system, fieldCore, options, system->basis);
    ASSERT_TRUE(fieldFree.has_value()) << fieldFree.error().message;
    ASSERT_TRUE(fieldPlain.has_value()) << fieldPlain.error().message;
    ASSERT_TRUE(fieldBlocked.has_value()) << fieldBlocked.error().message;
    ReportOutcome("field-free blocked", *fieldFree);
    ReportOutcome("field plain", *fieldPlain);
    ReportOutcome("field blocked", *fieldBlocked);
    ASSERT_TRUE(fieldFree->converged);
    ASSERT_TRUE(fieldPlain->converged);
    ASSERT_TRUE(fieldBlocked->converged);

    // THE DENSITY. The field moves it, and the guarded blocked path goes where
    // the plain path goes. The first assertion is the one the defect failed:
    // the unguarded blocked run returned exactly the field-free density.
    const double fieldMoved = (fieldBlocked->density - fieldFree->density).cwiseAbs().maxCoeff();
    EXPECT_GT(fieldMoved, 1e-3)
        << "the blocked run returned the field-free density: the applied field was projected out";
    EXPECT_NEAR((fieldBlocked->density - fieldPlain->density).cwiseAbs().maxCoeff(), 0.0, 1e-9)
        << "the guarded blocked run is not the plain run: a field-broken Fock was blocked";
    EXPECT_NEAR(fieldBlocked->totalEnergy, fieldPlain->totalEnergy, 1e-9);
    // The FIELD-POLARIZED solution is genuinely not group-adapted; a converged
    // Fock that still commuted would mean the field was never in it.
    EXPECT_GT(fieldBlocked->adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);

    // THE DISCLOSURE. In the field the guard refuses every solve and says so...
    EXPECT_EQ(fieldBlocked->blocking.action, qcx::scf::SymmetryBlockingAction::kRefused);
    EXPECT_EQ(fieldBlocked->blocking.blockedSolveCount, 0);
    EXPECT_GT(fieldBlocked->blocking.plainSolveCount, 0);
    EXPECT_EQ(fieldBlocked->blocking.tolerance, qcx::scf::internal::kSymmetryAdaptationTolerance);
    EXPECT_GT(fieldBlocked->blocking.maxGeneratorCommutatorNorm, fieldBlocked->blocking.tolerance);

    // ... and with no field the same route blocks every solve.
    EXPECT_EQ(fieldFree->blocking.action, qcx::scf::SymmetryBlockingAction::kUsed);
    EXPECT_GT(fieldFree->blocking.blockedSolveCount, 0);
    EXPECT_EQ(fieldFree->blocking.plainSolveCount, 0);
    EXPECT_LT(fieldFree->adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);
}
