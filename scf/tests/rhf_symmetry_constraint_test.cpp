// THE GUARD ON THE RESTRICTED PATH: measured, and NOT ported.
//
// The unrestricted path carries a guard (uhf.cpp's diagonalizeFock,
// d0b5ee3c): each iteration measures whether its Fock still commutes with the
// point group and refuses the irrep blocking when it does not. The restricted
// path has the same blocked/plain site (rhf.cpp's diagonalizeFock) and no
// guard, and the amendment left that
// as the open item, naming RhfOptions::initialScfState as the exposure.
//
// This file is that measurement. Its answer is a finding, in two parts.
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
// TWO - the guard's premise inverts here, so porting it would make the
// restricted path WORSE, not safer. the guard exists because blocking a
// non-adapted Fock PROJECTS OUT the solution that Fock belongs to, and on the
// unrestricted path that solution is the lower one (an RHF -> UHF instability
// is exactly a lower broken solution). On the restricted path the measured
// relation is the other way round: where the two paths disagree, the blocked
// path returns the LOWER solution and the plain path is the one that lands
// above it. Square H4 is the smallest case, and it is reachable from the
// default start: the plain walk converges to a symmetry-BROKEN fixed point at
// -1.633519911629 (attn 1.057e-01) while the blocked walk returns an ADAPTED
// one at -1.719789653762 (attn 4.358e-16), 86.27 mHa LOWER - and restarting
// the PLAIN path from that adapted solution stays there (dE -1.3e-15), so
// both are fixed points of one map and blocking is what selects the better.
//
// Ported verbatim (measure, refuse over tolerance - the unrestricted path's
// rule), the guard changed 21 of the 126 runs this file's scan measured and
// returned a HIGHER energy in every one of them, by 86 mHa up to 2.63 Ha: in
// each, the guarded blocked run reproduced the plain run exactly (square H4
// default start: -1.633519911629, dE 0.0, attn 1.057e-01). The cell below is
// the pin that fails if that port is ever made.
//
// The seam is the existing RunRhfScf basisSet argument (nullptr = plain) and
// the public EvaluateRhfDensityEnergy Fock builder; nothing is monkey-patched.
// The guard's own reading of a Fock - what the unrestricted path's guard would
// decide on - is taken through the internal seam
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
};

// Runs RHF through the one Fock builder, choosing the diagonalization path
// with the basisSet argument alone (nullptr = the plain n x n solve).
qcx::Result<RunOutcome> RunRhf(const System& system,
                               const qcx::scf::RhfOptions& options,
                               const qcx::basisset::BasisSet* basisSet) {
    const qcx::scf::FockBuilderFn builder =
        [&system](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto evaluated = qcx::scf::EvaluateRhfDensityEnergy(
            *system.molecule, system.oneElectron.coreHamiltonian, *system.eri, density);

        if (!evaluated.has_value())
        {
            return std::unexpected(evaluated.error());
        }

        return evaluated->fock;
    };

    auto result = qcx::scf::RunRhfScf(*system.molecule,
                                      system.oneElectron.overlap,
                                      system.oneElectron.coreHamiltonian,
                                      options,
                                      builder,
                                      basisSet);

    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }

    RunOutcome outcome;
    outcome.totalEnergy = result->totalEnergy;
    outcome.iterations = result->iterations;
    outcome.converged = result->converged;
    outcome.density = result->density;

    auto reEvaluated = qcx::scf::EvaluateRhfDensityEnergy(
        *system.molecule, system.oneElectron.coreHamiltonian, *system.eri, outcome.density);

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

// The default start the loop uses when no restart state is supplied, as a
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

// The scan's running tally. The defect signature this scan looks for is a
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
// the blocked path was never the HIGHER one.
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
    EXPECT_GT(tally.disagreements, 0) << "the frameworks in this scan must exercise a disagreement";
}

// The restart legs: both paths re-entered from each converged solution. A
// solution the plain path stays in while the blocked path walks away from is
// the defect the guard exists for. A solution BOTH paths keep - with the
// plain path's basin chosen by its start and the blocked path landing on the
// lower one - is the measured situation here, and it is what the guard would
// destroy.
TEST(RhfSymmetryConstraintTest, ScanRestartLegsOnTheDisagreeingFramework) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH4Rectangle(1.4, 1.4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto system = OpenSystem(*molecule, *basis);
    ASSERT_TRUE(system.has_value()) << system.error().message;
    ASSERT_TRUE(system->groupRealized);

    auto measured = MeasureCase(*system, nullptr, true);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;

    for (const auto* solution : {&measured->plain, &measured->blocked})
    {
        auto plainAgain = RunRhfFromDensity(*system, &solution->density, true, false);
        auto blockedAgain = RunRhfFromDensity(*system, &solution->density, true, true);
        ASSERT_TRUE(plainAgain.has_value()) << plainAgain.error().message;
        ASSERT_TRUE(blockedAgain.has_value()) << blockedAgain.error().message;

        std::printf("[rhf-probe] from E=%+.12f (attn %.3e): plain stays %+.12f (attn %.3e), "
                    "blocked goes to %+.12f (attn %.3e)\n",
                    solution->totalEnergy,
                    solution->adaptationNorm,
                    plainAgain->totalEnergy,
                    plainAgain->adaptationNorm,
                    blockedAgain->totalEnergy,
                    blockedAgain->adaptationNorm);
    }

    std::fflush(stdout);
}

// THE PIN. Square H4: the restricted path reaches a non-adapted Fock from the
// DEFAULT start, and the blocked diagonalization then returns the LOWER
// solution - the opposite of the unrestricted path's defect, and the reason
// the guard was not ported.
//
// If the unrestricted path's guard is ever ported to rhf.cpp's diagonalizeFock
// (measure with MeasureSymmetryAdaptation, refuse over
// kSymmetryAdaptationTolerance), this cell goes RED on both legs at once: the
// blocked run refuses at its first Fock - the start reading below is already
// over tolerance - follows the plain path, and returns -1.633519911629 with
// attn 1.057e-01 instead, a HIGHER energy by 86.27 mHa. Measured 2026-09-16.
TEST(RhfSymmetryConstraintTest, DefaultStartLeavesTheAdaptedManifoldAndBlockingReturnsTheLower) {
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
    ASSERT_TRUE(measured->plain.converged) << measured->plain.iterations;
    ASSERT_TRUE(measured->blocked.converged) << measured->blocked.iterations;

    // The plain walk converges to a symmetry-BROKEN solution: the guard's own
    // measure reads ~1e-1 on its converged Fock, seven orders over tolerance.
    EXPECT_GT(measured->plain.adaptationNorm, 1e-2)
        << "the plain path is expected to converge to a broken fixed point here";
    EXPECT_GT(measured->plain.offBlockFock, 1e-3);

    // The blocked walk returns an ADAPTED solution, and it is the LOWER one.
    // Both assertions together are the pin: a ported guard fails the first (its
    // refusal makes the blocked run the plain run) and the second with it.
    EXPECT_LT(measured->blocked.adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance)
        << "the blocked path returned a non-adapted solution: blocking would be constraining the "
           "search here, which is the unrestricted path's defect and not the measured situation on "
           "the restricted path";
    EXPECT_LT(measured->blocked.totalEnergy, measured->plain.totalEnergy - 1e-3)
        << "the blocked path no longer returns the lower solution: if the guard was ported to "
           "rhf.cpp's diagonalizeFock, this is the 86.27 mHa regression it causes";

    // The numbers themselves, so a change of framework or gate shows up as a
    // number rather than as a direction.
    EXPECT_NEAR(measured->plain.totalEnergy, -1.633519911629, 1e-8);
    EXPECT_NEAR(measured->blocked.totalEnergy, -1.719789653762, 1e-8);
}

// Both solutions are fixed points of the SAME map, and blocking is what picks
// the better basin: the plain path re-entered from the blocked path's adapted
// solution stays there (-1.719789653762, to rounding), and the blocked path
// re-entered from the plain path's broken solution moves to the adapted one.
// The blocked path is therefore not "the same answer, faster": on this
// framework it is the difference between the RHF minimum and a solution 86.27
// mHa above it.
TEST(RhfSymmetryConstraintTest, BothSolutionsArePlainPathFixedPointsAndBlockingPicksTheLower) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH4Rectangle(1.4, 1.4);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto system = OpenSystem(*molecule, *basis);
    ASSERT_TRUE(system.has_value()) << system.error().message;
    ASSERT_TRUE(system->groupRealized);

    auto measured = MeasureCase(*system, nullptr, true);
    ASSERT_TRUE(measured.has_value()) << measured.error().message;
    ASSERT_TRUE(measured->plain.converged);
    ASSERT_TRUE(measured->blocked.converged);
    ASSERT_LT(measured->blocked.totalEnergy, measured->plain.totalEnergy);

    // The broken solution is a fixed point of the plain map.
    auto plainFromBroken = RunRhfFromDensity(*system, &measured->plain.density, true, false);
    ASSERT_TRUE(plainFromBroken.has_value()) << plainFromBroken.error().message;
    ASSERT_TRUE(plainFromBroken->converged);
    EXPECT_NEAR(plainFromBroken->totalEnergy, measured->plain.totalEnergy, 1e-9);
    EXPECT_GT(plainFromBroken->adaptationNorm, 1e-2);

    // The adapted solution is a fixed point of the plain map TOO - so the
    // plain path's answer is a basin selection, not the map's only solution.
    auto plainFromAdapted = RunRhfFromDensity(*system, &measured->blocked.density, true, false);
    ASSERT_TRUE(plainFromAdapted.has_value()) << plainFromAdapted.error().message;
    ASSERT_TRUE(plainFromAdapted->converged);
    EXPECT_NEAR(plainFromAdapted->totalEnergy, measured->blocked.totalEnergy, 1e-9);
    EXPECT_LT(plainFromAdapted->adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);

    // And the blocked path walks from the broken solution to the adapted one.
    auto blockedFromBroken = RunRhfFromDensity(*system, &measured->plain.density, true, true);
    ASSERT_TRUE(blockedFromBroken.has_value()) << blockedFromBroken.error().message;
    ASSERT_TRUE(blockedFromBroken->converged);
    EXPECT_NEAR(blockedFromBroken->totalEnergy, measured->blocked.totalEnergy, 1e-9);
    EXPECT_LT(blockedFromBroken->adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);
}

// The other half, and the control that keeps the finding from being read as
// "the restricted path is broken": on a framework whose restricted solution IS
// symmetry-adapted, the default start is adapted (a guard would not fire), the
// two paths agree, and the converged Fock is adapted. The engagement evidence
// is not a self-report here - it is the cell above, where the two paths'
// answers can only differ because blocking ran.
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
    ASSERT_TRUE(measured->plain.converged);
    ASSERT_TRUE(measured->blocked.converged);
    EXPECT_NEAR(measured->blocked.totalEnergy, measured->plain.totalEnergy, 1e-11);
    EXPECT_LT(measured->blocked.adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);

    // A broken caller-supplied start on the same framework: the run relaxes to
    // that same adapted solution on both paths, so the exposed route
    // (initialScfState) is harmless wherever the solution is adapted - which is
    // why the finding above needed a framework with a degenerate HOMO to
    // appear.
    const Eigen::MatrixXd broken = MakeChargeSeed({4.0, 0.0, 0.0, 0.0});
    auto fromBroken = MeasureCase(*system, &broken, true);
    ASSERT_TRUE(fromBroken.has_value()) << fromBroken.error().message;
    ASSERT_TRUE(fromBroken->plain.converged) << fromBroken->plain.iterations;
    ASSERT_TRUE(fromBroken->blocked.converged) << fromBroken->blocked.iterations;
    EXPECT_NEAR(fromBroken->blocked.totalEnergy, measured->plain.totalEnergy, 1e-9);
    EXPECT_LT(fromBroken->blocked.adaptationNorm, qcx::scf::internal::kSymmetryAdaptationTolerance);
}
