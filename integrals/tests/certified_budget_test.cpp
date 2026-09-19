// The certified fp32 error-bound budget enforcement: the lane's accumulated
// bound is COMPARED against a preset-derived budget and the comparison
// DECIDES the routing.
//
// The gap it closes: the production path computed a bound and discarded it -
// "a bound-routed heuristic with an unused diagnostic". The
// enforcement is the fix's second half: one walk of the routing pass's own
// neighbor list accumulates the admitted candidates' certified bounds (and
// the screened-out ones', the budget's B_screen co-term), the sum is
// compared against CertifiedBoundBudget(accuracy, B_screen), and a build
// that does not fit runs the fp64 lane alone.
//
// THE ACT IS A DEGENERATION, NOT A MIXTURE. The fall-back revokes the lane
// BEFORE the screening pass runs, so every candidate lands in the fp64
// master exactly as it does with the lane switched off, and the build that
// follows is the lane-disabled build - the same quartet set on the same
// code path, so the two agree BIT-EXACTLY since the fixed-order join
// (internal/fixed_order_reduce.hpp); the test measures that agreement
// rather than assuming it. The first test pins it: no partly-fp32 build can
// exist, so the enforcement can never invent an intermediate precision the
// certification story does not cover.
//
// THE MEASUREMENT IS THE POINT. Two of the tests print the comparison's own
// numbers on a real fixture (C4H10/def2-SVP, 106 BF): the derived budget,
// the routed sum, and the verdict - once at kNormal, once across the whole
// preset ladder, because "the criterion is tight at kNormal" and "the lane
// is dead at every preset" are different findings. They are the evidence
// for whether the enforcement leaves the lane alive or turns it off, and
// the file asserts only what must hold whatever the numbers say (the budget
// is derived, the comparison is the one that decided, the count is the
// routing pass's own).

#include "alkane_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/precision_policy.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <utility>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

/// The fixture's carbon count: C(4)H(10)/def2-SVP, 106 basis functions -
/// the smallest alkane whose kNormal build routes fp32 quartets at
/// def2-SVP (the driver's STO-3G fixture needs only four carbons because
/// its basis is smaller).
constexpr std::size_t kFixtureCarbons = 4;

/// The enforcement fixture: molecule, basis, H = T + V and S - the same
/// construction the sibling harnesses use, minus everything this file does
/// not read.
struct EnforcementFixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    CpuTensor2 core;
    Eigen::MatrixXd coreH;
    Eigen::MatrixXd overlap;
    std::size_t occupiedCount = 0;
};

/// The def2-SVP orbital basis from the vendored corpus.
/// \returns The merged C + H def2-SVP basis, or an Error.
qcx::Result<qcx::basisset::BasisSet> MakeDef2SvpBasis() {
    const std::array<int, 2> elements{6, 1}; // C, H.
    const std::string directory = std::string(QcxBasisDataDir) + "/def2-svp";
    return qcx::basisset::ParseNwchemDirectoryFiltered(directory, elements);
}

/// H = T + V from the one-electron engines.
/// \param molecule The molecule.
/// \param basisSet The orbital basis.
/// \returns The core Hamiltonian, or an Error.
qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
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

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return std::move(*kinetic);
}

/// S^-1/2 through the overlap's eigenelements.
/// \param overlap The overlap matrix.
/// \returns X = S^-1/2, or an Error when the overlap is not positive definite.
qcx::Result<Eigen::MatrixXd> InverseSquareRoot(const Eigen::MatrixXd& overlap) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap);

    if (solver.info() != Eigen::Success || solver.eigenvalues()(0) <= 0.0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInternalError, "non-positive overlap eigenvalue"});
    }

    return solver.eigenvectors() * solver.eigenvalues().cwiseSqrt().cwiseInverse().asDiagonal() *
           solver.eigenvectors().transpose();
}

/// The occupied SPATIAL density rho = C_occ C_occ^T of a Fock matrix (the
/// builders' input convention - BuildFock takes rho = D/2).
/// \param fock The Fock matrix to diagonalize.
/// \param x S^-1/2.
/// \param occupiedCount The number of doubly-occupied orbitals.
/// \returns The spatial density.
Eigen::MatrixXd OccupiedDensity(const Eigen::MatrixXd& fock,
                                const Eigen::MatrixXd& x,
                                std::size_t occupiedCount) {
    const Eigen::MatrixXd transformed = x.transpose() * fock * x;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(transformed);
    const Eigen::MatrixXd coefficients = x * solver.eigenvectors();
    const Eigen::MatrixXd occupied =
        coefficients.leftCols(static_cast<Eigen::Index>(occupiedCount));
    return occupied * occupied.transpose();
}

/// Builds the enforcement fixture.
/// \returns The fixture, or the first Error of the chain.
qcx::Result<EnforcementFixture> MakeFixture() {
    auto molecule = MakeAlkaneSto3g(kFixtureCarbons);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto basis = MakeDef2SvpBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const Eigen::MatrixXd coreH = ToMatrix(*core);
    return EnforcementFixture{std::move(*molecule),
                              std::move(*basis),
                              std::move(*core),
                              coreH,
                              ToMatrix(*overlap),
                              (6 * kFixtureCarbons + (2 * kFixtureCarbons + 2)) / 2};
}

/// The build options of one probe: the given preset with the two switches
/// under test spelled explicitly.
/// \param preset The accuracy preset.
/// \param certifiedFp32 Route quartets through the certified lane.
/// \param enforce Run the global budget enforcement.
/// \returns The options.
qcx::integrals::FockBuildOptions PresetOptions(qcx::integrals::AccuracyPreset preset,
                                               bool certifiedFp32,
                                               bool enforce) {
    qcx::integrals::FockBuildOptions options;
    options.accuracy = preset;
    options.useCertifiedMixedPrecision = certifiedFp32;
    options.enforceCertifiedBoundBudget = enforce;
    return options;
}

/// The build options of one probe: kNormal (the production preset) with
/// the two switches under test spelled explicitly.
/// \param certifiedFp32 Route quartets through the certified lane.
/// \param enforce Run the global budget enforcement.
/// \returns The options.
qcx::integrals::FockBuildOptions ProbeOptions(bool certifiedFp32, bool enforce) {
    return PresetOptions(qcx::integrals::AccuracyPreset::kNormal, certifiedFp32, enforce);
}

/// One fixed-density build's observables: the Fock matrix, the delivered
/// bound sum, the fp32 task count, and the enforcement's outcome.
struct ProbeOutcome {
    Eigen::MatrixXd fock;
    double deliveredBoundHa = 0.0;
    std::size_t fp32Quartets = 0;
    std::size_t fp64Quartets = 0;
    qcx::integrals::CertifiedBudgetOutcome budget;
};

/// One fixed-density BuildFock on the fixture's core-H-guess density:
/// identical input density for every probe, so the Fock matrices are
/// directly comparable element by element.
/// \param fixture The fixture.
/// \param density The (fixed) density to contract.
/// \param options The build options.
/// \returns The probe's outcome, or an Error.
qcx::Result<ProbeOutcome> Probe(const EnforcementFixture& fixture,
                                const Eigen::MatrixXd& density,
                                const qcx::integrals::FockBuildOptions& options) {
    auto builder = qcx::integrals::DirectJkFockBuilder::Create(
        fixture.molecule, fixture.basis, fixture.core, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    qcx::integrals::FockBuildStats stats;
    double deliveredBound = 0.0;
    qcx::integrals::CertifiedBudgetOutcome budget;
    auto built = builder->BuildFock(*densityTensor, &deliveredBound, &stats, nullptr, &budget);

    if (!built.has_value())
    {
        return std::unexpected(built.error());
    }

    return ProbeOutcome{
        ToMatrix(*built), deliveredBound, stats.fp32QuartetCount, stats.fp64QuartetCount, budget};
}

// The act: a build whose routed sum does not fit the derived budget runs
// the fp64 lane alone, and that build IS the lane-disabled build - the
// same quartet set on the same code path, since the fall-back revokes the
// lane BEFORE the screening pass. The assertion is that the two Fock
// matrices differ by no more than the builder's OWN run-to-run
// reproducibility, measured in this test rather than assumed. Since the
// fixed-order join (internal/fixed_order_reduce.hpp) that
// reproducibility IS bit identity: two builds of one density on one code
// path agree exactly, and the reference below measures that rather than
// sampling a last-bit floor.
//
// FALSIFIABILITY: if the fall-back is reverted to a no-op (the comparison
// computed but not acted on), the enforced probe keeps its fp32 quartets
// and its Fock matrix moves by the fp32 lane's own deviation at this
// fixture (~1.7e-9) - six orders above the
// floor this test measures, so both the quartets == 0 and the
// separation assertions go red.
TEST(CertifiedBudgetTest, TheFallBackIsTheLaneDisabledBuild) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto fixtureResult = MakeFixture();
    ASSERT_TRUE(fixtureResult.has_value()) << fixtureResult.error().message;
    const EnforcementFixture& fixture = *fixtureResult;
    auto xResult = InverseSquareRoot(fixture.overlap);
    ASSERT_TRUE(xResult.has_value());
    const Eigen::MatrixXd density = OccupiedDensity(fixture.coreH, *xResult, fixture.occupiedCount);

    auto enforced = Probe(fixture, density, ProbeOptions(true, true));
    ASSERT_TRUE(enforced.has_value()) << enforced.error().message;

    // The route admits the enforcement (the FastPath is the default mode
    // at this size with no workspace budget): the outcome is a
    // measurement, never the zero defaults.
    ASSERT_TRUE(enforced->budget.enforced);
    EXPECT_GT(enforced->budget.routedQuartets, 0u)
        << "the fixture routes no fp32 quartet at kNormal: this test cannot "
           "exercise the comparison";

    // The comparison the act ran on: the same sum, the same budget, the
    // verdict the builder took.
    EXPECT_DOUBLE_EQ(enforced->budget.budgetHa,
                     qcx::integrals::CertifiedBoundBudget(qcx::integrals::AccuracyPreset::kNormal,
                                                          enforced->budget.screenedHa));
    EXPECT_EQ(enforced->budget.fellBackToFp64,
              enforced->budget.routedHa > enforced->budget.budgetHa);

    if (!enforced->budget.fellBackToFp64)
    {
        GTEST_SKIP() << "the routed sum fits the budget on this fixture: the act cannot be "
                        "exercised";
    }

    // The act, observed directly: the build routed nothing and delivered
    // no bound (0.0 is the true zero here, not a lost measurement).
    EXPECT_EQ(enforced->fp32Quartets, 0u);
    EXPECT_DOUBLE_EQ(enforced->deliveredBoundHa, 0.0);

    // The reference: two lane-disabled builds of the same density. Their
    // spread is this builder's reproducibility - asserted exactly zero,
    // because since the fixed-order join two builds of one density on
    // one code path ARE bit-identical (internal/fixed_order_reduce.hpp).
    // This row is the determinism pin, not a sampled tolerance.
    auto laneOffA = Probe(fixture, density, ProbeOptions(false, false));
    ASSERT_TRUE(laneOffA.has_value()) << laneOffA.error().message;
    auto laneOffB = Probe(fixture, density, ProbeOptions(false, false));
    ASSERT_TRUE(laneOffB.has_value()) << laneOffB.error().message;
    const double floor = (laneOffA->fock - laneOffB->fock).cwiseAbs().maxCoeff();
    const double delta = (enforced->fock - laneOffA->fock).cwiseAbs().maxCoeff();

    EXPECT_DOUBLE_EQ(floor, 0.0)
        << "the direct build's reduction order is not deterministic again";

    std::printf("[certified budget] fall-back: fp64_quartets=%zu (lane-off %zu), "
                "max|delta|=%.3e, reproducibility floor=%.3e\n",
                enforced->fp64Quartets,
                laneOffA->fp64Quartets,
                delta,
                floor);

    EXPECT_EQ(enforced->fp64Quartets, laneOffA->fp64Quartets);
    EXPECT_EQ(enforced->fock.rows(), laneOffA->fock.rows());
    // The measured agreement itself, against the deterministic builder's
    // zero reproducibility: the fall-back is the lane-disabled build, so
    // it must reproduce it exactly. The 1e-14 ceiling is kept as the bound
    // that carries the claim if a future arithmetic path admits a
    // last-bit difference - it sits four orders under the fp32 lane's own
    // deviation below.
    EXPECT_LE(delta, 1e-14) << "the fall-back diverges from the lane-disabled build: max |delta| "
                            << delta << " against the deterministic builder's zero floor";

    // The separation that makes the assertion above meaningful: a fall-back
    // that was NOT taken would leave the fp32 lane's own deviation behind
    // (1.7e-9 at this fixture), which is orders above the floor.
    EXPECT_LT(delta, 1e-12);
}

// The comparison's own numbers, and the pin that the two implementations
// of the routing decision agree: the enforcement walk's admitted count is
// the task-list size the screening pass produces for the same density
// (the drift guard between SumRoutingBounds and ScreenOne's gate - they
// are two spellings of one expression, and this is the test that fails if
// one of them is edited alone).
TEST(CertifiedBudgetTest, TheComparisonUsesTheRoutingPassesOwnAdmissions) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto fixtureResult = MakeFixture();
    ASSERT_TRUE(fixtureResult.has_value()) << fixtureResult.error().message;
    const EnforcementFixture& fixture = *fixtureResult;
    auto xResult = InverseSquareRoot(fixture.overlap);
    ASSERT_TRUE(xResult.has_value());
    const Eigen::MatrixXd density = OccupiedDensity(fixture.coreH, *xResult, fixture.occupiedCount);

    // The lane live with no enforcement: the routing pass's own count, and
    // the delivered bound the run record reports.
    auto plain = Probe(fixture, density, ProbeOptions(true, false));
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_FALSE(plain->budget.enforced);

    auto enforced = Probe(fixture, density, ProbeOptions(true, true));
    ASSERT_TRUE(enforced.has_value()) << enforced.error().message;
    ASSERT_TRUE(enforced->budget.enforced);

    std::printf("[certified budget] n_basis=%zu budget=%.6e screened=%.6e routed=%.6e "
                "routed_quartets=%zu delivered=%.6e verdict=%s\n",
                plain->fock.rows(),
                enforced->budget.budgetHa,
                enforced->budget.screenedHa,
                enforced->budget.routedHa,
                enforced->budget.routedQuartets,
                plain->deliveredBoundHa,
                enforced->budget.fellBackToFp64 ? "fallback-fp64" : "lane-live");

    EXPECT_EQ(enforced->budget.routedQuartets, plain->fp32Quartets)
        << "the enforcement's admission count disagrees with the routing pass's task list";
    EXPECT_GT(enforced->budget.screenedHa, 0.0);
    EXPECT_GT(enforced->budget.routedHa, 0.0);

    // The delivered bound is the routing pass's own quantity, untouched by
    // the enforcement's walk: an unenforced build still reports it.
    EXPECT_GT(plain->deliveredBoundHa, 0.0);
}

// Does the enforcement leave the lane alive at ANY preset? The budget is
// derived per preset (1e-6 / 1e-10 at the two that admit the lane at all),
// and the routed sum scales with the routing threshold, so a preset sweep
// is what separates "the criterion is tight at kNormal" from "the
// criterion refuses the lane everywhere". The probe prints the comparison
// at each preset and asserts only that the verdict the builder took is the
// comparison it reported - the numbers themselves are the evidence, and
// this test does not choose them.
TEST(CertifiedBudgetTest, TheVerdictHoldsAtEveryPresetThatAdmitsTheLane) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto fixtureResult = MakeFixture();
    ASSERT_TRUE(fixtureResult.has_value()) << fixtureResult.error().message;
    const EnforcementFixture& fixture = *fixtureResult;
    auto xResult = InverseSquareRoot(fixture.overlap);
    ASSERT_TRUE(xResult.has_value());
    const Eigen::MatrixXd density = OccupiedDensity(fixture.coreH, *xResult, fixture.occupiedCount);

    struct PresetCase {
        qcx::integrals::AccuracyPreset preset;
        const char* name;
        bool laneAdmitted;
    };

    const std::array<PresetCase, 3> cases{{
        {qcx::integrals::AccuracyPreset::kLoose, "kLoose", true},
        {qcx::integrals::AccuracyPreset::kNormal, "kNormal", true},
        {qcx::integrals::AccuracyPreset::kTight, "kTight", false},
    }};

    for (const PresetCase& testCase : cases)
    {
        auto plain = Probe(fixture, density, PresetOptions(testCase.preset, true, false));
        ASSERT_TRUE(plain.has_value()) << plain.error().message;
        auto enforced = Probe(fixture, density, PresetOptions(testCase.preset, true, true));
        ASSERT_TRUE(enforced.has_value()) << enforced.error().message;
        ASSERT_TRUE(enforced->budget.enforced);

        // kTight's gate is 0.0: the lane admits nothing, so there is
        // nothing for the budget to refuse and the enforcement runs
        // vacuously (the sums keep their observed-zero defaults).
        EXPECT_EQ(plain->fp32Quartets > 0, testCase.laneAdmitted) << testCase.name;

        std::printf("[certified budget] %-7s budget=%.6e screened=%.6e routed=%.6e "
                    "admitted=%zu delivered=%.6e verdict=%s\n",
                    testCase.name,
                    enforced->budget.budgetHa,
                    enforced->budget.screenedHa,
                    enforced->budget.routedHa,
                    enforced->budget.routedQuartets,
                    plain->deliveredBoundHa,
                    enforced->budget.fellBackToFp64 ? "fallback-fp64" : "lane-live");

        // The verdict is the comparison's, at every preset: the two are
        // one decision, never a value computed and then ignored.
        EXPECT_EQ(enforced->budget.fellBackToFp64,
                  enforced->budget.routedHa > enforced->budget.budgetHa);
        EXPECT_EQ(enforced->budget.routedQuartets, plain->fp32Quartets) << testCase.name;
    }
}

// The budget is DERIVED, not passed: every preset's budget follows from
// the preset and the screening co-term alone, and a build can never widen
// it to keep its lane alive. Pinned at the arithmetic level, where the
// enforcement's own comparison reads it.
TEST(CertifiedBudgetTest, TheBudgetIsThePresetsOwnArithmetic) {
    using qcx::integrals::AccuracyPreset;
    using qcx::integrals::CertifiedBoundBudget;
    using qcx::integrals::PresetEnergyBudget;

    for (const AccuracyPreset preset : {AccuracyPreset::kLoose, AccuracyPreset::kNormal})
    {
        const double presetBudget = PresetEnergyBudget(preset);

        // No screening co-term: the preset target less the ladder's slack.
        EXPECT_DOUBLE_EQ(CertifiedBoundBudget(preset, 0.0), 0.9 * presetBudget);

        // A co-term that eats the target leaves nothing - the budget floors
        // at zero rather than going negative (the ladder's own rule: a
        // non-positive per-batch budget routes fp64).
        EXPECT_DOUBLE_EQ(CertifiedBoundBudget(preset, presetBudget), 0.0);
        EXPECT_DOUBLE_EQ(CertifiedBoundBudget(preset, 10.0 * presetBudget), 0.0);
    }

    // kTight keeps the lane off and the budget arithmetic never runs; the
    // helper stays total for the presets it is asked about.
    EXPECT_DOUBLE_EQ(CertifiedBoundBudget(AccuracyPreset::kTight, 0.0), 0.9 * 1e-12);
}

} // namespace
