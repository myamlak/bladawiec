// The screening layer's contract at three levels: the significance test's own
// boundary arithmetic, the shell/range mapping the envelope build rests on, and
// the engine's screened assembly against the dense one.
//
// The engine-level gate is COUNTED rather than timed. A wall clock cannot say
// whether the setup ate the win, and it cannot say whether the run screened at
// all; the assertion here is that the engine's XcScreeningCounts agree exactly
// with a selection the test rebuilds from the public envelopes and the public
// predicate. A counter left at zero, or a selection built from the wrong
// shells, fails loudly instead of reading as a win.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/grid/shell_screening.hpp"
#include "qcx/grid/xc_grid_engine.hpp"
#include "synthetic_fixture.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

// One hydrogen at the origin: the smallest molecule the engine accepts.
qcx::Result<qcx::molecule::Molecule> MakeSingleHydrogen() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}}, std::move(*coordinates), 0, 1);
}

// Two hydrogens separated along z; the separation is what makes the screening
// bite, since each atom's shells decay at the other's grid points.
qcx::Result<qcx::molecule::Molecule> MakeHydrogenPair(double separation) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = separation;
    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// A rank-one density c c^T with c uniform and unit-normalized: positive
// semidefinite, so rho >= 0 on the whole grid, and dense enough that every
// shell carries weight.
Eigen::MatrixXd RankOneDensity(std::size_t aoCount) {
    const double entry = 1.0 / std::sqrt(static_cast<double>(aoCount));
    Eigen::MatrixXd density = Eigen::MatrixXd::Constant(
        static_cast<Eigen::Index>(aoCount), static_cast<Eigen::Index>(aoCount), entry);

    return density;
}

// A row-major copy, which is the layout ShellDensityWeights walks.
std::vector<double> RowMajor(const Eigen::MatrixXd& density) {
    std::vector<double> flat(static_cast<std::size_t>(density.rows() * density.cols()));
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(
        flat.data(), density.rows(), density.cols());
    map = density;

    return flat;
}

// One shell per angular momentum, s through i: the screened assembly's mapping
// has to hold for every angular momentum the AO tier ships, not for s and p.
constexpr std::string_view kAllMomentumBasis = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
1.0 1.0
H P
1.0 1.0
H D
1.0 1.0
H F
1.0 1.0
H G
1.0 1.0
H H
1.0 1.0
H I
1.0 1.0
END
)";

// One shell with two coefficient rows sharing an exponent array: a general
// contraction. The evaluator keeps both rows in ONE shell and lays their
// function sets out contiguously, so the shell's AO range spans both.
constexpr std::string_view kGeneralContractionBasis = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
1.0 1.0 1.0
END
)";

// One shell per angular momentum, all on one center: 1 + 3 + 5 + 7 + 9 + 11 + 13.
constexpr std::size_t kAllMomentumAoCount = 49;

TEST(XcScreeningTest, PredicateIsStrictAtTheToleranceAndFlatInsideTheExtent) {
    // The shell's extent is where r^l exp(-z r^2) peaks; inside it the decay
    // factor is exactly one, so the weight alone decides, and a test that lands
    // exactly on the tolerance must DROP its shell (ShellIsSignificant is a
    // strict inequality). Both sides of that line are pinned here.
    ShellEnvelope shell;
    shell.aoOffset = 0;
    shell.functionCount = 1;
    shell.center = {0.0, 0.0, 0.0};
    shell.minExponent = 0.5;
    shell.extentRadius = 1.5;

    const std::array<double, 3> inside = {1.0, 0.0, 0.0};
    const std::array<double, 3> outside = {0.0, 0.0, 1.9};

    EXPECT_DOUBLE_EQ(ShellDecayFactor(shell, inside), 1.0);
    EXPECT_TRUE(ShellIsSignificant(shell, inside, 1.0, 0.999));
    EXPECT_FALSE(ShellIsSignificant(shell, inside, 1.0, 1.0));
    EXPECT_FALSE(ShellIsSignificant(shell, inside, 1.0, 1.001));

    // Past the extent the factor is exp(-z d^2) with d the distance past it.
    const double past = 1.9 - shell.extentRadius;
    const double decay = std::exp(-shell.minExponent * past * past);
    EXPECT_DOUBLE_EQ(ShellDecayFactor(shell, outside), decay);
    EXPECT_LT(decay, 1.0);

    // Exactly on the tolerance drops; a hair below keeps.
    EXPECT_FALSE(ShellIsSignificant(shell, outside, 1.0, decay));
    EXPECT_TRUE(ShellIsSignificant(shell, outside, 1.0, decay * (1.0 - 1e-12)));

    // A shell carrying no density is dropped wherever it sits, and the test is
    // isotropic: the decay depends on the distance, not on the direction.
    EXPECT_FALSE(ShellIsSignificant(shell, inside, 0.0, 0.0));

    const std::array<double, 3> mirrored = {0.0, 0.0, -1.9};
    EXPECT_DOUBLE_EQ(ShellDecayFactor(shell, outside), ShellDecayFactor(shell, mirrored));
}

TEST(XcScreeningTest, EnvelopesSpanEveryContractionRowAndTileTheOrbitals) {
    // The envelope build maps the evaluator's shell order onto its AO ranges.
    // A general contraction is one shell whose block spans every coefficient
    // row, so a range that reported a single row would screen the wrong slots
    // and lose the others; this pins the width, and pins that the ranges tile
    // [0, AOCount()) with no gap and no overlap.
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());

    struct BasisCase {
        std::string_view label;
        std::string_view basis;
        std::size_t shellCount;
    };

    const std::array<BasisCase, 2> cases = {{
        {"general contraction", kGeneralContractionBasis, 1},
        {"s through i", kAllMomentumBasis, 7},
    }};

    for (const BasisCase& basisCase : cases)
    {
        const auto basis = qcx::basisset::ParseNwchemText(basisCase.basis);
        ASSERT_TRUE(basis.has_value()) << basisCase.label;
        const auto evaluator = AoEvaluator::Create(*molecule, *basis);
        ASSERT_TRUE(evaluator.has_value()) << basisCase.label;

        const auto ranges = evaluator->ShellRanges();
        ASSERT_EQ(ranges.size(), basisCase.shellCount) << basisCase.label;

        const auto envelopes = BuildShellEnvelopes(*molecule, *basis, ranges);
        ASSERT_TRUE(envelopes.has_value()) << basisCase.label;
        EXPECT_EQ(envelopes->size(), ranges.size()) << basisCase.label;

        std::size_t expectedOffset = 0;

        for (std::size_t shell = 0; shell < ranges.size(); ++shell)
        {
            EXPECT_EQ(ranges[shell].aoOffset, expectedOffset) << basisCase.label;
            EXPECT_EQ((*envelopes)[shell].aoOffset, ranges[shell].aoOffset) << basisCase.label;
            EXPECT_EQ((*envelopes)[shell].functionCount, ranges[shell].functionCount)
                << basisCase.label;
            expectedOffset += ranges[shell].functionCount;
        }

        EXPECT_EQ(expectedOffset, evaluator->AOCount()) << basisCase.label;
    }

    // The general contraction's block is two functions wide, not one, and both
    // slots belong to the same shell - the two rows carry identical
    // coefficients, so their values at a point are equal.
    const auto contractionBasis = qcx::basisset::ParseNwchemText(kGeneralContractionBasis);
    ASSERT_TRUE(contractionBasis.has_value());
    const auto contractionEvaluator = AoEvaluator::Create(*molecule, *contractionBasis);
    ASSERT_TRUE(contractionEvaluator.has_value());

    ASSERT_EQ(contractionEvaluator->AOCount(), 2u);
    ASSERT_EQ(contractionEvaluator->ShellRanges().size(), 1u);
    EXPECT_EQ(contractionEvaluator->ShellRanges()[0].functionCount, 2u);

    const std::array<std::size_t, 1> selection = {0};
    std::array<double, 2> selected{};
    contractionEvaluator->EvaluateSelected({0.7, 0.0, 0.0}, selection, selected);
    EXPECT_NE(selected[0], 0.0);
    EXPECT_DOUBLE_EQ(selected[0], selected[1]);
}

TEST(XcScreeningTest, ZeroToleranceReproducesTheDensePathForEveryAngularMomentum) {
    // The screened assembly is a restriction of the dense one to the surviving
    // shells; at a zero tolerance nothing that carries weight is dropped (only
    // decays that underflow to zero at the grid's far tail), so the two must
    // agree. Every angular momentum s through i is on this path, which is what
    // makes the shell-blocked contraction's AO bookkeeping load-bearing.
    const auto molecule = MakeSingleHydrogen();
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::basisset::ParseNwchemText(kAllMomentumBasis);
    ASSERT_TRUE(basis.has_value());
    ASSERT_EQ(basis->Find(1)->shells.size(), 7u);

    const Eigen::MatrixXd density = RankOneDensity(kAllMomentumAoCount);

    for (const std::string_view name : {"slater", "pbe"})
    {
        const auto engine = XcGridEngine::Create(*molecule, *basis, name);
        ASSERT_TRUE(engine.has_value()) << name;
        ASSERT_EQ(engine->AOCount(), kAllMomentumAoCount) << name;
        ASSERT_EQ(engine->ShellEnvelopes().size(), 7u) << name;

        const auto dense = engine->EvaluateClosedShell(density);
        ASSERT_TRUE(dense.has_value()) << name;
        const auto screened = engine->EvaluateClosedShellScreened(density, 0.0);
        ASSERT_TRUE(screened.has_value()) << name;

        ASSERT_TRUE(std::isfinite(screened->energy)) << name;
        ASSERT_TRUE(screened->potentialAlpha.allFinite()) << name;
        ASSERT_TRUE(screened->potentialBeta.allFinite()) << name;

        EXPECT_NEAR(screened->energy, dense->energy, 1e-12 * std::abs(dense->energy) + 1e-14)
            << name;
        EXPECT_NEAR((screened->potentialAlpha - dense->potentialAlpha).cwiseAbs().maxCoeff(),
                    0.0,
                    1e-12 * dense->potentialAlpha.cwiseAbs().maxCoeff())
            << name;
        EXPECT_NEAR((screened->potentialBeta - dense->potentialBeta).cwiseAbs().maxCoeff(),
                    0.0,
                    1e-12 * dense->potentialBeta.cwiseAbs().maxCoeff())
            << name;

        // The dense run reports its own cost in both columns and no screening
        // work at all; the screened run reports a partially covered selection.
        EXPECT_EQ(dense->counts.shellTests, 0u) << name;
        EXPECT_EQ(dense->counts.densityWeightTerms, 0u) << name;
        EXPECT_EQ(dense->counts.aoSlotsEvaluated, dense->counts.aoSlotsDense) << name;
        EXPECT_EQ(dense->counts.contractionPairs, dense->counts.contractionPairsDense) << name;

        EXPECT_GT(screened->counts.points, 0u) << name;
        EXPECT_EQ(screened->counts.shellTests,
                  screened->counts.points * engine->ShellEnvelopes().size())
            << name;
        EXPECT_GT(screened->counts.densityWeightTerms, 0u) << name;
        EXPECT_LT(screened->counts.aoSlotsEvaluated, screened->counts.aoSlotsDense) << name;
        EXPECT_LT(screened->counts.contractionPairs, screened->counts.contractionPairsDense)
            << name;

        // The structural half of the equivalence, with no neglect argument in
        // it at all: a tolerance below zero keeps EVERY shell, because every
        // weight and decay is non-negative - underflowed decays included - so
        // the selection is the whole basis and the shell-blocked assembly must
        // reproduce the dense one to rounding. This is the case that separates
        // a wrong sub-block formula from a neglected term, and only a
        // gradient-consuming functional can see the difference between them.
        const auto full = engine->EvaluateClosedShellScreened(density, -1.0);
        ASSERT_TRUE(full.has_value()) << name;

        EXPECT_EQ(full->counts.shellKeeps, full->counts.shellTests) << name;
        EXPECT_EQ(full->counts.aoSlotsEvaluated, full->counts.aoSlotsDense) << name;
        EXPECT_EQ(full->counts.contractionPairs, full->counts.contractionPairsDense) << name;
        EXPECT_NEAR(full->energy, dense->energy, 1e-12 * std::abs(dense->energy) + 1e-14) << name;
        EXPECT_NEAR((full->potentialAlpha - dense->potentialAlpha).cwiseAbs().maxCoeff(),
                    0.0,
                    1e-12 * dense->potentialAlpha.cwiseAbs().maxCoeff())
            << name;
        EXPECT_NEAR((full->potentialBeta - dense->potentialBeta).cwiseAbs().maxCoeff(),
                    0.0,
                    1e-12 * dense->potentialBeta.cwiseAbs().maxCoeff())
            << name;
    }
}

// The selection the engine should have made, rebuilt from the public API: the
// envelopes, the weights, and the predicate are the whole rule, so this is an
// independent statement of what the counters have to say.
struct SelectionPrediction {
    std::size_t points = 0;
    std::size_t livePoints = 0;
    std::size_t shellTests = 0;
    std::size_t shellKeeps = 0;
    std::size_t aoSlotsEvaluated = 0;
    std::size_t contractionPairs = 0;
    std::size_t maxKeptAos = 0;
    // The coverage account: the decay-weighted weight the list KEEPS, the
    // decay-weighted weight there is, and the worst per-point slack against the
    // bound a dropped shell is allowed (each dropped shell has
    // weight x decay <= tolerance, so the dropped part of the decay-weighted
    // sum is at most dropped x tolerance).
    double decayWeightKept = 0.0;
    double decayWeightTotal = 0.0;
    double coverageSlackMin = 0.0;
    bool coverageSlackSeen = false;
};

// The batch passes the walk should have used: one per (block, batch of points),
// which is what makes the batching structural rather than incidental.
std::size_t PredictBatches(const XcGridEngine& engine, const XcBatchSettings& batch) {
    std::size_t batches = 0;

    for (const excgrid::Block& block : engine.Grid().Blocks())
    {
        const std::size_t size = batch.pointBatch == 0 ? block.pointCount : batch.pointBatch;

        if (size == 0)
        {
            continue;
        }

        batches += (block.pointCount + size - 1) / size;
    }

    return batches;
}

SelectionPrediction PredictSelection(const XcGridEngine& engine,
                                     const Eigen::MatrixXd& density,
                                     double tolerance) {
    const std::span<const ShellEnvelope> envelopes = engine.ShellEnvelopes();
    const std::vector<double> flat = RowMajor(density);
    const auto weights = ShellDensityWeights(
        std::vector<ShellEnvelope>(envelopes.begin(), envelopes.end()), flat, engine.AOCount());
    EXPECT_TRUE(weights.has_value());

    SelectionPrediction prediction;

    for (const excgrid::Block& block : engine.Grid().Blocks())
    {
        for (std::size_t point = 0; point < block.pointCount; ++point)
        {
            if (block.weights[point] == 0.0)
            {
                continue;
            }

            ++prediction.points;
            std::size_t selectedAos = 0;
            std::size_t dropped = 0;
            double decayedKept = 0.0;
            double decayedTotal = 0.0;

            for (std::size_t shell = 0; shell < envelopes.size(); ++shell)
            {
                ++prediction.shellTests;

                const double decayed =
                    (*weights)[shell] * ShellDecayFactor(envelopes[shell], block.points[point]);
                decayedTotal += decayed;

                if (!ShellIsSignificant(
                        envelopes[shell], block.points[point], (*weights)[shell], tolerance))
                {
                    ++dropped;
                    continue;
                }

                ++prediction.shellKeeps;
                decayedKept += decayed;
                selectedAos += envelopes[shell].functionCount;
            }

            // The coverage bound, per point: every dropped shell contributes at
            // most its own tolerance, so the decay-weighted part the list does
            // not keep is at most dropped x tolerance. A negative slack would
            // mean the engine kept a list that does not cover what the
            // predicate promised it covers.
            const double slack =
                decayedKept + static_cast<double>(dropped) * tolerance - decayedTotal;

            if (!prediction.coverageSlackSeen || slack < prediction.coverageSlackMin)
            {
                prediction.coverageSlackMin = slack;
                prediction.coverageSlackSeen = true;
            }

            prediction.decayWeightKept += decayedKept;
            prediction.decayWeightTotal += decayedTotal;
            prediction.maxKeptAos = std::max(prediction.maxKeptAos, selectedAos);

            if (selectedAos == 0)
            {
                continue;
            }

            ++prediction.livePoints;
            prediction.aoSlotsEvaluated += selectedAos;
            prediction.contractionPairs += 2 * selectedAos * selectedAos;
        }
    }

    return prediction;
}

TEST(XcScreeningTest, CountsMatchAnIndependentlyRebuiltSelection) {
    // The counted gate. The engine's own columns are compared, exactly, against
    // a selection rebuilt here from the public envelopes and predicate - across
    // a tolerance sweep, so a counter that ignores the tolerance, or a
    // selection built from the wrong shells, cannot pass by accident.
    const auto molecule = MakeHydrogenPair(10.0);
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::basisset::ParseNwchemText(R"(
BASIS "ao basis" SPHERICAL PRINT
H S
3.42525091 0.15432897
0.62391373 0.53532814
0.16885540 0.44463454
END
)");
    ASSERT_TRUE(basis.has_value());

    Eigen::MatrixXd density(2, 2);
    density << 1.0, 1.0, 1.0, 1.0;

    const auto engine = XcGridEngine::Create(*molecule, *basis, "slater");
    ASSERT_TRUE(engine.has_value());
    ASSERT_EQ(engine->ShellEnvelopes().size(), 2u);

    for (const double tolerance : {0.0, 1e-10, 1e-8, 1e-6, 1e-4})
    {
        const auto evaluation = engine->EvaluateClosedShellScreened(density, tolerance);
        ASSERT_TRUE(evaluation.has_value()) << tolerance;
        ASSERT_TRUE(std::isfinite(evaluation->energy)) << tolerance;

        const SelectionPrediction predicted = PredictSelection(*engine, 0.5 * density, tolerance);

        EXPECT_EQ(evaluation->counts.points, predicted.points) << tolerance;
        EXPECT_EQ(evaluation->counts.shellTests, predicted.shellTests) << tolerance;
        EXPECT_EQ(evaluation->counts.shellKeeps, predicted.shellKeeps) << tolerance;
        EXPECT_EQ(evaluation->counts.aoSlotsEvaluated, predicted.aoSlotsEvaluated) << tolerance;
        EXPECT_EQ(evaluation->counts.contractionPairs, predicted.contractionPairs) << tolerance;

        // The fetch count is the batch API's gate: one AO evaluation per point
        // that kept a shell, and none for a point that kept none.
        EXPECT_EQ(evaluation->counts.aoFetches, predicted.livePoints) << tolerance;
        EXPECT_LE(evaluation->counts.aoFetches, evaluation->counts.points) << tolerance;
        EXPECT_EQ(evaluation->counts.batches, PredictBatches(*engine, XcBatchSettings{}))
            << tolerance;

        // A closed-shell pair is one matrix, so the weight pass runs once.
        EXPECT_EQ(evaluation->counts.densityWeightTerms, 4u) << tolerance;
        EXPECT_EQ(evaluation->counts.aoSlotsDense, evaluation->counts.points * 2) << tolerance;
    }
}

TEST(XcScreeningTest, BatchingChangesNoNumberAndFetchesEachPointOnce) {
    // The batch granularity changes how the grid is walked, never what is
    // computed: every point's arithmetic stays per point, so the results are
    // not merely close across batch sizes but bit-identical. The counters say
    // the same thing, except the batch count itself, which is the only column
    // that tracks the knob. And the fetch column is the batch API's own gate:
    // one AO evaluation per live point, so a consumer that re-entered the AO
    // tier for a second quantity would fail here by count.
    //
    // Two tolerances put both regimes on the path: at zero every visited point
    // keeps shells, and at 1e-4 some keep none and never reach the AO tier.
    const auto molecule = MakeHydrogenPair(10.0);
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::basisset::ParseNwchemText(R"(
BASIS "ao basis" SPHERICAL PRINT
H S
3.42525091 0.15432897
0.62391373 0.53532814
0.16885540 0.44463454
END
)");
    ASSERT_TRUE(basis.has_value());

    Eigen::MatrixXd density(2, 2);
    density << 1.0, 1.0, 1.0, 1.0;

    for (const std::string_view name : {"slater", "pbe"})
    {
        const auto engine = XcGridEngine::Create(*molecule, *basis, name);
        ASSERT_TRUE(engine.has_value()) << name;

        for (const double tolerance : {0.0, 1e-4})
        {
            const XcBatchSettings reference{64};
            const auto baseline =
                engine->EvaluateClosedShellScreened(density, tolerance, reference);
            ASSERT_TRUE(baseline.has_value()) << name << " " << tolerance;
            ASSERT_TRUE(std::isfinite(baseline->energy)) << name << " " << tolerance;

            const SelectionPrediction predicted =
                PredictSelection(*engine, 0.5 * density, tolerance);
            EXPECT_EQ(baseline->counts.aoFetches, predicted.livePoints) << name << " " << tolerance;

            // The empty selection is exercised at BOTH ends of this sweep, but
            // for different reasons. A zero tolerance still drops the grid's
            // far tail: there the shells' decays have underflowed to exactly
            // zero, so `weight * decay > 0` is false - on this fixture 2044 of
            // 42034 visited points. The looser end drops whole regions by
            // tolerance instead.
            EXPECT_LE(predicted.livePoints, predicted.points) << name << " " << tolerance;
            EXPECT_GT(predicted.livePoints, 0u) << name << " " << tolerance;

            if (tolerance > 0.0)
            {
                EXPECT_LT(predicted.livePoints, predicted.points) << name << " " << tolerance;
            }

            for (const std::size_t pointBatch : {0u, 1u, 3u, 7u, 4096u})
            {
                const XcBatchSettings settings{pointBatch};
                const auto batched =
                    engine->EvaluateClosedShellScreened(density, tolerance, settings);
                ASSERT_TRUE(batched.has_value()) << name << " " << pointBatch;
                ASSERT_TRUE(std::isfinite(batched->energy)) << name << " " << pointBatch;

                EXPECT_DOUBLE_EQ(batched->energy, baseline->energy) << name << " " << pointBatch;
                EXPECT_DOUBLE_EQ(
                    (batched->potentialAlpha - baseline->potentialAlpha).cwiseAbs().maxCoeff(), 0.0)
                    << name << " " << pointBatch;
                EXPECT_DOUBLE_EQ(
                    (batched->potentialBeta - baseline->potentialBeta).cwiseAbs().maxCoeff(), 0.0)
                    << name << " " << pointBatch;

                EXPECT_EQ(batched->counts.points, baseline->counts.points)
                    << name << " " << pointBatch;
                EXPECT_EQ(batched->counts.aoFetches, baseline->counts.aoFetches)
                    << name << " " << pointBatch;
                EXPECT_EQ(batched->counts.shellTests, baseline->counts.shellTests)
                    << name << " " << pointBatch;
                EXPECT_EQ(batched->counts.shellKeeps, baseline->counts.shellKeeps)
                    << name << " " << pointBatch;
                EXPECT_EQ(batched->counts.aoSlotsEvaluated, baseline->counts.aoSlotsEvaluated)
                    << name << " " << pointBatch;
                EXPECT_EQ(batched->counts.contractionPairs, baseline->counts.contractionPairs)
                    << name << " " << pointBatch;

                // The batch count is the one column the knob moves, and it must
                // be the number of passes the blocks actually need.
                EXPECT_EQ(batched->counts.batches, PredictBatches(*engine, settings))
                    << name << " " << pointBatch;
            }
        }
    }
}

TEST(XcScreeningTest, ScreeningDropsShellsAsTheToleranceGrowsAndStaysBounded) {
    // What the screening buys, and what it costs. The two atoms are 10 Bohr
    // apart, so each atom's shell decays by exp(-z r^2) at the other's points:
    // the selection shrinks monotonically as the tolerance grows, and the
    // energy error it introduces tracks that tolerance from below.
    //
    // Measured on this fixture (slater / pbe), AO slots out of the dense 84068:
    //   tol 0       slots 79980  error 0         / 0
    //   tol 1e-10   slots 71502  error 9.2e-13   / 1.6e-12
    //   tol 1e-8    slots 68108  error 5.4e-10   / 7.0e-10
    //   tol 1e-6    slots 38626  error 5.8e-8    / 5.7e-8
    //   tol 1e-4    slots 36576  error 6.8e-7    / 8.7e-7
    // The error stays under a quarter of the tolerance (worst ratio 0.058, at
    // 1e-6) and under 5e-6 Hartree everywhere - three orders below chemical
    // accuracy. The bounds below carry the headroom those ratios imply, so a
    // later change has to argue with the numbers rather than with the prose.
    const auto molecule = MakeHydrogenPair(10.0);
    ASSERT_TRUE(molecule.has_value());
    const auto basis = qcx::basisset::ParseNwchemText(R"(
BASIS "ao basis" SPHERICAL PRINT
H S
3.42525091 0.15432897
0.62391373 0.53532814
0.16885540 0.44463454
END
)");
    ASSERT_TRUE(basis.has_value());

    Eigen::MatrixXd density(2, 2);
    density << 1.0, 1.0, 1.0, 1.0;

    for (const std::string_view name : {"slater", "pbe"})
    {
        const auto engine = XcGridEngine::Create(*molecule, *basis, name);
        ASSERT_TRUE(engine.has_value()) << name;

        const auto dense = engine->EvaluateClosedShell(density);
        ASSERT_TRUE(dense.has_value()) << name;

        // The selection can only shrink as the tolerance grows, and no
        // screened run may evaluate more AO slots than the dense one.
        std::size_t previousSlots = dense->counts.aoSlotsDense;
        std::size_t previousKeeps = dense->counts.points * engine->ShellEnvelopes().size();

        for (const double tolerance : {0.0, 1e-10, 1e-8, 1e-6, 1e-4})
        {
            const auto screened = engine->EvaluateClosedShellScreened(density, tolerance);
            ASSERT_TRUE(screened.has_value()) << name << " " << tolerance;
            ASSERT_TRUE(std::isfinite(screened->energy)) << name << " " << tolerance;
            ASSERT_TRUE(screened->potentialAlpha.allFinite()) << name << " " << tolerance;

            EXPECT_LE(screened->counts.aoSlotsEvaluated, previousSlots) << name << " " << tolerance;
            EXPECT_LE(screened->counts.shellKeeps, previousKeeps) << name << " " << tolerance;
            EXPECT_LT(screened->counts.aoSlotsEvaluated, dense->counts.aoSlotsDense)
                << name << " " << tolerance;

            previousSlots = screened->counts.aoSlotsEvaluated;
            previousKeeps = screened->counts.shellKeeps;

            const double error = std::abs(screened->energy - dense->energy);
            EXPECT_LT(error, 5e-6) << name << " " << tolerance << " error " << error;

            if (tolerance > 0.0)
            {
                EXPECT_LT(error, 0.25 * tolerance)
                    << name << " " << tolerance << " error " << error;
            }
        }
    }
}

TEST(XcScreeningTest, ChainScreeningCoversTheDecayWeightedWeightItScreensAway) {
    // The counted form of the screening-cover gate, on a system where
    // screening has something to do: a 32-atom collinear hydrogen chain, whose
    // grid spans ~44 Bohr, so each point sees a neighbourhood rather than the
    // whole basis.
    //
    // "Covers" is stated in the units the predicate actually bounds - the
    // DECAY-WEIGHTED weight - and that is not pedantry: a dropped shell's raw
    // density weight is NOT bounded by the tolerance (a shell with a large
    // weight and a small decay is exactly what the test is meant to drop),
    // while its weight times its decay is bounded. So per point the gate
    // asserts
    //     kept decayed weight >= total decayed weight - dropped x tolerance,
    // with both sides computed HERE from the public envelopes, weights and
    // predicate, and the engine's counters required to match that same walk
    // exactly. A list built from the wrong shells, or one that keeps less than
    // the predicate promises, fails by count or by slack.
    constexpr std::size_t kChainAtoms = 32;

    auto system = MakeSyntheticChain(kChainAtoms);
    ASSERT_TRUE(system.has_value());

    // A coarse grid: the screening structure here is driven by the chain's
    // geometry and the decay radii, neither of which the grid resolution
    // touches, while the point count is what a 32-atom fixture costs. The
    // counts stay exact at every resolution because the engine and this test
    // walk the same points.
    XcGridSettings coarse;
    coarse.radialPoints = 30;
    coarse.angularPoints = 50;

    const auto engine = XcGridEngine::Create(system->molecule, system->basis, "slater", coarse);
    ASSERT_TRUE(engine.has_value());
    ASSERT_EQ(engine->AOCount(), kChainAtoms);
    ASSERT_EQ(engine->ShellEnvelopes().size(), kChainAtoms);

    // A uniform rank-one density puts the same weight on every shell, so the
    // screen is driven by geometry alone. That is the conservative case: a
    // localized density would make distant shells less significant, not more.
    const Eigen::MatrixXd density = RankOneDensity(kChainAtoms);

    for (const double tolerance : {1e-6, 1e-4})
    {
        const auto evaluation = engine->EvaluateClosedShellScreened(density, tolerance);
        ASSERT_TRUE(evaluation.has_value()) << tolerance;
        ASSERT_TRUE(std::isfinite(evaluation->energy)) << tolerance;

        const SelectionPrediction predicted = PredictSelection(*engine, 0.5 * density, tolerance);

        EXPECT_EQ(evaluation->counts.points, predicted.points) << tolerance;
        EXPECT_EQ(evaluation->counts.shellTests, predicted.shellTests) << tolerance;
        EXPECT_EQ(evaluation->counts.shellKeeps, predicted.shellKeeps) << tolerance;
        EXPECT_EQ(evaluation->counts.aoSlotsEvaluated, predicted.aoSlotsEvaluated) << tolerance;
        EXPECT_EQ(evaluation->counts.contractionPairs, predicted.contractionPairs) << tolerance;
        EXPECT_EQ(evaluation->counts.aoFetches, predicted.livePoints) << tolerance;

        // The cover bound, and the evidence that it is not vacuous: the test
        // must have seen points where shells were dropped, otherwise the
        // inequality holds for a reason that has nothing to do with screening.
        ASSERT_TRUE(predicted.coverageSlackSeen) << tolerance;
        EXPECT_GE(predicted.coverageSlackMin, -1e-12) << tolerance;
        EXPECT_LT(predicted.decayWeightKept, predicted.decayWeightTotal) << tolerance;
        EXPECT_LT(predicted.shellKeeps, predicted.shellTests) << tolerance;

        // The complexity bound: the pair terms are quadratic in the KEPT count
        // (bounded here by the largest selection the walk made), never in
        // AOCount(), and the run is cheaper than the dense one by that ratio.
        EXPECT_LE(evaluation->counts.contractionPairs,
                  2 * predicted.points * predicted.maxKeptAos * predicted.maxKeptAos)
            << tolerance;
        EXPECT_LT(evaluation->counts.contractionPairs, evaluation->counts.contractionPairsDense)
            << tolerance;
        EXPECT_LT(predicted.maxKeptAos, kChainAtoms) << tolerance;
    }
}

TEST(XcScreeningTest, ScreenedPairTermsScaleWithTheKeptCountNotTheBasisSize) {
    // The plan's claim is that the factor comes from the kept count staying
    // small while the basis grows. Two chain sizes at one tolerance measure it:
    // AOCount() doubles, the largest selection per point saturates, and the
    // screened share of the dense pair terms therefore falls with size.
    //
    // The tolerance matters to that measurement, and it was chosen against the
    // decay radius rather than by taste: a screen only bites once the system is
    // larger than the distance at which a shell stops mattering. At 1e-8 that
    // distance is ~10.2 Bohr for STO-3G hydrogen, so a 16-atom chain (21 Bohr
    // end to end) keeps every shell at its middle points and screens nothing -
    // measured, not guessed: maxKeptAos came back equal to AOCount() at that
    // size and tolerance. 1e-4 puts the radius at ~7.1 Bohr and 48 atoms put
    // the chain at 66 Bohr, so the kept neighbourhood is a small fraction of
    // the basis at the large size while the small one still screens.
    //
    // The smaller size is also where the dense path is affordable, so the
    // screened energy is checked against it there. At the larger size this gate
    // checks the SELECTION and the counts, not the arithmetic: that is pinned
    // against the dense path on the small fixtures and on the smaller chain
    // here, and re-running it at scale would be a benchmark.
    constexpr double kTolerance = 1e-4;

    XcGridSettings coarse;
    coarse.radialPoints = 30;
    coarse.angularPoints = 50;

    std::size_t smallKept = 0;
    std::size_t largeKept = 0;
    double smallShare = 0.0;
    double largeShare = 0.0;

    for (const std::size_t atoms : {24u, 48u})
    {
        auto system = MakeSyntheticChain(atoms);
        ASSERT_TRUE(system.has_value()) << atoms;
        const auto engine = XcGridEngine::Create(system->molecule, system->basis, "slater", coarse);
        ASSERT_TRUE(engine.has_value()) << atoms;

        const Eigen::MatrixXd density = RankOneDensity(atoms);
        const auto screened = engine->EvaluateClosedShellScreened(density, kTolerance);
        ASSERT_TRUE(screened.has_value()) << atoms;
        ASSERT_TRUE(std::isfinite(screened->energy)) << atoms;

        const SelectionPrediction predicted = PredictSelection(*engine, 0.5 * density, kTolerance);
        EXPECT_EQ(screened->counts.contractionPairs, predicted.contractionPairs) << atoms;
        EXPECT_EQ(screened->counts.contractionPairsDense, 2 * predicted.points * atoms * atoms)
            << atoms;

        const double share = static_cast<double>(screened->counts.contractionPairs) /
                             static_cast<double>(screened->counts.contractionPairsDense);

        EXPECT_LT(share, 1.0) << atoms;
        EXPECT_LT(predicted.maxKeptAos, atoms) << atoms;
        EXPECT_LT(predicted.shellKeeps, predicted.shellTests) << atoms;

        // The density comparison, at the size where it is affordable: a zero
        // tolerance keeps everything the tail has not underflowed, so the two
        // paths must agree to rounding.
        const auto dense = engine->EvaluateClosedShell(density);
        ASSERT_TRUE(dense.has_value()) << atoms;
        const auto exact = engine->EvaluateClosedShellScreened(density, 0.0);
        ASSERT_TRUE(exact.has_value()) << atoms;

        EXPECT_NEAR(exact->energy, dense->energy, 1e-12 * std::abs(dense->energy) + 1e-14) << atoms;
        EXPECT_LT((exact->potentialAlpha - dense->potentialAlpha).cwiseAbs().maxCoeff(),
                  1e-12 * dense->potentialAlpha.cwiseAbs().maxCoeff() + 1e-14)
            << atoms;

        if (atoms == 24)
        {
            smallKept = predicted.maxKeptAos;
            smallShare = share;
        } else
        {
            largeKept = predicted.maxKeptAos;
            largeShare = share;
        }
    }

    // Measured here (24 -> 48 atoms, tolerance 1e-4, 29384 -> 58280 points):
    //   largest selection   11 -> 12 AOs     (the decay radius sets it, not the
    //                                         system size; it does not double)
    //   shells kept         36.8% -> 19.8%   (of the tests the walk ran)
    //   share of dense pair terms  0.152 -> 0.043
    // So the factor improves with size, and the assertions below carry the
    // headroom those numbers imply: the neighbourhood grows by at most two AOs,
    // and the share at least halves.
    EXPECT_LE(largeKept, smallKept + 2);
    EXPECT_LT(largeShare, 0.5 * smallShare);
    EXPECT_GT(largeShare, 0.0);
}

// The bound, measured across a ladder wide enough to falsify it. The
// claim is O(nGrid x nSig): the point count grows with the system while the
// SIGNIFICANT-shell count does not. Four chains, each twice the last, span
// AOCount() by 8x - so a selection that tracked the basis would show the kept
// count climbing 12 -> 96 and the pair terms climbing with the square of that.
// The counted gate below is what tests it, at every rung, against a selection
// this test rebuilds from the public envelopes, weights and predicate.
//
// Measured 2026-09-13 (Release, radial 12, angular 26, tolerance 1e-4):
//   atoms     24      48      96     192
//   points  6178   12274   24466   48850
//   kept      12      12      12      12     <- saturated, not basis-sized
//   pairs   1.10e6  2.47e6  5.32e6  1.10e7   <- 10x over the 8x span
//   dense   7.12e6  5.66e7  4.51e8  3.60e9   <- 506x over the same span
//   share  .1547   .0437   .0118   .00305   <- the screening factor, growing
//   keeps   37.2%   20.0%   10.4%    5.3%
//   slack  +1.2e-3 +3.6e-3 +8.4e-3 +1.8e-2  <- coverage, all positive
// The grid is deliberately coarse: neither the resolution nor the tolerance
// reaches the quantity being measured. The decay radius that sets the kept
// neighbourhood comes from the exponent and the tolerance, and the counts stay
// exact at every resolution because the engine and the prediction walk the
// same points.
//
// The fixture this ladder does NOT reach is the 5000-class rung, and the
// reason is measured rather than assumed: that rung's cost is dominated by the
// grid BUILD, not by the screening. At the engine's own default resolution
// (radial 30, angular 50) the build runs 1.1 s / 2.9 s / 12.5 s / 78.3 s at
// 48 / 96 / 192 / 384 atoms while the kept count stays at 12 and the screened
// walk stays at 0.32 / 0.99 / 3.9 / 14.4 s - so the build's local exponent is
// 2.65 at the widest rung, and extrapolating 384 -> 5000 costs 3.7 h at a pure
// N^2 and ~19 h at that measured exponent. The screening half extrapolates to
// ~41 min, and the DENSE comparison priced at 10^15 pair terms is far
// past any window. That rung is deferred work,
// and its blocker is the per-point Becke
// partition, which walks every atom at every point. A caveat that keeps this
// honest: the chain is maximally extended - 5000 atoms span ~7000 Bohr - so it
// is the worst case for that loop, and a compact 5000-basis-function molecule
// would be far cheaper. What the ladder DOES establish is the half that was in
// doubt: nSig saturates over an 8x basis span and the assembly tracks it.
TEST(XcScreeningTest, SignificantShellCountSaturatesAcrossAChainLadder) {
    constexpr double kTolerance = 1e-4;
    constexpr std::size_t kLargestAtoms = 192;

    XcGridSettings ladderGrid;
    ladderGrid.radialPoints = 12;
    ladderGrid.angularPoints = 26;

    std::size_t smallestKept = 0;
    std::size_t largestKept = 0;
    double smallestShare = 0.0;
    double largestShare = 0.0;

    for (const std::size_t atoms : {24u, 48u, 96u, 192u})
    {
        auto system = MakeSyntheticChain(atoms);
        ASSERT_TRUE(system.has_value()) << atoms;

        const auto engine =
            XcGridEngine::Create(system->molecule, system->basis, "slater", ladderGrid);
        ASSERT_TRUE(engine.has_value()) << atoms;
        ASSERT_EQ(engine->AOCount(), atoms) << atoms;
        ASSERT_EQ(engine->ShellEnvelopes().size(), atoms) << atoms;

        // The same uniform rank-one density at every rung, so the screen is
        // driven by geometry and size rather than by a density that changes
        // shape with the ladder.
        const Eigen::MatrixXd density = RankOneDensity(atoms);
        const auto screened = engine->EvaluateClosedShellScreened(density, kTolerance);
        ASSERT_TRUE(screened.has_value()) << atoms;
        ASSERT_TRUE(std::isfinite(screened->energy)) << atoms;

        const SelectionPrediction predicted = PredictSelection(*engine, 0.5 * density, kTolerance);

        // The counters are the engine's; the prediction is the same walk rebuilt
        // from the public API. A selection built from the wrong shells fails
        // here by count rather than reading as a win.
        EXPECT_EQ(screened->counts.shellTests, predicted.shellTests) << atoms;
        EXPECT_EQ(screened->counts.shellKeeps, predicted.shellKeeps) << atoms;
        EXPECT_EQ(screened->counts.aoSlotsEvaluated, predicted.aoSlotsEvaluated) << atoms;
        EXPECT_EQ(screened->counts.contractionPairs, predicted.contractionPairs) << atoms;

        // The dense column is the anti-template's arithmetic - points x 2 x n^2
        // - carried here so the share below is the intended comparison and
        // not a ratio to a cheaper stand-in.
        EXPECT_EQ(screened->counts.contractionPairsDense, 2 * predicted.points * atoms * atoms)
            << atoms;

        // Screening must bite at every rung or the ladder measures nothing.
        ASSERT_TRUE(predicted.coverageSlackSeen) << atoms;
        EXPECT_GE(predicted.coverageSlackMin, -1e-12) << atoms;
        EXPECT_LT(predicted.shellKeeps, predicted.shellTests) << atoms;
        EXPECT_LT(predicted.decayWeightKept, predicted.decayWeightTotal) << atoms;
        EXPECT_LT(predicted.maxKeptAos, atoms) << atoms;

        // The bound: the pair terms are quadratic in the KEPT count, never in
        // AOCount().
        EXPECT_LE(screened->counts.contractionPairs,
                  2 * predicted.points * predicted.maxKeptAos * predicted.maxKeptAos)
            << atoms;

        const double share = static_cast<double>(screened->counts.contractionPairs) /
                             static_cast<double>(screened->counts.contractionPairsDense);
        EXPECT_LT(share, 1.0) << atoms;

        if (atoms == 24u)
        {
            smallestKept = predicted.maxKeptAos;
            smallestShare = share;
        }

        if (atoms == kLargestAtoms)
        {
            largestKept = predicted.maxKeptAos;
            largestShare = share;
        }
    }

    // The saturation, stated two ways: the kept count does not grow across the
    // ladder (measured flat at 12 for all four rungs, so the +2 carries real
    // headroom), and it is far smaller than the basis it is selected from
    // (measured 12 against 192, i.e. 16x, asserted here at 8x).
    EXPECT_LE(largestKept, smallestKept + 2);
    EXPECT_LT(largestKept * 8, kLargestAtoms);

    // The screening factor improves with size: the measured share falls by 50.7x
    // across the 8x span, asserted at 10x so the gate carries the headroom.
    EXPECT_GT(smallestShare, 0.0);
    EXPECT_LT(largestShare, 0.1 * smallestShare);
}

} // namespace
} // namespace qcx::grid
