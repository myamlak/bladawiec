// The QfmmJBuilder acceptance tests: the composed builder (near field = the direct Coulomb
// builder restricted to the octree near-field pair-pair set, far field = the
// multipole accumulation) must
//   1. recover the plain direct-sum Coulomb matrix BIT-IDENTICALLY at the
//      theta -> 0 gate (QfmmOptions::theta < 0 - everything near field, the
//      near-field restriction is a no-op and the far field contributes
//      exactly nothing) - the monopole gate re-verified through the new class,
//   2. resolve the accuracy-preset defaults exactly (theta = 0 and
//      lMult = -1 pick ThetaForPreset / LMultForPreset - the explicit
//      override builds are bit-identical to the default builds),
//   3. decrease the far-field error monotonically in theta at every fixed
//      L_mult and in L_mult at every fixed theta (the convergence properties
//      carried through the builder, on the C12 alkane fixture),
//   4. keep the accuracy-preset ladder strictly decreasing at the recorded
//      top rungs (kLoose > kNormal in both the relative Frobenius and the
//      J-energy error vs the direct ground truth) and inside the
//      QFMM budgets {1e-5, 1e-7, 1e-8}
//      (benchmarks/data/qfmm_ladder_sweep_full.csv; the kTight rung
//      re-derived from 1e-9 by the 2026-09-13 ruling). On this fixture
//      the kNormal/kTight boundary is degenerate: the (kNormal, C12) tooth
//      at the C24-pinned (0.3, 5) is vacuous (all-near - the near path's
//      screening thresholds match the reference's, so it is bit-exact and
//      measures 0.0 exactly), and the kTight rung is the theta -> 0
//      degenerate gate (the near-field-only path - its near path screens
//      at the preset's 1e-12, a strict superset of the kNormal-screened
//      reference's 1e-10, so its residual vs that reference is the
//      reference's OWN screening error, NOT a QFMM error: ~3.33e-9 on this
//      fixture, the sweep's tight-end floor, re-measured 3.33386e-9
//      on 2026-09-13). That floor is exactly why the kTight budget was
//      re-derived 1e-9 -> 1e-8: the residual now sits INSIDE the budget,
//      and the loop's kTight branch therefore pins the floor band
//      (1e-9, 1e-8) - its assertions are unchanged, only the reason each
//      bound is there. The C24 strictness evidence lives in the sweep
//      data.
// Plus the Create() contract (invalid lMult / maxLeafSize / NaN theta are
// rejected) and the determinism invariant of the composed build (serial
// pin: two instances and two calls are bit-identical - the far field is
// deterministic by construction and the near field inherits the serial
// pin's determinism).

#include "alkane_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/integrals/screening.hpp"
#include "tensor_conversions.hpp"

// The qfmm_fixture.hpp harness calls ToTensor / ToMatrix
// unqualified from inside qcx::integrals::test, while they live in
// qcx::testing - ADL cannot see them there (their arguments are Eigen /
// qcx::memory types), so they must be in scope BEFORE the fixture is
// included: lookup in the fixture's inline definitions happens at include
// time, not at use time.
namespace {

using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

} // namespace

#include "qfmm_fixture.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <map>
#include <utility>

namespace {

using qcx::integrals::test::BuildCoreHamiltonian;
using qcx::integrals::test::BuildDirectFock;
using qcx::integrals::test::PhysicalDensity;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The fp64-only serial configuration of every bit-exact comparison: the
// measurement configuration of the bit-exact gates (qfmm_fixture.hpp's
// comment - the chunked combine order is thread-schedule dependent, so the
// serial pin makes the near-field parts of the QFMM and the direct build
// bitwise equal and the measured difference exactly the multipole
// truncation).
qcx::integrals::QfmmOptions MeasurementOptions() {
    qcx::integrals::QfmmOptions options;
    options.useCertifiedMixedPrecision = false;
    options.maxParallelChunks = 1;
    return options;
}

// The QFMM J through the BUILDER under test (not the fixture harness): H +
// 2J_near + 2J_far in one QfmmJBuilder instance.
qcx::Result<Eigen::MatrixXd> BuildQfmmFock(const qcx::molecule::Molecule& molecule,
                                           const qcx::basisset::BasisSet& basisSet,
                                           const qcx::integrals::test::CpuTensor2& coreHamiltonian,
                                           const Eigen::MatrixXd& density,
                                           const qcx::integrals::QfmmOptions& options) {
    auto builder =
        qcx::integrals::QfmmJBuilder::Create(molecule, basisSet, coreHamiltonian, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        return std::unexpected(fock.error());
    }

    return ToMatrix(*fock);
}

// The gated-sweep reference: the fixture BuildQfmmFock shape under the
// CALLER's preset. The
// fixture hardcodes the kTight extent rung (QfmmExtentForPreset(kTight))
// with the default kNormal screening preset, while the bit-identity gate
// must compare the leaf-driven production near field against the gated
// sweep under the SAME geometry extent AND screening preset (the preset
// drives the octree boxes and the near-field screening thresholds alike).
// Everything else mirrors the fixture exactly: fp64-only, the serial pin,
// the near field as the full-CSR sweep gated by the leaf-pair bitset
// (restrictToPairPairs), and the same far-field accumulation on top.
qcx::Result<Eigen::MatrixXd> BuildQfmmFockGated(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::integrals::test::CpuTensor2& coreHamiltonian,
    const Eigen::MatrixXd& density,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    double theta,
    int lMult,
    qcx::integrals::AccuracyPreset preset,
    // The geometry model the PRODUCTION side runs: the equivalence claim below
    // is geometry-conditional (the gated sweep and the leaf-driven near field
    // only agree when both enumerate the SAME tree), so the caller passes the
    // model it gave QfmmOptions. Since the 2026-09-15 default that is the
    // product ball with the surface test; the recorded pair stays the default
    // here so the sweeps below keep comparing the recorded geometry.
    qcx::integrals::QfmmExtentMode extentModel = qcx::integrals::QfmmExtentMode::kMidpointBound,
    qcx::integrals::QfmmSeparationMode separationMode =
        qcx::integrals::QfmmSeparationMode::kWidthTheta,
    double separationK = 0.0) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto pairStore = qcx::integrals::internal::BuildPairData(molecule, basisSet, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    const auto geometries = qcx::integrals::internal::ComputePairGeometries(
        *pairStore,
        qcx::integrals::QfmmExtentForPreset(preset),
        extentModel == qcx::integrals::QfmmExtentMode::kProductBall
            ? qcx::integrals::internal::QfmmExtentModel::kProductBall
            : qcx::integrals::internal::QfmmExtentModel::kMidpointBound);
    const auto tree = qcx::integrals::internal::BuildQfmmTree(geometries);

    if (!tree.has_value())
    {
        return std::unexpected(tree.error());
    }

    // The same resolution QfmmJBuilder::Create applies: an explicit theta
    // selects the width test, the negative theta is the gate, and an absent
    // theta leaves the choice to the caller's mode and k.
    qcx::integrals::internal::QfmmSeparation separation;
    separation.test = separationMode == qcx::integrals::QfmmSeparationMode::kSurfaceBall
                          ? qcx::integrals::internal::QfmmSeparationTest::kSurfaceBall
                          : qcx::integrals::internal::QfmmSeparationTest::kWidthTheta;
    separation.theta = theta;
    separation.k = separationK;

    if (theta > 0.0)
    {
        separation.test = qcx::integrals::internal::QfmmSeparationTest::kWidthTheta;
    }

    if (theta < 0.0)
    {
        separation.k = -1.0;
    }

    if (theta == 0.0 && qcx::integrals::ThetaForPreset(preset) <= 0.0)
    {
        separation.k = -1.0;
    }

    std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
    qcx::integrals::internal::BuildInteractionLists(
        *tree, separation, farFieldPairs, nearFieldLeafPairs);

    qcx::integrals::PairPairRestriction nearFieldKeys;
    nearFieldKeys.nLeaves = tree->nodes.size();
    nearFieldKeys.leafOfPair = tree->leafOfPair;
    nearFieldKeys.words.assign((tree->nodes.size() * tree->nodes.size() + 63) / 64, 0);

    for (const auto& [leafA, leafB] : nearFieldLeafPairs)
    {
        nearFieldKeys.InsertBoth(leafA, leafB);
    }

    qcx::integrals::FockBuildOptions options;
    options.accuracy = preset;
    options.useCertifiedMixedPrecision = false;
    options.buildCoulombOnly = true;
    // The serial pin (the fixture's comment: the theta = 0 gate must be
    // BIT-exact, and the serial sweep is deterministic).
    options.maxParallelChunks = 1;
    options.restrictToPairPairs = std::move(nearFieldKeys);
    auto builder =
        qcx::integrals::DirectJkFockBuilder::Create(molecule, basisSet, coreHamiltonian, options);

    if (!builder.has_value())
    {
        return std::unexpected(builder.error());
    }

    auto densityTensor = ToTensor(density);

    if (!densityTensor.has_value())
    {
        return std::unexpected(densityTensor.error());
    }

    auto fock = builder->BuildFock(*densityTensor);

    if (!fock.has_value())
    {
        return std::unexpected(fock.error());
    }

    Eigen::MatrixXd result = ToMatrix(*fock);
    // The pass signatures: per-pair and per-node order vectors (a -1
    // order skips the pair's moments / node's contributions). The uniform
    // all-lMult vectors reproduce the fixed-order runs exactly.
    const std::vector<int> pairOrders(pairList->pairs.size(), lMult);
    const std::vector<int> nodeOrders(tree->nodes.size(), lMult);
    const auto table =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, pairOrders);

    if (!table.has_value())
    {
        return std::unexpected(table.error());
    }

    const auto leafMoments = qcx::integrals::internal::AggregateLeafMoments(
        *pairList, *tree, *table, geometries, density, nodeOrders);
    const auto nodeMoments =
        qcx::integrals::internal::AggregateNodeMoments(*tree, leafMoments, nodeOrders);
    const auto potentials = qcx::integrals::internal::BuildFarFieldPotentials(
        *tree, farFieldPairs, nodeMoments, pairOrders, nodeOrders);
    qcx::integrals::internal::AccumulateFarFieldJ(
        result, *pairList, *tree, *table, potentials, geometries, nodeOrders);

    return result;
}

// The largest absolute element difference (== 0.0 for the bit-exact gates;
// ±0.0 compares equal, so empty-far-field additions cannot disturb it).
double MaxAbsoluteDifference(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    double maximum = 0.0;

    for (Eigen::Index i = 0; i < a.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < a.cols(); ++j)
        {
            maximum = std::max(maximum, std::abs(a(i, j) - b(i, j)));
        }
    }

    return maximum;
}

// The J-energy error: E_J = 0.5 * Tr(D * J), so the error of the QFMM
// approximation vs the direct ground truth is 0.5 * |Tr(D * (J_qfmm -
// J_direct))| - the preset's budget metric (Eh). The core Hamiltonian cancels.
double EnergyError(const Eigen::MatrixXd& qfmm,
                   // (qfmm, direct) is the compared pair, density the fixed weight.
                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                   const Eigen::MatrixXd& direct,
                   const Eigen::MatrixXd& density) {
    const Eigen::MatrixXd delta = qfmm - direct;
    return 0.5 * std::abs((density.cwiseProduct(delta)).sum());
}

TEST(QfmmFockBuildTest, PresetConstantsMatchTheCalibratedLadder) {
    // The committed preset constants (recalibrated by the preset-ladder
    // sweep): the theta ladder {0.45, 0.3, 0.0} for
    // {kLoose, kNormal, kTight} was measured at the per-preset extents
    // {1e-6, 1e-8, 1e-10} on the C12/C24 alkane chains. The earlier
    // {1.05, 0.85, 0.7} held only at the all-tau = 1e-10 extent.
    // kLoose (0.45, 5) is the cheapest common in-budget cell (reduced
    // slack on C12, recorded); kNormal (0.3, 5) is the C24 pin (the C12
    // tooth is vacuous there); kTight theta 0.0 is the degenerate gate
    // (the near-field-only path - no (theta, L) reaches the 1e-9 budget
    // at tau 1e-10 in the committed kNormal-screened-direct comparison:
    // the tight-end residual is the reference's own screening floor).
    // The L_mult ladder {5, 5, 0}: kLoose/kNormal need L = 5 (kLoose C24 at
    // L = 4 overflows the budget), kTight rides the degenerate rung. The
    // extent ladder {1e-6, 1e-8, 1e-10} is unchanged (tau is not an
    // angle or order knob).
    EXPECT_DOUBLE_EQ(qcx::integrals::ThetaForPreset(qcx::integrals::AccuracyPreset::kLoose), 0.45);
    EXPECT_DOUBLE_EQ(qcx::integrals::ThetaForPreset(qcx::integrals::AccuracyPreset::kNormal), 0.3);
    EXPECT_DOUBLE_EQ(qcx::integrals::ThetaForPreset(qcx::integrals::AccuracyPreset::kTight), 0.0);
    EXPECT_EQ(qcx::integrals::LMultForPreset(qcx::integrals::AccuracyPreset::kLoose), 5);
    EXPECT_EQ(qcx::integrals::LMultForPreset(qcx::integrals::AccuracyPreset::kNormal), 5);
    EXPECT_EQ(qcx::integrals::LMultForPreset(qcx::integrals::AccuracyPreset::kTight), 0);
    EXPECT_DOUBLE_EQ(qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kLoose),
                     1e-6);
    EXPECT_DOUBLE_EQ(qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kNormal),
                     1e-8);
    EXPECT_DOUBLE_EQ(qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kTight),
                     1e-10);

    // The QfmmOptions defaults: the kNormal preset, the preset-resolution
    // sentinels (0 / -1), the default leaf cap.
    const qcx::integrals::QfmmOptions defaults;
    EXPECT_EQ(defaults.accuracy, qcx::integrals::AccuracyPreset::kNormal);
    EXPECT_DOUBLE_EQ(defaults.theta, 0.0);
    EXPECT_EQ(defaults.lMult, -1);
    EXPECT_EQ(defaults.maxLeafSize, 8u);
    EXPECT_TRUE(defaults.useDensityScreening);
    EXPECT_TRUE(defaults.useCertifiedMixedPrecision);
    EXPECT_EQ(defaults.maxParallelChunks, 0u);
}

TEST(QfmmFockBuildTest, CreateRejectsInvalidOptions) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::QfmmOptions options;

    // One past the cap (this test does not include the internal headers;
    // the public cap is kQfmmMaxLMult = 8, so 9 is rejected).
    options.lMult = 9;
    auto tooHigh = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(tooHigh.has_value());
    EXPECT_EQ(tooHigh.error().code, qcx::ErrorCode::kInvalidArgument);

    options.lMult = -2;
    auto tooLow = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(tooLow.has_value());
    EXPECT_EQ(tooLow.error().code, qcx::ErrorCode::kInvalidArgument);

    options = qcx::integrals::QfmmOptions();
    options.maxLeafSize = 0;
    auto zeroLeaf = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(zeroLeaf.has_value());
    EXPECT_EQ(zeroLeaf.error().code, qcx::ErrorCode::kInvalidArgument);

    options = qcx::integrals::QfmmOptions();
    options.theta = std::numeric_limits<double>::quiet_NaN();
    auto nanTheta = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(nanTheta.has_value());
    EXPECT_EQ(nanTheta.error().code, qcx::ErrorCode::kInvalidArgument);

    auto valid = qcx::integrals::QfmmJBuilder::Create(
        *molecule, *basis, *core, qcx::integrals::QfmmOptions());
    ASSERT_TRUE(valid.has_value()) << valid.error().message;
}

TEST(QfmmFockBuildTest, BuildFockRejectsMismatchedDensityShape) {
    // The BuildFock precondition (fock_build.cpp): the density must be
    // n x n in the pair-list function count. A wrong shape is a caller
    // bug and must fail loudly (kInvalidArgument), never read out of
    // bounds or silently produce a nonsense Fock.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto builder = qcx::integrals::QfmmJBuilder::Create(
        *molecule, *basis, *core, qcx::integrals::QfmmOptions());
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::size_t n = core->Shape()[0];

    // Build the mismatched tensor directly: qcx::testing::ToTensor is
    // square-only (it derives {n, n} from rows()), and passing a
    // non-square Eigen matrix through it reads out of bounds - an Eigen
    // assert in Debug (the CI failure this test was added with) and
    // silent garbage in Release.
    auto badTensor = qcx::integrals::test::CpuTensor2::Create({n + 1, n});
    ASSERT_TRUE(badTensor.has_value()) << badTensor.error().message;

    for (std::size_t i = 0; i < n + 1; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*badTensor)(i, j) = 0.0;
        }
    }

    badTensor->MarkHostDirty();

    auto fock = builder->BuildFock(*badTensor);
    ASSERT_FALSE(fock.has_value());
    EXPECT_EQ(fock.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(QfmmFockBuildTest, ThetaZeroGateRecoversTheDirectCoulombMatrix) {
    // THE wiring gate: at QfmmOptions::theta < 0 (the degenerate
    // theta -> 0 gate) everything is near field - the near-field
    // restriction covers every pair-pair (a no-op) and the far field is
    // empty, so the BUILDER's J must equal the plain direct-sum Coulomb
    // matrix BIT-IDENTICALLY (exact equality of doubles: the same
    // screening decisions, the same fp64 contractions, in the same serial
    // order - the monopole gate re-verified through the composed class).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    const auto direct = BuildDirectFock(*molecule, *basis, *core, density);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;

    qcx::integrals::QfmmOptions options = MeasurementOptions();
    options.theta = -1.0;
    const auto qfmm = BuildQfmmFock(*molecule, *basis, *core, density, options);
    ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

    EXPECT_TRUE(MaxAbsoluteDifference(*qfmm, *direct) == 0.0);
}

TEST(QfmmFockBuildTest, DefaultsResolveToTheNormalPreset) {
    // The resolution logic: theta = 0 / lMult = -1 pick
    // ThetaForPreset(kNormal) = 0.3 / LMultForPreset(kNormal) = 5 (the
    // recalibrated rungs), so the default-options build must be
    // bit-identical to the explicit-override build (same resolved theta ->
    // same interaction lists -> same near-field keys and the same far-field
    // pairs). The resolution path is what this test exercises (the C12
    // kNormal far field is all-near at 0.3 - the vacuous tooth - so
    // the far-field LIST machinery is not the point here; the far-alive
    // exercise lives on the C24 pin fixture in the sweep evidence).
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);

    qcx::integrals::QfmmOptions defaults = MeasurementOptions();
    const auto defaultBuild = BuildQfmmFock(*molecule, *basis, *core, density, defaults);
    ASSERT_TRUE(defaultBuild.has_value()) << defaultBuild.error().message;

    // The resolution path this test exercises: the ABSENT theta resolves to
    // ThetaForPreset(kNormal) = 0.3 - "not given" means the preset's own angle,
    // and the default classification (the centre-to-width form) is what reads
    // it - so the explicit spelling of that same angle must reproduce the
    // default build bit for bit (LMultForPreset still resolves lMult = -1).
    qcx::integrals::QfmmOptions explicitOptions = defaults;
    explicitOptions.lMult = qcx::integrals::LMultForPreset(qcx::integrals::AccuracyPreset::kNormal);
    explicitOptions.theta = qcx::integrals::ThetaForPreset(qcx::integrals::AccuracyPreset::kNormal);
    const auto explicitBuild = BuildQfmmFock(*molecule, *basis, *core, density, explicitOptions);
    ASSERT_TRUE(explicitBuild.has_value()) << explicitBuild.error().message;

    EXPECT_TRUE(MaxAbsoluteDifference(*defaultBuild, *explicitBuild) == 0.0);

    // The surface ball stays selectable and is a DIFFERENT build: what moved
    // was the default, not the availability of that form. Its pairs are a
    // different set from the angle's, so naming it must not silently equal the
    // default - the same fixture that reads 517 far pairs at the preset's angle
    // reads 2,864 on the surface ball at k = 1.
    qcx::integrals::QfmmOptions surfaceBall = defaults;
    surfaceBall.separationMode = qcx::integrals::QfmmSeparationMode::kSurfaceBall;
    surfaceBall.separationK = 1.0;
    const auto surfaceBallBuild = BuildQfmmFock(*molecule, *basis, *core, density, surfaceBall);
    ASSERT_TRUE(surfaceBallBuild.has_value()) << surfaceBallBuild.error().message;
    EXPECT_GT(MaxAbsoluteDifference(*surfaceBallBuild, *defaultBuild), 0.0);
}

TEST(QfmmFockBuildTest, CrossoverHeuristicRespectsThePlaceholderAndOverride) {
    // The crossover decision seam: the default (-1) resolves
    // to the UNMEASURED placeholder constant, the boundary is inclusive (at
    // the threshold the QFMM path IS recommended), and an explicit override
    // wins over the placeholder in both directions. A pure decision
    // function - no molecule or build needed.
    qcx::integrals::QfmmOptions defaults;

    EXPECT_FALSE(qcx::integrals::RecommendQfmmOverRiJ(
        qcx::integrals::kQfmmCrossoverBasisFunctionCount - 1, defaults));
    EXPECT_TRUE(qcx::integrals::RecommendQfmmOverRiJ(
        qcx::integrals::kQfmmCrossoverBasisFunctionCount, defaults));
    EXPECT_TRUE(qcx::integrals::RecommendQfmmOverRiJ(10000, defaults));

    // The override raises the threshold: a 500-function system that the
    // placeholder would send to QFMM now stays with RI-J.
    qcx::integrals::QfmmOptions overrideOff = defaults;
    overrideOff.crossoverBasisFunctionCount = 1000;
    EXPECT_FALSE(qcx::integrals::RecommendQfmmOverRiJ(
        qcx::integrals::kQfmmCrossoverBasisFunctionCount, overrideOff));
    EXPECT_TRUE(qcx::integrals::RecommendQfmmOverRiJ(1000, overrideOff));

    // The override lowers it to the floor: 0 recommends QFMM at any count.
    qcx::integrals::QfmmOptions overrideOn = defaults;
    overrideOn.crossoverBasisFunctionCount = 0;
    EXPECT_TRUE(qcx::integrals::RecommendQfmmOverRiJ(0, overrideOn));
    EXPECT_TRUE(qcx::integrals::RecommendQfmmOverRiJ(1, overrideOn));
}

TEST(QfmmFockBuildTest, FarFieldErrorDecreasesMonotonicallyInThetaAndLMult) {
    // The convergence properties through the BUILDER, on the C12 alkane fixture
    // (38 atoms, 86 functions, the acceptance fixture pulled forward by
    // the bit-exact gates): at every fixed L_mult the relative Frobenius
    // error must shrink as theta tightens {1.05, 0.85, 0.7}, and at every
    // fixed theta it must shrink as L_mult rises {0, 1, 2, 3}. The near-field
    // parts of the QFMM and the direct build are bitwise equal (same
    // options, same density, serial pin), so the measured error is purely
    // the far-field multipole truncation.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);

    const auto direct = BuildDirectFock(*molecule, *basis, *core, density);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const double directNorm = direct->norm();

    // errors[thetaIndex][lMult]: the theta rows {1.05, 0.85, 0.7} (looser
    // first - larger error), the L_mult columns {0, 1, 2, 3}.
    const std::array<double, 3> thetas = {1.05, 0.85, 0.7};
    std::array<std::array<double, 4>, 3> errors{};

    for (std::size_t thetaIndex = 0; thetaIndex < thetas.size(); ++thetaIndex)
    {
        for (int lMult = 0; lMult <= 3; ++lMult)
        {
            qcx::integrals::QfmmOptions options = MeasurementOptions();
            options.theta = thetas[thetaIndex];
            options.lMult = lMult;
            const auto qfmm = BuildQfmmFock(*molecule, *basis, *core, density, options);
            ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;
            errors[thetaIndex][static_cast<std::size_t>(lMult)] =
                (*direct - *qfmm).norm() / directNorm;
        }
    }

    std::cout << "QFMM relative Frobenius errors (rows theta 1.05/0.85/0.7, cols L 0/1/2/3):\n";

    for (std::size_t thetaIndex = 0; thetaIndex < thetas.size(); ++thetaIndex)
    {
        for (int lMult = 0; lMult <= 3; ++lMult)
        {
            std::cout << "  theta " << thetas[thetaIndex] << " L " << lMult << ": "
                      << errors[thetaIndex][static_cast<std::size_t>(lMult)] << "\n";
        }
    }

    // At every fixed theta: strictly decreasing in L_mult (0 > 1 > 2).
    for (std::size_t thetaIndex = 0; thetaIndex < thetas.size(); ++thetaIndex)
    {
        for (int lMult = 0; lMult < 3; ++lMult)
        {
            EXPECT_LT(errors[thetaIndex][static_cast<std::size_t>(lMult + 1)],
                      errors[thetaIndex][static_cast<std::size_t>(lMult)])
                << "theta = " << thetas[thetaIndex] << " lMult " << lMult << " -> " << lMult + 1;
        }
    }

    // At every fixed L_mult: strictly decreasing in theta (tighter theta,
    // smaller error).
    for (int lMult = 0; lMult <= 3; ++lMult)
    {
        for (std::size_t thetaIndex = 0; thetaIndex + 1 < thetas.size(); ++thetaIndex)
        {
            EXPECT_LT(errors[thetaIndex + 1][static_cast<std::size_t>(lMult)],
                      errors[thetaIndex][static_cast<std::size_t>(lMult)])
                << "lMult = " << lMult << " theta " << thetas[thetaIndex] << " -> "
                << thetas[thetaIndex + 1];
        }
    }

    // The machinery guard (the monopole-gate sanity bound carried forward): the
    // loosest rung (1.05, L = 0) is the monopole order at the loosest
    // theta - its error must be genuinely coarse (a silently dead far
    // field would look "perfect" and must be caught here) and finite.
    EXPECT_GT(errors[0][0], 1e-3);
    EXPECT_LT(errors[0][0], 0.99);
}

TEST(QfmmFockBuildTest, PresetLadderMatchesTheRecordedSweepRungs) {
    // The preset validation (recalibrated by the preset-ladder sweep):
    // at each accuracy preset the combination (ThetaForPreset,
    // LMultForPreset) must stay inside its committed budget - the
    // QFMM budgets {1e-5, 1e-7, 1e-8} Eh (the kTight rung re-derived from
    // 1e-9 by the 2026-09-13 ruling; the ladder is
    // QfmmBudgetForPreset, accuracy.hpp)
    // (benchmarks/data/qfmm_ladder_sweep_full.csv - the
    // QFMM-vs-preset budget split: the preset screening budgets keep applying to
    // the direct/RI rows only) - with a strict top rung (kLoose > kNormal
    // in both the relative Frobenius and the J-energy metric: measured
    // 1.78e-7 > 0.0 / 1.605e-6 > 0.0 on this fixture). The bottom boundary
    // is degenerate: the (kNormal, C12) tooth at the C24-pinned (0.3, 5)
    // is vacuous (all-near - the near path's screening thresholds match
    // the reference's, so it is the bit-exact direct path and measures
    // 0.0 exactly), and the kTight rung is the theta -> 0 degenerate gate
    // (the near-field-only path - its near path screens at the preset's
    // 1e-12, a strict superset of the kNormal-screened reference's 1e-10,
    // so the QFMM side is the MORE accurate one and the measured residual
    // is the reference's OWN screening error: ~3.33e-9 on this fixture,
    // deterministic, the sweep's tight-end floor (re-measured
    // 3.33386e-9 on 2026-09-13), asserted as the
    // (1e-9, 1e-8) band below - a strict monotonicity assertion
    // does not survive the measurement across the
    // degenerate boundary, so it is not made).
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);

    const auto direct = BuildDirectFock(*molecule, *basis, *core, density);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const double directNorm = direct->norm();

    std::cout << "QFMM preset sweep (C12 alkane, J-energy error in Eh):\n";
    double frobeniusByPreset[3]{};
    double energyByPreset[3]{};

    for (const qcx::integrals::AccuracyPreset preset : {qcx::integrals::AccuracyPreset::kLoose,
                                                        qcx::integrals::AccuracyPreset::kNormal,
                                                        qcx::integrals::AccuracyPreset::kTight})
    {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = preset;
        // The preset ladder is a record of the RECORDED model - its pins (the
        // vacuous kNormal tooth measuring 0.0 exactly, the kTight theta -> 0
        // floor band) describe the retired midpoint/width far field. Pin it so
        // the record keeps measuring what it recorded; the corrected default's
        // ladder accuracy is carried by TheCorrectedDefaultStaysInsideThe-
        // PresetBudgetsOnC12 below.
        options.extentModel = qcx::integrals::QfmmExtentMode::kMidpointBound;
        options.separationMode = qcx::integrals::QfmmSeparationMode::kWidthTheta;
        options.separationK = 0.0;
        const auto qfmm = BuildQfmmFock(*molecule, *basis, *core, density, options);
        ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

        const double frobenius = (*direct - *qfmm).norm() / directNorm;
        const double energy = EnergyError(*qfmm, *direct, density);
        frobeniusByPreset[static_cast<int>(preset)] = frobenius;
        energyByPreset[static_cast<int>(preset)] = energy;
        std::cout << "  " << static_cast<int>(preset) << ": theta "
                  << qcx::integrals::ThetaForPreset(preset) << " L "
                  << qcx::integrals::LMultForPreset(preset) << ": frobenius " << frobenius
                  << " energy " << energy << "\n";

        // The strict top rung: kLoose > kNormal in both metrics on this
        // fixture (measured 1.78e-7 > 0.0 / 1.605e-6 > 0.0). The
        // kNormal -> kTight transition is the degenerate boundary: the
        // kNormal tooth is the bit-exact direct path (0.0) while the
        // kTight rung's residual is the reference's own screening floor
        // (~3.33e-9) - the measured metric is NOT monotone across the
        // boundary; the kTight rung is asserted as the
        // recorded floor band in the budget block below.
        if (preset == qcx::integrals::AccuracyPreset::kNormal)
        {
            EXPECT_LT(frobenius,
                      frobeniusByPreset[static_cast<int>(qcx::integrals::AccuracyPreset::kLoose)])
                << "preset " << static_cast<int>(preset);
            EXPECT_LT(energy,
                      energyByPreset[static_cast<int>(qcx::integrals::AccuracyPreset::kLoose)])
                << "preset " << static_cast<int>(preset);
        }

        // The committed budgets (the preset-ladder recalibration): the QFMM
        // production budgets {1e-5, 1e-7, 1e-8} for {kLoose, kNormal,
        // kTight}, measured on the C12/C24 chains at the per-preset
        // extents {1e-6, 1e-8, 1e-10}. The earlier 0.71 / 0.38 / 1.5e-3
        // were calibrated at the all-tau = 1e-10 extent and are not
        // transferable. On this fixture: kLoose (0.45, 5)
        // measures 1.605e-6 - 6.2x slack, less than the pick rule's
        // >= 10x (recorded honestly - the budget is not widened);
        // kNormal (0.3, 5) is the bit-exact direct path (the vacuous
        // tooth - near thresholds match the reference's). kTight's
        // theta-0 residual (3.33e-9) is the recorded floor, which the
        // re-derived 1e-8 budget now CONTAINS - so the kTight branch below
        // asserts the floor band (1e-9, 1e-8), not a budget edge: the
        // floor is the comparison reference's own screening error and must
        // stay visible as a measurement, never be re-read as QFMM headroom
        // (the budget-versus-floor question was resolved by the 2026-09-13
        // ruling: the floor band stands, the budget is not widened).
        if (preset == qcx::integrals::AccuracyPreset::kLoose ||
            preset == qcx::integrals::AccuracyPreset::kNormal)
        {
            const double budget = preset == qcx::integrals::AccuracyPreset::kLoose ? 1e-5 : 1e-7;
            EXPECT_LT(energy, budget) << "preset " << static_cast<int>(preset);
        } else
        {
            // The kTight floor band: at theta 0.0 the
            // near-field-only path screens at the preset's 1e-12 - a
            // strict superset of the kNormal-screened reference (the
            // fixture's BuildDirectFock) - so the residual is the
            // reference's own dropped-quartet contribution (the QFMM side
            // is the more accurate one). The band pins the recorded C12
            // floor 3.33e-9 from BOTH sides: inside 1e-8 (the re-derived
            // kTight budget - a residual at or above it would mean the
            // budget is genuinely missed) and above 1e-9 (a floor, not a
            // measurement accident: the deterministic serial fp64
            // residual must stay above the threshold scale the reference
            // was screened at, so a drop to ~0 here would mean the
            // comparison had silently become equal-threshold and the floor
            // claim untested). Both bounds are UNCHANGED by the budget
            // re-derivation; only their reasons are - do not loosen the
            // upper one further, and never delete the lower one.
            EXPECT_LT(energy, 1e-8) << "preset " << static_cast<int>(preset);
            EXPECT_GT(energy, 1e-9) << "preset " << static_cast<int>(preset);
        }
    }

    // The degenerate bottom boundary on this fixture: the
    // (kNormal, C12) tooth at the C24-pinned (0.3, 5) is vacuous (zero far
    // pairs - the build is the bit-exact direct path, near thresholds
    // matching the reference's) so kNormal measures 0.0 exactly in both
    // metrics. The kTight rung (theta -> 0, the near-field-only path)
    // measures the recorded floor band instead - asserted in the loop's
    // budget block (its residual is the reference's own screening floor,
    // not a QFMM error). The C24 pin fixture's strictness
    // lives in the sweep evidence.
    EXPECT_DOUBLE_EQ(frobeniusByPreset[static_cast<int>(qcx::integrals::AccuracyPreset::kNormal)],
                     0.0);
    EXPECT_DOUBLE_EQ(energyByPreset[static_cast<int>(qcx::integrals::AccuracyPreset::kNormal)],
                     0.0);
}

TEST(QfmmFockBuildTest, FarFieldPairCountGrowsAsTheExtentLoosens) {
    // The far-alive gate: the pair extent is preset-driven, and on the C12
    // chain at a FIXED theta the far-pair count must rise as the extent
    // loosens - kLoose's tau 1e-6 shrinks the boxes, and smaller boxes pass
    // the well-separatedness test more often, which is the loosening
    // direction the measurement found. Both rungs must stay far-ALIVE
    // (count > 0) - a vacuous far field would hide the multipole truncation.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::QfmmOptions options = MeasurementOptions();
    // The fixed theta isolates the extent effect from the theta ladder.
    // The preset winners are unusable here: ThetaForPreset(kNormal) = 0.3 is
    // far-dead on C12 at every tau (the vacuous tooth) and the
    // kLoose winner 0.45 is far-alive only at tau 1e-6. The explicit 0.7
    // keeps all three per-preset extents far-alive on C12 with the
    // monotone measured counts 829 > 490 > 282 (kLoose/kNormal/kTight,
    // sweep evidence).
    options.theta = 0.7;

    // The extent ladder is a record of the RECORDED model (the sweep).
    // Pin it explicitly: since the 2026-09-15 flip the default geometry is the
    // product ball, whose radius is preset-driven too but whose coupling to
    // tau is not this record's (measured under the product ball the order
    // INVERTS on C12 - 5050 kNormal far pairs against 5207 kTight - a new
    // calibration question, not a re-reading of the old one).
    options.extentModel = qcx::integrals::QfmmExtentMode::kMidpointBound;
    options.separationMode = qcx::integrals::QfmmSeparationMode::kWidthTheta;
    options.separationK = 0.0;

    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    auto loose = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(loose.has_value()) << loose.error().message;
    const std::size_t looseFar = loose->FarFieldPairCount();

    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto normal = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(normal.has_value()) << normal.error().message;
    const std::size_t normalFar = normal->FarFieldPairCount();

    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto tight = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(tight.has_value()) << tight.error().message;
    const std::size_t tightFar = tight->FarFieldPairCount();

    std::cout << "C12 far-pair counts at theta 0.7: kLoose " << looseFar << ", kNormal "
              << normalFar << ", kTight " << tightFar << "\n";

    EXPECT_GT(looseFar, 0u);
    EXPECT_GT(normalFar, 0u);
    EXPECT_GT(tightFar, 0u);
    EXPECT_GE(looseFar, normalFar);
    EXPECT_GE(normalFar, tightFar);
    // The loose extent must actually shrink the boxes, not just break even
    // (the far-alive direction has to be exercised, not vacuous).
    EXPECT_GT(looseFar, tightFar);
}

TEST(QfmmFockBuildTest, BuildFockIsDeterministicAndIndependentOfInstance) {
    // The composed build's determinism invariant: with the serial pin the
    // near-field reduction order is fixed and the far field is
    // deterministic by construction, so two identical builder instances
    // must produce bit-identical results, and two consecutive calls on one
    // instance must too (the SCF delta-accumulation and DIIS consumers
    // depend on repeatable Focks).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    const qcx::integrals::QfmmOptions options = MeasurementOptions();
    auto first = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto second = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(second.has_value()) << second.error().message;

    auto firstCall = first->BuildFock(*densityTensor);
    ASSERT_TRUE(firstCall.has_value()) << firstCall.error().message;
    auto secondCall = first->BuildFock(*densityTensor);
    ASSERT_TRUE(secondCall.has_value()) << secondCall.error().message;
    auto otherInstance = second->BuildFock(*densityTensor);
    ASSERT_TRUE(otherInstance.has_value()) << otherInstance.error().message;

    EXPECT_TRUE(MaxAbsoluteDifference(ToMatrix(*firstCall), ToMatrix(*secondCall)) == 0.0);
    EXPECT_TRUE(MaxAbsoluteDifference(ToMatrix(*firstCall), ToMatrix(*otherInstance)) == 0.0);
}

TEST(QfmmFockBuildTest, ChunkedFarFieldBuildMatchesTheSerialPinWithinBudget) {
    // The chunked-far-field gate: the
    // chunked parallel far field (maxParallelChunks = 2 - the M2L chunked
    // over its far pairs with the fixed-order partial join, the L2L per
    // root-child subtree) must equal the serial pin within the preset
    // budget on the C12/C24 fixtures. The only chunked far-field
    // arithmetic difference is the fp grouping of each node's M2L adds
    // (last bits - budget-irrelevant for the approximate far field), and
    // the near field's own chunked merge scatter rides the same band (the
    // documented last-ulp tolerance). Every leg must be far-ALIVE
    // (FarFieldPairCount > 0) - a vacuous leg would not exercise the
    // chunked M2L/L2L at all. The band is budget / 100 - the assertion is
    // honest slack (measured deltas are ~1e-13 and below; a structural
    // chunking bug would land at the budget scale or worse).
    auto runLeg = [](const char* name,
                     const qcx::molecule::Molecule& molecule,
                     const qcx::basisset::BasisSet& basisSet,
                     const qcx::integrals::test::CpuTensor2& core,
                     const Eigen::MatrixXd& density,
                     double theta,
                     int lMult,
                     qcx::integrals::AccuracyPreset preset) {
        qcx::integrals::QfmmOptions serialOptions = MeasurementOptions();
        serialOptions.accuracy = preset;
        serialOptions.theta = theta;
        serialOptions.lMult = lMult;
        const auto serial = BuildQfmmFock(molecule, basisSet, core, density, serialOptions);
        ASSERT_TRUE(serial.has_value()) << serial.error().message;

        qcx::integrals::QfmmOptions chunkedOptions = serialOptions;
        chunkedOptions.maxParallelChunks = 2;
        const auto chunked = BuildQfmmFock(molecule, basisSet, core, density, chunkedOptions);
        ASSERT_TRUE(chunked.has_value()) << chunked.error().message;

        // The far-alive guard: the leg's octree must carry far pairs for
        // the chunked M2L/L2L to split (the liveness flag of the builder).
        auto builder =
            qcx::integrals::QfmmJBuilder::Create(molecule, basisSet, core, chunkedOptions);
        ASSERT_TRUE(builder.has_value()) << builder.error().message;
        ASSERT_GT(builder->FarFieldPairCount(), 0u) << name;

        const double energyVsSerial = EnergyError(*chunked, *serial, density);
        const double maxAbsVsSerial = MaxAbsoluteDifference(*chunked, *serial);
        std::cout << "chunked leg " << name << ": theta " << theta << " L " << lMult << " preset "
                  << static_cast<int>(preset) << ", far pairs " << builder->FarFieldPairCount()
                  << ": chunked-vs-serial J-energy " << energyVsSerial << ", max |diff| "
                  << maxAbsVsSerial << "\n";
        EXPECT_LT(energyVsSerial, qcx::integrals::QfmmBudgetForPreset(preset) * 1e-2) << name;
    };

    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // The far-alive legs of the swept comparison (the sweep evidence): C12 at the
    // explicit theta = 0.7 and theta = 0.45 configs (L = 5), C24 at the
    // same two. The theta is passed EXPLICITLY here; the preset argument
    // names the budget band only, and kNormal's own theta is 0.3 - so
    // these are not "the kNormal config".
    {
        auto molecule = MakeAlkaneSto3g(12);
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        auto basis = MakeAlkaneSto3gBasis();
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        auto core = BuildCoreHamiltonian(*molecule, *basis);
        ASSERT_TRUE(core.has_value()) << core.error().message;
        const std::size_t n = core->Shape()[0];
        const Eigen::MatrixXd density = PhysicalDensity(n);
        runLeg("C12",
               *molecule,
               *basis,
               *core,
               density,
               0.7,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
        runLeg("C12",
               *molecule,
               *basis,
               *core,
               density,
               0.45,
               5,
               qcx::integrals::AccuracyPreset::kLoose);
    }

    {
        auto molecule = MakeAlkaneSto3g(24);
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        auto basis = MakeAlkaneSto3gBasis();
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        auto core = BuildCoreHamiltonian(*molecule, *basis);
        ASSERT_TRUE(core.has_value()) << core.error().message;
        const std::size_t n = core->Shape()[0];
        const Eigen::MatrixXd density = PhysicalDensity(n);
        runLeg("C24",
               *molecule,
               *basis,
               *core,
               density,
               0.7,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
        runLeg("C24",
               *molecule,
               *basis,
               *core,
               density,
               0.45,
               5,
               qcx::integrals::AccuracyPreset::kLoose);
    }
}

TEST(QfmmFockBuildTest, ChunkedBuildsAreDeterministicWithinTheMergeScatterBand) {
    // The determinism gate in the chunked configuration: the far field
    // is bit-reproducible at any chunk count (the fixed-order M2L join and
    // the single-writer leaf/L2L/accumulation passes - pinned bit-exactly
    // by ChunkedFarFieldPassesAreSingleWriterBitExact below). The nested
    // direct builder's critical-section merge scatter that used to be the
    // residual source of run-to-run variation is GONE - the
    // fixed-order join (internal/fixed_order_reduce.hpp) pinned that
    // reduction's merge order, so this band is no longer carrying a
    // scatter - it is kept as the gate's own bound, with the measured
    // values printed. Two instances and two consecutive calls must agree
    // within it.
    auto runLeg = [](const char* name,
                     const qcx::molecule::Molecule& molecule,
                     const qcx::basisset::BasisSet& basisSet,
                     const qcx::integrals::test::CpuTensor2& core,
                     const Eigen::MatrixXd& density,
                     // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                     double theta,
                     int lMult,
                     qcx::integrals::AccuracyPreset preset) {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = preset;
        options.theta = theta;
        options.lMult = lMult;
        options.maxParallelChunks = 2;
        auto first = qcx::integrals::QfmmJBuilder::Create(molecule, basisSet, core, options);
        ASSERT_TRUE(first.has_value()) << first.error().message;
        auto second = qcx::integrals::QfmmJBuilder::Create(molecule, basisSet, core, options);
        ASSERT_TRUE(second.has_value()) << second.error().message;

        auto densityTensor = ToTensor(density);
        ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
        auto firstCall = first->BuildFock(*densityTensor);
        ASSERT_TRUE(firstCall.has_value()) << firstCall.error().message;
        auto secondCall = first->BuildFock(*densityTensor);
        ASSERT_TRUE(secondCall.has_value()) << secondCall.error().message;
        auto otherInstance = second->BuildFock(*densityTensor);
        ASSERT_TRUE(otherInstance.has_value()) << otherInstance.error().message;

        const double callVsCall =
            MaxAbsoluteDifference(ToMatrix(*firstCall), ToMatrix(*secondCall));
        const double instanceVsInstance =
            MaxAbsoluteDifference(ToMatrix(*firstCall), ToMatrix(*otherInstance));
        std::cout << "determinism leg " << name << ": chunked max |diff| call-vs-call "
                  << callVsCall << ", instance-vs-instance " << instanceVsInstance << "\n";
        EXPECT_LT(callVsCall, 1e-12) << name;
        EXPECT_LT(instanceVsInstance, 1e-12) << name;
    };

    // H2O (all-near - the chunked exercise is the near-field merge) and
    // the far-alive C12 leg (the chunked far field rides along).
    {
        auto molecule = MakeH2oSto3g();
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        auto basis = MakeH2oSto3gBasis();
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        auto core = BuildCoreHamiltonian(*molecule, *basis);
        ASSERT_TRUE(core.has_value()) << core.error().message;
        runLeg("H2O",
               *molecule,
               *basis,
               *core,
               PhysicalDensity(7),
               0.3,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
    }

    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    {
        auto molecule = MakeAlkaneSto3g(12);
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        auto basis = MakeAlkaneSto3gBasis();
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        auto core = BuildCoreHamiltonian(*molecule, *basis);
        ASSERT_TRUE(core.has_value()) << core.error().message;
        const std::size_t n = core->Shape()[0];
        const Eigen::MatrixXd density = PhysicalDensity(n);
        runLeg("C12",
               *molecule,
               *basis,
               *core,
               density,
               0.7,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
    }
}

TEST(QfmmFockBuildTest, ChunkedFarFieldPassesAreSingleWriterBitExact) {
    // The pass-level decomposition of the determinism gate: the leaf
    // aggregation and the far-field J accumulation are single-writer per
    // element, so their chunked evaluation is BIT-identical to the serial
    // in any schedule; the M2L is the one reduction, and its chunked
    // result is bit-reproducible run to run (the fixed-order partial
    // join) while differing from the serial only in the fp grouping of
    // each node's adds (last bits). The L2L is single-writer per node
    // block (each written by its parent's step), so it is bit-identical
    // in any schedule too - the chunked-vs-serial potential difference
    // below is exactly the M2L join grouping.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;
    const auto geometries = qcx::integrals::internal::ComputePairGeometries(
        *pairStore, qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kNormal));
    const auto tree = qcx::integrals::internal::BuildQfmmTree(geometries);
    ASSERT_TRUE(tree.has_value()) << tree.error().message;
    std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
    qcx::integrals::internal::BuildInteractionLists(*tree, 0.7, farFieldPairs, nearFieldLeafPairs);
    ASSERT_GT(farFieldPairs.size(), 0u);

    // The fixed-order (uniform) order vectors - the pass-level mirror of the
    // gated helper's uniform-order shape.
    const std::vector<int> pairOrders(pairList->pairs.size(), 5);
    const std::vector<int> nodeOrders(tree->nodes.size(), 5);
    const auto table =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, pairOrders);
    ASSERT_TRUE(table.has_value()) << table.error().message;

    auto maxAbsDiff = [](const std::vector<double>& a, const std::vector<double>& b) {
        double maximum = 0.0;

        for (std::size_t i = 0; i < a.size(); ++i)
        {
            maximum = std::max(maximum, std::abs(a[i] - b[i]));
        }

        return maximum;
    };

    // The leaf aggregation: single-writer per leaf - bit-identical across
    // every chunk count.
    const auto leafSerial = qcx::integrals::internal::AggregateLeafMoments(
        *pairList, *tree, *table, geometries, density, nodeOrders, 1);
    const auto leafChunked = qcx::integrals::internal::AggregateLeafMoments(
        *pairList, *tree, *table, geometries, density, nodeOrders, 2);
    const auto leafChunked4 = qcx::integrals::internal::AggregateLeafMoments(
        *pairList, *tree, *table, geometries, density, nodeOrders, 4);
    EXPECT_TRUE(maxAbsDiff(leafSerial, leafChunked) == 0.0);
    EXPECT_TRUE(maxAbsDiff(leafSerial, leafChunked4) == 0.0);

    // The M2M (serial by design - reverse pre-order) feeds every
    // potential build identically.
    const auto nodeMoments =
        qcx::integrals::internal::AggregateNodeMoments(*tree, leafSerial, nodeOrders);

    // The potentials: run-to-run bit-reproducible at every chunk count
    // (two consecutive calls at the same chunk count must be bit-equal);
    // chunked-vs-serial differs only in the M2L join grouping (the last
    // bits - the band below is 1e-10; a structural chunking bug would
    // land far above).
    const auto potentialsSerial = qcx::integrals::internal::BuildFarFieldPotentials(
        *tree, farFieldPairs, nodeMoments, pairOrders, nodeOrders, 1);
    const auto potentialsChunked = qcx::integrals::internal::BuildFarFieldPotentials(
        *tree, farFieldPairs, nodeMoments, pairOrders, nodeOrders, 2);
    const auto potentialsChunkedRepeat = qcx::integrals::internal::BuildFarFieldPotentials(
        *tree, farFieldPairs, nodeMoments, pairOrders, nodeOrders, 2);
    const auto potentialsChunked4 = qcx::integrals::internal::BuildFarFieldPotentials(
        *tree, farFieldPairs, nodeMoments, pairOrders, nodeOrders, 4);
    const auto potentialsChunked4Repeat = qcx::integrals::internal::BuildFarFieldPotentials(
        *tree, farFieldPairs, nodeMoments, pairOrders, nodeOrders, 4);
    EXPECT_TRUE(maxAbsDiff(potentialsChunked, potentialsChunkedRepeat) == 0.0);
    EXPECT_TRUE(maxAbsDiff(potentialsChunked4, potentialsChunked4Repeat) == 0.0);
    EXPECT_LT(maxAbsDiff(potentialsSerial, potentialsChunked), 1e-10);
    EXPECT_LT(maxAbsDiff(potentialsSerial, potentialsChunked4), 1e-10);
    std::cout << "potentials: chunked-vs-serial max |diff| at 2 chunks "
              << maxAbsDiff(potentialsSerial, potentialsChunked) << ", at 4 chunks "
              << maxAbsDiff(potentialsSerial, potentialsChunked4) << "\n";

    // The accumulation: single-writer per fock element (each canonical
    // pair lives in exactly one leaf and the pairs' blocks are disjoint)
    // - with the SAME potentials the chunked accumulation is bit-identical
    // to the serial.
    Eigen::MatrixXd fockSerial =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    Eigen::MatrixXd fockChunked =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    qcx::integrals::internal::AccumulateFarFieldJ(
        fockSerial, *pairList, *tree, *table, potentialsSerial, geometries, nodeOrders, 1);
    qcx::integrals::internal::AccumulateFarFieldJ(
        fockChunked, *pairList, *tree, *table, potentialsSerial, geometries, nodeOrders, 2);
    EXPECT_TRUE(MaxAbsoluteDifference(fockSerial, fockChunked) == 0.0);
    // And with the chunked potentials the composed far field lands within
    // the same last-bits band as the potentials themselves.
    Eigen::MatrixXd fockFromChunkedPotentials =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    qcx::integrals::internal::AccumulateFarFieldJ(fockFromChunkedPotentials,
                                                  *pairList,
                                                  *tree,
                                                  *table,
                                                  potentialsChunked,
                                                  geometries,
                                                  nodeOrders,
                                                  1);
    EXPECT_LT(MaxAbsoluteDifference(fockSerial, fockFromChunkedPotentials), 1e-10);
}

TEST(QfmmFockBuildTest, DefaultOptionsBuildMatchesDirectWithinTolerance) {
    // The production-path smoke test: the DEFAULT options (density
    // screening, the certified mixed-precision lane, auto-parallel) must run and match
    // the direct ground truth within a loose tolerance - that lane and
    // the reduction-order differences preclude bit-exactness here (that is
    // the serial fp64-only gates' job), but a gross wiring error would
    // show up as an O(1) mismatch.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    const auto direct = BuildDirectFock(*molecule, *basis, *core, density);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;

    const auto qfmm =
        BuildQfmmFock(*molecule, *basis, *core, density, qcx::integrals::QfmmOptions());
    ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

    EXPECT_LT(MaxAbsoluteDifference(*qfmm, *direct), 1e-6);
}

TEST(QfmmFockBuildTest, LeafDrivenNearFieldIsBitIdenticalToTheGatedSweep) {
    // The leaf-driven-versus-gated gate: the
    // production near field enumerates its candidate rows from the
    // near-field leaf pairs (QfmmJBuilder::Create sets the
    // LeafNearFieldDomain carrier) instead of sweeping the full pair space
    // and gating every candidate through the PairPairRestriction bitset.
    // On every QFMM fixture and near/far configuration the leaf-driven
    // build must be BIT-identical to the gated-sweep build under the same
    // geometry extent and screening preset (same chunking, serial pin):
    // the two enumerations deliver the same candidate rows in the same
    // order, so the screening, the routing and the contractions agree
    // exactly. The all-near legs (theta -1) re-verify the theta -> 0 gate
    // through the new path (the near-field-only build is the direct build);
    // the far-alive legs exercise the full near + far composition.
    auto runLeg = [](const char* name,
                     const qcx::molecule::Molecule& molecule,
                     const qcx::basisset::BasisSet& basisSet,
                     const qcx::integrals::test::CpuTensor2& core,
                     const Eigen::MatrixXd& density,
                     double theta,
                     int lMult,
                     qcx::integrals::AccuracyPreset preset) {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = preset;
        options.theta = theta;
        options.lMult = lMult;
        // Same geometry as the gated reference (see the pin below): the
        // equivalence is only a statement about one shared tree.
        options.extentModel = qcx::integrals::QfmmExtentMode::kMidpointBound;
        options.separationMode = qcx::integrals::QfmmSeparationMode::kWidthTheta;
        options.separationK = 0.0;
        const auto production = BuildQfmmFock(molecule, basisSet, core, density, options);
        ASSERT_TRUE(production.has_value()) << production.error().message;
        // Both sides are pinned to the RECORDED geometry for this gate: the
        // gated reference is a FULL-CSR sweep gated by the leaf-pair bitset
        // plus a far accumulation on top, and under the corrected default's
        // tree shape its far list reaches millions of node pairs, which takes
        // this test from minutes to tens of minutes per leg (measured
        // 2026-09-15). The enumeration-equivalence claim is geometry-parametric
        // (the two paths walk the same tree), so the recorded geometry still
        // tests it; the corrected default's own near-field behaviour is carried
        // by TheCorrectedPathIsTheDefaultAndTheDegenerateGateStaysEmpty, and the
        // reference harness itself needs a leaf-driven form before this gate
        // can cover the corrected geometry affordably (reported, not done).
        const auto gated = BuildQfmmFockGated(molecule,
                                              basisSet,
                                              core,
                                              density,
                                              theta,
                                              lMult,
                                              preset,
                                              options.extentModel,
                                              options.separationMode,
                                              options.separationK);
        ASSERT_TRUE(gated.has_value()) << gated.error().message;

        const double difference = MaxAbsoluteDifference(*production, *gated);
        std::cout << "gated leg " << name << ": theta " << theta << " L " << lMult << " preset "
                  << static_cast<int>(preset) << ": max |diff| " << difference << "\n";
        EXPECT_TRUE(difference == 0.0) << name;
    };

    // H2O (the 7-function fixture, all-near at any theta).
    {
        auto molecule = MakeH2oSto3g();
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        auto basis = MakeH2oSto3gBasis();
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        auto core = BuildCoreHamiltonian(*molecule, *basis);
        ASSERT_TRUE(core.has_value()) << core.error().message;
        const Eigen::MatrixXd density = PhysicalDensity(7);
        runLeg("H2O",
               *molecule,
               *basis,
               *core,
               density,
               -1.0,
               0,
               qcx::integrals::AccuracyPreset::kTight);
        runLeg("H2O",
               *molecule,
               *basis,
               *core,
               density,
               0.3,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
    }

    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    // C12 (86 functions, 3741 pairs): the vacuous-tooth all-near config
    // (kNormal theta 0.3 - the near set is the FULL pair space, the
    // largest leaf-driven enumeration), two far-alive configs (kNormal
    // theta 0.7 and kLoose theta 0.45 at the per-preset extents, the
    // committed sweep evidence), and the degenerate theta -> 0 gate at kTight.
    {
        auto molecule = MakeAlkaneSto3g(12);
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        auto basis = MakeAlkaneSto3gBasis();
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        auto core = BuildCoreHamiltonian(*molecule, *basis);
        ASSERT_TRUE(core.has_value()) << core.error().message;
        const std::size_t n = core->Shape()[0];
        const Eigen::MatrixXd density = PhysicalDensity(n);
        runLeg("C12",
               *molecule,
               *basis,
               *core,
               density,
               0.3,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
        runLeg("C12",
               *molecule,
               *basis,
               *core,
               density,
               0.7,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
        runLeg("C12",
               *molecule,
               *basis,
               *core,
               density,
               0.45,
               5,
               qcx::integrals::AccuracyPreset::kLoose);
        runLeg("C12",
               *molecule,
               *basis,
               *core,
               density,
               -1.0,
               0,
               qcx::integrals::AccuracyPreset::kTight);
    }

    // C24 (170 functions, 14535 pairs): the far-alive legs at the C24
    // kLoose/kNormal pin rungs.
    {
        auto molecule = MakeAlkaneSto3g(24);
        ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
        auto basis = MakeAlkaneSto3gBasis();
        ASSERT_TRUE(basis.has_value()) << basis.error().message;
        auto core = BuildCoreHamiltonian(*molecule, *basis);
        ASSERT_TRUE(core.has_value()) << core.error().message;
        const std::size_t n = core->Shape()[0];
        const Eigen::MatrixXd density = PhysicalDensity(n);
        runLeg("C24",
               *molecule,
               *basis,
               *core,
               density,
               0.7,
               5,
               qcx::integrals::AccuracyPreset::kNormal);
        runLeg("C24",
               *molecule,
               *basis,
               *core,
               density,
               0.45,
               5,
               qcx::integrals::AccuracyPreset::kLoose);
    }
}

TEST(QfmmFockBuildTest, LeafDrivenNeighborRowsAreTheGatedRowsEverywhere) {
    // The completeness gate: every near-field leaf pair's pair-pairs
    // appear in the leaf-driven candidate rows EXACTLY ONCE, in the gated
    // sweep's own order. Checked two ways on a far-alive C12 tree (both
    // diagonal and cross near-field leaf pairs present): (1) row-wise, the
    // leaf-driven CSR equals the full Schwarz CSR filtered by the
    // restriction bit test - the same rows, the same ascending order, no
    // duplicates; (2) per near-field leaf pair, the number of CSR entries
    // whose pair-pairs live in that leaf pair equals the leaf pair's own
    // product count under the same pair cutoff - the "exactly once" count
    // from the leaf side.
    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;

    constexpr qcx::integrals::AccuracyPreset preset = qcx::integrals::AccuracyPreset::kNormal;
    const double theta = 0.7;
    const auto geometries = qcx::integrals::internal::ComputePairGeometries(
        *pairStore, qcx::integrals::QfmmExtentForPreset(preset));
    const auto tree = qcx::integrals::internal::BuildQfmmTree(geometries);
    ASSERT_TRUE(tree.has_value()) << tree.error().message;

    std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
    qcx::integrals::internal::BuildInteractionLists(
        *tree, theta, farFieldPairs, nearFieldLeafPairs);

    qcx::integrals::PairPairRestriction restriction;
    restriction.nLeaves = tree->nodes.size();
    restriction.leafOfPair = tree->leafOfPair;
    restriction.words.assign((tree->nodes.size() * tree->nodes.size() + 63) / 64, 0);

    for (const auto& [leafA, leafB] : nearFieldLeafPairs)
    {
        restriction.InsertBoth(leafA, leafB);
    }

    qcx::integrals::LeafNearFieldDomain domain;
    domain.nLeaves = tree->nodes.size();
    domain.leafOfPair = tree->leafOfPair;
    domain.nearFieldLeafPairs = nearFieldLeafPairs;

    std::vector<std::size_t> fullOffsets;
    std::vector<std::size_t> fullIndices;
    std::vector<std::size_t> domainOffsets;
    std::vector<std::size_t> domainIndices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, preset, fullOffsets, fullIndices);
    qcx::integrals::internal::BuildLeafDrivenNeighborList(
        *pairList, *schwarz, preset, domain, domainOffsets, domainIndices);

    const std::size_t nPairs = pairList->pairs.size();
    EXPECT_EQ(domainOffsets.size(), nPairs + 1);

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        std::vector<std::size_t> expected;

        for (std::size_t idx = fullOffsets[bra]; idx < fullOffsets[bra + 1]; ++idx)
        {
            const std::size_t ket = fullIndices[idx];

            if (restriction.Contains(bra, ket))
            {
                expected.push_back(ket);
            }
        }

        const std::size_t rowStart = domainOffsets[bra];
        const std::size_t rowEnd = domainOffsets[bra + 1];
        ASSERT_EQ(rowEnd - rowStart, expected.size()) << "row " << bra;

        for (std::size_t k = 0; k < expected.size(); ++k)
        {
            EXPECT_EQ(domainIndices[rowStart + k], expected[k]) << "row " << bra << " ket " << k;
        }
    }

    // The leaf-side count: each CSR entry belongs to exactly one
    // near-field leaf pair (the pair-pair's own leaf pair), so per leaf
    // pair the CSR entries must count the leaf pair's cutoff-passing
    // products exactly once.
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> csrCountPerLeafPair;

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        for (std::size_t idx = domainOffsets[bra]; idx < domainOffsets[bra + 1]; ++idx)
        {
            const std::size_t leafA = tree->leafOfPair[bra];
            const std::size_t leafB = tree->leafOfPair[domainIndices[idx]];
            ++csrCountPerLeafPair[{std::min(leafA, leafB), std::max(leafA, leafB)}];
        }
    }

    // The near-field list is unordered with a <= b (qfmm_tree.cpp), so
    // each (leafA, leafB) pair indexes its own product set.
    for (const auto& [leafA, leafB] : nearFieldLeafPairs)
    {
        std::size_t expected = 0;

        for (const std::size_t p : tree->nodes[leafA].pairIndices)
        {
            for (const std::size_t q : tree->nodes[leafB].pairIndices)
            {
                const std::size_t bra = std::max(p, q);
                const std::size_t ket = std::min(p, q);

                if ((*schwarz)[bra] * (*schwarz)[ket] >=
                    qcx::integrals::SchwarzThreshold(preset) *
                        qcx::integrals::internal::kNeighborListSlack)
                {
                    // The diagonal leaf pair's unordered products: (p, p)
                    // once, (p, q) and (q, p) as the one canonical pair.
                    const bool diagonal = leafA == leafB;

                    if (diagonal && p < q)
                    {
                        continue;
                    }

                    ++expected;
                }
            }
        }

        const std::size_t counted = csrCountPerLeafPair[{leafA, leafB}];

        EXPECT_EQ(counted, expected) << "leaf pair (" << leafA << ", " << leafB << ")";
    }
}

TEST(QfmmFockBuildTest, TheCorrectedPathIsTheDefaultAndTheDegenerateGateStaysEmpty) {
    // The owner ruling of 2026-09-15 ("enable the corrected path now, both
    // keys on by default"), stated as a test, with the classification that
    // ruling's own default has since moved off the surface ball. Two
    // obligations ride the flip: the build must RUN the corrected model
    // (product-distribution extents, classified at the preset's own angle) and
    // SAY so (ModelRecord), and the degenerate gate must still produce the same
    // numbers it produced before the flip - the far field empty, the build the
    // exact restricted near-field direct build, bit for bit. The second is what
    // makes the default safe to trust: where the far field IS live the values
    // move, and where it is empty they must not.
    qcx::integrals::QfmmOptions defaults = MeasurementOptions();

    EXPECT_EQ(defaults.extentModel, qcx::integrals::QfmmExtentMode::kProductBall);
    // The classification default is the centre-to-width form driven by the
    // preset's OWN angle - an absent theta is "not given", not "zero degrees",
    // and the kSurfaceBall buffer is not consulted at all unless the caller
    // selects that form.
    EXPECT_EQ(defaults.separationMode, qcx::integrals::QfmmSeparationMode::kWidthTheta);
    EXPECT_DOUBLE_EQ(defaults.theta, 0.0);
    EXPECT_DOUBLE_EQ(qcx::integrals::ThetaForPreset(qcx::integrals::AccuracyPreset::kNormal), 0.3);

    // The empty-far-field fixture (H2O/STO-3G): at any setting the far field
    // holds nothing, so the flip cannot move a value here.
    auto h2o = MakeH2oSto3g();
    ASSERT_TRUE(h2o.has_value()) << h2o.error().message;
    auto h2oBasis = MakeH2oSto3gBasis();
    ASSERT_TRUE(h2oBasis.has_value()) << h2oBasis.error().message;
    auto h2oCore = BuildCoreHamiltonian(*h2o, *h2oBasis);
    ASSERT_TRUE(h2oCore.has_value()) << h2oCore.error().message;

    qcx::integrals::QfmmOptions gate = MeasurementOptions();
    gate.theta = -1.0;
    qcx::integrals::QfmmOptions retiredGate = gate;
    retiredGate.extentModel = qcx::integrals::QfmmExtentMode::kMidpointBound;
    retiredGate.separationMode = qcx::integrals::QfmmSeparationMode::kWidthTheta;
    retiredGate.separationK = 0.0;

    auto gateBuilder = qcx::integrals::QfmmJBuilder::Create(*h2o, *h2oBasis, *h2oCore, gate);
    ASSERT_TRUE(gateBuilder.has_value()) << gateBuilder.error().message;
    auto retiredGateBuilder =
        qcx::integrals::QfmmJBuilder::Create(*h2o, *h2oBasis, *h2oCore, retiredGate);
    ASSERT_TRUE(retiredGateBuilder.has_value()) << retiredGateBuilder.error().message;

    EXPECT_EQ(gateBuilder->FarFieldPairCount(), 0u);
    EXPECT_EQ(retiredGateBuilder->FarFieldPairCount(), 0u);

    const auto gateFock = BuildQfmmFock(*h2o, *h2oBasis, *h2oCore, PhysicalDensity(7), gate);
    ASSERT_TRUE(gateFock.has_value()) << gateFock.error().message;
    const auto retiredGateFock =
        BuildQfmmFock(*h2o, *h2oBasis, *h2oCore, PhysicalDensity(7), retiredGate);
    ASSERT_TRUE(retiredGateFock.has_value()) << retiredGateFock.error().message;

    EXPECT_TRUE(MaxAbsoluteDifference(*gateFock, *retiredGateFock) == 0.0);

    // The default names the model that classified -
    // the centre-to-width form at the preset's own angle, not the surface ball
    // the previous default used.
    const qcx::integrals::QfmmModelRecord record = gateBuilder->ModelRecord();
    EXPECT_EQ(record.extentModel, qcx::integrals::QfmmExtentMode::kProductBall);
    EXPECT_EQ(record.separationMode, qcx::integrals::QfmmSeparationMode::kWidthTheta);
    EXPECT_DOUBLE_EQ(record.separationK, 0.0);
    // theta < 0 is the gate under either form, and it resolves to the ZERO
    // angle: nothing is well separated, which is the identity a theta 0 reading
    // of this field means.
    EXPECT_DOUBLE_EQ(record.theta, 0.0);

    // The chain fixture where the default's far field IS live: 517 far pairs at
    // the preset's own angle (measured), so the default and the gate differ -
    // the far field carries real work - and the difference stays inside the
    // kNormal budget's order (measured 1.2e-10 Ha against the exact build,
    // C12H26/STO-3G).
    auto chain = MakeAlkaneSto3g(12);
    ASSERT_TRUE(chain.has_value()) << chain.error().message;
    auto chainBasis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(chainBasis.has_value()) << chainBasis.error().message;
    auto chainCore = BuildCoreHamiltonian(*chain, *chainBasis);
    ASSERT_TRUE(chainCore.has_value()) << chainCore.error().message;

    const std::size_t chainFunctions = chainCore->Shape()[0];
    const Eigen::MatrixXd chainDensity = PhysicalDensity(chainFunctions);
    auto chainBuilder =
        qcx::integrals::QfmmJBuilder::Create(*chain, *chainBasis, *chainCore, defaults);
    ASSERT_TRUE(chainBuilder.has_value()) << chainBuilder.error().message;
    EXPECT_GT(chainBuilder->FarFieldPairCount(), 0u);

    const auto chainDefault =
        BuildQfmmFock(*chain, *chainBasis, *chainCore, chainDensity, defaults);
    ASSERT_TRUE(chainDefault.has_value()) << chainDefault.error().message;
    const auto chainGate = BuildQfmmFock(*chain, *chainBasis, *chainCore, chainDensity, gate);
    ASSERT_TRUE(chainGate.has_value()) << chainGate.error().message;

    const double difference = MaxAbsoluteDifference(*chainDefault, *chainGate);
    EXPECT_GT(difference, 0.0);
    EXPECT_LT(difference, 1e-6);
}

TEST(QfmmFockBuildTest, TheCorrectedDefaultStaysInsideThePresetBudgetsOnC12) {
    // The flip's accuracy contract (the 2026-09-15 owner ruling), asserted as
    // a PROPERTY rather than a re-banded pin: with the default - an ABSENT
    // theta and no model keyword, so product-ball extents classified at the
    // preset's own angle - each preset's composed J must stay inside its
    // committed budget (QfmmBudgetForPreset). The recorded ladder pins are NOT
    // re-banded here; the measured default numbers are printed for the record
    // (C12H26, kNormal: 517 far pairs, 1.2e-10 Eh J-energy against the exact
    // near-field-only build, where the surface ball at k = 1 measures 2.1e-8).
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);
    const auto direct = BuildDirectFock(*molecule, *basis, *core, density);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const double directNorm = direct->norm();

    std::cout << "Corrected default, C12 preset budgets:\n";

    for (const qcx::integrals::AccuracyPreset preset : {qcx::integrals::AccuracyPreset::kLoose,
                                                        qcx::integrals::AccuracyPreset::kNormal,
                                                        qcx::integrals::AccuracyPreset::kTight})
    {
        qcx::integrals::QfmmOptions options = MeasurementOptions();
        options.accuracy = preset;
        auto builder = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(builder.has_value()) << builder.error().message;

        const auto qfmm = BuildQfmmFock(*molecule, *basis, *core, density, options);
        ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

        const double frobenius = (*direct - *qfmm).norm() / directNorm;
        const double energy = EnergyError(*qfmm, *direct, density);
        const double budget = qcx::integrals::QfmmBudgetForPreset(preset);
        std::cout << "  preset " << static_cast<int>(preset) << ": far pairs "
                  << builder->FarFieldPairCount() << ", frobenius " << frobenius << ", energy "
                  << energy << ", budget " << budget << "\n";

        EXPECT_LT(energy, budget) << "preset " << static_cast<int>(preset);
    }
}

} // namespace
