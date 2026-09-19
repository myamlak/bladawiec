// Incremental Fock wrapper tests (incremental_fock.hpp): the
// wrapper must match a direct full rebuild on EVERY call of a realistic SCF
// density sequence - up to fp64 accumulation-order roundoff. The parallel
// reduction makes the last bits schedule-dependent, so the bar is an
// element-wise 1e-10 envelope, not bit-identity; the one bit-exact leg is
// the repeated-density step, where the incremental contribution is zero and
// the accumulator must come back unchanged. Incremental Fock falls out of
// the linearity of J/K in the density (BuildFock(D1+D2) - H ==
// (BuildFock(D1)-H) + (BuildFock(D2)-H)), so any mismatch beyond that
// envelope is a real bug. The sequence covers the easy case (small,
// shrinking deltas) AND the trap case: a delta jumping above
// deltaNormEngagementGate mid-sequence (a DIIS kick), which must force a
// full rebuild and still land on the right answer, plus a repeated density
// (zero delta, incremental with an empty contribution).
//
// Screening and the certified fp32 lane are disabled (same options as
// the dense-reference pins in fock_build_test.cpp): the exactness claim is
// about the contraction linearity - the fp32 lane's delivered error is
// certified against the preset budget (up to ~1e-8 at kNormal), not against
// a 1e-10 bar.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/incremental_fock.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines (same as fock_build_test.cpp).
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

// A symmetric DEVIATION matrix: every element (diagonal included) is
// scale x U(-0.25, 0.25). scale=1 gives PhysicalDensity-like magnitudes
// (a full-scale density deviation), scale=0.01 an SCF-iteration-sized step
// (Frobenius norm ~0.01 on the 7-function fixtures, under the default 0.1
// engagement gate). The diagonal carries no 0.5 base - the base density is
// BaseDensity() below - so a "small step" really is small: a helper that
// embeds the 0.5 diagonal at every scale makes every delta huge and sends
// every call down the full-rebuild path (the easy-case trap).
// rng, n, scale: the RNG state, the matrix size, then the step scale.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd DensityStep(std::mt19937_64& rng, std::size_t n, double scale) {
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = scale * dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

// A base density with physical closed-shell magnitudes: 0.5 on the
// diagonal (D ~ N/2 in the spatial density), small deviations on top.
Eigen::MatrixXd BaseDensity(std::mt19937_64& rng, std::size_t n) {
    return 0.5 * Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n),
                                           static_cast<Eigen::Index>(n)) +
           DensityStep(rng, n, 1.0);
}

// Runs the density sequence through BOTH a direct builder (one fresh
// BuildFock per density, in the same order) and ONE persistent incremental
// builder (in sequence, the way a real SCF loop would call it), asserting
// they match element-wise on every step at 1e-10. H2O/STO-3G (7 functions).
// \param perElementScreening The per-element screening flag for both
// builders: on, the incremental delta passes rebuild the per-call pair-max
// vector per BuildFock call (the Delta-D density: a builder-level cache
// would screen the wrong density), and the envelope absorbs the filter's
// per-element skips (each inconsistent element carries <= 2*tau, far under
// the 1e-10 envelope at the delta step scales here).
void CheckSequenceMatchesDirect(const std::vector<Eigen::MatrixXd>& densities,
                                double gate,
                                int maxConsecutive,
                                bool perElementScreening = false) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions options;
    options.useDensityScreening = false;
    options.useCertifiedMixedPrecision = false;
    options.usePerElementScreening = perElementScreening;

    auto directBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(directBuilder.has_value()) << directBuilder.error().message;

    auto incrementalBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(incrementalBuilder.has_value()) << incrementalBuilder.error().message;

    qcx::integrals::IncrementalFockBuilder incremental(
        std::move(*incrementalBuilder), core->Shape()[0], gate, maxConsecutive);

    for (std::size_t step = 0; step < densities.size(); ++step)
    {
        auto densityTensor = ToTensor(densities[step]);
        ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

        auto directFock = directBuilder->BuildFock(*densityTensor);
        ASSERT_TRUE(directFock.has_value()) << directFock.error().message;

        auto incrementalFock = incremental.BuildFock(*densityTensor);
        ASSERT_TRUE(incrementalFock.has_value()) << incrementalFock.error().message;

        const Eigen::MatrixXd expected = ToMatrix(*directFock);
        const Eigen::MatrixXd actual = ToMatrix(*incrementalFock);

        for (std::size_t i = 0; i < static_cast<std::size_t>(expected.rows()); ++i)
        {
            for (std::size_t j = 0; j < static_cast<std::size_t>(expected.cols()); ++j)
            {
                EXPECT_NEAR(actual(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                            expected(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                            1e-10)
                    << "step " << step << " element (" << i << "," << j << ")";
            }
        }
    }
}

TEST(IncrementalFockBuilderTest, MatchesFullRebuildWithinRoundoff) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "full-sequence numerical pin - skipped in the fast smoke subset";
    }

    std::mt19937_64 rng(20260817);
    const std::size_t n = 7;
    const Eigen::MatrixXd base = BaseDensity(rng, n);

    std::vector<Eigen::MatrixXd> densities;
    densities.push_back(base); // call 1: first call ever - full rebuild.
    // The easy SCF-like ramp: small shrinking steps under the gate
    // (delta norms ~0.01, well under 0.1) - all incremental.
    densities.push_back(base + DensityStep(rng, n, 0.01));
    densities.push_back(base + DensityStep(rng, n, 0.005));
    densities.push_back(base + DensityStep(rng, n, 0.002));
    // The DIIS kick: a step well ABOVE deltaNormEngagementGate (0.1)
    // (delta norm ~0.5) - the incremental path must force a full rebuild
    // and still land exactly.
    densities.push_back(base + DensityStep(rng, n, 0.5));
    // And small steps after the kick: incremental again, accumulating on
    // the freshly rebuilt value.
    densities.push_back(densities.back() + DensityStep(rng, n, 0.01));
    densities.push_back(densities.back() + DensityStep(rng, n, 0.01));
    // A repeated density: zero delta - incremental with an empty
    // contribution must leave the accumulator exactly where it was.
    densities.push_back(densities.back());

    CheckSequenceMatchesDirect(densities, 0.1, 10);
}

// The same full sequence re-run with the
// per-element screening ON on both builders. The incremental delta passes
// then exercise the Delta-D-filter path: each BuildFock call - full and
// delta alike - rebuilds the shell-compressed pair-max vector from the
// density THAT call receives (a builder-level cache would silently
// screen the wrong density and the envelope would catch it). The direct
// reference filters against the full densities' maxima, the incremental
// against the deltas' - the per-element skips differ between the two paths
// by at most 2*tau per inconsistent element (the recorded bound), which is
// far under the 1e-10 envelope at the step scales here.
TEST(IncrementalFockBuilderTest, IncrementalWrapperMatchesDirectWithFilterOn) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "full-sequence numerical pin - skipped in the fast smoke subset";
    }

    std::mt19937_64 rng(20260817);
    const std::size_t n = 7;
    const Eigen::MatrixXd base = BaseDensity(rng, n);

    std::vector<Eigen::MatrixXd> densities;
    densities.push_back(base); // call 1: first call ever - full rebuild.
    densities.push_back(base + DensityStep(rng, n, 0.01));
    densities.push_back(base + DensityStep(rng, n, 0.005));
    densities.push_back(base + DensityStep(rng, n, 0.002));
    densities.push_back(base + DensityStep(rng, n, 0.5)); // the DIIS kick.
    densities.push_back(densities.back() + DensityStep(rng, n, 0.01));
    densities.push_back(densities.back() + DensityStep(rng, n, 0.01));
    densities.push_back(densities.back()); // zero delta - empty contribution.

    CheckSequenceMatchesDirect(densities, 0.1, 10, true);
}

TEST(IncrementalFockBuilderTest, RebuildTriggerFiresAfterMaxConsecutive) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "rebuild-trigger counter pin - skipped in the fast smoke subset";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions options;
    options.useDensityScreening = false;
    options.useCertifiedMixedPrecision = false;

    auto directBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(directBuilder.has_value()) << directBuilder.error().message;

    auto incrementalBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(incrementalBuilder.has_value()) << incrementalBuilder.error().message;

    constexpr int kMaxConsecutive = 10;
    qcx::integrals::IncrementalFockBuilder incremental(
        std::move(*incrementalBuilder), core->Shape()[0], 0.1, kMaxConsecutive);

    std::mt19937_64 rng(20260817);
    const Eigen::MatrixXd base = BaseDensity(rng, 7);

    // Call 1: first call ever - a full rebuild (count stays 0).
    auto densityTensor = ToTensor(base);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto firstFock = incremental.BuildFock(*densityTensor);
    ASSERT_TRUE(firstFock.has_value()) << firstFock.error().message;
    EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), 0);

    // Calls 2..11: ten tiny deltas, all incremental - the count climbs to
    // the cap (10) without ever tripping the gate (delta norm ~0.01).
    for (int step = 2; step <= 11; ++step)
    {
        auto stepTensor = ToTensor(base + DensityStep(rng, 7, 0.01));
        ASSERT_TRUE(stepTensor.has_value()) << stepTensor.error().message;
        auto fock = incremental.BuildFock(*stepTensor);
        ASSERT_TRUE(fock.has_value()) << fock.error().message;
        EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), step - 1);
    }

    // Call 12: the cap is reached (10 >= 10) - a full rebuild fires and the
    // count resets to 0, even though the delta is still tiny. The rebuilt
    // result must match the direct builder on the SAME density exactly.
    const Eigen::MatrixXd kickDensity = base + DensityStep(rng, 7, 0.01);
    auto kickTensor = ToTensor(kickDensity);
    ASSERT_TRUE(kickTensor.has_value()) << kickTensor.error().message;
    auto directKick = directBuilder->BuildFock(*kickTensor);
    ASSERT_TRUE(directKick.has_value()) << directKick.error().message;
    auto rebuiltFock = incremental.BuildFock(*kickTensor);
    ASSERT_TRUE(rebuiltFock.has_value()) << rebuiltFock.error().message;
    EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), 0);
    EXPECT_TRUE(ToMatrix(*rebuiltFock).isApprox(ToMatrix(*directKick), 1e-10));
}

TEST(IncrementalFockBuilderTest, ResetRestartsTheStateMachine) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "reset-state-machine pin - skipped in the fast smoke subset";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions options;
    options.useDensityScreening = false;
    options.useCertifiedMixedPrecision = false;

    auto directBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(directBuilder.has_value()) << directBuilder.error().message;

    auto incrementalBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(incrementalBuilder.has_value()) << incrementalBuilder.error().message;

    qcx::integrals::IncrementalFockBuilder incremental(
        std::move(*incrementalBuilder), core->Shape()[0], 0.1, 10);

    std::mt19937_64 rng(20260823);
    const Eigen::MatrixXd base = BaseDensity(rng, 7);

    // Two incremental steps on top of the base (small deltas, under the
    // gate): the first call is the base rebuild (count 0), each delta
    // call increments - the counter climbs to 2.
    auto firstTensor = ToTensor(base);
    ASSERT_TRUE(firstTensor.has_value()) << firstTensor.error().message;
    auto firstFock = incremental.BuildFock(*firstTensor);
    ASSERT_TRUE(firstFock.has_value()) << firstFock.error().message;

    const Eigen::MatrixXd stepped = base + DensityStep(rng, 7, 0.01);
    auto stepTensor = ToTensor(stepped);
    ASSERT_TRUE(stepTensor.has_value()) << stepTensor.error().message;
    auto stepFock = incremental.BuildFock(*stepTensor);
    ASSERT_TRUE(stepFock.has_value()) << stepFock.error().message;
    EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), 1);

    const Eigen::MatrixXd steppedTwice = stepped + DensityStep(rng, 7, 0.01);
    auto stepTwiceTensor = ToTensor(steppedTwice);
    ASSERT_TRUE(stepTwiceTensor.has_value()) << stepTwiceTensor.error().message;
    auto stepTwiceFock = incremental.BuildFock(*stepTwiceTensor);
    ASSERT_TRUE(stepTwiceFock.has_value()) << stepTwiceFock.error().message;
    EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), 2);

    // Reset(): the counter drops to 0 and the pre-reset accumulator is
    // gone - the next call on the CURRENT density must be a full rebuild
    // (first-call semantics), matching the direct builder.
    incremental.Reset();
    EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), 0);

    const Eigen::MatrixXd finalDensity = steppedTwice + DensityStep(rng, 7, 0.01);
    auto finalTensor = ToTensor(finalDensity);
    ASSERT_TRUE(finalTensor.has_value()) << finalTensor.error().message;
    auto directFock = directBuilder->BuildFock(*finalTensor);
    ASSERT_TRUE(directFock.has_value()) << directFock.error().message;
    auto rebuiltFock = incremental.BuildFock(*finalTensor);
    ASSERT_TRUE(rebuiltFock.has_value()) << rebuiltFock.error().message;

    EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), 0);
    EXPECT_TRUE(ToMatrix(*rebuiltFock).isApprox(ToMatrix(*directFock), 1e-10));
}

// The wrapper must surface the certified fp32-lane
// bound of the matrix it RETURNS - not just of the last contract. Each
// call's per-call bound (the sum of density-weighted quartet bounds the
// gate routed through the lane) bounds that call's own lane error
// element-wise, and the Fock accumulator is a linear sum of contracts, so
// the per-call bounds ADD: the cumulative bound must grow across
// incremental steps, reset to the single-call bound on a full rebuild,
// stay exactly flat on a zero-delta step, and dominate the actual
// element-wise deviation of the accumulated Fock from an fp64-only
// reference at the end of the sequence.
TEST(IncrementalFockBuilderTest, CertifiedBoundCumulatesAcrossIncrementalSteps) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "full-sequence numerical pin - skipped in the fast smoke subset";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::FockBuildOptions mixed;
    mixed.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    mixed.useDensityScreening = true;
    mixed.useCertifiedMixedPrecision = true;

    auto directBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixed);
    ASSERT_TRUE(directBuilder.has_value()) << directBuilder.error().message;

    auto incrementalBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, mixed);
    ASSERT_TRUE(incrementalBuilder.has_value()) << incrementalBuilder.error().message;

    qcx::integrals::IncrementalFockBuilder incremental(
        std::move(*incrementalBuilder), core->Shape()[0], 0.1, 10);

    std::mt19937_64 rng(20260829);
    const Eigen::MatrixXd base = BaseDensity(rng, 7);
    const Eigen::MatrixXd d1 = base + DensityStep(rng, 7, 0.01);
    const Eigen::MatrixXd d2 = d1 + DensityStep(rng, 7, 0.005);
    const Eigen::MatrixXd d3 = d2 + DensityStep(rng, 7, 0.002);
    const Eigen::MatrixXd kick = d3 + DensityStep(rng, 7, 0.5); // the DIIS kick.
    const Eigen::MatrixXd k1 = kick + DensityStep(rng, 7, 0.01);
    const Eigen::MatrixXd k2 = k1 + DensityStep(rng, 7, 0.01);
    const Eigen::MatrixXd k3 = k2 + DensityStep(rng, 7, 0.01);

    // The last density repeats: a zero-delta incremental step with an
    // empty contribution (the accumulator and its bound stay put).
    const std::vector<Eigen::MatrixXd> densities = {base, d1, d2, d3, kick, k1, k2, k3, k3};
    // Rebuild steps: 0 (first call ever) and 4 (the kick clears the gate);
    // the counts pin which path each call took, which the bound bookkeeping
    // depends on.
    const std::vector<int> expectedCounts = {0, 1, 2, 3, 0, 1, 2, 3, 4};
    std::vector<double> bounds;
    bounds.reserve(densities.size());
    Eigen::MatrixXd accumulated;

    for (std::size_t step = 0; step < densities.size(); ++step)
    {
        auto densityTensor = ToTensor(densities[step]);
        ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
        double bound = -1.0;
        auto fock = incremental.BuildFock(*densityTensor, &bound);
        ASSERT_TRUE(fock.has_value()) << fock.error().message;
        EXPECT_EQ(incremental.ConsecutiveIncrementalCountForTesting(), expectedCounts[step]);
        bounds.push_back(bound);
        accumulated = ToMatrix(*fock);
    }

    // Call 0 is a full rebuild of the base density: its bound is exactly
    // the direct mixed builder's single-call bound on the same density.
    double directBaseBound = -1.0;
    auto directBaseTensor = ToTensor(base);
    ASSERT_TRUE(directBaseTensor.has_value()) << directBaseTensor.error().message;
    auto directBaseFock = directBuilder->BuildFock(*directBaseTensor, &directBaseBound);
    ASSERT_TRUE(directBaseFock.has_value()) << directBaseFock.error().message;
    EXPECT_GT(bounds[0], 0.0); // the s-pair quartets must route through the lane
    EXPECT_DOUBLE_EQ(bounds[0], directBaseBound);

    // The incremental steps accumulate: each call adds its own non-negative
    // per-call bound to the cumulative sum.
    EXPECT_GE(bounds[1], bounds[0]);
    EXPECT_GE(bounds[2], bounds[1]);
    EXPECT_GE(bounds[3], bounds[2]);

    // The kick is a full rebuild: the cumulative sum RESETS to that call's
    // own bound (the previous contracts' lane errors no longer contribute
    // to the returned matrix).
    double freshKickBound = -1.0;
    auto freshKickTensor = ToTensor(kick);
    ASSERT_TRUE(freshKickTensor.has_value()) << freshKickTensor.error().message;
    auto freshKickFock = directBuilder->BuildFock(*freshKickTensor, &freshKickBound);
    ASSERT_TRUE(freshKickFock.has_value()) << freshKickFock.error().message;
    EXPECT_DOUBLE_EQ(bounds[4], freshKickBound);

    // Incremental accumulation resumes after the rebuild.
    EXPECT_GE(bounds[5], bounds[4]);
    EXPECT_GE(bounds[6], bounds[5]);
    EXPECT_GE(bounds[7], bounds[6]);

    // Zero delta: the empty contract's density-weighted bound is exactly
    // 0.0, so the cumulative sum stays put.
    EXPECT_DOUBLE_EQ(bounds[8], bounds[7]);

    // The cumulative bound must dominate the ACTUAL element-wise deviation
    // of the accumulated Fock from an fp64-only reference on the final
    // density: every contract's lane error lands inside its own per-call
    // bound, and the bounds add across the accumulation.
    qcx::integrals::FockBuildOptions fp64Only;
    fp64Only.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    fp64Only.useDensityScreening = true;
    fp64Only.useCertifiedMixedPrecision = false;
    auto fp64Builder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, fp64Only);
    ASSERT_TRUE(fp64Builder.has_value()) << fp64Builder.error().message;
    auto fp64Tensor = ToTensor(k3);
    ASSERT_TRUE(fp64Tensor.has_value()) << fp64Tensor.error().message;
    auto fp64Fock = fp64Builder->BuildFock(*fp64Tensor);
    ASSERT_TRUE(fp64Fock.has_value()) << fp64Fock.error().message;

    double maxDeviation = 0.0;

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            maxDeviation = std::max(
                maxDeviation,
                std::abs(accumulated(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) -
                         (*fp64Fock)(i, j)));
        }
    }

    EXPECT_LE(maxDeviation, bounds[8] + 1e-15)
        << "max deviation " << maxDeviation << " exceeds the certified sum " << bounds[8];
}

} // namespace
