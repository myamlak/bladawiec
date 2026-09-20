// The critical correctness gate: the L_mult = 0 QFMM J, assembled from the octree
// near field (the direct builder restricted to the near-field pair-pair
// set, buildCoulombOnly) and the far-field multipole accumulation
// (M2M/M2L/L2L at L_mult = 0 - the degenerate case of the general-L
// machinery), must
//   1. recover the plain direct-sum Coulomb matrix BIT-IDENTICALLY at
//      theta -> 0 (everything near field) - the single most important
//      test of the whole build; a failure here means a bug in the interaction-
//      list completeness or the moment aggregation, and must be found
//      here, not carried forward,
//   2. converge monotonically to the direct result as theta decreases
//      (kernel-level thetas 1.05 -> 0.85 -> 0.7 on the C12 alkane - the
//      acceptance fixture pulled forward; the extent-padded boxes keep the far
//      field alive only at these theta values on a 20-40 atom chain;
//      these are raw kernel inputs, not the production preset ladder,
//      which the production ladder recalibrated to {0.45, 0.3, 0.0} x {5, 5, 0}),
//   3. stay within a loose sanity bound - monopole order is genuinely
//      coarse, and the point of this step is the MACHINERY, not accuracy.
// Plus the moment table itself: a function pair's monopole IS its overlap
// integral, so the table must equal the trusted one-electron overlap
// matrix element for element. The shared build harness lives in
// qfmm_fixture.hpp (the general-L_mult acceptance test reuses it); this
// file pins the L_mult = 0 reduction.

#include "alkane_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_multipole.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qfmm_fixture.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>

namespace {

using qcx::integrals::test::BuildCoreHamiltonian;
using qcx::integrals::test::BuildDirectFock;
using qcx::integrals::test::BuildQfmmFock;
using qcx::integrals::test::PhysicalDensity;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;

TEST(QfmmMonopoleTest, MonopoleTableMatchesTheOverlapMatrix) {
    // A free correctness check: the monopole moment of a function
    // pair IS its overlap integral, so the moment table must equal the
    // already-tested one-electron overlap matrix element for element
    // (BuildOverlapMatrix contracts through the very BuildOverlapPair the
    // table calls - the equality is exact, not approximate).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    const auto geometries = qcx::integrals::internal::ComputePairGeometries(
        *pairStore, qcx::integrals::QfmmExtentForPreset(qcx::integrals::AccuracyPreset::kTight));
    const std::vector<int> zeroOrders(pairStore->size(), 0);
    const auto table =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, zeroOrders);
    ASSERT_TRUE(table.has_value()) << table.error().message;

    EXPECT_EQ(table->offsets.size(), pairList->pairs.size() + 1u);
    EXPECT_EQ(table->counts.size(), pairList->pairs.size());

    for (std::size_t p = 0; p < pairList->pairs.size(); ++p)
    {
        EXPECT_EQ(table->counts[p], 1u);
    }

    for (std::size_t p = 0; p < pairList->pairs.size(); ++p)
    {
        const qcx::integrals::ShellPairIndex& pair = pairList->pairs[p];
        const std::size_t oA = pairList->shells[pair.i].functionOffset;
        const std::size_t oB = pairList->shells[pair.j].functionOffset;
        const std::size_t nA = qcx::integrals::ShellFunctionCount(pairList->shells[pair.i]);
        const std::size_t nB = qcx::integrals::ShellFunctionCount(pairList->shells[pair.j]);

        for (std::size_t fb = 0; fb < nB; ++fb)
        {
            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                // The (0, 0) moment block sits at offset + 0.
                EXPECT_DOUBLE_EQ(table->blocks[table->offsets[p] + fa * nB + fb],
                                 (*overlap)(static_cast<Eigen::Index>(oA + fa),
                                            static_cast<Eigen::Index>(oB + fb)))
                    << "pair " << p << " element (" << fa << "," << fb << ")";
            }
        }
    }
}

TEST(QfmmMonopoleTest, ThetaZeroRecoversTheDirectCoulombMatrix) {
    // THE gate (the acceptance properties' first clause): at theta -> 0 everything is
    // near field, so the restricted near-field build covers every pair-
    // pair (the restriction is a no-op) and the far field contributes
    // nothing - the QFMM J must equal the plain direct-sum Coulomb matrix
    // BIT-IDENTICALLY (exact equality of doubles, not a tolerance: the
    // same screening decisions, the same fp64 contractions, in the same
    // order).
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    const auto direct = BuildDirectFock(*molecule, *basis, *core, density);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    const auto qfmm = BuildQfmmFock(*molecule, *basis, *core, density, 0.0, 0);
    ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

    const Eigen::MatrixXd diff = *direct - *qfmm;
    double maxDiff = 0.0;
    Eigen::Index maxI = 0;
    Eigen::Index maxJ = 0;

    for (Eigen::Index i = 0; i < diff.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < diff.cols(); ++j)
        {
            if (std::abs(diff(i, j)) > maxDiff)
            {
                maxDiff = std::abs(diff(i, j));
                maxI = i;
                maxJ = j;
            }
        }
    }

    EXPECT_TRUE(maxDiff == 0.0) << "largest deviation " << maxDiff << " at (" << maxI << "," << maxJ
                                << ")";
}

TEST(QfmmMonopoleTest, FarFieldErrorDecreasesMonotonicallyWithTighterTheta) {
    // The acceptance property (the monotonicity requirement): on a
    // spatially-extended molecule the QFMM error (relative Frobenius norm
    // vs the direct ground truth) must decrease monotonically as theta
    // decreases. The alkane chain C12H26 (38 atoms, 86 functions, 62
    // shells, 1953 shell pairs) is the acceptance fixture pulled
    // forward. The
    // near-field parts of the QFMM and the direct build make the SAME
    // screening decisions (same options, same density), so the difference
    // is purely the far-field multipole truncation - exactly what the
    // monotonicity measures.
    //
    // The ladder: with the extent-padded boxes (the padding requirement -
    // the box must contain the clouds for the multipole test to
    // be honest), every leaf holds a carbon pair or a C-H pair, so the
    // well-separated threshold is (wA + wB)/theta with w ~ 7 Bohr, and
    // theta <= 0.5 is far-dead on any chain within the 20-40 atom fixture
    // range (0.7 -> 0.5 -> 0.3 was the illustrative ladder, written
    // for raw octant boxes). The sweep thetas {1.05, 0.85, 0.7} keep the
    // far field alive and strictly shrinking at every rung - kernel-level
    // inputs, NOT the production preset ladder (the production rungs were
    // recalibrated to {0.45, 0.3, 0.0} x {5, 5, 0} at the per-preset taus);
    // 0.7 was the earlier tightest preset rung.
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

    // The monotonicity over the whole theta sweep, plus the sanity bounds
    // The sanity bound: monopole order is genuinely coarse and the error must NOT be
    // suspiciously accurate (a machinery bug that silently dropped the far
    // field would look "perfect" and must be caught here) nor gross garbage
    // (a sign or double-count bug). The real accuracy gate is the general-L
    // L_mult acceptance - these bounds only guard the machinery.
    double previousError = std::numeric_limits<double>::infinity();
    double errorLoose = 0.0;

    for (const double theta : {1.05, 0.85, 0.7})
    {
        const auto qfmm = BuildQfmmFock(*molecule, *basis, *core, density, theta, 0);
        ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;
        const double error = (*direct - *qfmm).norm() / directNorm;
        EXPECT_LT(error, previousError) << "theta = " << theta << " error = " << error;
        previousError = error;

        if (theta == 1.05)
        {
            errorLoose = error;
        }
    }

    EXPECT_GT(errorLoose, 1e-3);
    EXPECT_LT(errorLoose, 0.99);
}

} // namespace
