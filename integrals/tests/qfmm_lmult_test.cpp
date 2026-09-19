// The general-L_mult acceptance gate, the second half of the QFMM
// correctness acceptance: on the C12 alkane fixture at every multipole order
// L_mult in {0, ..., kQfmmMaxLMult} the QFMM far-field error must
//   1. stay BIT-IDENTICAL to the direct-sum Coulomb matrix at theta -> 0
//      (everything near field - the multipole machinery contributes
//      exactly nothing, unchanged by the moment order),
//   2. decrease monotonically in L_mult at every fixed theta - going from
//      monopole to dipole to quadrupole MUST reduce the error, and an
//      error increase pins a dipole-moment or M2L bug exactly,
//   3. decrease monotonically in theta at every fixed L_mult (the same
//      property the monopole-order test pins).
// Plus the translation-table sanity gate that makes the L_mult = 0
// reduction bit-exact by construction: the (0, 0) rows of the emitted
// kQfmmM2M / kQfmmM2L tables are exactly {1.0, 0, ..., 0} and
// kQfmmM2LDenomPower[0][0] == 1 (the monopole column is the bare Coulomb
// term, per the banner of qfmm_tables_gen.hpp).

#include "alkane_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "internal/md_tables_gen.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_multipole.hpp"
#include "internal/qfmm_tables_gen.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qfmm_fixture.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>

namespace {

using qcx::integrals::AccuracyPreset;
using qcx::integrals::QfmmExtentForPreset;
using qcx::integrals::internal::CartIndex;
using qcx::integrals::internal::kCartesianCount;
using qcx::integrals::internal::kCartesianIndices;
using qcx::integrals::internal::kMaxShellL;
using qcx::integrals::internal::kQfmmCartesianIndices;
using qcx::integrals::internal::kQfmmM2LCount;
using qcx::integrals::internal::kQfmmM2LDenomPower;
using qcx::integrals::internal::kQfmmM2LMonomial;
using qcx::integrals::internal::kQfmmM2LRunOffset;
using qcx::integrals::internal::kQfmmM2LValues;
using qcx::integrals::internal::kQfmmM2MCount;
using qcx::integrals::internal::kQfmmM2MMonomial;
using qcx::integrals::internal::kQfmmM2MRunOffset;
using qcx::integrals::internal::kQfmmM2MValues;
using qcx::integrals::internal::kQfmmMaxLMult;
using qcx::integrals::internal::kQfmmMomentCount;
using qcx::integrals::internal::kQfmmMonomialExponents;
using qcx::integrals::internal::kQfmmSolidHarmonicG;
using qcx::integrals::internal::kSolidHarmonicG;
using qcx::integrals::internal::QfmmM2LCoefficientStats;
using qcx::integrals::test::BuildCoreHamiltonian;
using qcx::integrals::test::BuildDirectFock;
using qcx::integrals::test::BuildQfmmFock;
using qcx::integrals::test::PhysicalDensity;
using qcx::integrals::test::ToMatrix;
using qcx::integrals::test::ToTensor;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;

TEST(QfmmLMultTest, TranslationTablesReduceToCoulombAtMonopoleOrder) {
    // The bit-exact L_mult = 0 reduction's foundation (pinned against the
    // emitted tables, never hand-derived): the (0, 0) runs of the
    // translation tables must be exactly the unit/coulomb runs - the M2M
    // and M2L coefficients at (t, s) = (0, 0) are exactly 1.0 for any
    // displacement (1.0 · 1.0 + the absent higher monomials), and the M2L
    // denominator power 1 divides by |d|^1 == the distance itself,
    // reproducing the bare Coulomb expression exactly. In the sparse
    // layout the (0, 0) row is the run [offset[0], offset[1]) of the flat
    // arrays: a single entry {(monomial 0, 1.0)} - the run length 1 IS the
    // "all other monomials are exactly 0" of the dense form.
    EXPECT_EQ(kQfmmMomentCount,
              static_cast<std::size_t>((kQfmmMaxLMult + 1) * (kQfmmMaxLMult + 1)));
    EXPECT_EQ(kQfmmMaxLMult, 8);
    EXPECT_EQ(kQfmmM2MRunOffset[0], 0u);
    EXPECT_EQ(kQfmmM2MRunOffset[1], 1u);
    EXPECT_EQ(kQfmmM2MValues[0], 1.0);
    EXPECT_EQ(kQfmmM2MMonomial[0], 0u);
    EXPECT_EQ(kQfmmM2LRunOffset[0], 0u);
    EXPECT_EQ(kQfmmM2LRunOffset[1], 1u);
    EXPECT_EQ(kQfmmM2LValues[0], 1.0);
    EXPECT_EQ(kQfmmM2LMonomial[0], 0u);
    EXPECT_EQ(kQfmmM2LDenomPower[0][0], 1);

    // The run offsets' last entry is the total nonzero count: the runs tile
    // the flat arrays exactly.
    EXPECT_EQ(kQfmmM2MRunOffset[kQfmmMomentCount * kQfmmMomentCount], kQfmmM2MCount);
    EXPECT_EQ(kQfmmM2LRunOffset[kQfmmMomentCount * kQfmmMomentCount], kQfmmM2LCount);
}

TEST(QfmmLMultTest, MomentSolidHarmonicRowsMatchTheShellTablesUpToShellL) {
    // The moment path's (l, m) contraction (ContractToSolidHarmonics in
    // qfmm_multipole.cpp) reads the QFMM header's own kQfmmCartesianIndices
    // / kQfmmSolidHarmonicG rows, which extend to l = kQfmmMaxLMult - the
    // shell tables kCartesianIndices / kSolidHarmonicG stop at
    // kMaxShellL = 6 (md_tables_gen.hpp), short of the lMult = 7 cap. The
    // l <= 6 rows of the two sources must be bit-identical: they come from
    // the same pinned gen_md_tables.solid_harmonic_table source, and any
    // divergence would silently change the trusted lMult <= 6 results (the
    // l > 6 rows exist only in the QFMM emission).
    for (int l = 0; l <= kMaxShellL; ++l)
    {
        const std::size_t cartCount = kCartesianCount[static_cast<std::size_t>(l)];

        for (std::size_t cart = 0; cart < cartCount; ++cart)
        {
            const auto& qfmmCart = kQfmmCartesianIndices[static_cast<std::size_t>(l)][cart];
            const CartIndex shellCart = kCartesianIndices[static_cast<std::size_t>(l)][cart];
            EXPECT_EQ(qfmmCart[0], shellCart.ix) << "l = " << l << " cart " << cart;
            EXPECT_EQ(qfmmCart[1], shellCart.iy) << "l = " << l << " cart " << cart;
            EXPECT_EQ(qfmmCart[2], shellCart.iz) << "l = " << l << " cart " << cart;

            for (int m = -l; m <= l; ++m)
            {
                EXPECT_EQ(kQfmmSolidHarmonicG[static_cast<std::size_t>(l)]
                                             [static_cast<std::size_t>(m + l)][cart],
                          kSolidHarmonicG[static_cast<std::size_t>(l)]
                                         [static_cast<std::size_t>(m + l)][cart])
                    << "l = " << l << " m = " << m << " cart " << cart;
            }
        }
    }
}

TEST(QfmmLMultTest, ThetaZeroRecoversTheDirectCoulombMatrixAtEveryLMult) {
    // The theta -> 0 exact recovery at every L_mult. The far field
    // is empty, so the multipole machinery contributes exactly nothing
    // (0.0 == -0.0 in the IEEE comparisons, so the ±0.0 far-field terms
    // cannot disturb the equality); the near-field build is the same
    // restricted direct build at every moment order.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7);
    const auto direct = BuildDirectFock(*molecule, *basis, *core, density);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;

    for (int lMult = 0; lMult <= kQfmmMaxLMult; ++lMult)
    {
        const auto qfmm = BuildQfmmFock(*molecule, *basis, *core, density, 0.0, lMult);
        ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

        double maxDiff = 0.0;

        for (Eigen::Index i = 0; i < direct->rows(); ++i)
        {
            for (Eigen::Index j = 0; j < direct->cols(); ++j)
            {
                maxDiff = std::max(maxDiff, std::abs((*direct)(i, j) - (*qfmm)(i, j)));
            }
        }

        EXPECT_TRUE(maxDiff == 0.0) << "lMult = " << lMult << " largest deviation " << maxDiff;
    }
}

TEST(QfmmLMultTest, FarFieldErrorDecreasesWithLMultAndTighterTheta) {
    // The acceptance's second half. The C12 alkane fixture, the theta ladder
    // {1.05, 0.85, 0.7} (loosest first) and the L_mult ladder {0, 1, 2}. The
    // near-field parts of the QFMM and the direct build are bitwise equal
    // (same options, same density, serial pin), so the measured error is
    // purely the far-field multipole truncation - and it must shrink strictly
    // in BOTH directions:
    //   - at every fixed theta: raising the moment order from monopole to
    //     dipole to quadrupole must reduce the error (a dipole or M2L
    //     defect shows up as an error INCREASE here),
    //   - at every fixed L_mult: tightening theta must reduce the error
    //     (the same property the monopole-order test pins).
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
    // first - larger error), the L_mult columns {0, ..., kQfmmMaxLMult}.
    const std::array<double, 3> thetas = {1.05, 0.85, 0.7};
    std::array<std::array<double, kQfmmMaxLMult + 1>, 3> errors{};

    for (std::size_t thetaIndex = 0; thetaIndex < thetas.size(); ++thetaIndex)
    {
        for (int lMult = 0; lMult <= kQfmmMaxLMult; ++lMult)
        {
            const auto qfmm =
                BuildQfmmFock(*molecule, *basis, *core, density, thetas[thetaIndex], lMult);
            ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;
            errors[thetaIndex][static_cast<std::size_t>(lMult)] =
                (*direct - *qfmm).norm() / directNorm;
        }
    }

    // At every fixed theta: strictly decreasing in L_mult (0 > 1 > ...).
    for (std::size_t thetaIndex = 0; thetaIndex < thetas.size(); ++thetaIndex)
    {
        for (int lMult = 0; lMult < kQfmmMaxLMult; ++lMult)
        {
            EXPECT_LT(errors[thetaIndex][static_cast<std::size_t>(lMult + 1)],
                      errors[thetaIndex][static_cast<std::size_t>(lMult)])
                << "theta = " << thetas[thetaIndex] << " lMult " << lMult << " -> " << lMult + 1;
        }
    }

    // At every fixed L_mult: strictly decreasing in theta (tighter theta,
    // smaller error).
    for (int lMult = 0; lMult <= kQfmmMaxLMult; ++lMult)
    {
        for (std::size_t thetaIndex = 0; thetaIndex + 1 < thetas.size(); ++thetaIndex)
        {
            EXPECT_LT(errors[thetaIndex + 1][static_cast<std::size_t>(lMult)],
                      errors[thetaIndex][static_cast<std::size_t>(lMult)])
                << "lMult = " << lMult << " theta " << thetas[thetaIndex] << " -> "
                << thetas[thetaIndex + 1];
        }
    }
}

TEST(QfmmLMultTest, MonopoleBlockAtLMultOneMatchesTheOverlap) {
    // The self-consistency check: at lMult = 1 the
    // raised path's (0, 0) moment block (a = b = c = 0: M_000 = O_000) is
    // the overlap through the same t = 0 contraction - it must match the
    // lMult = 0 table (the trusted overlap engine) to roundoff.
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;

    const auto geometries = qcx::integrals::internal::ComputePairGeometries(
        *pairStore, QfmmExtentForPreset(AccuracyPreset::kTight));
    const std::vector<int> zeroOrders(pairStore->size(), 0);
    const std::vector<int> oneOrders(pairStore->size(), 1);
    const auto monopole =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, zeroOrders);
    ASSERT_TRUE(monopole.has_value()) << monopole.error().message;
    const auto lMultOne =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, oneOrders);
    ASSERT_TRUE(lMultOne.has_value()) << lMultOne.error().message;

    // The on-demand layout: one count per pair, and the offsets advance
    // by counts[p] * nFuncs(p) element units (at the uniform order 1 that is
    // (1 + 1)^2 = 4 blocks per pair - the layout the whole machinery is
    // indexed with).
    ASSERT_EQ(lMultOne->counts.size(), pairList->pairs.size());

    for (std::size_t p = 0; p < pairList->pairs.size(); ++p)
    {
        EXPECT_EQ(lMultOne->counts[p], 4U) << "pair " << p;
        EXPECT_EQ(lMultOne->offsets[p + 1] - lMultOne->offsets[p],
                  lMultOne->counts[p] * pairStore->at(p).nFuncs)
            << "pair " << p;
    }

    ASSERT_EQ(monopole->counts.size(), pairList->pairs.size());

    for (std::size_t p = 0; p < pairList->pairs.size(); ++p)
    {
        EXPECT_EQ(monopole->counts[p], 1U) << "pair " << p;
    }

    for (std::size_t p = 0; p < pairList->pairs.size(); ++p)
    {
        const qcx::integrals::ShellPairIndex& pair = pairList->pairs[p];
        const std::size_t nA = qcx::integrals::ShellFunctionCount(pairList->shells[pair.i]);
        const std::size_t nB = qcx::integrals::ShellFunctionCount(pairList->shells[pair.j]);

        for (std::size_t fb = 0; fb < nB; ++fb)
        {
            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                EXPECT_NEAR(lMultOne->blocks[lMultOne->offsets[p] + fa * nB + fb],
                            monopole->blocks[monopole->offsets[p] + fa * nB + fb],
                            1e-12)
                    << "pair " << p << " element (" << fa << "," << fb << ")";
            }
        }
    }
}

// The synthetic two-center i-shell basis of the md pair-data benchmarks (a
// three-primitive I shell, so each (6, 6) pair runs nine primitive-table
// builds): the la = 6 bra raised by lMult = 3 exceeds the shared
// PerAxisETable bound (la <= kMaxShellL + 2) and takes the oversized-table
// FillRaisedPerAxisTable path - otherwise dead, no fixture basis has l = 6.
inline constexpr std::size_t kChainAtomCount = 6;
inline constexpr double kChainSpacing = 2.0;

inline constexpr std::string_view kIChainBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    I
      1.0000000000E+00       5.0000000000E-01
      5.0000000000E-01       3.0000000000E-01
      2.0000000000E-01       2.0000000000E-01
END
)";

qcx::Result<qcx::molecule::Molecule> MakeIChainMolecule() {
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({kChainAtomCount, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t i = 0; i < kChainAtomCount; ++i)
    {
        (*coordinates)(i, 0) = static_cast<double>(i) * kChainSpacing;
        (*coordinates)(i, 1) = 0.0;
        (*coordinates)(i, 2) = 0.0;
        atoms.push_back({"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

TEST(QfmmLMultTest, RaisedTablePathAtLMultThreeMatchesTheOverlap) {
    // The FillRaisedPerAxisTable path (qfmm_multipole.cpp): at lMult = 3
    // the (0, 0) moment block (M_000 = O_000) of every pair must equal the
    // lMult = 0 table (the trusted overlap engine) to roundoff - the
    // oversized per-axis tables are filled by the mirrored recurrence and
    // the rest of the moment pipeline is unchanged.
    if (!qcx::integrals::SupportsL(6))
    {
        GTEST_SKIP() << "no l = 6 classes in this build";
    }

    auto basis = qcx::basisset::ParseNwchemText(kIChainBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeIChainMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;

    const auto geometries = qcx::integrals::internal::ComputePairGeometries(
        *pairStore, QfmmExtentForPreset(AccuracyPreset::kTight));
    const std::vector<int> zeroOrders(pairStore->size(), 0);
    const std::vector<int> threeOrders(pairStore->size(), 3);
    const auto monopole =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, zeroOrders);
    ASSERT_TRUE(monopole.has_value()) << monopole.error().message;
    const auto lMultThree =
        qcx::integrals::internal::BuildMomentTable(*pairList, *pairStore, geometries, threeOrders);
    ASSERT_TRUE(lMultThree.has_value()) << lMultThree.error().message;

    // The ladder must actually cover the la = 6 -> laPrime = 9 overflow (a
    // future kMaxShellL + 2 change would silently take the shared path and
    // this test would stop guarding anything).
    bool sawSix = false;

    for (const qcx::integrals::internal::MdPairData& pair : *pairStore)
    {
        sawSix = sawSix || pair.la == 6;
    }

    ASSERT_TRUE(sawSix) << "the i-chain store has no la = 6 pair";

    for (std::size_t p = 0; p < pairList->pairs.size(); ++p)
    {
        const qcx::integrals::ShellPairIndex& pair = pairList->pairs[p];
        const std::size_t nA = qcx::integrals::ShellFunctionCount(pairList->shells[pair.i]);
        const std::size_t nB = qcx::integrals::ShellFunctionCount(pairList->shells[pair.j]);

        for (std::size_t fb = 0; fb < nB; ++fb)
        {
            for (std::size_t fa = 0; fa < nA; ++fa)
            {
                EXPECT_NEAR(lMultThree->blocks[lMultThree->offsets[p] + fa * nB + fb],
                            monopole->blocks[monopole->offsets[p] + fa * nB + fb],
                            1e-12)
                    << "pair " << p << " element (" << fa << "," << fb << ")";
            }
        }
    }
}

// The order-gate fixtures: an alkane chain's octree at a preset's extent
// rung, its interaction lists at a chosen theta, and the per-pair source
// ratio (sqrt(3) * max(halfWidth) / center distance) - the real-tree
// geometry the order selection and the on-demand table run on (the
// real form keeps the gates on the production path rather than on
// hand-built synthetic node pairs).
struct OrderGateGeometry {
    qcx::integrals::ShellPairList pairList;
    std::vector<qcx::integrals::internal::MdPairData> pairStore;
    std::vector<qcx::integrals::internal::QfmmPairGeometry> geometries;
    qcx::integrals::internal::QfmmTreeBuildResult tree;
    std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
};

qcx::Result<OrderGateGeometry> BuildOrderGateGeometry(std::size_t nC,
                                                      AccuracyPreset accuracy,
                                                      double theta) {
    OrderGateGeometry result;
    auto molecule = MakeAlkaneSto3g(nC);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto basis = MakeAlkaneSto3gBasis();

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(pairStore.error());
    }

    result.pairList = std::move(*pairList);
    result.pairStore = std::move(*pairStore);
    result.geometries = qcx::integrals::internal::ComputePairGeometries(
        result.pairStore, QfmmExtentForPreset(accuracy));
    auto tree = qcx::integrals::internal::BuildQfmmTree(result.geometries);

    if (!tree.has_value())
    {
        return std::unexpected(tree.error());
    }

    result.tree = std::move(*tree);
    std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
    qcx::integrals::internal::BuildInteractionLists(
        result.tree, theta, result.farFieldPairs, nearFieldLeafPairs);
    return result;
}

double FarPairRatio(const qcx::integrals::internal::QfmmTreeBuildResult& tree,
                    const std::pair<std::size_t, std::size_t>& pair) {
    const qcx::integrals::internal::QfmmTreeNode& nodeA = tree.nodes[pair.first];
    const qcx::integrals::internal::QfmmTreeNode& nodeB = tree.nodes[pair.second];
    const double dx = nodeA.centerX - nodeB.centerX;
    const double dy = nodeA.centerY - nodeB.centerY;
    const double dz = nodeA.centerZ - nodeB.centerZ;
    return std::sqrt(3.0) * std::max(nodeA.halfWidth, nodeB.halfWidth) /
           std::sqrt(dx * dx + dy * dy + dz * dz);
}

TEST(QfmmLMultTest, SelectMultipoleOrdersPinsTheMinimalOrderAndTheCap) {
    // The SelectMultipoleOrders contract, on a
    // real C24 tree at theta 0.21 (the far-pair onset band, where the
    // ratios ~ 0.18 make the per-interaction budget large enough for the
    // selector to genuinely lower orders below the cap): each returned
    // order is the MINIMAL L whose bound (r/d)^(L+1) fits epsInt (asserted
    // by the two-sided inequality, order+1 qualifies and order does not -
    // or the pair keeps the cap), no order exceeds the cap, the tiny-epsInt
    // limit saturates at the cap for every pair, and a generous epsInt
    // reaches L = 0 (bound of an order-0 pair is ratio <= 1, satisfied by
    // every well-separated pair).
    const auto geometry = BuildOrderGateGeometry(24, AccuracyPreset::kLoose, 0.21);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;

    if (geometry->farFieldPairs.empty())
    {
        GTEST_SKIP() << "no far pairs at theta 0.21 on this tree shape";
    }

    const int cap = qcx::integrals::LMultForPreset(AccuracyPreset::kLoose);
    ASSERT_EQ(cap, 5);

    for (const auto& pair : geometry->farFieldPairs)
    {
        EXPECT_GT(FarPairRatio(geometry->tree, pair), 0.0);
    }

    const std::vector<int> dropped = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, 2.2e-4, cap);
    ASSERT_EQ(dropped.size(), geometry->farFieldPairs.size());
    bool sawDropBelowCap = false;

    for (std::size_t i = 0; i < dropped.size(); ++i)
    {
        const double ratio = FarPairRatio(geometry->tree, geometry->farFieldPairs[i]);
        EXPECT_GE(dropped[i], 0) << "pair " << i;
        EXPECT_LE(dropped[i], cap) << "pair " << i;
        EXPECT_LE(std::pow(ratio, dropped[i] + 1), 2.2e-4)
            << "pair " << i << " order " << dropped[i] << " violates the bound";

        if (dropped[i] < cap)
        {
            // The minimality: the order below the chosen one does NOT fit.
            EXPECT_GT(std::pow(ratio, dropped[i]), 2.2e-4)
                << "pair " << i << " order " << dropped[i] << " is not minimal";
            sawDropBelowCap = true;
        }
    }

    EXPECT_TRUE(sawDropBelowCap)
        << "the 2.2e-4 budget should lower some pairs below the cap on this tree";

    // The cap fallback: a budget no order can fit leaves every pair at the
    // full resolved order (the preset configs saturate exactly this way).
    const std::vector<int> saturated = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, 1e-12, cap);

    for (const int order : saturated)
    {
        EXPECT_EQ(order, cap);
    }

    // L = 0 reachable: a budget above every well-separated pair's ratio.
    const std::vector<int> monopole = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, 1.0, cap);

    for (std::size_t i = 0; i < monopole.size(); ++i)
    {
        EXPECT_LE(FarPairRatio(geometry->tree, geometry->farFieldPairs[i]), 1.0)
            << "pair " << i << " is not well-separated (ratio above 1)";
        EXPECT_EQ(monopole[i], 0) << "pair " << i;
    }
}

TEST(QfmmLMultTest, SelectMultipoleOrdersIsMonotoneAndDeterministic) {
    // Same geometry, same pair list -> same selection (the determinism
    // clause of the order gate), and a looser per-interaction budget never
    // raises an order (larger epsInt -> smaller-or-equal minimal L).
    const auto geometry = BuildOrderGateGeometry(24, AccuracyPreset::kLoose, 0.21);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;

    if (geometry->farFieldPairs.empty())
    {
        GTEST_SKIP() << "no far pairs at theta 0.21 on this tree shape";
    }

    const int cap = qcx::integrals::LMultForPreset(AccuracyPreset::kLoose);
    const std::array<double, 3> epsInts = {1e-6, 2.2e-4, 1e-1};
    std::array<std::vector<int>, 3> orders{};

    for (std::size_t row = 0; row < epsInts.size(); ++row)
    {
        orders[row] = qcx::integrals::internal::SelectMultipoleOrders(
            geometry->tree, geometry->farFieldPairs, epsInts[row], cap);
    }

    // Determinism: two independent selections at the same budget agree
    // exactly.
    const std::vector<int> again = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, 2.2e-4, cap);
    EXPECT_EQ(again, orders[1]);

    // Monotonicity: elementwise non-increasing as the budget loosens.
    for (std::size_t i = 0; i < orders[0].size(); ++i)
    {
        EXPECT_GE(orders[0][i], orders[1][i]) << "pair " << i;
        EXPECT_GE(orders[1][i], orders[2][i]) << "pair " << i;
    }
}

TEST(QfmmLMultTest, ComputeNeededNodeOrdersMarksNearOnlyNodesAndKeepsBands) {
    // The per-node bands: -1 exactly for the nodes that neither are far
    // pairs endpoints nor descend from one (an empty far list leaves every
    // node -1), the nodes a pair touches carry at least that pair's order,
    // and the bands never fall between a node and its children (children
    // >= parents), so a -1 node's whole ancestor chain is -1 - it is never
    // an endpoint, nothing ever writes or reads content at it - while its
    // descendants may be active from their own far pairs: the -1 nodes
    // that bound an active subtree from above are exactly the inert
    // near-only interior nodes.
    const auto geometry = BuildOrderGateGeometry(24, AccuracyPreset::kLoose, 0.21);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;

    const std::vector<int> allNear =
        qcx::integrals::internal::ComputeNeededNodeOrders(geometry->tree, {}, {});

    for (const int order : allNear)
    {
        EXPECT_EQ(order, -1);
    }

    if (geometry->farFieldPairs.empty())
    {
        return;
    }

    const int cap = qcx::integrals::LMultForPreset(AccuracyPreset::kLoose);
    const std::vector<int> pairOrders = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, 2.2e-4, cap);
    const std::vector<int> nodeOrders = qcx::integrals::internal::ComputeNeededNodeOrders(
        geometry->tree, geometry->farFieldPairs, pairOrders);
    ASSERT_EQ(nodeOrders.size(), geometry->tree.nodes.size());

    for (std::size_t i = 0; i < pairOrders.size(); ++i)
    {
        EXPECT_GE(nodeOrders[geometry->farFieldPairs[i].first], pairOrders[i]) << "pair " << i;
        EXPECT_GE(nodeOrders[geometry->farFieldPairs[i].second], pairOrders[i]) << "pair " << i;
    }

    bool sawNearOnly = false;
    bool sawActive = false;
    bool sawNearOnlyAboveActive = false;

    for (std::size_t n = 0; n < geometry->tree.nodes.size(); ++n)
    {
        sawActive = sawActive || nodeOrders[n] >= 0;
        sawNearOnly = sawNearOnly || nodeOrders[n] < 0;

        for (const std::size_t child : geometry->tree.children[n])
        {
            // The band never falls between a node and its children (the
            // pre-order downward propagation), so an active node's whole
            // subtree is active and a -1 node's whole ancestry is -1.
            EXPECT_GE(nodeOrders[child], nodeOrders[n])
                << "node " << child << " falls below its parent " << n;
            sawNearOnlyAboveActive =
                sawNearOnlyAboveActive || (nodeOrders[n] < 0 && nodeOrders[child] >= 0);
        }
    }

    EXPECT_TRUE(sawActive);
    EXPECT_TRUE(sawNearOnly);
    EXPECT_TRUE(sawNearOnlyAboveActive)
        << "expected a -1 node with an active child (the inert near-only "
           "boundary above an active subtree)";
}

TEST(QfmmLMultTest, MomentTableAllocatesZeroBlocksForNearOnlyPairs) {
    // The memory gate: on the C12
    // chain at the kLoose preset (theta 0.45, a live far field of 14 pairs
    // saturating every order at the cap 5 - the preset config the probe
    // mapped), a pair in a near-field-only leaf holds ZERO moment blocks -
    // its moments never feed any interaction - while an active pair holds
    // (order + 1)^2 blocks; the strongest form, an empty far field, leaves
    // the table with no blocks at all.
    const auto geometry = BuildOrderGateGeometry(12, AccuracyPreset::kLoose, 0.45);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    ASSERT_FALSE(geometry->farFieldPairs.empty())
        << "the C12 kLoose preset far field must be live for this gate";
    const int cap = qcx::integrals::LMultForPreset(AccuracyPreset::kLoose);

    const double epsInt = qcx::integrals::QfmmBudgetForPreset(AccuracyPreset::kLoose) /
                          static_cast<double>(geometry->farFieldPairs.size());
    const std::vector<int> farPairOrders = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, epsInt, cap);

    for (const int order : farPairOrders)
    {
        EXPECT_EQ(order, cap) << "the preset budget must saturate on this tree";
    }

    const std::vector<int> nodeOrders = qcx::integrals::internal::ComputeNeededNodeOrders(
        geometry->tree, geometry->farFieldPairs, farPairOrders);
    std::vector<int> pairOrders(geometry->pairStore.size(), -1);
    std::size_t nearOnlyPairs = 0;
    std::size_t activePairs = 0;

    for (std::size_t p = 0; p < geometry->pairStore.size(); ++p)
    {
        // A canonical pair lives in exactly one leaf (its expansion center
        // sits in one box), whose band is the pair's.
        pairOrders[p] = nodeOrders[geometry->tree.leafOfPair[p]];

        if (pairOrders[p] < 0)
        {
            ++nearOnlyPairs;
        } else
        {
            ++activePairs;
        }
    }

    ASSERT_GT(nearOnlyPairs, 0u) << "this gate needs near-field-only leaves";
    ASSERT_GT(activePairs, 0u) << "this gate needs a live far field";

    const auto table = qcx::integrals::internal::BuildMomentTable(
        geometry->pairList, geometry->pairStore, geometry->geometries, pairOrders);
    ASSERT_TRUE(table.has_value()) << table.error().message;
    ASSERT_EQ(table->offsets.size(), geometry->pairStore.size() + 1u);
    ASSERT_EQ(table->counts.size(), geometry->pairStore.size());
    EXPECT_EQ(table->offsets.back(), table->blocks.size());

    std::size_t expectedElements = 0;

    for (std::size_t p = 0; p < geometry->pairStore.size(); ++p)
    {
        const std::size_t count = table->counts[p];
        const std::size_t expected = pairOrders[p] < 0
                                         ? 0u
                                         : static_cast<std::size_t>(pairOrders[p] + 1) *
                                               static_cast<std::size_t>(pairOrders[p] + 1);
        EXPECT_EQ(count, expected) << "pair " << p;
        EXPECT_EQ(table->offsets[p + 1] - table->offsets[p], count * geometry->pairStore[p].nFuncs)
            << "pair " << p;
        expectedElements += count * geometry->pairStore[p].nFuncs;
    }

    EXPECT_EQ(table->blocks.size(), expectedElements);

    // The empty-far-field degenerate case: every pair -1, zero blocks
    // anywhere (the moment table allocates nothing when no interaction
    // needs it).
    const std::vector<int> allNear(geometry->pairStore.size(), -1);
    const auto emptyTable = qcx::integrals::internal::BuildMomentTable(
        geometry->pairList, geometry->pairStore, geometry->geometries, allNear);
    ASSERT_TRUE(emptyTable.has_value()) << emptyTable.error().message;
    EXPECT_TRUE(emptyTable->blocks.empty());

    for (const std::size_t count : emptyTable->counts)
    {
        EXPECT_EQ(count, 0u);
    }
}

TEST(QfmmLMultTest, AdaptiveMomentTableContentMatchesTheFullTable) {
    // The content gate: "bit-identity
    // vs the full-table build (same lMult, same geometry)". At the
    // saturating preset config every active pair keeps the full resolved
    // order, so the adaptive table's blocks (the same BuildPairMoments at
    // the same order, only the (order+1)^2 stride and the zero-count
    // near-only pairs differ) must equal the uniform full table's content
    // element for element - the (0, 0) block is the overlap at every
    // nonzero count, exactly as in the full build.
    const auto geometry = BuildOrderGateGeometry(12, AccuracyPreset::kLoose, 0.45);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    ASSERT_FALSE(geometry->farFieldPairs.empty());
    const int cap = qcx::integrals::LMultForPreset(AccuracyPreset::kLoose);
    const double epsInt = qcx::integrals::QfmmBudgetForPreset(AccuracyPreset::kLoose) /
                          static_cast<double>(geometry->farFieldPairs.size());
    const std::vector<int> farPairOrders = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, epsInt, cap);
    const std::vector<int> nodeOrders = qcx::integrals::internal::ComputeNeededNodeOrders(
        geometry->tree, geometry->farFieldPairs, farPairOrders);
    std::vector<int> pairOrders(geometry->pairStore.size(), -1);

    for (std::size_t p = 0; p < geometry->pairStore.size(); ++p)
    {
        pairOrders[p] = nodeOrders[geometry->tree.leafOfPair[p]];
    }

    const std::vector<int> uniformOrders(geometry->pairStore.size(), cap);
    const auto adaptive = qcx::integrals::internal::BuildMomentTable(
        geometry->pairList, geometry->pairStore, geometry->geometries, pairOrders);
    ASSERT_TRUE(adaptive.has_value()) << adaptive.error().message;
    const auto full = qcx::integrals::internal::BuildMomentTable(
        geometry->pairList, geometry->pairStore, geometry->geometries, uniformOrders);
    ASSERT_TRUE(full.has_value()) << full.error().message;
    ASSERT_EQ(adaptive->counts.size(), full->counts.size());

    for (std::size_t p = 0; p < adaptive->counts.size(); ++p)
    {
        if (adaptive->counts[p] == 0)
        {
            // The near-only pair: the zero allocation - the full table's
            // content there exists but is never read by any interaction.
            EXPECT_GT(full->counts[p], 0u) << "pair " << p;
            continue;
        }

        EXPECT_EQ(full->counts[p], adaptive->counts[p]) << "pair " << p;
        const std::size_t nFuncs = geometry->pairStore[p].nFuncs;
        double maxDiff = 0.0;

        for (std::size_t b = 0; b < adaptive->counts[p]; ++b)
        {
            for (std::size_t e = 0; e < nFuncs; ++e)
            {
                maxDiff =
                    std::max(maxDiff,
                             std::abs(adaptive->blocks[adaptive->offsets[p] + b * nFuncs + e] -
                                      full->blocks[full->offsets[p] + b * nFuncs + e]));
            }
        }

        EXPECT_TRUE(maxDiff == 0.0) << "pair " << p << " largest deviation " << maxDiff;
    }
}

TEST(QfmmLMultTest, AdaptiveBuilderMatchesFixedWhenTheOrdersSaturate) {
    // The adaptive-order gate's first clause: "at
    // any fixed geometry, adaptive vs fixed-L-at-the-same-L are bit-
    // identical when the bound says 'full order everywhere'". The C12 chain
    // at the kLoose preset saturates (every far pair needs more than the
    // cap 5, so the selector keeps the full order everywhere - asserted
    // through FarFieldPairCount() > 0 plus the table shape of the gate
    // above), so the public builder's adaptive run must be bit-identical to
    // its useAdaptiveOrder = false run; two adaptive instances also pin the
    // cross-instance determinism clause. Both runs fp64-only and serial
    // (the determinism pin of the bit-exact gates).
    auto molecule = MakeAlkaneSto3g(12);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = MakeAlkaneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const std::size_t n = core->Shape()[0];
    const Eigen::MatrixXd density = PhysicalDensity(n);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::QfmmOptions adaptiveOptions;
    adaptiveOptions.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    adaptiveOptions.useCertifiedMixedPrecision = false;
    adaptiveOptions.maxParallelChunks = 1;
    qcx::integrals::QfmmOptions fixedOptions = adaptiveOptions;
    fixedOptions.useAdaptiveOrder = false;

    auto adaptiveOne =
        qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, adaptiveOptions);
    ASSERT_TRUE(adaptiveOne.has_value()) << adaptiveOne.error().message;
    ASSERT_GT(adaptiveOne->FarFieldPairCount(), 0u)
        << "the adaptive gate needs a live far field (vacuous-ladder trap)";
    auto adaptiveTwo =
        qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, adaptiveOptions);
    ASSERT_TRUE(adaptiveTwo.has_value()) << adaptiveTwo.error().message;
    auto fixed = qcx::integrals::QfmmJBuilder::Create(*molecule, *basis, *core, fixedOptions);
    ASSERT_TRUE(fixed.has_value()) << fixed.error().message;
    ASSERT_EQ(fixed->FarFieldPairCount(), adaptiveOne->FarFieldPairCount());

    const auto fockAdaptiveOne = adaptiveOne->BuildFock(*densityTensor);
    ASSERT_TRUE(fockAdaptiveOne.has_value()) << fockAdaptiveOne.error().message;
    const auto fockAdaptiveTwo = adaptiveTwo->BuildFock(*densityTensor);
    ASSERT_TRUE(fockAdaptiveTwo.has_value()) << fockAdaptiveTwo.error().message;
    const auto fockFixed = fixed->BuildFock(*densityTensor);
    ASSERT_TRUE(fockFixed.has_value()) << fockFixed.error().message;

    const Eigen::MatrixXd adaptiveMatrix = ToMatrix(*fockAdaptiveOne);
    const Eigen::MatrixXd adaptiveMatrixTwo = ToMatrix(*fockAdaptiveTwo);
    const Eigen::MatrixXd fixedMatrix = ToMatrix(*fockFixed);

    double adaptiveFixedDiff = 0.0;
    double instanceDiff = 0.0;

    for (Eigen::Index i = 0; i < adaptiveMatrix.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < adaptiveMatrix.cols(); ++j)
        {
            adaptiveFixedDiff =
                std::max(adaptiveFixedDiff, std::abs(adaptiveMatrix(i, j) - fixedMatrix(i, j)));
            instanceDiff =
                std::max(instanceDiff, std::abs(adaptiveMatrix(i, j) - adaptiveMatrixTwo(i, j)));
        }
    }

    EXPECT_TRUE(adaptiveFixedDiff == 0.0)
        << "adaptive vs fixed at the saturating preset, largest deviation " << adaptiveFixedDiff;
    EXPECT_TRUE(instanceDiff == 0.0)
        << "adaptive across instances, largest deviation " << instanceDiff;
}

// ---------------------------------------------------------------------------
// The M2L coefficient fold: one table evaluation per (t, s) row, the pair's
// reversed direction served from the row parity (-1)^(l_t + l_s).
// ---------------------------------------------------------------------------
//
// ApplyM2LTo used to evaluate EvalM2L(t, s, d) AND EvalM2L(t, s, -d) for every
// far pair. The emitted row (t, s) is a homogeneous polynomial of degree
// l_t + l_s (tools/gen_qfmm_translation_tables.py; the header banner of
// qfmm_tables_gen.hpp), so the reversed direction is the forward value times
// an exact sign and the second evaluation is redundant - positionally, not
// only up to a tolerance. The two cells below are the fold's two obligations:
// the table property it is licensed by, and the pass REACHING it (a folded
// pass reports equal row counters; a pass that evaluates the reversed
// direction separately reports twice the evaluations and no folds).

/// The l of a moment index (idx = l^2 + l + m - the emitted packing).
int LMultTestLOfIndex(int idx) {
    int l = 0;

    while ((l + 1) * (l + 1) <= idx)
    {
        ++l;
    }

    return l;
}

/// The pre-fold expression: the shipped M2L row value at (t, s) for the
/// displacement (dx, dy, dz), evaluated from the sparse run with the same
/// per-axis power-by-repeated-multiplication the pass uses. This is the
/// bit-identity reference of the fold - the pass must produce exactly this
/// value for the forward direction and its exact sign image for the reversed
/// one.
// (dx, dy, dz) is the x-then-y-then-z order of the displacement.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ReferenceM2LRowValue(int t, int s, double dx, double dy, double dz) {
    const auto power = [](double x, int k) {
        double value = 1.0;

        for (int i = 0; i < k; ++i)
        {
            value *= x;
        }

        return value;
    };

    double value = 0.0;
    const std::size_t row =
        static_cast<std::size_t>(t) * kQfmmMomentCount + static_cast<std::size_t>(s);

    for (std::uint32_t k = kQfmmM2LRunOffset[row]; k < kQfmmM2LRunOffset[row + 1]; ++k)
    {
        const std::array<int, 3>& exponents =
            kQfmmMonomialExponents[static_cast<std::size_t>(kQfmmM2LMonomial[k])];
        value += kQfmmM2LValues[k] *
                 (power(dx, exponents[0]) * power(dy, exponents[1]) * power(dz, exponents[2]));
    }

    return value;
}

TEST(QfmmLMultTest, TheM2LRowIsHomogeneousSoTheReversedDirectionIsTheForwardSign) {
    // The fold's licence, on the SHIPPED tables at every (t, s) of every
    // order: each row is homogeneous of degree l_t + l_s, so every monomial
    // of the run flips by the same sign under d -> -d and the reversed
    // direction is the forward value times exactly that sign. The identity
    // is asserted BIT FOR BIT (the doubles, not a tolerance): the fold is
    // only value-neutral if the two are the same double with a flipped
    // sign bit.
    const std::array<std::array<double, 3>, 4> displacements = {
        {{0.7, -1.3, 2.1}, {-0.25, 0.5, -3.75}, {2.0, 2.0, -0.125}, {-1.0, 0.0, 0.5}}};

    std::size_t degreeChecks = 0;
    std::size_t parityChecks = 0;

    for (int limit = 0; limit <= kQfmmMaxLMult; ++limit)
    {
        for (int lt = 0; lt <= limit; ++lt)
        {
            for (int mt = -lt; mt <= lt; ++mt)
            {
                const int t = lt * lt + lt + mt;

                for (int ls = 0; ls <= limit; ++ls)
                {
                    for (int ms = -ls; ms <= ls; ++ms)
                    {
                        const int s = ls * ls + ls + ms;
                        ASSERT_EQ(LMultTestLOfIndex(t), lt);
                        ASSERT_EQ(LMultTestLOfIndex(s), ls);
                        const std::size_t row = static_cast<std::size_t>(t) * kQfmmMomentCount +
                                                static_cast<std::size_t>(s);

                        for (std::uint32_t k = kQfmmM2LRunOffset[row];
                             k < kQfmmM2LRunOffset[row + 1];
                             ++k)
                        {
                            const std::array<int, 3>& exponents =
                                kQfmmMonomialExponents[static_cast<std::size_t>(
                                    kQfmmM2LMonomial[k])];
                            EXPECT_EQ(exponents[0] + exponents[1] + exponents[2], lt + ls)
                                << "row (" << lt << "," << mt << ") x (" << ls << "," << ms
                                << ") monomial " << k << " is not homogeneous of degree "
                                << lt + ls;
                            ++degreeChecks;
                        }

                        for (const auto& d : displacements)
                        {
                            const double forward = ReferenceM2LRowValue(t, s, d[0], d[1], d[2]);
                            const double reversed = ReferenceM2LRowValue(t, s, -d[0], -d[1], -d[2]);
                            const double expected = ((lt + ls) % 2 == 0) ? forward : -forward;
                            EXPECT_TRUE(reversed == expected)
                                << "row (" << lt << "," << mt << ") x (" << ls << "," << ms
                                << "): EvalM2L(-d) is not the forward value times "
                                << (((lt + ls) % 2 == 0) ? "+1" : "-1");
                            ++parityChecks;
                        }
                    }
                }
            }
        }
    }

    EXPECT_GT(degreeChecks, 0u);
    EXPECT_GT(parityChecks, 0u);
}

TEST(QfmmLMultTest, TheFarFieldPassFoldsTheReversedDirectionAndMovesNoValue) {
    // The fold's reach, on a real C12H26 octree and its real far-pair list at
    // a preset rung and the production order selection (the same geometry and
    // the same orders the QfmmJBuilder resolves). Three things are asserted,
    // and the LAST is the one that fails if the wiring accepts the fold and
    // drops it:
    //   1. the far field is NOT empty (the cell cannot pass vacuously - a
    //      fixture that stops reaching the far field must fail here, not
    //      silently skip),
    //   2. the pass's potentials are BIT-IDENTICAL to the pre-fold
    //      arithmetic (the reference below evaluates the reversed direction
    //      with a second table read, which is what the pass used to do),
    //   3. the pass reports one fold per (t, s) row, against the row count
    //      derived HERE from the pair orders - so the sink is tied to the
    //      loop the pass runs, not to a number the pass supplies about
    //      itself.
    // theta 0.7, not the production rung: on this fixture the far field is
    // EMPTY below theta ~ 0.5 at the kNormal extent rung (measured: 0 pairs
    // at 0.4, 4 at 0.5, 490 at 0.7 - the last is the recorded C12H26 count,
    // so this is the criterion's known regime, not a fixture accident). The
    // pass, the pair orders and the fold are the same at any theta; only the
    // pair count changes.
    const auto geometry = BuildOrderGateGeometry(12, AccuracyPreset::kNormal, 0.7);
    ASSERT_TRUE(geometry.has_value()) << geometry.error().message;
    ASSERT_FALSE(geometry->farFieldPairs.empty())
        << "the far field is empty on this fixture: the fold is not reached, so this cell "
           "cannot witness it - a fixture or criterion change that empties the far field "
           "must fail here rather than pass vacuously";

    const int cap = qcx::integrals::LMultForPreset(AccuracyPreset::kNormal);
    const double epsInt = qcx::integrals::QfmmBudgetForPreset(AccuracyPreset::kNormal) /
                          static_cast<double>(geometry->farFieldPairs.size());
    const std::vector<int> pairOrders = qcx::integrals::internal::SelectMultipoleOrders(
        geometry->tree, geometry->farFieldPairs, epsInt, cap);
    const std::vector<int> nodeOrders = qcx::integrals::internal::ComputeNeededNodeOrders(
        geometry->tree, geometry->farFieldPairs, pairOrders);

    // A deterministic non-trivial moment field (the fold must hold for any
    // content, not only for zero-adds).
    std::vector<double> moments(geometry->tree.nodes.size() * kQfmmMomentCount);
    std::uint64_t state = 0x2545F4914F6CDD1DULL;

    for (double& moment : moments)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        moment = static_cast<double>(state % 2000001) / 1000000.0 - 1.0;
    }

    QfmmM2LCoefficientStats stats;
    const std::vector<double> folded = qcx::integrals::internal::BuildFarFieldPotentials(
        geometry->tree, geometry->farFieldPairs, moments, pairOrders, nodeOrders, 1, &stats);

    // The row count the pass must have folded: (order + 1)^4 rows per pair,
    // one for each (l_t, m_t, l_s, m_s) with both l's bounded by the pair's
    // own order - each row's reversed direction is one fold.
    std::size_t expectedRows = 0;

    for (const int order : pairOrders)
    {
        const std::size_t side =
            static_cast<std::size_t>(order + 1) * static_cast<std::size_t>(order + 1);
        expectedRows += side * side;
    }

    ASSERT_GT(expectedRows, 0u);
    EXPECT_EQ(stats.foldedRows, expectedRows)
        << "the pass folded " << stats.foldedRows << " of " << expectedRows
        << " (t, s) rows: the reversed direction of the far pair's own row was evaluated "
           "from the table again instead of served by the row parity";

    // The bit-identity reference: the same arithmetic with BOTH directions
    // evaluated from the table, on the M2L alone. The L2L is a no-op under
    // all -1 node orders (a -1 node's subtree is skipped), which isolates
    // the pass that changed from the one that did not.
    const std::vector<int> noOrders(geometry->tree.nodes.size(), -1);
    const std::vector<double> foldedM2LOnly = qcx::integrals::internal::BuildFarFieldPotentials(
        geometry->tree, geometry->farFieldPairs, moments, pairOrders, noOrders, 1);
    std::vector<double> reference(geometry->tree.nodes.size() * kQfmmMomentCount, 0.0);

    for (std::size_t i = 0; i < geometry->farFieldPairs.size(); ++i)
    {
        const auto& [a, b] = geometry->farFieldPairs[i];
        const double dx = geometry->tree.nodes[a].centerX - geometry->tree.nodes[b].centerX;
        const double dy = geometry->tree.nodes[a].centerY - geometry->tree.nodes[b].centerY;
        const double dz = geometry->tree.nodes[a].centerZ - geometry->tree.nodes[b].centerZ;
        const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
        const int order = pairOrders[i];

        for (int lt = 0; lt <= order; ++lt)
        {
            for (int mt = -lt; mt <= lt; ++mt)
            {
                const int t = lt * lt + lt + mt;

                for (int ls = 0; ls <= order; ++ls)
                {
                    for (int ms = -ls; ms <= ls; ++ms)
                    {
                        const int s = ls * ls + ls + ms;
                        const double coeffA = ReferenceM2LRowValue(t, s, dx, dy, dz);
                        const double coeffB = ReferenceM2LRowValue(t, s, -dx, -dy, -dz);
                        double denom = distance;

                        for (int power = 1; power < kQfmmM2LDenomPower[t][s]; ++power)
                        {
                            denom *= distance;
                        }

                        reference[a * kQfmmMomentCount + t] +=
                            coeffA * (moments[b * kQfmmMomentCount + s] / denom);
                        reference[b * kQfmmMomentCount + t] +=
                            coeffB * (moments[a * kQfmmMomentCount + s] / denom);
                    }
                }
            }
        }
    }

    ASSERT_EQ(reference.size(), foldedM2LOnly.size());
    std::size_t differing = 0;
    double worst = 0.0;

    for (std::size_t k = 0; k < reference.size(); ++k)
    {
        if (reference[k] != foldedM2LOnly[k])
        {
            ++differing;
            worst = std::max(worst, std::abs(reference[k] - foldedM2LOnly[k]));
        }
    }

    EXPECT_EQ(differing, 0u) << "the fold moved " << differing
                             << " potentials entries, worst |delta| " << worst;
}

} // namespace
