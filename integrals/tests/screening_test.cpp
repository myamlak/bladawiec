// Schwarz screening: the per-pair bounds are positive, and the screened
// quartet list satisfies the bound property |(ab|cd)| <= Q_ab * Q_cd on
// every element of the synthetic s/p shellset (the mpmath-grid fixture of
// eri_batch_test.cpp).

#include "h2_sto3g.hpp"
#include "h2o_ccpvdz.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "internal/md_batch.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/sparsity_pattern.hpp"
#include "shellset_fixture.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

TEST(ScreeningTest, H2BoundsArePositive) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
    ASSERT_EQ(bounds->size(), 3u); // One bound per shell pair: (0,0), (0,1), (1,1).
    EXPECT_GT((*bounds)[0], 0.0);
    EXPECT_GT((*bounds)[1], 0.0);
    EXPECT_GT((*bounds)[2], 0.0);
}

TEST(ScreeningTest, ShellsetBoundsDominateEveryElement) {
    if (!qcx::integrals::SupportsL(2))
    {
        GTEST_SKIP() << "this build's Lmax is below the shellset's d shell";
    }

    auto basis = qcx::basisset::ParseNwchemText(qcx::testing::kShellsetBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    ASSERT_EQ(bounds->size(), pairList->pairs.size());

    // Every canonical quartet over the s/p shells (the d-shell pairs are
    // excluded - the class (2,2) needs the same 2e grid as the reference).
    const std::vector<std::size_t> sPShells = {0, 1, 3, 4};
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (std::size_t a : sPShells)
    {
        for (std::size_t b : sPShells)
        {
            if (b < a)
            {
                continue;
            }

            for (std::size_t c : sPShells)
            {
                for (std::size_t d : sPShells)
                {
                    if (d < c)
                    {
                        continue;
                    }

                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    auto batch = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->computed.size(), quartets.size());

    std::size_t offset = 0;

    for (const qcx::integrals::ShellQuartet& quartet : batch->computed)
    {
        const std::size_t braPair = qcx::integrals::PairIndexOf(quartet.i, quartet.j, *pairList);
        const std::size_t ketPair = qcx::integrals::PairIndexOf(quartet.k, quartet.l, *pairList);
        const double bound = (*bounds)[braPair] * (*bounds)[ketPair];
        const std::size_t nI = 2 * pairList->shells[quartet.i].angularMomentum + 1;
        const std::size_t nJ = 2 * pairList->shells[quartet.j].angularMomentum + 1;
        const std::size_t nK = 2 * pairList->shells[quartet.k].angularMomentum + 1;
        const std::size_t nL = 2 * pairList->shells[quartet.l].angularMomentum + 1;

        for (std::size_t fi = 0; fi < nI; ++fi)
        {
            for (std::size_t fj = 0; fj < nJ; ++fj)
            {
                for (std::size_t fk = 0; fk < nK; ++fk)
                {
                    for (std::size_t fl = 0; fl < nL; ++fl)
                    {
                        const double value =
                            batch->values[offset + (fj * nI + fi) * (nK * nL) + (fl * nK + fk)];
                        EXPECT_LE(std::abs(value), bound * (1.0 + 1e-12))
                            << "quartet (" << quartet.i << "," << quartet.j << "|" << quartet.k
                            << "," << quartet.l << ")";
                    }
                }
            }
        }

        offset += nI * nJ * nK * nL;
    }
}

TEST(ScreeningTest, RepresentsLinkStyleNeighborListFromRealScreening) {
    // LinK's production screening machinery (the actual
    // BuildNeighborList/ScreenAll of internal/fock_screen.hpp) represented
    // as a SparsityPattern<CpuTag> - not the C60-connectivity stand-in of
    // the 2026-08-16 prototype. The pattern-driven screening pass must
    // screen bit-identically to the raw-CSR form, on a real fixture (H2O
    // cc-pVDZ: the 24-shell basis of data/basis), in both the fp64-only and
    // the certified-mixed-precision lanes.
    if (!qcx::integrals::SupportsL(2))
    {
        GTEST_SKIP() << "this build's Lmax is below the cc-pVDZ d shells";
    }

    auto molecule = qcx::testing::MakeH2oCcpvdz();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oCcpvdzBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;
    ASSERT_EQ(schwarz->size(), pairList->pairs.size());

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kLoose;
    options.maxParallelChunks = 1; // Serial: exact decisions, canonical order.

    // The production neighbor list, exactly as the builders' Create builds
    // it...
    std::vector<std::size_t> rowOffsets;
    std::vector<std::size_t> neighborIndices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, rowOffsets, neighborIndices);

    // ... wrapped in the shared sparsity-pattern primitive (the migrated State
    // member's construction site: BuildSparsityFromAdjacency is the
    // documented LinK-shape builder).
    auto pattern =
        qcx::memory::BuildSparsityFromAdjacency<qcx::backend::CpuTag>(rowOffsets, neighborIndices);
    ASSERT_TRUE(pattern.has_value()) << pattern.error().message;
    EXPECT_EQ(pattern->RowCount(), pairList->pairs.size());
    EXPECT_EQ(pattern->IndexCount(), neighborIndices.size());
    EXPECT_EQ(pattern->RowOffsets().HostView(), rowOffsets);
    EXPECT_EQ(pattern->Indices().HostView(), neighborIndices);

    // A symmetric density with non-trivial block maxima: identity puts a
    // unit on every diagonal element, so the six-block density weights vary
    // per quartet and both screening gates take real decisions.
    const std::size_t n = pairList->functionCount;
    const Eigen::MatrixXd density =
        Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    const qcx::integrals::internal::ScreeningContext context{
        *pairList, *schwarz, *pairStore, options};
    const double densityThreshold = qcx::integrals::DensityThreshold(options.accuracy);
    const double mixedThreshold = qcx::integrals::MixedPrecisionThreshold(options.accuracy);

    for (const bool certifiedLane : {false, true})
    {
        std::vector<qcx::integrals::internal::MdQuartetTask> csr64;
        std::vector<qcx::integrals::internal::MdQuartetTask> csr32;
        std::vector<double> csrWeights;
        std::vector<qcx::integrals::internal::MdQuartetTask> pattern64;
        std::vector<qcx::integrals::internal::MdQuartetTask> pattern32;
        std::vector<double> patternWeights;

        qcx::integrals::internal::ScreenAll(context,
                                            density,
                                            rowOffsets,
                                            neighborIndices,
                                            densityThreshold,
                                            mixedThreshold,
                                            certifiedLane,
                                            csr64,
                                            csr32,
                                            csrWeights);
        qcx::integrals::internal::ScreenAll(context,
                                            density,
                                            *pattern,
                                            densityThreshold,
                                            mixedThreshold,
                                            certifiedLane,
                                            pattern64,
                                            pattern32,
                                            patternWeights);

        EXPECT_EQ(csr64.size(), pattern64.size());
        EXPECT_EQ(csr32.size(), pattern32.size());
        EXPECT_EQ(csrWeights.size(), patternWeights.size());

        for (std::size_t i = 0; i < csr64.size(); ++i)
        {
            EXPECT_EQ(csr64[i].braPair, pattern64[i].braPair);
            EXPECT_EQ(csr64[i].ketPair, pattern64[i].ketPair);
        }

        for (std::size_t i = 0; i < csr32.size(); ++i)
        {
            EXPECT_EQ(csr32[i].braPair, pattern32[i].braPair);
            EXPECT_EQ(csr32[i].ketPair, pattern32[i].ketPair);
        }

        for (std::size_t i = 0; i < csrWeights.size(); ++i)
        {
            EXPECT_EQ(csrWeights[i], patternWeights[i]);
        }
    }

    // The real screening output is genuinely a CSR neighbor list of the
    // LinK shape: offsets start at 0, end at the packed size, and every
    // entry is a valid ket pair index.
    const auto& offsets = pattern->RowOffsets().HostView();
    const auto& indices = pattern->Indices().HostView();
    EXPECT_EQ(offsets.front(), 0u);
    EXPECT_EQ(offsets.back(), indices.size());

    for (const auto ket : indices)
    {
        EXPECT_LT(ket, pairList->pairs.size());
    }
}

// The per-target gate vs the legacy six-block-max gate at
// uniform densities D = c * I. The products are the CERTIFIED
// per-quartet bounds of the mode's work (CORRECTED 2026-08-29 - the original
// P_ab * P_cd form under-bounded and was falsified by the C80H162/STO-3G
// kLoose sweep, 1.5e-2 deviation vs the 6.6e-9 budget): the J bound is
// (max|D_ab| + max|D_cd|) * Q_ab * Q_cd and the K bound is
// (max|D_ac| + max|D_bd| + max|D_ad| + max|D_bc|) * Q_ab * Q_cd. At
// D = c * I every diagonal shell-pair block max is exactly c and every
// off-diagonal max is exactly 0, so the per-class relationship to the legacy
// dMax * Q_bra * Q_ket gate is provable by inspection: (a,a|b,b) quartets
// bound at 2c * Q_ab * Q_cd (a keep-superset of the legacy value at every
// c); quartets with exactly one diagonal bra/ket pair bound at c * Q_ab * Q_cd
// (identical verdicts); quartets with neither bra nor ket diagonal have
// EXACTLY zero J work - the corrected gate drops them, where the legacy
// six-block max over-keeps them when a cross pair is diagonal (the full gate
// keeps at least the legacy-kept set: the K sum bound is >= the legacy max
// bound on every diagonal cross pair). The K products
// (Q_ac * Q_bd etc.) are INDEPENDENT Schwarz bounds of the same quartet - not
// comparable to Q_bra * Q_ket in general - so the K-mode assertions stay on
// the pure synthetic-maxima calls below.
TEST(ScreeningTest, PerTargetGateMatchesSixBlockMaxAtUniformDensity) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;
    ASSERT_EQ(schwarz->size(), pairList->pairs.size());

    // Every canonical quartet (bra >= ket in pair order - the screening
    // enumeration).
    const std::size_t nShells = pairList->shells.size();
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (std::size_t a = 0; a < nShells; ++a)
    {
        for (std::size_t b = 0; b <= a; ++b)
        {
            for (std::size_t c = 0; c < nShells; ++c)
            {
                for (std::size_t d = 0; d <= c; ++d)
                {
                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    ASSERT_FALSE(quartets.empty());
    const double threshold =
        qcx::integrals::DensityThreshold(qcx::integrals::AccuracyPreset::kLoose);
    const std::size_t n = pairList->functionCount;

    // Per-class exercise counters (asserted non-zero after the loop - an
    // over-fitted fixture could leave an assertion branch unexercised).
    std::size_t diagonalQuartets = 0;
    std::size_t mixedQuartets = 0;
    std::size_t offDiagonalQuartets = 0;
    std::size_t offDiagonalLegacyKept = 0;

    for (const double c : {0.5, 1.0, 2.0})
    {
        const Eigen::MatrixXd density = c * Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n),
                                                                      static_cast<Eigen::Index>(n));
        const std::vector<double> pairMax =
            qcx::integrals::internal::BuildShellPairMaxDensity(density, *pairList);

        for (const qcx::integrals::ShellQuartet& quartet : quartets)
        {
            const std::size_t bra = qcx::integrals::PairIndexOf(quartet.i, quartet.j, *pairList);
            const std::size_t ket = qcx::integrals::PairIndexOf(quartet.k, quartet.l, *pairList);
            const std::size_t ac =
                qcx::integrals::internal::CanonicalPairIndexOf(quartet.i, quartet.k, *pairList);
            const std::size_t bd =
                qcx::integrals::internal::CanonicalPairIndexOf(quartet.j, quartet.l, *pairList);
            const std::size_t ad =
                qcx::integrals::internal::CanonicalPairIndexOf(quartet.i, quartet.l, *pairList);
            const std::size_t bc =
                qcx::integrals::internal::CanonicalPairIndexOf(quartet.j, quartet.k, *pairList);

            // The legacy gate's value: the true SIX-block max (the pair-max
            // vector holds exactly the per-pair DensityBlockMax values, so
            // the cross pairs need no extra scans).
            const double dAb = qcx::integrals::internal::DensityBlockMax(
                density,
                pairList->shells[quartet.i].functionOffset,
                pairList->shells[quartet.j].functionOffset,
                qcx::integrals::ShellFunctionCount(pairList->shells[quartet.i]),
                qcx::integrals::ShellFunctionCount(pairList->shells[quartet.j]));
            const double dCd = qcx::integrals::internal::DensityBlockMax(
                density,
                pairList->shells[quartet.k].functionOffset,
                pairList->shells[quartet.l].functionOffset,
                qcx::integrals::ShellFunctionCount(pairList->shells[quartet.k]),
                qcx::integrals::ShellFunctionCount(pairList->shells[quartet.l]));
            const double dMax =
                std::max({dAb, dCd, pairMax[ac], pairMax[bd], pairMax[ad], pairMax[bc]});
            const bool sixBlockKeep = dMax * (*schwarz)[bra] * (*schwarz)[ket] >= threshold;

            // The corrected gate takes the RAW block maxima and the RAW
            // Schwarz pair values (the products are formed at the gate).
            const std::array<double, 6> pairBounds{
                dAb, dCd, pairMax[ac], pairMax[bd], pairMax[ad], pairMax[bc]};
            const std::array<double, 6> pairQ{(*schwarz)[bra],
                                              (*schwarz)[ket],
                                              (*schwarz)[ac],
                                              (*schwarz)[bd],
                                              (*schwarz)[ad],
                                              (*schwarz)[bc]};

            const bool jKeep = qcx::integrals::internal::QuartetDensityGate(
                pairBounds, pairQ, false, true, threshold);
            const bool fullKeep = qcx::integrals::internal::QuartetDensityGate(
                pairBounds, pairQ, false, false, threshold);

            const bool braDiagonal = quartet.i == quartet.j;
            const bool ketDiagonal = quartet.k == quartet.l;

            if (braDiagonal && ketDiagonal)
            {
                ++diagonalQuartets;
                // (a,a|b,b): the corrected J bound (c + c) * Q = 2c * Q is
                // a keep-superset of the legacy dMax * Q = c * Q at every c
                // (both J orientations contract a diagonal density block).
                EXPECT_TRUE(!sixBlockKeep || jKeep)
                    << "c = " << c << ": J-mode dropped a legacy-kept (diag,diag) quartet ("
                    << quartet.i << "," << quartet.j << "|" << quartet.k << "," << quartet.l << ")";
                EXPECT_TRUE(!sixBlockKeep || fullKeep)
                    << "c = " << c << ": full gate dropped a legacy-kept (diag,diag) quartet ("
                    << quartet.i << "," << quartet.j << "|" << quartet.k << "," << quartet.l << ")";
            } else if (braDiagonal || ketDiagonal)
            {
                ++mixedQuartets;
                // Exactly one diagonal bra/ket pair: the corrected J bound
                // (c + 0) * Q equals the legacy dMax * Q = c * Q exactly
                // (the +0.0 addition is exact) - identical verdicts; the
                // full gate can only add K-side keeps.
                EXPECT_EQ(jKeep, sixBlockKeep)
                    << "c = " << c
                    << ": J-mode verdict differs from the legacy gate on the mixed "
                       "quartet ("
                    << quartet.i << "," << quartet.j << "|" << quartet.k << "," << quartet.l << ")";
                EXPECT_TRUE(!sixBlockKeep || fullKeep)
                    << "c = " << c << ": full gate dropped a legacy-kept mixed quartet ("
                    << quartet.i << "," << quartet.j << "|" << quartet.k << "," << quartet.l << ")";
            } else
            {
                ++offDiagonalQuartets;
                // Neither bra nor ket diagonal: both J orientations contract
                // zero off-diagonal density blocks - the corrected J bound
                // is exactly 0 and drops EXACTLY, where the legacy six-block
                // max reads a diagonal cross pair (when one exists) and
                // over-keeps the zero-J-work quartet. The full gate is a
                // keep-superset of the legacy: a diagonal cross pair gives a
                // K bound k * c * Q with k >= 1 - at least the legacy
                // dMax * Q = c * Q, more when several cross pairs are
                // diagonal (the sum form is the certified bound of the four
                // K sections' total work); without one all six maxes are
                // zero and both products are zero.
                EXPECT_FALSE(jKeep)
                    << "c = " << c << ": J-mode kept a zero-J-work quartet (" << quartet.i << ","
                    << quartet.j << "|" << quartet.k << "," << quartet.l << ")";
                EXPECT_TRUE(!sixBlockKeep || fullKeep)
                    << "c = " << c << ": full gate dropped a legacy-kept off-diagonal quartet ("
                    << quartet.i << "," << quartet.j << "|" << quartet.k << "," << quartet.l << ")";
                offDiagonalLegacyKept += sixBlockKeep ? 1 : 0;
            }
        }
    }

    // The fixture exercises all three classes non-vacuously, and the legacy
    // over-keep direction is real: at D = c * I every off-diagonal quartet
    // with a shared shell (e.g. (a,b|a,b)) is kept by the legacy six-block
    // max (a cross pair is diagonal, dMax = c) while its J work is exactly
    // zero - the corrected gate drops it exactly.
    EXPECT_GT(diagonalQuartets, 0u) << "no (diag,diag) quartet on the H2O STO-3G fixture";
    EXPECT_GT(mixedQuartets, 0u) << "no mixed quartet on the H2O STO-3G fixture";
    EXPECT_GT(offDiagonalQuartets, 0u) << "no off-diagonal quartet on the H2O STO-3G fixture";
    EXPECT_GT(offDiagonalLegacyKept, 0u)
        << "no off-diagonal quartet over-kept by the legacy gate on the H2O STO-3G fixture";

    // Sparse synthetic maxima pin the corrected gate's certified-drop and
    // certified-keep directions (Q = 1 throughout so the pair bounds ARE the
    // block maxima). The legacy six-block max keeps on the largest single
    // block; the corrected gate decides on the mode's ACTUAL work bound:
    {
        const std::array<double, 6> q1{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        // A dense SOURCE pair keeps the J work even with a zero target-pair
        // max - (0.0 + 0.5) * 1 = 0.5 >= 0.1. This is the (a != b | c,c)
        // class at a diagonal density, which the original P_ab * P_cd
        // product dropped exactly (the H2O 6.29991 / alkane 1.5e-2
        // findings).
        const std::array<double, 6> denseSource{0.0, 0.5, 0.0, 0.0, 0.0, 0.0};
        EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
            denseSource, q1, false, true, 0.1)); // J: 0.5 keeps.
        EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
            denseSource, q1, true, false, 0.1)); // K: 0.0 drops.
        EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
            denseSource, q1, false, false, 0.1)); // Full: via J.
        // Genuinely tiny J pairs with dense K-side maxima: the J work
        // (0.04 + 0.04) = 0.08 < 0.1 drops - the certified J drop, where the
        // legacy six-block dMax = 0.5 keeps; the K work
        // (0.5 + 0.5 + 0.5 + 0.5) = 2.0 keeps.
        const std::array<double, 6> tinyJDenseK{0.04, 0.04, 0.5, 0.5, 0.5, 0.5};
        EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
            tinyJDenseK, q1, false, true, 0.1)); // J: 0.08 drops.
        EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
            tinyJDenseK, q1, true, false, 0.1)); // K: 2.0 keeps.
        EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
            tinyJDenseK, q1, false, false, 0.1)); // Full: via K.
        // The (diag,diag) superset-drop direction: the corrected J bound
        // 0.25 + 0.25 = 0.5 clears a threshold the legacy dMax = 0.25 does
        // not - the gate keeps what the legacy gate would drop.
        EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
            {0.25, 0.25, 0.0, 0.0, 0.0, 0.0}, q1, false, true, 0.4));
        // The equality boundary: (0.05 + 0.05) = 0.1 keeps at tau = 0.1.
        EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
            {0.05, 0.05, 0.0, 0.0, 0.0, 0.0}, q1, false, true, 0.1));
        EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
            {0.0, 0.0, 0.05, 0.05, 0.0, 0.0}, q1, true, false, 0.1));
    }
}

// The pure mode-awareness of QuartetDensityGate - J-only /
// K-only / full, the equality boundary (drop iff strictly less, matching the
// legacy gate's < drop convention), and the all-zero-maxima drop. Q = 1
// throughout so the pair bounds ARE the block maxima; the products are the
// CORRECTED certified bounds (2026-08-29): J = (b0 + b1) * q0 * q1, K =
// (b2 + b3 + b4 + b5) * q0 * q1.
TEST(ScreeningTest, PerTargetGateModeAware) {
    const double threshold = 0.1;
    const std::array<double, 6> q1{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

    // J-only: keep iff (b0 + b1) * q0 * q1 >= threshold; equality keeps.
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.25, 0.4, 0.0, 0.0, 0.0, 0.0}, q1, false, true, threshold));
    // The corrected sum form keeps a pair of sub-threshold blocks that the
    // original P_ab * P_cd product dropped (0.25 * 0.39 = 0.0975 < 0.1).
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.25, 0.39, 0.0, 0.0, 0.0, 0.0}, q1, false, true, threshold));
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.05, 0.049, 0.0, 0.0, 0.0, 0.0}, q1, false, true, threshold));
    // A dense SOURCE pair keeps the J work with a zero target-pair max - the
    // class the original product form dropped exactly (the H2O 6.29991 /
    // alkane 1.5e-2 findings).
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.5, 0.0, 0.0, 0.0, 0.0}, q1, false, true, threshold));
    // The K products are irrelevant in J-only mode.
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.25, 0.4, 1e-9, 1e-9, 1e-9, 1e-9}, q1, false, true, threshold));

    // K-only: keep iff (b2 + b3 + b4 + b5) * q0 * q1 >= threshold.
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.25, 0.4, 0.0, 0.0}, q1, true, false, threshold)); // (ac,bd).
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.0, 0.0, 0.25, 0.4}, q1, true, false, threshold)); // (ad,bc).
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.25, 0.39, 0.0, 0.0}, q1, true, false, threshold)); // Sum form keeps.
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.05, 0.049, 0.0, 0.0}, q1, true, false, threshold));
    // The J products are irrelevant in K-only mode: large J products with
    // zero K products still drop.
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.25, 0.4, 0.0, 0.0, 0.0, 0.0}, q1, true, false, threshold));
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {1e-9, 1e-9, 0.0, 0.0, 0.25, 0.4}, q1, true, false, threshold));

    // Full: keep iff EITHER product clears the threshold.
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.25, 0.4, 1e-9, 1e-9, 1e-9, 1e-9}, q1, false, false, threshold)); // via J.
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {1e-9, 1e-9, 0.25, 0.4, 1e-9, 1e-9}, q1, false, false, threshold)); // via K.
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.05, 0.049, 1e-9, 1e-9, 1e-9, 1e-9}, q1, false, false, threshold));

    // All-zero maxima drop in every mode (0 < threshold).
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, q1, false, true, threshold));
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, q1, true, false, threshold));
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, q1, false, false, threshold));

    // The boundary convention: equality keeps, at a zero threshold too (the
    // >= form, drop iff strictly less - the legacy gate's convention).
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, q1, false, false, 0.0));
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.25, 0.4, 0.0, 0.0, 0.0, 0.0}, q1, false, true, 0.0));
    EXPECT_TRUE(qcx::integrals::internal::QuartetDensityGate(
        {0.25, 0.39, 0.0, 0.0, 0.0, 0.0}, q1, false, false, 0.0));
    EXPECT_FALSE(qcx::integrals::internal::QuartetDensityGate(
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, q1, false, false, 1e-3));
}

// The chunked sweep (the unmodeled-pair-store finding): the sweep needs
// one diagonal (ab|ab) block per canonical pair, and a diagonal quartet's
// bra and ket pair are the SAME pair, so the pair data can be built and
// dropped one chunk at a time. The chunking is a partition of the same
// quartets through the same builder over the same pair data, so the values
// are the pre-chunking single pass's BYTE FOR BYTE - the property that makes
// the memory reduction free. chunkBytes = 0 is the single-pass reference
// (one chunk over the whole list); chunkBytes = 1 forces one pair per chunk,
// the maximal partition.
TEST(ScreeningTest, ChunkedSweepIsBitIdenticalToTheSinglePass) {
    auto molecule = qcx::testing::MakeH2oCcpvdz();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oCcpvdzBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto single = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis, 0);
    ASSERT_TRUE(single.has_value()) << single.error().message;
    auto perPair = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis, 1);
    ASSERT_TRUE(perPair.has_value()) << perPair.error().message;

    ASSERT_EQ(perPair->size(), single->size());
    ASSERT_GT(single->size(), 1u) << "the one-pair cap must actually cut this fixture";

    for (std::size_t pair = 0; pair < single->size(); ++pair)
    {
        EXPECT_EQ((*perPair)[pair], (*single)[pair]) << "pair " << pair;
    }
}

// The sweep's charge (footprint.hpp SchwarzSweepBytes): the phase-max check
// the estimate was missing. The chunked sweep's own peak - the geometry
// skeleton, the largest chunk's pair data, that chunk's batch arena and the
// bounds vector - is bounded by the pair store the same estimate already
// charges whole, so the schwarzSweepBytes term carries zero excess at every
// reachable size while the quantities stay honest: the charge grows with the
// chunk cap and never drops below the skeleton plus the bounds vector.
TEST(ScreeningTest, SchwarzSweepChargeIsBoundedByTheStoreTerm) {
    using qcx::integrals::internal::SchwarzSweepBytes;
    using qcx::integrals::internal::SchwarzSweepTerms;
    using qcx::integrals::internal::SchwarzSweepTermsOf;

    auto molecule = qcx::testing::MakeH2oCcpvdz();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::testing::MakeH2oCcpvdzBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t wholeStore =
        qcx::integrals::internal::PairStoreBytes(*molecule, *basis, *pairList);
    const std::size_t unchunked = SchwarzSweepBytes(*molecule, *basis, *pairList, 0);
    const std::size_t chunked = SchwarzSweepBytes(*molecule, *basis, *pairList);
    const SchwarzSweepTerms terms = SchwarzSweepTermsOf(*molecule, *basis, *pairList, 0);

    // The unchunked sweep is the whole store's payload plus the geometry
    // skeleton and the bounds vector - the pre-chunking allocation.
    EXPECT_GE(unchunked, wholeStore);
    // The skeleton is sizeof(MdPairData) per canonical pair plus headers,
    // and the bounds are 8 B per pair: the floor of every sweep.
    EXPECT_GE(chunked, nPairs * sizeof(qcx::integrals::internal::MdPairData) + 8 * nPairs);
    // The default (chunked) charge never exceeds the unchunked one, and the
    // term struct decomposes the charge exactly.
    EXPECT_LE(chunked, unchunked);
    EXPECT_LE(terms.chunkPayloadBytes, wholeStore);
    const SchwarzSweepTerms defaultTerms =
        SchwarzSweepTermsOf(*molecule, *basis, *pairList, qcx::integrals::kSchwarzChunkBytes);
    EXPECT_EQ(chunked,
              defaultTerms.skeletonBytes + defaultTerms.chunkPayloadBytes +
                  defaultTerms.chunkBatchBytes + defaultTerms.boundsBytes);
}

} // namespace
