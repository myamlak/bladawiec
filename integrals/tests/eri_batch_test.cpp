// The batched MD engine tests: the H2/STO-3G cross-engine regression (the
// batch reproduces the committed s-only CSV), the canonicalization contract,
// the certified fp32 bounds, the dense driver (8-fold mirroring,
// batch-scatter equivalence), the committed mpmath grids of
// tools/gen_md_reference.py (shellset at 1e-12, HF/H2O at 1e-10), and the
// unfolded reference path. Heavy grid walks are fast-mode-gated.

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2_sto3g_fixture.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_boys.hpp"
#include "internal/md_engine.hpp"
#include "internal/md_one_electron.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "shellset_fixture.hpp"
#include "unfolded_reference.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::test::LoadReferenceValues;

// One row of the new grid CSVs (kind, mol, i, j, k, l, a, b, c, d, value)
// where the 2e rows carry the packed (row, col) in a/b.
struct GridRow {
    std::string kind;
    std::string mol;
    std::size_t i, j, k, l;
    std::size_t a, b, c, d;
    double value;
};

std::vector<GridRow> LoadGrid(const std::string& fileName) {
    const std::string path = std::string(QcxIntegralsDataDir) + "/" + fileName;
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<GridRow> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        GridRow row{};
        std::getline(ss, row.kind, ',');
        std::getline(ss, row.mol, ',');
        std::getline(ss, cell, ',');
        row.i = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.j = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.k = cell.empty() ? 0 : std::stoull(cell);
        std::getline(ss, cell, ',');
        row.l = cell.empty() ? 0 : std::stoull(cell);
        std::getline(ss, cell, ',');
        row.a = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.b = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.c = cell.empty() ? 0 : std::stoull(cell);
        std::getline(ss, cell, ',');
        row.d = cell.empty() ? 0 : std::stoull(cell);
        std::getline(ss, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);
        rows.push_back(row);
    }

    return rows;
}

} // namespace

// ---------------------------------------------------------------------------
// H2/STO-3G: the cross-engine regression and the contracts (always run)
// ---------------------------------------------------------------------------

TEST(EriBatchTest, H2MatchesTheSOnlyCsv) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // All 16 requested forms canonicalize to one value-identical block.
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            for (std::size_t k = 0; k < 2; ++k)
            {
                for (std::size_t l = 0; l < 2; ++l)
                {
                    quartets.push_back({i, j, k, l});
                }
            }
        }
    }

    auto batch = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->computed.size(), 16u);
    EXPECT_EQ(batch->values.size(), 16u);
    const std::vector<qcx::integrals::test::ReferenceValue> reference = LoadReferenceValues();

    for (std::size_t t = 0; t < 16; ++t)
    {
        // The 16 requested forms canonicalize to the CSV's distinct H2
        // quartets ((0,0|0,0), (0,0|0,1), (0,1|0,1), ...); each request
        // compares against its canonical quartet's row.
        const qcx::integrals::ShellQuartet& canonical = batch->computed[t];
        double expected = 0.0;
        bool found = false;

        for (const auto& row : reference)
        {
            if (row.kind == "ERI" && row.i == canonical.i && row.j == canonical.j &&
                row.k == canonical.k && row.l == canonical.l)
            {
                expected = row.value;
                found = true;
            }
        }

        ASSERT_TRUE(found) << "request " << t << ": no CSV row for (" << canonical.i << ","
                           << canonical.j << "|" << canonical.k << "," << canonical.l << ")";
        EXPECT_NEAR(batch->values[t], expected, 1e-10) << "request " << t;
    }
}

TEST(EriBatchTest, CertifiedBoundsHoldOnH2) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    std::vector<qcx::integrals::ShellQuartet> quartets = {{0, 0, 0, 0}};

    auto fp64 = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(fp64.has_value()) << fp64.error().message;
    auto certified = qcx::integrals::ComputeEriBatchCertified(*molecule, *basis, quartets);
    ASSERT_TRUE(certified.has_value()) << certified.error().message;
    ASSERT_EQ(certified->errorBounds.size(), 1u);

    // The a-priori bound dominates the observed deviation (nothing is
    // approximate unless the bound covers it).
    const double deviation = std::abs(static_cast<double>(certified->values[0]) - fp64->values[0]);
    EXPECT_LE(deviation, certified->errorBounds[0]);
    EXPECT_GT(certified->errorBounds[0], 0.0);
}

TEST(EriBatchTest, DenseMatchesTheSOnlyCsvAndIsEightFoldBitExact) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto tensor = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;
    const std::vector<qcx::integrals::test::ReferenceValue> reference = LoadReferenceValues();

    for (const auto& row : reference)
    {
        if (row.kind != "ERI")
        {
            continue;
        }

        EXPECT_NEAR((*tensor)(row.i, row.j, row.k, row.l), row.value, 1e-10)
            << "ERI(" << row.i << "," << row.j << "," << row.k << "," << row.l << ")";
    }

    // The mirroring is bit-exact: permuted slots carry identical bits.
    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            for (std::size_t k = 0; k < 2; ++k)
            {
                for (std::size_t l = 0; l < 2; ++l)
                {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>((*tensor)(i, j, k, l)),
                              std::bit_cast<std::uint64_t>((*tensor)(j, i, k, l)));
                    EXPECT_EQ(std::bit_cast<std::uint64_t>((*tensor)(i, j, k, l)),
                              std::bit_cast<std::uint64_t>((*tensor)(i, j, l, k)));
                    EXPECT_EQ(std::bit_cast<std::uint64_t>((*tensor)(i, j, k, l)),
                              std::bit_cast<std::uint64_t>((*tensor)(k, l, i, j)));
                }
            }
        }
    }
}

TEST(EriBatchTest, ErrorPaths) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto empty = qcx::integrals::ComputeEriBatch(*molecule, *basis, {});
    EXPECT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, qcx::ErrorCode::kInvalidArgument);

    auto zeroCap = qcx::integrals::ComputeEriBatch(
        *molecule,
        *basis,
        {{0, 0, 0, 0}},
        qcx::integrals::EriBatchOptions{qcx::integrals::AccuracyPreset::kNormal, 0});
    EXPECT_FALSE(zeroCap.has_value());
    EXPECT_EQ(zeroCap.error().code, qcx::ErrorCode::kInvalidArgument);

    auto outOfRange = qcx::integrals::ComputeEriBatch(*molecule, *basis, {{0, 0, 0, 5}});
    EXPECT_FALSE(outOfRange.has_value());
    EXPECT_EQ(outOfRange.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The dense-tensor admission gate, exact-formula pin: H2/STO-3G has 2
// functions (n4 = 16, tensor 128 B, all-survive batch payload 128 B) and 3
// pairs (6 canonical quartets x 32 B = 192 B list), so the a-priori
// estimate is exactly 448 B - the cap admits at 448 and refuses one byte
// below. The refusal is the graceful kInvalidArgument (never an allocation
// death), and the default cap admits the build.
TEST(EriBatchTest, DenseTensorAdmissionGatePinsTheEstimate) {
    auto basis = qcx::testing::MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    qcx::integrals::EriDenseOptions admitted;
    admitted.maxTensorBytes = 448;
    auto fits = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis, admitted);
    ASSERT_TRUE(fits.has_value()) << fits.error().message;

    qcx::integrals::EriDenseOptions refused;
    refused.maxTensorBytes = 447;
    auto refusedBuild = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis, refused);
    ASSERT_FALSE(refusedBuild.has_value());
    EXPECT_EQ(refusedBuild.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(refusedBuild.error().message.find("maxTensorBytes"), std::string::npos);

    // The default cap (2 GiB) admits the same build - the existing
    // reference tests ride on it.
    auto defaulted = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis);
    ASSERT_TRUE(defaulted.has_value()) << defaulted.error().message;
}

// The engine-cap tripwire: no parser-valid basis can reach it (the parser
// caps l at 6), but a hand-built pair list can - pin the rejection so a
// future engine-cap regression cannot silently mis-batch an out-of-class
// quartet. l = kMaxEngineL + 1 (= 7 on the standard build) makes the
// quartet's total 2l > 2*kMaxEngineL unconditionally.
TEST(EriBatchTest, CanonicalizeQuartetOrderRejectsEngineCapExcess) {
    qcx::integrals::ShellPairList pairList;
    pairList.shells.push_back(
        qcx::integrals::ShellInfo{qcx::integrals::kMaxEngineL + 1, true, 1, 0, 0, 0});
    pairList.pairs.push_back(qcx::integrals::ShellPairIndex{0, 0});
    pairList.functionCount = qcx::integrals::ShellFunctionCount(pairList.shells[0]);

    auto ordered = qcx::integrals::CanonicalizeQuartetOrder(pairList, {{0, 0, 0, 0}});
    EXPECT_FALSE(ordered.has_value());
    EXPECT_EQ(ordered.error().code, qcx::ErrorCode::kUnimplemented);
}

// The shared-canonicalization contract: CanonicalizeQuartetOrder is
// the storage decorator's ordering source, so its output must reproduce
// EriBatch::computed exactly - including for a permuted request (the
// engine's computed order is SORTED, not request-ordered).
TEST(EriBatchTest, CanonicalizeQuartetOrderMatchesComputed) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    // The full 6-shell pair x pair request, deliberately permuted.
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (const qcx::integrals::ShellPairIndex& bra : pairList->pairs)
    {
        for (const qcx::integrals::ShellPairIndex& ket : pairList->pairs)
        {
            quartets.push_back({bra.i, bra.j, ket.i, ket.j});
        }
    }

    std::reverse(quartets.begin(), quartets.end());

    auto batch = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    auto ordered = qcx::integrals::CanonicalizeQuartetOrder(*pairList, quartets);
    ASSERT_TRUE(ordered.has_value()) << ordered.error().message;

    ASSERT_EQ(ordered->size(), batch->computed.size());

    for (std::size_t t = 0; t < ordered->size(); ++t)
    {
        const qcx::integrals::ShellQuartet& computed = batch->computed[t];
        const qcx::integrals::ShellQuartet& canonical = (*ordered)[t].quartet;
        EXPECT_EQ(computed.i, canonical.i);
        EXPECT_EQ(computed.j, canonical.j);
        EXPECT_EQ(computed.k, canonical.k);
        EXPECT_EQ(computed.l, canonical.l);
    }
}

// ---------------------------------------------------------------------------
// The mpmath grids (fast-mode-gated: heavy sweeps)
// ---------------------------------------------------------------------------

TEST(EriBatchTest, ShellsetMatchesMpmath) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = qcx::basisset::ParseNwchemText(qcx::testing::kShellsetBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<GridRow> grid = LoadGrid("md_shellset_reference.csv");
    ASSERT_GT(grid.size(), 0u);

    // The 1e matrices (S/T at 1e-11, V at 1e-11).
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    for (const GridRow& row : grid)
    {
        if (row.kind == "ERI")
        {
            continue;
        }

        const std::size_t offI = pairList->shells[row.i].functionOffset;
        const std::size_t offJ = pairList->shells[row.j].functionOffset;
        const double value = row.kind == "S"   ? (*overlap)(offI + row.a, offJ + row.b)
                             : row.kind == "T" ? (*kinetic)(offI + row.a, offJ + row.b)
                                               : (*nuclear)(offI + row.a, offJ + row.b);
        EXPECT_NEAR(value, row.value, 1e-11) << row.kind << " pair (" << row.i << "," << row.j
                                             << ") [" << row.a << "," << row.b << "]";
    }

    // The 2e blocks: request every canonical quartet once, map through the
    // batch's computed forms.
    std::vector<qcx::integrals::ShellQuartet> quartets;
    std::vector<std::pair<std::size_t, std::size_t>> keys; // {min pair, max pair}

    for (const GridRow& row : grid)
    {
        if (row.kind != "ERI")
        {
            continue;
        }

        const std::size_t bra = qcx::integrals::PairIndexOf(row.i, row.j, *pairList);
        const std::size_t ket = qcx::integrals::PairIndexOf(row.k, row.l, *pairList);

        if (std::find(keys.begin(), keys.end(), std::make_pair(bra, ket)) == keys.end())
        {
            keys.push_back({bra, ket});
            quartets.push_back({row.i, row.j, row.k, row.l});
        }
    }

    auto batch = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;

    // Block lookup by unordered pair key.
    std::size_t offset = 0;

    for (const qcx::integrals::ShellQuartet& quartet : batch->computed)
    {
        const std::size_t bra = qcx::integrals::PairIndexOf(quartet.i, quartet.j, *pairList);
        const std::size_t ket = qcx::integrals::PairIndexOf(quartet.k, quartet.l, *pairList);
        const std::size_t nI = ShellFunctionCount(pairList->shells[quartet.i]);
        const std::size_t nJ = ShellFunctionCount(pairList->shells[quartet.j]);
        const std::size_t nK = ShellFunctionCount(pairList->shells[quartet.k]);
        const std::size_t nL = ShellFunctionCount(pairList->shells[quartet.l]);

        for (const GridRow& row : grid)
        {
            if (row.kind != "ERI")
            {
                continue;
            }

            const std::size_t rowBra = qcx::integrals::PairIndexOf(row.i, row.j, *pairList);
            const std::size_t rowKet = qcx::integrals::PairIndexOf(row.k, row.l, *pairList);
            const bool swapped = rowBra == ket && rowKet == bra;

            if (!(rowBra == bra && rowKet == ket) && !swapped)
            {
                continue;
            }

            // The CSV row/col are packed for (row.i,row.j,row.k,row.l);
            // the class-swapped computed form stores (cd|ab) at (col, row).
            // The CSV (a, b) are the block row/col indices (row = fj*nI+fi,
            // col = fl*nK+fk); the swapped rows live in the (cd|ab) block
            // whose column count is the bra-pair count nI*nJ - the row-major
            // stride of the swapped cell [a, b] is therefore (nK * nL), the
            // canonical KET pair count (2026-08-19: an earlier version used
            // nI*nJ as the stride and read the transposed cell).
            const double value = swapped ? batch->values[offset + row.b * (nK * nL) + row.a]
                                         : batch->values[offset + row.a * (nK * nL) + row.b];
            EXPECT_NEAR(value, row.value, 1e-12)
                << "quartet (" << row.i << "," << row.j << "|" << row.k << "," << row.l << ") ["
                << row.a << "," << row.b << "]";
        }

        offset += nI * nJ * nK * nL;
    }
}

TEST(EriBatchTest, HighLSpotMatchesMpmath) {
    // The high-l spot grid (tools/gen_md_reference.py): shells f/h on H at
    // the origin and g/i on He at (1.4, 0, 0), single primitives, spherical
    // only. The grid uses the f/g/h shells only, so the classes it pins are
    // (6,6), (7,7), (8,8), (9,9) and (10,10); the i shell is in the basis but
    // in no grid quartet, and totals 11 and 12 have no independent reference
    // (FullClassMatrixCoversTheSupportedBound exercises them without one).
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(6))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below 6";
    }

    static constexpr std::string_view kHighLBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    F
      6.0000000000E-01       1.0000000000E+00
H    H
      4.0000000000E-01       1.0000000000E+00
He    G
      5.0000000000E-01       1.0000000000E+00
He    I
      3.0000000000E-01       1.0000000000E+00
END
)";

    auto basis = qcx::basisset::ParseNwchemText(kHighLBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<GridRow> grid = LoadGrid("md_high_l_reference.csv");
    ASSERT_GT(grid.size(), 0u);

    std::vector<qcx::integrals::ShellQuartet> quartets;
    std::vector<std::pair<std::size_t, std::size_t>> keys; // {min pair, max pair}

    for (const GridRow& row : grid)
    {
        const std::size_t bra = qcx::integrals::PairIndexOf(row.i, row.j, *pairList);
        const std::size_t ket = qcx::integrals::PairIndexOf(row.k, row.l, *pairList);

        if (std::find(keys.begin(), keys.end(), std::make_pair(bra, ket)) == keys.end())
        {
            keys.push_back({bra, ket});
            quartets.push_back({row.i, row.j, row.k, row.l});
        }
    }

    auto batch = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;

    // The VRR conditioning at l = 12 costs digits; the molecule-grid
    // tolerance (1e-10) is the high-l budget.
    std::size_t offset = 0;

    for (const qcx::integrals::ShellQuartet& quartet : batch->computed)
    {
        const std::size_t bra = qcx::integrals::PairIndexOf(quartet.i, quartet.j, *pairList);
        const std::size_t ket = qcx::integrals::PairIndexOf(quartet.k, quartet.l, *pairList);
        const std::size_t nI = ShellFunctionCount(pairList->shells[quartet.i]);
        const std::size_t nJ = ShellFunctionCount(pairList->shells[quartet.j]);
        const std::size_t nK = ShellFunctionCount(pairList->shells[quartet.k]);
        const std::size_t nL = ShellFunctionCount(pairList->shells[quartet.l]);

        for (const GridRow& row : grid)
        {
            const std::size_t rowBra = qcx::integrals::PairIndexOf(row.i, row.j, *pairList);
            const std::size_t rowKet = qcx::integrals::PairIndexOf(row.k, row.l, *pairList);
            const bool swapped = rowBra == ket && rowKet == bra;

            if (!(rowBra == bra && rowKet == ket) && !swapped)
            {
                continue;
            }

            // The CSV (a, b) are the block row/col indices (row = fj*nI+fi,
            // col = fl*nK+fk); the swapped rows live in the (cd|ab) block
            // whose column count is the bra-pair count nI*nJ - the row-major
            // stride of the swapped cell [a, b] is therefore (nK * nL), the
            // canonical KET pair count (2026-08-19: an earlier version used
            // nI*nJ as the stride and read the transposed cell).
            const double value = swapped ? batch->values[offset + row.b * (nK * nL) + row.a]
                                         : batch->values[offset + row.a * (nK * nL) + row.b];
            EXPECT_NEAR(value, row.value, 1e-10)
                << "quartet (" << row.i << "," << row.j << "|" << row.k << "," << row.l << ") ["
                << row.a << "," << row.b << "]";
        }

        offset += nI * nJ * nK * nL;
    }
}

TEST(EriBatchTest, IChainClass66ProducesFiniteValues) {
    // The minimal l = 6 smoke: class (6,6) is the first whose transforms
    // exceed the 16-wide micro kernel and reach the vendor seam - the gap
    // the class-(6,6) benchmark crash exposed (2026-08-18). No reference
    // values here (the mpmath high-l grid stays trimmed by design); this
    // test guards the dispatch and the transform seam.
    if (!qcx::integrals::SupportsL(6))
    {
        GTEST_SKIP() << "this build's Lmax is below the i shell";
    }

    static constexpr std::string_view kIChain = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    I
      5.0000000000E-01       1.0000000000E+00
END
)";
    auto basis = qcx::basisset::ParseNwchemText(kIChain);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({6, 3});
    ASSERT_TRUE(coordinates.has_value()) << coordinates.error().message;
    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t i = 0; i < 6; ++i)
    {
        (*coordinates)(i, 0) = static_cast<double>(i) * 2.0;
        (*coordinates)(i, 1) = 0.0;
        (*coordinates)(i, 2) = 0.0;
        atoms.push_back({"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();
    auto molecule =
        qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    std::vector<qcx::integrals::ShellQuartet> quartets;

    for (std::size_t atomA = 0; atomA < 6; ++atomA)
    {
        const std::size_t iShellA = 2 * atomA + 1;
        const std::size_t sShellA = 2 * atomA;

        for (std::size_t atomB = 0; atomB <= atomA; ++atomB)
        {
            const std::size_t iShellB = 2 * atomB + 1;
            const std::size_t sShellB = 2 * atomB;
            quartets.push_back({iShellA, sShellA, iShellB, sShellB});
        }
    }

    auto batch = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->computed.size(), quartets.size());

    for (const double value : batch->values)
    {
        EXPECT_TRUE(std::isfinite(value));
    }

    // A second run over the same quartets must reproduce the first
    // bit-for-bit: the class kernel keeps per-thread scratch (thread_local
    // vectors) and its ket accumulator must not carry state across calls
    // (the 2026-08-18 review flagged the accumulator as never zeroed).
    auto again = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(again.has_value()) << again.error().message;
    ASSERT_EQ(again->values.size(), batch->values.size());

    for (std::size_t i = 0; i < batch->values.size(); ++i)
    {
        EXPECT_EQ(again->values[i], batch->values[i]) << "element " << i;
    }
}

TEST(EriBatchTest, FullClassMatrixCoversTheSupportedBound) {
    // The shipped bound is l = 6: the 2e triangle covers every class
    // (Lbra <= Lket <= 12) and the 3c RI rectangle every class
    // (Lket <= 6, Lbra <= 12). The grid tests stop at (10,10), so totals 11
    // and 12 had no routine exercise at all. There is no independent
    // numerical reference for those totals; this pins the instantiation
    // matrix and the dispatch - the reachability a missing class fails -
    // and computes every triangle class once, checking the values are
    // finite. The 3c rectangle's tensor adds the metric solve and the
    // (u,v) symmetry. Release-only: the transform walks are long in Debug.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(6))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below 6";
    }

    // The instantiation matrix first: every class the capability needs has
    // a kernel at the shipped ceiling. 91 triangle classes plus 63 rectangle
    // classes with Lbra > Lket = 154, the shipped count.
    std::size_t classes = 0;

    for (int lBra = 0; lBra <= 12; ++lBra)
    {
        for (int lKet = 0; lKet <= 12; ++lKet)
        {
            const bool triangle = lBra <= lKet;
            const bool rectangle = lKet <= 6;

            if (!triangle && !rectangle)
            {
                continue;
            }

            const qcx::integrals::internal::MdClassFnF64* slot =
                qcx::integrals::internal::ClassSpecF64(lBra, lKet);
            ASSERT_NE(slot, nullptr)
                << "class (" << lBra << "," << lKet << ") is out of table range";

            // The slot itself is what carries the kernel: ClassSpecF64 returns
            // the address of the table entry whenever the indices are in range,
            // so a missing class is a null entry, not a null slot pointer.
            EXPECT_NE(*slot, nullptr) << "class (" << lBra << "," << lKet << ") has no kernel";
#if defined(QcxIntegralsF32) && (QcxIntegralsF32 == 1)
            // The certified fp32 lane is part of the shipped program, and its
            // table carries the same class matrix.
            const qcx::integrals::internal::MdClassFnF32* slot32 =
                qcx::integrals::internal::ClassSpecF32(lBra, lKet);
            ASSERT_NE(slot32, nullptr)
                << "class (" << lBra << "," << lKet << ") is out of table range";
            EXPECT_NE(*slot32, nullptr)
                << "class (" << lBra << "," << lKet << ") has no fp32 kernel";
#endif
            ++classes;
        }
    }

    EXPECT_EQ(classes, 154u);

    // s..i on both centers: single primitives with distinct exponents, spherical.
    static constexpr std::string_view kFullBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      2.0000000000E+00       1.0000000000E+00
H    P
      1.6000000000E+00       1.0000000000E+00
H    D
      1.3000000000E+00       1.0000000000E+00
H    F
      1.0000000000E+00       1.0000000000E+00
H    G
      8.0000000000E-01       1.0000000000E+00
H    H
      6.0000000000E-01       1.0000000000E+00
H    I
      4.5000000000E-01       1.0000000000E+00
He   S
      1.8000000000E+00       1.0000000000E+00
He   P
      1.4000000000E+00       1.0000000000E+00
He   D
      1.1000000000E+00       1.0000000000E+00
He   F
      9.0000000000E-01       1.0000000000E+00
He   G
      7.0000000000E-01       1.0000000000E+00
He   H
      5.5000000000E-01       1.0000000000E+00
He   I
      4.2000000000E-01       1.0000000000E+00
END
)";

    auto basis = qcx::basisset::ParseNwchemText(kFullBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    ASSERT_EQ(pairList->shells.size(), 14u);

    // H carries shells 0..6 (l = 0..6) and He 7..13, in input order.
    for (int l = 0; l <= 6; ++l)
    {
        EXPECT_EQ(pairList->shells[static_cast<std::size_t>(l)].angularMomentum, l);
        EXPECT_EQ(pairList->shells[static_cast<std::size_t>(7 + l)].angularMomentum, l);
    }

    // One shell of each l on H gives any bra or ket total 0..12: (T/2,
    // T - T/2), both halves at most 6.
    std::vector<qcx::integrals::ShellQuartet> quartets;
    std::vector<std::pair<int, int>> requested;

    for (int ketL = 0; ketL <= 12; ++ketL)
    {
        for (int braL = 0; braL <= ketL; ++braL)
        {
            const auto braLo = static_cast<std::size_t>(braL / 2);
            const auto braHi = static_cast<std::size_t>(braL - braL / 2);
            const auto ketLo = static_cast<std::size_t>(ketL / 2);
            const auto ketHi = static_cast<std::size_t>(ketL - ketL / 2);
            quartets.push_back({braLo, braHi, ketLo, ketHi});
            requested.emplace_back(braL, ketL);
        }
    }

    auto canonical = qcx::integrals::CanonicalizeQuartetOrder(*pairList, quartets);
    ASSERT_TRUE(canonical.has_value()) << canonical.error().message;
    ASSERT_EQ(canonical->size(), quartets.size());

    std::vector<std::pair<int, int>> covered;

    for (const qcx::integrals::CanonicalQuartetInfo& info : *canonical)
    {
        ASSERT_LE(info.lBra, info.lKet);
        covered.emplace_back(info.lBra, info.lKet);
    }

    std::sort(covered.begin(), covered.end());
    std::sort(requested.begin(), requested.end());
    EXPECT_EQ(covered, requested);
    EXPECT_EQ(covered.size(), 91u); // (13 * 14) / 2 classes, 0 <= bra <= ket <= 12.

    auto batch = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->computed.size(), quartets.size());

    std::size_t expectedValues = 0;

    for (const qcx::integrals::ShellQuartet& quartet : batch->computed)
    {
        expectedValues += qcx::integrals::ShellFunctionCount(pairList->shells[quartet.i]) *
                          qcx::integrals::ShellFunctionCount(pairList->shells[quartet.j]) *
                          qcx::integrals::ShellFunctionCount(pairList->shells[quartet.k]) *
                          qcx::integrals::ShellFunctionCount(pairList->shells[quartet.l]);
    }

    ASSERT_EQ(batch->values.size(), expectedValues);
    ASSERT_GT(batch->values.size(), 0u);

    for (const double value : batch->values)
    {
        EXPECT_TRUE(std::isfinite(value));
    }

    // The 3c rectangle: the same basis as the orbital and the auxiliary set
    // reaches every class (Lbra <= 12, Lket <= 6). The instantiation check
    // above covers the kernels; the tensor adds the metric solve and the
    // element checks below, and refuses by name if a class it needs is absent.
    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto riTensor = qcx::integrals::BuildRiTensor(*molecule, *basis, *basis, riOptions);
    ASSERT_TRUE(riTensor.has_value()) << riTensor.error().message;
    const auto shape = riTensor->Shape();
    ASSERT_EQ(shape[0], 98u); // 2 * (1 + 3 + 5 + 7 + 9 + 11 + 13)
    ASSERT_EQ(shape[1], 98u);
    ASSERT_EQ(shape[2], 98u);

    for (std::size_t p = 0; p < shape[2]; ++p)
    {
        for (std::size_t u = 0; u < shape[0]; ++u)
        {
            for (std::size_t v = u; v < shape[1]; ++v)
            {
                const double forward = (*riTensor)(u, v, p);
                EXPECT_TRUE(std::isfinite(forward));
                EXPECT_NEAR(
                    forward, (*riTensor)(v, u, p), 1e-12 * std::max(1.0, std::abs(forward)));
            }
        }
    }
}

TEST(EriBatchTest, UnfoldedPathMatchesTheEngine) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = qcx::basisset::ParseNwchemText(qcx::testing::kShellsetBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto pairStore = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value()) << pairStore.error().message;

    // All canonical quartets over the s/p shells (the d-shell pairs stay
    // out of the 2e grids).
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

    std::size_t offset = 0;

    for (const qcx::integrals::ShellQuartet& quartet : batch->computed)
    {
        const std::size_t braPair = qcx::integrals::PairIndexOf(quartet.i, quartet.j, *pairList);
        const std::size_t ketPair = qcx::integrals::PairIndexOf(quartet.k, quartet.l, *pairList);
        const std::vector<double> unfolded =
            qcx::integrals::test::UnfoldedEriBlock((*pairStore)[braPair], (*pairStore)[ketPair]);
        const std::size_t blockSize = unfolded.size();

        for (std::size_t element = 0; element < blockSize; ++element)
        {
            EXPECT_NEAR(batch->values[offset + element], unfolded[element], 1e-12)
                << "quartet (" << quartet.i << "," << quartet.j << "|" << quartet.k << ","
                << quartet.l << ") element " << element;
        }

        offset += blockSize;
    }
}

TEST(EriBatchTest, HfAndH2oMatchMpmath) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto hfBasis = qcx::testing::MakeHfSto3gBasis();
    ASSERT_TRUE(hfBasis.has_value()) << hfBasis.error().message;
    auto hfMolecule = qcx::testing::MakeHfSto3g();
    ASSERT_TRUE(hfMolecule.has_value()) << hfMolecule.error().message;
    auto h2oBasis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(h2oBasis.has_value()) << h2oBasis.error().message;
    auto h2oMolecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(h2oMolecule.has_value()) << h2oMolecule.error().message;

    const auto runGrid = [](const qcx::molecule::Molecule& molecule,
                            const qcx::basisset::BasisSet& basisSet,
                            const std::string& fileName) {
        auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);
        EXPECT_TRUE(pairList.has_value());

        if (!pairList.has_value())
        {
            return;
        }

        const std::vector<GridRow> grid = LoadGrid(fileName);
        ASSERT_GT(grid.size(), 0u);

        auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);
        ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
        auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);
        ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
        auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);
        ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

        for (const GridRow& row : grid)
        {
            if (row.kind == "ERI")
            {
                continue;
            }

            const std::size_t offI = pairList->shells[row.i].functionOffset;
            const std::size_t offJ = pairList->shells[row.j].functionOffset;
            const double value = row.kind == "S"   ? (*overlap)(offI + row.a, offJ + row.b)
                                 : row.kind == "T" ? (*kinetic)(offI + row.a, offJ + row.b)
                                                   : (*nuclear)(offI + row.a, offJ + row.b);
            EXPECT_NEAR(value, row.value, 1e-10)
                << row.kind << " pair (" << row.i << "," << row.j << ")";
        }

        std::vector<qcx::integrals::ShellQuartet> quartets;
        std::vector<std::pair<std::size_t, std::size_t>> keys;

        for (const GridRow& row : grid)
        {
            if (row.kind != "ERI")
            {
                continue;
            }

            const std::size_t bra = qcx::integrals::PairIndexOf(row.i, row.j, *pairList);
            const std::size_t ket = qcx::integrals::PairIndexOf(row.k, row.l, *pairList);

            if (std::find(keys.begin(), keys.end(), std::make_pair(bra, ket)) == keys.end())
            {
                keys.push_back({bra, ket});
                quartets.push_back({row.i, row.j, row.k, row.l});
            }
        }

        auto batch = qcx::integrals::ComputeEriBatch(molecule, basisSet, quartets);
        ASSERT_TRUE(batch.has_value()) << batch.error().message;

        std::size_t offset = 0;

        for (const qcx::integrals::ShellQuartet& quartet : batch->computed)
        {
            const std::size_t bra = qcx::integrals::PairIndexOf(quartet.i, quartet.j, *pairList);
            const std::size_t ket = qcx::integrals::PairIndexOf(quartet.k, quartet.l, *pairList);
            const std::size_t nI = ShellFunctionCount(pairList->shells[quartet.i]);
            const std::size_t nJ = ShellFunctionCount(pairList->shells[quartet.j]);
            const std::size_t nK = ShellFunctionCount(pairList->shells[quartet.k]);
            const std::size_t nL = ShellFunctionCount(pairList->shells[quartet.l]);

            for (const GridRow& row : grid)
            {
                if (row.kind != "ERI")
                {
                    continue;
                }

                const std::size_t rowBra = qcx::integrals::PairIndexOf(row.i, row.j, *pairList);
                const std::size_t rowKet = qcx::integrals::PairIndexOf(row.k, row.l, *pairList);
                const bool swapped = rowBra == ket && rowKet == bra;

                if (!(rowBra == bra && rowKet == ket) && !swapped)
                {
                    continue;
                }

                // The class-swapped computed form stores (cd|ab) at
                // (col, row): the CSV cell [a, b] lives at canonical
                // [b, a] with the row-major stride nK*nL (the canonical
                // ket-pair count) - 2026-08-19: an earlier version used
                // the bra-pair count nI*nJ as the stride and read the
                // transposed cell.
                const double value = swapped ? batch->values[offset + row.b * (nK * nL) + row.a]
                                             : batch->values[offset + row.a * (nK * nL) + row.b];
                EXPECT_NEAR(value, row.value, 1e-10)
                    << "quartet (" << row.i << "," << row.j << "|" << row.k << "," << row.l << ")";
            }

            offset += nI * nJ * nK * nL;
        }
    };

    runGrid(*hfMolecule, *hfBasis, "hf_sto3g_reference.csv");
    runGrid(*h2oMolecule, *h2oBasis, "h2o_sto3g_reference.csv");
}

// ---------------------------------------------------------------------------
// Parallel-schedule pins
// ---------------------------------------------------------------------------

// The parallel-schedule contract: the
// batch loop is chunked over the OpenMP team with a fixed static per-class
// chunk assignment. Every batch writes only its own disjoint output block
// (the caller's per-batch base pointers) and its own global certified bound slots
// (boundsBase + t, one write per task), so the run has NO cross-batch
// accumulation - the 1-vs-4-thread pin below judges bit-identity, not
// tolerance: a different team size must not move a single bit of the values
// or of the certified lane's bounds.
TEST(EriBatchTest, BatchRunIsBitIdenticalAcrossTeamSizes) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "parallel-schedule pin - fast smoke subset";
    }

    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    // The full canonical quartet set of the fixture (the certified-bound sweep's
    // enumeration), capped into many small batches so the chunked run has
    // real work to spread across the team.
    std::vector<qcx::integrals::ShellQuartet> quartets;
    const std::size_t nShells = pairList->shells.size();

    for (std::size_t a = 0; a < nShells; ++a)
    {
        for (std::size_t b = a; b < nShells; ++b)
        {
            for (std::size_t c = 0; c < nShells; ++c)
            {
                for (std::size_t d = c; d < nShells; ++d)
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

    // The process-wide team ceiling is sticky - restore the pre-test value
    // on every exit path (a failed pin must not leak a capped team into the
    // rest of the suite).
    struct CeilingGuard {
        int restoreValue;

        explicit CeilingGuard(int restore) : restoreValue(restore) {}

        // RAII guard: a copy or move would restore the ceiling twice.
        CeilingGuard(const CeilingGuard&) = delete;
        CeilingGuard& operator=(const CeilingGuard&) = delete;
        CeilingGuard(CeilingGuard&&) = delete;
        CeilingGuard& operator=(CeilingGuard&&) = delete;

        ~CeilingGuard() {
            qcx::backend::SetOmpThreadCeiling(restoreValue);
        }
    };

    const CeilingGuard guard(qcx::backend::OmpThreadCeiling());

    // The small cap forces many batches per class run - the shape that
    // exercises the static chunking across the whole team.
    qcx::integrals::EriBatchOptions options;
    options.maxBatchBytes = std::size_t{4} * 1024;

    qcx::backend::SetOmpThreadCeiling(1);
    auto single64 = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets, options);
    ASSERT_TRUE(single64.has_value()) << single64.error().message;
    auto singleCertified =
        qcx::integrals::ComputeEriBatchCertified(*molecule, *basis, quartets, options);
    ASSERT_TRUE(singleCertified.has_value()) << singleCertified.error().message;

    qcx::backend::SetOmpThreadCeiling(4);
    auto multi64 = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets, options);
    ASSERT_TRUE(multi64.has_value()) << multi64.error().message;
    auto multiCertified =
        qcx::integrals::ComputeEriBatchCertified(*molecule, *basis, quartets, options);
    ASSERT_TRUE(multiCertified.has_value()) << multiCertified.error().message;

    // The computed forms, the packed fp64 values, the certified fp32 values
    // and the per-quartet certified bounds must all be bit-identical across the
    // two team sizes.
    ASSERT_EQ(multi64->computed.size(), single64->computed.size());
    ASSERT_EQ(multi64->values.size(), single64->values.size());

    for (std::size_t t = 0; t < single64->computed.size(); ++t)
    {
        const qcx::integrals::ShellQuartet& expected = single64->computed[t];
        const qcx::integrals::ShellQuartet& actual = multi64->computed[t];
        EXPECT_EQ(actual.i, expected.i) << "quartet " << t;
        EXPECT_EQ(actual.j, expected.j) << "quartet " << t;
        EXPECT_EQ(actual.k, expected.k) << "quartet " << t;
        EXPECT_EQ(actual.l, expected.l) << "quartet " << t;
    }

    for (std::size_t i = 0; i < single64->values.size(); ++i)
    {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(multi64->values[i]),
                  std::bit_cast<std::uint64_t>(single64->values[i]))
            << "fp64 element " << i;
    }

    ASSERT_EQ(multiCertified->computed.size(), singleCertified->computed.size());
    ASSERT_EQ(multiCertified->values.size(), singleCertified->values.size());
    ASSERT_EQ(multiCertified->errorBounds.size(), singleCertified->errorBounds.size());

    for (std::size_t i = 0; i < singleCertified->values.size(); ++i)
    {
        EXPECT_EQ(std::bit_cast<std::uint32_t>(multiCertified->values[i]),
                  std::bit_cast<std::uint32_t>(singleCertified->values[i]))
            << "fp32 element " << i;
    }

    for (std::size_t i = 0; i < singleCertified->errorBounds.size(); ++i)
    {
        EXPECT_EQ(std::bit_cast<std::uint64_t>(multiCertified->errorBounds[i]),
                  std::bit_cast<std::uint64_t>(singleCertified->errorBounds[i]))
            << "certified bound " << i;
    }
}
