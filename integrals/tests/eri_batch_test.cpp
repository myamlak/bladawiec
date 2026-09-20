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
#include "internal/md_one_electron.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "shellset_fixture.hpp"
#include "unfolded_reference.hpp"

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
    // The high-l spot grid (tools/gen_md_reference.py): the l = 3..6
    // classes - f/h on H at the origin, g/i on He at (1.4, 0, 0), single
    // primitives, spherical only - covering the (6,6) i-shell class at a
    // fixed pattern of function indices per canonical quartet.
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
