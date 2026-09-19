// The gpu_split batch-partition primitives of internal/md_batch.hpp
// (BatchElementCount, EqualKetGroupStraddles, GpuSplitBatchIndex): the split
// index is deterministic and cost-based (the fp64 block element mass), lands
// at the nearest equal-ket group boundary (never inside a group), and
// degenerates to the empty suffix at the default zero device share (the
// CPU-only fallback) and the empty prefix at one. The fixture exercises real
// assembled batches (the byte-cap cuts of AssembleClassBatches land where they
// land); the hand-built case pins the exact boundary semantics.

#include "h2o_sto3g.hpp"
#include "internal/md_batch.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::ShellQuartet;
using qcx::integrals::internal::MdClassBatch;
using qcx::integrals::internal::MdPairData;
using qcx::integrals::internal::MdQuartetTask;

// A zeroed MdPrimPair: only its presence in primPairs (the prim count) is
// read by the partition primitives.
qcx::integrals::internal::MdPrimPair PrimPairPlaceholder() {
    return {};
}

// The element mass of one task, read straight off the pair store - the
// test-side accumulation the expectations are checked against.
std::size_t TaskMass(const std::vector<MdPairData>& store, const MdQuartetTask& task) {
    return store[task.braPair].nFuncs * store[task.ketPair].nFuncs;
}

// The batch's element mass by raw task accumulation (the oracle route -
// never through BatchElementCount itself).
std::size_t BatchMassFromTasks(const std::vector<MdPairData>& store, const MdClassBatch& batch) {
    std::size_t total = 0;

    for (const MdQuartetTask& task : batch.tasks)
    {
        total += TaskMass(store, task);
    }

    return total;
}

// The maximal equal-ket run walk over the flattened canonical task stream
// (the kernel's group rule of md_vrr.hpp): true at the interior boundary
// between \p before and \p after when the boundary tasks - the last of
// the earlier batch and the first of the next - share the class and the
// group key (ket pair, bra row-pair count, bra prim-pair count).
bool BoundaryTasksShareAGroup(const std::vector<MdPairData>& store,
                              const MdClassBatch& before,
                              const MdClassBatch& after) {
    if (before.lBra != after.lBra || before.lKet != after.lKet)
    {
        return false;
    }

    if (before.tasks.empty() || after.tasks.empty())
    {
        return false;
    }

    const MdQuartetTask& last = before.tasks.back();
    const MdQuartetTask& first = after.tasks.front();

    if (last.ketPair != first.ketPair)
    {
        return false;
    }

    const MdPairData& lastBra = store[last.braPair];
    const MdPairData& firstBra = store[first.braPair];
    return lastBra.rowPairs == firstBra.rowPairs &&
           lastBra.primPairs.size() == firstBra.primPairs.size();
}

// The assembled fixture: the pair store and the class batches at one byte
// cap. Filled in place (never returned by value - the batches carry raw
// pointers into the pair store member, so any move of the struct would
// leave them dangling at the source's dead storage).
struct AssembledBatches {
    std::vector<MdPairData> pairStore;
    std::vector<MdClassBatch> batches;
};

// Counts the straddling interior boundaries of one assembly, so the tests
// can assert the fixture actually exercises straddles (a sweep over a
// straddle-free vector would not bite).
std::size_t StraddleCount(const AssembledBatches& assembled) {
    std::size_t straddles = 0;

    for (std::size_t i = 1; i < assembled.batches.size(); ++i)
    {
        if (BoundaryTasksShareAGroup(
                assembled.pairStore, assembled.batches[i - 1], assembled.batches[i]))
        {
            ++straddles;
        }
    }

    return straddles;
}

// Assembles the fixture's full canonical quartet set at the byte cap into
// \p out in place.
qcx::Result<void> AssembleH2o(const qcx::molecule::Molecule& molecule,
                              const qcx::basisset::BasisSet& basis,
                              std::size_t maxBatchBytes,
                              AssembledBatches& out) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basis);

    if (!pairList.has_value())
    {
        return std::unexpected(std::move(pairList.error()));
    }

    auto pairStore = qcx::integrals::internal::BuildPairData(molecule, basis, *pairList);

    if (!pairStore.has_value())
    {
        return std::unexpected(std::move(pairStore.error()));
    }

    out.pairStore = *pairStore;

    std::vector<ShellQuartet> quartets;

    for (const qcx::integrals::ShellPairIndex& bra : pairList->pairs)
    {
        for (const qcx::integrals::ShellPairIndex& ket : pairList->pairs)
        {
            quartets.push_back({bra.i, bra.j, ket.i, ket.j});
        }
    }

    std::vector<ShellQuartet> computed;
    auto batches = qcx::integrals::internal::AssembleClassBatches(
        out.pairStore, *pairList, quartets, maxBatchBytes, computed);

    if (!batches.has_value())
    {
        return std::unexpected(std::move(batches.error()));
    }

    out.batches = *batches;
    return {};
}

// ---------------------------------------------------------------------------
// The hand-built case: five pairs in one shared store, three class (0, 0)
// batches with known masses. Pair 0 (ket k0): nFuncs 1; pair 1 (the only
// two-function pair): nFuncs 2; pair 2 (ket k2): nFuncs 2; pairs 3 and 4
// (the bras b3 and b4): nFuncs 1 and identical group keys (rowPairs 1,
// one prim). Batches: b0 = [(k0,b3) mass 1, (k0,b1) mass 2] -> 3; b1 =
// [(k0,b4) mass 1, (k1,p0) mass 2] -> 3; b2 = [(k2,p0) mass 2] -> 2;
// total 8. Boundary 1 straddles (b3 and b4 share the k0 group key) and
// its prefix mass 3 sits CLOSEST to the half-share target 4 - excluded
// anyway; boundaries 0, 2 and 3 are clean.
struct HandBuiltFixture {
    std::vector<MdPairData> store;
    std::vector<MdClassBatch> batches;
};

void FillHandBuiltFixture(HandBuiltFixture& fixture) {
    fixture.store.resize(5);

    for (MdPairData& pair : fixture.store)
    {
        pair.rowPairs = 1;
        pair.primPairs.push_back(PrimPairPlaceholder());
    }

    fixture.store[0].nFuncs = 1; // Ket k0.
    fixture.store[1].nFuncs = 2; // The two-function pair (bra b1, ket k1).
    fixture.store[2].nFuncs = 2; // Ket k2.
    fixture.store[3].nFuncs = 1; // Bra b3.
    fixture.store[4].nFuncs = 1; // Bra b4.

    MdClassBatch b0;
    b0.lBra = 0;
    b0.lKet = 0;
    b0.pairStore = &fixture.store;
    b0.tasks = {{3, 0, 0}, {1, 0, 1}};
    fixture.batches.push_back(b0);

    MdClassBatch b1;
    b1.lBra = 0;
    b1.lKet = 0;
    b1.pairStore = &fixture.store;
    b1.tasks = {{4, 0, 0}, {0, 1, 1}};
    fixture.batches.push_back(b1);

    MdClassBatch b2;
    b2.lBra = 0;
    b2.lKet = 0;
    b2.pairStore = &fixture.store;
    b2.tasks = {{0, 2, 0}};
    fixture.batches.push_back(b2);
}

TEST(BatchPartitionTest, BatchElementCountSumsTheLayoutMass) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // Two different caps assemble the same canonical task multiset, so the
    // summed element mass must agree - the mass is layout-independent.
    AssembledBatches wide;
    auto wideResult = AssembleH2o(*molecule, *basis, 8192, wide);
    ASSERT_TRUE(wideResult.has_value()) << wideResult.error().message;
    AssembledBatches narrow;
    auto narrowResult = AssembleH2o(*molecule, *basis, 1024, narrow);
    ASSERT_TRUE(narrowResult.has_value()) << narrowResult.error().message;

    std::size_t wideTotal = 0;

    for (const MdClassBatch& batch : wide.batches)
    {
        wideTotal += qcx::integrals::internal::BatchElementCount(batch);
    }

    std::size_t narrowTotal = 0;

    for (const MdClassBatch& batch : narrow.batches)
    {
        narrowTotal += qcx::integrals::internal::BatchElementCount(batch);
    }

    EXPECT_GT(wideTotal, 0u);
    EXPECT_EQ(wideTotal, narrowTotal);

    // The per-task accumulation over the raw tasks agrees with the
    // per-batch sums.
    std::size_t taskTotal = 0;

    for (const MdClassBatch& batch : narrow.batches)
    {
        taskTotal += BatchMassFromTasks(narrow.pairStore, batch);
    }

    EXPECT_EQ(narrowTotal, taskTotal);
    EXPECT_GT(narrow.batches.size(), 1u);
}

TEST(BatchPartitionTest, StraddleMapMatchesTheBoundaryTaskWalk) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // The 1024-byte cap cuts the larger equal-ket runs of the fixture
    // (the split-index sweep below would not bite otherwise).
    AssembledBatches assembled;
    auto result = AssembleH2o(*molecule, *basis, 1024, assembled);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_GT(assembled.batches.size(), 1u);
    EXPECT_GT(StraddleCount(assembled), 0u);

    for (std::size_t i = 1; i < assembled.batches.size(); ++i)
    {
        const bool verdict = qcx::integrals::internal::EqualKetGroupStraddles(
            assembled.batches[i - 1], assembled.batches[i]);
        const bool oracle = BoundaryTasksShareAGroup(
            assembled.pairStore, assembled.batches[i - 1], assembled.batches[i]);
        EXPECT_EQ(verdict, oracle) << "boundary " << i << " straddle verdict mismatch";
    }
}

TEST(BatchPartitionTest, SplitIndexIsDeterministicAndNeverInsideAGroup) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    AssembledBatches assembled;
    auto result = AssembleH2o(*molecule, *basis, 1024, assembled);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // Identical inputs produce the identical index (the determinism pin).
    const double shares[] = {0.0, 0.125, 0.2, 0.5, 0.9, 1.0};

    for (const double share : shares)
    {
        const std::size_t first =
            qcx::integrals::internal::GpuSplitBatchIndex(assembled.batches, share);
        const std::size_t second =
            qcx::integrals::internal::GpuSplitBatchIndex(assembled.batches, share);
        EXPECT_EQ(first, second) << "share " << share;
    }

    // The sweep: the index never lands on an interior straddling boundary
    // (never inside a group), the degenerate shares pin the endpoints (the
    // empty suffix at zero = the CPU-only fallback; the empty prefix at
    // one), and the index is non-increasing in the share.
    std::size_t previous = assembled.batches.size() + 1;

    for (int step = 0; step <= 20; ++step)
    {
        const double share = static_cast<double>(step) / 20.0;
        const std::size_t index =
            qcx::integrals::internal::GpuSplitBatchIndex(assembled.batches, share);

        if (index > 0 && index < assembled.batches.size())
        {
            const bool straddles = qcx::integrals::internal::EqualKetGroupStraddles(
                assembled.batches[index - 1], assembled.batches[index]);
            EXPECT_FALSE(straddles)
                << "share " << share << " split index " << index << " cuts a group";
        }

        if (step == 0)
        {
            EXPECT_EQ(index, assembled.batches.size());
        }

        if (step == 20)
        {
            EXPECT_EQ(index, 0u);
        }

        EXPECT_LE(index, previous) << "share " << share << " must not raise the index";
        previous = index;
    }
}

TEST(BatchPartitionTest, SplitIndexIsTheNearestCleanBoundaryByCumulativeMass) {
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    AssembledBatches assembled;
    auto result = AssembleH2o(*molecule, *basis, 1024, assembled);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The oracle: prefix masses accumulated boundary by boundary over the
    // raw task stream, and the argmin over the CLEAN boundaries only,
    // earliest tie winning - independently of GpuSplitBatchIndex's scan.
    const std::size_t n = assembled.batches.size();
    std::vector<double> prefixAt(n + 1, 0.0);
    std::vector<bool> clean(n + 1, false);

    clean[0] = true;
    clean[n] = true;

    for (std::size_t i = 1; i < n; ++i)
    {
        clean[i] = !BoundaryTasksShareAGroup(
            assembled.pairStore, assembled.batches[i - 1], assembled.batches[i]);
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        prefixAt[i + 1] =
            prefixAt[i] +
            static_cast<double>(BatchMassFromTasks(assembled.pairStore, assembled.batches[i]));
    }

    const double shares[] = {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.95};

    for (const double share : shares)
    {
        const double target = (1.0 - share) * prefixAt[n];
        std::size_t expected = 0;
        double bestDistance = std::abs(prefixAt[0] - target);

        for (std::size_t i = 1; i <= n; ++i)
        {
            if (!clean[i])
            {
                continue;
            }

            const double distance = std::abs(prefixAt[i] - target);

            if (distance < bestDistance)
            {
                bestDistance = distance;
                expected = i;
            }
        }

        EXPECT_EQ(qcx::integrals::internal::GpuSplitBatchIndex(assembled.batches, share), expected)
            << "share " << share;
    }
}

TEST(BatchPartitionTest, HandBuiltCaseNeverSplitsInsideAGroup) {
    // The exact boundary semantics on known masses: boundary 1 straddles
    // and its prefix 3 is CLOSEST to the half-share target 4 - the split
    // must skip it for the clean boundary 2 (prefix 6) instead.
    HandBuiltFixture fixture;
    FillHandBuiltFixture(fixture);

    EXPECT_EQ(qcx::integrals::internal::BatchElementCount(fixture.batches[0]), 3u);
    EXPECT_EQ(qcx::integrals::internal::BatchElementCount(fixture.batches[1]), 3u);
    EXPECT_EQ(qcx::integrals::internal::BatchElementCount(fixture.batches[2]), 2u);

    EXPECT_TRUE(
        qcx::integrals::internal::EqualKetGroupStraddles(fixture.batches[0], fixture.batches[1]));
    EXPECT_FALSE(
        qcx::integrals::internal::EqualKetGroupStraddles(fixture.batches[1], fixture.batches[2]));

    // A class-mismatched and an empty-batch boundary never straddle.
    MdClassBatch otherClass;
    otherClass.lBra = 1;
    otherClass.lKet = 0;
    otherClass.pairStore = &fixture.store;
    EXPECT_FALSE(qcx::integrals::internal::EqualKetGroupStraddles(fixture.batches[0], otherClass));

    MdClassBatch empty;
    empty.lBra = 0;
    empty.lKet = 0;
    empty.pairStore = &fixture.store;
    EXPECT_FALSE(qcx::integrals::internal::EqualKetGroupStraddles(fixture.batches[0], empty));

    // share 0.5: target 4; clean candidates 0 (0), 2 (6), 3 (8) - the
    // straddling boundary 1 (prefix 3, distance 1) is never chosen; the
    // nearest clean boundary is 2 (distance 2).
    EXPECT_EQ(qcx::integrals::internal::GpuSplitBatchIndex(fixture.batches, 0.5), 2u);

    // share 0.625: target 3; boundaries 0 and 2 tie at distance 3 - the
    // earlier boundary wins.
    EXPECT_EQ(qcx::integrals::internal::GpuSplitBatchIndex(fixture.batches, 0.625), 0u);

    // share 0.375: target 5; boundary 2 (6, distance 1) wins over 0 and 3.
    EXPECT_EQ(qcx::integrals::internal::GpuSplitBatchIndex(fixture.batches, 0.375), 2u);

    // The degenerate shares: zero -> the empty suffix (the CPU-only
    // fallback), one -> the empty prefix.
    EXPECT_EQ(qcx::integrals::internal::GpuSplitBatchIndex(fixture.batches, 0.0), 3u);
    EXPECT_EQ(qcx::integrals::internal::GpuSplitBatchIndex(fixture.batches, 1.0), 0u);
}

} // namespace
