// The Create-time screened pre-warm of a full-ERI store
// (screened_eri_store.hpp): the admission rule and its monotonicity in the
// preset, the dense-bound denominator, the payload the plan predicts against
// what the fill writes, the class-run cut (and its byte cap), and the pin the
// whole disk tier rests on - a load equals a recompute, bit for bit.
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/storage/eri_store.hpp"
#include "qcx/storage/screened_eri_store.hpp"
#include "temp_store.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <tuple>
#include <vector>

namespace {

using qcx::integrals::CanonicalQuartetInfo;
using qcx::integrals::ComputeEriBatch;
using qcx::integrals::EriBatchOptions;
using qcx::integrals::ShellQuartet;
using qcx::storage::EriStore;
using qcx::storage::FillScreenedEriStore;
using qcx::storage::PlanScreenedEriStore;
using qcx::storage::ScreenedEriStorePlan;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ScopedTempStoreFile;
using Preset = qcx::integrals::AccuracyPreset;

// The canonical sort key, exactly the fields CanonicalizeQuartetOrder orders
// by - so two plans' quartet lists compare as ordered sequences and a
// threshold-only difference reads as a subsequence.
using OrderKey = std::tuple<int, int, std::size_t, std::size_t, std::size_t>;

OrderKey KeyOf(const CanonicalQuartetInfo& info) {
    return {info.lKet, info.lBra, info.ketPair, info.braRowPairs, info.braPair};
}

std::vector<OrderKey> KeysOf(const ScreenedEriStorePlan& plan) {
    std::vector<OrderKey> keys;
    keys.reserve(plan.quartets.size());

    for (const CanonicalQuartetInfo& info : plan.quartets)
    {
        keys.push_back(KeyOf(info));
    }

    return keys;
}

std::vector<ShellQuartet> QuartetsOf(const ScreenedEriStorePlan& plan) {
    std::vector<ShellQuartet> quartets;
    quartets.reserve(plan.quartets.size());

    for (const CanonicalQuartetInfo& info : plan.quartets)
    {
        quartets.push_back(info.quartet);
    }

    return quartets;
}

// Plans H2O/STO-3G and asserts the plan succeeded; the fixture is shared, so
// the local helper keeps each test's body to its own claim.
ScreenedEriStorePlan PlannedH2oSto3g(Preset preset) {
    auto molecule = MakeH2oSto3g();
    auto basis = MakeH2oSto3gBasis();

    EXPECT_TRUE(molecule.has_value());
    EXPECT_TRUE(basis.has_value());

    auto plan = PlanScreenedEriStore(*molecule, *basis, preset);

    EXPECT_TRUE(plan.has_value());

    return *plan;
}

TEST(ScreenedEriStoreTest, TheDenseBoundIsTheUnreducedFourthPower) {
    // STO-3G water is 7 functions: 2 (O 1s, O 2s) + 3 (O 2p) + 2 x 1 (H 1s).
    const ScreenedEriStorePlan plan = PlannedH2oSto3g(Preset::kNormal);

    EXPECT_EQ(plan.denseBytes, 8 * 7 * 7 * 7 * 7);
    EXPECT_GT(plan.quartets.size(), 0u);

    // The pre-warm exists to stay far below the bound it reports.
    EXPECT_LT(plan.payloadBytes, plan.denseBytes);
}

TEST(ScreenedEriStoreTest, TheAdmissionIsMonotoneInThePreset) {
    // SchwarzThreshold is 1e-8 / 1e-10 / 1e-12 for kLoose / kNormal / kTight,
    // so a tighter preset can only admit more, and a threshold-only
    // difference leaves the canonical order intact: the looser plan's
    // quartets are a SUBSEQUENCE of the tighter one's.
    const ScreenedEriStorePlan loose = PlannedH2oSto3g(Preset::kLoose);
    const ScreenedEriStorePlan normal = PlannedH2oSto3g(Preset::kNormal);
    const ScreenedEriStorePlan tight = PlannedH2oSto3g(Preset::kTight);

    EXPECT_LE(loose.quartets.size(), normal.quartets.size());
    EXPECT_LE(normal.quartets.size(), tight.quartets.size());

    const std::vector<OrderKey> looseKeys = KeysOf(loose);
    const std::vector<OrderKey> normalKeys = KeysOf(normal);
    const std::vector<OrderKey> tightKeys = KeysOf(tight);

    EXPECT_TRUE(
        std::includes(normalKeys.begin(), normalKeys.end(), looseKeys.begin(), looseKeys.end()));
    EXPECT_TRUE(
        std::includes(tightKeys.begin(), tightKeys.end(), normalKeys.begin(), normalKeys.end()));
}

TEST(ScreenedEriStoreTest, ThePayloadIsWhatTheFillWrites) {
    ScopedTempStoreFile file("qcx-screened-eri-payload");
    auto molecule = MakeH2oSto3g();
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    const ScreenedEriStorePlan plan = PlannedH2oSto3g(Preset::kNormal);
    ASSERT_GT(plan.quartets.size(), 0u);
    ASSERT_GT(plan.payloadBytes, 0u);

    auto written =
        FillScreenedEriStore(file.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal);
    ASSERT_TRUE(written.has_value());

    // The plan's model IS the write: the admission arithmetic is the number
    // of bytes the fill appends, not a separate estimate.
    EXPECT_EQ(*written, plan.payloadBytes);

    auto store = EriStore::Open(file.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(store.has_value());
    EXPECT_EQ(store->StoredQuartetCount(), plan.quartets.size());
}

TEST(ScreenedEriStoreTest, TheLoadEqualsTheRecomputeBitForBit) {
    // The property the disk tier's value-preservation rests on (the
    // decorator's own contract): stored bytes are the engine's verbatim, so
    // a reload of the pre-warm reproduces a fresh evaluation exactly.
    ScopedTempStoreFile file("qcx-screened-eri-bitexact");
    auto molecule = MakeH2oSto3g();
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    const ScreenedEriStorePlan plan = PlannedH2oSto3g(Preset::kNormal);
    ASSERT_GT(plan.quartets.size(), 0u);

    auto written =
        FillScreenedEriStore(file.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal);
    ASSERT_TRUE(written.has_value());

    auto store = EriStore::Open(file.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(store.has_value());

    auto loaded = store->LoadBatch(plan.quartets);
    ASSERT_TRUE(loaded.has_value());

    auto recomputed = ComputeEriBatch(*molecule, *basis, QuartetsOf(plan));
    ASSERT_TRUE(recomputed.has_value());

    ASSERT_EQ(loaded->values.size(), recomputed->values.size());
    ASSERT_GT(loaded->values.size(), 0u);
    EXPECT_EQ(std::memcmp(loaded->values.data(),
                          recomputed->values.data(),
                          loaded->values.size() * sizeof(double)),
              0);
}

TEST(ScreenedEriStoreTest, TheFillCutsOneRunPerClass) {
    // STO-3G water spans several (lBra, lKet) classes, and AppendBatch
    // requires one class per call - so a fill that cut the runs wrongly would
    // be refused by the store rather than silently mixed. More than one
    // stored chunk IS the observable that the cut happened.
    ScopedTempStoreFile file("qcx-screened-eri-classruns");
    auto molecule = MakeH2oSto3g();
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    const ScreenedEriStorePlan plan = PlannedH2oSto3g(Preset::kNormal);

    auto written =
        FillScreenedEriStore(file.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal);
    ASSERT_TRUE(written.has_value());

    auto store = EriStore::Open(file.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(store.has_value());
    EXPECT_GT(store->StoredChunkCount(), 1u);
}

TEST(ScreenedEriStoreTest, TheByteCapCutsTheRunsAndLosesNothing) {
    // The run cut is a byte budget, not a value filter: shrinking it divides
    // the same quartets among more chunks and stores the same payload.
    ScopedTempStoreFile coarseFile("qcx-screened-eri-cap-coarse");
    ScopedTempStoreFile fineFile("qcx-screened-eri-cap-fine");
    auto molecule = MakeH2oSto3g();
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    const ScreenedEriStorePlan plan = PlannedH2oSto3g(Preset::kNormal);
    ASSERT_GT(plan.quartets.size(), 0u);

    // The unsigned-int product is 64 MiB: in range, the widening is the assignment's.
    // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result)
    constexpr std::size_t kCoarseCap = 64u * 1024u * 1024u;
    // One block is 8 bytes per element, so a cap of 8 bytes admits exactly
    // one quartet per run: the cut is at its finest and the run count is
    // pinned to the quartet count rather than to this fixture's class
    // sizes. (The measured payload of this fixture is small enough that a
    // four-figure cap never cut at all - the fine cap is deliberately below
    // a single block, not below a class.)

    constexpr std::size_t kFineCap = 8u;

    auto coarse = FillScreenedEriStore(
        coarseFile.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal, kCoarseCap);
    ASSERT_TRUE(coarse.has_value());
    auto fine = FillScreenedEriStore(
        fineFile.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal, kFineCap);
    ASSERT_TRUE(fine.has_value());

    EXPECT_EQ(*coarse, *fine);
    EXPECT_EQ(*coarse, plan.payloadBytes);

    auto coarseStore = EriStore::Open(coarseFile.Path(), *molecule, *basis, "sto-3g", "");
    auto fineStore = EriStore::Open(fineFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(coarseStore.has_value());
    ASSERT_TRUE(fineStore.has_value());

    // Every block is at least 8 bytes, so a cap of 8 bytes closes each run
    // after its first quartet: the fine fill's chunk count IS the quartet
    // count, exactly. (A lone over-cap block may exceed the cap - the guard
    // that keeps a run non-empty - so no arithmetic floor over the payload
    // would be sound here; this identity is the sound one.)
    EXPECT_EQ(fineStore->StoredChunkCount(), plan.quartets.size());
    EXPECT_GT(fineStore->StoredChunkCount(), coarseStore->StoredChunkCount());
    EXPECT_EQ(fineStore->StoredQuartetCount(), plan.quartets.size());
}

TEST(ScreenedEriStoreTest, TheFillOwnsNoStalePath) {
    // The freshness contract is the caller's: Create refuses an existing file
    // (kIOError) and the fill does not delete one behind the caller's back.
    ScopedTempStoreFile file("qcx-screened-eri-freshness");
    auto molecule = MakeH2oSto3g();
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    const ScreenedEriStorePlan plan = PlannedH2oSto3g(Preset::kNormal);

    auto first =
        FillScreenedEriStore(file.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal);
    ASSERT_TRUE(first.has_value());

    auto second =
        FillScreenedEriStore(file.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal);
    EXPECT_FALSE(second.has_value());
}

TEST(ScreenedEriStoreTest, AZeroByteCapIsRefused) {
    ScopedTempStoreFile file("qcx-screened-eri-zerocap");
    auto molecule = MakeH2oSto3g();
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    const ScreenedEriStorePlan plan = PlannedH2oSto3g(Preset::kNormal);

    auto refused =
        FillScreenedEriStore(file.Path(), *molecule, *basis, "sto-3g", plan, Preset::kNormal, 0u);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
