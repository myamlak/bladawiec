// The store-level contract: bit-identical one-electron and
// chunk round trips, the wrong-system rejection, corruption detection at
// open (store checksum) and on serve (per-chunk checksums), the schema
// derived-dataset slot contract, and the append-only refusals.
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/engine_version.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/storage/eri_store.hpp"
#include "qcx/storage/schema.hpp"
#include "temp_store.hpp"
#include "test_hooks.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include <string>
#include <vector>

namespace {

using qcx::integrals::BuildOverlapMatrix;
using qcx::integrals::BuildShellPairs;
using qcx::integrals::CanonicalizeQuartetOrder;
using qcx::integrals::ComputeEriBatch;
using qcx::integrals::EriBatch;
using qcx::integrals::ShellFunctionCount;
using qcx::integrals::ShellPairList;
using qcx::integrals::ShellQuartet;
using qcx::storage::EriStore;
using qcx::storage::StoreOptions;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ScopedTempStoreFile;

// The block element count of one canonical quartet (the store derives the
// same sizes from the pair list; mirrors the decorator's helper).
std::size_t BlockElementCount(const ShellPairList& pairList, const ShellQuartet& quartet) {
    return ShellFunctionCount(pairList.shells[quartet.i]) *
           ShellFunctionCount(pairList.shells[quartet.j]) *
           ShellFunctionCount(pairList.shells[quartet.k]) *
           ShellFunctionCount(pairList.shells[quartet.l]);
}

// The class of a canonical quartet.
std::pair<int, int> ClassOf(const ShellPairList& pairList, const ShellQuartet& quartet) {
    return {pairList.shells[quartet.i].angularMomentum + pairList.shells[quartet.j].angularMomentum,
            pairList.shells[quartet.k].angularMomentum +
                pairList.shells[quartet.l].angularMomentum};
}

// The first contiguous class run of the engine's full-batch output: the
// canonical quartets of the run and their packed values slice.
struct ClassRun {
    std::vector<ShellQuartet> quartets;
    std::vector<double> values;
};

ClassRun FirstClassRun(const ShellPairList& pairList, const EriBatch& batch) {
    const std::pair<int, int> firstClass = ClassOf(pairList, batch.computed.front());
    std::size_t runEnd = 1;

    while (runEnd < batch.computed.size() &&
           ClassOf(pairList, batch.computed[runEnd]) == firstClass)
    {
        ++runEnd;
    }

    std::size_t valueStart = 0;

    for (std::size_t i = 0; i < runEnd; ++i)
    {
        valueStart += BlockElementCount(pairList, batch.computed[i]);
    }

    return ClassRun{
        std::vector<ShellQuartet>(batch.computed.begin(),
                                  batch.computed.begin() + static_cast<long long>(runEnd)),
        std::vector<double>(batch.values.begin(),
                            batch.values.begin() + static_cast<long long>(valueStart))};
}

// The SECOND contiguous class run of the engine's full-batch output (the
// C1 regression needs two runs of different sizes, so a misplaced
// firstQuartet from a failed append cannot coincidentally align with the
// next run's rows). H2O/sto-3g has six classes (the s/p pair l-sums
// {0,1,2} x {0,1,2}), so a second run always exists for this fixture.
ClassRun SecondClassRun(const ShellPairList& pairList, const EriBatch& batch) {
    const std::pair<int, int> firstClass = ClassOf(pairList, batch.computed.front());
    std::size_t position = 1;
    std::size_t valueStart = BlockElementCount(pairList, batch.computed.front());

    while (position < batch.computed.size() &&
           ClassOf(pairList, batch.computed[position]) == firstClass)
    {
        valueStart += BlockElementCount(pairList, batch.computed[position]);
        ++position;
    }

    const std::pair<int, int> secondClass = ClassOf(pairList, batch.computed[position]);
    std::size_t runEnd = position;

    while (runEnd < batch.computed.size() &&
           ClassOf(pairList, batch.computed[runEnd]) == secondClass)
    {
        ++runEnd;
    }

    std::size_t valueEnd = valueStart;

    for (std::size_t i = position; i < runEnd; ++i)
    {
        valueEnd += BlockElementCount(pairList, batch.computed[i]);
    }

    return ClassRun{
        std::vector<ShellQuartet>(batch.computed.begin() + static_cast<long long>(position),
                                  batch.computed.begin() + static_cast<long long>(runEnd)),
        std::vector<double>(batch.values.begin() + static_cast<long long>(valueStart),
                            batch.values.begin() + static_cast<long long>(valueEnd))};
}

TEST(EriStoreTest, OneElectronRoundTripIsBitIdentical) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto overlap = BuildOverlapMatrix(*molecule, *basis);
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    ScopedTempStoreFile tempFile("qcx_one_electron");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
        auto written = store->WriteOneElectron(*overlap, *kinetic, *nuclear);
        ASSERT_TRUE(written.has_value()) << written.error().message;
    }

    auto reopened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;

    auto readOverlap = reopened->ReadOverlapMatrix();
    ASSERT_TRUE(readOverlap.has_value()) << readOverlap.error().message;
    auto readKinetic = reopened->ReadKineticMatrix();
    ASSERT_TRUE(readKinetic.has_value()) << readKinetic.error().message;
    auto readNuclear = reopened->ReadNuclearAttractionMatrix();
    ASSERT_TRUE(readNuclear.has_value()) << readNuclear.error().message;

    // HostView is mutable-only: the Results are deliberately non-const.
    const std::vector<double>& expectedOverlap = overlap->HostView();
    const std::vector<double>& expectedKinetic = kinetic->HostView();
    const std::vector<double>& expectedNuclear = nuclear->HostView();
    const std::vector<double>& actualOverlap = readOverlap->HostView();
    const std::vector<double>& actualKinetic = readKinetic->HostView();
    const std::vector<double>& actualNuclear = readNuclear->HostView();

    EXPECT_EQ(actualOverlap.size(), expectedOverlap.size());
    EXPECT_EQ(std::memcmp(actualOverlap.data(),
                          expectedOverlap.data(),
                          expectedOverlap.size() * sizeof(double)),
              0);
    EXPECT_EQ(std::memcmp(actualKinetic.data(),
                          expectedKinetic.data(),
                          expectedKinetic.size() * sizeof(double)),
              0);
    EXPECT_EQ(std::memcmp(actualNuclear.data(),
                          expectedNuclear.data(),
                          expectedNuclear.size() * sizeof(double)),
              0);
}

TEST(EriStoreTest, WrongSystemIsRejected) {
    const auto water = MakeH2oSto3g();
    ASSERT_TRUE(water.has_value()) << water.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    ScopedTempStoreFile tempFile("qcx_wrong_system");
    {
        auto store = EriStore::Create(tempFile.Path(), *water, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
    }

    const auto hydrogen = MakeH2Sto3g();
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;

    const auto opened = EriStore::Open(tempFile.Path(), *hydrogen, *basis, "sto-3g", "");
    ASSERT_FALSE(opened.has_value());
    EXPECT_EQ(opened.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(EriStoreTest, SchemaVersionMismatchIsRejected) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    ScopedTempStoreFile tempFile("qcx_schema_tamper");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
    }

    // The raw-HighFive tamper session (the store must be closed - the
    // Windows file-lock rule).
    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet schema = file.getDataSet("/metadata/schema_version");
        schema.write(std::uint32_t{qcx::storage::kStoreSchemaVersion + 1});
    }

    // The checksum covers the schema-version bytes: the tamper fails open.
    const auto opened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_FALSE(opened.has_value());
    EXPECT_EQ(opened.error().code, qcx::ErrorCode::kIOError);
}

TEST(EriStoreTest, DerivedSlotsExistUnfilled) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    ScopedTempStoreFile tempFile("qcx_derived_slots");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
    }

    HighFive::File file(tempFile.Path().string(), HighFive::File::ReadOnly);
    // The schema shapes pinned in eri_store.cpp
    // CreateDerivedSlots: the MO slots are rank-4 zero-extent, the AO
    // grid slot {0,2} - zero extent in the row dim is the unfilled marker.
    const std::vector<std::pair<std::string, std::vector<std::size_t>>> slots = {
        {"/derived/mo_integrals/iajb", {0, 0, 0, 0}},
        {"/derived/mo_integrals/abcd", {0, 0, 0, 0}},
        {"/derived/mo_integrals/kpqrs", {0, 0, 0, 0}},
        {"/derived/grid_properties/ao_values", {0, 2}},
    };

    for (const auto& [path, expectedDims] : slots)
    {
        HighFive::DataSet slot = file.getDataSet(path);
        const std::vector<std::size_t> dims = slot.getSpace().getDimensions();
        EXPECT_EQ(dims, expectedDims) << path;

        // The source-fingerprint contract: empty means unfilled.
        std::string sourceFingerprint = "sentinel";
        slot.getAttribute("source_fingerprint").read(sourceFingerprint);
        EXPECT_TRUE(sourceFingerprint.empty()) << path;

        std::uint32_t engineVersion = 0;
        slot.getAttribute("engine_version").read(engineVersion);
        EXPECT_EQ(engineVersion, qcx::integrals::kIntegralEngineVersion) << path;

        std::uint32_t schemaVersion = 0;
        slot.getAttribute("schema_version").read(schemaVersion);
        EXPECT_EQ(schemaVersion, qcx::storage::kStoreSchemaVersion) << path;
    }
}

TEST(EriStoreTest, WriteOneElectronTwiceIsRejected) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const auto overlap = BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    ScopedTempStoreFile tempFile("qcx_one_electron_twice");
    auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(store.has_value()) << store.error().message;

    auto first = store->WriteOneElectron(*overlap, *kinetic, *nuclear);
    ASSERT_TRUE(first.has_value()) << first.error().message;

    const auto second = store->WriteOneElectron(*overlap, *kinetic, *nuclear);
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, qcx::ErrorCode::kInvalidArgument);
}

// Create owns fresh files only: a second Create over the same path must
// refuse (kIOError) instead of truncating the existing store - the store
// is append-only and Open is the read-write entry point.
TEST(EriStoreTest, CreateRefusesToClobberAnExistingFile) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    ScopedTempStoreFile tempFile("qcx_create_clobber");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
    }

    auto second = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_FALSE(second.has_value());
    EXPECT_EQ(second.error().code, qcx::ErrorCode::kIOError);

    // The first store survives untouched and still opens.
    auto reopened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;
}

TEST(EriStoreTest, TruncatedFileIsRejected) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    const auto overlap = BuildOverlapMatrix(*molecule, *basis);
    const auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    const auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    ScopedTempStoreFile tempFile("qcx_truncated");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
        auto written = store->WriteOneElectron(*overlap, *kinetic, *nuclear);
        ASSERT_TRUE(written.has_value()) << written.error().message;
    }

    const std::uintmax_t size = std::filesystem::file_size(tempFile.Path());
    std::error_code resizeError;
    std::filesystem::resize_file(tempFile.Path(), size / 2, resizeError);
    ASSERT_FALSE(resizeError) << resizeError.message();

    // Half the file is gone: Open must fail (kIOError). If HDF5 still
    // opens the truncated store, the failure surfaces on the first read -
    // never a silent success.
    auto opened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");

    if (opened.has_value())
    {
        const auto read = opened->ReadOverlapMatrix();
        ASSERT_FALSE(read.has_value());
        EXPECT_EQ(read.error().code, qcx::ErrorCode::kIOError);
    } else
    {
        EXPECT_EQ(opened.error().code, qcx::ErrorCode::kIOError);
    }
}

TEST(EriStoreTest, AppendServeRoundTripIsBitIdentical) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<ShellQuartet> all;

    for (const auto& bra : pairList->pairs)
    {
        for (const auto& ket : pairList->pairs)
        {
            all.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
        }
    }

    auto batch = ComputeEriBatch(*molecule, *basis, all);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    const ClassRun run = FirstClassRun(*pairList, *batch);

    ScopedTempStoreFile tempFile("qcx_append_serve");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
        auto appended = store->AppendBatch(run.quartets, run.values);
        ASSERT_TRUE(appended.has_value()) << appended.error().message;
        EXPECT_EQ(store->StoredChunkCount(), std::size_t{1});
        EXPECT_EQ(store->StoredQuartetCount(), run.quartets.size());
    }

    auto reopened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;

    auto ordered = CanonicalizeQuartetOrder(*pairList, run.quartets);
    ASSERT_TRUE(ordered.has_value()) << ordered.error().message;
    auto served = reopened->LoadBatch(*ordered);
    ASSERT_TRUE(served.has_value()) << served.error().message;

    ASSERT_EQ(served->values.size(), run.values.size());
    EXPECT_EQ(
        std::memcmp(served->values.data(), run.values.data(), run.values.size() * sizeof(double)),
        0);
    EXPECT_EQ(served->computed, run.quartets);
}

// The C1 placement regression: a failure between the quartets and values
// writes leaves orphan rows on the on-disk tables. The placement of the
// NEXT append must come from the mirrors (which never saw the failed
// append) - deriving it from the on-disk tables instead would write a
// manifest row whose firstQuartet points past the mirrors, and every later
// load of that chunk would compute block offsets from the wrong quartets
// (and Open's checksum would cover different rows than the tables hold).
// Arm the injection for the failed append A, append B normally, and pin
// that B serves bit-identical values.
TEST(EriStoreTest, FailedAppendDoesNotCorruptTheNextPlacement) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<ShellQuartet> all;

    for (const auto& bra : pairList->pairs)
    {
        for (const auto& ket : pairList->pairs)
        {
            all.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
        }
    }

    auto batch = ComputeEriBatch(*molecule, *basis, all);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    const ClassRun failedRun = FirstClassRun(*pairList, *batch);
    const ClassRun goodRun = SecondClassRun(*pairList, *batch);
    // The two runs must differ in size: a same-sized second run would
    // overwrite the orphans in place and mask the placement regression.
    ASSERT_NE(failedRun.quartets.size(), goodRun.quartets.size());

    ScopedTempStoreFile tempFile("qcx_append_failure");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;

        // Append A fails between the quartets and values writes, leaving
        // its quartets as orphan rows on the tables; the mirrors never
        // commit (the injection flag is reset before any assertion, so a
        // regression that lets the append through cannot leak it).
        qcx::storage::internal::failAfterQuartetsWrite = true;
        auto failed = store->AppendBatch(failedRun.quartets, failedRun.values);
        qcx::storage::internal::failAfterQuartetsWrite = false;
        ASSERT_FALSE(failed.has_value());
        EXPECT_EQ(failed.error().code, qcx::ErrorCode::kIOError);
        EXPECT_EQ(store->StoredChunkCount(), std::size_t{0});

        // Append B: the mirror-derived placement must truncate the orphan
        // rows and land B's values exactly where its manifest row points.
        auto appended = store->AppendBatch(goodRun.quartets, goodRun.values);
        ASSERT_TRUE(appended.has_value()) << appended.error().message;
        EXPECT_EQ(store->StoredChunkCount(), std::size_t{1});
    }

    auto reopened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;

    auto ordered = CanonicalizeQuartetOrder(*pairList, goodRun.quartets);
    ASSERT_TRUE(ordered.has_value()) << ordered.error().message;
    auto served = reopened->LoadBatch(*ordered);
    ASSERT_TRUE(served.has_value()) << served.error().message;

    ASSERT_EQ(served->values.size(), goodRun.values.size());
    EXPECT_EQ(std::memcmp(served->values.data(),
                          goodRun.values.data(),
                          goodRun.values.size() * sizeof(double)),
              0);
    EXPECT_EQ(served->computed, goodRun.quartets);
}

// The class-canonical guard: a run in the MIRRORED class
// (L_bra > L_ket) is the value-identical bra/ket swap of a canonical run -
// the engine always emits the canonical class, so the store refuses the
// mirror instead of storing a manifest row whose class flips the run's.
TEST(EriStoreTest, MirroredClassRunIsRejected) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<ShellQuartet> all;

    for (const auto& bra : pairList->pairs)
    {
        for (const auto& ket : pairList->pairs)
        {
            all.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
        }
    }

    auto batch = ComputeEriBatch(*molecule, *basis, all);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    // The second class is (0, 1) for this fixture (the sort keys by lKet):
    // its mirror is (1, 0), L_bra > L_ket.
    const ClassRun canonical = SecondClassRun(*pairList, *batch);
    const ShellQuartet& head = canonical.quartets.front();
    const int lBra =
        pairList->shells[head.i].angularMomentum + pairList->shells[head.j].angularMomentum;
    const int lKet =
        pairList->shells[head.k].angularMomentum + pairList->shells[head.l].angularMomentum;
    ASSERT_LT(lBra, lKet);
    std::vector<ShellQuartet> mirrored;
    mirrored.reserve(canonical.quartets.size());

    for (const ShellQuartet& quartet : canonical.quartets)
    {
        mirrored.push_back(ShellQuartet{quartet.k, quartet.l, quartet.i, quartet.j});
    }

    ScopedTempStoreFile tempFile("qcx_mirrored_class");
    auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(store.has_value()) << store.error().message;

    auto appended = store->AppendBatch(mirrored, canonical.values);
    ASSERT_FALSE(appended.has_value());
    EXPECT_EQ(appended.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(store->StoredChunkCount(), std::size_t{0});
}

TEST(EriStoreTest, CorruptChunkIsRejectedOnServe) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<ShellQuartet> all;

    for (const auto& bra : pairList->pairs)
    {
        for (const auto& ket : pairList->pairs)
        {
            all.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
        }
    }

    auto batch = ComputeEriBatch(*molecule, *basis, all);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    const ClassRun run = FirstClassRun(*pairList, *batch);

    ScopedTempStoreFile tempFile("qcx_corrupt_chunk");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
        auto appended = store->AppendBatch(run.quartets, run.values);
        ASSERT_TRUE(appended.has_value()) << appended.error().message;
    }

    // Flip one bit of the chunk's first double, found through the manifest
    // (column 3 = value_offset_bytes, 4 = value_bytes).
    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet manifest = file.getDataSet("/integrals/ao/eri/fp64/manifest");
        std::vector<std::uint64_t> row(10);
        manifest.select({0, 0}, {1, 10}).read(row);
        ASSERT_GE(row[4], sizeof(double));

        HighFive::DataSet values = file.getDataSet("/integrals/ao/eri/fp64/values");
        const std::size_t element = row[3] / sizeof(double);
        double original = 0.0;
        values.select({element}, {1}).read(original);
        std::uint64_t bits = 0;
        std::memcpy(&bits, &original, sizeof(bits));
        bits ^= std::uint64_t{1};
        double flipped = 0.0;
        std::memcpy(&flipped, &bits, sizeof(flipped));
        values.select({element}, {1}).write(flipped);
    }

    // The values are outside the store checksum (per-chunk checksums cover
    // them on serve): Open succeeds, the serve must refuse.
    auto reopened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;

    auto ordered = CanonicalizeQuartetOrder(*pairList, run.quartets);
    ASSERT_TRUE(ordered.has_value()) << ordered.error().message;
    auto served = reopened->LoadBatch(*ordered);
    ASSERT_FALSE(served.has_value());
    EXPECT_EQ(served.error().code, qcx::ErrorCode::kIOError);

    // And a manifest tamper is caught at open (the store checksum covers
    // the manifest bytes).
    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet manifest = file.getDataSet("/integrals/ao/eri/fp64/manifest");
        std::vector<std::uint64_t> row(10);
        manifest.select({0, 0}, {1, 10}).read(row);
        row[8] ^= std::uint64_t{1}; // engine_version column
        manifest.select({0, 0}, {1, 10}).write(row);
    }

    const auto tampered = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_FALSE(tampered.has_value());
    EXPECT_EQ(tampered.error().code, qcx::ErrorCode::kIOError);
}

// Coverage gap: the certified bounds dataset was outside every
// checksum - the per-chunk checksums covered only the values chunk, and the
// store checksum covers quartets + manifests + scalars - so a flipped bound
// byte would silently under-report the certified error (or over-commit a
// budget). The fp32 manifest now carries a bounds_checksum column, verified
// on serve exactly like the values checksum.
TEST(EriStoreTest, CorruptCertifiedBoundsAreRejectedOnServe) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<ShellQuartet> all;

    for (const auto& bra : pairList->pairs)
    {
        for (const auto& ket : pairList->pairs)
        {
            all.push_back(ShellQuartet{bra.i, bra.j, ket.i, ket.j});
        }
    }

    auto batch = ComputeEriBatch(*molecule, *basis, all);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    const ClassRun run = FirstClassRun(*pairList, *batch);

    // Certified lane: fp32 values verbatim plus one bound per quartet (the
    // values are stored, not recomputed; arbitrary positive bounds are
    // fine here - the store validates sizes, not numbers).
    std::vector<float> values32(run.values.begin(), run.values.end());
    std::vector<double> bounds(run.quartets.size(), 1e-7);

    ScopedTempStoreFile tempFile("qcx_corrupt_bounds");
    {
        auto store = EriStore::Create(tempFile.Path(), *molecule, *basis, "sto-3g", "");
        ASSERT_TRUE(store.has_value()) << store.error().message;
        auto appended = store->AppendCertifiedBatch(run.quartets, values32, bounds);
        ASSERT_TRUE(appended.has_value()) << appended.error().message;
    }

    // Flip one bit of the chunk's first bound, found through the manifest
    // (column 10 = bounds_offset_bytes, 11 = bounds_bytes, 13 columns wide).
    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet manifest = file.getDataSet("/integrals/ao/eri/fp32/manifest");
        std::vector<std::uint64_t> row(13);
        manifest.select({0, 0}, {1, 13}).read(row);
        ASSERT_GE(row[11], sizeof(double));

        HighFive::DataSet boundsTable = file.getDataSet("/integrals/ao/eri/fp32/bounds");
        const std::size_t element = row[10] / sizeof(double);
        double original = 0.0;
        boundsTable.select({element}, {1}).read(original);
        std::uint64_t bits = 0;
        std::memcpy(&bits, &original, sizeof(bits));
        bits ^= std::uint64_t{1};
        double flipped = 0.0;
        std::memcpy(&flipped, &bits, sizeof(flipped));
        boundsTable.select({element}, {1}).write(flipped);
    }

    // The bounds are outside the store checksum (their own per-chunk
    // checksum covers them on serve): Open succeeds, the serve must refuse.
    auto reopened = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;

    auto ordered = CanonicalizeQuartetOrder(*pairList, run.quartets);
    ASSERT_TRUE(ordered.has_value()) << ordered.error().message;
    auto served = reopened->LoadCertifiedBatch(*ordered);
    ASSERT_FALSE(served.has_value());
    EXPECT_EQ(served.error().code, qcx::ErrorCode::kIOError);
    EXPECT_NE(served.error().message.find("bounds checksum mismatch"), std::string::npos);

    // And a bounds_checksum column tamper is caught at open (the manifest
    // bytes are inside the store checksum).
    {
        HighFive::File file(tempFile.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet manifest = file.getDataSet("/integrals/ao/eri/fp32/manifest");
        std::vector<std::uint64_t> row(13);
        manifest.select({0, 0}, {1, 13}).read(row);
        row[12] ^= std::uint64_t{1}; // bounds_checksum column
        manifest.select({0, 0}, {1, 13}).write(row);
    }

    const auto tampered = EriStore::Open(tempFile.Path(), *molecule, *basis, "sto-3g", "");
    ASSERT_FALSE(tampered.has_value());
    EXPECT_EQ(tampered.error().code, qcx::ErrorCode::kIOError);
}

} // namespace
