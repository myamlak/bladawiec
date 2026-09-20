// The cached batch-engine decorator:
// acceptance at memcmp level (reload equals recompute bit for bit), the
// ordering pin (computed is SORTED, not request-ordered), the
// partial-miss append semantics, the corruption policy (kIOError, never
// silent recompute), the certified fp32 lane with the budget
// re-check, and the empty-request pass-through.
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/storage/cached_eri_batch_engine.hpp"
#include "temp_store.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::BuildShellPairs;
using qcx::integrals::ComputeEriBatch;
using qcx::integrals::ComputeEriBatchCertified;
using qcx::integrals::ShellPairList;
using qcx::integrals::ShellQuartet;
using qcx::storage::CachedEriBatchEngine;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::TempStoreFile;

// The class of a canonical quartet.
std::pair<int, int> ClassOf(const ShellPairList& pairList, const ShellQuartet& quartet) {
    return {pairList.shells[quartet.i].angularMomentum + pairList.shells[quartet.j].angularMomentum,
            pairList.shells[quartet.k].angularMomentum +
                pairList.shells[quartet.l].angularMomentum};
}

// Every canonical quartet of the shell set (the 8-fold form plus the class
// canonical form L_bra <= L_ket, exactly as CanonicalizeQuartetOrder
// defines it - canonical requests are no-ops under canonicalization).
std::vector<ShellQuartet> AllCanonicalQuartets(const ShellPairList& pairList) {
    std::vector<ShellQuartet> quartets;

    for (std::size_t i = 0; i < pairList.shells.size(); ++i)
    {
        for (std::size_t j = i; j < pairList.shells.size(); ++j)
        {
            for (std::size_t k = 0; k < pairList.shells.size(); ++k)
            {
                for (std::size_t l = k; l < pairList.shells.size(); ++l)
                {
                    ShellQuartet candidate{i, j, k, l};

                    if (qcx::integrals::PairIndexOf(candidate.i, candidate.j, pairList) <
                        qcx::integrals::PairIndexOf(candidate.k, candidate.l, pairList))
                    {
                        continue;
                    }

                    const int lBra =
                        pairList.shells[i].angularMomentum + pairList.shells[j].angularMomentum;
                    const int lKet =
                        pairList.shells[k].angularMomentum + pairList.shells[l].angularMomentum;

                    if (lBra > lKet)
                    {
                        continue;
                    }

                    quartets.push_back(candidate);
                }
            }
        }
    }

    return quartets;
}

// The canonical quartets of one class.
std::vector<ShellQuartet> ClassQuartets(const ShellPairList& pairList, int lBra, int lKet) {
    std::vector<ShellQuartet> quartets;

    for (const ShellQuartet& quartet : AllCanonicalQuartets(pairList))
    {
        if (ClassOf(pairList, quartet) == std::make_pair(lBra, lKet))
        {
            quartets.push_back(quartet);
        }
    }

    return quartets;
}

TEST(CachedEriEngineTest, ReloadEqualsRecomputeBitForBit) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const std::vector<ShellQuartet> quartets = AllCanonicalQuartets(*pairList);
    ASSERT_GT(quartets.size(), std::size_t{1});

    int engineCalls = 0;
    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_reload_test.h5");
    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            ++engineCalls;
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const auto first = cached->ComputeEriBatch(quartets);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    EXPECT_EQ(engineCalls, 1);

    const auto second = cached->ComputeEriBatch(quartets);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    EXPECT_EQ(engineCalls, 1); // the second call was served from disk

    ASSERT_EQ(first->values.size(), second->values.size());
    EXPECT_EQ(std::memcmp(first->values.data(),
                          second->values.data(),
                          first->values.size() * sizeof(double)),
              0);
    EXPECT_EQ(first->computed, second->computed);
}

// The ordering-critical test: the engine's `computed` is SORTED, not
// request-ordered. A reversed request must still be served byte-identical
// to the engine's own output.
TEST(CachedEriEngineTest, PermutedRequestServesSameOrderAsEngine) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<ShellQuartet> quartets = AllCanonicalQuartets(*pairList);
    std::reverse(quartets.begin(), quartets.end());

    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_permuted_test.h5");
    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    // First call: miss, persists the class runs.
    const auto first = cached->ComputeEriBatch(quartets);
    ASSERT_TRUE(first.has_value()) << first.error().message;

    // Second call: full hit, served in the engine's canonical order.
    const auto served = cached->ComputeEriBatch(quartets);
    const auto reference = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(served.has_value()) << served.error().message;
    ASSERT_TRUE(reference.has_value()) << reference.error().message;
    EXPECT_EQ(served->computed, reference->computed);
    EXPECT_EQ(served->computed, first->computed);
    EXPECT_EQ(std::memcmp(served->values.data(),
                          reference->values.data(),
                          reference->values.size() * sizeof(double)),
              0);
}

TEST(CachedEriEngineTest, PartialMissRecomputesAndAppends) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    // Request A: the (0,0) class only. Request B: half of A's (0,0) class
    // plus the not-yet-stored (0,1) class - the partial miss. ((0,2)
    // cannot appear for the STO-3G H2O shell set: with the s shells at
    // {0,1,2} and the p shells at {3,4,5} every (s,s) pair index is below
    // every (p,p) pair index, so the canonical pair rule pair(i,j) >=
    // pair(k,l) leaves the class empty.)
    const std::vector<ShellQuartet> class00 = ClassQuartets(*pairList, 0, 0);
    const std::vector<ShellQuartet> class01 = ClassQuartets(*pairList, 0, 1);
    ASSERT_GT(class00.size(), std::size_t{1});
    ASSERT_GT(class01.size(), std::size_t{0});

    const std::vector<ShellQuartet>& requestA = class00;

    std::vector<ShellQuartet> requestB(
        class00.begin(), class00.begin() + static_cast<long long>(class00.size() / 2));
    requestB.insert(requestB.end(), class01.begin(), class01.end());

    int engineCalls = 0;
    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            ++engineCalls;
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_partial_miss.h5");
    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const auto firstA = cached->ComputeEriBatch(requestA);
    ASSERT_TRUE(firstA.has_value()) << firstA.error().message;
    EXPECT_EQ(engineCalls, 1);
    const std::size_t storedAfterA = firstA->computed.size();

    // B is a partial miss: one engine call, the store grows, and the
    // served B is bit-identical to the engine's B.
    const auto firstB = cached->ComputeEriBatch(requestB);
    ASSERT_TRUE(firstB.has_value()) << firstB.error().message;
    EXPECT_EQ(engineCalls, 2);

    const auto referenceB = qcx::integrals::ComputeEriBatch(*molecule, *basis, requestB);
    ASSERT_TRUE(referenceB.has_value()) << referenceB.error().message;
    EXPECT_EQ(firstB->computed, referenceB->computed);
    EXPECT_EQ(std::memcmp(firstB->values.data(),
                          referenceB->values.data(),
                          referenceB->values.size() * sizeof(double)),
              0);

    // A later full request of B is a hit (no third engine call).
    const auto secondB = cached->ComputeEriBatch(requestB);
    ASSERT_TRUE(secondB.has_value()) << secondB.error().message;
    EXPECT_EQ(engineCalls, 2);
    EXPECT_EQ(std::memcmp(secondB->values.data(),
                          firstB->values.data(),
                          firstB->values.size() * sizeof(double)),
              0);

    // The store grew on disk.
    auto store = qcx::storage::EriStore::Open(storePath, *molecule, *basis, "sto-3g", "");
    ASSERT_TRUE(store.has_value()) << store.error().message;
    EXPECT_GT(store->StoredQuartetCount(), storedAfterA);
}

TEST(CachedEriEngineTest, CountingEngineSeesOneCall) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<ShellQuartet> quartets = ClassQuartets(*pairList, 0, 0);
    ASSERT_GT(quartets.size(), std::size_t{0});

    int engineCalls = 0;
    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            ++engineCalls;
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_counting_engine.h5");
    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const auto first = cached->ComputeEriBatch(quartets);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    EXPECT_EQ(engineCalls, 1);

    const auto second = cached->ComputeEriBatch(quartets);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    EXPECT_EQ(engineCalls, 1); // served from disk, no recompute

    // A fresh decorator over the same file never recomputes either.
    auto reopened =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(reopened.has_value()) << reopened.error().message;
    const auto third = reopened->ComputeEriBatch(quartets);
    ASSERT_TRUE(third.has_value()) << third.error().message;
    EXPECT_EQ(engineCalls, 1);
}

TEST(CachedEriEngineTest, StatsCountHitsAndMisses) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<ShellQuartet> quartets = ClassQuartets(*pairList, 0, 0);
    ASSERT_GT(quartets.size(), std::size_t{0});

    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_stats.h5");
    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const auto first = cached->ComputeEriBatch(quartets);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    const auto second = cached->ComputeEriBatch(quartets);
    ASSERT_TRUE(second.has_value()) << second.error().message;

    const qcx::storage::CachedEriStats& stats = cached->Stats();
    EXPECT_EQ(stats.missQuartets, quartets.size());
    EXPECT_EQ(stats.hitQuartets, quartets.size());
    // Single-class requests fill the per-class maps.
    EXPECT_EQ(stats.missQuartetsByClass.at({0, 0}), quartets.size());
    EXPECT_EQ(stats.hitQuartetsByClass.at({0, 0}), quartets.size());
    EXPECT_GE(stats.recomputeMs, 0.0);
    EXPECT_GE(stats.readMs, 0.0);

    std::cout << "recomputeMs=" << stats.recomputeMs << " readMs=" << stats.readMs << "\n";
}

TEST(CachedEriEngineTest, CorruptChunkIsRejectedOnServe) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<ShellQuartet> quartets = ClassQuartets(*pairList, 0, 0);
    ASSERT_GT(quartets.size(), std::size_t{0});

    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_corrupt_serve.h5");
    {
        auto cached =
            CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
        ASSERT_TRUE(cached.has_value()) << cached.error().message;
        const auto first = cached->ComputeEriBatch(quartets);
        ASSERT_TRUE(first.has_value()) << first.error().message;
    } // the store's file handle is released before the tamper

    // Flip one bit of the first stored double, found through the manifest.
    {
        HighFive::File file(storePath.string(), HighFive::File::ReadWrite);
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

    // The decorator reopens (the checksum still passes - values are outside
    // it) and the full-hit serve must refuse: kIOError, never a silent
    // recompute.
    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;
    const auto served = cached->ComputeEriBatch(quartets);
    ASSERT_FALSE(served.has_value());
    EXPECT_EQ(served.error().code, qcx::ErrorCode::kIOError);
}

TEST(CachedEriEngineTest, EmptyRequestMatchesEngineError) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_empty_request.h5");
    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const std::vector<ShellQuartet> empty;
    const auto decorated = cached->ComputeEriBatch(empty);
    const auto plain = qcx::integrals::ComputeEriBatch(*molecule, *basis, empty);
    ASSERT_FALSE(decorated.has_value());
    ASSERT_FALSE(plain.has_value());
    EXPECT_EQ(decorated.error().code, plain.error().code);
    EXPECT_EQ(decorated.error().code, qcx::ErrorCode::kInvalidArgument);
}

#if QcxIntegralsF32

TEST(CachedEriEngineTest, CertifiedRoundTripIsBitIdentical) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<ShellQuartet> quartets = ClassQuartets(*pairList, 0, 0);
    ASSERT_GT(quartets.size(), std::size_t{0});

    int certifiedCalls = 0;
    CachedEriBatchEngine::UnderlyingCertifiedEngine certifiedEngine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            ++certifiedCalls;
            return qcx::integrals::ComputeEriBatchCertified(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_certified.h5");
    auto cached = CachedEriBatchEngine::Create(
        storePath, *molecule, *basis, "sto-3g", "", {}, certifiedEngine);
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const auto first = cached->ComputeEriBatchCertified(quartets);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    EXPECT_EQ(certifiedCalls, 1);

    const auto second = cached->ComputeEriBatchCertified(quartets);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    EXPECT_EQ(certifiedCalls, 1); // served from the fp32 lane

    EXPECT_EQ(first->computed, second->computed);
    ASSERT_EQ(first->values.size(), second->values.size());
    // fp32 values are stored verbatim: byte-identical round trip.
    EXPECT_EQ(std::memcmp(first->values.data(),
                          second->values.data(),
                          first->values.size() * sizeof(float)),
              0);

    // Bounds are stored as double in quartets-table order: bit-identical
    // on load.
    ASSERT_EQ(first->errorBounds.size(), second->errorBounds.size());
    EXPECT_EQ(std::memcmp(first->errorBounds.data(),
                          second->errorBounds.data(),
                          first->errorBounds.size() * sizeof(double)),
              0);
}

// The consumer-side budget re-check:
// certification survives a session boundary, it never becomes
// unconditional - the stored bound is re-checked against the caller's
// budget on load.
TEST(CachedEriEngineTest, StoredBoundRecheckedByConsumer) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<ShellQuartet> quartets = ClassQuartets(*pairList, 0, 0);
    ASSERT_GT(quartets.size(), std::size_t{0});

    CachedEriBatchEngine::UnderlyingCertifiedEngine certifiedEngine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            return qcx::integrals::ComputeEriBatchCertified(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_budget_recheck.h5");
    auto cached = CachedEriBatchEngine::Create(
        storePath, *molecule, *basis, "sto-3g", "", {}, certifiedEngine);
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const auto first = cached->ComputeEriBatchCertified(quartets);
    ASSERT_TRUE(first.has_value()) << first.error().message;

    // The second call is served from disk; the consumer-side check must
    // run identically against stored and computed batches. A deliberately
    // tiny budget is exceeded by every real a-priori bound.
    const auto served = cached->ComputeEriBatchCertified(quartets);
    ASSERT_TRUE(served.has_value()) << served.error().message;
    const double tinyBudget = 1e-30;
    bool refused = false;

    for (std::size_t q = 0; q < served->computed.size(); ++q)
    {
        if (served->errorBounds[q] > tinyBudget)
        {
            refused = true;
            break;
        }
    }

    EXPECT_TRUE(refused) << "every stored bound must exceed the tiny budget";
}

TEST(CachedEriEngineTest, CertifiedWithoutEngineIsUnimplemented) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    const std::vector<ShellQuartet> quartets = ClassQuartets(*pairList, 0, 0);
    ASSERT_GT(quartets.size(), std::size_t{0});

    CachedEriBatchEngine::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request) {
            return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
        };

    const std::filesystem::path storePath = TempStoreFile::MakePath("qcx_no_certified.h5");
    auto cached =
        CachedEriBatchEngine::Create(storePath, *molecule, *basis, "sto-3g", "", engine, {});
    ASSERT_TRUE(cached.has_value()) << cached.error().message;

    const auto result = cached->ComputeEriBatchCertified(quartets);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
}

#endif // QcxIntegralsF32

} // namespace
