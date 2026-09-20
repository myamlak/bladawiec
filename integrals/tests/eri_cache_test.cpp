// The in-memory ERI-value cache tests (eri_cache.hpp):
//   - the bit-parity contract: a cached ComputeEriBatch/ComputeEriBatchCertified
//     returns the EXACT layout a direct engine call produces - all-miss,
//     all-hit, and mixed requests, both lanes,
//   - the budget behavior: exhaustion leaves correctness untouched (only
//     the hit rate suffers), a zero budget is kInvalidArgument at Create,
//     an empty request reproduces the engine's error, and a missing engine
//     (either lane) is kUnimplemented,
//   - the builder level: a DirectJkFockBuilder with the cache engaged is
//     bit-identical to the uncached builder on the same densities (the
//     serial fallback pin - ParallelReduce's parallel combine order is
//     unspecified, so exact equality is tested single-threaded), with the
//     per-call hit counts filling in from the second call on.
//
// The decorator tests are light (a 7-function water walk in milliseconds),
// so they run in fast mode; the builder walk follows the other H2O builder
// tests' fast-mode gate.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_cache.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <functional>
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

// H = T + V from the one-electron engines.
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

// A symmetric density with |D| <= 0.75 and a diagonal near 0.5 - the
// magnitudes the screening and certified-lane gates expect (fock_build_test's pattern,
// with a seed knob for distinct densities). Test helper; the (n, seed) order
// is the call-site convention.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd PhysicalDensity(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

// Every 8-fold canonical quartet of the pair list (i <= j, k <= l,
// pair (i, j) >= (k, l)) - the full canonical request space of one system.
std::vector<qcx::integrals::ShellQuartet> AllCanonicalQuartets(
    const qcx::integrals::ShellPairList& pairList) {
    std::vector<qcx::integrals::ShellQuartet> quartets;
    const std::size_t nShells = pairList.shells.size();

    for (std::size_t i = 0; i < nShells; ++i)
    {
        for (std::size_t j = i; j < nShells; ++j)
        {
            const std::size_t bra = qcx::integrals::PairIndexOf(i, j, pairList);

            for (std::size_t k = 0; k < nShells; ++k)
            {
                for (std::size_t l = k; l < nShells; ++l)
                {
                    if (qcx::integrals::PairIndexOf(k, l, pairList) <= bra)
                    {
                        quartets.push_back(qcx::integrals::ShellQuartet{i, j, k, l});
                    }
                }
            }
        }
    }

    return quartets;
}

bool BitIdentical(const std::vector<double>& a, const std::vector<double>& b) {
    return a == b;
}

bool BitIdentical(const std::vector<float>& a, const std::vector<float>& b) {
    return a == b;
}

bool BitIdentical(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    return a.rows() == b.rows() && a.cols() == b.cols() && (a.array() == b.array()).all();
}

// The engine-call counter: the cache must evaluate a quartet exactly once
// across all-hit repeats.
struct CountingEngine {
    std::size_t calls = 0;
};

} // namespace

TEST(EriCacheTest, BatchParityAllMissThenAllHit) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    const std::vector<qcx::integrals::ShellQuartet> quartets = AllCanonicalQuartets(*pairList);

    ASSERT_GT(quartets.size(), 0u);

    CountingEngine counting;
    qcx::integrals::EriBatchCache::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request)
        -> qcx::Result<qcx::integrals::EriBatch> {
        ++counting.calls;
        return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
    };

    auto cache =
        qcx::integrals::EriBatchCache::Create(std::size_t{1024} * 1024, *pairList, engine, {});

    ASSERT_TRUE(cache.has_value());

    // The reference: a direct engine call on the same request.
    auto plain = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);

    ASSERT_TRUE(plain.has_value());

    auto first = cache->ComputeEriBatch(quartets);

    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(counting.calls, 1u) << "the first call must be an all-miss compute";
    EXPECT_EQ(first->computed, plain->computed);
    EXPECT_TRUE(BitIdentical(first->values, plain->values));
    EXPECT_EQ(cache->Stats().fp64HitQuartets, 0u);
    EXPECT_EQ(cache->Stats().fp64MissQuartets, quartets.size());

    // The second call on the same request must be served entirely from the
    // arena - no engine call, byte-identical output.
    auto second = cache->ComputeEriBatch(quartets);

    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(counting.calls, 1u) << "the second call must be served from the cache";
    EXPECT_EQ(second->computed, plain->computed);
    EXPECT_TRUE(BitIdentical(second->values, plain->values));
    EXPECT_EQ(cache->Stats().fp64HitQuartets, quartets.size());
    EXPECT_EQ(cache->Stats().fp64MissQuartets, quartets.size());
    EXPECT_GT(cache->Stats().fp64PayloadBytes, 0u);
}

TEST(EriCacheTest, CertifiedLaneParity) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    const std::vector<qcx::integrals::ShellQuartet> quartets = AllCanonicalQuartets(*pairList);

    CountingEngine counting;
    qcx::integrals::EriBatchCache::UnderlyingCertifiedEngine certifiedEngine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request)
        -> qcx::Result<qcx::integrals::CertifiedBatch> {
        ++counting.calls;
        return qcx::integrals::ComputeEriBatchCertified(*molecule, *basis, request);
    };

    auto cache = qcx::integrals::EriBatchCache::Create(
        std::size_t{1024} * 1024, *pairList, {}, certifiedEngine);

    ASSERT_TRUE(cache.has_value());

    auto plain = qcx::integrals::ComputeEriBatchCertified(*molecule, *basis, quartets);

    ASSERT_TRUE(plain.has_value());

    auto first = cache->ComputeEriBatchCertified(quartets);

    ASSERT_TRUE(first.has_value());
    EXPECT_EQ(counting.calls, 1u) << "the first call must be an all-miss compute";
    EXPECT_EQ(first->computed, plain->computed);
    EXPECT_TRUE(BitIdentical(first->values, plain->values));
    EXPECT_TRUE(BitIdentical(first->errorBounds, plain->errorBounds));
    EXPECT_EQ(cache->Stats().fp32HitQuartets, 0u);
    EXPECT_EQ(cache->Stats().fp32MissQuartets, quartets.size());

    auto second = cache->ComputeEriBatchCertified(quartets);

    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(counting.calls, 1u) << "the second call must be served from the cache";
    EXPECT_EQ(second->computed, plain->computed);
    EXPECT_TRUE(BitIdentical(second->values, plain->values));
    EXPECT_TRUE(BitIdentical(second->errorBounds, plain->errorBounds));
    EXPECT_EQ(cache->Stats().fp32HitQuartets, quartets.size());
    EXPECT_GT(cache->Stats().fp32PayloadBytes, 0u);
}

TEST(EriCacheTest, MixedRequestScattersHitsAndMisses) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    const std::vector<qcx::integrals::ShellQuartet> quartets = AllCanonicalQuartets(*pairList);

    ASSERT_GT(quartets.size(), 1u);

    CountingEngine counting;
    qcx::integrals::EriBatchCache::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request)
        -> qcx::Result<qcx::integrals::EriBatch> {
        ++counting.calls;
        return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
    };

    auto cache =
        qcx::integrals::EriBatchCache::Create(std::size_t{1024} * 1024, *pairList, engine, {});

    ASSERT_TRUE(cache.has_value());

    // Warm a strict subset, then request the full list: the warm keys must
    // hit while the rest recompute, and the full result must still match
    // the direct engine byte for byte.
    const std::vector<qcx::integrals::ShellQuartet> warm(
        quartets.begin(), quartets.begin() + static_cast<std::ptrdiff_t>(quartets.size() / 2));
    auto warmCall = cache->ComputeEriBatch(warm);

    ASSERT_TRUE(warmCall.has_value());

    auto plain = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);

    ASSERT_TRUE(plain.has_value());

    auto full = cache->ComputeEriBatch(quartets);

    ASSERT_TRUE(full.has_value());
    EXPECT_EQ(counting.calls, 2u) << "the warm request and the miss half of the full request";
    EXPECT_EQ(full->computed, plain->computed);
    EXPECT_TRUE(BitIdentical(full->values, plain->values));
    EXPECT_EQ(cache->Stats().fp64HitQuartets, warm.size());
    EXPECT_EQ(cache->Stats().fp64MissQuartets, quartets.size());
}

TEST(EriCacheTest, BudgetExhaustionStaysCorrect) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    const std::vector<qcx::integrals::ShellQuartet> quartets = AllCanonicalQuartets(*pairList);

    CountingEngine counting;
    qcx::integrals::EriBatchCache::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request)
        -> qcx::Result<qcx::integrals::EriBatch> {
        ++counting.calls;
        return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
    };

    // A budget far below the union: only a handful of the smallest blocks
    // fit, the rest recompute on every request - correctness must not
    // depend on the budget.
    auto cache = qcx::integrals::EriBatchCache::Create(24, *pairList, engine, {});

    ASSERT_TRUE(cache.has_value());

    auto plain = qcx::integrals::ComputeEriBatch(*molecule, *basis, quartets);

    ASSERT_TRUE(plain.has_value());

    auto first = cache->ComputeEriBatch(quartets);

    ASSERT_TRUE(first.has_value());
    EXPECT_TRUE(BitIdentical(first->values, plain->values));

    auto second = cache->ComputeEriBatch(quartets);

    ASSERT_TRUE(second.has_value());
    EXPECT_EQ(counting.calls, 2u) << "budget-exhausted quartets recompute on every call";
    EXPECT_TRUE(BitIdentical(second->values, plain->values));
    EXPECT_GT(cache->Stats().fp64HitQuartets, 0u) << "the smallest blocks must still cache";
    EXPECT_LT(cache->Stats().fp64HitQuartets, quartets.size())
        << "the budget must not hold the whole union";
}

TEST(EriCacheTest, EmptyRequestMatchesEngineError) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    CountingEngine counting;
    qcx::integrals::EriBatchCache::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request)
        -> qcx::Result<qcx::integrals::EriBatch> {
        ++counting.calls;
        return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
    };

    auto cache =
        qcx::integrals::EriBatchCache::Create(std::size_t{1024} * 1024, *pairList, engine, {});

    ASSERT_TRUE(cache.has_value());

    const std::vector<qcx::integrals::ShellQuartet> empty;
    const auto plain = qcx::integrals::ComputeEriBatch(*molecule, *basis, empty);
    const auto decorated = cache->ComputeEriBatch(empty);

    ASSERT_FALSE(plain.has_value());
    ASSERT_FALSE(decorated.has_value());
    EXPECT_EQ(decorated.error().code, plain.error().code)
        << "the empty request must reproduce the engine's error";
    EXPECT_EQ(decorated.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_EQ(counting.calls, 0u) << "the error must be produced without the engine";
}

TEST(EriCacheTest, CertifiedWithoutEngineIsUnimplemented) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    auto cache = qcx::integrals::EriBatchCache::Create(std::size_t{1024} * 1024, *pairList, {}, {});

    ASSERT_TRUE(cache.has_value());

    const auto result = cache->ComputeEriBatchCertified(AllCanonicalQuartets(*pairList));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(EriCacheTest, Fp64WithoutEngineIsUnimplemented) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    auto cache = qcx::integrals::EriBatchCache::Create(std::size_t{1024} * 1024, *pairList, {}, {});

    ASSERT_TRUE(cache.has_value());

    const auto result = cache->ComputeEriBatch(AllCanonicalQuartets(*pairList));

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(EriCacheTest, ZeroBudgetCreateFailsWithoutEngineCall) {
    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value());

    CountingEngine counting;
    qcx::integrals::EriBatchCache::UnderlyingEngine engine =
        [&](const std::vector<qcx::integrals::ShellQuartet>& request)
        -> qcx::Result<qcx::integrals::EriBatch> {
        ++counting.calls;
        return qcx::integrals::ComputeEriBatch(*molecule, *basis, request);
    };

    const auto result =
        qcx::integrals::EriBatchCache::Create(std::size_t{0}, *pairList, engine, {});

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kInvalidArgument)
        << "a zero budget is the only Create error path";
    EXPECT_EQ(counting.calls, 0u) << "the engine must never be called for a rejected Create";
}

TEST(EriCacheTest, BuilderParityCachedVsUncached) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto molecule = MakeH2oSto3g();

    ASSERT_TRUE(molecule.has_value());

    auto basis = MakeH2oSto3gBasis();

    ASSERT_TRUE(basis.has_value());

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    ASSERT_TRUE(core.has_value());

    qcx::integrals::FockBuildOptions plainOptions;
    // The serial fallback pin: ParallelReduce's parallel combine order is
    // unspecified (it sums the chunk partials in thread-completion order),
    // so exact equality is tested single-threaded - the deterministic
    // path, bit-identical between the two builders by construction.
    plainOptions.maxParallelChunks = 1;
    // The certified lane is requested explicitly: its default is the
    // device's now (the device probe's verdict), and with the lane off on
    // both legs the bound-sum parity below would compare 0.0 to 0.0 - a
    // green pin that no longer covers what it names.
    plainOptions.useCertifiedMixedPrecision = true;
    qcx::integrals::FockBuildOptions cachedOptions = plainOptions;
    // Far above the water union (~50 quartets): every screened quartet
    // caches after the first call.
    cachedOptions.maxCacheBytes = std::size_t{16} * 1024 * 1024;

    auto plainBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, plainOptions);

    ASSERT_TRUE(plainBuilder.has_value());
    EXPECT_EQ(plainBuilder->CacheStats(), nullptr) << "maxCacheBytes 0 keeps the cache off";

    auto cachedBuilder =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, cachedOptions);

    ASSERT_TRUE(cachedBuilder.has_value());
    ASSERT_NE(cachedBuilder->CacheStats(), nullptr) << "the cache must be engaged";

    // Two densities: the repeat of density A must be served entirely from
    // the cache (same screened set, all hits); density B exercises the
    // partial-hit path (whatever the screen keeps, the served blocks must
    // match the uncached builder bit for bit).
    const Eigen::MatrixXd densityA = PhysicalDensity(7, 20260817);
    const Eigen::MatrixXd densityB = PhysicalDensity(7, 20260818);

    for (int call = 1; call <= 3; ++call)
    {
        const Eigen::MatrixXd& density = (call == 3) ? densityB : densityA;
        qcx::integrals::FockBuildStats plainStats;
        qcx::integrals::FockBuildStats cachedStats;
        double plainBoundSum = -1.0;
        double cachedBoundSum = -1.0;
        auto plainFock = plainBuilder->BuildFock(*ToTensor(density), &plainBoundSum, &plainStats);

        ASSERT_TRUE(plainFock.has_value());

        auto cachedFock =
            cachedBuilder->BuildFock(*ToTensor(density), &cachedBoundSum, &cachedStats);

        ASSERT_TRUE(cachedFock.has_value());
        EXPECT_TRUE(BitIdentical(ToMatrix(*plainFock), ToMatrix(*cachedFock)))
            << "call " << call << ": the cached path must be bit-identical to the plain path";
        EXPECT_EQ(cachedBoundSum, plainBoundSum)
            << "call " << call << ": the certified bound sum must match bit for bit";

        const std::size_t screenedQuartets =
            cachedStats.fp64QuartetCount + cachedStats.fp32QuartetCount;
        const std::size_t cachedHits =
            cachedStats.cacheHitFp64QuartetCount + cachedStats.cacheHitFp32QuartetCount;

        EXPECT_GT(screenedQuartets, 0u) << "the water screen must not be empty";

        if (call == 1)
        {
            EXPECT_EQ(cachedHits, 0u) << "the first call must be cold";
        } else if (call == 2)
        {
            // The density repeat: the identical screen (identical keys,
            // identical lane routing), so EVERY screened quartet must be
            // served from the cache.
            EXPECT_EQ(cachedHits, screenedQuartets)
                << "the repeated density must be served entirely from the cache";
        } else
        {
            // Density B may screen a different set than A - what matters is
            // the served blocks still match the uncached builder bit for
            // bit (asserted above) and that at least the recurring quartets
            // hit.
            EXPECT_GT(cachedHits, 0u) << "the recurring quartets must hit";
        }
    }

    const qcx::integrals::EriCacheStats& stats = *cachedBuilder->CacheStats();
    EXPECT_GT(stats.fp64HitQuartets, 0u);
    EXPECT_GT(stats.fp64PayloadBytes + stats.fp32PayloadBytes, 0u)
        << "the union must be stored in the dual-lane arenas";
    EXPECT_EQ(stats.maxCacheBytes, std::size_t{16} * 1024 * 1024)
        << "the budget must report what was actually engaged";
}
