// The engine-decorator seam's acceptance (FockBuildOptions::engineDecorator,
// ): the disk tier's missing path.
//
// WHY THIS TEST LIVES IN storage. The seam itself is integrals-side, but the
// tier that needed it is storage's CachedEriBatchEngine, and the DAG runs
// integrals -> storage (never the reverse) - so this module's test binary
// is the one that can hold BOTH the direct builder and the disk decorator in
// one process. What the cells pin is exactly the capability the seam exists
// for: a decorator built OUTSIDE the builder reaches the builder, becomes the
// engine every later request runs through, and the disk tier is therefore
// reachable through production code (tools/bench/probes/disk_tier_probe.cpp
// had to drive the decorator directly for want of this seam, and says so).
//
// EVERY CELL IS WRITTEN TO FAIL IF THE DECORATOR IS IGNORED - the defect
// shape this repo has already met in this family, a knob accepted and
// silently dropped. The positive cells assert on the DECORATOR's own
// counters, which move only if the engines the builder called are the ones
// the factory returned; the refusal cell asserts the factory's reason
// survives to the caller, which it cannot if the result is discarded.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_cache.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/storage/cached_eri_batch_engine.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::CertifiedEriBatchEngineFn;
using qcx::integrals::DecoratedEngines;
using qcx::integrals::EngineDecoratorFactory;
using qcx::integrals::EriBatch;
using qcx::integrals::EriBatchEngineFn;
using qcx::integrals::ShellQuartet;

// H = T + V for the fixture (the same two-integrals sum the builder tests
// build by hand).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCore(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
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

    return std::move(*kinetic);
}

// A dense symmetric spatial density: every block has a non-zero max |D|, so
// the product screen keeps quartets and the engine is genuinely reached (an
// identity density would screen every off-diagonal block away and make the
// call-count cells vacuous).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> MakeDensity(std::size_t n) {
    auto tensor = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*tensor)(i, j) = 0.05;
        }

        (*tensor)(i, i) += 0.1;
    }

    tensor->MarkHostDirty();
    return std::move(*tensor);
}

// The fixture's builder options: the fp64 lane forced, so every quartet is
// the decorator's business (the device-driven certified default would
// otherwise move quartets to the fp32 lane and make the counters
// machine-dependent - the seam's cells must not be).
qcx::integrals::FockBuildOptions SeamOptions(std::size_t maxCacheBytes) {
    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false;
    options.maxCacheBytes = maxCacheBytes;
    return options;
}

// The fixture, built once per cell (cheap: H2O/STO-3G is 7 functions).
struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> core;
};

std::optional<Fixture> MakeFixture() {
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();

    if (!molecule.has_value() || !basis.has_value())
    {
        return std::nullopt;
    }

    auto core = BuildCore(*molecule, *basis);

    if (!core.has_value())
    {
        return std::nullopt;
    }

    return Fixture{std::move(*molecule), std::move(*basis), std::move(*core)};
}

} // namespace

// The factory is called ONCE, at Create, and the pair it returns is the engine
// every later BuildFock runs through. A wiring that accepted the factory and
// dropped its result leaves both counters at zero after Create and one
// BuildFock - this cell is red.
TEST(EngineDecoratorSeam, TheDecoratorEnginesAreTheOnesTheBuilderRuns) {
    const auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << "H2O/STO-3G fixture construction failed";

    std::size_t factoryCalls = 0;
    std::size_t decoratedFp64Calls = 0;
    std::size_t decoratedQuartets = 0;

    // The store-only shape: the RAM tier's budget is zero, so the decorator
    // is the whole tier. That is the case the seam exists for - without it a
    // zero budget would drop the decorator silently.
    qcx::integrals::FockBuildOptions options = SeamOptions(0);
    options.engineDecorator =
        [&factoryCalls, &decoratedFp64Calls, &decoratedQuartets](
            const EriBatchEngineFn& rawFp64,
            const CertifiedEriBatchEngineFn& rawFp32) -> qcx::Result<DecoratedEngines> {
        ++factoryCalls;
        DecoratedEngines engines;
        // rawFp64 by VALUE: the returned lambda outlives the reference.
        engines.fp64 = [&decoratedFp64Calls, &decoratedQuartets, rawFp64](
                           const std::vector<ShellQuartet>& quartets) -> qcx::Result<EriBatch> {
            ++decoratedFp64Calls;
            decoratedQuartets += quartets.size();
            return rawFp64(quartets);
        };
        engines.fp32 = rawFp32;
        return engines;
    };

    auto builder = qcx::integrals::DirectJkFockBuilder::Create(
        fixture->molecule, fixture->basis, fixture->core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    EXPECT_EQ(factoryCalls, 1u) << "the factory is called ONCE, at Create - not per BuildFock";
    EXPECT_EQ(decoratedFp64Calls, 0u) << "Create must not evaluate any batch";

    auto density = MakeDensity(fixture->core.Shape()[0]);
    ASSERT_TRUE(density.has_value()) << density.error().message;

    auto first = builder->BuildFock(*density);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    EXPECT_GT(decoratedFp64Calls, 0u)
        << "the returned engine pair is the one the builder calls (a dropped decorator "
           "leaves this at zero)";
    EXPECT_GT(decoratedQuartets, 0u) << "the decorator saw a non-empty request";

    // The zero-budget engagement itself: the tier EXISTS (a decorator with no
    // RAM grant is granted the smallest budget the cache accepts, which admits
    // no block - a passthrough, not a disabled tier), which is what keeps a
    // store-only run from having its request silently dropped.
    const qcx::integrals::EriCacheStats* const cacheStats = builder->CacheStats();
    ASSERT_NE(cacheStats, nullptr)
        << "a set decorator engages the engine tier even at a zero RAM budget";
    EXPECT_EQ(cacheStats->maxCacheBytes, 1u) << "the passthrough is the smallest accepted budget";
    EXPECT_EQ(cacheStats->fp64HitQuartets, 0u)
        << "a passthrough admits nothing, so nothing is ever served from RAM";

    // Every later call goes through it too - the seam is not a one-shot.
    const std::size_t afterFirst = decoratedFp64Calls;
    auto second = builder->BuildFock(*density);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    EXPECT_GT(decoratedFp64Calls, afterFirst) << "every call after Create runs through the pair";
}

// A factory that cannot build its decorator stops the run and the reason
// survives to the caller. A wiring that accepted the factory and ignored a
// failure would compute on the raw engines while the caller believed a store
// was in force - the run-record defect - and this cell is red.
TEST(EngineDecoratorSeam, AFailingFactoryRefusesTheCreate) {
    const auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << "H2O/STO-3G fixture construction failed";

    qcx::integrals::FockBuildOptions options = SeamOptions(0);
    options.engineDecorator =
        [](const EriBatchEngineFn&,
           const CertifiedEriBatchEngineFn&) -> qcx::Result<DecoratedEngines> {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "seam refusal marker: the store could not be opened"});
    };

    auto builder = qcx::integrals::DirectJkFockBuilder::Create(
        fixture->molecule, fixture->basis, fixture->core, options);
    ASSERT_FALSE(builder.has_value()) << "a refused decorator must not yield a builder";
    EXPECT_EQ(builder.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(builder.error().message.find("seam refusal marker"), std::string::npos)
        << "the factory's own reason must survive to the caller: " << builder.error().message;
}

// The capability the seam was landed for: storage's disk decorator, wired the
// way the driver wires it, reaches the builder and serves the run - and the
// Fock it produces is bit-identical to the same run without any decorator
// (the store returns the engine's own bytes verbatim, so swapping the tier
// cannot move a value). Before the seam this cell could not be written: the
// decorator had no caller in the Fock path.
TEST(EngineDecoratorSeam, TheDiskTierReachesTheBuilderThroughTheSeam) {
    const auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << "H2O/STO-3G fixture construction failed";

    auto density = MakeDensity(fixture->core.Shape()[0]);
    ASSERT_TRUE(density.has_value()) << density.error().message;

    // The control leg: no decorator, no RAM tier - today's plain path. Both
    // legs run the SERIAL fallback: the repo's own cached-vs-uncached
    // bit-identity pin states why (ParallelReduce's parallel combine order is
    // unspecified, so exact equality is a single-threaded statement -
    // integrals/tests/eri_cache_test.cpp, the builder-level parity cell).
    qcx::integrals::FockBuildOptions plainOptions = SeamOptions(0);
    plainOptions.maxParallelChunks = 1;
    auto plainBuilder = qcx::integrals::DirectJkFockBuilder::Create(
        fixture->molecule, fixture->basis, fixture->core, plainOptions);
    ASSERT_TRUE(plainBuilder.has_value()) << plainBuilder.error().message;
    auto plain = plainBuilder->BuildFock(*density);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    // The seam leg: storage's own decorator, built with the two engines the
    // builder hands the factory - the driver's wiring, verbatim in shape.
    std::error_code tempError;
    const std::filesystem::path storePath =
        std::filesystem::temp_directory_path(tempError) / "qcx_seam_disk_tier_test.h5";
    ASSERT_FALSE(tempError) << "no writable temp directory: " << tempError.message();
    std::error_code ignored;
    std::filesystem::remove(storePath, ignored);

    std::optional<qcx::storage::CachedEriBatchEngine> decorator;
    std::size_t rawEngineCalls = 0;
    Eigen::MatrixXd seamMatrix;

    {
        // The decorator outlives the builder's every call; it is dropped
        // before the store file is removed (the probe's measured lesson - a
        // store removed under a live handle fails silently).
        qcx::integrals::FockBuildOptions options = SeamOptions(0);
        options.maxParallelChunks = 1;
        options.engineDecorator =
            [&decorator, &rawEngineCalls, &fixture, &storePath](
                const EriBatchEngineFn& rawFp64,
                const CertifiedEriBatchEngineFn& rawFp32) -> qcx::Result<DecoratedEngines> {
            auto created = qcx::storage::CachedEriBatchEngine::Create(
                storePath, fixture->molecule, fixture->basis, "sto-3g", "", rawFp64, rawFp32);

            if (!created.has_value())
            {
                return std::unexpected(created.error());
            }

            decorator = std::move(*created);

            DecoratedEngines engines;
            engines.fp64 = [&decorator, &rawEngineCalls](
                               const std::vector<ShellQuartet>& quartets) -> qcx::Result<EriBatch> {
                ++rawEngineCalls;
                return decorator->ComputeEriBatch(quartets);
            };
            return engines;
        };

        auto builder = qcx::integrals::DirectJkFockBuilder::Create(
            fixture->molecule, fixture->basis, fixture->core, options);
        ASSERT_TRUE(builder.has_value()) << builder.error().message;

        auto fock = builder->BuildFock(*density);
        ASSERT_TRUE(fock.has_value()) << fock.error().message;
        seamMatrix = qcx::testing::ToMatrix(*fock);
    }

    ASSERT_TRUE(decorator.has_value()) << "the factory never ran";
    EXPECT_GT(rawEngineCalls, 0u) << "the builder never reached the decorator";
    EXPECT_GT(decorator->Stats().missQuartets, 0u)
        << "the disk tier never recomputed anything, so it never served the run";

    const Eigen::MatrixXd plainMatrix = qcx::testing::ToMatrix(*plain);

    ASSERT_EQ(plainMatrix.rows(), seamMatrix.rows());
    ASSERT_EQ(plainMatrix.cols(), seamMatrix.cols());

    for (Eigen::Index i = 0; i < plainMatrix.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < plainMatrix.cols(); ++j)
        {
            EXPECT_EQ(plainMatrix(i, j), seamMatrix(i, j))
                << "the store tier moved a value at (" << i << "," << j << ")";
        }
    }

    std::filesystem::remove(storePath, ignored);
}
