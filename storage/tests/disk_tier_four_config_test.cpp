// The DISK variant of the four-configuration measurement.
//
// THE QUESTION. The specification asks for the work
// and memory of the four configurations the ERI-value cache × class-aware
// reduction produce (plain, plain + cache, class path, class path with the cache
// asked), and then: "Where the disk store is involved (`eri_store.hpp`: one
// file per system, append-only, per-chunk FNV-1a-64) the same four
// configurations are reported with `CachedEriBatchEngine` in place of the
// in-memory decorator." The in-memory half is
// tools/bench/probes/four_config_probe.cpp. This file is the disk half.
//
// WHY THE MEASUREMENT LIVES IN storage RATHER THAN BESIDE THE PROBE. The disk
// tier is storage's CachedEriBatchEngine and the DAG runs integrals -> storage
// (never the reverse), so this module - the one below the driver that can
// link both sides - is the only place the direct builder and the disk decorator
// meet in one process. That is the same reason engine_decorator_seam_test.cpp is
// here, and the shape is the one scf/tests/rks_convergence_probe_test.cpp
// established for a section's own numbers: a TEST rather than a throwaway probe,
// because what it prints is the record and the assertions are the invariant that
// survives a re-run on another machine.
//
// WHAT IT MEASURES (counts and footprint - bars wall-clock claims from
// this evidence, so no timing is reported and none is needed: the
// quantity it is about is already a count):
//   C1-C4 are configuration labels, deliberately not decision references.
//   C1 plain                   no decorator, no RAM grant: the reference.
//   C2 plain + disk tier       the CachedEriBatchEngine decorator and NO RAM
//                              grant - the store-only shape, where the RAM tier
//                              is a passthrough and the disk tier serves the run.
//   C3 class path              the real detected reduction
//                              (scf::BuildSymmetryReduction), no decorator.
//   C4 class path + disk asked C3 with the decorator set as well.
//
// THE PREDICTION, stated before the run so a cell can fail: the class-aware path
// disengages the WHOLE engine tier (the contract on
// FockBuildOptions::engineDecorator, fock_build.hpp; the engagement guard in
// fock_build.cpp), so C4 must run exactly what C3 runs - the factory must never
// be called, no store must be opened, and the builder must report no tier - while
// C2's Fock must be bit-identical to C1's, because the store returns the engine's
// own bytes verbatim. C2 also has to show the tier actually serving: a decorator
// that is accepted and then never reached would satisfy a value comparison
// vacuously.
//
// THE FIXTURE is the probe's own (H2O/STO-3G, C2v, groupOrder 4) on the probe's
// own core-guess density, so the class-path cells here are comparable with the
// in-memory table's by construction rather than by resemblance. The runs are
// SERIAL (maxParallelChunks = 1): ParallelReduce's parallel combine order is
// unspecified, so exact equality is a single-threaded statement - the same reason
// engine_decorator_seam_test.cpp's parity cell is serial. The in-memory probe
// runs its cells in parallel, which is a difference of the instrument, not of the
// configurations; the counts below are the comparable quantity.
//
// HOW TO RUN IT (the one command that reproduces the table):
//   build/windows-msvc/storage/Release/qcx-storage-tests.exe
//       --gtest_filter=DiskTierFourConfig.*

#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_cache.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/symmetry_reduction.hpp"
#include "qcx/storage/cached_eri_batch_engine.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Eigenvalues>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::CertifiedEriBatchEngineFn;
using qcx::integrals::DecoratedEngines;
using qcx::integrals::EriBatch;
using qcx::integrals::EriBatchEngineFn;
using qcx::integrals::ShellQuartet;

constexpr std::size_t kBudgetBytes = std::size_t{16} * 1024 * 1024 * 1024;

struct Fixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    CpuTensor2 core;
    CpuTensor2 density;
};

struct ConfigOutcome {
    std::string label;
    bool ran = false;
    std::string error;
    Eigen::MatrixXd fock;
    qcx::integrals::FockBuildStats call1;
    qcx::integrals::FockBuildStats call2;
    bool modePresent = false;
    qcx::integrals::FockModeInfo mode{};
    bool tierPresent = false;
    qcx::integrals::EriCacheStats cache{};
    bool diskRequested = false;
    std::size_t decoratorFactoryCalls = 0;
    std::size_t diskHitQuartets = 0;
    std::size_t diskMissQuartets = 0;
    std::uintmax_t storeBytes = 0;
};

// H = T + V for the fixture (the same two-integrals sum the module's other
// builder tests build by hand).
qcx::Result<CpuTensor2> BuildCore(const qcx::molecule::Molecule& molecule,
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

// The probe's core-guess density: solve H C = S C e, occupy the lowest nOcc
// orbitals, rho = C_occ C_occ^T (the builder's spatial rho = D/2). Reached
// through public headers only, and built ONCE for every configuration - the
// rows are comparable only if the density is the same one.
qcx::Result<CpuTensor2> BuildCoreGuessDensity(const qcx::molecule::Molecule& molecule,
                                              const qcx::basisset::BasisSet& basisSet,
                                              const CpuTensor2& core) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    const Eigen::MatrixXd hMatrix = qcx::testing::ToMatrix(core);
    const Eigen::MatrixXd sMatrix = qcx::testing::ToMatrix(*overlap);

    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(hMatrix, sMatrix);
    const std::size_t nOccupied = static_cast<std::size_t>(molecule.ElectronCount()) / 2;
    const Eigen::MatrixXd occupied =
        solver.eigenvectors().leftCols(static_cast<Eigen::Index>(nOccupied));
    const Eigen::MatrixXd rho = occupied * occupied.transpose();

    return qcx::testing::ToTensor(rho);
}

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

    auto density = BuildCoreGuessDensity(*molecule, *basis, *core);

    if (!density.has_value())
    {
        return std::nullopt;
    }

    return Fixture{std::move(*molecule), std::move(*basis), std::move(*core), std::move(*density)};
}

// One store file per disk configuration, so a later configuration can never be
// served out of an earlier one's misses: the decorator's unit is the CALL, and a
// shared file would turn the four rows into one sequence.
std::filesystem::path StorePath(std::string_view slug) {
    std::error_code error;
    const std::filesystem::path directory = std::filesystem::temp_directory_path(error);

    if (error)
    {
        return std::filesystem::path(std::string(slug) + ".h5");
    }

    return directory / (std::string(slug) + ".h5");
}

ConfigOutcome RunConfig(const std::string& label,
                        std::string_view storeSlug,
                        const Fixture& fixture,
                        const qcx::integrals::SymmetryReduction* reduction,
                        bool disk) {
    ConfigOutcome out;
    out.label = label;
    out.diskRequested = disk;

    auto budget = qcx::memory::WorkspaceBudget::Create(kBudgetBytes);

    if (!budget.has_value())
    {
        out.error = "WorkspaceBudget::Create failed";
        return out;
    }

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    options.useCertifiedMixedPrecision = false; // one lane, deterministic
    options.maxParallelChunks = 1; // exact equality is a serial statement
    options.workspaceBudget = &*budget;
    options.maxCacheBytes = 0; // the store-only shape: no RAM grant at all
    options.symmetryReduction = reduction;

    // The decorator is declared BEFORE the builder so that the builder (and the
    // engine pair it holds, which captures the decorator by reference) is
    // destroyed first - a store removed under a live handle fails silently.
    std::optional<qcx::storage::CachedEriBatchEngine> decorator;
    const std::filesystem::path storePath = StorePath(storeSlug);

    if (disk)
    {
        std::error_code ignored;
        std::filesystem::remove(storePath, ignored);

        options.engineDecorator =
            [&decorator, &out, &storePath, &fixture](
                const EriBatchEngineFn& rawFp64,
                const CertifiedEriBatchEngineFn& rawFp32) -> qcx::Result<DecoratedEngines> {
            ++out.decoratorFactoryCalls;

            auto created = qcx::storage::CachedEriBatchEngine::Create(
                storePath, fixture.molecule, fixture.basis, "sto-3g", "", rawFp64, rawFp32);

            if (!created.has_value())
            {
                return std::unexpected(created.error());
            }

            decorator = std::move(*created);

            DecoratedEngines engines;
            engines.fp64 =
                [&decorator](const std::vector<ShellQuartet>& quartets) -> qcx::Result<EriBatch> {
                return decorator->ComputeEriBatch(quartets);
            };
            // The certified fp32 lane is unreachable in this configuration
            // (useCertifiedMixedPrecision is false), so it is passed through
            // rather than routed - an untested route would be dead code.
            engines.fp32 = rawFp32;
            return engines;
        };
    }

    auto builder = qcx::integrals::DirectJkFockBuilder::Create(
        fixture.molecule, fixture.basis, fixture.core, options);

    if (!builder.has_value())
    {
        out.error = builder.error().message;
        return out;
    }

    out.modePresent = builder->ModeInfo().has_value();

    if (out.modePresent)
    {
        out.mode = *builder->ModeInfo();
    }

    const qcx::integrals::EriCacheStats* const created = builder->CacheStats();
    out.tierPresent = created != nullptr;

    if (out.tierPresent)
    {
        out.cache = *created;
    }

    auto fock1 = builder->BuildFock(fixture.density, nullptr, &out.call1);

    if (!fock1.has_value())
    {
        out.error = fock1.error().message;
        return out;
    }

    out.fock = qcx::testing::ToMatrix(*fock1);

    auto fock2 = builder->BuildFock(fixture.density, nullptr, &out.call2);

    if (!fock2.has_value())
    {
        out.error = fock2.error().message;
        return out;
    }

    out.ran = true;

    if (builder->CacheStats() != nullptr)
    {
        out.cache = *builder->CacheStats();
    }

    if (decorator.has_value())
    {
        out.diskHitQuartets = decorator->Stats().hitQuartets;
        out.diskMissQuartets = decorator->Stats().missQuartets;
    }

    if (std::filesystem::exists(storePath))
    {
        std::error_code sizeError;
        const std::uintmax_t bytes = std::filesystem::file_size(storePath, sizeError);

        if (!sizeError)
        {
            out.storeBytes = bytes;
        }
    }

    return out;
}

void PrintCall(const char* which, const qcx::integrals::FockBuildStats& stats) {
    std::printf("  %-8s fp64Quartets=%zu fp32Quartets=%zu cacheHits=%zu sigPairs=%zu "
                "primProducts=%zu elementDrops=%zu\n",
                which,
                stats.fp64QuartetCount,
                stats.fp32QuartetCount,
                stats.cacheHitFp64QuartetCount + stats.cacheHitFp32QuartetCount,
                stats.significantPairCount,
                stats.primitiveProductSum,
                stats.elementDrops);
}

double MaxAbsDiff(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.size() != b.size())
    {
        return -1.0;
    }

    return (a - b).cwiseAbs().maxCoeff();
}

void PrintOutcome(const ConfigOutcome& out) {
    std::printf("\n=== CONFIG %s ===\n", out.label.c_str());

    if (!out.ran && !out.error.empty())
    {
        std::printf("  REFUSED  %s\n", out.error.c_str());
        return;
    }

    if (!out.ran)
    {
        std::printf("  DID NOT RUN\n");
        return;
    }

    if (out.modePresent)
    {
        const char* modeName =
            out.mode.mode == qcx::integrals::FockBuildMode::kFastPath
                ? "fast"
                : (out.mode.mode == qcx::integrals::FockBuildMode::kLightPath ? "light" : "disk");
        std::printf("  create   mode=%s predicted=%zu reserved=%zu cache=%zu classTable=%zu "
                    "classPathDisengaged=%d\n",
                    modeName,
                    out.mode.predictedBytes,
                    out.mode.reservedBytes,
                    out.mode.cacheBytes,
                    out.mode.classTableBytes,
                    out.mode.classPathDisengaged ? 1 : 0);
    } else
    {
        std::printf("  create   (no mode record: no budget)\n");
    }

    std::printf("  tier     present=%d ramGrantBytes=%zu\n",
                out.tierPresent ? 1 : 0,
                out.tierPresent ? out.cache.maxCacheBytes : std::size_t{0});
    PrintCall("call 1", out.call1);
    PrintCall("call 2", out.call2);
    std::printf("  disk     requested=%d factoryCalls=%zu hits=%zu misses=%zu storeBytes=%llu\n",
                out.diskRequested ? 1 : 0,
                out.decoratorFactoryCalls,
                out.diskHitQuartets,
                out.diskMissQuartets,
                static_cast<unsigned long long>(out.storeBytes));
}

} // namespace

// The four configurations with the disk tier in place of the
// in-memory decorator, on the symmetric fixture, on the counts-plus-footprint
// basis. The printed rows ARE the disk half of the table.
TEST(DiskTierFourConfig, TheFourConfigurationsWithTheDiskTier) {
    const auto fixture = MakeFixture();
    ASSERT_TRUE(fixture.has_value()) << "H2O/STO-3G fixture construction failed";

    const auto reduction = qcx::scf::BuildSymmetryReduction(fixture->molecule, fixture->basis);
    ASSERT_TRUE(reduction.has_value()) << reduction.error().message;

    const auto pairList = qcx::integrals::BuildShellPairs(fixture->molecule, fixture->basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const qcx::integrals::SymmetryReduction* const reductionPtr = &*reduction;

    std::printf("disk-tier four-configuration measurement - H2O/STO-3G\n");
    std::printf(
        "fixture  atoms=%zu functions=%zu pairs=%zu reduction groupOrder=%zu isTrivial=%d\n",
        fixture->molecule.AtomCount(),
        pairList->functionCount,
        pairList->pairs.size(),
        reductionPtr->groupOrder,
        reductionPtr->isTrivial ? 1 : 0);
    std::printf(
        "budget   %zu B per configuration, no RAM grant on any row (the store-only shape)\n",
        kBudgetBytes);

    const ConfigOutcome plain = RunConfig("C1 plain", "r36_disk_c1", *fixture, nullptr, false);
    const ConfigOutcome disk =
        RunConfig("C2 plain + disk tier", "r36_disk_c2", *fixture, nullptr, true);
    const ConfigOutcome classPath =
        RunConfig("C3 class path", "r36_disk_c3", *fixture, reductionPtr, false);
    const ConfigOutcome classPathDisk =
        RunConfig("C4 class path + disk asked", "r36_disk_c4", *fixture, reductionPtr, true);

    PrintOutcome(plain);
    PrintOutcome(disk);
    PrintOutcome(classPath);
    PrintOutcome(classPathDisk);

    std::printf("\n=== SUMMARY (disk variant, H2O/STO-3G) ===\n");
    std::printf("  disk     control C1 has no tier: present=%d\n", plain.tierPresent ? 1 : 0);
    std::printf("  value    C2 vs C1 max |dF| = %.6e\n", MaxAbsDiff(disk.fock, plain.fock));
    std::printf("  value    C4 vs C3 max |dF| = %.6e\n",
                MaxAbsDiff(classPathDisk.fock, classPath.fock));
    std::printf("  counters C3 call1=%zu/C4 call1=%zu  C3 call2=%zu/C4 call2=%zu\n",
                classPath.call1.fp64QuartetCount,
                classPathDisk.call1.fp64QuartetCount,
                classPath.call2.fp64QuartetCount,
                classPathDisk.call2.fp64QuartetCount);
    std::printf("  the exclusion, read off the decorator rather than argued:\n");
    std::printf("    C4 factoryCalls=%zu storesOpened=%s -> the class path drops the disk tier "
                "BEFORE the factory is reached\n",
                classPathDisk.decoratorFactoryCalls,
                classPathDisk.storeBytes == 0 ? "no" : "yes");
    std::printf("DISK_TIER_TABLE_DONE\n");

    ASSERT_TRUE(plain.ran) << plain.error;
    ASSERT_TRUE(disk.ran) << disk.error;
    ASSERT_TRUE(classPath.ran) << classPath.error;
    ASSERT_TRUE(classPathDisk.ran) << classPathDisk.error;

    // C1 is the reference: no decorator, no grant, so there is no tier at all.
    EXPECT_FALSE(plain.tierPresent) << "a plain run with no cache and no decorator has no tier";
    EXPECT_EQ(plain.decoratorFactoryCalls, 0u);

    // C2: the disk tier must actually SERVE the run - a decorator that is
    // accepted and then never reached would satisfy the value cell vacuously.
    EXPECT_EQ(disk.decoratorFactoryCalls, 1u) << "the factory is called ONCE, at Create";
    EXPECT_GT(disk.diskMissQuartets, 0u)
        << "the store recomputed nothing, so it never served the run";
    EXPECT_TRUE(disk.tierPresent) << "a set decorator engages the engine tier even at a zero grant";
    EXPECT_EQ(disk.cache.maxCacheBytes, 1u) << "the RAM tier is the smallest accepted budget";
    EXPECT_GT(disk.storeBytes, 0u) << "the store file was never written";
    EXPECT_EQ(MaxAbsDiff(disk.fock, plain.fock), 0.0)
        << "the store returns the engine's own bytes verbatim, so swapping the tier cannot move a "
           "value";

    // C3 vs C4: the class-aware path disengages the WHOLE engine tier, so the
    // disk request is dropped before the factory is reached - no store opened,
    // no decorator built, and the run is C3's run exactly.
    EXPECT_FALSE(classPath.tierPresent) << "the class path keeps the whole engine tier disengaged";
    EXPECT_FALSE(classPathDisk.tierPresent) << "asking for the disk tier does not engage it here";
    EXPECT_EQ(classPathDisk.decoratorFactoryCalls, 0u)
        << "the class path must drop the request BEFORE the factory, not inside it";
    EXPECT_EQ(classPathDisk.storeBytes, 0u) << "no store may be opened on the class path";
    EXPECT_EQ(MaxAbsDiff(classPathDisk.fock, classPath.fock), 0.0)
        << "C4 must run exactly what C3 runs";
    EXPECT_EQ(classPathDisk.call1.fp64QuartetCount, classPath.call1.fp64QuartetCount);
    EXPECT_EQ(classPathDisk.call2.fp64QuartetCount, classPath.call2.fp64QuartetCount);

    std::error_code ignored;
    std::filesystem::remove(StorePath("r36_disk_c2"), ignored);
    std::filesystem::remove(StorePath("r36_disk_c3"), ignored);
    std::filesystem::remove(StorePath("r36_disk_c4"), ignored);
}
