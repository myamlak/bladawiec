// The direct-vs-dense Fock-build benchmark: DirectJkFockBuilder::BuildFock
// against the retained dense path (BuildEriTensorGeneral + a supermatrix
// contraction) on H2O/def2-SVP - the acceptance target is at least 2x the
// dense path at this size.
//
// Plus the per-element rescreen rows: the per-element filter's before-number
// (single-level: usePerElementScreening=false, the legacy six-block gate) and
// after-number (two-level: default flag on) on H2O/def2-SVP and on the
// production-size fixture C80H162/STO-3G (562 functions; no dense supermatrix
// leg at that size - the dense path would need ~800 GB). Each row reports one
// stats capture (total/eri/contract wall time, per-lane quartet counts, ERI
// cache hits, survived-quartet keys, element drops) before the timed loop.
//
// Plus the Tensor <-> Eigen conversion micro-benchmark: the boundary copies
// between Tensor and Eigen::MatrixXd, comparing the old
// i-outer/j-inner loop order against the storage-order-aware helper loop
// (j-outer/i-inner for TensorToEigen, i-outer kept for EigenToTensor) and
// the Eigen::Map row-major-view alternative, across n = 50, 500, 1000,
// 2000 - spanning the L2/L3 boundary (n x n doubles: 20 KB, 2 MB, 8 MB,
// 32 MB). The crossover n below which the loop order does not matter is the
// number this section exists to record.
//
// Plus the AccumulateBlock micro-benchmark: FockContractor::AccumulateBlock
// in isolation at a representative block size. The kernel is ONE template
// compiled twice (internal/fock_contract_kernel.hpp): the scalar copy (no
// /arch flag) and the /arch:AVX2 copy (auto-vectorization only - no hand
// intrinsics exist). Both copies register the same per-iteration batch range
// over the (d,d,d,d) fixture - BenchAccumBlockScalar (the no-/arch baseline)
// and BenchAccumBlockAvx2 (the /arch:AVX2 copy) each at 1 / 8 / 64 kernel
// calls per iteration. Batch 1 is the original ~5 µs floor row
// (MEASURED-INCONCLUSIVE at the timing-resolution floor); batches 8 and 64
// are the larger fixture - they lift the per-iteration time into the
// 40-320 µs class where the timer read stops dominating, which is what the
// scalar-vs-AVX2 ratio at those batches resolves the question with. The
// hand-intrinsics row is gated on that ratio (not written).

#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_contract_kernel.hpp"
#include "internal/tensor_eigen_bridge.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <utility>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

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

struct FockFixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd core;
    Eigen::MatrixXd density;
    CpuTensor2 densityTensor;
    qcx::integrals::DirectJkFockBuilder direct;
    qcx::integrals::DirectJkFockBuilder
        singleLevel; ///< usePerElementScreening=false (legacy gate).
    Eigen::MatrixXd coulomb; ///< The dense supermatrix of (uv|ws).
    Eigen::MatrixXd exchange; ///< The dense exchange supermatrix (u w|v s).
    Eigen::MatrixXd dVec;
    Eigen::Index eigenN;
};

std::unique_ptr<FockFixture> gFixture;

void PrepareFixture(benchmark::State& state) {
    if (gFixture != nullptr)
    {
        return;
    }

    auto molecule = MakeH2oSto3g();
    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / "def2-svp").string());

    if (!molecule.has_value() || !basis.has_value())
    {
        state.SkipWithError("fixture construction failed");
        return;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        state.SkipWithError(core.error().message.c_str());
        return;
    }

    const std::size_t n = core->Shape()[0];
    Eigen::MatrixXd density =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        density(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 0.5;
    }

    auto densityTensor = ToTensor(density);

    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto direct =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, directOptions);

    qcx::integrals::FockBuildOptions singleLevelOptions = directOptions;
    singleLevelOptions.usePerElementScreening = false;
    auto singleLevel =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, singleLevelOptions);

    qcx::integrals::EriDenseOptions eriOptions;
    eriOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto eri = qcx::integrals::BuildEriTensorGeneral(*molecule, *basis, eriOptions);

    if (!densityTensor.has_value() || !direct.has_value() || !singleLevel.has_value() ||
        !eri.has_value())
    {
        state.SkipWithError("builder preparation failed");
        return;
    }

    const Eigen::Index eigenN = static_cast<Eigen::Index>(n);
    Eigen::MatrixXd coulomb(eigenN * eigenN, eigenN * eigenN);
    Eigen::MatrixXd exchange(eigenN * eigenN, eigenN * eigenN);

    for (Eigen::Index mu = 0; mu < eigenN; ++mu)
    {
        for (Eigen::Index nu = 0; nu < eigenN; ++nu)
        {
            const Eigen::Index row = mu * eigenN + nu;

            for (Eigen::Index l = 0; l < eigenN; ++l)
            {
                for (Eigen::Index s = 0; s < eigenN; ++s)
                {
                    const Eigen::Index col = l * eigenN + s;
                    coulomb(row, col) = (*eri)(mu, nu, l, s);
                    exchange(row, col) = (*eri)(mu, l, s, nu);
                }
            }
        }
    }

    Eigen::MatrixXd dVec(eigenN * eigenN, 1);

    for (Eigen::Index mu = 0; mu < eigenN; ++mu)
    {
        for (Eigen::Index nu = 0; nu < eigenN; ++nu)
        {
            dVec(mu * eigenN + nu, 0) = density(mu, nu);
        }
    }

    gFixture = std::make_unique<FockFixture>(FockFixture{std::move(*molecule),
                                                         std::move(*basis),
                                                         ToMatrix(*core),
                                                         std::move(density),
                                                         std::move(*densityTensor),
                                                         std::move(*direct),
                                                         std::move(*singleLevel),
                                                         std::move(coulomb),
                                                         std::move(exchange),
                                                         std::move(dVec),
                                                         eigenN});
}

// One BuildFock with a stats capture, printed as one report row: the
// total/eri/contract wall-time split (screen + infrastructure is the
// total-minus-eri-minus-contract residual at k = 1 - there is no separate
// screenWallTime field), the serialized merge-chain time
// (FockBuildStats::mergeWallTime - a split of the contract phase, never a
// separate span), the per-lane quartet counts, the ERI cache hits,
// the survived-quartet key count, and the per-element drops (0 on the
// single-level path by the flag contract). Under the k > 1 concurrent-slot
// regime (stats.concurrentSlots > 1) the eri/contract fields are
// SLOT-ACCUMULATED sums of the overlapping per-slot spans - not wall spans -
// so the k = 1 identity (split sum <= total, the screen+infra residual) is
// dropped there instead of printing its meaningless negative artifact; total
// (one clock over the call) and merge (strictly serialized by the gate) stay
// the wall-comparable numbers.
void ReportSingleBuildStats(const char* label, const qcx::integrals::FockBuildStats& stats) {
    const double totalMs = std::chrono::duration<double, std::milli>(stats.totalWallTime).count();
    const double eriMs = std::chrono::duration<double, std::milli>(stats.eriWallTime).count();
    const double contractMs =
        std::chrono::duration<double, std::milli>(stats.contractWallTime).count();
    const double mergeMs = std::chrono::duration<double, std::milli>(stats.mergeWallTime).count();
    const std::size_t keys = stats.quartetKeysOut != nullptr ? stats.quartetKeysOut->size() : 0;

    std::cout << label << ": total " << totalMs << " ms";

    if (stats.concurrentSlots > 1)
    {
        std::cout << " (wall) | k " << stats.concurrentSlots << " | eri " << eriMs
                  << " ms (slot-accumulated, NOT wall) | contract " << contractMs
                  << " ms (slot-accumulated, incl. its gate waits) | merge " << mergeMs
                  << " ms (serialized wall)";
    } else
    {
        const double residualMs = totalMs - eriMs - contractMs;
        std::cout << " | eri " << eriMs << " ms | contract " << contractMs << " ms | merge "
                  << mergeMs << " ms | screen+infra " << residualMs << " ms";
    }

    std::cout << " | fp64Q " << stats.fp64QuartetCount << " | fp32Q " << stats.fp32QuartetCount
              << " | cacheHits "
              << (stats.cacheHitFp64QuartetCount + stats.cacheHitFp32QuartetCount) << " | keys "
              << keys << " | elementDrops " << stats.elementDrops << '\n';
}

// The mode line: one line per alkane builder at fixture build, printing the
// Create-time mode record - the rung, the authorized k (concurrentSlots),
// the clamp-origin team read (defaultTeamSize; k == team means the fired k
// was team-clamped, below it budget-clamped), the fitted batch cap, and the
// charged estimate and reservation. Rows without a workspace budget carry no
// mode record and print nothing (the k = 1 legacy path needs no regime
// label).
void ReportModeInfo(const char* label, const qcx::integrals::DirectJkFockBuilder& builder) {
    const std::optional<qcx::integrals::FockModeInfo>& info = builder.ModeInfo();

    if (!info.has_value())
    {
        return;
    }

    const char* modeName =
        info->mode == qcx::integrals::FockBuildMode::kLightPath
            ? "kLightPath"
            : (info->mode == qcx::integrals::FockBuildMode::kDisk ? "kDisk" : "kFastPath");
    const bool teamClamped = info->defaultTeamSize != 0 && info->concurrentSlots != 0 &&
                             info->concurrentSlots >= info->defaultTeamSize;
    std::cout << label << " mode: " << modeName << " | k " << info->concurrentSlots << " | team "
              << info->defaultTeamSize << (teamClamped ? " (team-clamped)" : " (budget-clamped)")
              << " | cap " << (info->maxBatchBytes / (1024 * 1024)) << " MiB | predicted "
              << (info->predictedBytes / (1024 * 1024)) << " MiB | reserved "
              << (info->reservedBytes / (1024 * 1024)) << " MiB\n";
}

// The per-configuration run: one stats capture (the record row) followed by
// the timed BuildFock loop. GoogleBenchmark invokes the function several
// times per registration, so the capture is guarded to fire once per label
// (the call sites pass string literals - stable addresses).
template <typename Builder>
void RunTimedFock(benchmark::State& state,
                  const char* label,
                  const Builder& builder,
                  const CpuTensor2& densityTensor) {
    static std::set<const char*> gCapturedLabels;

    if (gCapturedLabels.insert(label).second)
    {
        std::vector<std::size_t> keys;
        qcx::integrals::FockBuildStats stats;
        stats.quartetKeysOut = &keys;
        auto capture = builder.BuildFock(densityTensor, nullptr, &stats);

        if (!capture.has_value())
        {
            state.SkipWithError(capture.error().message.c_str());
            return;
        }

        ReportSingleBuildStats(label, stats);
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        auto fock = builder.BuildFock(densityTensor);

        if (!fock.has_value())
        {
            state.SkipWithError(fock.error().message.c_str());
            return;
        }

        sink += (*fock)(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

// The alkane workspace budget: the k > 1 slot count fires only on the
// budgeted fast path (k = min(max(1, remaining / cap), team)). The band must
// clear the budget decision's a-priori exclusion. That gate used to be the
// all-survive pattern bound - AllSurvivePatternBytes(nPairs) = 8 * nPairs *
// (nPairs + 1) / 2 - and the alkane geometry of 402 shells (the NWChem SP
// carbon shell parses into two shells, basis_set.cpp ParseShellType returns
// {0,1}; 81,003 canonical pairs) bounded it at 26,246,268,048 B (24.44 GiB):
// a 12 GiB band then excluded the count unconditionally (dump-proven
// excluded=1, chunkPairs=6) and the all-survive-bound light rung produced the
// wall-serial 341-s run, so the band was raised to 25 GiB. The exclusion now
// counts the Schwarz-screened neighbor pattern itself
// (CountSchwarzSurvivingPairs, fock_build.cpp decision site): the ~4.7M real
// rows (~38 MB) admit any band above them, and the band returns to 12 GiB
// (the timed re-measurement at that band is still pending). The null-budget
// rows (H2O fixture) keep the legacy k = 1 path - the k = 1 byte pins stay
// intact by construction.
inline constexpr std::size_t kAlkaneWorkspaceBudgetBytes = std::size_t{12} * 1024 * 1024 * 1024;

struct AlkaneFixture {
    qcx::molecule::Molecule molecule;
    qcx::basisset::BasisSet basis;
    Eigen::MatrixXd core;
    Eigen::MatrixXd density;
    CpuTensor2 densityTensor;
    qcx::integrals::DirectJkFockBuilder direct;
    qcx::integrals::DirectJkFockBuilder
        singleLevel; ///< usePerElementScreening=false (legacy gate).
};

std::unique_ptr<AlkaneFixture> gAlkaneFixture;

// The production-size fixture: C80H162/STO-3G (562 functions, 402 shells,
// 81,003 canonical shell pairs - the SP-shell split of the NWChem carbon
// basis; see the kAlkaneWorkspaceBudgetBytes narrative). No dense supermatrix
// path at this size (n^4 storage would be ~800 GB) - the alkane leg is
// direct-only.
void PrepareAlkaneFixture(benchmark::State& state) {
    if (gAlkaneFixture != nullptr)
    {
        return;
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(80);
    auto basis = qcx::testing::MakeAlkaneSto3gBasis();

    if (!molecule.has_value() || !basis.has_value())
    {
        state.SkipWithError("alkane fixture construction failed");
        return;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        state.SkipWithError(core.error().message.c_str());
        return;
    }

    const std::size_t n = core->Shape()[0];
    Eigen::MatrixXd density =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        density(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) = 0.5;
    }

    auto densityTensor = ToTensor(density);

    // Both alkane builders run the budgeted fast path, where the k > 1
    // concurrent-slot batch loop fires. Each builder needs its OWN band
    // (12 GiB, kAlkaneWorkspaceBudgetBytes - the counted-pattern exclusion
    // gate): the Create-time fixed point clamps its cap until the k = team
    // charge fits the band and the Reserve is monotone, so a shared budget
    // would leave the second builder nothing and refuse its Create. The
    // budget outlives its Create call (the seam contract); the builder
    // captures the decision, not the pointer.
    auto directWorkspaceBudget = qcx::memory::WorkspaceBudget::Create(kAlkaneWorkspaceBudgetBytes);
    auto singleLevelWorkspaceBudget =
        qcx::memory::WorkspaceBudget::Create(kAlkaneWorkspaceBudgetBytes);

    if (!directWorkspaceBudget.has_value() || !singleLevelWorkspaceBudget.has_value())
    {
        state.SkipWithError("alkane workspace budget creation failed");
        return;
    }

    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    // The cap clamp: the fixed point at the default 512 MB cap would hold up
    // to ~24 GiB live (slots x cap-scale arenas plus stores) - over the
    // 16 GiB benchmark ceiling. 256 MB caps keep the live peak near ~6 GiB
    // while the k > 1 concurrent-slot loop still fires (k = 5 was measured at
    // the 25 GiB band; the 12 GiB band awaits its own timed run), so the
    // throughput question is answered against the 65 s k = 1 baseline.
    directOptions.maxBatchBytes = 256 * 1024 * 1024;
    directOptions.workspaceBudget = &*directWorkspaceBudget;
    auto direct =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, directOptions);

    qcx::integrals::FockBuildOptions singleLevelOptions = directOptions;
    singleLevelOptions.usePerElementScreening = false;
    singleLevelOptions.workspaceBudget = &*singleLevelWorkspaceBudget;
    auto singleLevel =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, singleLevelOptions);

    if (!densityTensor.has_value() || !direct.has_value() || !singleLevel.has_value())
    {
        state.SkipWithError("alkane builder preparation failed");
        return;
    }

    // The mode lines: the fired k and the clamp-origin team read per alkane
    // builder (the k = 5 team-size read; the fixture's early return above
    // keeps this one-shot).
    ReportModeInfo("C80H162/STO-3G two-level", *direct);
    ReportModeInfo("C80H162/STO-3G single-level", *singleLevel);

    gAlkaneFixture = std::make_unique<AlkaneFixture>(AlkaneFixture{std::move(*molecule),
                                                                   std::move(*basis),
                                                                   ToMatrix(*core),
                                                                   std::move(density),
                                                                   std::move(*densityTensor),
                                                                   std::move(*direct),
                                                                   std::move(*singleLevel)});
}

void BenchFockDense(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    const Eigen::Index n = gFixture->eigenN;
    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        // The dense supermatrix path: J = E d, K = E_x d.
        Eigen::MatrixXd fock = gFixture->core;
        const Eigen::MatrixXd j = gFixture->coulomb * gFixture->dVec;
        const Eigen::MatrixXd k = gFixture->exchange * gFixture->dVec;

        for (Eigen::Index mu = 0; mu < n; ++mu)
        {
            for (Eigen::Index nu = 0; nu < n; ++nu)
            {
                const Eigen::Index flat = mu * n + nu;
                fock(mu, nu) += j(flat, 0) - 0.5 * k(flat, 0);
            }
        }

        sink += fock(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

void BenchFockDirect(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    RunTimedFock(state, "H2O/def2-SVP two-level", gFixture->direct, gFixture->densityTensor);
}

void BenchFockDirectSingleLevel(benchmark::State& state) {
    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    RunTimedFock(
        state, "H2O/def2-SVP single-level", gFixture->singleLevel, gFixture->densityTensor);
}

void BenchFockDirectAlkane(benchmark::State& state) {
    PrepareAlkaneFixture(state);

    if (gAlkaneFixture == nullptr)
    {
        return;
    }

    RunTimedFock(
        state, "C80H162/STO-3G two-level", gAlkaneFixture->direct, gAlkaneFixture->densityTensor);
}

void BenchFockDirectAlkaneSingleLevel(benchmark::State& state) {
    PrepareAlkaneFixture(state);

    if (gAlkaneFixture == nullptr)
    {
        return;
    }

    RunTimedFock(state,
                 "C80H162/STO-3G single-level",
                 gAlkaneFixture->singleLevel,
                 gAlkaneFixture->densityTensor);
}

BENCHMARK(BenchFockDense)->Unit(benchmark::kMillisecond)->MinTime(2.0);
BENCHMARK(BenchFockDirect)->Unit(benchmark::kMillisecond)->MinTime(2.0);
BENCHMARK(BenchFockDirectSingleLevel)->Unit(benchmark::kMillisecond)->MinTime(2.0);
BENCHMARK(BenchFockDirectAlkane)->Unit(benchmark::kMillisecond)->MinTime(5.0);
BENCHMARK(BenchFockDirectAlkaneSingleLevel)->Unit(benchmark::kMillisecond)->MinTime(5.0);

// ---------------------------------------------------------------------------
// The AccumulateBlock micro-benchmark.
//
// FockContractor::AccumulateBlock in isolation at a representative block
// size - the (d,d,d,d) self-quartet of the H2O/def2-SVP pair list (625
// contiguous doubles per block: the class whose 5x5x5x5
// multiply-accumulate loops are the SIMD-width question's target). The
// kernel is one template compiled twice: the scalar copy instantiates in
// fock_build.cpp (no /arch flag) and the /arch:AVX2 copy in
// fock_contract_simd.cpp (AccumulateBlockAvx2 - auto-vectorized, no hand
// intrinsics). Running BOTH copies on one binary gives the scalar-vs-AVX2
// comparison at every batch; the AVX2 row skips itself on CPUs without AVX2
// (FockAvx2Available - the runtime dispatch gate). The
// self-quartet exercises the J bra section and all four K targets; the J
// ket section is the same loop shape as J bra and is skipped only because
// bra == ket.
//
// The fixture size knob is the per-iteration batch (state.range(0)): each
// timed iteration performs one pass over a batch of kernel calls, the
// per-chunk sequence of calls a real Fock build makes over one chunk's
// quartets, rotating through the fixture's block pool. Batch 1
// reproduces the original single-call row - the ~5 µs timing-resolution
// floor; batches 8 and 64 are the larger fixture, lifting the per-iteration
// time to ~40 µs / ~320 µs so the scalar-vs-AVX2 comparison resolves the
// question that the floor row left MEASURED-INCONCLUSIVE.
// ---------------------------------------------------------------------------

// The block pool size the batched pass rotates through (each entry is one
// synthetic (d,d,d,d) block). Larger than the L1-resident working set of
// any single call, small enough to stay cache-warm across the rotation.
inline constexpr std::size_t kAccumBlockPoolSize = 16;

// The fixture: the (d,d,d,d) self-quartet over a pool of synthetic
// blocks. The per-element filter is off (pairMaxDensity null), so every
// block element accumulates - the block values are a deterministic,
// non-zero, non-constant dyadic ramp (they shape the arithmetic, not the
// timing).
struct AccumBlockFixture {
    qcx::integrals::ShellPairList pairList;
    Eigen::MatrixXd density;
    Eigen::MatrixXd fock;
    qcx::integrals::ShellQuartet quartet;
    std::size_t pairIndex;
    std::vector<std::vector<double>> blocks;
};

std::unique_ptr<AccumBlockFixture> gAccumBlockFixture;

void PrepareAccumBlockFixture(benchmark::State& state) {
    if (gAccumBlockFixture != nullptr)
    {
        return;
    }

    PrepareFixture(state);

    if (gFixture == nullptr)
    {
        return;
    }

    auto pairList = qcx::integrals::BuildShellPairs(gFixture->molecule, gFixture->basis);

    if (!pairList.has_value())
    {
        state.SkipWithError("pair construction failed");
        return;
    }

    // The oxygen d shell - the only l == 2 shell in H2O/def2-SVP.
    std::size_t dShell = 0;
    bool found = false;

    for (std::size_t i = 0; i < pairList->shells.size(); ++i)
    {
        if (pairList->shells[i].angularMomentum == 2)
        {
            dShell = i;
            found = true;
            break;
        }
    }

    if (!found)
    {
        state.SkipWithError("no d shell in the H2O/def2-SVP fixture");
        return;
    }

    const std::size_t nFuncs = qcx::integrals::ShellFunctionCount(pairList->shells[dShell]);
    std::vector<std::vector<double>> blocks(kAccumBlockPoolSize,
                                            std::vector<double>(nFuncs * nFuncs * nFuncs * nFuncs));

    for (std::vector<double>& block : blocks)
    {
        for (std::size_t k = 0; k < block.size(); ++k)
        {
            block[k] = 0.001 * static_cast<double>(1 + (k % 16));
        }
    }

    const Eigen::Index eigenN = static_cast<Eigen::Index>(pairList->functionCount);
    Eigen::MatrixXd fock = Eigen::MatrixXd::Zero(eigenN, eigenN);

    gAccumBlockFixture = std::make_unique<AccumBlockFixture>(
        AccumBlockFixture{std::move(*pairList),
                          gFixture->density,
                          std::move(fock),
                          qcx::integrals::ShellQuartet{dShell, dShell, dShell, dShell},
                          qcx::integrals::PairIndexOf(dShell, dShell, *pairList),
                          std::move(blocks)});
}

// The shared timed body: one pass over a batch of kernel calls per
// iteration (the benchmark range is the batch - the larger fixture that
// lifts the per-iteration time off the timer-resolution floor). The pass
// rotates through the fixture's block pool: the per-chunk sequence of
// kernel calls a real Fock build performs over one chunk's quartets, each
// with its own ERI block. A sink read of the accumulated Fock matrix keeps
// the work live.
template <typename Kernel> void RunAccumBlock(benchmark::State& state, Kernel kernel) {
    PrepareAccumBlockFixture(state);

    if (gAccumBlockFixture == nullptr)
    {
        return;
    }

    const std::size_t batch = static_cast<std::size_t>(state.range(0));
    const std::vector<std::vector<double>>& blocks = gAccumBlockFixture->blocks;
    const qcx::integrals::internal::FockContractContext context{gAccumBlockFixture->pairList,
                                                                gAccumBlockFixture->density,
                                                                gAccumBlockFixture->fock,
                                                                false,
                                                                false};
    const qcx::integrals::ShellQuartet& quartet = gAccumBlockFixture->quartet;
    const std::size_t pairIndex = gAccumBlockFixture->pairIndex;
    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        for (std::size_t b = 0; b < batch; ++b)
        {
            kernel(blocks[b % kAccumBlockPoolSize].data(), quartet, pairIndex, pairIndex, context);
        }

        sink += gAccumBlockFixture->fock(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

void BenchAccumBlockScalar(benchmark::State& state) {
    RunAccumBlock(state,
                  qcx::integrals::internal::AccumulateBlockKernel<
                      qcx::integrals::internal::AccumulateBlockScalarTag>);
}

void BenchAccumBlockAvx2(benchmark::State& state) {
    if (!qcx::integrals::internal::FockAvx2Available())
    {
        state.SkipWithError("this CPU has no AVX2 (the /arch:AVX2 copy's row)");
        return;
    }

    RunAccumBlock(state, qcx::integrals::internal::AccumulateBlockAvx2);
}

// The per-iteration batch is the fixture size knob: 1 / 8 / 64 kernel
// calls per timed iteration. Batch 1 keeps the original floor row; the
// larger batches are the fixture that lifts the per-iteration time into
// the 50-500 µs class the comparison needs. Both copies share the range,
// so the scalar-vs-AVX2 ratio reads at every batch on one binary.
BENCHMARK(BenchAccumBlockScalar)
    ->RangeMultiplier(8)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2.0);
BENCHMARK(BenchAccumBlockAvx2)
    ->RangeMultiplier(8)
    ->Range(1, 64)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(2.0);

// ---------------------------------------------------------------------------
// The Tensor <-> Eigen conversion micro-benchmark.
//
// Tensor is row-major, Eigen::MatrixXd is column-major. The pre-consolidation
// copies all used i-outer/j-inner; the shared helper (tensor_eigen_bridge.hpp)
// uses j-outer/i-inner for TensorToEigen (contiguous writes into the Eigen
// destination) and keeps i-outer/j-inner for EigenToTensor (contiguous writes
// into the Tensor destination). The Eigen::Map row-major-view alternative
// wraps the tensor's host buffer zero-copy and lets Eigen perform the
// relayout internally. All three variants must produce bit-identical output -
// checked once per size in the fixture.
// ---------------------------------------------------------------------------

// The pre-consolidation loop order, kept here for comparison (i-outer /
// j-inner: contiguous reads from the row-major Tensor, strided writes into
// the column-major Eigen matrix).
Eigen::MatrixXd OldTensorToEigen(const CpuTensor2& tensor) {
    const std::size_t n = tensor.Shape()[0];
    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = tensor(i, j);
        }
    }

    return matrix;
}

// The hand-written EigenToTensor scalar loop (i-outer/j-inner) - the order
// the pre-consolidation copies used AND the order the shared helper keeps for
// this direction (the destination-oriented rule says Tensor, the row-major
// destination, gets the contiguous writes; only TensorToEigen changed). Writes
// into a pre-allocated destination tensor (the benchmark isolates the copy
// cost; production allocates the tensor once per call, equally for every
// variant).
void HandEigenToTensor(const Eigen::MatrixXd& matrix, CpuTensor2& tensorOut) {
    const std::size_t n = static_cast<std::size_t>(matrix.rows());

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            tensorOut(i, j) = matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        }
    }

    tensorOut.MarkHostDirty();
}

// The Eigen::Map row-major-view alternative for the Tensor -> Eigen
// direction: zero-copy view of the tensor's host buffer (row-major), copied
// into a plain column-major MatrixXd - Eigen performs the relayout internally
// instead of the hand-written scalar loop.
Eigen::MatrixXd MapTensorToEigen(CpuTensor2& tensor) {
    const std::size_t n = tensor.Shape()[0];
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        rowMajorView(
            tensor.HostView().data(), static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    return rowMajorView;
}

// The Eigen::Map alternative for the Eigen -> Tensor direction: the tensor's
// host buffer is written through a row-major map directly.
void MapEigenToTensor(const Eigen::MatrixXd& matrix, CpuTensor2& tensorOut) {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> rowMajorView(
        tensorOut.HostView().data(), matrix.rows(), matrix.cols());
    rowMajorView = matrix;
    tensorOut.MarkHostDirty();
}

// The conversion fixture: source tensor + source matrix + a pre-allocated
// destination tensor, with values chosen to be a) non-trivial (a smooth
// dyadic ramp, so neither loop order can alias a constant pattern) and
// b) exactly representable in fp64, so the bit-identity check compares
// exact bytes. Construction verifies all three variants produce bit-identical
// output before any timing starts.
struct ConversionFixture {
    CpuTensor2 tensor;
    CpuTensor2 tensorOut;
    Eigen::MatrixXd matrix;
};

qcx::Result<ConversionFixture> MakeConversionFixture(std::size_t n) {
    auto tensor = CpuTensor2::Create({n, n});

    if (!tensor.has_value())
    {
        return std::unexpected(tensor.error());
    }

    auto tensorOut = CpuTensor2::Create({n, n});

    if (!tensorOut.has_value())
    {
        return std::unexpected(tensorOut.error());
    }

    Eigen::MatrixXd matrix(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            // dyadic: (2i + j) / 2^15, exactly representable in fp64.
            const double value = static_cast<double>(2 * i + j) / 32768.0;
            (*tensor)(i, j) = value;
            matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
        }
    }

    tensor->MarkHostDirty();

    // Bit-identity across all three TensorToEigen variants.
    const Eigen::MatrixXd newCopy = qcx::integrals::internal::TensorToEigen(*tensor);
    const Eigen::MatrixXd oldCopy = OldTensorToEigen(*tensor);
    const Eigen::MatrixXd mapCopy = MapTensorToEigen(*tensor);

    if ((newCopy - oldCopy).cwiseAbs().maxCoeff() != 0.0 ||
        (newCopy - mapCopy).cwiseAbs().maxCoeff() != 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "TensorToEigen variants are not bit-identical"});
    }

    // Bit-identity across the EigenToTensor variants: the hand loop, the Map
    // assignment, and the production helper itself.
    HandEigenToTensor(matrix, *tensorOut);
    const Eigen::MatrixXd handBack = qcx::testing::ToMatrix(*tensorOut);
    MapEigenToTensor(matrix, *tensorOut);
    const Eigen::MatrixXd mapBack = qcx::testing::ToMatrix(*tensorOut);
    auto newTensor = qcx::integrals::internal::EigenToTensor(matrix);

    if (!newTensor.has_value())
    {
        return std::unexpected(newTensor.error());
    }

    const Eigen::MatrixXd newBack = qcx::testing::ToMatrix(*newTensor);

    if ((newBack - handBack).cwiseAbs().maxCoeff() != 0.0 ||
        (newBack - mapBack).cwiseAbs().maxCoeff() != 0.0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "EigenToTensor variants are not bit-identical"});
    }

    return ConversionFixture{std::move(*tensor), std::move(*tensorOut), std::move(matrix)};
}

void BenchConvertTensorToEigenOld(benchmark::State& state) {
    auto fixture = MakeConversionFixture(static_cast<std::size_t>(state.range(0)));

    if (!fixture.has_value())
    {
        state.SkipWithError(fixture.error().message.c_str());
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        const Eigen::MatrixXd m = OldTensorToEigen(fixture->tensor);
        sink += m(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

void BenchConvertTensorToEigenNew(benchmark::State& state) {
    auto fixture = MakeConversionFixture(static_cast<std::size_t>(state.range(0)));

    if (!fixture.has_value())
    {
        state.SkipWithError(fixture.error().message.c_str());
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        const Eigen::MatrixXd m = qcx::integrals::internal::TensorToEigen(fixture->tensor);
        sink += m(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

void BenchConvertTensorToEigenMap(benchmark::State& state) {
    auto fixture = MakeConversionFixture(static_cast<std::size_t>(state.range(0)));

    if (!fixture.has_value())
    {
        state.SkipWithError(fixture.error().message.c_str());
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        const Eigen::MatrixXd m = MapTensorToEigen(fixture->tensor);
        sink += m(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

void BenchConvertEigenToTensorHand(benchmark::State& state) {
    auto fixture = MakeConversionFixture(static_cast<std::size_t>(state.range(0)));

    if (!fixture.has_value())
    {
        state.SkipWithError(fixture.error().message.c_str());
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        HandEigenToTensor(fixture->matrix, fixture->tensorOut);
        sink += fixture->tensorOut(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

void BenchConvertEigenToTensorMap(benchmark::State& state) {
    auto fixture = MakeConversionFixture(static_cast<std::size_t>(state.range(0)));

    if (!fixture.has_value())
    {
        state.SkipWithError(fixture.error().message.c_str());
        return;
    }

    double sink = 0.0;

    for (auto _ : state) // NOLINT(clang-analyzer-deadcode.DeadStores): the GoogleBenchmark loop
                         // variable is deliberately unused.
    {
        MapEigenToTensor(fixture->matrix, fixture->tensorOut);
        sink += fixture->tensorOut(0, 0);
    }

    benchmark::DoNotOptimize(sink);
}

BENCHMARK(BenchConvertTensorToEigenOld)
    ->Arg(50)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);
BENCHMARK(BenchConvertTensorToEigenNew)
    ->Arg(50)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);
BENCHMARK(BenchConvertTensorToEigenMap)
    ->Arg(50)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);
BENCHMARK(BenchConvertEigenToTensorHand)
    ->Arg(50)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);
BENCHMARK(BenchConvertEigenToTensorMap)
    ->Arg(50)
    ->Arg(500)
    ->Arg(1000)
    ->Arg(2000)
    ->Unit(benchmark::kMillisecond)
    ->MinTime(1.0);

} // namespace

BENCHMARK_MAIN();
