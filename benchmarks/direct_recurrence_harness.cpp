// The integral-tensor-cache measurement: whether the survived-quartet set of
// the DIRECT Fock path recurs across SCF iterations, and what fraction of the
// run's ERI wall time a compute-once-on-first-occurrence value cache would
// eliminate. That cache is implemented (EriBatchCache in DirectJkFockBuilder)
// - this harness measures BOTH: the recurrence analysis on the uncached pass,
// and the implemented tier on the cached pass (--cache).
//
// A full RHF run through the FockBuilderFn seam with the
// DirectJkFockBuilder at kNormal - the driver's default builder for a plain
// qcx run (BuilderKind::kDirect is the RunInput default) - recording per
// iteration: the screened-in fp64/fp32 quartet counts, the cumulative-union
// hit rate (how many of this iteration's quartets were already computed in
// an earlier iteration), the cache hit counts of the in-memory tier, the
// ERI/contraction/call wall-time split (the FockBuildStats measurement
// seam), and the exact value-store footprint of the union (block sizes
// from the public shell-pair API). The host-memory probe (backend) is
// reported alongside so the footprint can be read against actual available
// RAM. The cached pass sizes the cache from a fresh DetectHostMemory() probe
// (a conservative fraction of the available bytes; the builder clamps to
// half) and reports the effective budget and the fp64/fp32 payload split.
//
// Plain main() like the accuracy-preset sweep; local-only like the other
// benchmarks. Sizes: C20H42 STO-3G (142 functions) by default, C80H162
// STO-3G (562 functions, the 500+ benchmark floor) with --large.

#include "alkane_sto3g.hpp"
#include "qcx/backend/memory_topology.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::MakeAlkaneSto3g;
using qcx::testing::MakeAlkaneSto3gBasis;
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

double Milliseconds(std::chrono::nanoseconds duration) {
    return static_cast<double>(duration.count()) / 1.0e6;
}

// One iteration's measurement: the screened quartet counts, the cumulative-
// union recurrence, the in-memory cache hits (zero on the uncached pass),
// and the wall-time split of the call.
struct IterationRecord {
    int index;
    std::size_t fp64Quartets;
    std::size_t fp32Quartets;
    std::size_t hits; // Keys already in the cumulative union before this call.
    std::size_t unionSize; // Cumulative union size AFTER this call.
    std::size_t cacheHits64; // Served from the fp64 cache lane.
    std::size_t cacheHits32; // Served from the fp32 cache lane.
    double eriMilliseconds;
    double contractMilliseconds;
    double totalMilliseconds;
};

// One pass's totals - the before/after comparison needs them out of RunOne.
struct RunTotals {
    double totalEriMilliseconds = 0.0;
    double totalContractMilliseconds = 0.0;
    double totalCallMilliseconds = 0.0;
};

int gMaxIterations = 100;

// Runs one RHF calculation (converged or max-iteration-bounded - C20H42
// STO-3G does not converge, it settles into a 2-cycle from iteration 6) and
// prints the per-iteration table and the recurrence summary. With useCache
// the run repeats the SCF through a cache-enabled builder (the Fock output
// is bit-identical to the uncached run, so the two passes share one
// trajectory) and the table gains the cache columns. Returns the totals, or
// nullopt on failure. Caveat: the uncached pass runs first and the cached
// pass second, so the second pass enjoys warm OS page cache/TLB; and the
// bit-identical-Fock claim holds strictly for the serial fallback - the
// parallel path's last bits are schedule-dependent.
std::optional<RunTotals> RunOne(std::string_view label, std::size_t carbonCount, bool useCache) {
    std::cout << "\n=== " << label << " (C" << carbonCount << "H" << 2 * carbonCount + 2
              << " STO-3G) ===\n";

    auto molecule = MakeAlkaneSto3g(carbonCount);

    if (!molecule.has_value())
    {
        std::cerr << "molecule construction failed: " << molecule.error().message << "\n";
        return std::nullopt;
    }

    auto basis = MakeAlkaneSto3gBasis();

    if (!basis.has_value())
    {
        std::cerr << "basis construction failed: " << basis.error().message << "\n";
        return std::nullopt;
    }

    auto core = BuildCoreHamiltonian(*molecule, *basis);

    if (!core.has_value())
    {
        std::cerr << "core Hamiltonian failed: " << core.error().message << "\n";
        return std::nullopt;
    }

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    if (!overlap.has_value())
    {
        std::cerr << "overlap failed: " << overlap.error().message << "\n";
        return std::nullopt;
    }

    qcx::integrals::FockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kNormal;

    if (useCache)
    {
        // The cache sizing policy in action: a FRESH DetectHostMemory() probe
        // (a point-in-time snapshot, not a reservation - memory_topology.hpp),
        // a conservative fraction of the available bytes; the builder clamps
        // to half of available as the final safety. The effective budget is
        // reported in the summary (CacheStats().maxCacheBytes).
        const qcx::backend::HostMemoryInfo host = qcx::backend::DetectHostMemory();
        options.maxCacheBytes =
            static_cast<std::size_t>(0.25 * static_cast<double>(host.availableBytes));
    }

    auto builder = qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, options);

    if (!builder.has_value())
    {
        std::cerr << "builder creation failed: " << builder.error().message << "\n";
        return std::nullopt;
    }

    // The exact value-store footprint of the union needs the per-pair block
    // function products; the public shell-pair API provides both lists.
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        std::cerr << "pair list failed: " << pairList.error().message << "\n";
        return std::nullopt;
    }

    const std::size_t nPairs = pairList->pairs.size();
    std::vector<std::size_t> pairFunctionProducts(nPairs);

    for (std::size_t pair = 0; pair < nPairs; ++pair)
    {
        const qcx::integrals::ShellPairIndex& indices = pairList->pairs[pair];
        pairFunctionProducts[pair] =
            qcx::integrals::ShellFunctionCount(pairList->shells[indices.i]) *
            qcx::integrals::ShellFunctionCount(pairList->shells[indices.j]);
    }

    // No reserve: the union size is the measured quantity (nPairs^2/8
    // buckets would be ~1.4 GB on the large fixture), and a few rehashes
    // are noise next to the ERI work.
    std::unordered_set<std::size_t> unionSet;
    std::vector<std::size_t> keys;
    std::vector<IterationRecord> records;
    double totalEriMilliseconds = 0.0;
    double totalContractMilliseconds = 0.0;
    double totalCallMilliseconds = 0.0;

    qcx::scf::RhfOptions scfOptions;
    scfOptions.maxIterations = gMaxIterations;

    const qcx::scf::FockBuilderFn fockBuilder =
        [&](const Eigen::MatrixXd& density) -> qcx::Result<Eigen::MatrixXd> {
        auto densityTensor = ToTensor(0.5 * density);

        if (!densityTensor.has_value())
        {
            return std::unexpected(densityTensor.error());
        }

        qcx::integrals::FockBuildStats stats;
        stats.quartetKeysOut = &keys;
        auto fock = builder->BuildFock(*densityTensor, nullptr, &stats);

        if (!fock.has_value())
        {
            return std::unexpected(fock.error());
        }

        std::size_t hits = 0;

        for (const std::size_t key : keys)
        {
            if (unionSet.contains(key))
            {
                ++hits;
            } else
            {
                unionSet.insert(key);
            }
        }

        records.push_back(IterationRecord{static_cast<int>(records.size()),
                                          stats.fp64QuartetCount,
                                          stats.fp32QuartetCount,
                                          hits,
                                          unionSet.size(),
                                          stats.cacheHitFp64QuartetCount,
                                          stats.cacheHitFp32QuartetCount,
                                          Milliseconds(stats.eriWallTime),
                                          Milliseconds(stats.contractWallTime),
                                          Milliseconds(stats.totalWallTime)});
        totalEriMilliseconds += Milliseconds(stats.eriWallTime);
        totalContractMilliseconds += Milliseconds(stats.contractWallTime);
        totalCallMilliseconds += Milliseconds(stats.totalWallTime);
        return ToMatrix(*fock);
    };

    const auto runStarted = std::chrono::steady_clock::now();
    const auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*core), scfOptions, fockBuilder);
    const double runWallSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - runStarted).count();

    if (!result.has_value())
    {
        std::cerr << "scf failed: " << result.error().message << "\n";
        return std::nullopt;
    }

    std::cout << "run wall: " << runWallSeconds << " s, converged " << result->converged
              << ", iterations " << result->iterations << ", E = " << result->totalEnergy << "\n\n";
    std::cout << "iter  fp64Quartets  fp32Quartets  hits  unionSize";

    if (useCache)
    {
        std::cout << "  cacheH64  cacheH32  cacheHit%";
    }

    std::cout << "  eriMs  contractMs  totalMs\n";

    for (const IterationRecord& record : records)
    {
        std::cout << std::setw(4) << record.index << std::setw(14) << record.fp64Quartets
                  << std::setw(14) << record.fp32Quartets << std::setw(7) << record.hits
                  << std::setw(12) << record.unionSize;

        if (useCache)
        {
            const std::size_t quartets = record.fp64Quartets + record.fp32Quartets;
            const std::size_t cacheHits = record.cacheHits64 + record.cacheHits32;
            std::cout << std::setw(10) << record.cacheHits64 << std::setw(10) << record.cacheHits32
                      << std::setw(11) << std::fixed << std::setprecision(1)
                      << 100.0 * static_cast<double>(cacheHits) /
                             (quartets > 0 ? static_cast<double>(quartets) : 1.0);
        }

        std::cout << std::setw(8) << std::fixed << std::setprecision(1) << record.eriMilliseconds
                  << std::setw(11) << record.contractMilliseconds << std::setw(9)
                  << record.totalMilliseconds << "\n";
    }

    // The cache-eliminable share: each iteration's ERI time scaled by the
    // fraction of its quartets that were already computed earlier (a
    // compute-once-on-first-occurrence cache would serve those from memory).
    double eliminatedEriMilliseconds = 0.0;
    std::size_t cumulativeQuartets = 0;

    for (std::size_t i = 0; i < records.size(); ++i)
    {
        const IterationRecord& record = records[i];
        const std::size_t quartets = record.fp64Quartets + record.fp32Quartets;
        cumulativeQuartets += quartets;

        if (quartets > 0)
        {
            eliminatedEriMilliseconds += record.eriMilliseconds * static_cast<double>(record.hits) /
                                         static_cast<double>(quartets);
        }
    }

    // The exact fp64 value-store footprint of the union: per quartet, the
    // block is pairFunctionProducts[bra] * pairFunctionProducts[ket]
    // doubles; the fp32 lane would halve its share, but the fp64 number is
    // the honest upper bound.
    std::uint64_t storeBytes = 0;

    for (const std::size_t key : unionSet)
    {
        storeBytes += static_cast<std::uint64_t>(pairFunctionProducts[key / nPairs] *
                                                 pairFunctionProducts[key % nPairs]);
    }

    storeBytes *= 8;

    const qcx::backend::HostMemoryInfo host = qcx::backend::DetectHostMemory();
    std::cout << "\nsummary:\n"
              << "  total quartets computed across the run: " << cumulativeQuartets << "\n"
              << "  union (unique quartets ever computed): " << unionSet.size() << "\n"
              << "  total ERI time: " << totalEriMilliseconds / 1.0e3 << " s\n"
              << "  total contract time: " << totalContractMilliseconds / 1.0e3 << " s\n"
              << "  total BuildFock time: " << totalCallMilliseconds / 1.0e3 << " s\n"
              << "  ERI time eliminated by a first-occurrence cache: "
              << eliminatedEriMilliseconds / 1.0e3 << " s ("
              << 100.0 * eliminatedEriMilliseconds /
                     (totalEriMilliseconds > 0.0 ? totalEriMilliseconds : 1.0)
              << "% of ERI time)\n"
              << "  union value-store footprint (fp64 blocks, exact): " << storeBytes / 1048576
              << " MiB\n";

    if (useCache)
    {
        // The implemented tier's actual numbers: the effective
        // budget after the builder's clamp, the live dual-lane payload, and
        // the cumulative hits - the fp32 lane's share answers the
        // mixed-precision lane-split question. A failed host-memory probe
        // (zero available bytes) makes maxCacheBytes 0 and disables the
        // cache in Create - CacheStats() then returns nullptr and the cached
        // pass degenerates to a second uncached run.
        const qcx::integrals::EriCacheStats* cacheStats = builder->CacheStats();

        if (cacheStats != nullptr)
        {
            std::cout << "  cache: budget " << cacheStats->maxCacheBytes / 1048576
                      << " MiB, fp64 payload " << cacheStats->fp64PayloadBytes / 1048576
                      << " MiB, fp32 payload " << cacheStats->fp32PayloadBytes / 1048576
                      << " MiB, fp64 hits " << cacheStats->fp64HitQuartets << ", fp32 hits "
                      << cacheStats->fp32HitQuartets << "\n";
        } else
        {
            std::cout << "  cache: disabled (host-memory probe failed) - the cached pass "
                         "is a second uncached run\n";
        }
    }

    std::cout << "  host RAM probe: total " << host.totalBytes / 1048576 << " MiB, available "
              << host.availableBytes / 1048576 << " MiB\n";

    RunTotals totals;
    totals.totalEriMilliseconds = totalEriMilliseconds;
    totals.totalContractMilliseconds = totalContractMilliseconds;
    totals.totalCallMilliseconds = totalCallMilliseconds;
    return totals;
}

} // namespace

int main(int argc, char** argv) {
    bool large = false;
    bool cache = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument(argv[i]);

        if (argument == "--large")
        {
            large = true;
        } else if (argument == "--cache")
        {
            cache = true;
        } else if (argument == "--max-iters" && i + 1 < argc)
        {
            gMaxIterations = std::atoi(argv[++i]);
        } else
        {
            std::cerr << "usage: qcx-bench-direct-recurrence [--large] [--cache] "
                         "[--max-iters N]\n";
            return 1;
        }
    }

    const auto small = RunOne("small", 20, false);

    if (!small.has_value())
    {
        return 1;
    }

    if (cache)
    {
        const auto smallCached = RunOne("small-cached", 20, true);

        if (!smallCached.has_value())
        {
            return 1;
        }

        // The before/after the cache-benefit question asks for: the
        // same fixture, the same SCF trajectory (the cached path is
        // bit-identical), totals side by side. Caveat: the uncached pass
        // runs first, the cached pass second - warm OS page cache/TLB
        // favor the second pass - and bit-identical Fock holds strictly
        // only for the serial fallback (the parallel path's last bits are
        // schedule-dependent).
        std::cout << "\n=== cached vs uncached (small fixture) ===\n"
                  << "  total ERI time: " << small->totalEriMilliseconds / 1.0e3
                  << " s (uncached) -> " << smallCached->totalEriMilliseconds / 1.0e3
                  << " s (cached)\n"
                  << "  total contract time: " << small->totalContractMilliseconds / 1.0e3
                  << " s (uncached) -> " << smallCached->totalContractMilliseconds / 1.0e3
                  << " s (cached)\n"
                  << "  total BuildFock time: " << small->totalCallMilliseconds / 1.0e3
                  << " s (uncached) -> " << smallCached->totalCallMilliseconds / 1.0e3
                  << " s (cached)\n";
    }

    if (large)
    {
        const auto largeResult = RunOne("large", 80, cache);

        if (!largeResult.has_value())
        {
            return 1;
        }
    }

    return 0;
}
