// The read-vs-recompute crossover measurement: per class of the
// H2O/cc-pVDZ disk-integral surface, wall time of serving the whole class
// from the store vs recomputing it through the underlying engine. This is
// the measurement, not an assumption.
//
// One decorator persists the classes once (warm-up), then every serve call
// is a full hit (checksums included). The recompute side calls the raw
// engine directly - the miss path's append work is not the crossover that
// matters. Plain main() program (the preset_sweep.cpp pattern), not a
// google-benchmark run: the crossover table is the deliverable.
//
// Default run: H2O/cc-pVDZ (the classes the store tests persist).
// `--alkane` adds C12/STO-3G (the large-shell story, ~50 shells) - opt-in,
// because persisting its classes is minutes, not seconds.
//
// The RI-J leg measures the SAME crossover for the disk rung's chunk
// stream: per
// BuildFock iteration, wall time of serving every chunk from
// the chunked store (read + FNV-1a-64 checksum + dataset open, twice
// per iteration - the two-pass shape of DiskRiFockBuilder) vs recomputing
// the same chunks through the BuildRiTensorChunk entry point. The two
// sides differ ONLY in the chunk source; the common two-GEMM base and the
// direct-exchange half cancel (identical code on both
// rungs) and are not timed. The checksum column measures the FNV-1a-64
// tax separately over the same byte volume (data-independent linear hash
// over the real per-chunk sizes); "read ms" is the serve total minus it.
// The H2O/def2-SVP rows smoke the leg; the C24H50/def2-SVP and C60 rows
// of the reinstatement path are the clean-machine runs (the
// in-memory monolithic reference is the cap-broken profile).
#include "alkane_sto3g.hpp"
#include "h2o_ccpvdz.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/storage/cached_eri_batch_engine.hpp"
#include "qcx/storage/disk_ri_fock_build.hpp"
#include "qcx/storage/fingerprint.hpp"
#include "qcx/storage/ri_tensor_chunk_store.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using qcx::integrals::BuildShellPairs;
using qcx::integrals::ComputeEriBatch;
using qcx::integrals::ShellFunctionCount;
using qcx::integrals::ShellPairList;
using qcx::integrals::ShellQuartet;
using qcx::storage::CachedEriBatchEngine;

constexpr int kServeReps = 10;
constexpr int kRecomputeReps = 3;

// The class of a canonical quartet.
std::pair<int, int> ClassOf(const ShellPairList& pairList, const ShellQuartet& quartet) {
    return {pairList.shells[quartet.i].angularMomentum + pairList.shells[quartet.j].angularMomentum,
            pairList.shells[quartet.k].angularMomentum +
                pairList.shells[quartet.l].angularMomentum};
}

// The canonical quartets of one class (the same enumeration as the store
// tests' helpers).
std::vector<ShellQuartet> ClassQuartets(const ShellPairList& pairList, int lBra, int lKet) {
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

                    if (ClassOf(pairList, candidate) != std::make_pair(lBra, lKet))
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

std::size_t BlockElementCount(const ShellPairList& pairList, const ShellQuartet& quartet) {
    return ShellFunctionCount(pairList.shells[quartet.i]) *
           ShellFunctionCount(pairList.shells[quartet.j]) *
           ShellFunctionCount(pairList.shells[quartet.k]) *
           ShellFunctionCount(pairList.shells[quartet.l]);
}

// Mean wall time of one call, over reps.
double MeasureMs(const std::function<void()>& call, int reps) {
    double total = 0.0;

    for (int i = 0; i < reps; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        call();
        const auto end = std::chrono::steady_clock::now();
        total += std::chrono::duration<double, std::milli>(end - start).count();
    }

    return total / static_cast<double>(reps);
}

void RunSystem(const std::string& name,
               const qcx::molecule::Molecule& molecule,
               const qcx::basisset::BasisSet& basis,
               const std::vector<std::pair<int, int>>& classes) {
    auto pairList = BuildShellPairs(molecule, basis);

    if (!pairList.has_value())
    {
        std::printf(
            "%s: BuildShellPairs failed: %s\n", name.c_str(), pairList.error().message.c_str());
        return;
    }

    // The store path carries the system name (the two systems of one run
    // must not share a file), and any stale file from an earlier run is
    // removed: EriStore::Create refuses to clobber an existing file. The
    // '/' of the display names ("H2O/cc-pVDZ") must not reach the path -
    // it implies a directory that does not exist, and HDF5 cannot create
    // the file under it (the 4d7fe9d regression that stopped the leg
    // from running on a clean machine).
    std::string fileName = name;
    std::replace(fileName.begin(), fileName.end(), '/', '_');
    const std::filesystem::path storePath =
        std::filesystem::temp_directory_path() / ("qcx_io_crossover_" + fileName + ".h5");
    std::error_code ignored;
    std::filesystem::remove(storePath, ignored);

    CachedEriBatchEngine::UnderlyingEngine engine = [&](const std::vector<ShellQuartet>& request) {
        return ComputeEriBatch(molecule, basis, request);
    };

    auto cached =
        CachedEriBatchEngine::Create(storePath, molecule, basis, "cc-pvdz", "", engine, {}, {});

    if (!cached.has_value())
    {
        std::printf("%s: CachedEriBatchEngine::Create failed: %s\n",
                    name.c_str(),
                    cached.error().message.c_str());
        return;
    }

    // Warm-up: persist every class through the decorator (one engine call
    // per class - the recompute+append path).
    std::vector<std::vector<ShellQuartet>> classQuartets;

    for (const auto& klass : classes)
    {
        classQuartets.push_back(ClassQuartets(*pairList, klass.first, klass.second));
        const auto persisted = cached->ComputeEriBatch(classQuartets.back());

        if (!persisted.has_value())
        {
            std::printf("%s: persist class (%d,%d) failed: %s\n",
                        name.c_str(),
                        klass.first,
                        klass.second,
                        persisted.error().message.c_str());
            return;
        }
    }

    std::printf("%s\n%-6s %10s %12s %12s %14s %8s\n",
                name.c_str(),
                "class",
                "quartets",
                "elements",
                "serve ms",
                "recompute ms",
                "serves?");

    bool anyReadFaster = false;

    for (std::size_t ci = 0; ci < classes.size(); ++ci)
    {
        const auto& quartets = classQuartets[ci];
        std::size_t elements = 0;

        for (const ShellQuartet& quartet : quartets)
        {
            elements += BlockElementCount(*pairList, quartet);
        }

        const double serveMs = MeasureMs(
            [&]() {
                const auto served = cached->ComputeEriBatch(quartets);

                if (!served.has_value())
                {
                    std::printf("%s: serve class (%d,%d) failed: %s\n",
                                name.c_str(),
                                classes[ci].first,
                                classes[ci].second,
                                served.error().message.c_str());
                    std::exit(2);
                }
            },
            kServeReps);
        const double recomputeMs = MeasureMs(
            [&]() {
                const auto recomputed = engine(quartets);

                if (!recomputed.has_value())
                {
                    std::printf("%s: recompute class (%d,%d) failed: %s\n",
                                name.c_str(),
                                classes[ci].first,
                                classes[ci].second,
                                recomputed.error().message.c_str());
                    std::exit(2);
                }
            },
            kRecomputeReps);

        const bool readsFaster = serveMs < recomputeMs;
        anyReadFaster = anyReadFaster || readsFaster;

        std::printf("%-6s %10zu %12zu %12.3f %14.3f %8s\n",
                    ("(" + std::to_string(classes[ci].first) + "," +
                     std::to_string(classes[ci].second) + ")")
                        .c_str(),
                    quartets.size(),
                    elements,
                    serveMs,
                    recomputeMs,
                    readsFaster ? "yes" : "no");
    }

    std::printf("%s: %s\n",
                name.c_str(),
                anyReadFaster ? "disk reads beat recompute in at least one class"
                              : "recompute beats disk reads in every class");

    // Windows file lock: the store must be gone before the temp path can
    // be reused by a later system. The engine's EriStore file handle
    // closes when the engine value is destroyed, so the engine is moved
    // into a discarded temporary here.
    std::ignore = std::move(cached);
    std::error_code removeError;
    std::filesystem::remove(storePath, removeError);
}

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

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

// The RI-J leg of the crossover: per-iteration
// wall time of the disk rung's chunk stream vs the recompute rung's,
// over the SAME chunk plan. One measured iteration replays the two-pass
// shape of the streamed builder (disk_ri_fock_build.hpp): pass 1 reads
// every chunk, pass 2 reads every chunk again (the real per-iteration
// volume is 2x; the second pass hits the OS page cache, exactly as
// here). The store molecule is taken by value (the builder Create moves
// it in); the loads verify against `molecule`, an identical-geometry
// fixture.
void RunRiJLeg(std::string_view name,
               qcx::molecule::Molecule storeMolecule,
               const qcx::molecule::Molecule& molecule,
               const qcx::basisset::BasisSet& orbitalBasis,
               const qcx::basisset::BasisSet& auxBasis,
               std::string_view orbitalName,
               std::string_view auxName,
               std::size_t chunkBytes) {
    const std::string label(name);

    auto core = BuildCoreHamiltonian(molecule, orbitalBasis);

    if (!core.has_value())
    {
        std::printf(
            "%s: core Hamiltonian failed: %s\n", label.c_str(), core.error().message.c_str());
        return;
    }

    // Persist once: the Create-time chunked build + chunked store is the
    // warm-up, and its chunk plan is what the reps
    // replay. Any stale file from an earlier run is removed first (the
    // store is append-only and refuses to clobber).
    const std::filesystem::path storePath =
        std::filesystem::temp_directory_path() / ("qcx_io_crossover_ri_" + label + ".h5");
    std::error_code ignored;
    std::filesystem::remove(storePath, ignored);

    qcx::storage::DiskRiFockOptions options;
    options.chunkBytes = chunkBytes;
    auto builder = qcx::storage::DiskRiFockBuilder::Create(storePath,
                                                           std::move(storeMolecule),
                                                           orbitalBasis,
                                                           auxBasis,
                                                           orbitalName,
                                                           auxName,
                                                           *core,
                                                           options);

    if (!builder.has_value())
    {
        std::printf("%s: DiskRiFockBuilder::Create failed: %s\n",
                    label.c_str(),
                    builder.error().message.c_str());
        return;
    }

    const std::vector<qcx::storage::RiChunkMeta>& plan = builder->Chunks();
    const std::size_t n2 = plan.front().orbitalFunctionCount * plan.front().orbitalFunctionCount;
    const std::size_t kTot = plan.back().auxFunctionEnd;
    const double volumeMiB =
        2.0 * static_cast<double>(kTot) * static_cast<double>(n2) * 8.0 / (1024.0 * 1024.0);

    // The streaming side: two passes of real chunk loads (each load
    // carries its dataset open, its read, and the store's own fused
    // FNV-1a-64 checksum verification).
    const double serveMs = MeasureMs(
        [&]() {
            for (int pass = 0; pass < 2; ++pass)
            {
                for (const qcx::storage::RiChunkMeta& meta : plan)
                {
                    auto chunk = qcx::storage::LoadRiTensorChunk(
                        storePath, molecule, orbitalName, auxName, meta);

                    if (!chunk.has_value())
                    {
                        std::printf("%s: serve load failed: %s\n",
                                    label.c_str(),
                                    chunk.error().message.c_str());
                        std::exit(2);
                    }
                }
            }
        },
        kServeReps);

    // The checksum column: the FNV-1a-64 tax over the SAME per-chunk byte
    // volume, measured on its own. The hash is data-independent (linear
    // over the stream), so the standing buffer's content does not matter;
    // the sink's address escapes into the callable, so the pass cannot be
    // optimized away.
    std::size_t maxChunkBytes = 0;

    for (const qcx::storage::RiChunkMeta& meta : plan)
    {
        maxChunkBytes =
            std::max(maxChunkBytes, (meta.auxFunctionEnd - meta.auxFunctionStart) * n2 * 8);
    }

    std::vector<std::byte> hashed(maxChunkBytes);
    double hashSink = 0.0;
    const double hashMs = MeasureMs(
        [&]() {
            for (int pass = 0; pass < 2; ++pass)
            {
                for (const qcx::storage::RiChunkMeta& meta : plan)
                {
                    const std::size_t metaBytes = std::max<std::size_t>(
                        1, (meta.auxFunctionEnd - meta.auxFunctionStart) * n2 * 8);
                    hashSink += static_cast<double>(
                        qcx::storage::internal::Fnv1a64(hashed.data(), metaBytes));
                }
            }
        },
        kServeReps);

    // The recompute side: the same two passes through the chunked
    // build entry point (the recompute rung's per-iteration 3c cost).
    const double recomputeMs = MeasureMs(
        [&]() {
            for (int pass = 0; pass < 2; ++pass)
            {
                for (const qcx::storage::RiChunkMeta& meta : plan)
                {
                    qcx::integrals::RiEngineOptions chunkOptions;
                    auto chunk = qcx::integrals::BuildRiTensorChunk(molecule,
                                                                    orbitalBasis,
                                                                    auxBasis,
                                                                    meta.auxShellStart,
                                                                    meta.auxShellEnd,
                                                                    chunkOptions);

                    if (!chunk.has_value())
                    {
                        std::printf("%s: recompute chunk failed: %s\n",
                                    label.c_str(),
                                    chunk.error().message.c_str());
                        std::exit(2);
                    }
                }
            }
        },
        kRecomputeReps);

    const double readMs = serveMs - hashMs;
    const bool streamsFaster = serveMs < recomputeMs;
    (void)hashSink;

    std::printf("%-34s %6zu %10.1f %12.3f %12.3f %12.3f %14.3f %8s\n",
                label.c_str(),
                plan.size(),
                volumeMiB,
                serveMs,
                hashMs,
                readMs,
                recomputeMs,
                streamsFaster ? "yes" : "no");
    std::printf("%s: %s\n",
                label.c_str(),
                streamsFaster ? "streaming the stored chunks beats recomputing them"
                              : "recomputing the chunks beats streaming them");
}

} // namespace

int main(int argc, char** argv) {
    const bool withAlkane = argc > 1 && std::string(argv[1]) == "--alkane";
    const std::vector<std::pair<int, int>> classes = {{0, 0}, {0, 1}, {0, 2}, {1, 1}};

    const auto water = qcx::testing::MakeH2oCcpvdz();
    const auto waterBasis = qcx::testing::MakeH2oCcpvdzBasis();

    if (water.has_value() && waterBasis.has_value())
    {
        RunSystem("H2O/cc-pVDZ", *water, *waterBasis, classes);
    } else
    {
        std::printf("H2O/cc-pVDZ fixtures failed: %s / %s\n",
                    water.has_value() ? "" : water.error().message.c_str(),
                    waterBasis.has_value() ? "" : waterBasis.error().message.c_str());
        return 1;
    }

    if (withAlkane)
    {
        const auto alkane = qcx::testing::MakeAlkaneSto3g(12);
        const auto alkaneBasis = qcx::testing::MakeAlkaneSto3gBasis();

        if (alkane.has_value() && alkaneBasis.has_value())
        {
            RunSystem("C12/STO-3G", *alkane, *alkaneBasis, classes);
        } else
        {
            std::printf("C12/STO-3G fixtures failed: %s / %s\n",
                        alkane.has_value() ? "" : alkane.error().message.c_str(),
                        alkaneBasis.has_value() ? "" : alkaneBasis.error().message.c_str());
            return 1;
        }
    }

    // The RI-J leg: the streaming-vs-recompute
    // crossover instrument. The H2O/def2-SVP rows smoke the leg - the
    // C24H50/def2-SVP and C60 rows of the reinstatement path are the
    // clean-machine runs. Two chunk modes per
    // system: per-shell chunks (the floor) and a single chunk.
    if (!qcx::integrals::SupportsL(4))
    {
        std::printf("RI-J leg skipped: this build's kMaxEngineL is below the jfit g shells\n");
        return 0;
    }

    const std::filesystem::path basisRoot(QcxBasisDataDir);
    const std::array<int, 2> elements{1, 8};
    auto orbital =
        qcx::basisset::ParseNwchemDirectoryFiltered((basisRoot / "def2-svp").string(), elements);
    auto aux = qcx::basisset::ParseNwchemDirectoryFiltered(
        (basisRoot / "def2-universal-jfit").string(), elements);
    auto storeMoleculeA = qcx::testing::MakeH2oSto3g();
    auto storeMoleculeB = qcx::testing::MakeH2oSto3g();
    auto loadMolecule = qcx::testing::MakeH2oSto3g();

    if (!orbital.has_value() || !aux.has_value() || !storeMoleculeA.has_value() ||
        !storeMoleculeB.has_value() || !loadMolecule.has_value())
    {
        std::printf("RI-J leg fixtures failed: %s / %s / %s / %s / %s\n",
                    orbital.has_value() ? "" : orbital.error().message.c_str(),
                    aux.has_value() ? "" : aux.error().message.c_str(),
                    storeMoleculeA.has_value() ? "" : storeMoleculeA.error().message.c_str(),
                    storeMoleculeB.has_value() ? "" : storeMoleculeB.error().message.c_str(),
                    loadMolecule.has_value() ? "" : loadMolecule.error().message.c_str());
        return 1;
    }

    std::printf("%-34s %6s %10s %12s %12s %12s %14s %8s\n",
                "RI-J chunk stream (Section 6.6)",
                "chunks",
                "MiB/iter",
                "serve ms",
                "hash ms",
                "read ms",
                "recompute ms",
                "streams?");
    RunRiJLeg("H2O-def2-SVP shell-chunks",
              std::move(*storeMoleculeA),
              *loadMolecule,
              *orbital,
              *aux,
              "def2-svp",
              "def2-universal-jfit",
              1);
    RunRiJLeg("H2O-def2-SVP one-chunk",
              std::move(*storeMoleculeB),
              *loadMolecule,
              *orbital,
              *aux,
              "def2-svp",
              "def2-universal-jfit",
              256 * 1024 * 1024);

    return 0;
}
