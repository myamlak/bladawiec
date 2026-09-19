// The cached batch-engine decorator (cached_eri_batch_engine.hpp):
// reads hit the store's manifest; misses recompute through the injected
// engine callback and append. Bit identity: the engine's bytes are stored
// verbatim and served back in the canonical request order (the shared
// CanonicalizeQuartetOrder - EriBatch::computed is SORTED, never
// request-ordered).

#include "qcx/storage/cached_eri_batch_engine.hpp"

#include "qcx/storage/fingerprint.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <utility>
#include <vector>

namespace qcx::storage {

namespace {

// The block element count of a canonical quartet (the engine's base
// accumulation rule - the store derives the same sizes from its pair list,
// so the decorator's run slices line up with the stored blocks exactly).
std::size_t BlockElementCount(const qcx::integrals::ShellPairList& pairList,
                              const qcx::integrals::ShellQuartet& quartet) {
    return qcx::integrals::ShellFunctionCount(pairList.shells[quartet.i]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.j]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.k]) *
           qcx::integrals::ShellFunctionCount(pairList.shells[quartet.l]);
}

// The class of a canonical quartet (the chunk's run key).
std::pair<int, int> ClassOf(const qcx::integrals::ShellPairList& pairList,
                            const qcx::integrals::ShellQuartet& quartet) {
    return {pairList.shells[quartet.i].angularMomentum + pairList.shells[quartet.j].angularMomentum,
            pairList.shells[quartet.k].angularMomentum +
                pairList.shells[quartet.l].angularMomentum};
}

// The packed value-slice bounds of one class run [runStart, runEnd) of the
// engine's computed order (the prefix sums of the block sizes).
// (runStart, runEnd) are the half-open run bounds - start before end by
// construction.
std::pair<std::size_t, std::size_t> RunValueSlice(
    const qcx::integrals::ShellPairList& pairList,
    const std::vector<qcx::integrals::ShellQuartet>& computed,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::size_t runStart,
    std::size_t runEnd) {
    std::size_t start = 0;

    for (std::size_t i = 0; i < runStart; ++i)
    {
        start += BlockElementCount(pairList, computed[i]);
    }

    std::size_t end = start;

    for (std::size_t i = runStart; i < runEnd; ++i)
    {
        end += BlockElementCount(pairList, computed[i]);
    }

    return {start, end};
}

} // namespace

CachedEriBatchEngine::CachedEriBatchEngine(EriStore store,
                                           std::shared_ptr<qcx::integrals::ShellPairList> pairList,
                                           UnderlyingEngine engine,
                                           UnderlyingCertifiedEngine certifiedEngine,
                                           std::string fingerprint) :
    _store(std::move(store)), _pairList(std::move(pairList)), _engine(std::move(engine)),
    _certifiedEngine(std::move(certifiedEngine)), _fingerprint(std::move(fingerprint)) {}

qcx::Result<CachedEriBatchEngine> CachedEriBatchEngine::Create(
    std::filesystem::path storePath,
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::string_view orbitalBasisName,
    std::string_view auxBasisName,
    UnderlyingEngine engine,
    UnderlyingCertifiedEngine certifiedEngine,
    const StoreOptions& options) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);

    if (!pairList.has_value())
    {
        return std::unexpected(pairList.error());
    }

    // Open the existing store; a missing file means the first run of this
    // system - create it. A corrupt or wrong-system file fails Open loudly
    // (never silently recreated). EriStore is not default-constructible, so
    // the branch resolves the store value directly.
    qcx::Result<EriStore> storeResult = [&]() -> qcx::Result<EriStore> {
        if (!std::filesystem::exists(storePath))
        {
            return EriStore::Create(
                storePath, molecule, basisSet, orbitalBasisName, auxBasisName, options);
        }

        return EriStore::Open(
            storePath, molecule, basisSet, orbitalBasisName, auxBasisName, options);
    }();

    if (!storeResult.has_value())
    {
        return std::unexpected(storeResult.error());
    }

    EriStore store = std::move(*storeResult);

    const std::string fingerprint = ComputeFingerprint(molecule, orbitalBasisName, auxBasisName);

    return CachedEriBatchEngine(
        std::move(store),
        std::make_shared<qcx::integrals::ShellPairList>(std::move(*pairList)),
        std::move(engine),
        std::move(certifiedEngine),
        fingerprint);
}

qcx::Result<EriBatch> CachedEriBatchEngine::ComputeEriBatch(
    const std::vector<ShellQuartet>& quartets, const EriBatchOptions& options) {
    (void)options;

    auto ordered = qcx::integrals::CanonicalizeQuartetOrder(*_pairList, quartets);

    if (!ordered.has_value())
    {
        // Empty request / invalid shells: the engine's own errors.
        return std::unexpected(ordered.error());
    }

    // Full hit: every canonical quartet already stored.
    bool fullHit = true;

    for (const auto& info : *ordered)
    {
        if (!_store.ContainsFp64(info.quartet))
        {
            fullHit = false;
            break;
        }
    }

    if (fullHit)
    {
        const auto t0 = std::chrono::steady_clock::now();
        auto served = _store.LoadBatch(*ordered);
        const auto t1 = std::chrono::steady_clock::now();

        if (!served.has_value())
        {
            return std::unexpected(served.error());
        }

        const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        _stats.hitQuartets += ordered->size();
        _stats.readMs += elapsedMs;

        // Per-class stats only for single-class requests (the benchmark's
        // shape; the sorted order makes "single class" a first/last check).
        const std::pair<int, int> firstClass = ClassOf(*_pairList, ordered->front().quartet);

        if (ClassOf(*_pairList, ordered->back().quartet) == firstClass)
        {
            _stats.hitQuartetsByClass[firstClass] += ordered->size();
            _stats.readMsByClass[firstClass] += elapsedMs;
        }

        return served;
    }

    // Miss: recompute the WHOLE request through the underlying engine and
    // append. The engine's batch is handed back verbatim - bit identity on
    // the miss path is the engine's own guarantee.
    const auto t0 = std::chrono::steady_clock::now();
    auto batch = _engine(quartets);
    const auto t1 = std::chrono::steady_clock::now();

    if (!batch.has_value())
    {
        return std::unexpected(batch.error());
    }

    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    _stats.missQuartets += batch->computed.size();
    _stats.recomputeMs += elapsedMs;

    const std::pair<int, int> firstClass = ClassOf(*_pairList, batch->computed.front());
    const bool singleClass = ClassOf(*_pairList, batch->computed.back()) == firstClass;

    if (singleClass)
    {
        _stats.missQuartetsByClass[firstClass] += batch->computed.size();
        _stats.recomputeMsByClass[firstClass] += elapsedMs;
    }

    // Split the computed order into contiguous class runs and append one
    // chunk per run (the values slice is contiguous per run).
    std::size_t runStart = 0;

    for (std::size_t i = 1; i <= batch->computed.size(); ++i)
    {
        const bool runEnds =
            i == batch->computed.size() || ClassOf(*_pairList, batch->computed[i]) !=
                                               ClassOf(*_pairList, batch->computed[runStart]);

        if (!runEnds)
        {
            continue;
        }

        const auto [valueStart, valueEnd] = RunValueSlice(*_pairList, batch->computed, runStart, i);
        std::vector<ShellQuartet> runQuartets(batch->computed.begin() +
                                                  static_cast<long long>(runStart),
                                              batch->computed.begin() + static_cast<long long>(i));
        std::vector<double> runValues(batch->values.begin() + static_cast<long long>(valueStart),
                                      batch->values.begin() + static_cast<long long>(valueEnd));

        if (auto appended = _store.AppendBatch(runQuartets, runValues); !appended.has_value())
        {
            return std::unexpected(appended.error());
        }

        runStart = i;
    }

    return batch;
}

qcx::Result<CertifiedBatch> CachedEriBatchEngine::ComputeEriBatchCertified(
    const std::vector<ShellQuartet>& quartets, const EriBatchOptions& options) {
    (void)options;

    auto ordered = qcx::integrals::CanonicalizeQuartetOrder(*_pairList, quartets);

    if (!ordered.has_value())
    {
        return std::unexpected(ordered.error());
    }

    // Full hit: every canonical quartet already stored in the fp32 lane.
    bool fullHit = true;

    for (const auto& info : *ordered)
    {
        if (!_store.ContainsFp32(info.quartet))
        {
            fullHit = false;
            break;
        }
    }

    if (fullHit)
    {
        const auto t0 = std::chrono::steady_clock::now();
        auto served = _store.LoadCertifiedBatch(*ordered);
        const auto t1 = std::chrono::steady_clock::now();

        if (!served.has_value())
        {
            return std::unexpected(served.error());
        }

        const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        _stats.hitQuartets += ordered->size();
        _stats.readMs += elapsedMs;
        const std::pair<int, int> firstClass = ClassOf(*_pairList, ordered->front().quartet);

        if (ClassOf(*_pairList, ordered->back().quartet) == firstClass)
        {
            _stats.hitQuartetsByClass[firstClass] += ordered->size();
            _stats.readMsByClass[firstClass] += elapsedMs;
        }

        return served;
    }

    if (!_certifiedEngine)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented, "no certified engine was injected"});
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto batch = _certifiedEngine(quartets);
    const auto t1 = std::chrono::steady_clock::now();

    if (!batch.has_value())
    {
        return std::unexpected(batch.error());
    }

    const double elapsedMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    _stats.missQuartets += batch->computed.size();
    _stats.recomputeMs += elapsedMs;

    const std::pair<int, int> firstClass = ClassOf(*_pairList, batch->computed.front());
    const bool singleClass = ClassOf(*_pairList, batch->computed.back()) == firstClass;

    if (singleClass)
    {
        _stats.missQuartetsByClass[firstClass] += batch->computed.size();
        _stats.recomputeMsByClass[firstClass] += elapsedMs;
    }

    // Append one chunk per contiguous class run: values slice + the
    // matching stored bounds slice.
    std::size_t runStart = 0;

    for (std::size_t i = 1; i <= batch->computed.size(); ++i)
    {
        const bool runEnds =
            i == batch->computed.size() || ClassOf(*_pairList, batch->computed[i]) !=
                                               ClassOf(*_pairList, batch->computed[runStart]);

        if (!runEnds)
        {
            continue;
        }

        const auto [valueStart, valueEnd] = RunValueSlice(*_pairList, batch->computed, runStart, i);
        std::vector<ShellQuartet> runQuartets(batch->computed.begin() +
                                                  static_cast<long long>(runStart),
                                              batch->computed.begin() + static_cast<long long>(i));
        std::vector<float> runValues(batch->values.begin() + static_cast<long long>(valueStart),
                                     batch->values.begin() + static_cast<long long>(valueEnd));
        std::vector<double> runBounds(batch->errorBounds.begin() + static_cast<long long>(runStart),
                                      batch->errorBounds.begin() + static_cast<long long>(i));

        if (auto appended = _store.AppendCertifiedBatch(runQuartets, runValues, runBounds);
            !appended.has_value())
        {
            return std::unexpected(appended.error());
        }

        runStart = i;
    }

    return batch;
}

const CachedEriStats& CachedEriBatchEngine::Stats() const noexcept {
    return _stats;
}

const std::string& CachedEriBatchEngine::Fingerprint() const noexcept {
    return _fingerprint;
}

} // namespace qcx::storage
