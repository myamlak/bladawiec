// The in-memory ERI-value cache (eri_cache.hpp): the
// integrals-side decorator over the batch engine pair. The request is
// canonicalized in FULL - CanonicalizeQuartetOrder, the exact order the
// batch engine produces - then split by the canonical pair key:
// hits are copied out of the dual-lane arenas, misses run through the
// injected engine in their canonical-order subsequence (a total order
// restricted to a subset - the engine's own order for that subset), and
// the engine's blocks are scattered back by key. The returned layout
// therefore matches a direct engine call byte for byte - the bit-parity
// contract the builder's cached/uncached paths pin.

#include "qcx/integrals/eri_cache.hpp"

#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace qcx::integrals {

namespace {

// The packed block element count of one quartet: nI * nJ * nK * nL.
std::size_t BlockSizeOf(const ShellPairList& pairList, const ShellQuartet& quartet) noexcept {
    const std::size_t nI = ShellFunctionCount(pairList.shells[quartet.i]);
    const std::size_t nJ = ShellFunctionCount(pairList.shells[quartet.j]);
    const std::size_t nK = ShellFunctionCount(pairList.shells[quartet.k]);
    const std::size_t nL = ShellFunctionCount(pairList.shells[quartet.l]);
    return nI * nJ * nK * nL;
}

} // namespace

EriBatchCache::EriBatchCache(std::size_t maxCacheBytes,
                             const ShellPairList* pairList,
                             UnderlyingEngine engine,
                             UnderlyingCertifiedEngine certifiedEngine) :
    _pairList(pairList), _nPairs(pairList->pairs.size()), _engine(std::move(engine)),
    _certifiedEngine(std::move(certifiedEngine)), _maxCacheBytes(maxCacheBytes) {
    _stats.maxCacheBytes = maxCacheBytes;
}

qcx::Result<EriBatchCache> EriBatchCache::Create(std::size_t maxCacheBytes,
                                                 const ShellPairList& pairList,
                                                 UnderlyingEngine engine,
                                                 UnderlyingCertifiedEngine certifiedEngine) {
    if (maxCacheBytes == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "maxCacheBytes must be positive"});
    }

    // The cache family (eri_cache.hpp): the dual-lane payload arenas
    // capture the tag at their construction inside the cache object below,
    // so every later arena insert attributes to kCache no matter which
    // thread serves the hits.
    qcx::memory::AllocationTagScope cacheScope(qcx::memory::AllocationTag::kCache);

    return EriBatchCache(maxCacheBytes, &pairList, std::move(engine), std::move(certifiedEngine));
}

std::size_t EriBatchCache::PairKeyOf(const ShellQuartet& quartet) const noexcept {
    std::size_t i = quartet.i;
    std::size_t j = quartet.j;
    std::size_t k = quartet.k;
    std::size_t l = quartet.l;

    if (i > j)
    {
        std::swap(i, j);
    }

    if (k > l)
    {
        std::swap(k, l);
    }

    std::size_t bra = PairIndexOf(i, j, *_pairList);
    std::size_t ket = PairIndexOf(k, l, *_pairList);

    if (bra < ket)
    {
        std::swap(bra, ket);
    }

    return bra * _nPairs + ket;
}

// The (key, block, blockSize) order is fixed by the two compute-path call
// sites; a swap would silently misplace a block in the arena.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool EriBatchCache::TryInsertFp64(std::size_t key, const double* block, std::size_t blockSize) {
    const std::size_t blockBytes = blockSize * sizeof(double);

    if (_fp64Bytes + blockBytes > _maxCacheBytes)
    {
        // The budget is exhausted for this block: leave the key uncached
        // (the entry may exist with the fp32 lane only - never half-fill a
        // lane). Every later request recomputes - correctness is untouched.
        return false;
    }

    const auto found = _entries.find(key);
    Entry entry;
    entry.fp64Offset = _fp64Arena.size();
    entry.blockSize = blockSize;
    entry.hasFp64 = true;

    if (found != _entries.end())
    {
        if (found->second.hasFp64)
        {
            return true;
        }

        // Merge with the existing fp32 lane (a certified lane can survive a
        // budget-rejected fp64 insert - lanes budget independently).
        entry.fp32Offset = found->second.fp32Offset;
        entry.hasFp32 = found->second.hasFp32;
        entry.fp32Bound = found->second.fp32Bound;
        found->second = entry;
    } else
    {
        _entries.emplace(key, entry);
    }

    _fp64Arena.insert(_fp64Arena.end(), block, block + blockSize);
    _fp64Bytes += blockBytes;
    _stats.fp64PayloadBytes = _fp64Bytes;
    return true;
}

// Mirrors TryInsertFp64: the (key, block, blockSize, bound) order is fixed
// by the certified compute path's call site. BEGIN/END because the
// diagnostic lands on the multi-line signature's second parameter line.
// NOLINTBEGIN(bugprone-easily-swappable-parameters)
bool EriBatchCache::TryInsertFp32(std::size_t key,
                                  const float* block,
                                  std::size_t blockSize,
                                  double bound) {
    const std::size_t blockBytes = blockSize * sizeof(float);

    if (_fp32Bytes + blockBytes > _maxCacheBytes)
    {
        return false;
    }

    const auto found = _entries.find(key);
    Entry entry;
    entry.fp32Offset = _fp32Arena.size();
    entry.blockSize = blockSize;
    entry.hasFp32 = true;
    entry.fp32Bound = bound;

    if (found != _entries.end())
    {
        if (found->second.hasFp32)
        {
            return true;
        }

        entry.fp64Offset = found->second.fp64Offset;
        entry.hasFp64 = found->second.hasFp64;
        found->second = entry;
    } else
    {
        _entries.emplace(key, entry);
    }

    _fp32Arena.insert(_fp32Arena.end(), block, block + blockSize);
    _fp32Bytes += blockBytes;
    _stats.fp32PayloadBytes = _fp32Bytes;
    return true;
}

// NOLINTEND(bugprone-easily-swappable-parameters)

qcx::Result<EriBatch> EriBatchCache::ComputeEriBatch(const std::vector<ShellQuartet>& quartets,
                                                     const EriBatchOptions&) {
    // An empty engine is an Error here exactly as on the certified lane: an
    // empty std::function would throw std::bad_function_call (barred by the
    // no-exceptions rule), so mirror the certified guard (kUnimplemented).
    if (!_engine)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kUnimplemented, "no engine was injected into the cache"});
    }

    // An empty request is an Error here exactly as it is for the engines:
    // CanonicalizeQuartetOrder rejects it (kInvalidArgument) - the
    // decorated call reproduces the engine's error, the storage decorator's
    // empty-request parity contract.
    auto ordered = CanonicalizeQuartetOrder(*_pairList, quartets);

    if (!ordered.has_value())
    {
        return std::unexpected(ordered.error());
    }

    const std::size_t n = ordered->size();
    // The full canonical order's block offsets - the layout the engine
    // would have produced for the same request.
    std::vector<std::size_t> fullOffsets(n);
    std::size_t total = 0;

    for (std::size_t pos = 0; pos < n; ++pos)
    {
        fullOffsets[pos] = total;
        total += BlockSizeOf(*_pairList, (*ordered)[pos].quartet);
    }

    EriBatch result;
    result.computed.reserve(n);
    result.values.resize(total);

    for (const CanonicalQuartetInfo& info : *ordered)
    {
        result.computed.push_back(info.quartet);
    }

    // The hit/miss split by key: hits are copied from the arena now, the
    // miss quartets go to the engine as one request and scatter back by
    // position (the m-th engine result is the m-th miss - the miss list is
    // the full order filtered to the misses, and the engine's own
    // canonicalization of a canonical list is the identity).
    std::vector<ShellQuartet> misses;
    std::vector<std::size_t> missPositions;
    misses.reserve(n);
    missPositions.reserve(n);
    std::size_t hitCount = 0;

    for (std::size_t pos = 0; pos < n; ++pos)
    {
        const std::size_t key = PairKeyOf((*ordered)[pos].quartet);
        const auto found = _entries.find(key);

        if (found != _entries.end() && found->second.hasFp64)
        {
            const Entry& entry = found->second;
            std::memcpy(result.values.data() + fullOffsets[pos],
                        _fp64Arena.data() + entry.fp64Offset,
                        entry.blockSize * sizeof(double));
            ++hitCount;
        } else
        {
            misses.push_back((*ordered)[pos].quartet);
            missPositions.push_back(pos);
        }
    }

    _stats.fp64HitQuartets += hitCount;

    if (!misses.empty())
    {
        auto engineBatch = _engine(misses);

        if (!engineBatch.has_value())
        {
            return std::unexpected(engineBatch.error());
        }

        // The miss count accrues only on a successful engine call: an engine
        // error must not inflate "computed by the engine" (EriCacheStats).
        _stats.fp64MissQuartets += misses.size();

        if (engineBatch->computed.size() != misses.size())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInternalError,
                           "the engine's computed list does not match the miss request"});
        }

        std::size_t missOffset = 0;

        for (std::size_t m = 0; m < misses.size(); ++m)
        {
            // The engine must return computed in the request's order, one
            // block per request quartet - AssembleClassBatches preserves the
            // canonical order, so this holds for the injected engines; a
            // reordering engine would silently misplace every block (the
            // parity contract's boundary). The per-quartet key and block size
            // come from the reported quartet, a self-consistent pair with the
            // reported values.
            const ShellQuartet& quartet = engineBatch->computed[m];
            const std::size_t blockSize = BlockSizeOf(*_pairList, quartet);
            std::memcpy(result.values.data() + fullOffsets[missPositions[m]],
                        engineBatch->values.data() + missOffset,
                        blockSize * sizeof(double));
            TryInsertFp64(PairKeyOf(quartet), engineBatch->values.data() + missOffset, blockSize);
            missOffset += blockSize;
        }
    }

    return result;
}

qcx::Result<CertifiedBatch> EriBatchCache::ComputeEriBatchCertified(
    const std::vector<ShellQuartet>& quartets, const EriBatchOptions&) {
    if (!_certifiedEngine)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                          "no certified engine was injected into the cache"});
    }

    auto ordered = CanonicalizeQuartetOrder(*_pairList, quartets);

    if (!ordered.has_value())
    {
        return std::unexpected(ordered.error());
    }

    const std::size_t n = ordered->size();
    std::vector<std::size_t> fullOffsets(n);
    std::size_t total = 0;

    for (std::size_t pos = 0; pos < n; ++pos)
    {
        fullOffsets[pos] = total;
        total += BlockSizeOf(*_pairList, (*ordered)[pos].quartet);
    }

    CertifiedBatch result;
    result.computed.reserve(n);
    // assign (not resize), like the uncached path: the fp32 lanes are read
    // back in full by the caller, including any pads the batch machinery
    // writes only partially - zero-init so stale data never leaks through
    // a pad. The bounds share the discipline (every position is either a
    // hit's stored bound, a miss's engine bound, or the zero pad).
    result.values.assign(total, 0.f);
    result.errorBounds.assign(n, 0.0);

    for (const CanonicalQuartetInfo& info : *ordered)
    {
        result.computed.push_back(info.quartet);
    }

    std::vector<ShellQuartet> misses;
    std::vector<std::size_t> missPositions;
    misses.reserve(n);
    missPositions.reserve(n);
    std::size_t hitCount = 0;

    for (std::size_t pos = 0; pos < n; ++pos)
    {
        const std::size_t key = PairKeyOf((*ordered)[pos].quartet);
        const auto found = _entries.find(key);

        if (found != _entries.end() && found->second.hasFp32)
        {
            const Entry& entry = found->second;
            std::memcpy(result.values.data() + fullOffsets[pos],
                        _fp32Arena.data() + entry.fp32Offset,
                        entry.blockSize * sizeof(float));
            result.errorBounds[pos] = entry.fp32Bound;
            ++hitCount;
        } else
        {
            misses.push_back((*ordered)[pos].quartet);
            missPositions.push_back(pos);
        }
    }

    _stats.fp32HitQuartets += hitCount;

    if (!misses.empty())
    {
        auto engineBatch = _certifiedEngine(misses);

        if (!engineBatch.has_value())
        {
            return std::unexpected(engineBatch.error());
        }

        // Same discipline as the fp64 lane: the miss count is "computed by
        // the engine" and accrues only on a successful call (EriCacheStats).
        _stats.fp32MissQuartets += misses.size();

        if (engineBatch->computed.size() != misses.size())
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInternalError,
                           "the certified engine's computed list does not match the miss request"});
        }

        std::size_t missOffset = 0;

        for (std::size_t m = 0; m < misses.size(); ++m)
        {
            const ShellQuartet& quartet = engineBatch->computed[m];
            const std::size_t blockSize = BlockSizeOf(*_pairList, quartet);
            std::memcpy(result.values.data() + fullOffsets[missPositions[m]],
                        engineBatch->values.data() + missOffset,
                        blockSize * sizeof(float));
            result.errorBounds[missPositions[m]] = engineBatch->errorBounds[m];
            TryInsertFp32(PairKeyOf(quartet),
                          engineBatch->values.data() + missOffset,
                          blockSize,
                          engineBatch->errorBounds[m]);
            missOffset += blockSize;
        }
    }

    return result;
}

const EriCacheStats& EriBatchCache::Stats() const noexcept {
    return _stats;
}

} // namespace qcx::integrals
