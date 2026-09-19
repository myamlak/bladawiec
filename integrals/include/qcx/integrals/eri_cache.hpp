#pragma once

/// \file
/// The in-memory ERI-value cache of the direct Fock path: a
/// CachedEriBatchEngine-shaped decorator
/// (storage/cached_eri_batch_engine.hpp) over the integrals batch
/// engine pair. Every quartet key is computed once on first occurrence and
/// served from the dual-lane arenas on every later iteration - the direct
/// builder's screened set recurs ~98% across SCF iterations,
/// and the cache turns that recomputation into memcpy traffic.
///
/// The storage tier and this cache share the DECORATOR SHAPE but not the
/// implementation: storage lives downstream of integrals in the module DAG
/// (core -> ... -> integrals -> storage), so the integrals-side tier is
/// the one the direct builder can depend on; storage's CachedEriBatchEngine
/// remains the disk tier. RI-J (RiJkFockBuilder) is untouched - its
/// contraction is density-expanded, not value-cacheable.
///
/// Value semantics: for the same request the returned EriBatch /
/// CertifiedBatch layout (computed order, packed values, error bounds) is
/// byte-identical to a direct engine call - the bit-parity contract the
/// direct builder's cached/uncached paths pin in tests. The fp32 pads are
/// zeroed in both paths: the cache states the zeroing explicitly (assign,
/// eri_cache.cpp) where the direct entry point gets it implicitly from
/// resize's value-initialization of its fresh buffer (pads are never
/// semantically read, so the parity contract stands). The fp32
/// lane's stored blocks and their certified bounds are density-independent
/// (the bound formula), so both lanes are cacheable across iterations;
/// only the certified-bound BOOKKEEPING is per-iteration (recomputed from
/// the stored norm-free bound and the live density by the caller).
///
/// Threading: no locks, no atomics. The direct builder's batch evaluation
/// is serial - one ERI assembly per call, read-mostly within a call, writes
/// across calls - so a cache written only from within ComputeEriBatch* and
/// read from the same thread is race-free by construction; concurrent
/// callers are not supported.
///
/// Capacity: no eviction. The union of computed quartets saturates for the
/// single-SCF-run pattern this tier serves - one molecule, one basis, fixed
/// for the run (the stated assumption) - so a budget that holds the
/// steady-state union bounds the cache. The caller sizes the budget from a
/// fresh DetectHostMemory() probe (backend/memory_topology.hpp) and a
/// conservative fraction of the available bytes - a point-in-time snapshot,
/// not a reservation; the budget here is the PAYLOAD budget (hash and arena
/// overhead are extra, documented at the builder). The budget is PER LANE:
/// each lane's payload (fp64 as doubles, fp32 as floats) is capped at it
/// independently - lanes budget independently, a block that exhausts one
/// lane's budget is left uncached in that lane only - so the two lanes
/// together can reach 2x the budget; callers sizing maxCacheBytes against
/// available memory must size for the union of both lanes' blocks.

#include "qcx/error.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/allocation_instrument.hpp"

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

namespace qcx::integrals {

/// Cumulative counters and live footprint of one EriBatchCache:
/// hit/miss per lane over the cache's whole lifetime, the live payload
/// bytes of the two arenas, and the payload budget at Create.
/// \ingroup qcx-integrals
struct EriCacheStats {
    std::size_t fp64HitQuartets = 0; ///< fp64-lane quartets served from the arena.
    std::size_t fp64MissQuartets = 0; ///< fp64-lane quartets computed by the engine.
    std::size_t fp32HitQuartets = 0; ///< fp32-lane quartets served from the arena.
    std::size_t fp32MissQuartets = 0; ///< fp32-lane quartets computed by the certified engine.
    std::size_t fp64PayloadBytes = 0; ///< Live fp64 arena payload bytes.
    std::size_t fp32PayloadBytes = 0; ///< Live fp32 arena payload bytes.
    std::size_t maxCacheBytes = 0; ///< The payload budget at Create.
};

/// The fp64 batch engine's SHAPE: evaluates one non-empty quartet list and
/// returns the packed batch (eri_batch.hpp layout). ONE HOME - this cache's
/// UnderlyingEngine, storage's CachedEriBatchEngine::UnderlyingEngine and the
/// decorator seam below all name this type, so a decorator written against one
/// is accepted by the other without a conversion.
/// \ingroup qcx-integrals
using EriBatchEngineFn = std::function<qcx::Result<EriBatch>(const std::vector<ShellQuartet>&)>;

/// The certified fp32 batch engine's shape: evaluates one non-empty
/// quartet list and returns the fp32 blocks with their a-priori per-quartet
/// error bounds. One home, as EriBatchEngineFn.
/// \ingroup qcx-integrals
using CertifiedEriBatchEngineFn =
    std::function<qcx::Result<CertifiedBatch>(const std::vector<ShellQuartet>&)>;

/// The engine pair a decorated factory hands back: the engines the builder
/// calls for every request after Create. An empty member leaves that lane
/// unimplemented (kUnimplemented on its call), exactly as an empty injected
/// engine does today.
/// \ingroup qcx-integrals
struct DecoratedEngines {
    EriBatchEngineFn fp64; ///< The decorated fp64 lane.
    CertifiedEriBatchEngineFn fp32; ///< The decorated certified fp32 lane.
};

/// The engine-decorator seam: the builder takes the two RAW engines
/// it would have used itself, calls a factory ONCE at Create, and uses the
/// returned pair for every later call - so the tier standing between the
/// builder and the basis is chosen from outside the module, and `integrals`
/// never names it.
///
/// This is the shape the module layering requires rather than a preference:
/// the disk tier is storage's CachedEriBatchEngine and storage is downstream
/// of integrals in the DAG, so the builder
/// cannot construct it - but the DRIVER may link both sides, so it constructs
/// the decorator, owns it (and therefore can read its Stats after the SCF),
/// and hands the builder an opaque engine pair. Dependency inversion performed
/// by the party able to perform it.
///
/// Empty (the default) is the in-memory tier and today's behaviour exactly.
/// A factory MAY compose - a RAM tier in front of a store, say - and that
/// composition belongs to its caller, the driver, not here.
/// \ingroup qcx-integrals
using EngineDecoratorFactory = std::function<qcx::Result<DecoratedEngines>(
    const EriBatchEngineFn& fp64, const CertifiedEriBatchEngineFn& fp32)>;

/// The in-memory ERI-value cache: the integrals-side mirror
/// of storage's CachedEriBatchEngine - same engine-function shape, same
/// compute-once-on-first-occurrence semantics, same bit-parity contract -
/// keyed on the 8-fold canonical pair index (bra * nPairs + ket, bra >=
/// ket), which is invariant under every 8-fold partner and the class swap,
/// so the block stored for a key is the same canonical quartet's block for
/// every later request. Miss quartets run through the injected engines in
/// their canonical-order subsequence (the engine's own order for that
/// subset), and the returned layout is the FULL request's canonical order -
/// identical to what a direct engine call produces.
/// \ingroup qcx-integrals
class EriBatchCache {
public:
    /// The fp64 lane's underlying batch engine (EriBatchEngineFn - the one
    /// home for the shape).
    using UnderlyingEngine = EriBatchEngineFn;

    /// The certified fp32 lane's underlying batch engine
    /// (CertifiedEriBatchEngineFn - the one home for the shape).
    using UnderlyingCertifiedEngine = CertifiedEriBatchEngineFn;

    /// Creates the cache.
    /// \param maxCacheBytes The per-lane payload budget: each lane's stored
    /// block bytes (fp64 as doubles, fp32 as floats) never exceed it - the
    /// lanes budget independently, so the two together can reach 2x it;
    /// size for the union of both lanes' blocks (the capacity note in the
    /// file header).
    /// Must be positive; blocks that would exceed the remaining budget are
    /// left uncached (recomputed on every later request - correctness is
    /// budget-independent, only the hit rate suffers).
    /// The guard is kept for a caller that wants NO RAM tier: the smallest
    /// positive budget already admits no block (every stored block is at
    /// least one double), so one byte is a passthrough tier that serves
    /// every request through the injected engines and leaves the arenas
    /// empty. The equality-only zero check below stays an error so a
    /// forgotten argument cannot be mistaken for that request.
    /// \param pairList The system's canonical pair list (BuildShellPairs);
    /// BORROWED - the pair list must outlive the cache. The direct
    /// builder's State owns both, so the cache dies with the State.
    /// \param engine The fp64 lane engine; an empty std::function leaves the
    /// lane unimplemented (kUnimplemented on ComputeEriBatch).
    /// \param certifiedEngine The certified fp32 lane engine; an empty
    /// std::function leaves the certified lane unimplemented (kUnimplemented
    /// on ComputeEriBatchCertified).
    /// \returns The cache, or an Error.
    static qcx::Result<EriBatchCache> Create(std::size_t maxCacheBytes,
                                             const ShellPairList& pairList,
                                             UnderlyingEngine engine,
                                             UnderlyingCertifiedEngine certifiedEngine);

    /// The fp64 lane: evaluates the request through the cache - hits served
    /// from the arena, misses through the engine - and returns the batch in
    /// the exact layout a direct engine call produces for the same request.
    /// \param quartets The request, in any form (each quartet is
    /// canonicalized; an empty list is an Error - the engine's own
    /// contract).
    /// \param options Accepted for API compatibility with the storage
    /// decorator and otherwise ignored: the engines carry their own
    /// batch options at injection.
    /// \returns The packed batch, or an Error.
    qcx::Result<EriBatch> ComputeEriBatch(const std::vector<ShellQuartet>& quartets,
                                          const EriBatchOptions& options = {});

    /// The certified fp32 lane: the ComputeEriBatch semantics with the
    /// fp32 blocks and their per-quartet error bounds (bit-identical to a
    /// direct certified engine call).
    /// \param quartets The request, in any form (empty = Error, as above).
    /// \param options Accepted for API compatibility and ignored (as above).
    /// \returns The certified batch, or an Error (kUnimplemented when no
    /// certified engine was injected).
    qcx::Result<CertifiedBatch> ComputeEriBatchCertified(const std::vector<ShellQuartet>& quartets,
                                                         const EriBatchOptions& options = {});

    /// The cumulative stats (EriCacheStats). The reference is stable for
    /// the cache's lifetime.
    /// \returns The stats.
    const EriCacheStats& Stats() const noexcept;

private:
    // One key's dual-lane bookkeeping: the block offsets into the two
    // append-only arenas (element offsets; the block size is the same in
    // both lanes), the per-lane presence flags, and the stored certified
    // bound of the fp32 block (density-independent, so cacheable).
    struct Entry {
        std::size_t fp64Offset = 0;
        std::size_t fp32Offset = 0;
        std::size_t blockSize = 0;
        bool hasFp64 = false;
        bool hasFp32 = false;
        double fp32Bound = 0.0;
    };

    EriBatchCache(std::size_t maxCacheBytes,
                  const ShellPairList* pairList,
                  UnderlyingEngine engine,
                  UnderlyingCertifiedEngine certifiedEngine);

    // The 8-fold canonical pair key: i <= j, k <= l, bra >= ket, then
    // bra * nPairs + ket - invariant under every 8-fold partner and the
    // class swap (the class swap only reorders which pair is bra).
    std::size_t PairKeyOf(const ShellQuartet& quartet) const noexcept;

    bool TryInsertFp64(std::size_t key, const double* block, std::size_t blockSize);
    bool TryInsertFp32(std::size_t key, const float* block, std::size_t blockSize, double bound);

    const ShellPairList* _pairList;
    std::size_t _nPairs;
    UnderlyingEngine _engine;
    UnderlyingCertifiedEngine _certifiedEngine;
    // The key -> entry table; the entries are stable under arena growth
    // (the arenas hold the VALUES; the offsets into them never move).
    std::unordered_map<std::size_t, Entry> _entries;
    // The dual-lane payload arenas (the cache family - DirectFootprint
    // charges the two lanes' payload caps as 2 x maxCacheBytes of the 3x
    // cacheBytes term; the entry-map allowance is the third): the tagged
    // allocator captures kCache at the cache's construction (the Create
    // member opens the family scope), so the arenas' growth through the
    // hits of later calls attributes to the cache family.
    std::vector<double, qcx::memory::TaggedAllocator<double>>
        _fp64Arena; // Append-only, no eviction.
    std::vector<float, qcx::memory::TaggedAllocator<float>> _fp32Arena; // Append-only, no eviction.
    std::size_t _fp64Bytes = 0; // Live fp64 payload bytes.
    std::size_t _fp32Bytes = 0; // Live fp32 payload bytes.
    EriCacheStats _stats;
    std::size_t _maxCacheBytes = 0;
};

} // namespace qcx::integrals
