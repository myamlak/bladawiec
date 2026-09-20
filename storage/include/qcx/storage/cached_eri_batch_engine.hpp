#pragma once

/// \file
/// The cached batch-engine decorator: reads hit the store's manifest;
/// misses recompute through an injected engine callback and append. A
/// drop-in decorator over the integrals batch API.

#include "qcx/error.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/storage/eri_store.hpp"

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace qcx::storage {

/// The read-vs-recompute crossover measurement: per-class counters
/// and wall times. recomputeMsByClass is filled only for single-class
/// requests (the benchmark's shape); multi-class misses contribute to the
/// totals only.
/// \ingroup qcx-storage
struct CachedEriStats {
    std::size_t hitQuartets = 0; ///< Served from disk.
    std::size_t missQuartets = 0; ///< Recomputed.
    double readMs = 0.0; ///< Wall time serving hits (checksums included).
    double recomputeMs = 0.0; ///< Wall time in the underlying engine.
    std::map<std::pair<int, int>, std::size_t> hitQuartetsByClass; ///< Hits by (lBra, lKet) class.
    std::map<std::pair<int, int>, std::size_t> missQuartetsByClass; ///< Misses by class.
    std::map<std::pair<int, int>, double> readMsByClass; ///< Serve wall time by class.
    std::map<std::pair<int, int>, double> recomputeMsByClass; ///< Engine wall time by class.
};

/// Drop-in decorator over the batch API: reads hit the store's
/// manifest; misses recompute through an injected engine callback and
/// append. Opt-in by construction - the direct/LinK Fock builders never use
/// it (LinK keeps recomputing by design). A corrupt store fails with
/// kIOError.
///
/// One engine call is the decorator's unit: a request is canonicalized and
/// sorted (the shared CanonicalizeQuartetOrder); when every quartet is
/// already stored the whole request is served from disk, otherwise the
/// WHOLE request is recomputed through the underlying engine and appended
/// as new chunks (chunk = contiguous class run of the sorted result). The
/// stored bytes are the engine's verbatim - reload equals recompute bit
/// for bit.
/// \ingroup qcx-storage
class CachedEriBatchEngine {
public:
    /// The injected fp64 batch engine (misses are recomputed through it).
    using UnderlyingEngine = std::function<qcx::Result<EriBatch>(const std::vector<ShellQuartet>&)>;
    /// The injected certified fp32 engine ({} for none).
    using UnderlyingCertifiedEngine =
        std::function<qcx::Result<CertifiedBatch>(const std::vector<ShellQuartet>&)>;

    /// Creates the decorator over an existing or fresh store file. Opens
    /// the EriStore (fingerprint-verified) and canonicalizes the system's
    /// shell pairs once; the store is ready for hit/miss traffic.
    /// \param storePath The HDF5 store file (created when missing).
    /// \param molecule The system (fingerprint identity).
    /// \param basisSet The basis (function counts; the fingerprint).
    /// \param orbitalBasisName The orbital basis-set name.
    /// \param auxBasisName The auxiliary basis-set name, or empty.
    /// \param engine The fp64 batch engine to recompute misses.
    /// \param certifiedEngine The certified fp32 engine, or {} for none.
    /// \param options Store creation options (ignored for an existing store).
    /// \returns The decorator, or an Error (kInvalidArgument for a
    /// fingerprint mismatch, kIOError for HDF5 failures).
    static qcx::Result<CachedEriBatchEngine> Create(std::filesystem::path storePath,
                                                    const qcx::molecule::Molecule& molecule,
                                                    const qcx::basisset::BasisSet& basisSet,
                                                    std::string_view orbitalBasisName,
                                                    std::string_view auxBasisName,
                                                    UnderlyingEngine engine,
                                                    UnderlyingCertifiedEngine certifiedEngine,
                                                    const StoreOptions& options = {});

    /// Mirrors qcx::integrals::ComputeEriBatch minus the fixed system
    /// arguments. Returns the underlying engine's EriBatch verbatim on a
    /// miss; on a full hit, values/computed byte-identical to what the
    /// engine would return for the same request.
    /// \param quartets The request quartets (any order; canonicalized here).
    /// \param options Accepted for API compatibility with
    /// qcx::integrals::ComputeEriBatch and ignored.
    /// \returns The EriBatch, or an Error (kIOError on a corrupt store).
    qcx::Result<EriBatch> ComputeEriBatch(const std::vector<ShellQuartet>& quartets,
                                          const EriBatchOptions& options = {});
    /// The certified analog; kUnimplemented when no certified engine was
    /// injected. Stored fp32 blocks carry their stored error bounds; the
    /// consumer's existing budget check applies unchanged on load.
    /// \param quartets The request quartets (any order; canonicalized here).
    /// \param options Accepted for API compatibility with
    /// qcx::integrals::ComputeEriBatchCertified and ignored.
    /// \returns The CertifiedBatch, or an Error (kUnimplemented without a
    /// certified engine, kIOError on a corrupt store).
    qcx::Result<CertifiedBatch> ComputeEriBatchCertified(const std::vector<ShellQuartet>& quartets,
                                                         const EriBatchOptions& options = {});

    /// The cumulative hit/miss counters and wall times.
    /// \returns The stats.
    const CachedEriStats& Stats() const noexcept;
    /// The store's system fingerprint.
    /// \returns The 16-character lowercase hex fingerprint.
    const std::string& Fingerprint() const noexcept;

private:
    // EriStore is not default-constructible; Create opens the store and
    // moves it in. HighFive types stay inside the store's PImpl - the
    // decorator only ever holds the opaque store value.
    CachedEriBatchEngine(EriStore store,
                         std::shared_ptr<qcx::integrals::ShellPairList> pairList,
                         UnderlyingEngine engine,
                         UnderlyingCertifiedEngine certifiedEngine,
                         std::string fingerprint);

    EriStore _store;
    std::shared_ptr<qcx::integrals::ShellPairList> _pairList;
    UnderlyingEngine _engine;
    UnderlyingCertifiedEngine _certifiedEngine;
    CachedEriStats _stats;
    std::string _fingerprint;
};

} // namespace qcx::storage
