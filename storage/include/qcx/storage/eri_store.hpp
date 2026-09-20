#pragma once

/// \defgroup qcx-storage Storage module
/// \brief The disk-integral engine: HDF5 persistence.
///
/// The HDF5 integral store behind the cached batch decorator, the RI-3c
/// tensor and SCF checkpoint persistence, gated on QCX_ENABLE_IO.

/// \file
/// The HDF5 integral store: one file per system, append-only,
/// with per-chunk FNV-1a-64 checksums verified on every serve and a
/// /metadata/store_checksum verified at open. The schema lives in
/// `schema.hpp`; this header is the schema owner's public face.
///
/// > **Derived-dataset contract.** A derived dataset
/// > (`/derived/mo_integrals/iajb`, `/derived/mo_integrals/abcd`,
/// > `/derived/mo_integrals/kpqrs`, `/derived/grid_properties/ao_values`)
/// > is written only by the module that owns its schema (the future post-HF
/// > / properties modules). Every write sets the dataset's
/// > `source_fingerprint` attribute to the file's `/molecule/fingerprint`.
/// > A reader MUST verify `source_fingerprint == /molecule/fingerprint`
/// > before consuming; an empty `source_fingerprint` means unfilled and MUST
/// > NOT be read. A derived dataset is versioned by its own
/// > `engine_version`/`schema_version` attributes against the AO source
/// > fingerprint - the store itself never interprets or migrates derived
/// > content. This store deliberately ships no writer: the slots tolerate
/// > the derived-dataset owner modules not existing.
///
/// Integrity policy: reject = refuse to serve, fail with kIOError,
/// never silently recompute. Corruption is a user-visible error; the caller
/// decides whether to fall back. A healthy file that belongs to a different
/// system is kInvalidArgument.
///
/// Endianness: native layout; files are portable across machines of the
/// same endianness (PHDF5-ready note, cross-endian pinning deferred).

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/storage/schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace qcx::storage {

// The integrals types the store API speaks (qcx::integrals; unqualified
// below for readability, like the test suite does).
using qcx::integrals::CanonicalQuartetInfo;
using qcx::integrals::CertifiedBatch;
using qcx::integrals::EriBatch;
using qcx::integrals::EriBatchOptions;
using qcx::integrals::ShellQuartet;

/// Store options.
/// \ingroup qcx-storage
struct StoreOptions {
    qcx::integrals::AccuracyPreset accuracy =
        qcx::integrals::AccuracyPreset::kNormal; ///< Recorded in /metadata (provenance).
    std::string provenance = "qcx"; ///< Free-form creator tag.
};

/// One HDF5 integral store for one system. Create for a fresh file,
/// Open for append. Single-writer, single-process, single-threaded.
/// Every read verifies the chunk checksum; a corrupt store fails with
/// kIOError and is never silently recomputed around.
/// \ingroup qcx-storage
class EriStore {
public:
    /// Creates a fresh store file (fails with kIOError when one already
    /// exists) and writes the schema header, /molecule identity, the
    /// fingerprint and /metadata. The system fingerprint is computed from
    /// the molecule and basis names and stored as the file's identity.
    /// \param path The store file to create.
    /// \param molecule The system (canonical atom order, Bohr coordinates).
    /// \param basisSet The basis (function counts go into the header).
    /// \param orbitalBasisName The orbital basis-set name.
    /// \param auxBasisName The auxiliary basis-set name, or empty.
    /// \param options Provenance metadata recorded in /metadata.
    /// \returns The store, or an Error (kIOError when the file already
    /// exists or HDF5 fails).
    static qcx::Result<EriStore> Create(const std::filesystem::path& path,
                                        const qcx::molecule::Molecule& molecule,
                                        const qcx::basisset::BasisSet& basisSet,
                                        std::string_view orbitalBasisName,
                                        std::string_view auxBasisName,
                                        const StoreOptions& options = {});

    /// Opens an existing store for append. Re-verifies the fingerprint AND
    /// every readable /molecule field element-wise (kInvalidArgument on
    /// mismatch - "store belongs to a different system"), then the
    /// /metadata/store_checksum (kIOError on corruption). The options are
    /// ignored here - they are Create-time metadata.
    /// \param path The store file to open.
    /// \param molecule The expected system.
    /// \param basisSet The expected basis.
    /// \param orbitalBasisName The expected orbital basis-set name.
    /// \param auxBasisName The expected auxiliary basis-set name, or empty.
    /// \param options Ignored (Create-time metadata).
    /// \returns The store, or an Error (kInvalidArgument for a system or
    /// basis mismatch, kIOError for corruption or HDF5 failures).
    static qcx::Result<EriStore> Open(const std::filesystem::path& path,
                                      const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basisSet,
                                      std::string_view orbitalBasisName,
                                      std::string_view auxBasisName,
                                      const StoreOptions& options = {});

    /// Writes the three one-electron matrices as {n, n} float64 datasets
    /// under /one_electron (drop-in mirrors of one_electron.hpp naming).
    /// Refuses when any of them already exists (kInvalidArgument;
    /// append-only store semantics). The shapes must be {n, n} with n the
    /// basis function count.
    /// \param overlap The overlap matrix.
    /// \param kinetic The kinetic-energy matrix.
    /// \param nuclear The nuclear-attraction matrix.
    /// \returns An Error (kInvalidArgument when a dataset already exists or
    /// a shape is wrong, kIOError for HDF5 failures).
    qcx::Result<void> WriteOneElectron(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& overlap,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& kinetic,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& nuclear);
    /// Reads the stored overlap matrix.
    /// \returns The {n, n} matrix, or an Error (kIOError when missing or
    /// corrupt).
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> ReadOverlapMatrix() const;
    /// Reads the stored kinetic-energy matrix.
    /// \returns The {n, n} matrix, or an Error (kIOError when missing or
    /// corrupt).
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> ReadKineticMatrix() const;
    /// Reads the stored nuclear-attraction matrix.
    /// \returns The {n, n} matrix, or an Error (kIOError when missing or
    /// corrupt).
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> ReadNuclearAttractionMatrix()
        const;

    /// Appends one class run: \p quartets are the canonical quartets of the
    /// run in computed order (one contiguous slice of a batch), \p values the
    /// run's packed blocks. One manifest row is written. All quartets must
    /// share one (lBra, lKet) class, be in canonical form, and block sizes
    /// must sum to values.size() (kInvalidArgument otherwise). Duplicate
    /// quartets across appends are stored twice (byte-identical blocks) -
    /// the v1 dedupe is deliberately deferred.
    /// \param quartets The canonical quartets of the run in computed order.
    /// \param values The run's packed blocks (sizes must match the quartets).
    /// \returns An Error (kInvalidArgument on class/order/block-size
    /// violations, kIOError for HDF5 failures).
    qcx::Result<void> AppendBatch(const std::vector<ShellQuartet>& quartets,
                                  const std::vector<double>& values);
    /// Serves a full-hit request: \p request is the canonical, ordered
    /// CanonicalizeQuartetOrder output; every quartet must already be
    /// stored (kInvalidArgument otherwise). Values are assembled in \p
    /// request order, checksum-verified per chunk (kIOError on corruption).
    /// \param request The canonical, ordered quartets (all stored).
    /// \returns The assembled EriBatch, or an Error (kInvalidArgument when
    /// a quartet is unstored, kIOError on corruption or HDF5 failures).
    qcx::Result<EriBatch> LoadBatch(const std::vector<CanonicalQuartetInfo>& request) const;
    /// Appends one certified fp32 class run: values stored verbatim with
    /// their error bounds.
    /// \param quartets The canonical quartets of the run.
    /// \param values The run's packed fp32 blocks.
    /// \param errorBounds The bounds in quartets order.
    /// \returns An Error (kInvalidArgument on class/order/size violations,
    /// kIOError for HDF5 failures).
    qcx::Result<void> AppendCertifiedBatch(const std::vector<ShellQuartet>& quartets,
                                           const std::vector<float>& values,
                                           const std::vector<double>& errorBounds);
    /// The certified analog of LoadBatch: fp32 values stored verbatim (never
    /// promoted through double) with their bounds in quartets-table
    /// order. A chunk whose engine-version stamp differs from the current
    /// kIntegralEngineVersion is never served (kIOError) - the a-priori
    /// bound is only valid for the pipeline that produced it. The returned
    /// batch carries the STORED bounds; the consumer re-checks its own
    /// budget against them (the contract, unchanged on load).
    /// \param request The canonical, ordered quartets (all stored).
    /// \returns The assembled CertifiedBatch, or an Error (kInvalidArgument
    /// when a quartet is unstored, kIOError on version/corruption issues or
    /// HDF5 failures).
    qcx::Result<CertifiedBatch> LoadCertifiedBatch(
        const std::vector<CanonicalQuartetInfo>& request) const;

    /// Whether the canonical quartet has a stored fp64 chunk block (the
    /// decorator's full-hit test; the manifest-derived lookup).
    /// \param quartet The canonical quartet.
    /// \returns True when a fp64 block is stored.
    bool ContainsFp64(const ShellQuartet& quartet) const noexcept;
    /// The fp32-chunk analog (the certified path's full-hit test).
    /// \param quartet The canonical quartet.
    /// \returns True when a fp32 block is stored.
    bool ContainsFp32(const ShellQuartet& quartet) const noexcept;

    /// The number of stored manifest rows (chunks).
    /// \returns The chunk count.
    std::size_t StoredChunkCount() const noexcept;
    /// The number of stored quartets (manifest quartetsPerChunk summed).
    /// \returns The quartet count.
    std::size_t StoredQuartetCount() const noexcept;

    /// The implementation state lives in the .cpp (HighFive stays
    /// implementation-only; the store holds exactly one file handle for its
    /// lifetime - Windows locks the file while it lives). The name is public
    /// because the .cpp's free helpers take it by reference; the definition
    /// stays in the .cpp - nothing outside can do anything with it.
    struct State;

private:
    explicit EriStore(std::shared_ptr<State> state);
    std::shared_ptr<State> _state;
};

} // namespace qcx::storage
