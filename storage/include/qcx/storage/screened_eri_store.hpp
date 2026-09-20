#pragma once

/// \file
/// The Create-time screened pre-warm of a full-ERI store - the storage half
/// of the disk-full member of the disk family (the disk rung is the
/// ladder's LAST).
///
/// WHAT THIS IS, AND WHAT IT IS NOT. A full 4-index ERI tensor is O(n^4) and
/// cannot be materialized (8 n^4 bytes: 5e15 at 5000 functions), so this is
/// not "every quartet on disk". It is a **pre-warm**: the shell quartets a
/// Create-time Schwarz screen admits are evaluated once and written into an
/// EriStore (eri_store.hpp), so the disk tier's later reads hit.
///
/// The pre-warm is a HIT-RATE decision and never a correctness boundary. The
/// direct Fock builders gate per iteration on a density-weighted test
/// (fock_screen.hpp:323, `dMax * Q_bra * Q_ket >= densityThreshold`, with
/// DensityThreshold == SchwarzThreshold at every preset), so the Schwarz set
/// is NOT a superset of what a build asks for and the pre-warm can omit
/// quartets a run wants. That is by design and is harmless: the store is a
/// cache whose consumer recomputes a miss and appends it
/// (cached_eri_batch_engine.hpp:48-54), so an omitted quartet is computed on
/// demand and is present from the second iteration on. Nothing here filters
/// an answer; the two functions below decide what is computed early and how
/// much disk it costs.
///
/// Deliberately NOT the RI-J disk shape. The RI disk builder stores a fixed
/// row x aux chunk plan (ri_tensor_chunk_store.hpp) because its tensor has a
/// dense scaffold; a full-ERI store has none, so the pre-warmed set is stored
/// sparsely - the chunks are the contiguous class runs the store's own
/// AppendBatch contract already defines (eri_store.hpp), and the reduction
/// that makes them sparse is a STORE-SHAPE reduction, never a point-group
/// mechanism (disk_ri_fock_build.hpp:80-89 is the precedent for that
/// labelling).

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/storage/eri_store.hpp"

#include <cstddef>
#include <filesystem>
#include <string_view>
#include <vector>

namespace qcx::storage {

/// The pre-warm plan of one full-ERI store: the significant canonical
/// quartets and the raw ERI payload they imply.
///
/// The quartet list is held in RAM, exactly as the store's own append-only
/// tables are (EriStore::State::quartets, eri_store.cpp:81 - 32 bytes per
/// stored quartet): the store scales the VALUES out of RAM, not the
/// manifest. A caller sizing for a very large pre-warm must account for the
/// list as well as the payload.
/// \ingroup qcx-storage
struct ScreenedEriStorePlan {
    /// The significant canonical quartets in the store's own canonical
    /// order (CanonicalizeQuartetOrder), which is also the order
    /// EriBatch::computed reports - so a plan is directly appendable.
    std::vector<qcx::integrals::CanonicalQuartetInfo> quartets;
    /// The RAW ERI payload the plan implies: 8 bytes per element summed over
    /// the significant quartets (nI*nJ*nK*nL each). This is the
    /// on-disk-vs-modeled reconciliation target's numerator - and it is the
    /// payload, not the file size: the file adds its manifest, its quartets
    /// table, the per-chunk checksums and HDF5 overhead, so a reconciliation
    /// must compare against the HDF5 dataset bytes rather than file_size.
    std::size_t payloadBytes = 0;
    /// The unreduced 4-index bound, 8 n^4 over the basis function count n
    /// (no 8-fold symmetry folded in): the number the pre-warm exists to
    /// stay far below, and the reason the disk tier cannot mean "every
    /// quartet". Purely a denominator for the pre-warm's realized saving -
    /// no code reads it.
    std::size_t denseBytes = 0;
};

/// Plans the Create-time pre-warm: builds the shell pairs and the Schwarz
/// bounds, admits every canonical pair-pair whose Schwarz product reaches
/// SchwarzThreshold(accuracy), and canonicalizes the admitted quartets into
/// the store's own order.
///
/// The enumeration is the same quadratic pair-pair construction the direct
/// builders already run at their own Create (their cached Schwarz neighbor
/// list, fock_build.hpp:703): this path adds no new scaling problem to the
/// family, and above 5000 functions the construction is the one the QFMM
/// leaf-driven domain (fock_build.hpp:543-556) exists to replace.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet The orbital basis (no auxiliary basis is involved: this
/// is the full-ERI path).
/// \param accuracy The screening preset; its Schwarz threshold is the
/// admission rule. kTight admits the most and costs the most disk.
/// \returns The plan, or an Error (from the pair-list or Schwarz-bound
/// construction).
/// \ingroup qcx-storage
qcx::Result<ScreenedEriStorePlan> PlanScreenedEriStore(const qcx::molecule::Molecule& molecule,
                                                       const qcx::basisset::BasisSet& basisSet,
                                                       qcx::integrals::AccuracyPreset accuracy);

/// Executes a plan: creates the store file, evaluates the significant
/// quartets class run by class run, and appends each run.
///
/// A class run is one maximal run of consecutive equal (lBra, lKet) in the
/// plan's canonical order, cut again at maxBatchBytes - the unit
/// EriStore::AppendBatch requires (one run = one chunk = one manifest row)
/// and the unit the decorator's own contract names.
/// \param storePath The store file to create. A fresh file is REQUIRED:
/// EriStore::Create refuses an existing one (kIOError) and this function
/// does not delete a file behind the caller's back - the caller owns the
/// path's freshness, exactly as the driver's disk route does when it clears
/// its scratch path before constructing the builder.
/// \param molecule The system (the store's identity; written to /molecule).
/// \param basisSet The orbital basis (function counts; the fingerprint).
/// \param orbitalBasisName The orbital basis-set name recorded in the store.
/// \param plan The plan to execute (PlanScreenedEriStore).
/// \param accuracy The preset the runs are evaluated at.
/// \param maxBatchBytes The per-run byte cap, and the batch cap forwarded to
/// the engine (must be positive).
/// \returns The raw ERI bytes appended - equal to payloadBytes
/// when the plan came from PlanScreenedEriStore with the same system and
/// preset - or an Error (kInvalidArgument for a zero cap, kIOError for HDF5
/// failures, or the engine's and the store's own errors).
/// \ingroup qcx-storage
qcx::Result<std::size_t> FillScreenedEriStore(const std::filesystem::path& storePath,
                                              const qcx::molecule::Molecule& molecule,
                                              const qcx::basisset::BasisSet& basisSet,
                                              std::string_view orbitalBasisName,
                                              const ScreenedEriStorePlan& plan,
                                              qcx::integrals::AccuracyPreset accuracy,
                                              std::size_t maxBatchBytes = 512 * 1024 * 1024);

} // namespace qcx::storage
