#pragma once

/// \file
/// The chunked RI 3-center tensor store (the disk-backed rung):
/// /integrals/ao/eri/ri_tensor_chunks/{index}
/// - one (storedRowCount x k_c) float64 dataset per chunk in the
/// contraction layout (chunk-scoped), each carrying the per-chunk integrity
/// attributes: the aux shell/function ranges, the stored row count, the
/// FNV-1a-64 checksum over the chunk's byte stream, and the
/// kIntegralEngineVersion stamp: an engine bump invalidates stored
/// chunks - refuse, never serve. Dense format: the
/// Create-time-screened-out blocks are stored as zeros (the
/// storage-not-saved caveat; the dense-format verdict is
/// FORMAT-v1-SCOPED).
///
/// The row space may be the FULL n^2 (pair (u,v) at row u*n + v) or the
/// UV-REDUCED n*(n+1)/2 - see RiChunkMeta. The reduced layout is NOT a
/// point-group mechanism: it is the 3-center integral's own bra-pair
/// permutation symmetry, (uv|P) = (vu|P), which the full layout duplicates
/// byte for byte. Nothing about a chunk's COLUMN space changes with it.

#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>
#include <string_view>

namespace qcx::storage {

/// The chunk identity and range metadata. The chunk covers one contiguous
/// range of aux SHELLS [auxShellStart, auxShellEnd) in the molecule-scoped
/// aux pair-list shell order (a task's block spans a whole shell, so chunk
/// boundaries never split a shell); the dataset's columns are the chunk's
/// aux FUNCTIONS [auxFunctionStart, auxFunctionEnd) of the n^2 x nAux
/// contraction layout, and its rows are the orbital-function pairs
/// (n = orbitalFunctionCount).
///
/// Chunks partition the AUX (column) index only - every chunk's dataset
/// carries the SAME row space, so the row layout below is a per-store
/// property and never a per-chunk one.
///
/// TWO ROW LAYOUTS are allowed, selected by storedRowCount:
///   - `0` (the default, the original contract): the full n^2 rows, pair
///     (u, v) at row u*n + v.
///   - `n*(n+1)/2`: the uv-reduced layout, one row per UNORDERED pair,
///     pair (u, v) with u <= v at row u*n - u*(u-1)/2 + (v-u). The 3-center
///     integral is symmetric in its bra pair, so the full layout stores
///     every off-diagonal row twice - byte for byte, the same doubles - and
///     the reduced layout is the same tensor without the duplicate half.
/// Any other value is refused (the layout's row order is a contract the
/// store cannot verify from a shape alone).
/// \ingroup qcx-storage
struct RiChunkMeta {
    std::size_t chunkIndex = 0; ///< The dataset index (append-only per index).
    std::size_t auxShellStart = 0; ///< First aux shell of the chunk (inclusive).
    std::size_t auxShellEnd = 0; ///< One past the last aux shell of the chunk.
    std::size_t auxFunctionStart = 0; ///< Global aux function offset of column 0.
    std::size_t auxFunctionEnd = 0; ///< One past the last global aux function.
    std::size_t orbitalFunctionCount = 0; ///< The orbital-basis function count n.
    /// The rows the dataset carries: 0 for the full n^2 layout, or
    /// n*(n+1)/2 for the uv-reduced one (see the struct note).
    std::size_t storedRowCount = 0;
};

/// The row count a chunk's dataset carries under its own meta: the meta's
/// storedRowCount when set, n^2 otherwise.
/// \param meta The chunk metadata.
/// \returns The dataset's row count.
/// \ingroup qcx-storage
inline std::size_t RiChunkStoredRowCount(const RiChunkMeta& meta) noexcept {
    return meta.storedRowCount != 0 ? meta.storedRowCount
                                    : meta.orbitalFunctionCount * meta.orbitalFunctionCount;
}

/// The uv-reduced row count of an orbital function count: one row per
/// unordered function pair.
/// \param n The orbital function count.
/// \returns n*(n+1)/2.
/// \ingroup qcx-storage
inline std::size_t RiChunkReducedRowCount(std::size_t n) noexcept {
    return n * (n + 1) / 2;
}

/// The uv-reduced row index of the unordered pair (u, v). Requires u <= v;
/// the caller owns that precondition (a swapped pair is out of contract,
/// not a lookup that silently returns a neighbour's row).
/// \param n The orbital function count.
/// \param u The first function index.
/// \param v The second function index (>= u).
/// \returns The pair's row in the reduced layout.
/// \ingroup qcx-storage
inline std::size_t RiChunkReducedRowIndex(std::size_t n, std::size_t u, std::size_t v) noexcept {
    return u * n - u * (u - 1) / 2 + (v - u);
}

/// Saves one chunk of the RI 3-center tensor as
/// /integrals/ao/eri/ri_tensor_chunks/{meta.chunkIndex}. The matrix is the
/// (storedRowCount x k_c) contraction-layout slice in the meta's row
/// layout (full n^2 rows u*n+v, or uv-reduced rows u<=v; columns are the
/// chunk's aux functions either way); its byte stream (row-major on disk)
/// is covered by the stored FNV-1a-64 checksum. Refuses when the chunk
/// dataset already exists (append-only). The system fingerprint must match
/// the file's /molecule (written when the file is fresh, verified
/// otherwise).
/// \param path The store file (created when missing).
/// \param molecule The system (written to the store's /molecule).
/// \param orbitalBasisName The orbital basis-set name.
/// \param auxBasisName The auxiliary basis-set name.
/// \param meta The chunk identity (validated: the matrix shape must equal
/// {RiChunkStoredRowCount(meta), auxFunctionEnd - auxFunctionStart}).
/// \param chunkMatrix The chunk in the meta's row layout.
/// \returns An Error (kInvalidArgument for shape/range inconsistencies or
/// an existing chunk dataset, kIOError for HDF5 failures).
/// \ingroup qcx-storage
qcx::Result<void> SaveRiTensorChunk(const std::filesystem::path& path,
                                    const qcx::molecule::Molecule& molecule,
                                    std::string_view orbitalBasisName,
                                    std::string_view auxBasisName,
                                    const RiChunkMeta& meta,
                                    const Eigen::MatrixXd& chunkMatrix);

/// Loads one stored chunk. Every serve path is checksummed (the
/// fp32-bounds hole of the EriStore pattern is not repeated) and the
/// shape is validated against the chunk metadata: a dataset whose
/// dimensions disagree with meta is kInvalidArgument (never UB), a
/// checksum mismatch is kIOError (never a silent recompute - the
/// integrity policy), and a chunk stamped under a different
/// kIntegralEngineVersion refuses to serve (kInvalidArgument).
/// \param path The store file.
/// \param molecule The system (must match the stored fingerprint).
/// \param orbitalBasisName The orbital basis-set name.
/// \param auxBasisName The auxiliary basis-set name.
/// \param meta The chunk identity; the loaded dataset's shape, row-layout
/// and range attributes must equal it exactly (a store whose row layout
/// disagrees with meta refuses rather than returning a wrongly-shaped
/// matrix).
/// \returns The chunk in the meta's row layout, or an Error.
/// \ingroup qcx-storage
qcx::Result<Eigen::MatrixXd> LoadRiTensorChunk(const std::filesystem::path& path,
                                               const qcx::molecule::Molecule& molecule,
                                               std::string_view orbitalBasisName,
                                               std::string_view auxBasisName,
                                               const RiChunkMeta& meta);

} // namespace qcx::storage
