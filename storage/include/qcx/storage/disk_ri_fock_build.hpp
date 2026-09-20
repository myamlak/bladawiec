#pragma once

/// \file
/// The disk-backed RI-J Fock builder: the (uv|P) 3-center
/// tensor's n^2 x nAux contraction
/// matrix lives on disk as the per-chunk datasets of ri_tensor_chunk_store.hpp
/// instead of in RAM, and each BuildFock runs the two RI products as two
/// streamed passes over the chunks (the disk rung - the ladder's LAST rung).
/// Create chunk-builds and chunk-stores the tensor once (BuildRiTensorChunk,
/// then SaveRiTensorChunk - the
/// full-pair-list/tensor materializations of the in-memory fast path never
/// exist); BuildFock then loads every chunk twice per call:
///
///   1. pass 1 accumulates v_c^T = d^T * I_c per chunk (the no-transpose
///      orientation - no per-chunk transpose copy) in
///      fixed serial chunk order. Under the uv-reduced row layout it is
///      the same orientation on a shorter row vector, gathered once per
///      call OUTSIDE the chunk loop, so the invariant is untouched: no
///      per-chunk transpose copy is introduced, and pass 2's result is
///      scattered once per call by the same argument,
///   2. w = V diag(invLambda) V^T v - the in-memory floored eigen-inverse
///      solve, unchanged (the metric stays in RAM at C60-class scope),
///   3. pass 2 accumulates J = sum_c I_c * w_c in the SAME chunk order,
///   4. Fock = exchangeFock + 2J - the exchange half runs through the
///      direct builder's exchange-only mode, exactly like the in-memory
///      RiJkFockBuilder.
///
/// Tensor values are bit-identical to the monolithic path per task; v is
/// bit-identical up to BLAS-internal k-loop blocking; J alone accumulates
/// across chunks and drifts in the last ulps - the product
/// tolerance is a J-only quantity, measured and pinned at
/// implementation (never asserted a priori). The uv-reduced row layout
/// changes none of that: it stores no new value and recomputes none (the
/// rows it drops are byte-for-byte duplicates of the rows it keeps), so
/// J is bit-identical to the full layout's and only v re-associates the
/// same addends.

#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/storage/ri_tensor_chunk_store.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

namespace qcx::storage {

/// Settings of the disk-backed RI-J builder.
/// \ingroup qcx-storage
struct DiskRiFockOptions {
    /// Screening preset - the same semantics as
    /// qcx::integrals::RiEngineOptions::accuracy: it feeds the 3c screen of
    /// every chunk build and the nested direct-exchange half (the
    /// fp32 lane gate, fock_build.hpp).
    qcx::integrals::AccuracyPreset accuracy = qcx::integrals::AccuracyPreset::kNormal;
    /// Per-class batch cap of the chunk builds and of the nested
    /// exchange half (bytes; must be positive).
    std::size_t maxBatchBytes = 512 * 1024 * 1024;
    /// Target chunk size (bytes) of the chunking formula.
    /// The plan's chunk-size floor applies on top: the effective chunk
    /// size is max(chunkBytes, 8 * n^2 * maxShellFunctionCount(aux)) - one
    /// def2-jfit aux shell column is 8 * n^2 bytes, so at 5000-function
    /// orbital scales the shell floor dominates and the 256 MiB default is
    /// unattainable by design. The chunk plan is a pure function of (basis
    /// data, options); chunk boundaries never split an aux shell.
    std::size_t chunkBytes = 256 * 1024 * 1024;
    /// Relative floor for the (P|Q) metric inverse - identical semantics
    /// to qcx::integrals::RiEngineOptions::metricFloorEpsilon (the default
    /// 1e-10 equals kMetricFloorEpsilon). The eigen-inverse recipe is
    /// duplicated from the in-memory engine (its implementation is
    /// file-local there); the values match.
    double metricFloorEpsilon = 1e-10;
    /// Store the chunks in the uv-reduced row layout (RiChunkMeta) instead
    /// of the full n^2: one row per unordered bra pair, dropping the
    /// byte-for-byte duplicate half the 3-center integral's own bra-pair
    /// permutation symmetry creates. This is NOT a point-group mechanism -
    /// the group takes no part in it. Turning it off restores the original
    /// n^2 store, which is the layout the on-disk model
    /// (`8 n^2 nAux`, synthesized driver-side) describes; with it on, the
    /// store's payload is n(n+1)/2 / n^2 of that model. Both layouts run
    /// the same two streamed passes, and the reduced one is the same
    /// numbers: the tensor values are untouched (the dropped rows are
    /// duplicates) and J is bit-identical (each pair's dot product sees
    /// the identical row); v only re-associates the same addends.
    bool reducedRowLayout = true;
};

/// The streamed RI-J Fock builder over chunked on-disk 3-center integrals.
///
/// Create chunk-builds and chunk-stores the (uv|P) tensor (BuildRiTensorChunk
/// into the chunked store - per-chunk FNV-1a-64 checksum,
/// kIntegralEngineVersion stamp)
/// and precomputes the metric's floored eigen-inverse and the direct
/// exchange half; each BuildFock call streams the two RI contractions over
/// the stored chunks (two passes, fixed serial chunk order) plus the direct
/// exchange build - no integrals are evaluated per iteration. The builder
/// consumes F(D) = H + 2J_RI(rho) - K(rho) densities in the SPATIAL
/// closed-shell convention rho = D/2, exactly like the in-memory
/// RiJkFockBuilder (qcx/integrals/ri_engine.hpp documents the seam).
///
/// **It also exposes that fused build's two halves separately**, which is
/// what makes the disk rung reachable from the Kohn-Sham energy seam and
/// from the unrestricted per-spin assembly rather than only from a
/// closed-shell Hartree-Fock run: `BuildCoulombOnly` is the RI-J half and
/// `BuildExchangeOnly` is the nested direct builder's exchange-only result.
/// The pair is `RiJkFockBuilder`'s own pair, entry point for entry point and
/// number for number - this class IS the ri_j_link family's disk rung - so
/// the accounting is that family's and NOT the driver's `HalfFockFn`
/// contract: **the Coulomb half carries the factor of two and no core
/// Hamiltonian, and the exchange half is the sole H carrier**. The driver's
/// adapters state that difference at their call sites
/// (`MakeRiJLinkKsHalf` adds H to the Coulomb half). Both halves are the
/// same code the fused call runs - `BuildFock` is their sum - so a half and
/// the fused build cannot drift about the rung they took or the convention's
/// factor of two.
/// \ingroup qcx-storage
class DiskRiFockBuilder {
public:
    /// Prepares the builder: chunk-builds and chunk-stores the 3c tensor,
    /// computes the metric's floored eigen-inverse, and prepares the
    /// direct-exchange half.
    /// \param storePath The HDF5 store file. A fresh file is created; an
    /// existing ERI/chunk store file is allowed (its /molecule group is
    /// verified element-wise against this system - the shared
    /// store_molecule machinery). A file that already carries any chunk
    /// dataset refuses (the append-only store contract) - a store path is
    /// single-builder scratch.
    /// \param molecule The molecule providing the atom coordinates (Bohr),
    /// taken BY VALUE: qcx::molecule::Molecule is move-only and the builder
    /// retains it for the per-load /molecule verification of every
    /// BuildFock chunk read (callers move it in).
    /// \param orbitalBasis Orbital basis (same contract as the in-memory
    /// builder: every shell must satisfy l <= kMaxEngineL).
    /// \param auxBasis Auxiliary basis (every shell must satisfy
    /// l <= kMaxEngineL; every atom must have an entry).
    /// \param orbitalBasisName The orbital basis-set name recorded in the
    /// store's /molecule group.
    /// \param auxBasisName The auxiliary basis-set name.
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options Build settings; maxBatchBytes must be positive.
    /// \returns The builder, or an Error (kInvalidArgument for a
    /// degenerate metric below the floor, a zero maxBatchBytes, or a store
    /// file that already carries chunks; kIOError for genuine HDF5
    /// failures).
    static qcx::Result<DiskRiFockBuilder> Create(
        const std::filesystem::path& storePath,
        qcx::molecule::Molecule molecule,
        const qcx::basisset::BasisSet& orbitalBasis,
        const qcx::basisset::BasisSet& auxBasis,
        std::string_view orbitalBasisName,
        std::string_view auxBasisName,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const DiskRiFockOptions& options = {});

    /// Builds F(D) = H + 2J_RI(rho) - K(rho) for the given closed-shell
    /// density: the direct exchange half plus the two streamed RI passes
    /// over the stored chunks.
    /// \param density The SPATIAL closed-shell density rho = D/2
    /// (spin-summed density over 2) - the in-memory builder's convention,
    /// forwarded unchanged to the nested exchange half. Must be SYMMETRIC:
    /// the RI-J half below accepts any density, but the exchange half it is
    /// summed with does not (see BuildExchangeOnly), and this call runs that
    /// half first. An asymmetric density is REFUSED with kInvalidArgument
    /// before any work, in Release as well as Debug - the nested builder's
    /// own check is a Debug assert, which NDEBUG removes.
    /// \param statsOut Optional per-call diagnostics sink
    /// (qcx::integrals::FockBuildStats): the nested direct-exchange build's
    /// quartet counts are forwarded when a pointer is given. The RI
    /// contractions contribute no quartet counts.
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        qcx::integrals::FockBuildStats* statsOut = nullptr) const;

    /// Builds the Coulomb half on its own: the RI-J term 2 J_RI(rho) the
    /// fused BuildFock adds to the exchange half, with nothing else in it -
    /// no core Hamiltonian and no exchange. This is
    /// RiJkFockBuilder::BuildCoulombOnly's accounting verbatim
    /// (qcx/integrals/ri_engine.hpp), which the driver's per-spin assembly
    /// and Kohn-Sham composition are already written against: the
    /// unrestricted seam contracts it ONCE on the half-summed density
    /// 0.5 (P_alpha + P_beta), whose 2 J_RI is the J[D_total] of the
    /// unrestricted Fock, and the Kohn-Sham composition contracts it for the
    /// J[D] of its energy seam.
    ///
    /// **The core Hamiltonian is deliberately absent**, and that is the
    /// accounting difference from the direct and qfmm families, whose
    /// Coulomb halves carry a full H: on this path the exchange-only call is
    /// the sole H carrier, so an assembly adds the two halves with no
    /// subtraction. Folding H in here moves an assembled Fock by an entire
    /// core Hamiltonian and the energy by Tr[D H], with nothing in the
    /// result to show it.
    /// \param density The density to contract, host-canonical rank-2 tensor
    /// with shape {n, n}. The caller halves it for an unrestricted assembly
    /// (the factor of two is in this half, so it is passed rho, not P_sigma).
    /// NO SYMMETRY IS REQUIRED, and that is this half's own property rather
    /// than the builder's: the uv-reduced gather reads d(u,v) + d(v,u), the
    /// exact sum of the two terms the full layout contributes at that pair,
    /// and J is symmetric for any D (disk_ri_fock_build.cpp,
    /// BuildRiCoulombVector). The exchange half does require a symmetric
    /// density and refuses anything else - see BuildExchangeOnly.
    /// \param statsOut Optional per-call diagnostics sink
    /// (qcx::integrals::FockBuildStats). The streamed contractions evaluate
    /// no quartets, so this call WRITES a default (all-zero) struct rather
    /// than leaving a stale one behind - the measured zero that is the
    /// structural record of the absent exchange half, the same rule the
    /// in-memory builder's split call follows. Null skips the write.
    /// \returns The n x n Coulomb half 2 J_RI(rho), or an Error (a failed
    /// chunk load - the checksum or engine stamp of the chunk store).
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoulombOnly(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        qcx::integrals::FockBuildStats* statsOut = nullptr) const;

    /// Builds the exchange half on its own: the nested direct-exchange
    /// builder's result verbatim - H - K(rho), the exchange-only mode of
    /// qcx::integrals::DirectJkFockBuilder (fock_build.hpp) that Create
    /// configured from this builder's own options - and nothing else in it.
    /// No chunk is read: this half is the direct build, so it costs what the
    /// fused call's exchange half costs and no RI streaming at all.
    ///
    /// It is the SAME nested builder the fused BuildFock runs, so the K half
    /// of an unrestricted or Kohn-Sham run is the K half of the restricted
    /// one - one accuracy preset, one batch cap, one certified-lane default,
    /// one exchange option recipe - rather than a second wiring that could
    /// drift from it.
    /// \param density The spin density P_sigma, host-canonical rank-2 tensor
    /// with shape {n, n}: the caller passes it RAW (no halving - the
    /// exchange-only builder's -K is linear in its input, so P_sigma is the
    /// K(P_sigma) of the unrestricted channel, not the K(rho) of the closed
    /// shell). Must be SYMMETRIC - the nested direct-exchange builder's
    /// documented precondition (qcx::integrals::DirectJkFockBuilder: the
    /// canonical-pair contractions read both orientations of every unordered
    /// pair block, and the K transpose-writes assume the symmetry). The
    /// unrestricted assembly's per-spin P_sigma and the closed-shell rho are
    /// both symmetric, so the precondition holds on every production path.
    /// An asymmetric density is REFUSED with kInvalidArgument here, in
    /// Release as well as Debug: the nested builder's check is a Debug
    /// assert, and without this one a Release run would return a K that its
    /// transpose-writes had locked symmetric - a wrong matrix, silently.
    /// The Coulomb half (BuildCoulombOnly) carries no such requirement and
    /// accepts any density.
    /// \param statsOut Optional per-call diagnostics sink, forwarded to the
    /// nested exchange build, which fills the quartet counts; null keeps the
    /// zero-cost path.
    /// \returns The n x n exchange half H - K(rho), or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildExchangeOnly(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        qcx::integrals::FockBuildStats* statsOut = nullptr) const;

    /// The Create-time chunk plan: one meta per stored chunk
    /// in chunk order, each with its aux shell/function range. The plan is
    /// a pure function of (basis data, options) - exposed for the
    /// determinism tests and the on-disk-vs-modeled reconciliation.
    /// \returns The chunk metas of this builder's store.
    const std::vector<RiChunkMeta>& Chunks() const noexcept;

private:
    struct State;
    explicit DiskRiFockBuilder(std::shared_ptr<const State> state);

    /// The RI-J contraction itself, in the fused call's own flat layout (the
    /// u*n + v column-major vector), returned as the 2 J_RI(rho) the Fock
    /// assembly adds: the two streamed chunk passes and the metric's floored
    /// eigen-inverse solve in between, with the uv-reduced layout's scatter
    /// already applied. One code path serves the fused BuildFock and the
    /// split BuildCoulombOnly, so the two cannot drift about the rung they
    /// took, the chunk order they accumulated in, or the convention's factor
    /// of two - the in-memory builder's BuildRiCoulombVector rule, which
    /// this mirrors.
    /// \param density The SPATIAL density, shape {n, n}; the caller has
    /// already applied whatever halving its convention requires.
    /// \returns The flat n^2 vector 2 J_RI(rho), or an Error (a failed chunk
    /// load - the checksum or engine stamp of the chunk store).
    qcx::Result<Eigen::VectorXd> BuildRiCoulombVector(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const;

    std::shared_ptr<const State> _state;
};

} // namespace qcx::storage
