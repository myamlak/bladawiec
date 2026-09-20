#pragma once

/// \file
/// The CUDA lane of the matrix-form MD ERI engine: the same per-class
/// {L_bra, L_ket} pipeline as the CPU kernels of
/// eri_batch.hpp, executed on a CUDA device behind QCX_ENABLE_CUDA. The
/// engine owns the device-side pair data (uploaded once at Create) and
/// evaluates caller-supplied quartet lists per call - the same
/// canonicalization semantics as the CPU path (the batch machinery reports
/// the canonicalized form actually evaluated).
///
/// Three device variants are selected per class by the heuristic of
/// eri_cuda.cpp (override-able through EriCudaOptions::variant):
///   kV0 (L <= 4):   register-resident fused kernel, 128 threads per block.
///   kV1 (5..8):     fused kernel with shared-memory intermediates, 256
///                   threads per block.
///   kV2 (L >= 9):   VRR kernel to global memory + strided-batched cuBLAS
///                   GEMMs for the ket transform, per-task GEMMs for the bra.
/// kV0/kV1 overrides stay within the fused family; a kV2 override on a fused
/// class (or the reverse) is kUnimplemented.
///
/// The certified fp32 lane mirrors ComputeEriBatchCertified: the
/// fp32 values with the a-priori per-quartet error bounds, exact same bound
/// formula as the CPU lane. The 3-center RI path (ComputeRiBatch) is fp64
/// only and runs bra pairs x aux shells without canonicalization (the
/// md_vrr_3c.hpp contract, i <= j validated).
///
/// This header is CUDA-runtime-free: streams pass as void* and the device
/// state lives behind a pimpl. The engine is movable, not copyable.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/device_workspace_budget.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <memory>
#include <vector>

/// The instantiation cap of the CUDA matrix: classes with lKet <= 2 *
/// QcxIntegralsCudaLMax (and lBra <= 2 * QcxIntegralsCudaLMax when lKet <=
/// QcxIntegralsCudaLMax - the RI rectangle). 0 means the knob was not set
/// (the build defines it PUBLIC on the CUDA target; a consumer without the
/// define then sees the full matrix as unimplemented).
#ifndef QcxIntegralsCudaLMax
#define QcxIntegralsCudaLMax 0
#endif

namespace qcx::integrals {

/// The device variant override (EriCudaOptions::variant); kAuto selects per
/// class by the L = lBra + lKet heuristic.
/// \ingroup qcx-integrals
enum class EriCudaVariant {
    kAuto, ///< The heuristic: kV0 for L <= 4, kV1 for 5 <= L <= 8, kV2 for L >= 9.
    kV0, ///< Fused, register-resident (128 threads).
    kV1, ///< Fused, shared-memory intermediates (256 threads).
    kV2, ///< Global-memory VRR + cuBLAS transform GEMMs.
};

/// CUDA engine settings.
/// \ingroup qcx-integrals
struct EriCudaOptions {
    std::size_t maxBatchBytes =
        512 * 1024 * 1024; ///< Per-class batch cap (512 MB, same default as the CPU).
    EriCudaVariant variant = EriCudaVariant::kAuto; ///< Variant override; see EriCudaVariant.
    /// The device workspace budget mirror the Create-time mode decision
    /// runs against - the FastPath/LightPath surface
    /// generalized to the device context. Null (the default) keeps the
    /// legacy behavior: the engine always takes the fast rung unless
    /// forceLightPath forces the light rung. The caller owns the budget.
    qcx::memory::DeviceWorkspaceBudget* deviceWorkspaceBudget = nullptr;
    /// The forced-LightPath knob - true FORCES the
    /// per-call-statics rung (the mode-forcing test surface, the same role
    /// as FockBuildOptions::lightPathChunkPairs on the CPU). The engine
    /// then uploads the pair tables and Boys copy per call (EnsureStatics
    /// in the per-call paths) and releases them at the call end instead of
    /// retaining them from Create.
    bool forceLightPath = false;
};

/// The Create-time mode record of the GPU engine: the
/// that decision generalized to the device context, plus the byte terms
/// the decision ran on. Purely observational - no later code reads the
/// mode or the terms; the decision is consumed at Create (the mode governs
/// whether the per-call paths upload the statics per call or use the
/// retained copies).
/// \ingroup qcx-integrals
struct EriCudaModeInfo {
    FockBuildMode mode = FockBuildMode::kFastPath; ///< The decision.
    std::size_t fastEstimateBytes = 0; ///< The fast rung: the full footprint
                                       ///< (GpuDeviceFootprint::Total) - everything resident.
    std::size_t lightEstimateBytes = 0; ///< The light rung: the per-call working set
                                        ///< (matrices + batch scratch + per-call output) - the
                                        ///< statics are uploaded per call and released.
    std::size_t remainingAtDecision = 0; ///< The budget's Remaining() at the decision (0 when
                                         ///< no budget was set).
};

/// The per-call device-host transfer accounting - the residency contract:
/// per-iteration bus traffic = density up, Fock up/down, screened lists -
/// and nothing else, at any system
/// size). Every transfer site of the Fock-build path is wired into these
/// sums; the byte counts are the exact buffer sizes at the sites, so the
/// record is exact by construction. The statics upload once per run on the
/// FastPath (0 per call - the retained copy) and per call on the LightPath
/// (the per-call upload). The certified lane's per-quartet bound-sum
/// read-back is the bounds class; the engine's batch read-back APIs
/// (ComputeBatch and friends) are outside the residency contract but
/// accumulate into this record (their task/ranges uploads are the list
/// class); the BuildFock entry reset wipes them. Covers the last
/// BuildFock call; the debug
/// assert in BuildFock verifies the accounted math against this call's
/// in-scope quantities (a transfer added without a wiring increment breaks
/// the contract loudly in debug builds).
/// \ingroup qcx-integrals
struct EriCudaTransferAccounting {
    std::size_t densityUploadBytes = 0; ///< The density matrix up (n^2 doubles).
    std::size_t fockUploadBytes = 0; ///< The Fock buffer up (the H-seeded pre-fill).
    std::size_t fockDownloadBytes = 0; ///< The Fock matrix down (the read-back).
    std::size_t listUploadBytes = 0; ///< The screened task/ranges/meta lists up
                                     ///< (the screened quartets only, never the all-survive set).
    std::size_t boundsDownloadBytes = 0; ///< The certified lane's per-quartet bound-sum read-back
                                         ///< (0 when the fp32 lane did not run).
    std::size_t staticsUploadBytes = 0; ///< The statics upload: 0 per call on the FastPath
                                        ///< (uploaded once at Create), the full table + Boys bytes
                                        ///< per call on the LightPath.

    /// The call's total bus traffic: the sum of the named classes.
    /// \returns The per-call transfer total in bytes.
    std::size_t Total() const noexcept {
        return densityUploadBytes + fockUploadBytes + fockDownloadBytes + listUploadBytes +
               boundsDownloadBytes + staticsUploadBytes;
    }
};

/// The engine's Create-time device footprint estimate: the byte terms the
/// device workspace budget is charged against - the exact uploaded table
/// sizes, the fixed Boys copy, the per-call density/Fock matrices, the kV2
/// per-batch global scratch, the per-call output mass and the structural
/// allowance. Computed once at Create from the pair data,
/// the options and the variant heuristic; every term is a byte bound, never
/// a guess (the formulas mirror the allocation shapes of eri_cuda.cpp).
/// \ingroup qcx-integrals
struct GpuDeviceFootprint {
    std::size_t tablesBytes = 0; ///< The uploaded pair tables: the pair meta, the primitive
                                 ///< data and the bra/ket transforms of both lanes (the exact
                                 ///< hostTables sizes).
    std::size_t boysBytes = 0; ///< The fixed Boys tables copy.
    std::size_t matricesBytes = 0; ///< The per-call density and Fock device copies (2 x 8 n^2).
    std::size_t batchScratchBytes = 0; ///< The kV2 per-batch pq/acc/bound arena (one batch at a
                                       ///< time; 0 when no class maps to kV2).
    std::size_t perCallOutputBytes = 0; ///< The per-call out/bounds/task buffers of one
                                        ///< contraction pass (the screened-exchange envelope).
    std::size_t riResidencyBytes = 0; ///< RI/tree device residency (the named slot; 0 until the
                                      ///< RI path's device residency lands).
    std::size_t structuralBytes = 0; ///< The cuBLAS workspace (kV2 only) and the driver pool
                                     ///< (the first-kernel module loads, retained by the context).

    /// The estimated device footprint of one BuildFock call: the resident
    /// terms (tables, Boys, structural) plus the per-call working set
    /// (matrices, batch scratch, per-call output).
    /// \returns The sum of all terms in bytes.
    std::size_t Total() const noexcept {
        return tablesBytes + boysBytes + matricesBytes + batchScratchBytes + perCallOutputBytes +
               riResidencyBytes + structuralBytes;
    }
};

/// One fp64 batch result of the CUDA engine: values packed per the
/// eri_batch.hpp layout contract, and the canonicalized quartets in the same
/// order (same semantics as EriBatch).
/// \ingroup qcx-integrals
struct EriCudaBatch {
    std::vector<double> values; ///< Packed blocks, one per computed quartet.
    std::vector<ShellQuartet> computed; ///< The canonicalized quartets actually evaluated.
};

/// One certified-mixed-precision batch result: the fp32 values with
/// the a-priori per-quartet error bounds (same formula as the CPU lane).
/// \ingroup qcx-integrals
struct EriCudaCertifiedBatch {
    std::vector<float> values; ///< Packed blocks (same layout contract).
    std::vector<double> errorBounds; ///< A-priori |error| bound per computed quartet.
    std::vector<ShellQuartet> computed; ///< The canonicalized quartets actually evaluated.
};

/// One fp64 3c batch result: packed blocks in the ri_engine.hpp order
/// (per task: bra.nFuncs x aux.nFuncs, (f_b * n_i + f_a) rows, (row * nAng +
/// fang) columns), with the (i, j, k) task list actually evaluated.
/// \ingroup qcx-integrals
struct EriCudaRiBatch {
    std::vector<double> values; ///< Packed blocks, one per computed triple.
    std::vector<ShellTriple> computed; ///< The (i, j, k) triples in value order.
};

/// The CUDA MD ERI engine: owns the uploaded device pair data and runs
/// quartet/triple batches per call on the current CUDA device of the
/// creating thread.
///
/// \note The RI path keeps the caller's Molecule/BasisSet/auxiliary-BasisSet
/// alive only for the duration of the call - but the pair store (and thus
/// the atom coordinates) is uploaded at Create, so the molecule/basis must
/// outlive the engine. The auxiliary basis is used per call only.
/// \ingroup qcx-integrals
class EriCudaEngine {
public:
    /// Creates the engine: builds the pair data of every canonical shell
    /// pair (same construction as the CPU path), flattens it into the
    /// device tables, uploads the Boys tables, and opens the cuBLAS handle.
    /// \param molecule Molecule providing the atom coordinates (Bohr); must
    /// outlive the engine.
    /// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
    /// (kUnimplemented otherwise) and every atom must have a basis entry
    /// (kInvalidArgument otherwise).
    /// \param options Engine settings; maxBatchBytes must be positive.
    /// \returns The engine, or an Error (kDeviceError when no CUDA device
    /// is available or a device allocation/handle step fails).
    /// \ingroup qcx-integrals
    static Result<EriCudaEngine> Create(const qcx::molecule::Molecule& molecule,
                                        const qcx::basisset::BasisSet& basisSet,
                                        const EriCudaOptions& options = {});

    /// Evaluates the given quartets through the fp64 device pipeline. Same
    /// canonicalization semantics as ComputeEriBatch: each quartet is
    /// canonicalized by total class and reported through
    /// EriCudaBatch::computed; entries of the same class and contraction
    /// degree are contiguous.
    /// \param quartets The quartets to evaluate, in any form.
    /// \param stream The CUDA stream to execute on (nullptr = the default
    /// stream); results are synchronously copied back before returning.
    /// \returns The packed batch, or an Error (kUnimplemented for classes
    /// outside the instantiated matrix or for unsupported variant
    /// overrides, kInvalidArgument for empty lists).
    /// \ingroup qcx-integrals
    Result<EriCudaBatch> ComputeBatch(const std::vector<ShellQuartet>& quartets,
                                      void* stream = nullptr);

    /// Evaluates the given quartets through the certified fp32 device
    /// pipeline: every returned block carries its a-priori error
    /// bound, same formula as the CPU lane. Requires the fp32 pipeline to be
    /// instantiated (the QCX_INTEGRALS_F32 knob, on by default).
    /// \param quartets The quartets to evaluate, in any form.
    /// \param stream The CUDA stream to execute on (nullptr = default).
    /// \returns The certified batch, or an Error (kUnimplemented when this
    /// build has no fp32 pipeline).
    /// \ingroup qcx-integrals
    Result<EriCudaCertifiedBatch> ComputeBatchCertified(const std::vector<ShellQuartet>& quartets,
                                                        void* stream = nullptr);

    /// Evaluates the 3-center integrals (uv|P) of the given triples through
    /// the fp64 device pipeline: bra shell pair (i, j) with i <= j, ket
    /// shell k of the auxiliary basis. No canonicalization (the RI path has
    /// no 8-fold symmetry); the computed triples come back in value order.
    /// \param auxBasis The auxiliary basis set (RI fitting basis).
    /// \param triples The (i, j, k) triples to evaluate.
    /// \param stream The CUDA stream to execute on (nullptr = default).
    /// \returns The packed batch, or an Error (kUnimplemented for classes
    /// outside the instantiated matrix, kInvalidArgument for empty lists or
    /// out-of-range triples).
    /// \ingroup qcx-integrals
    Result<EriCudaRiBatch> ComputeRiBatch(const qcx::basisset::BasisSet& auxBasis,
                                          const std::vector<ShellTriple>& triples,
                                          void* stream = nullptr);

    /// The Create-time device footprint estimate: the byte terms the
    /// device workspace budget is charged against. Computed
    /// once at Create; valid for the engine's lifetime.
    /// \returns The estimate (all fields; zero only on a never-created
    /// engine).
    /// \ingroup qcx-integrals
    GpuDeviceFootprint DeviceFootprint() const;

    /// The Create-time mode: the FastPath/LightPath
    /// decision generalized to the device context. The light rung uploads
    /// the statics per call and releases them at the call end; the fast
    /// rung retains them from Create.
    /// \returns The mode (kFastPath when no decision surface was given).
    /// \ingroup qcx-integrals
    FockBuildMode Mode() const noexcept;

    /// The Create-time mode record: the decision plus
    /// the byte terms it ran on (the rung estimates and the budget's
    /// remaining bytes at the decision). Purely observational.
    /// \returns The record.
    /// \ingroup qcx-integrals
    EriCudaModeInfo ModeInfo() const noexcept;

    /// The per-call transfer accounting: the device-host bus traffic of
    /// the last BuildFock call, by
    /// class - the residency contract's per-iteration statement (density
    /// up, Fock up/down, screened lists, and the statics on the LightPath;
    /// nothing else). Reset at every BuildFock entry.
    /// \returns The last Fock-build call's accounting.
    /// \ingroup qcx-integrals
    EriCudaTransferAccounting TransferAccounting() const noexcept;

    /// The engine is movable, not copyable (owns device resources).
    EriCudaEngine(EriCudaEngine&&) noexcept = default;
    /// The move-assignment is defaulted: it moves the device resources.
    /// \returns *this (the defaulted move-assignment contract).
    EriCudaEngine& operator=(EriCudaEngine&&) noexcept = default;
    EriCudaEngine(const EriCudaEngine&) = delete;
    EriCudaEngine& operator=(const EriCudaEngine&) = delete;
    /// Defined out-of-line in eri_cuda.cpp where Impl is complete (the
    /// pimpl rule: a defaulted in-header destructor would be instantiated
    /// in every consumer TU and fail on the incomplete Impl).
    ~EriCudaEngine();

private:
    struct Impl;

    // The GPU Fock builder reuses the engine's device tables
    // and stream for its per-iteration contraction: BuildFock drives the
    // same RunBatchOnDevice batches the ComputeBatch path uses, so it needs
    // the Impl internals (pair store, host pair meta, device handles).
    friend class GpuJkFockBuilder;

    explicit EriCudaEngine(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> _impl;
};

} // namespace qcx::integrals
