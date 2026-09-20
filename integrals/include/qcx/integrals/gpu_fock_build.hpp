#pragma once

/// \file
/// The GPU J/K Fock builder: the DirectJkFockBuilder contract with the
/// J/K contraction moved onto the CUDA device. The per-iteration
/// screening is the SHARED path of
/// internal/fock_screen.hpp (density weight, density gate, certified
/// routing - bit-identical decisions and candidate lists to the CPU
/// builder), the quartets are evaluated by the EriCudaEngine (eri_cuda.hpp),
/// and the contraction runs as the generic per-quartet kernel of
/// eri_cuda_fock.cu (one block per quartet, atomicAdd accumulations).
/// BuildFock is the same two-pass build as the CPU path: the fp64 pass,
/// then the certified fp32 lane with the per-quartet a-priori bounds
/// accumulated into the density-weighted certified sum.
///
/// Requires a CUDA device at Create (kDeviceError otherwise - the
/// self-skipping device-test convention). This header is CUDA-runtime-free
/// (the device state lives inside the builder); no Eigen in this header
/// (integrals public headers keep Eigen implementation-only - the stated
/// CMake policy).

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_cuda.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/precision_policy.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <memory>

namespace qcx::integrals {

/// Builds F(D) = H + 2J(rho) - K(rho) on the CUDA device, with the same
/// public contract as DirectJkFockBuilder (fock_build.hpp): the
/// input density is the SPATIAL closed-shell density rho = D/2 (the
/// direct/RI builder convention - the scf FockBuilderFn seam hands over
/// the spin-summed D, so adapters must scale by 1/2 before calling
/// BuildFock, rhf.hpp documents the seam contract), and the certified
/// bound sum has the same meaning and accumulation as the CPU lane (an
/// upper bound on every Fock-element error from the fp32 lane).
///
/// J and K are accumulated with the 8-fold symmetry folded in, exactly as
/// the CPU builder: every canonical quartet block contributes to the bra
/// and ket pair blocks of J and to the four exchange targets of K (the
/// papers' rule - the density is permuted per batch, integrals are never
/// transposed). The device contraction evaluates the identical formula
/// elementwise (the CPU-vs-device parity gate).
/// \ingroup qcx-integrals
class GpuJkFockBuilder {
public:
    /// Prepares the builder: creates the CUDA engine (pair data, Schwarz
    /// bounds, the device tables), copies the core Hamiltonian, and builds
    /// the cached Schwarz neighbor list - the same preparations as
    /// DirectJkFockBuilder::Create plus the device side.
    /// \param molecule Molecule providing the atom coordinates (Bohr).
    /// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
    /// (kUnimplemented otherwise).
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options Build settings; maxBatchBytes must be positive. The
    /// buildExchangeOnly / buildCoulombOnly splits are honored by the
    /// device contraction like the CPU builder honors them.
    /// \returns The builder, or an Error (kDeviceError when no CUDA device
    /// is available).
    static qcx::Result<GpuJkFockBuilder> Create(
        const qcx::molecule::Molecule& molecule,
        const qcx::basisset::BasisSet& basisSet,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const FockBuildOptions& options = {});

    /// Builds F(D) = H + 2J(rho) - K(rho) on the device for the given
    /// closed-shell density, with the CPU builder's screening semantics
    /// (the shared internal::ScreenAll path) and its two-pass fp64 +
    /// certified-fp32 structure.
    /// \param density The SPATIAL closed-shell density rho = D/2,
    /// host-canonical rank-2 tensor with shape {n, n}. Must be SYMMETRIC:
    /// the canonical-pair contractions read both orientations of every
    /// unordered pair block and the K transpose-writes assume it (the
    /// builder Debug-asserts it like the CPU builder).
    /// \param certifiedBoundSumOut Optional out-parameter receiving the
    /// sum of the DENSITY-WEIGHTED a-priori certified bounds of every
    /// quartet the certified gate routed through the fp32 lane (0.0 when the
    /// lane is disabled): each quartet's kernel bound times its max-|D|
    /// block weight - the same weight the certified routing gate uses, the same
    /// accumulation as the CPU lane. With the precision ladder engaged the
    /// out-parameter carries the ladder's committed band-scaled sum
    /// instead.
    /// \param ladderInputs The per-call precision-ladder inputs
    /// (PrecisionLadderInputs): the SCF's density-convergence
    /// error, the composed RI path's measured error, and the run's
    /// escalation schedule. Null (the default) keeps the pre-ladder
    /// two-pass build bit-identical; engaged, the band classification
    /// replaces the certified routing and the per-band passes dispatch between
    /// class runs.
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        double* certifiedBoundSumOut = nullptr,
        const PrecisionLadderInputs* ladderInputs = nullptr) const;

    /// The core Hamiltonian this builder was constructed with (H in
    /// F = H + 2J(rho) - K(rho)).
    /// \returns The construction-time H, host-canonical rank-2 tensor with
    /// shape {n, n}; a reference into this builder's state (valid for the
    /// builder's lifetime, stable across calls).
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& CoreHamiltonian() const;

    /// The engine's Create-time device footprint estimate: the byte terms
    /// the device workspace budget is charged against.
    /// Computed once at Create; valid for the builder's lifetime.
    /// \returns The estimate.
    /// \ingroup qcx-integrals
    GpuDeviceFootprint DeviceFootprint() const;

    /// The Create-time mode: the FastPath/LightPath
    /// decision generalized to the device context. The light rung uploads
    /// the statics per call and releases them at the call end; the fast
    /// rung retains them from Create.
    /// \returns The mode (kFastPath when no decision surface was given).
    FockBuildMode Mode() const noexcept;

    /// The Create-time mode record: the decision plus
    /// the byte terms it ran on. Purely observational.
    /// \returns The record.
    EriCudaModeInfo ModeInfo() const noexcept;

    /// The per-call transfer accounting: the device-host bus traffic of
    /// the last BuildFock call, by
    /// class - the residency contract's per-iteration statement (density
    /// up, Fock up/down, screened lists, and the statics on the LightPath;
    /// nothing else). Reset at every BuildFock entry.
    /// \returns The last Fock-build call's accounting.
    EriCudaTransferAccounting TransferAccounting() const noexcept;

    /// The device-timeline span of the last BuildFock call's device passes
    /// (the host/device-split instrument,
    /// benchmarks/gpu_eri_benchmark.cpp): cudaEventElapsedTime between
    /// event records on the default stream bracketing each pass's launches
    /// - every device op of the Fock path enqueues on the default stream.
    /// Host-only stretches of the call (screening, the pre-pass assembly,
    /// the read-backs) fall outside the brackets, so wall - span is the
    /// concurrent path's host share and can never underflow. Purely
    /// observational, like the transfer accounting; 0.0 when no pass ran
    /// device work or the engine's event creation failed.
    /// \returns The span in milliseconds of the last BuildFock call.
    double LastBuildDeviceSpanMs() const noexcept;

private:
    // The implementation state lives in eri_cuda.cpp (Eigen stays
    // implementation-only); it owns the EriCudaEngine, whose private
    // section friends this class so the device orchestration can reuse the
    // engine's tables and stream.
    struct State;
    explicit GpuJkFockBuilder(std::shared_ptr<const State> state);
    std::shared_ptr<const State> _state;
};

} // namespace qcx::integrals
