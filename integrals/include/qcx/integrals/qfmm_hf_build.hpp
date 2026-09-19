#pragma once

/// \file
/// The QFMM composed HF builder: the RIJCOSX-style combination point applied
/// to QFMM - the
/// Coulomb half through the QfmmJBuilder (near field + multipole far field,
/// H + 2J) and the exchange half through a DirectJkFockBuilder in its
/// exchange-only mode (H - K), composed as H + 2J_QFMM(rho) - K(rho) with
/// one core-Hamiltonian copy subtracted back (the double-H trap of the
/// direct-UHF adapter - every split builder result carries its own
/// full H copy). The QFMM builder computes Coulomb ONLY (its multipole far
/// field has no meaningful exchange analog - qfmm_fock_build.hpp), so K
/// stays with the direct exchange-only builder. The same two nested states
/// serve the UHF channels: the Coulomb half runs once on the
/// half-summed density 0.5 (P_alpha + P_beta) and the exchange half once
/// per spin on the raw spin densities, one H copy subtracted back per
/// channel - the MakeDirectUhfFockBuilder assembly of the driver
/// with the QFMM builder in the coulomb slot.
///
/// The three nested states (the QFMM outer store, its near-field direct
/// store, and this exchange store) are the footprint note - the
/// double store the adaptive design already counts. With a
/// workspace budget (QfmmOptions::workspaceBudget) the shared counter
/// prices all three stores at Create time: the QFMM half's estimate
/// (outer store plus its nested near-field builder) reserves first, then
/// this exchange half's own Create-time estimate - nesting order =
/// reservation order, the same composition the direct-UHF driver path
/// uses (fock_build.hpp FockModeInfo).
///
/// No Eigen in this header (integrals public headers keep Eigen
/// implementation-only - the stated CMake policy).

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

namespace qcx::integrals {

/// Builds F(D) = H + 2J(rho) - K(rho) for closed-shell RHF (the
/// RIJCOSX-style composition): the Coulomb half is the
/// QfmmJBuilder (the octree near field through the restricted direct
/// Coulomb builder plus the multipole far field - qfmm_fock_build.hpp) and
/// the exchange half is a DirectJkFockBuilder in its exchange-only mode
/// (fock_build.hpp buildExchangeOnly, the H - K(rho) form). Each
/// BuildFock call composes the two split results and subtracts one core-
/// Hamiltonian copy (every split result carries one full H - the double-H
/// trap); the composed result is the fused direct builder's Fock matrix up
/// to the QFMM far-field approximation (the recorded per-preset QFMM
/// budgets {1e-5, 1e-7, 1e-8} Eh, the kTight rung re-derived
/// from 1e-9 in the 2026-09-13 measurement) and the split-pass summation
/// order. The input density is the SPATIAL closed-shell density rho = D/2,
/// the DirectJkFockBuilder convention (the scf FockBuilderFn seam's
/// adapters scale by 1/2).
/// \ingroup qcx-integrals
class QfmmHfFockBuilder {
public:
    /// Prepares the builder: constructs the QfmmJBuilder (pairs, octree,
    /// interaction lists, moment table, restricted near-field direct
    /// builder) and the exchange-only DirectJkFockBuilder - all one-time,
    /// geometry-only work. The exchange half inherits the QfmmOptions
    /// pass-throughs the QFMM near field gets (useDensityScreening,
    /// useCertifiedMixedPrecision, maxParallelChunks - the serial pin
    /// reaches both halves) and the accuracy preset.
    /// \param molecule Molecule providing the atom coordinates (Bohr).
    /// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
    /// (kUnimplemented otherwise, from the nested builders).
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options QFMM build settings (validated by the nested QFMM
    /// Create: theta must not be NaN, lMult must be -1 or in
    /// 0..kQfmmMaxLMult, maxLeafSize must be >= 1). The adaptive-memory
    /// workspaceBudget, when set, is shared with both nested Creates (the
    /// QFMM half first - nesting order = reservation order, fock_build.hpp
    /// FockModeInfo): the QFMM Create-time estimate prices the outer store
    /// plus its nested near-field builder, then the exchange half's own
    /// Create-time estimate follows, so the triple-store footprint
    /// is budgeted rather than left to the legacy path. Null (the
    /// default) keeps the legacy behavior exactly.
    /// \returns The builder, or an Error.
    static qcx::Result<QfmmHfFockBuilder> Create(
        const qcx::molecule::Molecule& molecule,
        const qcx::basisset::BasisSet& basisSet,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const QfmmOptions& options = {});

    /// Builds F(D) = H + 2J_QFMM(rho) - K(rho) for the given closed-shell
    /// density: the QfmmJBuilder result (H + 2J, near + far) plus the
    /// exchange-only direct result (H - K) minus one core-Hamiltonian copy.
    /// The two summands ARE BuildCoulombOnly and BuildExchangeOnly at this
    /// density (this call is written as exactly those two, in that order),
    /// so a Kohn-Sham caller composing the halves has the Fock's own
    /// Coulomb half rather than a second contraction of it.
    /// \param density The SPATIAL closed-shell density rho = D/2, host-
    /// canonical rank-2 tensor with shape {n, n}. Must be SYMMETRIC (the
    /// nested builders' documented precondition - the canonical-pair
    /// contractions read both orientations of every unordered pair block).
    /// \returns The Fock matrix as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const;

    /// Builds the composed builder's Coulomb half ALONE, in the direct
    /// family's buildCoulombOnly convention: F_J(rho) = H + 2J_QFMM(rho).
    ///
    /// This is the SAME nested QfmmJBuilder call BuildFock opens with, so the
    /// Coulomb half a caller composes with is that Fock's own - not a second
    /// contraction of the same density that happens to agree. It exists for
    /// the Kohn-Sham energy seam, whose formula contracts J[D] separately
    /// from the Fock (scf/rhf.hpp, the seam line restated in the driver's
    /// internal/ks_composition.hpp): a fused BuildFock result cannot be split
    /// back into the halves, and a caller that rebuilt the J half from its
    /// own Create would have a second builder, a second set of Create-time
    /// decisions and a second budget reservation to keep in step by hand.
    ///
    /// \param density The SPATIAL density in the family's own convention: the
    /// closed-shell rho = D/2 (direct-JkFockBuilder's), or the
    /// half-summed 0.5 (P_alpha + P_beta) of an unrestricted Coulomb call -
    /// the J loop's baked 2.0 turns either into the physical J of the
    /// spin-summed density. Host-canonical rank-2 tensor with shape {n, n};
    /// must be SYMMETRIC (the nested builder's documented precondition).
    /// \returns The half as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoulombOnly(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const;

    /// Builds the composed builder's exchange half ALONE, in the direct
    /// family's buildExchangeOnly convention: F_K(rho) = H - K(rho).
    ///
    /// As BuildCoulombOnly: the nested exchange-only DirectJkFockBuilder call
    /// BuildFock adds, at the density handed in. The density is the RAW spin
    /// density of the channel the half serves - the -K is linear in its
    /// input, so an unrestricted caller passes P_sigma with no halving (the
    /// BuildUhfFock convention) - and the half is H - K[D]/2 for a
    /// closed-shell caller passing rho = D/2.
    /// \param density The density the exchange is contracted against:
    /// host-canonical rank-2 tensor with shape {n, n}, must be SYMMETRIC.
    /// \returns The half as a rank-2 tensor, or an Error.
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildExchangeOnly(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density) const;

    /// Builds the per-spin Fock matrices of an unrestricted run,
    /// F_sigma = H + J(P_alpha + P_beta) - K(P_sigma) for sigma in
    /// {alpha, beta}: the Coulomb half runs ONCE on the half-summed density
    /// 0.5 (P_alpha + P_beta) - the QFMM far field sees the spin-summed
    /// charge (the J loop's baked 2.0 turns 2J(0.5 P_tot) into J(P_tot),
    /// the MakeDirectUhfFockBuilder convention of the driver) - and
    /// the exchange half runs once per spin on the RAW spin densities (the
    /// exchange-only builder's -K is linear in its input, so P_sigma needs
    /// no halving - the K(P) of the UHF channel, not the K(rho) of the
    /// closed shell). Each channel subtracts one core-Hamiltonian copy back
    /// (the double-H trap, exactly like BuildFock: every split result
    /// carries its own full H).
    /// \param alphaDensity The spin-up density matrix P_alpha in the scf
    /// UhfFockBuilderFn convention (the actual per-spin D_sigma, NOT the
    /// spatial half-density), host-canonical rank-2 tensor with shape
    /// {n, n}.
    /// \param betaDensity The spin-down density matrix P_beta, same shape.
    /// Both must be SYMMETRIC (the nested builders' documented
    /// precondition).
    /// \returns The alpha and beta Fock matrices as a pair of rank-2
    /// tensors, or an Error.
    qcx::Result<std::pair<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>,
                          qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>>>
    BuildUhfFock(const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& alphaDensity,
                 const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& betaDensity) const;

    /// The number of far-field pair-pair interactions in the built octree
    /// (the liveness flag, passed through from the nested QfmmJBuilder):
    /// nonzero means the multipole far field is live for this geometry and
    /// theta, zero means the QFMM half degenerates to the near-field direct
    /// build (the vacuous-ladder trap - the composed builder is then
    /// the split-pass direct build).
    /// \returns The far-field pair count from the last successful Create().
    std::size_t FarFieldPairCount() const noexcept;

    /// The Create-time mode record of the Coulomb (QFMM) half
    /// (fock_build.hpp FockModeInfo); nullopt when no workspace budget was
    /// given (the legacy path - no decision was made).
    /// \returns The mode record, or nullopt on the legacy path.
    const std::optional<FockModeInfo>& ModeInfo() const noexcept;

    /// The Create-time mode record of the exchange half (the exchange-only
    /// DirectJkFockBuilder - fock_build.hpp FockModeInfo); nullopt when no
    /// workspace budget was given (the legacy path - no decision was made).
    /// The composed builder's shared budget prices this half AFTER the
    /// Coulomb half's Create (nesting order = reservation order).
    /// \returns The mode record, or nullopt on the legacy path.
    const std::optional<FockModeInfo>& ExchangeModeInfo() const noexcept;

    /// What the Coulomb (QFMM) half actually ran (fock_build.hpp
    /// QfmmModelRecord): the geometry model, the separation test and its
    /// parameter, the resolved theta. Forwarded from the nested QfmmJBuilder
    /// like ModeInfo() above, so a consumer reads the engine's own resolution
    /// instead of recomputing it - the theta resolution (ThetaForPreset, and
    /// the kTight-preset gate) lives in the engine, and a second copy of it in
    /// a caller could disagree with the engine about what ran.
    /// \returns The record from the last successful Create().
    QfmmModelRecord ModelRecord() const noexcept;

private:
    // The implementation state lives in the .cpp (Eigen stays
    // implementation-only).
    struct State;
    explicit QfmmHfFockBuilder(std::shared_ptr<const State> state);
    std::shared_ptr<const State> _state;
};

} // namespace qcx::integrals
