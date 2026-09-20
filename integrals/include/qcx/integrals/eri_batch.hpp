#pragma once

/// \file
/// The batched ERI entry points of the matrix-form MD engine:
/// caller-supplied shell-quartet lists evaluated in
/// per-{L_bra, L_ket}-class batches. The engine performs no screening
/// internally - the caller supplies screened, canonicalized lists.
///
/// Layout contract: per computed quartet (shells i, j, k, l with n_i..n_l
/// functions) a packed block of n_i*n_j*n_k*n_l values, row-major over the
/// two pair dimensions: bra rows (f_j*n_i + f_i), ket columns
/// (f_l*n_k + f_k) - EriBlockIndex below is the single implementation
/// every consumer must use (2026-08-18: an earlier column-major contract
/// here drifted from the engine and misled two consumers - see the
/// unfolded-path test that pins the layout element-wise).
/// \c computed holds the form actually evaluated - the batch machinery
/// canonicalizes each quartet by total class (L_bra <= L_ket), which may
/// swap bra/ket (one of the 8-fold partners, value-identical); consumers
/// must use \c computed, not the requested form. Entries of the same
/// (L_bra, L_ket) class and contraction degree are contiguous.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <vector>

namespace qcx::integrals {

/// The bra row index of the packed block.
/// \param fi The bra function index of shell i.
/// \param fj The bra function index of shell j.
/// \param nI The function count of shell i.
/// \returns f_j * nI + f_i.
/// \ingroup qcx-integrals
inline std::size_t EriBlockRow(std::size_t fi, std::size_t fj, std::size_t nI) noexcept {
    return fj * nI + fi;
}

/// The ket column index of the packed block.
/// \param fk The ket function index of shell k.
/// \param fl The ket function index of shell l.
/// \param nK The function count of shell k.
/// \returns f_l * nK + f_k.
/// \ingroup qcx-integrals
inline std::size_t EriBlockCol(std::size_t fk, std::size_t fl, std::size_t nK) noexcept {
    return fl * nK + fk;
}

/// The packed-block offset of the (fi, fj | fk, fl) element of one quartet
/// block: bra rows (f_j*n_i + f_i), ket columns (f_l*n_k + f_k), row-major.
/// The single layout implementation - consumers and tests must index the
/// batch through this (and EriBlockRow/EriBlockCol), never by hand.
/// \param fi The bra function index of shell i.
/// \param fj The bra function index of shell j.
/// \param fk The ket function index of shell k.
/// \param fl The ket function index of shell l.
/// \param nI The function count of shell i.
/// \param nJ The function count of shell j (bra block row stride).
/// \param nK The function count of shell k.
/// \param nL The function count of shell l (ket block column stride).
/// \returns The flat offset in the packed block.
/// \ingroup qcx-integrals
inline std::size_t EriBlockIndex(std::size_t fi,
                                 std::size_t fj,
                                 std::size_t fk,
                                 std::size_t fl,
                                 std::size_t nI,
                                 std::size_t nJ,
                                 std::size_t nK,
                                 std::size_t nL) noexcept {
    return (EriBlockRow(fi, fj, nI) * (nK * nL)) + EriBlockCol(fk, fl, nK);
}

/// Batch evaluation settings.
/// \ingroup qcx-integrals
struct EriBatchOptions {
    AccuracyPreset accuracy =
        AccuracyPreset::kNormal; ///< Screening preset (future Fock consumers).
    std::size_t maxBatchBytes = 512 * 1024 * 1024; ///< Per-class batch cap (512 MB).
};

/// One fp64 batch result: values packed per the layout contract, and the
/// canonicalized quartets in the same order.
/// \ingroup qcx-integrals
struct EriBatch {
    std::vector<double> values; ///< Packed blocks, one per computed quartet.
    std::vector<ShellQuartet> computed; ///< The canonicalized quartets actually evaluated.
};

/// One certified-mixed-precision batch result: the fp32 pipeline
/// values with the a-priori per-quartet error bounds.
/// \ingroup qcx-integrals
struct CertifiedBatch {
    std::vector<float> values; ///< Packed blocks (same layout contract).
    std::vector<double> errorBounds; ///< A-priori |error| bound per computed quartet.
    std::vector<ShellQuartet> computed; ///< The canonicalized quartets actually evaluated.
};

/// The canonical form and ordering keys of one quartet, exactly as the
/// batch engine computes them: the 8-fold form (i <= j, k <= l,
/// pair (i,j) >= (k,l)), then the class canonical form L_bra <= L_ket
/// (bra/ket swap, value-identical). The ordering keys reproduce the
/// engine's class-major, ket-grouped sort.
/// \ingroup qcx-integrals
struct CanonicalQuartetInfo {
    ShellQuartet quartet; ///< The canonical form the engine evaluates.
    int lBra; ///< Total bra class: la + lb.
    int lKet; ///< Total ket class: lc + ld.
    std::size_t braPair; ///< PairIndexOf(i, j) of the canonical pair.
    std::size_t ketPair; ///< PairIndexOf(k, l) of the canonical pair.
    std::size_t braRowPairs; ///< Bra-pair contraction row pairs: rows_i * rows_j.
};

/// Canonicalizes and orders a quartet request exactly as the batch engine
/// does (the single implementation shared by ComputeEriBatch and the
/// storage decorator): validates shell indices and the
/// l <= 2*kMaxEngineL cap, then sorts by (lKet, lBra, ketPair, braRowPairs,
/// braPair). The returned order matches EriBatch::computed for the same
/// request.
/// \param pairList The pair list of the system (BuildShellPairs).
/// \param quartets The request, in any form; must be non-empty.
/// \returns The canonical, sorted infos, or an Error (kInvalidArgument for
/// out-of-range shells or an empty list, kUnimplemented over the L cap).
/// \ingroup qcx-integrals
qcx::Result<std::vector<CanonicalQuartetInfo>> CanonicalizeQuartetOrder(
    const ShellPairList& pairList, const std::vector<ShellQuartet>& quartets);

/// Evaluates the two-electron repulsion integrals of the given shell
/// quartets through the fp64 MD pipeline.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise) and every atom must have a basis entry
/// (kInvalidArgument otherwise).
/// \param quartets The quartets to evaluate, in any form; each is
/// canonicalized and reported through EriBatch::computed.
/// \param options Batch settings; maxBatchBytes must be positive.
/// \returns The packed batch, or an Error.
/// \ingroup qcx-integrals
qcx::Result<EriBatch> ComputeEriBatch(const qcx::molecule::Molecule& molecule,
                                      const qcx::basisset::BasisSet& basisSet,
                                      const std::vector<ShellQuartet>& quartets,
                                      const EriBatchOptions& options = {});

/// Evaluates the given quartets through the certified fp32 pipeline:
/// every returned block carries its a-priori error bound - consumers accept
/// the values only where their own accuracy budget covers the bounds.
/// Requires the fp32 pipeline to be instantiated (the QCX_INTEGRALS_F32
/// CMake knob, on by default).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set (same contract as ComputeEriBatch).
/// \param quartets The quartets to evaluate, in any form.
/// \param options Batch settings; maxBatchBytes must be positive.
/// \returns The certified batch, or an Error (kUnimplemented when this build
/// has no fp32 pipeline).
/// \ingroup qcx-integrals
qcx::Result<CertifiedBatch> ComputeEriBatchCertified(const qcx::molecule::Molecule& molecule,
                                                     const qcx::basisset::BasisSet& basisSet,
                                                     const std::vector<ShellQuartet>& quartets,
                                                     const EriBatchOptions& options = {});

} // namespace qcx::integrals
