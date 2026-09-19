#pragma once

/// \file
/// The general-l dense ERI tensor: the replacement
/// for the s-only BuildEriTensor, internally batched with Schwarz screening.
/// The dense driver is the screening caller; the batch API (eri_batch.hpp)
/// takes caller-supplied lists.

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

namespace qcx::integrals {

/// The default dense-tensor admission cap (2 GiB). The build materializes
/// three Theta(n^4)-class structures - the canonical quartet list
/// (nPairs^2/2 x sizeof(ShellQuartet) entries, the list itself the death
/// allocation at scale), the batch payload (8 B per surviving function
/// quartet, 8 n^4 worst case) and the rank-4 tensor (8 n^4 B) - so the
/// Create-time estimate is the sum of the three, checked against this cap
/// before anything is enumerated. The dense tensor is a reference path (the
/// direct screened builder is the scale path); the cap admits n up to ~120
/// functions with the all-survive bound, and every current dense-tensor
/// fixture (H2O/def2-SVP benchmarks, the driver's atomic-guess fragments)
/// sits orders of magnitude below it. Raise the option to admit a larger
/// reference build deliberately.
inline constexpr std::size_t kDenseEriTensorByteCap = 2ull * 1024 * 1024 * 1024;

/// Dense-tensor build settings.
/// \ingroup qcx-integrals
struct EriDenseOptions {
    AccuracyPreset accuracy = AccuracyPreset::kNormal; ///< Schwarz screening preset.
    std::size_t maxBatchBytes = 512 * 1024 * 1024; ///< Per-class batch cap (512 MB).
    bool screen = true; ///< Schwarz-screen canonical quartets (false = all of them).
    std::size_t maxTensorBytes = kDenseEriTensorByteCap; ///< The admission cap for the
                                                         ///< tensor plus its batch payload
                                                         ///< plus the canonical quartet list.
};

/// The a-priori worst-case byte estimate of the dense ERI tensor build:
/// the canonical quartet list (nPairs(nPairs + 1) / 2 ShellQuartet
/// entries), the all-survive batch payload (8 n^4 B) and the rank-4
/// tensor (8 n^4 B), each product saturating at size_t max (an exact
/// refusal bound - anything that saturates cannot fit any cap), so the
/// sum can never under-estimate through wraparound. The engine's
/// admission gate (BuildEriTensorGeneral) and the driver's memory-model
/// charge (the NOCV dense tensor, driver memory_model.hpp) share this
/// formula - one estimate everywhere.
/// \param nBasis The basis-function count (the tensor's n).
/// \param nPairs The shell-pair count (the canonical quartet list's
/// cardinality driver).
/// \returns The worst-case estimate in bytes.
/// \ingroup qcx-integrals
std::size_t DenseEriTensorEstimateBytes(std::size_t nBasis, std::size_t nPairs) noexcept;

/// Builds the two-electron repulsion tensor (uv|ws) in chemist's notation
/// over all functions, dense rank-4 with shape {n, n, n, n} (function
/// ordering as in shell_pairs.hpp).
///
/// Canonical quartets are computed once and mirrored into their 8-fold
/// partners bit-exactly (pure index transpositions); Schwarz screening drops
/// quartets with Q_ij * Q_kl below SchwarzThreshold(options.accuracy).
///
/// Admission gate: before the quartet enumeration the build estimates the
/// canonical-quartet list bytes, the all-survive batch payload (8 n^4 B)
/// and the tensor (8 n^4 B) - each saturated at size_t max, never under -
/// and refuses (kInvalidArgument) when the sum exceeds
/// options.maxTensorBytes, so an unaffordable dense build fails gracefully
/// instead of dying mid-enumeration.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise) and every atom must have a basis entry
/// (kInvalidArgument otherwise).
/// \param options Build settings; maxBatchBytes must be positive.
/// \returns The repulsion tensor, host-canonical, or an Error (kOutOfMemory
/// propagates from Tensor::Create, kInvalidArgument from the admission
/// gate).
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>> BuildEriTensorGeneral(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const EriDenseOptions& options = {});

} // namespace qcx::integrals
