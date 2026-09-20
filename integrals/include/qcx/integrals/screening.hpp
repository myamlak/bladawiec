#pragma once

/// \file
/// Schwarz screening: the
/// per-shell-pair bounds Q_ab = sqrt of the largest diagonal element of the
/// (ab|ab) block. |(ab|cd)| <= Q_ab * Q_cd for every function quadruple.
///
/// The engine performs no screening internally - the caller (the dense
/// driver today, the direct Fock builders) supplies screened
/// quartet lists built from these bounds.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"

#include <vector>

namespace qcx::integrals {

/// The Schwarz sweep's pair-data chunk cap: the byte mass of contracted pair
/// data (the E tables, the transforms and the weights BuildContractedPairTransform
/// commits) the sweep materializes per chunk, 24 B x the Hermite tables'
/// shape per primitive pair. The sweep needs the diagonal (ab|ab) block of
/// each canonical pair and nothing else, and a diagonal block's bra and ket
/// pair are the SAME pair, so the sweep is chunkable by pair at no
/// algorithmic cost: the pair store is built and dropped one chunk at a
/// time instead of whole (the 4,974-function C42H86/def2-QZVP point's whole
/// store is 9.95 GiB, which no path can afford as a screening pre-pass).
/// 512 MiB is the repo's standard batch cap
/// (EriBatchOptions::maxBatchBytes), so the sweep chunks at the same
/// granularity every other batched allocation uses.
/// \ingroup qcx-integrals
inline constexpr std::size_t kSchwarzChunkBytes = 512 * 1024 * 1024;

/// Computes the Schwarz bound of every canonical pair, in
/// ShellPairList::pairs order.
///
/// The diagonal (ab|ab) quartets are evaluated through the fp64 batch
/// pipeline, chunked over the pair list: the bra and ket pair of a diagonal
/// quartet are the same pair, so each chunk keeps only its own pairs'
/// contracted data resident (the geometry-only skeleton, sizeof(MdPairData)
/// per canonical pair, plus the chunk's pair data). A pair list whose whole
/// payload fits \p chunkBytes runs in ONE chunk and produces byte-for-byte
/// the values of the pre-chunking single-pass form.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \param chunkBytes The pair-data chunk cap (kSchwarzChunkBytes by
/// default); 0 disables chunking - one chunk covering the whole pair list,
/// the pre-chunking form (the bit-identity reference of the chunked path).
/// \returns Q_ab per canonical pair, or an Error (a chunk's diagonal
/// quartets are evaluated through the fp64 batch pipeline).
/// \ingroup qcx-integrals
qcx::Result<std::vector<double>> ComputeSchwarzBounds(const qcx::molecule::Molecule& molecule,
                                                      const qcx::basisset::BasisSet& basisSet,
                                                      std::size_t chunkBytes = kSchwarzChunkBytes);

} // namespace qcx::integrals
