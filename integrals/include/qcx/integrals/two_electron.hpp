#pragma once

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>

namespace qcx::integrals {

/// \file
/// Two-electron repulsion integrals: the dense s-only reference tensor, and
/// the nuclear-coordinate gradient of the density-contracted Coulomb energy.
///
/// The s-s-s-s primitive integral is the closed-form Gaussian-product
/// expression (Helgaker2000) evaluated through
/// the F0 evaluation of the Boys kernel. The gradient differentiates the
/// density contraction of the general-l packed blocks through the
/// two-electron derivative tier, one shell quartet at a time.

/// Builds the two-electron repulsion tensor (uv|ws) in chemist's notation
/// over all s functions, dense rank-4 with shape {n, n, n, n} (function
/// ordering as in one_electron.hpp).
///
/// Known limitation (documented, not a defect): the tensor is stored dense
/// with no permutational-symmetry compression yet; the 8-fold symmetry is
/// nevertheless exact to rounding, and compression lands with the
/// SymmetryPolicy storage tier.
///
/// Admission gate: the tensor is 8 n^4 B, so the build refuses
/// (kInvalidArgument) when that estimate exceeds \p maxTensorBytes instead
/// of dying on the allocation (the dense tensor is a reference path; the
/// direct screened builder is the scale path).
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must be an s shell.
/// \param maxTensorBytes The admission cap for the 8 n^4 tensor
/// (kDenseEriTensorByteCap by default, shared with the general-l dense
/// builder).
/// \returns The repulsion tensor, or an Error (kUnimplemented for a non-s
/// shell, kInvalidArgument when an atom has no basis entry or the admission
/// gate refuses).
/// \ingroup qcx-integrals
qcx::Result<qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>> BuildEriTensor(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    std::size_t maxTensorBytes = kDenseEriTensorByteCap);

/// What one two-electron ERI gradient assembly counted.
///
/// The columns the walk keeps so its cost stays readable: the canonical pair
/// pairs it offered, the quartets it differentiated, and the ones its
/// derivative-aware screen discarded. A gradient that walked a different task
/// set than the one the accuracy preset asks for differs here first.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct EriGradientCounts {
    std::size_t braPairs = 0; ///< Canonical bra pairs walked.
    std::size_t ketPairs = 0; ///< Canonical ket pairs offered, the pairs' upper triangle.
    std::size_t quartets = 0; ///< Shell quartets differentiated.
    std::size_t screenedQuartets = 0; ///< Quartets the derivative screen discarded.
};

/// One two-electron ERI gradient, at fixed density.
///
/// The density matrix is held fixed: `gradient` is the derivative of
///
///   E = 1/2 sum_uvls D_uv D_ls (uv|ls)
///
/// with respect to the nuclear positions, so the density matrix's own
/// response to the displacement - the SCF's business, not this walk's - is
/// not in it. It is the Coulomb energy's contribution to the total nuclear
/// gradient, the skeleton term [Helgaker2000].
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-integrals
struct EriGradient {
    /// dE/dR, 3N entries in Hartree/Bohr, atom-major: entry 3*i + d is
    /// direction d of atom i, the same ordering the nuclear coordinates use.
    Eigen::VectorXd gradient;
    /// The energy this gradient belongs to, Hartree: the density-contracted
    /// Coulomb energy, accumulated over the same retained quartets the
    /// gradient was assembled from.
    ///
    /// Reported on its own because it is the one number that separates the
    /// two ways a gradient can be wrong: a gradient that differentiates a
    /// different contraction differs from this energy's finite differences,
    /// whereas one that agrees with its own energy's finite differences
    /// while the energy is not the Coulomb contraction differs nowhere else.
    double energy = 0.0;
    /// The counted cost of this assembly.
    EriGradientCounts counts;
};

/// Settings of one two-electron ERI gradient assembly.
/// \ingroup qcx-integrals
struct EriGradientOptions {
    /// The screening preset. Its Schwarz threshold, weighted by the two
    /// canonical pairs' largest density elements, decides which pair pairs
    /// are differentiated; the default is the energy path's own.
    AccuracyPreset accuracy = AccuracyPreset::kNormal;
    /// Walk the whole canonical pair-pair space. Off, no quartet is
    /// discarded and every pair pair is differentiated - the exact
    /// contraction, whose cost the preset exists to cut. Both halves of the
    /// returned energy and gradient come from the same retained set either
    /// way, so the finite difference of the reported energy differentiates
    /// exactly what the gradient assembled.
    bool useDensityScreening = true;
};

/// Assembles the nuclear-coordinate gradient of the density-contracted
/// Coulomb energy, at fixed density.
///
/// The walk enumerates the canonical shell pairs of the basis set and, for
/// each unordered pair of them, one shell quartet (i, j | k, l). The
/// ordered shell sum the functional carries is recovered from the canonical
/// walk by the pair multiplicities: a pair (i, j) with i != j stands for two
/// ordered pairs, and a pair pair (P, Q) with P != Q for two ordered pairs of
/// pairs - the density matrix is symmetric, so both orientations carry the
/// same block.
///
/// Each quartet's packed block is differentiated through the two-electron
/// derivative tier, and the block is contracted with the density before any
/// accumulation: the walk holds one quartet's value block, its derivative
/// blocks and one weight block at a time, never the four-index tensor.
///
/// The energy is accumulated from the same blocks and the same weights, so
/// `energy` and `gradient` are one functional reached two ways - the
/// property the acceptance test reads when it finite-differences the
/// reported energy.
/// \param molecule Molecule providing the atom coordinates (Bohr).
/// \param basisSet Basis set; every shell must satisfy l <= kMaxEngineL
/// (kUnimplemented otherwise).
/// \param density The density matrix, nBasis x nBasis, symmetric, in the
/// basis set's function order.
/// \param options The assembly settings; the accuracy preset is read exactly
/// as the energy path reads it.
/// \returns The gradient, the energy it belongs to and the counted cost, or
/// an Error: kInvalidArgument for a density whose shape does not match the
/// basis set, kUnimplemented for a shell beyond kMaxEngineL or a quartet
/// whose derivative reaches past the derivative tier's Hermite table, and
/// the pair and quartet builders' own errors otherwise.
/// \ingroup qcx-integrals
qcx::Result<EriGradient> ComputeEriGradient(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basisSet,
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
    const EriGradientOptions& options = {});

} // namespace qcx::integrals
