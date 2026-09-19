#pragma once

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

namespace qcx::integrals {

/// \file
/// Two-electron repulsion integrals over s functions.
///
/// The s-s-s-s primitive integral is the closed-form Gaussian-product
/// expression (Helgaker2000) evaluated through
/// the F0 evaluation of the Boys kernel.

/// Builds the two-electron repulsion tensor (uv|ws) in chemist's notation
/// over all s functions, dense rank-4 with shape {n, n, n, n} (function
/// ordering as in one_electron.hpp).
///
/// Known limitation (documented, not a defect): the tensor is stored dense
/// with no permutational-symmetry compression yet; the 8-fold symmetry is
/// nevertheless exact to rounding.
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

} // namespace qcx::integrals
