#pragma once

// The class dispatch table of the matrix-form MD engine
// one instantiated kernel per {L_bra, L_ket}
// class, reached through compile-time function pointers so scf/tests never
// re-instantiate the hot templates. The table itself is generated
// (md_dispatch_gen.hpp, tools/gen_md_tables.py) and gated on the
// QcxIntegralsLMax/QcxIntegralsF32 compile definitions - classes above the
// configured Lmax are nullptr (kUnimplemented), and no CI configuration
// ever links the full class matrix.

#include "md_batch.hpp"

#include <array>

namespace qcx::integrals::internal {

/// The fp64 class-kernel signature: writes the packed blocks through the
/// batch's outF64 pointer; the Error is the transform-GEMM seam's
/// (kInvalidArgument, excluded by construction in normal operation).
using MdClassFnF64 = qcx::Result<void> (*)(const MdClassBatch&);

/// The certified fp32 class-kernel signature: writes the packed blocks
/// through outF32 and the a-priori per-quartet bounds through errorBounds
/// (the certified bound); the Error is as for MdClassFnF64.
using MdClassFnF32 = qcx::Result<void> (*)(const MdClassBatch&);

/// The fp64 pipeline of one class (defined in md_vrr.hpp, instantiated in
/// the per-class object libraries).
/// \param batch The class batch to evaluate.
/// \returns An Error from the transform GEMM seam (excluded by
/// construction in normal operation).
template <int LBra, int LKet> qcx::Result<void> ComputeEriClass(const MdClassBatch& batch);

/// The certified fp32 pipeline of one class.
/// \param batch The class batch to evaluate.
/// \returns An Error from the transform GEMM seam (excluded by
/// construction in normal operation).
template <int LBra, int LKet> qcx::Result<void> ComputeEriClassF32(const MdClassBatch& batch);

/// The kernel of class (L_bra, L_ket), or nullptr above the configured Lmax.
/// \param lBra Total class of the bra pair.
/// \param lKet Total class of the ket pair.
/// \returns A pointer to the table slot holding the kernel (a
/// pointer-to-function-pointer), or nullptr when not instantiated.
const MdClassFnF64* ClassSpecF64(int lBra, int lKet) noexcept;

/// The certified fp32 kernel of class (L_bra, L_ket).
/// \param lBra Total class of the bra pair.
/// \param lKet Total class of the ket pair.
/// \returns A pointer to the table slot holding the kernel (a
/// pointer-to-function-pointer), or nullptr when not instantiated.
const MdClassFnF32* ClassSpecF32(int lBra, int lKet) noexcept;

} // namespace qcx::integrals::internal
