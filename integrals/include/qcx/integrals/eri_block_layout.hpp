#pragma once

/// \file
/// The packed-ERI-block layout, in one place: per computed quartet (shells i,
/// j, k, l with n_i..n_l functions) a block of n_i*n_j*n_k*n_l values,
/// row-major over the two pair dimensions - bra rows (f_j*n_i + f_i), ket
/// columns (f_l*n_k + f_k). EriBlockIndex is the single implementation every
/// consumer must use (2026-08-18: an earlier column-major contract drifted
/// from the engine and misled two consumers - see the unfolded-path test that
/// pins the layout element-wise).
///
/// The helpers live in their own header so a consumer that only indexes a
/// block does not pull the batch entry points with it: the Fock-contraction
/// kernel needs this layout and nothing from the Boys-backed batch surface,
/// and it is compiled as the one first-party unit without the AVX2 tier,
/// where including a Boys header would be refused by the library's tier
/// registry at start-up.

#include <cstddef>

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
                                 [[maybe_unused]] std::size_t nJ,
                                 std::size_t nK,
                                 std::size_t nL) noexcept {
    return (EriBlockRow(fi, fj, nI) * (nK * nL)) + EriBlockCol(fk, fl, nK);
}

} // namespace qcx::integrals
