#pragma once

/// \file
/// Engine angular-momentum limits.
///
/// The library supports shells up to l = 6; that is a capability boundary, and
/// a basis above it is refused by name. The matrix-form MD 2e/3c engines are
/// instantiated per total {L_bra, L_ket} class, and \c QcxIntegralsLMax (a
/// PUBLIC compile definition on qcx-integrals, set by the QCX_INTEGRALS_LMAX
/// CMake knob) is the instantiation ceiling of this build. The shipped default
/// is 6 - the full class matrix - and that is what CI builds; a reduced-size
/// build may lower it to trade instantiation size for compile time, and is not
/// a narrower product contract. The 1e engines are runtime-l and always support
/// the parser cap l <= 6.

namespace qcx::integrals {

#ifndef QcxIntegralsLMax
/// Header-only consumers (no CMake definition) get the full shipped default.
#define QcxIntegralsLMax 6
#endif

/// Highest shell angular momentum the 2e/3c engines of this build are
/// instantiated for. The shipped value is 6, the supported bound; a lower
/// value is a reduced-size build.
inline constexpr int kMaxEngineL = QcxIntegralsLMax;

/// Whether the 2e/3c engines of this build cover a shell of angular
/// momentum l.
/// \param l Angular momentum to check.
/// \returns True when 0 <= l <= kMaxEngineL.
/// \ingroup qcx-integrals
constexpr bool SupportsL(int l) noexcept {
    return l >= 0 && l <= kMaxEngineL;
}

} // namespace qcx::integrals
