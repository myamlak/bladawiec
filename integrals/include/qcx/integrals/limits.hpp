#pragma once

/// \file
/// Engine angular-momentum limits.
///
/// The matrix-form MD 2e/3c engines are instantiated per total
/// {L_bra, L_ket} class; \c QcxIntegralsLMax (a PUBLIC compile definition on
/// qcx-integrals, set by the QCX_INTEGRALS_LMAX CMake knob) caps the shell
/// angular momentum those instantiations cover. CI configures Lmax = 2 so
/// routine jobs never compile the full l <= 6 class matrix; a local
/// build compiles all of it. The 1e engines are runtime-l and always support
/// the parser cap l <= 6.

namespace qcx::integrals {

#ifndef QcxIntegralsLMax
/// Header-only consumers (no CMake definition) get the full local default.
#define QcxIntegralsLMax 6
#endif

/// Highest shell angular momentum the 2e/3c engines support in this build.
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
