#pragma once

/// \file
/// The integral-generator version stamp.

#include <cstdint>

namespace qcx::integrals {

/// Integral-generator version stamp. Bump this constant
/// whenever ANY integral-generator output can change for identical input:
/// MD kernel changes, Boys-table regeneration, certified fp32 bound changes,
/// fp32-lane changes. The storage module refuses to serve chunks written
/// under a different stamp (the reload==recompute bit-identity contract).
/// \ingroup qcx-integrals
inline constexpr std::uint32_t kIntegralEngineVersion = 1;

} // namespace qcx::integrals
