#pragma once

/// \file
/// The deterministic system fingerprint of the HDF5 integral store:
/// FNV-1a-64 over the canonical byte layout of {schema version, engine
/// version, charge, multiplicity, canonical atom Z list, raw coordinate
/// bits, basis names}, hex-encoded. Identical input bits -> identical
/// fingerprint; any bit change -> a different one. The hash gates file
/// selection; the readable fields under /molecule are re-verified on open
/// (a collision must match them too).

#include "qcx/integrals/engine_version.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/storage/schema.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace qcx::storage {

/// The deterministic system fingerprint. Coordinates enter as raw
/// uint64 bits (never through arithmetic), so -0.0 vs 0.0 and any
/// last-ulp difference produce different fingerprints - bit-identity is
/// the contract.
/// \param molecule The system (canonical atom order, Bohr coordinates).
/// \param orbitalBasisName The basis-set name (the bundled corpus names,
/// e.g. "cc-pvdz" - see the basis version table).
/// \param auxBasisName Auxiliary-basis name, or empty when none.
/// \returns The 16-character lowercase hex fingerprint.
/// \ingroup qcx-storage
std::string ComputeFingerprint(const qcx::molecule::Molecule& molecule,
                               std::string_view orbitalBasisName,
                               std::string_view auxBasisName);

namespace internal {

/// The FNV-1a-64 offset basis constant.
inline constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
/// The FNV-1a-64 prime.
inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

/// FNV-1a-64 over the raw bytes (deterministic, dependency-free).
/// \param data The byte span to hash.
/// \param size The span length in bytes.
/// \returns The FNV-1a-64 hash of the span.
inline std::uint64_t Fnv1a64(const std::byte* data, std::size_t size) noexcept {
    std::uint64_t hash = kFnvOffsetBasis;

    for (std::size_t i = 0; i < size; ++i)
    {
        hash ^= static_cast<std::uint8_t>(data[i]);
        hash *= kFnvPrime;
    }

    return hash;
}

} // namespace internal
} // namespace qcx::storage
