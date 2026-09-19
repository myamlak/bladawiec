#pragma once

/// \file
/// The storage schema version stamp: the on-disk layout of
/// every HDF5 store this module writes. Bump this constant whenever the
/// schema changes in a way that breaks older readers (new required groups,
/// renamed datasets, changed dtypes); a store written under a different
/// schema version has a different fingerprint and is refused.

#include <cstdint>

namespace qcx::storage {

/// The HDF5 store schema version. Part of the system fingerprint
/// (offset 8), so a version bump invalidates every existing store for the
/// same system - the documented price of a schema change.
/// \ingroup qcx-storage
///
/// Version 2: the fp32 manifest gained a 13th column (bounds_checksum,
/// the per-chunk FNV-1a-64 over the certified bounds dataset, verified on
/// serve - st-1). Version-1 stores are refused by the schema-version stamp
/// gate; the store checksum covers the manifest bytes, so a version-1 store
/// also fails the fingerprint comparison at open.
inline constexpr std::uint32_t kStoreSchemaVersion = 2;

} // namespace qcx::storage
