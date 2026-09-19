// Internal: the /molecule group writer and verifier shared by the store,
// the SCF-checkpoint writer, and the RI-tensor writer (the readable
// double-check of the fingerprint design).

#pragma once

#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <highfive/H5File.hpp>
#include <string>
#include <string_view>

namespace qcx::storage::internal {

/// Writes /molecule (the fingerprint plus every readable field) into a
/// freshly created store file. Returns the fingerprint.
/// \returns The fingerprint, or an Error.
qcx::Result<std::string> WriteMoleculeGroup(HighFive::File& file,
                                            const qcx::molecule::Molecule& molecule,
                                            std::string_view orbitalBasisName,
                                            std::string_view auxBasisName);

/// Recomputes the fingerprint for \p molecule and verifies it AND every
/// readable field of /molecule against the caller's system element-wise
/// (coordinate bits via memcmp; a collision must match them too).
/// \returns The stored fingerprint, or an Error (kInvalidArgument "store
/// belongs to a different system" on any mismatch; kIOError for HDF5
/// failures).
qcx::Result<std::string> VerifyMoleculeGroup(HighFive::File& file,
                                             const qcx::molecule::Molecule& molecule,
                                             std::string_view orbitalBasisName,
                                             std::string_view auxBasisName);

} // namespace qcx::storage::internal
