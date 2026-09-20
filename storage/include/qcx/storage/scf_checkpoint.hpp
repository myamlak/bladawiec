#pragma once

/// \file
/// The SCF checkpoint writer/reader: /scf/state carries the
/// serializable restart state (scf_state.hpp) plus a zero-filled Fock
/// dataset (spec compliance - the Fock is NOT consumed on restart; the
/// previous energies ARE, they seed the convergence gate) and a
/// state_checksum attribute (FNV-1a-64 over the group's fixed-order byte
/// layout, verified on load - a corrupted checkpoint is refused).

#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/scf/scf_state.hpp"

#include <filesystem>
#include <string_view>

namespace qcx::storage {

/// Saves one SCF run's restart state under /scf/state. The
/// fingerprint is computed from the molecule and basis names; the density
/// convention is the SCF loop's own (RHF: spin-summed D = 2 C_occ C_occ^T;
/// UHF: per-spin). Refuses when /scf/state already exists.
/// \param path The store file (created when missing).
/// \param molecule The system (written to the store's /molecule).
/// \param orbitalBasisName The orbital basis-set name.
/// \param auxBasisName The auxiliary basis-set name, or empty.
/// \param state The serializable restart state (scf_state.hpp).
/// \returns An Error (kInvalidArgument when the checkpoint already exists
/// or the state's matrices disagree in shape - densities or DIIS
/// histories - kIOError for HDF5 failures).
/// \ingroup qcx-storage
qcx::Result<void> SaveScfCheckpoint(const std::filesystem::path& path,
                                    const qcx::molecule::Molecule& molecule,
                                    std::string_view orbitalBasisName,
                                    std::string_view auxBasisName,
                                    const qcx::scf::ScfRestartState& state);

/// Loads a checkpoint; the caller hands the state back to a fresh run's
/// RhfOptions/UhfOptions restart seed (their `initialScfState` member).
/// The system fingerprint must match (kInvalidArgument
/// otherwise).
/// \param path The store file.
/// \param molecule The expected system.
/// \param orbitalBasisName The expected orbital basis-set name.
/// \param auxBasisName The expected auxiliary basis-set name, or empty.
/// \returns The restart state, or an Error (kInvalidArgument for a
/// fingerprint mismatch or when no checkpoint exists, kIOError for HDF5
/// failures or a corrupted checkpoint - wrong dataset shapes, a missing
/// state_checksum, or a checksum mismatch).
/// \ingroup qcx-storage
qcx::Result<qcx::scf::ScfRestartState> LoadScfCheckpoint(const std::filesystem::path& path,
                                                         const qcx::molecule::Molecule& molecule,
                                                         std::string_view orbitalBasisName,
                                                         std::string_view auxBasisName);

} // namespace qcx::storage
