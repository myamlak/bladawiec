#pragma once

/// \file
/// The RI 3-center tensor persistence: the n^2 x nAux
/// contraction layout is the persistence contract - the {n,n,nAux}
/// tensor form is deliberately not stored (one layout).

#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>
#include <filesystem>
#include <string_view>

namespace qcx::storage {

/// Saves the RI 3-center tensor in the n^2 x nAux contraction layout
/// as /integrals/ao/eri/ri_tensor. Bit-identical round trip on
/// the same machine. Refuses when the dataset already exists.
/// \param path The store file (created when missing).
/// \param molecule The system (written to the store's /molecule).
/// \param orbitalBasisName The orbital basis-set name.
/// \param auxBasisName The auxiliary basis-set name.
/// \param riMatrix The n^2 x nAux contraction-layout matrix.
/// \returns An Error (kInvalidArgument when the dataset already exists,
/// kIOError for HDF5 failures).
/// \ingroup qcx-storage
qcx::Result<void> SaveRiTensor(const std::filesystem::path& path,
                               const qcx::molecule::Molecule& molecule,
                               std::string_view orbitalBasisName,
                               std::string_view auxBasisName,
                               const Eigen::MatrixXd& riMatrix);

/// Loads a stored RI tensor; the shape is validated against the file's
/// own shape (kInvalidArgument on mismatch). The system fingerprint must
/// match the caller's system (kInvalidArgument otherwise) - the caller
/// saves only when an aux basis exists, so the fingerprint carries it.
/// \param path The store file.
/// \param molecule The system (must match the stored fingerprint).
/// \param orbitalBasisName The orbital basis-set name.
/// \param auxBasisName The auxiliary basis-set name.
/// \returns The n^2 x nAux contraction matrix, or an Error (kInvalidArgument
/// for shape/fingerprint mismatches, kIOError for HDF5 failures).
/// \ingroup qcx-storage
qcx::Result<Eigen::MatrixXd> LoadRiTensor(const std::filesystem::path& path,
                                          const qcx::molecule::Molecule& molecule,
                                          std::string_view orbitalBasisName,
                                          std::string_view auxBasisName);

} // namespace qcx::storage
