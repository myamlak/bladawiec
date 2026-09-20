// The SCF checkpoint writer/reader (scf_checkpoint.hpp):
// /scf/state carries the restart state as fixed-shape float64 datasets
// (density {n,n}, per-spin densities, DIIS histories {k,n,n} with k the
// stored history size - 0-length is legal HDF5 - and the gate's previous
// energies), plus a zero-filled Fock for spec compliance and the
// state_checksum attribute (FNV-1a-64 over the group's fixed-order byte
// layout, verified on load). The file may be a fresh file or an
// existing ERI store file: /molecule is written when missing and verified
// element-wise when present. The checkpoint is append-only - a second
// Save refuses.

#include "qcx/storage/scf_checkpoint.hpp"

#include "hdf5_util.hpp"
#include "qcx/storage/fingerprint.hpp"
#include "store_molecule.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::storage {

namespace {

// One {n,n} matrix as a row-major flat vector (the HDF5 layout; the Eigen
// column-major data is copied element-wise - never a raw data() blit).
std::vector<double> FlattenMatrix(const Eigen::MatrixXd& matrix) {
    std::vector<double> flat(static_cast<std::size_t>(matrix.rows() * matrix.cols()));

    for (Eigen::Index row = 0; row < matrix.rows(); ++row)
    {
        for (Eigen::Index col = 0; col < matrix.cols(); ++col)
        {
            flat[static_cast<std::size_t>(row * matrix.cols() + col)] = matrix(row, col);
        }
    }

    return flat;
}

// The reversed element-wise copy.
Eigen::MatrixXd UnflattenMatrix(const std::vector<double>& flat, std::size_t n) {
    Eigen::MatrixXd matrix =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t col = 0; col < n; ++col)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                flat[row * n + col];
        }
    }

    return matrix;
}

// One {k,n,n} history as a flat vector (k matrices, each row-major).
std::vector<double> FlattenHistory(const std::vector<Eigen::MatrixXd>& history) {
    std::vector<double> flat;

    if (history.empty())
    {
        return flat;
    }

    const std::size_t n = static_cast<std::size_t>(history.front().rows());
    flat.reserve(history.size() * n * n);

    for (const Eigen::MatrixXd& matrix : history)
    {
        const std::vector<double> flattened = FlattenMatrix(matrix);
        flat.insert(flat.end(), flattened.begin(), flattened.end());
    }

    return flat;
}

// The reversed element-wise copy.
std::vector<Eigen::MatrixXd> UnflattenHistory(const std::vector<double>& flat, std::size_t n) {
    std::vector<Eigen::MatrixXd> history;

    if (flat.empty())
    {
        return history;
    }

    history.reserve(flat.size() / (n * n));

    for (std::size_t i = 0; i < flat.size(); i += n * n)
    {
        const std::vector<double> block(flat.begin() + static_cast<long long>(i),
                                        flat.begin() + static_cast<long long>(i + n * n));
        history.push_back(UnflattenMatrix(block, n));
    }

    return history;
}

// Writes the {k,n,n} history dataset (0-length k is legal HDF5).
qcx::Result<void> WriteHistory(HighFive::Group& group,
                               const std::string& name,
                               const std::vector<Eigen::MatrixXd>& history,
                               std::size_t n) {
    const std::vector<double> flat = FlattenHistory(history);
    const std::size_t k = history.size();
    const std::vector<std::size_t> shape = {k, n, n};

    return WrapH5([&]() {
        HighFive::DataSet dataSet =
            group.createDataSet(name, HighFive::DataSpace(shape), HighFive::AtomicType<double>());

        if (k > 0)
        {
            dataSet.write_raw(flat.data(), dataSet.getDataType());
        }
    });
}

// Reads the {k,n,n} history dataset back.
qcx::Result<std::vector<Eigen::MatrixXd>> ReadHistory(HighFive::Group& group,
                                                      const std::string& name,
                                                      std::size_t n) {
    std::vector<double> flat;

    auto read = WrapH5([&]() {
        HighFive::DataSet dataSet = group.getDataSet(name);
        const std::vector<std::size_t> dims = dataSet.getSpace().getDimensions();
        flat.resize(std::accumulate(
            dims.begin(), dims.end(), std::size_t{1}, std::multiplies<std::size_t>()));

        if (!flat.empty())
        {
            dataSet.read_raw(flat.data(), dataSet.getDataType());
        }
    });

    if (!read.has_value())
    {
        return std::unexpected(read.error());
    }

    return UnflattenHistory(flat, n);
}

// The /scf/state checksum: FNV-1a-64 over the fixed-order byte
// layout of everything the group carries - the method bytes, the channel
// densities (RHF: density; UHF: density_alpha then density_beta), the
// fock, the two previous energies as IEEE-754 bit patterns, then the six
// DIIS history flats in the fixed order fock/error/alpha-fock/alpha-error/
// beta-fock/beta-error. Computed identically at save (from the in-memory
// state) and at load (from the read-back data), so any on-disk change
// flips the hash and a mismatch means a corrupted checkpoint.
// (previousTotalEnergy, previousElectronicEnergy) are the two previous
// energies hashed in fixed order.
std::uint64_t ComputeStateChecksum(std::string_view method,
                                   const std::vector<double>& density,
                                   const std::vector<double>& densityAlpha,
                                   const std::vector<double>& densityBeta,
                                   const std::vector<double>& fock,
                                   // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                   double previousTotalEnergy,
                                   double previousElectronicEnergy,
                                   const std::array<std::vector<double>, 6>& historyFlats) {
    std::uint64_t hash = internal::kFnvOffsetBasis;

    const auto addBytes = [&hash](const std::byte* data, std::size_t size) {
        for (std::size_t i = 0; i < size; ++i)
        {
            hash ^= static_cast<std::uint8_t>(data[i]);
            hash *= internal::kFnvPrime;
        }
    };

    const auto addDoubles = [&addBytes](const std::vector<double>& values) {
        addBytes(reinterpret_cast<const std::byte*>(values.data()), values.size() * sizeof(double));
    };

    addBytes(reinterpret_cast<const std::byte*>(method.data()), method.size());
    addDoubles(density);
    addDoubles(densityAlpha);
    addDoubles(densityBeta);
    addDoubles(fock);
    addBytes(reinterpret_cast<const std::byte*>(&previousTotalEnergy), sizeof(double));
    addBytes(reinterpret_cast<const std::byte*>(&previousElectronicEnergy), sizeof(double));

    for (const std::vector<double>& flat : historyFlats)
    {
        addDoubles(flat);
    }

    return hash;
}

} // namespace

qcx::Result<void> SaveScfCheckpoint(const std::filesystem::path& path,
                                    const qcx::molecule::Molecule& molecule,
                                    std::string_view orbitalBasisName,
                                    std::string_view auxBasisName,
                                    const qcx::scf::ScfRestartState& state) {
    // The function-count consistency check: every non-empty
    // matrix must be square on the same n (n from the
    // spin-summed density, or the per-spin densities when the
    // RHF field is empty).
    const std::size_t n =
        state.density.size() != 0
            ? static_cast<std::size_t>(state.density.rows())
            : (state.densityAlpha.size() != 0 ? static_cast<std::size_t>(state.densityAlpha.rows())
                                              : 0);

    const auto squareOf = [](const Eigen::MatrixXd& matrix, std::size_t expected) {
        return matrix.size() == 0 || (static_cast<std::size_t>(matrix.rows()) == expected &&
                                      static_cast<std::size_t>(matrix.cols()) == expected);
    };

    if (n == 0 || !squareOf(state.density, n) || !squareOf(state.densityAlpha, n) ||
        !squareOf(state.densityBeta, n))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the checkpoint densities must agree on the "
                                          "function count"});
    }

    // The DIIS histories must agree with the density n as well
    // WriteHistory declares the {k,n,n} dataspace from n
    // while FlattenHistory flattens each matrix's own size - a
    // mismatched matrix would write a mis-sized buffer against
    // the declared shape.
    const auto validHistory = [n](const std::vector<Eigen::MatrixXd>& history) {
        return std::all_of(history.begin(), history.end(), [n](const Eigen::MatrixXd& matrix) {
            return static_cast<std::size_t>(matrix.rows()) == n &&
                   static_cast<std::size_t>(matrix.cols()) == n;
        });
    };

    if (!validHistory(state.diis.fockHistory) || !validHistory(state.diis.errorHistory) ||
        !validHistory(state.diisAlpha.fockHistory) || !validHistory(state.diisAlpha.errorHistory) ||
        !validHistory(state.diisBeta.fockHistory) || !validHistory(state.diisBeta.errorHistory))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the checkpoint DIIS histories must be square on "
                                          "the function count"});
    }

    std::shared_ptr<HighFive::File> file;

    auto open = WrapH5([&]() {
        file = std::make_shared<HighFive::File>(path.string(),
                                                HighFive::File::ReadWrite | HighFive::File::Create);
    });

    if (!open.has_value())
    {
        return std::unexpected(open.error());
    }

    // Append-only: refuse an existing checkpoint.
    bool exists = false;

    if (auto existsCheck = WrapH5([&]() { exists = file->exist("scf/state"); });
        !existsCheck.has_value())
    {
        return std::unexpected(existsCheck.error());
    }

    if (exists)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the checkpoint already exists"});
    }

    // The /molecule group: written for a fresh file, verified
    // element-wise when the file already carries one (an ERI
    // store file).
    bool hasMolecule = false;

    if (auto hasMoleculeCheck = WrapH5([&]() { hasMolecule = file->exist("molecule"); });
        !hasMoleculeCheck.has_value())
    {
        return std::unexpected(hasMoleculeCheck.error());
    }

    if (hasMolecule)
    {
        if (auto verified =
                internal::VerifyMoleculeGroup(*file, molecule, orbitalBasisName, auxBasisName);
            !verified.has_value())
        {
            return std::unexpected(verified.error());
        }
    } else if (auto written =
                   internal::WriteMoleculeGroup(*file, molecule, orbitalBasisName, auxBasisName);
               !written.has_value())
    { return std::unexpected(written.error()); }

    // /scf/state: all datasets are always written - empty
    // {0,n,n} histories for the unused spin channel (the load
    // side reads unconditionally; the "method" scalar is
    // informational).
    const bool isUhf = state.densityAlpha.size() != 0;

    // The state checksum: FNV-1a-64 over the fixed-order
    // layout of the group's content (ComputeStateChecksum
    // above), stored as the state_checksum attribute and
    // verified on load.
    const std::array<std::vector<double>, 6> historyFlats = {
        FlattenHistory(state.diis.fockHistory),
        FlattenHistory(state.diis.errorHistory),
        FlattenHistory(state.diisAlpha.fockHistory),
        FlattenHistory(state.diisAlpha.errorHistory),
        FlattenHistory(state.diisBeta.fockHistory),
        FlattenHistory(state.diisBeta.errorHistory),
    };
    const std::uint64_t stateChecksum = ComputeStateChecksum(
        isUhf ? std::string_view("uhf") : std::string_view("rhf"),
        state.density.size() != 0 ? FlattenMatrix(state.density) : std::vector<double>{},
        state.densityAlpha.size() != 0 ? FlattenMatrix(state.densityAlpha) : std::vector<double>{},
        state.densityBeta.size() != 0 ? FlattenMatrix(state.densityBeta) : std::vector<double>{},
        std::vector<double>(n * n, 0.0),
        state.previousTotalEnergy,
        state.previousElectronicEnergy,
        historyFlats);

    auto group = WrapH5Value([&]() { return file->createGroup("scf/state"); });

    if (!group.has_value())
    {
        return std::unexpected(group.error());
    }

    // The method marker is a fixed-width char dataset
    // (hdf5_util.hpp).
    if (auto written = WriteScalarString(*group, "method", isUhf ? "uhf" : "rhf");
        !written.has_value())
    {
        return std::unexpected(written.error());
    }

    return WrapH5([&]() {
               // The {n,n} schema shapes (write() would reject a
               // flat buffer against the rank-2 dataspace - its
               // dimension check only squeezes singleton dims -
               // so the raw overload writes the flat layout).
               if (state.density.size() != 0)
               {
                   const std::vector<double> flat = FlattenMatrix(state.density);
                   HighFive::DataSet dataSet = group->createDataSet(
                       "density", HighFive::DataSpace({n, n}), HighFive::AtomicType<double>());
                   dataSet.write_raw(flat.data(), dataSet.getDataType());
               }

               if (state.densityAlpha.size() != 0)
               {
                   const std::vector<double> flat = FlattenMatrix(state.densityAlpha);
                   HighFive::DataSet dataSet = group->createDataSet("density_alpha",
                                                                    HighFive::DataSpace({n, n}),
                                                                    HighFive::AtomicType<double>());
                   dataSet.write_raw(flat.data(), dataSet.getDataType());
               }

               if (state.densityBeta.size() != 0)
               {
                   const std::vector<double> flat = FlattenMatrix(state.densityBeta);
                   HighFive::DataSet dataSet = group->createDataSet(
                       "density_beta", HighFive::DataSpace({n, n}), HighFive::AtomicType<double>());
                   dataSet.write_raw(flat.data(), dataSet.getDataType());
               }

               // The zero-filled Fock: spec compliance and
               // diagnostics only - the restart never consumes
               // it.
               group->createDataSet("fock", std::vector<double>(n * n, 0.0));

               group->createDataSet("previous_total_energy", state.previousTotalEnergy);
               group->createDataSet("previous_electronic_energy", state.previousElectronicEnergy);
               group->createAttribute("state_checksum", stateChecksum);
           })
        .and_then([&]() -> qcx::Result<void> {
            HighFive::Group group = file->getGroup("scf/state");

            if (auto written = WriteHistory(group, "diis_fock_history", state.diis.fockHistory, n);
                !written.has_value())
            {
                return std::unexpected(written.error());
            }

            if (auto written =
                    WriteHistory(group, "diis_error_history", state.diis.errorHistory, n);
                !written.has_value())
            {
                return std::unexpected(written.error());
            }

            if (auto written =
                    WriteHistory(group, "diis_fock_history_alpha", state.diisAlpha.fockHistory, n);
                !written.has_value())
            {
                return std::unexpected(written.error());
            }

            if (auto written = WriteHistory(
                    group, "diis_error_history_alpha", state.diisAlpha.errorHistory, n);
                !written.has_value())
            {
                return std::unexpected(written.error());
            }

            if (auto written =
                    WriteHistory(group, "diis_fock_history_beta", state.diisBeta.fockHistory, n);
                !written.has_value())
            {
                return std::unexpected(written.error());
            }

            if (auto written =
                    WriteHistory(group, "diis_error_history_beta", state.diisBeta.errorHistory, n);
                !written.has_value())
            {
                return std::unexpected(written.error());
            }

            return WrapH5([&]() { file->flush(); });
        });
}

qcx::Result<qcx::scf::ScfRestartState> LoadScfCheckpoint(const std::filesystem::path& path,
                                                         const qcx::molecule::Molecule& molecule,
                                                         std::string_view orbitalBasisName,
                                                         std::string_view auxBasisName) {
    std::shared_ptr<HighFive::File> file;

    auto open = WrapH5([&]() {
        file = std::make_shared<HighFive::File>(path.string(), HighFive::File::ReadOnly);
    });

    if (!open.has_value())
    {
        return std::unexpected(open.error());
    }

    if (auto verified =
            internal::VerifyMoleculeGroup(*file, molecule, orbitalBasisName, auxBasisName);
        !verified.has_value())
    {
        return std::unexpected(verified.error());
    }

    qcx::scf::ScfRestartState state;
    auto group = WrapH5Value([&]() { return file->getGroup("scf/state"); });

    if (!group.has_value())
    {
        return std::unexpected(group.error());
    }

    std::string method;

    if (auto value = ReadScalarString(*group, "method"); value.has_value())
    {
        method = *value;
    } else
    {
        return std::unexpected(value.error());
    }

    const bool isUhf = method == "uhf";

    // All raw reads happen in one wrapped block; shape
    // validation and the checksum comparison follow outside it. A
    // malformed checkpoint must surface as a clean
    // kIOError - never as an out-of-bounds access
    // (UnflattenMatrix/UnflattenHistory index by the declared n,
    // so a non-square or wrong-rank dataset would read past its
    // buffer).
    std::vector<double> densityFlat;
    std::vector<double> densityBetaFlat;
    std::vector<double> fockFlat;
    std::vector<std::size_t> densityDims;
    std::vector<std::size_t> densityBetaDims;
    std::vector<std::size_t> fockDims;
    std::array<std::vector<double>, 6> historyFlats;
    std::array<std::vector<std::size_t>, 6> historyDims;
    double previousTotalEnergy = 0.0;
    double previousElectronicEnergy = 0.0;
    std::uint64_t storedChecksum = 0;

    const auto resizeToDims = [](std::vector<double>& flat, const std::vector<std::size_t>& dims) {
        flat.resize(std::accumulate(
            dims.begin(), dims.end(), std::size_t{1}, std::multiplies<std::size_t>()));
    };

    auto read = WrapH5([&]() {
        const std::string densityName = isUhf ? "density_alpha" : "density";
        HighFive::DataSet densityDataSet = group->getDataSet(densityName);
        densityDims = densityDataSet.getSpace().getDimensions();
        resizeToDims(densityFlat, densityDims);

        if (!densityFlat.empty())
        {
            densityDataSet.read_raw(densityFlat.data(), densityDataSet.getDataType());
        }

        if (isUhf)
        {
            HighFive::DataSet betaDataSet = group->getDataSet("density_beta");
            densityBetaDims = betaDataSet.getSpace().getDimensions();
            resizeToDims(densityBetaFlat, densityBetaDims);

            if (!densityBetaFlat.empty())
            {
                betaDataSet.read_raw(densityBetaFlat.data(), betaDataSet.getDataType());
            }
        }

        HighFive::DataSet fockDataSet = group->getDataSet("fock");
        fockDims = fockDataSet.getSpace().getDimensions();
        resizeToDims(fockFlat, fockDims);

        if (!fockFlat.empty())
        {
            fockDataSet.read_raw(fockFlat.data(), fockDataSet.getDataType());
        }

        group->getDataSet("previous_total_energy").read(previousTotalEnergy);
        group->getDataSet("previous_electronic_energy").read(previousElectronicEnergy);

        // The six histories in the checksum's fixed order.
        const std::array<std::string_view, 6> historyNames = {
            "diis_fock_history",
            "diis_error_history",
            "diis_fock_history_alpha",
            "diis_error_history_alpha",
            "diis_fock_history_beta",
            "diis_error_history_beta",
        };

        for (std::size_t i = 0; i < historyNames.size(); ++i)
        {
            HighFive::DataSet historyDataSet = group->getDataSet(std::string(historyNames[i]));
            historyDims[i] = historyDataSet.getSpace().getDimensions();
            resizeToDims(historyFlats[i], historyDims[i]);

            if (!historyFlats[i].empty())
            {
                historyDataSet.read_raw(historyFlats[i].data(), historyDataSet.getDataType());
            }
        }

        HighFive::Attribute checksumAttribute = group->getAttribute("state_checksum");
        checksumAttribute.read(storedChecksum);
    });

    if (!read.has_value())
    {
        return std::unexpected(read.error());
    }

    // Shape validation: the density must be square {n,n}
    // with n its first dimension, the fock flat {n*n}, the
    // histories {k,n,n}.
    if (densityDims.size() != 2 || densityDims[0] == 0 || densityDims[0] != densityDims[1])
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                          "the checkpoint density dataset must be "
                                          "square {n,n}"});
    }

    const std::size_t n = densityDims[0];

    if (isUhf &&
        (densityBetaDims.size() != 2 || densityBetaDims[0] != n || densityBetaDims[1] != n))
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                          "the checkpoint density_beta dataset "
                                          "must be square {n,n}"});
    }

    if (fockDims.size() != 1 || fockDims[0] != n * n)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, "the checkpoint fock dataset must be flat {n*n}"});
    }

    for (std::size_t i = 0; i < historyDims.size(); ++i)
    {
        if (historyDims[i].size() != 3 || historyDims[i][1] != n || historyDims[i][2] != n)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                              "a checkpoint DIIS history dataset "
                                              "must be {k,n,n}"});
        }
    }

    // The checksum: recompute over the read-back data in
    // the fixed layout and compare against the stored attribute.
    const std::uint64_t expectedChecksum =
        ComputeStateChecksum(method,
                             isUhf ? std::vector<double>{} : densityFlat,
                             isUhf ? densityFlat : std::vector<double>{},
                             densityBetaFlat,
                             fockFlat,
                             previousTotalEnergy,
                             previousElectronicEnergy,
                             historyFlats);

    if (expectedChecksum != storedChecksum)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                          "the checkpoint state checksum mismatch "
                                          "(corrupt store)"});
    }

    // Assemble the restart state from the verified data. Only
    // the stored method's channels are loaded - the RHF channel
    // for an RHF checkpoint, the alpha/beta channels for a UHF
    // one - so a restart never inherits the other method's DIIS
    // history.
    state.previousTotalEnergy = previousTotalEnergy;
    state.previousElectronicEnergy = previousElectronicEnergy;

    if (isUhf)
    {
        state.densityAlpha = UnflattenMatrix(densityFlat, n);
        state.densityBeta = UnflattenMatrix(densityBetaFlat, n);
        state.diisAlpha.fockHistory = UnflattenHistory(historyFlats[2], n);
        state.diisAlpha.errorHistory = UnflattenHistory(historyFlats[3], n);
        state.diisBeta.fockHistory = UnflattenHistory(historyFlats[4], n);
        state.diisBeta.errorHistory = UnflattenHistory(historyFlats[5], n);
    } else
    {
        state.density = UnflattenMatrix(densityFlat, n);
        state.diis.fockHistory = UnflattenHistory(historyFlats[0], n);
        state.diis.errorHistory = UnflattenHistory(historyFlats[1], n);
    }

    return state;
}

} // namespace qcx::storage
