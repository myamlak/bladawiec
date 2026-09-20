// The RI 3-center tensor persistence (ri_tensor_store.hpp):
// /integrals/ao/eri/ri_tensor as a fixed {rows, cols} float64 dataset in
// the contraction layout. The file may be a fresh file or an existing
// ERI store file: /molecule is written when missing and verified
// element-wise when present. Append-only - a second Save refuses.

#include "qcx/storage/ri_tensor_store.hpp"

#include "hdf5_util.hpp"
#include "store_molecule.hpp"

#include <cstddef>
#include <filesystem>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <memory>
#include <utility>
#include <vector>

namespace qcx::storage {

qcx::Result<void> SaveRiTensor(const std::filesystem::path& path,
                               const qcx::molecule::Molecule& molecule,
                               std::string_view orbitalBasisName,
                               std::string_view auxBasisName,
                               const Eigen::MatrixXd& riMatrix) {
    const std::size_t rows = static_cast<std::size_t>(riMatrix.rows());
    const std::size_t cols = static_cast<std::size_t>(riMatrix.cols());

    if (rows == 0 || cols == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the RI matrix must not be empty"});
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

    // Append-only: refuse an existing dataset.
    bool exists = false;

    if (auto existsCheck = WrapH5([&]() { exists = file->exist("integrals/ao/eri/ri_tensor"); });
        !existsCheck.has_value())
    {
        return std::unexpected(existsCheck.error());
    }

    if (exists)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the RI tensor is already stored"});
    }

    // The /molecule group: written for a fresh file, verified element-wise
    // when the file already carries one (an ERI store file).
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

    // Eigen is column-major, HDF5 is row-major: the element-wise copy
    // `data()[r + c*rows]` -> flat position `r*cols + c` - never a raw
    // data() blit across layouts.
    std::vector<double> flat(rows * cols);

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            flat[row * cols + col] =
                riMatrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
        }
    }

    return WrapH5([&]() {
        // The /integrals/ao/eri path exists in an ERI store file; a fresh
        // file must build it level by level.
        HighFive::Group integrals =
            file->exist("integrals") ? file->getGroup("integrals") : file->createGroup("integrals");
        HighFive::Group ao =
            integrals.exist("ao") ? integrals.getGroup("ao") : integrals.createGroup("ao");
        HighFive::Group eri = ao.exist("eri") ? ao.getGroup("eri") : ao.createGroup("eri");
        HighFive::DataSet dataSet = eri.createDataSet(
            "ri_tensor", HighFive::DataSpace({rows, cols}), HighFive::AtomicType<double>());
        // write() would reject the flat buffer against the rank-2 dataspace.
        dataSet.write_raw(flat.data(), dataSet.getDataType());
        file->flush();
    });
}

qcx::Result<Eigen::MatrixXd> LoadRiTensor(const std::filesystem::path& path,
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

    std::vector<std::size_t> dims;
    std::vector<double> flat;

    auto read = WrapH5([&]() {
        HighFive::DataSet dataSet = file->getDataSet("integrals/ao/eri/ri_tensor");
        dims = dataSet.getSpace().getDimensions();
        flat.resize(dims.at(0) * dims.at(1));

        if (!flat.empty())
        {
            dataSet.read_raw(flat.data(), dataSet.getDataType());
        }
    });

    if (!read.has_value())
    {
        return std::unexpected(read.error());
    }

    // The reversed element-wise copy (column-major Eigen from the row-major
    // on-disk layout).
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(dims.at(0)),
                                                   static_cast<Eigen::Index>(dims.at(1)));

    for (std::size_t row = 0; row < dims.at(0); ++row)
    {
        for (std::size_t col = 0; col < dims.at(1); ++col)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                flat[row * dims.at(1) + col];
        }
    }

    return matrix;
}

} // namespace qcx::storage
