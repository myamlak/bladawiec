// The chunked RI 3-center tensor store (ri_tensor_chunk_store.hpp):
// /integrals/ao/eri/ri_tensor_chunks/{chunk_index} as one
// storedRowCount x k_c float64 dataset per chunk in the contraction
// layout, with the per-chunk attributes (aux shell/function ranges, the
// stored row count, the FNV-1a-64 checksum over the chunk's byte stream,
// the kIntegralEngineVersion stamp). The file may be a fresh file or an
// existing ERI store file: /molecule is written when missing and verified
// element-wise when present (the shared store_molecule machinery of the
// SaveRiTensor path). Append-only - a second Save of the same chunk index
// refuses.

#include "qcx/storage/ri_tensor_chunk_store.hpp"

#include "hdf5_util.hpp"
#include "qcx/storage/fingerprint.hpp"
#include "store_molecule.hpp"

#include <cstdint>
#include <highfive/H5Attribute.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5Group.hpp>
#include <memory>
#include <string>
#include <vector>

namespace qcx::storage {

namespace {

// The dataset path of one chunk under the file's /integrals/ao/eri tree.
std::string ChunkDatasetPath(std::size_t chunkIndex) {
    return "integrals/ao/eri/ri_tensor_chunks/" + std::to_string(chunkIndex);
}

// The scalar range checks shared by Save and Load (the chunk's shell and
// function ranges are never derived from a matrix on the load side - the
// dataset's own dimensions are validated against them separately).
qcx::Result<void> ValidateMetaRanges(const RiChunkMeta& meta) {
    if (meta.auxShellStart >= meta.auxShellEnd)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the chunk aux shell range must be non-empty"});
    }

    if (meta.auxFunctionStart >= meta.auxFunctionEnd)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the chunk aux function range must be non-empty"});
    }

    if (meta.orbitalFunctionCount == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "the chunk's orbital function count must be positive"});
    }

    // Two row layouts exist and only two: the full n^2 (storedRowCount 0)
    // and the uv-reduced n(n+1)/2. The reduced layout's row ORDER is a
    // convention a shape cannot verify, so no third value is admitted.
    if (meta.storedRowCount != 0 &&
        meta.storedRowCount != RiChunkReducedRowCount(meta.orbitalFunctionCount))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the chunk's stored row count must be 0 (full n^2) or n*(n+1)/2 "
                       "(uv-reduced)"});
    }

    return {};
}

qcx::Result<void> ValidateMeta(const RiChunkMeta& meta, const Eigen::MatrixXd& chunkMatrix) {
    if (auto ranges = ValidateMetaRanges(meta); !ranges.has_value())
    {
        return std::unexpected(ranges.error());
    }

    const std::size_t rows = static_cast<std::size_t>(chunkMatrix.rows());
    const std::size_t cols = static_cast<std::size_t>(chunkMatrix.cols());
    const std::size_t expectedCols = meta.auxFunctionEnd - meta.auxFunctionStart;

    if (rows != RiChunkStoredRowCount(meta))
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the chunk matrix rows must equal the chunk's stored row count (n^2 for the "
                       "full layout, n*(n+1)/2 for the uv-reduced one)"});
    }

    if (cols != expectedCols)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the chunk matrix columns must equal the chunk's aux function count"});
    }

    return {};
}

// The stored per-chunk attributes of one dataset (all writes and reads are
// fixed-width/scalar HighFive calls inside WrapH5 - HDF5 failures surface
// as kIOError).
struct StoredChunkAttributes {
    std::uint32_t engineVersion = 0;
    std::uint64_t checksum = 0;
    std::uint64_t auxShellStart = 0;
    std::uint64_t auxShellEnd = 0;
    std::uint64_t auxFunctionStart = 0;
    std::uint64_t auxFunctionEnd = 0;
    /// Absent on a store written before the row layouts existed, where the
    /// only layout was the full n^2 the other attributes imply.
    std::uint64_t storedRowCount = 0;
    bool hasStoredRowCount = false;
};

qcx::Result<void> WriteChunkAttributes(HighFive::DataSet& dataSet,
                                       const RiChunkMeta& meta,
                                       std::uint64_t checksum) {
    return WrapH5([&]() {
        auto versionAttr = dataSet.createAttribute<std::uint32_t>(
            "engine_version", HighFive::DataSpace::From(qcx::integrals::kIntegralEngineVersion));
        versionAttr.write(qcx::integrals::kIntegralEngineVersion);

        auto checksumAttr = dataSet.createAttribute<std::uint64_t>(
            "chunk_checksum", HighFive::DataSpace::From(checksum));
        checksumAttr.write(checksum);

        auto shellStartAttr = dataSet.createAttribute<std::uint64_t>(
            "aux_shell_start", HighFive::DataSpace::From(meta.auxShellStart));
        shellStartAttr.write(static_cast<std::uint64_t>(meta.auxShellStart));

        auto shellEndAttr = dataSet.createAttribute<std::uint64_t>(
            "aux_shell_end", HighFive::DataSpace::From(meta.auxShellEnd));
        shellEndAttr.write(static_cast<std::uint64_t>(meta.auxShellEnd));

        auto functionStartAttr = dataSet.createAttribute<std::uint64_t>(
            "aux_function_start", HighFive::DataSpace::From(meta.auxFunctionStart));
        functionStartAttr.write(static_cast<std::uint64_t>(meta.auxFunctionStart));

        auto functionEndAttr = dataSet.createAttribute<std::uint64_t>(
            "aux_function_end", HighFive::DataSpace::From(meta.auxFunctionEnd));
        functionEndAttr.write(static_cast<std::uint64_t>(meta.auxFunctionEnd));

        // The row layout, stated explicitly rather than inferred from the
        // dataset's first dimension: a reduced store must not be readable
        // as a full one whose n merely looks wrong.
        const std::uint64_t storedRows = RiChunkStoredRowCount(meta);
        auto storedRowsAttr = dataSet.createAttribute<std::uint64_t>(
            "stored_row_count", HighFive::DataSpace::From(storedRows));
        storedRowsAttr.write(storedRows);
    });
}

// The semantic refuse-to-serve checks (stale engine stamp, range
// attributes disagreeing with the requested chunk metadata) are
// kInvalidArgument - only genuine HDF5 failures are kIOError.
qcx::Result<void> ValidateStoredAttributes(const StoredChunkAttributes& attributes,
                                           const RiChunkMeta& meta) {
    if (attributes.engineVersion != qcx::integrals::kIntegralEngineVersion)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the chunk carries a stale kIntegralEngineVersion - refusing to serve"});
    }

    if (attributes.auxShellStart != meta.auxShellStart ||
        attributes.auxShellEnd != meta.auxShellEnd ||
        attributes.auxFunctionStart != meta.auxFunctionStart ||
        attributes.auxFunctionEnd != meta.auxFunctionEnd)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the chunk's stored range attributes disagree with the requested chunk metadata"});
    }

    // A store without the attribute predates the row layouts, so the only
    // layout it can hold is the full n^2; one that has it must agree.
    const std::uint64_t expectedRows = RiChunkStoredRowCount(meta);

    if (attributes.hasStoredRowCount && attributes.storedRowCount != expectedRows)
    {
        return std::unexpected(qcx::Error{
            qcx::ErrorCode::kInvalidArgument,
            "the chunk's stored row-layout attribute disagrees with the requested chunk metadata"});
    }

    if (!attributes.hasStoredRowCount &&
        expectedRows != meta.orbitalFunctionCount * meta.orbitalFunctionCount)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the chunk predates the uv-reduced row layout and cannot serve a reduced "
                       "metadata request"});
    }

    return {};
}

} // namespace

qcx::Result<void> SaveRiTensorChunk(const std::filesystem::path& path,
                                    const qcx::molecule::Molecule& molecule,
                                    std::string_view orbitalBasisName,
                                    std::string_view auxBasisName,
                                    const RiChunkMeta& meta,
                                    const Eigen::MatrixXd& chunkMatrix) {
    auto validated = ValidateMeta(meta, chunkMatrix);

    if (!validated.has_value())
    {
        return std::unexpected(validated.error());
    }

    if (chunkMatrix.rows() == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the chunk matrix must not be empty"});
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

    // Append-only: refuse an existing chunk dataset.
    bool exists = false;

    if (auto existsCheck =
            WrapH5([&]() { exists = file->exist(ChunkDatasetPath(meta.chunkIndex)); });
        !existsCheck.has_value())
    {
        return std::unexpected(existsCheck.error());
    }

    if (exists)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "the RI tensor chunk is already stored"});
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
    } else
    {
        if (auto written =
                internal::WriteMoleculeGroup(*file, molecule, orbitalBasisName, auxBasisName);
            !written.has_value())
        {
            return std::unexpected(written.error());
        }
    }

    // Eigen is column-major, HDF5 is row-major: the element-wise copy
    // `data()[r + c*rows]` -> flat position `r*cols + c` - never a raw
    // data() blit across layouts. The checksum covers this exact byte
    // stream (the chunk's bytes as written to the dataset).
    const std::size_t rows = static_cast<std::size_t>(chunkMatrix.rows());
    const std::size_t cols = static_cast<std::size_t>(chunkMatrix.cols());
    std::vector<double> flat(rows * cols);

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            flat[row * cols + col] =
                chunkMatrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col));
        }
    }

    const std::uint64_t checksum = internal::Fnv1a64(
        reinterpret_cast<const std::byte*>(flat.data()), flat.size() * sizeof(double));

    std::shared_ptr<HighFive::DataSet> dataSet;

    auto written = WrapH5([&]() {
        // The /integrals/ao/eri path exists in an ERI store file; a fresh
        // file must build it level by level.
        HighFive::Group integrals =
            file->exist("integrals") ? file->getGroup("integrals") : file->createGroup("integrals");
        HighFive::Group ao =
            integrals.exist("ao") ? integrals.getGroup("ao") : integrals.createGroup("ao");
        HighFive::Group eri = ao.exist("eri") ? ao.getGroup("eri") : ao.createGroup("eri");
        HighFive::Group chunks = eri.exist("ri_tensor_chunks")
                                     ? eri.getGroup("ri_tensor_chunks")
                                     : eri.createGroup("ri_tensor_chunks");
        dataSet = std::make_shared<HighFive::DataSet>(
            chunks.createDataSet(std::to_string(meta.chunkIndex),
                                 HighFive::DataSpace({rows, cols}),
                                 HighFive::AtomicType<double>()));
        // write() would reject the flat buffer against the rank-2 dataspace.
        dataSet->write_raw(flat.data(), dataSet->getDataType());
        file->flush();
    });

    if (!written.has_value())
    {
        return std::unexpected(written.error());
    }

    return WriteChunkAttributes(*dataSet, meta, checksum);
}

qcx::Result<Eigen::MatrixXd> LoadRiTensorChunk(const std::filesystem::path& path,
                                               const qcx::molecule::Molecule& molecule,
                                               std::string_view orbitalBasisName,
                                               std::string_view auxBasisName,
                                               const RiChunkMeta& meta) {
    if (auto ranges = ValidateMetaRanges(meta); !ranges.has_value())
    {
        return std::unexpected(ranges.error());
    }

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

    const std::size_t rows = RiChunkStoredRowCount(meta);
    const std::size_t cols = meta.auxFunctionEnd - meta.auxFunctionStart;

    // Shape validation on load: the dataset's own dimensions must equal
    // the chunk metadata's before any buffer is sized from them
    // (kInvalidArgument on disagreement - never UB on a corrupt header).
    std::vector<std::size_t> dims;

    auto readDims = WrapH5([&]() {
        dims = file->getDataSet(ChunkDatasetPath(meta.chunkIndex)).getSpace().getDimensions();
    });

    if (!readDims.has_value())
    {
        return std::unexpected(readDims.error());
    }

    if (dims.size() != 2 || dims.at(0) != rows || dims.at(1) != cols)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "the stored chunk's shape disagrees with the requested chunk metadata"});
    }

    std::vector<double> flat(rows * cols);
    StoredChunkAttributes stored;

    auto read = WrapH5([&]() {
        HighFive::DataSet dataSet = file->getDataSet(ChunkDatasetPath(meta.chunkIndex));
        dataSet.read_raw(flat.data(), dataSet.getDataType());

        // A failed attribute read is a genuine HDF5 failure (kIOError);
        // the stored values' SEMANTIC validation happens below.
        dataSet.getAttribute("engine_version").read(stored.engineVersion);
        dataSet.getAttribute("chunk_checksum").read(stored.checksum);
        dataSet.getAttribute("aux_shell_start").read(stored.auxShellStart);
        dataSet.getAttribute("aux_shell_end").read(stored.auxShellEnd);
        dataSet.getAttribute("aux_function_start").read(stored.auxFunctionStart);
        dataSet.getAttribute("aux_function_end").read(stored.auxFunctionEnd);

        // Optional: a store written before the row layouts existed carries
        // no such attribute and can only hold the full n^2 layout.
        stored.hasStoredRowCount = dataSet.hasAttribute("stored_row_count");

        if (stored.hasStoredRowCount)
        {
            dataSet.getAttribute("stored_row_count").read(stored.storedRowCount);
        }
    });

    if (!read.has_value())
    {
        return std::unexpected(read.error());
    }

    if (auto semantic = ValidateStoredAttributes(stored, meta); !semantic.has_value())
    {
        return std::unexpected(semantic.error());
    }

    // The checksum covers the byte stream just read: a mismatch is kIOError
    // (the integrity policy - never a silent recompute).
    const std::uint64_t computedChecksum = internal::Fnv1a64(
        reinterpret_cast<const std::byte*>(flat.data()), flat.size() * sizeof(double));

    if (computedChecksum != stored.checksum)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kIOError, "the RI tensor chunk failed its checksum"});
    }

    // The reversed element-wise copy (column-major Eigen from the row-major
    // on-disk layout).
    Eigen::MatrixXd matrix =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                flat[row * cols + col];
        }
    }

    return matrix;
}

} // namespace qcx::storage
