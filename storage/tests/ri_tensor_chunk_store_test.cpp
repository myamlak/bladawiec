// The chunked RI-tensor store (ri_tensor_chunk_store.hpp):
// the per-chunk round-trip is bit-identical, a tampered chunk byte fails
// the checksum (kIOError - never a silent recompute), a stale
// kIntegralEngineVersion refuses to serve, an existing chunk dataset
// refuses a rewrite (append-only), and the shape validation on load is
// kInvalidArgument (never UB). The chunk matrices here are hand-built
// (the store is layout-agnostic below the n^2 x k_c contract -
// ri_tensor_test.cpp covers the tensor side of the values).

#include "h2o_sto3g.hpp"
#include "qcx/integrals/engine_version.hpp"
#include "qcx/storage/ri_tensor_chunk_store.hpp"
#include "temp_store.hpp"

#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5File.hpp>
#include <memory>
#include <string>
#include <vector>

namespace {

using qcx::storage::LoadRiTensorChunk;
using qcx::storage::RiChunkMeta;
using qcx::storage::SaveRiTensorChunk;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::ScopedTempStoreFile;

constexpr std::string_view kOrbitalName = "sto-3g";
constexpr std::string_view kAuxName = "def2-universal-jfit";

// Two adjacent chunk metas over a fictitious aux split (shells 0..2 then
// 2..5; functions 0..7 then 7..13 of an orbital basis with n = 4 - the
// store validates shapes against these, it does not derive them).
RiChunkMeta FirstChunkMeta() {
    RiChunkMeta meta;
    meta.chunkIndex = 0;
    meta.auxShellStart = 0;
    meta.auxShellEnd = 2;
    meta.auxFunctionStart = 0;
    meta.auxFunctionEnd = 7;
    meta.orbitalFunctionCount = 4;
    return meta;
}

RiChunkMeta SecondChunkMeta() {
    RiChunkMeta meta = FirstChunkMeta();
    meta.chunkIndex = 1;
    meta.auxShellStart = 2;
    meta.auxShellEnd = 5;
    meta.auxFunctionStart = 7;
    meta.auxFunctionEnd = 13;
    return meta;
}

// A deterministic n^2 x k matrix (exactly representable doubles - the
// store must round-trip the raw bits).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd MakeChunkMatrix(std::size_t rows, std::size_t cols, double seed) {
    Eigen::MatrixXd matrix =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(cols));

    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t col = 0; col < cols; ++col)
        {
            matrix(static_cast<Eigen::Index>(row), static_cast<Eigen::Index>(col)) =
                seed + static_cast<double>(row * 31 + col * 17);
        }
    }

    return matrix;
}

TEST(RiTensorChunkStoreTest, RoundTripIsBitIdentical) {
    ScopedTempStoreFile temp("ri_chunk_roundtrip");

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const RiChunkMeta firstMeta = FirstChunkMeta();
    const RiChunkMeta secondMeta = SecondChunkMeta();
    const Eigen::MatrixXd first = MakeChunkMatrix(16, 7, 0.25);
    const Eigen::MatrixXd second = MakeChunkMatrix(16, 6, -1.5);

    auto savedFirst =
        SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, firstMeta, first);
    ASSERT_TRUE(savedFirst.has_value()) << savedFirst.error().message;
    auto savedSecond =
        SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, secondMeta, second);
    ASSERT_TRUE(savedSecond.has_value()) << savedSecond.error().message;

    auto loadedFirst = LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, firstMeta);
    ASSERT_TRUE(loadedFirst.has_value()) << loadedFirst.error().message;
    auto loadedSecond =
        LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, secondMeta);
    ASSERT_TRUE(loadedSecond.has_value()) << loadedSecond.error().message;

    EXPECT_EQ(loadedFirst->rows(), first.rows());
    EXPECT_EQ(loadedFirst->cols(), first.cols());
    EXPECT_EQ(loadedSecond->rows(), second.rows());
    EXPECT_EQ(loadedSecond->cols(), second.cols());
    // Bit-identity: the whole storage row-major byte stream round-trips.
    EXPECT_EQ(std::memcmp(
                  loadedFirst->data(), first.data(), first.rows() * first.cols() * sizeof(double)),
              0);
    EXPECT_EQ(std::memcmp(loadedSecond->data(),
                          second.data(),
                          second.rows() * second.cols() * sizeof(double)),
              0);
}

TEST(RiTensorChunkStoreTest, TamperedChunkByteFailsTheChecksum) {
    ScopedTempStoreFile temp("ri_chunk_tamper");

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const RiChunkMeta meta = FirstChunkMeta();
    const Eigen::MatrixXd matrix = MakeChunkMatrix(16, 7, 2.0);

    auto saved = SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta, matrix);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    // Flip one byte of the dataset's data through a raw HighFive handle
    // (the store handle must be gone - the Windows single-handle rule).
    auto tampered = [&]() {
        auto file =
            std::make_shared<HighFive::File>(temp.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet dataSet = file->getDataSet("integrals/ao/eri/ri_tensor_chunks/0");
        std::vector<double> flat(static_cast<std::size_t>(16 * 7));
        dataSet.read_raw(flat.data(), dataSet.getDataType());
        flat[40] += 1.0;
        dataSet.write_raw(flat.data(), dataSet.getDataType());
        file->flush();
        return true;
    }();

    ASSERT_TRUE(tampered);

    auto loaded = LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta);
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kIOError);
    EXPECT_NE(loaded.error().message.find("checksum"), std::string::npos);
}

TEST(RiTensorChunkStoreTest, StaleEngineVersionRefusesToServe) {
    ScopedTempStoreFile temp("ri_chunk_stale_version");

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const RiChunkMeta meta = FirstChunkMeta();
    const Eigen::MatrixXd matrix = MakeChunkMatrix(16, 7, 3.0);

    auto saved = SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta, matrix);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    // Bump the engine stamp through a raw handle.
    {
        auto file =
            std::make_shared<HighFive::File>(temp.Path().string(), HighFive::File::ReadWrite);
        HighFive::DataSet dataSet = file->getDataSet("integrals/ao/eri/ri_tensor_chunks/0");
        HighFive::Attribute attribute = dataSet.getAttribute("engine_version");
        attribute.write(qcx::integrals::kIntegralEngineVersion + 1);
        file->flush();
    }

    auto loaded = LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta);
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(loaded.error().message.find("stale"), std::string::npos);
}

TEST(RiTensorChunkStoreTest, ExistingChunkDatasetRefusesRewrite) {
    ScopedTempStoreFile temp("ri_chunk_append_only");

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const RiChunkMeta meta = FirstChunkMeta();
    const Eigen::MatrixXd matrix = MakeChunkMatrix(16, 7, 4.0);

    auto savedFirst =
        SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta, matrix);
    ASSERT_TRUE(savedFirst.has_value()) << savedFirst.error().message;

    // The same chunk index again - even with different data - refuses.
    const Eigen::MatrixXd other = MakeChunkMatrix(16, 7, -4.0);
    auto savedSecond =
        SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta, other);
    ASSERT_FALSE(savedSecond.has_value());
    EXPECT_EQ(savedSecond.error().code, qcx::ErrorCode::kInvalidArgument);
    EXPECT_NE(savedSecond.error().message.find("already stored"), std::string::npos);

    // The original data is untouched.
    auto loaded = LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta);
    ASSERT_TRUE(loaded.has_value()) << loaded.error().message;
    EXPECT_EQ(
        std::memcmp(loaded->data(), matrix.data(), matrix.rows() * matrix.cols() * sizeof(double)),
        0);
}

TEST(RiTensorChunkStoreTest, ShapeMismatchIsRefused) {
    ScopedTempStoreFile temp("ri_chunk_shape");

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const RiChunkMeta meta = FirstChunkMeta();
    const Eigen::MatrixXd matrix = MakeChunkMatrix(16, 7, 5.0);

    // Save refuses a matrix whose columns disagree with the meta's aux
    // function range (a shape/range inconsistency, before any HDF5 call).
    const Eigen::MatrixXd wrongCols = MakeChunkMatrix(16, 6, 5.0);
    auto saved = SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta, wrongCols);
    ASSERT_FALSE(saved.has_value());
    EXPECT_EQ(saved.error().code, qcx::ErrorCode::kInvalidArgument);

    auto savedGood =
        SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta, matrix);
    ASSERT_TRUE(savedGood.has_value()) << savedGood.error().message;

    // Load with a meta whose function range disagrees with the stored
    // dataset's shape - kInvalidArgument, never a mis-sized read.
    RiChunkMeta wrongMeta = meta;
    wrongMeta.auxFunctionEnd = 8;
    auto loadedWrong = LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, wrongMeta);
    ASSERT_FALSE(loadedWrong.has_value());
    EXPECT_EQ(loadedWrong.error().code, qcx::ErrorCode::kInvalidArgument);

    // Load with a meta whose RANGE ATTRIBUTES disagree (same shape, wrong
    // identity) - kInvalidArgument as well.
    RiChunkMeta wrongIdentity = meta;
    wrongIdentity.auxShellStart = 1;
    auto loadedIdentity =
        LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, wrongIdentity);
    ASSERT_FALSE(loadedIdentity.has_value());
    EXPECT_EQ(loadedIdentity.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(RiTensorChunkStoreTest, WrongSystemFingerprintIsRefused) {
    ScopedTempStoreFile temp("ri_chunk_fingerprint");

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const RiChunkMeta meta = FirstChunkMeta();
    const Eigen::MatrixXd matrix = MakeChunkMatrix(16, 7, 6.0);

    auto saved = SaveRiTensorChunk(temp.Path(), *molecule, kOrbitalName, kAuxName, meta, matrix);
    ASSERT_TRUE(saved.has_value()) << saved.error().message;

    // The fingerprint carries the aux basis name: a different name is a
    // different system.
    auto loaded =
        LoadRiTensorChunk(temp.Path(), *molecule, kOrbitalName, "def2-universal-jkfit", meta);
    ASSERT_FALSE(loaded.has_value());
    EXPECT_EQ(loaded.error().code, qcx::ErrorCode::kInvalidArgument);
}

} // namespace
