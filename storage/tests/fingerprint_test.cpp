// The system-fingerprint determinism contract: identical
// input bits -> identical hex; any single-field change -> a different one.
// The fingerprint is the file-selection gate - a collision must be
// impossible in practice AND re-verified element-wise on open, so this
// test pins the sensitivity at the bit level.
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"
#include "qcx/storage/fingerprint.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using qcx::testing::MakeH2oSto3g;

// A second H2O molecule with exactly one coordinate bit flipped (the O x
// coordinate from 0.0 to the smallest positive double).
qcx::Result<qcx::molecule::Molecule> MakeH2oWithFlippedCoordinate() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = std::nextafter(0.0, 1.0);
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.430428808474167;
    (*coordinates)(1, 1) = 1.107157044080814;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -1.430428808474167;
    (*coordinates)(2, 1) = 1.107157044080814;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// A charged H2O molecule (the only other single-field change).
qcx::Result<qcx::molecule::Molecule> MakeChargedH2o() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.430428808474167;
    (*coordinates)(1, 1) = 1.107157044080814;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -1.430428808474167;
    (*coordinates)(2, 1) = 1.107157044080814;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        1,
        1);
}

TEST(FingerprintTest, IsDeterministicAndSensitive) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const std::string reference = qcx::storage::ComputeFingerprint(*molecule, "sto-3g", "");

    // Deterministic: same inputs, same hex.
    EXPECT_EQ(qcx::storage::ComputeFingerprint(*molecule, "sto-3g", ""), reference);
    // The 16-character lowercase hex shape.
    EXPECT_EQ(reference.size(), std::size_t{16});
    EXPECT_EQ(reference.find_first_not_of("0123456789abcdef"), std::string::npos);

    // One coordinate bit flips the fingerprint.
    const auto flipped = MakeH2oWithFlippedCoordinate();
    ASSERT_TRUE(flipped.has_value()) << flipped.error().message;
    EXPECT_NE(qcx::storage::ComputeFingerprint(*flipped, "sto-3g", ""), reference);

    // The charge flips it.
    const auto charged = MakeChargedH2o();
    ASSERT_TRUE(charged.has_value()) << charged.error().message;
    EXPECT_NE(qcx::storage::ComputeFingerprint(*charged, "sto-3g", ""), reference);

    // The multiplicity flips it.
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});
    ASSERT_TRUE(coordinates.has_value()) << coordinates.error().message;
    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.430428808474167;
    (*coordinates)(1, 1) = 1.107157044080814;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -1.430428808474167;
    (*coordinates)(2, 1) = 1.107157044080814;
    (*coordinates)(2, 2) = 0.0;
    coordinates->MarkHostDirty();
    const auto triplet = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        3);
    ASSERT_TRUE(triplet.has_value()) << triplet.error().message;
    EXPECT_NE(qcx::storage::ComputeFingerprint(*triplet, "sto-3g", ""), reference);

    // Either basis name flips it.
    EXPECT_NE(qcx::storage::ComputeFingerprint(*molecule, "cc-pvdz", ""), reference);
    EXPECT_NE(qcx::storage::ComputeFingerprint(*molecule, "sto-3g", "cc-pvdz-rifit"), reference);

    // An atom's Z flips it (H2 instead of H2O: a one-atom change of the
    // canonical Z list).
    const auto hydrogen = qcx::testing::MakeH2Sto3g();
    ASSERT_TRUE(hydrogen.has_value()) << hydrogen.error().message;
    EXPECT_NE(qcx::storage::ComputeFingerprint(*hydrogen, "sto-3g", ""), reference);
}

} // namespace
