// The chunked RI-tensor build (ri_engine.hpp BuildRiTensorChunk): every chunk
// slice equals the monolithic BuildRiTensor byte-for-byte on the same tasks -
// across a two-chunk split AND across one-shell chunks (the chunk boundaries
// never split a shell), and the chunk-range validation refuses the degenerate
// ranges. The chunk-scoped task-list contract (the full nPairs x nAuxShells
// list never materialized) is structural in the implementation; the values
// below verify the partition-equivalence the gate rests on, and the
// two-chunk cell also sums the term counters over the chunks and compares
// them with the monolithic pass's (the sink's summation contract, with the
// one pass's dedup state threaded - RiScreenedPassState).

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/molecule/molecule.hpp"

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;

// The tiny s/p auxiliary set of the generator's 3c grid (the shared
// ri_engine_test fixture, inlined here): every H2O atom has an entry.
constexpr std::string_view kTinyAux = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      2.5000000000E+00       3.0000000000E-01
      8.0000000000E-01       8.0000000000E-01
O    P
      1.2000000000E+00       1.0000000000E+00
H    S
      1.0000000000E+00       1.0000000000E+00
H    P
      6.0000000000E-01       1.0000000000E+00
END
)";

// The water C2v reduction over the H-first STO-3G function order
// [H1s, H2s, O1s, O2s, Opy, Opz, Opx] - the hand-built fixture the module's
// own reductions carry (the integrals module cannot include scf headers).
// Used by the refusal test below, where its CONTENTS are never read: the
// refusal fires on the pointer, before any basis work.
qcx::integrals::SymmetryReduction MakeWaterC2vReduction() {
    qcx::integrals::SymmetryReduction reduction;
    reduction.groupOrder = 4;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5, 6}, // I.
        {0, 1, 2, 3, 4, 5, 6}, // sigma_z: identity permutation.
        {1, 0, 2, 3, 4, 5, 6}, // sigma_x: swaps the two H functions.
        {1, 0, 2, 3, 4, 5, 6}, // C2: swaps the two H functions.
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1, 1}, // I.
        {1, 1, 1, 1, 1, -1, 1}, // sigma_z: p_z -> -p_z.
        {1, 1, 1, 1, 1, 1, -1}, // sigma_x: p_x -> -p_x.
        {1, 1, 1, 1, 1, -1, -1}, // C2: p_z and p_x flip.
    };
    return reduction;
}

// The monolith's {n, n, nAux} tensor flattened into the n^2 x nAux
// contraction layout (rows u*n + v, columns P) - the exact Create-time
// flatten of the monolithic path (ri_engine.cpp).
Eigen::MatrixXd FlattenTensor(const qcx::memory::Tensor<double, 3, qcx::backend::CpuTag>& tensor) {
    const std::size_t n = tensor.Shape()[0];
    const std::size_t nAux = tensor.Shape()[2];
    Eigen::MatrixXd flat =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n * n), static_cast<Eigen::Index>(nAux));

    for (std::size_t u = 0; u < n; ++u)
    {
        for (std::size_t v = 0; v < n; ++v)
        {
            for (std::size_t p = 0; p < nAux; ++p)
            {
                flat(static_cast<Eigen::Index>(u * n + v), static_cast<Eigen::Index>(p)) =
                    tensor(u, v, p);
            }
        }
    }

    return flat;
}

// Concatenates the chunk slices' columns in chunk order into the full
// n^2 x nAux contraction matrix (chunk c's columns are the global aux
// functions of the c-th chunk - contiguous and ordered).
Eigen::MatrixXd ConcatenateChunks(const std::vector<Eigen::MatrixXd>& chunks) {
    const Eigen::Index rows = chunks.front().rows();
    Eigen::Index totalCols = 0;

    for (const Eigen::MatrixXd& chunk : chunks)
    {
        EXPECT_EQ(chunk.rows(), rows);
        totalCols += chunk.cols();
    }

    Eigen::MatrixXd full = Eigen::MatrixXd::Zero(rows, totalCols);
    Eigen::Index columnBase = 0;

    for (const Eigen::MatrixXd& chunk : chunks)
    {
        full.block(0, columnBase, rows, chunk.cols()) = chunk;
        columnBase += chunk.cols();
    }

    return full;
}

TEST(RiEngineChunkTest, ChunkedEqualsMonolithicOnTheSameTasks) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The vendored jfit carries g shells on O: the CI Lmax=2 builds cannot
    // instantiate the classes those need.
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    // The term counters of the same partition: a chunked caller's
    // chunks sum to the monolithic build's totals when ONE pass's dedup state
    // spans the calls (ri_engine.hpp RiScreenedPassState). For x and g3 that is
    // the sink's own arithmetic (the chunk ranges partition the aux shells and
    // the tasks); for p3 it is the state's - on the route's standalone per-call
    // stamps each chunk re-counts every orbital pair it survives with, so a
    // chunked caller's sum reads the pairs once per chunk.
    qcx::integrals::RiTermCounters monolithicCounters;
    auto monolithic =
        qcx::integrals::BuildRiTensor(*molecule, *basis, *aux, {}, &monolithicCounters);
    ASSERT_TRUE(monolithic.has_value()) << monolithic.error().message;
    const Eigen::MatrixXd expected = FlattenTensor(*monolithic);

    // The aux shell count (molecule-scoped pair-list order) and the
    // per-chunk function offsets.
    auto auxPairs = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairs.has_value()) << auxPairs.error().message;
    const std::size_t nAuxShells = auxPairs->shells.size();
    ASSERT_GE(nAuxShells, 3u);

    // A two-chunk split in the middle of the aux shells.
    std::vector<Eigen::MatrixXd> chunks;
    const std::size_t split = nAuxShells / 2;
    qcx::integrals::RiTermCounters chunkedCounters;
    qcx::integrals::RiScreenedPassState passState;

    for (const std::pair<std::size_t, std::size_t> range : {
             std::pair<std::size_t, std::size_t>{0, split},
             std::pair<std::size_t, std::size_t>{split, nAuxShells},
         })
    {
        auto chunk = qcx::integrals::BuildRiTensorChunk(
            *molecule, *basis, *aux, range.first, range.second, {}, &chunkedCounters, &passState);
        ASSERT_TRUE(chunk.has_value()) << chunk.error().message;
        EXPECT_EQ(chunk->cols(),
                  static_cast<Eigen::Index>(
                      auxPairs->shells[range.second - 1].functionOffset +
                      qcx::integrals::ShellFunctionCount(auxPairs->shells[range.second - 1]) -
                      auxPairs->shells[range.first].functionOffset));
        chunks.push_back(std::move(*chunk));
    }

    const Eigen::MatrixXd assembled = ConcatenateChunks(chunks);
    EXPECT_EQ(assembled.rows(), expected.rows());
    EXPECT_EQ(assembled.cols(), expected.cols());
    // Byte-for-byte on the same tasks.
    EXPECT_EQ(
        std::memcmp(assembled.data(),
                    expected.data(),
                    static_cast<std::size_t>(expected.rows() * expected.cols()) * sizeof(double)),
        0);

    // The counters of that partition, against the monolithic build's: the
    // documented summation contract ("a chunked caller sums its chunks'").
    // The p3 floor is what keeps the equality from holding vacuously.
    EXPECT_GT(monolithicCounters.p3, 0u) << "the monolithic pass engaged no orbital pair";
    EXPECT_EQ(chunkedCounters.p3, monolithicCounters.p3)
        << "a two-chunk caller's p3 sum is not the pass's p3 (" << chunks.size() << " chunks)";
    EXPECT_EQ(chunkedCounters.x, monolithicCounters.x)
        << "the chunks reached a different auxiliary shell set from the monolithic pass";
    EXPECT_EQ(chunkedCounters.g3, monolithicCounters.g3)
        << "the chunks evaluated different kernel work from the monolithic pass";
}

TEST(RiEngineChunkTest, OneShellChunksConcatenateToTheMonolith) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto monolithic = qcx::integrals::BuildRiTensor(*molecule, *basis, *aux);
    ASSERT_TRUE(monolithic.has_value()) << monolithic.error().message;
    const Eigen::MatrixXd expected = FlattenTensor(*monolithic);

    // One-shell chunks: every shell boundary is a chunk boundary, and an
    // all-screened shell (an empty task list) must come back as a zero
    // slice, not an error.
    auto auxPairs = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairs.has_value()) << auxPairs.error().message;
    const std::size_t nAuxShells = auxPairs->shells.size();
    ASSERT_GT(nAuxShells, 1u);
    std::vector<Eigen::MatrixXd> chunks;

    for (std::size_t shell = 0; shell < nAuxShells; ++shell)
    {
        auto chunk = qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, shell, shell + 1);
        ASSERT_TRUE(chunk.has_value()) << chunk.error().message;
        chunks.push_back(std::move(*chunk));
    }

    const Eigen::MatrixXd assembled = ConcatenateChunks(chunks);
    EXPECT_EQ(assembled.rows(), expected.rows());
    EXPECT_EQ(assembled.cols(), expected.cols());
    EXPECT_EQ(
        std::memcmp(assembled.data(),
                    expected.data(),
                    static_cast<std::size_t>(expected.rows() * expected.cols()) * sizeof(double)),
        0);
}

TEST(RiEngineChunkTest, DegenerateChunkRangesAreRefused) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    // An empty range.
    auto empty = qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, 3, 3);
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, qcx::ErrorCode::kInvalidArgument);

    // A range past the aux shell count.
    auto pastEnd = qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, 4, 9);
    ASSERT_FALSE(pastEnd.has_value());
    EXPECT_EQ(pastEnd.error().code, qcx::ErrorCode::kInvalidArgument);

    // A zero maxBatchBytes.
    qcx::integrals::RiEngineOptions options;
    options.maxBatchBytes = 0;
    auto zeroBatch = qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, 0, 1, options);
    ASSERT_FALSE(zeroBatch.has_value());
    EXPECT_EQ(zeroBatch.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The point-group reduction fields are REFUSED BY NAME on this entry point
// (the RiFullFockBuilder Create refusal's shape at the source): the
// option struct is the ri_j_link family's, so its three symmetry fields are
// in this call's type and are read nowhere in the chunked build - and the
// disk rung's own Create is a caller of this function, so a caller that
// ported the in-memory option set here would otherwise lose the reduction
// without a word.
//
// The refusal fires on the POINTER, before any basis work and before any
// validation of the reductions' contents. The first assertion is what keeps
// that honest: the disk rung's own option recipe - accuracy and
// maxBatchBytes, both fields null - still BUILDS, so the outcome difference
// between the two calls is the field itself and not a malformed fixture.
// The last cell is the one that goes red if the refusal is dropped: the full
// engaged shape (both reductions and the permission) is exactly the option
// set RiJkFockBuilder consumes on the in-memory path, and without the
// refusal this call returns a tensor with the reduction gone and nothing in
// the result to say so.
TEST(RiEngineChunkTest, ReductionFieldsAreRefusedByNameNotSilentlyDropped) {
    const auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    // The disk rung's own option recipe (disk_ri_fock_build.cpp Create): the
    // family's batch vocabulary, no reduction. This call must keep working.
    qcx::integrals::RiEngineOptions diskRungOptions;
    diskRungOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto legal = qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, 0, 1, diskRungOptions);
    ASSERT_TRUE(legal.has_value())
        << "the refusal narrowed the entry point's own contract: " << legal.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReduction();
    ASSERT_FALSE(reduction.isTrivial);

    // The orbital field alone, and then the aux one alone.
    qcx::integrals::RiEngineOptions orbitalOnly;
    orbitalOnly.symmetryReduction = &reduction;
    auto orbitalRefused =
        qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, 0, 1, orbitalOnly);
    ASSERT_FALSE(orbitalRefused.has_value());
    EXPECT_EQ(orbitalRefused.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(orbitalRefused.error().message.find("BuildRiTensorChunk"), std::string::npos);
    EXPECT_NE(orbitalRefused.error().message.find("symmetryReduction"), std::string::npos);

    qcx::integrals::RiEngineOptions auxOnly;
    auxOnly.auxSymmetryReduction = &reduction;
    auto auxRefused = qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, 0, 1, auxOnly);
    ASSERT_FALSE(auxRefused.has_value());
    EXPECT_EQ(auxRefused.error().code, qcx::ErrorCode::kUnimplemented);

    // The full engaged shape: both reductions plus the permission, the option
    // set the in-memory ri_j_link run builds under method.ri_orbit_expansion.
    qcx::integrals::RiEngineOptions engaged;
    engaged.symmetryReduction = &reduction;
    engaged.auxSymmetryReduction = &reduction;
    engaged.symmetryOrbitExpansion = true;
    auto engagedRefused =
        qcx::integrals::BuildRiTensorChunk(*molecule, *basis, *aux, 0, 1, engaged);
    ASSERT_FALSE(engagedRefused.has_value())
        << "a fully specified orbit-expansion option set was accepted by an entry point that "
           "runs no orbit expansion";
    EXPECT_EQ(engagedRefused.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(engagedRefused.error().message.find("silently ignored"), std::string::npos);
}

} // namespace
