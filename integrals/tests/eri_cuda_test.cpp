// CUDA MD lane tests (eri_cuda.hpp):
//   - H2/STO-3G vs the committed CSV (all 16 forms, s-only) and the
//     canonicalization contract,
//   - per-class GPU-vs-CPU fp64 parity on a synthetic s/p/d shellset,
//   - the H2 RHF pin (-1.1167143252 at 1e-8) through GPU batches assembled
//     into the dense ERI tensor,
//   - the 3c mpmath grid (kind "R" rows): the GPU covers the full RI
//     rectangle INCLUDING lBra > lKet classes; the CPU-supported classes
//     (lBra <= lKet) are checked against BuildRiTensor on H2/STO-3G x the
//     tiny s/p aux set,
//   - the certified fp32 lane: |fp32 - fp64| <= errorBounds per element,
//     bounds positive,
//   - the error paths: variant overrides across families, empty lists,
//     out-of-matrix classes (the instantiated matrix is the 2e triangle
//     lBra <= lKet <= 2*Lmax UNION the RI rectangle; a class beyond it can
//     only be built when QcxIntegralsCudaLMax < 6 - the parser cap l <= 6
//     keeps every reachable class inside the Lmax=6 matrix).
//
// The engine's Create reports kDeviceError when no CUDA device is present,
// so the suite gates on that (the host TU stays CUDA-header-free).

#include "h2_sto3g.hpp"
#include "h2_sto3g_fixture.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/eri_cuda.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/scf/rhf.hpp"
#include "shellset_fixture.hpp"
#include "tensor_conversions.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using qcx::ErrorCode;
using qcx::integrals::ComputeEriBatch;
using qcx::integrals::EriCudaEngine;
using qcx::integrals::EriCudaOptions;
using qcx::integrals::EriCudaVariant;
using qcx::integrals::ShellFunctionCount;
using qcx::integrals::ShellQuartet;
using qcx::integrals::ShellTriple;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// Skips the test when the machine has no usable CUDA device (the engine's
// Create is the gate: it probes cudaGetDeviceCount at construction).
void RequireCudaDevice(const qcx::Result<EriCudaEngine>& engine) {
    if (!engine.has_value() && engine.error().code == ErrorCode::kDeviceError)
    {
        GTEST_SKIP() << "no CUDA device: " << engine.error().message;
    }

    ASSERT_TRUE(engine.has_value()) << engine.error().message;
}

// The mpmath grid rows of tools/gen_md_reference.py (kind "R").
struct GridRow {
    std::string kind;
    std::size_t i, j, p, ri, rj, rp;
    double value;
};

std::vector<GridRow> Load3cGrid(const std::string& fileName) {
    const std::string path = std::string(QcxIntegralsDataDir) + "/" + fileName;
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<GridRow> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        GridRow row{};
        std::getline(ss, row.kind, ',');
        std::getline(ss, cell, ',');
        row.i = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.j = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.p = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.ri = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.rj = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.rp = std::stoull(cell);
        std::getline(ss, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);
        rows.push_back(row);
    }

    return rows;
}

// The tiny s/p auxiliary set of the generator's 3c grid (kTinyAux of
// ri_engine_test.cpp): per atom one s and one p shell.
inline constexpr std::string_view kTinyAux = R"(BASIS "ao basis" SPHERICAL PRINT
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

// An f-shell-only basis (one O atom): the sole pair class is (6,6), L = 12 -
// a kV2 class on every instantiation matrix that covers it.
inline constexpr std::string_view kFShellBasis = R"(BASIS "ao basis" SPHERICAL PRINT
O    F
      1.0000000000E+00       1.0000000000E+00
END
)";

qcx::Result<qcx::molecule::Molecule> MakeSingleOxygen() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({1, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}}, std::move(*coordinates), 0, 1);
}

// ---------------------------------------------------------------------------
// The 2e lane
// ---------------------------------------------------------------------------

TEST(EriCudaEngineTest, H2MatchesTheSOnlyCsv) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    // All 16 requested forms land in the (0,0) class but as distinct pair
    // blocks: (0,0|0,0), (0,0|0,1), (0,1|0,1), ... - each with its own ERI.
    std::vector<ShellQuartet> quartets;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            for (std::size_t k = 0; k < 2; ++k)
            {
                for (std::size_t l = 0; l < 2; ++l)
                {
                    quartets.push_back({i, j, k, l});
                }
            }
        }
    }

    auto batch = engine->ComputeBatch(quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->computed.size(), 16u);
    ASSERT_EQ(batch->values.size(), 16u);
    const std::vector<qcx::integrals::test::ReferenceValue> reference =
        qcx::integrals::test::LoadReferenceValues();

    for (std::size_t t = 0; t < 16; ++t)
    {
        // The engine returns values in its canonical sorted order
        // (AssembleClassBatches); values[t] aligns with computed[t], NOT
        // with the request order. Index the CSV expectation through
        // computed[t] - the old lookup keyed by quartets[t] compared sorted
        // values against request-order expectations (6 of 16 mismatched,
        // all with CORRECT ERI values at the wrong positions).
        double expected = 0.0;

        for (const auto& row : reference)
        {
            if (row.kind == "ERI" && row.i == batch->computed[t].i &&
                row.j == batch->computed[t].j && row.k == batch->computed[t].k &&
                row.l == batch->computed[t].l)
            {
                expected = row.value;
            }
        }

        EXPECT_NEAR(batch->values[t], expected, 1e-10) << "request " << t;
    }
}

TEST(EriCudaEngineTest, ShellsetMatchesTheCpu) {
    auto basis = qcx::basisset::ParseNwchemText(qcx::testing::kShellsetBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    // All canonical quartets over the five shells (classes up to (4,4)).
    std::vector<ShellQuartet> quartets;

    for (std::size_t a = 0; a < 5; ++a)
    {
        for (std::size_t b = 0; b < 5; ++b)
        {
            if (b < a)
            {
                continue;
            }

            for (std::size_t c = 0; c < 5; ++c)
            {
                for (std::size_t d = 0; d < 5; ++d)
                {
                    if (d < c)
                    {
                        continue;
                    }

                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    ASSERT_GT(quartets.size(), 0u);
    auto gpu = engine->ComputeBatch(quartets);
    ASSERT_TRUE(gpu.has_value()) << gpu.error().message;
    auto cpu = ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(cpu.has_value()) << cpu.error().message;
    ASSERT_EQ(gpu->computed.size(), cpu->computed.size());
    ASSERT_EQ(gpu->values.size(), cpu->values.size());

    for (std::size_t w = 0; w < gpu->values.size(); ++w)
    {
        // The fp64 lanes agree to ~1-2 ulp (device pow/exp vs MSVC); the
        // scale-aware tolerance keeps the comparison meaningful for the
        // near-zero elements.
        const double expected = cpu->values[w];
        EXPECT_NEAR(gpu->values[w], expected, 1e-12 * (1.0 + std::abs(expected)))
            << "packed element " << w;
    }
}

TEST(EriCudaEngineTest, H2RHFpinThroughGpuBatches) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    std::vector<ShellQuartet> quartets;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            for (std::size_t k = 0; k < 2; ++k)
            {
                for (std::size_t l = 0; l < 2; ++l)
                {
                    quartets.push_back({i, j, k, l});
                }
            }
        }
    }

    auto batch = engine->ComputeBatch(quartets);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->values.size(), 16u);

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    // s-only, but the dense (uv|ws) tensor is NOT constant - the symmetry
    // classes genuinely differ (the recorded lesson: (01|01) = 0.2970 vs
    // (00|11) = 0.5697). Values come back in the engine's canonical order
    // (one representative per 8-fold orbit, md_batch.cpp AssembleClassBatches);
    // the ERI is identical across the orbit, so stamp all 8 partners.
    auto eri = qcx::memory::Tensor<double, 4, qcx::backend::CpuTag>::Create({2, 2, 2, 2});
    ASSERT_TRUE(eri.has_value()) << eri.error().message;
    std::array<double, 16> valueByKey{};

    for (std::size_t t = 0; t < 16; ++t)
    {
        const auto& c = batch->computed[t];
        const double v = batch->values[t];
        const std::array<std::size_t, 4> shells{c.i, c.j, c.k, c.l};
        // (i,j|k,l), (j,i|k,l), (i,j|l,k), (j,i|l,k) and the pair swap.
        const std::array<std::array<std::size_t, 4>, 8> partners = {
            {{shells[0], shells[1], shells[2], shells[3]},
             {shells[1], shells[0], shells[2], shells[3]},
             {shells[0], shells[1], shells[3], shells[2]},
             {shells[1], shells[0], shells[3], shells[2]},
             {shells[2], shells[3], shells[0], shells[1]},
             {shells[3], shells[2], shells[0], shells[1]},
             {shells[2], shells[3], shells[1], shells[0]},
             {shells[3], shells[2], shells[1], shells[0]}}};

        for (const auto& p : partners)
        {
            valueByKey[(p[0] << 3) | (p[1] << 2) | (p[2] << 1) | p[3]] = v;
        }
    }

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            for (std::size_t k = 0; k < 2; ++k)
            {
                for (std::size_t l = 0; l < 2; ++l)
                {
                    (*eri)(i, j, k, l) = valueByKey[(i << 3) | (j << 2) | (k << 1) | l];
                }
            }
        }
    }

    eri->MarkHostDirty();
    auto result = qcx::scf::RunRhfScf(
        *molecule, ToMatrix(*overlap), ToMatrix(*kinetic) + ToMatrix(*nuclear), *eri);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_NEAR(result->totalEnergy, -1.1167143252, 1e-8);
}

// ---------------------------------------------------------------------------
// The certified fp32 lane
// ---------------------------------------------------------------------------

TEST(EriCudaEngineTest, CertifiedBoundsHoldOnH2) {
#if !QcxIntegralsF32
    GTEST_SKIP() << "the fp32 pipeline is not instantiated (QCX_INTEGRALS_F32)";
#else
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    std::vector<ShellQuartet> quartets;
    quartets.push_back({0, 0, 0, 0});

    auto fp64 = engine->ComputeBatch(quartets);
    ASSERT_TRUE(fp64.has_value()) << fp64.error().message;
    auto certified = engine->ComputeBatchCertified(quartets);
    ASSERT_TRUE(certified.has_value()) << certified.error().message;
    ASSERT_EQ(certified->errorBounds.size(), 1u);
    ASSERT_EQ(certified->values.size(), 1u);

    // The a-priori bound dominates the observed deviation (nothing is
    // approximate unless the bound covers it).
    const double deviation = std::abs(static_cast<double>(certified->values[0]) - fp64->values[0]);
    EXPECT_LE(deviation, certified->errorBounds[0]);
    EXPECT_GT(certified->errorBounds[0], 0.0);
#endif
}

// A contracted H/He shellset: 6-primitive s, 4-primitive p, 3-primitive d
// on both atoms. The 0.1f restructure's multi-chunk path needs rows >
// chunkRows in a fused class - the uncontracted/single-primitive fixtures
// above always fit one chunk, so the accSh/braTp chunk indexing the
// restructure fixed never ran under an assertion (found 2026-08-23).
// With 6 bra primitives and 3 bra pairs per (0,0)-class batch (18 slots vs
// the ~3-row (4,4) chunk limit), every fused class here chunks at least
// twice.
inline constexpr std::string_view kContractedShellsetBasis = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      8.0000000000E+00       1.0000000000E+00
      2.0000000000E+00       1.0000000000E+00
      6.0000000000E-01       1.0000000000E+00
      2.0000000000E-01       1.0000000000E+00
      6.0000000000E-02       1.0000000000E+00
      2.0000000000E-02       1.0000000000E+00
H    P
      2.0000000000E+00       1.0000000000E+00
      6.0000000000E-01       1.0000000000E+00
      2.0000000000E-01       1.0000000000E+00
      6.0000000000E-02       1.0000000000E+00
H    D
      1.0000000000E+00       1.0000000000E+00
      3.0000000000E-01       1.0000000000E+00
      1.0000000000E-01       1.0000000000E+00
He   S
      8.0000000000E+00       1.0000000000E+00
      2.0000000000E+00       1.0000000000E+00
      6.0000000000E-01       1.0000000000E+00
      2.0000000000E-01       1.0000000000E+00
      6.0000000000E-02       1.0000000000E+00
      2.0000000000E-02       1.0000000000E+00
He   P
      2.0000000000E+00       1.0000000000E+00
      6.0000000000E-01       1.0000000000E+00
      2.0000000000E-01       1.0000000000E+00
      6.0000000000E-02       1.0000000000E+00
He   D
      1.0000000000E+00       1.0000000000E+00
      3.0000000000E-01       1.0000000000E+00
      1.0000000000E-01       1.0000000000E+00
END
)";

TEST(EriCudaEngineTest, ContractedShellsetMultiChunkMatchesTheCpu) {
    auto basis = qcx::basisset::ParseNwchemText(kContractedShellsetBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    // All canonical quartets over the six shells (the ShellsetMatchesTheCpu
    // pattern; the fused classes top out at (4,4) here).
    std::vector<ShellQuartet> quartets;

    for (std::size_t a = 0; a < 6; ++a)
    {
        for (std::size_t b = 0; b < 6; ++b)
        {
            if (b < a)
            {
                continue;
            }

            for (std::size_t c = 0; c < 6; ++c)
            {
                for (std::size_t d = 0; d < 6; ++d)
                {
                    if (d < c)
                    {
                        continue;
                    }

                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    ASSERT_GT(quartets.size(), 0u);
    auto gpu = engine->ComputeBatch(quartets);
    ASSERT_TRUE(gpu.has_value()) << gpu.error().message;
    auto cpu = ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(cpu.has_value()) << cpu.error().message;
    ASSERT_EQ(gpu->computed.size(), cpu->computed.size());
    ASSERT_EQ(gpu->values.size(), cpu->values.size());

    for (std::size_t w = 0; w < gpu->values.size(); ++w)
    {
        // The fp64 lanes agree to ~1-2 ulp (device pow/exp vs MSVC); the
        // scale-aware tolerance keeps the comparison meaningful for the
        // near-zero elements.
        const double expected = cpu->values[w];
        EXPECT_NEAR(gpu->values[w], expected, 1e-12 * (1.0 + std::abs(expected)))
            << "packed element " << w;
    }
}

// ---------------------------------------------------------------------------
// The kV2 (global-memory + cuBLAS) lane
// ---------------------------------------------------------------------------

TEST(EriCudaEngineTest, KV2ClassMatchesTheCpu) {
    // The f-shell pair class (6,6), L = 12: the kV2 happy path (the
    // restructure reworked its geometry - run key, bra GEMM k-dim,
    // certified-bound finalize - and only the benchmarks ever ran it;
    // found 2026-08-23).
    auto basis = qcx::basisset::ParseNwchemText(kFShellBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeSingleOxygen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    if (!qcx::integrals::SupportsL(6))
    {
        GTEST_SKIP() << "the CPU side has no l = 6 classes";
    }

    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);
    std::vector<ShellQuartet> quartets = {{0, 0, 0, 0}};
    auto gpu = engine->ComputeBatch(quartets);
    ASSERT_TRUE(gpu.has_value()) << gpu.error().message;
    // The quartet is four spherical f shells: 7 * 7 bra by 7 * 7 ket.
    ASSERT_EQ(gpu->values.size(), 2401u);
    auto cpu = ComputeEriBatch(*molecule, *basis, quartets);
    ASSERT_TRUE(cpu.has_value()) << cpu.error().message;
    ASSERT_EQ(cpu->values.size(), gpu->values.size());

    for (std::size_t w = 0; w < gpu->values.size(); ++w)
    {
        // The fp64 lanes agree to ~1-2 ulp (device pow/exp vs MSVC); the
        // scale-aware tolerance keeps the comparison meaningful for the
        // near-zero elements.
        const double expected = cpu->values[w];
        EXPECT_NEAR(gpu->values[w], expected, 1e-12 * (1.0 + std::abs(expected)))
            << "packed element " << w;
    }
}

TEST(EriCudaEngineTest, KV2CertifiedBoundsHold) {
#if !QcxIntegralsF32
    GTEST_SKIP() << "the fp32 pipeline is not instantiated (QCX_INTEGRALS_F32)";
#else
    auto basis = qcx::basisset::ParseNwchemText(kFShellBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeSingleOxygen();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    std::vector<ShellQuartet> quartets = {{0, 0, 0, 0}};
    auto fp64 = engine->ComputeBatch(quartets);
    ASSERT_TRUE(fp64.has_value()) << fp64.error().message;
    auto certified = engine->ComputeBatchCertified(quartets);
    ASSERT_TRUE(certified.has_value()) << certified.error().message;
    // One a-priori bound per quartet (taskBound semantics, md_vrr.hpp) - the
    // single bound covers every value of its quartet.
    ASSERT_EQ(certified->values.size(), 2401u);
    ASSERT_EQ(certified->errorBounds.size(), quartets.size());
    EXPECT_GT(certified->errorBounds[0], 0.0);

    for (std::size_t w = 0; w < certified->values.size(); ++w)
    {
        const double deviation =
            std::abs(static_cast<double>(certified->values[w]) - fp64->values[w]);
        EXPECT_LE(deviation, certified->errorBounds[0]) << "packed element " << w;
    }
#endif
}

// ---------------------------------------------------------------------------
// The 3c (RI) lane
// ---------------------------------------------------------------------------

TEST(EriCudaEngineTest, RiMatchesTheMpmathGrid) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const std::vector<GridRow> grid = Load3cGrid("md_3c_reference.csv");
    ASSERT_GT(grid.size(), 0u);

    // The unique triples in first-seen order; the values come back in the
    // engine's sorted order, so each triple maps through computed[].
    std::vector<ShellTriple> triples;

    for (const GridRow& row : grid)
    {
        if (row.kind != "R")
        {
            continue;
        }

        const ShellTriple triple{row.i, row.j, row.p};
        bool seen = false;

        for (const ShellTriple& t : triples)
        {
            if (t.i == triple.i && t.j == triple.j && t.k == triple.k)
            {
                seen = true;
                break;
            }
        }

        if (!seen)
        {
            triples.push_back(triple);
        }
    }

    ASSERT_GT(triples.size(), 0u);
    auto batch = engine->ComputeRiBatch(*aux, triples);
    ASSERT_TRUE(batch.has_value()) << batch.error().message;
    ASSERT_EQ(batch->computed.size(), triples.size());

    // The block base per computed triple (bra.nFuncs x aux.nFuncs packed).
    std::vector<std::size_t> blockBase(batch->computed.size(), 0);
    std::size_t base = 0;

    for (std::size_t t = 0; t < batch->computed.size(); ++t)
    {
        blockBase[t] = base;
        base += ShellFunctionCount(pairList->shells[batch->computed[t].i]) *
                ShellFunctionCount(pairList->shells[batch->computed[t].j]) *
                ShellFunctionCount(auxPairList->shells[batch->computed[t].k]);
    }

    int checked = 0;

    for (const GridRow& row : grid)
    {
        if (row.kind != "R")
        {
            continue;
        }

        std::size_t task = batch->computed.size();

        for (std::size_t t = 0; t < batch->computed.size(); ++t)
        {
            if (batch->computed[t].i == row.i && batch->computed[t].j == row.j &&
                batch->computed[t].k == row.p)
            {
                task = t;
                break;
            }
        }

        ASSERT_NE(task, batch->computed.size())
            << "missing task (" << row.i << "," << row.j << "," << row.p << ")";
        const std::size_t nI = ShellFunctionCount(pairList->shells[row.i]);
        const std::size_t nP = ShellFunctionCount(auxPairList->shells[row.p]);
        // The (f_b * n_i + f_a) bra-major row, the f_p column.
        const std::size_t offset = blockBase[task] + (row.rj * nI + row.ri) * nP + row.rp;
        ASSERT_LT(offset, batch->values.size());
        EXPECT_NEAR(batch->values[offset], row.value, 1e-10)
            << "triple (" << row.i << "," << row.j << "|" << row.p << ") functions (" << row.ri
            << "," << row.rj << "," << row.rp << ")";
        ++checked;
    }

    // Every "R" row was checked (the grid covers lBra > lKet classes the
    // CPU lane cannot run - see the file comment).
    ASSERT_GT(checked, 0);
}

TEST(EriCudaEngineTest, RiMatchesCpuBuildRiTensorOnH2) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    // All (i, j, p): H2 has two s shells, the aux set four s/p shells; every
    // bra class is (0,0) so all classes are CPU-supported (lBra <= lKet).
    std::vector<ShellTriple> triples;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = i; j < 2; ++j)
        {
            for (std::size_t p = 0; p < 4; ++p)
            {
                triples.push_back({i, j, p});
            }
        }
    }

    auto cpu = qcx::integrals::BuildRiTensor(*molecule, *basis, *aux);
    ASSERT_TRUE(cpu.has_value()) << cpu.error().message;
    auto gpu = engine->ComputeRiBatch(*aux, triples);
    ASSERT_TRUE(gpu.has_value()) << gpu.error().message;
    ASSERT_EQ(gpu->computed.size(), triples.size());
    std::size_t base = 0;

    for (std::size_t t = 0; t < gpu->computed.size(); ++t)
    {
        const std::size_t i = gpu->computed[t].i;
        const std::size_t j = gpu->computed[t].j;
        const std::size_t p = gpu->computed[t].k;
        const std::size_t nI = ShellFunctionCount(pairList->shells[i]);
        const std::size_t nJ = ShellFunctionCount(pairList->shells[j]);
        const std::size_t nP = ShellFunctionCount(auxPairList->shells[p]);
        const std::size_t block = nI * nJ * nP;

        for (std::size_t fi = 0; fi < nI; ++fi)
        {
            for (std::size_t fj = 0; fj < nJ; ++fj)
            {
                for (std::size_t fp = 0; fp < nP; ++fp)
                {
                    const std::size_t oI = pairList->shells[i].functionOffset + fi;
                    const std::size_t oJ = pairList->shells[j].functionOffset + fj;
                    const std::size_t oP = auxPairList->shells[p].functionOffset + fp;
                    const double expected = (*cpu)(oI, oJ, oP);
                    EXPECT_NEAR(gpu->values[base + (fj * nI + fi) * nP + fp], expected, 1e-10)
                        << "triple (" << i << "," << j << "|" << p << ") functions (" << fi << ","
                        << fj << "," << fp << ")";
                }
            }
        }

        base += block;
    }
}

// ---------------------------------------------------------------------------
// The error paths
// ---------------------------------------------------------------------------

TEST(EriCudaEngineTest, VariantOverridesAcrossFamilies) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // kV2 on the fused H2 class (0,0): the kV2 instantiation does not exist.
    EriCudaOptions options;
    options.variant = EriCudaVariant::kV2;
    auto engine = EriCudaEngine::Create(*molecule, *basis, options);
    RequireCudaDevice(engine);
    std::vector<ShellQuartet> quartets = {{0, 0, 0, 0}};
    auto batch = engine->ComputeBatch(quartets);
    ASSERT_FALSE(batch.has_value());
    EXPECT_EQ(batch.error().code, ErrorCode::kUnimplemented);

    // A kV2 class (the f-shell single pair class (6,6), L = 12) with a
    // fused override: kUnimplemented as well.
    auto fBasis = qcx::basisset::ParseNwchemText(kFShellBasis);
    ASSERT_TRUE(fBasis.has_value()) << fBasis.error().message;
    auto oxygen = MakeSingleOxygen();
    ASSERT_TRUE(oxygen.has_value()) << oxygen.error().message;
    EriCudaOptions fusedOptions;
    fusedOptions.variant = EriCudaVariant::kV0;
    auto v2engine = EriCudaEngine::Create(*oxygen, *fBasis, fusedOptions);
    RequireCudaDevice(v2engine);
    auto v2batch = v2engine->ComputeBatch(quartets);
    ASSERT_FALSE(v2batch.has_value());
    EXPECT_EQ(v2batch.error().code, ErrorCode::kUnimplemented);
}

TEST(EriCudaEngineTest, OutOfMatrixClassesAndEmptyLists) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    // Empty lists are rejected up front.
    auto empty = engine->ComputeBatch({});
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, ErrorCode::kInvalidArgument);

    // A class beyond the instantiated matrix needs shells with l beyond
    // what the parser can feed it when the matrix cap is 6 (the f-shell
    // pair class (6,6) fits lKet = 6 <= 2*Lmax), so this branch only runs
    // on builds with a smaller cap (CI never configures CUDA, but the knob
    // exists for the local matrix experiments).
    if (QcxIntegralsCudaLMax < 6)
    {
        auto fBasis = qcx::basisset::ParseNwchemText(kFShellBasis);
        ASSERT_TRUE(fBasis.has_value()) << fBasis.error().message;
        auto oxygen = MakeSingleOxygen();
        ASSERT_TRUE(oxygen.has_value()) << oxygen.error().message;
        auto v2engine = EriCudaEngine::Create(*oxygen, *fBasis);
        RequireCudaDevice(v2engine);
        std::vector<ShellQuartet> quartets = {{0, 0, 0, 0}};
        auto batch = v2engine->ComputeBatch(quartets);
        ASSERT_FALSE(batch.has_value());
        EXPECT_EQ(batch.error().code, ErrorCode::kUnimplemented);
    }
}

TEST(EriCudaEngineTest, RiValidation) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto engine = EriCudaEngine::Create(*molecule, *basis);
    RequireCudaDevice(engine);

    // Empty triple list.
    auto empty = engine->ComputeRiBatch(*aux, {});
    ASSERT_FALSE(empty.has_value());
    EXPECT_EQ(empty.error().code, ErrorCode::kInvalidArgument);

    // j < i violates the RI contract.
    std::vector<ShellTriple> triples = {{1, 0, 0}};
    auto bad = engine->ComputeRiBatch(*aux, triples);
    ASSERT_FALSE(bad.has_value());
    EXPECT_EQ(bad.error().code, ErrorCode::kInvalidArgument);

    // An out-of-range aux shell.
    triples = {{0, 0, 8}};
    bad = engine->ComputeRiBatch(*aux, triples);
    ASSERT_FALSE(bad.has_value());
    EXPECT_EQ(bad.error().code, ErrorCode::kInvalidArgument);
}

} // namespace
