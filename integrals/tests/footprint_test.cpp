// Create-time footprint formula tests (footprint.hpp): the exact per-pair byte
// formulas against hand
// arithmetic (the (g,g) ~214 KB pair among them), the all-survive pattern
// bound and the scratch clamp, and the never-under floor - the estimated
// pair store must cover the engine's real allocation on a small fixture
// (H2O/STO-3G), measured by capacity. The mode-selection tests cover
// the deterministic Create-time decision: FastPath with an exact
// reservation on a generous budget, clamping to fit, the a-priori
// exclusions (i)/(ii), the four-rung ladder diagnostics, the legacy null
// path's bit identity, and the RI-J/QFMM nested-reservation sums. The
// LightPath tests cover the chunk-knob mode forcing (the
// budget-driven LightPath is unreachable at the small end, where the
// pattern saving is zero), the bit-identity pin at chunkPairs = nPairs,
// the chunk-size reduction-order agreement, the knob-path refusal, and
// the 70-water STO-3G counted-admission band (the exclusion tests the
// exact Schwarz-screened pattern bytes, so
// 12 GiB admits the ~4.4 MB counted pattern to the fast rung where the
// 15.1 GB all-survive bound used to force LightPath, while 1 MiB still
// excludes and the light rung also cannot fit - refusal). The
// exclusion-count tests pin CountSchwarzSurvivingPairs and
// CountNearFieldPatternEntries against the sweeps' own counts and
// against a brute-force O(n^2) reference, at all three presets. The
// The count tests pin CountSchwarzSurvivingRiTasks against the RI-J
// task grid's per-cell walk (the screened taskListBytes charge - 16 B per
// surviving task, never-under by equality with the engine's exact
// reserve); the band fixtures pass the counted value to every RiFootprint
// recompute, and the new count tests cover the sweep match, the synthetic
// brute-force reference, the dense/degenerate shapes and the 5000-shape
// sparse grid.

#include "alkane_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "internal/light_footprint.hpp"
#include "internal/md_batch.hpp"
#include "internal/md_defs.hpp"
#include "internal/md_vrr_3c.hpp"
#include "internal/qfmm_geometry.hpp"
#include "internal/qfmm_tree.hpp"
#include "qcx/backend/cpu_backend.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/qfmm_fock_build.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "tensor_conversions.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <vector>

using qcx::integrals::AccuracyPreset;
using qcx::integrals::BuildShellPairs;
using qcx::integrals::DirectJkFockBuilder;
using qcx::integrals::FockBuildMode;
using qcx::integrals::FockBuildOptions;
using qcx::integrals::FockModeInfo;
using qcx::integrals::LeafNearFieldDomain;
using qcx::integrals::QfmmExtentForPreset;
using qcx::integrals::QfmmJBuilder;
using qcx::integrals::QfmmOptions;
using qcx::integrals::RiEngineOptions;
using qcx::integrals::RiJkFockBuilder;
using qcx::integrals::SchwarzThreshold;
using qcx::integrals::internal::AllSurvivePatternBytes;
using qcx::integrals::internal::BuildAuxPairData;
using qcx::integrals::internal::BuildInteractionLists;
using qcx::integrals::internal::BuildLeafDrivenNeighborList;
using qcx::integrals::internal::BuildNeighborList;
using qcx::integrals::internal::BuildPairData;
using qcx::integrals::internal::BuildQfmmTree;
using qcx::integrals::internal::ClampBatchBytes;
using qcx::integrals::internal::ClassTableBytes;
using qcx::integrals::internal::ComputePairGeometries;
using qcx::integrals::internal::CountNearFieldPatternEntries;
using qcx::integrals::internal::CountSchwarzPeakRowWidth;
using qcx::integrals::internal::CountSchwarzSurvivingPairs;
using qcx::integrals::internal::CountSchwarzSurvivingRiTasks;
using qcx::integrals::internal::ETableBytes;
using qcx::integrals::internal::FlattenShells;
using qcx::integrals::internal::Hermite3DCount;
using qcx::integrals::internal::kNeighborListSlack;
using qcx::integrals::internal::kVectorHeaderBytes;
using qcx::integrals::internal::MdPairData;
using qcx::integrals::internal::MdPrimPair;
using qcx::integrals::internal::MdShellInput;
using qcx::integrals::internal::MetricPairStoreBytes;
using qcx::integrals::internal::PairStoreBytes;
using qcx::integrals::internal::PairStoreBytesPerPair;
using qcx::integrals::internal::QfmmOuterStoreBytes;
using qcx::integrals::internal::TransformBytes;
using qcx::integrals::internal::WeightsBytes;
using qcx::memory::WorkspaceBudget;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

namespace {

// H = T + V from the one-electron engines (the eri_cache_test helper's
// shape - BuildCoreHamiltonian is test-local there, not engine API).
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildCoreHamiltonian(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    auto core = ToTensor(ToMatrix(*kinetic) + ToMatrix(*nuclear));

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    return core;
}

// The engine's real per-pair allocation, measured by capacity: the
// MdPairData struct (all vector headers included), the prim-pair structs
// (their per-axis E-table vector headers included), the E-table payloads
// (the PerAxisETable constructor allocates exactly (la+1)(lb+1)(la+lb+1)
// doubles per axis - md_hermite.hpp), the bra transform, the per-prim
// weights and ket transforms, and the inner vector headers. Every vector
// on this path is sized by one assign/reserve, so capacity == size.
std::size_t MeasuredPairBytes(const MdPairData& pair, int la, int lb, std::size_t nPrimPairs) {
    const std::size_t eTablePayload = std::size_t{3} * 8 * static_cast<std::size_t>(la + 1) *
                                      static_cast<std::size_t>(lb + 1) *
                                      static_cast<std::size_t>(la + lb + 1) * nPrimPairs;
    std::size_t inner = 0;

    for (std::size_t prim = 0; prim < pair.primPairs.size(); ++prim)
    {
        inner += pair.braWeights[prim].capacity() * sizeof(double);
        inner += pair.ketTransforms[prim].capacity() * sizeof(double);
    }

    return sizeof(MdPairData) + pair.primPairs.capacity() * sizeof(MdPrimPair) +
           pair.braTransform.capacity() * sizeof(double) + inner +
           (pair.braWeights.capacity() + pair.ketTransforms.capacity()) * kVectorHeaderBytes +
           eTablePayload;
}

// The smallest aux basis the RI machinery exercises (the ri_engine_test
// fixture, inlined here): four shells (O S, O P, H S, H P), 8 functions.
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

// The firing-rule band aux (the ri_engine_test fixture,
// inlined here): a per-element 3-shell set (O and H both S,P,D; 648
// functions at 24 waters). The band's positive fast-light floor gap comes
// from the kWaterSto3g orbital (the 5-function O shells make 2 x tensor +
// riMatrix - slice ~ 320 MB), NOT from aux angular momentum - d shells are
// the ceiling, so the pins also run on the CI Lmax=2 builds (a guard
// would be needed only for l >= 3 aux shells).
inline constexpr std::string_view kRiBandAux = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      2.5000000000E+00       3.0000000000E-01
      8.0000000000E-01       8.0000000000E-01
O    P
      1.2000000000E+00       1.0000000000E+00
O    D
      1.2000000000E+00       1.0000000000E+00
H    S
      1.0000000000E+00       1.0000000000E+00
H    P
      6.0000000000E-01       1.0000000000E+00
H    D
      6.0000000000E-01       1.0000000000E+00
END
)";

// A symmetric physical-ish density (|D| <= 0.75, diagonal ~0.5), the
// magnitudes the screening and certified-bound gates expect (deterministic).
Eigen::MatrixXd PhysicalDensity(std::size_t n) {
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                (i == j) ? 0.5 : 0.05 * static_cast<double>((i + j) % 3);
        }
    }

    return d;
}

// The default batch arena per thread (the FockBuildOptions default batch
// cap times the Create-time team size) - the reference the mode tests size
// their budgets against.
std::size_t DefaultArenaBytes() {
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    return FockBuildOptions{}.maxBatchBytes * threadCount;
}

// The smallest aux the nested-all-survive band needs: one S function per
// oxygen atom (no H functions) - small enough that the (uv|P) tensor and
// the task list stay far below the all-survive pattern bound, which the
// spurious-nested-refusal band requires (the full tiny aux would push the
// tensor over the bound and the clamp would refuse before the nesting).
// A minimal s-only orbital basis (one S per atom, two primitives each).
// The nested-all-survive
// band needs every store term (two full pair-store copies, the task list,
// both (uv|P) copies) far below the all-survive pattern bound - with the
// full STO-3G basis the per-pair store dwarfs the bound at any reachable
// scale and the outer clamp refuses before any nesting. The s-only basis
// makes the band reachable at the 14-water scale (~1.2 KB per pair).
inline constexpr std::string_view kTinyOrbitalSOnly = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      2.5000000000E+00       3.0000000000E-01
      8.0000000000E-01       8.0000000000E-01
H    S
      1.2000000000E+00       5.0000000000E-01
      4.0000000000E-01       5.0000000000E-01
END
)";

// STO-3G O and H in one parsed basis (the same published literals the
// shared fixture carries - kSto3gOxygen and the H text of h2_sto3g.hpp):
// the water-cluster LightPath tests need both elements in one basis.
inline constexpr std::string_view kWaterSto3g = R"(BASIS "ao basis" SPHERICAL PRINT
O    S
      0.1307093214E+03       0.1543289673E+00
      0.2380886605E+02       0.5353281423E+00
      0.6443608313E+01       0.4446345422E+00
O    SP
      0.5033151319E+01      -0.9996722919E-01       0.1559162750E+00
      0.1169596125E+01       0.3995128261E+00       0.6076837186E+00
      0.3803889600E+00       0.7001154689E+00       0.3919573931E+00
H    S
      3.4252509140E+00       1.5432896730E-01
      6.2391372980E-01       5.3532814230E-01
      1.6885540400E-01       4.4463454220E-01
END
)";

// N H2O monomers at the experimental geometry, 20 Bohr apart along y (no
// inter-monomer overlap at that separation) - the multi-molecule scale the
// RI-J mode tests need beyond the single-molecule fixtures.
qcx::Result<qcx::molecule::Molecule> MakeWaterCluster(std::size_t count) {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3 * count, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;

    for (std::size_t k = 0; k < count; ++k)
    {
        const double y = 20.0 * static_cast<double>(k);
        (*coordinates)(3 * k + 0, 0) = 0.0;
        (*coordinates)(3 * k + 0, 1) = y;
        (*coordinates)(3 * k + 0, 2) = 0.0;
        (*coordinates)(3 * k + 1, 0) = 1.430428808474167;
        (*coordinates)(3 * k + 1, 1) = y + 1.107157044080814;
        (*coordinates)(3 * k + 1, 2) = 0.0;
        (*coordinates)(3 * k + 2, 0) = -1.430428808474167;
        (*coordinates)(3 * k + 2, 1) = y + 1.107157044080814;
        (*coordinates)(3 * k + 2, 2) = 0.0;
        atoms.push_back(qcx::molecule::Atom{"O", 8, 0.0});
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

/// A byte count in GiB, for the 586-point structural print's record line.
double Gibibytes(std::size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

/// One printed footprint term: the field's own name, its bytes and its GiB.
void PrintFootprintTerm(const char* name, std::size_t bytes) {
    std::printf("    %-22s %14zu B  %9.4f GiB\n", name, bytes, Gibibytes(bytes));
}

} // namespace

TEST(Footprint, ETableBytesArithmetic) {
    // Three per-axis tables of (la+1)(lb+1)(la+lb+1) doubles per prim pair.
    EXPECT_EQ(ETableBytes(0, 0, 1), 24);
    EXPECT_EQ(ETableBytes(4, 4, 1), 24 * 5 * 5 * 9);
    EXPECT_EQ(ETableBytes(4, 4, 2), 24 * 5 * 5 * 9 * 2);
    EXPECT_EQ(ETableBytes(6, 6, 1), 24 * 7 * 7 * 13);
    EXPECT_EQ(ETableBytes(0, 4, 3), 24 * 1 * 5 * 5 * 3);
}

TEST(Footprint, TransformBytesArithmetic) {
    // Hermite3DCount(l) = (l+1)(l+2)(l+3)/6, checked independently here.
    EXPECT_EQ(Hermite3DCount(0), 1);
    EXPECT_EQ(Hermite3DCount(2), 10);
    EXPECT_EQ(Hermite3DCount(8), 165);
    EXPECT_EQ(Hermite3DCount(10), 286);
    EXPECT_EQ(Hermite3DCount(12), 455);

    // The design-fact (g,g) pair: spherical, one primitive. nFuncs = 9x9,
    // nHerm = 165, rowPairs = 1: bra 8x81x1x1x165 + ket 8x1x165x81 =
    // 213,840 bytes = the ~214 KB design-fact pair.
    EXPECT_EQ(TransformBytes(4, 4, 81, 1, 1), 213840);
    // (h,h): nFuncs = 121, nHerm = 286.
    EXPECT_EQ(TransformBytes(5, 5, 121, 1, 1), 2 * 8 * 121 * 286);
    // Two rows, two prims: bra grows by rowPairs x nPrimPairs.
    EXPECT_EQ(TransformBytes(0, 0, 1, 4, 2), 8 * 2 * 1 * 1 * (4 + 1));
}

TEST(Footprint, WeightsBytesArithmetic) {
    EXPECT_EQ(WeightsBytes(1, 1), 8);
    EXPECT_EQ(WeightsBytes(4, 3), 8 * 3 * 4);
}

TEST(Footprint, AllSurvivePatternBound) {
    // 8 bytes per surviving upper-triangle pair (ket <= bra).
    EXPECT_EQ(AllSurvivePatternBytes(1), 8);
    EXPECT_EQ(AllSurvivePatternBytes(2), 24);
    EXPECT_EQ(AllSurvivePatternBytes(28), 3248);
    // H2O/STO-3G: 5 shells (the SP shell parses as S + P) -> 15 pairs
    // -> 8 * 15 * 16 / 2.
    EXPECT_EQ(AllSurvivePatternBytes(15), 960);
}

TEST(Footprint, MetricPairStoreBytesArithmetic) {
    // The compact shared-phantom metric store: one MdPairData entry per aux
    // shell plus the unused (0, 0) slot - the phantom-hole tail of the
    // former per-shell phantom layout (~3 nAuxShells^2 / 2 entries) is
    // gone. The 24-water band fixture: 72 S shells.
    EXPECT_EQ(MetricPairStoreBytes(1), 2 * sizeof(MdPairData));
    EXPECT_EQ(MetricPairStoreBytes(72), 73 * sizeof(MdPairData));
    // The former layout's tail at the same shell counts (2 * nAuxShells
    // shells): 3 entries vs 2 * 3 * 3 / 2 = 9 for nAuxShells = 3.
    EXPECT_LT(MetricPairStoreBytes(3), 9 * sizeof(MdPairData));
}

TEST(Footprint, ScratchClamp) {
    // The clamp shrinks the batch so the per-thread arena closes the
    // deficit; a fitting estimate leaves the batch untouched; a deficit
    // larger than the whole arena returns 0 (LightPath mandatory).
    EXPECT_EQ(ClampBatchBytes(1000000, 4, 4500000, 4100000), 900000);
    EXPECT_EQ(ClampBatchBytes(1000000, 4, 4100001, 4100000), 999999);
    EXPECT_EQ(ClampBatchBytes(1000000, 4, 4200000, 4000000), 950000);
    EXPECT_EQ(ClampBatchBytes(1000000, 4, 3000000, 4100000), 1000000);
    EXPECT_EQ(ClampBatchBytes(10, 4, 100, 1), 0);
    EXPECT_EQ(ClampBatchBytes(1000000, 4, 5000000, 1000), 0);
}

TEST(Footprint, ClampAcrossTwoArenas) {
    // The RI-J stack runs TWO sequential batch arenas (the 3c builder's and
    // the nested exchange's), so its clamp spreads the deficit across both:
    // the per-arena shrink is ceil(deficit / (arenaCount * threads)) and the
    // re-estimate (2 * clamped * threads) still closes the deficit. A
    // single-arena clamp on the same deficit would over-refuse the whole
    // band one arena wide.
    EXPECT_EQ(ClampBatchBytes(10, 4, 1040, 1000, 2), 5);
    EXPECT_EQ(ClampBatchBytes(10, 4, 1060, 1000, 2), 2);
    EXPECT_EQ(ClampBatchBytes(10, 4, 1072, 1000, 2), 1);
    EXPECT_EQ(ClampBatchBytes(10, 4, 1073, 1000, 2), 0);
    // The defaulted arenaCount keeps the single-arena semantics.
    EXPECT_EQ(ClampBatchBytes(10, 4, 1020, 1000), 5);
    EXPECT_EQ(ClampBatchBytes(10, 4, 1040, 1000), 0);
}

TEST(Footprint, H2oPairStoreFloorCoversTheEngineAllocation) {
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    // 5 shells (the O SP shell parses as S + P) -> 15 canonical pairs.
    ASSERT_EQ(pairList->pairs.size(), 15);

    // The primitive counts come from the basis set through the same lookup
    // the engine uses (FlattenShells) - ShellInfo itself carries only the
    // contraction-row count.
    auto shells = FlattenShells(*molecule, *basis, *pairList);
    ASSERT_TRUE(shells.has_value());
    ASSERT_EQ(shells->size(), pairList->shells.size());

    auto pairStore = BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value());
    ASSERT_EQ(pairStore->size(), 15);

    const std::size_t predicted = PairStoreBytes(*molecule, *basis, *pairList);
    std::size_t measured = 0;

    for (std::size_t pairIndex = 0; pairIndex < pairStore->size(); ++pairIndex)
    {
        const qcx::integrals::ShellPairIndex& shellPair = pairList->pairs[pairIndex];
        const qcx::integrals::ShellInfo& a = pairList->shells[shellPair.i];
        const qcx::integrals::ShellInfo& b = pairList->shells[shellPair.j];
        const std::size_t nPrimPairs = (*shells)[shellPair.i].contractions.exponents.size() *
                                       (*shells)[shellPair.j].contractions.exponents.size();
        const std::size_t predictedPair = PairStoreBytesPerPair(a.angularMomentum,
                                                                b.angularMomentum,
                                                                a.isSpherical,
                                                                b.isSpherical,
                                                                a.contractionCount,
                                                                b.contractionCount,
                                                                nPrimPairs);
        const std::size_t measuredPair = MeasuredPairBytes(
            (*pairStore)[pairIndex], a.angularMomentum, b.angularMomentum, nPrimPairs);
        EXPECT_GE(predictedPair, measuredPair) << "pair " << pairIndex;
        measured += measuredPair;
    }

    // The never-under floor: the estimate covers the engine's real store.
    EXPECT_GE(predicted, measured);

    // The engine's own function counts agree with the pair list's (the
    // rows the formulas read are the contraction rows).
    for (std::size_t pairIndex = 0; pairIndex < pairStore->size(); ++pairIndex)
    {
        const MdPairData& pair = (*pairStore)[pairIndex];
        EXPECT_EQ(pair.nFuncs, pair.nFuncsA * pair.nFuncsB);
        EXPECT_GE(pair.nFuncsA, 1u);
        EXPECT_GE(pair.nFuncsB, 1u);
        EXPECT_GE(pair.primPairs.capacity(), 1u);
        EXPECT_GE(pair.braTransform.capacity(), 1u);
    }
}

TEST(Footprint, AuxPairDataFloorCoversTheEngineAllocation) {
    // The RI-J aux kets are single-shell pair data (BuildAuxPairData):
    // (la, 0) pairs with one phantom ket row. The formula must cover the
    // engine's aux pair allocation shell by shell.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());

    auto auxShells = FlattenShells(*molecule, *basis, *pairList);
    ASSERT_TRUE(auxShells.has_value());

    for (const qcx::integrals::internal::MdShellInput& shell : *auxShells)
    {
        const std::array<double, 3> center = {shell.cx, shell.cy, shell.cz};
        const MdPairData auxPair = BuildAuxPairData(shell.contractions, center);
        const std::size_t nPrimPairs = shell.contractions.exponents.size();
        const std::size_t predicted = PairStoreBytesPerPair(shell.contractions.angularMomentum,
                                                            0,
                                                            shell.contractions.isSpherical,
                                                            false,
                                                            shell.contractions.rows,
                                                            1,
                                                            nPrimPairs);
        const std::size_t measured =
            MeasuredPairBytes(auxPair, shell.contractions.angularMomentum, 0, nPrimPairs);
        EXPECT_GE(predicted, measured) << "aux shell l = " << shell.contractions.angularMomentum;
        EXPECT_GT(predicted, 0u);
    }
}

TEST(Footprint, ModeSelectionHugeBudgetFastPathReservesExactly) {
    // The direct builder on a generous budget (the exchange term is
    // k-folded - k = team at a fitting budget, k x (2|3) x cap per slot - so
    // the unclamped k = team charge needs four per-thread arenas (scratch +
    // the k x 3 x cap exchange bound at the 3:1 lane ratio), plus a margin
    // for the non-scratch terms, a few tens of KB): the estimate fits
    // unclamped, FastPath is selected with the full slot authorization, and
    // the reservation equals the estimate exactly (the CWA sum of the fired
    // terms).
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    const std::size_t budgetBytes = 4 * DefaultArenaBytes() + std::size_t{1024} * 1024;
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::optional<FockModeInfo>& info = builder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kFastPath);
    EXPECT_EQ(info->budgetBytes, budgetBytes);
    EXPECT_EQ(info->remainingAtDecision, budgetBytes);
    // No clamp: the estimate fits the unclamped arena, and the per-thread
    // scratch is the full arena. The k-folded charge fits exactly at the
    // four-arena budget (the k = team x (1 + 3) cap product equals the four
    // arenas at any team size), so the slot authorization saturates at the
    // team size.
    const std::size_t teamSize = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    EXPECT_EQ(info->concurrentSlots, teamSize);
    // The clamp-origin team read: the fired k is the whole
    // team (k == defaultTeamSize - team-clamped, not budget-clamped), and
    // the record carries the same read the k = min(...) clamped against.
    EXPECT_EQ(info->defaultTeamSize, teamSize);
    EXPECT_EQ(info->maxBatchBytes, FockBuildOptions{}.maxBatchBytes);
    EXPECT_EQ(info->scratchBytes, DefaultArenaBytes());
    // The pattern is the swept count: positive and inside the all-survive
    // bound (15 pairs -> 960 bytes upper bound).
    EXPECT_GT(info->patternBytes, 0u);
    EXPECT_LE(info->patternBytes, AllSurvivePatternBytes(15));
    EXPECT_GT(info->pairStoreBytes, 0u);
    EXPECT_GT(info->structuralBytes, 0u);
    EXPECT_EQ(info->cacheBytes, 0u);
    // The reservation is the full estimate - the exact CWA charge.
    EXPECT_EQ(info->reservedBytes, info->predictedBytes);
    EXPECT_EQ(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_EQ(budget->Remaining(), budgetBytes - info->predictedBytes);
}

TEST(Footprint, ModeSelectionScarceBudgetClampsTheBatch) {
    // A budget of exactly one arena: the per-thread scratch alone needs the
    // full arena, so the non-scratch terms force a clamp - the first rung
    // response - and the clamped estimate must fit the budget.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    const std::size_t budgetBytes = DefaultArenaBytes();
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::optional<FockModeInfo>& info = builder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kFastPath);
    // The clamp fired: the batch cap dropped below the default (any thread
    // count - the estimate exceeds one arena by the non-scratch terms).
    EXPECT_LT(info->maxBatchBytes, FockBuildOptions{}.maxBatchBytes);
    EXPECT_GT(info->maxBatchBytes, 0u);
    // The clamped estimate fits, the scratch is the clamped batch times
    // the team size, and the reservation is exact.
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    EXPECT_LE(info->predictedBytes, budgetBytes);
    EXPECT_EQ(info->scratchBytes, info->maxBatchBytes * threadCount);
    EXPECT_EQ(budget->CommittedBytes(), info->predictedBytes);
}

TEST(Footprint, ModeSelectionTinyBudgetRefusesWithTheLadder) {
    // A one-byte budget: exclusion (i) - the exact-count test fires (for
    // dense H2O every one of the 120 pair-pairs survives, so the counted
    // bound equals the all-survive bound) - refuses with the full
    // four-rung ladder diagnostics and nothing charged.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto budget = WorkspaceBudget::Create(1);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = builder.error().message;
    // The four ladder rungs, by name.
    EXPECT_NE(message.find("direct screened"), std::string::npos);
    EXPECT_NE(message.find("batched/blocked"), std::string::npos);
    EXPECT_NE(message.find("recompute"), std::string::npos);
    EXPECT_NE(message.find("disk"), std::string::npos);
    // Exclusion (i): the counted pattern sentence, no estimate made.
    EXPECT_NE(message.find("counted Schwarz-screened neighbor pattern"), std::string::npos);
    EXPECT_NE(message.find("light rung"), std::string::npos);
    EXPECT_EQ(message.find("footprint estimate is"), std::string::npos);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, ModeSelectionSweepRefusalNamesTheEstimate) {
    // A budget above the all-survive bound (960 bytes for the 15 H2O
    // pairs) but far below the estimate: the sweep runs, the clamp bottoms
    // out, and the refusal carries the estimate sentence - no exclusion
    // clause, the four rungs still named.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto budget = WorkspaceBudget::Create(2000);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = builder.error().message;
    EXPECT_NE(message.find("footprint estimate is"), std::string::npos);
    EXPECT_NE(message.find("direct screened"), std::string::npos);
    EXPECT_NE(message.find("batched/blocked"), std::string::npos);
    EXPECT_NE(message.find("recompute"), std::string::npos);
    EXPECT_NE(message.find("disk"), std::string::npos);
    // No exclusion fired - the estimate sentence replaced the exclusions.
    EXPECT_EQ(message.find("neighbor pattern alone"), std::string::npos);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, ModeSelectionLegacyNullKeepsBitIdentity) {
    // The hard gate: a null budget keeps the legacy path bit-for-bit - the
    // Create-time decision must not perturb the engine's numerics (the
    // SCF-level pins like the H2O/STO-3G kTight -74.96292827 run on the
    // legacy path; this is the builder-level identity).
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value());

    // Serial schedule on both sides (maxParallelChunks = 1, the
    // fock_build_test pin pattern): the gate here is that the decision
    // layer leaves the numerics untouched, so the two legs are compared on
    // one schedule. Since the fixed-order join
    // (internal/fixed_order_reduce.hpp) the chunked schedule is
    // deterministic too, so the serial pin is now a choice of comparison
    // schedule rather than a requirement for bit identity.
    FockBuildOptions legacyOptions;
    legacyOptions.maxParallelChunks = 1;
    auto legacy = DirectJkFockBuilder::Create(*molecule, *basis, *core, legacyOptions);
    ASSERT_TRUE(legacy.has_value()) << legacy.error().message;
    EXPECT_FALSE(legacy->ModeInfo().has_value());
    auto legacyFock = legacy->BuildFock(*densityTensor);
    ASSERT_TRUE(legacyFock.has_value());

    auto budget = WorkspaceBudget::Create(2 * DefaultArenaBytes() + std::size_t{1024} * 1024);
    ASSERT_TRUE(budget.has_value());
    FockBuildOptions budgetOptions;
    budgetOptions.workspaceBudget = &*budget;
    budgetOptions.maxParallelChunks = 1;
    auto budgeted = DirectJkFockBuilder::Create(*molecule, *basis, *core, budgetOptions);
    ASSERT_TRUE(budgeted.has_value()) << budgeted.error().message;
    auto budgetedFock = budgeted->BuildFock(*densityTensor);
    ASSERT_TRUE(budgetedFock.has_value());

    EXPECT_EQ((ToMatrix(*budgetedFock) - ToMatrix(*legacyFock)).norm(), 0.0);
}

// The ClassTableBytes never-under formula (the pair-class root fix):
// 208 B per canonical member quartet (the member, a
// class-pair entry and an orbit entry - each with its vector header -
// the worst case of one class pair per quartet), 72 B per canonical pair
// (the classOfPair entry, the class's member-list entry, the PairClass
// with its vector header), 8 B per function for the shellOfFunction
// support data (with its vector header - the on-demand orbit generation's
// per-function shell index), 16 B per pair-pair for the BuildOrbits
// building transient (the raw mP x mQ quartet list, charged as if
// concurrent with the table), and the four vector headers.
TEST(Footprint, ClassTableBytesArithmetic) {
    // The hand-pinned points: 1 pair / 1 function (1 quartet), 10 pairs /
    // 4 functions (55 quartets), 15 pairs / 5 functions (120 quartets),
    // and the 24-water cluster (7,260 pairs, 168 functions, 26,357,430
    // quartets - the c60-shape class space whose ~6.7 GiB charge the
    // admission gate enforces). The pins are Release-calibrated: the
    // class-table containers carry the tagged allocator, and
    // every std::vector<T, TaggedAllocator<T>> is +8 B over the plain
    // vector (the allocator's tag member; ClassTableBytes is sizeof-based
    // by design, so the formula tracks the layout), exactly the +16 B per
    // member quartet and +8 B per pair the measured delta shows - the
    // Release pins read at the plain-layout values plus that delta, and
    // the MSVC Debug STL adds another 8 B per vector (the _Container_proxy
    // pointer, the same +16/+8 over Release) - the Debug pins below, the
    // larger estimate in the never-under direction.
#ifdef _DEBUG
    EXPECT_EQ(ClassTableBytes(1, 1), 448u);
    EXPECT_EQ(ClassTableBytes(10, 4), 15'808u);
    EXPECT_EQ(ClassTableBytes(15, 5), 33'856u);
    EXPECT_EQ(ClassTableBytes(7260, 168), 7'169'745'120u);
#else
    EXPECT_EQ(ClassTableBytes(1, 1), 424u);
    EXPECT_EQ(ClassTableBytes(10, 4), 14'848u);
    EXPECT_EQ(ClassTableBytes(15, 5), 31'816u);
    EXPECT_EQ(ClassTableBytes(7260, 168), 6'747'968'160u);
#endif
}

// The class-table admission gate (the pair-class root fix):
// the class path's Create-time table is charged against the budget's
// remaining bytes, and a table that cannot fit DISENGAGES the class path -
// the plain screened path builds instead of the pre-fix 0xC0000409 death.
// The class path's materialization charge at the 24-water cluster, via
// CountClassMaterialization on the fixture's reduction and the production
// Schwarz bounds, through
// internal::ClassMaterializationBytes. 69,696 ordered kept class pairs and
// 129,600 ordered member quartets over 4,956 classes. The Release pin; the
// Debug STL's per-vector _Container_proxy pointer raises it by a few MiB (the
// band is asserted there).
constexpr std::size_t kMaterializationChargeBytes = 28'898'352;

// The class-table admission gate (the pair-class root fix): the class path's
// Create-time charge is the MINIMUM of two never-under upper bounds of the
// same allocation - the eager singleton-orbit ceiling
// (internal::ClassTableBytes, the only shape Create-time-safe without the
// class decomposition) and the bound of the shape the path actually reaches
// (internal::ClassMaterializationBytes from CountClassMaterialization, which
// counts the Schwarz-screened class pairs and their member mass by a sorted
// sweep and materializes nothing). The 24-water cluster (7,260 pairs, the
// c60-shape class space) pins the gap the fix closes: the ceiling charges
// 6,747,968,160 B = 6.28 GiB for a table whose live peak is 8,375,712 B =
// 7.99 MiB (measured through the class_table allocation family), and the
// materialization bound is 28,898,352 B = 27.6 MiB - 3.45x the live table,
// so never-under still holds. A 2 GiB budget therefore ENGAGES the class
// path where the ceiling alone disengaged it. The never-under property
// survives: a budget below the materialization charge still disengages, and
// the charge can never engage the class path LESS often than the ceiling did.
TEST(Footprint, ClassTableAdmissionGateDisengagesWhenTheTableCannotFit) {
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    ASSERT_EQ(pairList->functionCount, 168u);
    ASSERT_EQ(pairList->pairs.size(), 7260u);

    // The hand-built Cs reduction: the identity plus the x-mirror. The
    // flattened function layout follows the canonical atom order
    // (Molecule::Create renumbers by Z, then position): the 2W H atoms
    // come first - H Wk-x at function k, H Wk+x at function waters+k
    // (functions 0..47) - then per water the O s1 (48+5k), O s2 (49+5k),
    // and the O p shell (50+5k..52+5k in the codebase's Cartesian p
    // order py, pz, px - the m-ascending convention the
    // WaterC2vClassPathMatchesPlain sign rows pin). The mirror swaps the
    // two H functions (the cluster's H Wk-x / H Wk+x monomers) and flips
    // the O p_x (52+5k, sign -1): the sign the expansion pattern needs,
    // the WaterC2vClassPathMatchesPlain convention. The reduction IS a
    // true symmetry of the cluster - the engaged leg's class-vs-plain
    // equivalence is only defined for one - and BuildPairClasses
    // validates the rows, indices and signs (the tables must hold
    // groupOrder rows).
    const std::size_t nFunctions = pairList->functionCount;
    std::vector<std::size_t> identity(nFunctions);
    std::vector<std::size_t> swap(nFunctions);
    std::vector<int> signsIdentity(nFunctions, 1);
    std::vector<int> signsMirror(nFunctions, 1);

    for (std::size_t i = 0; i < nFunctions; ++i)
    {
        identity[i] = i;
        swap[i] = i;
    }

    for (std::size_t k = 0; k < 24; ++k)
    {
        swap[k] = k + 24;
        swap[k + 24] = k;
        signsMirror[52 + 5 * k] = -1;
    }

    const qcx::integrals::SymmetryReduction reduction{
        {std::move(identity), std::move(swap)},
        {std::move(signsIdentity), std::move(signsMirror)},
        2,
        false};

    // The plain path's Fock: the reference every class-path leg below is
    // compared against - created without the reduction and without a budget.
    // The certified fp32 lane is OFF on every leg (fp64 everywhere): the
    // class-vs-plain comparison must not mix in the lane's per-quartet
    // precision routing (the WaterC2vClassPathMatchesPlain pattern).
    FockBuildOptions plainOptions;
    plainOptions.useCertifiedMixedPrecision = false;
    plainOptions.maxBatchBytes = std::size_t{2} * 1024 * 1024;
    plainOptions.maxParallelChunks = 1;
    auto plain = DirectJkFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(nFunctions);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value());
    auto plainFock = plain->BuildFock(*densityTensor);
    ASSERT_TRUE(plainFock.has_value());

    // The 2 GiB leg: the eager ceiling (~6.28 GiB) cannot fit this budget -
    // which is why the pre-fix gate DISENGAGED here - but the materialization
    // charge (~27.6 MiB, the production counter's bound of what the path
    // actually allocates) clears it by two orders of magnitude, so the class
    // path ENGAGES. This leg is the fix's contract: the symmetry is no longer
    // lost to an accounting bound that describes a table the path never
    // builds. The class path's own equivalence to the plain path is the
    // documented 1e-12 (WaterC2vClassPathMatchesPlain), not bit identity -
    // the two paths contract in a different order.
    auto twoGiBBudget = WorkspaceBudget::Create(2ull * 1024 * 1024 * 1024);
    ASSERT_TRUE(twoGiBBudget.has_value());
    FockBuildOptions twoGiBOptions;
    twoGiBOptions.workspaceBudget = &*twoGiBBudget;
    twoGiBOptions.symmetryReduction = &reduction;
    twoGiBOptions.useCertifiedMixedPrecision = false;
    twoGiBOptions.maxBatchBytes = std::size_t{2} * 1024 * 1024;
    twoGiBOptions.maxParallelChunks = 1;
    auto twoGiB = DirectJkFockBuilder::Create(*molecule, *basis, *core, twoGiBOptions);
    ASSERT_TRUE(twoGiB.has_value()) << twoGiB.error().message;
    const std::optional<FockModeInfo>& twoGiBInfo = twoGiB->ModeInfo();
    ASSERT_TRUE(twoGiBInfo.has_value());
    EXPECT_FALSE(twoGiBInfo->classPathDisengaged);
#ifdef _DEBUG
    // The MSVC Debug STL adds 8 B per vector to every charged struct (the
    // _Container_proxy pointer - ClassTableBytesArithmetic), so the Debug
    // charge sits a few MiB above the Release pin; the band is what is
    // asserted on that calibration.
    EXPECT_GT(twoGiBInfo->classTableBytes, kMaterializationChargeBytes);
    EXPECT_LT(twoGiBInfo->classTableBytes, 40'000'000u);
#else
    EXPECT_EQ(twoGiBInfo->classTableBytes, kMaterializationChargeBytes);
#endif
    auto twoGiBFock = twoGiB->BuildFock(*densityTensor);
    ASSERT_TRUE(twoGiBFock.has_value());
    EXPECT_NEAR((ToMatrix(*twoGiBFock) - ToMatrix(*plainFock)).norm(), 0.0, 1e-12);

    // The never-under leg: a budget one MiB BELOW the materialization charge
    // still disengages the class path (nothing charged for the table). The
    // bounded shape is the point: the gate must never materialize a table it
    // has not first bounded, and it must fail clean - the plain path
    // completing inside the leftover budget or the ladder refusing are both
    // legitimate (this fixture's plain working set is the same order as the
    // charge), and neither may be a charged table or a crash.
    auto belowChargeBudget = WorkspaceBudget::Create(
        static_cast<std::size_t>(kMaterializationChargeBytes) - (std::size_t{1} << 20));
    ASSERT_TRUE(belowChargeBudget.has_value());
    FockBuildOptions belowChargeOptions;
    belowChargeOptions.workspaceBudget = &*belowChargeBudget;
    belowChargeOptions.symmetryReduction = &reduction;
    belowChargeOptions.useCertifiedMixedPrecision = false;
    belowChargeOptions.maxBatchBytes = std::size_t{2} * 1024 * 1024;
    belowChargeOptions.maxParallelChunks = 1;
    auto belowCharge = DirectJkFockBuilder::Create(*molecule, *basis, *core, belowChargeOptions);
    std::printf("    the below-charge budget: %s\n",
                belowCharge.has_value() ? "the plain path fitted - disengaged"
                                        : belowCharge.error().message.c_str());
    std::fflush(stdout);

    if (belowCharge.has_value())
    {
        const std::optional<FockModeInfo>& belowChargeInfo = belowCharge->ModeInfo();
        ASSERT_TRUE(belowChargeInfo.has_value());
        EXPECT_TRUE(belowChargeInfo->classPathDisengaged);
        EXPECT_EQ(belowChargeInfo->classTableBytes, 0u);
        auto belowChargeFock = belowCharge->BuildFock(*densityTensor);
        ASSERT_TRUE(belowChargeFock.has_value());
        EXPECT_EQ((ToMatrix(*belowChargeFock) - ToMatrix(*plainFock)).norm(), 0.0);
    }

    // The cross-budget pin: the charge is the path's bound, not the budget's,
    // so a 4x larger budget charges the identical bytes and engages the same
    // class path - the contraction runs the FULL class path (the on-demand
    // orbit generation per screened class pair - the root restructure's
    // laziness, exercised end to end at the c60-shape class space) and
    // reproduces the plain path's Fock element-wise: the screening drops
    // exactly the pairs that cannot contain a screened member quartet, and
    // the generated orbit sets are the eager table's.
    auto engagedBudget = WorkspaceBudget::Create(8ull * 1024 * 1024 * 1024);
    ASSERT_TRUE(engagedBudget.has_value());
    FockBuildOptions engagedOptions;
    engagedOptions.workspaceBudget = &*engagedBudget;
    engagedOptions.symmetryReduction = &reduction;
    engagedOptions.useCertifiedMixedPrecision = false;
    engagedOptions.maxBatchBytes = std::size_t{2} * 1024 * 1024;
    engagedOptions.maxParallelChunks = 1;
    auto engaged = DirectJkFockBuilder::Create(*molecule, *basis, *core, engagedOptions);
    ASSERT_TRUE(engaged.has_value()) << engaged.error().message;
    const std::optional<FockModeInfo>& engagedInfo = engaged->ModeInfo();
    ASSERT_TRUE(engagedInfo.has_value());
    EXPECT_FALSE(engagedInfo->classPathDisengaged);
    EXPECT_EQ(engagedInfo->classTableBytes, twoGiBInfo->classTableBytes);
    auto engagedFock = engaged->BuildFock(*densityTensor);
    ASSERT_TRUE(engagedFock.has_value());

    // The fast-path class leg: the same class path WITHOUT the budget (no
    // light mode) at the same scale - the on-demand orbit generation and
    // the class contraction over the FULL screened set, the C2v fixture's
    // regime at the cluster's scale. The engaged leg's chunked class pass
    // and this full-list class pass must both reproduce the plain Fock.
    FockBuildOptions fastClassOptions;
    fastClassOptions.symmetryReduction = &reduction;
    fastClassOptions.useCertifiedMixedPrecision = false;
    fastClassOptions.maxBatchBytes = std::size_t{2} * 1024 * 1024;
    fastClassOptions.maxParallelChunks = 1;
    auto fastClass = DirectJkFockBuilder::Create(*molecule, *basis, *core, fastClassOptions);
    ASSERT_TRUE(fastClass.has_value()) << fastClass.error().message;
    auto fastClassFock = fastClass->BuildFock(*densityTensor);
    ASSERT_TRUE(fastClassFock.has_value());

    const Eigen::MatrixXd plainMatrix = ToMatrix(*plainFock);
    const Eigen::MatrixXd engagedMatrix = ToMatrix(*engagedFock);
    const Eigen::MatrixXd fastClassMatrix = ToMatrix(*fastClassFock);

    for (std::size_t i = 0; i < nFunctions; ++i)
    {
        for (std::size_t j = 0; j < nFunctions; ++j)
        {
            EXPECT_NEAR(engagedMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        plainMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        1e-12)
                << "element (" << i << "," << j << ")";

            EXPECT_NEAR(fastClassMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        plainMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)),
                        1e-12)
                << "fast-class element (" << i << "," << j << ")";
        }
    }
}

// The null-budget escape-hatch host-RAM guard: the legacy no-budget path
// (workspaceBudget
// null - memory_cap_gib = 0, the documented opt-out) engages the class
// path unconditionally - the budget-path admission gate never runs, so a
// c60-shaped request would still die as a raw machine-OOM allocation
// growing the canonical member-quartet space. The guard charges the same
// never-under estimate (ClassTableBytes) against the node's TOTAL physical
// RAM and refuses CLEANLY (kOutOfMemory) when the table cannot fit. The
// 60-water cluster IS the c60 STO-3G shell shape (300 shells, 45,150
// shell pairs, 420 functions): the never-under charge ~228 GiB exceeds the
// total physical RAM of any machine this suite runs on
// (up to ~128 GiB CI runners), so the guard fires deterministically. The
// refusal is Create-time only - BuildFock is never reached, no Fock is
// built.
TEST(Footprint, NullBudgetClassTableRefusesWhenTheChargeExceedsHostRam) {
    auto molecule = MakeWaterCluster(60);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    ASSERT_EQ(pairList->functionCount, 420u);
    ASSERT_EQ(pairList->pairs.size(), 45150u);

    // The Cs reduction, built for 60 waters on the canonical atom order
    // (Molecule::Create renumbers by Z, then position) exactly like the
    // admission-gate test's 24-water construction: the 120 H atoms first -
    // H Wk-x at function k, H Wk+x at function waters+k - then per water
    // the O s1 (120+5k), O s2 (121+5k) and the O p shell (122+5k..124+5k
    // in the m-ascending order py, pz, px): the mirror swaps the two H
    // functions and flips the O p_x (124+5k, sign -1). The rows are never
    // consulted - the guard fires before BuildPairClasses - but the
    // reduction is the same true symmetry the admission-gate test builds,
    // so the fixture stays honest.
    const std::size_t nFunctions = pairList->functionCount;
    const std::size_t nWaters = 60;
    std::vector<std::size_t> identity(nFunctions);
    std::vector<std::size_t> swap(nFunctions);
    std::vector<int> signsIdentity(nFunctions, 1);
    std::vector<int> signsMirror(nFunctions, 1);

    for (std::size_t i = 0; i < nFunctions; ++i)
    {
        identity[i] = i;
        swap[i] = i;
    }

    for (std::size_t k = 0; k < nWaters; ++k)
    {
        swap[k] = k + nWaters;
        swap[k + nWaters] = k;
        signsMirror[2 * nWaters + 4 + 5 * k] = -1;
    }

    const qcx::integrals::SymmetryReduction reduction{
        {std::move(identity), std::move(swap)},
        {std::move(signsIdentity), std::move(signsMirror)},
        2,
        false};

    // No budget - the escape hatch - and a groupOrder-2 reduction: the
    // class path is requested, the admission gate never runs, and the
    // host-RAM guard is the only gate left. The ~228 GiB never-under
    // charge cannot fit any real machine, so Create must refuse cleanly
    // (kOutOfMemory) instead of dying on a raw allocation.
    FockBuildOptions options;
    options.symmetryReduction = &reduction;
    auto result = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, qcx::ErrorCode::kOutOfMemory);
    EXPECT_NE(result.error().message.find("physical RAM"), std::string::npos);
}

TEST(Footprint, ModeSelectionIsDeterministic) {
    // Two fresh budgets, same options: the same decision record and the
    // same Fock - the mode selection is a pure function of (options,
    // budget state at Create).
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value());

    const std::size_t budgetBytes = 2 * DefaultArenaBytes() + std::size_t{1024} * 1024;

    auto makeRecord = [&]() -> std::pair<std::optional<FockModeInfo>, Eigen::MatrixXd> {
        auto budget = WorkspaceBudget::Create(budgetBytes);
        EXPECT_TRUE(budget.has_value());
        FockBuildOptions options;
        options.workspaceBudget = &*budget;
        // Serial schedule: bit-identical Focks (see the bit-identity test).
        options.maxParallelChunks = 1;
        auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
        EXPECT_TRUE(builder.has_value()) << builder.error().message;
        auto fock = builder->BuildFock(*densityTensor);
        EXPECT_TRUE(fock.has_value());
        return {builder->ModeInfo(), ToMatrix(*fock)};
    };

    const auto [firstInfo, firstFock] = makeRecord();
    const auto [secondInfo, secondFock] = makeRecord();
    ASSERT_TRUE(firstInfo.has_value());
    ASSERT_TRUE(secondInfo.has_value());

    EXPECT_EQ(firstInfo->mode, secondInfo->mode);
    EXPECT_EQ(firstInfo->predictedBytes, secondInfo->predictedBytes);
    EXPECT_EQ(firstInfo->reservedBytes, secondInfo->reservedBytes);
    EXPECT_EQ(firstInfo->maxBatchBytes, secondInfo->maxBatchBytes);
    EXPECT_EQ(firstInfo->pairStoreBytes, secondInfo->pairStoreBytes);
    EXPECT_EQ(firstInfo->patternBytes, secondInfo->patternBytes);
    EXPECT_EQ(firstInfo->scratchBytes, secondInfo->scratchBytes);
    EXPECT_EQ(firstInfo->structuralBytes, secondInfo->structuralBytes);
    EXPECT_EQ((firstFock - secondFock).norm(), 0.0);
}

TEST(Footprint, RiModeSelectionTensorExclusionEngagesTheLightRung) {
    // Exclusion (ii): a budget above the all-survive bound (960) but below
    // the (uv|P) tensor term alone (8 * 7^2 * 12 = 4704 for H2O/STO-3G with
    // the 12-function tiny aux). The fast path is skipped a priori and the
    // light rung's own estimate runs - at H2O scale it cannot fit either
    // (the 512 MB x threads batch arenas dominate the estimate, so the
    // clamp bottoms out), and Create refuses with the light-rung
    // diagnostic, nothing charged. The rung CONSTRUCTS only where the
    // tensor dominates the residual quadratic terms (nt84-class); the
    // unit-scale construction is validated through the forcing seam in
    // RiEngineTest.LightRungMatchesMaterializedWithinThePresetBudget.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto budget = WorkspaceBudget::Create(2000);
    ASSERT_TRUE(budget.has_value());

    RiEngineOptions options;
    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_FALSE(ri.has_value());
    EXPECT_EQ(ri.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = ri.error().message;
    EXPECT_NE(message.find("light rung cannot fit"), std::string::npos);
    EXPECT_NE(message.find("light-rung Create-time footprint estimate is"), std::string::npos);
    EXPECT_NE(message.find("(uv|P) tensor alone exceeds"), std::string::npos);
    EXPECT_NE(message.find("direct screened"), std::string::npos);
    EXPECT_NE(message.find("batched/blocked"), std::string::npos);
    EXPECT_NE(message.find("recompute"), std::string::npos);
    EXPECT_NE(message.find("disk"), std::string::npos);
    EXPECT_NE(message.find("blocked-metric rung"), std::string::npos);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, RiModeSelectionReservesTheFullStack) {
    // The RI-J nested reservation: the outer Create charges the estimate
    // minus the exchange half, and the nested direct-exchange Create
    // charges its own estimate - the stack total is the full estimate and
    // the exchange half's scratch mirrors its clamped batch.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    // Three arenas: one for the RI 3c scratch, one for the exchange
    // scratch, one of slack for the stores - no clamp on either half.
    const std::size_t budgetBytes = 3 * DefaultArenaBytes() + std::size_t{1024} * 1024;
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    RiEngineOptions options;
    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    const std::optional<FockModeInfo>& info = ri->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kFastPath);
    // The (uv|P) tensor term: 8 * n^2 * nAuxFuncs = 8 * 49 * 12 - the
    // kTinyAux text applies to ALL three atoms of H2O (O gets S+P, each H
    // gets S+P), so the aux basis has 1+3+1+3+1+3 = 12 functions, not 8.
    EXPECT_EQ(info->tensorBytes, 4704);
    EXPECT_EQ(info->riMatrixBytes, 4704);
    EXPECT_GT(info->taskListBytes, 0u);
    EXPECT_GT(info->metricBytes, 0u);
    EXPECT_GT(info->orbitalAuxBytes, 0u);
    // The exchange half is charged at ITS Create, inside the outer one.
    // That nested decision fires k = team into its
    // post-reservation band, so the stack's committed charge rides at or
    // above the k = 1 estimate (predicted) and up to the budget edge - the
    // saturated fixed point claims the three-arena slack and its own
    // in-band cap re-derivation holds the CWA. Never-under is the honest
    // form: the k-fold excess is platform-arithmetic boundary-sensitive,
    // and the Debug rounding lands the total exactly ON the estimate on
    // the clang leg - the committed == predicted equality is a legitimate
    // boundary, not a violation.
    EXPECT_GT(info->exchangeBytes, 0u);
    EXPECT_EQ(info->exchangeScratchBytes, info->scratchBytes);
    EXPECT_EQ(info->reservedBytes, info->predictedBytes - info->exchangeBytes);
    EXPECT_GE(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(budget->CommittedBytes(), budgetBytes);
    EXPECT_LE(info->predictedBytes, budgetBytes);
}

TEST(Footprint, RiAuxStoreModelsThePerShellKets) {
    // The F2 shape: the aux term is the PER-SHELL ket
    // store - one BuildAuxPairData per aux shell (BuildAuxMetric and
    // BuildRiTensor of ri_engine.cpp) - never the canonical aux PAIR list
    // (the nAuxShells^2/2-pair sum over-counted by the 10^3-10^4 factor the
    // audit measured and dominated the RI-J fit band where the tensor is
    // marginal). The pin re-derives the expected sum shell by shell with the
    // engine's phantom-ket parameters (l = 0 bra, one row, the aux shell's
    // own primitives) and checks it against both auxStoreBytes and the
    // metric's filled-pair term (one shared ground truth).
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());
    // kTinyAux spans all three atoms: O S+P and two H S+P - 6 aux shells,
    // 12 functions.
    ASSERT_EQ(auxPairList->shells.size(), 6u);
    ASSERT_EQ(auxPairList->functionCount, 12u);

    std::size_t perShellSum = 0;

    for (const qcx::integrals::ShellInfo& shell : auxPairList->shells)
    {
        perShellSum += PairStoreBytesPerPair(
            0,
            shell.angularMomentum,
            false,
            shell.isSpherical,
            1,
            shell.contractionCount,
            qcx::integrals::internal::ShellPrimitiveCount(*molecule, *aux, shell));
    }

    const qcx::integrals::internal::RiFootprintTerms terms = qcx::integrals::internal::RiFootprint(
        *molecule, *basis, *aux, *pairList, *auxPairList, FockBuildOptions{}.maxBatchBytes, 1);
    EXPECT_EQ(terms.auxStoreBytes, perShellSum);
    // The wrong-shape term: the canonical aux pair store sums 21 full pair
    // stores - the far larger sum the metric used to charge before F2.
    EXPECT_LT(terms.auxStoreBytes,
              qcx::integrals::internal::PairStoreBytes(*molecule, *aux, *auxPairList));
    // The metric's filled (phantom, P) pairs are the SAME per-shell ket
    // store (the shared-phantom metric, ri_engine.cpp): the metric term
    // is the filled-pair store plus the metric/eigen-class peak - the
    // three live nAuxFuncs^2 matrices of the unblocked path (the
    // TensorToEigen input retained through the State eigenvector copy,
    // the solver's m_eivec and the State copy; the solver holds exactly
    // one working matrix).
    EXPECT_EQ(terms.metricBytes,
              MetricPairStoreBytes(6) + std::size_t{3} * 8 * 12 * 12 + perShellSum);
}

TEST(Footprint, RiFirstIterationBytesArithmetic) {
    // The estimate terms: the first
    // Fock iteration's tensor-class allocations, folded into
    // RiFootprintTerms::Total so the reservation covers the per-iteration
    // peak. Fast: the riMatrix.transpose() copy is exactly the tensorBytes
    // class again (8 n^2 nAuxFuncs) plus the small-vector allowance
    // (8 nAuxFuncs^2 + 48 n^2 + 40 nAuxFuncs - the dVec/fock/dMatrix and j
    // product copies plus the returned Fock tensor, the nAux-class v/w
    // vectors with slack, and a conservative w-step allowance; the
    // metric-eigenvector products are a lazy Eigen view, never
    // materialized). Light: the AssembleRiBatches recompute slice at a
    // 104 B/task envelope over the SCREENED task count - the counted
    // counted treatment, the list the per-iteration batch loops iterate
    // (the Item + computed + batch-task copies, 96 B verified against the
    // struct sizes below); a no-count call charges the unconditional dense
    // grid, the never-under reference - plus the batch-capped values
    // buffer and the reduced small-vector allowance (8 nAuxFuncs^2 +
    // 32 n^2 - no dMatrix/jColumn on the light rung, the products run
    // batch-at-a-time). The retained-list term closes the
    // footprint.hpp:355-357 reconciliation item.
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());

    // H2O/STO-3G: n = 7 functions, 3 shells; kTinyAux on all three atoms:
    // 6 shells, 12 functions.
    ASSERT_EQ(pairList->functionCount, 7u);
    ASSERT_EQ(auxPairList->functionCount, 12u);
    ASSERT_EQ(auxPairList->shells.size(), 6u);
    const std::size_t n = pairList->functionCount;
    const std::size_t nAuxFuncs = auxPairList->functionCount;
    const std::size_t nOrbitalPairs = pairList->pairs.size();
    const std::size_t nAuxShells = auxPairList->shells.size();
    const std::size_t batch = FockBuildOptions{}.maxBatchBytes;

    // The 104 B/task envelope vs the allocation-site struct sizes: Item
    // (RiTask + 2 ints + 4 size_ts, 56 B),
    // the computed RiTask copy (16 B) and the per-batch MdQuartetTask
    // (24 B) - 96 <= 104, the conservative full-count envelope.
    struct MirrorItem {
        qcx::integrals::internal::RiTask task;
        int lBra;
        int lKet;
        std::size_t braRowPairs;
        std::size_t outputSize;
        std::size_t convertSize;
        std::size_t scratchSize;
    };

    EXPECT_EQ(sizeof(qcx::integrals::internal::RiTask), 16u);
    EXPECT_EQ(sizeof(MirrorItem), 56u);
    EXPECT_EQ(sizeof(qcx::integrals::internal::MdQuartetTask), 24u);
    EXPECT_LE(sizeof(MirrorItem) + sizeof(qcx::integrals::internal::RiTask) +
                  sizeof(qcx::integrals::internal::MdQuartetTask),
              104u);

    const qcx::integrals::internal::RiFirstIterationTerms fast =
        qcx::integrals::internal::RiFirstIterationBytes(*pairList, *auxPairList, batch, false);
    EXPECT_EQ(fast.transposeBytes, 8 * n * n * nAuxFuncs);
    EXPECT_EQ(fast.smallVectorsBytes, 8 * nAuxFuncs * nAuxFuncs + 48 * n * n + 40 * nAuxFuncs);
    EXPECT_EQ(fast.sliceBytes, 0u);
    EXPECT_EQ(fast.valuesBufferBytes, 0u);

    const qcx::integrals::internal::RiFirstIterationTerms light =
        qcx::integrals::internal::RiFirstIterationBytes(*pairList, *auxPairList, batch, true);
    EXPECT_EQ(light.sliceBytes, 104 * nOrbitalPairs * nAuxShells);
    EXPECT_EQ(light.valuesBufferBytes, batch);
    EXPECT_EQ(light.smallVectorsBytes, 8 * nAuxFuncs * nAuxFuncs + 32 * n * n);
    EXPECT_EQ(light.transposeBytes, 0u);

    // The counted treatment on the slice (the direct-call pin):
    // a caller-supplied screened count charges exactly 104 B per surviving
    // task - the per-iteration batch loops iterate that list, so the charge
    // equals the realized slice. The no-count dense form above and this one
    // differ by exactly 104 B per dropped grid cell; everything else is
    // count-inert.
    const std::size_t screenedTaskCount = nOrbitalPairs * nAuxShells - 7;
    const qcx::integrals::internal::RiFirstIterationTerms screenedLight =
        qcx::integrals::internal::RiFirstIterationBytes(
            *pairList, *auxPairList, batch, true, screenedTaskCount);
    EXPECT_EQ(screenedLight.sliceBytes, 104 * screenedTaskCount);
    EXPECT_EQ(light.sliceBytes - screenedLight.sliceBytes,
              104 * (nOrbitalPairs * nAuxShells - screenedTaskCount));
    EXPECT_EQ(screenedLight.valuesBufferBytes, light.valuesBufferBytes);
    EXPECT_EQ(screenedLight.smallVectorsBytes, light.smallVectorsBytes);
    EXPECT_EQ(screenedLight.transposeBytes, 0u);
    EXPECT_EQ(screenedLight.transposeBytes + screenedLight.sliceBytes,
              screenedLight.Total() - screenedLight.valuesBufferBytes -
                  screenedLight.smallVectorsBytes);

    // The folded terms inside RiFootprint: the fast rung's transpose is
    // exactly the tensorBytes class; the light rung carries the slice and
    // the values buffer instead; the retained-list term is charged on both
    // rungs (16 B per pair + 40 B per shell over BOTH lists) as a
    // worst case - the lists are retained only via the light payload (the
    // heap-retention note).
    const qcx::integrals::internal::RiFootprintTerms fastTerms =
        qcx::integrals::internal::RiFootprint(
            *molecule, *basis, *aux, *pairList, *auxPairList, batch, 1);
    EXPECT_EQ(fastTerms.transposeBytes, fastTerms.tensorBytes);
    EXPECT_EQ(fastTerms.sliceBytes, 0u);
    EXPECT_EQ(fastTerms.valuesBufferBytes, 0u);
    EXPECT_EQ(fastTerms.smallVectorsBytes, fast.smallVectorsBytes);

    const qcx::integrals::internal::RiFootprintTerms lightTerms =
        qcx::integrals::internal::RiFootprint(
            *molecule, *basis, *aux, *pairList, *auxPairList, batch, 1, true);
    EXPECT_EQ(lightTerms.transposeBytes, 0u);
    EXPECT_EQ(lightTerms.sliceBytes, light.sliceBytes);
    EXPECT_EQ(lightTerms.valuesBufferBytes, batch);
    EXPECT_EQ(lightTerms.smallVectorsBytes, light.smallVectorsBytes);

    // The count forwards through RiFootprint into the first-iteration fold
    // (the slice fix): the light rung's slice charges 104 B per
    // surviving task and the task list 16 B per surviving task, so the
    // counted total sits exactly 120 B per dropped grid cell under the
    // no-count dense form above - the delta the item-3 razor budgets in the
    // engine-path tests below move with.
    const qcx::integrals::internal::RiFootprintTerms screenedLightTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              1,
                                              true,
                                              false,
                                              screenedTaskCount);
    EXPECT_EQ(screenedLightTerms.sliceBytes, 104 * screenedTaskCount);
    EXPECT_EQ(screenedLightTerms.taskListBytes, screenedTaskCount * 2 * sizeof(std::size_t));
    EXPECT_EQ(lightTerms.Total() - screenedLightTerms.Total(),
              120 * (nOrbitalPairs * nAuxShells - screenedTaskCount));

    const std::size_t retainedListBytes =
        (pairList->pairs.size() + auxPairList->pairs.size()) *
            sizeof(qcx::integrals::ShellPairIndex) +
        (pairList->shells.size() + auxPairList->shells.size()) * sizeof(qcx::integrals::ShellInfo);
    EXPECT_EQ(fastTerms.retainedListBytes, retainedListBytes);
    EXPECT_EQ(lightTerms.retainedListBytes, retainedListBytes);

    // Total() folds the first-iteration terms in - the reservation carries
    // the full per-iteration peak, one number.
    EXPECT_EQ(fastTerms.Total(),
              fastTerms.orbitalStoreBytes + fastTerms.auxStoreBytes + fastTerms.taskListBytes +
                  fastTerms.tensorBytes + fastTerms.riMatrixBytes + fastTerms.metricBytes +
                  fastTerms.scratchBytes + fastTerms.transposeBytes + fastTerms.sliceBytes +
                  fastTerms.valuesBufferBytes + fastTerms.smallVectorsBytes +
                  fastTerms.retainedListBytes + fastTerms.schwarzSweepBytes);
    EXPECT_EQ(lightTerms.Total(),
              lightTerms.orbitalStoreBytes + lightTerms.auxStoreBytes + lightTerms.taskListBytes +
                  lightTerms.tensorBytes + lightTerms.riMatrixBytes + lightTerms.metricBytes +
                  lightTerms.scratchBytes + lightTerms.transposeBytes + lightTerms.sliceBytes +
                  lightTerms.valuesBufferBytes + lightTerms.smallVectorsBytes +
                  lightTerms.retainedListBytes + lightTerms.schwarzSweepBytes);
}

TEST(Footprint, RiFirstIterationFiresTheLightRung) {
    // The firing-rule band on the 24-water
    // kWaterSto3g family: a budget where the fast rung's FOLDED
    // first-iteration total cannot fit - the clamp-ineligible mass
    // (transpose + small vectors + the exchange term) exceeds the
    // two-arena capacity, so the clamp's refusal region binds - while the
    // light rung fits and reserves the folded total. The clamp's refusal
    // selects the light rung in the same inputs the floor check was
    // drafted for; the check itself proved decision-inert (the clamp's
    // refusal region strictly contains the floor's) and
    // was deleted, so this band documents the folded-total geometry the
    // clamp actually enforces.
    //
    // The band's geometry (derived, 2026-08-30): the fast clamp's fit
    // condition cancels the batch - clamped > 0 iff remaining >
    // fastTotal - 2T(batch-1) - so at ANY budget below the folded total's
    // clamp-fit zone today's clamp cannot absorb the NEW total; the "today
    // fast fires" property must be pinned against the OLD total (the
    // pre-fix estimate, without the first-iteration terms). The razor's
    // edge therefore sits one clamped arena under the old total: the
    // budget is oldTotal - threadCount x (batch - 1), the midpoint of the
    // today-clamp zone [oldTotal - 2T(batch-1), oldTotal]. The
    // literal 24-water kTinyOrbitalSOnly family cannot host that budget
    // (the light rung's own clamp fit needs budget >= lightTotal -
    // 2T(batch-1), which the s-only gap between the rungs' clamp-ineligible
    // masses ~ 2 MB cannot clear at team size 4) - the STO-3G orbital's
    // 5-function O
    // shells widen the gap to ~ 320 MB (2 x tensorBytes + riMatrixBytes -
    // sliceBytes), a documented deviation from the literal family.
    // The aux stays d-max (l <= 2), so the pin runs on the CI Lmax=2
    // builds.
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kRiBandAux);
    ASSERT_TRUE(aux.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());
    // 24 waters x 5 shells per water (O S, the SP parsing as separate
    // S and P shells, two H S) = 120 shells, 7 functions per water
    // (O 5: S + spherical SP; two H S) = 168 functions, 7260 canonical
    // orbital pairs; the 3-shell aux: 24 x 3 + 48 x 3 = 216 shells,
    // 648 functions.
    ASSERT_EQ(pairList->functionCount, 168u);
    ASSERT_EQ(pairList->pairs.size(), 7260u);
    ASSERT_EQ(auxPairList->shells.size(), 216u);
    ASSERT_EQ(auxPairList->functionCount, 648u);

    RiEngineOptions options;
    const std::size_t batch = std::size_t{2} * 1024 * 1024;
    options.maxBatchBytes = batch;

    // The counted sweep, exactly as the engine's decision block runs it
    // (ri_engine.cpp:1036-1047) - the exchange half's estimate.
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value());
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> indices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, offsets, indices);

    // The screened task-grid count, exactly as the engine's decision block
    // computes it: the footprint's taskListBytes
    // charges the surviving (orbital pair, aux shell) cells x 16 B, and
    // this band's delta-cancellation holds only while the recomputed terms
    // carry the SAME count the engine charges (the razor budgets and the
    // fired-term equalities below all ride these totals).
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value());
    std::vector<double> auxShellBounds(auxPairList->shells.size());

    for (std::size_t shell = 0; shell < auxPairList->shells.size(); ++shell)
    {
        auxShellBounds[shell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(shell, shell, *auxPairList)];
    }

    const std::size_t screenedTaskCount = qcx::integrals::internal::CountSchwarzSurvivingRiTasks(
        *schwarz, auxShellBounds, options.accuracy);
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const qcx::integrals::internal::RiFootprintTerms fastTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              false,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::RiFootprintTerms lightTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              true,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::DirectFootprintTerms exchange =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  indices.size(),
                                                  batch,
                                                  threadCount,
                                                  false,
                                                  0,
                                                  options.accuracy);

    // The clamp-ineligible masses: the folded totals minus exactly the
    // clamp-eligible mass (the per-thread scratch on both rungs plus the
    // light values buffer - both ride the clamped batch). The old total
    // is today's (pre-fix) estimate: the folded total minus the first-
    // iteration terms the fix folds in.
    const std::size_t fastClampIneligible = fastTerms.Total() - fastTerms.scratchBytes;
    const std::size_t lightClampIneligible =
        lightTerms.Total() - lightTerms.scratchBytes - lightTerms.valuesBufferBytes;
    // The follow-up folds the exchange per-call term into Total() too -
    // subtract it with the first-iteration terms so the "old" total stays
    // today's true pre-fix estimate (the Total()-inheritance contract:
    // the consumers see the term without logic changes, and the pin's
    // razor's-edge budget shifts with it - the cancellation keeps every
    // assertion at its former margin). The screened-quartet charge
    // does NOT follow the same cancellation: the charge (446,092,738 B at
    // this 168-function fixture) is ~40x the one-arena razor
    // (teamSize x (batch - 1) = 10.5 MB at the 2 MiB batch), so a
    // cancelled budget (404,466,925 B) sits BELOW the charged light
    // total (766,812,223 B) - the light rung cannot fit and the razor
    // collapses (measured: "the RI-J light rung cannot fit the remaining
    // workspace budget (404466925 bytes): the light-rung Create-time
    // footprint estimate is 766812223 bytes"). The budget must MOVE WITH
    // the charge instead: the razor rides the charged old total (the
    // charged fast total minus the per-call and first-iteration terms),
    // one arena below it - the light total fits, the fast rung's refusal
    // comes from its charged folded total (the per-call, transpose and
    // small-vector terms, far above the two-arena capacity), and the
    // fast clamp-ineligible mass sits BELOW the budget - its band role
    // is gone, the charge's task machinery carries it.
    //
    // The honest screened-quartet charge: the whole-call
    // charge drops from 446,092,738 B to 63,727,534 B at this n=168
    // fixture (the 24 B x n64 + 32 B x nF32 survivor), and the charged
    // totals sink with it (the razor budget measures 458,116,763 B; the
    // light total 384,447,019 B - 766,812,223 B minus the 382,365,204 B
    // delta - fits with the full batch). The fast rung's clamp-ineligible
    // mass (506,797,184 B - the fast terms carry no screened charge) now
    // sits ABOVE the razor: the counted survivor no longer lifts the charged
    // totals over the ineligible mass, its band role is back, and the
    // fast refusal at this budget is by mass - even a unit-batch fast
    // cannot fit (the assertion below inverts with the geometry).
    const std::size_t oldFastTotal = fastTerms.Total() + exchange.Total() -
                                     exchange.exchangePerCallBytes - fastTerms.transposeBytes -
                                     fastTerms.smallVectorsBytes;
    const std::size_t fastTotal = fastTerms.Total() + exchange.Total();
    const std::size_t lightTotal = lightTerms.Total() + exchange.Total();

    // The razor's-edge budget: one arena below the charged old total
    // (the clamped arena is teamSize x (batch - 1) bytes wide - the
    // old-total clamp fits with a one-arena shrink), clears both
    // a-priori exclusions (the fast attempt RAN - its clamp refused),
    // and admits the charged light total with the full batch - the
    // light rung is the only survivor.
    const std::size_t budgetBytes = oldFastTotal - threadCount * (batch - 1);
    // The band's geometry (measured 2026-09-03): the fast rung's
    // ineligible mass (506,797,184 B) sits above the 458,116,763 B razor
    // - its refusal is by mass now (the band role the charge had
    // taken is back); the light rung's ineligible mass and full total
    // still fit (384,447,019 B at the full batch).
    EXPECT_GT(fastClampIneligible, budgetBytes);
    EXPECT_LT(lightClampIneligible, budgetBytes);
    // The budget clears both a-priori exclusions (the light rung fired, so
    // neither refused it). The all-survive floor carries the counted-form
    // discount (dense - screened) x 104 - the same slice delta the counted
    // charge applies to the totals above, and the identical form
    // RiFirstIterationRefusesTheLightRung's own preamble states, so the two
    // sibling fixtures assert the exclusion the model actually computes
    // rather than the a-priori one it replaced. The tensor floor below is
    // the fast rung's own basis-fixed term, count-independent by
    // construction, and is unchanged between the two siblings.
    //
    // Measured on this fixture (2026-09-17, Release, this binary): budget
    // 709,496,075 B; all-survive 210,859,440 B (dense 1,568,160 tasks,
    // screened 77,760); discounted floor 55,857,840 B; tensor floor
    // 146,313,216 B. NEITHER floor binds here - margins 498,636,635 B and
    // 653,638,235 B - and the all-survive one cannot: the razor must admit
    // the light total, which sits above the all-survive pattern bound. The
    // exclusion this fixture actually exercises is the clamp's, which is
    // what the assertions around these two are for; the floors are
    // preambles recording that the run is not in the excluded band. The
    // numbers are stated because a reader would otherwise derive them by
    // hand from a budget the folded-total comments above no longer quote.
    EXPECT_GT(budgetBytes,
              qcx::integrals::internal::AllSurvivePatternBytes(pairList->pairs.size()) -
                  104 * (pairList->pairs.size() * auxPairList->shells.size() - screenedTaskCount));
    EXPECT_GT(budgetBytes,
              8 * pairList->functionCount * pairList->functionCount * auxPairList->functionCount);
    // The clamp fits the old total (the batch shrinks one arena) and
    // rejects the fast rung's folded total - the folded terms + two-arena
    // clamp are the mechanism.
    EXPECT_GT(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, oldFastTotal, budgetBytes, 2),
        0u);
    EXPECT_EQ(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, fastTotal, budgetBytes, 2),
        0u);
    EXPECT_GT(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, lightTotal, budgetBytes, 2),
        0u);

    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());
    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    const std::optional<FockModeInfo>& info = ri->ModeInfo();
    ASSERT_TRUE(info.has_value());
    // The fast attempt's clamp bound - the light rung fired and reserved
    // the folded total (the stack's full charge, the nested exchange
    // included). The nested exchange's k = team
    // fire claims its k-fold expansion of the post-reservation band, so
    // the stack's committed charge rides at or above the folded k = 1
    // estimate while its in-band cap re-derivation keeps it inside the
    // razor budget. Never-under is the honest form: the k-fold excess is
    // platform-arithmetic boundary-sensitive, and the Debug rounding
    // lands the total exactly ON the estimate on the clang leg - the
    // committed == predicted equality is a legitimate boundary.
    EXPECT_EQ(info->mode, FockBuildMode::kLightPath);
    EXPECT_FALSE(info->tensorExcluded);
    EXPECT_EQ(ri->RiMatrix().size(), 0);
    EXPECT_LE(info->predictedBytes, budgetBytes);
    EXPECT_GE(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(budget->CommittedBytes(), budgetBytes);
}

TEST(Footprint, RiFirstIterationFiresTheBlockedMetricRung) {
    // The blocked-metric rung's firing: the
    // blocked-metric estimate (the light rung with the metric built
    // strip-wise - the 3 x 8 nAuxFuncs^2 eigen-class term drops to the
    // 2 x 8 nAuxFuncs^2 the blocked peak carries, plus the per-strip
    // arena) sits below the light rung's by the
    // metric term's own gap, so the budget band between the rungs'
    // clamp-fit floors - the blocked rung fires where the light rung's
    // clamp refuses. The band's geometry on the 24-water kWaterSto3g
    // family (the deviation documented in
    // RiFirstIterationFiresTheLightRung): lightTotal - blockedTotal =
    // 1 x 8 nAuxFuncs^2 - max(batch, largest metric row mass) =
    // 1,262,080 bytes here (the largest row mass, ~64 KB at 216 aux
    // shells, rides under the 2 MiB batch), narrower than the two-arena
    // clamp capacity 2T(batch-1) = 16,777,208 bytes, so the blocked rung
    // always fires with a clamped batch in this band (never at the full
    // batch - the light rung would fit first). The razor budget one byte
    // below the light clamp-fit floor - lightTotal - 2T(batch-1) - 1 -
    // makes the light rung's clamp refuse by exactly one byte, and the
    // blocked rung's clamp absorbs the residual gap (its estimate sits
    // 15,515,129 bytes over the budget, inside the two-arena capacity):
    // the fired batch shrinks by ceil(15,515,129 / 2T) = 1,939,392 to
    // 157,760 bytes - healthy, and the fired estimate is re-evaluated at
    // it. Same construction as the band tests.
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kRiBandAux);
    ASSERT_TRUE(aux.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());
    // The kRiBandAux family: 216 aux shells, 648 functions (see
    // RiFirstIterationFiresTheLightRung).
    ASSERT_EQ(pairList->functionCount, 168u);
    ASSERT_EQ(auxPairList->shells.size(), 216u);
    ASSERT_EQ(auxPairList->functionCount, 648u);

    RiEngineOptions options;
    const std::size_t batch = std::size_t{2} * 1024 * 1024;
    options.maxBatchBytes = batch;

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value());
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> indices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, offsets, indices);

    // The screened task-grid count, exactly as the engine's decision block
    // computes it: the footprint's taskListBytes
    // charges the surviving (orbital pair, aux shell) cells x 16 B, and
    // this band's delta-cancellation holds only while the recomputed terms
    // carry the SAME count the engine charges (the razor budgets and the
    // fired-term equality below all ride these totals). The rungs differ by
    // the metric term only - the task charge cancels between them.
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value());
    std::vector<double> auxShellBounds(auxPairList->shells.size());

    for (std::size_t shell = 0; shell < auxPairList->shells.size(); ++shell)
    {
        auxShellBounds[shell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(shell, shell, *auxPairList)];
    }

    const std::size_t screenedTaskCount = qcx::integrals::internal::CountSchwarzSurvivingRiTasks(
        *schwarz, auxShellBounds, options.accuracy);
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const qcx::integrals::internal::RiFootprintTerms lightTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              true,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::RiFootprintTerms blockedTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              true,
                                              true,
                                              screenedTaskCount);
    const qcx::integrals::internal::DirectFootprintTerms exchange =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  indices.size(),
                                                  batch,
                                                  threadCount,
                                                  false,
                                                  0,
                                                  options.accuracy);
    const qcx::integrals::internal::RiFootprintTerms fastTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              false,
                                              false,
                                              screenedTaskCount);
    const std::size_t fastTotal = fastTerms.Total() + exchange.Total();
    const std::size_t lightTotal = lightTerms.Total() + exchange.Total();
    const std::size_t blockedTotal = blockedTerms.Total() + exchange.Total();

    // The rungs differ by the metric term only - the blocked eigen-class
    // 2 x 8 nAuxFuncs^2 plus the strip arena against the unblocked
    // 3 x 8 nAuxFuncs^2 (the stores cancel). The strip arena is the batch
    // cap: the largest metric row's byte mass (176 B per phantom quartet
    // plus 8 B per value - the last aux shell's row carries 216 quartets
    // and ~648 values) rides far under the 2 MiB cap.
    std::size_t maxRowMass = 0;

    for (std::size_t q = 0; q < auxPairList->shells.size(); ++q)
    {
        maxRowMass = std::max(
            maxRowMass, qcx::integrals::internal::MetricRowStripBytes(q, auxPairList->shells[q]));
    }

    EXPECT_LE(maxRowMass, batch);
    EXPECT_EQ(lightTerms.metricBytes - blockedTerms.metricBytes,
              8 * static_cast<std::size_t>(648u) * 648u - batch);
    EXPECT_EQ(lightTerms.Total() - blockedTerms.Total(),
              lightTerms.metricBytes - blockedTerms.metricBytes);

    // The razor's edge: the fast rung's and the light rung's clamps
    // refuse (the light deficit is 2T(batch-1) + 1 - one byte over the
    // two-arena capacity), the blocked rung's clamp absorbs the residual
    // gap (its deficit 8,796,665 sits inside the capacity).
    const std::size_t budgetBytes = lightTotal - 2 * threadCount * (batch - 1) - 1;
    EXPECT_EQ(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, fastTotal, budgetBytes, 2),
        0u);
    EXPECT_EQ(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, lightTotal, budgetBytes, 2),
        0u);
    const std::size_t firedBatch =
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, blockedTotal, budgetBytes, 2);
    EXPECT_GT(firedBatch, 0u);
    EXPECT_LT(firedBatch, batch);
    // The fired terms: the engine re-estimates both halves at the clamped
    // batch (ri_engine.cpp attempt()).
    const qcx::integrals::internal::RiFootprintTerms firedBlockedTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              firedBatch,
                                              threadCount,
                                              true,
                                              true,
                                              screenedTaskCount);
    const qcx::integrals::internal::DirectFootprintTerms firedExchange =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  indices.size(),
                                                  firedBatch,
                                                  threadCount,
                                                  false,
                                                  0,
                                                  options.accuracy);
    // The budget clears both a-priori exclusions (the fast attempt RAN).
    // The all-survive floor carries the counted-form discount (dense -
    // screened) x 104 - the slice delta the counted charge applies to the
    // totals above, so the budget and the floor shrink together and the
    // assertion keeps its pre-count margin. The tensor floor below is the
    // fast rung's own basis-fixed term, count-independent by construction.
    EXPECT_GT(budgetBytes,
              qcx::integrals::internal::AllSurvivePatternBytes(pairList->pairs.size()) -
                  104 * (pairList->pairs.size() * auxPairList->shells.size() - screenedTaskCount));
    EXPECT_GT(budgetBytes,
              8 * pairList->functionCount * pairList->functionCount * auxPairList->functionCount);

    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());
    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    const std::optional<FockModeInfo>& info = ri->ModeInfo();
    ASSERT_TRUE(info.has_value());
    // The blocked-metric rung fired and reserved: the mode stays the
    // light rung (the per-iteration recompute is the light payload), the
    // marker records the blocked metric build, no tensor is held, and the
    // fired estimate (both halves at the clamped batch) was reserved.
    // The nested exchange fires k = team into its
    // band, so the stack's committed charge rides at or above the k = 1
    // estimate while staying inside the razor budget - never-under is the
    // honest form (the k-fold excess is platform-arithmetic boundary-
    // sensitive; the equality committed == predicted is a legitimate
    // boundary).
    EXPECT_EQ(info->mode, FockBuildMode::kLightPath);
    EXPECT_FALSE(info->tensorExcluded);
    EXPECT_TRUE(info->blockedMetricRung);
    EXPECT_EQ(info->maxBatchBytes, firedBatch);
    EXPECT_EQ(ri->RiMatrix().size(), 0);
    EXPECT_EQ(info->metricBytes, firedBlockedTerms.metricBytes);
    EXPECT_LE(info->predictedBytes, budgetBytes);
    EXPECT_EQ(info->predictedBytes, firedBlockedTerms.Total() + firedExchange.Total());
    EXPECT_GE(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(budget->CommittedBytes(), budgetBytes);
}

TEST(Footprint, RiFirstIterationRefusesTheLightRung) {
    // The ladder's refusal: the budget one byte below
    // the BLOCKED-metric rung's clamp fit floor - the final rung's
    // deficit against its folded total exceeds the two-arena capacity
    // 2T(batch-1), so the refusal stands even though the clamp could
    // shrink the scratch - refuses with the standing LightRungRefusal
    // tokens (the four-rung ladder intact; the next-rung sentence keeps
    // its wording, including the "(named, not built)" of the rung this
    // test's budget now refuses past); nothing charged. The blocked floor
    // sits below the light rung's by the light-vs-blocked metric gap
    // (lightTotal - blockedTotal = 1 x 8 nAuxFuncs^2 - max(batch, the
    // largest metric row's mass) = 1,262,080 bytes at this fixture), so
    // the light rung's own one-byte razor (lightTotal - 2T(batch-1) - 1,
    // 1,262,079 bytes above this floor) would FIRE the blocked rung -
    // its deficit 15,515,129 sits inside the two-arena capacity - the
    // reason this budget sits one byte below the blocked floor instead.
    // Same construction as the band tests.
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kRiBandAux);
    ASSERT_TRUE(aux.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());

    RiEngineOptions options;
    const std::size_t batch = std::size_t{2} * 1024 * 1024;
    options.maxBatchBytes = batch;

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value());
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> indices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, offsets, indices);

    // The screened task-grid count, exactly as the engine's decision block
    // computes it: the footprint's taskListBytes
    // charges the surviving (orbital pair, aux shell) cells x 16 B, and
    // this band's delta-cancellation holds only while the recomputed terms
    // carry the SAME count the engine charges (the razor budget below rides
    // these totals; the rungs differ by the metric term only - the task
    // charge cancels between them).
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value());
    std::vector<double> auxShellBounds(auxPairList->shells.size());

    for (std::size_t shell = 0; shell < auxPairList->shells.size(); ++shell)
    {
        auxShellBounds[shell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(shell, shell, *auxPairList)];
    }

    const std::size_t screenedTaskCount = qcx::integrals::internal::CountSchwarzSurvivingRiTasks(
        *schwarz, auxShellBounds, options.accuracy);
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const qcx::integrals::internal::RiFootprintTerms lightTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              true,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::RiFootprintTerms blockedTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              true,
                                              true,
                                              screenedTaskCount);
    const qcx::integrals::internal::DirectFootprintTerms exchange =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  indices.size(),
                                                  batch,
                                                  threadCount,
                                                  false,
                                                  0,
                                                  options.accuracy);
    const std::size_t lightTotal = lightTerms.Total() + exchange.Total();
    const std::size_t blockedTotal = blockedTerms.Total() + exchange.Total();
    // One byte below the blocked rung's clamp-fit floor: both rungs'
    // deficits (blockedTotal - budgetBytes = 2T(batch-1) + 1) exceed the
    // two-arena capacity.
    const std::size_t budgetBytes = blockedTotal - 2 * threadCount * (batch - 1) - 1;
    // The refusal budget still clears both a-priori exclusions: the fast
    // attempt RAN and its clamp refused before the light one - both floors
    // are stated against the budget, matching the two sibling fixtures
    // (FiresTheLightRung, ExchangePerCallTermRidesTheBatchClamp). The
    // all-survive floor carries the counted-form discount (dense -
    // screened) x 120 - the same discount the screened charge applies to
    // the totals above (16 B per surviving task list plus 104 B per
    // surviving slice task), so the delta cancels and the assertion keeps
    // its dense-world meaning. The tensor floor below is the fast rung's
    // own basis-fixed term, count-independent by construction: the light
    // rung's clamp-ineligible mass no longer clears it under the counted
    // slice (206,622,272 B -> 51,620,672 B at this fixture; the dense
    // over-charge is what once lifted it above 146,313,216 B), so the
    // mass-vs-tensor comparison is retired and the exclusion is stated
    // against the budget the exclusion actually gates on.
    EXPECT_GT(budgetBytes,
              qcx::integrals::internal::AllSurvivePatternBytes(pairList->pairs.size()) -
                  120 * (pairList->pairs.size() * auxPairList->shells.size() - screenedTaskCount));
    EXPECT_GT(budgetBytes,
              8 * pairList->functionCount * pairList->functionCount * auxPairList->functionCount);

    EXPECT_EQ(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, blockedTotal, budgetBytes, 2),
        0u);
    EXPECT_EQ(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, lightTotal, budgetBytes, 2),
        0u);
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());
    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_FALSE(ri.has_value());
    EXPECT_EQ(ri.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = ri.error().message;
    EXPECT_NE(message.find("light rung cannot fit"), std::string::npos);
    EXPECT_NE(message.find("light-rung Create-time footprint estimate is"), std::string::npos);
    EXPECT_NE(message.find("direct screened"), std::string::npos);
    EXPECT_NE(message.find("batched/blocked"), std::string::npos);
    EXPECT_NE(message.find("recompute"), std::string::npos);
    EXPECT_NE(message.find("disk"), std::string::npos);
    EXPECT_NE(message.find("blocked-metric rung"), std::string::npos);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, ExchangePerCallBytesArithmetic) {
    // The values-term arithmetic pins (the structural redesign of
    // the S3-era fit, 2026-09-03): ExchangePerCallBytes now charges the
    // per-batch bound, not the whole-call fitted envelope - the
    // 0.0088 kExchangeSurvivalFraction / 1.42e-4 kExchangeQuartetFraction
    // arithmetic is gone from the term (the constants survive only as
    // the GPU lane's never-under floors and the fraction law's anchor,
    // device_footprint.hpp). The structural terms are F64Bound = 2 x
    // maxBatchBytes (the fp64 half's per-call bound - the measured
    // C24H50 fp64 bound ~1 GiB at the 512 MiB default cap) and
    // F32Live = maxBatchBytes (the certified fp32 half's per-call bound
    // - its valuesF32 + per-quartet bounds are chunked per batch and
    // cache slice at the batch cap like the fp64 half's). The preset
    // splits the lanes: kTight runs the certified lane OFF (all fp64 -
    // F64Bound only), kLoose/kNormal route
    // through the certified lane and charge both halves (F64Bound +
    // F32Live - they run sequentially per call, so the CWA sums them).
    // The literals are exact (the fit's truncating cast sums are gone).
    const std::size_t batch = FockBuildOptions{}.maxBatchBytes;
    // The default 512 MiB cap (the C24H50 admission class).
    EXPECT_EQ(qcx::integrals::internal::ExchangePerCallBytes(
                  batch, qcx::integrals::AccuracyPreset::kTight),
              1073741824u);
    EXPECT_EQ(qcx::integrals::internal::ExchangePerCallBytes(
                  batch, qcx::integrals::AccuracyPreset::kNormal),
              1610612736u);
    EXPECT_EQ(qcx::integrals::internal::ExchangePerCallBytes(
                  batch, qcx::integrals::AccuracyPreset::kLoose),
              1610612736u);
    // The lane-shape relation kNormal - kTight = F32Live = maxBatchBytes
    // (the certified lane's bound) - exact, no rounding.
    EXPECT_EQ(qcx::integrals::internal::ExchangePerCallBytes(
                  batch, qcx::integrals::AccuracyPreset::kNormal) -
                  qcx::integrals::internal::ExchangePerCallBytes(
                      batch, qcx::integrals::AccuracyPreset::kTight),
              batch);
    // The terms scale with the batch cap (they track the clamped batch
    // when the ladder re-estimates - the shrink absorbs the term): the
    // band tests' 2 MiB batch.
    EXPECT_EQ(qcx::integrals::internal::ExchangePerCallBytes(
                  std::size_t{2} * 1024 * 1024, qcx::integrals::AccuracyPreset::kTight),
              4194304u);
    EXPECT_EQ(qcx::integrals::internal::ExchangePerCallBytes(
                  std::size_t{2} * 1024 * 1024, qcx::integrals::AccuracyPreset::kNormal),
              6291456u);
}

TEST(Footprint, ScreenedQuartetFractionLawArithmetic) {
    // The fraction-law arithmetic pins: the
    // piecewise log-log law through the measured anchors - flat
    // 2.5e-3 below n = 298 (the C12H26 bracket 2.225e-3..2.336e-3,
    // charged at the conservative max), 1.42e-4 at n = 586 (the C24H50
    // fit point, equals the shipped kExchangeQuartetFraction), 1/n^2 =
    // 4e-8 at and above n = 5000 (the measured ~n^2 survival anchor).
    // The boundary continuity: at n = 586 the segments agree to the
    // exponent-rounding residual ~5e-6 relative (s1 evaluated AT 586
    // gives 1.420007226e-4 - 2.5e-3 x (586/298)^-4.2415 with the
    // 4-decimal exponent - not the anchor 1.42e-4 exactly; the charge
    // pin below follows the law, not the anchor). The literals are the
    // exact doubles, derived 2026-09-01 (n=298: 2.5e-3 x 298^4 = 19,715,376.04
    // quartets x 224 B = 4,416,244,232.96; n=586: 1.42e-4 x 586^4 =
    // 16,744,755.42 quartets x 224 B = 3,750,844,301.21; n=5000:
    // 4e-8 x 6.25e14 = 25,000,000 quartets x 224 B = 5,600,000,000;
    // n=7: 2.5e-3 x 7^4 = 6.0025 quartets x 224 B = 1344.56), re-derived
    // 2026-09-03 for the counted survivor charge (the same quartet counts x
    // 24 B - kScreenedTaskBytesPerQuartet, the fp64 master - and x 32 B
    // with the kCertifiedLaneBytesPerQuartet delta at kLoose/kNormal:
    // n=298 x 32 = 630,892,033.28, n=586 x 32 = 535,834,900.16, n=5000 x
    // 32 = 800,000,000, n=7 x 32 = 192.08; n=298 x 24 = 473,169,024.96,
    // n=586 x 24 = 401,876,175.12, n=5000 x 24 = 600,000,000, n=7 x 24 =
    // 144.06 - size_t truncation).
    EXPECT_DOUBLE_EQ(qcx::integrals::internal::ScreenedQuartetFraction(7), 2.5e-3);
    EXPECT_DOUBLE_EQ(qcx::integrals::internal::ScreenedQuartetFraction(298), 2.5e-3);
    // The segment-1 law evaluated AT 586 (the exact computed value - the
    // 585 pin's pattern) and the anchor agreement (the 4-decimal-exponent
    // residual, 1e-9 absolute).
    EXPECT_NEAR(qcx::integrals::internal::ScreenedQuartetFraction(586),
                2.5e-3 * std::pow(586.0 / 298.0, -4.2415),
                1e-12);
    EXPECT_NEAR(qcx::integrals::internal::ScreenedQuartetFraction(586), 1.42e-4, 1e-9);
    EXPECT_DOUBLE_EQ(qcx::integrals::internal::ScreenedQuartetFraction(5000), 4e-8);
    EXPECT_DOUBLE_EQ(qcx::integrals::internal::ScreenedQuartetFraction(8000), 1.0 / 64e6);
    // The segment-1 boundary agrees with the anchor at 586 (5e-7 relative).
    EXPECT_NEAR(qcx::integrals::internal::ScreenedQuartetFraction(585),
                2.5e-3 * std::pow(585.0 / 298.0, -4.2415),
                1e-12);

    // The whole-run screened-quartet charge at the anchors, both lane
    // states: kTight (fp64 master only - the 24 B per-quartet survivor)
    // and kNormal (the certified-lane delta rides the full charged set -
    // the never-under end of the split).
    // convention).
    EXPECT_EQ(
        qcx::integrals::internal::ScreenedQuartetBytes(7, qcx::integrals::AccuracyPreset::kTight),
        144u);
    EXPECT_EQ(
        qcx::integrals::internal::ScreenedQuartetBytes(298, qcx::integrals::AccuracyPreset::kTight),
        473169024u);
    EXPECT_EQ(
        qcx::integrals::internal::ScreenedQuartetBytes(586, qcx::integrals::AccuracyPreset::kTight),
        401876175u);
    EXPECT_EQ(qcx::integrals::internal::ScreenedQuartetBytes(
                  5000, qcx::integrals::AccuracyPreset::kTight),
              600000000u);
    EXPECT_EQ(
        qcx::integrals::internal::ScreenedQuartetBytes(7, qcx::integrals::AccuracyPreset::kNormal),
        192u);
    EXPECT_EQ(qcx::integrals::internal::ScreenedQuartetBytes(
                  298, qcx::integrals::AccuracyPreset::kNormal),
              630892033u);
    EXPECT_EQ(qcx::integrals::internal::ScreenedQuartetBytes(
                  586, qcx::integrals::AccuracyPreset::kNormal),
              535834900u);
    EXPECT_EQ(qcx::integrals::internal::ScreenedQuartetBytes(
                  5000, qcx::integrals::AccuracyPreset::kNormal),
              800000000u);
}

TEST(Footprint, CoulombOnlyStackCarriesNoExchangeCharge) {
    // The pin: the exchange per-call term
    // is charged only when the estimated stack runs an exchange pass. A
    // Coulomb-only stack (buildCoulombOnly - the QFMM near-field) has no
    // exchange mass, so the term must be ABSENT from its estimate - the
    // phantom charge the flag threading removes. The engaged estimate
    // carries the term (H2O/STO-3G n=7 kNormal at the default 512 MiB
    // batch: the structural bound F64Bound + F32Live = 3 x maxBatchBytes
    // = 1,610,612,736 B - the flag gate keeps the per-batch
    // bound, not the old 87 B whole-call fit); the Coulomb-only estimate
    // zeroes exactly it - every other term is identical. Small arithmetic
    // pin - no gate run needed.
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value());
    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());

    const std::size_t batch = FockBuildOptions{}.maxBatchBytes;
    const qcx::integrals::internal::DirectFootprintTerms engaged =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  0,
                                                  batch,
                                                  1,
                                                  false,
                                                  0,
                                                  qcx::integrals::AccuracyPreset::kNormal,
                                                  true);
    const qcx::integrals::internal::DirectFootprintTerms coulombOnly =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  0,
                                                  batch,
                                                  1,
                                                  false,
                                                  0,
                                                  qcx::integrals::AccuracyPreset::kNormal,
                                                  false);
    EXPECT_EQ(engaged.exchangePerCallBytes, 1610612736u);
    EXPECT_EQ(coulombOnly.exchangePerCallBytes, 0u);
    EXPECT_EQ(coulombOnly.Total(), engaged.Total() - engaged.exchangePerCallBytes);
    EXPECT_EQ(coulombOnly.pairStoreBytes, engaged.pairStoreBytes);
    EXPECT_EQ(coulombOnly.patternBytes, engaged.patternBytes);
    EXPECT_EQ(coulombOnly.scratchBytes, engaged.scratchBytes);
    EXPECT_EQ(coulombOnly.structuralBytes, engaged.structuralBytes);
    EXPECT_EQ(coulombOnly.cacheBytes, engaged.cacheBytes);
}

TEST(Footprint, ExchangePerCallTermRidesTheBatchClamp) {
    // The band pin (the flip test, reframed): the
    // earlier fit charged a WHOLE-CALL term (28,945,045 B at n=168 kNormal)
    // that no batch shrink could absorb - it exceeded the two-arena
    // clamp capacity at this fixture's midpoint budget, so the post-term
    // light estimate refused where the pre-term one admitted
    // (ExchangePerCallTermFlipsTheFitBand). The restructured term is the
    // structural per-batch bound (F64Bound + F32Live = 3 x maxBatchBytes
    // = 6,291,456 B at the 2 MiB batch): it rides the clamp, so the flip
    // at that budget is gone by construction - the razor budget shrinks
    // the batch and the term shrinks with it, and the engine ADMITS at
    // the very budget the S3-era fit refused (the fix's observable:
    // the model no longer refuses admissions the chunked code fits). The
    // razor is the margin-based form: one full batch cap of the two-arena
    // clamp under the
    // POST-term light total (margin = teamSize x batch) - derived from
    // the platform-computed terms, it fires the light clamp at the half-
    // batch point at EVERY team size (the derivation and the CI numbers
    // are at the budget line - the pre-band razor held only at
    // teamSize >= 4 and the CI clang leg runs team size 1). The light-vs-
    // blocked metric gap (1,262,080 B after the counts)
    // plays no role here: the light attempt precedes the blocked one and
    // its clamp fits (its deficit is exactly the razor margin teamSize x
    // batch, whose half-batch shrink sits inside the two-arena capacity),
    // so the light rung fires regardless of where the blocked floor sits.
    // The fast rung's clamp stays pinned: its pre-term and post-term
    // totals refuse this budget alike (its first-iteration excess over
    // the light rung - the tensor, riMatrix and transpose terms versus
    // the slice and the batch-capped values buffer - is ~435 MB at
    // n=168, far above the two-arena capacity), and the clamp-fail falls
    // through to the light attempt, whose clamp decides - deterministic.
    // The carriage at the DirectFootprint level is pinned below (the
    // member equals the structural term and Total() folds it); the
    // pre-term arithmetic (lightTotal - term) cannot be run (the code is
    // post-term) - it is pinned at the clamp level, where the light
    // clamp absorbs the full post-term deficit (the razor margin
    // teamSize x batch, inside the two-arena capacity).
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kRiBandAux);
    ASSERT_TRUE(aux.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());

    RiEngineOptions options;
    const std::size_t batch = std::size_t{2} * 1024 * 1024;
    options.maxBatchBytes = batch;

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value());
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> indices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, offsets, indices);

    // The screened task-grid count, exactly as the engine's decision block
    // computes it: the footprint's taskListBytes
    // charges the surviving (orbital pair, aux shell) cells x 16 B, and
    // this band's delta-cancellation holds only while the recomputed terms
    // carry the SAME count the engine charges (the razor budget and the
    // fired-term equality below all ride these totals).
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value());
    std::vector<double> auxShellBounds(auxPairList->shells.size());

    for (std::size_t shell = 0; shell < auxPairList->shells.size(); ++shell)
    {
        auxShellBounds[shell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(shell, shell, *auxPairList)];
    }

    const std::size_t screenedTaskCount = qcx::integrals::internal::CountSchwarzSurvivingRiTasks(
        *schwarz, auxShellBounds, options.accuracy);
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const qcx::integrals::internal::RiFootprintTerms fastTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              false,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::RiFootprintTerms lightTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              true,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::DirectFootprintTerms exchange =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  indices.size(),
                                                  batch,
                                                  threadCount,
                                                  false,
                                                  0,
                                                  options.accuracy);

    // The carriage at the DirectFootprint level: the member is set to the
    // structural bound (3 x maxBatchBytes at this fixture's 2 MiB batch -
    // F64Bound + F32Live, kNormal) and Total() folds it. The fraction-law
    // screened-quartet charge is unconditional (the task machinery exists
    // on every direct path) - 63,727,534 B at this n=168 fixture (2.5e-3
    // x 168^4 x 32 B at the default kNormal lane split - the survivor
    // charge, the ScreenedQuartetFractionLawArithmetic anchor).
    const std::size_t term =
        qcx::integrals::internal::ExchangePerCallBytes(batch, options.accuracy);
    EXPECT_EQ(term, 6291456u);
    EXPECT_EQ(exchange.exchangePerCallBytes, term);
    EXPECT_EQ(exchange.screenedQuartetBytes, 63727534u);
    EXPECT_EQ(exchange.Total(),
              exchange.pairStoreBytes + exchange.patternBytes + exchange.scratchBytes +
                  exchange.structuralBytes + exchange.cacheBytes + exchange.exchangePerCallBytes +
                  exchange.screenedQuartetBytes + exchange.schwarzSweepBytes);

    const std::size_t preLightTotal = lightTerms.Total() + exchange.Total() - term;
    const std::size_t lightTotal = lightTerms.Total() + exchange.Total();
    const std::size_t lightClampIneligible =
        lightTerms.Total() - lightTerms.scratchBytes - lightTerms.valuesBufferBytes;
    // The razor budget: the margin-based form, derived from the
    // platform-computed
    // terms - one full batch cap of the two-arena clamp below the
    // POST-term light total (margin = teamSize x batch). The pre-band
    // razor sat one clamped arena (teamSize x (batch - 1)) below the
    // PRE-term total, whose admission zone is empty below teamSize = 4
    // (the post-term deficit term + teamSize x (batch - 1) must fit the
    // two-arena capacity 2 x teamSize x (batch - 1), and the structural
    // term 3 x batch overruns it alone at team sizes 1-3). The clang
    // Debug CI leg runs team size 1 (the runner's topology probe): the
    // fixture budget landed BELOW the Create-time light estimate there
    // and the engine refused the very admission the test pins (run
    // 34155831480: budget 326,549,911 B < the light-rung estimate
    // 334,938,518 B). The margin-based budget fires the light clamp at
    // the half-batch point at EVERY team size (fired = batch -
    // ceil(teamSize x batch / (2 x teamSize)) = batch / 2 - the deficit
    // always sits inside the two-arena capacity), so the intent - the
    // engine admits at the very budget the earlier whole-call term refused
    // - holds on every platform; at teamSize >= 4 the budget stays below
    // the pre-term total (the F3 razor character), at team size 1 the
    // pre-term total fits at the full batch and the post-term admission
    // still needs the clamp.
    const std::size_t budgetBytes = lightTotal - threadCount * batch;
    // The budget clears both a-priori exclusions (the attempt flow runs -
    // the margin-based budget sits teamSize x batch below the post-term
    // light total, far above both floors at every team size). The
    // all-survive floor carries the counted-form discount (dense -
    // screened) x 104 - the slice delta the counted charge applies to the
    // totals above, so the budget and the floor shrink together and the
    // assertion keeps its pre-count margin. The tensor floor below is the
    // fast rung's own basis-fixed term, count-independent by construction.
    EXPECT_GT(budgetBytes,
              qcx::integrals::internal::AllSurvivePatternBytes(pairList->pairs.size()) -
                  104 * (pairList->pairs.size() * auxPairList->shells.size() - screenedTaskCount));
    EXPECT_GT(budgetBytes,
              8 * pairList->functionCount * pairList->functionCount * auxPairList->functionCount);
    // The light attempt reaches its clamp (the fast rung's clamp refusal
    // is unpinned - see the header comment).
    EXPECT_LT(lightClampIneligible, budgetBytes);
    // THE ABSORPTION (the flip's inverse, at the clamp level): the
    // pre-term light total admits at every team size (its deficit
    // (teamSize - 3) x batch is <= 0 at team sizes 1-3 - it fits
    // unclamped - and rides the two-arena shrink at teamSize >= 4), and
    // the POST-term light total now admits too: its deficit is exactly
    // the razor margin teamSize x batch, INSIDE the two-arena capacity
    // 2 x teamSize x (batch - 1) for every batch >= 2, so the clamp
    // fires at the half-batch point (shrink = ceil(teamSize x batch /
    // (2 x teamSize)) = batch / 2) and the batch-scaled term shrinks
    // with the batch (the earlier whole-call term's 28,945,045 B exceeded the
    // capacity and refused this budget; the restructured term restores the
    // admission). The fast rung's pre-term and post-term totals refuse
    // this budget alike: its first-iteration excess over the light rung -
    // the tensor, riMatrix and transpose terms (three 8 n^2 nAux =
    // 146,313,216 B each at n=168, nAux=648) versus the slice and the
    // batch-capped values buffer - is ~435 MB, far above the two-arena
    // capacity, so the fast side has no admission at a budget razored on
    // the light total.
    EXPECT_EQ(qcx::integrals::internal::ClampBatchBytes(
                  batch, threadCount, fastTerms.Total() + exchange.Total() - term, budgetBytes, 2),
              0u);
    EXPECT_GT(qcx::integrals::internal::ClampBatchBytes(
                  batch, threadCount, preLightTotal, budgetBytes, 2),
              0u);
    EXPECT_EQ(qcx::integrals::internal::ClampBatchBytes(
                  batch, threadCount, fastTerms.Total() + exchange.Total(), budgetBytes, 2),
              0u);
    const std::size_t firedBatch =
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, lightTotal, budgetBytes, 2);
    EXPECT_GT(firedBatch, 0u);
    EXPECT_LT(firedBatch, batch);

    // The Create-level observable: the post-term estimate ADMITS (the
    // S3-era refusal at this budget is gone) - the light rung fires with
    // the clamped batch, and the fired estimate re-evaluates BOTH halves
    // at it (the exchange term scales with the fired batch).
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());
    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    const std::optional<FockModeInfo>& info = ri->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kLightPath);
    EXPECT_FALSE(info->tensorExcluded);
    EXPECT_EQ(info->maxBatchBytes, firedBatch);
    EXPECT_EQ(ri->RiMatrix().size(), 0);

    // The fired terms at the clamped batch (the engine re-estimates both
    // halves there - the same recompute as the blocked-rung pin).
    const qcx::integrals::internal::RiFootprintTerms firedLightTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              firedBatch,
                                              threadCount,
                                              true,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::DirectFootprintTerms firedExchange =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  indices.size(),
                                                  firedBatch,
                                                  threadCount,
                                                  false,
                                                  0,
                                                  options.accuracy);
    EXPECT_EQ(info->predictedBytes, firedLightTerms.Total() + firedExchange.Total());
    EXPECT_LE(info->predictedBytes, budgetBytes);
    // The nested exchange's k = team charge
    // reaches or exceeds the k = 1 estimate, yet its in-band re-derivation
    // keeps the stack's total commit inside the budget - never-under is
    // the honest form (the k-fold excess is platform-arithmetic boundary-
    // sensitive; the equality committed == predicted is a legitimate
    // boundary).
    EXPECT_GE(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(budget->CommittedBytes(), budgetBytes);
}

TEST(Footprint, CacheBytesModelsTheEntryMapAllowance) {
    // The cache-entry-map shape: the EriBatchCache entry map is
    // outside the payload arenas - the unordered_map's Entry (~48 B) plus
    // node overhead run ~90-100 B per cached quartet, 1-3 GB against the
    // 10-30M-quartet screening law of the 5000-scale projections - so the
    // cache term is the two payload lanes PLUS one maxCacheBytes allowance
    // (3x), never the 2x under-count. Disengaged: exactly zero.
    auto molecule = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value());
    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());

    const std::size_t maxCacheBytes = std::size_t{512} * 1024u * 1024u;
    const qcx::integrals::internal::DirectFootprintTerms withCache =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  0,
                                                  FockBuildOptions{}.maxBatchBytes,
                                                  1,
                                                  true,
                                                  maxCacheBytes,
                                                  qcx::integrals::AccuracyPreset::kNormal);
    EXPECT_EQ(withCache.cacheBytes, 3 * maxCacheBytes);
    const qcx::integrals::internal::DirectFootprintTerms withoutCache =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  0,
                                                  FockBuildOptions{}.maxBatchBytes,
                                                  1,
                                                  false,
                                                  maxCacheBytes,
                                                  qcx::integrals::AccuracyPreset::kNormal);
    EXPECT_EQ(withoutCache.cacheBytes, 0u);
}

TEST(Footprint, QfmmModeSelectionReservesTheOuterStore) {
    // The QFMM nested reservation (scheme (a)): the outer Create charges
    // exactly the outer store - the full estimate minus the near-field
    // half - and the nested near-field direct Create charges its own
    // estimate, so the stack total is the full estimate and the nested
    // fits whenever the outer does.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    const std::size_t budgetBytes = 3 * DefaultArenaBytes() + std::size_t{1024} * 1024;
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    QfmmOptions options;
    options.workspaceBudget = &*budget;
    auto qfmm = QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

    const std::optional<FockModeInfo>& info = qfmm->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kFastPath);
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    // 15 pairs, the default leaf cap 8: 2 leaves, 3 nodes.
    EXPECT_EQ(info->outerStoreBytes, QfmmOuterStoreBytes(*molecule, *basis, *pairList, 8));
    EXPECT_EQ(info->reservedBytes, info->outerStoreBytes);
    EXPECT_EQ(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(info->predictedBytes, budgetBytes);
}

TEST(Footprint, QfmmModeSelectionTinyBudgetRefuses) {
    // The QFMM family's exclusion (i): a one-byte budget refuses with the
    // ladder and the builder name.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto budget = WorkspaceBudget::Create(1);
    ASSERT_TRUE(budget.has_value());

    QfmmOptions options;
    options.workspaceBudget = &*budget;
    auto qfmm = QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(qfmm.has_value());
    EXPECT_EQ(qfmm.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = qfmm.error().message;
    EXPECT_NE(message.find("QFMM"), std::string::npos);
    EXPECT_NE(message.find("counted Schwarz-screened neighbor pattern"), std::string::npos);
    EXPECT_NE(message.find("direct screened"), std::string::npos);
    EXPECT_NE(message.find("batched/blocked"), std::string::npos);
    EXPECT_NE(message.find("recompute"), std::string::npos);
    EXPECT_NE(message.find("disk"), std::string::npos);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, QfmmModeSelectionNestedCarriesTheSweepCount) {
    // The spurious-nested-refusal band for the QFMM family (the asymmetry:
    // the RI-J's nested exchange got its carrier while the QFMM's nested
    // near-field did not): the budget fits the
    // outer's all-survive bound by a hair, but the outer's reservation
    // leaves the nested remaining below that bound, so the nested's
    // a-priori exclusion (i) would refuse a stack the outer proved feasible
    // with its counted sweep. The QFMM's outer now carries the counted
    // sweep (qfmm_fock_build.cpp), the nested runs its own counted sweep
    // and the stack constructs as FastPath - the same band shape as the
    // RI-J's RiModeSelectionNestedAllSurviveBandConstructs (the comment
    // there works out the numbers: 72 S shells, 2,628 pairs, a 27.6 MB
    // all-survive bound against a ~4 MB outer store).
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kTinyOrbitalSOnly);
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    // 24 waters x 3 S shells = 72 shells -> 2628 canonical pairs.
    const std::size_t nPairs = pairList->pairs.size();
    ASSERT_EQ(nPairs, 2628u);

    // The band: a budget that fits the all-survive bound by 1 KB.
    const std::size_t budgetBytes = AllSurvivePatternBytes(nPairs) + 1024;
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    QfmmOptions options;
    options.workspaceBudget = &*budget;
    auto qfmm = QfmmJBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(qfmm.has_value()) << qfmm.error().message;

    const std::optional<FockModeInfo>& info = qfmm->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kFastPath);
    EXPECT_LE(info->predictedBytes, budgetBytes);
    EXPECT_EQ(info->reservedBytes, info->outerStoreBytes);
    EXPECT_EQ(budget->CommittedBytes(), info->predictedBytes);
    // The band is real: after the outer's reservation the remaining is far
    // below the all-survive bound, so the nested's a-priori exclusion would
    // have refused without the carried sweep count.
    EXPECT_LT(budget->Remaining(), AllSurvivePatternBytes(nPairs));
}

TEST(Footprint, ModeSelectionValidatedSweepSkipsExclusionOne) {
    // A nested builder carrying the enclosing RI-J's
    // validated sweep count must not be refused by the a-priori all-survive
    // exclusion - the outer already proved the stack with its counted sweep,
    // and the all-survive bound can exceed the post-reservation remaining
    // even when the counted pattern fits. The nested still runs its own
    // counting pass and estimate, so the refusal (when the fit is
    // impossible) names the estimate, not the exclusion.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto budget = WorkspaceBudget::Create(1);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    options.validatedSweepCount = 1;
    auto direct = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(direct.has_value());
    EXPECT_EQ(direct.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = direct.error().message;
    EXPECT_NE(message.find("footprint estimate is"), std::string::npos);
    EXPECT_EQ(message.find("all-survive neighbor pattern"), std::string::npos);

    // The control: without the carrier the a-priori exclusion fires first.
    FockBuildOptions control;
    control.workspaceBudget = &*budget;
    auto refused = DirectJkFockBuilder::Create(*molecule, *basis, *core, control);
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(refused.error().message.find("counted Schwarz-screened neighbor pattern"),
              std::string::npos);
    // Neither refusal charged anything.
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, RiModeSelectionNestedAllSurviveBandConstructs) {
    // The spurious-nested-refusal band, end-to-end):
    // the budget fits the outer's all-survive bound by a hair, but the
    // outer's reservation leaves the nested remaining below that bound - the
    // nested's a-priori exclusion would refuse a stack the outer proved
    // feasible. With the validated sweep count carried, the nested runs its
    // own counted sweep and the stack constructs; the total charge is the
    // full estimate (nesting order = reservation order).
    //
    // The aux basis must span every atom (the RI-J engine maps it over the
    // molecule - BuildShellPairs refuses a partial aux), so the s-only O+H
    // text serves as both the orbital and the aux basis. The band needs the
    // all-survive bound (quadratic in the pair count) to dominate the
    // estimate's cubic terms (the two (uv|P) copies and the task list), so
    // it opens only at the 24-water scale: 72 S shells, 2,628 pairs,
    // AllSurvivePatternBytes = 27,636,048; the non-scratch estimate is
    // 19,225,672 + 8 x (counted survivors) - the shared-phantom metric
    // store shaved 1,733,536 B off the former phantom-hole estimate -
    // leaving an 8.4 MB band for the clamped batch arenas. At 14 waters
    // (903 pairs) the cubic terms win and the outer clamp refuses first -
    // with STO-3G the per-pair store dwarfs the bound.
    //
    // The screened-quartet charge lands in Total(): the screened-quartet
    // charge (15,049,359 B at n = 72) rides the exchange half's estimate,
    // pushing the fast rung's clamp-ineligible total above the all-survive
    // bound - the razor re-derives on the charged ineligible mass, the same
    // pattern as RiFirstIterationFiresTheLightRung).
    //
    // The structural F64Bound + F32Live per-batch terms replace the
    // whole-call fit: the fixture's batch is
    // the 2 MiB razor geometry, not the engine default - the exchange
    // half's mirror charge is now 3 x maxBatchBytes (6,291,456 B at 2 MiB
    // kNormal), and the band-reality assertion below needs that charge to
    // sit BELOW the 27.6 MB all-survive bound (the nested-exclusion
    // premise: the outer's reservation leaves the exchange charge as the
    // remaining). At the default 512 MiB batch the structural charge is
    // 1.61 GB - above the bound, so the exclusion is not the binding
    // constraint there and the earlier band moves to the batch-clamp
    // fixtures (ExchangePerCallTermRidesTheBatchClamp); the engine's own
    // attempt batch is pinned to the fixture's here (the razor cancels
    // the mirror's full-batch term only when both sides charge the same
    // batch).
    //
    // The honest screened-quartet charge (the 224 B/q control mass
    // replaced by the ~24-32 B/q survivor): the exchange
    // half's charge drops to 2,149,908 B (n = 72 at the kNormal lane
    // split), and the clamp-ineligible total (27,508,060 B - measured
    // 2026-09-03) sinks back BELOW the all-survive bound by 127,988 B -
    // the razor's mass is gone and the razor re-derives on the
    // bound itself once more (the earlier shape, restored): the budget
    // sits 1024 B above AllSurvivePatternBytes (27,636,048 B), clearing
    // exclusion (i) by the hair while the nested-exclusion premise still
    // binds (the post-reservation remaining stays far below the bound),
    // and the fast rung's clamp absorbs the two per-thread arenas
    // (deficit ~20.8 MB at the fixture's team, a ~12.9 KB fired batch at
    // team 5 - the clamp's arena geometry, the ineligible mass now
    // charge-free of the razor).
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kTinyOrbitalSOnly);
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kTinyOrbitalSOnly);
    ASSERT_TRUE(aux.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());
    // 24 waters x 3 S shells = 72 shells -> 2628 canonical pairs.
    const std::size_t nPairs = pairList->pairs.size();
    ASSERT_EQ(nPairs, 2628u);

    RiEngineOptions options;

    // The band: the budget rides the all-survive bound (the razor
    // razor re-derivation - the bound is the decisive mass again since
    // the honest charge sank the clamp-ineligible total below it).
    // The budget sits 1024 B above AllSurvivePatternBytes (27,636,048
    // B) - the razor's hair over exclusion (i) - clearing the a-priori
    // exclusions (the bound and the tensor term, 2,985,984 B) and
    // leaving the fast rung's clamp to absorb the two per-thread arenas
    // (the ineligible mass 27,508,060 B is no longer charge-carried;
    // the fired batch is the clamp's arena geometry, ~12.9 KB at the
    // fixture's team size 5).
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const std::size_t batch = std::size_t{2} * 1024 * 1024;
    options.maxBatchBytes = batch;
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value());
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> indices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, offsets, indices);
    // The screened task-grid count, exactly as the engine's decision block
    // computes it: the mirror recompute below
    // charges the surviving (orbital pair, aux shell) cells x 16 B - the
    // same count the engine charges (the band's geometric premises are
    // all bound-independent of the task term, but the mirror must stay
    // honest).
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value());
    std::vector<double> auxShellBounds(auxPairList->shells.size());

    for (std::size_t shell = 0; shell < auxPairList->shells.size(); ++shell)
    {
        auxShellBounds[shell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(shell, shell, *auxPairList)];
    }

    const std::size_t screenedTaskCount = qcx::integrals::internal::CountSchwarzSurvivingRiTasks(
        *schwarz, auxShellBounds, options.accuracy);
    const qcx::integrals::internal::RiFootprintTerms fastTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              batch,
                                              threadCount,
                                              false,
                                              false,
                                              screenedTaskCount);
    const qcx::integrals::internal::DirectFootprintTerms exchange =
        qcx::integrals::internal::DirectFootprint(*molecule,
                                                  *basis,
                                                  *pairList,
                                                  indices.size(),
                                                  batch,
                                                  threadCount,
                                                  false,
                                                  0,
                                                  options.accuracy);
    // The clamp-ineligible mass of the band comment's derivation (the
    // 27,508,060 B figure above): computed but deliberately never
    // asserted - the razor no longer charge-carries it - so it reads
    // as a dead store to the analyzer.
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    const std::size_t ineligibleFast =
        fastTerms.Total() + exchange.Total() - fastTerms.scratchBytes - exchange.scratchBytes;
    // The band fixture's budget, re-read after the Schwarz pre-pass charge
    // landed: the fixture's premise is a budget that fits the all-survive
    // bound by 1 KB, and every Create now also carries its chunked screening
    // sweep. The budget adds exactly the charge the model adds - the phase-max
    // excess of the sweeps over the store terms the footprint already carries
    // - so the remaining after the outer's reservation is still what the band
    // comment below describes.
    const std::size_t storeBytes =
        qcx::integrals::internal::PairStoreBytes(*molecule, *basis, *pairList) +
        qcx::integrals::internal::AuxKetsStoreBytes(*molecule, *aux, *auxPairList);
    const std::size_t sweepBytes =
        qcx::integrals::internal::SchwarzSweepBytes(*molecule, *basis, *pairList) +
        qcx::integrals::internal::SchwarzSweepBytes(*molecule, *aux, *auxPairList);
    const std::size_t sweepChargeBytes = sweepBytes > storeBytes ? sweepBytes - storeBytes : 0;
    const std::size_t budgetBytes = AllSurvivePatternBytes(nPairs) + 1024 + sweepChargeBytes;
#ifdef _DEBUG
    // The MSVC Debug STL's inflated per-vector bookkeeping (the
    // _Container_proxy class - see the ClassTableBytesArithmetic Debug pins)
    // grows the platform-computed light-rung terms beyond what the razor
    // can fit under the AllSurvive floor at this fixture's scale (measured:
    // the 61.9 MB Debug light estimate vs the 27.6 MB remaining budget; the
    // Release leg fits and pins the band machinery end to end). The Debug
    // leg skips the geometric premise it cannot hold.
    GTEST_SKIP() << "Debug STL inflation breaks the razor geometry (Release pins the band)";
#endif
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());
    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    const std::optional<FockModeInfo>& info = ri->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kFastPath);
    EXPECT_LE(info->predictedBytes, budgetBytes);
    EXPECT_EQ(info->reservedBytes, info->predictedBytes - info->exchangeBytes);
    // The nested exchange fires k = team at the
    // fired batch - the stack's committed charge is the k = 1 estimate
    // plus the k-fold expansion of that batch (155,684 B here), held
    // inside the band - so the total never sits below the k = 1 estimate
    // nor above the budget. Never-under is the honest form: the k-fold
    // excess is platform-arithmetic boundary-sensitive, and the equality
    // committed == predicted is a legitimate boundary.
    EXPECT_GE(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(budget->CommittedBytes(), budgetBytes);
    // The band is real: the exchange's k = team charge at the inherited
    // ~12.9-KB fired batch fits without any cap descent, leaving 6.1 MB
    // of the band unclaimed - while the remaining budget still sits below
    // the all-survive bound, so the nested's a-priori exclusion would
    // have refused without the validated sweep count.
    EXPECT_LT(budget->Remaining(), AllSurvivePatternBytes(nPairs));
}

TEST(Footprint, LightPathKnobForcesTheModeAndReserves) {
    // A non-zero lightPathChunkPairs FORCES the LightPath mode
    // with that chunk size - the budget-driven LightPath is unreachable at
    // the small end, where the pattern saving is zero and the light rung's
    // floor sits above the fast path's. The forced Create charges the
    // light estimate exactly and records every LightPath term.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    // Two arenas plus slack: the light path's non-scratch terms (the peak
    // chunk's arena at the knob's all-survive bounds) are larger than the
    // fast path's, so the slack must cover them for the unclamped pin.
    const std::size_t budgetBytes = 2 * DefaultArenaBytes() + std::size_t{4} * 1024 * 1024;
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    options.lightPathChunkPairs = 5;
    options.maxParallelChunks = 1;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::optional<FockModeInfo>& info = builder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kLightPath);
    EXPECT_EQ(info->chunkPairs, 5u);
    EXPECT_FALSE(info->patternExcluded);
    // The generous budget fits the unclamped arena - the reservation is
    // the exact CWA charge.
    EXPECT_EQ(info->scratchBytes, DefaultArenaBytes());
    EXPECT_EQ(info->reservedBytes, info->predictedBytes);
    EXPECT_EQ(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(info->predictedBytes, budgetBytes);
    // The light store: the 15 geometry-only entries. The chunk terms carry
    // the knob path's all-survive bounds (no counting pass on the forced
    // path): the arena pairs are the chunk's rows plus its kets, the
    // pattern is chunkPairs x nPairs rows.
    EXPECT_EQ(info->lightStoreBytes, 15 * sizeof(MdPairData));
    EXPECT_GT(info->chunkArenaBytes, 0u);
    EXPECT_EQ(info->chunkPatternBytes, 5 * 15 * 8u);
    // The per-call chunk-pair bookkeeping (the stamp vector and the chunk
    // pair set) and the retained flattened shells (5 shells) - the C1
    // terms, formula-consistent.
    EXPECT_EQ(info->chunkIndexBytes, 2 * (15 * sizeof(std::size_t) + kVectorHeaderBytes));
    EXPECT_EQ(info->lightShellsBytes, 5 * sizeof(MdShellInput) + kVectorHeaderBytes);
    EXPECT_GT(info->structuralBytes, 0u);
    EXPECT_EQ(info->cacheBytes, 0u);
    // The forced builder is usable end to end: the chunk loop screens,
    // fills and clears against the light store.
    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value());
    auto fock = builder->BuildFock(*densityTensor);
    ASSERT_TRUE(fock.has_value()) << fock.error().message;
}

TEST(Footprint, LightPathKnobBitIdentityPin) {
    // The mode-independent physics pin. The knob forces the
    // LightPath with chunkPairs = nPairs - ONE chunk, whose serial task
    // concatenation (the fp64 half, then the fp32 half) matches the
    // FastPath's exactly, so the contraction accumulation order is
    // identical and the Fock matrices are bit-identical. Smaller chunks
    // interleave the per-chunk fp64/fp32 halves and reorder the
    // accumulation - within the documented reduction-order tolerance
    // (fock_screen.hpp:358-362), not bit-identity - so the bit pin pins
    // the single-chunk form. Both presets are pinned: kNormal keeps the
    // certified fp32 lane on (the concatenation order matters), kTight is
    // the preset with the certified lane off (pure fp64).
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value());

    for (const qcx::integrals::AccuracyPreset preset :
         {qcx::integrals::AccuracyPreset::kNormal, qcx::integrals::AccuracyPreset::kTight})
    {
        FockBuildOptions legacyOptions;
        legacyOptions.accuracy = preset;
        legacyOptions.maxParallelChunks = 1;
        auto legacy = DirectJkFockBuilder::Create(*molecule, *basis, *core, legacyOptions);
        ASSERT_TRUE(legacy.has_value()) << legacy.error().message;
        EXPECT_FALSE(legacy->ModeInfo().has_value());
        auto legacyFock = legacy->BuildFock(*densityTensor);
        ASSERT_TRUE(legacyFock.has_value()) << legacyFock.error().message;

        // Two arenas plus slack - the light estimate must fit unclamped:
        // a clamp would change the batch composition against the legacy
        // reference and break the bit-identity comparison.
        auto budget =
            WorkspaceBudget::Create(2 * DefaultArenaBytes() + std::size_t{4} * 1024 * 1024);
        ASSERT_TRUE(budget.has_value());
        FockBuildOptions lightOptions;
        lightOptions.accuracy = preset;
        lightOptions.workspaceBudget = &*budget;
        // The knob forces the mode with the FULL bra range: one chunk.
        lightOptions.lightPathChunkPairs = 15;
        lightOptions.maxParallelChunks = 1;
        auto light = DirectJkFockBuilder::Create(*molecule, *basis, *core, lightOptions);
        ASSERT_TRUE(light.has_value()) << light.error().message;
        ASSERT_TRUE(light->ModeInfo().has_value());
        EXPECT_EQ(light->ModeInfo()->mode, FockBuildMode::kLightPath);
        auto lightFock = light->BuildFock(*densityTensor);
        ASSERT_TRUE(lightFock.has_value()) << lightFock.error().message;

        EXPECT_EQ((ToMatrix(*lightFock) - ToMatrix(*legacyFock)).norm(), 0.0);
    }
}

TEST(Footprint, LightPathKnobChunkSizesAgreeWithinReductionOrder) {
    // Every forced chunk size delivers the same Fock up to the
    // reduction-order tolerance (fock_screen.hpp:358-362). The smaller
    // chunks contract the per-chunk fp64/fp32 halves interleaved, so the
    // accumulation order differs from the full-space pass in the last
    // bits; the bit pin (chunkPairs = nPairs) is the separate forced-chunk test.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value());

    FockBuildOptions legacyOptions;
    legacyOptions.maxParallelChunks = 1;
    auto legacy = DirectJkFockBuilder::Create(*molecule, *basis, *core, legacyOptions);
    ASSERT_TRUE(legacy.has_value()) << legacy.error().message;
    auto legacyFock = legacy->BuildFock(*densityTensor);
    ASSERT_TRUE(legacyFock.has_value()) << legacyFock.error().message;

    for (const std::size_t knob : {std::size_t{1}, std::size_t{3}, std::size_t{15}})
    {
        // Unclamped for every knob size (the same slack note as the pin).
        auto budget =
            WorkspaceBudget::Create(2 * DefaultArenaBytes() + std::size_t{4} * 1024 * 1024);
        ASSERT_TRUE(budget.has_value());
        FockBuildOptions options;
        options.workspaceBudget = &*budget;
        options.lightPathChunkPairs = knob;
        options.maxParallelChunks = 1;
        auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
        ASSERT_TRUE(builder.has_value()) << builder.error().message;
        ASSERT_TRUE(builder->ModeInfo().has_value());
        EXPECT_EQ(builder->ModeInfo()->mode, FockBuildMode::kLightPath);
        EXPECT_EQ(builder->ModeInfo()->chunkPairs, knob);
        auto fock = builder->BuildFock(*densityTensor);
        ASSERT_TRUE(fock.has_value()) << fock.error().message;
        EXPECT_LE((ToMatrix(*fock) - ToMatrix(*legacyFock)).norm(), 1e-12) << "knob " << knob;
    }
}

TEST(Footprint, LightPathKnobTinyBudgetRefusesWithTheEstimate) {
    // The knob-forced light rung that still cannot fit: the forced path
    // skips the a-priori exclusion (the knob bypasses the fast estimate
    // entirely), so the refusal carries the fired light estimate sentence
    // - no exclusion clause - and charges nothing.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto budget = WorkspaceBudget::Create(1);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    options.lightPathChunkPairs = 5;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = builder.error().message;
    EXPECT_NE(message.find("footprint estimate is"), std::string::npos);
    EXPECT_NE(message.find("direct screened"), std::string::npos);
    EXPECT_NE(message.find("batched/blocked"), std::string::npos);
    EXPECT_NE(message.find("recompute"), std::string::npos);
    EXPECT_NE(message.find("disk"), std::string::npos);
    EXPECT_NE(message.find("light rung"), std::string::npos);
    EXPECT_EQ(message.find("neighbor pattern alone"), std::string::npos);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, LightPathCountedAdmissionAtScaleSelectsTheFastRung) {
    // The exclusion fix at the 70-water shape scale (the direct-builder
    // form): the 70-water STO-3G cluster has 350 shells and 61,425
    // canonical pairs, whose all-survive neighbor bound is ~15.1 GB - the
    // all-survive exclusion (i) would fire on a 12 GiB budget. The
    // monomers sit 20 Bohr apart, so the Schwarz-exact surviving-pair
    // count is ~552k (a ~4.4 MB pattern at 8 bytes a row), and the
    // counted exclusion admits the fast rung instead of forcing the
    // LightPath (the 80-carbon chain's misfire shape: a 24.44 GiB all-survive
    // bound against a ~38 MB counted pattern). The all-survive bound is
    // the premise (the pre-gate runs); the admission is the fix.
    // Create-only (no BuildFock - the fixture is the 70-water scale); the
    // counted sweep is the decision's own CSR build.
    auto molecule = MakeWaterCluster(70);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    // 70 waters x 5 shells (O S + O S + O P + H S + H S) = 350 shells ->
    // 350 x 351 / 2 = 61,425 canonical pairs.
    const std::size_t nPairs = pairList->pairs.size();
    ASSERT_EQ(nPairs, 61425u);

    const std::size_t budgetBytes = 12ull * 1024 * 1024 * 1024;
    // The exclusion premise: the all-survive bound alone exceeds the
    // budget (8 x 61,425 x 61,426 / 2 = 15,092,692,200 bytes), so the
    // pre-gate runs and the counted test decides.
    EXPECT_GT(AllSurvivePatternBytes(nPairs), budgetBytes);

    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::optional<FockModeInfo>& info = builder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    // The counted admission: the fast rung, no exclusion, the counted
    // pattern a tiny slice of the band (the fix's whole point).
    EXPECT_EQ(info->mode, FockBuildMode::kFastPath);
    EXPECT_FALSE(info->patternExcluded);
    EXPECT_GT(info->patternBytes, 0u);
    EXPECT_LE(info->patternBytes, AllSurvivePatternBytes(nPairs));
    EXPECT_LT(info->patternBytes, budgetBytes / 64);
    EXPECT_LE(info->predictedBytes, budgetBytes);
    EXPECT_EQ(info->reservedBytes, info->predictedBytes);
    EXPECT_EQ(budget->CommittedBytes(), info->predictedBytes);
}

TEST(Footprint, LightPathExclusionAtScaleRefusesBelowTheLightFloor) {
    // The same 70-water fixture at 1 MiB: the counted exclusion fires
    // (~552k surviving rows at 8 bytes = ~4.4 MB still cannot fit 1 MiB),
    // the light rung's own estimate (the light store plus the peak-chunk
    // terms) also cannot fit, and the refusal names the ladder with the
    // exclusion clause - nothing charged.
    auto molecule = MakeWaterCluster(70);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto budget = WorkspaceBudget::Create(std::size_t{1024} * 1024);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_FALSE(builder.has_value());
    EXPECT_EQ(builder.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string& message = builder.error().message;
    EXPECT_NE(message.find("counted Schwarz-screened neighbor pattern"), std::string::npos);
    EXPECT_NE(message.find("light rung"), std::string::npos);
    EXPECT_NE(message.find("direct screened"), std::string::npos);
    EXPECT_NE(message.find("batched/blocked"), std::string::npos);
    EXPECT_NE(message.find("recompute"), std::string::npos);
    EXPECT_NE(message.find("disk"), std::string::npos);
    EXPECT_EQ(budget->CommittedBytes(), 0u);
}

TEST(Footprint, LightPathExclusionBoundsTheChunkByTheCountedPeakRow) {
    // The exclusion path's peak-chunk bound, on the fixture whose own
    // window opens: C24H50 / def2-SVP, 586 basis functions and 43,365
    // canonical pairs (the point the RI-J print below uses). The
    // counted pattern is ~122M rows, so the budget the test derives below
    // (~978 MB, the exclusion band's top) fires the a-priori exclusion -
    // the position where the light rung is the ONLY in-memory fit left,
    // and so exactly the position where its own bound must not throw it
    // away.
    //
    // It did. The peak-chunk markers fell back to the whole pair space
    // (nPairs kets: (1 + 43,365) x the (d,d) pair's 14,936 B, ~647 MB of
    // arena, 1.06 GB with the fixed terms) - over this budget, so Create
    // refused. They now ride the exclusion's own counted peak row width (a
    // chunk's ket set is the union of its rows', each bounded by the widest
    // row), and the rung engages.
    //
    // The window is why THIS fixture: the exclusion's band is
    // (floor, 8 x counted) = (~553 MB, ~978 MB), wide here because the
    // pattern term grows super-quadratically while the light rung's fixed
    // terms do not. On the sparse 70-water cluster above, the same two
    // bands do not overlap at all - the quartet term alone (518 MB) exceeds
    // the whole exclusion band (4.4 MB) - so that fixture can never select
    // this rung; the sub-test there pins the refusal, this one the landing.
    //
    // The batch cap is pinned tiny on purpose: the light estimate's only
    // thread-scaled term is the batch arena (cap x team), and the
    // engagement must not turn on the machine's core count.
    auto molecule = qcx::testing::MakeAlkaneSto3g(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const std::array<int, 2> elements{6, 1}; // C, H.
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-svp", elements);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    const std::size_t nPairs = pairList->pairs.size();
    ASSERT_EQ(nPairs, 43365u);
    ASSERT_EQ(pairList->functionCount, 586u);

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;

    // The exclusion premise on the counted form itself, and the budget
    // DERIVED from it: the pre-gate fires iff `8 x counted > remaining`, so
    // the last budget that fires it is one byte under the band's top. The
    // fixture's own numbers choose the budget - no magic constant - and the
    // rung must fit inside that band or this fixture cannot host it at all.
    const std::size_t countedPattern =
        CountSchwarzSurvivingPairs(*schwarz, AccuracyPreset::kNormal);
    ASSERT_GT(countedPattern, 0u);
    const std::size_t exclusionBandTop = 8 * countedPattern;
    const std::size_t budgetBytes = exclusionBandTop - 1;
    EXPECT_GT(AllSurvivePatternBytes(nPairs), budgetBytes);

    // The bound that replaces the fallback, and the premise that it bites:
    // it is far under the pair space on this screened alkane.
    const std::size_t peakRow = CountSchwarzPeakRowWidth(*schwarz, AccuracyPreset::kNormal);
    ASSERT_LT(peakRow, nPairs);

    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    FockBuildOptions options;
    options.workspaceBudget = &*budget;
    options.maxBatchBytes = 4096;
    options.accuracy = AccuracyPreset::kNormal;
    auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(builder.has_value()) << "budget=" << budgetBytes << " counted=" << countedPattern
                                     << " peakRow=" << peakRow << ": " << builder.error().message;

    const std::optional<FockModeInfo>& info = builder->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kLightPath);
    EXPECT_TRUE(info->patternExcluded);
    EXPECT_GT(info->chunkPairs, 0u);
    // The peak chunk's counted rows are bounded by its rows' count times
    // the widest row - the all-survive product chunkPairs x nPairs is the
    // bound this replaces.
    EXPECT_LE(info->chunkPatternBytes, info->chunkPairs * peakRow * 8);
    EXPECT_LT(info->chunkPatternBytes, info->chunkPairs * nPairs * 8);
    EXPECT_LE(info->predictedBytes, budgetBytes);
}

TEST(Footprint, RiJkNestedExchangeReachesItsOwnLightRungOnTheCountedPreGate) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    // The RI-JK's nested exchange half at the band where its counted pattern cannot fit
    // but its LightPath can - the position the engine's own exclusion (i) used to REFUSE
    // (the counted pattern's 183.74 GiB of
    // fast-rung CSR charged against an 11.73 GB grant at the 4,974 manifest case, while
    // the same builder constructs the stack on the light rung one byte under that band -
    // LightPathExclusionBoundsTheChunkByTheCountedPeakRow above, which proves exactly
    // that for the DIRECT builder). The RI-JK must leave its nested half on the rung that
    // half's own rule selects (the validatedSweepCount skip is gone: it forced the fast
    // rung's whole-pattern CSR to be materialized before the light decision could be
    // made) and price it there (ri_engine.cpp priceExchange) - the three coupled changes.
    //
    // The band is FORCED by the budget, not by the system size, and both rungs are
    // recorded: the OUTER's tensor term
    // is 4.74 GB against a ~1.9 GB budget, so exclusion (ii) takes it to the LightPath,
    // and the nested's counted pre-gate takes ITS half to the LightPath on the pattern -
    // two independent decisions on two different objects, which is the point.
    //
    // The fixture is this file's own counted-admission fixture (the 586-point anchor,
    // LightPathExclusionBoundsTheChunkByTheCountedPeakRow's): C24H50 / def2-SVP, 586
    // functions and 43,365 pairs, whose counted pattern (122,319,961 rows) and
    // screened-quartet charge (535,834,900 B) are the two numbers that make the window
    // open at all - the band must clear the light rung's fixed terms, which is why the
    // small s-only fixtures cannot host it (at 24 waters the counted charge is 2.1 MB against
    // a 50 KB pattern). Both are asserted below from the fixture's own data.
    auto molecule = qcx::testing::MakeAlkaneSto3g(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const std::array<int, 2> elements{6, 1}; // C, H.
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-svp", elements);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-universal-jfit", elements);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());
    const std::size_t n = pairList->functionCount;
    const std::size_t nPairs = pairList->pairs.size();
    ASSERT_EQ(n, 586u);
    ASSERT_EQ(nPairs, 43365u);

    // The batch cap is pinned tiny on purpose (the mirror's note): the light estimate's
    // only thread-scaled term is the batch arena, and the band arithmetic must not turn
    // on the machine's core count.
    RiEngineOptions options;
    options.maxBatchBytes = 4096;
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value());
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value()) << schwarzAux.error().message;
    auto auxShells = FlattenShells(*molecule, *aux, *auxPairList);
    ASSERT_TRUE(auxShells.has_value());
    std::vector<double> auxShellBounds(auxShells->size());

    for (std::size_t shell = 0; shell < auxShells->size(); ++shell)
    {
        auxShellBounds[shell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(shell, shell, *auxPairList)];
    }

    const std::size_t screenedTaskCount =
        CountSchwarzSurvivingRiTasks(*schwarz, auxShellBounds, options.accuracy);
    const std::size_t countedPattern = CountSchwarzSurvivingPairs(*schwarz, options.accuracy);
    ASSERT_GT(countedPattern, 0u);
    // The window, from the fixture's own numbers: the counted pattern cannot fit the band
    // (the pre-gate's light decision) while the all-survive bound exceeds it (the
    // pre-gate's own trigger, so the count that decides is the counted one), and the
    // light rung's dominant fixed term sits under the band's top - the third condition is
    // what chooses this fixture rather than a smaller one.
    const std::size_t band = 8 * countedPattern - 1;
    EXPECT_GT(8 * countedPattern,
              qcx::integrals::internal::ScreenedQuartetBytes(n, options.accuracy));
    EXPECT_LT(8 * countedPattern, AllSurvivePatternBytes(nPairs));

    const qcx::integrals::internal::RiFootprintTerms outerTerms =
        qcx::integrals::internal::RiFootprint(*molecule,
                                              *basis,
                                              *aux,
                                              *pairList,
                                              *auxPairList,
                                              options.maxBatchBytes,
                                              threadCount,
                                              true,
                                              false,
                                              screenedTaskCount);
    const std::size_t budgetBytes = outerTerms.Total() + band;

    std::printf("\n  the nested exchange half's own band (C24H50 / def2-SVP)\n");
    std::printf("    counted pattern %zu | 8 x counted %zu B | light floor's dominant term %zu B\n",
                countedPattern,
                8 * countedPattern,
                qcx::integrals::internal::ScreenedQuartetBytes(n, options.accuracy));
    std::printf("    band %zu B | outer light RI terms %zu B | budget %zu B\n",
                band,
                outerTerms.Total(),
                budgetBytes);
    auto budget = WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value());

    options.workspaceBudget = &*budget;
    auto ri = RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << "budget=" << budgetBytes << " band=" << band
                                << " counted=" << countedPattern << ": " << ri.error().message;

    // The OUTER's record: the tensor exclusion took it to the light rung.
    const std::optional<FockModeInfo>& info = ri->ModeInfo();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->mode, FockBuildMode::kLightPath);
    EXPECT_TRUE(info->tensorExcluded);
    EXPECT_LE(info->predictedBytes, budgetBytes);

    // The nested half's own record - the rung the OUTER's record cannot name (the two are
    // decided on different objects: the outer on the RI tensor, the nested on the counted
    // pattern). patternExcluded is set in exactly one place (the counted pre-gate), so
    // this is the assertion that fails if the nested is forced off its own decision again.
    const std::optional<FockModeInfo>& nested = ri->ExchangeModeInfo();
    ASSERT_TRUE(nested.has_value());
    EXPECT_TRUE(nested->patternExcluded) << "the counted pre-gate was skipped";
    EXPECT_EQ(nested->mode, FockBuildMode::kLightPath);
    EXPECT_GT(nested->chunkPairs, 0u);
    EXPECT_EQ(nested->lightStoreBytes, nPairs * sizeof(MdPairData));
    const std::size_t peakRow = CountSchwarzPeakRowWidth(*schwarz, options.accuracy);
    ASSERT_LT(peakRow, nPairs);
    // The chunk's counted rows, bounded by its rows x the widest row - the all-survive
    // product chunkPairs x nPairs is the bound this replaces, and the whole-pattern term
    // (8 x counted, the budget itself here) is the object the fast rung would hold.
    EXPECT_LE(nested->chunkPatternBytes, nested->chunkPairs * peakRow * 8);
    EXPECT_LT(nested->chunkPatternBytes, nested->chunkPairs * nPairs * 8);
    EXPECT_LT(nested->chunkPatternBytes, 8 * countedPattern);
    // The charge the outer made is the object the nested half holds: the same functions
    // price both (internal/light_footprint.hpp), so the engine's exchange term and the
    // nested's own estimate are the same number - never the fast rung's whole-pattern
    // object, which is what the exclusion used to charge.
    EXPECT_EQ(info->exchangeBytes, nested->predictedBytes);
    EXPECT_LT(info->exchangeBytes, 8 * countedPattern);
    // The charge is checked against the builder that RUNS the rung: the direct
    // builder's own light decision at the same band, same batch, same accuracy is the
    // nested half's estimate by construction (the nested IS a DirectJkFockBuilder in
    // exchange-only mode, and internal/light_footprint.hpp gives the two decision layers
    // one definition of the arithmetic). Its budget is the band itself, which is exactly
    // the bytes the outer's reservation leaves the nested.
    {
        auto directBudget = WorkspaceBudget::Create(band);
        ASSERT_TRUE(directBudget.has_value());
        FockBuildOptions directOptions;
        directOptions.workspaceBudget = &*directBudget;
        directOptions.maxBatchBytes = options.maxBatchBytes;
        directOptions.accuracy = options.accuracy;
        auto direct = DirectJkFockBuilder::Create(*molecule, *basis, *core, directOptions);
        ASSERT_TRUE(direct.has_value()) << direct.error().message << " (band " << band << ")";
        const std::optional<FockModeInfo>& directInfo = direct->ModeInfo();
        ASSERT_TRUE(directInfo.has_value());
        EXPECT_EQ(directInfo->mode, FockBuildMode::kLightPath);
        EXPECT_TRUE(directInfo->patternExcluded);
        EXPECT_EQ(directInfo->predictedBytes, info->exchangeBytes);
    }

    EXPECT_LE(nested->predictedBytes, budgetBytes);

    // Reported, not asserted here: the movement this rung choice carries. The LightPath's
    // chunks interleave the per-chunk fp64/fp32 halves, so its accumulation order is not
    // the fast rung's single full-space pass and the Fock moves in the last bits - the
    // documented reduction-order tolerance (fock_screen.hpp:358-362), pinned for this same
    // chunk loop by LightPathKnobChunkSizesAgreeWithinReductionOrder (matrix norm <= 1e-12)
    // with bit-identity only at chunkPairs == nPairs (LightPathKnobBitIdentityPin). A
    // BuildFock pair at 586 functions is minutes and several GB - the fast reference alone
    // charges 5.62 GiB of nested exchange - so it does not run inside the routine suite.
    std::printf("    chunk %zu rows | nested light estimate %zu B | the movement is the light "
                "rung's documented reduction-order one\n",
                nested->chunkPairs,
                nested->predictedBytes);
}

TEST(Footprint, CountSchwarzSurvivingPairsMatchesTheSweep) {
    // The exclusion-count bit-identity on a real fixture (the H2O
    // decision fixture): the helper's count equals the sweep's own
    // candidateCount at every preset - same doubles, same cutoff, so a
    // pattern the exclusion admits is exactly the rows the builder
    // builds.
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;

    for (const AccuracyPreset preset :
         {AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight})
    {
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> indices;
        BuildNeighborList(*pairList, *schwarz, preset, offsets, indices);
        EXPECT_EQ(CountSchwarzSurvivingPairs(*schwarz, preset), indices.size())
            << static_cast<int>(preset);
    }
}

TEST(Footprint, CountSchwarzSurvivingPairsMatchesBruteForce) {
    // The two-pointer count against an O(n^2) reference over synthetic
    // bound vectors, at every preset: a random log-uniform draw (the
    // screening-profile shape), a quantized draw (heavy ties), and an
    // all-survive draw. The reference replicates the sweep's per-pair
    // product test over the canonical (bra, ket <= bra) pairs.
    std::mt19937_64 rng(20260905);
    std::uniform_real_distribution<double> logUniform(-14.0, 0.0);

    auto bruteForce = [](const std::vector<double>& schwarz, double threshold) {
        std::size_t count = 0;

        for (std::size_t bra = 0; bra < schwarz.size(); ++bra)
        {
            for (std::size_t ket = 0; ket <= bra; ++ket)
            {
                if (schwarz[bra] * schwarz[ket] >= threshold)
                {
                    ++count;
                }
            }
        }

        return count;
    };

    const std::size_t n = 220;

    for (const AccuracyPreset preset :
         {AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight})
    {
        const double threshold = SchwarzThreshold(preset) * kNeighborListSlack;

        for (std::size_t draw = 0; draw < 3; ++draw)
        {
            std::vector<double> values(n);

            for (std::size_t i = 0; i < n; ++i)
            {
                if (draw == 1)
                {
                    values[i] = std::pow(10.0, std::round(logUniform(rng)));
                } else if (draw == 2)
                {
                    values[i] = 1.0;
                } else
                {
                    values[i] = std::pow(10.0, logUniform(rng));
                }
            }

            EXPECT_EQ(CountSchwarzSurvivingPairs(values, preset), bruteForce(values, threshold))
                << "preset " << static_cast<int>(preset) << ", draw " << draw;
        }
    }
}

TEST(Footprint, CountSchwarzSurvivingPairsDenseAndDegenerate) {
    // The dense reproduction of the all-survive bound - the never-under
    // pole of the fix: the exact count must never admit a pattern the
    // all-survive form would have refused (admission only widens) - and
    // the degenerate shapes: empty, single-element, and all-dead bounds
    // (zero count at any positive cutoff).
    const std::vector<double> dense(37, 1.0);
    const std::size_t allSurviveCount = 37 * 38 / 2;
    EXPECT_EQ(CountSchwarzSurvivingPairs(dense, AccuracyPreset::kTight), allSurviveCount);
    EXPECT_EQ(AllSurvivePatternBytes(37), 8 * allSurviveCount);

    const std::vector<double> empty;
    EXPECT_EQ(CountSchwarzSurvivingPairs(empty, AccuracyPreset::kLoose), 0u);

    const std::vector<double> single{1.0};
    EXPECT_EQ(CountSchwarzSurvivingPairs(single, AccuracyPreset::kNormal), 1u);

    const std::vector<double> dead(64, 0.0);
    EXPECT_EQ(CountSchwarzSurvivingPairs(dead, AccuracyPreset::kLoose), 0u);
}

TEST(Footprint, CountSchwarzPeakRowWidthIsNeverUnderTheSweep) {
    // The LightPath peak-chunk ket bound: the counted replacement for the
    // all-survive nPairs fallback. The property that matters is
    // never-under - the bound must be at least the widest row the sweep
    // emits, or a chunk could be sized under what it must hold - and the
    // all-survive reproduction (the fallback it stands in for) at the
    // dense pole. The real-fixture arm builds the CSR and compares.
    const std::vector<double> dense(37, 1.0);
    EXPECT_EQ(CountSchwarzPeakRowWidth(dense, AccuracyPreset::kTight), 37u);

    const std::vector<double> empty;
    EXPECT_EQ(CountSchwarzPeakRowWidth(empty, AccuracyPreset::kLoose), 0u);

    const std::vector<double> single{1.0};
    EXPECT_EQ(CountSchwarzPeakRowWidth(single, AccuracyPreset::kNormal), 1u);

    const std::vector<double> dead(64, 0.0);
    EXPECT_EQ(CountSchwarzPeakRowWidth(dead, AccuracyPreset::kLoose), 0u);

    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;

    for (const AccuracyPreset preset :
         {AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight})
    {
        std::vector<std::size_t> offsets;
        std::vector<std::size_t> indices;
        BuildNeighborList(*pairList, *schwarz, preset, offsets, indices);

        std::size_t widestRow = 0;

        for (std::size_t bra = 0; bra + 1 < offsets.size(); ++bra)
        {
            widestRow = std::max(widestRow, offsets[bra + 1] - offsets[bra]);
        }

        const std::size_t bound = CountSchwarzPeakRowWidth(*schwarz, preset);
        EXPECT_GE(bound, widestRow) << static_cast<int>(preset);
        EXPECT_LE(bound, schwarz->size()) << static_cast<int>(preset);
    }
}

TEST(Footprint, CountSchwarzSurvivingPairsB5ShapeAdmitsTheBenchBand) {
    // The misfire numbers: the
    // C80H162/STO-3G chain's 402 shells -> 81,003 canonical pairs, whose
    // all-survive bound (26,246,268,048 B = 24.44 GiB) fired the exclusion
    // on any band below it and forced the LightPath on a system whose real
    // screened pattern was ~4.7M rows (~38 MB). The synthetic shape below
    // reproduces the geometry's sparsity class - a ~1,000-pair chemically
    // live core, everything else decayed below any preset's cutoff (the
    // inter-monomer pairs of a well-separated cluster): the exact count is
    // the live core's all-survive triangle, 500,500 rows, 4,004,000 B at
    // 8 bytes a row - admitted by the 12 GiB bench band that the
    // all-survive bound exceeds by more than 2x.
    const std::size_t nPairs = 81003;
    const std::size_t live = 1000;
    std::vector<double> schwarz(nPairs, 1e-300);

    for (std::size_t i = 0; i < live; ++i)
    {
        schwarz[i] = 1.0;
    }

    const std::size_t count = CountSchwarzSurvivingPairs(schwarz, AccuracyPreset::kTight);
    EXPECT_EQ(count, live * (live + 1) / 2);
    EXPECT_EQ(AllSurvivePatternBytes(nPairs), 26246268048u);
    EXPECT_GT(AllSurvivePatternBytes(nPairs), 12ull * 1024 * 1024 * 1024);
    EXPECT_LE(count * 8, 12ull * 1024 * 1024 * 1024);
}

TEST(Footprint, CountNearFieldPatternEntriesMatchesTheLeafDrivenBuild) {
    // The leaf-driven count's bit-identity with the CSR build it replaces
    // (the QFMM exclusion's bound): on a real H2O octree the helper
    // returns the pass-1 candidate count exactly - the same per-leaf
    // lists, the same product cutoff over the same near-field leaf pairs -
    // at every preset and across the near-field extent (theta = 0 covers
    // every leaf pair; theta = 0.7 the restricted set).
    auto molecule = qcx::testing::MakeH2oSto3g();
    auto basis = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(molecule.has_value());
    ASSERT_TRUE(basis.has_value());
    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto pairStore = BuildPairData(*molecule, *basis, *pairList);
    ASSERT_TRUE(pairStore.has_value());
    auto geometries =
        ComputePairGeometries(*pairStore, QfmmExtentForPreset(AccuracyPreset::kTight));
    auto tree = BuildQfmmTree(geometries);
    ASSERT_TRUE(tree.has_value());
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;

    for (const double theta : {0.0, 0.7})
    {
        std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
        std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
        BuildInteractionLists(*tree, theta, farFieldPairs, nearFieldLeafPairs);
        LeafNearFieldDomain domain;
        domain.nLeaves = tree->nodes.size();
        domain.leafOfPair = tree->leafOfPair;
        domain.nearFieldLeafPairs = std::move(nearFieldLeafPairs);

        for (const AccuracyPreset preset :
             {AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight})
        {
            std::vector<std::size_t> offsets;
            std::vector<std::size_t> indices;
            BuildLeafDrivenNeighborList(*pairList, *schwarz, preset, domain, offsets, indices);
            EXPECT_EQ(CountNearFieldPatternEntries(*schwarz, preset, domain),
                      offsets[pairList->pairs.size()])
                << "theta " << theta << ", preset " << static_cast<int>(preset);
        }
    }
}

TEST(Footprint, CountSchwarzSurvivingRiTasksMatchesTheSweep) {
    // The count's bit-identity on a real RI-J fixture (the
    // 24-water kWaterSto3g + kRiBandAux band fixture): the helper's count
    // equals the build's per-cell walk at every preset - same doubles,
    // same cutoff (plain SchwarzThreshold, no neighbor-list slack), so the
    // exact reserve the engine realizes is exactly the cells the fill
    // pushes. The walk replicates BuildScreenedRiTaskList's per-cell test
    // over the (orbital pair x aux shell) grid; the aux side is the pair
    // list's diagonal (the kets are single shells).
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value());
    auto basis = qcx::basisset::ParseNwchemText(kWaterSto3g);
    ASSERT_TRUE(basis.has_value());
    auto aux = qcx::basisset::ParseNwchemText(kRiBandAux);
    ASSERT_TRUE(aux.has_value());

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value());
    auto auxPairList = BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value());
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value());
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value());
    std::vector<double> auxShellBounds(auxPairList->shells.size());

    for (std::size_t shell = 0; shell < auxPairList->shells.size(); ++shell)
    {
        auxShellBounds[shell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(shell, shell, *auxPairList)];
    }

    for (const AccuracyPreset preset :
         {AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight})
    {
        const double threshold = SchwarzThreshold(preset);
        std::size_t sweepCount = 0;

        for (const double orbitalBound : *schwarz)
        {
            for (const double auxBound : auxShellBounds)
            {
                if (orbitalBound * auxBound >= threshold)
                {
                    ++sweepCount;
                }
            }
        }

        EXPECT_EQ(CountSchwarzSurvivingRiTasks(*schwarz, auxShellBounds, preset), sweepCount)
            << static_cast<int>(preset);
        EXPECT_LE(sweepCount, pairList->pairs.size() * auxPairList->shells.size());
    }
}

TEST(Footprint, CountSchwarzSurvivingRiTasksMatchesBruteForce) {
    // The two-pointer count against an O(rows x cols) reference over
    // synthetic bound vectors, at every preset: a random log-uniform draw
    // (the screening-profile shape), a quantized draw (heavy ties), and an
    // all-survive draw. The reference replicates BuildScreenedRiTaskList's
    // per-cell product test - plain SchwarzThreshold(preset), NO
    // kNeighborListSlack (the 3c screen is preset-level, unlike the
    // neighbor-list cutoff, which multiplies in the slack).
    std::mt19937_64 rng(20260908);
    std::uniform_real_distribution<double> logUniform(-14.0, 0.0);

    // (orbital, auxShell) name the bra orbital and the aux shell, in that order.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    auto bruteForce = [](const std::vector<double>& orbital,
                         const std::vector<double>& auxShell,
                         double threshold) {
        std::size_t count = 0;

        for (const double orbitalBound : orbital)
        {
            for (const double auxBound : auxShell)
            {
                if (orbitalBound * auxBound >= threshold)
                {
                    ++count;
                }
            }
        }

        return count;
    };

    const std::size_t rows = 250;
    const std::size_t cols = 90;

    for (const AccuracyPreset preset :
         {AccuracyPreset::kLoose, AccuracyPreset::kNormal, AccuracyPreset::kTight})
    {
        const double threshold = SchwarzThreshold(preset);

        for (std::size_t draw = 0; draw < 3; ++draw)
        {
            std::vector<double> orbital(rows);
            std::vector<double> auxShell(cols);

            for (std::size_t i = 0; i < rows; ++i)
            {
                if (draw == 1)
                {
                    orbital[i] = std::pow(10.0, std::round(logUniform(rng)));
                } else if (draw == 2)
                {
                    orbital[i] = 1.0;
                } else
                {
                    orbital[i] = std::pow(10.0, logUniform(rng));
                }
            }

            for (std::size_t i = 0; i < cols; ++i)
            {
                if (draw == 1)
                {
                    auxShell[i] = std::pow(10.0, std::round(logUniform(rng)));
                } else if (draw == 2)
                {
                    auxShell[i] = 1.0;
                } else
                {
                    auxShell[i] = std::pow(10.0, logUniform(rng));
                }
            }

            EXPECT_EQ(CountSchwarzSurvivingRiTasks(orbital, auxShell, preset),
                      bruteForce(orbital, auxShell, threshold))
                << "preset " << static_cast<int>(preset) << ", draw " << draw;
        }
    }
}

TEST(Footprint, CountSchwarzSurvivingRiTasksDenseAndDegenerate) {
    // The dense reproduction of the unconditional grid - the never-under
    // pole of the fix: an all-survive grid must count every
    // (row, col) cell, reproducing the pre-fix reserve the footprint's
    // taskListBytes used to charge unconditionally (admission only widens;
    // the charge and the realized exact reserve are the same number) - and
    // the degenerate shapes: empty vectors and all-dead bounds (zero count
    // at any positive cutoff).
    const std::vector<double> denseRows(37, 1.0);
    const std::vector<double> denseCols(53, 1.0);
    EXPECT_EQ(CountSchwarzSurvivingRiTasks(denseRows, denseCols, AccuracyPreset::kTight),
              37u * 53u);

    const std::vector<double> empty;
    EXPECT_EQ(CountSchwarzSurvivingRiTasks(empty, denseCols, AccuracyPreset::kLoose), 0u);
    EXPECT_EQ(CountSchwarzSurvivingRiTasks(denseRows, empty, AccuracyPreset::kLoose), 0u);

    const std::vector<double> single{1.0};
    EXPECT_EQ(CountSchwarzSurvivingRiTasks(single, single, AccuracyPreset::kNormal), 1u);

    const std::vector<double> dead(64, 0.0);
    const std::vector<double> live(8, 1.0);
    EXPECT_EQ(CountSchwarzSurvivingRiTasks(dead, live, AccuracyPreset::kLoose), 0u);
    EXPECT_EQ(CountSchwarzSurvivingRiTasks(live, dead, AccuracyPreset::kLoose), 0u);
}

TEST(Footprint, CountSchwarzSurvivingRiTasksSparseGridShrinksTheDenseReserve) {
    // The 5000-shape story: the pre-fix
    // taskListBytes charged the unconditional grid - ~127 GB at n = 5000
    // (2.646e6 pairs x ~3,000 shells x 16 B). The synthetic shape below
    // reproduces the geometry's sparsity class - a ~2,000-row chemically
    // live core over a wide grid, everything else decayed below any
    // preset's cutoff: the counted charge is the live rows' cells x 16 B
    // (the number the engine's exact reserve realizes - never-under by
    // equality), while the dense grid stays the footprint's no-count
    // reference.
    const std::size_t rows = 500000;
    const std::size_t cols = 3000;
    const std::size_t liveRows = 2000;
    std::vector<double> orbital(rows, 1e-300);
    std::vector<double> auxShell(cols, 1.0);

    for (std::size_t i = 0; i < liveRows; ++i)
    {
        orbital[i] = 1.0;
    }

    const std::size_t count =
        CountSchwarzSurvivingRiTasks(orbital, auxShell, AccuracyPreset::kTight);
    EXPECT_EQ(count, liveRows * cols);
    const std::size_t denseBytes = rows * cols * 2 * sizeof(std::size_t);
    EXPECT_GT(denseBytes, 16ull * 1024 * 1024 * 1024);
    EXPECT_LE(count * 2 * sizeof(std::size_t), 100ull * 1024 * 1024);
    EXPECT_EQ(count * 2 * sizeof(std::size_t), count * 16);
}

// The free 586-point structural print: the engine's
// own Create-time footprint at C24H50 / def2-SVP, straight from the RI-J
// family's own estimate functions, with NO SCF run.
//
// The caveat it closes: every 586-basis-function memory number on record
// before this print is a WHOLE-RUN RSS proxy - the job-object peak commit of
// c24h50/def2-SVP manifest runs, 8.473-10.818 GiB - never the engine's own
// structural Create-time figure. The print composes its inputs exactly as the
// RI-J family's own Create-time decision does (ri_engine.cpp's attempt
// lambda): the two canonical pair lists, the Schwarz bounds both charges are
// counted from, the counted screened task list, the engine's own default
// batch cap and the resolved OpenMP team. Its nested-exchange term is the
// fast rung's price (the engine prices that half at the rung its own rule
// selects - priceExchange; at this manifest band the counted pattern cannot
// fit, so an admitted run charges the LightPath's chunk-sized object
// instead).
//
// The fixture is the shared alkane builder at 24 carbons - C24H50, whose
// geometry reproduces the manifest case's exactly (C-C 1.538 A, C-H 1.09 A,
// tetrahedral backbone; 24 x 14 + 50 x 5 = 586 def2-SVP basis functions) -
// with the manifest's def2-universal-jfit aux set.
//
// The number is a MEASUREMENT, so this test PRINTS it and pins only what must
// hold of any honest print: the 586 count, the screening having engaged, the
// counted-charge identity, the documented tensor term, and the
// rung ordering. The byte values themselves are deliberately NOT pinned -
// footprint.hpp's formulas are documented to change with the engine, and a
// frozen copy here would re-introduce exactly the drift this print exists to
// expose. Both rungs are printed because the engine's choice between them is
// the budget decision's, not this print's.
TEST(RiFootprint, PrintsTheEngineCreateTimeFootprintAtC24H50Def2Svp) {
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const std::array<int, 2> elements{6, 1}; // C, H.
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-svp", elements);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-universal-jfit", elements);
    ASSERT_TRUE(auxBasis.has_value()) << auxBasis.error().message;

    auto pairList = BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = BuildShellPairs(*molecule, *auxBasis);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const std::size_t n = pairList->functionCount;
    ASSERT_EQ(n, 586u) << "the fixture must land on the manifest case's 586 basis functions";

    // The Schwarz bounds and the counted task list, built exactly as
    // BuildScreenedRiTaskList builds them (ri_engine.cpp): the aux kets are
    // single shells, so the per-shell bound is the diagonal pair Q_P.
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *auxBasis);
    ASSERT_TRUE(schwarzAux.has_value()) << schwarzAux.error().message;
    auto auxShells = FlattenShells(*molecule, *auxBasis, *auxPairList);
    ASSERT_TRUE(auxShells.has_value()) << auxShells.error().message;

    std::vector<double> schwarzAuxShell(auxShells->size());

    for (std::size_t auxShell = 0; auxShell < auxShells->size(); ++auxShell)
    {
        schwarzAuxShell[auxShell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(auxShell, auxShell, *auxPairList)];
    }

    const AccuracyPreset accuracy = AccuracyPreset::kNormal; // the manifest case's preset.
    const std::size_t screenedTaskCount =
        CountSchwarzSurvivingRiTasks(*schwarz, schwarzAuxShell, accuracy);
    const std::size_t patternCount = CountSchwarzSurvivingPairs(*schwarz, accuracy);
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const std::size_t maxBatchBytes = RiEngineOptions{}.maxBatchBytes;
    const std::size_t denseTaskGrid = pairList->pairs.size() * auxShells->size();

    using qcx::integrals::internal::DirectFootprint;
    using qcx::integrals::internal::DirectFootprintTerms;
    using qcx::integrals::internal::RiFootprint;
    using qcx::integrals::internal::RiFootprintTerms;

    const RiFootprintTerms fast = RiFootprint(*molecule,
                                              *basis,
                                              *auxBasis,
                                              *pairList,
                                              *auxPairList,
                                              maxBatchBytes,
                                              threadCount,
                                              false,
                                              false,
                                              screenedTaskCount);
    const RiFootprintTerms lightBlocked = RiFootprint(*molecule,
                                                      *basis,
                                                      *auxBasis,
                                                      *pairList,
                                                      *auxPairList,
                                                      maxBatchBytes,
                                                      threadCount,
                                                      true,
                                                      true,
                                                      screenedTaskCount);
    // The nested direct-exchange half: the RI-J always runs it in
    // buildExchangeOnly mode, and it does not depend on the RI rung (the
    // engine passes it the same batch cap on both).
    const DirectFootprintTerms exchange = DirectFootprint(*molecule,
                                                          *basis,
                                                          *pairList,
                                                          patternCount,
                                                          maxBatchBytes,
                                                          threadCount,
                                                          false,
                                                          0,
                                                          accuracy,
                                                          true);

    std::printf("\n  the engine's own Create-time footprint, C24H50/def2-SVP\n");
    std::printf("    (no SCF run: the pair lists and the estimates only)\n");
    std::printf("  n = %zu | orbital pairs %zu | aux shells %zu | aux functions %zu\n",
                n,
                pairList->pairs.size(),
                auxShells->size(),
                auxPairList->functionCount);
    std::printf("  screened RI tasks %zu of the dense %zu | surviving pattern %zu\n",
                screenedTaskCount,
                denseTaskGrid,
                patternCount);
    std::printf(
        "  OpenMP team %zu | batch cap %zu B (the engine default)\n", threadCount, maxBatchBytes);

    std::printf("\n  RiFootprintTerms - the FAST rung (the first attempt)\n");
    PrintFootprintTerm("orbitalStoreBytes", fast.orbitalStoreBytes);
    PrintFootprintTerm("auxStoreBytes", fast.auxStoreBytes);
    PrintFootprintTerm("taskListBytes", fast.taskListBytes);
    PrintFootprintTerm("tensorBytes", fast.tensorBytes);
    PrintFootprintTerm("riMatrixBytes", fast.riMatrixBytes);
    PrintFootprintTerm("metricBytes", fast.metricBytes);
    PrintFootprintTerm("scratchBytes", fast.scratchBytes);
    PrintFootprintTerm("transposeBytes", fast.transposeBytes);
    PrintFootprintTerm("sliceBytes", fast.sliceBytes);
    PrintFootprintTerm("valuesBufferBytes", fast.valuesBufferBytes);
    PrintFootprintTerm("smallVectorsBytes", fast.smallVectorsBytes);
    PrintFootprintTerm("retainedListBytes", fast.retainedListBytes);
    PrintFootprintTerm("RiFootprintTerms::Total", fast.Total());

    std::printf("\n  RiFootprintTerms - the LIGHT + BLOCKED-METRIC rung (the ladder's next)\n");
    PrintFootprintTerm("orbitalStoreBytes", lightBlocked.orbitalStoreBytes);
    PrintFootprintTerm("auxStoreBytes", lightBlocked.auxStoreBytes);
    PrintFootprintTerm("taskListBytes", lightBlocked.taskListBytes);
    PrintFootprintTerm("tensorBytes", lightBlocked.tensorBytes);
    PrintFootprintTerm("riMatrixBytes", lightBlocked.riMatrixBytes);
    PrintFootprintTerm("metricBytes", lightBlocked.metricBytes);
    PrintFootprintTerm("scratchBytes", lightBlocked.scratchBytes);
    PrintFootprintTerm("transposeBytes", lightBlocked.transposeBytes);
    PrintFootprintTerm("sliceBytes", lightBlocked.sliceBytes);
    PrintFootprintTerm("valuesBufferBytes", lightBlocked.valuesBufferBytes);
    PrintFootprintTerm("smallVectorsBytes", lightBlocked.smallVectorsBytes);
    PrintFootprintTerm("retainedListBytes", lightBlocked.retainedListBytes);
    PrintFootprintTerm("RiFootprintTerms::Total", lightBlocked.Total());

    // The nested half is the one term of the estimate the engine prices at the RUNG
    // (ri_engine.cpp priceExchange): this print shows the fast rung's DirectFootprint,
    // the object a nested on the fast rung holds - the LightPath's chunk-sized price
    // for the same band is what RiJkNestedExchangeReachesItsOwnLightRungOnTheCountedPreGate
    // pins.
    std::printf("\n  DirectFootprintTerms - the nested exchange half, the FAST rung's price\n");
    PrintFootprintTerm("pairStoreBytes", exchange.pairStoreBytes);
    PrintFootprintTerm("patternBytes", exchange.patternBytes);
    PrintFootprintTerm("scratchBytes", exchange.scratchBytes);
    PrintFootprintTerm("structuralBytes", exchange.structuralBytes);
    PrintFootprintTerm("cacheBytes", exchange.cacheBytes);
    PrintFootprintTerm("exchangePerCallBytes", exchange.exchangePerCallBytes);
    PrintFootprintTerm("classTableBytes", exchange.classTableBytes);
    PrintFootprintTerm("screenedQuartetBytes", exchange.screenedQuartetBytes);
    PrintFootprintTerm("DirectFootprintTerms::Total", exchange.Total());

    // The engine's own predictedBytes on each rung (ri_engine.cpp: the RI
    // terms plus the nested exchange half's, the reservation subject).
    std::printf("\n  predictedBytes (fast)          %14zu B  %9.4f GiB\n",
                fast.Total() + exchange.Total(),
                Gibibytes(fast.Total() + exchange.Total()));
    std::printf("  predictedBytes (light+blocked) %14zu B  %9.4f GiB\n\n",
                lightBlocked.Total() + exchange.Total(),
                Gibibytes(lightBlocked.Total() + exchange.Total()));

    // The invariants any honest print must satisfy. The screening engaged:
    // the counted charge is the realized reserve only if it is the surviving
    // subset, never the dense grid (or the whole point of the count is lost).
    EXPECT_GT(screenedTaskCount, 0u);
    EXPECT_LT(screenedTaskCount, denseTaskGrid);
    EXPECT_GT(patternCount, 0u);

    // The counted treatment, never-under by equality: the
    // charged task list is the surviving count x 16 B per RiTask.
    EXPECT_EQ(fast.taskListBytes, screenedTaskCount * 2 * sizeof(std::size_t));
    EXPECT_EQ(lightBlocked.taskListBytes, fast.taskListBytes);

    // The rungs' defining difference: the light rung never builds the (uv|P)
    // values buffer or the retained Eigen copy.
    EXPECT_EQ(lightBlocked.tensorBytes, 0u);
    EXPECT_EQ(lightBlocked.riMatrixBytes, 0u);
    EXPECT_EQ(fast.tensorBytes, 8 * n * n * auxPairList->functionCount);
    EXPECT_EQ(fast.riMatrixBytes, fast.tensorBytes);

    // The rung ordering the evidence ledger reads: light+blocked is below
    // fast on the same inputs, and the exchange half is charged on top.
    EXPECT_LT(lightBlocked.Total(), fast.Total());
    EXPECT_GT(exchange.Total(), 0u);
}

// ---------------------------------------------------------------------------
// The 5000-basis scale print
//
// The 5000-basis scale target names five paths - direct screened, QFMM,
// RI-J fast, RI-J light, and the streaming rung - and asks each path's peak
// under the 16 GiB cap regime. This test answers it from
// the ENGINE'S OWN estimate functions: no SCF, no benchmark, no cap and no
// timed pass - the pair lists, the Schwarz bounds and the estimates every
// Create-time decision already builds, and nothing else.
//
// The instrument is the free 586-point structural print of
// RiFootprint.PrintsTheEngineCreateTimeFootprintAtC24H50Def2Svp, applied at
// scale. The target fixture is tools/bench/cases/c42h86_def2qzvp_rhf.toml -
// C42H86 def2-QZVP, 4,974 spherical functions over 1,574 shells - whose
// geometry is the shared alkane builder at 42 carbons (the manifest file is
// that builder's Bohr geometry converted to Angstrom). Two smaller rungs
// of the SAME generator in the SAME def2-QZVP basis carry the scaling check
// (C6H14, C12H26), so each term's measured growth can be read against the
// growth its own formula predicts without a basis-set change confounding it;
// the 586-function def2-SVP anchor is printed first so
// every ratio is also readable against the one already-measured point.
//
// The numbers are a MEASUREMENT: this test PRINTS them and pins only counts
// and identities that must hold of any honest print. It is gated OFF by
// default (QCX_FOOTPRINT_SCALE_PRINT=1 - the QCX_REFERENCE_REGENERATION_CHECK
// convention, local-only and never in CI) because the C42H86 point's Schwarz
// sweep is minutes, far outside a routine suite's budget.
// ---------------------------------------------------------------------------
namespace {

/// One rung of the scale ladder: the alkane carbon count and the two basis
/// directories.
struct ScaleLadderPoint {
    const char* label;
    std::size_t carbonCount;
    const char* orbitalBasisDir;
    const char* auxBasisDir;
};

/// Prints every path's every term at one ladder point and returns the point's
/// orbital function count (0 when an input failed to build - the error is
/// printed and the caller skips the point).
std::size_t PrintScaleLadderPoint(const ScaleLadderPoint& point) {
    using qcx::integrals::internal::QfmmOuterStoreBytes;
    using qcx::integrals::internal::RiFootprint;
    using qcx::integrals::internal::RiFootprintTerms;
    using qcx::integrals::internal::ScreenedQuartetBytes;

    auto molecule = qcx::testing::MakeAlkaneSto3g(point.carbonCount);

    if (!molecule.has_value())
    {
        std::printf(
            "  %s: the molecule failed: %s\n", point.label, molecule.error().message.c_str());
        return 0;
    }

    const std::array<int, 2> elements{6, 1}; // C, H.
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/" + point.orbitalBasisDir, elements);

    if (!basis.has_value())
    {
        std::printf(
            "  %s: the orbital basis failed: %s\n", point.label, basis.error().message.c_str());
        return 0;
    }

    auto auxBasis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/" + point.auxBasisDir, elements);

    if (!auxBasis.has_value())
    {
        std::printf(
            "  %s: the aux basis failed: %s\n", point.label, auxBasis.error().message.c_str());
        return 0;
    }

    auto pairList = BuildShellPairs(*molecule, *basis);

    if (!pairList.has_value())
    {
        std::printf(
            "  %s: the pair list failed: %s\n", point.label, pairList.error().message.c_str());
        return 0;
    }

    auto auxPairList = BuildShellPairs(*molecule, *auxBasis);

    if (!auxPairList.has_value())
    {
        std::printf("  %s: the aux pair list failed: %s\n",
                    point.label,
                    auxPairList.error().message.c_str());
        return 0;
    }

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    if (!schwarz.has_value())
    {
        std::printf(
            "  %s: the Schwarz bounds failed: %s\n", point.label, schwarz.error().message.c_str());
        return 0;
    }

    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *auxBasis);

    if (!schwarzAux.has_value())
    {
        std::printf("  %s: the aux Schwarz bounds failed: %s\n",
                    point.label,
                    schwarzAux.error().message.c_str());
        return 0;
    }

    auto auxShells = FlattenShells(*molecule, *auxBasis, *auxPairList);

    if (!auxShells.has_value())
    {
        std::printf(
            "  %s: the aux shells failed: %s\n", point.label, auxShells.error().message.c_str());
        return 0;
    }

    const std::size_t n = pairList->functionCount;
    const std::size_t nShells = pairList->shells.size();
    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t nAuxShells = auxShells->size();
    const std::size_t nAuxFuncs = auxPairList->functionCount;
    std::vector<double> auxShellBounds(nAuxShells);

    for (std::size_t auxShell = 0; auxShell < nAuxShells; ++auxShell)
    {
        auxShellBounds[auxShell] =
            (*schwarzAux)[qcx::integrals::PairIndexOf(auxShell, auxShell, *auxPairList)];
    }

    const AccuracyPreset accuracy = AccuracyPreset::kNormal; // the manifest cases' preset.
    const std::size_t screenedTaskCount =
        CountSchwarzSurvivingRiTasks(*schwarz, auxShellBounds, accuracy);
    const std::size_t patternCount = CountSchwarzSurvivingPairs(*schwarz, accuracy);
    const std::size_t threadCount = static_cast<std::size_t>(qcx::backend::DefaultOmpTeamSize());
    const std::size_t maxBatchBytes = RiEngineOptions{}.maxBatchBytes;

    std::printf("\n===== %s =====\n", point.label);
    std::printf("  n = %zu | shells %zu | orbital pairs %zu | aux shells %zu | aux functions %zu\n",
                n,
                nShells,
                nPairs,
                nAuxShells,
                nAuxFuncs);
    std::printf(
        "  screened RI tasks %zu of the dense %zu | surviving pattern %zu of the dense %zu\n",
        screenedTaskCount,
        nPairs * nAuxShells,
        patternCount,
        nPairs * (nPairs + 1) / 2);
    std::printf("  OpenMP team %zu | batch cap %zu B\n", threadCount, maxBatchBytes);
    std::fflush(stdout);

    // The lean direct default (the no-builder
    // default at nBasis <= 1000): its envelope is a SINGLE opaque total - the
    // builder's own five-term decomposition is computed in
    // lean_fock_build.cpp's EstimatePeakBytes and is NOT exported.
    auto lean = qcx::integrals::EstimateLeanEnvelope(
        *molecule, *basis, qcx::integrals::LeanFockBuildOptions{});

    if (!lean.has_value())
    {
        std::printf("  the lean envelope failed: %s\n", lean.error().message.c_str());
    } else
    {
        std::printf("\n  LEAN direct (EstimateLeanEnvelope, terms not exported)\n");
        PrintFootprintTerm("lean total", static_cast<std::size_t>(lean->totalBytes));
    }

    // The direct screened family's FastPath estimate, composed exactly as
    // fock_build.cpp's decision block composes it: the counted survivor
    // pattern, the engine's batch cap, no cache grant (the cache is a
    // budget-derived add-on, and at this size no budget survives it), the k=1
    // slot model.
    const auto directFast = qcx::integrals::internal::DirectFootprint(*molecule,
                                                                      *basis,
                                                                      *pairList,
                                                                      patternCount,
                                                                      maxBatchBytes,
                                                                      threadCount,
                                                                      false,
                                                                      0,
                                                                      accuracy,
                                                                      true);

    std::printf("\n  DIRECT SCREENED - the FastPath rung\n");
    PrintFootprintTerm("pairStoreBytes", directFast.pairStoreBytes);
    PrintFootprintTerm("patternBytes", directFast.patternBytes);
    PrintFootprintTerm("scratchBytes", directFast.scratchBytes);
    PrintFootprintTerm("structuralBytes", directFast.structuralBytes);
    PrintFootprintTerm("cacheBytes", directFast.cacheBytes);
    PrintFootprintTerm("exchangePerCallBytes", directFast.exchangePerCallBytes);
    PrintFootprintTerm("classTableBytes", directFast.classTableBytes);
    PrintFootprintTerm("screenedQuartetBytes", directFast.screenedQuartetBytes);
    PrintFootprintTerm("DirectFootprintTerms::Total", directFast.Total());
    std::printf("    (the class-table ceiling ClassTableBytes = %zu B would DISENGAGE the\n",
                ClassTableBytes(nPairs, n));
    std::printf("     class path at this size; charged 0 above, the plain-path form)\n");
    std::fflush(stdout);

    // The RI-J family's two rungs, composed exactly as ri_engine.cpp's
    // attempt lambda composes them, plus the nested direct-exchange half the
    // RI-J always runs (buildExchangeOnly mode, rung-independent).
    const auto exchange = qcx::integrals::internal::DirectFootprint(*molecule,
                                                                    *basis,
                                                                    *pairList,
                                                                    patternCount,
                                                                    maxBatchBytes,
                                                                    threadCount,
                                                                    false,
                                                                    0,
                                                                    accuracy,
                                                                    true);
    const RiFootprintTerms fast = RiFootprint(*molecule,
                                              *basis,
                                              *auxBasis,
                                              *pairList,
                                              *auxPairList,
                                              maxBatchBytes,
                                              threadCount,
                                              false,
                                              false,
                                              screenedTaskCount);
    const RiFootprintTerms lightBlocked = RiFootprint(*molecule,
                                                      *basis,
                                                      *auxBasis,
                                                      *pairList,
                                                      *auxPairList,
                                                      maxBatchBytes,
                                                      threadCount,
                                                      true,
                                                      true,
                                                      screenedTaskCount);

    std::printf("\n  RI-J - the FAST rung (the tensor rung)\n");
    PrintFootprintTerm("orbitalStoreBytes", fast.orbitalStoreBytes);
    PrintFootprintTerm("auxStoreBytes", fast.auxStoreBytes);
    PrintFootprintTerm("taskListBytes", fast.taskListBytes);
    PrintFootprintTerm("tensorBytes", fast.tensorBytes);
    PrintFootprintTerm("riMatrixBytes", fast.riMatrixBytes);
    PrintFootprintTerm("metricBytes", fast.metricBytes);
    PrintFootprintTerm("scratchBytes", fast.scratchBytes);
    PrintFootprintTerm("transposeBytes", fast.transposeBytes);
    PrintFootprintTerm("sliceBytes", fast.sliceBytes);
    PrintFootprintTerm("valuesBufferBytes", fast.valuesBufferBytes);
    PrintFootprintTerm("smallVectorsBytes", fast.smallVectorsBytes);
    PrintFootprintTerm("retainedListBytes", fast.retainedListBytes);
    PrintFootprintTerm("RiFootprintTerms::Total", fast.Total());

    std::printf("\n  RI-J - the LIGHT + BLOCKED-METRIC rung (the recompute rung)\n");
    PrintFootprintTerm("orbitalStoreBytes", lightBlocked.orbitalStoreBytes);
    PrintFootprintTerm("auxStoreBytes", lightBlocked.auxStoreBytes);
    PrintFootprintTerm("taskListBytes", lightBlocked.taskListBytes);
    PrintFootprintTerm("tensorBytes", lightBlocked.tensorBytes);
    PrintFootprintTerm("riMatrixBytes", lightBlocked.riMatrixBytes);
    PrintFootprintTerm("metricBytes", lightBlocked.metricBytes);
    PrintFootprintTerm("scratchBytes", lightBlocked.scratchBytes);
    PrintFootprintTerm("transposeBytes", lightBlocked.transposeBytes);
    PrintFootprintTerm("sliceBytes", lightBlocked.sliceBytes);
    PrintFootprintTerm("valuesBufferBytes", lightBlocked.valuesBufferBytes);
    PrintFootprintTerm("smallVectorsBytes", lightBlocked.smallVectorsBytes);
    PrintFootprintTerm("retainedListBytes", lightBlocked.retainedListBytes);
    PrintFootprintTerm("RiFootprintTerms::Total", lightBlocked.Total());

    std::printf(
        "\n  the nested direct-exchange half (rung-independent, on top of either RI rung)\n");
    PrintFootprintTerm("DirectFootprintTerms::Total", exchange.Total());

    std::printf("\n  the engine's own predictedBytes per RI rung (RI terms + the exchange half)\n");
    std::printf("    fast           %14zu B  %9.4f GiB\n",
                fast.Total() + exchange.Total(),
                Gibibytes(fast.Total() + exchange.Total()));
    std::printf("    light+blocked  %14zu B  %9.4f GiB\n",
                lightBlocked.Total() + exchange.Total(),
                Gibibytes(lightBlocked.Total() + exchange.Total()));

    // The three analytic ceilings the admission reads beside the composed
    // terms: the streaming rung's whole-call task-master survivor, the class
    // table and the all-survive pattern bound.
    std::printf("\n  the analytic ceilings\n");
    PrintFootprintTerm("ScreenedQuartetBytes", ScreenedQuartetBytes(n, accuracy));
    PrintFootprintTerm("ClassTableBytes", ClassTableBytes(nPairs, n));
    PrintFootprintTerm("AllSurvivePatternBytes", AllSurvivePatternBytes(nPairs));
    std::fflush(stdout);

    // The batched rung the ladder would actually fire for the direct family
    // and for the RI-J nested exchange: the chunked LightPath. Its own
    // footprint function
    // (internal/light_footprint.hpp LightFootprint) is the builders' own and
    // the mode decision is theirs too, so the only honest instrument is a
    // real Create against a budget - still no SCF: Create builds the pair
    // lists, the Schwarz bounds, the core Hamiltonian and the mode record,
    // and stops.
    // The budget is what the LADDER owns, and the LightPath here is
    // budget-driven rather than knob-forced, so the chunk size reported is
    // the builder's own auto-sizing. A refusal is itself a datum - the
    // refusal text names WHICH gate refused - so a small ladder of budgets
    // is walked and every outcome is printed: the answer the question
    // needs is whether the batched rung fits 16 GiB at this size, and a
    // single budget cannot say that.
    {
        auto core = BuildCoreHamiltonian(*molecule, *basis);

        if (!core.has_value())
        {
            std::printf("\n  LIGHT PATH: the core Hamiltonian failed: %s\n",
                        core.error().message.c_str());
        } else
        {
            for (const std::size_t budgetGib : {std::size_t{16}, std::size_t{48}})
            {
                const std::size_t budgetBytes = budgetGib << 30;
                auto budget = WorkspaceBudget::Create(budgetBytes);

                if (!budget.has_value())
                {
                    std::printf("\n  LIGHT PATH: the %zu GiB budget failed: %s\n",
                                budgetGib,
                                budget.error().message.c_str());
                    continue;
                }

                FockBuildOptions options;
                options.workspaceBudget = &*budget;
                auto builder = DirectJkFockBuilder::Create(*molecule, *basis, *core, options);

                std::printf("\n  DIRECT SCREENED - the budget-driven mode at a %zu GiB budget\n",
                            budgetGib);

                if (!builder.has_value())
                {
                    std::printf("    refused: %s\n", builder.error().message.c_str());
                    continue;
                }

                const std::optional<FockModeInfo>& info = builder->ModeInfo();

                if (!info.has_value())
                {
                    std::printf("    no mode record\n");
                } else if (info->mode != FockBuildMode::kLightPath)
                {
                    std::printf("    the FASTPATH was admitted - the batched rung is "
                                "unreachable at this budget\n");
                } else
                {
                    std::printf("    the LIGHTPATH was selected | auto chunkPairs %zu\n",
                                info->chunkPairs);
                    PrintFootprintTerm("lightStoreBytes", info->lightStoreBytes);
                    PrintFootprintTerm("chunkArenaBytes", info->chunkArenaBytes);
                    PrintFootprintTerm("chunkPatternBytes", info->chunkPatternBytes);
                    PrintFootprintTerm("chunkIndexBytes", info->chunkIndexBytes);
                    PrintFootprintTerm("lightShellsBytes", info->lightShellsBytes);
                    PrintFootprintTerm("scratchBytes", info->scratchBytes);
                    PrintFootprintTerm("structuralBytes", info->structuralBytes);
                    PrintFootprintTerm("cacheBytes", info->cacheBytes);
                    PrintFootprintTerm("predictedBytes", info->predictedBytes);
                    PrintFootprintTerm("reservedBytes", info->reservedBytes);
                }
            }
        }
    }

    std::fflush(stdout);

    // The QFMM family: the octree outer store plus the nested near-field
    // direct estimate in buildCoulombOnly mode (exchangeEngaged false - the
    // J-only stack, so the exchange term would be phantom). The near-field
    // pattern count is the leaf-driven survivor count the outer's own
    // exclusion (i) runs.
    std::size_t qfmmNearFieldCount = 0;
    std::size_t qfmmOuterStore = 0;

    {
        auto pairStore = BuildPairData(*molecule, *basis, *pairList);

        if (!pairStore.has_value())
        {
            std::printf("\n  QFMM: the pair store failed: %s\n", pairStore.error().message.c_str());
        } else
        {
            auto geometries = ComputePairGeometries(*pairStore, QfmmExtentForPreset(accuracy));
            auto tree = BuildQfmmTree(geometries);

            if (!tree.has_value())
            {
                std::printf("\n  QFMM: the tree failed: %s\n", tree.error().message.c_str());
            } else
            {
                std::vector<std::pair<std::size_t, std::size_t>> farFieldPairs;
                std::vector<std::pair<std::size_t, std::size_t>> nearFieldLeafPairs;
                BuildInteractionLists(
                    *tree, ThetaForPreset(accuracy), farFieldPairs, nearFieldLeafPairs);
                LeafNearFieldDomain domain;
                domain.nLeaves = tree->nodes.size();
                domain.leafOfPair = tree->leafOfPair;
                domain.nearFieldLeafPairs = std::move(nearFieldLeafPairs);
                qfmmNearFieldCount = CountNearFieldPatternEntries(*schwarz, accuracy, domain);
                qfmmOuterStore =
                    QfmmOuterStoreBytes(*molecule, *basis, *pairList, QfmmOptions{}.maxLeafSize);
            }
        }
    }

    if (qfmmOuterStore != 0)
    {
        const auto nearField = qcx::integrals::internal::DirectFootprint(*molecule,
                                                                         *basis,
                                                                         *pairList,
                                                                         qfmmNearFieldCount,
                                                                         maxBatchBytes,
                                                                         threadCount,
                                                                         false,
                                                                         0,
                                                                         accuracy,
                                                                         false);

        std::printf("\n  QFMM - octree outer store + the nested near-field (Coulomb-only)\n");
        std::printf("    near-field surviving pattern %zu of the full-screened %zu\n",
                    qfmmNearFieldCount,
                    patternCount);
        PrintFootprintTerm("outerStoreBytes", qfmmOuterStore);
        PrintFootprintTerm("pairStoreBytes", nearField.pairStoreBytes);
        PrintFootprintTerm("patternBytes", nearField.patternBytes);
        PrintFootprintTerm("scratchBytes", nearField.scratchBytes);
        PrintFootprintTerm("structuralBytes", nearField.structuralBytes);
        PrintFootprintTerm("cacheBytes", nearField.cacheBytes);
        PrintFootprintTerm("exchangePerCallBytes", nearField.exchangePerCallBytes);
        PrintFootprintTerm("classTableBytes", nearField.classTableBytes);
        PrintFootprintTerm("screenedQuartetBytes", nearField.screenedQuartetBytes);
        PrintFootprintTerm("nearField::Total", nearField.Total());
        PrintFootprintTerm("QFMM stack Total", nearField.Total() + qfmmOuterStore);
    }

    std::fflush(stdout);

    return n;
}

} // namespace

TEST(RiFootprint, PrintsTheScaleLadderAtThe5000BasisFixture) {
    if (std::getenv("QCX_FOOTPRINT_SCALE_PRINT") == nullptr)
    {
        GTEST_SKIP() << "set QCX_FOOTPRINT_SCALE_PRINT=1 to run the 5000-basis scale print "
                        "(the C42H86 Schwarz sweep is minutes)";
    }

    // Largest first, and deliberately so: the 4,974 point's first allocation is
    // ComputeSchwarzBounds' FULL contracted pair store (10.68 GiB), and two
    // runs of this test aborted that point with a bare `bad allocation` after
    // the three smaller points had churned the heap - the smaller points'
    // Creates leave the 64-bit address space fragmented enough that a
    // contiguous 10.68 GiB block no longer exists even with ~18 GiB free.
    // Running the biggest point against a fresh heap is part of the
    // instrument, not a convenience.
    const std::array<ScaleLadderPoint, 4> ladder{{
        {"the scale target: C42H86 / def2-QZVP (the 4,974-function fixture)",
         42,
         "def2-qzvp",
         "def2-universal-jfit"},
        {"the scale ladder 2: C12H26 / def2-QZVP", 12, "def2-qzvp", "def2-universal-jfit"},
        {"the scale ladder 1: C6H14 / def2-QZVP", 6, "def2-qzvp", "def2-universal-jfit"},
        {"the def2-SVP anchor: C24H50 / def2-SVP (n = 586)", 24, "def2-svp", "def2-universal-jfit"},
    }};

    std::printf("\n  the scale print - the engine's own Create-time footprint, no SCF\n");
    std::printf("  16 GiB = %zu B; every GiB column below is / 2^30\n", std::size_t{16} << 30);

    std::size_t anchorFunctions = 0;
    std::size_t targetFunctions = 0;

    for (const ScaleLadderPoint& point : ladder)
    {
        const std::size_t n = PrintScaleLadderPoint(point);

        if (n == 0)
        {
            continue;
        }

        if (point.carbonCount == 24 && anchorFunctions == 0)
        {
            anchorFunctions = n;
        }

        if (point.carbonCount == 42)
        {
            targetFunctions = n;
        }
    }

    // The two counts the fixture owes: the def2-SVP anchor's 586 (so the
    // printed ratios are read against the already-measured point) and the
    // 4,974 (the manifest file's own count - a print that lands elsewhere is
    // not the fixture this print names).
    EXPECT_EQ(anchorFunctions, 586u);
    EXPECT_EQ(targetFunctions, 4974u);
}
