// RI engine tests (ri_engine.hpp):
//   - the 3c grid of tools/gen_md_reference.py (kind "R" rows) at 1e-10
//     against BuildRiTensor over a tiny s/p auxiliary fixture,
//   - the (P|Q) function metric (kind "M" rows) against BuildAuxMetric,
//     both through the phantom (P, s_0) construction,
//   - the RI Coulomb against the direct Coulomb on H2O/STO-3G with the
//     vendored def2-universal-jfit (fast-mode-gated): the builders share
//     the direct exchange at kTight, so F_RI - F_direct = 2 (J_RI - J_direct)
//     and J_RI - J_direct must stay within the RI-J approximation error
//     (the floored eigen-inverse at kMetricFloorEpsilon = 1e-10 plus the
//     jfit fit error, ~1e-4 - measured 2026-08-22),
//   - the aux-shell angular-momentum rejection (skipped when this build's
//     kMaxEngineL covers everything the parser accepts),
//   - the Create-time 3c screen: stretched H2 cross-pair blocks dropped
//     below the kNormal budget come back exactly ZERO where kTight keeps
//     them.
// The RI-J SCF pin lives in scf/tests/ri_rhf_test.cpp (the integrals
// module cannot see scf - the module DAG).

#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "internal/footprint.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/limits.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/workspace_budget.hpp"
#include "qcx/molecule/molecule.hpp"
#include "tensor_conversions.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#ifdef QcxHasCuda
#include <cuda_runtime.h>
#endif

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The tiny s/p auxiliary set of the generator's 3c grid (tiny_aux_shells):
// per atom one s shell (two primitives on O) and one p shell.
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

// An auxiliary set with an f shell: beyond kMaxEngineL on the CI builds.
// Every H2O atom needs an entry (BuildShellPairs rejects a basis missing
// an atom with kInvalidArgument, which would mask the kUnimplemented
// rejection this fixture is meant to reach - the Lmax=2 fix).
inline constexpr std::string_view kFShellAux = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
O    F
      1.0000000000E+00       1.0000000000E+00
END
)";

// The band orbital basis (the footprint_test fixture, inlined here):
// STO-3G O and H in one parsed basis - the same published literals the
// shared fixtures carry (kSto3gOxygen and the H text of h2_sto3g.hpp).
// The 5-function O shells widen the fast-light floor gap to ~ 320 MB
// (2 x tensor + riMatrix - slice), which the literal s-only
// family cannot reach at team size 4 (the light rung's own clamp-fit
// needs budget >= lightClampIneligible + batch + exchangeFixed + 2T); documented
// deviation in footprint_test.cpp RiFirstIterationFiresTheLightRung.
inline constexpr std::string_view kRiBandOrbital = R"(BASIS "ao basis" SPHERICAL PRINT
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

// The firing-rule band aux (the footprint_test fixture,
// inlined here): a per-element 3-shell set (O and H both S,P,D; 648
// functions at 24 waters). The band's positive gap comes from the
// kRiBandOrbital above, NOT from aux angular momentum - d shells are the
// ceiling, so the pins also run on the CI Lmax=2 builds.
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

// A rank-deficient auxiliary set: two identical s shells on one center make
// the (P|Q) metric singular (one zero eigenvalue).
inline constexpr std::string_view kDegenerateAux = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
O    S
      1.0000000000E+00       1.0000000000E+00
O    S
      1.0000000000E+00       1.0000000000E+00
END
)";

// The metric-floor epsilon-sweep fixture: a near-degenerate (P|Q) with
// a small-but-nonzero eigenvalue inside the epsilon sweep window. Two s
// shells per H atom with exponents 1.0 and 1.0 + 2e-4 make each per-atom
// metric pair nearly singular. The same-center s-pair eigenvalue scales as
// delta^2, not delta: the closed form (a|b) = 2^(5/2) pi (ab)^(-1/4)
// (a+b)^(-1/2) gives (a|a) - (a|b) = (a|b) - (b|b) to leading order, so
// lambdaMin ~ (pi/4) delta^2 - measured 1.1e-9 * lambdaMax at delta = 2e-4,
// inside the sweep window between the 1e-10 default floor and the 1e-8/1e-6
// sweep points (a 1e-7 floor measures 2.16e-16 - the delta^2
// eigenvalue sits at solver roundoff, below every floor; 2026-08-27). The
// stretched geometry (R = 17 bohr, the Schwarz-test pattern) keeps the
// cross-atom metric blocks at ~1e-63, so the near-degenerate pair's
// eigenvalue IS the metric's exact minimum (unlike H2O, where the C2v
// symmetry makes the H-antisymmetric combination exactly null).
inline constexpr std::string_view kNearDegenerateAux = R"(BASIS "ao basis" SPHERICAL PRINT
H    S
      1.0000000000E+00       1.0000000000E+00
H    S
      1.0002000000E+00       1.0000000000E+00
END
)";

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

qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
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

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return std::move(*kinetic);
}

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

// N H2O monomers at the experimental geometry, 20 Bohr apart along y (no
// inter-monomer overlap at that separation) - the band fixture (the
// footprint_test helper's shape, inlined here).
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

TEST(RiEngineTest, RiTensorMatchesMpmathReference) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto tensor = qcx::integrals::BuildRiTensor(*molecule, *basis, *aux);
    ASSERT_TRUE(tensor.has_value()) << tensor.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const std::vector<GridRow> grid = Load3cGrid("md_3c_reference.csv");
    ASSERT_GT(grid.size(), 0u);
    int checked = 0;

    for (const GridRow& row : grid)
    {
        if (row.kind != "R")
        {
            continue;
        }

        const std::size_t oI = pairList->shells[row.i].functionOffset + row.ri;
        const std::size_t oJ = pairList->shells[row.j].functionOffset + row.rj;
        const std::size_t oP = auxPairList->shells[row.p].functionOffset + row.rp;
        EXPECT_NEAR((*tensor)(oI, oJ, oP), row.value, 1e-10)
            << "3c (" << row.i << "," << row.j << "|" << row.p << ") function (" << row.ri << ","
            << row.rj << "," << row.rp << ")";
        ++checked;
    }

    EXPECT_GT(checked, 0);
}

TEST(RiEngineTest, AuxMetricMatchesMpmathReference) {
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kTinyAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto metric = qcx::integrals::BuildAuxMetric(*molecule, *aux);
    ASSERT_TRUE(metric.has_value()) << metric.error().message;

    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;

    const std::vector<GridRow> grid = Load3cGrid("md_3c_reference.csv");
    int checked = 0;

    for (const GridRow& row : grid)
    {
        if (row.kind != "M")
        {
            continue;
        }

        const std::size_t oI = auxPairList->shells[row.i].functionOffset + row.ri;
        const std::size_t oJ = auxPairList->shells[row.j].functionOffset + row.rj;
        EXPECT_NEAR((*metric)(oI, oJ), row.value, 1e-10)
            << "metric (" << row.i << "," << row.j << ") function (" << row.ri << "," << row.rj
            << ")";
        EXPECT_NEAR((*metric)(oJ, oI), row.value, 1e-10);
        ++checked;
    }

    EXPECT_GT(checked, 0);
}

TEST(RiEngineTest, RiCoulombMatchesDirectCoulomb) {
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

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // kTight on both sides: the shared direct exchange is identical, so
    // F_RI - F_direct = 2 (J_RI - J_direct) exactly.
    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto ri = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, riOptions);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto direct =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, directOptions);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;

    auto fRi = ri->BuildFock(*densityTensor);
    ASSERT_TRUE(fRi.has_value()) << fRi.error().message;
    auto fDirect = direct->BuildFock(*densityTensor);
    ASSERT_TRUE(fDirect.has_value()) << fDirect.error().message;

    // The honest RI-J bound: the error is the approximation itself - the
    // floored eigen-inverse at kMetricFloorEpsilon = 1e-10 plus the jfit
    // fit residual, measured at 2.6e-4 max over the 7x7 block on 2026-08-22
    // (pyscf's own RI-vs-direct control on this system: 2.3e-4 - the jfit
    // residual dominates, the pre-fix engine was off by 256). The engine's
    // 3c tensor itself agrees with pyscf elementwise at ~1e-5; the direct
    // comparison cannot do better than the fit. 1e-3 leaves a ~4x margin.
    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_LE(std::abs((*fRi)(i, j) - (*fDirect)(i, j)) / 2.0, 1e-3)
                << "J_RI - J_direct at (" << i << "," << j << ")";
        }
    }
}

TEST(RiEngineTest, RiJkSplitHalvesComposeToTheFusedBuild) {
    // The two halves the unrestricted leg's per-spin adapter contracts
    // (ri_engine.hpp BuildCoulombOnly / BuildExchangeOnly) are the SAME
    // halves the fused BuildFock composes, and this row pins that as
    // algebra rather than as a claim:
    //
    //     C(rho) + X(P_a) + C(rho) + X(P_b)  ==  2 F(rho),
    //     C(d) = BuildCoulombOnly(d), X(d) = BuildExchangeOnly(d),
    //     rho = 0.5 (P_a + P_b),    F = BuildFock.
    //
    // The right side is the RESTRICTED leg's own entry point, so the row
    // ties the adapter's three-call assembly to the fused build the RHF
    // path runs - a half that is not the half it claims (a Coulomb call
    // that carries exchange, an exchange call that carries J, a J half
    // missing its convention's factor of two, a density read with the wrong
    // halving) breaks it at the first element.
    //
    // The pair is deliberately POLARIZED (P_a != P_b) and synthetic: the
    // identity is algebra on a linear operator, and the builders contract
    // whatever symmetric density they are handed, so a non-N-representable
    // pair is a legitimate input here. What it does NOT establish, stated
    // because the sum is symmetric under exchanging the two spins: an
    // alpha/beta swap in an ASSEMBLY leaves the sum invariant. That failure
    // is caught where the spins are seen separately - the driver's
    // open-shell row (O2UhfRiJLinkWiresThePerSpinAdapterEndToEnd) reads the
    // per-atom spin populations of a polarized run.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    // The vendored jfit carries g shells on O: the CI Lmax=2 builds cannot
    // instantiate the classes those need (the neighbor test's guard).
    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // kTight: the density screen is at its mildest, which is what keeps the
    // separate calls' screen decisions as close as this identity needs them
    // (the fused call screens the half-summed density, the split calls
    // screen each spin's - a screen difference, not an algebra difference,
    // is the residual this row's tolerance carries).
    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto builder =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, riOptions);
    ASSERT_TRUE(builder.has_value()) << builder.error().message;

    const std::size_t n = 7;
    const Eigen::MatrixXd pAlpha = 0.6 * PhysicalDensity(n);
    const Eigen::MatrixXd pBeta = 1.4 * PhysicalDensity(n);
    const Eigen::MatrixXd halfSummed = 0.5 * (pAlpha + pBeta);

    auto pAlphaTensor = ToTensor(pAlpha);
    ASSERT_TRUE(pAlphaTensor.has_value()) << pAlphaTensor.error().message;
    auto pBetaTensor = ToTensor(pBeta);
    ASSERT_TRUE(pBetaTensor.has_value()) << pBetaTensor.error().message;
    auto halfSummedTensor = ToTensor(halfSummed);
    ASSERT_TRUE(halfSummedTensor.has_value()) << halfSummedTensor.error().message;

    auto coulomb = builder->BuildCoulombOnly(*halfSummedTensor);
    ASSERT_TRUE(coulomb.has_value()) << coulomb.error().message;
    auto exchangeAlpha = builder->BuildExchangeOnly(*pAlphaTensor);
    ASSERT_TRUE(exchangeAlpha.has_value()) << exchangeAlpha.error().message;
    auto exchangeBeta = builder->BuildExchangeOnly(*pBetaTensor);
    ASSERT_TRUE(exchangeBeta.has_value()) << exchangeBeta.error().message;
    auto fused = builder->BuildFock(*halfSummedTensor);
    ASSERT_TRUE(fused.has_value()) << fused.error().message;

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double composed =
                2.0 *
                    ToMatrix(*coulomb)(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) +
                ToMatrix(*exchangeAlpha)(static_cast<Eigen::Index>(i),
                                         static_cast<Eigen::Index>(j)) +
                ToMatrix(*exchangeBeta)(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
            const double fusedTwice =
                2.0 * ToMatrix(*fused)(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
            EXPECT_NEAR(composed, fusedTwice, 1e-10)
                << "the split halves do not compose to the fused build at (" << i << "," << j
                << ")";
        }
    }

    // The convention that lets the adapter hand each exchange call a RAW
    // spin density and no halved one, as its own statement: the exchange
    // half is linear in the density, so X(P_a) + X(P_b) = 2 X(0.5 (P_a +
    // P_b)) - the two channels' K, read off the same half-summed density
    // the Coulomb call took. The composition identity above is this
    // linearity with the shared C(rho) around it; stated separately because
    // THIS is the property the adapter's density choice rests on, and a
    // reader checking that choice should not have to factor it out of the
    // composition.
    auto exchangeHalfSummed = builder->BuildExchangeOnly(*halfSummedTensor);
    ASSERT_TRUE(exchangeHalfSummed.has_value()) << exchangeHalfSummed.error().message;

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const Eigen::Index row = static_cast<Eigen::Index>(i);
            const Eigen::Index column = static_cast<Eigen::Index>(j);
            EXPECT_NEAR(ToMatrix(*exchangeAlpha)(row, column) +
                            ToMatrix(*exchangeBeta)(row, column),
                        2.0 * ToMatrix(*exchangeHalfSummed)(row, column),
                        1e-10)
                << "the exchange half is not linear in the density at (" << i << "," << j << ")";
        }
    }
}

TEST(RiEngineTest, FilteredAuxIsBitIdenticalToUnfiltered) {
    // The per-element aux parse filter is value-neutral: on the same
    // molecule the engine built over the filtered parse must be
    // bit-identical to the engine built over the full parse. The filter
    // keeps H and O - the molecule's elements - whose entries are re-parsed
    // through the identical per-file path.
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    if (!qcx::integrals::SupportsL(4))
    {
        GTEST_SKIP() << "this build's kMaxEngineL is below the jfit g shells";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto auxFull = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(auxFull.has_value()) << auxFull.error().message;
    const std::array<int, 2> auxElements{1, 8};
    auto auxFiltered = qcx::basisset::ParseNwchemDirectoryFiltered(
        (root / "def2-universal-jfit").string(), auxElements);
    ASSERT_TRUE(auxFiltered.has_value()) << auxFiltered.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    qcx::integrals::RiEngineOptions riOptions;
    riOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto builderFull =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *auxFull, *core, riOptions);
    ASSERT_TRUE(builderFull.has_value()) << builderFull.error().message;
    auto builderFiltered =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *auxFiltered, *core, riOptions);
    ASSERT_TRUE(builderFiltered.has_value()) << builderFiltered.error().message;

    const Eigen::MatrixXd& full = builderFull->RiMatrix();
    const Eigen::MatrixXd& filtered = builderFiltered->RiMatrix();
    ASSERT_EQ(full.rows(), filtered.rows());
    ASSERT_EQ(full.cols(), filtered.cols());
    EXPECT_TRUE(full.cwiseEqual(filtered).all()) << "the filtered aux parse changed the RI tensor";
}

#ifdef QcxHasCuda
TEST(RiEngineTest, CudaGemmMatchesCpuGemm) {
    int deviceCount = 0;

    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

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

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The only difference between the two builders is the product backend
    // of the two RI contractions: cuBLAS vs the CPU BLAS seam. Both are
    // dense double Dgemms over the same col-major operands, so the Fock
    // matrices must agree elementwise at double-rounding level. (1e-12
    // absolute on elements of magnitude ~10 - 1e-12 relative -
    // leaves a ~100x margin over the ~1e-14 rounding spread.)
    qcx::integrals::RiEngineOptions cpuOptions;
    cpuOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto cpuRi =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, cpuOptions);
    ASSERT_TRUE(cpuRi.has_value()) << cpuRi.error().message;

    qcx::integrals::RiEngineOptions gpuOptions = cpuOptions;
    gpuOptions.useCudaGemm = true;
    auto gpuRi =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, gpuOptions);
    ASSERT_TRUE(gpuRi.has_value()) << gpuRi.error().message;

    auto fCpu = cpuRi->BuildFock(*densityTensor);
    ASSERT_TRUE(fCpu.has_value()) << fCpu.error().message;
    auto fGpu = gpuRi->BuildFock(*densityTensor);
    ASSERT_TRUE(fGpu.has_value()) << fGpu.error().message;

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR((*fGpu)(i, j), (*fCpu)(i, j), 1e-12)
                << "cuBLAS vs CPU contraction at (" << i << "," << j << ")";
        }
    }
}
#endif

TEST(RiEngineTest, SchwarzScreenedTensorDropsDistantBlocks) {
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

    // Stretched H2 (STO-3G, R = 17 bohr): the cross-pair Schwarz bound
    // Q_uv = sqrt(max diag of (uv|uv)) decays as exp(-zeta R^2 / 2) in the
    // most diffuse primitive (0.169), so Q_uv * Q_P crosses the budgets
    // only at very large separations. With Q_uv ~ 0.45 * exp(-0.0844 R^2)
    // and the aux self-bounds Q_P ~ 0.3..1.5: kNormal drops the cross
    // blocks once Q_uv * Q_P < 1e-10 (R > ~16.4) and kTight keeps them
    // while Q_uv * Q_P >= 1e-12 (R < ~18.1). R = 17 sits inside with
    // 3..17x margins on both sides (R = 18 was tried first: it is inside
    // the bound window, but the ELEMENTS themselves sit below the kTight
    // screen, so the reference tensor is zero there and nothing counts as
    // dropped - the window on meaningful elements is narrower). If a
    // future screening change shifts the prefactor, tune R here (the
    // droppedBlocks EXPECT_GT below prints the actual count).
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        FAIL() << coordinates.error().message;
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 17.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    auto molecule = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    // The kTight tensor is the unscreened reference: its Schwarz budget
    // (1e-14) lies below everything the kNormal screen can drop, so a
    // block kept at kTight with a nonzero value that comes back EXACTLY
    // zero at kNormal is a screened block.
    qcx::integrals::RiEngineOptions tightOptions;
    tightOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto tightTensor = qcx::integrals::BuildRiTensor(*molecule, *basis, *aux, tightOptions);
    ASSERT_TRUE(tightTensor.has_value()) << tightTensor.error().message;

    qcx::integrals::RiEngineOptions normalOptions;
    normalOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    auto normalTensor = qcx::integrals::BuildRiTensor(*molecule, *basis, *aux, normalOptions);
    ASSERT_TRUE(normalTensor.has_value()) << normalTensor.error().message;

    const auto& tight = *tightTensor;
    const auto& normal = *normalTensor;
    const std::size_t n = normal.Shape()[0];
    const std::size_t nAux = normal.Shape()[2];
    std::size_t droppedBlocks = 0;
    std::size_t checkedElements = 0;

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t p = 0; p < nAux; ++p)
            {
                const double tightValue = tight(i, j, p);
                const double normalValue = normal(i, j, p);

                if (normalValue != 0.0)
                {
                    // Both budgets kept the block (kNormal's screen is the
                    // looser of the two, so a kept block implies kTight
                    // kept it too) - the values agree to fp64 last bits.
                    EXPECT_NEAR(normalValue, tightValue, 1e-12)
                        << "retained element (" << i << "," << j << "," << p << ")";
                    ++checkedElements;
                } else if (tightValue != 0.0)
                {
                    // A screened block: exactly zero at kNormal where
                    // kTight computed something nonzero (the cross-pair
                    // elements, e.g. ~8e-12 at this R, sit between the
                    // kTight keep and the kNormal drop).
                    ++droppedBlocks;
                }
            }
        }
    }

    EXPECT_GT(droppedBlocks, 0u)
        << "R=17 bohr must push the cross-pair x aux blocks below the kNormal budget";
    EXPECT_GT(checkedElements, 0u) << "the retained (same-center) part must be nonempty";
}

TEST(RiEngineTest, DegenerateAuxMetricIsFlooredNotRejected) {
    // The BUG-2 protection pin (ri_engine.cpp): a singular (P|Q) - the
    // duplicated function gives one zero eigenvalue - must be floored by
    // kMetricFloorEpsilon, not amplified: the build succeeds and the Fock
    // stays finite (the raw factorized solve would produce O(1e9) null
    // components that cancel). The kInvalidArgument branch of Create
    // (lambdaMax <= 0) is defensive: the metric of any nonzero real basis
    // has a positive self-overlap, so it is unreachable through the parser
    // and has no input that would exercise it.
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kDegenerateAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    qcx::integrals::RiEngineOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto ri = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    auto f = ri->BuildFock(*densityTensor);
    ASSERT_TRUE(f.has_value()) << f.error().message;

    for (const double value : f->HostView())
    {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(RiEngineTest, NearDegenerateMetricEpsilonSweep) {
    // RiEngineOptions::metricFloorEpsilon must be
    // runtime-configurable with the kMetricFloorEpsilon = 1e-10 default
    // unchanged. Sweeping it across the near-degenerate metric's small
    // eigenvalue must not crash at any epsilon; epsilon > 0 must keep
    // every Fock element finite (the floor protects); and the unfloored
    // small-epsilon point must be at least as accurate as the floored
    // large-epsilon point.
    // Stretched H2 (R = 17 bohr, the Schwarz-test geometry): the cross-atom
    // (P|Q) blocks decay as exp(-R^2/2) ~ 1e-63, so each atom's near-
    // degenerate pair owns its metric eigenvalue cleanly - the fixture's
    // small eigenvalue is exactly the sweep target.
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({2, 3});

    if (!coordinates.has_value())
    {
        FAIL() << coordinates.error().message;
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 17.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = 0.0;
    coordinates->MarkHostDirty();
    auto molecule = qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kNearDegenerateAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(2);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The direct reference at kTight: the direct and the RI builder's
    // exchange-only half run the identical K, so F_RI - F_direct =
    // 2 (J_RI - J_direct) exactly (the RiCoulombMatchesDirectCoulomb
    // accounting).
    qcx::integrals::FockBuildOptions directOptions;
    directOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto direct =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, directOptions);
    ASSERT_TRUE(direct.has_value()) << direct.error().message;
    auto fDirect = direct->BuildFock(*densityTensor);
    ASSERT_TRUE(fDirect.has_value()) << fDirect.error().message;

    // The fixture's small eigenvalue ratio: self-documenting window for
    // the sweep - it must sit inside the sweep, strictly between the 1e-12
    // low point (unfloored) and the 1e-8/1e-6 high points (floored), or
    // the sweep never crosses the floor transition.
    auto metric = qcx::integrals::BuildAuxMetric(*molecule, *aux);
    ASSERT_TRUE(metric.has_value()) << metric.error().message;
    const std::size_t nAux = metric->Shape()[0];
    Eigen::MatrixXd metricMatrix(static_cast<Eigen::Index>(nAux), static_cast<Eigen::Index>(nAux));

    for (std::size_t i = 0; i < nAux; ++i)
    {
        for (std::size_t j = 0; j < nAux; ++j)
        {
            metricMatrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                (*metric)(i, j);
        }
    }

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(metricMatrix);
    ASSERT_EQ(solver.info(), Eigen::Success);
    const double lambdaRatio = solver.eigenvalues().minCoeff() / solver.eigenvalues().maxCoeff();
    std::cout << "near-degenerate metric lambdaMin/lambdaMax = " << lambdaRatio << "\n";
    EXPECT_GT(lambdaRatio, 1e-12);
    EXPECT_LT(lambdaRatio, 1e-8);

    constexpr double kEpsilonLow = 1e-12;
    constexpr double kEpsilonHigh = 1e-6;
    double maxErrorAtLow = 0.0;
    double maxErrorAtHigh = 0.0;

    for (const double epsilon : {0.0, kEpsilonLow, 1e-10, 1e-8, kEpsilonHigh})
    {
        qcx::integrals::RiEngineOptions options;
        options.accuracy = qcx::integrals::AccuracyPreset::kTight;
        options.metricFloorEpsilon = epsilon;
        auto ri = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);

        if (epsilon == 0.0)
        {
            // The pinned epsilon == 0 contract: the
            // near-degenerate metric is still SPD (lambdaMin > 0), so the
            // unfloored inverse is computable - Create and BuildFock
            // SUCCEED with finite (amplified) values. The engine does not
            // crash; it just loses the floor's protection.
            ASSERT_TRUE(ri.has_value()) << ri.error().message;
            auto f = ri->BuildFock(*densityTensor);
            ASSERT_TRUE(f.has_value()) << f.error().message;

            for (std::size_t i = 0; i < 2; ++i)
            {
                for (std::size_t j = 0; j < 2; ++j)
                {
                    EXPECT_TRUE(std::isfinite((*f)(i, j)));
                }
            }

            continue;
        }

        ASSERT_TRUE(ri.has_value()) << ri.error().message;
        auto f = ri->BuildFock(*densityTensor);
        ASSERT_TRUE(f.has_value()) << f.error().message;
        double maxError = 0.0;

        for (std::size_t i = 0; i < 2; ++i)
        {
            for (std::size_t j = 0; j < 2; ++j)
            {
                EXPECT_TRUE(std::isfinite((*f)(i, j)));
                maxError = std::max(maxError, std::abs((*f)(i, j) - (*fDirect)(i, j)) / 2.0);
            }
        }

        std::cout << "metricFloorEpsilon = " << epsilon << ": max |J_RI - J_direct| = " << maxError
                  << "\n";

        if (epsilon == kEpsilonLow)
        {
            maxErrorAtLow = maxError;
        }

        if (epsilon == kEpsilonHigh)
        {
            maxErrorAtHigh = maxError;
        }
    }

    // Monotonicity sanity: a larger floor zeroes more of the near-null
    // metric direction, so the RI-J error vs the direct path grows (or
    // stays) as epsilon grows; the unfloored 1e-12 point must be at
    // least as accurate as the floored 1e-6 point. (Measured spread
    // ~5e-4; the 1e-9 slack sits ~5 orders below it and absorbs roundoff
    // in the cancellation-prone w solve.)
    EXPECT_LE(maxErrorAtLow, maxErrorAtHigh + 1e-9);
}

TEST(RiEngineTest, AuxShellBeyondEngineLIsRejected) {
    // The parser accepts l <= 6; when this build's kMaxEngineL is below
    // that, an auxiliary shell above the cap must be rejected with
    // kUnimplemented.
    if (qcx::integrals::SupportsL(3))
    {
        GTEST_SKIP() << "this build covers the test f shell";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kFShellAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto tensor = qcx::integrals::BuildRiTensor(*molecule, *basis, *aux);
    EXPECT_FALSE(tensor.has_value());
    EXPECT_EQ(tensor.error().code, qcx::ErrorCode::kUnimplemented);

    auto metric = qcx::integrals::BuildAuxMetric(*molecule, *aux);
    EXPECT_FALSE(metric.has_value());
    EXPECT_EQ(metric.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(RiEngineTest, LightRungMatchesMaterializedWithinThePresetBudget) {
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

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(aux.has_value()) << aux.error().message;

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The materialized fast path and the forced light rung (the budget seam
    // only engages the rung at nt84-class scales where the tensor dominates
    // the residual quadratic terms - the flag is the unit-scale validation
    // seam, RiEngineOptions::forceLightRung).
    qcx::integrals::RiEngineOptions fastOptions;
    fastOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto fast =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, fastOptions);
    ASSERT_TRUE(fast.has_value()) << fast.error().message;

    qcx::integrals::RiEngineOptions lightOptions;
    lightOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    lightOptions.forceLightRung = true;
    auto light =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, lightOptions);
    ASSERT_TRUE(light.has_value()) << light.error().message;

    // The light rung holds no tensor (RiMatrix is empty) and the forced
    // seam records no mode decision.
    EXPECT_EQ(light->RiMatrix().size(), 0);
    EXPECT_FALSE(light->ModeInfo().has_value());

    auto fFast = fast->BuildFock(*densityTensor);
    ASSERT_TRUE(fFast.has_value()) << fFast.error().message;
    auto fLight = light->BuildFock(*densityTensor);
    ASSERT_TRUE(fLight.has_value()) << fLight.error().message;

    // The recompute contracts the SAME 3c values (the shared assembly order)
    // in a different accumulation order than the materialized BLAS products -
    // not bit-identical by design, validated within the error budget.
    // The difference is summation-order noise (~1e-15 relative), far below
    // the RI approximation itself (2.6e-4 measured); 1e-10 leaves ~1e5 of
    // headroom. kTight on both sides makes the shared exchange half
    // identical, so the Fock difference is exactly 2 (J_light - J_fast).
    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR((*fLight)(i, j), (*fFast)(i, j), 1e-10);
        }
    }

    // The per-iteration contract: a second iteration reuses the retained
    // payload (the combined store, the screened task list and the pair
    // lists) and reproduces the first - same inputs, same assembly order,
    // same accumulation. The pin is a tolerance, not bit equality: the
    // shared exchange half's per-call combine is thread-schedule dependent
    // by design (fock_screen.hpp), so its contribution can differ in the
    // last ulp across calls at the default team size (measured 5e-17).
    // The light rung's own terms are deterministic; 1e-13 covers the
    // exchange's noise with ~1e3 headroom.
    auto fLightSecond = light->BuildFock(*densityTensor);
    ASSERT_TRUE(fLightSecond.has_value()) << fLightSecond.error().message;

    for (std::size_t i = 0; i < 7; ++i)
    {
        for (std::size_t j = 0; j < 7; ++j)
        {
            EXPECT_NEAR((*fLightSecond)(i, j), (*fLight)(i, j), 1e-13);
        }
    }
}

TEST(RiEngineTest, TermCountersAccumulateOverTheRun) {
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

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::filesystem::path root(QcxBasisDataDir);
    auto aux = qcx::basisset::ParseNwchemDirectory((root / "def2-universal-jfit").string());
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;
    const std::size_t nAuxShells = auxPairList->shells.size();

    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;
    const Eigen::MatrixXd density = PhysicalDensity(7);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // The term-counter pins (ri_engine.hpp RiTermCounters):
    // the materialized fast path and the forced light rung screen the SAME
    // task list (identical inputs), so the light rung's per-call
    // recompute partial is exactly twice the fast path's single Create-time
    // pass, and both runs' metric occurrence is the same full (P|Q)
    // triangle (x == nAuxShells, p3 == 0). kTight on both sides: the
    // exchange halves are identical, so the qx/gx per-call counts match
    // exactly and equal the caller-side statsOut totals.
    qcx::integrals::RiEngineOptions fastOptions;
    fastOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    auto fast =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, fastOptions);
    ASSERT_TRUE(fast.has_value()) << fast.error().message;

    qcx::integrals::RiEngineOptions lightOptions;
    lightOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    lightOptions.forceLightRung = true;
    auto light =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, lightOptions);
    ASSERT_TRUE(light.has_value()) << light.error().message;

    // The Create-time snapshots (no BuildFock call yet): the light rung
    // holds exactly the metric occurrence (its tensor-pass partial enters
    // only per call), the fast path holds the metric plus its single
    // tensor pass.
    const qcx::integrals::RiTermCounters fastC0 = fast->TermCounters();
    const qcx::integrals::RiTermCounters lightC0 = light->TermCounters();
    EXPECT_EQ(lightC0.x, nAuxShells);
    EXPECT_EQ(lightC0.p3, 0u);
    EXPECT_GT(lightC0.g3, 0u);
    EXPECT_EQ(lightC0.qx, 0u);
    EXPECT_EQ(lightC0.gx, 0u);
    EXPECT_EQ(fastC0.qx, 0u);
    EXPECT_EQ(fastC0.gx, 0u);
    EXPECT_GT(fastC0.x, lightC0.x);
    EXPECT_GT(fastC0.g3, lightC0.g3);
    EXPECT_GE(fastC0.g3, fastC0.p3);
    EXPECT_GT(fastC0.p3, 0u);

    // One instrumented call on each builder (per-call statsOut given).
    qcx::integrals::FockBuildStats fastStats;
    auto fFast = fast->BuildFock(*densityTensor, &fastStats);
    ASSERT_TRUE(fFast.has_value()) << fFast.error().message;
    qcx::integrals::FockBuildStats lightStats;
    auto fLight = light->BuildFock(*densityTensor, &lightStats);
    ASSERT_TRUE(fLight.has_value()) << fLight.error().message;

    const qcx::integrals::RiTermCounters fastC1 = fast->TermCounters();
    const qcx::integrals::RiTermCounters lightC1 = light->TermCounters();

    // The recompute's two passes per call over the same list: the light
    // run's per-call tensor-pass partial is exactly twice the fast path's
    // single Create-time pass. That pass is fastC0 - lightC0 (both rungs
    // share the metric occurrence, so lightC0 is the metric alone and
    // fastC0 is the metric plus the one fast pass); the fastC1 - fastC0
    // deltas below are the fast path's PER-CALL movement, zero by design
    // (the materialized tensor is built once at Create).
    EXPECT_EQ(lightC1.p3, lightC0.p3 + 2 * (fastC0.p3 - lightC0.p3));
    EXPECT_EQ(lightC1.x, lightC0.x + 2 * (fastC0.x - lightC0.x));
    EXPECT_EQ(lightC1.g3, lightC0.g3 + 2 * (fastC0.g3 - lightC0.g3));
    EXPECT_EQ(fastC1.x, fastC0.x);
    EXPECT_EQ(fastC1.p3, fastC0.p3);
    EXPECT_EQ(fastC1.g3, fastC0.g3);

    // The qx/gx occurrences: identical exchange inputs -> identical
    // per-call counts, each exactly the caller-side statsOut totals (the
    // instrumented-call gate).
    EXPECT_EQ(fastC1.qx, lightC1.qx);
    EXPECT_EQ(fastC1.gx, lightC1.gx);
    EXPECT_EQ(fastC1.qx,
              static_cast<std::size_t>(fastStats.fp64QuartetCount + fastStats.fp32QuartetCount));
    EXPECT_EQ(fastC1.gx, static_cast<std::size_t>(fastStats.primitiveProductSum));
    EXPECT_EQ(lightC1.qx,
              static_cast<std::size_t>(lightStats.fp64QuartetCount + lightStats.fp32QuartetCount));
    EXPECT_EQ(lightC1.gx, static_cast<std::size_t>(lightStats.primitiveProductSum));
    EXPECT_GT(fastC1.qx, 0u);
    EXPECT_GE(fastC1.gx, fastC1.qx);
    EXPECT_GE(fastC1.g3, fastC1.p3);
    EXPECT_GE(lightC1.g3, lightC1.p3);

    // Multi-call: the fast path's 3c counts stay at their Create-time
    // values while the qx/gx exchange counts accumulate per instrumented
    // call (the driver seam passes a statsOut on every call).
    qcx::integrals::FockBuildStats fastStatsSecond;
    auto fFastSecond = fast->BuildFock(*densityTensor, &fastStatsSecond);
    ASSERT_TRUE(fFastSecond.has_value()) << fFastSecond.error().message;
    const qcx::integrals::RiTermCounters fastC2 = fast->TermCounters();
    EXPECT_EQ(fastC2.x, fastC1.x);
    EXPECT_EQ(fastC2.p3, fastC1.p3);
    EXPECT_EQ(fastC2.g3, fastC1.g3);
    EXPECT_EQ(fastC2.qx, 2 * fastC1.qx);
    EXPECT_EQ(fastC2.gx, 2 * fastC1.gx);

    // Two UNINSTRUMENTED light calls (no statsOut): the 3c recompute
    // passes still count - the occurrence is the kernel evaluation itself -
    // while the qx/gx exchange occurrences do not (their per-call counts
    // exist only on instrumented calls): the statsOut gate, pinned.
    auto fLightSecond = light->BuildFock(*densityTensor);
    ASSERT_TRUE(fLightSecond.has_value()) << fLightSecond.error().message;
    auto fLightThird = light->BuildFock(*densityTensor);
    ASSERT_TRUE(fLightThird.has_value()) << fLightThird.error().message;
    const qcx::integrals::RiTermCounters lightC3 = light->TermCounters();
    EXPECT_EQ(lightC3.p3, 3 * lightC1.p3);
    EXPECT_EQ(lightC3.g3, lightC1.g3 + 2 * (lightC1.g3 - lightC0.g3));
    EXPECT_EQ(lightC3.x, lightC1.x + 2 * (lightC1.x - lightC0.x));
    EXPECT_EQ(lightC3.qx, lightC1.qx);
    EXPECT_EQ(lightC3.gx, lightC1.gx);

    // The blocked-metric rung's strip-wise build counts the SAME metric
    // occurrence as the unblocked build (the strip partition is
    // value-neutral): x == nAuxShells, p3 == 0, equal g3.
    qcx::integrals::RiTermCounters blockedCounters;
    auto blocked = qcx::integrals::BuildBlockedAuxMetric(*molecule, *aux, {}, &blockedCounters);
    ASSERT_TRUE(blocked.has_value()) << blocked.error().message;
    EXPECT_EQ(blockedCounters.x, nAuxShells);
    EXPECT_EQ(blockedCounters.p3, 0u);
    EXPECT_GT(blockedCounters.g3, 0u);

    qcx::integrals::RiTermCounters metricCounters;
    auto metric = qcx::integrals::BuildAuxMetric(*molecule, *aux, {}, &metricCounters);
    ASSERT_TRUE(metric.has_value()) << metric.error().message;
    EXPECT_EQ(metricCounters.x, blockedCounters.x);
    EXPECT_EQ(metricCounters.g3, blockedCounters.g3);
    EXPECT_EQ(metricCounters.p3, 0u);
    // Every rung counts the SAME metric occurrence (the full (P|Q)
    // triangle): the light snapshot above equals the direct builds'.
    EXPECT_EQ(lightC0.x, metricCounters.x);
    EXPECT_EQ(lightC0.g3, metricCounters.g3);
}

TEST(RiEngineTest, FirstIterationSelectsTheLightRung) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "band construction: Release-only";
    }

    // The mode-selection pin: a budget where TODAY
    // the fast rung fires (the clamp absorbs the overflow by shrinking the
    // scratch arenas) and the fast rung's FOLDED first-iteration total
    // cannot fit - the two-arena clamp's refusal selects the light rung
    // (the clamp-IRREDUCIBLE floor check drafted for this zone is
    // decision-inert and was deleted; the distinctive
    // zone it would isolate does not exist - the clamp refuses in the same
    // inputs), which fires, reserves the folded total and builds Fock
    // against the retained payload. The 24-water kRiBandOrbital (STO-3G)
    // family with the 3-shell d-max kRiBandAux (the deviations documented
    // in footprint_test.cpp RiFirstIterationFiresTheLightRung); the budget
    // is the razor's edge: one clamped arena under the OLD (pre-fix)
    // total - the midpoint of today's clamp-fit zone, which sits between
    // the rungs' clamp-ineligible masses.
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::basisset::ParseNwchemText(kRiBandOrbital);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kRiBandAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;
    // 24 waters x 5 shells per water (O S, the SP parsing as separate
    // S and P shells, two H S) = 120 shells, 7 functions per water
    // (O 5: S + spherical SP; two H S) = 168 functions, 7260 canonical
    // orbital pairs; the aux: 216 shells, 648 functions.
    ASSERT_EQ(pairList->functionCount, 168u);
    ASSERT_EQ(pairList->pairs.size(), 7260u);
    ASSERT_EQ(auxPairList->shells.size(), 216u);
    ASSERT_EQ(auxPairList->functionCount, 648u);

    qcx::integrals::RiEngineOptions options;
    const std::size_t batch = std::size_t{2} * 1024 * 1024;
    options.maxBatchBytes = batch;

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> indices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, offsets, indices);

    // The screened task-grid count, exactly as the engine's decision block
    // computes it: the footprint's taskListBytes
    // charges the surviving (orbital pair, aux shell) cells x 16 B, and
    // this band's delta-cancellation holds only while the recomputed terms
    // carry the SAME count the engine charges (the razor budget below
    // rides these totals).
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value()) << schwarzAux.error().message;
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
    const std::size_t fastClampIneligible = fastTerms.Total() - fastTerms.scratchBytes;
    const std::size_t lightClampIneligible =
        lightTerms.Total() - lightTerms.scratchBytes - lightTerms.valuesBufferBytes;
    // The exchange per-call term is folded into Total() too - subtract it
    // with the first-iteration terms so the "old" total stays today's true
    // pre-change estimate (the budget's razor's edge keeps its exact
    // margins). The n34b screened-quartet charge does NOT
    // follow the same cancellation: the charge (446,092,738 B at this
    // 168-function fixture) is ~40x the one-arena razor
    // (teamSize x (batch - 1) = 10.5 MB at the 2 MiB batch), so a
    // cancelled budget (404,466,925 B) sits BELOW the charged light
    // total (766,812,223 B) - the light rung cannot fit and the razor
    // collapses (the footprint_test.cpp mirror measured the same
    // refusal). The budget must MOVE WITH the charge instead: the razor
    // rides the charged old total, one arena below it - the light total
    // fits, the fast rung's refusal comes from its charged folded total
    // (the per-call, transpose and small-vector terms, far above the
    // two-arena capacity), and the fast clamp-ineligible mass sits
    // BELOW the budget - its band role is gone, the charge's task
    // machinery carries it.
    //
    // The honest screened-quartet charge (the footprint_test mirror does the
    // same): the whole-call charge drops by
    // 382,365,204 B at this fixture, the razor budget measures
    // ~458,116,763 B and the fast clamp-ineligible mass (~506,797,184 B,
    // charge-free) sits ABOVE it - the fast refusal at this budget is by
    // mass (its band role is back), while the light total (~384.4 MB)
    // still fits with the full batch; the assertion below inverts with
    // the geometry.
    const std::size_t oldFastTotal = fastTerms.Total() + exchange.Total() -
                                     exchange.exchangePerCallBytes - fastTerms.transposeBytes -
                                     fastTerms.smallVectorsBytes;
    const std::size_t fastTotal = fastTerms.Total() + exchange.Total();
    const std::size_t lightTotal = lightTerms.Total() + exchange.Total();
    EXPECT_GT(fastClampIneligible, lightClampIneligible);
    // The razor's edge: the clamp fits the old total (shrinking one
    // arena), the fast rung's folded total fails the clamp - by its
    // ineligible mass alone since the honest charge (the mirror's
    // own geometry) - and the light rung fits with the full batch.
    const std::size_t budgetBytes = oldFastTotal - threadCount * (batch - 1);
    EXPECT_GT(fastClampIneligible, budgetBytes);
    EXPECT_LT(lightClampIneligible, budgetBytes);
    EXPECT_GT(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, oldFastTotal, budgetBytes, 2),
        0u);
    EXPECT_EQ(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, fastTotal, budgetBytes, 2),
        0u);
    EXPECT_GT(
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, lightTotal, budgetBytes, 2),
        0u);

    auto budget = qcx::memory::WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    options.workspaceBudget = &*budget;
    auto ri = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(ri.has_value()) << ri.error().message;

    const std::optional<qcx::integrals::FockModeInfo>& info = ri->ModeInfo();
    ASSERT_TRUE(info.has_value());
    // The fast attempt's clamp bound: the light rung fired, holds no
    // tensor and reserved the folded total. Post-lift the
    // nested exchange fires k = team into its post-reservation band - the
    // committed charge rides at or above the k = 1 estimate and stays
    // inside the razor budget. Never-under is the honest form: the k-fold
    // excess is platform-arithmetic boundary-sensitive, and the Debug
    // rounding lands the total exactly ON the estimate on the clang leg -
    // the committed == predicted equality is a legitimate boundary.
    EXPECT_EQ(info->mode, qcx::integrals::FockBuildMode::kLightPath);
    EXPECT_FALSE(info->tensorExcluded);
    EXPECT_EQ(ri->RiMatrix().size(), 0);
    EXPECT_LE(info->predictedBytes, budgetBytes);
    EXPECT_GE(budget->CommittedBytes(), info->predictedBytes);
    EXPECT_LE(budget->CommittedBytes(), budgetBytes);

    // The light payload built: BuildFock runs the recompute against the
    // retained combined store and the screened task list.
    const Eigen::MatrixXd density = PhysicalDensity(pairList->functionCount);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto f = ri->BuildFock(*densityTensor);
    ASSERT_TRUE(f.has_value()) << f.error().message;
    EXPECT_EQ(f->Shape()[0], pairList->functionCount);
}

TEST(RiEngineTest, BlockedMetricRungMatchesTheLightRung) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "band construction: Release-only";
    }

    // The blocked-metric rung's identity contract: the
    // blocked build's (P|Q) matrix is BIT-IDENTICAL to the unblocked
    // BuildAuxMetric's (the same phantom quartets through the same kernels;
    // the strip partition is value-neutral by contract), so the rung's
    // only observable is its memory shape, never its numbers. Two pins:
    // (a) the direct metric compare - BuildAuxMetric vs
    // BuildBlockedAuxMetric on the same inputs, byte for byte over the
    // full nAux x nAux matrix (648^2 doubles at the kRiBandAux fixture);
    // (b) the end-to-end compare - a builder engaged through the budget
    // seam (the firing razor of footprint_test.cpp
    // RiFirstIterationFiresTheBlockedMetricRung) against a forced-light
    // builder without a budget, same batch and same kTight preset (the
    // screening and the certified lane are preset-consistent on both sides):
    // the Fock matrices agree. The Fock pin is a tolerance, not bit
    // equality - the shared exchange half's per-call combine is
    // thread-schedule dependent by design (fock_screen.hpp), and the two
    // builders' exchanges run different clamped batches; the RI halves
    // themselves are bit-identical by construction (same task order, same
    // accumulation - the partitions only split contiguous runs). kTight
    // keeps the exchange identical in both builders (same screen, certified
    // lane off); 1e-11 leaves ~1e3 headroom over the measured last-ulp noise
    // and sits ~7 orders below the RI approximation error (2.6e-4), which
    // a divergent metric would show up as.
    auto molecule = MakeWaterCluster(24);
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto basis = qcx::basisset::ParseNwchemText(kRiBandOrbital);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto aux = qcx::basisset::ParseNwchemText(kRiBandAux);
    ASSERT_TRUE(aux.has_value()) << aux.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto auxPairList = qcx::integrals::BuildShellPairs(*molecule, *aux);
    ASSERT_TRUE(auxPairList.has_value()) << auxPairList.error().message;
    ASSERT_EQ(pairList->functionCount, 168u);
    ASSERT_EQ(pairList->pairs.size(), 7260u);
    ASSERT_EQ(auxPairList->shells.size(), 216u);
    ASSERT_EQ(auxPairList->functionCount, 648u);

    // (a) The metric pin: the blocked matrix is the unblocked one, byte
    // for byte. Both builders run the same batch cap; the blocked build's
    // 2 MiB strips exercise the greedy partition over the 216 metric
    // rows (each strip's rows' byte masses sum to at most the cap - the
    // ~64 KB rows close strips of ~30 rows).
    qcx::integrals::RiEngineOptions metricOptions;
    const std::size_t batch = std::size_t{2} * 1024 * 1024;
    metricOptions.maxBatchBytes = batch;
    auto unblockedMetric = qcx::integrals::BuildAuxMetric(*molecule, *aux, metricOptions);
    ASSERT_TRUE(unblockedMetric.has_value()) << unblockedMetric.error().message;
    auto blockedMetric = qcx::integrals::BuildBlockedAuxMetric(*molecule, *aux, metricOptions);
    ASSERT_TRUE(blockedMetric.has_value()) << blockedMetric.error().message;
    const Eigen::MatrixXd unblocked = ToMatrix(*unblockedMetric);
    ASSERT_EQ(unblocked.rows(), 648);
    EXPECT_EQ(blockedMetric->rows(), unblocked.rows());
    EXPECT_EQ(blockedMetric->cols(), unblocked.cols());
    EXPECT_EQ(std::memcmp(unblocked.data(),
                          blockedMetric->data(),
                          unblocked.rows() * unblocked.cols() * sizeof(double)),
              0);

    // (b) The end-to-end pin: the blocked-rung builder through the budget
    // seam vs the forced-light builder without a budget. The firing razor
    // (footprint_test.cpp RiFirstIterationFiresTheBlockedMetricRung): one
    // byte below the light clamp-fit floor, at kTight (the exchange term
    // shifts with the preset - everything is computed at runtime, the
    // metric gap lightTotal - blockedTotal = 3 x 8 nAuxFuncs^2 -
    // max(batch, largest row mass) is preset-independent).
    qcx::integrals::RiEngineOptions options;
    options.maxBatchBytes = batch;
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;

    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> indices;
    qcx::integrals::internal::BuildNeighborList(
        *pairList, *schwarz, options.accuracy, offsets, indices);
    // The screened task-grid count, exactly as the engine's decision block
    // computes it: the footprint's taskListBytes
    // charges the surviving (orbital pair, aux shell) cells x 16 B - the
    // same count the engine charges at this fixture's kTight preset (the
    // razor budget below rides these totals; the rungs differ by the
    // metric term only, so the task charge cancels between them).
    auto schwarzAux = qcx::integrals::ComputeSchwarzBounds(*molecule, *aux);
    ASSERT_TRUE(schwarzAux.has_value()) << schwarzAux.error().message;
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
    const std::size_t budgetBytes = lightTotal - 2 * threadCount * (batch - 1) - 1;
    const std::size_t firedBatch =
        qcx::integrals::internal::ClampBatchBytes(batch, threadCount, blockedTotal, budgetBytes, 2);
    EXPECT_GT(firedBatch, 0u);
    auto budget = qcx::memory::WorkspaceBudget::Create(budgetBytes);
    ASSERT_TRUE(budget.has_value()) << budget.error().message;
    options.workspaceBudget = &*budget;
    auto blocked = qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, options);
    ASSERT_TRUE(blocked.has_value()) << blocked.error().message;

    const std::optional<qcx::integrals::FockModeInfo>& blockedInfo = blocked->ModeInfo();
    ASSERT_TRUE(blockedInfo.has_value());
    EXPECT_EQ(blockedInfo->mode, qcx::integrals::FockBuildMode::kLightPath);
    EXPECT_TRUE(blockedInfo->blockedMetricRung);
    EXPECT_EQ(blocked->RiMatrix().size(), 0);

    qcx::integrals::RiEngineOptions lightOptions = options;
    lightOptions.workspaceBudget = nullptr;
    lightOptions.forceLightRung = true;
    auto light =
        qcx::integrals::RiJkFockBuilder::Create(*molecule, *basis, *aux, *core, lightOptions);
    ASSERT_TRUE(light.has_value()) << light.error().message;
    // The forced-light reference runs the legacy path: no budget, no mode
    // record, no blocked rung - it is the unblocked light rung at the full
    // 2 MiB batch, the exact counterfactual of the budget-engaged builder.
    EXPECT_FALSE(light->ModeInfo().has_value());

    const Eigen::MatrixXd density = PhysicalDensity(pairList->functionCount);
    auto densityTensor = ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;
    auto fBlocked = blocked->BuildFock(*densityTensor);
    ASSERT_TRUE(fBlocked.has_value()) << fBlocked.error().message;
    auto fLight = light->BuildFock(*densityTensor);
    ASSERT_TRUE(fLight.has_value()) << fLight.error().message;

    for (std::size_t i = 0; i < pairList->functionCount; ++i)
    {
        for (std::size_t j = 0; j < pairList->functionCount; ++j)
        {
            EXPECT_NEAR((*fBlocked)(i, j), (*fLight)(i, j), 1e-11);
        }
    }
}

} // namespace
