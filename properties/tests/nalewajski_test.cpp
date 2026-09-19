// Nalewajski-Mrozek bond-order tests: the quadratic
// valence indices of the one-determinantal difference approach with the
// frozen-SAL reference ([Nalewajski1996], Can. J. Chem. 74, 1121-1130),
// ported from niedoida's NalewajskiAnalysis, on real converged RHF
// densities (properties links scf PRIVATE; the fragment SCFs run internally
// per distinct element).
//
// What is pinned EXACTLY is what the method guarantees by construction:
//   the H2 molecule is algebraically exact: v_i = 1/4, v_ab = 1/2,
//     B(H,H) = 1.0 - the single-bond unit;
//   the H2O and benzene fixture geometries carry exact point-group
//     symmetry, so equivalent pairs must receive EQUAL bond orders and
//     equivalent atoms EQUAL (v_i_a + v_c_a) - the invariant combination;
//   v_ab is a plain sum of the orthogonalized density squares, with
//     total_ab(a) = sum_b diatomicCovalent(a, b) by construction, and the
//     H fragments have a single function, so their v_c_a = 0 identically.
// What is pinned TIGHTLY is the numpy cross-validation
// (tools/properties/nalewajski_numpy_reference.py reproduces
// nalewajski.cpp bit-for-bit on pyscf densities). The config averaging is
// invariant under rotations of the degenerate-shell spanning set
// (v_i_a + v_c_a = 1/2 <||Delta||_F^2>_cfg), so the bond orders and the
// invariant combos pin the C++; v_i_a and v_c_a individually depend on the
// spanning set the fragment SCF happens to produce (both codes produce the
// AO-aligned canonical set, but the test pins only the invariant - the C
// atoms' individual v_i_a scatter by ~3e-7 in the reference itself).
// What is pinned LOOSELY is the magnitude class against the paper's Fig. 3
// (STO-6G, GAMES; the paper's bonding valences are negative): benzene C-C
// ~ -1.0 and C-H ~ -0.9..-1.0, water O-H = -1.2 (3-21G UHF). The STO-3G
// values land on the O-H and C-H magnitudes; the aromatic C-C reads 1.44
// (between the single and double bond units, above the STO-6G -1.005 - a
// documented basis difference).
#include "benzene_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/eri_dense.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/properties/nalewajski.hpp"
#include "qcx/scf/rhf.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <utility>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeBenzeneSto3g;
using qcx::testing::MakeBenzeneSto3gBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeSto3gBasis;
using qcx::testing::ToMatrix;

// The numpy cross-validation pins (nalewajski_numpy_reference.py =
// nalewajski.cpp bit-for-bit, pyscf densities, unit="Bohr", conv 1e-10).
constexpr double kH2oOhBondOrder = 1.19996670;
constexpr double kH2oHhBondOrder = 6.75636403e-4;
constexpr double kH2oOhDiatomic = 0.491642221;
constexpr double kH2oHhDiatomic = 3.32395961e-4;
constexpr double kH2oOxygenTotal = 0.98328444;
constexpr double kH2oHydrogenTotal = 0.49197462;
constexpr double kH2oOxygenValence = 0.90896682;
constexpr double kH2oHydrogenValence = 0.25401269;
constexpr double kBenzeneCcBondOrder = 1.44319823;
// The GJ forward-root texture is non-monotone in the ring: the 120-degree
// (meta) C-C block reads 7.9e-4 while the 180-degree (para) block reads
// 0.116 - the orthogonalizing root amplifies the ring's long-range blocks
// (the documented working-basis effect of the EDDB episode too). The pins
// follow the numbers, not the chemistry intuition.
constexpr double kBenzeneCcMetaBondOrder = 7.88749267e-4;
constexpr double kBenzeneCcParaBondOrder = 0.115828431;
constexpr double kBenzeneChBondOrder = 0.980867168;
constexpr double kBenzeneHhOrthoBondOrder = 2.19967302e-3;
constexpr double kBenzeneCcDiatomic = 0.721976888;
constexpr double kBenzeneChDiatomic = 0.490335361;
constexpr double kBenzeneCarbonTotal = 1.99946392;
constexpr double kBenzeneHydrogenTotal = 0.49953848;
constexpr double kBenzeneCarbonValence = 0.99868574;
constexpr double kBenzeneHydrogenValence = 0.25023076;

// H = T + V from the one-electron engines (the dense_rhf_test.cpp pattern,
// shared with the sibling properties tests).
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

// The dense general-l RHF path (eri_dense.hpp): the N-M tests run the real
// converged density, then analyze it.
qcx::Result<qcx::scf::HfResult> RunDenseRhf(const qcx::molecule::Molecule& molecule,
                                            const qcx::basisset::BasisSet& basisSet) {
    auto overlap = qcx::integrals::BuildOverlapMatrix(molecule, basisSet);

    if (!overlap.has_value())
    {
        return std::unexpected(overlap.error());
    }

    auto core = BuildCoreHamiltonian(molecule, basisSet);

    if (!core.has_value())
    {
        return std::unexpected(core.error());
    }

    auto eri = qcx::integrals::BuildEriTensorGeneral(molecule, basisSet);

    if (!eri.has_value())
    {
        return std::unexpected(eri.error());
    }

    return qcx::scf::RunRhfScf(molecule, ToMatrix(*overlap), ToMatrix(*core), *eri);
}

// The per-spin densities from a converged RHF result (the nalewajski.hpp /
// populations.hpp density convention: P_sigma = C C^T, D = 2 P_alpha).
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> SpinDensities(const qcx::scf::HfResult& scf) {
    const Eigen::MatrixXd perSpin = scf.density / 2.0;
    return {perSpin, perSpin};
}

// The indices of the atoms of the given element (the canonical order is
// a sort, so tests must never assume an index layout).
std::vector<std::size_t> ElementIndices(const qcx::molecule::Molecule& molecule, int atomicNumber) {
    const auto& atoms = molecule.Atoms();
    std::vector<std::size_t> indices;

    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        if (atoms[i].atomicNumber == atomicNumber)
        {
            indices.push_back(i);
        }
    }

    return indices;
}

// Mean of the per-atom values over the atoms of the given element.
double MeanOverElement(const qcx::molecule::Molecule& molecule,
                       const Eigen::VectorXd& values,
                       int atomicNumber) {
    const auto indices = ElementIndices(molecule, atomicNumber);
    EXPECT_FALSE(indices.empty());

    double sum = 0.0;

    for (const std::size_t i : indices)
    {
        sum += values(static_cast<Eigen::Index>(i));
    }

    return sum / static_cast<double>(indices.size());
}

// The partner atoms of one atom among the atoms of the given element,
// grouped by distance class (0 = nearest, 1 = second, ...). Geometry-based
// pairing: the canonical order is a sort, so the tests never assume an
// index layout.
std::vector<std::vector<std::size_t>> PartnerDistanceClasses(
    const qcx::molecule::Molecule& molecule, std::size_t atomIndex, int atomicNumber) {
    const auto& atoms = molecule.Atoms();
    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<std::pair<double, std::size_t>> distances;

    for (std::size_t j = 0; j < atoms.size(); ++j)
    {
        if (j == atomIndex || atoms[j].atomicNumber != atomicNumber)
        {
            continue;
        }

        double distanceSquared = 0.0;

        for (std::size_t k = 0; k < 3; ++k)
        {
            const double delta = coordinates(atomIndex, k) - coordinates(j, k);
            distanceSquared += delta * delta;
        }

        distances.emplace_back(distanceSquared, j);
    }

    std::sort(distances.begin(), distances.end());
    std::vector<std::vector<std::size_t>> classes;
    double previousDistance = -1.0;

    for (const auto& [distanceSquared, j] : distances)
    {
        const double distance = std::sqrt(distanceSquared);

        if (classes.empty() || std::abs(distance - previousDistance) > 1e-9)
        {
            classes.push_back({});
            previousDistance = distance;
        }

        classes.back().push_back(j);
    }

    return classes;
}

// All unordered pairs (i, j) with i < j and j in the k-th distance class of
// i's partners of the other element. For the mutual nearest-neighbor classes
// of a symmetric ring this collects every symmetry-equivalent pair exactly
// once.
std::vector<std::pair<std::size_t, std::size_t>> PartnerPairs(
    const qcx::molecule::Molecule& molecule,
    // element is the anchor atom's species, otherElement the partner species.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    int element,
    int otherElement,
    std::size_t distanceClass) {
    const auto& atoms = molecule.Atoms();
    std::vector<std::pair<std::size_t, std::size_t>> pairs;

    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        if (atoms[i].atomicNumber != element)
        {
            continue;
        }

        const auto classes = PartnerDistanceClasses(molecule, i, otherElement);

        if (classes.size() <= distanceClass)
        {
            continue;
        }

        for (const std::size_t j : classes[distanceClass])
        {
            if (i < j)
            {
                pairs.emplace_back(i, j);
            }
        }
    }

    return pairs;
}

// Mean of a symmetric matrix over a class of symmetry-equivalent pairs,
// asserting the matrix symmetry and that every pair in the class carries
// the same value (the converged density at the exact symmetry geometry).
double MeanPairValues(const Eigen::MatrixXd& matrix,
                      const std::vector<std::pair<std::size_t, std::size_t>>& pairs) {
    EXPECT_FALSE(pairs.empty());
    double first = 0.0;
    double sum = 0.0;

    for (std::size_t p = 0; p < pairs.size(); ++p)
    {
        const auto& [i, j] = pairs[p];
        const double value = matrix(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j));
        EXPECT_NEAR(matrix(j, i), value, 1e-12);

        if (p == 0)
        {
            first = value;
        }

        EXPECT_NEAR(value, first, 1e-6);
        sum += value;
    }

    return sum / static_cast<double>(pairs.size());
}

} // namespace

TEST(NalewajskiTest, H2Sto3gSingleBondIsExact) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto [alpha, beta] = SpinDensities(*scf);
    auto result = qcx::properties::AnalyzeNalewajskiBondOrders(*molecule, *basis, alpha, beta);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The H2/STO-3G case is algebraically exact: the symmetric MO gives the
    // orthogonalized density P1 = 1/2 (I + X), so v_i = 1/4, v_ab = 1/2, and
    // the Scheme-III weighting (1 + 2 v_i/total_ab = 2) doubles the
    // diatomic contribution: B = 1.0 - the single-bond unit.
    EXPECT_EQ(result->bondOrders.rows(), 2);
    EXPECT_NEAR(result->bondOrders(0, 1), 1.0, 1e-6);
    EXPECT_NEAR(result->bondOrders(1, 0), 1.0, 1e-6);
    EXPECT_NEAR(result->bondOrders(0, 0), 0.0, 1e-12);
    EXPECT_NEAR(result->diatomicCovalent(0, 1), 0.5, 1e-6);
    EXPECT_NEAR(result->totalValence(0), 0.5, 1e-6);
    EXPECT_NEAR(result->atomicIonicValence(0), 0.25, 1e-6);
    EXPECT_NEAR(result->atomicIonicValence(1), 0.25, 1e-6);
    EXPECT_NEAR(result->atomicCovalentValence(0), 0.0, 1e-12);
    EXPECT_NEAR(result->atomicCovalentValence(1), 0.0, 1e-12);
}

TEST(NalewajskiTest, H2oSto3gSymmetryAndPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto [alpha, beta] = SpinDensities(*scf);
    auto result = qcx::properties::AnalyzeNalewajskiBondOrders(*molecule, *basis, alpha, beta);
    ASSERT_TRUE(result.has_value()) << result.error().message;

    // The C2v-symmetric converged density: the two O-H bonds are exactly
    // equivalent (MeanPairValues asserts the pair equality).
    const auto ohPairs = PartnerPairs(*molecule, 1, 8, 0);
    const auto hhPairs = PartnerPairs(*molecule, 1, 1, 0);
    ASSERT_EQ(ohPairs.size(), 2u);
    ASSERT_EQ(hhPairs.size(), 1u);

    const double oh = MeanPairValues(result->bondOrders, ohPairs);
    const double hh = MeanPairValues(result->bondOrders, hhPairs);

    // The numpy cross-validation pins: B(O-H) = 1.2000, B(H-H) = 6.76e-4.
    EXPECT_NEAR(oh, kH2oOhBondOrder, 1e-3);
    EXPECT_NEAR(hh, kH2oHhBondOrder, 1e-5);
    EXPECT_NEAR(MeanPairValues(result->diatomicCovalent, ohPairs), kH2oOhDiatomic, 1e-5);
    EXPECT_NEAR(MeanPairValues(result->diatomicCovalent, hhPairs), kH2oHhDiatomic, 1e-6);

    // The one-center valences: the invariant combo v_i + v_c, plus the
    // hydrogen's exact zeros (a single-function fragment has no off-diagonal
    // pairs).
    const Eigen::VectorXd oneCenter = result->atomicIonicValence + result->atomicCovalentValence;
    EXPECT_NEAR(MeanOverElement(*molecule, oneCenter, 8), kH2oOxygenValence, 1e-4);
    EXPECT_NEAR(MeanOverElement(*molecule, oneCenter, 1), kH2oHydrogenValence, 1e-5);
    EXPECT_NEAR(MeanOverElement(*molecule, result->atomicCovalentValence, 1), 0.0, 1e-12);
    EXPECT_NEAR(
        MeanOverElement(*molecule, result->atomicIonicValence, 1), kH2oHydrogenValence, 1e-5);

    // total_ab(a) = sum_b v_ab(a, b) by construction.
    EXPECT_NEAR(MeanOverElement(*molecule, result->totalValence, 8), kH2oOxygenTotal, 1e-5);
    EXPECT_NEAR(MeanOverElement(*molecule, result->totalValence, 1), kH2oHydrogenTotal, 1e-5);

    // The paper's magnitude class (Fig. 3, water O-H = -1.2, 3-21G UHF;
    // the paper's bonding valences are negative): a near-single O-H bond,
    // and the H-H pair is not a bond.
    EXPECT_GT(oh, 1.0);
    EXPECT_LT(oh, 1.4);
    EXPECT_LT(hh, 0.01 * oh);
}

TEST(NalewajskiTest, BenzeneSto3gSymmetryAndPins) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "real SCF: Release-only";
    }

    auto basis = MakeBenzeneSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeBenzeneSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto scf = RunDenseRhf(*molecule, *basis);
    ASSERT_TRUE(scf.has_value()) << scf.error().message;
    EXPECT_TRUE(scf->converged);

    auto [alpha, beta] = SpinDensities(*scf);
    auto result = qcx::properties::AnalyzeNalewajskiBondOrders(*molecule, *basis, alpha, beta);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->bondOrders.rows(), 12);

    // The D6h ring geometry mapped by distance classes (no index-layout
    // assumptions): six bonded C-C, six meta, three para; six bonded C-H;
    // six H-H ortho edges, six meta, three para (each H has two ortho and
    // two meta partners, so the distinct unordered pairs are the six edges
    // of the two hexagons).
    const auto ccBonded = PartnerPairs(*molecule, 6, 6, 0);
    const auto ccMeta = PartnerPairs(*molecule, 6, 6, 1);
    const auto ccPara = PartnerPairs(*molecule, 6, 6, 2);
    const auto chBonded = PartnerPairs(*molecule, 1, 6, 0);
    const auto hhOrtho = PartnerPairs(*molecule, 1, 1, 0);
    const auto hhMeta = PartnerPairs(*molecule, 1, 1, 1);
    const auto hhPara = PartnerPairs(*molecule, 1, 1, 2);
    ASSERT_EQ(ccBonded.size(), 6u);
    ASSERT_EQ(ccMeta.size(), 6u);
    ASSERT_EQ(ccPara.size(), 3u);
    ASSERT_EQ(chBonded.size(), 6u);
    ASSERT_EQ(hhOrtho.size(), 6u);
    ASSERT_EQ(hhMeta.size(), 6u);
    ASSERT_EQ(hhPara.size(), 3u);

    // The numpy cross-validation pins (MeanPairValues asserts the D6h pair
    // equality): bonded C-C 1.4432, meta 0.1158, para 7.89e-4; C-H 0.9809;
    // H-H ortho 2.20e-3.
    const double cc = MeanPairValues(result->bondOrders, ccBonded);
    const double ccMetaBond = MeanPairValues(result->bondOrders, ccMeta);
    const double ccParaBond = MeanPairValues(result->bondOrders, ccPara);
    const double ch = MeanPairValues(result->bondOrders, chBonded);
    const double hhOrthoBond = MeanPairValues(result->bondOrders, hhOrtho);
    EXPECT_NEAR(cc, kBenzeneCcBondOrder, 1e-3);
    EXPECT_NEAR(ccMetaBond, kBenzeneCcMetaBondOrder, 1e-3);
    EXPECT_NEAR(ccParaBond, kBenzeneCcParaBondOrder, 1e-5);
    EXPECT_NEAR(ch, kBenzeneChBondOrder, 1e-3);
    EXPECT_NEAR(hhOrthoBond, kBenzeneHhOrthoBondOrder, 1e-5);
    EXPECT_NEAR(MeanPairValues(result->bondOrders, hhMeta), 5.24176007e-4, 1e-5);
    EXPECT_NEAR(MeanPairValues(result->bondOrders, hhPara), 8.14858958e-5, 1e-5);

    // The diatomic contributions and the row totals.
    EXPECT_NEAR(MeanPairValues(result->diatomicCovalent, ccBonded), kBenzeneCcDiatomic, 1e-5);
    EXPECT_NEAR(MeanPairValues(result->diatomicCovalent, chBonded), kBenzeneChDiatomic, 1e-5);
    EXPECT_NEAR(MeanOverElement(*molecule, result->totalValence, 6), kBenzeneCarbonTotal, 1e-5);
    EXPECT_NEAR(MeanOverElement(*molecule, result->totalValence, 1), kBenzeneHydrogenTotal, 1e-5);

    // The one-center invariant combo v_i + v_c: equal across the six
    // carbons and across the six hydrogens (the D6h density), with the
    // numpy pins.
    const Eigen::VectorXd oneCenter = result->atomicIonicValence + result->atomicCovalentValence;
    const auto carbonIndices = ElementIndices(*molecule, 6);
    const auto hydrogenIndices = ElementIndices(*molecule, 1);
    double carbonExtent = 0.0;
    double hydrogenExtent = 0.0;

    for (const std::size_t i : carbonIndices)
    {
        const double value = oneCenter(static_cast<Eigen::Index>(i));
        carbonExtent = std::max(
            carbonExtent, std::abs(value - oneCenter(static_cast<Eigen::Index>(carbonIndices[0]))));
    }

    for (const std::size_t i : hydrogenIndices)
    {
        const double value = oneCenter(static_cast<Eigen::Index>(i));
        hydrogenExtent =
            std::max(hydrogenExtent,
                     std::abs(value - oneCenter(static_cast<Eigen::Index>(hydrogenIndices[0]))));
    }

    EXPECT_LT(carbonExtent, 1e-6);
    EXPECT_LT(hydrogenExtent, 1e-6);
    EXPECT_NEAR(MeanOverElement(*molecule, oneCenter, 6), kBenzeneCarbonValence, 1e-4);
    EXPECT_NEAR(MeanOverElement(*molecule, oneCenter, 1), kBenzeneHydrogenValence, 1e-5);
    EXPECT_NEAR(MeanOverElement(*molecule, result->atomicCovalentValence, 1), 0.0, 1e-12);

    // The magnitude ordering: aromatic C-C between the single and double
    // bond units (the paper's STO-6G value is -1.005, Fig. 3; the STO-3G
    // minimal basis reads higher), C-H near one (paper -0.909..-1.008),
    // the H-H pairs decaying with distance.
    EXPECT_GT(cc, 1.0);
    EXPECT_LT(cc, 2.0);
    EXPECT_GT(cc, ch);
    EXPECT_GT(ch, 0.7);
    EXPECT_LT(ch, 1.3);
    EXPECT_GT(ch, hhOrthoBond);
    EXPECT_GT(hhOrthoBond, MeanPairValues(result->bondOrders, hhMeta));
}

TEST(NalewajskiTest, RejectsInvalidInputs) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // Wrong density shape: the H2O/STO-3G AO count is 7.
    Eigen::MatrixXd badDensity = Eigen::MatrixXd::Zero(3, 3);
    auto badShape =
        qcx::properties::AnalyzeNalewajskiBondOrders(*molecule, *basis, badDensity, badDensity);
    ASSERT_FALSE(badShape.has_value());
    EXPECT_EQ(badShape.error().code, qcx::ErrorCode::kInvalidArgument);

    // Mismatched spin channels are rejected too.
    auto mixed = qcx::properties::AnalyzeNalewajskiBondOrders(
        *molecule, *basis, Eigen::MatrixXd::Identity(7, 7), Eigen::MatrixXd::Zero(8, 8));
    ASSERT_FALSE(mixed.has_value());
    EXPECT_EQ(mixed.error().code, qcx::ErrorCode::kInvalidArgument);
}
