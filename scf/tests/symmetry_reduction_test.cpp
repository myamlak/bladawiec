// The petite-list AO-space symmetry reduction (the qcx::scf
// seam). The tests pin: the water C2v extraction (group order 4, every
// element an involution, the extracted actions matching the hand-built
// set, the end-to-end pair-class tables over the real STO-3G pair list),
// the C1 trivial fallback, the axis-aligned-subgroup fallback (a rotated
// molecule keeps the elements that survive as signed coordinate
// permutations - the molecular plane's mirror for water turned about z, C2h
// for benzene), and the kUnimplemented gate when no non-identity element
// survives (the automatic correctness check of the signed-permutation
// applicability).

#include "benzene_sto3g.hpp"
#include "corpus.hpp"
#include "h2o_sto3g.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/scf/symmetry_reduction.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace {

using qcx::integrals::BuildPairClasses;
using qcx::integrals::BuildShellPairs;
using qcx::integrals::ClassPair;
using qcx::integrals::GenerateClassPairOrbits;
using qcx::integrals::PairClass;
using qcx::integrals::PairClassTable;
using qcx::integrals::SymmetryReduction;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// The lean builder's core-Hamiltonian argument type (direct_rhf_test.cpp's
// alias, repeated here - test TUs do not share one).
using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;

// The table's containers carry the class_table family's tagged
// allocator (symmetry_reduction.hpp); expected-value literals name it so
// the gtest comparisons type-match (vector operator== is same-type).
using TaggedSizeVector = std::vector<std::size_t, qcx::memory::TaggedAllocator<std::size_t>>;

// STO-3G merged from the vendored corpus files (the symmetry_block_test
// pattern; single-source for C, H, F, Cl alike).
qcx::Result<qcx::basisset::BasisSet> MakeSto3g(const std::vector<std::string_view>& symbols) {
    qcx::basisset::BasisSet merged;

    for (const std::string_view symbol : symbols)
    {
        const std::string path =
            std::string(QcxBasisDataDir) + "/sto-3g/" + std::string(symbol) + ".nwchem";
        auto parsed = qcx::basisset::ParseNwchemFile(path);

        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }

        auto mergeResult = merged.Merge(*parsed);

        if (!mergeResult.has_value())
        {
            return std::unexpected(mergeResult.error());
        }
    }

    return merged;
}

// The hand-built C2v action on the 7 water STO-3G functions. The merged
// basis (MakeH2oSto3gBasis) carries the two H s-shells first, so the
// function order is [H1s, H2s, O1s, O2s, Opy, Opz, Opx] (the module's
// spherical-function convention: m ascending -l..+l, so m=-1 ~ y, m=0 ~ z,
// m=+1 ~ x). With the molecule in the xy plane (H at (+-d, h, 0)), the C2v
// elements are {I, sigma_z (z-flip), sigma_x (x-flip), C2 (pi about y)};
// sigma_x and C2 both swap the two H functions - the H atoms differ only in
// x. The extracted elements must equal this set as a multiset - the
// detector's element order is not pinned.
struct HandBuiltElement {
    std::vector<std::size_t> permutation;
    std::vector<int> sign;
};

const std::vector<HandBuiltElement> kHandBuiltC2vActions = {
    {{0, 1, 2, 3, 4, 5, 6}, {1, 1, 1, 1, 1, 1, 1}}, // I.
    {{0, 1, 2, 3, 4, 5, 6}, {1, 1, 1, 1, 1, -1, 1}}, // sigma(z-flip): p_z -> -p_z.
    {{1, 0, 2, 3, 4, 5, 6}, {1, 1, 1, 1, 1, 1, -1}}, // sigma(x-flip): H swap, p_x -> -p_x.
    {{1, 0, 2, 3, 4, 5, 6}, {1, 1, 1, 1, 1, -1, -1}}, // C2: H swap, p_z and p_x flip.
};

// True when an extracted element equals one hand-built action; the match is
// consumed (multiset semantics) by removing it from the candidates.
bool MatchAndConsume(const HandBuiltElement& element, std::vector<HandBuiltElement>& candidates) {
    for (std::size_t i = 0; i < candidates.size(); ++i)
    {
        if (candidates[i].permutation == element.permutation && candidates[i].sign == element.sign)
        {
            candidates.erase(candidates.begin() + static_cast<std::ptrdiff_t>(i));
            return true;
        }
    }

    return false;
}

// The rotated water: the fixture geometry turned by 30 degrees about the z
// axis, so the C2v mirror normals are no longer global axes. The C2 axis is
// the H-O-H bisector - the global y for the unrotated fixture, see the
// hand-built table below - and the turn takes it off the global frame, so
// the C2 does NOT survive. What survives is the molecular PLANE: the
// molecule stays in z = 0, the z-flip maps every atom to itself, and the
// gate's axis-aligned-subgroup fallback therefore selects {E, sigma_z} - a
// subgroup that permutes no function and so acts trivially on every shell
// pair.
qcx::Result<qcx::molecule::Molecule> MakeRotatedWater() {
    constexpr double kCos30 = 0.8660254037844387;
    constexpr double kSin30 = 0.5;
    constexpr double kX = 1.430428808474167;
    constexpr double kY = 1.107157044080814;

    Eigen::MatrixXd coordinates(3, 3);
    coordinates.row(0) << 0.0, 0.0, 0.0;

    for (const double sign : {-1.0, 1.0})
    {
        const double x = sign * kX;
        const Eigen::Index row = sign < 0 ? 1 : 2;
        coordinates.row(row) << x * kCos30 - kY * kSin30, x * kSin30 + kY * kCos30, 0.0;
    }

    auto tensorResult = ToTensor(coordinates);

    if (!tensorResult.has_value())
    {
        return std::unexpected(tensorResult.error());
    }

    return qcx::molecule::Molecule::Create({qcx::molecule::Atom{"O", 8, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0}},
                                           std::move(*tensorResult),
                                           0,
                                           1);
}

// Water turned 45 degrees about the (1,1,1)/sqrt(3) axis on top of the
// base geometry (the H atoms at +-(kX, kY, 0)): the C2 axis and both mirror
// normals leave the global frame, so no non-identity element is a signed
// coordinate permutation - the fallback finds no axis-aligned subgroup and
// the gate reports kUnimplemented. (A rotation about a global axis would
// keep that axis' element aligned - e.g. 45 degrees about x keeps the
// yz-plane mirror whose normal is the rotation axis.)
qcx::Result<qcx::molecule::Molecule> MakeRotatedWater3d() {
    constexpr double kX = 1.430428808474167;
    constexpr double kY = 1.107157044080814;
    constexpr double kCos45 = 0.7071067811865476;
    constexpr double kInvSqrt3 = 0.5773502691896258;

    // The 45-degree rotation about n = (1,1,1)/sqrt(3):
    // R = cos(theta) I + (1 - cos(theta)) n n^T + sin(theta) [n]_x.
    const Eigen::Matrix3d skew =
        kInvSqrt3 *
        (Eigen::Matrix3d() << 0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0).finished();
    const Eigen::Matrix3d rotation = kCos45 * Eigen::Matrix3d::Identity() +
                                     (1.0 - kCos45) / 3.0 * Eigen::Matrix3d::Ones() + kCos45 * skew;

    Eigen::MatrixXd coordinates(3, 3);
    coordinates.row(0) << 0.0, 0.0, 0.0;

    for (const double sign : {-1.0, 1.0})
    {
        const Eigen::Vector3d hydrogen{kX * sign, kY, 0.0};
        const Eigen::Index row = sign < 0 ? 1 : 2;
        coordinates.row(row) = (rotation * hydrogen).transpose();
    }

    auto tensorResult = ToTensor(coordinates);

    if (!tensorResult.has_value())
    {
        return std::unexpected(tensorResult.error());
    }

    return qcx::molecule::Molecule::Create({qcx::molecule::Atom{"O", 8, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0}},
                                           std::move(*tensorResult),
                                           0,
                                           1);
}

// The planar HOCl fixture: O at the origin, H 0.97 A along +x, Cl 1.69 A
// from O at 103 degrees - EVERY atom in the xy plane, so the molecular
// point group is Cs with the molecular plane itself as the mirror. This is
// the shape the mechanism-value audit measured the engagement defect on
// (its hocl/Cs row: 987 ERI blocks either way, exactly 1.0000x).
qcx::Result<qcx::molecule::Molecule> MakePlanarHocl() {
    constexpr double kBohrPerAngstrom = 1.8897259886;
    constexpr double kOhBohr = 0.97 * kBohrPerAngstrom;
    constexpr double kOclBohr = 1.69 * kBohrPerAngstrom;
    constexpr double kAngle = 103.0 * std::numbers::pi / 180.0;

    Eigen::MatrixXd coordinates(3, 3);
    coordinates.row(0) << 0.0, 0.0, 0.0;
    coordinates.row(1) << kOhBohr, 0.0, 0.0;
    coordinates.row(2) << kOclBohr * std::cos(kAngle), kOclBohr * std::sin(kAngle), 0.0;

    auto tensorResult = ToTensor(coordinates);

    if (!tensorResult.has_value())
    {
        return std::unexpected(tensorResult.error());
    }

    return qcx::molecule::Molecule::Create({qcx::molecule::Atom{"O", 8, 0.0},
                                            qcx::molecule::Atom{"H", 1, 0.0},
                                            qcx::molecule::Atom{"Cl", 17, 0.0}},
                                           std::move(*tensorResult),
                                           0,
                                           1);
}

// THE ENGAGEMENT TEST IS THE GROUP'S ACTION ON SHELL PAIRS (the
// mechanism-value audit, 2026-09-13), and this is its measured shape: a
// group can be non-C1 and still unable to reduce anything. Every element of
// a Cs group whose mirror plane contains every atom acts on the basis
// functions as a pure sign flip - no index moves - so it maps every
// canonical shell pair onto itself, every orbit is a singleton, and the
// reduction buys exactly 1.0000x while still paying for the classification,
// the class tables and the orbit-action tables.
//
// The fixture is the audit's own: hocl, planar, 13 BF in its row. Three
// things are pinned TOGETHER, because the defect was that the third did not
// follow from the first two: the group IS found (not the C1 fallback), the
// tables carry a truthful action (the permutation rows are the identity and
// the signs are not - the mirror really does flip p_z), and the flag the
// consumers read says what that means - nothing to exploit.
TEST(SymmetryReductionTest, PlanarCsGroupActsTriviallyOnEveryShellPair) {
    const auto moleculeResult = MakePlanarHocl();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = MakeSto3g({"O", "H", "Cl"});
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_TRUE(reductionResult.has_value()) << reductionResult.error().message;
    const SymmetryReduction& reduction = *reductionResult;

    // Found, and it is Cs: the molecular plane is the mirror.
    EXPECT_EQ(reduction.groupOrder, 2u);
    ASSERT_EQ(reduction.permutation.size(), 2u);
    ASSERT_EQ(reduction.sign.size(), 2u);

    bool anySignFlip = false;

    for (std::size_t g = 0; g < reduction.groupOrder; ++g)
    {
        for (std::size_t f = 0; f < reduction.permutation[g].size(); ++f)
        {
            EXPECT_EQ(reduction.permutation[g][f], f)
                << "element " << g << " moves function " << f
                << " - a planar Cs element permutes no function";
            anySignFlip = anySignFlip || reduction.sign[g][f] != 1;
        }
    }

    EXPECT_TRUE(anySignFlip) << "the mirror is a sign flip, not the identity";
    EXPECT_TRUE(reduction.isTrivial)
        << "a group whose every element fixes every shell pair removes no work: the "
           "engagement flag must say so";
}

// The core Hamiltonian H = T + V for the planar fixture, from the public
// one-electron seam (the lean tests' helper shape).
qcx::Result<CpuTensor2> PlanarHoclCore(const qcx::molecule::Molecule& molecule,
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

    return kinetic;
}

// A deterministic symmetric probe density (the lean tests' PhysicalDensity
// shape, local to this fixture): symmetric by construction, which is the
// builder's stated input contract.
// (n, seed) is the size-then-seed order of the random-density helper.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd PlanarHoclDensity(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

// THE MEASURED EFFECT, on the audit's own fixture class: with the planar Cs
// group reading trivial, a lean builder handed that reduction AND asked for
// the orbit expansion must produce the NO-SYMMETRY build - the same ERI block
// count and the same Fock matrix BYTE for byte, not merely within a
// tolerance. The audit verified the C1 control bit-identical on every
// counter; this is the same path reached by a group that exists.
//
// It is a byte pin deliberately: the orbit expansion is a NUMERICS change
// where it engages (it replaces a member's own evaluation with the
// representative's), so equality of the count alone would not say the
// mechanism is off. Byte identity says it.
TEST(SymmetryReductionTest, PlanarCsEngagedLeanBuildIsTheNoSymmetryBuildByteForByte) {
    const auto moleculeResult = MakePlanarHocl();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = MakeSto3g({"O", "H", "Cl"});
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_TRUE(reductionResult.has_value()) << reductionResult.error().message;
    ASSERT_TRUE(reductionResult->isTrivial);
    ASSERT_EQ(reductionResult->groupOrder, 2u) << "the fixture must reach the engagement gate";

    const auto coreResult = PlanarHoclCore(*moleculeResult, *basisResult);
    ASSERT_TRUE(coreResult.has_value()) << coreResult.error().message;

    const auto pairListResult = BuildShellPairs(*moleculeResult, *basisResult);
    ASSERT_TRUE(pairListResult.has_value());

    qcx::integrals::LeanFockBuildOptions plainOptions;
    plainOptions.accuracy = qcx::integrals::AccuracyPreset::kNormal;
    plainOptions.maxParallelChunks = 1;
    auto plain = qcx::integrals::LeanDirectFockBuilder::Create(
        *moleculeResult, *basisResult, *coreResult, plainOptions);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    qcx::integrals::LeanFockBuildOptions engagedOptions = plainOptions;
    engagedOptions.symmetryReduction = &*reductionResult;
    engagedOptions.symmetryOrbitExpansion = true;
    auto engaged = qcx::integrals::LeanDirectFockBuilder::Create(
        *moleculeResult, *basisResult, *coreResult, engagedOptions);
    ASSERT_TRUE(engaged.has_value()) << engaged.error().message;

    const Eigen::MatrixXd density = PlanarHoclDensity(pairListResult->functionCount, 20260913);
    qcx::integrals::FockBuildStats plainStats;
    auto plainFock = plain->BuildFock(*ToTensor(density), &plainStats);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    qcx::integrals::FockBuildStats engagedStats;
    auto engagedFock = engaged->BuildFock(*ToTensor(density), &engagedStats);
    ASSERT_TRUE(engagedFock.has_value()) << engagedFock.error().message;

    const Eigen::MatrixXd plainMatrix = ToMatrix(*plainFock);
    const Eigen::MatrixXd engagedMatrix = ToMatrix(*engagedFock);
    ASSERT_EQ(plainMatrix.size(), engagedMatrix.size());

    EXPECT_EQ(engagedStats.fp64QuartetCount, plainStats.fp64QuartetCount)
        << "a pair-trivial group must not change the ERI block count";
    EXPECT_EQ(std::memcmp(plainMatrix.data(),
                          engagedMatrix.data(),
                          static_cast<std::size_t>(plainMatrix.size()) * sizeof(double)),
              0)
        << "the expanded leg moved a byte: the pair-trivial group was engaged";
}

TEST(SymmetryReductionTest, WaterC2vExtractionMatchesHandBuilt) {
    const auto moleculeResult = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_TRUE(reductionResult.has_value());

    const SymmetryReduction& reduction = *reductionResult;
    EXPECT_EQ(reduction.groupOrder, 4u);
    EXPECT_FALSE(reduction.isTrivial);
    ASSERT_EQ(reduction.permutation.size(), 4u);
    ASSERT_EQ(reduction.sign.size(), 4u);

    for (std::size_t g = 0; g < reduction.groupOrder; ++g)
    {
        ASSERT_EQ(reduction.permutation[g].size(), 7u);
        ASSERT_EQ(reduction.sign[g].size(), 7u);

        // Every element is an involution: applying it twice returns each
        // function to itself with a positive sign (A(g)^2 = I).
        for (std::size_t i = 0; i < 7; ++i)
        {
            const std::size_t image = reduction.permutation[g][i];
            EXPECT_EQ(reduction.permutation[g][image], i);
            EXPECT_EQ(reduction.sign[g][i] * reduction.sign[g][image], 1);
        }
    }

    // The extracted action set equals the hand-built one (multiset).
    std::vector<HandBuiltElement> candidates = kHandBuiltC2vActions;

    for (std::size_t g = 0; g < reduction.groupOrder; ++g)
    {
        EXPECT_TRUE(MatchAndConsume(HandBuiltElement{reduction.permutation[g], reduction.sign[g]},
                                    candidates))
            << "extracted element " << g << " matches no hand-built action";
    }

    EXPECT_TRUE(candidates.empty());
}

TEST(SymmetryReductionTest, WaterC2vEndToEndClassTables) {
    const auto moleculeResult = qcx::testing::MakeH2oSto3g();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_TRUE(reductionResult.has_value());

    const auto pairListResult = BuildShellPairs(*moleculeResult, *basisResult);
    ASSERT_TRUE(pairListResult.has_value());
    // The no-screening table: all-ones Schwarz bounds keep every class
    // pair (the class-bound test product >= 0.0).
    const std::vector<double> schwarz(pairListResult->pairs.size(), 1.0);
    auto tableResult = BuildPairClasses(*reductionResult, *pairListResult, schwarz, 0.0);
    ASSERT_TRUE(tableResult.has_value());

    PairClassTable& table = *tableResult;
    // The 5 shells (H1s, H2s, O1s, O2s, Op) of the merged basis give 15
    // shell pairs over the 7 functions.
    const std::size_t nPairs = pairListResult->pairs.size();
    EXPECT_EQ(nPairs, 15u);

    // shellOfFunction[f] = the shell containing function f: the image of a
    // shell under g is the shell containing the image of its first
    // function - the corrected semantics (the reduction acts
    // on FUNCTIONS, the pair list indexes SHELLS). The H-first fixture has
    // shell index == function offset, which masks a shell-as-function walk,
    // so the closure check below uses the corrected form explicitly.
    std::vector<std::size_t> shellOfFunction(pairListResult->functionCount);

    for (std::size_t s = 0; s < pairListResult->shells.size(); ++s)
    {
        const auto& shell = pairListResult->shells[s];

        for (std::size_t f = shell.functionOffset;
             f < shell.functionOffset + qcx::integrals::ShellFunctionCount(shell);
             ++f)
        {
            shellOfFunction[f] = s;
        }
    }

    // Every pair lands in exactly one class and the class members are
    // closed under the group action: the image of any member under any
    // element stays in the class.
    std::size_t memberTotal = 0;

    for (const PairClass& cls : table.classes)
    {
        memberTotal += cls.members.size();
        EXPECT_EQ(cls.members[0], cls.repPair);

        for (const std::size_t member : cls.members)
        {
            for (std::size_t g = 0; g < reductionResult->groupOrder; ++g)
            {
                const auto& memberPair = pairListResult->pairs[member];
                const std::size_t imageI =
                    shellOfFunction[reductionResult->permutation
                                        [g][pairListResult->shells[memberPair.i].functionOffset]];
                const std::size_t imageJ =
                    shellOfFunction[reductionResult->permutation
                                        [g][pairListResult->shells[memberPair.j].functionOffset]];
                const std::size_t imagePair = qcx::integrals::PairIndexOf(
                    std::min(imageI, imageJ), std::max(imageI, imageJ), *pairListResult);
                EXPECT_EQ(table.classOfPair[imagePair], table.classOfPair[member]);
            }
        }
    }

    EXPECT_EQ(memberTotal, nPairs);

    // The harness-verified structure of the real water pair list: 11
    // classes, 66 class pairs, 76 member-quartet orbits, 10 class pairs
    // carrying more than one orbit (the harness's n_multi; the counts are
    // convention-independent - the same values the integrals test pins on
    // its hand-built O-first list).
    EXPECT_EQ(table.classes.size(), 11u);
    EXPECT_EQ(table.classPairs.size(), 66u);

    // The root restructure's laziness: Create stores the class pairs with
    // EMPTY orbit vectors (the on-demand generation fills them per reached
    // pair).
    for (const ClassPair& classPair : table.classPairs)
    {
        EXPECT_TRUE(classPair.orbits.empty())
            << "class pair (" << classPair.p << ", " << classPair.q << ") materialized at Create";
    }

    // The on-demand fill: generate every kept class pair (all 66 - the
    // no-screening table), the deterministic pure function of the class
    // members - the pinned structure below is the eager table's.
    for (std::size_t cp = 0; cp < table.classPairs.size(); ++cp)
    {
        GenerateClassPairOrbits(table, *reductionResult, *pairListResult, cp);
    }

    std::size_t orbitTotal = 0;
    std::size_t multiOrbitClassPairs = 0;

    for (const auto& classPair : table.classPairs)
    {
        orbitTotal += classPair.orbits.size();

        if (classPair.orbits.size() > 1)
        {
            ++multiOrbitClassPairs;
        }
    }

    EXPECT_EQ(orbitTotal, 76u);
    EXPECT_EQ(multiOrbitClassPairs, 10u);

    // THE COVERAGE INVARIANT, pinned as a SET (the mechanism-value audit's
    // 119-of-120 lead, 2026-09-13): the class path emits one task per orbit
    // and distributes that orbit's SCREENED members, so a cell the orbit
    // expansion does not carry is a cell nothing accumulates. The union of
    // every orbit's members must therefore be EXACTLY the canonical cell
    // set - not merely of equal size, which a swap would satisfy. The count
    // is the no-screening cell count n(n+1)/2 = 120 here, and the machinery
    // side of the lean tests' cross-check asserts the same equality from
    // the other end (memberQuartets == the plain walk's surviving cells).
    std::vector<bool> cellSeen(nPairs * nPairs, false);
    std::size_t memberQuartetTotal = 0;
    std::size_t memberDuplicates = 0;

    for (const auto& classPair : table.classPairs)
    {
        for (const auto& orbit : classPair.orbits)
        {
            for (const auto& member : orbit.members)
            {
                ++memberQuartetTotal;
                const std::size_t key = member.braPair * nPairs + member.ketPair;

                if (cellSeen[key])
                {
                    ++memberDuplicates;
                }

                cellSeen[key] = true;
            }
        }
    }

    std::size_t cellsCovered = 0;

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (cellSeen[bra * nPairs + ket])
            {
                ++cellsCovered;
            }
        }
    }

    EXPECT_EQ(memberDuplicates, 0u) << "an orbit member is claimed by two orbits";
    EXPECT_EQ(memberQuartetTotal, nPairs * (nPairs + 1) / 2)
        << "the orbit members are not the full canonical cell set";
    EXPECT_EQ(cellsCovered, memberQuartetTotal)
        << "some canonical cell is in NO orbit - it would contribute nothing to F";

    // The (H1s, H1s) class = {pair (0, 0), pair (1, 1)} = pair indices
    // {0, 5}: the H-first basis carries the two H s-shells at indices 0
    // and 1, and sigma_x/C2 swap them.
    const std::size_t h1sH1s = qcx::integrals::PairIndexOf(0, 0, *pairListResult);
    EXPECT_EQ(h1sH1s, 0u);
    EXPECT_EQ(qcx::integrals::PairIndexOf(1, 1, *pairListResult), 5u);
    const PairClass& hhClass = table.classes[table.classOfPair[h1sH1s]];
    EXPECT_EQ(table.classOfPair[5], table.classOfPair[h1sH1s]);
    EXPECT_EQ(hhClass.repPair, 0u);
    EXPECT_EQ(hhClass.members, (TaggedSizeVector{0, 5}));

    // The two-member class of (H1s, O2s) = pair (0, 3): sigma_x and C2
    // swap the hydrogen shells (mapping the pair to (H2s, O2s) = pair
    // (1, 3)), everything else fixes the pair.
    const std::size_t h1sO2s = qcx::integrals::PairIndexOf(0, 3, *pairListResult);
    const PairClass& hydrogenClass = table.classes[table.classOfPair[h1sO2s]];
    EXPECT_EQ(hydrogenClass.members.size(), 2u);
    EXPECT_EQ(table.classOfPair[qcx::integrals::PairIndexOf(1, 3, *pairListResult)],
              table.classOfPair[h1sO2s]);
}

TEST(SymmetryReductionTest, C1MoleculeGivesTrivialReduction) {
    const auto moleculeResult = qcx::symmetry::testing::MakeGenericC1();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = MakeSto3g({"C", "H", "F", "Cl"});
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_TRUE(reductionResult.has_value());

    const SymmetryReduction& reduction = *reductionResult;
    EXPECT_TRUE(reduction.isTrivial);
    EXPECT_EQ(reduction.groupOrder, 1u);

    // The trivial reduction is the default-constructed SymmetryReduction:
    // empty tables, the identity action implicit (BuildPairClasses
    // materializes it - an order-1 group has only the identity element).
    EXPECT_TRUE(reduction.permutation.empty());
    EXPECT_TRUE(reduction.sign.empty());

    // The class tables of a trivial reduction are the plain enumeration.
    const auto pairListResult = BuildShellPairs(*moleculeResult, *basisResult);
    ASSERT_TRUE(pairListResult.has_value());
    const std::vector<double> schwarz(pairListResult->pairs.size(), 1.0);
    const auto tableResult = BuildPairClasses(reduction, *pairListResult, schwarz, 0.0);
    ASSERT_TRUE(tableResult.has_value());
    EXPECT_EQ(tableResult->classes.size(), pairListResult->pairs.size());
}

TEST(SymmetryReductionTest, RotatedWaterFallsBackToItsPlaneMirrorSubgroup) {
    const auto moleculeResult = MakeRotatedWater();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_TRUE(reductionResult.has_value()) << reductionResult.error().message;
    const SymmetryReduction& reduction = *reductionResult;
    EXPECT_EQ(reduction.groupOrder, 2u);
    ASSERT_EQ(reduction.permutation.size(), 2u);
    ASSERT_EQ(reduction.sign.size(), 2u);

    // The identity element: the trivial action.
    for (std::size_t i = 0; i < reduction.permutation[0].size(); ++i)
    {
        EXPECT_EQ(reduction.permutation[0][i], i);
        EXPECT_EQ(reduction.sign[0][i], 1);
    }

    // WHAT THE SURVIVING ELEMENT IS, measured rather than named (the
    // mechanism-value audit, 2026-09-13, corrected this comment). The
    // 30-degree turn about z takes the ORIGINAL C2 axis (the global y, the
    // H-O-H bisector) off the global frame, so the C2 does NOT survive. What
    // does is the molecular PLANE: the molecule stays in z = 0, so the
    // z-flip is a symmetry of the turned geometry, it maps every atom to
    // itself, and its action is a pure sign flip - the identity permutation
    // with p_z negated. (An earlier version of this comment read "the C2
    // axis is the global z ... the H functions swap"; the assertions below
    // never said that, and the pair-action verdict exposes it.)
    for (std::size_t i = 0; i < reduction.permutation[1].size(); ++i)
    {
        EXPECT_EQ(reduction.permutation[1][i], i)
            << "function " << i << " moves - the surviving element permutes no function, so the "
            << "subgroup acts trivially on every shell pair";
        EXPECT_EQ(reduction.permutation[1][reduction.permutation[1][i]], i);
        EXPECT_EQ(reduction.sign[1][reduction.permutation[1][i]] * reduction.sign[1][i], 1);
    }

    // Something does happen - the sign flip - so this is a real group and
    // not the C1 fallback, and it is STILL trivial in the sense the
    // consumers read: no shell pair moves, every orbit is a singleton, and
    // the reduction removes no ERI work.
    bool anySignFlip = false;

    for (std::size_t i = 0; i < reduction.sign[1].size(); ++i)
    {
        anySignFlip = anySignFlip || reduction.sign[1][i] != 1;
    }

    EXPECT_TRUE(anySignFlip);
    EXPECT_TRUE(reduction.isTrivial) << "the surviving subgroup acts trivially on every shell pair";
}

TEST(SymmetryReductionTest, FullyRotatedWaterStillFallsBackToPlainPath) {
    const auto moleculeResult = MakeRotatedWater3d();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = qcx::testing::MakeH2oSto3gBasis();
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_FALSE(reductionResult.has_value());
    EXPECT_EQ(reductionResult.error().code, qcx::ErrorCode::kUnimplemented);
}

TEST(SymmetryReductionTest, RotatedBenzeneFallsBackToC2h) {
    const auto moleculeResult = qcx::testing::MakeRotatedBenzeneSto3g();
    ASSERT_TRUE(moleculeResult.has_value());

    const auto basisResult = qcx::testing::MakeBenzeneSto3gBasis();
    ASSERT_TRUE(basisResult.has_value());

    const auto reductionResult = qcx::scf::BuildSymmetryReduction(*moleculeResult, *basisResult);
    ASSERT_TRUE(reductionResult.has_value()) << reductionResult.error().message;
    const SymmetryReduction& reduction = *reductionResult;
    EXPECT_FALSE(reduction.isTrivial);
    // The D6h -> D2h realization's in-plane C2 axes are no longer global
    // axes after the 30-degree turn; the C6-axis C2, the inversion, and
    // their product (the molecular-plane mirror) survive as the C2h
    // subgroup.
    EXPECT_EQ(reduction.groupOrder, 4u);
    ASSERT_EQ(reduction.permutation.size(), 4u);
    ASSERT_EQ(reduction.sign.size(), 4u);

    // The identity element: the trivial action.
    for (std::size_t i = 0; i < reduction.permutation[0].size(); ++i)
    {
        EXPECT_EQ(reduction.permutation[0][i], i);
        EXPECT_EQ(reduction.sign[0][i], 1);
    }

    // Every other element is an involution, and at least one of them moves
    // or flips a function.
    bool anyNontrivial = false;

    for (std::size_t g = 1; g < reduction.groupOrder; ++g)
    {
        for (std::size_t i = 0; i < reduction.permutation[g].size(); ++i)
        {
            const std::size_t image = reduction.permutation[g][i];
            EXPECT_EQ(reduction.permutation[g][image], i);
            EXPECT_EQ(reduction.sign[g][image] * reduction.sign[g][i], 1);
            anyNontrivial = anyNontrivial || image != i || reduction.sign[g][i] != 1;
        }
    }

    EXPECT_TRUE(anyNontrivial);

    // THE COVERAGE INVARIANT ON A SECOND GROUP AND A FIVE-TIMES-LARGER CELL
    // SPACE (the mechanism-value audit's 119-of-120 lead, 2026-09-13): the
    // water C2v pin in WaterC2vEndToEndClassTables measures the union of
    // every orbit's members as a SET against the canonical cells, and this
    // repeats it for the C2h realization - a different group, a different
    // order, 36 functions against 7. The claim the audit needed is whether
    // the ORBIT CONSTRUCTION can lose a cell; one fixture cannot settle it,
    // and a group whose elements permute atoms can fail differently from a
    // C2v.
    const auto pairListResult = BuildShellPairs(*moleculeResult, *basisResult);
    ASSERT_TRUE(pairListResult.has_value());
    const std::size_t nPairsHere = pairListResult->pairs.size();
    const std::vector<double> schwarz(pairListResult->pairs.size(), 1.0);
    auto tableResult = BuildPairClasses(reduction, *pairListResult, schwarz, 0.0);
    ASSERT_TRUE(tableResult.has_value()) << tableResult.error().message;
    PairClassTable& table = *tableResult;

    for (std::size_t cp = 0; cp < table.classPairs.size(); ++cp)
    {
        GenerateClassPairOrbits(table, reduction, *pairListResult, cp);
    }

    std::vector<bool> cellSeen(nPairsHere * nPairsHere, false);
    std::size_t memberTotal = 0;
    std::size_t duplicates = 0;

    for (const auto& classPair : table.classPairs)
    {
        for (const auto& orbit : classPair.orbits)
        {
            for (const auto& member : orbit.members)
            {
                ++memberTotal;
                const std::size_t key = member.braPair * nPairsHere + member.ketPair;

                if (cellSeen[key])
                {
                    ++duplicates;
                }

                cellSeen[key] = true;
            }
        }
    }

    std::size_t cellsCovered = 0;

    for (std::size_t bra = 0; bra < nPairsHere; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (cellSeen[bra * nPairsHere + ket])
            {
                ++cellsCovered;
            }
        }
    }

    EXPECT_EQ(duplicates, 0u) << "an orbit member is claimed by two orbits";
    EXPECT_EQ(memberTotal, nPairsHere * (nPairsHere + 1) / 2)
        << "the orbit members are not the full canonical cell set";
    EXPECT_EQ(cellsCovered, memberTotal)
        << "some canonical cell is in NO orbit - it would contribute nothing to F";
}

} // namespace
