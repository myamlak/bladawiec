// The pair-class decomposition tests (symmetry_reduction.hpp):
// the class and per-orbit-member structure of the petite-list class tables
// (the member quartets' generators and slot permutations the class-aware
// contraction expands with). The reductions are hand-built here (the scf
// module owns the extraction); the scf test exercises the real extraction
// -> tables flow end to end.

#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "qcx/memory/allocation_instrument.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <unordered_set>
#include <vector>

namespace {

using qcx::integrals::BuildPairClasses;
using qcx::integrals::ClassPair;
using qcx::integrals::CountClassMaterialization;
using qcx::integrals::GenerateClassPairOrbits;
using qcx::integrals::PairClass;
using qcx::integrals::PairClassTable;
using qcx::integrals::ShellInfo;
using qcx::integrals::ShellPairIndex;
using qcx::integrals::ShellPairList;
using qcx::integrals::SymmetryReduction;

// The table's containers carry the class_table family's tagged
// allocator (symmetry_reduction.hpp); expected-value literals below name it
// so the gtest comparisons type-match (vector operator== is same-type).
using TaggedSizeVector = std::vector<std::size_t, qcx::memory::TaggedAllocator<std::size_t>>;

// Enables the attribution instrument for one measurement and restores the
// prior state on every exit path (the instrument is process-global, so a
// failed ASSERT_ must not leave it on for the rest of the binary's tests).
// A scoped RAII guard; copy is already deleted and it is never moved.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class ScopedInstrumentEnable {
public:
    ScopedInstrumentEnable() : _wasEnabled(qcx::memory::AllocationInstrumentEnabled()) {
        if (!_wasEnabled)
        {
            _enabled =
                qcx::memory::AllocationInstrumentEnable(qcx::memory::AllocationInstrumentOptions{})
                    .has_value();
        }
    }

    ~ScopedInstrumentEnable() {
        if (_enabled && !_wasEnabled)
        {
            // The Result is deliberately discarded: a destructor cannot report it.
            // NOLINTNEXTLINE(bugprone-unused-return-value)
            (void)qcx::memory::AllocationInstrumentDisable();
        }
    }

    ScopedInstrumentEnable(const ScopedInstrumentEnable&) = delete;
    ScopedInstrumentEnable& operator=(const ScopedInstrumentEnable&) = delete;

    bool Ok() const noexcept {
        return _wasEnabled || _enabled;
    }

private:
    bool _wasEnabled = false;
    bool _enabled = false;
};

// A toy basis of three s shells (functions 0, 1, 2) with the group {I, swap
// (1 <-> 2)}: every pair orbit and member sign in the resulting table is
// hand-computable, including a pair with a non-trivial stabilizer (the
// (1,2) class).
ShellPairList MakeToyPairList() {
    ShellPairList pairList;
    pairList.shells.push_back(ShellInfo{0, true, 1, 0, 0, 0});
    pairList.shells.push_back(ShellInfo{0, true, 1, 1, 1, 0});
    pairList.shells.push_back(ShellInfo{0, true, 1, 2, 2, 0});
    pairList.pairs.push_back(ShellPairIndex{0, 0});
    pairList.pairs.push_back(ShellPairIndex{0, 1});
    pairList.pairs.push_back(ShellPairIndex{0, 2});
    pairList.pairs.push_back(ShellPairIndex{1, 1});
    pairList.pairs.push_back(ShellPairIndex{1, 2});
    pairList.pairs.push_back(ShellPairIndex{2, 2});
    pairList.functionCount = 3;
    return pairList;
}

SymmetryReduction MakeToyReduction() {
    SymmetryReduction reduction;
    reduction.groupOrder = 2;
    reduction.isTrivial = false;
    reduction.permutation = {{0, 1, 2}, {0, 2, 1}};
    reduction.sign = {{1, 1, 1}, {1, 1, 1}};
    return reduction;
}

// The water STO-3G basis (the MakeH2oSto3g geometry: O at the origin, H at
// (+-d, h, 0), so the C2 axis is y and the two mirror planes are the global
// yz and xz planes): 7 functions [O1s, O2s, Opy, Opz, Opx, Hs1, Hs2] with
// the C2v elements {I, sigma_x (x-flip), sigma_z (z-flip), C2 (pi about
// y)}. Hand-built from the module's spherical-function convention (m
// ascending -l..+l: m=-1 = sin(phi) ~ y, m=0 ~ z, m=+1 = cos(phi) ~ x).
// sigma_x and C2 swap the two H functions: the H atoms differ only in x,
// so the x-flip maps one onto the other.
SymmetryReduction MakeWaterC2vReduction() {
    SymmetryReduction reduction;
    reduction.groupOrder = 4;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5, 6}, // I
        {0, 1, 2, 3, 4, 6, 5}, // sigma_x: swaps the two H functions...
        {0, 1, 2, 3, 4, 5, 6}, // sigma_z: identity permutation...
        {0, 1, 2, 3, 4, 6, 5}, // C2: swaps the two H functions.
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1, 1}, // I.
        {1, 1, 1, 1, -1, 1, 1}, // sigma_x: p_x -> -p_x.
        {1, 1, 1, -1, 1, 1, 1}, // sigma_z: p_z -> -p_z.
        {1, 1, 1, -1, -1, 1, 1}, // C2: p_x and p_z flip.
    };
    return reduction;
}

TEST(PairClassesTest, ToyReductionOrbitStructure) {
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction = MakeToyReduction();
    // The no-screening table: all-ones Schwarz bounds keep every class
    // pair (the class-bound test product >= 0.0).
    const auto table = BuildPairClasses(reduction, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0);

    ASSERT_TRUE(table.has_value());

    // The four orbits: {(0,0)}, {(0,1), (0,2)}, {(1,1), (2,2)}, {(1,2)}.
    ASSERT_EQ(table->classes.size(), 4u);
    EXPECT_EQ(table->classes[0].repPair, 0u);
    EXPECT_EQ(table->classes[1].repPair, 1u);
    EXPECT_EQ(table->classes[2].repPair, 3u);
    EXPECT_EQ(table->classes[3].repPair, 4u);

    EXPECT_EQ(table->classes[0].members, (TaggedSizeVector{0}));
    EXPECT_EQ(table->classes[1].members, (TaggedSizeVector{1, 2}));
    // The (1,1) orbit maps to (2,2) under the swap.
    EXPECT_EQ(table->classes[2].members, (TaggedSizeVector{3, 5}));
    // The (1,2) orbit is the singleton {(1,2)}: the swap element maps the
    // pair to itself (the stabilizer case).
    EXPECT_EQ(table->classes[3].members, (TaggedSizeVector{4}));

    EXPECT_EQ(table->classOfPair, (TaggedSizeVector{0, 1, 1, 2, 3, 2}));
}

TEST(PairClassesTest, ToyReductionClassPairs) {
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction = MakeToyReduction();
    // The no-screening table: all-ones Schwarz bounds keep every class
    // pair (the class-bound test product >= 0.0).
    const auto table = BuildPairClasses(reduction, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0);

    ASSERT_TRUE(table.has_value());
    ASSERT_EQ(table->classPairs.size(), 10u); // 4 * 5 / 2.

    const auto findClassPair = [&table](std::size_t p, std::size_t q) {
        const auto it = std::find_if(
            table->classPairs.begin(), table->classPairs.end(), [p, q](const ClassPair& classPair) {
                return classPair.p == p && classPair.q == q;
            });
        return it == table->classPairs.end() ? ClassPair{} : *it;
    };

    // (class 0, class 0): the rep pair (0, 0) is diagonal on both sides and
    // the quartet is self-role-swapped - the full kMult = 8.
    const ClassPair diagonal = findClassPair(0, 0);
    EXPECT_EQ(diagonal.diagonalPair, true);
    EXPECT_EQ(diagonal.pDiagonal, true);
    EXPECT_EQ(diagonal.qDiagonal, true);
    EXPECT_EQ(diagonal.kMult, 8);

    // The hand-computed multiplicity of every class pair (the kMult table of
    // the toy group: 2 per degenerate pair, 2 more when p == q).
    const int expectedKMult[4][4] = {
        {8, 2, 4, 2},
        {2, 2, 2, 1},
        {4, 2, 8, 2},
        {2, 1, 2, 2},
    };

    for (std::size_t p = 0; p < table->classes.size(); ++p)
    {
        for (std::size_t q = 0; q <= p; ++q)
        {
            const ClassPair& classPair = findClassPair(p, q);
            EXPECT_EQ(classPair.kMult, expectedKMult[p][q]) << "p = " << p << ", q = " << q;
        }
    }

    // The canonical orientation: the bra class carries the larger
    // representative pair index (the plain path's bra >= ket).
    for (const ClassPair& classPair : table->classPairs)
    {
        EXPECT_GE(table->classes[classPair.p].repPair, table->classes[classPair.q].repPair);
    }
}

TEST(PairClassesTest, ToyReductionOrbitPartition) {
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction = MakeToyReduction();
    // The no-screening table: all-ones Schwarz bounds keep every class
    // pair (the class-bound test product >= 0.0).
    const auto table = BuildPairClasses(reduction, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0);

    ASSERT_TRUE(table.has_value());
    const std::size_t nPairs = pairList.pairs.size();

    // Every pair lands in exactly one class and the class representatives
    // are the orbit minima.
    std::size_t memberTotal = 0;

    for (const PairClass& cls : table->classes)
    {
        memberTotal += cls.members.size();
        EXPECT_EQ(cls.members[0], cls.repPair);

        for (std::size_t m = 1; m < cls.members.size(); ++m)
        {
            EXPECT_GT(cls.members[m], cls.repPair);
        }
    }

    EXPECT_EQ(memberTotal, nPairs);

    for (std::size_t p = 0; p < nPairs; ++p)
    {
        EXPECT_LT(table->classOfPair[p], table->classes.size());
        const PairClass& cls = table->classes[table->classOfPair[p]];
        EXPECT_NE(std::find(cls.members.begin(), cls.members.end(), p), cls.members.end());
    }
}

TEST(PairClassesTest, TrivialReductionIsThePlainEnumeration) {
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction; // groupOrder 1, isTrivial - the C1 case.
    // The no-screening table: all-ones Schwarz bounds keep every class
    // pair (the class-bound test product >= 0.0).
    const auto table = BuildPairClasses(reduction, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0);

    ASSERT_TRUE(table.has_value());
    // One class per pair, in pair-index order.
    ASSERT_EQ(table->classes.size(), pairList.pairs.size());

    for (std::size_t p = 0; p < pairList.pairs.size(); ++p)
    {
        EXPECT_EQ(table->classes[p].repPair, p);
        EXPECT_EQ(table->classes[p].members, (TaggedSizeVector{p}));
        EXPECT_EQ(table->classOfPair[p], p);
    }

    // Every class pair, canonical orientation (the full plain enumeration).
    const std::size_t nClasses = table->classes.size();
    ASSERT_EQ(table->classPairs.size(), nClasses * (nClasses + 1) / 2);
    std::size_t index = 0;

    for (std::size_t p = 0; p < nClasses; ++p)
    {
        for (std::size_t q = 0; q <= p; ++q)
        {
            const ClassPair& classPair = table->classPairs[index++];
            EXPECT_EQ(classPair.p, p);
            EXPECT_EQ(classPair.q, q);
            EXPECT_EQ(classPair.repBraPair, p);
            EXPECT_EQ(classPair.repKetPair, q);
            EXPECT_EQ(classPair.diagonalPair, p == q);
        }
    }
}

TEST(PairClassesTest, MalformedReductionRejected) {
    const ShellPairList pairList = MakeToyPairList();

    SymmetryReduction outOfRange = MakeToyReduction();
    outOfRange.permutation[1][1] = 3; // Function index 3 does not exist.
    EXPECT_EQ(
        BuildPairClasses(outOfRange, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0).error().code,
        qcx::ErrorCode::kInvalidArgument);

    SymmetryReduction badSign = MakeToyReduction();
    badSign.sign[1][0] = 2;
    EXPECT_EQ(BuildPairClasses(badSign, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0).error().code,
              qcx::ErrorCode::kInvalidArgument);

    SymmetryReduction wrongSize = MakeToyReduction();
    wrongSize.permutation[1].pop_back();
    EXPECT_EQ(
        BuildPairClasses(wrongSize, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0).error().code,
        qcx::ErrorCode::kInvalidArgument);

    SymmetryReduction wrongOrder = MakeToyReduction();
    wrongOrder.groupOrder = 3;
    EXPECT_EQ(
        BuildPairClasses(wrongOrder, pairList, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}, 0.0).error().code,
        qcx::ErrorCode::kInvalidArgument);

    // A Schwarz array that does not span the pair list is rejected too.
    EXPECT_EQ(BuildPairClasses(MakeToyReduction(), pairList, {1.0, 1.0, 1.0}, 0.0).error().code,
              qcx::ErrorCode::kInvalidArgument);
}

// The water C2v pair list in the O-first shell order [O1s, O2s, Op, Hs1,
// Hs2]: the H shells sit at indices 3, 4 with function offsets 5, 6 - the
// unmasked case for the shell-of-image-function lookup (the committed code
// applied the function permutation to the SHELL indices, correct only when
// shell index == function offset; qcx's own water fixture is H-first with
// offsets [0, 1, 2, 3, 4] and accidentally masks the bug - this hand-built
// list is the pin). The function action matches the real water basis
// ([O1s, O2s, Opy, Opz, Opx, Hs1, Hs2] with sigma_x swapping the two H
// functions and flipping p_x, sigma_z flipping p_z, C2 both), so the class
// and orbit counts are the harness-verified 11/66/76/10.
ShellPairList MakeWaterPairListOFunctionsFirst() {
    ShellPairList pairList;
    pairList.functionCount = 7;
    pairList.shells.push_back(ShellInfo{0, true, 1, 0, 0, 0}); // O1s.
    pairList.shells.push_back(ShellInfo{0, true, 1, 1, 1, 0}); // O2s.
    pairList.shells.push_back(ShellInfo{1, true, 1, 2, 2, 0}); // Op (3 functions).
    pairList.shells.push_back(ShellInfo{0, true, 1, 5, 3, 0}); // Hs1.
    pairList.shells.push_back(ShellInfo{0, true, 1, 6, 4, 0}); // Hs2.

    for (std::size_t i = 0; i < pairList.shells.size(); ++i)
    {
        for (std::size_t j = i; j < pairList.shells.size(); ++j)
        {
            pairList.pairs.push_back(ShellPairIndex{i, j});
        }
    }

    return pairList;
}

TEST(PairClassesTest, WaterC2vOrbitStructureOFunctionsFirst) {
    // The corrected shell-of-image-function walk over the unmasked O-first
    // shell layout: the class and orbit structure of the real water C2v
    // basis (the harness-verified counts), plus the (H1s, H1s) class - the
    // pair whose image under sigma_x/C2 the buggy shell-as-function walk
    // gets wrong (it would report a singleton class).
    const ShellPairList pairList = MakeWaterPairListOFunctionsFirst();
    const SymmetryReduction reduction = MakeWaterC2vReduction();
    // The no-screening table (15 bounds - the 15 canonical pairs of the
    // 5-shell list - keep every class pair).
    auto table = BuildPairClasses(
        reduction,
        pairList,
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
        0.0);

    ASSERT_TRUE(table.has_value());
    ASSERT_EQ(table->classes.size(), 11u);
    ASSERT_EQ(table->classPairs.size(), 66u);

    // The root restructure's laziness: Create stores the class pairs with
    // EMPTY orbit vectors (the on-demand generation fills them per reached
    // pair).
    for (const ClassPair& classPair : table->classPairs)
    {
        EXPECT_TRUE(classPair.orbits.empty())
            << "class pair (" << classPair.p << ", " << classPair.q << ") materialized at Create";
    }

    // The on-demand fill: generate every kept class pair (all 66 - the
    // no-screening table), the deterministic pure function of the class
    // members - the pinned structure below is the eager table's.
    for (std::size_t cp = 0; cp < table->classPairs.size(); ++cp)
    {
        GenerateClassPairOrbits(*table, reduction, pairList, cp);
    }

    // The (H1s, H1s) class: shell pair (3, 3) = pair index 12 and its image
    // (H2s, H2s) = shell pair (4, 4) = pair index 14 under sigma_x/C2 (the
    // H swap). The buggy walk's image of shell 3 under sigma_x is shell 3
    // (function 3 is Opz - the swap only moves functions 5 and 6), which
    // would split the class in two.
    const PairClass& hydrogenClass = table->classes[table->classOfPair[12]];
    EXPECT_EQ(table->classOfPair[14], table->classOfPair[12]);
    EXPECT_EQ(hydrogenClass.repPair, 12u);
    EXPECT_EQ(hydrogenClass.members, (TaggedSizeVector{12, 14}));

    // The orbit structure: every canonical member quartet of every class
    // pair lands in exactly one orbit; 76 orbits over the 66 class pairs,
    // 10 of the class pairs carrying more than one orbit (the diagonal
    // (P, P) quartets and the off-diagonal ones split; the
    // harness's n_multi).
    std::size_t orbitTotal = 0;
    std::size_t multiOrbitClassPairs = 0;
    std::unordered_set<std::size_t> repKeys;
    const std::size_t nPairs = pairList.pairs.size();

    for (const ClassPair& classPair : table->classPairs)
    {
        orbitTotal += classPair.orbits.size();

        if (classPair.orbits.size() > 1)
        {
            ++multiOrbitClassPairs;
        }

        for (const auto& orbit : classPair.orbits)
        {
            // The orbit representative is the first member with the identity
            // generator, and the rep quartet keys are globally unique (the
            // contraction's order-agnostic lookup relies on it).
            ASSERT_FALSE(orbit.members.empty());
            EXPECT_EQ(orbit.members[0].braPair, orbit.repBraPair);
            EXPECT_EQ(orbit.members[0].ketPair, orbit.repKetPair);
            EXPECT_EQ(orbit.members[0].generator, 0u);
            EXPECT_TRUE(repKeys.insert(orbit.repBraPair * nPairs + orbit.repKetPair).second)
                << "duplicate orbit representative " << orbit.repBraPair << ", "
                << orbit.repKetPair;

            for (const auto& member : orbit.members)
            {
                // Canonical member quartets, closed under the action: every
                // member's pair classes are the orbit's class pair (either
                // orientation - the canonical bra >= ket reorder can swap
                // them when the members' indices interleave).
                EXPECT_GE(member.braPair, member.ketPair);
                const bool classesMatch = (table->classOfPair[member.braPair] == classPair.p &&
                                           table->classOfPair[member.ketPair] == classPair.q) ||
                                          (table->classOfPair[member.braPair] == classPair.q &&
                                           table->classOfPair[member.ketPair] == classPair.p);
                EXPECT_TRUE(classesMatch)
                    << "member (" << member.braPair << ", " << member.ketPair
                    << ") escapes its class pair (" << classPair.p << ", " << classPair.q << ")";

                // slotPerm is a permutation of the four slots.
                std::array<std::size_t, 4> slots = member.slotPerm;
                std::sort(slots.begin(), slots.end());
                EXPECT_EQ(slots, (std::array<std::size_t, 4>{0, 1, 2, 3}));
            }
        }
    }

    EXPECT_EQ(orbitTotal, 76u);
    EXPECT_EQ(multiOrbitClassPairs, 10u);
}

TEST(PairClassesTest, ClassBoundScreeningDropsUnaffordablePairs) {
    // The Schwarz class-bound screening: a class pair is kept iff the
    // product of its two classes' worst member bounds clears the threshold.
    // The toy group's classes over the 6-pair list are {0} (pair 0), {1, 2}
    // (pairs 1, 2), {3, 5} (pairs 3, 5), {4} (pair 4); the bounds below make
    // class 3's worst product 0.01 - under the 0.02 cutoff in every
    // combination - so the four class pairs involving class 3 drop and the
    // other six stay.
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction = MakeToyReduction();
    const std::vector<double> schwarz = {1.0, 0.5, 0.01, 0.5, 0.01, 0.01};
    auto table = BuildPairClasses(reduction, pairList, schwarz, 0.02);

    ASSERT_TRUE(table.has_value());

    // The screening touches only the class-pair list: the classes and the
    // per-pair class map are the unscreened decomposition's.
    ASSERT_EQ(table->classes.size(), 4u);
    EXPECT_EQ(table->classOfPair, (TaggedSizeVector{0, 1, 1, 2, 3, 2}));

    // The kept set: the 6 class pairs with classWorst product >= 0.02 -
    // all of the 10 except the four (3, *) pairs (class 3's only member,
    // pair 4 = (1, 2), carries bound 0.01).
    ASSERT_EQ(table->classPairs.size(), 6u);

    for (const ClassPair& classPair : table->classPairs)
    {
        EXPECT_NE(classPair.p, 3u);
        EXPECT_NE(classPair.q, 3u);
        EXPECT_TRUE(classPair.orbits.empty()) << "screened-in class pair (" << classPair.p << ", "
                                              << classPair.q << ") materialized at Create";
    }

    // Soundness: every canonical member quartet whose product of member
    // bounds clears the neighbor-list cutoff lives in a KEPT class pair -
    // the screening drops exactly the pairs that cannot contain a screened
    // quartet (the plain path's per-quartet test would keep exactly these).
    const auto keptPair = [&table](std::size_t p, std::size_t q) {
        const auto it = std::find_if(
            table->classPairs.begin(), table->classPairs.end(), [p, q](const ClassPair& classPair) {
                return classPair.p == p && classPair.q == q;
            });
        return it != table->classPairs.end();
    };

    for (std::size_t m = 0; m < pairList.pairs.size(); ++m)
    {
        for (std::size_t n = 0; n <= m; ++n)
        {
            if (schwarz[m] * schwarz[n] < 0.02)
            {
                continue;
            }

            const std::size_t braClass = table->classOfPair[m];
            const std::size_t ketClass = table->classOfPair[n];
            const std::size_t p = std::max(braClass, ketClass);
            const std::size_t q = std::min(braClass, ketClass);
            EXPECT_TRUE(keptPair(p, q)) << "screened quartet (" << m << ", " << n
                                        << ") in dropped class pair (" << p << ", " << q << ")";
        }
    }

    // The on-demand fill works on the kept pairs and is idempotent.
    GenerateClassPairOrbits(*table, reduction, pairList, 0);
    ASSERT_FALSE(table->classPairs[0].orbits.empty());
    const std::size_t orbitCount = table->classPairs[0].orbits.size();
    GenerateClassPairOrbits(*table, reduction, pairList, 0);
    EXPECT_EQ(table->classPairs[0].orbits.size(), orbitCount);
}

// The class-table admission gate's materialization counter
// (CountClassMaterialization): the gate charges
// these counts instead of the eager table's singleton-orbit ceiling, so the
// counts must bound the table BuildPairClasses actually builds - at every
// screening threshold, and by construction rather than by luck.
TEST(PairClassesTest, MaterializationCountsBoundTheBuiltTable) {
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction = MakeToyReduction();
    // Distinct per-pair bounds, so the class-bound products separate the
    // class pairs at the interior thresholds.
    const std::vector<double> schwarz{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // 0.0 is the no-screening table (every class-bound product clears it);
    // 37.0 exceeds every product (the largest class bound is 6.0), so
    // nothing survives.
    for (const double threshold : {0.0, 4.0, 9.0, 16.0, 26.0, 37.0})
    {
        const auto counts = CountClassMaterialization(reduction, pairList, schwarz, threshold);
        ASSERT_TRUE(counts.has_value()) << counts.error().message;
        const auto table = BuildPairClasses(reduction, pairList, schwarz, threshold);
        ASSERT_TRUE(table.has_value()) << table.error().message;

        std::size_t quartets = 0;

        for (const ClassPair& classPair : table->classPairs)
        {
            const std::size_t membersP = table->classes[classPair.p].members.size();
            const std::size_t membersQ = table->classes[classPair.q].members.size();
            quartets +=
                (classPair.p == classPair.q) ? membersP * (membersP + 1) / 2 : membersP * membersQ;
        }

        EXPECT_EQ(counts->classes, table->classes.size());
        EXPECT_GE(counts->classPairs, table->classPairs.size()) << "threshold " << threshold;
        EXPECT_GE(counts->memberQuartets, quartets) << "threshold " << threshold;
    }
}

// The counter's exact values on the no-screening toy: 4 classes, so 16
// ORDERED class pairs (the never-under form of the table's 10 unordered
// ones) and a member-quartet mass of 6 x 6 = 36 (the never-under form of
// the table's 21 canonical quartets).
TEST(PairClassesTest, MaterializationCountsAreExactWithoutScreening) {
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction = MakeToyReduction();
    const std::vector<double> schwarz{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const auto counts = CountClassMaterialization(reduction, pairList, schwarz, 0.0);
    ASSERT_TRUE(counts.has_value()) << counts.error().message;
    EXPECT_EQ(counts->classes, 4u);
    EXPECT_EQ(counts->classPairs, 16u);
    EXPECT_EQ(counts->memberQuartets, 36u);
}

// The reduction's own sizing instrument (ClassMaterializationCounts): the
// pair axis is exact on both sides, so the two counts read as the petite
// list's saving - `classes` of `canonicalPairs` - instead of being inferred
// from a flag. The measurement's second job is the one the engagement fix
// created it for: `classes == canonicalPairs` is the verdict that the
// reduction removes NOTHING, which covers the pair-trivial group (the
// sign-only, function-fixing action the planar fixtures hit) that a
// `groupOrder > 1` test gets wrong, not only the C1 case. The toy list: 6
// canonical pairs, 4 classes under the swap of shells 1 and 2 - the orbits
// {(0,0)}, {(0,1),(0,2)}, {(1,1),(2,2)}, {(1,2)} - and 6 classes under the
// default C1 reduction, the plain enumeration.
TEST(PairClassesTest, MaterializationCountsCarryTheReductionsOwnDenominator) {
    const ShellPairList pairList = MakeToyPairList();
    const std::vector<double> schwarz(pairList.pairs.size(), 1.0);

    const auto counts = CountClassMaterialization(MakeToyReduction(), pairList, schwarz, 0.0);
    ASSERT_TRUE(counts.has_value()) << counts.error().message;
    EXPECT_EQ(counts->canonicalPairs, 6u);
    EXPECT_EQ(counts->classes, 4u);
    EXPECT_LT(counts->classes, counts->canonicalPairs)
        << "the swap group must remove pairs, and the count must say so";

    const auto plain = CountClassMaterialization(SymmetryReduction{}, pairList, schwarz, 0.0);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    EXPECT_EQ(plain->canonicalPairs, 6u);
    EXPECT_EQ(plain->classes, 6u);
}

// The same measurement on the real water basis (the O-first 5-shell list, 15
// canonical pairs) under three groups of the SAME molecule: the C2v
// extraction's action (11 classes - the two H shells swap, the p functions
// flip sign without moving), a SIGN-ONLY group of order 2 (15 classes - the
// mirror every atom lies on, whose every function image is the function
// itself), and the C1 default (15 classes). The sign-only leg is the one the
// defect class lives in: the group is REAL and of order 2, so an engagement
// test keyed on the order calls it reduced while the count says 15 of 15 -
// and `isTrivial` is the caller's assertion while these counts are the
// measurement, so where the two can be compared they must agree. Scope: this
// pins the agreement on the three groups the suite governs; a foreign
// hand-built reduction that sets the flag against its own action is outside
// what any count can police.
TEST(PairClassesTest, ThePairCountIsTheMeasurementBehindTheTrivialFlag) {
    const ShellPairList pairList = MakeWaterPairListOFunctionsFirst();
    const std::vector<double> schwarz(pairList.pairs.size(), 1.0);

    const SymmetryReduction c2v = MakeWaterC2vReduction();
    const auto c2vCounts = CountClassMaterialization(c2v, pairList, schwarz, 0.0);
    ASSERT_TRUE(c2vCounts.has_value()) << c2vCounts.error().message;
    EXPECT_EQ(c2vCounts->canonicalPairs, 15u);
    EXPECT_EQ(c2vCounts->classes, 11u);
    EXPECT_FALSE(c2v.isTrivial);
    EXPECT_LT(c2vCounts->classes, c2vCounts->canonicalPairs);

    // The sign-only group: the identity permutation on every element, one of
    // them flipping Opz - a real order-2 group that fixes every canonical
    // pair, so every orbit is a singleton and nothing is removed.
    SymmetryReduction signOnly;
    signOnly.groupOrder = 2;
    signOnly.isTrivial = true;
    signOnly.permutation = {{0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 6}};
    signOnly.sign = {{1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, -1, 1, 1, 1}};

    const auto signOnlyCounts = CountClassMaterialization(signOnly, pairList, schwarz, 0.0);
    ASSERT_TRUE(signOnlyCounts.has_value()) << signOnlyCounts.error().message;
    EXPECT_EQ(signOnlyCounts->canonicalPairs, 15u);
    EXPECT_EQ(signOnlyCounts->classes, 15u) << "a sign-only group removes no pair";
    ASSERT_GT(signOnly.groupOrder, 1u) << "this fixture must not be the C1 fallback";
    EXPECT_TRUE(signOnly.isTrivial);

    const SymmetryReduction c1;
    const auto c1Counts = CountClassMaterialization(c1, pairList, schwarz, 0.0);
    ASSERT_TRUE(c1Counts.has_value()) << c1Counts.error().message;
    EXPECT_EQ(c1Counts->canonicalPairs, 15u);
    EXPECT_EQ(c1Counts->classes, 15u);
}

// The counter rejects exactly what the table build rejects: the gate must
// never engage on a shape BuildPairClasses would refuse.
TEST(PairClassesTest, MaterializationCountsRejectTheMalformedInputs) {
    const ShellPairList pairList = MakeToyPairList();
    const SymmetryReduction reduction = MakeToyReduction();
    // The Schwarz array must span the pair list.
    const auto shortSchwarz =
        CountClassMaterialization(reduction, pairList, std::vector<double>{1.0, 1.0}, 0.0);
    ASSERT_FALSE(shortSchwarz.has_value());
    EXPECT_EQ(shortSchwarz.error().code, qcx::ErrorCode::kInvalidArgument);
    // A reduction row that is not a signed permutation over the basis.
    SymmetryReduction malformed = reduction;
    malformed.permutation[1][2] = 7;
    const auto badReduction =
        CountClassMaterialization(malformed, pairList, std::vector<double>(6, 1.0), 0.0);
    ASSERT_FALSE(badReduction.has_value());
    EXPECT_EQ(badReduction.error().code, qcx::ErrorCode::kInvalidArgument);
    // A zero group order is the empty-table shape BuildPairClasses refuses.
    SymmetryReduction empty = reduction;
    empty.groupOrder = 0;
    const auto badOrder =
        CountClassMaterialization(empty, pairList, std::vector<double>(6, 1.0), 0.0);
    ASSERT_FALSE(badOrder.has_value());
    EXPECT_EQ(badOrder.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The class-path gate's SCOPE-HYGIENE invariant: the tag its own transients
// land in. The gate counts the class decomposition with three containers of its
// own - the per-function shell map, the classes, the per-class member lists -
// and frees them on return; the model charges the class_table family for
// the TABLE the trace reports (BuildPairClasses / GenerateClassPairOrbits),
// so the counting transient stays UNTRACKED. A tagged container built with
// no scope open captures kUnclassified - the bucket that counts wrapper
// traffic with no scope, NOT a coverage check (the instrument header is the
// contract's home) - which is how this guard fires: on the
// pre-fix binary this water fixture charged 23 kUnclassified allocations and
// a 1,048 B high-water, the same two figures the driver's instrumented H2O
// run recorded for the direct family (the driver's
// InstrumentedH2RunStaysOnThePinAndKeepsUnclassifiedEmpty pin was red on
// exactly them; its H2 case reads 6 / 160 B). The measure is the tag DELTA
// across one gate call, so the guard covers any tagged site the gate grows,
// not the three it had.
TEST(PairClassesTest, ClassPathGateLeavesTheUnclassifiedBucketEmpty) {
    const ScopedInstrumentEnable instrument;
    ASSERT_TRUE(instrument.Ok()) << "the allocation attribution instrument did not enable";

    const ShellPairList pairList = MakeWaterPairListOFunctionsFirst();
    const SymmetryReduction reduction = MakeWaterC2vReduction();
    const std::vector<double> schwarz(pairList.pairs.size(), 1.0);

    const std::uint64_t before =
        qcx::memory::TagAllocationCount(qcx::memory::AllocationTag::kUnclassified);
    const std::uint64_t peakBefore =
        qcx::memory::TagPeakBytes(qcx::memory::AllocationTag::kUnclassified);
    const auto counts = CountClassMaterialization(reduction, pairList, schwarz, 0.0);
    const std::uint64_t after =
        qcx::memory::TagAllocationCount(qcx::memory::AllocationTag::kUnclassified);

    ASSERT_TRUE(counts.has_value()) << counts.error().message;
    EXPECT_EQ(after, before) << "the gate's transient charged " << (after - before)
                             << " allocations to the unclassified leak bucket";
    // The peak is the binding reading of a transient: it is freed before the
    // call returns, so only the high-water shows it was ever there.
    EXPECT_EQ(qcx::memory::TagPeakBytes(qcx::memory::AllocationTag::kUnclassified), peakBefore)
        << "the gate's transient raised the unclassified high-water";
    EXPECT_EQ(qcx::memory::TagCurrentBytes(qcx::memory::AllocationTag::kUnclassified), 0u);
}

// The table build's OWN allocation is the opposite case: the class_table
// family must carry it, so the same fixture's BuildPairClasses has to charge
// kClassTable (the guard above would pass vacuously if the table's storage
// ever went untagged).
TEST(PairClassesTest, TheTableBuildChargesTheClassTableFamily) {
    const ScopedInstrumentEnable instrument;
    ASSERT_TRUE(instrument.Ok()) << "the allocation attribution instrument did not enable";

    const ShellPairList pairList = MakeWaterPairListOFunctionsFirst();
    const SymmetryReduction reduction = MakeWaterC2vReduction();
    const std::vector<double> schwarz(pairList.pairs.size(), 1.0);

    const std::uint64_t before =
        qcx::memory::TagAllocationCount(qcx::memory::AllocationTag::kClassTable);
    auto table = BuildPairClasses(reduction, pairList, schwarz, 0.0);
    const std::uint64_t after =
        qcx::memory::TagAllocationCount(qcx::memory::AllocationTag::kClassTable);

    ASSERT_TRUE(table.has_value()) << table.error().message;
    EXPECT_GT(after - before, 0u) << "the table build charged the class_table family nothing";
    GenerateClassPairOrbits(*table, reduction, pairList, 0);
    EXPECT_GT(qcx::memory::TagAllocationCount(qcx::memory::AllocationTag::kClassTable), before)
        << "the on-demand orbit fill charged the class_table family nothing";
}

} // namespace
