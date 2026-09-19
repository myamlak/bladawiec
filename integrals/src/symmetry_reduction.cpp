// The pair-class decomposition (symmetry_reduction.hpp): the
// orbits of the canonical pair list under the group action, the unordered
// class pairs with their exchange multiplicities, and the per-class-pair
// orbits of canonical member quartets under the diagonal action - the data
// the class-aware Fock contraction enumerates (one ERI block per quartet
// orbit, expanded per member).

#include "qcx/integrals/symmetry_reduction.hpp"

#include "internal/md_attribution.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace qcx::integrals {
namespace {

// The canonicalized image of canonical shell pair p under element g: the
// pair of the two IMAGE SHELLS - the shells containing the image functions
// permutation[g][functionOffset[s]] of the pair's shells - ordered by shell
// index. The reduction acts on the FUNCTIONS while the pair list indexes
// SHELLS; the committed code applied the function permutation to the shell
// indices, which is only correct when shell index == function offset (the
// O-first water ordering [O1s, O2s, Op, Hs1, Hs2] - offsets [0,1,2,5,6],
// the H shells at indices 3,4 - is the unmasked case; qcx's own water
// fixture is H-first and accidentally masks the bug, so the unmasked case
// is pinned by the hand-built pair list in the integrals test).
// The block value of a canonical quartet is invariant under the
// exchange of the two functions of a pair, so the pair canonicalization
// carries no extra position permutation here.
std::size_t ImagePairIndex(const SymmetryReduction& reduction,
                           const ShellPairList& pairList,
                           std::span<const std::size_t> shellOfFunction,
                           // (g, p) are the generator and the pair index - distinct
                           // quantities; every call site passes them in fixed order.
                           //
                           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                           std::size_t g,
                           std::size_t p) {
    const ShellPairIndex& pair = pairList.pairs[p];
    const std::size_t imageI =
        shellOfFunction[reduction.permutation[g][pairList.shells[pair.i].functionOffset]];
    const std::size_t imageJ =
        shellOfFunction[reduction.permutation[g][pairList.shells[pair.j].functionOffset]];
    return PairIndexOf(std::min(imageI, imageJ), std::max(imageI, imageJ), pairList);
}

// The canonicalized image of canonical quartet (bra, ket) under the
// diagonal action of element g: the canonical quartet (bra >= ket) of the
// two image pairs.
std::pair<std::size_t, std::size_t> ImageQuartetIndex(const SymmetryReduction& reduction,
                                                      const ShellPairList& pairList,
                                                      std::span<const std::size_t> shellOfFunction,
                                                      std::size_t g,
                                                      std::size_t bra,
                                                      std::size_t ket) {
    const std::size_t imageBra = ImagePairIndex(reduction, pairList, shellOfFunction, g, bra);
    const std::size_t imageKet = ImagePairIndex(reduction, pairList, shellOfFunction, g, ket);
    return imageBra >= imageKet ? std::pair{imageBra, imageKet} : std::pair{imageKet, imageBra};
}

// The slot mapping of element g on the rep quartet: the member-block slot
// receiving each source slot. The source slots are the rep quartet's
// positions in shell order (0, 1 = the bra pair's two shells, 2, 3 = the
// ket pair's two shells); the member slots index the member quartet's
// canonical shell order (the canonicalized image pairs' shells). The map
// encodes the two within-pair flips (an image pair's shells out of
// canonical order) and the whole-quartet role swap (the image bra pair's
// index below the image ket pair's - the canonical quartet swaps bra/ket).
// The caller has verified that ImageQuartetIndex(g, repBra, repKet) ==
// (memberBra, memberKet).
std::array<std::size_t, 4> ComputeSlotPerm(const SymmetryReduction& reduction,
                                           const ShellPairList& pairList,
                                           std::span<const std::size_t> shellOfFunction,
                                           // (g, repBra) are the generator and the bra
                                           // index - distinct quantities; every call site
                                           // passes them in fixed order.
                                           //
                                           // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                           std::size_t g,
                                           std::size_t repBra,
                                           std::size_t repKet) {
    const ShellPairIndex& braPair = pairList.pairs[repBra];
    const ShellPairIndex& ketPair = pairList.pairs[repKet];
    const std::size_t imageBraShell[2] = {
        shellOfFunction[reduction.permutation[g][pairList.shells[braPair.i].functionOffset]],
        shellOfFunction[reduction.permutation[g][pairList.shells[braPair.j].functionOffset]]};
    const std::size_t imageKetShell[2] = {
        shellOfFunction[reduction.permutation[g][pairList.shells[ketPair.i].functionOffset]],
        shellOfFunction[reduction.permutation[g][pairList.shells[ketPair.j].functionOffset]]};
    const std::size_t imageBraPair = PairIndexOf(std::min(imageBraShell[0], imageBraShell[1]),
                                                 std::max(imageBraShell[0], imageBraShell[1]),
                                                 pairList);
    const std::size_t imageKetPair = PairIndexOf(std::min(imageKetShell[0], imageKetShell[1]),
                                                 std::max(imageKetShell[0], imageKetShell[1]),
                                                 pairList);

    std::array<std::size_t, 4> slotPerm{};
    const bool flipBra = imageBraShell[0] > imageBraShell[1];
    const bool flipKet = imageKetShell[0] > imageKetShell[1];
    slotPerm[0] = flipBra ? 1 : 0;
    slotPerm[1] = flipBra ? 0 : 1;
    slotPerm[2] = flipKet ? 3 : 2;
    slotPerm[3] = flipKet ? 2 : 3;

    if (imageBraPair < imageKetPair)
    {
        // The canonical quartet is (imageKetPair, imageBraPair): the roles
        // swap, so the bra source slots land in the member ket slots and
        // vice versa.
        slotPerm[0] += 2;
        slotPerm[1] += 2;
        slotPerm[2] -= 2;
        slotPerm[3] -= 2;
    }

    return slotPerm;
}

// shellOfFunction[f] = the shell containing function f: the shells are
// ordered by ascending functionOffset and cover contiguous function ranges,
// so one pass fills the table (the function-level permutation maps to the
// shell level through it). Shared by BuildPairClasses (stored
// in the table) and the class-path gate's counting transient
// (CountClassMaterialization). The container type is the CALLER's, so each
// call site states the storage it needs: the family a container is
// charged to follows from where it lands, never from this fill - the
// table's vector is tagged (BuildPairClasses' kClassTable scope), the
// gate's is plain (its transient is charged by no family).
template <typename Container> Container BuildShellOfFunction(const ShellPairList& pairList) {
    Container shellOfFunction(pairList.functionCount);

    for (std::size_t s = 0; s < pairList.shells.size(); ++s)
    {
        const ShellInfo& shell = pairList.shells[s];

        for (std::size_t f = shell.functionOffset;
             f < shell.functionOffset + ShellFunctionCount(shell);
             ++f)
        {
            shellOfFunction[f] = s;
        }
    }

    return shellOfFunction;
}

// The orbits of the canonical member quartets of two classes under the
// diagonal action: each orbit's representative quartet (the lexicographic
// minimum, the one ERI block the contraction computes) and every member's
// expansion data (the diagonal action is generally NOT
// transitive - the diagonal P == Q quartets and the off-diagonal ones split
// into separate orbits). The members include the representative itself
// (the identity generator). Every member's block equals the rep block up to
// the pattern product and slot permutation of ANY element mapping the rep
// to it (all maps of a member produce the identical block, so
// the first element in group order is picked). The returned orbit entries
// land in the class pair's orbits vector, so the class_table family
// types them (the tag is the scope active at this call -
// GenerateClassPairOrbits' kClassTable); the member-list parameters are
// spans - the caller's table members bind regardless of allocator.
internal::TaggedVector<ClassOrbit> BuildOrbits(
    const SymmetryReduction& reduction,
    const ShellPairList& pairList,
    // (shellOfFunction, membersP) are the
    // function-map and the member lists -
    // distinct quantities, fixed call order.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    std::span<const std::size_t> shellOfFunction,
    std::span<const std::size_t> membersP,
    std::span<const std::size_t> membersQ) {
    // The class pair's canonical member quartets, deduplicated and sorted.
    std::vector<std::pair<std::size_t, std::size_t>> quartets;

    for (const std::size_t m : membersP)
    {
        for (const std::size_t n : membersQ)
        {
            quartets.emplace_back(std::max(m, n), std::min(m, n));
        }
    }

    std::sort(quartets.begin(), quartets.end());
    quartets.erase(std::unique(quartets.begin(), quartets.end()), quartets.end());

    // The unassigned quartets, breadth-first over the diagonal action. The
    // orbit of a quartet is its connected component in the member-set
    // graph (the images of a member quartet stay within the class pair's
    // member set: classes are closed under the action).
    std::vector<bool> assigned(quartets.size(), false);
    internal::TaggedVector<ClassOrbit> orbits;

    for (std::size_t start = 0; start < quartets.size(); ++start)
    {
        if (assigned[start])
        {
            continue;
        }

        std::vector<std::size_t> memberIndices{start};
        assigned[start] = true;
        std::size_t explored = 0;

        while (explored < memberIndices.size())
        {
            const std::size_t index = memberIndices[explored++];
            const auto [bra, ket] = quartets[index];

            for (std::size_t g = 0; g < reduction.groupOrder; ++g)
            {
                const auto image =
                    ImageQuartetIndex(reduction, pairList, shellOfFunction, g, bra, ket);
                const auto found = std::find(quartets.begin(), quartets.end(), image);

                // The image is always a member quartet (class closure);
                // skip the already-assigned ones.
                if (found == quartets.end())
                {
                    continue;
                }

                const std::size_t imageIndex =
                    static_cast<std::size_t>(std::distance(quartets.begin(), found));

                if (!assigned[imageIndex])
                {
                    assigned[imageIndex] = true;
                    memberIndices.push_back(imageIndex);
                }
            }
        }

        // The representative: the lexicographic minimum of the orbit (the
        // BFS started from it - the quartets are scanned in sorted order).
        const auto [repBra, repKet] = quartets[memberIndices[0]];

        ClassOrbit orbit;
        orbit.repBraPair = repBra;
        orbit.repKetPair = repKet;

        for (const std::size_t memberIndex : memberIndices)
        {
            const auto [memberBra, memberKet] = quartets[memberIndex];

            for (std::size_t g = 0; g < reduction.groupOrder; ++g)
            {
                if (ImageQuartetIndex(reduction, pairList, shellOfFunction, g, repBra, repKet) !=
                    std::pair{memberBra, memberKet})
                {
                    continue;
                }

                ClassOrbitMember member;
                member.braPair = memberBra;
                member.ketPair = memberKet;
                member.generator = g;
                member.slotPerm =
                    ComputeSlotPerm(reduction, pairList, shellOfFunction, g, repBra, repKet);
                orbit.members.push_back(member);
                break;
            }
        }

        orbits.push_back(std::move(orbit));
    }

    return orbits;
}

// The reduction's validation and the trivial-case materialization: an
// order-1 group may carry empty permutation/sign tables (the
// default-constructed SymmetryReduction and the shape the C1 extraction
// returns), and an order-1 group has only the identity element - so
// materialize it and let the walks below see a uniform signed permutation.
// On success \p work is the validated reduction the caller walks. Shared by
// BuildPairClasses and CountClassMaterialization: the admission gate must
// reject exactly what the table build rejects.
qcx::Result<void> PrepareReduction(SymmetryReduction& work, const ShellPairList& pairList) {
    const std::size_t order = work.groupOrder;
    const std::size_t n = pairList.functionCount;

    if (order == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "reduction permutation/sign tables mismatch the group order"});
    }

    if (order == 1)
    {
        std::vector<std::size_t> identityPermutation(n);
        std::iota(identityPermutation.begin(), identityPermutation.end(), std::size_t{0});
        work.permutation = {std::move(identityPermutation)};
        work.sign = {std::vector<int>(n, 1)};
    }

    if (work.permutation.size() != order || work.sign.size() != order)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "reduction permutation/sign tables mismatch the group order"});
    }

    for (std::size_t g = 0; g < order; ++g)
    {
        if (work.permutation[g].size() != n || work.sign[g].size() != n)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "reduction rows do not span the basis function count"});
        }

        for (std::size_t i = 0; i < n; ++i)
        {
            if (work.permutation[g][i] >= n || (work.sign[g][i] != -1 && work.sign[g][i] != 1))
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "reduction is not a signed permutation over the basis"});
            }
        }
    }

    return {};
}

// The class record of one pair class, as the class-path GATE counts it
// (CountClassMaterialization): the shape of PairClass with UNTAGGED member
// storage. The gate's three containers - the per-function shell map, the
// classes, the per-class member lists - are counting scratch freed on
// return, and the class_table family charges the TABLE the trace reports
// (the BuildPairClasses build and the GenerateClassPairOrbits fills), so the
// gate's transients must carry no family at all. Tagged containers built
// with no scope open would capture kUnclassified - the bucket that counts
// wrapper traffic with no scope, and NOT a coverage check (the instrument
// header is the contract's home) - which is the defect this
// record's fix closes (the driver's instrumented-run pin and this module's
// ClassPathGateLeavesTheUnclassifiedBucketEmpty pin that scope hygiene).
struct UntaggedPairClass {
    std::size_t repPair = 0;
    std::vector<std::size_t> members;
};

// The pair classes of one canonical pair list under one reduction: the
// orbits, breadth-first over the group from every unvisited pair in
// ascending index order (the first unvisited pair is the orbit's minimum -
// its representative). \p classOfPair is the nPairs-sized pair -> class
// lookup, pre-filled with the nPairs "unvisited" sentinel (no class index
// can reach it: a list of n pairs has at most n classes); the members land
// in \p classes. Shared by BuildPairClasses (which stores both in the table,
// through PairClass and its tagged member vector) and
// CountClassMaterialization (which needs the member counts and their Schwarz
// maxima only, through UntaggedPairClass). The member storage follows the
// class record's own type: the tagged vector captures the scope active at
// THIS call, so the table build charges kClassTable and the gate charges
// nothing.
template <typename PairClassT, typename ClassesContainer>
void FillPairClasses(const SymmetryReduction& work,
                     const ShellPairList& pairList,
                     std::span<const std::size_t> shellOfFunction,
                     ClassesContainer& classes,
                     std::span<std::size_t> classOfPair) {
    // The class record's own member-container type (decltype of a
    // non-static data member is its declared type, exactly).
    using MembersVector = decltype(PairClassT::members);
    const std::size_t order = work.groupOrder;
    const std::size_t nPairs = pairList.pairs.size();

    for (std::size_t p = 0; p < nPairs; ++p)
    {
        if (classOfPair[p] != nPairs)
        {
            continue;
        }

        MembersVector members{p};
        std::size_t explored = 0;

        while (explored < members.size())
        {
            const std::size_t current = members[explored];
            ++explored;

            for (std::size_t g = 0; g < order; ++g)
            {
                const std::size_t image =
                    ImagePairIndex(work, pairList, shellOfFunction, g, current);

                if (std::find(members.begin(), members.end(), image) == members.end())
                {
                    members.push_back(image);
                }
            }
        }

        PairClassT cls;
        cls.repPair = p;
        cls.members = std::move(members);

        const std::size_t classIndex = classes.size();

        for (const std::size_t member : cls.members)
        {
            classOfPair[member] = classIndex;
        }

        classes.push_back(std::move(cls));
    }
}

} // namespace

qcx::Result<PairClassTable> BuildPairClasses(const SymmetryReduction& reduction,
                                             const ShellPairList& pairList,
                                             const std::vector<double>& schwarz,
                                             double schwarzThreshold) {
    if (schwarz.size() != pairList.pairs.size())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "Schwarz bounds do not span the canonical pair list"});
    }

    SymmetryReduction work = reduction;
    const auto prepared = PrepareReduction(work, pairList);

    if (!prepared.has_value())
    {
        return std::unexpected(prepared.error());
    }

    // The class_table family (the DirectFootprint classTableBytes
    // term): the table's containers capture the tag at their construction
    // below, so the Create-time build and the later on-demand orbit
    // generation (GenerateClassPairOrbits) both attribute to kClassTable -
    // the table exists only on the class path, exactly the legs the term
    // charges.
    qcx::memory::AllocationTagScope classTableScope(qcx::memory::AllocationTag::kClassTable);
    PairClassTable table;
    const std::size_t nPairs = pairList.pairs.size();
    table.shellOfFunction = BuildShellOfFunction<internal::TaggedVector<std::size_t>>(pairList);
    const internal::TaggedVector<std::size_t>& shellOfFunction = table.shellOfFunction;
    // nPairs doubles as the "unvisited" sentinel: no valid class index can
    // reach it (a canonical pair list of n pairs has at most n classes).
    table.classOfPair.assign(nPairs, nPairs);

    FillPairClasses<PairClass>(work, pairList, shellOfFunction, table.classes, table.classOfPair);

    // The unordered class pairs, canonical orientation repBra >= repKet:
    // the classes are in ascending rep order, so p >= q is the bra class
    // first - the class-level match of the plain path's canonical quartet
    // enumeration (bra pair index >= ket pair index).
    const std::size_t nClasses = table.classes.size();

    // The Schwarz class bound: the worst per-member Schwarz bound of each
    // class. The class-pair screening test (the product of the two classes'
    // worst bounds vs schwarzThreshold) is the sound superset of the plain
    // path's per-quartet neighbor-list test (schwarz[bra] * schwarz[ket] >=
    // the same threshold): every neighbor-list quartet's bound is <= the
    // product of its two classes' worst bounds, so every screened member
    // quartet lives in a kept class pair - the screening drops exactly the
    // class pairs that cannot contain a screened quartet, and the
    // contraction's (quartet, block) pairs are unchanged.
    std::vector<double> classWorstBound(nClasses, 0.0);

    for (std::size_t c = 0; c < nClasses; ++c)
    {
        for (const std::size_t member : table.classes[c].members)
        {
            classWorstBound[c] = std::max(classWorstBound[c], schwarz[member]);
        }
    }

    for (std::size_t p = 0; p < nClasses; ++p)
    {
        for (std::size_t q = 0; q <= p; ++q)
        {
            if (classWorstBound[p] * classWorstBound[q] < schwarzThreshold)
            {
                // No member quartet of this class pair can clear the
                // neighbor-list cutoff in any iteration: drop the pair
                // (it contributes no task to the contraction - the
                // screened quartets are all in kept pairs).
                continue;
            }

            ClassPair classPair;
            classPair.p = p;
            classPair.q = q;
            classPair.repBraPair = table.classes[p].repPair;
            classPair.repKetPair = table.classes[q].repPair;
            const ShellPairIndex& bra = pairList.pairs[classPair.repBraPair];
            const ShellPairIndex& ket = pairList.pairs[classPair.repKetPair];
            classPair.pDiagonal = bra.i == bra.j;
            classPair.qDiagonal = ket.i == ket.j;
            classPair.diagonalPair = p == q;
            // The exchange multiplicity: 2 per degenerate pair and 2 more
            // for the self-role-swapped quartet (the plain path's
            // kMultiplicity, evaluated at the class level - degeneracy is a
            // class property).
            classPair.kMult = (classPair.pDiagonal ? 2 : 1) * (classPair.qDiagonal ? 2 : 1) *
                              (classPair.diagonalPair ? 2 : 1);
            // The orbits are NOT built here (the root restructure): the
            // class pair is stored with an empty orbits vector and the
            // contraction generates the expansions on demand
            // (GenerateClassPairOrbits) the first time it reaches the pair.
            table.classPairs.push_back(classPair);
        }
    }

    return table;
}

qcx::Result<ClassMaterializationCounts> CountClassMaterialization(
    const SymmetryReduction& reduction,
    const ShellPairList& pairList,
    const std::vector<double>& schwarz,
    double schwarzThreshold) {
    if (schwarz.size() != pairList.pairs.size())
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "Schwarz bounds do not span the canonical pair list"});
    }

    SymmetryReduction work = reduction;
    const auto prepared = PrepareReduction(work, pairList);

    if (!prepared.has_value())
    {
        return std::unexpected(prepared.error());
    }

    // The class decomposition the table build will produce, at O(nPairs)
    // bytes - the same order as the pair list the caller already holds,
    // never the theta(N^4) class-pair or member space. The gate's own
    // transient is deliberately NOT attributed to the class_table
    // family: the family measures the TABLE the trace reports, and these
    // vectors are freed on return - so they are PLAIN containers, never
    // tagged ones. A tagged container here would capture the ambient tag,
    // and with no scope open that is kUnclassified - wrapper traffic with no
    // scope, not a coverage reading (UntaggedPairClass; the instrument header
    // is the contract's home).
    const std::vector<std::size_t> shellOfFunction =
        BuildShellOfFunction<std::vector<std::size_t>>(pairList);
    std::vector<UntaggedPairClass> classes;
    std::vector<std::size_t> classOfPair(pairList.pairs.size(), pairList.pairs.size());

    FillPairClasses<UntaggedPairClass>(work, pairList, shellOfFunction, classes, classOfPair);

    const std::size_t nClasses = classes.size();

    // The Schwarz class bound of each class (the same worst-member maximum
    // BuildPairClasses screens with) and its member count.
    std::vector<double> classWorstBound(nClasses, 0.0);
    std::vector<std::size_t> classMembers(nClasses, 0);

    for (std::size_t c = 0; c < nClasses; ++c)
    {
        classMembers[c] = classes[c].members.size();

        for (const std::size_t member : classes[c].members)
        {
            classWorstBound[c] = std::max(classWorstBound[c], schwarz[member]);
        }
    }

    // The classes by descending Schwarz bound: the kept-pair test
    // (bound[p] * bound[q] >= schwarzThreshold) puts the qualifying partners
    // of a class in a PREFIX of this order, so one non-increasing
    // two-pointer sweep counts every qualifying ordered pair (and its
    // member-quartet mass) in O(nClasses log nClasses) plus O(nClasses) -
    // the class-pair vector and the orbit expansions are never touched.
    std::vector<std::size_t> classOrder(nClasses);
    std::iota(classOrder.begin(), classOrder.end(), std::size_t{0});
    std::sort(
        classOrder.begin(), classOrder.end(), [&classWorstBound](std::size_t lhs, std::size_t rhs) {
            if (classWorstBound[lhs] != classWorstBound[rhs])
            {
                return classWorstBound[lhs] > classWorstBound[rhs];
            }

            return lhs < rhs;
        });

    std::vector<std::size_t> prefixMass(nClasses + 1, 0);

    for (std::size_t i = 0; i < nClasses; ++i)
    {
        prefixMass[i + 1] = prefixMass[i] + classMembers[classOrder[i]];
    }

    ClassMaterializationCounts counts;
    // The reduction's own sizing (symmetry_reduction.hpp): the pair axis is
    // exact on both sides - the plain pair list against the classes the group
    // action leaves of it - so a consumer reads the saving off the record
    // instead of inferring it from the flag. `classes == canonicalPairs` is
    // the pair-trivial case (nothing removed), which is the sign-only group
    // the engagement fix stopped gating on the group ORDER.
    counts.canonicalPairs = pairList.pairs.size();
    counts.classes = nClasses;
    // The qualifying-prefix length: non-increasing as the sweep walks the
    // descending bounds (a smaller bound p admits a shorter prefix of
    // partners), so the pointer only ever moves down.
    std::size_t qualifying = nClasses;

    for (std::size_t i = 0; i < nClasses; ++i)
    {
        while (qualifying > 0 &&
               classWorstBound[classOrder[qualifying - 1]] * classWorstBound[classOrder[i]] <
                   schwarzThreshold)
        {
            --qualifying;
        }

        // The ordered pairs: the unordered kept set is bounded by them, and
        // its member mass by the corresponding ordered mass (each
        // off-diagonal member product is counted twice, each diagonal's
        // triangular member count is bounded by its square).
        counts.classPairs += qualifying;
        counts.memberQuartets += classMembers[classOrder[i]] * prefixMass[qualifying];
    }

    return counts;
}

void GenerateClassPairOrbits(PairClassTable& table,
                             const SymmetryReduction& reduction,
                             const ShellPairList& pairList,
                             std::size_t classPairIndex) {
    assert(classPairIndex < table.classPairs.size());

    // The class_table family: the on-demand orbit entries (and their
    // nested member vectors) construct under this scope, so their
    // allocators capture kClassTable like the Create-time table did.
    qcx::memory::AllocationTagScope classTableScope(qcx::memory::AllocationTag::kClassTable);
    ClassPair& classPair = table.classPairs[classPairIndex];

    if (!classPair.orbits.empty())
    {
        // Already generated (the fp64 and fp32 passes reach the same
        // pairs; the orbit set is a pure function of the class members).
        return;
    }

    // The validation the eager build ran lives in BuildPairClasses; the
    // table it produced is valid by construction (the work reduction is
    // the materialized identity for the trivial order-1 case).
    SymmetryReduction work = reduction;

    if (work.groupOrder == 1 && work.permutation.empty())
    {
        std::vector<std::size_t> identityPermutation(pairList.functionCount);
        std::iota(identityPermutation.begin(), identityPermutation.end(), std::size_t{0});
        work.permutation = {std::move(identityPermutation)};
        work.sign = {std::vector<int>(pairList.functionCount, 1)};
    }

    classPair.orbits = BuildOrbits(work,
                                   pairList,
                                   table.shellOfFunction,
                                   table.classes[classPair.p].members,
                                   table.classes[classPair.q].members);
}

} // namespace qcx::integrals
