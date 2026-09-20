#pragma once

/// \file
/// The AO-space symmetry reduction and the shell-pair classes of one
/// molecule (the petite list): the data the class-aware Fock
/// contraction enumerates to evaluate each symmetry-unique ERI class once
/// and expand the result over the group permutations. The reduction is
/// built by qcx::scf::BuildSymmetryReduction (the scf module owns the group
/// realization); a trivial reduction (C1) means the plain path.

#include "qcx/error.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/memory/allocation_instrument.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace qcx::integrals {

/// The action of the computational point group on the basis functions.
///
/// For the eight Abelian computational groups (C1, Cs, Ci, C2, C2v, C2h,
/// D2, D2h) realized with the molecule's symmetry planes/axes aligned with
/// the global coordinate frame, every element acts on each basis function
/// as either a sign flip of the function itself or a sign flip of its image
/// under the element's coordinate permutation - a signed permutation over
/// the function indices. The ERI invariance (g mu g nu | g lambda g sigma)
/// = (mu nu | lambda sigma) is then expressible purely in terms of these
/// permutations and signs, which is what the petite list consumes. A
/// molecule whose symmetry frame is not the global frame (or a group whose
/// realized elements are not signed coordinate permutations) cannot be
/// reduced this way; the reduction construction reports kUnimplemented and
/// the caller falls back to the plain path.
/// \ingroup qcx-integrals
struct SymmetryReduction {
    /// [g][i]: the image function index of function i under element g.
    /// elements[0] is the identity, so row 0 is always the identity
    /// permutation.
    std::vector<std::vector<std::size_t>> permutation;
    /// [g][i]: the sign of the image, +1 or -1: the action maps function i
    /// to sign[g][i] times the function permutation[g][i].
    std::vector<std::vector<int>> sign;
    /// The group order (2^k for the Abelian computational groups).
    std::size_t groupOrder = 1;
    /// True when the group's action on the CANONICAL SHELL PAIRS is the
    /// identity - every orbit is a singleton, so the reduction removes
    /// nothing and the petite list is the plain pair list.
    ///
    /// It is NOT the "the group is C1" test, and the two are not
    /// interchangeable: a group the molecule really has reads TRUE here
    /// whenever the realized elements fix every canonical shell pair. The
    /// case this repo has met is a SIGN-ONLY element (one that permutes no
    /// function and only flips signs - a planar molecule's mirror), but the
    /// predicate is wider than sign-only, and both readings agree with the
    /// flag because both ask the same question of the same pair action. The
    /// flag's question is "is there symmetry to EXPLOIT", not "was a group
    /// found" - such a group costs the classification, the class tables and
    /// the orbit tables and removes no contribution, so the consumers read
    /// this verdict instead of the group ORDER
    /// (integrals/src/lean_fock_build.cpp IsTrivialReduction,
    /// ri_engine.cpp RiOrbitExpansionEngaged; the lean predicate keeps
    /// `groupOrder <= 1` as a second guard for a hand-built reduction that
    /// never went through the builder).
    ///
    /// scf::BuildSymmetryReduction sets it from the walk of that action; a
    /// caller that HAND-BUILDS a reduction must set it (the default is the
    /// fail-safe TRUE = the plain path). Its MEASURED twin is
    /// CountClassMaterialization's counts: `pairClasses == canonicalPairs`
    /// is exactly this verdict, from the pair list alone - so a record
    /// quotes the count rather than the flag (the flag is the caller's
    /// assertion; the count is the measurement).
    bool isTrivial = true;
};

/// One orbit of canonical shell pairs under the group action: the
/// symmetry-unique pair class of the petite list.
/// \ingroup qcx-integrals
struct PairClass {
    /// The canonical pair index of the class representative (the smallest
    /// pair index in the orbit).
    std::size_t repPair = 0;
    /// The pair indices of every member, members[0] == repPair. The
    /// class_table family's tagged allocator (the tag is the scope active
    /// when the entry is built - BuildPairClasses' kClassTable scope).
    std::vector<std::size_t, qcx::memory::TaggedAllocator<std::size_t>> members;
};

/// One member of one orbit of canonical member quartets (see ClassOrbit):
/// the data the class-aware contraction needs to expand the orbit
/// representative's ERI block into this member's block. The expansion is
/// `memberBlock[memberPos] = pattern[srcPos] * repBlock[srcPos]` with
/// memberPos = pi_g(srcPos) the position permutation induced by the
/// generator g (the function images of the source position, canonicalized
/// by the within-pair function sorts and the whole-quartet bra/ket role
/// swap) and pattern[srcPos] the outer product of the representative's
/// four shells' per-position sign vectors under g - NOT a scalar multiple
/// of the rep block, because sign-mixed shells make the member block a
/// position-dependent pattern product.
/// \ingroup qcx-integrals
struct ClassOrbitMember {
    /// The canonical member bra pair index of the quartet (>= ketPair).
    std::size_t braPair = 0;
    /// The canonical member ket pair index of the quartet.
    std::size_t ketPair = 0;
    /// A group element mapping the orbit representative quartet to this
    /// member: the canonicalized image of (repBraPair, repKetPair) under
    /// generator is (braPair, ketPair). Every map of a member produces the
    /// identical expanded block, so any valid element works; the
    /// representative's own member carries the identity element 0.
    std::size_t generator = 0;
    /// slotPerm[s] = the member-block slot receiving the source slot s,
    /// where slots are the quartet positions in shell order (0, 1 = the
    /// bra pair's two shells, 2, 3 = the ket pair's two shells) and the
    /// member slots index the member quartet's canonical shell order. The
    /// expansion reads the source position's functions through
    /// permutation[generator] and scatters them into slotPerm's member
    /// slots; the two within-pair flips and the whole-quartet role swap of
    /// the canonicalization appear here as the slot mapping (the pattern
    /// factors commute with the position permutation).
    std::array<std::size_t, 4> slotPerm{0, 1, 2, 3};
};

/// One orbit of canonical member quartets of one class pair under the
/// diagonal group action (g acting on both pairs): every member's ERI
/// block equals the orbit representative's block up to the per-member
/// pattern product and position permutation (ClassOrbitMember), so the
/// class-aware contraction computes the representative block once and
/// expands it per member - one value per orbit, not per class pair
/// (the diagonal action is generally NOT transitive - e.g.
/// the diagonal P == Q quartets {(m, m)} and the off-diagonal ones split).
/// \ingroup qcx-integrals
struct ClassOrbit {
    /// The bra pair index of the orbit's representative quartet (the
    /// lexicographically smallest canonical member quartet; >= repKetPair).
    std::size_t repBraPair = 0;
    /// The ket pair index of the orbit's representative quartet.
    std::size_t repKetPair = 0;
    /// Every member quartet of the orbit, including the representative
    /// itself (members[0] is the rep with the identity generator). The
    /// members partition the class pair's canonical member quartets and
    /// are closed under the group action. The class_table family's
    /// tagged allocator (captured when the orbit entry is built under the
    /// GenerateClassPairOrbits scope).
    std::vector<ClassOrbitMember, qcx::memory::TaggedAllocator<ClassOrbitMember>> members;
};

/// One unordered pair of pair classes in canonical orientation (the
/// class-level analog of the plain path's canonical quartet bra >= ket:
/// the class with the larger representative pair index is p).
/// \ingroup qcx-integrals
struct ClassPair {
    /// The first (bra) class index; classes[p].repPair >= classes[q].repPair.
    std::size_t p = 0;
    /// The second (ket) class index.
    std::size_t q = 0;
    /// The pair index of the bra class representative (classes[p].repPair).
    std::size_t repBraPair = 0;
    /// The pair index of the ket class representative (classes[q].repPair).
    std::size_t repKetPair = 0;
    /// The exchange multiplicity of the K contraction: 2 per degenerate
    /// pair and 2 more for the self-role-swapped quartet (p == q) - the
    /// class-level form of the plain path's kMultiplicity.
    int kMult = 1;
    /// True when the bra representative pair is diagonal (i, i) - a class
    /// property (diagonal pairs map to diagonal pairs).
    bool pDiagonal = false;
    /// True when the ket representative pair is diagonal (i, i).
    bool qDiagonal = false;
    /// True when p == q (the class quartet (P, P)).
    bool diagonalPair = false;
    /// The orbits of this class pair's canonical member quartets under the
    /// diagonal group action - the per-orbit representative quartets the
    /// class-aware contraction evaluates (one ERI block per orbit) and
    /// their member expansions. The union of all orbits' members is the
    /// class pair's full canonical member quartet set.
    ///
    /// The orbit expansions are generated ON DEMAND (the root restructure,
    /// 2026-08-31): BuildPairClasses stores the pair classes and the
    /// Schwarz-screened class pairs with EMPTY orbits, and
    /// GenerateClassPairOrbits fills them the first time the contraction
    /// reaches a pair that can contain a screened member quartet. The
    /// generation is a pure function of (the pair's class members, the
    /// reduction, the pair list), so a lazy fill is deterministic - the
    /// orbit set, representatives, member order, and generators of a
    /// generated pair are identical to the eager table's - and the
    /// Create-time table never materializes the full member-quartet space
    /// (the c60 0xC0000409 death class). The class_table family's
    /// tagged allocator (the Create-time BuildPairClasses scope captures
    /// the tag; the on-demand fills ride it).
    std::vector<ClassOrbit, qcx::memory::TaggedAllocator<ClassOrbit>> orbits;
};

/// The pair-class decomposition of one molecule's canonical pair list under
/// one reduction.
/// \ingroup qcx-integrals
struct PairClassTable {
    // The class_table family (the DirectFootprint classTableBytes
    // term): every container of the table uses the tagged allocator, and
    // the builders open the kClassTable scope (BuildPairClasses and the
    // on-demand GenerateClassPairOrbits), so the Create-time build and the
    // later orbit fills both attribute to the family - the table exists
    // only on the class path, exactly the legs the term charges.
    /// The pair classes, ordered by ascending representative pair index.
    std::vector<PairClass, qcx::memory::TaggedAllocator<PairClass>> classes;
    /// classOfPair[p] = the index of the class containing canonical pair p.
    std::vector<std::size_t, qcx::memory::TaggedAllocator<std::size_t>> classOfPair;
    /// shellOfFunction[f] = the shell containing function f: the support
    /// data of the on-demand orbit generation (GenerateClassPairOrbits),
    /// computed once at BuildPairClasses - the function-level permutation
    /// maps to the shell level through it.
    std::vector<std::size_t, qcx::memory::TaggedAllocator<std::size_t>> shellOfFunction;
    /// The unordered class pairs, canonical orientation p >= q, each
    /// enumerated once - the Schwarz class-bound screening (the worst
    /// per-member Schwarz product of the pair's two classes vs
    /// schwarzThreshold) drops the class pairs that cannot contain a
    /// screened member quartet, and the surviving pairs carry their orbit
    /// expansions lazily.
    std::vector<ClassPair, qcx::memory::TaggedAllocator<ClassPair>> classPairs;
};

/// Builds the pair-class table of one molecule under one reduction.
///
/// The pair classes are the orbits of the canonical pair list under the
/// group action (a pair maps to the canonicalized pair of the shells
/// containing its two image functions); the class pairs enumerate each
/// unordered pair of classes once with the bra class first (the plain
/// path's bra >= ket convention). The trivial reduction (groupOrder 1)
/// yields one class per pair and the full class-pair set - the plain
/// enumeration (each orbit holds a single member).
///
/// The class-pair table is SCREENED by the Schwarz class bound: a class
/// pair is kept iff the product of its two classes' worst member Schwarz
/// bounds clears schwarzThreshold - the class-level analog of the plain
/// path's neighbor-list cutoff (BuildNeighborList's
/// schwarz[bra] * schwarz[ket] >= SchwarzThreshold(accuracy) * slack test,
/// evaluated over the class members, the sound superset: every
/// neighbor-list quartet lives in a kept class pair). A dropped class pair
/// contains no screened member quartet in any iteration, so the
/// contraction's (quartet, block) pairs are unchanged - the screening is
/// a throughput-only saving. The class pairs' orbit expansions are NOT
/// built here (see ClassPair::orbits and GenerateClassPairOrbits).
/// \param reduction The reduction; must be a valid signed permutation over
/// the basis (rows of the right size, indices in range, signs +-1).
/// \param pairList The canonical pair list the classes decompose.
/// \param schwarz The per-pair Schwarz bounds over pairList (size
/// pairList.pairs.size()).
/// \param schwarzThreshold The class-bound cutoff: the neighbor-list
/// threshold the plain path screens with (SchwarzThreshold(accuracy) times
/// the neighbor-list slack).
/// \returns The class table, or an Error (kInvalidArgument for a malformed
/// reduction or a Schwarz array of the wrong size).
/// \ingroup qcx-integrals
qcx::Result<PairClassTable> BuildPairClasses(const SymmetryReduction& reduction,
                                             const ShellPairList& pairList,
                                             const std::vector<double>& schwarz,
                                             double schwarzThreshold);

/// The class path's materialization counts, computed WITHOUT building the
/// table.
///
/// The class path's Create-time admission gate charges a never-under bound
/// of what the path can allocate. The eager table's ceiling
/// (internal::ClassTableBytes) is the singleton-orbit worst case - one class
/// pair and one orbit entry per canonical member quartet - and the root
/// restructure (2026-08-31) made it a gross over-charge: BuildPairClasses
/// stores only the Schwarz-screened class pairs with EMPTY orbit vectors and
/// the members generate on demand per REACHED class pair, so a well-screened
/// molecule materializes a tiny fraction of the quartet space. This is that
/// fraction's never-under measure, from the class decomposition and the
/// Schwarz bounds alone - never the theta(N^4) table.
/// \ingroup qcx-integrals
struct ClassMaterializationCounts {
    /// The plain path's canonical pair list, EXACT: the denominator of the
    /// reduction's own saving. `classes` of these pairs survive the group
    /// action, so `classes == canonicalPairs` is the count's verdict that
    /// the reduction removed nothing (the pair-trivial case, and the
    /// quantity SymmetryReduction::isTrivial asserts as a flag).
    ///
    /// The axis matters and only this one is exact: the pair count is
    /// unordered and screened by nothing. The two quartet-axis members below
    /// are ORDERED never-under bounds of a SCREENED mass, so neither may be
    /// divided by a canonical quartet count - a ratio built that way
    /// compares a bound of a screened sum against an exact unscreened
    /// triangle, which is two different screens and two different
    /// orderings. The ERI-block ratio (orbit representatives over canonical
    /// quartets) is NOT this counter's reading: the representatives exist
    /// only after the orbits are generated, and this counter's contract is
    /// that it never materializes them.
    std::size_t canonicalPairs = 0;
    /// The kept class pairs (never-under: the unordered kept set is bounded
    /// by the ORDERED pair count of the same Schwarz test).
    std::size_t classPairs = 0;
    /// The canonical member quartets the kept class pairs cover (never-under
    /// by the same ordered-pair argument; a reached class pair materializes
    /// its whole member-quartet set, and reached is a subset of kept).
    std::size_t memberQuartets = 0;
    /// The pair classes.
    std::size_t classes = 0;
};

/// Counts what the class path can materialize, without materializing it.
///
/// The measure of `ClassMaterializationCounts`: the classes and their Schwarz
/// maxima, then the kept class pairs of the same Schwarz class-bound test
/// BuildPairClasses screens with (bound[p] * bound[q] >= schwarzThreshold,
/// bound = the worst member bound) - enumerated by a sorted two-pointer sweep
/// over the classes, so the count costs O(nClasses log nClasses) and touches
/// neither the class-pair vector nor the orbit expansions. Every returned
/// count is a never-under bound of the corresponding materialized quantity:
/// the unordered kept set and its member mass are each bounded by the
/// corresponding ORDERED sum, and the diagonal's triangular member count by
/// its square.
///
/// It is also the reduction's SIZING instrument, and that reading is the one
/// to quote: `classes` of `canonicalPairs` are both EXACT, over the same pair
/// list and with no screening involved, so the ratio is the petite list's own
/// saving - and `classes == canonicalPairs` is the measurement that the
/// reduction removes nothing (SymmetryReduction::isTrivial's flag asserts
/// that verdict; this count decides it). The quartet-axis members are bounds
/// of a SCREENED ordered mass and have no exact denominator here (the struct's
/// own note); a consumer that wants an ERI-block ratio must count orbit
/// representatives, which this counter deliberately never builds.
/// \param reduction The reduction; must be a valid signed permutation over
/// the basis (the same validation BuildPairClasses runs).
/// \param pairList The canonical pair list the classes decompose.
/// \param schwarz The per-pair Schwarz bounds over pairList (size
/// pairList.pairs.size()).
/// \param schwarzThreshold The class-bound cutoff BuildPairClasses will
/// screen with.
/// \returns The counts, or an Error (the same kInvalidArgument shapes
/// BuildPairClasses reports).
/// \ingroup qcx-integrals
qcx::Result<ClassMaterializationCounts> CountClassMaterialization(
    const SymmetryReduction& reduction,
    const ShellPairList& pairList,
    const std::vector<double>& schwarz,
    double schwarzThreshold);

/// Generates the orbits of one class pair on demand.
///
/// The root restructure's laziness (ClassPair::orbits): the contraction
/// calls this the first time it reaches a class pair that can contain a
/// screened member quartet. Idempotent - a pair whose orbits are already
/// generated returns without work, and the generation is deterministic:
/// the pure function of (the pair's class members, the reduction, the pair
/// list) that BuildPairClasses used eagerly, so the lazy table contracts
/// exactly the (quartet, block) pairs of the eager table's.
/// \param table The class table (BuildPairClasses output).
/// \param reduction The reduction the table was built with.
/// \param pairList The canonical pair list the table decomposes.
/// \param classPairIndex The index into table.classPairs.
/// \ingroup qcx-integrals
void GenerateClassPairOrbits(PairClassTable& table,
                             const SymmetryReduction& reduction,
                             const ShellPairList& pairList,
                             std::size_t classPairIndex);

} // namespace qcx::integrals
