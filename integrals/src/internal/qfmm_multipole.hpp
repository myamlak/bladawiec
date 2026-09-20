#pragma once

// The multipole machinery of the QFMM octree: the moment table, the per-iteration
// density-weighted leaf aggregation, the M2M/M2L/L2L passes at general
// L_mult (0..kQfmmMaxLMult), and the far-field J accumulation. The
// monopole gate is the L_mult = 0 degenerate case of the same seams: the
// (0, 0) moment block IS the overlap block through the trusted overlap
// engine, the emitted translation tables' (0, 0) rows are exactly
// {1.0, 0, ...} with kQfmmM2LDenomPower[0][0] = 1 (so the M2M reduces to
// the exact 1.0 coefficient, the M2L to Coulomb's law at the node centers,
// and the L2L to the identity), and every pass degrades to the monopole
// code path bit-identically. The acceptance properties: with theta -> 0
// the far field is empty, so the near-field direct build plus this
// machinery must reproduce the plain direct-sum Coulomb matrix
// BIT-IDENTICALLY; at theta > 0 the far-field error must decrease
// monotonically with theta at every fixed L_mult, and with L_mult at every
// fixed theta (the acceptance gates).

#include "md_batch.hpp"
#include "md_one_electron.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qfmm_geometry.hpp"
#include "qfmm_tree.hpp"

#include <Eigen/Core>
#include <cstddef>
#include <utility>
#include <vector>

namespace qcx::integrals::internal {

/// The moments of every function pair of every canonical shell pair, on
/// demand: pair p carries
/// counts[p] blocks (counts[p] = (order(p) + 1)^2 for a pair whose leaf
/// needs order order(p), 0 for a pair in a near-field-only leaf - its
/// moments never feed any interaction, so it allocates nothing). For the
/// function pair (mu, nu) the Cartesian moments M_{a,b,c} = ∫ (x-Ax)^a
/// (y-Ay)^b (z-Az)^c mu(r) nu(r) dr about the shell-pair expansion center,
/// contracted to the Racah real solid harmonics
/// R_{l,m} = Σ_cart kSolidHarmonicG[l][m+l][cart] · x^cart (the same
/// transform the codebase's spherical folds use). Block (l, m) of pair p
/// is the (nFuncsA x nFuncsB) row-major moment matrix at
/// blocks[offsets[p] + (l^2+l+m) * nFuncs] (offsets in ELEMENT units:
/// offsets[p + 1] - offsets[p] == counts[p] * nFuncs(p)). At order 0 the
/// single (0, 0) block is the overlap matrix: the monopole moment of a
/// function pair's density IS the overlap integral, computed by calling
/// the trusted overlap engine (BuildOverlapPair, md_one_electron.hpp -
/// one_electron.cpp's BuildOverlapMatrix uses the same entry point, so the
/// moment table IS the already-tested overlap ground truth, the
/// "single biggest risk-reduction"); at every nonzero count the (0, 0)
/// block is that overlap, and the blocks with l beyond the pair's order
/// simply do not exist.
struct QfmmMomentTable {
    /// The flat block storage; pair p's counts[p] blocks live at
    /// blocks[offsets[p]] .. blocks[offsets[p + 1]] (offsets in ELEMENT
    /// units). Value-initialized; a zero-count pair allocates nothing.
    std::vector<double> blocks;
    /// Per-pair element offsets; offsets.size() == nPairs + 1 and
    /// offsets[p + 1] - offsets[p] == counts[p] * nFuncs(p).
    std::vector<std::size_t> offsets;
    /// Per-pair block counts, parallel to the pair ordering: counts[p] == 0
    /// for a pair in a near-field-only leaf, (order + 1)^2 otherwise.
    std::vector<std::size_t> counts;
};

/// Builds the moment table for every canonical pair at its per-pair order
/// \p pairOrders: pair p
/// gets counts[p] = (pairOrders[p] + 1)^2 blocks when pairOrders[p] >= 0,
/// zero blocks when pairOrders[p] == -1 (a pair in a near-field-only leaf
/// - its moments never feed any interaction, and the zero count is what
/// lets the consumers skip it without ever touching its nonexistent
/// storage). The (0, 0) block of every nonzero-count pair with order 0 is
/// the overlap block through BuildOverlapPair - the exact monopole path (the
/// block offsets make the (0, 0) block sit at blocks[offsets[p]],
/// byte-identical to the old monopole table). At order > 0 the moments are
/// built per primitive pair from the raised (la + order, lb) per-axis
/// Hermite E tables (the "add one angular momentum" trick - E
/// coefficients up to la + lb + order): the t = 0 slot of the raised E3d
/// gives the overlap-type integrals of the pair with the bra raised by
/// (p, q, r), the Cartesian moments about the pair center follow by the
/// binomial shift (x - P)^a = Σ_p C(a, p) (x - A)^p (A - P)^{a-p}, and the
/// solid-harmonic contraction lands in the (l, m) blocks (the (0, 0) block
/// comes out of this path too: a = b = c = 0 gives the overlap). Uniform
/// pairOrders (every pair the same order) reproduce the fixed-order
/// full-table build bit-for-bit: same content, only the (order + 1)^2
/// stride instead of the always-full kQfmmMomentCount stride.
/// \param pairList The canonical shell pairs.
/// \param pairStore The pair data (pair ordering must match pairList).
/// \param pairGeometries The pair expansion centers (qfmm_geometry.hpp),
/// one per pair - the moments are about these centers.
/// \param pairOrders One order per canonical pair (size == pairStore
/// size), each -1 (zero blocks) or in 0..kQfmmMaxLMult.
qcx::Result<QfmmMomentTable> BuildMomentTable(const ShellPairList& pairList,
                                              const std::vector<MdPairData>& pairStore,
                                              const std::vector<QfmmPairGeometry>& pairGeometries,
                                              const std::vector<int>& pairOrders);

/// What the order selector did on one call - the loud record of the
/// fall-through that used to be silent.
///
/// The selector can only LOWER an interaction's order; when even the cap's
/// bound exceeds epsInt the interaction runs at the cap with the budget
/// MISSED, and nothing in a run's record said so. Under the error-aware
/// admission (QfmmSeparation's arm) this count is 0 by construction - every
/// admitted pair has an allowed degree within budget - so the field is the
/// proof the two agree, and any nonzero value means the caller ran the
/// geometric admission (an explicit theta) at a cap the budget does not
/// cover.
struct QfmmOrderSelectionStats {
    std::size_t pairCount = 0; ///< Far pairs the selector saw.
    std::size_t fellThroughToCap = 0; ///< Pairs whose bound misses epsInt at the cap.
    int orderCap = 0; ///< The cap the selection ran at.
    double epsInt = 0.0; ///< The per-interaction budget the selection ran at.
    double worstRatio = 0.0; ///< The largest radius/distance over the pairs (0 with no pairs).
    double worstBound = 0.0; ///< MultipoleTruncationBound at the cap for the worst pair.
};

/// The adaptive per-interaction multipole order: for each far-field pair (A, B) the minimal order
/// L whose a-priori geometric bound (r / d)^(L+1) <= epsInt, with r = the
/// larger source CHARGE radius max(rA, rB) (the max bounds BOTH M2L
/// directions of the unordered pair, and the nodes' radii are the radii of the
/// distributions the moments are of - the box circumradius sqrt(3) * max(wA,
/// wB) this used to read over-states that by the box's own slack, so the
/// bound was loose by exactly the amount that hid the budget miss), d = the
/// center distance, and epsInt the per-interaction error budget (the preset's
/// QfmmBudgetForPreset divided by the far-pair count). L is capped at orderCap
/// - the resolved multipole order of the build
/// (min(kQfmmMaxLMult, LMultForPreset) on the default path) - so the preset
/// ladder stays the accuracy contract and the selector only LOWERS an
/// interaction's order where the bound proves it safe; an interaction whose
/// bound needs more than the cap keeps the full order AND IS COUNTED in
/// \p stats, because at the cap the budget is not met. The same bound decides
/// admission (qfmm_tree.hpp MinimalMultipoleOrder), so with the error arm live
/// no pair reaches here that the cap cannot represent.
/// Geometry-only and deterministic (same tree and pair list, same selection),
/// so the result joins the far-field pair list as the per-pair order vector.
/// \param tree The octree result (the node charge radii and centers).
/// \param farFieldPairs The unordered well-separated node pairs (each
/// appears once).
/// \param epsInt The per-interaction budget (positive).
/// \param orderCap The order cap, 0..kQfmmMaxLMult (0 selects the
/// monopole-only order for every pair - the kTight-rung shape).
/// \param stats Optional out-parameter receiving the selection's
/// fall-through record (nullptr = not asked for).
/// \returns One order per far pair, each in [0, orderCap].
std::vector<int> SelectMultipoleOrders(
    const QfmmTreeBuildResult& tree,
    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
    double epsInt,
    int orderCap,
    QfmmOrderSelectionStats* stats = nullptr);

/// The per-node needed order: node n's needed order is the maximum over the far pairs touching n
/// or any ancestor of n of (pairOrder + 1), minus 1 - the far pair (A, B)
/// demands order pairOrder of A's and B's moments (M2L source content) and
/// potentials, and of every node on the two root paths (the M2M and L2L
/// passes carry that content down), so the propagated maximum is exactly
/// the content each node must carry. The pre-order propagation (parents
/// before children in the tree's node order) makes the bands non-increasing
/// upward: children always need at least their parents' order. -1 marks a
/// node no far pair touches (a near-field-only node); its whole subtree is
/// -1 (a far pair touching a descendant would propagate up), so the passes
/// may skip it entirely - its content is never read by any interaction.
/// \param tree The octree result (nodes and children).
/// \param farFieldPairs The unordered well-separated node pairs (each
/// appears once).
/// \param pairOrders The per-pair orders from SelectMultipoleOrders (one
/// per far pair, 0..kQfmmMaxLMult).
/// \returns One order per node, each in [-1, kQfmmMaxLMult].
std::vector<int> ComputeNeededNodeOrders(
    const QfmmTreeBuildResult& tree,
    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
    const std::vector<int>& pairOrders);

/// The per-LEAF aggregate moments (the M2M source): leaf moment t = Σ over
/// the leaf's pairs of EvalM2M(t, s, leafCenter - pairCenter) ·
/// pairMoment_s with pairMoment_s = Σ over the pair's (mu, nu) function
/// pairs of D_{mu,nu} · block_s(mu, nu). The density element follows the
/// direct builder's J convention (AccumulateBlock in fock_build.cpp): both
/// orientations of an unordered pair block summed for i != j canonical
/// pairs. Density-dependent - recomputed every SCF iteration (the octree
/// and the moment table are geometry-only and built once). At order 0 the
/// M2M coefficient is exactly 1.0 (kQfmmM2M[0][0][0] = 1, all other
/// monomials exact 0), so a leaf's moment 0 is the monopole charge sum in
/// the old expression/order.
/// \param pairList The shell-pair list the tree was built over (the pair
/// indices of the leaves' pairIndices).
/// \param tree The octree result (leaf membership).
/// \param table The per-pair moment blocks from BuildMomentTable.
/// \param pairGeometries The pair expansion centers (one per pair).
/// \param density The SPATIAL closed-shell density rho (the BuildFock
/// convention), symmetric n x n.
/// \param nodeOrders One order per node (ComputeNeededNodeOrders): -1
/// skips the leaf entirely (a near-field-only leaf - its moments are
/// never read); a leaf of order o aggregates its pairs' blocks to t, s
/// <= o (the blocks exist exactly to the pair's order, which equals the
/// leaf's by construction).
/// \param chunkCount The parallel chunk override, the far-field mirror
/// of FockBuildOptions::maxParallelChunks on this pass's own task count
/// (the nodes): 0 = auto (min(taskCount, DefaultOmpTeamSize())); 1 = the
/// serial fallback (the default - direct-call callers keep the
/// bit-identical serial path); >= 2 a fixed split (min(taskCount,
/// chunkCount)). Each node's moments are written by exactly one chunk
/// (single-writer per element), so the chunked evaluation is bit-identical
/// to the serial in any schedule.
/// \returns kQfmmMomentCount moments per node (internal nodes zero), layout
/// moments[node * kQfmmMomentCount + idx(l, m)].
std::vector<double> AggregateLeafMoments(const ShellPairList& pairList,
                                         const QfmmTreeBuildResult& tree,
                                         const QfmmMomentTable& table,
                                         const std::vector<QfmmPairGeometry>& pairGeometries,
                                         const Eigen::MatrixXd& density,
                                         const std::vector<int>& nodeOrders,
                                         std::size_t chunkCount = 1);

/// The M2M pass: a node's moments are the M2M-translated sum of its
/// children's - moments[t] = Σ_children Σ_s EvalM2M(t, s, cNode - cChild)
/// · moments[child][s]. Node indices are pre-order (parents before
/// children), so the reverse iteration visits children before parents. At
/// order 0 the M2M coefficient is exactly 1.0, so a node's moment 0 is the
/// monopole child-charge sum in the old expression/order.
/// \param leafMoments The per-node moments from AggregateLeafMoments.
/// \param nodeOrders One order per node: -1 nodes (whole subtrees - the
/// orders are non-increasing upward, so a node's entire subtree shares its
/// -1) aggregate nothing - their content is never read. An order-o node
/// computes t, s <= o; its children carry at least order o (children's
/// orders never fall below their parents'), so every M2M read sits inside
/// the child's content.
/// \returns kQfmmMomentCount moments per node, layout
/// moments[node * kQfmmMomentCount + idx(l, m)].
std::vector<double> AggregateNodeMoments(const QfmmTreeBuildResult& tree,
                                         const std::vector<double>& leafMoments,
                                         const std::vector<int>& nodeOrders);

/// The M2L coefficient work of one far-field pass: the reversed-direction
/// (t, s) rows it served from the row parity fold instead of a second table
/// evaluation. The emitted M2L row of (t, s) is a HOMOGENEOUS polynomial of
/// degree l_t + l_s (the generator's convention - tools/
/// gen_qfmm_translation_tables.py), so the reversed direction is the forward
/// row up to the exact sign (-1)^(l_t + l_s) and one evaluation serves both
/// directions of every far pair: EvalM2L(t, s, -d) == (-1)^(l_t + l_s) ·
/// EvalM2L(t, s, d), bit for bit.
///
/// The count is incremented where the fold is APPLIED, so it is the fold's
/// witness rather than a restatement of the loop bounds: a pass that accepts
/// the fold and drops it (the reversed direction evaluated from the table
/// again) reports zero here while the same pairs still run. The expected
/// value for a pass over a known pair list is one fold per (t, s) row, which
/// a caller derives from the pair orders.
struct QfmmM2LCoefficientStats {
    /// The reversed-direction rows the pass served by the fold.
    std::size_t foldedRows = 0;
};

/// The M2L and L2L passes: for every far-field node pair (a, b) the M2L
/// translates b's moments into a's local expansion and a's into b's - the
/// unordered far pair appears once, both directions with the SAME
/// coefficient table: potentials[a][t] += EvalM2L(t, s, cA - cB) ·
/// (moments[b][s] / |cA - cB|^kQfmmM2LDenomPower[t][s]) and the symmetric
/// write at b with the REVERSED displacement (the emitted polynomial is
/// the target-side local-expansion coefficient with d = cTarget - cSource
/// - the generator's V7 convention - and the odd-parity terms flip sign
/// between the two directions; at order 0 the two reads coincide
/// exactly). The two directions share ONE evaluation: the row homogeneity
/// above makes the reversed direction the forward row times the exact sign
/// (-1)^(l_t + l_s), so the pass evaluates the table once per (t, s) row
/// instead of twice - a reuse that assumes no symmetry of the geometry,
/// the tree or the density, and moves no value (see
/// QfmmM2LCoefficientStats). The pair's truncation is ITS order (the
/// per-pair orders): both the source moments and the target writes stay within
/// t, s <= pairOrders[i], and pairOrders[i] never exceeds the two nodes'
/// orders (ComputeNeededNodeOrders takes the maximum over the touching
/// pairs), so both sides read and write existing content. The L2L then
/// descends each local expansion to its descendants: potentials[child][t]
/// += Σ_s EvalL2L(s, t, cChild - cNode) · potentials[node][s] (the
/// top-down pass; pre-order indices), capped at t, s <= order(child)
/// (parent content beyond order(parent) is exactly zero - the L2L's
/// zero-adds there are inert). At order 0 the M2L reduces to Coulomb's law
/// between the node centers (coefficient exactly 1.0, denominator exactly
/// |cA - cB| - the old nodeCharges[b] / distance expression) and the L2L
/// to the identity (potentials[child] += potentials[node]), both in the
/// monopole expression/order.
/// \param tree The octree result (nodes and children).
/// \param farFieldPairs The unordered well-separated node pairs from the
/// tree build (each appears once; both directions are written).
/// \param nodeMoments The per-node moments (kQfmmMomentCount per node) from
/// AggregateNodeMoments.
/// \param pairOrders One order per far-field pair (SelectMultipoleOrders -
/// the M2L truncation of both directions of the pair).
/// \param nodeOrders One order per node: a -1 child's subtree is skipped
/// by the L2L (its potentials stay the exact zeros nothing reads).
/// \param chunkCount The parallel chunk override (see
/// AggregateLeafMoments): the M2L chunks over the far pairs with per-chunk
/// partial potentials joined in FIXED chunk order after the parallel phase
/// (deterministic - not the completion-order merge of ParallelReduce), so
/// the chunked result differs from the serial only in the fp grouping of
/// each node's adds (last bits - budget-irrelevant for the approximate far
/// field) and is bit-reproducible run to run. The L2L chunks per
/// root-child subtree after the M2L join (the root step first, then each
/// subtree descends independently - every node block is written exactly
/// once, by its parent's step, so the L2L is bit-identical to the serial
/// in any schedule). 1 (the default) forces the serial path.
/// \param coefficientStats Optional out-parameter, reset here: the M2L
/// coefficient work of THIS call (QfmmM2LCoefficientStats) - the fold's
/// witness, and the only way a caller can see that the pass folded the
/// reversed direction rather than evaluating it. Null (the default) counts
/// nothing and the pass runs the unfolded-instrumentation instantiation.
/// The counts are summed over the chunks in chunk order after the parallel
/// phase, so they are deterministic at every chunk count.
/// \returns kQfmmMomentCount potentials per node (only leaves are consumed
/// downstream), layout potentials[node * kQfmmMomentCount + idx(l, m)].
std::vector<double> BuildFarFieldPotentials(
    const QfmmTreeBuildResult& tree,
    const std::vector<std::pair<std::size_t, std::size_t>>& farFieldPairs,
    const std::vector<double>& nodeMoments,
    const std::vector<int>& pairOrders,
    const std::vector<int>& nodeOrders,
    std::size_t chunkCount = 1,
    QfmmM2LCoefficientStats* coefficientStats = nullptr);

/// Accumulates the far-field Fock contribution 2·J_far for every function
/// pair: F_{mu,nu} += Σ_{l,m: l <= order(leaf)} 2 · momentBlock'_{l,m}(mu,nu)
/// · potential(node of (mu,nu))_{l,m}, written to both orientations for
/// i != j - the direct builder's J convention (the 2J factor and the
/// transposed write, AccumulateBlock in fock_build.cpp), so the restricted
/// near-field build and this accumulation compose to the full J. The
/// moment blocks are about the PAIR centers while the potentials are the
/// local expansion about the NODE centers: each block is first translated
/// to its node center by the same EvalM2M shift the leaf aggregation
/// applies (the identity at order 0 - the monopole single 2 · q · pot term,
/// unchanged bit-for-bit). The per-pair block reads stay within the
/// pair's count (the pair's order equals its leaf's), and the potential
/// reads within the leaf's order.
/// \param fock The spatial closed-shell Fock matrix, updated in place.
/// \param pairList The shell-pair list the tree was built over (the pair
/// indices of the leaves' pairIndices).
/// \param tree The octree result (leaf membership).
/// \param table The per-pair moment blocks from BuildMomentTable.
/// \param potentials The per-node local expansions from
/// BuildFarFieldPotentials.
/// \param pairGeometries The pair expansion centers (one per pair - the
/// translation displacement).
/// \param nodeOrders One order per node: -1 leaves are skipped entirely -
/// their pairs hold zero blocks (nothing to dereference), and their
/// potentials are the exact zeros nothing wrote.
/// \param chunkCount The parallel chunk override (see
/// AggregateLeafMoments): the pass chunks over the leaf nodes. Each
/// canonical pair lives in exactly one leaf and the pairs' function-block
/// index sets are disjoint, so the fock writes are single-writer per
/// element - the chunked evaluation is bit-identical to the serial in any
/// schedule. 1 (the default) forces the serial path.
void AccumulateFarFieldJ(Eigen::MatrixXd& fock,
                         const ShellPairList& pairList,
                         const QfmmTreeBuildResult& tree,
                         const QfmmMomentTable& table,
                         const std::vector<double>& potentials,
                         const std::vector<QfmmPairGeometry>& pairGeometries,
                         const std::vector<int>& nodeOrders,
                         std::size_t chunkCount = 1);

} // namespace qcx::integrals::internal
