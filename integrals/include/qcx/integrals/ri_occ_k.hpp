#pragma once

/// \file
/// The occ-RI-K contraction core: the auxiliary-metric transform B = I s, the
/// occupied-orbital transform B^P_ui = sum_l B^P_ul C_li, and the exchange
/// contraction K = sum_P B^P_occ (B^P_occ)^T.
///
/// These are module-level entry points. The composed full-RI builder
/// (ri_full_fock.hpp RiFullFockBuilder) runs the first three as
/// its per-iteration chain; the batched build (BuildBatchedRiContractions)
/// is the cache-tuned re-expression of that chain, and the
/// builder's DENSE per-call path runs it in place of the chain
/// when its caller asked for it (ri_full_fock.hpp RiContractionBatchOptions,
/// whose default of 0 selects the chain). The shipped RiJkFockBuilder
/// (ri_engine.hpp - the ri_j_link family, RI-J with direct exchange) is
/// untouched by this header.
/// The composition is
///   B    = BuildMetricTransformedTensor(I, M, floor)
///   Bocc = TransformToOccupiedOrbitals(B, C_occ)
///   K    = BuildRiExchangeMatrix(Bocc, nOcc)
/// where I is the (uv|P) tensor in the contraction layout (rows u*n + v,
/// columns P - ri_engine.hpp RiMatrix(), BuildRiTensorChunk), M the (P|Q)
/// auxiliary metric (BuildAuxMetric / BuildBlockedAuxMetric), and C_occ the
/// occupied block of the SCF's MO coefficient matrix.
///
/// Two exact algebraic identities underpin this transform. Both carry zero
/// approximation error, so
/// between them they separate a transform defect from the auxiliary fit error
/// (the discrimination the accuracy cell needs when it misses):
/// - the J-from-B identity: with d in the same u*n + v layout, J = B (B^T d)
///   equals the shipped eigen-path J = I V diag(invLambda) V^T (I^T d) to
///   round-off (the shipped path's BUG-2 floored eigen-inverse, ri_engine.cpp);
/// - the occ-form/AO-form identity: for rho = C_occ C_occ^T the occ contraction
///   sum_P (B^P C_occ)(B^P C_occ)^T equals the AO-basis form
///   sum_P B^P rho B^P exactly, for any C_occ.
///
/// The single approximation on this path is the auxiliary-basis fit, and it is
/// not measured here: it is the caller's acceptance cell, which records the
/// measured aux K-fit error
/// per fixture it touches.
/// \ingroup qcx-integrals

#include "qcx/error.hpp"

#include <Eigen/Dense>
#include <cstddef>

namespace qcx::integrals {

/// The relative floor of the auxiliary-metric eigen-inverse (the shipped RI
/// path's metricFloorEpsilon default, ri_engine.cpp kMetricFloorEpsilon):
/// eigenvalues below kDefaultMetricFloorEpsilon * lambdaMax are zeroed in the
/// inverse. Public so a caller's own documentation and a test's fixture can
/// name the value the shipped path uses without re-deriving it.
/// \ingroup qcx-integrals
inline constexpr double kDefaultMetricFloorEpsilon = 1e-10;

/// The transformed SLAB's byte target for one auxiliary block of the batched
/// contraction build (BuildBatchedRiContractions).
///
/// What it sizes: the block's columns of B, `n^2 * width * 8` bytes. That slab
/// is the one array the block's three passes reuse - the Coulomb weights
/// `blk^T d`, the Coulomb product `blk w`, and the occ transform of each of its
/// columns - so its residency is what decides whether the block is read from
/// DRAM once or three times. The target is therefore the cache-tuned quantity
/// of this build, and it is deliberately MUCH smaller than
/// `RiEngineOptions::maxBatchBytes`: the memory-fit seam asks "what fits the
/// budget", this asks "what stays resident".
///
/// It is a TARGET chosen before measurement, not a measured cache size: 4 MiB
/// is a conservative fraction of the last-level cache of the machines this
/// engine is developed on, and it degenerates to a ONE-COLUMN block on a large
/// basis (n = 600: one column is 2.75 MiB, so the width floors at 1), which is
/// the strongest form of the reuse this build can have - a single column
/// held while all three passes run over it. The width it produces is only ever
/// an upper bound on the reuse; the block still has to fit, and the suite's
/// timing cell is what reports whether it does. The acceptance for this
/// build is a TIMING claim (>= 20% at 600 BF, paired-interleaved against
/// the ri_j_link row) and is not made here.
/// \ingroup qcx-integrals
inline constexpr std::size_t kRiContractionBlockBytes = 4 * 1024 * 1024;

/// The auxiliary block width of the batched contraction build: the number of
/// columns of B one block carries.
///
/// Three terms, in the order they bind at scale:
/// - the CACHE term (\p blockTargetBytes, kRiContractionBlockBytes by
///   default): the width whose transformed slab `n^2 * width * 8` fits the
///   residency target - the cache-tuned part;
/// - the SEAM term (\p maxBatchBytes): the same per-class cap the 3-center
///   route and the RI-J light rung run under (ri_engine.hpp
///   RiEngineOptions::maxBatchBytes), so no block here can exceed the budget
///   the caller already declared;
/// - the EXTENT term (\p columnCount): a width past the auxiliary count is a
///   single block.
///
/// Both byte arguments are clamped to one column's worth before the minimum is
/// taken, so the result is never 0 for a positive extent - a target or a cap
/// below one column degrades to one column per block, which is value-neutral
/// (asserted by the suite's partition cell) and not a refusal.
/// \param nSquared The contraction layout's row count (n^2) - the slab's rows.
/// \param columnCount The number of auxiliary functions - the slab's maximum
/// columns. Zero returns 0.
/// \param maxBatchBytes The seam's per-class cap in bytes.
/// \param blockTargetBytes The transformed slab's residency target in bytes.
/// \returns The block width in columns, in [1, columnCount].
/// \ingroup qcx-integrals
std::size_t RiContractionBlockWidth(std::size_t nSquared,
                                    std::size_t columnCount,
                                    std::size_t maxBatchBytes,
                                    std::size_t blockTargetBytes = kRiContractionBlockBytes);

/// The two contractions of the metric-transformed tensor B, built in one
/// traversal of the auxiliary index (the batched transform+K build).
/// \ingroup qcx-integrals
struct RiBatchedContractions {
    /// J = B (B^T d), the Coulomb half through B - the same matrix the
    /// J-from-B identity of the file comment names, and the one the Fock
    /// composition multiplies by two.
    Eigen::MatrixXd coulomb;
    /// K = sum_P B^P_occ (B^P_occ)^T, the occ-RI-K exchange half.
    Eigen::MatrixXd exchange;
    /// The auxiliary block width the traversal actually ran at, in columns -
    /// `RiContractionBlockWidth`'s answer from this call's arguments, reported
    /// by the call that ran it rather than re-derived by its caller (one home
    /// for the partition, and the number a record of this build states).
    std::size_t blockWidth = 0;
    /// The number of blocks the auxiliary index was partitioned into
    /// (`ceil(columns / blockWidth)`). 1 means the block covered the whole
    /// auxiliary index, which is the partition under which this build is the
    /// chain's arithmetic in the chain's order - bit-identical, and pinned so.
    std::size_t blockCount = 0;
};

/// Builds BOTH contractions of the metric-transformed tensor in one batched
/// traversal of the auxiliary index - the cache-tuned re-expression of the
/// transform+K chain, and the batched entry point.
///
/// What it does that the chain it replaces
/// (TransformToOccupiedOrbitals + BuildRiExchangeMatrix + the Coulomb
/// contraction) does not:
/// - the auxiliary index is walked in BLOCKS whose width is
///   RiContractionBlockWidth's cache-tuned answer, and every pass of a block
///   runs while the block's slab is the recently-touched array - the three
///   passes over B become one DRAM pass plus two cache passes at the intended
///   widths;
/// - the occ-transformed tensor is NEVER materialized: each block's occ columns
///   are formed, contracted into K, and dropped. The `n * nOcc * nAux` array
///   the chain allocates - the term the memory ladder's own class charges as
///   RiFullFockModeInfo::occTransformBytes - does not exist here. The working
///   set is the slab plus one `n * nOcc * width` occ block.
///
/// The build is exact: it is the same arithmetic on the same operands, and the
/// auxiliary partition is value-neutral. At a width covering the whole
/// auxiliary index the result is bit-identical to the chain's (the same
/// products in the same order); at a narrower width the Coulomb accumulation is
/// regrouped per block, which is a round-off difference and not an
/// approximation - `blockTargetBytes` moves no physics. The suite's
/// partition cell measures both: MEASURED on the water/def2-SVP and
/// H2O/STO-3G fixtures, the full-width build returns both products
/// bit-identical to the chain's (deviation exactly 0), the one-column build
/// returns the exchange bit-identical as well (the occ transform and the
/// contraction are the chain's expressions in the chain's order) and the
/// Coulomb to 2.2e-15 at scale 3.8.
///
/// **MEASURED 2026-09-16, single run under heavy concurrent load (CPU
/// at 100%), n = 200 / nAux = 1000 / nOcc = 30:** the chain's own cost is 16%
/// the two Coulomb passes over B and 84% the occ transform plus the exchange
/// contraction (0.066 s of 0.40 s, a parts probe of this fixture), so the
/// traversal this build fuses has a **~16% ceiling** there - and at a
/// T1-shaped occupied fraction (nOcc/n about 0.5 against the fixture's 0.15)
/// roughly a quarter of it, because the same n^2 nAux traffic buys more of the
/// compute. **Comparing the arms was NOT possible at that contention:** over
/// three interleaved runs one arm moved by more than 20x (the full-width build
/// 0.48 s in the first run, 12.1 s in the third, against a 0.38-0.66 s chain),
/// which is a machine-state reading and not a reading about the build. So the
/// speed question this build's acceptance asks is still OPEN here: it is a
/// paired-interleaved whole-builder comparison at 600 BF against the ri_j_link
/// row, it needs a settled machine, and the cost cell's numbers are an input to
/// that decision rather than a substitute for it. What the build adds
/// independently of any timing is the smaller working set below.
///
/// The run's term counters are untouched: this entry point evaluates no
/// 3-center block and calls no quartet kernel, so it has no occurrence to
/// count.
/// \param transformed The metric-transformed tensor B: rows u*n + v, columns P
/// - the output of BuildMetricTransformedTensor.
/// \param density The SPATIAL closed-shell density rho = D/2 (the
/// ri_engine.hpp convention), n x n for n = the occupied block's row count.
/// \param occupiedOrbitals The occupied MO coefficient block C_occ, n x nOcc in
/// the AO basis (the same block TransformToOccupiedOrbitals takes; nOcc must be
/// positive).
/// \param maxBatchBytes The seam's per-class byte cap; must be positive. It
/// bounds the block's transformed slab (`n^2 * width * 8`) - see
/// RiContractionBlockWidth.
/// \param blockTargetBytes The transformed slab's residency target; see
/// kRiContractionBlockBytes. Must be positive.
/// \returns The Coulomb and exchange contractions, or an Error
/// (kInvalidArgument for an empty C_occ, a B whose row count is not n*n for
/// n = C_occ's row count, a density whose shape is not n x n, or a zero
/// maxBatchBytes or blockTargetBytes - the chain's own guards, kept in the same
/// order and with the same messages where the operation is the same one).
/// \ingroup qcx-integrals
qcx::Result<RiBatchedContractions> BuildBatchedRiContractions(
    const Eigen::MatrixXd& transformed,
    const Eigen::MatrixXd& density,
    const Eigen::MatrixXd& occupiedOrbitals,
    std::size_t maxBatchBytes,
    std::size_t blockTargetBytes = kRiContractionBlockBytes);

/// The square root of the auxiliary metric's FLOORED eigen-inverse,
/// s = V diag(sqrt(invLambda)) V^T (M = V diag(Lambda) V^T; invLambda_i =
/// 1/Lambda_i for Lambda_i >= floor * lambdaMax, else 0 - the shipped RI
/// path's rule, ri_engine.cpp, so sqrt(invLambda) keeps the floored
/// directions exactly zero). This is the single home of the regularization
/// rule: `BuildMetricTransformedTensor` is one product against the matrix
/// this returns, and a caller that must transform in BLOCKS - the
/// RiFullFockBuilder's blocked rung, whose whole point is that the raw
/// tensor is never fully live - retains the root and accumulates
/// B += I_chunk * s_chunk instead of re-deriving the rule.
///
/// Cost is one nAux x nAux self-adjoint eigendecomposition. The root is
/// density- and tensor-independent: one build per SCF run.
/// \param auxMetric The (P|Q) auxiliary metric, nAux x nAux symmetric
/// positive definite (BuildAuxMetric's matrix; the same matrix the shipped RI
/// path factorizes).
/// \param metricFloorEpsilon The relative floor of the eigen-inverse
/// (kDefaultMetricFloorEpsilon by default, matching the shipped path's
/// default); zero disables the floor. Must not be negative.
/// \returns s as an nAux x nAux symmetric matrix, or an Error
/// (kInvalidArgument for a negative floor, an empty or non-square metric, a
/// failed eigendecomposition, or a metric degenerate below the floor - the
/// shipped path's own two guards).
/// \ingroup qcx-integrals
qcx::Result<Eigen::MatrixXd> MetricInverseRoot(
    const Eigen::MatrixXd& auxMetric, double metricFloorEpsilon = kDefaultMetricFloorEpsilon);

/// Builds the metric-transformed 3-center tensor B = I s with
/// s = V diag(sqrt(invLambda)) V^T the square root of the auxiliary metric's
/// FLOORED eigen-inverse (M = V diag(Lambda) V^T; invLambda_i = 1/Lambda_i for
/// Lambda_i >= floor * lambdaMax, else 0 - the shipped RI path's rule,
/// ri_engine.cpp, so sqrt(invLambda) keeps the floored directions exactly
/// zero; the rule itself lives in MetricInverseRoot, which this is one
/// product against). B is what the occ transform and the exchange contraction work from,
/// and it is also the tensor the J-from-B identity of the file comment uses:
/// with M held fixed, s s = V diag(invLambda) V^T exactly, so the shipped
/// eigen-path J = I (V diag(invLambda) V^T) (I^T d) equals B (B^T d) up to
/// round-off.
///
/// Cost is one nAux x nAux self-adjoint eigendecomposition plus one
/// n^2 x nAux by nAux x nAux product; B replaces I rather than joining it, and
/// B is density-independent (one build per SCF run, not per iteration).
/// \param riTensor The (uv|P) 3-center tensor in the contraction layout:
/// rows u*n + v, columns P (ri_engine.hpp RiMatrix(), BuildRiTensorChunk).
/// \param auxMetric The (P|Q) auxiliary metric, nAux x nAux symmetric
/// positive definite (BuildAuxMetric's matrix; the same matrix the shipped RI
/// path factorizes).
/// \param metricFloorEpsilon The relative floor of the eigen-inverse
/// (kDefaultMetricFloorEpsilon by default, matching the shipped path's
/// default); zero disables the floor. Must not be negative.
/// \returns B as an n^2 x nAux matrix in the same contraction layout as
/// \p riTensor, or an Error (kInvalidArgument for a negative floor, an empty
/// or non-square metric, a metric/tensor column mismatch, a failed
/// eigendecomposition, or a metric degenerate below the floor - the shipped
/// path's own two guards).
/// \ingroup qcx-integrals
qcx::Result<Eigen::MatrixXd> BuildMetricTransformedTensor(
    const Eigen::MatrixXd& riTensor,
    const Eigen::MatrixXd& auxMetric,
    double metricFloorEpsilon = kDefaultMetricFloorEpsilon);

/// The occupied-orbital transform B^P_ui = sum_l B^P_ul C_li: the half
/// transformation that takes the transformed 3-center tensor into the occupied
/// basis, one n x n by n x nOcc product per auxiliary function. Only the ket
/// index is transformed - the exchange contraction
/// needs no more, and that is what halves the per-iteration flops of the
/// AO-basis form.
///
/// The result's row index is u*nOcc + i (orbital function outer, occupied
/// inner), i.e. the same "contraction layout" convention as the input's
/// u*n + v, so the exchange contraction is a plain product over its columns.
/// The transform is algebraic and density-independent: it takes the MO
/// coefficient block, not a density, and is exact for any C_occ (the
/// occ-form/AO-form identity of the file comment needs C_occ only to define
/// rho = C_occ C_occ^T).
/// \param transformed The metric-transformed tensor B: rows u*n + v, columns
/// P - the output of BuildMetricTransformedTensor.
/// \param occupiedOrbitals The occupied MO coefficient block C_occ, n x nOcc
/// in the AO basis (the SCF's orbital coefficients; nOcc must be positive).
/// \returns The occ-transformed tensor, (n*nOcc) x nAux, or an Error
/// (kInvalidArgument for an empty C_occ, or a B whose row count is not
/// n*n for n = C_occ's row count).
/// \ingroup qcx-integrals
qcx::Result<Eigen::MatrixXd> TransformToOccupiedOrbitals(const Eigen::MatrixXd& transformed,
                                                         const Eigen::MatrixXd& occupiedOrbitals);

/// The exchange contraction K = sum_P B^P_occ (B^P_occ)^T over the auxiliary
/// functions: the RI exchange matrix for the closed-shell spatial density
/// rho = C_occ C_occ^T the occ transform was built from (the density
/// convention of ri_engine.hpp - the spin-summed D over 2). The K matrix is
/// the one the Fock composition F = H + 2J - K consumes, and it is symmetric
/// by construction (each rank-nOcc update contributes a symmetric block).
///
/// This is the unscreened dense contraction: no auxiliary-pair
/// screening and no batching. The contraction is one n x nOcc by nOcc x n
/// update per auxiliary
/// function.
/// \param occTransformed The occ-transformed tensor: rows u*nOcc + i, columns
/// P - the output of TransformToOccupiedOrbitals.
/// \param occupiedCount The occupied orbital count nOcc the row layout was
/// built with (the columns of the C_occ handed to
/// TransformToOccupiedOrbitals); must be positive and divide the row count.
/// \returns The n x n exchange matrix with n = rows / nOcc, or an Error
/// (kInvalidArgument for a zero occupiedCount, an empty tensor, or a row
/// count that is not n*nOcc).
/// \ingroup qcx-integrals
qcx::Result<Eigen::MatrixXd> BuildRiExchangeMatrix(const Eigen::MatrixXd& occTransformed,
                                                   std::size_t occupiedCount);

} // namespace qcx::integrals
