#pragma once

/// \file
/// The composed full-RI Fock builder: the builder that evaluates BOTH halves
/// of the Kohn-Sham matrix through the auxiliary basis,
///   F(D) = H + 2 J_RI(rho) - K_RI(rho),
/// with **no nested direct-exchange call** - the property that separates this
/// class from RiJkFockBuilder (ri_engine.hpp), which is the `ri_j_link` family:
/// RI-J for the Coulomb part and the direct screened exchange for K.
///
/// The exchange half is the occ-RI-K contraction (`ri_occ_k.hpp`):
///   B    = BuildMetricTransformedTensor(I, M, floor)
///   Bocc = TransformToOccupiedOrbitals(B, C_occ)
///   K    = BuildRiExchangeMatrix(Bocc, nOcc)
/// The Coulomb half reads the SAME `B`: J = B (B^T d), which is algebraically
/// identical to the shipped eigen-path J = I V diag(invLambda) V^T (I^T d).
/// One `n^2 x nAux` array therefore serves both halves - `B` replaces the raw
/// 3-center tensor at Create rather than joining it - and no
/// second tensor layout is introduced.
///
/// The two halves are held as SEPARATE locals and meet only at the composition
/// line, so the builder hands them out as well as composing them:
/// `BuildFockHalves` is the same call `BuildFock` makes (the same validation,
/// the same traversal, the same per-call sinks) returning the pair, and
/// `BuildFock` is that pair plus the one-line composition. A caller that needs
/// the halves separately - the unrestricted leg's per-spin assembly, which
/// needs a spin's exchange at that spin's density and no second traversal to
/// get it - therefore pays what the fused call pays and no more.
/// `RiFullFockHalves` states the accounting of the pair.
///
/// Two properties a caller must not have to infer:
/// - **The exchange result depends on C_occ only through rho = C_occ C_occ^T.**
///   The occ-form contraction sum_P (B^P C_occ)(B^P C_occ)^T equals the AO form
///   sum_P B^P rho B^P exactly for any C_occ (the identity verified at
///   4.4e-16). The two arguments of BuildFock are therefore not independent
///   inputs, and a caller must hand over the pair from ONE SCF iterate - a rho
///   from iterate k beside a C_occ from iterate j silently evaluates the
///   Coulomb and exchange halves at two different densities. This builder
///   cannot detect that cheaply (the consistency residual rho - C_occ C_occ^T
///   costs n^2*nOcc to form, and a DIIS-extrapolated rho is legitimately not of
///   the form C C^T at all), so the contract is stated rather than enforced.
/// - **No auxiliary-quality check is performed here, and none can be.** An
///   `ri_jk` request whose auxiliary basis in
///   effect is not a JK-optimized fit is refused by the predicate over the
///   RESOLVED AUX NAME (`aux_basis.hpp` IsJkOptimizedAux). A BasisSet carries
///   no name, so this class receives an already-parsed basis and cannot see
///   which family it came from: the refusal belongs to the site that resolves
///   the name (the io/driver wiring), not to the builder. Running this builder
///   against a J-only fit is numerically legal and is the ~10x-worse error
///   class that refusal detects.
///
/// What is available here: the auxiliary-pair density screen
/// (RiAuxScreenOptions), which is off at its
/// default threshold of 0, so this class's dense path is unchanged unless a
/// caller asks. **The auxiliary-index batching for CACHE is available and
/// WIRED** - `RiContractionBatchOptions` selects the batched traversal
/// (`BuildBatchedRiContractions`, ri_occ_k.hpp) in place of the three-call
/// chain, and the per-call `RiContractionStats` sink states which one ran. It
/// too is OFF at its own default (a residency target of 0 selects the chain)
/// and because its acceptance is UNMET: the target is
/// >= 20% at 600 BF, the parts probe bounds the fused traversal at
/// ~16% of the chain on its fixture and ~4% at a T1-shaped occupied fraction,
/// and its cost cell could not order its arms on a contended machine - so the
/// traversal is a caller's choice and never a default this class assumes. What
/// a run may NOT do is ask for both treatments at once: the batched traversal
/// takes the auxiliary index as its block partition and does not carry the
/// screen's surviving (pair, aux shell) structure, so `Create` REFUSES the
/// combination by name rather than accepting the screen and quietly dropping
/// it (the silent-substitution class).
///
/// The MEMORY LADDER is engaged: the 3-center tensor is always materialized at
/// Create (both rungs retain B, which is what makes the per-iteration
/// contraction cheap), and the workspace-budget seam selects between the two
/// rungs of `RiFullFockRung` - the ladder's batched/blocked rung before any
/// per-iteration recompute. The recompute rung itself (no retained tensor,
/// every iteration re-evaluating the 3-center blocks - `forceLightRung` on
/// the RI-J path, ri_engine.hpp) is NOT implemented here and is refused by
/// name; see Create's option contract.
///
/// **The term counters are CARRIED here, and they cover the terms this
/// class actually evaluates.** The counter contract (ri_engine.hpp) is that a
/// term whose work never occurred reads 0 and that the 3-center occurrences
/// count UNCONDITIONALLY - so a zero on a counter whose work did happen is a
/// false statement, not a missing one. `Create` populates `x`, `p3` and `g3`
/// (`TermCounters()`) from the accumulation the monolithic build runs, through
/// `BuildRiTensorChunk`'s sink and `BuildAuxMetric`'s - both wired
/// 2026-09-15, since before them this builder had nowhere to put the metric
/// and tensor-pass counts and every one of those fields was unpopulated rather
/// than measured-zero. `qx` and `gx` stay zero: this builder calls no quartet
/// kernel, and `BuildFock`'s stats sink states that structurally with a
/// written default. These are exactly the terms a point-group reduction would
/// shrink, which is why the reduction request is REFUSED rather than accepted
/// (see `Create`'s option contract).
/// \ingroup qcx-integrals

#include "qcx/backend/tags.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/ri_engine.hpp"
#include "qcx/memory/tensor.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace qcx::integrals {

/// The Create-time rung of the composed full-RI builder (the memory ladder;
/// ri_full_fock.hpp's Create contract).
///
/// The rungs differ only in how the (uv|P) values and their transform overlap
/// - both build the same tensor, from the same kernels, and every rung's B,
/// K, J and F are equal to round-off (the rung-identity cell of
/// integrals/tests/ri_full_fock_test.cpp):
/// - **kFast**: the raw 3-center tensor and its transform are live at once
///   (Eigen's product allocates its output while its input is alive). This is
///   the landing path, and the one a builder without a workspace budget takes.
/// - **kBlocked**: the transform is accumulated one auxiliary SHELL RANGE at a
///   time - B += I_chunk s_chunk against the retained root - so the raw tensor
///   is never fully live and the peak drops by one full 8 n^2 nAux array. It is
///   the ladder's SECOND rung (batched/blocked), the one that costs no
///   additional flops: the accumulated product has the same n^2 nAux^2 total as
///   the single one.
/// \ingroup qcx-integrals
enum class RiFullFockRung : std::uint8_t {
    kFast = 0, ///< The raw tensor and its transform live at once.
    kBlocked = 1, ///< The transform accumulated in auxiliary shell ranges.
};

/// The Create-time rung decision of a builder created WITH a workspace budget
/// (the adaptive-memory mode record - ri_engine.hpp FockModeInfo's RI-K
/// counterpart, with this builder's own rung vocabulary). A builder created
/// without a budget reports `engaged == false`: the legacy path consults no
/// budget and takes `kFast`, so there is no decision to record and no bytes
/// charged on the caller's behalf.
/// \ingroup qcx-integrals
struct RiFullFockModeInfo {
    bool engaged = false; ///< A budget was given and a rung was decided.
    RiFullFockRung rung = RiFullFockRung::kFast; ///< The rung that engaged.
    std::size_t predictedBytes = 0; ///< The engaged rung's Create-plus-first-call peak.
    std::size_t reservedBytes = 0; ///< Bytes charged to the budget (0 when Reserve failed).
    std::size_t budgetBytes = 0; ///< The budget's capacity at Create.
    std::size_t remainingAtDecision = 0; ///< Remaining() read at the decision.
    /// The batch cap the decision ran at (the scratch clamp's result;
    /// `options.maxBatchBytes` when the clamp did not bind). Every Create-time
    /// build runs at this cap, so the charged 3c arena is the realized one.
    std::size_t maxBatchBytes = 0;
    /// The fired estimate's terms, in bytes - the RI-K counterpart of
    /// FockModeInfo's decomposition, so a reader adds them up and lands on
    /// `predictedBytes` instead of taking the total on trust. This one is the
    /// footprint's pair stores, screened task list and metric/eigen cluster.
    /// The blocked rung's slice and accumulation temporary are the only terms
    /// not named here - they are `predictedBytes` minus the sum.
    std::size_t structuralBytes = 0;
    std::size_t rootBytes = 0; ///< The retained metric-inverse root (nAux x nAux).
    /// ONE n^2 x nAux array (the raw tensor on the fast rung, the retained
    /// transform on the blocked one, where the fast rung's second copy is the
    /// difference between the rungs).
    std::size_t tensorBytes = 0;
    std::size_t occTransformBytes = 0; ///< BuildFock's n x nOcc x nAux half transform.
    std::size_t fockBytes = 0; ///< The rank-2 working class.
    std::size_t arenaBytes = 0; ///< The 3c batch arena at `maxBatchBytes`.
    std::size_t sliceFunctions = 0; ///< kBlocked: the shell range's auxiliary function width.
    std::size_t sliceCount = 0; ///< kBlocked: the number of auxiliary shell ranges built.
    /// The auxiliary-pair density screen this builder was created with
    /// (RiAuxScreenOptions). 0 = the screen is off, and a per-call
    /// RiAuxScreenStats sink then records `engaged == false` with a surviving
    /// set equal to the whole grid - the record is written either way, so a
    /// reader never has to infer the treatment from the numbers. A non-zero
    /// threshold is charged in `predictedBytes`.
    double screenThreshold = 0.0;
    /// The screen's retained Create-time stores (the orbital pair list, its
    /// Schwarz bounds, the auxiliary shell bounds and ranges) - 0 when the
    /// screen is off. The per-call surviving structure is charged separately,
    /// in `screenStructureBytes`.
    std::size_t screenStoreBytes = 0;
    /// The screen's per-call surviving structure at its WORST case (every cell
    /// surviving: one entry per (pair, aux shell) cell plus the shell
    /// offsets). The realized structure is the size the gate gives, reported
    /// per call by `RiAuxScreenStats`; the charge is the worst case, the same
    /// never-under form the task-list charge uses.
    std::size_t screenStructureBytes = 0;
    /// The batched contraction traversal's residency target this builder was
    /// created with (RiContractionBatchOptions::blockTargetBytes). **0 is the
    /// default and selects the three-call chain**; a non-zero value selects
    /// the batched traversal, which then runs on every DENSE call - the one
    /// combination `Create` refuses is that traversal beside an engaged
    /// screen (the screen's field above). The traversal's block width and
    /// count are Create-time consequences of this number and are NOT restated
    /// here: they are execution facts about a call, and the per-call
    /// `RiContractionStats` sink is their home.
    std::size_t contractionBlockBytes = 0;
};

/// The auxiliary-pair density screen's settings.
///
/// The screen is the density-weighted form of the cutoff the 3-center tensor
/// is BUILT at. `BuildScreenedRiTaskList` fills the (orbital pair x auxiliary
/// shell) grid at `SchwarzThreshold` and skips every cell below it, so the
/// dense tensor already carries exact zeros outside that grid - the cells the
/// contractions then multiply by nothing. That cutoff is BASIS-ONLY by
/// construction ("density factors (max_d over the block) are
/// per-iteration quantities, vacuous here: BuildRiTensor is basis-only,
/// called once at Create", ri_engine.cpp's 8b note). This screen is the
/// per-iteration half the same note names: the grid test gains the density
/// weight `max |rho|` over the pair block (`internal::BuildShellPairMaxDensity`
/// - the direct family's own density bound, reused verbatim rather than
/// re-derived) and runs at `threshold` instead of the Schwarz cutoff.
///
/// The gate is the direct family's product form on the same three factors it
/// uses everywhere (`fock_screen.hpp`'s QuartetDensityGate reads the density
/// max and the two Schwarz pair bounds): a cell survives iff
///   `densityWeight(pair) * Q_pair * Q_auxShell >= threshold`
/// with `Q_pair` the orbital pair's Schwarz bound and `Q_auxShell` the
/// auxiliary shell's (the two factors `BuildScreenedRiTaskList` already
/// multiplies). Drop iff strictly less, keep at equality - the legacy gate's
/// boundary convention. The quartet gate itself has no (pair, aux) instance:
/// its K arm is the sum of four CROSS-pair density maxima, and a 3-center cell
/// has no second pair to cross with, so what is reused is the bound and the
/// product form, not the quartet expression.
///
/// `threshold = 0` disables the screen: every cell survives (no bound product
/// is negative) and `BuildFock` runs the dense contractions verbatim.
/// The default is 0 because the VALUE is a policy question (the
/// screening-induced converged-energy move must stay at or below a tenth of
/// the measured auxiliary-fit error on the same fixture) and no number is
/// pre-registered; the preset's own `DensityThreshold` is the
/// proposed home of it, which is why this struct carries the value rather than
/// a preset enum - a caller writes the threshold it means and the record
/// states it, instead of a default the record cannot show.
/// \ingroup qcx-integrals
struct RiAuxScreenOptions {
    /// The screen's cutoff. 0 (the default) disables the screen entirely.
    double threshold = 0.0;
};

/// The per-call record of the auxiliary-pair density screen.
/// Written by `BuildFock` into a caller-supplied sink, never retained on the
/// builder: the surviving structure is rebuilt from the density the CALL
/// receives (the `fock_screen.hpp` rule - the max-density vector is never
/// cached across calls, because the incremental wrapper alternates full D and
/// Delta-D between them), so a per-call record is the only shape that cannot
/// describe a previous call's density.
/// \ingroup qcx-integrals
struct RiAuxScreenStats {
    /// Whether the screen ran on this call (`threshold > 0`).
    bool engaged = false;
    /// The cutoff this call screened at (0 when not engaged).
    double threshold = 0.0;
    /// The grid's cells: every canonical orbital pair x every auxiliary shell.
    std::size_t totalCells = 0;
    /// The cells the gate kept.
    std::size_t survivingCells = 0;
    /// The cells the gate dropped: `totalCells - survivingCells`, the cells
    /// whose (pair, aux shell) bound product fell below the threshold.
    std::size_t droppedCells = 0;
    /// The retained structure: one entry per surviving (pair, aux shell) cell,
    /// plus the auxiliary shell offsets. `structureBytes` is what the entry
    /// costs (8 B each) plus the offsets.
    std::size_t retainedEntries = 0;
    /// The retained structure's size in bytes: the entries' own 8 B each plus
    /// the auxiliary shell offsets.
    std::size_t structureBytes = 0;
    /// `droppedCells / totalCells` (0 when the grid is empty).
    double droppedFraction = 0.0;
};

/// The batched contraction traversal's settings (auxiliary-index batching for
/// cache).
///
/// What the option selects: `BuildBatchedRiContractions` (ri_occ_k.hpp) walks
/// the auxiliary index in BLOCKS and runs the three passes the dense chain
/// makes over the metric-transformed tensor inside one traversal per block -
/// the cache-tuned re-expression of the chain, whose second effect is
/// structural: the `n x nOcc x nAux` occ-transformed array the chain allocates
/// (the term `RiFullFockModeInfo::occTransformBytes` charges) is never
/// materialized. The arithmetic is the chain's: at a width covering the whole
/// auxiliary index the two agree **bit for bit** (deviation exactly 0, pinned
/// on both fixtures), and a narrower width regroups the Coulomb
/// accumulation only - MEASURED 2.22045e-15 ABSOLUTE on water/def2-SVP at
/// scale 3.76952 (5.9e-16 relative), inside the chain's own 1e-12 identity
/// tolerance. `blockTargetBytes` therefore moves no physics at any width; what
/// it moves is the position on a time/working-set tradeoff.
///
/// **A target of 0 - the default - selects the three-call chain**, and the
/// reason is the acceptance, which is UNMET and unclaimable. The target is
/// >= 20% faster than exact screened exchange at 600 BF, on a
/// paired-interleaved statistic; the parts probe bounds the fused
/// traversal at **~16% of the chain** on the fixture it ran on (n = 200 /
/// nAux = 1000 / nOcc = 30) and **~4% at a T1-shaped occupied fraction**,
/// because 84% of the chain is the occ transform plus the exchange
/// contraction, which the batching does not touch - that number is not
/// reachable by this optimization. The cost cell could not order its arms at
/// all (one arm moved by more than 20x between three interleaved runs on a
/// machine at 100% load), so **no ratio is claimed anywhere**. A traversal
/// whose benefit is unmeasured and whose narrow widths read slower does not
/// become the default on the strength of its own wiring: a caller that wants
/// it writes the target it means, and `ModeInfo` plus the per-call
/// `RiContractionStats` then state what ran.
/// \ingroup qcx-integrals
struct RiContractionBatchOptions {
    /// The batched traversal's transformed-slab residency target, in bytes:
    /// one auxiliary block's columns of B cost `n^2 * width * 8`, so this is
    /// the number that decides how many columns a block carries. 0 - the
    /// default - selects the three-call chain instead. A non-zero value is
    /// clamped to at least one column, so any positive target is legal, and
    /// `kRiContractionBlockBytes` is the target that entry point defaults to
    /// when a caller names none.
    std::size_t blockTargetBytes = 0;
};

/// The per-call record of the contraction traversal, written
/// into a caller-supplied sink by every `BuildFock` call that reaches a
/// contraction.
///
/// **It is written from the branch that RUNS, never from the option that
/// selected it** - so `batched` is a fact about the call rather than a
/// restatement of the builder's configuration, which is what makes it usable
/// as the evidence that the wiring is reached (a run record
/// that can silently differ from what ran is a defect). A builder created
/// without a target reports `batched == false` and a width and count of 0 on
/// every call; a builder created with one reports `batched == true` and the
/// partition `BuildBatchedRiContractions` actually used.
/// \ingroup qcx-integrals
struct RiContractionStats {
    /// Whether the batched traversal ran on this call.
    bool batched = false;
    /// The auxiliary block width the traversal ran at, in columns (0 when the
    /// chain ran, which has no block partition).
    std::size_t blockWidth = 0;
    /// The number of blocks the auxiliary index was partitioned into (0 when
    /// the chain ran). 1 means one block covered the whole auxiliary index,
    /// the partition under which the traversal is the chain's arithmetic in
    /// the chain's order.
    std::size_t blockCount = 0;
    /// The residency target in effect (RiContractionBatchOptions: 0 when the
    /// chain ran).
    std::size_t blockTargetBytes = 0;
    /// The seam cap the width was clamped against - the effective
    /// `RiEngineOptions::maxBatchBytes` of this builder, which the rung
    /// decision narrows when a workspace budget engaged.
    std::size_t maxBatchBytes = 0;
};

/// The two halves of the composed build, from ONE traversal of the auxiliary
/// index - the pair `BuildFock` fuses at its own composition line.
///
/// **Neither member carries the core Hamiltonian, and neither carries a factor
/// of two**: `coulomb` is J_RI(rho) and `exchange` is K_RI(rho) for the density
/// the call received, and the fused build composes them as
///   F = H + 2 * coulomb - exchange
/// with H the caller's. That accounting is STATED because both of the families
/// beside this one do something different and a reader will carry their habit
/// in: the RI-J link's split halves bake the factor of two into the Coulomb
/// half and a full H into the exchange half (`ri_engine.hpp` BuildCoulombOnly
/// returns 2 J_RI(rho) and BuildExchangeOnly returns H - K(rho)), while the
/// direct and composed-QFMM families put a full H copy into EACH half (the
/// driver's per-spin assemblies subtract one copy per channel). Folding H or
/// the factor of two in here moves an assembled Fock by a whole core
/// Hamiltonian or by an entire J.
///
/// The bare form is what an unrestricted caller needs: both halves are LINEAR
/// in the density, so one spin channel's Fock,
///   F_sigma = H + C(P_alpha) + C(P_beta) - K(P_sigma),   C = coulomb, K = exchange,
/// is assembled from TWO calls of the pair entry point at the two per-spin
/// densities, with no factor or H to correct for. A half that carried either
/// could not be summed across spins without the correction, which is why this
/// shape is what the per-spin leg wants rather than a wrapper over the fused
/// result.
/// \ingroup qcx-integrals
struct RiFullFockHalves {
    /// J_RI(rho), n x n: the Coulomb half the fused build scales by two.
    qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> coulomb;
    /// K_RI(rho), n x n: the exchange half the fused build subtracts.
    qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> exchange;
};

/// The composed full-RI Fock builder: RI-J and
/// occ-RI-K from one metric-transformed 3-center tensor, composed as
/// F = H + 2 J_RI - K_RI with no direct-exchange call anywhere on the path.
///
/// Create pays the whole auxiliary cost once: the screened (uv|P) 3-center
/// tensor, the (P|Q) metric, and the metric transform B. Each BuildFock call
/// then runs two dense products (J = B (B^T d)) plus the occ transform and the
/// exchange contraction - no integrals are evaluated per iteration.
///
/// Density convention: exactly RiJkFockBuilder's. The density argument is the
/// SPATIAL closed-shell density rho = D/2 (spin-summed D over 2), which is also
/// the convention the in-tree SCF instrument and the scf FockBuilderFn seam
/// both use.
///
/// The composition's error class is the AUXILIARY FIT's, not the screening
/// budget's: the fit is the single approximation
/// on this path, and it is not improved by tightening `accuracy`.
/// \ingroup qcx-integrals
class RiFullFockBuilder {
public:
    /// Prepares the builder: the screened 3-center tensor, the auxiliary
    /// metric, and the metric-transformed tensor B the two halves share.
    ///
    /// The peak of this call holds the raw 3-center tensor and its transform
    /// at once (two n^2 x nAux arrays); the raw tensor is released when Create
    /// returns, so the retained state is one such array plus the metric-class
    /// working set. That is the arithmetic - B replaces the
    /// tensor rather than joining it - and it is the class that decides
    /// admit-vs-refuse at the large tiers.
    /// \param molecule Molecule providing the atom coordinates (Bohr).
    /// \param basisSet Orbital basis (same contract as BuildRiTensor).
    /// \param auxBasisSet Auxiliary basis (same contract as BuildAuxMetric).
    /// \param coreHamiltonian H = T + V, host-canonical rank-2 tensor with
    /// shape {n, n}.
    /// \param options Build settings. `accuracy` and `maxBatchBytes` reach the
    /// Create-time 3-center build (its task-list screen and its batch cap);
    /// `metricFloorEpsilon` reaches the metric's floored eigen-inverse, the
    /// same rule and default as the shipped path. `workspaceBudget` selects the
    /// MEMORY LADDER's rung (RiFullFockRung): the fast rung is attempted
    /// first, and when its Create-time estimate does not fit the budget's
    /// remaining bytes the blocked rung is attempted, sized from what is left.
    /// The decision, its numbers and the rung that engaged are reported by
    /// ModeInfo(). A budget too small for either rung's minimum is refused with
    /// the ladder named. Two other fields are REFUSED rather than
    /// accepted-and-ignored, because accepting them would state a mode this
    /// builder does not have: `forceLightRung` (the per-iteration 3-center
    /// RECOMPUTE rung - the RI-J path's landed light rung, ri_engine.hpp; this
    /// builder's ladder stops one rung earlier, so the knob cannot be honoured),
    /// and a non-null `symmetryReduction` or `auxSymmetryReduction` (the
    /// point-group reduction: this path's 3-center tensor is counted but never
    /// reduced, so the fields would be accepted and silently ignored - the same
    /// silent-substitution defect class). Both refusals return
    /// kUnimplemented with the mode named. `useCudaGemm` is accepted and does
    /// not reach this class's contractions, which are host Eigen products - the
    /// same disclosure the field's own doc already makes for a CUDA-less build.
    /// \param screenOptions The auxiliary-pair density screen
    /// (RiAuxScreenOptions). A non-zero threshold makes Create retain the
    /// screen's stores - the orbital pair list, its Schwarz bounds, the
    /// auxiliary shell bounds and function ranges - and charges them in the
    /// rung record's `screenStoreBytes`/`screenStructureBytes`; a zero
    /// threshold retains nothing and every call runs the dense contractions.
    /// A non-zero threshold beside a non-zero \p batchOptions target is
    /// REFUSED - see that parameter.
    /// The retained bounds cost one `ComputeSchwarzBounds` sweep over each
    /// basis, the sweep the budget decision below already pays when it runs.
    /// \param batchOptions The batched contraction traversal
    /// (RiContractionBatchOptions). A zero target - the default - leaves the
    /// dense path on the three-call chain; a non-zero one makes it run
    /// `BuildBatchedRiContractions` in place of that chain at the width the
    /// target, the seam cap and the auxiliary extent decide. Create retains
    /// nothing for it (the width is a function of numbers it already has), and
    /// the choice and what then ran are reported by `ModeInfo()` and per call
    /// by `RiContractionStats`. **A non-zero target beside an engaged screen
    /// (a non-zero \p screenOptions threshold) is REFUSED with kUnimplemented
    /// and both options named**: the batched traversal takes the auxiliary
    /// index as its block partition and does not carry the screen's surviving
    /// (pair, aux shell) structure, so the two treatments are alternatives
    /// here, and accepting both would silently drop one.
    /// \returns The builder, or an Error (the refusals above; the
    /// screen-and-batch refusal; the ladder's refusal when neither rung fits
    /// the budget; the tensor, metric or transform entry points' own errors).
    static qcx::Result<RiFullFockBuilder> Create(
        const qcx::molecule::Molecule& molecule,
        const qcx::basisset::BasisSet& basisSet,
        const qcx::basisset::BasisSet& auxBasisSet,
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coreHamiltonian,
        const RiEngineOptions& options = {},
        const RiAuxScreenOptions& screenOptions = {},
        const RiContractionBatchOptions& batchOptions = {});

    /// Builds F = H + 2 J_RI(rho) - K_RI for the given closed-shell density
    /// and occupied block.
    /// \param density The SPATIAL closed-shell density rho = D/2, host-
    /// canonical rank-2 tensor with shape {n, n}. It must come from the same
    /// SCF iterate as \p occupiedOrbitals (the coherence contract in the file
    /// comment).
    /// \param occupiedOrbitals The occupied MO coefficient block C_occ, n x
    /// nOcc in the AO basis: whichever block satisfies
    /// rho = C_occ C_occ^T for the density above. The exchange half is
    /// invariant under an occupied-space rotation, so any such block is
    /// accepted - it is the density it describes that must match, not the
    /// basis choice within the occupied space.
    /// \param statsOut Optional per-call diagnostics sink. This builder
    /// evaluates NO quartets: no direct exchange is called, so the sink is
    /// WRITTEN with a default (all-zero) FockBuildStats rather than left
    /// untouched - the measured zero is the structural record of the absent
    /// direct-exchange half (the `qx`/`gx` rule), and
    /// writing it keeps a caller from reading a stale struct as this call's
    /// counts. Null (the default) skips the write.
    /// \param screenOut Optional per-call screen record
    /// (RiAuxScreenStats): the surviving/dropped cell counts and the retained
    /// structure's size, built from THIS call's density. Null (the default)
    /// skips the write, exactly as `statsOut` does; a builder created with a
    /// zero screen threshold writes a record whose `engaged` is false and
    /// whose counts are the whole grid.
    /// \param contractionOut Optional per-call contraction record
    /// (RiContractionStats): which traversal ran - the three-call chain or
    /// the batched one - and, for the batched one, the auxiliary block width
    /// and count. Null (the default) skips the write. It is written from the
    /// branch that ran, so it is the record of this call's execution and not a
    /// restatement of the builder's options: **a builder created with a
    /// batched target that routed this call through the chain would report
    /// `batched == false`**, which is the fact a reader needs and cannot infer
    /// from the numbers (the two paths agree to round-off at every width).
    ///
    /// The screen narrows the two contraction halves that read the 3-center
    /// tensor's nonzero structure: the Coulomb weights `u_P = sum_uv B^P_uv
    /// rho_uv` and the occupied transform `Bocc[(u,i),P] = sum_v B^P_uv C_vi`
    /// are accumulated over the surviving pair blocks only. The exchange
    /// contraction `K = sum_P Bocc_P Bocc_P^T` stays the dense rank-nOcc
    /// update: a dropped pair block zeroes a BLOCK ROW of `Bocc_P`, but the
    /// surviving u-rows of a dropped-block pair are still the whole AO range
    /// on any compact molecule (a localized function always has a
    /// significant partner), so restricting that product would cost a gather
    /// and a scatter to buy nothing. The consequence is stated rather than
    /// implied: the screen's ceiling on this path is the Coulomb half plus
    /// the occupied transform - half the per-iteration flops - and no more.
    /// \returns The Fock matrix as a rank-2 tensor, or an Error
    /// (kInvalidArgument for a density or orbital block whose shape does not
    /// match the basis the builder was created with, or for an empty
    /// occupied block).
    qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> BuildFock(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        const Eigen::MatrixXd& occupiedOrbitals,
        FockBuildStats* statsOut = nullptr,
        RiAuxScreenStats* screenOut = nullptr,
        RiContractionStats* contractionOut = nullptr) const;

    /// Builds the two halves of the composition - J_RI(rho) and K_RI(rho) -
    /// in ONE traversal of the auxiliary index, and returns them as the pair
    /// `BuildFock` fuses.
    ///
    /// **It is the same call `BuildFock` makes**: the parameters, the shape and
    /// coherence checks, the refusals Create already made, the three per-call
    /// sinks and the traversal branch are that entry point's, verbatim, and
    /// `BuildFock` is this result plus the composition line its own contract
    /// states. Distinguishing the two is therefore a caller's choice and never
    /// a difference in what was computed: for one density the pair and the
    /// fused Fock come from one traversal either way.
    ///
    /// The pair is FREE, and the reason is the builder's own structure: it
    /// holds the two halves as separate locals on every path - the dense
    /// chain's two products, the screened sweep's two contractions, the
    /// batched traversal's one report - and combines them at a single line.
    /// What is NOT free is calling this entry point TWICE at the same density
    /// to obtain the two halves one at a time: each call is a full traversal
    /// and each computes both halves, because the contraction path does not
    /// have a Coulomb-only mode. A caller that wants one density's two halves
    /// takes them from ONE call (the pair), and a caller that wants one half
    /// of one density takes the fused entry point and reads the half it needs.
    ///
    /// What the pair is FOR is the density it is called at: an unrestricted
    /// caller has one density per spin and needs each spin's exchange at THAT
    /// spin's density, while the Coulomb half of the Fock it assembles is the
    /// spin-summed one. Both halves being linear in the density, the Coulomb
    /// halves of the two per-spin calls already sum to the spin-summed one
    /// (C(P_a) + C(P_b) = C(D_total), the closed-shell 2 J_RI(rho) of the
    /// convention) - so two pair calls serve the whole per-spin assembly, and
    /// every one of them pairs a density with an occupied block describing the
    /// SAME spin. That last property is the reason to prefer this entry point
    /// on the unrestricted leg rather than the fused one at the half-summed
    /// density: the fused route would hand the builder a sum over both spins
    /// beside one spin's block, which is the mismatch the coherence contract
    /// in the file comment warns a caller away from.
    /// \param density The SPATIAL density rho (a SPIN density P_sigma on the
    /// unrestricted leg), host-canonical rank-2 tensor with shape {n, n}, the
    /// same convention and the same coherence requirement as BuildFock's.
    /// \param occupiedOrbitals The occupied block C_occ the exchange half is
    /// contracted against, n x nOcc: whichever block satisfies
    /// rho = C_occ C_occ^T for the density above. The exchange half reads it
    /// through that product alone (see the file comment), so a per-spin block
    /// derived from a per-spin density is the same object as the block the
    /// loop's own diagonalization produced.
    /// \param statsOut Optional per-call diagnostics sink, written exactly as
    /// BuildFock writes it - the measured all-zero FockBuildStats of a builder
    /// that calls no quartet kernel. Null skips the write.
    /// \param screenOut Optional per-call screen record (RiAuxScreenStats),
    /// built from THIS call's density as BuildFock's is, and reporting the
    /// same surviving structure: the screen is computed before the halves are
    /// and cannot differ between the two entry points.
    /// \param contractionOut Optional per-call contraction record
    /// (RiContractionStats): the traversal this call ran, as BuildFock reports
    /// it - the pair and the fused build reach the same branch for the same
    /// options.
    /// \returns The pair (RiFullFockHalves), or an Error - the same errors from
    /// the same checks BuildFock's contract lists (a density or block whose
    /// shape does not match the basis, an empty occupied block, and the
    /// traversal's own failures).
    qcx::Result<RiFullFockHalves> BuildFockHalves(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        const Eigen::MatrixXd& occupiedOrbitals,
        FockBuildStats* statsOut = nullptr,
        RiAuxScreenStats* screenOut = nullptr,
        RiContractionStats* contractionOut = nullptr) const;

    /// The metric-transformed 3-center tensor B = I s this builder works
    /// from: rows u*n + v, columns P, in the contraction layout the raw
    /// tensor uses. Exposed for the composition's tests and for a caller that
    /// wants to verify or persist the one array both halves consume; it is
    /// NOT a second tensor layout and nothing else is derived from it here.
    /// \returns The n^2 x nAux matrix (column-major).
    const Eigen::MatrixXd& MetricTransformedTensor() const noexcept;

    /// The Create-time rung decision (RiFullFockModeInfo): which rung of the
    /// memory ladder engaged, and the numbers the decision read. `engaged` is
    /// false for a builder created without a workspace budget - that path takes
    /// the fast rung with no budget consulted, and recording it as a decision
    /// would state a budget read that never happened.
    /// \returns The mode record.
    const RiFullFockModeInfo& ModeInfo() const noexcept;

    /// The Create-time term counters of this builder (the class
    /// comment's contract): the 3-center tensor pass's `x`/`p3`/`g3` totals
    /// over the screened task list, plus the metric occurrence's `x`/`g3`
    /// contributions, as `BuildRiTensorChunk` and `BuildAuxMetric` accumulate
    /// them. They cover the work a point-group reduction would shrink, which is
    /// why the reduction is refused here rather than accepted: any later
    /// reduction work is aimed by these numbers, not by an estimate.
    ///
    /// **They are exact on both rungs, and equal between them.** The blocked
    /// rung's chunks partition ONE tensor-pass occurrence, so the builder
    /// threads its dedup state through every chunk call (`RiScreenedPassState`)
    /// and an orbital pair is counted once per pass instead of once per chunk.
    /// On the route's standalone per-call stamps the blocked rung's `p3` read
    /// each pair once per chunk: MEASURED on the H2O/STO-3G +
    /// `def2-universal-jkfit` fixture, `p3` 15 (fast, one call) against 555
    /// (blocked, 37 calls) - the deviation
    /// `RiFullFockTest.TheBlockedRungReproducesTheFastRungAndChargesItsBudget`
    /// pinned with an inequality while it stood, and pins as an equality now.
    /// `x` (the distinct AUXILIARY shells, which the chunk ranges partition -
    /// every aux shell belongs to exactly one chunk) and `g3` (a sum over the
    /// tasks) are additive from the sink alone, and always were. A counter that
    /// describes the work must not move with the memory partition: the chunked
    /// route's documented summation contract (ri_engine.hpp
    /// BuildRiTensorChunk, "a chunked caller sums its chunks'") therefore holds
    /// for all three terms.
    /// \returns The counters; `qx`/`gx` are always zero (no quartet kernel).
    const RiTermCounters& TermCounters() const noexcept;

private:
    // The implementation state lives in the .cpp (Eigen stays
    // implementation-only).
    struct State;
    explicit RiFullFockBuilder(std::shared_ptr<const State> state);

    /// The shared body of the two public entry points: the shape and coherence
    /// checks, the per-call sinks, the traversal branch and the two
    /// contractions, in the working layout. `BuildFock` composes its
    /// result at one line and `BuildFockHalves` converts it, so the pair and
    /// the fused build cannot drift about what they computed - a second copy
    /// of this body is the defect this declaration exists to prevent, and the
    /// pair's zero-cost claim rests on there being only one.
    /// \param density The SPATIAL density, shape {n, n}.
    /// \param occupiedOrbitals The occupied block, n x nOcc.
    /// \param statsOut The per-call diagnostics sink, written on every call.
    /// \param screenOut The per-call screen record, or null for no write.
    /// \param contractionOut The per-call contraction record, or null for no
    /// write.
    /// \returns The Coulomb half first and the exchange half second, or the
    /// error the call refused with.
    qcx::Result<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> BuildHalves(
        const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& density,
        const Eigen::MatrixXd& occupiedOrbitals,
        FockBuildStats* statsOut,
        RiAuxScreenStats* screenOut,
        RiContractionStats* contractionOut) const;

    std::shared_ptr<const State> _state;
};

} // namespace qcx::integrals
