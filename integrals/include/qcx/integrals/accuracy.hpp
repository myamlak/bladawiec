#pragma once

/// \file
/// The unified accuracy presets and the per-technique thresholds of the MD
/// engine.

namespace qcx::integrals {

/// The one user-facing accuracy knob.
/// \ingroup qcx-integrals
enum class AccuracyPreset {
    kLoose, ///< J/K energy target 1e-6 Eh.
    kNormal, ///< J/K energy target 1e-10 Eh (the default).
    kTight, ///< J/K energy target 1e-12 Eh.
};

// The exhaustiveness guard around this file's preset mappers - the ONE place
// a user-named accuracy preset (the [method].accuracy word) becomes the
// numbers the engines run on. The diagnostics have to be promoted here to be
// worth anything: MSVC emits C4062/C4061 for an unhandled enumerator at
// neither /W3 nor /W4 (both are off-by-default), GCC emits nothing without
// -Wall, and Clang's -Wswitch is a warning nobody reads. C4061 and
// -Wswitch-enum are the variants that also catch a `default:` arm added to
// silence the check. The region spans exactly the nine mappers below and
// nothing else; every other switch in every translation unit that includes
// this header keeps the project's default diagnostic settings.
//
// Every mapper here is exhaustive BY CONSTRUCTION - no `default:` arm - so a
// slice that adds an AccuracyPreset enumerator cannot leave it unplaced; the
// compiler stops it. Without the guard it could: each mapper's tail after the
// switch is the runtime half, reached only by a value no enumerator names (a
// programmatic caller's out-of-range cast), and a mapper that FORGOT the new
// enumerator would fall into that same tail and answer with kNormal's number
// - so a user asking for a stricter or looser preset would silently run at
// normal accuracy, with nothing in the record able to tell the two apart.
// That is the silent substitution this contract forbids; the guard makes it
// a build failure instead. The tails stay documented rather than refusing by
// value (the BuilderConsumesAux posture, io/src/parse_input.cpp): a
// constexpr numeric mapper has no error channel, and its tail is
// unreachable for every value the enum names.
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(error : 4061)
#pragma warning(error : 4062)
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic error "-Wswitch"
#pragma clang diagnostic error "-Wswitch-enum"
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

/// The Schwarz screening threshold of a preset: shell quartets with
/// Q_ij * Q_kl below it are dropped. The threshold tracks the preset's J/K
/// energy target, and the mapping was validated on the pins by the committed
/// preset sweep (2026-08-23).
/// \param preset The accuracy preset.
/// \returns The Schwarz product threshold.
/// \ingroup qcx-integrals
inline constexpr double SchwarzThreshold(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 1e-8;
    case AccuracyPreset::kNormal:
        return 1e-10;
    case AccuracyPreset::kTight:
        return 1e-12;
    }

    return 1e-10;
}

/// The density-weighted screening threshold of a preset: density
/// products |D_element| * Q_ij * Q_kl below it are dropped by the direct
/// Fock builders. The values mirror SchwarzThreshold's
/// J/K energy budgets (kLoose 1e-6, kNormal 1e-10, kTight 1e-12 Eh - the
/// PRESET targets, not the delivered errors: the sweep measured 6.6e-9 /
/// 6.3e-12 / ~1e-14 energy error on the fixtures, each more than an order
/// of magnitude inside its budget). Validated on the pins by the committed
/// preset sweep (2026-08-23).
/// \param preset The accuracy preset.
/// \returns The density-weighted threshold.
/// \ingroup qcx-integrals
inline constexpr double DensityThreshold(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 1e-8;
    case AccuracyPreset::kNormal:
        return 1e-10;
    case AccuracyPreset::kTight:
        return 1e-12;
    }

    return 1e-10;
}

/// The certified-mixed-precision gate of a preset: a shell quartet
/// may use the certified fp32 lane when its a-priori density-weighted bound
/// C_class * eps_fp32 * Q_ij * Q_kl * |D_element| stays at or below this
/// threshold. 0 disables the fp32 lane entirely - kTight keeps the lane off
/// so the strict preset reproduces the fp64 path (the pins depend on it).
/// The kLoose/kNormal gates were tightened by the committed preset sweep:
/// at the original 1e-6/1e-8 the lane
/// delivered 1.5e-6 / 6.6e-9 of ENERGY error on the sweep fixtures, over
/// the preset budgets (1e-6 / 1e-10) - the current values land every preset
/// more than an order of magnitude inside its budget.
/// \param preset The accuracy preset.
/// \returns The per-element mixed-precision budget (0 = fp32 off).
/// \ingroup qcx-integrals
inline constexpr double MixedPrecisionThreshold(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 1e-8;
    case AccuracyPreset::kNormal:
        return 1e-10;
    case AccuracyPreset::kTight:
        return 0.0;
    }

    return 1e-10;
}

/// The QFMM well-separatedness parameter of a preset (the fast-multipole
/// method; recalibrated by the committed preset-ladder sweep): smaller theta
/// means a stricter near/far classification - more pairs computed exactly,
/// higher accuracy, more cost; theta -> 0 degenerates to "everything is
/// near field" - the near-field-only path, bit-exact when the near path's
/// screening thresholds match the comparison's (the (kNormal, C12)
/// vacuous-tooth 0.0 pins that case). The recalibrated
/// ladder {0.45, 0.3, 0.0} for {kLoose, kNormal, kTight} was measured at
/// the per-preset extents {1e-6, 1e-8, 1e-10} on the C12/C24 alkane chains
/// - the earlier {1.05, 0.85, 0.7} held only at the all-tau = 1e-10
/// extent, and the extent reclassifies near/far pairs, so those values
/// are not transferable (the 567-cell committed grid is
/// benchmarks/data/qfmm_ladder_sweep_full.csv). kLoose (0.45, 5) is the
/// cheapest common in-budget cell (reduced slack on C12 - measured 6.2x
/// inside the budget, which is not widened); kNormal (0.3, 5) is pinned by
/// C24 (the C12 tooth is vacuous there - all-near, bit-exact; the
/// far-alive pin uses the next-larger fixture); kTight 0.0 is
/// the degenerate gate - the near-field-only path, the designed
/// converge-to-direct fallback. At the ORIGINAL
/// kTight budget (1e-9) no (theta, L) reached it at tau 1e-10 in the
/// committed kNormal-screened-direct comparison, and that budget has since
/// been re-derived to 1e-8 precisely because that comparison's ground truth
/// screens coarser (1e-10) than the kTight near path (1e-12) - the
/// measured residual ~3.33e-9 C12 / ~6.1-7.5e-9 C24 was the REFERENCE's own
/// dropped-quartet mass, so the tight rung MEETS the re-derived 1e-8 budget
/// (QfmmBudgetForPreset). The rung stays theta -> 0 on its own design
/// ground - zero multipole benefit at kTight - NOT on unreachability.
/// Consequence recorded for the accuracy-ladder owner: at 1e-8 the tight
/// end now has LIVE in-budget cells on the committed sweep grid (C12
/// (0.7, 7) 3.19e-9 with 282 far pairs; C24 (0.3, 4) 6.13e-9 with 107), so
/// the "no vacuous ladders" acceptance clause is satisfiable at kTight only
/// by moving the rung - a rung change, not part of the budget
/// re-derivation.
/// \param preset The accuracy preset.
/// \returns The QFMM well-separatedness parameter.
/// \ingroup qcx-integrals
inline constexpr double ThetaForPreset(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 0.45;
    case AccuracyPreset::kNormal:
        return 0.3;
    case AccuracyPreset::kTight:
        return 0.0;
    }

    return 0.3;
}

/// The QFMM multipole expansion order of a preset (the fast-multipole
/// method; recalibrated by the committed sweep): higher L_mult means a more
/// accurate far field at more cost per interaction. {5, 5, 0} for
/// {kLoose, kNormal, kTight} - the ladder winners: kLoose and kNormal need
/// L = 5 (kLoose C24 at L = 4 overflows the 1e-5 budget, 4.54e-5 - the
/// order-cap evidence); kTight's 0 rides the degenerate theta-0
/// rung, where the multipole machinery never runs.
/// \param preset The accuracy preset.
/// \returns The QFMM multipole expansion order L_mult.
/// \ingroup qcx-integrals
inline constexpr int LMultForPreset(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 5;
    case AccuracyPreset::kNormal:
        return 5;
    case AccuracyPreset::kTight:
        return 0;
    }

    return 5;
}

/// The QFMM pair-extent density cutoff of a preset: the tau of
/// ComputeExtent's Gaussian falloff
/// radius sqrt(-ln(tau) / p). This is the GEOMETRIC "how big is this
/// pair's footprint" cutoff of the octree boxes - deliberately distinct
/// from the preset's Schwarz/density screening thresholds, which answer a
/// different question ("is this quartet worth computing"); conflating the
/// two is a known trap, and this ladder keeps them separate by mechanism.
/// The coupling is the documented tail-charge argument: the pair's charge
/// outside its box is bounded by the Gaussian tail integral at tau, and
/// that missed charge lands in the far-field expansion as an error term at
/// the preset's own budget scale - so the extent tightens (smaller boxes,
/// more far-field pairs) as the preset loosens. kTight = 1e-10 is the
/// skeleton's original kQfmmExtentThreshold (the skeleton's committed
/// value, kept as the tight rung); kLoose = 1e-6 and kNormal = 1e-8 shrink
/// the boxes (a looser preset admits a larger missed-charge tail at its
/// own coarser budget).
/// \param preset The accuracy preset.
/// \returns The QFMM pair-extent density cutoff tau.
/// \ingroup qcx-integrals
inline constexpr double QfmmExtentForPreset(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 1e-6;
    case AccuracyPreset::kNormal:
        return 1e-8;
    case AccuracyPreset::kTight:
        return 1e-10;
    }

    return 1e-8;
}

/// The QFMM far-field error budget of a preset (the QFMM budgets
/// {1e-5, 1e-7, 1e-8} of the committed
/// ladder sweep, benchmarks/data/qfmm_ladder_sweep_full.csv, the kTight
/// rung re-derived 1e-9 -> 1e-8 on 2026-09-13): the
/// per-interaction multipole-truncation budget of the adaptive-order
/// selector is this budget divided by the far-field pair count. The
/// selector only LOWERS an interaction's order where the geometric
/// bound (source-radius / distance)^(L+1) proves the order sufficient at
/// the per-interaction share of this budget - the preset ladder stays the
/// accuracy contract (the per-preset fixed L_mult of LMultForPreset remains
/// the order cap).
///
/// kTight's rung is the degenerate theta-0 gate, where the multipole
/// machinery never runs (FarFieldPairCount() is 0 by construction), so this
/// value selects nothing at kTight - it is the recorded contract the
/// acceptance rows compare against. WHY IT MOVED: the 1e-9 it replaced was
/// unreachable in the committed comparison because the comparison's own
/// ground truth is screened coarser than the thing it judges.
/// BuildDirectFock (qfmm_fixture.hpp) takes default FockBuildOptions, whose
/// accuracy is kNormal (fock_build.hpp:375) and so drops quartets at
/// SchwarzThreshold(kNormal) = 1e-10, while the kTight rung's near path
/// screens at SchwarzThreshold(kTight) = 1e-12 - a strict superset - so the
/// measured tight-end residual IS the reference's own dropped-quartet mass,
/// not a QFMM error. Measured on this tree 2026-09-13: the SAME theta-0 /
/// L-0 rung reads 3.33386e-9 against the kNormal-screened reference but
/// 9.83143e-13 against a kTight-screened one (the equal-preset protocol,
/// qfmm_hf_build_test.cpp) - tightening the REFERENCE 100x drops the
/// residual ~3400x, which no QFMM-side error could do. The sweep agrees at
/// the tight end: all 36 C12 far-dead cells read the identical
/// 3.3338633737e-9 (theta-independent, the signature of a reference-side
/// floor), and on C24H50 - where every cell is far-alive at tau 1e-10 -
/// the best cells are 6.1285072515e-9 at (0.3, 4) and 7.4817334393e-9 at
/// (0.3, 8). SCOPE of that agreement, stated because the contract is a
/// per-rung acceptance and NOT a whole-grid claim: no cell of either chain
/// at tau 1e-10 reaches 1e-9 (0 of 63 on each - the old budget was
/// genuinely unreachable), but the re-derived 1e-8 is met by the TIGHT end
/// only - 23 of the 63 C12 cells and 54 of the 63 C24 cells at this tau
/// still sit above 1e-8, in the loose theta/L corner where a coarse
/// expansion is the designed behaviour. C24H50 also has no far-dead cell
/// on the sweep grid (its theta rungs stop at 0.3, so the committed theta
/// 0.0 rung is off-grid), so its engaged theta-0 value is not directly
/// measured there; the re-derivation rests on the C12 direct measurement
/// plus C24's best live cells, and 1e-8 holds with 1.63x slack at the
/// tightest C24 cell measured.
/// \param preset The accuracy preset.
/// \returns The QFMM far-field error budget.
/// \ingroup qcx-integrals
inline constexpr double QfmmBudgetForPreset(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 1e-5;
    case AccuracyPreset::kNormal:
        return 1e-7;
    case AccuracyPreset::kTight:
        return 1e-8;
    }

    return 1e-7;
}

/// The per-atom exchange-error budget of a preset on an APPROXIMATED-exchange
/// path: the acceptance bars
/// {0.5, 0.02, 0.005} kcal/mol per atom for {kLoose, kNormal, kTight},
/// carried here in Eh per atom so a measurement is compared against the same
/// table it is recorded in.
///
/// **What this budget bars, and what it does not.** It is the SCREENING
/// CO-TERM's budget. On an approximated-exchange path the preset governs the
/// screening truncation, while the run's error against exact K is the error
/// of the AUXILIARY FIT: tightening the preset
/// does not improve that fit, so the fit's measured per-atom error is the
/// number to read against these bars and NOT against a preset-invariant
/// absolute. A mapper that read as a whole-path accuracy budget would promise
/// exactly the improvement the auxiliary fit rules out - the misreading this
/// distinction exists to forbid.
///
/// **The screen-threshold arm of this mapping states no number of its own.**
/// The thresholds an approximated-exchange run screens at ARE the preset's,
/// read from their own homes: the (uv|P) task-grid screen is
/// `SchwarzThreshold` (the cutoff `BuildScreenedRiTaskList` fills at) and the
/// auxiliary-pair density screen is `DensityThreshold` once the
/// approximated-exchange path screens its auxiliary pairs. This file
/// deliberately carries no second table for them: the
/// committed preset sweep's pins and the gate-tolerance check both compare
/// against the mapped values, and a copy is the drift the one-home rule
/// prevents.
///
/// The entry below is a per-atom ENERGY bar, not a convergence gate: it
/// stages no (energy, density) tolerance pair, so the convergence ladder the
/// gate-tolerance check enforces is not engaged by it. An SCF driven at this
/// preset still converges at the preset's own gate.
/// \param preset The accuracy preset.
/// \returns The per-atom exchange-energy budget in Eh.
/// \ingroup qcx-integrals
inline constexpr double RiExchangeErrorBudgetPerAtom(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
        return 7.97e-4;
    case AccuracyPreset::kNormal:
        return 3.19e-5;
    case AccuracyPreset::kTight:
        return 7.97e-6;
    }

    return 3.19e-5;
}

/// The MEASURED per-atom exchange error of the approximated-exchange path and
/// the fixture it was measured on: the counterpart of
/// `RiExchangeErrorBudgetPerAtom`, which is the bar. The measurement is the
/// approximated-exchange accuracy cell
/// (`integrals/tests/ri_jk_validation_test.cpp`; the two-cell table and the
/// ratios are in the run-schema document, not restated here).
///
/// **The measurement is the PATH's, never a run's.** Nothing on the run path
/// can produce this number: it is the deviation of the fitted exchange from
/// exact screened exchange at the same preset, and evaluating exact K is the
/// cost the path exists to avoid. The builder states the same limit from its
/// own side - no auxiliary-quality check is performed there and none can be
/// (`ri_full_fock.hpp:34`). A run therefore DISCLOSES this characterized error
/// beside its own preset's bar rather than a number of its own, and its record
/// NAMES the fixture so the number can never be read as a claim about the run
/// that printed it (`RunExchangeError`).
///
/// **Both cells, and why the error is no preset's.** Measured with
/// `def2-universal-jkfit` on both sides and the same exact screened
/// comparator, at every preset. The cell records the fixture's TOTAL exchange
/// move and its per-atom reading; the TOTAL of each is `h2o_sto3g` 3.52432e-4 Eh
/// and `water_def2svp` 8.51135e-5 Eh, and each fixture is water, so the per-atom
/// values this struct carries are those totals over three atoms -
/// 1.174773e-4 and 2.837117e-5 Eh/atom, the cell's own 0.0737 and 0.0178
/// kcal/mol/atom at 627.5095 kcal/mol per Eh. **The distinction is stated
/// because reading the total as the per-atom number is a factor of three**: the
/// bars are per atom and the recorded ratios are per atom, so a record
/// that paired a total with a per-atom bar would understate the miss by 3x.
/// The decisive reading is the PRESET-INVARIANCE - `h2o_sto3g` reads the
/// bit-identical total 3.52432e-4 at all three presets, and `water_def2svp`
/// moves 3.6e-8 Eh (0.04% of itself) across the whole range - because the
/// auxiliary FIT sets the error while the preset governs the screening co-term
/// only. One achieved value is therefore read
/// against every preset's bar unchanged, and 3 of the 6 (fixture x preset)
/// cells meet the bar they are read against: 1.174773e-4 is 0.147x the kLoose
/// bar, 3.68x kNormal and 14.7x kTight; 2.837117e-5 is 0.0356x, 0.890x and
/// 3.56x the same three.
/// \ingroup qcx-integrals
struct RiExchangeErrorMeasurement {
    /// The fixture label the accuracy cell recorded the measurement under - the
    /// cell's own label vocabulary, so the two spellings cannot drift.
    const char* fixture = "";
    /// The auxiliary fit BOTH sides of the measurement used.
    const char* auxBasis = "";
    /// The achieved PER-ATOM exchange error, Eh. The cell's totals over the
    /// three atoms are in the struct's own note; this member is never the total.
    double perAtomEh = 0.0;
};

/// The `h2o_sto3g` cell: 1.174773e-4 Eh/atom = 0.0737 kcal/mol/atom (the
/// recorded total 3.52432e-4 Eh over three atoms), and the LARGER of the two
/// characterized cells.
/// \ingroup qcx-integrals
inline constexpr RiExchangeErrorMeasurement kRiExchangeErrorH2oSto3g{
    "h2o_sto3g", "def2-universal-jkfit", 1.174773e-4};

/// The `water_def2svp` cell: 2.837117e-5 Eh/atom = 0.0178 kcal/mol/atom (the
/// recorded total 8.51135e-5 Eh over three atoms).
/// \ingroup qcx-integrals
inline constexpr RiExchangeErrorMeasurement kRiExchangeErrorWaterDef2Svp{
    "water_def2svp", "def2-universal-jkfit", 2.837117e-5};

/// The LARGEST achieved per-atom error among the characterized cells: the value
/// an approximated-exchange run discloses beside its preset's bar.
///
/// Largest, computed rather than asserted, and not "the" error: each number is a
/// measurement on a fixture, and a fixture whose error is smaller does not make
/// the path's larger one go away. **It is not a bound.** It is the worst of the
/// cells measured so far, and a run on an uncharacterized fixture may sit on
/// either side of it - which is exactly why the record ships the fixture name
/// beside the value instead of presenting it as the run's own error.
/// \returns The worst characterized per-atom error with its fixture and fit.
/// \ingroup qcx-integrals
inline constexpr RiExchangeErrorMeasurement RiExchangeWorstMeasuredPerAtomError() noexcept {
    return kRiExchangeErrorH2oSto3g.perAtomEh >= kRiExchangeErrorWaterDef2Svp.perAtomEh
               ? kRiExchangeErrorH2oSto3g
               : kRiExchangeErrorWaterDef2Svp;
}

/// Whether an approximated-exchange run at this preset requires a
/// JK-optimized auxiliary fit: **true at every preset, and that invariance is
/// the mapping's content** (the auxiliary-quality arm).
///
/// The auxiliary quality is not a preset knob, and the reason is the same one
/// `RiExchangeErrorBudgetPerAtom` states from the other side: the fit is the
/// single approximation on the path, the preset governs the screening co-term
/// only, so no preset can be served by a
/// coarser fit. Marking it per preset rather than leaving it implied is what
/// stops the two knobs being conflated - a `kTight` request is not a request
/// for a better fit, and a `kLoose` request does not license a J-only one
/// (which the auxiliary-quality rule refuses outright, `aux_basis.hpp`
/// IsJkOptimizedAux).
/// \param preset The accuracy preset.
/// \returns True at every preset; the mapper exists to state the invariance.
/// \ingroup qcx-integrals
inline constexpr bool RiExchangeRequiresJkFit(AccuracyPreset preset) noexcept {
    switch (preset)
    {
    case AccuracyPreset::kLoose:
    case AccuracyPreset::kNormal:
    case AccuracyPreset::kTight:
        return true;
    }

    return true;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

} // namespace qcx::integrals
