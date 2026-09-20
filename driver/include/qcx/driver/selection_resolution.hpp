#pragma once

/// \file
/// The builder resolution (the auto-selection rule,
/// under the owner's ruling of 2026-09-13 ~20:35 - the size LADDER that
/// replaced the lean-direct flip, 2026-09-08): turns the parsed
/// input into the builder the run wires, plus the selection record the
/// driver reports in resources_resolved.
///
/// The no-key ladder - what a run wires when the input names no
/// [method].fock_builder - is by SIZE:
/// | nBasis | tier |
/// |---|---|
/// | <= kLeanTierMaxBasisFunctions (1000) | the direct family's lean (Schwarz-only) member |
/// | 1000 < n <= kRiJTierMaxBasisFunctions (2000) | ri_j_link |
/// | n > 2000 | qfmm |
///
/// The 1000 boundary is the existing lean ceiling (the lean boundary,
/// carried over unchanged). The 2000 boundary is a CHOSEN value with NO
/// measurement behind it: the qfmm crossover - where a linear-scaling
/// method's advantage actually begins - is unmeasured, so 2000 must never
/// be read as a derived number. `direct` (the budgeted machinery) and both
/// disk routes (RI-J disk, full/cached ERI) are explicit opt-ins. The
/// model-driven auto-selection stays retired from the default path (the
/// estimator, selection_heuristic.hpp, stays untouched and testable and
/// may return only behind measured crossover data).
///
/// THE ONE EXCEPTION, and what it is keyed on (the owner's ruling of
/// 2026-09-13, refining the same day's ladder: "all should have the same
/// defaults (RHF/UHF/RKS/UKS), lean, ri-j and so on, the only exception are
/// non-hybrid KS, then put ri-j before lean"). The ladder is the SAME for
/// every method - RHF, UHF, RKS and UKS share one order, with no
/// method-specific variation - and the single exception is a NON-HYBRID
/// (pure) Kohn-Sham run, which promotes ri_j_link ABOVE the lean member:
/// a non-hybrid functional needs J and no K, so the RI-J path is the cheap
/// one there while the lean member's screened direct build would do work the
/// method does not need. Hybrid Kohn-Sham (B3LYP, PBE0) keeps the shared
/// order because it needs K. The key is the functional's CHARACTER, never
/// the method: keying on "is it Kohn-Sham" would silently mis-order every
/// hybrid DFT run. The caller states the fact (SelectionResolutionInput::
/// nonHybridKohnSham) because resolving a functional name is the registry's
/// job, not this module's.
///
/// The exception moves the ORDER only. The disclosure rule still governs: whichever tier
/// runs, the record's reasoning says so, and a run resolved against the
/// two-tier order carries that ordering in its record even when the promoted
/// tier turns out to be unwirable.
///
/// The ladder demotes rather than refusing (the disclosure rule demote-with-
/// disclosure arm): a tier this run cannot WIRE is skipped for the tier
/// below, the reasoning states which tier the size named and why it could
/// not run, and the warning carries the same divergence to stderr. The
/// unavailable cases are the caller's to state (SelectionResolutionInput):
/// an auxiliary basis the wiring cannot resolve for ri_j_link (since the
/// The aux-auto rule (owner's ruling 2026-09-13) is a quality tier that always
/// produces a default, so this reason is no longer reachable through an
/// unmatched ORBITAL NAME - it survives for the degenerate cases, and the
/// ladder keeps the arm so a future rule that refuses again is demoted
/// rather than run), and the v1 wiring has no ri_j link arm for UHF or for
/// the Kohn-Sham lanes, and no qfmm arm for the Kohn-Sham lanes. The
/// ladder's floor - the direct family's lean member - is wired for every
/// method (RHF, both unrestricted legs through the unrestricted seam, and
/// Kohn-Sham through the same half-mode flags), so the walk always
/// terminates and nothing refuses here.
///
/// An explicit [method].fock_builder wins verbatim (the ladder
/// governs the ABSENT case only) and is never demoted - an explicit
/// ri_j_link without an aux keeps the wiring's own refusal by name. The
/// The device-less fallback GPU auto-fallback survives, retargeted: an explicit "gpu" with no
/// device falls back to the LADDER's pick with a loud warning instead of
/// failing the run. The explicit "lean" spelling is a within-family
/// request, not a family word (RunMethodInput::leanDirect): it resolves to
/// the direct family's lean member like an absent key at or below the lean
/// boundary does, and only the reasoning carries it - above that boundary
/// the flag is what makes the member reachable at all.
///
/// Pure and total: no probing, no I/O - the caller probes the topology and
/// states the tiers' runnability once at run start, and copies the record
/// into the io schema (result_json.hpp RunSelection). Nothing refuses here:
/// every memory admission fires downstream of the resolution - the wiring's
/// own gates at the latest, and on the UHF and RHF paths the lean member's
/// own Create-time envelope one step earlier, at the driver's pre-gate setup
/// admission (CheckPreGateSetupAdmission), BEFORE the setup ramp that used
/// to run ahead of every gate).

#include "qcx/backend/topology_profile.hpp"
#include "qcx/error.hpp"
#include "qcx/io/run_input.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace qcx::driver {

/// The lean tier's ceiling (the owner's ruling 2026-09-13): the no-key
/// ladder's bottom boundary, nBasis <= 1000 - the existing lean
/// ceiling, carried over unchanged. It has history behind it (the lean
/// rule was decided against the measured routes at this boundary); unlike
/// kRiJTierMaxBasisFunctions below, it is the boundary the ladder started
/// from rather than a value chosen for the new shape.
/// \ingroup qcx-driver
inline constexpr std::size_t kLeanTierMaxBasisFunctions = 1000;

/// The ri_j_link tier's ceiling (the owner's ruling 2026-09-13): the no-key
/// ladder picks ri_j_link for 1000 < nBasis <= 2000 and qfmm above.
///
/// 2000 is a CHOSEN value with NO MEASUREMENT behind it - nothing measured
/// says where qfmm's crossover begins, and the owner's ruling states the
/// value as chosen ("the cutoffs are lean <= 1000, ri_j_link to 2000, qfmm
/// above", with both cutoffs labelled unmeasured). A reader must not be
/// able to mistake it for a derived number: no measurement, no benchmark
/// and no model produced it.
/// \ingroup qcx-driver
inline constexpr std::size_t kRiJTierMaxBasisFunctions = 2000;

/// The resolution's output: the effective builder and the selection
/// record the driver copies into resources_resolved (the resolution rule -
/// the pick, its reasoning, and the override/fallback flags).
/// \ingroup qcx-driver
struct ResolvedBuilderSelection {
    qcx::io::BuilderKind kind = qcx::io::BuilderKind::kDirect; ///< The builder the run wires.
    /// The pick the record reports: the builder the run wires WITHOUT an
    /// explicit fock_builder - the no-key ladder's outcome for this run
    /// (never the retired model's pick). It is the ladder's own answer,
    /// demotion included: on a run whose named tier could not be wired it
    /// names the tier that actually runs, and `reasoning` states the
    /// demotion. On a run with an explicit request it is still the ladder's
    /// outcome - the builder the run would have wired without the key.
    std::optional<qcx::io::BuilderKind> picked;
    /// True when the run wires the direct family's LEAN (Schwarz-only)
    /// member: the ladder's floor for this run, or an explicit
    /// `fock_builder = "lean"` request. This is the ONE statement of the
    /// member choice: the RHF wiring branch, the UHF runner's lean arm, the
    /// pre-gate setup admission, the certified-lane route judgement and the
    /// record's builder_member all read THIS value, so none of them can
    /// disagree about which member runs (before the 2026-09-13 ladder they
    /// each re-derived it from the input, which a demoted run would have
    /// got wrong).
    bool leanMember = false;
    /// The resolution's reasoning string, verbatim (the record's
    /// "reasoning" field; the driver reports it unchanged).
    std::string reasoning;
    /// True when the input named a [method].fock_builder FAMILY word
    /// explicitly. The lean carve-out: the explicit `fock_builder =
    /// "lean"` spelling is a within-family request, not a family word
    /// (RunMethodInput::leanDirect - the kGpuSplit precedent), so it
    /// leaves this flag FALSE by design, exactly as an absent key does;
    /// the request is carried in reasoning instead, and the choice it
    /// makes is carried in the record's selection.builder_member.
    bool explicitBuilder = false;
    /// The warning text when the run diverged from the resolution the
    /// policy names: the retargeted device-less gpu fallback, or a
    /// ladder DEMOTION (see this header's file note). Empty = no
    /// divergence. The driver emits it on stderr and serializes it in the
    /// record, so a demotion is never silent.
    std::string warning;
};

/// Everything the resolution needs, probed once by the caller. The sizes
/// (nBasis/nAux/nPairs) and the ri_j availability were consumed by the
/// retired estimator's pricing; nBasis now steers the no-key ladder, and
/// the two unavailable-reason fields state which of its tiers this run can
/// actually wire - both are facts the driver knows and this module must
/// not re-derive (the ladder itself reads the topology only for the device-less fallback
/// condition).
/// \ingroup qcx-driver
struct SelectionResolutionInput {
    qcx::backend::TopologyProfile topology; ///< The run-start topology probe.
    qcx::io::RunInput input; ///< The parsed input (builder absent = auto).
    /// Orbital basis-function count: the no-key ladder's own input (the
    /// tier boundaries in this header are counted in basis functions).
    std::size_t nBasis = 0;
    /// Auxiliary basis-function count (the estimator's pricing input).
    std::size_t nAux = 0;
    /// Why the ri_j_link tier cannot be wired (empty = it can): the
    /// method's v1 wiring has no ri_j arm (UHF's whitelist and the
    /// Kohn-Sham lanes), or the wiring cannot resolve an auxiliary basis
    /// for the run. The text is the reason the record prints, so the caller
    /// states it in the words its own refusal uses.
    ///
    /// The aux half of this field carries a dated note: an early dry run
    /// measured it as the LIVE reason (the pre-ruling rule refused every
    /// orbital name outside def2-* and cc-*), and the ruling of
    /// 2026-09-13 replaced that rule with a quality tier that always
    /// produces a default, so an unmatched orbital NAME no longer reaches
    /// it. The field and the ladder arm stay: the reason is still the
    /// caller's to state, and a rule that refuses again must be demoted,
    /// not run.
    std::optional<std::string> riJUnavailableReason;
    /// Why the qfmm tier cannot be wired (empty = it can): the v1 wiring's
    /// Kohn-Sham restriction - a Kohn-Sham run is wired on the direct
    /// family only, so the composed QFMM has no Kohn-Sham composition to
    /// run. Stated by the caller for the same reason as the field above.
    std::optional<std::string> qfmmUnavailableReason;
    /// True when this run is a Kohn-Sham run whose functional carries NO
    /// exact exchange (a pure / non-hybrid functional: PBE, LDA, ...). This
    /// is the ladder's ONE exception and it is keyed on the functional's
    /// CHARACTER, never on the method (see this header's file note): true
    /// promotes ri_j_link ABOVE the lean member, false keeps the shared
    /// order. The caller resolves the functional NAME against the registry
    /// (the driver's `ResolveKsFunctional`) and states the fact here; this
    /// module is pure policy and never sees a functional, and an
    /// unresolved name is stated as false - the shared order - with the
    /// run's own refusal naming the name it could not resolve.
    bool nonHybridKohnSham = false;
    std::size_t nPairs = 0; ///< Shell-pair count (the pair-store sizing input).
    int effectiveThreads = 1; ///< The threadCap-clamped team.
};

/// Resolves the run's builder (see ResolvedBuilderSelection). Total: the
/// ladder's floor - the direct family's lean member - is wired for every
/// method, so the auto path neither refuses nor walks off the ladder; the
/// explicit requests resolve too (the unwired UHF-unsafe kinds were
/// rejected up front by the driver's ValidateCombination). All memory
/// admission fires downstream of the resolution - the engine's own
/// Create-time estimates at the builder Creates - and on both the RHF and
/// the UHF paths the lean member's envelope one step earlier at
/// CheckPreGateSetupAdmission, before the setup ramp.
/// \param input The probed topology, input, sizes and tier runnability.
/// \returns The effective builder with the selection record, or the
/// refusal.
/// \ingroup qcx-driver
qcx::Result<ResolvedBuilderSelection> ResolveBuilderSelection(
    const SelectionResolutionInput& input) noexcept;

} // namespace qcx::driver
