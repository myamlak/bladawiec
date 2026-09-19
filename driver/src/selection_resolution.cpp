// The builder resolution (selection_resolution.hpp): the no-builder
// default is a size LADDER - the
// direct family's lean member at nBasis <= 1000, ri_j_link up to 2000,
// qfmm above, the SAME for RHF, UHF, RKS and UKS - with ONE exception keyed
// on the functional's character rather than the method (a NON-HYBRID
// Kohn-Sham run promotes ri_j_link above the lean member; hybrid Kohn-Sham
// keeps the shared order). direct (the budgeted machinery), both disk routes
// and the gpu/GpuSplit candidates run only as explicit opt-ins. The
// model-driven auto-selection stays retired from the default path: the
// estimator (selection_heuristic.hpp) stays untouched and testable, but
// nothing here steers on it - it may return only behind measured crossover
// data. The device-less GPU fallback survives, retargeted to the
// ladder's pick. A tier this run cannot WIRE is demoted, disclosed, never
// silently substituted. Pure: the caller probes the topology, counts the
// sizes and states the tiers' runnability; everything here is
// deterministic policy.

#include "qcx/driver/selection_resolution.hpp"

#include <string>

namespace qcx::driver {
namespace {

// The short name of a builder kind, for the warning texts.
std::string NameOf(qcx::io::BuilderKind kind) {
    return std::string(qcx::io::ToString(kind));
}

// The standing default-ladder text, verbatim (the record's "reasoning"
// field; the driver reports it unchanged). Speaker-neutral by design: only
// the openers below say WHO asked for the family, and the body never claims
// a key was or was not given - the record's explicit_builder flag carries
// that, and a text that denied it while the driver named the request in the
// same sentence would read as a lie.
//
// The PROVENANCE of the two boundaries is stated here because this text is
// where the policy is read: what is not measured must say so wherever the
// policy is stated. 1000 is the existing lean
// ceiling; 2000 is a chosen value with no measurement behind it, and a
// reader who took the two for the same kind of number would be reading a
// choice as a finding.
constexpr const char* kDefaultFamilyReasoning =
    "the run defaults to the size ladder: the direct family's lean member at nBasis <= 1000 "
    "(the existing lean ceiling, carried over unchanged), ri_j_link at 1000 < nBasis <= 2000, "
    "qfmm above 2000; the 2000 boundary is a CHOSEN value with NO measurement behind it (where "
    "qfmm's crossover begins is unmeasured, so it must not be read as a derived number), and "
    "the 1000 boundary is the one boundary with history; direct (the budgeted machinery) and "
    "both disk routes (RI-J disk, full/cached ERI) are explicit opt-ins; model-driven "
    "auto-selection stays retired pending measured crossover data";

// The genuine no-key opener: the record of a run whose input named no
// builder at all.
constexpr const char* kNoBuilderOpener = "no fock_builder given: ";

// The standing explicit reasoning (a record for an opt-in request never
// pretends a model ranked it). An explicit request leaves the ladder
// entirely: it is honored by name, demotion and all.
std::string ExplicitReasoning(qcx::io::BuilderKind kind) {
    return "explicit fock_builder " + NameOf(kind) +
           " honored (the ladder governs the absent case only - an explicit request is never "
           "demoted, and the record's pick names the tier the ladder would have wired for this "
           "run without the key)";
}

// The explicit lean request's reasoning: the request IS the opener (the
// key was given, so no text here may claim otherwise). The lean member is
// within-family - the request names no family word - so it leaves the
// builder slot absent and explicitBuilder false, and this text, never the
// pick or the builder slot, is the record of the request. Above the lean
// boundary the request is what reaches the member at all (the ladder picks
// a higher tier there). Pinned by the tests (selection_resolution_test.cpp,
// run_driver_test.cpp).
std::string ExplicitLeanReasoning() {
    return std::string("explicit fock_builder lean honored (the direct family's within-family "
                       "lean member - the key was given and named no family word, so the "
                       "builder slot stays absent and the record's pick stays the ladder's; "
                       "above the 1000-basis-function lean boundary this request is what "
                       "reaches the member): ") +
           kDefaultFamilyReasoning;
}

// The ladder's rungs, in the order the policy names them.
// Not a bulk-storage type; shrinking the base type is a deferred micro-optimization.
// NOLINTNEXTLINE(performance-enum-size)
enum class Tier {
    kLean, ///< The direct family's within-family lean (Schwarz-only) member.
    kRiJLink,
    kQfmm,
};

// The tier the size policy names for this count (its table, with
// its ONE exception - see the nonHybridKohnSham clause below).
//
// The ladder is the SAME for RHF, UHF, RKS and UKS, so the
// method does not enter this function. The exception is keyed on the
// functional's CHARACTER, never on the method: a NON-HYBRID (pure) Kohn-Sham
// run promotes ri_j_link ABOVE the lean member, because a non-hybrid
// functional needs J and no K - the RI-J path is the cheap one there, while
// the lean member's Schwarz-screened direct build would do work the method
// does not need. Hybrid Kohn-Sham (B3LYP, PBE0) keeps the shared order,
// because it needs K. Keying on "is it Kohn-Sham" instead would silently
// mis-order every hybrid DFT run.
//
// The exception moves the ORDER only: whichever tier runs, the demotion walk
// below still discloses it.
Tier NamedTier(std::size_t nBasis, bool nonHybridKohnSham) {
    if (nBasis <= kLeanTierMaxBasisFunctions)
    {
        return nonHybridKohnSham ? Tier::kRiJLink : Tier::kLean;
    }

    if (nBasis <= kRiJTierMaxBasisFunctions)
    {
        return Tier::kRiJLink;
    }

    return Tier::kQfmm;
}

// The exception's own clause, appended to the reasoning of the run it applies
// to and to no other: a reader of a pure-Kohn-Sham record must be able to see
// that the order it was resolved against is the two-tier one, and why.
constexpr const char* kNonHybridKohnShamClause =
    "; this is a NON-HYBRID (pure) Kohn-Sham run, so the ladder puts ri_j_link BEFORE the lean "
    "member (a pure functional needs J and no K: the RI-J path is the cheap one there, while the "
    "lean member's screened direct build would do work the method does not need); hybrid "
    "functionals keep the shared order, because they need K";

// The policy's own name for a tier: the text a demotion's reasoning and
// warning use, so a reader ties the record's words back to the table.
const char* TierName(Tier tier) {
    switch (tier)
    {
    case Tier::kLean:
        return "the direct family's lean member";
    case Tier::kRiJLink:
        return "ri_j_link";
    case Tier::kQfmm:
        return "qfmm";
    }

    return "the direct family's lean member";
}

// The tier below this one - the demotion step. The lean member is the
// ladder's floor: the direct family is wired for every method (RHF, both
// unrestricted legs through the UHF seam, and Kohn-Sham through the same
// half-mode flags the machinery member carries), so the walk down always
// terminates and the resolution never refuses.
Tier TierBelow(Tier tier) {
    switch (tier)
    {
    case Tier::kQfmm:
        return Tier::kRiJLink;
    case Tier::kRiJLink:
    case Tier::kLean:
        return Tier::kLean;
    }

    return Tier::kLean;
}

// Why this tier cannot be wired for this run, or nothing when it can. The
// two facts come from the caller (SelectionResolutionInput): they are the
// driver's own wiring knowledge, never re-derived here. The lean member
// has no entry - it is the floor, and a floor that could be unavailable
// would leave the ladder with nothing to demote to.
std::optional<std::string> TierUnavailableReason(const SelectionResolutionInput& input, Tier tier) {
    switch (tier)
    {
    case Tier::kLean:
        return std::nullopt;
    case Tier::kRiJLink:
        return input.riJUnavailableReason;
    case Tier::kQfmm:
        return input.qfmmUnavailableReason;
    }

    return std::nullopt;
}

// The no-key ladder's answer for one run: the tier that runs, whether it is
// the lean member, and the reasoning/warning the demotion (if any) owes.
struct LadderPick {
    qcx::io::BuilderKind kind = qcx::io::BuilderKind::kDirect;
    bool leanMember = false;
    std::string reasoning;
    std::string warning;
};

// Walks the ladder down from the tier the size names to the first tier this
// run can wire (the demote-with-disclosure arm). The demotion is stated
// twice on purpose: in the reasoning (the record's own why, read by anyone
// who reads the record) and in the warning (the same divergence on stderr,
// because a run that silently ran a tier the policy does not name for it is
// the defect the disclosure exists to prevent).
LadderPick PickFromLadder(const SelectionResolutionInput& input) {
    const Tier named = NamedTier(input.nBasis, input.nonHybridKohnSham);
    Tier running = named;

    // The tiers the walk passed, each with its reason, in walk order.
    std::string passed;

    while (const std::optional<std::string> reason = TierUnavailableReason(input, running))
    {
        if (!passed.empty())
        {
            passed += "; ";
        }

        passed += std::string(TierName(running)) + " (" + *reason + ")";

        if (running == Tier::kLean)
        {
            // Unreachable by construction (TierUnavailableReason returns
            // nothing for the floor). Left as the loop's own guard rather
            // than an assertion: refusing to imagine it is one thing, and
            // walking in a circle is another.
            break;
        }

        running = TierBelow(running);
    }

    LadderPick pick;
    pick.leanMember = running == Tier::kLean;
    pick.kind = pick.leanMember ? qcx::io::BuilderKind::kDirect
                                : (running == Tier::kRiJLink ? qcx::io::BuilderKind::kRiJLink
                                                             : qcx::io::BuilderKind::kQfmm);

    if (running == named)
    {
        pick.reasoning = kDefaultFamilyReasoning;
    } else
    {
        const std::string demotion =
            "nBasis = " + std::to_string(input.nBasis) + " names the " + TierName(named) +
            " tier of that ladder, which this run cannot wire: " + passed +
            ", so the ladder demoted to " + TierName(running) +
            " - the run is not refused, and the record names the member that ran";
        pick.reasoning = std::string(kDefaultFamilyReasoning) + "; " + demotion;
        pick.warning = "the no-fock_builder size ladder names the " + std::string(TierName(named)) +
                       " tier at nBasis = " + std::to_string(input.nBasis) +
                       ", which this run cannot wire: " + passed + "; demoting to " +
                       TierName(running) +
                       " instead of substituting silently (name fock_builder explicitly to "
                       "override the ladder)";
    }

    // The one exception's own disclosure, on the runs it applies to and no
    // others: the record of a pure-Kohn-Sham run states that it was resolved
    // against the two-tier order, whether or not the promoted tier could be
    // wired (an ordering the record does not mention is an ordering nobody
    // can audit).
    if (input.nonHybridKohnSham)
    {
        pick.reasoning += kNonHybridKohnShamClause;
    }

    return pick;
}

} // namespace

qcx::Result<ResolvedBuilderSelection> ResolveBuilderSelection(
    const SelectionResolutionInput& input) noexcept {
    const std::optional<qcx::io::BuilderKind> requested = input.input.method.builder;

    // The ladder answers first, whether or not a key was given: the
    // record's pick IS this answer (the builder the run wires without an
    // explicit fock_builder), and the explicit paths below read its
    // member choice - the GPU fallback lands on the ladder's tier.
    const LadderPick ladder = PickFromLadder(input);

    ResolvedBuilderSelection resolved;
    resolved.explicitBuilder = requested.has_value();
    resolved.picked = ladder.kind;
    resolved.leanMember = ladder.leanMember;
    resolved.warning = ladder.warning;

    if (!requested.has_value())
    {
        // The absent case: the ladder's own answer, nothing else. Nothing
        // to refuse here - the memory admission fires downstream of the
        // resolution: the engine's own Create-time estimates at the builder
        // Creates, and the lean member's envelope first at the driver's
        // pre-gate setup admission (CheckPreGateSetupAdmission), ahead of
        // the setup ramp.
        resolved.kind = ladder.kind;

        // The explicit lean request is the one within-family spelling that
        // reaches the member from the absent-key side: no family word was
        // given, so explicitBuilder stays false and the record's pick stays
        // the ladder's - the request's own reasoning alone records that
        // this was not the bare no-key default.
        if (input.input.method.leanDirect)
        {
            resolved.kind = qcx::io::BuilderKind::kDirect;
            resolved.leanMember = true;
            resolved.reasoning = ExplicitLeanReasoning();
        } else
        {
            resolved.reasoning = std::string(kNoBuilderOpener) + ladder.reasoning;
        }

        return resolved;
    }

    // The GPU auto-fallback, retargeted to the ladder: an explicit
    // "gpu" with no device present falls back to the tier the LADDER names
    // for this run (with its member choice - the fallback can land on the
    // lean member at or below the boundary) with a loud warning instead of
    // failing the run - misconfiguration does not fail it, and the fallback
    // target is the policy's own default, never a model pick (the model is
    // retired). The old nothing-to-fall-back-to refusal cannot fire
    // anymore: the ladder's floor is always there. (The probe reports no
    // devices without the CUDA runtime, so the plain builds land here too -
    // the wiring's "requires the CUDA build" refusal stays the defensive
    // net for a CUDA build whose probe found a device, not this path.)
    if (*requested == qcx::io::BuilderKind::kGpu && input.topology.devices.empty())
    {
        // kind/leanMember/picked are already the ladder's (set above); only
        // the texts below are the fallback's own.
        //
        // The fallback's reasoning starts from the request too: the key WAS
        // given (the warning below names it as well), so the no-key opener
        // would contradict both this text and the record's explicit_builder.
        resolved.reasoning =
            std::string("explicit fock_builder gpu requested but no CUDA device is present, so "
                        "the fallback is the ladder's own default for this run - ") +
            kDefaultFamilyReasoning;
        // The demotion warning the ladder produced (if this run's ladder
        // pass walked down) is NOT dropped: both divergences are facts
        // about the run, and the fallback's own text leads.
        resolved.warning =
            "fock_builder = \"gpu\" requested but no CUDA device is present; falling back to "
            "the no-fock_builder size ladder's tier for this run (" +
            (resolved.leanMember ? std::string(TierName(Tier::kLean))
                                 : "\"" + NameOf(resolved.kind) + "\"") +
            " - gpu is an explicit opt-in, and the model auto-selection that used to name the "
            "fallback is retired)" +
            (ladder.warning.empty() ? std::string() : std::string("; " + ladder.warning));
        return resolved;
    }

    // An explicit opt-in is honored as requested. The UHF-unsafe kinds
    // never arrive here (ValidateCombination rejects them up front); the
    // explicit direct request keeps resolving to the machinery family
    // byte-for-byte, and every pre-existing pin is untouched.
    //
    // The member choice is cleared with it, and that is not a formality: the
    // ladder answer computed above carries the lean member for every run at
    // or below the lean boundary, and an explicit fock_builder = "direct" is
    // exactly the request that must NOT inherit it. Leaving it set sent an
    // explicit "direct" at nBasis <= 1000 to the lean builder while the
    // reasoning claimed the machinery - a silent substitution, caught by the
    // record test (the member name read "lean" for an
    // explicit direct request).
    resolved.leanMember = false;
    resolved.kind = *requested;
    resolved.reasoning = ExplicitReasoning(*requested);
    // The ladder's demotion warning is cleared with it. It speaks about the
    // ABSENT case ("the no-fock_builder size ladder names the ri_j_link
    // tier...") and this run named a builder, so carrying it would put a
    // false statement about the input into the record of a run the ladder
    // never steered - the same speaker-neutrality rule the no-key opener
    // follows in the other direction.
    resolved.warning.clear();
    return resolved;
}

} // namespace qcx::driver
