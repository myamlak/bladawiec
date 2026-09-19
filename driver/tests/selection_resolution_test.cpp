// The builder resolution (selection_resolution.hpp): the no-builder default
// is a size LADDER - the direct family's lean (Schwarz-only) member at
// nBasis <= 1000, ri_j_link up to 2000, qfmm above - and direct (the budgeted
// machinery), both disk routes and gpu run only as explicit opt-ins. A tier
// this run cannot WIRE is DEMOTED with disclosure (the record's reasoning plus the
// warning), never refused for an absent key and never silently substituted;
// an explicit request is honored or refused by name and is never demoted.
// The model-driven auto-selection stays retired from the default path.
// Nothing refuses at the resolution: the ladder's floor is wired for every
// method, so every memory admission fires downstream of it - in the wiring,
// or one step earlier at the driver's pre-gate setup admission
// (CheckPreGateSetupAdmission), which consults the same estimate before the
// setup ramp.
//
// THE ONE EXCEPTION: the order above is the SAME for RHF, UHF, RKS and UKS,
// and the single exception is a NON-HYBRID Kohn-Sham run, which puts
// ri_j_link BEFORE the lean member (a pure functional needs J and no K).
// The key is the functional's character, never the method - hybrid Kohn-Sham
// keeps the shared order. The exception tests state both sides of that key.
//
// WHAT IS MEASURED AND WHAT IS CHOSEN, pinned here because this file is
// where the policy is read: 1000 is the existing lean ceiling carried over
// unchanged (it has history); 2000 is a CHOSEN value with no
// measurement behind it. The reasoning text says so, and
// ReasoningStatesWhichBoundaryIsMeasured pins that it keeps saying so - a
// reader must not be able to take the chosen number for a derived one.
//
// Pure function tests with injected topologies - deterministic on every
// machine, CUDA or not.

#include "qcx/driver/selection_resolution.hpp"
#include "qcx/io/parse_input.hpp"

#include <gtest/gtest.h>
#include <string>

namespace {

using qcx::backend::DeviceProfile;
using qcx::backend::TopologyProfile;
using qcx::driver::kLeanTierMaxBasisFunctions;
using qcx::driver::kRiJTierMaxBasisFunctions;
using qcx::driver::ResolveBuilderSelection;
using qcx::driver::SelectionResolutionInput;
using qcx::io::BuilderKind;
using qcx::io::MethodType;
using qcx::io::RunInput;

constexpr double kGiB = 1073741824.0;

// The aux-refusal text the driver used to state when the auto-selection rule
// covered no aux for the orbital basis - the measured aux-coverage
// constraint. That rule is now a quality tier that always produces a
// default, so the driver no longer produces these words for any orbital
// name the parser admits. The ladder arm is still the contract - the reason
// is the caller's to state, and a rule that refuses again must be demoted
// rather than run - so the test injects the text rather than invoking the
// rule.
constexpr const char* kNoAuxReason = "no aux basis for \"sto-3g\"; specify [basis].aux explicitly";

// A stand-in for "this tier is unwired for the run" on the Kohn-Sham lanes -
// the resolution's demotion machinery is what these tests exercise, and the
// reason is the CALLER's to state (the comment above: the test injects the
// text rather than invoking the rule). It does not have to be the driver's
// live words, and it is not any of them now: qfmm was the last
// family the driver produced a reason of this shape for on a Kohn-Sham run
// ("no half for the energy seam"), and its composition landed, so a Kohn-Sham
// run now wires every tier the size ladder names. What the driver still
// refuses on this path are the ri_j_link DISK RUNGS - a rung, not a tier, and
// the ladder's word for it is the rung refusal's own text. The rows below
// keep injecting the shape because that is the property they pin: a caller
// that states a tier reason gets a demotion with that reason in it.
constexpr const char* kKsReason =
    "a Kohn-Sham run (method.type = \"rks\"/\"uks\") is not wired with this family";

// A 4 GiB device with 3.5 GiB free; the headroom constant (0.5 GiB)
// leaves a 3.0 GiB device budget.
DeviceProfile FakeDevice() {
    DeviceProfile device;
    device.deviceId = 0;
    device.totalBytes = static_cast<std::size_t>(4.0 * kGiB);
    device.freeBytes = static_cast<std::size_t>(3.5 * kGiB);
    return device;
}

TopologyProfile DeviceTopology() {
    TopologyProfile topology = {};
    topology.devices.push_back(FakeDevice());
    return topology;
}

TopologyProfile DeviceLessTopology() {
    return TopologyProfile{};
}

SelectionResolutionInput MakeInput(const TopologyProfile& topology, const RunInput& input) {
    SelectionResolutionInput resolved;
    resolved.topology = topology;
    resolved.input = input;
    return resolved;
}

SelectionResolutionInput SmallInput(const TopologyProfile& topology, const RunInput& input) {
    auto resolved = MakeInput(topology, input);
    resolved.nBasis = 2;
    resolved.nPairs = 3;
    resolved.effectiveThreads = 8;
    return resolved;
}

SelectionResolutionInput LargeInput(const TopologyProfile& topology, const RunInput& input) {
    auto resolved = MakeInput(topology, input);
    resolved.nBasis = 562;
    resolved.nAux = 0;
    resolved.nPairs = 81003;
    resolved.effectiveThreads = 12;
    return resolved;
}

// The ladder's own input: a run of a given size, everything else default.
SelectionResolutionInput SizedInput(std::size_t nBasis,
                                    const RunInput& input,
                                    const TopologyProfile& topology = DeviceLessTopology()) {
    auto resolved = MakeInput(topology, input);
    resolved.nBasis = nBasis;
    resolved.nPairs = nBasis / 2;
    resolved.effectiveThreads = 8;
    return resolved;
}

RunInput AutoInput(MethodType method = MethodType::kRhf) {
    RunInput input;
    input.method.method = method;
    return input;
}

// A run whose middle tier is runnable: the auxiliary basis the wiring would
// auto-select exists, so the ladder stays on the ri_j_link tier.
SelectionResolutionInput RiJRunnable(std::size_t nBasis, const RunInput& input) {
    auto resolved = SizedInput(nBasis, input);
    resolved.riJUnavailableReason = std::nullopt;
    return resolved;
}

} // namespace

TEST(SelectionResolutionTest, AutoAtOrBelowTheLeanBoundaryPicksTheLeanMember) {
    // The ladder's floor: an absent key at nBasis <= 1000 wires the direct
    // family's lean member, silently
    // (the ladder named this tier and the run can wire it), and the record
    // carries the family word plus the member flag.
    const auto resolved = ResolveBuilderSelection(SmallInput(DeviceLessTopology(), AutoInput()));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_FALSE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("defaults to the size ladder"), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("model-driven auto-selection stays retired"),
              std::string::npos);
}

TEST(SelectionResolutionTest, AutoWithADeviceAtScaleStaysOnTheLadder) {
    // The retired estimator's device pick (gpu at n = 562 on a 4 GiB
    // device) is gone: device presence does not steer the ladder, so the
    // lean tier resolves here too, silently.
    const auto resolved = ResolveBuilderSelection(LargeInput(DeviceTopology(), AutoInput()));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_FALSE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("defaults to the size ladder"), std::string::npos);
}

TEST(SelectionResolutionTest, TheBoundariesAreWhereTheRulingPutsThem) {
    // The table, read at its own edges: 1000 is the lean ceiling (the tier
    // is chosen AT the boundary, not above it), 2000 ends the ri_j_link
    // tier. Every leg is run with the middle tier runnable, so the tiers
    // below are the size policy's answer and never a demotion.
    const struct Leg {
        std::size_t nBasis;
        BuilderKind kind;
        bool lean;
    } legs[] = {
        {kLeanTierMaxBasisFunctions, BuilderKind::kDirect, true},
        {kLeanTierMaxBasisFunctions + 1, BuilderKind::kRiJLink, false},
        {kRiJTierMaxBasisFunctions, BuilderKind::kRiJLink, false},
        {kRiJTierMaxBasisFunctions + 1, BuilderKind::kQfmm, false},
    };

    for (const Leg& leg : legs)
    {
        const auto resolved = ResolveBuilderSelection(RiJRunnable(leg.nBasis, AutoInput()));

        ASSERT_TRUE(resolved.has_value()) << resolved.error().message << " at n = " << leg.nBasis;
        EXPECT_EQ(resolved->kind, leg.kind) << " at n = " << leg.nBasis;
        EXPECT_EQ(resolved->leanMember, leg.lean) << " at n = " << leg.nBasis;
        EXPECT_TRUE(resolved->warning.empty()) << " at n = " << leg.nBasis;
    }
}

TEST(SelectionResolutionTest, NonHybridKohnShamPromotesRiJAboveLean) {
    // The ladder's ONE exception: a NON-HYBRID
    // Kohn-Sham run puts ri_j_link before
    // the lean member, because a pure functional needs J and no K. At a size
    // inside the lean band the tier is therefore ri_j_link - with the
    // exception's own clause in the reasoning, so the record says which order
    // it was resolved against.
    RunInput input = AutoInput(MethodType::kRks);
    auto selectionInput = SizedInput(200, input);
    selectionInput.nonHybridKohnSham = true;

    const auto resolved = ResolveBuilderSelection(selectionInput);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kRiJLink);
    EXPECT_FALSE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kRiJLink);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("NON-HYBRID (pure) Kohn-Sham run"), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("puts ri_j_link BEFORE the lean member"), std::string::npos);
}

TEST(SelectionResolutionTest, HybridKohnShamKeepsTheSharedOrder) {
    // The key is the functional's CHARACTER, not the method: a hybrid
    // Kohn-Sham run (B3LYP, PBE0) needs K, so it keeps the shared order
    // - the lean member inside its band - and carries NO exception clause.
    // Keying on "is it Kohn-Sham" would have mis-ordered this run.
    RunInput input = AutoInput(MethodType::kRks);
    auto selectionInput = SizedInput(200, input);
    selectionInput.nonHybridKohnSham = false;

    const auto resolved = ResolveBuilderSelection(selectionInput);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    EXPECT_EQ(resolved->reasoning.find("NON-HYBRID"), std::string::npos);
}

TEST(SelectionResolutionTest, TheNonHybridOrderDisclosesWhenThePromotedTierIsUnwirable) {
    // The exception changes the ORDER, not the disclosure. The v1
    // wiring has no Kohn-Sham ri_j arm, so a pure-Kohn-Sham run whose
    // promoted tier cannot be wired gets the demotion - naming the tier the
    // pure-functional order promoted, the reason, and where the run landed -
    // AND the clause that says why the order was the two-tier one. A record
    // that silently ran lean would hide both facts.
    RunInput input = AutoInput(MethodType::kRks);
    auto selectionInput = SizedInput(200, input);
    selectionInput.nonHybridKohnSham = true;
    selectionInput.riJUnavailableReason = kKsReason;

    const auto resolved = ResolveBuilderSelection(selectionInput);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    EXPECT_NE(resolved->reasoning.find("names the ri_j_link tier"), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("NON-HYBRID (pure) Kohn-Sham run"), std::string::npos);
    EXPECT_NE(resolved->warning.find("names the ri_j_link tier at nBasis = 200"),
              std::string::npos);
}

TEST(SelectionResolutionTest, ReasoningStatesWhichBoundaryIsMeasured) {
    // Pinned as text: the reasoning must say that the 2000 boundary is
    // CHOSEN and unmeasured, and that 1000 is the boundary with history.
    // A future edit that quietly promotes the chosen
    // number to a derived one fails here.
    const auto resolved =
        ResolveBuilderSelection(RiJRunnable(kRiJTierMaxBasisFunctions + 1, AutoInput()));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_NE(resolved->reasoning.find("CHOSEN value with NO measurement"), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("the 1000 boundary is the one boundary with history"),
              std::string::npos);
}

TEST(SelectionResolutionTest, MissingAuxDemotesTheMiddleTierToTheLeanMember) {
    // The measured aux-coverage constraint: ri_j_link needs an auxiliary
    // basis, and while the older aux rule covered no aux for the orbital
    // name a no-key run in the middle tier could not enter it. The aux rule
    // is now a quality tier that always resolves, so this fixture injects
    // the reason instead of provoking it - what is under test is the ladder's
    // contract, not the aux rule's coverage: a tier this run cannot wire
    // DEMOTES to the tier below, disclosed in the reasoning AND carried on
    // the warning, never silently and never by refusing a run that named no
    // builder.
    auto input = SizedInput(1500, AutoInput());
    input.riJUnavailableReason = kNoAuxReason;

    const auto resolved = ResolveBuilderSelection(input);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_FALSE(resolved->explicitBuilder);

    // The reasoning: the tier the size named, why, and where the run landed.
    EXPECT_NE(resolved->reasoning.find("nBasis = 1500 names the ri_j_link tier"),
              std::string::npos);
    EXPECT_NE(resolved->reasoning.find(kNoAuxReason), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("the ladder demoted to the direct family's lean member"),
              std::string::npos);
    EXPECT_NE(resolved->reasoning.find("specify [basis].aux explicitly"), std::string::npos);

    // And the same divergence on the warning the driver prints to stderr.
    EXPECT_NE(resolved->warning.find("names the ri_j_link tier at nBasis = 1500"),
              std::string::npos);
    EXPECT_NE(resolved->warning.find("demoting to the direct family's lean member"),
              std::string::npos);
}

TEST(SelectionResolutionTest, AnUnavailableTopTierWalksDownOneStepAtATime) {
    // A tier that cannot run is skipped for the tier BELOW it, not for the
    // floor: with qfmm unavailable and ri_j_link runnable, a run above the
    // 2000 boundary lands on ri_j_link.
    auto input = SizedInput(4000, AutoInput());
    input.qfmmUnavailableReason = kKsReason;

    const auto resolved = ResolveBuilderSelection(input);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kRiJLink);
    EXPECT_FALSE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kRiJLink);
    EXPECT_NE(resolved->reasoning.find("nBasis = 4000 names the qfmm tier"), std::string::npos);
    EXPECT_NE(resolved->warning.find("demoting to ri_j_link"), std::string::npos);
}

TEST(SelectionResolutionTest, TheWalkReachesTheFloorAndNamesEveryTierItPassed) {
    // A run whose two upper tiers are both unwired lands on the lean member,
    // and the disclosure names BOTH tiers it passed - a reader must be able
    // to see that the table said qfmm and neither upper rung was reachable,
    // not merely that the run ended up small. The Kohn-Sham lanes supply no
    // run of that shape at all any more - a Kohn-Sham run wires qfmm and
    // ri_j_link both, and only the ri_j_link disk RUNGS are left, which are
    // a rung rather than a tier. The fixture below is therefore synthetic on
    // both members, which is what it always was: the reason is the caller's to
    // state, and what this row pins is the walk's own behaviour - two stated
    // reasons, two tiers named, the floor
    // reached and disclosed.
    auto input = SizedInput(4000, AutoInput(MethodType::kRks));
    input.riJUnavailableReason = kKsReason;
    input.qfmmUnavailableReason = kKsReason;

    const auto resolved = ResolveBuilderSelection(input);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    EXPECT_NE(resolved->reasoning.find("qfmm ("), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("ri_j_link ("), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("demoted to the direct family's lean member"),
              std::string::npos);
}

TEST(SelectionResolutionTest, UhfAboveTheLeanBoundaryDemotesWhenTheMiddleTierIsUnwired) {
    // The demotion machinery, exercised on the UHF leg. The middle tier is
    // no longer unwired FOR UHF in general - the per-spin adapter landed
    // (RunRiJLinkUhfScf), so a plain no-key UHF run at this size now
    // resolves to ri_j_link - and what still makes the tier unwired for an
    // unrestricted run is the family's disk rungs (no per-spin adapter for
    // the storage-module store), whose reason the driver's own
    // LadderRunnabilityFor produces and this test feeds by hand, exactly as
    // it always fed the old one. What the row asserts is the ladder's
    // behaviour on a reason: a no-key UHF run at 1500 basis functions is
    // demoted to the lean member - the UHF seam wires it - rather than
    // refused or silently run on a family with no UHF arm.
    auto input = SizedInput(1500, AutoInput(MethodType::kUhf));
    input.riJUnavailableReason = "the RI disk rung is wired on the restricted ri_j_link leg only";

    const auto resolved = ResolveBuilderSelection(input);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    EXPECT_NE(resolved->warning.find("the RI disk rung"), std::string::npos);
}

TEST(SelectionResolutionTest, AnExplicitRequestIsNeverDemoted) {
    // The ladder governs the ABSENT case. An explicit ri_j_link on a
    // run whose aux cannot be resolved keeps resolving to ri_j_link - the
    // wiring refuses it by name, which is where an explicit request is
    // judged - and it is never quietly moved to another tier by the
    // ladder's own reasoning about what is runnable.
    RunInput input = AutoInput();
    input.method.builder = BuilderKind::kRiJLink;

    auto selectionInput = SizedInput(1500, input);
    selectionInput.riJUnavailableReason = kNoAuxReason;

    const auto resolved = ResolveBuilderSelection(selectionInput);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kRiJLink);
    EXPECT_FALSE(resolved->leanMember);
    EXPECT_TRUE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    // The demotion DISCLOSURE is what must not be there - the reasoning is
    // allowed to say that an explicit request is never demoted.
    EXPECT_EQ(resolved->reasoning.find("the ladder demoted to"), std::string::npos);
}

TEST(SelectionResolutionTest, ExplicitDirectKeepsTheMachineryMemberAtEverySize) {
    // An explicit machinery spelling stays byte-stable, and that
    // includes the MEMBER - the ladder's lean member for small runs must
    // not leak into an explicit "direct" request, which would run one
    // builder while the request named another.
    RunInput input = AutoInput();
    input.method.builder = BuilderKind::kDirect;

    for (const std::size_t nBasis : {std::size_t{2}, kRiJTierMaxBasisFunctions + 1})
    {
        const auto resolved = ResolveBuilderSelection(SizedInput(nBasis, input));
        ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
        EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
        EXPECT_FALSE(resolved->leanMember) << " at n = " << nBasis;
        // The record's pick is still the ladder's own answer for the run.
        ASSERT_TRUE(resolved->picked.has_value());
        EXPECT_EQ(*resolved->picked,
                  nBasis <= kLeanTierMaxBasisFunctions ? BuilderKind::kDirect : BuilderKind::kQfmm);
    }
}

TEST(SelectionResolutionTest, MultiNodeAutoResolvesOnTheLadder) {
    // The estimator's distributed-execution stub refusal is not consulted:
    // the resolution reads no node count, so a nodeCount = 2 topology
    // resolves like any other.
    RunInput input = AutoInput();
    auto selectionInput = SmallInput(DeviceTopology(), input);
    selectionInput.topology.nodeCount = 2;

    const auto resolved = ResolveBuilderSelection(selectionInput);
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    EXPECT_FALSE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
}

TEST(SelectionResolutionTest, ExplicitDirectRequestIsHonoredVerbatimAndSilently) {
    // An explicit builder wins verbatim (the machinery family for an
    // explicit "direct"); the record still reports the pick the run would
    // have wired without it (the ladder's answer), and the old
    // explicit-override divergence warning is retired with the model - an
    // explicit request never diverges from the policy's own answer.
    RunInput input = AutoInput();
    input.method.builder = BuilderKind::kDirect;

    const auto resolved = ResolveBuilderSelection(LargeInput(DeviceTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("explicit fock_builder direct honored"), std::string::npos);
}

TEST(SelectionResolutionTest, TheTierWordInMemoryResolvesExactlyLikeTheFamilyWordDirect) {
    // The two axes of the builder vocabulary: `direct` names the FAMILY and
    // runs its in-memory tier; `in_memory` names that TIER. One request in two
    // spellings, so both must reach ONE
    // resolution - checked here through the REAL parser and the real resolver,
    // on one TOML text that differs in nothing but the word. A struct the test
    // filled in twice could not show that: it would pin the resolver's
    // behaviour on a state, not the words' equivalence.
    const auto parseWord = [](const std::string& word) {
        const std::string toml =
            "[molecule]\n"
            "atoms = [[\"H\", 0.0, 0.0, 0.0], [\"H\", 0.74084809526419992, 0.0, 0.0]]\n"
            "[basis]\n"
            "orbital = \"sto-3g\"\n"
            "[method]\n"
            "type = \"rhf\"\n"
            "fock_builder = \"" +
            word + "\"\naccuracy = \"kNormal\"\n";
        return qcx::io::ParseRunInput(toml);
    };

    const auto directInput = parseWord("direct");
    ASSERT_TRUE(directInput.has_value()) << directInput.error().message;
    const auto inMemoryInput = parseWord("in_memory");
    ASSERT_TRUE(inMemoryInput.has_value()) << inMemoryInput.error().message;

    // The parse half: one builder slot value, one clear lean flag. The tier
    // word fills the slot with kDirect (it names no enumerator of its own) and
    // must NOT touch the lean flag - the lean member is the same axis's other
    // word, and reaching it from `in_memory` would run a builder the request
    // did not name.
    ASSERT_TRUE(inMemoryInput->method.builder.has_value());
    EXPECT_EQ(*inMemoryInput->method.builder, BuilderKind::kDirect);
    EXPECT_FALSE(inMemoryInput->method.leanDirect);
    EXPECT_EQ(inMemoryInput->method.builder, directInput->method.builder);
    EXPECT_EQ(inMemoryInput->method.leanDirect, directInput->method.leanDirect);

    // The resolution half, on both sides of the lean boundary: the same kind,
    // the same member, the same record text. Above the boundary the ladder
    // would pick qfmm - the explicit tier word keeps the machinery there, as
    // the family word does, and the record still reports the ladder's pick.
    for (const std::size_t nBasis : {std::size_t{2}, kRiJTierMaxBasisFunctions + 1})
    {
        const auto direct = ResolveBuilderSelection(SizedInput(nBasis, *directInput));
        ASSERT_TRUE(direct.has_value()) << direct.error().message;
        const auto inMemory = ResolveBuilderSelection(SizedInput(nBasis, *inMemoryInput));
        ASSERT_TRUE(inMemory.has_value()) << inMemory.error().message;

        EXPECT_EQ(inMemory->kind, direct->kind) << " at n = " << nBasis;
        EXPECT_EQ(inMemory->leanMember, direct->leanMember) << " at n = " << nBasis;
        EXPECT_EQ(inMemory->explicitBuilder, direct->explicitBuilder) << " at n = " << nBasis;
        EXPECT_EQ(inMemory->picked, direct->picked) << " at n = " << nBasis;
        EXPECT_EQ(inMemory->reasoning, direct->reasoning) << " at n = " << nBasis;
        EXPECT_EQ(inMemory->warning, direct->warning) << " at n = " << nBasis;

        // Said outright as well, so the pairing above cannot pass by both sides
        // going wrong together: the request wires the MACHINERY member, not the
        // ladder's lean default for a small run.
        EXPECT_EQ(inMemory->kind, BuilderKind::kDirect) << " at n = " << nBasis;
        EXPECT_FALSE(inMemory->leanMember) << " at n = " << nBasis;
        EXPECT_TRUE(inMemory->explicitBuilder) << " at n = " << nBasis;
    }
}

TEST(SelectionResolutionTest, ExplicitLeanRequestResolvesToTheLeanMember) {
    // The explicit fock_builder = "lean" names no family word (the lean
    // member is within-family), so it leaves the builder slot absent,
    // explicitBuilder absent and the record's pick the ladder's - and ONLY
    // the reasoning records the request, which must not deny that a key was
    // given.
    RunInput input = AutoInput();
    input.method.leanDirect = true;

    const auto resolved = ResolveBuilderSelection(LargeInput(DeviceTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_FALSE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("defaults to the size ladder"), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("explicit fock_builder lean honored"), std::string::npos);
    EXPECT_NE(resolved->reasoning.find("within-family lean member"), std::string::npos);
    // The request reaches the member above the boundary, where the ladder
    // alone would not pick it - said in the text, since that is the one
    // thing the flag changes there.
    EXPECT_NE(resolved->reasoning.find("this request is what reaches the member"),
              std::string::npos);
    EXPECT_EQ(resolved->reasoning.find("no fock_builder given"), std::string::npos);
}

TEST(SelectionResolutionTest, ExplicitRiJLinkRequestIsHonoredVerbatimAndSilently) {
    // ri_j_link is an explicit opt-in under the ladder: honored as
    // requested, recorded as such, no warning (nothing diverged - the
    // divergence warnings died with the heuristic pick; the ladder's own
    // demotion warning belongs to the absent case).
    RunInput input = AutoInput();
    input.method.builder = BuilderKind::kRiJLink;

    const auto resolved = ResolveBuilderSelection(LargeInput(DeviceTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kRiJLink);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("explicit fock_builder ri_j_link honored"),
              std::string::npos);
}

TEST(SelectionResolutionTest, ExplicitGpuOnADeviceMachineIsHonoredVerbatimAndSilently) {
    // gpu is an explicit opt-in: on a machine with a device the request
    // wires gpu, recorded as explicit - the no-builder default (the
    // record's pick) is still the ladder's answer, and no warning fires.
    RunInput input = AutoInput();
    input.method.builder = BuilderKind::kGpu;

    const auto resolved = ResolveBuilderSelection(LargeInput(DeviceTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kGpu);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("explicit fock_builder gpu honored"), std::string::npos);
}

TEST(SelectionResolutionTest, DeviceLessGpuRequestFallsBackToTheLadderWithALoudWarning) {
    // The device-less GPU fallback: an explicit "gpu" without a device
    // falls back to the LADDER's own tier for this run (never a model pick,
    // the model is retired) with a loud warning, and the record's reasoning
    // starts from the request too (the key WAS given - the no-key opener
    // would contradict the warning and the record's explicit_builder).
    RunInput input = AutoInput();
    input.method.builder = BuilderKind::kGpu;

    const auto resolved = ResolveBuilderSelection(SmallInput(DeviceLessTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->explicitBuilder);
    EXPECT_NE(resolved->warning.find("no CUDA device is present"), std::string::npos);
    EXPECT_NE(resolved->warning.find("falling back to the no-fock_builder size ladder's tier"),
              std::string::npos);
    EXPECT_NE(resolved->reasoning.find("defaults to the size ladder"), std::string::npos);
    // The fallback's reasoning must not deny the given key either (the
    // warning right above names it as the request).
    EXPECT_EQ(resolved->reasoning.find("no fock_builder given"), std::string::npos);
}

TEST(SelectionResolutionTest, DeviceLessGpuRequestUnderATinyCapStillFallsBack) {
    // The old nothing-to-fall-back-to refusal cannot fire anymore: the
    // ladder's floor always exists, so even a tiny cap leaves the
    // resolution to the wiring's own admission - the fallback resolves
    // with its loud warning.
    RunInput input = AutoInput();
    input.method.builder = BuilderKind::kGpu;
    input.resources.memoryCapGiB = 0.001;

    const auto resolved = ResolveBuilderSelection(SmallInput(DeviceLessTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->explicitBuilder);
    EXPECT_NE(resolved->warning.find("no CUDA device is present"), std::string::npos);
}

TEST(SelectionResolutionTest, AutoNeverRefusesAtTheResolution) {
    // Auto + a tiny cap that used to prune every candidate: the resolution
    // refuses nothing (the ladder's floor always exists) - the memory
    // admission lives downstream of it (the engine's own Create-time
    // last-resort check or the ri_j engine's workspace estimate, and first
    // at the driver's pre-gate setup admission).
    RunInput input = AutoInput();
    input.resources.memoryCapGiB = 0.001;

    const auto resolved = ResolveBuilderSelection(SmallInput(DeviceLessTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;
    EXPECT_EQ(resolved->kind, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->leanMember);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_FALSE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
}

TEST(SelectionResolutionTest, UhfExplicitQfmmRequestIsHonoredVerbatimAndSilently) {
    // The composed-QFMM UHF adapter is an explicit opt-in: an
    // explicit fock_builder = "qfmm" on a UHF run resolves to qfmm on
    // every topology, recorded as explicit, with the ladder's answer as the
    // record's pick - no divergence warning.
    RunInput input = AutoInput(MethodType::kUhf);
    input.method.builder = BuilderKind::kQfmm;

    const auto resolved = ResolveBuilderSelection(LargeInput(DeviceTopology(), input));
    ASSERT_TRUE(resolved.has_value()) << resolved.error().message;

    EXPECT_EQ(resolved->kind, BuilderKind::kQfmm);
    ASSERT_TRUE(resolved->picked.has_value());
    EXPECT_EQ(*resolved->picked, BuilderKind::kDirect);
    EXPECT_TRUE(resolved->explicitBuilder);
    EXPECT_TRUE(resolved->warning.empty());
    EXPECT_NE(resolved->reasoning.find("explicit fock_builder qfmm honored"), std::string::npos);
}
