// The bundled aux-basis selection table, exercised exhaustively against the
// real data/basis directory names (the table below mirrors the vendored
// corpus; a renamed or removed aux set breaks the existence assertions instead
// of silently decaying).
//
// TWO TESTS IN THIS FILE WERE REPLACED, not deleted:
// `CcFamilyAppendsRifitRegardlessOfBuilderKind` and
// `UnsupportedFamiliesAreHardErrorsForEveryBuilderKind` both pinned the OLD
// provenance rule, and the old rule's stated ground is REFUTED by measurement.
// Their replacements are
// `CcFamilyAppendsRifitOnlyForAJPureRequest` and
// `UncoveredFamiliesResolveToTheUniversalFitForEveryBuilderKind`. The first
// old test was not merely stale, it was DANGEROUS to keep: it asserted that a
// `cc-*` orbital basis with a JK request resolves to a J-ONLY `-rifit`, which
// is exactly the "silently using a J-fit for K" substitution that is the real
// risk. Its replacement pins the opposite, and
// `NoSelectionEverHandsAJOnlyFitToAJkRequest` sweeps every bundled name to
// hold that line.
//
// A SECOND contract changed with the corpus, when it gained the sets the
// tiered rule had no member for. Two expectations in this file moved with it,
// and both were RE-POINTED rather than dropped:
// `CcFamilyAppendsRifitOnlyForAJPureRequest` had pinned `def2-universal-jkfit`
// for every cc-* JK request (true only while the family had no JK fit), and the
// `AuxSelectionNoticeTest` JK case had pinned `cc-pvdz` as the unmatched
// example (true only for the same reason). The replacements pin the NEW
// behaviour at least as tightly - the cc JK arm is now pinned by exact name,
// and the notice arm by a name above every vendored JK rung - and the cases
// that became silent are pinned as silent in their own test.

#include "qcx/integrals/aux_basis.hpp"

#include <array>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

namespace {

using qcx::integrals::AuxSelectionNotice;
using qcx::integrals::FockBuilderKind;
using qcx::integrals::IsDiffuseCapableAux;
using qcx::integrals::IsJkOptimizedAux;
using qcx::integrals::kTier3DiffuseAux;
using qcx::integrals::SelectAuxBasis;

// Every bundled orbital basis directory. The
// sweeps below run over this list, so a newly vendored basis that breaks the
// "always produce a default" or the "never a J-fit for K" property fails here
// rather than in a run.
const std::vector<std::string> kAllBundledOrbitalBases = {"3-21g",
                                                          "6-311g-dstar",
                                                          "6-31g-dstar",
                                                          "6-31g-star",
                                                          "aug-cc-pvdz",
                                                          "aug-cc-pvtz",
                                                          "cc-pvdz",
                                                          "cc-pvtz",
                                                          "def2-qzvp",
                                                          "def2-svp",
                                                          "def2-tzvp",
                                                          "def2-tzvpd",
                                                          "def2-tzvppd",
                                                          "pcseg-1",
                                                          "sto-3g",
                                                          "sto-6g"};

// The bundled orbital bases with a def2-family aux rule: one universal aux
// family serves every zeta level.
const std::vector<std::string> kDef2OrbitalBases = {"def2-svp", "def2-qzvp", "def2-tzvp"};

// The bundled orbital bases with a cc-family aux rule: each has its own
// matching J-only -rifit set, including the aug- prefix.
const std::vector<std::pair<std::string, std::string>> kCcOrbitalBases = {
    {"cc-pvdz", "cc-pvdz-rifit"},
    {"cc-pvtz", "cc-pvtz-rifit"},
    {"aug-cc-pvdz", "aug-cc-pvdz-rifit"},
    {"aug-cc-pvtz", "aug-cc-pvtz-rifit"},
};

// The bundled orbital bases with no family-specific aux rule: the tier-2 set,
// which the OLD rule refused outright.
const std::vector<std::string> kUncoveredOrbitalBases = {
    "sto-3g",
    "sto-6g",
    "3-21g",
    "6-31g-dstar",
    "6-31g-star",
    "6-311g-dstar",
    "pcseg-1",
};

constexpr std::array<FockBuilderKind, 2> kAllKinds = {FockBuilderKind::kDefault,
                                                      FockBuilderKind::kRiJk};

void ExpectBundledAuxExists(const std::string& auxName) {
    EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(QcxBasisDataDir) / auxName))
        << "aux basis " << auxName << " is not bundled in data/basis";
}

} // namespace

TEST(AuxBasisSelectionTest, Def2FamilyMapsToUniversalJFitForTheDefaultPath) {
    for (const auto& orbital : kDef2OrbitalBases)
    {
        const auto selected = SelectAuxBasis(orbital, FockBuilderKind::kDefault);
        ASSERT_TRUE(selected.has_value()) << orbital;
        EXPECT_EQ(*selected, "def2-universal-jfit") << orbital;
        ExpectBundledAuxExists(*selected);
    }
}

TEST(AuxBasisSelectionTest, Def2FamilyMapsToUniversalJkFitOnlyForFullRiExchange) {
    for (const auto& orbital : kDef2OrbitalBases)
    {
        const auto selected = SelectAuxBasis(orbital, FockBuilderKind::kRiJk);
        ASSERT_TRUE(selected.has_value()) << orbital;
        EXPECT_EQ(*selected, "def2-universal-jkfit") << orbital;
        ExpectBundledAuxExists(*selected);
    }
}

// The -rifit is a J-ONLY fit, so it answers a J request and NOT a JK
// one. The JK request falls through to the universal JK fit instead.
TEST(AuxBasisSelectionTest, CcFamilyAppendsRifitOnlyForAJPureRequest) {
    for (const auto& [orbital, expected] : kCcOrbitalBases)
    {
        const auto jSelected = SelectAuxBasis(orbital, FockBuilderKind::kDefault);
        ASSERT_TRUE(jSelected.has_value()) << orbital;
        EXPECT_EQ(*jSelected, expected) << orbital;
        ExpectBundledAuxExists(*jSelected);

        // The JK arm must NOT hand back the J-only -rifit this arm just
        // returned. Stated as that property rather than as a fixed string,
        // because the fixed string is a corpus fact: it changed when the cc
        // family gained a JK fit, and the property did not.
        const auto jkSelected = SelectAuxBasis(orbital, FockBuilderKind::kRiJk);
        ASSERT_TRUE(jkSelected.has_value()) << orbital;
        EXPECT_NE(*jkSelected, *jSelected) << orbital;
        EXPECT_TRUE(IsJkOptimizedAux(*jkSelected)) << orbital;
        ExpectBundledAuxExists(*jkSelected);
    }
}

// The corpus now carries the cc family's JK fit; before it,
// every name below resolved to the universal def2-universal-jkfit. Each now
// resolves to the vendored Dunning rung at its own cardinal or the NEXT one up,
// and cc-pvdz is the case that shows the direction matters: the vendor's JK-fit
// family starts at triple (BSE 0.12 carries no cc-pVDZ-JKFIT under any name),
// so a DZ orbital basis takes the TZ fit rather than reaching DOWN a cardinal.
TEST(AuxBasisSelectionTest, CcFamilyJkRequestResolvesToTheVendoredDunningRung) {
    const std::vector<std::string> ccBases = {"cc-pvdz", "cc-pvtz", "aug-cc-pvdz", "aug-cc-pvtz"};

    for (const std::string& orbital : ccBases)
    {
        const auto selected = SelectAuxBasis(orbital, FockBuilderKind::kRiJk);
        ASSERT_TRUE(selected.has_value()) << orbital;
        EXPECT_EQ(*selected, "cc-pvtz-jkfit") << orbital;
        ExpectBundledAuxExists(*selected);
    }

    // The arm stops where the vendored rungs stop. A cardinal above triple has
    // no vendored JK fit at or above it, so it falls to the universal fit -
    // which is the honest outcome. Taking the TZ fit for a QZ orbital basis
    // would be a fit SMALLER than the density it has to span, the one direction
    // the "cardinal matching is preferred, never required" rule does
    // not cover.
    const auto uncovered = SelectAuxBasis("cc-pvqz", FockBuilderKind::kRiJk);
    ASSERT_TRUE(uncovered.has_value());
    EXPECT_EQ(*uncovered, "def2-universal-jkfit");
}

// The diffuse arm, both halves of it. Before this, every diffuse
// def2 basis took the NON-diffuse def2-universal-jfit - the second half of the
// named hazard, "a non-diffuse aux for a diffuse orbital
// basis" - and the notice said no diffuse-capable aux was vendored. Now the
// matched fit is taken where it is vendored, and the tier-3 member otherwise.
TEST(AuxBasisSelectionTest, DiffuseDef2ResolvesToADiffuseCapableFit) {
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"def2-tzvpd", "def2-tzvpd-rifit"}, // matched fit, vendored
        {"def2-qzvppd", "def2-qzvppd-rifit"}, // matched fit, vendored
        {"def2-svpd", "def2-qzvppd-rifit"}, // no vendored match: tier 3
        {"def2-tzvppd", "def2-qzvppd-rifit"}, // no vendored match: tier 3
        {"def2-qzvpd", "def2-qzvppd-rifit"}}; // no vendored match: tier 3

    for (const auto& [orbital, expected] : cases)
    {
        const auto selected = SelectAuxBasis(orbital, FockBuilderKind::kDefault);
        ASSERT_TRUE(selected.has_value()) << orbital;
        EXPECT_EQ(*selected, expected) << orbital;
        EXPECT_TRUE(IsDiffuseCapableAux(*selected))
            << orbital << " was handed \"" << *selected << "\", which is not diffuse-capable";
        ExpectBundledAuxExists(*selected);
    }

    // def2-tzvppd is the bundled diffuse base with NO vendored matched fit, so
    // it is the one that proves tier 3 is reachable in a real run rather than
    // only on a name. Both diffuse def2 spellings are bundled orbital bases.
    for (const std::string& orbital : {"def2-tzvpd", "def2-tzvppd"})
    {
        EXPECT_TRUE(std::filesystem::exists(std::filesystem::path(QcxBasisDataDir) / orbital))
            << orbital << " is expected to be a bundled orbital basis";
    }
}

// Tier 3 is a real member now, not a label: the name the rule resolves diffuse
// cases to must be bundled, diffuse-capable, and J-only. The last is not
// cosmetic - a JK fit here would let a diffuse J request reach the exchange
// path through the tier-3 arm.
TEST(AuxBasisSelectionTest, TheTier3MemberIsBundledAndDiffuseCapable) {
    ExpectBundledAuxExists(std::string{kTier3DiffuseAux});
    EXPECT_TRUE(IsDiffuseCapableAux(kTier3DiffuseAux)) << std::string{kTier3DiffuseAux};

    EXPECT_FALSE(IsJkOptimizedAux(kTier3DiffuseAux))
        << "tier 3 must stay a J-only fit so it can never reach a JK request";
}

// The measured limit, stated as a test so it cannot decay into an assumption:
// NO diffuse-capable JK fit is vendored under any name, so the diffuse def2 JK
// path keeps the universal JK fit. If a diffuse JK fit is ever vendored, this
// test fails and the tier-3 JK arm is the thing to write.
TEST(AuxBasisSelectionTest, NoDiffuseJkFitIsVendoredSoTheDiffuseJkPathStaysUniversal) {
    for (const std::string& orbital : {"def2-tzvpd", "def2-tzvppd"})
    {
        const auto selected = SelectAuxBasis(orbital, FockBuilderKind::kRiJk);
        ASSERT_TRUE(selected.has_value()) << orbital;
        EXPECT_EQ(*selected, "def2-universal-jkfit") << orbital;
        EXPECT_FALSE(IsDiffuseCapableAux(*selected)) << orbital;
    }
}

// The old rule refused these with kInvalidArgument. It now defaults
// them, because the refutation showed the refusal's stated ground was false
// and the residual degradation is confined to the two smallest bases.
TEST(AuxBasisSelectionTest, UncoveredFamiliesResolveToTheUniversalFitForEveryBuilderKind) {
    for (const auto& orbital : kUncoveredOrbitalBases)
    {
        for (const FockBuilderKind kind : kAllKinds)
        {
            const auto selected = SelectAuxBasis(orbital, kind);
            ASSERT_TRUE(selected.has_value()) << orbital;
            EXPECT_EQ(*selected,
                      kind == FockBuilderKind::kRiJk ? "def2-universal-jkfit"
                                                     : "def2-universal-jfit")
                << orbital;
            ExpectBundledAuxExists(*selected);
        }
    }
}

// THE HAZARD PIN (the named risk, first half). Sweeping
// every bundled orbital basis against a JK request, the selection must never
// hand back a fit that is not JK-optimized. This is the assertion that would
// fail if the tier table ever grew a J-only arm on the JK kind - the
// substitution the whole rule exists to prevent, and the one the OLD rule
// committed by construction on every cc-* basis.
TEST(AuxBasisSelectionTest, NoSelectionEverHandsAJOnlyFitToAJkRequest) {
    for (const auto& orbital : kAllBundledOrbitalBases)
    {
        const auto selected = SelectAuxBasis(orbital, FockBuilderKind::kRiJk);
        ASSERT_TRUE(selected.has_value()) << orbital;
        EXPECT_TRUE(IsJkOptimizedAux(*selected)) << orbital << " resolved a JK request to \""
                                                 << *selected << "\", which is not a JK fit";
    }
}

// The rule behind this file: "if for some calculations we have no default
// aux basis, set one as default". Every bundled orbital basis resolves for
// both kinds, and the resolved name is bundled.
TEST(AuxBasisSelectionTest, EveryBundledOrbitalBasisResolvesToABundledAux) {
    for (const auto& orbital : kAllBundledOrbitalBases)
    {
        for (const FockBuilderKind kind : kAllKinds)
        {
            const auto selected = SelectAuxBasis(orbital, kind);
            ASSERT_TRUE(selected.has_value()) << orbital;
            ExpectBundledAuxExists(*selected);
        }
    }
}

// The one name-level refusal that survives, FIRED rather than assumed: an
// empty name is not a name, so "always produce a default" has nothing to
// default for. A guard nobody has seen fire is a guard nobody has seen behave.
TEST(AuxBasisSelectionTest, EmptyOrbitalNameIsRefusedByName) {
    for (const FockBuilderKind kind : kAllKinds)
    {
        const auto selected = SelectAuxBasis("", kind);
        ASSERT_FALSE(selected.has_value()) << "an empty orbital basis name must not resolve";
        EXPECT_EQ(selected.error().code, qcx::ErrorCode::kInvalidArgument);
        EXPECT_FALSE(selected.error().message.empty());
    }
}

// The notice must be SILENT where the selection is matched and the region is
// not on the weak list, or it becomes noise a reader learns to skip.
TEST(AuxSelectionNoticeTest, NoticeIsSilentForMatchedSelectionsOutsideWeakRegions) {
    EXPECT_FALSE(AuxSelectionNotice("def2-svp", FockBuilderKind::kDefault).has_value());
    EXPECT_FALSE(AuxSelectionNotice("def2-tzvp", FockBuilderKind::kRiJk).has_value());
    EXPECT_FALSE(AuxSelectionNotice("cc-pvtz", FockBuilderKind::kDefault).has_value());
    EXPECT_FALSE(AuxSelectionNotice("pcseg-1", FockBuilderKind::kDefault).has_value());
    EXPECT_FALSE(AuxSelectionNotice("6-31g-star", FockBuilderKind::kDefault).has_value());
}

TEST(AuxSelectionNoticeTest, NoticeNamesTheWeakRegionForEachListedFamily) {
    const auto sto = AuxSelectionNotice("sto-3g", FockBuilderKind::kDefault);
    ASSERT_TRUE(sto.has_value());
    EXPECT_NE(sto->find("minimal basis"), std::string::npos) << *sto;

    const auto unpolarized = AuxSelectionNotice("3-21g", FockBuilderKind::kDefault);
    ASSERT_TRUE(unpolarized.has_value());
    EXPECT_NE(unpolarized->find("no polarization functions"), std::string::npos) << *unpolarized;

    const auto diffuse = AuxSelectionNotice("aug-cc-pvdz", FockBuilderKind::kDefault);
    ASSERT_TRUE(diffuse.has_value());
    EXPECT_NE(diffuse->find("diffuse-augmented"), std::string::npos) << *diffuse;

    const auto def2Diffuse = AuxSelectionNotice("def2-tzvpd", FockBuilderKind::kDefault);
    ASSERT_TRUE(def2Diffuse.has_value());
    EXPECT_NE(def2Diffuse->find("diffuse-augmented"), std::string::npos) << *def2Diffuse;

    const auto ecp = AuxSelectionNotice("lanl2dz", FockBuilderKind::kDefault);
    ASSERT_TRUE(ecp.has_value());
    EXPECT_NE(ecp->find("ECP/relativistic"), std::string::npos) << *ecp;

    // The second half of the named hazard: a JK request without a matched JK
    // fit. It runs, and it says so.
    //
    // RE-POINTED, not dropped. The case used here was "cc-pvdz",
    // which is no longer in this region: the cc family gained a vendored JK fit
    // (cc-pvtz-jkfit), so a cc-pvdz JK request IS matched now and is silent by
    // design. A cc-pvqz orbital basis is above every vendored JK rung, so it is
    // the name that still exercises this arm - and the assertion that the cc
    // family became silent is pinned separately below, so the re-point narrows
    // nothing.
    const auto unmatchedJk = AuxSelectionNotice("cc-pvqz", FockBuilderKind::kRiJk);
    ASSERT_TRUE(unmatchedJk.has_value());
    EXPECT_NE(unmatchedJk->find("no matched JK fit"), std::string::npos) << *unmatchedJk;

    // Every other family is still unmatched on the JK kind.
    const auto otherFamilyJk = AuxSelectionNotice("sto-3g", FockBuilderKind::kRiJk);
    ASSERT_TRUE(otherFamilyJk.has_value());
    EXPECT_NE(otherFamilyJk->find("no matched JK fit"), std::string::npos) << *otherFamilyJk;
}

// The cc family STOPPED being a warned-on JK region when it gained a vendored
// JK fit. This is the other side of the re-point above, and it is the assertion
// that would fail if the notice kept flagging a now-matched pairing - noise a
// reader learns to skip, which the notice's own doc names as the failure mode.
TEST(AuxSelectionNoticeTest, CcFamilyJkRequestIsSilentNowThatItIsMatched) {
    // The non-diffuse cc names: a matched JK pairing is no longer a warned
    // region at all, so these go quiet.
    for (const std::string& orbital : {"cc-pvdz", "cc-pvtz"})
    {
        EXPECT_FALSE(AuxSelectionNotice(orbital, FockBuilderKind::kRiJk).has_value())
            << orbital << " is a MATCHED JK pairing now and must not be warned about";
    }

    // The aug- names are NOT silent, and that is the correct outcome rather
    // than a leftover. The JK arm hands them cc-pvtz-jkfit, which carries no
    // diffuse functions, so the diffuse region still applies to them. This is
    // the measured limit of the vendor listing - BSE 0.12 has no augmented JK
    // fit under any name - and the notice states it rather than letting a
    // family-matched label read as a diffuse match too.
    for (const std::string& orbital : {"aug-cc-pvdz", "aug-cc-pvtz"})
    {
        const auto notice = AuxSelectionNotice(orbital, FockBuilderKind::kRiJk);
        ASSERT_TRUE(notice.has_value()) << orbital;
        EXPECT_NE(notice->find("NOT diffuse-capable"), std::string::npos) << *notice;
        EXPECT_EQ(notice->find("no matched JK fit"), std::string::npos)
            << orbital << " IS family-matched on the JK kind; only its diffuseness is warned about";
    }
}

// The diffuse clause used to say "no diffuse-capable aux is vendored" for every
// diffuse name. That was false even before this change - aug-cc-pvdz already
// resolved to a diffuse-capable fit under a warning saying none existed - and
// the clause is now decided by what the selection actually returned.
TEST(AuxSelectionNoticeTest, DiffuseClauseReportsTheCaseTheRunIsActuallyIn) {
    // Served: the fit handed back is diffuse-capable, and the notice says so.
    for (const std::string& orbital : {"aug-cc-pvdz", "aug-cc-pvtz", "def2-tzvpd", "def2-tzvppd"})
    {
        const auto notice = AuxSelectionNotice(orbital, FockBuilderKind::kDefault);
        ASSERT_TRUE(notice.has_value()) << orbital;
        EXPECT_NE(notice->find("diffuse-augmented"), std::string::npos) << *notice;
        EXPECT_NE(notice->find("served by a diffuse-capable fit"), std::string::npos) << *notice;
        EXPECT_EQ(notice->find("NOT diffuse-capable"), std::string::npos) << *notice;
    }

    // Not served: a diffuse basis on the JK kind, where the vendor listing
    // carries no diffuse-capable fit at all. The clause must say the opposite,
    // because the run really is in the other case.
    const auto notServed = AuxSelectionNotice("def2-tzvpd", FockBuilderKind::kRiJk);
    ASSERT_TRUE(notServed.has_value());
    EXPECT_NE(notServed->find("NOT diffuse-capable"), std::string::npos) << *notServed;
    EXPECT_EQ(notServed->find("served by a diffuse-capable fit"), std::string::npos) << *notServed;
}

// The notice READS the selected name from the rule instead of restating the
// tier table. This is the regression pin for a real defect caught while
// writing it: an aug-cc-* J request resolves to its own -rifit, and a second
// copy of the table in the message had claimed the universal J-fit instead.
TEST(AuxSelectionNoticeTest, NoticeNamesTheAuxTheSelectionActuallyReturns) {
    // aug-cc-pvdz is the one bundled name that is BOTH diffuse-flagged and
    // resolved through the cc-family arm, so it is the case that separates
    // "the notice restates the tier table" from "the notice reads it".
    const auto selected = SelectAuxBasis("aug-cc-pvdz", FockBuilderKind::kDefault);
    ASSERT_TRUE(selected.has_value());
    EXPECT_EQ(*selected, "aug-cc-pvdz-rifit");

    const auto notice = AuxSelectionNotice("aug-cc-pvdz", FockBuilderKind::kDefault);
    ASSERT_TRUE(notice.has_value());
    EXPECT_NE(notice->find(*selected), std::string::npos)
        << "the notice must name \"" << *selected << "\"; got: " << *notice;
}
