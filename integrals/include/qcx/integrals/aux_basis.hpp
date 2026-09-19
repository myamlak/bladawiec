#pragma once

/// \file
/// Automatic selection of the RI auxiliary basis family for a bundled
/// orbital basis. The RI engines need an aux basis; the driver can default
/// one from the orbital basis name instead of requiring the user to name an
/// aux set explicitly.
///
/// The rule is QUALITY-TIERED, not provenance-based. See SelectAuxBasis for
/// the tiers and AuxSelectionNotice for the weak regions that warn but still
/// run.

#include "qcx/error.hpp"

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace qcx::integrals {

/// Which Fock builder the aux basis will feed: the aux family choice depends
/// on whether exchange also comes from RI. The
/// default production path is RI-J + direct/screened LinK exchange, which
/// needs the J-only fit; the full-RI exchange path needs the J+K fit. An
/// unused K-fitting block is harmless numerically (it just sits unused in
/// the aux basis), so this distinction exists to avoid silently loading a
/// larger-than-needed aux set on the default path, not to prevent a wrong
/// answer.
///
/// The distinction is load-bearing in the other direction too, and that is
/// the half that matters: a J-only fit reaching an exchange path is a ~10x
/// worse error class, which is why kRiJk resolves to a JK fit and never to a
/// J fit - see IsJkOptimizedAux, the predicate that refuses the pairing if it
/// ever arrives by another road.
/// \ingroup qcx-integrals
enum class FockBuilderKind {
    kDefault, ///< The default production path: RI-J + direct LinK exchange.
    kRiJk, ///< Full RI for both Coulomb and exchange (RiJkFockBuilder).
};

/// True when \p name ends with \p suffix. Spelled with rfind rather than the
/// standard ends_with because the project's static checks allow starts_with
/// on a string_view but not ends_with (the same reason IsJkOptimizedAux
/// spells its own test this way).
/// \param name The name to test.
/// \param suffix The suffix to test for.
/// \returns True when the name ends with the suffix.
/// \ingroup qcx-integrals
inline bool AuxNameEndsWith(std::string_view name, std::string_view suffix) {
    return name.size() >= suffix.size() && name.rfind(suffix) == name.size() - suffix.size();
}

/// Selects the bundled auxiliary basis directory for an orbital basis name.
///
/// **The rule is a quality tier, not a provenance test.** Until 2026-09-13
/// an earlier rule refused every orbital basis outside the `def2-*` and
/// `cc-*` families, on the stated ground that pairing, say, STO-3G with a
/// def2 fit "would run but produce a numerically nonsensical RI
/// approximation - the silent-wrong-answer failure mode the screening/
/// tolerance discipline exists to avoid". **That ground is REFUTED by
/// measurement**, so the tiered rule replaces it, and it ALWAYS produces a
/// default rather than leaving a run with no aux basis at all.
///
/// **CORRECTION, 2026-09-13 (measurement).** The refuted claim is corrected
/// here rather than deleted, because the paragraph was wrong in its
/// conclusion and partly right in its observation. Measured, H2O at one
/// geometry, `ri_j_link` against the exact `direct` path,
/// `def2-universal-jfit` held fixed:
/// - **"would run" - TRUE.** 7 of 7 previously-refused bases converge.
/// - **"numerically nonsensical" - REFUTED.** Every error is NEGATIVE
///   (E_RI - E_exact = -1/2 (J_exact - J_RI)), which is the variational
///   signature of a CORRECT fit, not of a broken one. Five of the seven land
///   within 1.6x of the supported control `def2-svp` (-1.092e-04 Eh), and
///   `6-31g-star` is BETTER than it (0.97x). The strongest single counter-fact
///   is this repo's own committed test, which builds STO-3G with a
///   `def2-universal-jkfit` as the module's RI instrument anchor
///   (`integrals/tests/ri_full_fock_test.cpp:743-752`).
/// - **RESIDUAL, and it is not nothing: the two smallest bases DO degrade.**
///   sto-3g sits at **6.99x** the control and sto-6g at **2.65x**, and no
///   larger set is available to close that gap within the vendored universe.
///   Those two rows are the reason AuxSelectionNotice exists: they are warned
///   about, not refused, and the measured magnitude is reported rather than
///   rounded away.
/// - **SCOPE, stated in the same sentence as the conclusion.** These numbers
///   are RI-J total-energy errors on one small molecule. They say nothing
///   about RI-K or RI-JK (a separate, ~10x worse class), about density-
///   sensitive properties, about gradients, or about diffuse and ECP bases -
///   and RI-K/JK and diffuse bases are the regions most likely to REVERSE
///   this conclusion. The rule below is therefore a default with warnings,
///   explicitly not a certification.
///
/// The Dunning cardinal letters in ASCENDING order (double, triple, quadruple,
/// quintuple). The \c cc-pVnZ JK-fit family starts at triple: BSE 0.12 carries
/// \c cc-pVTZ-JKFIT, \c cc-pVQZ-JKFIT and \c cc-pV5Z-JKFIT, and **no
/// \c cc-pVDZ-JKFIT under any name** (nor any \c aug-cc-pV*Z-JKFIT). That is a
/// measured property of the vendor, not a choice made here - see
/// \c data/basis/VERSIONS.md, which records the search.
/// \ingroup qcx-integrals
inline constexpr std::string_view kDunningCardinalOrder = "dtqz5";

/// The vendored Dunning JK fits, ascending by cardinal. Tier 1 for the
/// \c cc-* / \c aug-cc-* families' RI-JK and RI-K paths, where before the
/// 2026-09-13 corpus addition the family had **no** matched JK fit and every
/// JK request fell through to the universal \c def2-universal-jkfit. The
/// RI-K error class is the one this arm exists for: 5.82x the RI-J class on
/// the same fixture and aux.
/// \ingroup qcx-integrals
inline constexpr std::array<std::string_view, 1> kVendoredCcJkfits = {"cc-pvtz-jkfit"};

/// The vendored diffuse-capable def2 fits. Each is the matched fit for the
/// def2 basis of the same name minus its \c -rifit suffix, and each carries
/// the extra diffuse shell block the non-diffuse def2 fits lack (verified
/// against BSE 0.12: \c def2-TZVPD-RIFIT is 92 functions on oxygen against
/// \c def2-TZVP-RIFIT's 76, the difference being its trailing
/// \c s1 p1 d1 f1 diffuse block).
/// \ingroup qcx-integrals
inline constexpr std::array<std::string_view, 2> kVendoredDef2DiffuseRifits = {"def2-tzvpd-rifit",
                                                                               "def2-qzvppd-rifit"};

/// Tier 3's member: the largest vendored diffuse-capable fit. It serves the
/// diffuse def2 spellings that have no vendored matched fit (\c def2-svpd,
/// \c def2-tzvppd, \c def2-qzvpd), which before this member existed were handed
/// the NON-diffuse universal J-fit - the pairing this tier exists to prevent
/// (a non-diffuse aux for a diffuse orbital basis). It is a \c -rifit, so it
/// is J-only by construction and can never reach a JK request.
/// \ingroup qcx-integrals
inline constexpr std::string_view kTier3DiffuseAux = "def2-qzvppd-rifit";

/// Whether an orbital basis NAME is diffuse-augmented, by the two vendor
/// conventions this listing uses: the Dunning \c aug- prefix, and the def2
/// diffuse \c d suffix (def2-SVPD, def2-TZVPD, def2-QZVPPD). Both are the
/// vendors' own spellings.
///
/// Shared by SelectAuxBasis and AuxSelectionNotice on purpose: the two must
/// agree on which region a name is in, and a second copy of this test is
/// exactly how they would stop agreeing.
/// \param orbitalBasisName The orbital basis name to test.
/// \returns True when the name is diffuse-augmented.
/// \ingroup qcx-integrals
inline bool IsDiffuseOrbitalBasis(std::string_view orbitalBasisName) {
    return orbitalBasisName.starts_with("aug-") ||
           (orbitalBasisName.starts_with("def2-") && AuxNameEndsWith(orbitalBasisName, "d"));
}

/// Whether a RESOLVED auxiliary basis name is a diffuse-capable fit - the
/// predicate the diffuse-region notice keys on, so it can stop claiming no
/// diffuse-capable aux exists once the selection has in fact handed one back.
/// The Dunning \c -rifit sets carry their orbital set's \c aug- prefix, so the
/// prefix survives into the fit name and a prefix test is enough for them. The
/// def2 diffuse fits have no such marker: \c def2-tzvpd-rifit is told from
/// \c def2-tzvp-rifit only by the cardinal letter inside the name, so the
/// vendored set is listed explicitly here rather than inferred from a suffix.
/// \param auxBasisName The resolved auxiliary basis directory name.
/// \returns True when the fit is diffuse-capable.
/// \ingroup qcx-integrals
inline bool IsDiffuseCapableAux(std::string_view auxBasisName) {
    if (auxBasisName.starts_with("aug-"))
    {
        return true;
    }

    for (const std::string_view diffuseRifit : kVendoredDef2DiffuseRifits)
    {
        if (auxBasisName == diffuseRifit)
        {
            return true;
        }
    }

    return false;
}

/// The vendored Dunning JK fit a \c cc-* / \c aug-cc-* orbital basis resolves
/// to for a JK request, or nothing when no vendored rung covers it.
///
/// The mapping is by CARDINAL NUMBER, and it takes the smallest vendored rung
/// at or above the orbital basis's own, because the JK-fit family starts at
/// triple: \c cc-pvdz has no JDZ fit to match, so it gets the TZ one. A
/// \c cc-pvqz orbital basis finds no vendored rung above triple and reports
/// nothing, so the caller falls through to tier 2 - which is the honest
/// outcome, not a silent cardinal mismatch in the other direction (a fit
/// SMALLER than the orbital basis is the one direction the rule's
/// "cardinal matching is preferred, never required" does not cover).
/// \param orbitalBasisName The orbital basis name to map.
/// \returns The vendored JK fit name, or nothing when uncovered.
/// \ingroup qcx-integrals
inline std::optional<std::string_view> DunningJkfitRung(std::string_view orbitalBasisName) {
    // The aug- prefix selects the diffuse VARIANT of a set whose compact part
    // keeps the same cardinal, so it is skipped rather than matched: there is
    // no aug-cc-pV*Z-JKFIT in the vendor listing to match against. Written as a
    // substr offset rather than string_view::remove_prefix, whose name carries
    // an underscore and is not on the allowed library-token list (the same
    // constraint AuxNameEndsWith documents above).
    const std::string_view core =
        orbitalBasisName.starts_with("aug-") ? orbitalBasisName.substr(4) : orbitalBasisName;

    // "cc-pv<cardinal>z" - the cardinal letter sits at a fixed offset.
    if (!core.starts_with("cc-pv") || core.size() < 7)
    {
        return std::nullopt;
    }

    const std::size_t wanted = kDunningCardinalOrder.find(core[5]);

    if (wanted == std::string_view::npos)
    {
        return std::nullopt;
    }

    for (const std::string_view rung : kVendoredCcJkfits)
    {
        const std::size_t available = kDunningCardinalOrder.find(rung[5]);

        if (available != std::string_view::npos && available >= wanted)
        {
            return rung;
        }
    }

    return std::nullopt;
}

/// The tiers:
/// \li **Tier 1 - dedicated matched aux.** \c def2-* orbital sets share one
///     universal fit for the whole family: \c def2-universal-jfit for RI-J and
///     \c def2-universal-jkfit when \p kind is FockBuilderKind::kRiJk - EXCEPT
///     that a diffuse def2 basis takes its own vendored diffuse-capable fit,
///     \c &lt;name&gt;-rifit, where that fit is vendored (def2-tzvpd, def2-qzvppd).
///     The \c cc-* / \c aug-cc-* sets each have their own matching J-only
///     \c -rifit set, whose name is the orbital name with \c -rifit appended
///     (any \c aug- prefix is preserved by construction); for a JK request they
///     take the vendored Dunning JK fit at their own cardinal or the next one
///     up (DunningJkfitRung). **A \c cc-* orbital basis above the vendored JK
///     rungs still has no dedicated JK fit**, and that pairing falls through to
///     tier 2 rather than reaching for the J-only \c -rifit - the substitution
///     this rule exists to prevent. AuxSelectionNotice reports the
///     fall-through.
/// \li **Tier 2 - universal default.** Every other orbital basis (sto-3g,
///     3-21g, 6-31g-star, pcseg-1, ...) resolves to \c def2-universal-jfit for
///     RI-J and \c def2-universal-jkfit for RI-JK/K. The universal J-set is
///     the right default for arbitrary orbital bases: aux quality dominates
///     family match by roughly an order of magnitude, and cross-family
///     cardinal-number mapping is ambiguous, so cardinal matching is
///     PREFERRED where a dedicated fit exists, never REQUIRED.
/// \li **Tier 3 - larger universal aux for diffuse or high-accuracy cases.**
///     Its member is \c def2-qzvppd-rifit (kTier3DiffuseAux), the largest
///     vendored diffuse-capable fit. A diffuse orbital basis whose own matched
///     fit is not vendored is served by it rather than by the non-diffuse
///     universal J-fit - the hazard this tier exists to prevent, and the reason
///     this tier is not left as a label with no member. It is J-only: for a JK
///     request the tier-3 arm does not apply, because **no diffuse-capable JK
///     fit is vendored under any name**, and that is a measured limit of the
///     vendor listing rather than a gap this rule can close.
///
/// Comparison is case-sensitive: names come from the data/basis directory
/// listing, which is lowercase.
///
/// \param orbitalBasisName A bundled orbital basis directory name
/// (e.g. "def2-svp", "aug-cc-pvdz").
/// \param kind Which Fock builder the aux basis will feed.
/// \returns The aux basis directory name. The error arm fires only for an
/// EMPTY orbital basis name, which names no family and no cardinal number and
/// so cannot be given a default; every name the input parser can produce
/// resolves. The fit-quality refusals this function no longer owns do not
/// disappear - they move to the site that can judge them (a singular fit
/// matrix, or a request whose tolerance even the largest vendored aux cannot
/// meet), which is where the hard refusal belongs.
/// \ingroup qcx-integrals
inline qcx::Result<std::string> SelectAuxBasis(std::string_view orbitalBasisName,
                                               FockBuilderKind kind) {
    // The one name-level refusal that survives: an empty name is not a name,
    // so "always produce a default" has nothing to default FOR. Tested rather
    // than assumed - a guard nobody has seen fire is a guard nobody has seen
    // behave.
    if (orbitalBasisName.empty())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "empty orbital basis name; there is no family or cardinal "
                       "number to match an aux basis against"});
    }

    const bool wantsJkFit = kind == FockBuilderKind::kRiJk;

    // Tier 1a: the def2 family's own universal fit, J or JK as asked - with the
    // diffuse exception. A JK request has no diffuse arm at all: the vendor
    // listing carries no diffuse-capable JK fit, so that request keeps the
    // universal JK fit and the notice reports the region.
    if (orbitalBasisName.starts_with("def2-"))
    {
        if (wantsJkFit)
        {
            return std::string{"def2-universal-jkfit"};
        }

        if (IsDiffuseOrbitalBasis(orbitalBasisName))
        {
            const std::string matched = std::string{orbitalBasisName} + "-rifit";

            for (const std::string_view diffuseRifit : kVendoredDef2DiffuseRifits)
            {
                if (matched == diffuseRifit)
                {
                    return matched;
                }
            }

            return std::string{kTier3DiffuseAux};
        }

        return std::string{"def2-universal-jfit"};
    }

    const bool isCorrelationConsistent =
        orbitalBasisName.starts_with("cc-") || orbitalBasisName.starts_with("aug-cc-");

    // Tier 1b: the Dunning families' per-set -rifit. It is a J-ONLY fit, so it
    // answers a J request and NOT a JK one - a JK request goes to the JK arm
    // below, or to the tier-2 JK fit when no rung covers it. Returning it for
    // both kinds was the old rule, and it is the one pairing the tiered rule
    // exists to prevent.
    if (isCorrelationConsistent && !wantsJkFit)
    {
        return std::string{orbitalBasisName} + "-rifit";
    }

    // Tier 1c: the Dunning families' JK fit, at the orbital basis's own
    // cardinal or the next one up. This arm is what keeps a cc-* RI-JK request
    // off the universal fit where a same-family JK fit exists.
    if (isCorrelationConsistent)
    {
        const auto rung = DunningJkfitRung(orbitalBasisName);

        if (rung.has_value())
        {
            return std::string{*rung};
        }
    }

    // Tier 2: the universal default, for every uncovered basis and for a JK
    // request on a family with no dedicated JK fit.
    return wantsJkFit ? std::string{"def2-universal-jkfit"} : std::string{"def2-universal-jfit"};
}

/// The weak-region notice for an auxiliary basis selection: the text an
/// `ri_j_link` / `ri_jk` run must carry into its record when the selection
/// landed in a region whose fit quality is known to be less reliable. Returns
/// nothing when the selection is in no such region.
///
/// **Warn, do not refuse**: every region below RUNS, and the boundary is the
/// error class rather than the provenance of a name. The hard refusal is
/// reserved for a singular fit matrix or a tolerance even the largest
/// vendored aux cannot meet - neither is decidable from a name, so neither
/// lives here.
///
/// The disclosure obligation, stated where it bites: a notice the run record
/// cannot show is not a warning, it is the silence this contract exists to
/// end. This function RETURNS the text; the resolution point that owns the
/// record is responsible for writing it there (NOT stderr-only). The record's
/// `resources_resolved.selection.warning` is NOT that place - it has a single
/// producer, the device-less gpu fallback, so a second writer would clobber
/// one of the two texts (the member's own note, `result_json.hpp`: the member
/// name is the citation). The repo's own answer to a second writer, taken
/// twice already (`riTensorMode`, `riChunkBytes`), is a SIBLING block on
/// `resources_resolved` - and that block is what this notice is shaped to
/// feed.
///
/// **CORRECTION, 2026-09-13.** The diffuse clause used
/// to read "no diffuse-capable aux is vendored (tier 3)" for EVERY diffuse
/// name. That was false as written - `aug-cc-pvdz` already resolved to
/// `aug-cc-pvdz-rifit`, a diffuse-capable fit, under a warning saying no such
/// fit existed - and it became false for the rest of the region when the
/// diffuse def2 fits and the tier-3 member landed. The clause is now decided by
/// IsDiffuseCapableAux on the RESOLVED name, so it reports which of the two
/// cases the run is actually in. The region itself is still warned either way:
/// a diffuse orbital basis is a region RI serves less reliably, and the notice
/// is not a certification that a diffuse-capable fit makes it safe (diffuse
/// bases are one of the two regions most likely to reverse the RI-J finding
/// above).
///
/// What is NOT implemented, named rather than hidden: a "basis without
/// polarization" test is not decidable from a name in general, so this
/// function does not guess at one outside the Pople family, where the
/// polarization marker IS the vendor's own convention. Guessing there would
/// warn on correct runs, which is worse than a named gap.
/// \param orbitalBasisName The orbital basis the selection was made for.
/// \param kind Which Fock builder the aux basis will feed.
/// \returns The notice text, or nothing when the selection is in no weak
/// region.
/// \ingroup qcx-integrals
inline std::optional<std::string> AuxSelectionNotice(std::string_view orbitalBasisName,
                                                     FockBuilderKind kind) {
    // The selected name is READ from the rule rather than restated here, so
    // the notice cannot name a different aux than the selection returns - the
    // `aug-cc-*` path resolves to its own `-rifit` under a J request, and a
    // second copy of the tier table in this message would have got that wrong.
    // It is read BEFORE the regions are built because one of them now depends
    // on where the selection landed: see the diffuse clause.
    const auto selected = SelectAuxBasis(orbitalBasisName, kind);

    if (!selected.has_value())
    {
        return std::nullopt;
    }

    std::string regions;

    // The Pople split-valence marker: the family spells polarization with `*`,
    // and this listing spells it out as `-star` (and `-dstar`, which contains
    // it). A Pople name carrying neither is one without polarization. Scoped to
    // the family on purpose - the same test applied to arbitrary names would
    // warn on sets that do polarize.
    const bool isPopleSplitValence =
        orbitalBasisName.size() >= 2 && orbitalBasisName.front() >= '0' &&
        orbitalBasisName.front() <= '9' && orbitalBasisName.find('g') != std::string_view::npos;
    const bool isPolarizedPople = orbitalBasisName.find('*') != std::string_view::npos ||
                                  orbitalBasisName.find("star") != std::string_view::npos;

    if (orbitalBasisName.starts_with("sto-"))
    {
        regions += "minimal basis;";
    }

    if (isPopleSplitValence && !isPolarizedPople)
    {
        regions += "no polarization functions;";
    }

    // Diffuse. The region is a WEAK one either way, but the notice now says
    // which of the two it is, because the remedy differs and the old text
    // claimed the second case unconditionally. Corrected 2026-09-13: the old
    // clause read "no diffuse-capable aux is vendored", which was false even
    // then for `aug-cc-pvdz` - the selection handed back `aug-cc-pvdz-rifit`,
    // a diffuse-capable fit, under a warning saying none existed.
    if (IsDiffuseOrbitalBasis(orbitalBasisName))
    {
        if (IsDiffuseCapableAux(*selected))
        {
            regions += "diffuse-augmented basis, served by a diffuse-capable fit;";
        } else
        {
            regions += "diffuse-augmented basis, and the selection is NOT diffuse-capable;";
        }
    }

    // ECP / relativistic: the families that carry a core potential, each by
    // its own published name prefix.
    if (orbitalBasisName.starts_with("lanl") || orbitalBasisName.starts_with("sdd") ||
        orbitalBasisName.starts_with("x2c") || orbitalBasisName.starts_with("dyall") ||
        orbitalBasisName.starts_with("ano-rcc") || AuxNameEndsWith(orbitalBasisName, "-pp"))
    {
        regions += "ECP/relativistic basis;";
    }

    // A JK request on a family with no matched JK fit: the universal JK fit is
    // a DEFAULT, not a match, and this pairing - a J-fit silently reused for K
    // - is the risk the whole rule exists to prevent. It runs, and it says so.
    // "Matched" is decided by the same
    // predicate SelectAuxBasis chooses by, so the cc family stops being flagged
    // exactly when it gains a vendored JK rung.
    const bool hasMatchedJkFit =
        orbitalBasisName.starts_with("def2-") || DunningJkfitRung(orbitalBasisName).has_value();

    if (kind == FockBuilderKind::kRiJk && !hasMatchedJkFit)
    {
        regions += "RI-JK with no matched JK fit;";
    }

    if (regions.empty())
    {
        return std::nullopt;
    }

    return "orbital basis \"" + std::string{orbitalBasisName} +
           "\" is in a region the RI fit "
           "serves less reliably, and the selection resolved to \"" +
           *selected + "\" for it: " + regions +
           " the run proceeds, the RI error class is reported rather than assumed, and refusal "
           "is reserved for a singular fit matrix or an unmeetable tolerance";
}

/// Whether a RESOLVED auxiliary basis name is a JK-optimized fit - the
/// predicate an `ri_jk` request must pass: "refuse when the aux in effect for
/// the JK kind is not a JK-optimized fit".
///
/// The test is over the NAME, and it is deliberately a suffix test on the fit
/// family rather than a test of the orbital basis's prefix. Both alternatives
/// were ruled out for measured reasons:
/// - An orbital-prefix test (`cc-*` refuses) hard-codes today's vendor
///   listing, so a later `cc-pvdz-jkfit` would keep refusing until someone
///   remembered to edit the string test. This predicate lifts it the moment
///   the name is vendored, with no code change.
/// - A prefix test cannot see the other door into the same defect: an explicit
///   `[basis].aux` override naming a `-rifit` for a `def2-*` orbital base
///   never touches the orbital prefix. A resolved-name predicate catches both
///   doors.
///
/// **This predicate is the guard the rule's named hazard turns on**
/// - "the real risk is ... silently using a J-fit for K" - and it survives
/// the tiered rule unchanged, because it tests fit FAMILY while
/// SelectAuxBasis now tests quality tier. They agree by construction and are
/// deliberately not the same test: SelectAuxBasis CHOOSES, this REFUSES.
///
/// **Note what the auto-selection used to return for a `cc-*` orbital basis,
/// and what it returns now.** An earlier version mapped
/// every `cc-*`/`aug-cc-*` name to `<name>-rifit` for BOTH kinds, so a `cc-*`
/// + FockBuilderKind::kRiJk request resolved to a name this predicate REFUSES
/// - and the refusal was doing the work. Under the tiered rule that pairing
/// resolves to `def2-universal-jkfit`, which this predicate ACCEPTS: the J-fit
/// is no longer produced for a K request at all, so the refusal has become the
/// second line of defence rather than the first. It still owns both explicit
/// doors (an `[basis].aux` naming a `-rifit` under `ri_jk`), which is precisely
/// why the check must run on the resolved name rather than on the kind asked
/// for.
/// \param auxBasisName The resolved auxiliary basis directory name (the value
/// SelectAuxBasis returns, or an explicit `[basis].aux`).
/// \returns True when the name is a JK-optimized fit, false otherwise.
/// \ingroup qcx-integrals
inline bool IsJkOptimizedAux(std::string_view auxBasisName) {
    // The bundled listing carries exactly two fit families: `*-jkfit` (a J+K
    // fit) and `*-jfit` / `*-rifit` (J-only). A JK request needs the first.
    return AuxNameEndsWith(auxBasisName, "-jkfit");
}

/// The refusal message an `ri_jk` request carries when its auxiliary basis in
/// effect is not JK-optimized. Kept beside the predicate so the wording and
/// the test cannot drift apart, and so the message can name its own remedy
/// rather than only the fault - a refusal names the condition that fired.
/// \param auxBasisName The resolved auxiliary basis name that failed
/// IsJkOptimizedAux.
/// \returns The refusal message.
/// \ingroup qcx-integrals
inline std::string JkOptimizedAuxRefusal(std::string_view auxBasisName) {
    // The message describes the TEST that failed, not a claim about the name:
    // a J-fit is the observed case, but an explicitly named aux from outside
    // the bundled families is not a J-fit and must not be told that it is.
    //
    // The remedy no longer needs the orbital basis changed: under the tiered
    // rule a `cc-*` orbital basis with `ri_jk` resolves to the universal JK fit
    // on its own, so an arrival here means the name came from an explicit
    // [basis].aux. Naming the auto-selection as the remedy is therefore the
    // accurate instruction, and it is one the user can act on.
    return "fock_builder = \"ri_jk\" requires a JK-optimized auxiliary basis, and the aux in "
           "effect is \"" +
           std::string{auxBasisName} +
           "\", which is not a JK fit: full-RI exchange built from a J-only fit lands in a ~10x "
           "worse error class than the RI-J class on the same fixture. The remedy is to drop "
           "the [basis].aux override and let the selection "
           "default (def2-* resolves to def2-universal-jkfit, and every other orbital basis "
           "resolves to it too), or to vendor a -jkfit aux for this orbital basis";
}

} // namespace qcx::integrals
