// Semantic validation of the parsed run input. See
// validate_input.hpp for the parser/validator/driver division of labor.

#include "qcx/io/validate_input.hpp"

#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/aux_basis.hpp"
#include "qcx/molecule/elements.hpp"

#include <filesystem>
#include <optional>
#include <string_view>

namespace qcx::io {
namespace {

// The bundled basis data root: the same compile definition the driver and
// the basisset tests use (io/CMakeLists.txt) - never a hardcoded path.
const std::filesystem::path kDataRoot(QcxBasisDataDir);

// Resolves one basis name exactly as the driver does (ParseNwchemDirectory
// on the bundled data root). Returns the issue message when the name does
// not resolve; std::nullopt when it does.
std::optional<std::string> UnresolvedBasisIssue(std::string_view name) {
    const auto basis =
        qcx::basisset::ParseNwchemDirectory((kDataRoot / std::string(name)).string());

    if (basis.has_value())
    {
        return std::nullopt;
    }

    return "cannot load basis \"" + std::string(name) + "\": " + basis.error().message;
}

// The closed-shell multiplicity rule of one method word, carried by a method
// whose restricted density has no spin polarization (so a multiplicity above
// one would be run as the singlet state and reported as the state that was
// asked for): the method's own word, and the open-shell-native method an
// open-shell request belongs to instead.
struct ClosedShellRule {
    std::string_view word; ///< The [method] word the rule is for.
    std::string_view openShellWord; ///< Where an open-shell request belongs.
};

// The exhaustiveness guard around the classifier below. The diagnostics have
// to be promoted here to be worth anything: MSVC emits C4062/C4061 for an
// unhandled enumerator at neither /W3 nor /W4 (both are off-by-default), GCC
// emits nothing without -Wall, and Clang's -Wswitch is a warning nobody
// reads. C4061 and -Wswitch-enum are the variants that also catch a
// `default:` arm added to silence the check. Only this function is in the
// region; every other switch in the translation unit keeps the project's
// default diagnostic settings.
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

// The spin rule of one method word, or nullopt for a method that carries the
// unpaired electrons natively (no multiplicity this layer could restrict).
//
// This is the ONE place where a method word's spin rule is decided: the
// closed-shell refusal below reads nothing else. The switch is exhaustive BY
// CONSTRUCTION - no `default:` arm, and the region above makes an unhandled
// enumerator a BUILD FAILURE - so a slice that adds a MethodType enumerator
// cannot leave it unclassified; the compiler stops it. The equality tests
// this replaced could not: `method == kRhf` answers false for every word
// nobody had thought about yet, and false means "no restriction" here, so a
// future closed-shell method would have run an open-shell request as the
// singlet state with nothing downstream able to tell the two apart.
std::optional<ClosedShellRule> ClosedShellRuleFor(MethodType method) {
    switch (method)
    {
    case MethodType::kRhf:
        return ClosedShellRule{"rhf", "uhf"};

    // The unrestricted lanes have no rule to state: the spin polarization
    // their density carries is what an open-shell request needs. They name
    // the answer explicitly instead of inheriting it from a test that
    // happens to answer false for them.
    case MethodType::kUhf:
    case MethodType::kUks:
        return std::nullopt;

    case MethodType::kRks:
        return ClosedShellRule{"rks", "uks"};
    }

    // Reached only by a value no enumerator names (a programmatic caller's
    // out-of-range cast): this layer has no method rule to apply and stays
    // silent on purpose - the driver's own total refusal (ResolveScfPath,
    // the single exhaustive method dispatch) reports that value by number
    // with the words this build runs, so the caller reads one refusal rather
    // than two. DriverErrorTest's unclassified-method row pins that message
    // through RunDriver, which validates first.
    return std::nullopt;
}

// The method word of one MethodType, for refusals that must name what the
// author actually wrote. Exhaustive for the same reason as the classifier
// below: a new enumerator fails the BUILD here rather than putting a number or
// an empty word into a message the reader has to decode.
std::string_view MethodWord(MethodType method) {
    switch (method)
    {
    case MethodType::kRhf:
        return "rhf";

    case MethodType::kUhf:
        return "uhf";

    case MethodType::kRks:
        return "rks";

    case MethodType::kUks:
        return "uks";
    }

    return "unknown";
}

// Which optional Kohn-Sham keys one method word admits.
//
// This is the ONE place a method word's Kohn-Sham key policy is decided, and
// the switch is exhaustive BY CONSTRUCTION - no `default:` arm, with the
// region's unhandled-enumerator diagnostics promoted to errors above, so a
// slice that adds a MethodType enumerator cannot leave it unclassified. The
// list this replaced would have: a list answers "takes the keys" for every
// word nobody has thought about yet, so the refusal would silently not fire -
// the same equality-selection weakness that let an unwired method run
// Hartree-Fock under a DFT label.
//
// The policy is a pure function of the method word, which is why it is a
// switch here rather than a condition at the call site.
struct XcKeyPolicy {
    bool takesFunctional = false;
    bool takesScreeningTolerance = false;
    bool takesXcGrid = false;
};

std::optional<XcKeyPolicy> XcKeyPolicyFor(MethodType method) {
    switch (method)
    {
    // The Hartree-Fock lanes take no key of this family. A `functional` on one
    // of them means the author expected a DFT run, and running HF under that
    // label is the failure the kRks/kUks refusal exists to prevent; the
    // tolerance and the `[grid]` block are refused with it so all three share
    // one rule rather than presenting a reader with an arbitrary asymmetry
    // (the grid belongs to the family: an rhf/uhf run that wrote a
    // quadrature would silently run the engine's default one, and the record's
    // `xc_grid` - which exists precisely so a grid is never unnamed - would be
    // absent exactly where the file had asked). Refused, never ignored (the
    // no-silent-substitution posture).
    case MethodType::kRhf:
    case MethodType::kUhf:
        return XcKeyPolicy{false, false, false};

    case MethodType::kRks:
    case MethodType::kUks:
        return XcKeyPolicy{true, true, true};
    }

    // A value no enumerator names (a programmatic caller's out-of-range cast):
    // this layer has no key policy to apply, so it applies none - returning a
    // refusing policy here would make the comment above false and would double
    // the driver's own total refusal for that value. Same shape and same reason
    // as ClosedShellRuleFor.
    return std::nullopt;
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__clang__)
#pragma clang diagnostic pop
#else
#pragma GCC diagnostic pop
#endif

} // namespace

ValidationReport ValidateInput(const RunInput& input) {
    ValidationReport report;

    int sumZ = 0;
    bool allSymbolsKnown = true;

    for (std::size_t i = 0; i < input.molecule.atoms.size(); ++i)
    {
        const auto& atom = input.molecule.atoms[i];
        const auto* element = qcx::molecule::FindElement(atom.symbol);

        if (element == nullptr)
        {
            report.issues.push_back("molecule.atoms[" + std::to_string(i) +
                                    "]: unknown element symbol \"" + atom.symbol + "\"");
            allSymbolsKnown = false;
            continue;
        }

        sumZ += element->atomicNumber;
    }

    // The electron-count checks are only meaningful when every symbol
    // resolved; the symbol errors above are the actionable ones.
    if (allSymbolsKnown)
    {
        const int electronCount = sumZ - input.molecule.charge;

        if (input.molecule.charge > sumZ)
        {
            report.issues.push_back("molecule: charge " + std::to_string(input.molecule.charge) +
                                    " exceeds the sum of atomic numbers " + std::to_string(sumZ));
        } else if (electronCount < 1)
        {
            report.issues.push_back(
                "molecule: no electrons (charge equals the sum of atomic numbers)");
        } else if ((input.molecule.multiplicity % 2 == 1) != (electronCount % 2 == 0))
        {
            report.issues.push_back(
                "molecule: multiplicity " + std::to_string(input.molecule.multiplicity) +
                " is incompatible with " + std::to_string(electronCount) +
                " electrons (multiplicity and electron count must have opposite parity)");
        }
    }

    // The RHF lane is closed-shell by construction (rhf.cpp fills
    // numOccupied = ElectronCount() / 2 and never consults the
    // multiplicity), and the restricted Kohn-Sham lane is closed-shell for
    // the same reason: the restricted density has no spin polarization to
    // carry the unpaired electrons. Either one with multiplicity != 1 would
    // run the singlet state and report it as correct - the parity check
    // above lets such a request with an even electron count through, which
    // is exactly the silent-wrong-spin-state case. Reject it here (the
    // unrestricted lanes handle M > 1 natively). The rule comes from the
    // classifier above, so the word and its open-shell counterpart are
    // stated once, in the arm that states the rule.
    if (const auto closedShell = ClosedShellRuleFor(input.method.method);
        closedShell.has_value() && input.molecule.multiplicity != 1)
    {
        report.issues.push_back(
            "molecule: multiplicity " + std::to_string(input.molecule.multiplicity) +
            " is incompatible with method " + std::string(closedShell->word) + " (" +
            std::string(closedShell->word) + " is closed-shell only; use " +
            std::string(closedShell->openShellWord) + ")");
    }

    // The Kohn-Sham keys on a method word that does not take them.
    // The policy comes from the exhaustive classifier above, so the
    // word and its answer are stated once, in the arm that states the rule.
    // The keys are REFUSED rather than ignored: a `functional` the run silently
    // drops would leave the document asking for one method and the run
    // performing another - the failure mode this whole slice exists to prevent,
    // and the same reason the kRks/kUks words are refused until the run path is
    // wired rather than run as Hartree-Fock. A nullopt policy is a method value
    // no enumerator names: nothing to apply here, the driver reports it.
    const std::optional<XcKeyPolicy> xcKeys = XcKeyPolicyFor(input.method.method);

    if (xcKeys.has_value() && !xcKeys->takesFunctional && input.method.functional.has_value())
    {
        report.issues.push_back(
            "method.functional: " + std::string(MethodWord(input.method.method)) +
            " does not take a functional (the Kohn-Sham methods rks and uks do; "
            "remove the key or change method.type)");
    }

    if (xcKeys.has_value() && !xcKeys->takesScreeningTolerance &&
        input.method.screeningTolerance.has_value())
    {
        report.issues.push_back(
            "method.screening_tolerance: " + std::string(MethodWord(input.method.method)) +
            " does not take a screening tolerance (nothing is evaluated on a grid; the "
            "Kohn-Sham methods rks and uks do)");
    }

    // The `[grid]` block belongs to the same family: the XC integration
    // grid is built by the Kohn-Sham composition, which is the only caller of
    // XcGridEngine::Create, so a grid key on an rhf/uhf run is a request no
    // path can honour. Refused with its two neighbours, for their reason: a
    // silently dropped quadrature is a run on a grid the file did not ask for,
    // and - since schema 34 discloses the grid a run BUILT - the one request
    // the record could not have disclosed as ignored.
    if (xcKeys.has_value() && !xcKeys->takesXcGrid && input.grid.has_value())
    {
        report.issues.push_back(
            "[grid]: " + std::string(MethodWord(input.method.method)) +
            " does not take an XC integration grid (no density functional is integrated, so "
            "nothing is evaluated on a grid; the Kohn-Sham methods rks and uks do)");
    }

    if (auto issue = UnresolvedBasisIssue(input.basis.orbital); issue.has_value())
    {
        report.issues.push_back("basis.orbital: " + *issue);
    }

    // The aux basis is validated when the run puts one in effect, and v1 has
    // exactly TWO such combinations, both RHF: ri_j_link (RI-J with a direct
    // exchange half) and ri_jk (the composed full-RI builder). Both are
    // explicit-only opt-ins: the
    // no-key size ladder (selection_resolution.hpp) resolves to the direct
    // family's lean member at every size and never picks either, so a run with
    // no family word is never a consumer. The absent key AND the within-family
    // "lean" word both resolve to the direct family, whose members wire no
    // aux. A PRESENT aux name is therefore validated exactly when the run can
    // consume it - one rule for every spelling, so explicit "direct" and
    // explicit "lean" (the same family, one carrying a slot word and one not)
    // are treated alike, and an inert aux name stays inert. Nothing here
    // REQUIRES an aux: each wiring refuses its own missing aux with its own
    // naming message, not the static validation. The combination rules fire
    // first for the not-yet-wired ones.
    //
    // What this predicate is, and what it is deliberately not: it is not a
    // dispatch over MethodType, and it
    // is not written as one. The question it answers - does this run put an
    // aux basis in effect - belongs to the builder the driver RESOLVES
    // (ResolveBuilderSelection: size, topology and device dependent), and the
    // driver's own single place for it, AuxNameInEffect (run_driver.cpp), is a
    // function of the builder kind alone. What this line covers is exactly the
    // reachable set: ValidateCombination refuses every UHF builder except
    // direct and qfmm, so the RHF ri_j_link and ri_jk requests are the only
    // combinations that can reach an aux consumer, and this rule and the
    // driver's agree there. A method-only switch cannot express that rule, and
    // a builder-only switch would be worse than this line: ValidateInput runs
    // BEFORE ValidateCombination, so validating the aux for uhf + ri_j_link
    // would replace the driver's actionable combination refusal with a
    // basis.aux message about a name no run reads.
    //
    // ri_jk became the second reachable consumer, which this
    // predicate's own note predicted and required ("If such a consumer becomes
    // reachable, this predicate has to widen with it, and
    // tests/validate_input_test.cpp pins the boundary from the outside"). The
    // widening is therefore not a new rule, it is the sentence above being
    // applied - and the pin moved with it, as it said it would.
    const bool usesAux = input.method.method == MethodType::kRhf &&
                         input.method.builder.has_value() &&
                         (*input.method.builder == BuilderKind::kRiJLink ||
                          *input.method.builder == BuilderKind::kRiJk);
    // The full-RI exchange request carries a second rule the ri_j_link one
    // does not: the rule refuses it when the aux IN EFFECT is not a
    // JK-optimized fit, because occ-RI-K contracts the exchange through the
    // same (uv|P) tensor and a J-only fit lands it in a ~10x worse error
    // class. The predicate is over the resolved NAME (integrals
    // IsJkOptimizedAux), not over the orbital prefix and not over the kind
    // asked for - see that function for why both alternatives were ruled out.
    const bool isJkRequest = usesAux && *input.method.builder == BuilderKind::kRiJk;

    if (usesAux)
    {
        std::optional<std::string> auxName;

        if (input.basis.aux.has_value())
        {
            if (input.basis.aux->empty())
            {
                report.issues.push_back("basis.aux: must be a non-empty basis name when present");
            } else
            {
                auxName = *input.basis.aux;
            }
        } else
        {
            auto selected = qcx::integrals::SelectAuxBasis(
                input.basis.orbital,
                isJkRequest ? qcx::integrals::FockBuilderKind::kRiJk
                            : qcx::integrals::FockBuilderKind::kDefault);

            if (!selected.has_value())
            {
                report.issues.push_back("basis.aux: no auto-selected aux basis for \"" +
                                        input.basis.orbital + "\"; specify [basis].aux explicitly");
            } else
            {
                auxName = *selected;
            }
        }

        if (auxName.has_value())
        {
            if (auto issue = UnresolvedBasisIssue(*auxName); issue.has_value())
            {
                report.issues.push_back("basis.aux: " + *issue);
            }

            if (isJkRequest && !qcx::integrals::IsJkOptimizedAux(*auxName))
            {
                report.issues.push_back("basis.aux: " +
                                        qcx::integrals::JkOptimizedAuxRefusal(*auxName));
            }
        }
    }

    return report;
}

} // namespace qcx::io
