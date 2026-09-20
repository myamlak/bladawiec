// The semantic-validation layer over the parsed run input. The
// messages ARE the contract the CLI prints, so the tests assert them
// verbatim. No SCF work happens here - these run in every mode including
// Debug/ASAN.

#include "qcx/io/parse_input.hpp"
#include "qcx/io/validate_input.hpp"

#include <gtest/gtest.h>
#include <optional>
#include <string>

namespace {

using qcx::io::ValidationReport;

// Parses and validates the fixture TOML. A parse failure is a fixture
// bug: the helper fails the test and returns an empty report, so the
// test's own assertions then fail loudly on it.
ValidationReport Validate(const char* toml) {
    const auto input = qcx::io::ParseRunInput(toml);

    if (!input.has_value())
    {
        ADD_FAILURE() << "fixture must parse: " << input.error().message;
        return {};
    }

    return qcx::io::ValidateInput(*input);
}

constexpr const char* kHeader = R"(
[molecule]
)";

TEST(RunInputValidationTest, ValidH2InputPasses) {
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, ValidAuxAutoSelectionResolves) {
    const auto report = Validate(R"(
[molecule]
atoms = [["C", 0.0, 0.0, 0.0]]
[basis]
orbital = "def2-svp"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, UnknownAtomSymbolReported) {
    const auto report = Validate(R"(
[molecule]
atoms = [["Xx", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0], R"(molecule.atoms[0]: unknown element symbol "Xx")");
}

TEST(RunInputValidationTest, EvenElectronCountNeedsOddMultiplicity) {
    // The parity rule is method-independent, so the fixture runs under uhf:
    // under rhf this input would also trip the RHF closed-shell rule
    // (multiplicity != 1), and this test pins the parity rule alone.
    const auto report = Validate(R"(
[molecule]
multiplicity = 2
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0],
              "molecule: multiplicity 2 is incompatible with 2 electrons "
              "(multiplicity and electron count must have opposite parity)");
}

TEST(RunInputValidationTest, RhfRejectsNonSingletMultiplicity) {
    // The parity check lets RHF + M=3 with an even electron count through
    // (O2 triplet: 16 electrons, multiplicity 3 - opposite parity, valid by
    // the parity rule alone); the RHF lane is closed-shell by construction,
    // so the request must be rejected here instead of silently running the
    // singlet state.
    const auto report = Validate(R"(
[molecule]
multiplicity = 3
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2075]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0],
              "molecule: multiplicity 3 is incompatible with method rhf "
              "(rhf is closed-shell only; use uhf)");
}

TEST(RunInputValidationTest, RksRejectsNonSingletMultiplicity) {
    // The restricted Kohn-Sham lane is closed-shell for the same reason the
    // restricted Hartree-Fock lane is: a restricted density carries no spin
    // polarization, so a triplet request would be run as a singlet and
    // reported as the state that was asked for. The UKS lane is where an
    // open-shell density functional run belongs.
    const auto report = Validate(R"(
[molecule]
multiplicity = 3
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2075]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0],
              "molecule: multiplicity 3 is incompatible with method rks "
              "(rks is closed-shell only; use uks)");
}

TEST(RunInputValidationTest, UhfTripletIsNotAValidationError) {
    // The same molecule under uhf is the intended way to run a triplet; the
    // RHF-only restriction must not leak into the UHF lane.
    const auto report = Validate(R"(
[molecule]
multiplicity = 3
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2075]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, OddElectronCountNeedsEvenMultiplicity) {
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0],
              "molecule: multiplicity 1 is incompatible with 1 electrons "
              "(multiplicity and electron count must have opposite parity)");
}

TEST(RunInputValidationTest, OverchargedMoleculeReported) {
    const auto report = Validate(R"(
[molecule]
charge = 2
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0], "molecule: charge 2 exceeds the sum of atomic numbers 1");
}

TEST(RunInputValidationTest, ZeroElectronsReported) {
    const auto report = Validate(R"(
[molecule]
charge = 1
atoms = [["H", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0], "molecule: no electrons (charge equals the sum of atomic numbers)");
}

TEST(RunInputValidationTest, UnknownOrbitalBasisReported) {
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3gx"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_NE(report.issues[0].find(R"(basis.orbital: cannot load basis "sto-3gx")"),
              std::string::npos);
}

TEST(RunInputValidationTest, UnknownAuxBasisReported) {
    const auto report = Validate(R"(
[molecule]
atoms = [["C", 0.0, 0.0, 0.0]]
[basis]
orbital = "def2-svp"
aux = "x2c-jfit"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_NE(report.issues[0].find(R"(basis.aux: cannot load basis "x2c-jfit")"),
              std::string::npos);
}

TEST(RunInputValidationTest, AuxAutoSelectionCoversEveryOrbitalFamily) {
    // This row used to assert the opposite: that an orbital basis no
    // auto-selection rule covered was REPORTED. That premise is gone - the
    // universal-fallback rule ("the aux basis rule becomes a quality tier")
    // replaced the hard error, so sto-3g, the exact basis this row was
    // written against, now resolves the universal J fit and the run
    // validates.
    //
    // The row is kept rather than deleted because the FACT it now records is the
    // one the old row inverted, and it is load-bearing for the aux predicate's
    // reach: the selection is TOTAL over the bundled families, so the only
    // refusal left in it is the empty name (EmptyAuxNameReported).
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, EmptyAuxNameReported) {
    const auto report = Validate(R"(
[molecule]
atoms = [["C", 0.0, 0.0, 0.0]]
[basis]
orbital = "def2-svp"
aux = ""
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 1u);
    EXPECT_EQ(report.issues[0], "basis.aux: must be a non-empty basis name when present");
}

TEST(RunInputValidationTest, CollectsAllIssuesInOnePass) {
    const auto report = Validate(R"(
[molecule]
atoms = [["Xx", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3gx"
[method]
type = "rhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    ASSERT_EQ(report.issues.size(), 2u);
    EXPECT_NE(report.issues[0].find("unknown element symbol"), std::string::npos);
    EXPECT_NE(report.issues[1].find("basis.orbital: cannot load basis"), std::string::npos);
    // The third issue this row used to collect was the aux auto-selection
    // refusal for the unloadable orbital name. The universal fallback
    // removed that refusal, so the pass now collects two -
    // and the row still holds what it was written for, which is that the pass
    // reports EVERY independent issue rather than stopping at the first.
}

TEST(RunInputValidationTest, AuxIgnoredForDirectBuilder) {
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
aux = "definitely-not-a-basis"
[method]
type = "rhf"
fock_builder = "direct"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, AuxIsInertWithoutAnExplicitRiJLinkRequest) {
    // Under the lean-direct flip the aux has exactly one consumer,
    // the explicit ri_j_link opt-in - the absent key resolves to the direct
    // family (lean at <= 1000 basis functions, the machinery above) and that
    // family wires no aux, so a PRESENT aux name is inert for every spelling
    // that does not name ri_j_link. One rule for all of them: an explicit
    // "direct" and an explicit "lean" (the same family, the second carrying
    // no slot word) are treated alike - the old asymmetry validated the
    // typo for the auto/lean spellings and ignored it for explicit direct.
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
aux = "definitely-not-a-basis"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, TheDefaultFamilyNeverRequiresAnAux) {
    // The no-key default is the direct family at every size, and no member
    // of that family wires an aux - demanding
    // one up front would veto every ordinary run. A run that DOES need an
    // aux names ri_j_link explicitly, and its own wiring refuses a missing
    // aux with the naming message (never this static pass).
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

// The ri_jk request is the aux predicate's SECOND reachable consumer.
// The cells below hold the widened predicate and the JK-fit rule from the
// outside, at the layer that runs FIRST:
// ValidateInput runs ahead of the driver's ValidateCombination, so a rule stated
// here and not in the wiring (or the reverse) is a run that validates and then
// dies - the disagreement this layer split exists to prevent.
TEST(RunInputValidationTest, TheRiJkRequestResolvesTheJkFitAndValidates) {
    // def2-* is the one family whose kRiJk auto-selection is a JK fit, so this
    // cell proves the predicate widened WITHOUT over-refusing. The KIND the aux
    // resolves for is load-bearing: SelectAuxBasis answers
    // def2-universal-jkfit for kRiJk and def2-universal-jfit for kDefault, so a
    // predicate that kept passing kDefault here would refuse this run, and one
    // that passed kRiJk for the link would refuse ordinary ri_j_link runs.
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "def2-svp"
[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, RiJkAutoSelectionAlwaysResolvesAJkFit) {
    // What the aux rule now guarantees ("the aux basis rule becomes a
    // quality tier"), and the reason the JK-fit rule has only ONE door left.
    //
    // The old rule refused an uncovered orbital basis outright and mapped every
    // cc-* name to a -rifit for both kinds. The new rule answers a JK request
    // with a JK fit for EVERY family: the def2 family's own, and the tier-2
    // universal JK fit for everything else - the cc-* -rifit and the uncovered
    // bases included, because -rifit is a J-only fit and now answers a J request
    // only. So the AUTO door can no longer produce a name the rule would refuse, and
    // only an explicit [basis].aux can (the next cell).
    //
    // This cell pins that guarantee from the outside, on the two shapes the old
    // rule handled differently: a cc-* base (whose own fit is J-only) and an
    // uncovered base (which used to be a hard error at all).
    const auto cc = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "cc-pvdz"
[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");
    EXPECT_TRUE(cc.IsValid());

    const auto uncovered = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");
    EXPECT_TRUE(uncovered.IsValid());
}

TEST(RunInputValidationTest, RiJkRefusesAnExplicitAuxThatIsNotAJkFit) {
    // The rule's remaining door, and the one that was always the load-bearing one: a
    // def2-* orbital base, whose own kRiJk mapping IS the JK fit, with a J-only
    // name forced through [basis].aux. An orbital-PREFIX test could not see this
    // at all - the override never touches the prefix - which is why the
    // predicate is over the RESOLVED name.
    //
    // With the tier in place this is the ONLY shape that reaches the
    // predicate: the auto door always answers a JK request with a JK fit.
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "def2-svp"
aux = "def2-universal-jfit"
[method]
type = "rhf"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");
    ASSERT_FALSE(report.IsValid());
    EXPECT_NE(report.issues[0].find("requires a JK-optimized auxiliary basis"), std::string::npos)
        << report.issues[0];
    EXPECT_NE(report.issues[0].find("def2-universal-jfit"), std::string::npos) << report.issues[0];
}

TEST(RunInputValidationTest, RiJkStaysInertForTheUnrestrictedLeg) {
    // What this cell pins NOW: the aux predicate is scoped to the RHF leg
    // (method.type = "rhf"), so a uhf + ri_jk deck raises no basis.aux issue,
    // and the typo'd name below proves the predicate did not fire. The reason
    // the cell was WRITTEN for is gone, and it is worth stating rather than
    // leaving the old one to be read as current: the driver's UHF whitelist
    // refused every builder but direct and qfmm, so no unrestricted run could
    // read the aux at all - it admits ri_j_link and ri_jk now, and this
    // family's unrestricted runner resolves the aux itself (RunRiJkUhfScf ->
    // AuxNameInEffect), so a typo'd name here is read by a run that refuses it
    // later rather than by no run. The predicate's own note
    // (validate_input.cpp) says it has to widen WITH a reachable consumer;
    // until it does, this cell pins that method clause, not a wiring refusal.
    const auto report = Validate(R"(
[molecule]
multiplicity = 3
atoms = [["O", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
aux = "definitely-not-a-basis"
[method]
type = "uhf"
fock_builder = "ri_jk"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, UnwiredCombinationsAreNotValidationErrors) {
    const auto report = Validate(R"(
[molecule]
multiplicity = 3
atoms = [["O", 0.0, 0.0, 0.0]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
fock_builder = "qfmm"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, AuxStaysInertForACombinationTheWiringRefuses) {
    // The boundary the aux predicate deliberately stops at. ValidateInput runs
    // BEFORE the driver's ValidateCombination, and the driver refuses every UHF
    // builder except direct and qfmm - so validating the aux here for
    // uhf + ri_j_link would replace "UHF is wired with fock_builder = \"direct\"
    // and \"qfmm\" in v1" with a basis.aux message about a name no run would
    // read, and an absent aux would raise the auto-selection issue for a run
    // that is refused anyway. This row holds that boundary in place: widening
    // the predicate to the builder half alone (the shape of the driver's own
    // builder-only AuxNameInEffect rule) makes it fail.
    const auto report = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
aux = "definitely-not-a-basis"
[method]
type = "uhf"
fock_builder = "ri_j_link"
accuracy = "kNormal"
)");
    EXPECT_TRUE(report.IsValid());
}

TEST(RunInputValidationTest, HartreeFockRefusesTheKohnShamKeys) {
    // The two method-scoped keys are REFUSED on the
    // Hartree-Fock words, not ignored. A `functional` on an rhf run means the
    // author expected a DFT run, and dropping the key would leave the document
    // asking for one method while the run performs another - the failure the
    // kRks/kUks refusal exists to prevent, one layer earlier. The message is
    // pinned verbatim so a later narrowing of the refusal cannot pass quietly.
    const auto functional = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
functional = "pbe"
)");
    ASSERT_EQ(functional.issues.size(), 1u);
    EXPECT_EQ(functional.issues[0],
              "method.functional: rhf does not take a functional (the Kohn-Sham methods rks and "
              "uks do; remove the key or change method.type)");

    // The tolerance is refused on the other Hartree-Fock word, so the rule is
    // shown to come from the classifier rather than from one word's arm.
    const auto tolerance = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "uhf"
accuracy = "kNormal"
screening_tolerance = 1e-10
)");
    ASSERT_EQ(tolerance.issues.size(), 1u);
    EXPECT_EQ(tolerance.issues[0],
              "method.screening_tolerance: uhf does not take a screening tolerance (nothing is "
              "evaluated on a grid; the Kohn-Sham methods rks and uks do)");

    // And the `[grid]` block, which joins the same family. The XC grid
    // engine is built by the Kohn-Sham composition alone, so a quadrature on an
    // rhf run is a request no path can honour - and the one this validator must
    // not let through silently, because schema 34's `xc_grid` block exists so a
    // grid is never unnamed: absent there, it would read as "no grid was
    // involved", which is true, while the file's own keys vanished without a
    // word.
    const auto grid = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rhf"
accuracy = "kNormal"
[grid]
radial_points = 40
)");
    ASSERT_EQ(grid.issues.size(), 1u);
    EXPECT_EQ(grid.issues[0],
              "[grid]: rhf does not take an XC integration grid (no density functional is "
              "integrated, so nothing is evaluated on a grid; the Kohn-Sham methods rks and uks "
              "do)");
}

TEST(RunInputValidationTest, KohnShamMethodsAcceptTheKohnShamKeys) {
    // The other half of the rule: the keys are legal exactly where they have a
    // consumer. Both Kohn-Sham words accept both keys, so the refusal above is
    // a method rule and not a blanket rejection of the schema.
    const auto rks = Validate(R"(
[molecule]
atoms = [["H", 0.0, 0.0, 0.0], ["H", 0.0, 0.0, 0.74]]
[basis]
orbital = "sto-3g"
[method]
type = "rks"
accuracy = "kNormal"
functional = "pbe"
screening_tolerance = 1e-10
[grid]
radial_points = 40
angular_points = 50
)");
    EXPECT_TRUE(rks.IsValid()) << (rks.issues.empty() ? "" : rks.issues[0]);

    const auto uks = Validate(R"(
[molecule]
atoms = [["O", 0.0, 0.0, 0.0], ["O", 0.0, 0.0, 1.2075]]
[basis]
orbital = "sto-3g"
[method]
type = "uks"
accuracy = "kNormal"
functional = "pbe"
screening_tolerance = 0.0
)");
    EXPECT_TRUE(uks.IsValid()) << (uks.issues.empty() ? "" : uks.issues[0]);
}

} // namespace
