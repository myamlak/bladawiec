// The property coverage record, checked cell by cell.
//
// The properties module cannot link the driver - the module DAG puts driver to
// its right - so the driver's outcome is written out here as an expected table,
// transcribed from the driver code paths and driver tests the registry's
// evidence column names. The registry must agree with every cell. The two
// gradient properties are also exercised through the io input validator, which
// properties can link: that is the code that produces their user-facing
// refusal, so those eight cells are observed rather than transcribed.
//
// The last test prints the matrix the coverage record exists to show.

#include "qcx/io/run_input.hpp"
#include "qcx/io/validate_input.hpp"
#include "qcx/properties/property_registry.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using qcx::io::MethodType;
using qcx::io::RunInput;
using qcx::io::ValidateInput;
using qcx::io::ValidationReport;
using qcx::properties::AllDeclarations;
using qcx::properties::DeclarationFor;
using qcx::properties::ElementScope;
using qcx::properties::PropertyDeclaration;
using qcx::properties::PropertyKind;
using qcx::properties::RefusalFor;
using qcx::properties::RefusalReason;
using qcx::properties::ToString;

constexpr std::array<MethodType, qcx::properties::kMethodKindCount> kMethodKinds{
    MethodType::kRhf,
    MethodType::kUhf,
    MethodType::kRks,
    MethodType::kUks,
    MethodType::kRohf,
    MethodType::kRoks,
};

constexpr std::size_t kPropertyCount = static_cast<std::size_t>(PropertyKind::kTotalGradient) + 1;

// Whether a sorted-copy duplicate check finds a repeated word. A plain scan
// rather than std::adjacent_find: the style guard reserves snake_case tokens
// for library names it has admitted one at a time, and this check does not
// need one admitted on its behalf.
bool HasDuplicateWord(std::vector<std::string_view> words) {
    std::sort(words.begin(), words.end());

    for (std::size_t i = 1; i < words.size(); ++i)
    {
        if (words[i] == words[i - 1])
        {
            return true;
        }
    }

    return false;
}

struct ExpectedRow {
    PropertyKind kind;
    std::array<RefusalReason, qcx::properties::kMethodKindCount> byMethod;
};

// The driver's current outcome, one row per property, in the column order
// rhf, uhf, rks, uks, rohf, roks. The refusals are transcribed from the
// driver's pre-run checks (ETS-NOCV on a per-spin run and on a
// functional-naming method) and the input validator's two
// gradient policies plus the restricted open-shell key policy; the element
// limits are declared beside the table and checked separately.
constexpr std::array<ExpectedRow, kPropertyCount> kExpected{{
    {PropertyKind::kPopulations,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kMultipoles,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kHirshfeld,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kVoronoi,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kEsp,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kEddb,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kFukui,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kNalewajski,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kDensityAtNuclei,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kQtaim,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    // The closed-shell check precedes the functional check, so uks reports the
    // closed-shell reason and only rks reports the Hartree-Fock one; the
    // restricted open-shell pair is per spin and reports the closed-shell
    // reason like uks.
    {PropertyKind::kEtsNocv,
     {RefusalReason::kNone,
      RefusalReason::kEtsNocvClosedShellOnly,
      RefusalReason::kEtsNocvHartreeFockOnly,
      RefusalReason::kEtsNocvClosedShellOnly,
      RefusalReason::kEtsNocvClosedShellOnly,
      RefusalReason::kEtsNocvClosedShellOnly}},
    {PropertyKind::kMolden,
     {RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kNone}},
    {PropertyKind::kXcGradient,
     {RefusalReason::kXcGradientRequiresFunctional,
      RefusalReason::kXcGradientRequiresFunctional,
      RefusalReason::kNone,
      RefusalReason::kNone,
      RefusalReason::kXcGradientRequiresFunctional,
      RefusalReason::kNone}},
    {PropertyKind::kTotalGradient,
     {RefusalReason::kNone,
      RefusalReason::kTotalGradientRequiresClosedShell,
      RefusalReason::kNone,
      RefusalReason::kTotalGradientRequiresClosedShell,
      RefusalReason::kTotalGradientRequiresClosedShell,
      RefusalReason::kTotalGradientRequiresClosedShell}},
}};

// True when any issue starts with the given property key.
bool HasIssueFor(const ValidationReport& report, std::string_view keyPrefix) {
    return std::any_of(
        report.issues.begin(), report.issues.end(), [keyPrefix](const std::string& issue) {
            return issue.starts_with(keyPrefix);
        });
}

// The one issue for a property key; empty when the validator raised none.
std::string IssueFor(const ValidationReport& report, std::string_view keyPrefix) {
    for (const std::string& issue : report.issues)
    {
        if (issue.starts_with(keyPrefix))
        {
            return issue;
        }
    }

    return {};
}

RunInput MakeInput(MethodType method) {
    RunInput input;
    input.method.method = method;
    return input;
}

TEST(PropertyRegistryTest, TheTableHoldsOneRowPerPropertyKind) {
    const auto declarations = AllDeclarations();

    ASSERT_EQ(declarations.size(), kPropertyCount);
    ASSERT_EQ(kExpected.size(), kPropertyCount);

    for (std::size_t i = 0; i < declarations.size(); ++i)
    {
        EXPECT_EQ(declarations[i].kind, kExpected[i].kind) << "row " << i;
        EXPECT_EQ(&DeclarationFor(kExpected[i].kind), &declarations[i]);
    }
}

TEST(PropertyRegistryTest, EveryCellMatchesTheDriversOutcome) {
    for (const ExpectedRow& expected : kExpected)
    {
        for (std::size_t column = 0; column < kMethodKinds.size(); ++column)
        {
            const MethodType method = kMethodKinds[column];

            EXPECT_EQ(RefusalFor(expected.kind, method), expected.byMethod[column])
                << ToString(expected.kind) << " on " << qcx::io::ToString(method) << ": registry "
                << ToString(RefusalFor(expected.kind, method)) << ", driver "
                << ToString(expected.byMethod[column]);
        }
    }
}

TEST(PropertyRegistryTest, EveryDeclaredWordIsUnique) {
    std::vector<std::string_view> propertyWords;

    for (const PropertyDeclaration& declaration : AllDeclarations())
    {
        EXPECT_EQ(declaration.name, ToString(declaration.kind));
        EXPECT_FALSE(declaration.name.empty());
        propertyWords.push_back(declaration.name);
    }

    EXPECT_FALSE(HasDuplicateWord(propertyWords)) << "two properties share a word";

    std::vector<std::string_view> refusalWords;

    for (const RefusalReason reason : {RefusalReason::kNone,
                                       RefusalReason::kEtsNocvClosedShellOnly,
                                       RefusalReason::kEtsNocvHartreeFockOnly,
                                       RefusalReason::kHirshfeldSadTableZLimit,
                                       RefusalReason::kNalewajskiFragmentTableZLimit,
                                       RefusalReason::kEtsNocvFragmentTableZLimit,
                                       RefusalReason::kXcGradientRequiresFunctional,
                                       RefusalReason::kTotalGradientRequiresClosedShell,
                                       RefusalReason::kUnclassifiedMethod})
    {
        EXPECT_FALSE(ToString(reason).empty());
        refusalWords.push_back(ToString(reason));
    }

    EXPECT_FALSE(HasDuplicateWord(refusalWords)) << "two refusal reasons share a word";
}

TEST(PropertyRegistryTest, EveryDeclarationCarriesEvidence) {
    for (const PropertyDeclaration& declaration : AllDeclarations())
    {
        EXPECT_FALSE(declaration.evidence.empty()) << ToString(declaration.kind);
    }
}

TEST(PropertyRegistryTest, TheFunctionalRuleMatchesTheKohnShamCells) {
    for (const PropertyDeclaration& declaration : AllDeclarations())
    {
        if (!declaration.refusesWhenFunctionalNamed)
        {
            continue;
        }

        // Every Kohn-Sham method word names a functional; the rule says every
        // one of them is refused, whichever reason has precedence.
        for (const MethodType method : {MethodType::kRks, MethodType::kUks, MethodType::kRoks})
        {
            EXPECT_NE(RefusalFor(declaration.kind, method), RefusalReason::kNone)
                << ToString(declaration.kind) << " on " << qcx::io::ToString(method);
        }
    }
}

TEST(PropertyRegistryTest, ElementLimitsNameTheirReason) {
    for (const PropertyDeclaration& declaration : AllDeclarations())
    {
        if (declaration.maxAtomicNumber == 0)
        {
            EXPECT_EQ(declaration.elementRefusal, RefusalReason::kNone)
                << ToString(declaration.kind);
            EXPECT_EQ(declaration.elementScope, ElementScope::kMolecule)
                << ToString(declaration.kind);
            continue;
        }

        EXPECT_NE(declaration.elementRefusal, RefusalReason::kNone) << ToString(declaration.kind);
    }

    EXPECT_EQ(DeclarationFor(PropertyKind::kHirshfeld).maxAtomicNumber, 19);
    EXPECT_EQ(DeclarationFor(PropertyKind::kHirshfeld).elementScope, ElementScope::kMolecule);
    EXPECT_EQ(DeclarationFor(PropertyKind::kHirshfeld).elementRefusal,
              RefusalReason::kHirshfeldSadTableZLimit);
    EXPECT_EQ(DeclarationFor(PropertyKind::kNalewajski).maxAtomicNumber, 19);
    EXPECT_EQ(DeclarationFor(PropertyKind::kNalewajski).elementScope, ElementScope::kMolecule);
    EXPECT_EQ(DeclarationFor(PropertyKind::kNalewajski).elementRefusal,
              RefusalReason::kNalewajskiFragmentTableZLimit);
    EXPECT_EQ(DeclarationFor(PropertyKind::kEtsNocv).maxAtomicNumber, 19);
    EXPECT_EQ(DeclarationFor(PropertyKind::kEtsNocv).elementScope,
              ElementScope::kSingleAtomFragment);
    EXPECT_EQ(DeclarationFor(PropertyKind::kEtsNocv).elementRefusal,
              RefusalReason::kEtsNocvFragmentTableZLimit);
}

// The two gradient properties' refusals are produced by the input validator,
// which this module links: the declaration is checked against the real
// refusal rather than against a transcription.
TEST(PropertyRegistryTest, TheGradientCellsMatchTheInputValidator) {
    for (const MethodType method : kMethodKinds)
    {
        RunInput xcInput = MakeInput(method);
        xcInput.properties.xcGradient = true;
        const ValidationReport xcReport = ValidateInput(xcInput);
        const bool xcRefused = HasIssueFor(xcReport, "properties.xc_gradient");

        EXPECT_EQ(xcRefused, RefusalFor(PropertyKind::kXcGradient, method) != RefusalReason::kNone)
            << qcx::io::ToString(method);

        if (xcRefused)
        {
            const std::string issue = IssueFor(xcReport, "properties.xc_gradient");
            EXPECT_NE(issue.find(qcx::io::ToString(method)), std::string::npos) << issue;
            EXPECT_NE(issue.find("does not take an exchange-correlation gradient"),
                      std::string::npos)
                << issue;
        }

        RunInput totalInput = MakeInput(method);
        totalInput.properties.totalGradient = true;
        const ValidationReport totalReport = ValidateInput(totalInput);
        const bool totalRefused = HasIssueFor(totalReport, "properties.total_gradient");

        EXPECT_EQ(totalRefused,
                  RefusalFor(PropertyKind::kTotalGradient, method) != RefusalReason::kNone)
            << qcx::io::ToString(method);

        if (totalRefused)
        {
            const std::string issue = IssueFor(totalReport, "properties.total_gradient");
            EXPECT_NE(issue.find(qcx::io::ToString(method)), std::string::npos) << issue;
            EXPECT_NE(issue.find("does not take the total analytic gradient"), std::string::npos)
                << issue;
        }
    }
}

TEST(PropertyRegistryTest, TheMatrixPrints) {
    std::ostringstream matrix;
    matrix << "property";

    for (const MethodType method : kMethodKinds)
    {
        matrix << " | " << qcx::io::ToString(method);
    }

    // The cross-cutting conditions are columns too: a matrix that showed only
    // the method axis would hide the element caps and the functional rule the
    // same declaration states.
    matrix << " | max-Z | element-scope | functional-refused";

    for (const ExpectedRow& expected : kExpected)
    {
        matrix << "\n" << ToString(expected.kind);

        for (const MethodType method : kMethodKinds)
        {
            matrix << " | " << ToString(RefusalFor(expected.kind, method));
        }

        const PropertyDeclaration& declaration = DeclarationFor(expected.kind);
        matrix << " | " << declaration.maxAtomicNumber << " | "
               << (declaration.elementScope == ElementScope::kMolecule ? "molecule"
                                                                       : "single-atom-fragment")
               << " | " << (declaration.refusesWhenFunctionalNamed ? "yes" : "no");
    }

    const std::string printed = matrix.str();

    EXPECT_EQ(static_cast<std::size_t>(std::count(printed.begin(), printed.end(), '\n')),
              kPropertyCount);

    RecordProperty("coverage_matrix", printed);
    std::cout << printed << "\n";
}

} // namespace
