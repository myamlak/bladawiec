#include "qcx/properties/property_registry.hpp"

#include <array>
#include <cstddef>

namespace qcx::properties {
namespace {

using qcx::io::MethodType;

// The coverage record: one row per PropertyKind, in enumerator order. Every
// cell states what the driver does TODAY, refusals included, so the record is
// evidence rather than an aspiration. A row whose cell changes without this
// table changing is the drift the coverage test exists to catch.
constexpr std::array<PropertyDeclaration, 14> kDeclarations{{
    // Populations and moments are part of every run: no key asks for them and
    // no method is refused them.
    {
        .kind = PropertyKind::kPopulations,
        .name = "populations",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "FillProperties; both lanes, every method",
    },
    {
        .kind = PropertyKind::kMultipoles,
        .name = "multipoles",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "FillProperties; both lanes, every method",
    },
    // Hirshfeld's promolecular densities are the SAD fragment densities, whose
    // multiplicity table stops at neon.
    {
        .kind = PropertyKind::kHirshfeld,
        .name = "hirshfeld",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 10,
        .elementRefusal = RefusalReason::kHirshfeldSadTableZLimit,
        .evidence = "BuildChargesBlock with BuildAtomicInputs; "
                    "test DriverErrorTest.RejectsUnwiredCombinations",
    },
    {
        .kind = PropertyKind::kVoronoi,
        .name = "voronoi",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildChargesBlock; test DriverPinTest.H2oDirectRhfStage72BlocksPins",
    },
    {
        .kind = PropertyKind::kEsp,
        .name = "esp",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildEspBlock; test DriverPinTest.H2oDirectRhfStage72BlocksPins",
    },
    {
        .kind = PropertyKind::kEddb,
        .name = "eddb",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildEddbBlock; test DriverPinTest.H2oDirectRhfStage72BlocksPins",
    },
    // Fukui runs the N +- 1 species through the direct-UHF runner, so the
    // species of a Kohn-Sham parent would be Hartree-Fock densities.
    {
        .kind = PropertyKind::kFukui,
        .name = "fukui",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kFukuiKohnShamSpecies,
                     RefusalReason::kFukuiKohnShamSpecies},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildFukuiBlock; tests H2oDirectRhfFukuiSumRules, "
                    "O2UhfFukuiExercisesTheTripletBranches",
    },
    {
        .kind = PropertyKind::kNalewajski,
        .name = "nalewajski",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 19,
        .elementRefusal = RefusalReason::kNalewajskiFragmentTableZLimit,
        .evidence = "BuildNalewajskiBlock; ClassifyFragmentStates",
    },
    {
        .kind = PropertyKind::kDensityAtNuclei,
        .name = "density-at-nuclei",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildDensityAtNucleiBlock; test DriverPinTest.DensityAtNucleiBlockEndToEnd",
    },
    {
        .kind = PropertyKind::kQtaim,
        .name = "qtaim",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildQtaimBlock; test DriverPinTest.H2QtaimEndToEnd",
    },
    // The closed-shell check runs first, so a UKS run reports the closed-shell
    // reason and only RKS reaches the Hartree-Fock reason for ETS-NOCV. The
    // Aufbau table bounds a fragment built as one atom only; a multi-atom
    // fragment takes its multiplicity from its electron count.
    {
        .kind = PropertyKind::kEtsNocv,
        .name = "ets-nocv",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kEtsNocvClosedShellOnly,
                     RefusalReason::kEtsNocvHartreeFockOnly,
                     RefusalReason::kEtsNocvClosedShellOnly},
        .refusesWhenFunctionalNamed = true,
        .maxAtomicNumber = 19,
        .elementScope = ElementScope::kSingleAtomFragment,
        .elementRefusal = RefusalReason::kEtsNocvFragmentTableZLimit,
        .evidence = "BuildNocvBlock; tests H2oDirectRhfNocvPartitionPins, "
                    "RejectsUhfNocvCombination",
    },
    {
        .kind = PropertyKind::kMolden,
        .name = "molden",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "WriteMoldenExport; tests H2MoldenEndToEnd, O2MoldenUhfTwoBlocks",
    },
    // Both gradients refuse by the method's Kohn-Sham key policy before the
    // driver is reached.
    {
        .kind = PropertyKind::kXcGradient,
        .name = "xc-gradient",
        .byMethod = {RefusalReason::kXcGradientRequiresFunctional,
                     RefusalReason::kXcGradientRequiresFunctional,
                     RefusalReason::kNone,
                     RefusalReason::kNone},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildXcGradientBlock; test KsInputFileTest.TheGradientKeyOnAHartreeFockFileIs"
                    "RefusedByName",
    },
    {
        .kind = PropertyKind::kTotalGradient,
        .name = "total-gradient",
        .byMethod = {RefusalReason::kNone,
                     RefusalReason::kTotalGradientRequiresClosedShell,
                     RefusalReason::kNone,
                     RefusalReason::kTotalGradientRequiresClosedShell},
        .refusesWhenFunctionalNamed = false,
        .maxAtomicNumber = 0,
        .elementRefusal = RefusalReason::kNone,
        .evidence = "BuildTotalGradientBlock; tests TheKeyIsRefusedOnTheUnrestrictedMethods, "
                    "TheKohnShamGradientMatchesTheEnergyItDifferentiates",
    },
}};

static_assert(kDeclarations.size() == static_cast<std::size_t>(PropertyKind::kTotalGradient) + 1,
              "the declaration table must hold one row per PropertyKind, in enumerator order");

// The position of a method kind in a declaration's column order, which is
// MethodType's own order. A value outside the four returns kMethodKindCount:
// the lookup below reports it as kUnclassifiedMethod rather than reading past
// the row.
constexpr std::size_t MethodIndex(MethodType method) noexcept {
    switch (method)
    {
    case MethodType::kRhf:
        return 0;
    case MethodType::kUhf:
        return 1;
    case MethodType::kRks:
        return 2;
    case MethodType::kUks:
        return 3;
    }

    return kMethodKindCount;
}

} // namespace

const PropertyDeclaration& DeclarationFor(PropertyKind kind) noexcept {
    return kDeclarations[static_cast<std::size_t>(kind)];
}

std::span<const PropertyDeclaration> AllDeclarations() noexcept {
    return kDeclarations;
}

RefusalReason RefusalFor(PropertyKind kind, qcx::io::MethodType method) noexcept {
    const std::size_t column = MethodIndex(method);

    if (column == kMethodKindCount)
    {
        return RefusalReason::kUnclassifiedMethod;
    }

    return DeclarationFor(kind).byMethod[column];
}

std::string_view ToString(PropertyKind kind) noexcept {
    switch (kind)
    {
    case PropertyKind::kPopulations:
        return "populations";
    case PropertyKind::kMultipoles:
        return "multipoles";
    case PropertyKind::kHirshfeld:
        return "hirshfeld";
    case PropertyKind::kVoronoi:
        return "voronoi";
    case PropertyKind::kEsp:
        return "esp";
    case PropertyKind::kEddb:
        return "eddb";
    case PropertyKind::kFukui:
        return "fukui";
    case PropertyKind::kNalewajski:
        return "nalewajski";
    case PropertyKind::kDensityAtNuclei:
        return "density-at-nuclei";
    case PropertyKind::kQtaim:
        return "qtaim";
    case PropertyKind::kEtsNocv:
        return "ets-nocv";
    case PropertyKind::kMolden:
        return "molden";
    case PropertyKind::kXcGradient:
        return "xc-gradient";
    case PropertyKind::kTotalGradient:
        return "total-gradient";
    }

    return "unclassified-property";
}

std::string_view ToString(RefusalReason reason) noexcept {
    switch (reason)
    {
    case RefusalReason::kNone:
        return "none";
    case RefusalReason::kFukuiKohnShamSpecies:
        return "fukui-kohn-sham-species";
    case RefusalReason::kEtsNocvClosedShellOnly:
        return "ets-nocv-closed-shell-only";
    case RefusalReason::kEtsNocvHartreeFockOnly:
        return "ets-nocv-hartree-fock-only";
    case RefusalReason::kHirshfeldSadTableZLimit:
        return "hirshfeld-sad-table-z-limit";
    case RefusalReason::kNalewajskiFragmentTableZLimit:
        return "nalewajski-fragment-table-z-limit";
    case RefusalReason::kEtsNocvFragmentTableZLimit:
        return "ets-nocv-fragment-table-z-limit";
    case RefusalReason::kXcGradientRequiresFunctional:
        return "xc-gradient-requires-functional";
    case RefusalReason::kTotalGradientRequiresClosedShell:
        return "total-gradient-requires-closed-shell";
    case RefusalReason::kUnclassifiedMethod:
        return "unclassified-method";
    }

    return "unclassified-refusal";
}

} // namespace qcx::properties
