#pragma once

/// \file
/// The Fock-build planner: the provider interface over the builders' existing
/// J and K half adapters, and the constraint vocabulary that decides whether a
/// requested pair of halves can be built together.
///
/// **Why a planner rather than a wider builder enum.** A `BuilderKind` names a
/// whole family and takes both halves from it. The independent choices are the
/// Coulomb half and the exchange half; every coupling between them (the
/// Kohn-Sham energy seam's separate J[D], the per-spin assembly, the auxiliary
/// fit a fitted half needs, the disk rung, the device boundary) is an EXPLICIT
/// constraint here, so a request no provider pair satisfies is refused by
/// naming the constraint it broke rather than resolved to something else.
///
/// **The smallest useful version: the direct screened and the density-fitted
/// providers.** Those two cover the owner's example - a direct Coulomb half
/// against a density-fitted exchange half - and the remaining families are
/// registered only when their half adapters exist and are exercised. A
/// substitution is never made across provider classes: a half whose provider
/// the plan cannot supply is a refusal, not a fallback, because a fitted half
/// and an exact half are different calculations.
///
/// **The plan is a decision, not a substitution.** It names the provider that
/// supplies each half, the accounting its adapter speaks, and the constraints
/// that were checked. The driver records it beside the builder selection and
/// refuses a plan its wiring cannot execute; it must never record a plan and
/// then execute a different one.
///
/// The provider interface is deliberately a DESCRIPTOR rather than a type
/// erased callable pair: the driver's half adapters are typed over concrete
/// builders (`MakeDirectKsHalf`, `MakeRiFullKsHalf`, ...), and hiding them
/// behind `std::function` here would move the adapter contract one layer away
/// from the only place that can check it. A provider declares which half it
/// builds, the accounting it returns, and its capabilities; the wiring builds
/// the adapter from that declaration.

#include "qcx/error.hpp"
#include "qcx/io/run_input.hpp"

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::driver {

/// Which half of the Fock matrix a provider supplies.
/// \ingroup qcx-driver
enum class FockHalfRole {
    kCoulomb, ///< J, the Coulomb half.
    kExchange ///< K, the exchange half.
};

/// The provider class a half belongs to (the equivalence classes the design
/// names: the direct screened, the density-fitted, the combined and the
/// fast-multipole providers). Substitution is legal only inside one class, and
/// this planner makes none: the word is recorded so a reader can see the
/// class of each half a mixed plan joined.
/// \ingroup qcx-driver
enum class FockProviderClass {
    kDirectScreened, ///< Exact screened quartets (the direct family, lean included).
    kDensityFitted, ///< Both halves through one auxiliary fit (occ-RI-K included).
    kCombined, ///< A fitted Coulomb half beside an exact exchange half (the RI-J link).
    kFastMultipole ///< The QFMM far field (both halves from the composed builder).
};

/// A provider's NATIVE accounting: what the family's own builder returns,
/// BEFORE the driver's half adapter converts it to the composition contract
/// (`driver/src/internal/ks_composition.hpp`, HalfFockFn).
///
/// The record carries the native word because that is what distinguishes the
/// families - after conversion every half is the contract's H + 2 J or H - K,
/// so a converted word would say nothing. The live conventions differ by a full
/// core Hamiltonian and by a factor of two, and a plan that did not state its
/// accounting would let a reader assemble the Fock from the wrong halves without
/// any number looking wrong. The adapter's conversion is fixed per family and
/// documented at its definition: `MakeRiJLinkKsHalf` adds H to the Coulomb half
/// only, and `MakeRiFullKsHalf` adds H to both halves and doubles the Coulomb
/// one.
/// \ingroup qcx-driver
enum class FockHalfAccounting {
    kCoulombWithCore, ///< Native H + 2 J(rho) (the direct family and the composed QFMM builder).
    kCoulombBare, ///< Native 2 J(rho), no core Hamiltonian (the RI-J link; ri_jk emits bare J).
    kExchangeWithCore, ///< Native H - K(rho) (the direct family and the RI-J link).
    kExchangeBare ///< Native K(rho), no core Hamiltonian (the full-RI family emits bare K).
};

/// What a half provider can and cannot do, stated once so the planner's
/// constraints read capabilities rather than family names.
/// \ingroup qcx-driver
struct FockHalfCapabilities {
    /// The builder exposes this half as a call of its own. False when the
    /// family only builds a fused Fock (the case the Kohn-Sham energy seam
    /// refuses).
    bool exposesHalf = false;
    /// A per-spin adapter exists (the unrestricted and restricted open-shell
    /// assemblies call one half per spin).
    bool perSpinAdapter = false;
    /// The provider's Create consumes an auxiliary basis even when only this
    /// half runs.
    bool auxiliaryRequired = false;
    /// The family's DISK rung can serve this half through the driver's
    /// half wiring. False marks the rung the driver documents as exposing no
    /// per-iteration half.
    bool diskRungSuppliesHalf = false;
    /// The provider supplies the Coulomb half only (the fast-multipole far
    /// field has no exchange half of its own).
    bool coulombOnly = false;
    /// The half's kernel is exact screened quartets. False for every fitted or
    /// approximated half; the record states the word so a mixed plan's result
    /// class is visible.
    bool exactKernel = false;
};

/// One registered half provider: the family, the half it supplies, its
/// equivalence class, the accounting its adapter returns and its capabilities.
/// Aggregate by design - the fields are the interface (the aggregate-struct
/// exemption).
/// \ingroup qcx-driver
struct FockHalfProvider {
    qcx::io::IntegralFamily family; ///< The family word the provider answers to.
    FockHalfRole role; ///< The half it builds.
    FockProviderClass providerClass; ///< Its equivalence class.
    FockHalfAccounting accounting; ///< Its NATIVE convention (before the driver adapter converts).
    FockHalfCapabilities capabilities; ///< Its capabilities and constraints.
};

/// The registered providers, in the family vocabulary's order.
///
/// Eight entries: the direct family's two halves, the RI-J link's two, the
/// full-RI family's two, and the QFMM pair. A family with no entry for a role
/// cannot supply that half, and the planner refuses the request by name.
/// \returns The provider table.
/// \ingroup qcx-driver
inline std::span<const FockHalfProvider> FockHalfProviders() noexcept {
    static constexpr std::array<FockHalfProvider, 8> kProviders{{
        // The direct family: exact screened quartets, the composition's own
        // native accounting (both halves carry one full H).
        {qcx::io::IntegralFamily::kDirect,
         FockHalfRole::kCoulomb,
         FockProviderClass::kDirectScreened,
         FockHalfAccounting::kCoulombWithCore,
         {true, true, false, true, false, true}},
        {qcx::io::IntegralFamily::kDirect,
         FockHalfRole::kExchange,
         FockProviderClass::kDirectScreened,
         FockHalfAccounting::kExchangeWithCore,
         {true, true, false, true, false, true}},
        // The RI-J link: fitted Coulomb (bare 2 J_RI) beside an exact nested
        // direct exchange (the sole H carrier). Its disk rung is the driver's
        // documented no-half route.
        {qcx::io::IntegralFamily::kRiJLink,
         FockHalfRole::kCoulomb,
         FockProviderClass::kCombined,
         FockHalfAccounting::kCoulombBare,
         {true, true, true, false, false, false}},
        {qcx::io::IntegralFamily::kRiJLink,
         FockHalfRole::kExchange,
         FockProviderClass::kCombined,
         FockHalfAccounting::kExchangeWithCore,
         {true, true, true, false, false, true}},
        // The full-RI family: both halves bare and fitted, one traversal of
        // the auxiliary index.
        {qcx::io::IntegralFamily::kRiJk,
         FockHalfRole::kCoulomb,
         FockProviderClass::kDensityFitted,
         FockHalfAccounting::kCoulombBare,
         {true, true, true, true, false, false}},
        {qcx::io::IntegralFamily::kRiJk,
         FockHalfRole::kExchange,
         FockProviderClass::kDensityFitted,
         FockHalfAccounting::kExchangeBare,
         {true, true, true, true, false, false}},
        // The QFMM far field: both halves are exposed by the composed builder
        // (BuildCoulombOnly / BuildExchangeOnly), each already carrying one
        // full H copy. The exchange half is K only, so it is a mixed plan's
        // exchange provider on the same terms as the whole-family run.
        {qcx::io::IntegralFamily::kQfmm,
         FockHalfRole::kCoulomb,
         FockProviderClass::kFastMultipole,
         FockHalfAccounting::kCoulombWithCore,
         {true, true, false, true, true, false}},
        {qcx::io::IntegralFamily::kQfmm,
         FockHalfRole::kExchange,
         FockProviderClass::kFastMultipole,
         FockHalfAccounting::kExchangeWithCore,
         {true, true, false, true, false, false}},
    }};

    return kProviders;
}

/// Finds the provider that supplies one role for one family.
/// \param providers The provider table to search.
/// \param family The family word.
/// \param role The half requested.
/// \returns The provider, or nullopt when the family has no entry for that
/// half in the table.
/// \ingroup qcx-driver
inline std::optional<FockHalfProvider> FindFockHalfProvider(
    std::span<const FockHalfProvider> providers,
    qcx::io::IntegralFamily family,
    FockHalfRole role) noexcept {
    for (const FockHalfProvider& provider : providers)
    {
        if (provider.family == family && provider.role == role)
        {
            return provider;
        }
    }

    return std::nullopt;
}

/// Finds a provider in the registered table.
/// \param family The family word.
/// \param role The half requested.
/// \returns The provider, or nullopt.
/// \ingroup qcx-driver
inline std::optional<FockHalfProvider> FindFockHalfProvider(qcx::io::IntegralFamily family,
                                                            FockHalfRole role) noexcept {
    return FindFockHalfProvider(FockHalfProviders(), family, role);
}

/// One half request and the run context the constraints are read against.
/// Aggregate by design (the aggregate-struct exemption).
/// \ingroup qcx-driver
struct FockPlanRequest {
    /// The Coulomb half's family. Required: every Fock has a Coulomb half.
    std::optional<qcx::io::IntegralFamily> coulombFamily;
    /// The exchange half's family. Required when `exchangeRequired`; when it is
    /// absent and the run needs an exchange, the resolved family supplied no K
    /// half and the planner refuses by name.
    std::optional<qcx::io::IntegralFamily> exchangeFamily;
    /// True when the run builds an exchange term at all: always on Hartree-Fock,
    /// and on a Kohn-Sham run exactly when the functional carries a non-zero
    /// exact-exchange fraction.
    bool exchangeRequired = true;
    /// True on the Kohn-Sham lanes, whose energy seam contracts J[D]
    /// SEPARATELY from the Fock.
    bool kohnSham = false;
    /// True on the per-spin lanes (UHF, UKS, ROHF, ROKS).
    bool perSpin = false;
    /// The auxiliary basis resolved for the Coulomb half, when that provider
    /// requires one; nullopt otherwise and on a resolution failure.
    std::optional<std::string> coulombAuxBasis;
    /// The auxiliary basis resolved for the exchange half, under the same rule.
    std::optional<std::string> exchangeAuxBasis;
    /// True when the run asked for the RI disk rung (`method.ri_tensor_mode =
    /// "disk"` or `[diagnostics] force_disk_ri`).
    bool diskRungRequested = false;
    /// True when the run asked for the RI orbit expansion.
    bool orbitExpansionRequested = false;
    /// True when the run asked for the disk-tier ERI store.
    bool eriStoreRequested = false;
    /// The execution backend the run requested.
    qcx::io::ExecutionBackend backend = qcx::io::ExecutionBackend::kCpu;
    /// True when the resolved selection wires the direct family's lean member.
    bool leanMemberRequested = false;
};

/// One half of a resolved plan, in the record's own words.
/// \ingroup qcx-driver
struct FockHalfPlan {
    /// "coulomb" | "exchange" - the half this record describes.
    std::string role;
    /// The family word of the provider that supplies it.
    std::string provider;
    /// The provider's equivalence class: "direct_screened" | "density_fitted" |
    /// "combined" | "fast_multipole".
    std::string providerClass;
    /// The half's kernel: "exact" | "density_fitted" | "fast_multipole".
    std::string kernel;
    /// The adapter's accounting: "H+2J" | "2J" | "H-K" | "K".
    std::string accounting;
    /// The auxiliary basis the half's provider Create consumes; absent when
    /// the half needs none.
    std::optional<std::string> auxBasis;
};

/// The planner's decision: the provider per half, whether one provider builds
/// both halves (one shared intermediate and one Create cost), the constraints
/// that were checked, and the reasoning the record carries.
/// \ingroup qcx-driver
struct FockBuildPlan {
    FockHalfPlan coulomb; ///< The Coulomb half's provider record.
    /// The exchange half's provider record; absent exactly when the run needs
    /// no exchange (a pure Kohn-Sham functional).
    std::optional<FockHalfPlan> exchange;
    /// True when one family supplies both halves - one builder, one Create,
    /// and one shared shell-quartet pass.
    bool sameProvider = true;
    /// True when the two halves come from different providers. The halves are
    /// then separate builders and share no intermediate.
    bool mixed = false;
    /// The constraints the planner evaluated, in check order, machine words:
    /// "coulomb_matrix", "per_spin_adapter", "auxiliary_basis", "disk_rung",
    /// "device_boundary", "option_scope", "exchange_capability".
    std::vector<std::string> constraintsChecked;
    /// One sentence naming the request, the providers and the sharing taken.
    std::string reasoning;
};

/// The provider class word the record carries.
/// \param providerClass The class.
/// \returns "direct_screened" | "density_fitted" | "combined" | "fast_multipole".
/// \ingroup qcx-driver
inline std::string_view ToString(FockProviderClass providerClass) noexcept {
    switch (providerClass)
    {
    case FockProviderClass::kDirectScreened:
        return "direct_screened";

    case FockProviderClass::kDensityFitted:
        return "density_fitted";

    case FockProviderClass::kCombined:
        return "combined";

    case FockProviderClass::kFastMultipole:
        return "fast_multipole";
    }

    return "direct_screened";
}

/// The accounting word the record carries.
/// \param accounting The adapter convention.
/// \returns "H+2J" | "2J" | "H-K" | "K".
/// \ingroup qcx-driver
inline std::string_view ToString(FockHalfAccounting accounting) noexcept {
    switch (accounting)
    {
    case FockHalfAccounting::kCoulombWithCore:
        return "H+2J";

    case FockHalfAccounting::kCoulombBare:
        return "2J";

    case FockHalfAccounting::kExchangeWithCore:
        return "H-K";

    case FockHalfAccounting::kExchangeBare:
        return "K";
    }

    return "H+2J";
}

namespace detail {

/// Builds a qcx::Error from a code and message.
///
/// The planner's own two-argument helper, named distinctly from the driver
/// translation unit's anonymous-namespace `Err` so an unqualified call in that
/// unit can never become ambiguous with this header's.
/// \param code The error code.
/// \param message The refusal text.
/// \returns The error value.
inline qcx::Error PlanError(qcx::ErrorCode code, std::string message) {
    return qcx::Error{code, std::move(message)};
}

/// The kernel word a provider's half carries.
/// \param provider The half provider.
/// \returns "exact" | "density_fitted" | "fast_multipole".
inline std::string_view KernelWord(const FockHalfProvider& provider) noexcept {
    if (provider.capabilities.exactKernel)
    {
        return "exact";
    }

    if (provider.providerClass == FockProviderClass::kFastMultipole)
    {
        return "fast_multipole";
    }

    return "density_fitted";
}

/// The list of family words that can supply one role, for a refusal message.
/// \param providers The provider table to read.
/// \param role The half asked for.
/// \returns The provider words, comma-separated.
inline std::string RoleProviderList(std::span<const FockHalfProvider> providers,
                                    FockHalfRole role) {
    std::string names;

    for (const FockHalfProvider& provider : providers)
    {
        if (provider.role != role)
        {
            continue;
        }

        if (!names.empty())
        {
            names += ", ";
        }

        names += std::string(qcx::io::ToString(provider.family));
    }

    return names;
}

/// Builds the record for one resolved half.
/// \param provider The resolved provider.
/// \param auxBasis The auxiliary basis resolved for the half, when the
/// provider requires one.
/// \returns The half's record.
inline FockHalfPlan MakeHalfPlan(const FockHalfProvider& provider,
                                 const std::optional<std::string>& auxBasis) {
    return FockHalfPlan{provider.role == FockHalfRole::kCoulomb ? "coulomb" : "exchange",
                        std::string(qcx::io::ToString(provider.family)),
                        std::string(ToString(provider.providerClass)),
                        std::string(KernelWord(provider)),
                        std::string(ToString(provider.accounting)),
                        auxBasis};
}

} // namespace detail

/// Plans the Fock build against a provider table, or refuses by naming the
/// unsatisfied constraint and the half it was read against.
///
/// The table is an argument rather than a hard-wired read so the planner's
/// honesty mechanism is testable: a provider that declares it exposes no half
/// (the fused-only case) exercises the Kohn-Sham seam constraint, and a test
/// provider can do that without a second production registration. The
/// production overload below always passes `FockHalfProviders()`.
///
/// The checks run in a fixed order so two runs of the same request refuse with
/// the same sentence: the halves resolve first (an unregistered half is a
/// vocabulary error), then the per-spin adapter, the Kohn-Sham Coulomb seam,
/// the auxiliary basis, the disk rung, the device boundary and the option
/// scope.
/// \param request The half families and the run context.
/// \param providers The provider table the plan chooses from.
/// \returns The plan, or an Error (kInvalidArgument for an unregistered half,
/// kUnimplemented for a constraint no provider pair satisfies) whose message
/// names the constraint, the half, and the providers that could have supplied
/// it.
/// \ingroup qcx-driver
inline qcx::Result<FockBuildPlan> PlanFockBuild(const FockPlanRequest& request,
                                                std::span<const FockHalfProvider> providers) {
    if (!request.coulombFamily.has_value())
    {
        return std::unexpected(
            detail::PlanError(qcx::ErrorCode::kInvalidArgument,
                              "the Fock-build plan needs a Coulomb half, and none was "
                              "resolved for the run"));
    }

    const auto coulomb =
        FindFockHalfProvider(providers, *request.coulombFamily, FockHalfRole::kCoulomb);

    if (!coulomb.has_value())
    {
        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kInvalidArgument,
            "builder.coulomb_half = \"" + std::string(qcx::io::ToString(*request.coulombFamily)) +
                "\" is not a Coulomb-half provider: the registered Coulomb halves are " +
                detail::RoleProviderList(providers, FockHalfRole::kCoulomb)));
    }

    std::optional<FockHalfProvider> exchange;

    if (request.exchangeFamily.has_value())
    {
        exchange =
            FindFockHalfProvider(providers, *request.exchangeFamily, FockHalfRole::kExchange);
    }

    if (!request.exchangeRequired && request.exchangeFamily.has_value())
    {
        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kUnimplemented,
            "builder.exchange_half = \"" + std::string(qcx::io::ToString(*request.exchangeFamily)) +
                "\" was requested, but this run builds no exchange term (a pure Kohn-Sham "
                "functional carries no exact-exchange fraction), so the half would never be "
                "built: the request is refused rather than dropped"));
    }

    if (request.exchangeRequired && !exchange.has_value())
    {
        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kUnimplemented,
            "the exchange half is required by this run and no provider supplies it: " +
                (request.exchangeFamily.has_value()
                     ? "builder.exchange_half = \"" +
                           std::string(qcx::io::ToString(*request.exchangeFamily)) +
                           "\" names a family with no exchange half"
                     : std::string("the resolved family supplies no exchange half")) +
                " (the registered exchange halves are " +
                detail::RoleProviderList(providers, FockHalfRole::kExchange) + ")"));
    }

    FockBuildPlan plan;
    plan.constraintsChecked.reserve(7);

    // The Kohn-Sham energy seam contracts J[D] separately from the Fock, so the
    // Coulomb half must be callable on its own. This is the constraint the
    // fused-only providers fail.
    if (request.kohnSham)
    {
        plan.constraintsChecked.emplace_back("coulomb_matrix");

        if (!coulomb->capabilities.exposesHalf)
        {
            return std::unexpected(detail::PlanError(
                qcx::ErrorCode::kUnimplemented,
                "the Kohn-Sham energy seam needs the Coulomb matrix J[D] separately from the "
                "Fock, and builder.coulomb_half = \"" +
                    std::string(qcx::io::ToString(coulomb->family)) +
                    "\" supplies only a fused Fock: no Coulomb-half adapter exists for this "
                    "provider, so the run is refused rather than given a seam it cannot "
                    "contract"));
        }
    }

    // The per-spin assemblies call one half per spin.
    if (request.perSpin)
    {
        plan.constraintsChecked.emplace_back("per_spin_adapter");

        if (!coulomb->capabilities.perSpinAdapter ||
            (exchange.has_value() && !exchange->capabilities.perSpinAdapter))
        {
            const char* const half = !coulomb->capabilities.perSpinAdapter ? "coulomb" : "exchange";

            return std::unexpected(detail::PlanError(
                qcx::ErrorCode::kUnimplemented,
                "this run builds its Focks per spin, and the " + std::string(half) +
                    " half's provider has no per-spin adapter: the halved composition "
                    "cannot be stated for this pair, so the run is refused"));
        }
    }

    // A fitted half's Create consumes the auxiliary basis; the driver resolves
    // the name and hands it in so this check is about the RESOLUTION, not a
    // second selection rule.
    plan.constraintsChecked.emplace_back("auxiliary_basis");

    if (coulomb->capabilities.auxiliaryRequired && !request.coulombAuxBasis.has_value())
    {
        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kUnimplemented,
            "the Coulomb half's provider (builder.coulomb_half = \"" +
                std::string(qcx::io::ToString(coulomb->family)) +
                "\") fits its integrals through an auxiliary basis, and no auxiliary basis "
                "could be resolved for it: the half cannot be built without its fit"));
    }

    if (exchange.has_value() && exchange->capabilities.auxiliaryRequired &&
        !request.exchangeAuxBasis.has_value())
    {
        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kUnimplemented,
            "the exchange half's provider (builder.exchange_half = \"" +
                std::string(qcx::io::ToString(exchange->family)) +
                "\") fits its integrals through an auxiliary basis, and no auxiliary basis "
                "could be resolved for it: the half cannot be built without its fit"));
    }

    if (!request.exchangeRequired)
    {
        exchange.reset();
    }

    // The mixed plan: both halves exist and come from different families. Read
    // here rather than after the device check because the disk-rung rule below
    // needs it - the rung is one whole-family builder's storage route, and a
    // mixed plan has no single builder to put on it.
    const bool mixed = exchange.has_value() && exchange->family != coulomb->family;

    // The disk rung is a rung of the RI-J link family, and the driver's disk
    // route exposes no per-iteration half to either the energy seam or the
    // per-spin assembly. A plan that reaches it is refused here.
    if (request.diskRungRequested)
    {
        plan.constraintsChecked.emplace_back("disk_rung");

        const bool diskHalfRequested =
            (coulomb->family == qcx::io::IntegralFamily::kRiJLink &&
             !coulomb->capabilities.diskRungSuppliesHalf) ||
            (exchange.has_value() && exchange->family == qcx::io::IntegralFamily::kRiJLink &&
             !exchange->capabilities.diskRungSuppliesHalf);

        if (diskHalfRequested)
        {
            return std::unexpected(detail::PlanError(
                qcx::ErrorCode::kUnimplemented,
                "the RI disk rung was requested (method.ri_tensor_mode = \"disk\" or "
                "[diagnostics] force_disk_ri), and a half of this plan is supplied by the "
                "RI-J link family whose disk route exposes no per-iteration half: the run "
                "would have no J[D] (or no per-spin exchange) for its seam. Drop the rung "
                "key, or name a provider whose halves are wired on the rung"));
        }

        // The rung's route is ONE whole-family builder (the storage-module disk
        // builder). A plan assembled from two providers has no such builder, so
        // the request could only be dropped - refused rather than run as if the
        // key were absent.
        if (mixed)
        {
            return std::unexpected(detail::PlanError(
                qcx::ErrorCode::kUnimplemented,
                "the RI disk rung was requested (method.ri_tensor_mode = \"disk\" or "
                "[diagnostics] force_disk_ri), and this plan draws its halves from two "
                "providers: the rung is one whole-family builder's storage route, and a "
                "mixed-half plan has no single builder to put on it. Drop the rung key, or "
                "request one family for both halves"));
        }
    }

    // The device boundary: the GPU builder runs the direct family's algorithm
    // and exposes no half. A half request on the device is therefore refused.
    if (request.backend != qcx::io::ExecutionBackend::kCpu)
    {
        plan.constraintsChecked.emplace_back("device_boundary");

        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kUnimplemented,
            "builder.execution_backend = \"" + std::string(qcx::io::ToString(request.backend)) +
                "\" names the device path, whose builder exposes no Fock half: a half "
                "request cannot be served on the device. Drop the half keys and request the "
                "whole device family, or run the halves on the CPU"));
    }

    // The mixed-plan option scope: a key that configures one family's whole
    // builder cannot be attached to one half of a pair drawn from two families.
    if (mixed)
    {
        plan.constraintsChecked.emplace_back("option_scope");
    }

    if (mixed && request.orbitExpansionRequested)
    {
        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kUnimplemented,
            "method.ri_orbit_expansion configures a whole RI family builder's task grid, and "
            "this plan draws its halves from two providers: there is no single grid to "
            "expand. Drop the key, or request the whole ri_jk family"));
    }

    if (mixed && request.eriStoreRequested)
    {
        return std::unexpected(detail::PlanError(
            qcx::ErrorCode::kUnimplemented,
            "method.eri_cache_store decorates the direct family's engine pair and binds the "
            "system fingerprint through that family's auxiliary selection; a mixed-half plan "
            "has no single pair to decorate. Drop the key, or request the whole direct "
            "family"));
    }

    plan.constraintsChecked.emplace_back("exchange_capability");

    plan.coulomb = detail::MakeHalfPlan(*coulomb, request.coulombAuxBasis);

    if (exchange.has_value())
    {
        plan.exchange = detail::MakeHalfPlan(*exchange, request.exchangeAuxBasis);
    }

    plan.sameProvider = exchange.has_value() && !mixed;
    plan.mixed = mixed;
    plan.reasoning =
        "builder.coulomb_half = \"" + plan.coulomb.provider + "\", builder.exchange_half = \"" +
        (plan.exchange.has_value() ? plan.exchange->provider : std::string("none")) +
        "\": the Coulomb half is supplied by the " + plan.coulomb.provider + " provider (" +
        plan.coulomb.kernel + " kernel, " + plan.coulomb.accounting + " accounting)" +
        (plan.exchange.has_value()
             ? ", the exchange half by the " + plan.exchange->provider + " provider (" +
                   plan.exchange->kernel + " kernel, " + plan.exchange->accounting + " accounting)"
             : std::string(", and the run builds no exchange term")) +
        (plan.mixed ? "; the halves are separate builders and share no intermediate"
                    : "; both halves come from one builder and share its intermediate");

    if (request.leanMemberRequested && plan.mixed)
    {
        plan.reasoning +=
            "; the resolved lean member supplies the direct family's half (the other provider's "
            "member is that family's own)";
    }

    // The fitted provider's own cost, stated because a mixed plan pays it
    // without consuming the result: the RI-J link's exchange half is its exact
    // nested direct kernel and the family's Create still builds the Coulomb
    // tensor, while the full-RI pair computes both halves in one traversal and
    // this plan reads the exchange one.
    if (plan.exchange.has_value() && plan.exchange->provider == "ri_j_link")
    {
        plan.reasoning +=
            "; the ri_j_link provider's Create builds its RI-J tensor, which this plan does not "
            "consume - only its nested exact exchange half is used";
    }

    if (plan.exchange.has_value() && plan.exchange->provider == "ri_jk")
    {
        plan.reasoning +=
            "; the full-RI pair computes both halves in one traversal and this plan consumes the "
            "exchange half only";
    }

    return plan;
}

/// Plans the Fock build against the registered provider table.
/// \param request The half families and the run context.
/// \returns The plan, or the refusal (see the table overload).
/// \ingroup qcx-driver
inline qcx::Result<FockBuildPlan> PlanFockBuild(const FockPlanRequest& request) {
    return PlanFockBuild(request, FockHalfProviders());
}

} // namespace qcx::driver
