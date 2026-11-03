// The Fock-build planner: the provider interface, the constraint vocabulary and
// the owner's acceptance pair.
//
// The arithmetic the planner decides is pinned here at the DECISION level: the
// provider per half, the accounting each adapter speaks, and the constraint a
// refused pair broke. The execution side of the acceptance pair is pinned in
// run_driver_test.cpp (the json carries `fock_plan`), because a planner test
// that stopped at the plan would be the "plans something it cannot execute"
// failure the design names.

#include "qcx/driver/fock_planner.hpp"

#include <algorithm>
#include <array>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <vector>

namespace {

using qcx::driver::FockBuildPlan;
using qcx::driver::FockHalfAccounting;
using qcx::driver::FockHalfCapabilities;
using qcx::driver::FockHalfProvider;
using qcx::driver::FockHalfRole;
using qcx::driver::FockPlanRequest;
using qcx::driver::FockProviderClass;
using qcx::driver::PlanFockBuild;
using qcx::io::ExecutionBackend;
using qcx::io::IntegralFamily;

// The owner's acceptance example in this tree's precise vocabulary: an exact
// direct Coulomb half against the full-RI family's occ-RI-K exchange half.
// (The brief's shorthand "ri_j_link" names the RI-J link family, whose exchange
// half is the exact nested direct kernel - see the literal-spelling case
// below.)
TEST(FockPlannerTest, DirectCoulombAgainstDensityFittedExchangeIsPlanned) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kRiJk;
    request.exchangeRequired = true;
    request.exchangeAuxBasis = std::string("def2-universal-jkfit");

    auto plan = PlanFockBuild(request);

    ASSERT_TRUE(plan.has_value());
    EXPECT_TRUE(plan->mixed);
    EXPECT_FALSE(plan->sameProvider);
    EXPECT_EQ(plan->coulomb.provider, "direct");
    EXPECT_EQ(plan->coulomb.providerClass, "direct_screened");
    EXPECT_EQ(plan->coulomb.kernel, "exact");
    EXPECT_EQ(plan->coulomb.accounting, "H+2J");
    EXPECT_FALSE(plan->coulomb.auxBasis.has_value());
    ASSERT_TRUE(plan->exchange.has_value());
    EXPECT_EQ(plan->exchange->provider, "ri_jk");
    EXPECT_EQ(plan->exchange->providerClass, "density_fitted");
    EXPECT_EQ(plan->exchange->kernel, "density_fitted");
    EXPECT_EQ(plan->exchange->accounting, "K");
    ASSERT_TRUE(plan->exchange->auxBasis.has_value());
    EXPECT_EQ(*plan->exchange->auxBasis, "def2-universal-jkfit");
    // The auxiliary check ran (the exchange half is fitted); the Kohn-Sham
    // seam check did not (this is a restricted Hartree-Fock run).
    EXPECT_NE(std::find(plan->constraintsChecked.begin(),
                        plan->constraintsChecked.end(),
                        "auxiliary_basis"),
              plan->constraintsChecked.end());
    EXPECT_EQ(std::find(plan->constraintsChecked.begin(),
                        plan->constraintsChecked.end(),
                        "coulomb_matrix"),
              plan->constraintsChecked.end());
}

// The spelling the task brief names literally: direct Coulomb against the
// RI-J link's exchange half. Its K is the family's EXACT nested direct kernel -
// the record says so - so the density-fitted result class is the ri_jk case
// above, not this one.
TEST(FockPlannerTest, DirectCoulombAgainstRiJLinkExchangeIsPlannedAndExact) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kRiJLink;
    request.exchangeAuxBasis = std::string("def2-universal-jfit");

    auto plan = PlanFockBuild(request);

    ASSERT_TRUE(plan.has_value());
    EXPECT_TRUE(plan->mixed);
    ASSERT_TRUE(plan->exchange.has_value());
    EXPECT_EQ(plan->exchange->provider, "ri_j_link");
    EXPECT_EQ(plan->exchange->providerClass, "combined");
    EXPECT_EQ(plan->exchange->kernel, "exact");
    EXPECT_EQ(plan->exchange->accounting, "H-K");
}

TEST(FockPlannerTest, OneFamilyForBothHalvesSharesTheIntermediate) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kDirect;

    auto plan = PlanFockBuild(request);

    ASSERT_TRUE(plan.has_value());
    EXPECT_TRUE(plan->sameProvider);
    EXPECT_FALSE(plan->mixed);
    ASSERT_TRUE(plan->exchange.has_value());
    EXPECT_EQ(plan->exchange->accounting, "H-K");
}

// The Kohn-Sham energy seam's constraint: a Coulomb provider that exposes no
// half of its own cannot supply J[D], and the refusal names the constraint.
TEST(FockPlannerTest, KohnShamRefusesAFusedCoulombProvider) {
    const std::array<FockHalfProvider, 2> fused{{
        {IntegralFamily::kDirect,
         FockHalfRole::kCoulomb,
         FockProviderClass::kDirectScreened,
         FockHalfAccounting::kCoulombWithCore,
         FockHalfCapabilities{false, true, false, true, false, true}},
        {IntegralFamily::kDirect,
         FockHalfRole::kExchange,
         FockProviderClass::kDirectScreened,
         FockHalfAccounting::kExchangeWithCore,
         FockHalfCapabilities{true, true, false, true, false, true}},
    }};

    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kDirect;
    request.kohnSham = true;

    auto plan = PlanFockBuild(request, fused);

    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(plan.error().message.find("Kohn-Sham energy seam"), std::string::npos);
    EXPECT_NE(plan.error().message.find("Coulomb-half"), std::string::npos);
}

// The per-spin constraint: an exchange provider with no per-spin adapter.
TEST(FockPlannerTest, PerSpinRefusesAnExchangeProviderWithNoAdapter) {
    const std::array<FockHalfProvider, 2> fused{{
        {IntegralFamily::kDirect,
         FockHalfRole::kCoulomb,
         FockProviderClass::kDirectScreened,
         FockHalfAccounting::kCoulombWithCore,
         FockHalfCapabilities{true, true, false, true, false, true}},
        {IntegralFamily::kDirect,
         FockHalfRole::kExchange,
         FockProviderClass::kDirectScreened,
         FockHalfAccounting::kExchangeWithCore,
         FockHalfCapabilities{true, false, false, true, false, true}},
    }};

    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kDirect;
    request.perSpin = true;

    auto plan = PlanFockBuild(request, fused);

    ASSERT_FALSE(plan.has_value());
    EXPECT_NE(plan.error().message.find("per-spin adapter"), std::string::npos);
    EXPECT_NE(plan.error().message.find("exchange half"), std::string::npos);
}

// The fast-multipole provider exposes both halves (BuildCoulombOnly and
// BuildExchangeOnly), so a half request may name it for K: the whole-family
// QFMM hybrid already runs that exchange half, and the registered table must
// not refuse a request the builder serves.
TEST(FockPlannerTest, QfmmSuppliesTheExchangeHalf) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kQfmm;

    auto plan = PlanFockBuild(request);

    ASSERT_TRUE(plan.has_value()) << plan.error().message;
    EXPECT_TRUE(plan->mixed);
    ASSERT_TRUE(plan->exchange.has_value());
    EXPECT_EQ(plan->exchange->provider, "qfmm");
    EXPECT_EQ(plan->exchange->accounting, "H-K");
}

TEST(FockPlannerTest, PureKohnShamRefusesAnExchangeHalfRequest) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kRiJk;
    request.kohnSham = true;
    request.exchangeRequired = false;
    request.exchangeAuxBasis = std::string("def2-universal-jkfit");

    auto plan = PlanFockBuild(request);

    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(plan.error().message.find("builds no exchange term"), std::string::npos);
}

TEST(FockPlannerTest, AFittedHalfWithoutAnAuxIsRefusedByConstraint) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kRiJk;

    auto plan = PlanFockBuild(request);

    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(plan.error().message.find("auxiliary basis"), std::string::npos);
    EXPECT_NE(plan.error().message.find("exchange half"), std::string::npos);
}

TEST(FockPlannerTest, TheDiskRungRefusesARiJLinkHalf) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kRiJLink;
    request.exchangeAuxBasis = std::string("def2-universal-jfit");
    request.diskRungRequested = true;

    auto plan = PlanFockBuild(request);

    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(plan.error().message.find("disk rung"), std::string::npos);
}

// The disk rung is ONE whole-family builder's storage route. A mixed plan whose
// halves name no ri_j_link family still cannot carry it: the request would be
// dropped, and an explicit rung key is never dropped.
TEST(FockPlannerTest, TheDiskRungRefusesAMixedPlanWithNoRiJLinkHalf) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kRiJk;
    request.exchangeAuxBasis = std::string("def2-universal-jkfit");
    request.diskRungRequested = true;

    auto plan = PlanFockBuild(request);

    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(plan.error().message.find("disk rung"), std::string::npos);
    EXPECT_NE(plan.error().message.find("mixed-half plan"), std::string::npos);
}

// A pure Kohn-Sham functional builds no exchange term: the plan is Coulomb-only,
// neither same-provider nor mixed, and the record's exchange member is absent.
TEST(FockPlannerTest, APureKohnShamPlanHasNoExchangeMember) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.kohnSham = true;
    request.exchangeRequired = false;

    auto plan = PlanFockBuild(request);

    ASSERT_TRUE(plan.has_value());
    EXPECT_FALSE(plan->exchange.has_value());
    EXPECT_FALSE(plan->mixed);
    EXPECT_FALSE(plan->sameProvider);
    EXPECT_NE(std::find(plan->constraintsChecked.begin(),
                        plan->constraintsChecked.end(),
                        "coulomb_matrix"),
              plan->constraintsChecked.end());
}

TEST(FockPlannerTest, TheDeviceBackendCannotServeAHalf) {
    FockPlanRequest request;
    request.coulombFamily = IntegralFamily::kDirect;
    request.exchangeFamily = IntegralFamily::kDirect;
    request.backend = ExecutionBackend::kGpu;

    auto plan = PlanFockBuild(request);

    ASSERT_FALSE(plan.has_value());
    EXPECT_EQ(plan.error().code, qcx::ErrorCode::kUnimplemented);
    EXPECT_NE(plan.error().message.find("device"), std::string::npos);
}

} // namespace
