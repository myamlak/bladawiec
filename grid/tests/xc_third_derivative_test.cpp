// The excgrid kernel seam as this consumer pins it: the contract version, the
// third-derivative entry point's signature, its default refusal, and the wire
// values of Request and KernelStatus. No third-derivative tier is built, so
// the default refusal and the wrapper that reports it are the behaviour this
// tree ships; the pins are what make a seam change move this tree instead of
// silently reinterpreting it.

#include "qcx/grid/xc_third_derivative.hpp"

#include <array>
#include <cstddef>
#include <excgrid/kernel.hpp>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace qcx::grid {
namespace {

// The seam's signature, pinned as a member-pointer type: a changed parameter
// list, constness or noexcept specification stops this translation unit.
using ThirdDerivativeEntry =
    excgrid::KernelStatus (excgrid::XcFunctional::*)(const excgrid::PointInputs&,
                                                     std::span<const double>,
                                                     excgrid::PointResult&,
                                                     std::span<double>) const;

static_assert(
    std::is_same_v<decltype(&excgrid::XcFunctional::EvaluatePointWithThirdDerivatives),
                   ThirdDerivativeEntry>,
    "the excgrid third-derivative seam changed: move the pinned tag and this consumer together");

static_assert(
    std::is_same_v<decltype(&excgrid::ContractVersion), excgrid::SchemaVersion (*)() noexcept>,
    "the excgrid contract-version entry point changed");

// The request and status values cross a boundary, so their numbers are part of
// the contract rather than an implementation detail.
static_assert(static_cast<int>(excgrid::Request::kFirstDerivatives) == 0);
static_assert(static_cast<int>(excgrid::Request::kSecondDerivativeContraction) == 1);
static_assert(static_cast<int>(excgrid::Request::kSecondDerivativeMatrix) == 2);
static_assert(static_cast<int>(excgrid::Request::kThirdDerivativeContraction) == 3);

static_assert(static_cast<int>(excgrid::KernelStatus::kOk) == 0);
static_assert(static_cast<int>(excgrid::KernelStatus::kRefusedUnsupportedCapability) == 1);
static_assert(static_cast<int>(excgrid::KernelStatus::kRefusedMissingProvider) == 2);
static_assert(static_cast<int>(excgrid::KernelStatus::kRefusedExhaustedCapacity) == 3);
static_assert(static_cast<int>(excgrid::KernelStatus::kRefusedVersionMismatch) == 4);
static_assert(static_cast<int>(excgrid::KernelStatus::kRefusedUnsupportedCombination) == 5);
static_assert(static_cast<int>(excgrid::KernelStatus::kRefusedSecondDerivativeCoverage) == 6);
static_assert(static_cast<int>(excgrid::KernelStatus::kRefusedThirdDerivativeCoverage) == 7);

// A functional with an order-1 answer and no derivative tier at all: it
// overrides nothing optional, so every tier question gets the seam's default.
class FirstTierOnlyFunctional : public excgrid::XcFunctional {
public:
    [[nodiscard]] bool UsesGradient() const override {
        return false;
    }

    [[nodiscard]] double ExchangeFraction() const override {
        return 0.0;
    }

    [[nodiscard]] excgrid::ComponentMask RequiredMask() const override {
        excgrid::ComponentMask mask;
        mask.Set(excgrid::Component::RhoA);
        mask.Set(excgrid::Component::RhoB);

        return mask;
    }

    [[nodiscard]] excgrid::KernelStatus EvaluatePoint(const excgrid::PointInputs& inputs,
                                                      excgrid::PointResult& result) const override {
        result.version = excgrid::kSchemaVersion;
        result.mask = RequiredMask();
        result.exc = inputs.Get(excgrid::Component::RhoA) + inputs.Get(excgrid::Component::RhoB);
        result.first[excgrid::IndexOf(excgrid::Component::RhoA)] = 1.0;
        result.first[excgrid::IndexOf(excgrid::Component::RhoB)] = 1.0;

        return excgrid::KernelStatus::kOk;
    }
};

// A functional whose tier answers, so the wrapper's success path is exercised
// without a shipped tier existing: the wrapper forwards what the seam returned
// and contributes no arithmetic of its own.
class AnsweringThirdTierFunctional final : public FirstTierOnlyFunctional {
public:
    [[nodiscard]] excgrid::KernelStatus EvaluatePointWithThirdDerivatives(
        const excgrid::PointInputs& inputs,
        std::span<const double> rightHandSide,
        excgrid::PointResult& result,
        std::span<double> contracted) const override {
        result.version = excgrid::kSchemaVersion;
        result.exc = inputs.Get(excgrid::Component::RhoA);
        contracted[0] = rightHandSide.front();

        return excgrid::KernelStatus::kOk;
    }
};

TEST(ExcgridKernelContractTest, ContractVersionIsPinned) {
    const excgrid::SchemaVersion version = excgrid::ContractVersion();

    EXPECT_EQ(version.major, 1);
    EXPECT_EQ(version.minor, 0);
    EXPECT_TRUE(version.CompatibleWith(excgrid::kSchemaVersion));
    EXPECT_TRUE(version.CompatibleWith(excgrid::PointResult{}.version));
}

TEST(ExcgridKernelContractTest, TheSeamRefusesByDefaultAndLeavesTheOutputsUntouched) {
    const FirstTierOnlyFunctional functional;

    excgrid::PointInputs inputs;
    inputs.Set(excgrid::Component::RhoA, 1.5);
    inputs.Set(excgrid::Component::RhoB, 0.5);

    excgrid::PointResult result;
    result.version = {9, 9};
    result.mask.Set(excgrid::Component::RhoA);
    result.exc = 7.0;
    result.first[0] = 3.0;

    std::array<double, 4> contracted{};
    contracted.fill(5.0);
    const std::array<double, 2> rightHandSide{0.25, -0.5};

    EXPECT_EQ(
        functional.EvaluatePointWithThirdDerivatives(inputs, rightHandSide, result, contracted),
        excgrid::KernelStatus::kRefusedUnsupportedCapability);

    // A refusal is not an answer: no output slot moved.
    EXPECT_EQ(result.version.major, 9);
    EXPECT_EQ(result.version.minor, 9);
    EXPECT_EQ(result.exc, 7.0);
    EXPECT_EQ(result.first[0], 3.0);

    for (const double value : contracted)
    {
        EXPECT_EQ(value, 5.0);
    }
}

TEST(ExcgridKernelContractTest, EveryShippedFunctionalRefusesTheThirdTier) {
    const std::span<const std::string_view> names = excgrid::FunctionalNames();

    ASSERT_FALSE(names.empty());

    for (const std::string_view name : names)
    {
        const excgrid::XcFunctional* functional = excgrid::FindFunctional(name);

        ASSERT_NE(functional, nullptr) << name;

        excgrid::PointInputs inputs;
        inputs.Set(excgrid::Component::RhoA, 0.5);
        inputs.Set(excgrid::Component::RhoB, 0.5);

        excgrid::PointResult result;
        std::array<double, 2> contracted{};
        const std::array<double, 1> rightHandSide{1.0};

        EXPECT_EQ(functional->EvaluatePointWithThirdDerivatives(
                      inputs, rightHandSide, result, contracted),
                  excgrid::KernelStatus::kRefusedUnsupportedCapability)
            << name;
    }
}

TEST(XcThirdDerivativeContractionTest, TheWrapperReportsTheRefusalAsAnError) {
    const FirstTierOnlyFunctional functional;

    excgrid::PointInputs inputs;
    inputs.Set(excgrid::Component::RhoA, 1.0);

    excgrid::PointResult result;
    result.exc = 7.0;
    std::array<double, 2> contracted{};
    contracted.fill(5.0);
    const std::array<double, 1> rightHandSide{1.0};

    const qcx::Result<void> outcome = EvaluateXcKernelThirdDerivativeContraction(
        functional, inputs, rightHandSide, result, contracted);

    ASSERT_FALSE(outcome.has_value());
    EXPECT_EQ(outcome.error().code, qcx::ErrorCode::kUnimplemented);

    const std::string_view refusal =
        excgrid::DescribeStatus(excgrid::KernelStatus::kRefusedUnsupportedCapability);

    EXPECT_NE(outcome.error().message.find(refusal), std::string::npos);

    // The wrapper translated a refusal; it did not fill the outputs.
    EXPECT_EQ(result.exc, 7.0);

    for (const double value : contracted)
    {
        EXPECT_EQ(value, 5.0);
    }
}

TEST(XcThirdDerivativeContractionTest, TheWrapperForwardsASuccessfulAnswer) {
    const AnsweringThirdTierFunctional functional;

    excgrid::PointInputs inputs;
    inputs.Set(excgrid::Component::RhoA, 1.25);

    excgrid::PointResult result;
    std::array<double, 1> contracted{};
    const std::array<double, 1> rightHandSide{0.75};

    const qcx::Result<void> outcome = EvaluateXcKernelThirdDerivativeContraction(
        functional, inputs, rightHandSide, result, contracted);

    ASSERT_TRUE(outcome.has_value());
    EXPECT_EQ(result.exc, 1.25);
    EXPECT_EQ(contracted[0], 0.75);
}

} // namespace
} // namespace qcx::grid
