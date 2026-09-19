// The F16/Bf16 conversion contract (integrals/include/qcx/integrals/f16.hpp):
// round-to-nearest-even float -> half on the I/O boundary, verified bit by
// bit against the IEEE-754 binary16 / bfloat16 rounding rules. The cases
// cover every branch of FloatToHalf (Inf/NaN quieting, the 65520 RNE
// overflow threshold, normal rounding, the 2^-25 zero-threshold tie, the
// subnormal carry into 0x0400) and the RNE bit trick of FloatToBf16. The
// assertions go through detail::F16Bits / F16FromBits so the same table
// runs on the MSVC wrapper path and the stdfloat path (GCC/Clang CI).

#include "qcx/integrals/f16.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace {

using qcx::integrals::Bf16;
using qcx::integrals::F16;
using qcx::integrals::detail::Bf16Bits;
using qcx::integrals::detail::F16Bits;
using qcx::integrals::detail::F16FromBits;

TEST(F16ConversionTest, FloatToHalfRoundsToNearestEven) {
    // Not constexpr: MSVC's STL does not make std::ldexp constexpr (GCC/
    // Clang do, so a constexpr table would compile on CI and fail locally).
    struct Case {
        float input;
        std::uint16_t expectedBits;
        const char* note;
    };

    const Case cases[] = {
        {1.0f, 0x3C00u, "1.0"},
        {-1.0f, 0xBC00u, "-1.0"},
        {0.0f, 0x0000u, "0.0"},
        {-0.0f, 0x8000u, "-0.0 (sign preserved)"},
        {std::ldexp(1.0f, -25), 0x0000u, "the 2^-25 tie rounds to even (zero)"},
        {std::ldexp(1.0f, -25) + std::ldexp(1.0f, -48), 0x0001u, "just above the tie"},
        {std::ldexp(1.0f, -24), 0x0001u, "the smallest subnormal"},
        {std::ldexp(1.0f, -14), 0x0400u, "the smallest normal half"},
        {std::ldexp(1.0f, -14) - std::ldexp(1.0f, -25),
         0x0400u,
         "exact tie between 0x03FF and 0x0400 - even mantissa wins"},
        {65504.0f, 0x7BFFu, "the half maximum"},
        {65520.0f, 0x7C00u, "the RNE overflow threshold rounds to Inf"},
        {std::numeric_limits<float>::infinity(), 0x7C00u, "+Inf"},
        {-std::numeric_limits<float>::infinity(), 0xFC00u, "-Inf"},
        {std::numeric_limits<float>::quiet_NaN(),
         0x7E00u,
         "NaN keeps the payload, sets the quiet bit"},
    };

    for (const Case& c : cases)
    {
        EXPECT_EQ(F16Bits(F16(c.input)), c.expectedBits) << c.note;
    }
}

TEST(F16ConversionTest, FloatToBf16RoundsToNearestEven) {
    // Not constexpr: MSVC's STL does not make std::ldexp constexpr (GCC/
    // Clang do, so a constexpr table would compile on CI and fail locally).
    struct Case {
        float input;
        std::uint16_t expectedBits;
        const char* note;
    };

    const Case cases[] = {
        {1.0f, 0x3F80u, "1.0"},
        {-1.0f, 0xBF80u, "-1.0"},
        {-0.0f, 0x8000u, "-0.0 (sign preserved)"},
        {std::bit_cast<float>(0x3F7FFFFFu), 0x3F80u, "the value just below 1.0 rounds up"},
        {std::bit_cast<float>(0x7F7FFFFFu), 0x7F80u, "the float maximum rounds up to Inf"},
        {std::bit_cast<float>(0x7F800000u), 0x7F80u, "+Inf"},
        {std::bit_cast<float>(0x7F800001u), 0x7F80u, "NaN"},
    };

    for (const Case& c : cases)
    {
        EXPECT_EQ(Bf16Bits(Bf16(c.input)), c.expectedBits) << c.note;
    }
}

TEST(F16ConversionTest, WideningIsExact) {
    // binary16 -> binary32 is exact: the widened value must reproduce the
    // stored pattern's exact value (checked against the raw-bit widening).
    EXPECT_EQ(static_cast<float>(F16FromBits(0x3C00u)), 1.0f);
    EXPECT_EQ(static_cast<float>(F16FromBits(0x0001u)), std::ldexp(1.0f, -24));
    EXPECT_EQ(static_cast<float>(F16FromBits(0x0400u)), std::ldexp(1.0f, -14));
    EXPECT_EQ(static_cast<float>(F16FromBits(0x7BFFu)), 65504.0f);
    EXPECT_EQ(static_cast<float>(F16FromBits(0x8000u)), -0.0f);
    EXPECT_TRUE(std::isnan(static_cast<float>(F16FromBits(0x7E00u))));
    EXPECT_TRUE(std::isinf(static_cast<float>(F16FromBits(0x7C00u))));
}

TEST(F16ConversionTest, RoundTripThroughTheBoundary) {
    // A value exactly representable in half survives float -> F16 -> float
    // bit-identically in both directions.
    EXPECT_EQ(static_cast<float>(F16(0.5f)), 0.5f);
    EXPECT_EQ(static_cast<float>(F16(-2.0f)), -2.0f);
    EXPECT_EQ(static_cast<float>(F16(65504.0f)), 65504.0f);
}

} // namespace
