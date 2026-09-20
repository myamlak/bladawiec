#include "qcx/backend/cpu_backend.hpp"
#include "qcx/memory/first_touch.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

using qcx::memory::AllocatorTraits;
using qcx::memory::kCacheLineBytes;
using qcx::memory::TouchPagesAcrossTeam;

namespace {

// A deterministic per-index pattern: exercises every byte value class
// (zero and non-zero) so the non-destructiveness pin is not vacuous.
double PatternOf(std::size_t i) {
    return (i % 7 == 0) ? 0.0 : static_cast<double>(i % 101) * 0.5;
}

} // namespace

TEST(FirstTouchTest, WindowPassPreservesEveryByte) {
    // 1 MiB of doubles plus an odd tail: the window is not a multiple of
    // the line size, so the ragged last line is covered by the pin.
    const std::size_t count = (1u << 20) + 3;
    std::vector<double> block(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        block[i] = PatternOf(i);
    }

    AllocatorTraits<qcx::backend::CpuTag>::TouchPages(block.data(), count * sizeof(double));

    for (std::size_t i = 0; i < count; ++i)
    {
        ASSERT_EQ(block[i], PatternOf(i)) << "byte at index " << i;
    }
}

TEST(FirstTouchTest, WindowPassDegenerateInputsAreNoOps) {
    std::vector<double> block(256, 1.0);
    double* data = block.data();

    AllocatorTraits<qcx::backend::CpuTag>::TouchPages(nullptr, 1024);
    AllocatorTraits<qcx::backend::CpuTag>::TouchPages(data, 0);

    for (double value : block)
    {
        ASSERT_EQ(value, 1.0);
    }
}

TEST(FirstTouchTest, WindowPassCoversSubLineWindows) {
    // Windows smaller than one cache line still get their single store,
    // and it lands inside the window.
    double single = PatternOf(3);

    AllocatorTraits<qcx::backend::CpuTag>::TouchPages(&single, sizeof(double));
    ASSERT_EQ(single, PatternOf(3));
}

TEST(FirstTouchTest, AcrossTeamPassPreservesEveryByte) {
    // 2 MiB: larger than any cached team's stripe footprint, so the pass
    // stripes on multi-threaded machines and degenerates to the serial
    // path on single-threaded ones — both must preserve the contents.
    const std::size_t count = (2u << 20) + 5;
    std::vector<double> block(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        block[i] = PatternOf(i);
    }

    TouchPagesAcrossTeam(block.data(), count * sizeof(double));

    for (std::size_t i = 0; i < count; ++i)
    {
        ASSERT_EQ(block[i], PatternOf(i)) << "byte at index " << i;
    }
}

TEST(FirstTouchTest, AcrossTeamPassCoversBlocksBelowOneStripePerThread) {
    // Fewer lines than team threads: the stripe count collapses to the
    // line count, every stripe owns at least one line, and the ragged
    // tail is covered.
    const std::size_t count = 100;
    std::vector<double> block(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        block[i] = PatternOf(i);
    }

    TouchPagesAcrossTeam(block.data(), count * sizeof(double));

    for (std::size_t i = 0; i < count; ++i)
    {
        ASSERT_EQ(block[i], PatternOf(i)) << "byte at index " << i;
    }
}

TEST(FirstTouchTest, AcrossTeamPassDegenerateInputsAreNoOps) {
    TouchPagesAcrossTeam(nullptr, 1024);
    TouchPagesAcrossTeam(nullptr, 0);
    TouchPagesAcrossTeam(nullptr, kCacheLineBytes);
}
