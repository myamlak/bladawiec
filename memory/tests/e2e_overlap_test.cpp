#include "qcx/backend/cpu_backend.hpp"
#include "qcx/memory/tensor.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace {

// Closed-form primitive s-s Gaussian overlap (App. A, [Szabo1989]). Used
// only to validate core+memory+backend end-to-end - not the real
// integrals implementation, which belongs in a future basisset/integrals
// module.
double SOverlap(double a, double b, double rSquared) {
    constexpr double kPi = 3.14159265358979323846;
    double p = a + b;
    return std::pow(kPi / p, 1.5) * std::exp(-a * b / p * rSquared);
}

} // namespace

TEST(EndToEndOverlapTest, SymmetricAndDecaysWithDistance) {
    std::vector<double> exponents = {1.0, 0.5, 2.0};
    std::vector<double> centersR = {0.0, 1.0, 2.0}; // 1D positions, for simplicity
    const std::size_t n = exponents.size();

    auto overlap = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({n, n});
    ASSERT_TRUE(overlap.has_value());
    qcx::backend::Backend<qcx::backend::CpuTag> backend;

    backend.ParallelFor(n, [&](std::size_t i) {
        for (std::size_t j = 0; j < n; ++j)
        {
            double diff = centersR[i] - centersR[j];
            (*overlap)(i, j) = SOverlap(exponents[i], exponents[j], diff * diff);
        }
    });

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            EXPECT_DOUBLE_EQ((*overlap)(i, j), (*overlap)(j, i));
        }
    }

    // Same-exponent-pair overlap should shrink as the centers move apart.
    EXPECT_GT((*overlap)(0, 0), (*overlap)(0, 2));
}
