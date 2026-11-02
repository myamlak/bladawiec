// The wavefunction view's contract: it copies no density, it reports the
// driver's method word and convergence state unchanged, and it carries spin
// densities only when the spin-resolved constructor supplied them.

#include "qcx/io/run_input.hpp"
#include "qcx/properties/wavefunction_view.hpp"

#include <Eigen/Core>
#include <gtest/gtest.h>
#include <type_traits>
#include <utility>

namespace {

using qcx::io::MethodType;
using qcx::properties::WavefunctionView;

Eigen::MatrixXd MakeDensity(double scale) {
    Eigen::MatrixXd density(2, 2);
    density << scale, 0.25 * scale, 0.25 * scale, 1.5 * scale;
    return density;
}

TEST(WavefunctionViewTest, TheAccessorsAreConstReferences) {
    static_assert(std::is_same_v<decltype(std::declval<const WavefunctionView&>().TotalDensity()),
                                 const Eigen::MatrixXd&>);
    static_assert(std::is_same_v<decltype(std::declval<const WavefunctionView&>().AlphaDensity()),
                                 const Eigen::MatrixXd&>);
    static_assert(std::is_same_v<decltype(std::declval<const WavefunctionView&>().BetaDensity()),
                                 const Eigen::MatrixXd&>);
    static_assert(
        std::is_same_v<decltype(std::declval<const WavefunctionView&>().Method()), MethodType>);
    static_assert(
        std::is_same_v<decltype(std::declval<const WavefunctionView&>().Converged()), bool>);
}

TEST(WavefunctionViewTest, TheDensitiesAreTheDriversMatrices) {
    const Eigen::MatrixXd total = MakeDensity(2.0);
    const WavefunctionView view(total, MethodType::kRhf, true);

    EXPECT_EQ(&view.TotalDensity(), &total);
    EXPECT_EQ(view.TotalDensity().data(), total.data());
}

TEST(WavefunctionViewTest, ARestrictedViewHasNoSpinDensities) {
    const Eigen::MatrixXd total = MakeDensity(2.0);

    for (const MethodType method : {MethodType::kRhf, MethodType::kRks})
    {
        const WavefunctionView view(total, method, true);

        EXPECT_FALSE(view.HasSpinDensities()) << qcx::io::ToString(method);
        EXPECT_FALSE(view.IsUnrestricted()) << qcx::io::ToString(method);
    }
}

TEST(WavefunctionViewTest, ASpinResolvedViewCarriesBothSpins) {
    const Eigen::MatrixXd total = MakeDensity(3.0);
    const Eigen::MatrixXd alpha = MakeDensity(2.0);
    const Eigen::MatrixXd beta = MakeDensity(1.0);

    for (const MethodType method :
         {MethodType::kUhf, MethodType::kUks, MethodType::kRohf, MethodType::kRoks})
    {
        const WavefunctionView view(total, alpha, beta, method, true);

        EXPECT_TRUE(view.HasSpinDensities());
        EXPECT_EQ(&view.TotalDensity(), &total);
        EXPECT_EQ(&view.AlphaDensity(), &alpha);
        EXPECT_EQ(&view.BetaDensity(), &beta);
    }
}

// "Restricted" is not "equal spins": the restricted open-shell pair has one
// spatial orbital set but a real spin density, so it is spin-resolved and NOT
// unrestricted. A property that needs the spins asks HasSpinDensities().
TEST(WavefunctionViewTest, RestrictedOpenShellIsSpinResolvedButNotUnrestricted) {
    const Eigen::MatrixXd total = MakeDensity(3.0);
    const Eigen::MatrixXd alpha = MakeDensity(2.0);
    const Eigen::MatrixXd beta = MakeDensity(1.0);

    for (const MethodType method : {MethodType::kRohf, MethodType::kRoks})
    {
        const WavefunctionView view(total, alpha, beta, method, true);

        EXPECT_TRUE(view.HasSpinDensities()) << qcx::io::ToString(method);
        EXPECT_FALSE(view.IsUnrestricted()) << qcx::io::ToString(method);
    }
}

TEST(WavefunctionViewTest, ACopyAddressesTheSameMatrices) {
    const Eigen::MatrixXd total = MakeDensity(2.0);
    const Eigen::MatrixXd alpha = MakeDensity(1.5);
    const Eigen::MatrixXd beta = MakeDensity(0.5);
    const WavefunctionView view(total, alpha, beta, MethodType::kUks, false);

    const WavefunctionView copy = view;

    EXPECT_EQ(&copy.TotalDensity(), &total);
    EXPECT_EQ(&copy.AlphaDensity(), &alpha);
    EXPECT_EQ(&copy.BetaDensity(), &beta);
}

TEST(WavefunctionViewTest, MethodAndConvergenceAreTheDrivers) {
    const Eigen::MatrixXd total = MakeDensity(2.0);

    for (const MethodType method : {MethodType::kRhf,
                                    MethodType::kUhf,
                                    MethodType::kRks,
                                    MethodType::kUks,
                                    MethodType::kRohf,
                                    MethodType::kRoks})
    {
        const WavefunctionView view(total, method, false);

        EXPECT_EQ(view.Method(), method);
        EXPECT_FALSE(view.Converged()) << qcx::io::ToString(method);
    }
}

} // namespace
