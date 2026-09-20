// The certified-bound sweep: for every canonical quartet class of the
// shellset fixture (s/p/d - every class up to (4,4)) and of the HF/H2O
// fixtures, the certified fp32 batch must deliver a-priori bounds that
// dominate the actual element-wise deviation from the fp64 batch. This is
// the systematic stress walk the routing gate of fock_build.hpp relies on:
// a bound that fails to dominate anywhere means the gate can admit an
// integral whose error exceeds the preset budget. The HF/H2O walks are
// fast-mode-gated like the other heavy sweeps.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "shellset_fixture.hpp"

#include <cmath>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeHfSto3g;
using qcx::testing::MakeHfSto3gBasis;

void SweepFixture(const qcx::molecule::Molecule& molecule,
                  const qcx::basisset::BasisSet& basisSet) {
    auto pairList = qcx::integrals::BuildShellPairs(molecule, basisSet);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    std::vector<qcx::integrals::ShellQuartet> quartets;
    const std::size_t nShells = pairList->shells.size();

    for (std::size_t a = 0; a < nShells; ++a)
    {
        for (std::size_t b = a; b < nShells; ++b)
        {
            for (std::size_t c = 0; c < nShells; ++c)
            {
                for (std::size_t d = c; d < nShells; ++d)
                {
                    if (qcx::integrals::PairIndexOf(a, b, *pairList) <
                        qcx::integrals::PairIndexOf(c, d, *pairList))
                    {
                        continue;
                    }

                    quartets.push_back({a, b, c, d});
                }
            }
        }
    }

    auto fp64Batch = qcx::integrals::ComputeEriBatch(molecule, basisSet, quartets);
    ASSERT_TRUE(fp64Batch.has_value()) << fp64Batch.error().message;
    auto fp32Batch = qcx::integrals::ComputeEriBatchCertified(molecule, basisSet, quartets);
    ASSERT_TRUE(fp32Batch.has_value()) << fp32Batch.error().message;
    ASSERT_EQ(fp32Batch->computed.size(), fp64Batch->computed.size());
    ASSERT_EQ(fp32Batch->errorBounds.size(), fp32Batch->computed.size());

    std::size_t offset = 0;

    for (std::size_t t = 0; t < fp64Batch->computed.size(); ++t)
    {
        const qcx::integrals::ShellQuartet& quartet = fp64Batch->computed[t];
        const std::size_t nI = ShellFunctionCount(pairList->shells[quartet.i]);
        const std::size_t nJ = ShellFunctionCount(pairList->shells[quartet.j]);
        const std::size_t nK = ShellFunctionCount(pairList->shells[quartet.k]);
        const std::size_t nL = ShellFunctionCount(pairList->shells[quartet.l]);
        const std::size_t blockSize = nI * nJ * nK * nL;
        const double bound = fp32Batch->errorBounds[t];
        EXPECT_GT(bound, 0.0) << "quartet " << t << " has no certified bound";

        double maxDeviation = 0.0;

        for (std::size_t element = 0; element < blockSize; ++element)
        {
            const double fp64Value = fp64Batch->values[offset + element];
            const double fp32Value = static_cast<double>(fp32Batch->values[offset + element]);
            maxDeviation = std::max(maxDeviation, std::abs(fp32Value - fp64Value));
        }

        // The bound carries a 2x derivation margin; it must dominate the
        // measured deviation outright.
        EXPECT_LE(maxDeviation, bound)
            << "quartet " << t << " shells (" << quartet.i << "," << quartet.j << "," << quartet.k
            << "," << quartet.l << "): deviation " << maxDeviation
            << " exceeds the certified bound " << bound;

        offset += blockSize;
    }
}

TEST(CertifiedBoundSweepTest, ShellsetBoundsDominate) {
    auto basis = qcx::basisset::ParseNwchemText(qcx::testing::kShellsetBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = qcx::testing::MakeShellsetMolecule();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    SweepFixture(*molecule, *basis);
}

TEST(CertifiedBoundSweepTest, HfBoundsDominate) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    SweepFixture(*molecule, *basis);
}

TEST(CertifiedBoundSweepTest, H2oBoundsDominate) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    SweepFixture(*molecule, *basis);
}

} // namespace
