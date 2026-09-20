// Generated-table invariants of the MD engine (md_tables_gen.hpp,
// tools/gen_md_tables.py): the counts against their closed forms, the
// Hermite parity masks, the index-map bijectivity, and the solid-harmonic
// orthonormality G W G^T = I (W = the angular monomial overlap; the
// generator verifies the same identity at 30 digits). The byte-identity
// regeneration check runs as the md-tables-regeneration ctest.

#include "internal/md_defs.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace {

int ClosedHermite2D(int l) {
    return (l + 1) * (l + 2) / 2;
}

int ClosedHermite3D(int l) {
    return (l + 1) * (l + 2) * (l + 3) / 6;
}

TEST(MdTablesTest, CountsMatchClosedForms) {
    for (int l = 0; l <= qcx::integrals::internal::kMaxShellL; ++l)
    {
        EXPECT_EQ(qcx::integrals::internal::kCartesianCount[l],
                  qcx::integrals::internal::CartesianCount(l));
        EXPECT_EQ(qcx::integrals::internal::kSphericalCount[l],
                  qcx::integrals::internal::SphericalCount(l));
    }

    for (int l = 0; l <= 12; ++l)
    {
        EXPECT_EQ(qcx::integrals::internal::kHermite2DCount[l], ClosedHermite2D(l));
        EXPECT_EQ(qcx::integrals::internal::kHermite3DCount[l], ClosedHermite3D(l));
        EXPECT_EQ(qcx::integrals::internal::kH2Prefix[l], l * (l + 1) * (l + 2) / 6);
    }
}

TEST(MdTablesTest, CartesianIndicesAreBijective) {
    for (int l = 0; l <= qcx::integrals::internal::kMaxShellL; ++l)
    {
        for (int index = 0; index < qcx::integrals::internal::CartesianCount(l); ++index)
        {
            const auto component = qcx::integrals::internal::kCartesianIndices[l][index];
            EXPECT_EQ(component.ix + component.iy + component.iz, l);
            // The descending-lexicographic order: (ix, iy, iz) strictly
            // decreases as triples.
            if (index + 1 < qcx::integrals::internal::CartesianCount(l))
            {
                const auto next = qcx::integrals::internal::kCartesianIndices[l][index + 1];
                const bool ordered =
                    component.ix > next.ix || (component.ix == next.ix && component.iy > next.iy) ||
                    (component.ix == next.ix && component.iy == next.iy && component.iz > next.iz);
                EXPECT_TRUE(ordered);
            }
        }
    }
}

TEST(MdTablesTest, Hermite3DIndexIsBijective) {
    // Every triple with |t| <= L lands on a distinct index in
    // [0, Hermite3DCount(L)).
    constexpr int kL = 12;
    std::vector<int> seen(qcx::integrals::internal::Hermite3DCount(kL), 0);

    for (int tx = 0; tx <= kL; ++tx)
    {
        for (int ty = 0; ty <= kL - tx; ++ty)
        {
            for (int tz = 0; tz <= kL - tx - ty; ++tz)
            {
                const int index = qcx::integrals::internal::Hermite3DIndex(tx, ty, tz);
                ASSERT_GE(index, 0);
                ASSERT_LT(index, static_cast<int>(seen.size()));
                EXPECT_EQ(seen[index], 0)
                    << "collision at (" << tx << "," << ty << "," << tz << ")";
                seen[index] = 1;
            }
        }
    }

    for (int v : seen)
    {
        EXPECT_EQ(v, 1);
    }
}

TEST(MdTablesTest, ParityMasksMatchTheParityRule) {
    // E_t^{(i,j)} vanishes unless t = i + j (mod 2) - bit t of
    // kParityMask[i][j] is set exactly then.
    for (int i = 0; i <= qcx::integrals::internal::kMaxShellL; ++i)
    {
        for (int j = 0; j <= qcx::integrals::internal::kMaxShellL; ++j)
        {
            const unsigned int mask = qcx::integrals::internal::kParityMask[i][j];

            for (int t = 0; t <= i + j; ++t)
            {
                EXPECT_EQ((mask >> t) & 1u, static_cast<unsigned int>((t - i - j) % 2 == 0 ? 1 : 0))
                    << "la=" << i << " lb=" << j << " t=" << t;
            }
        }
    }
}

TEST(MdTablesTest, SolidHarmonicsAreOrthonormal) {
    // The physics-defining property: the tables are the orthonormal real
    // spherical harmonics, so <Y_lm | Y_l'm'> over the unit sphere is
    // delta. In the monomial basis that is G W G^T = I with
    // W[a][b] = int x^{i+i'} y^{j+j'} z^{k+k'} dOmega - the closed form
    // 2 Gamma((A+1)/2) Gamma((B+1)/2) Gamma((C+1)/2) / Gamma((A+B+C+3)/2),
    // zero for odd exponents. l = 0 is the s-identity exception (a
    // spherical s function equals the Cartesian s function; its sphere
    // integral is 4 pi, not 1).
    for (int l = 1; l <= qcx::integrals::internal::kMaxShellL; ++l)
    {
        const int nSpherical = 2 * l + 1;
        const int nCartesian = qcx::integrals::internal::CartesianCount(l);
        std::vector<std::vector<double>> w(nCartesian, std::vector<double>(nCartesian, 0.0));

        for (int a = 0; a < nCartesian; ++a)
        {
            const qcx::integrals::internal::CartIndex ca =
                qcx::integrals::internal::kCartesianIndices[l][a];

            for (int b = 0; b < nCartesian; ++b)
            {
                const qcx::integrals::internal::CartIndex cb =
                    qcx::integrals::internal::kCartesianIndices[l][b];
                const int ax = ca.ix + cb.ix;
                const int ay = ca.iy + cb.iy;
                const int az = ca.iz + cb.iz;

                if (ax % 2 != 0 || ay % 2 != 0 || az % 2 != 0)
                {
                    continue;
                }

                w[a][b] = 2.0 * std::tgamma((ax + 1) * 0.5) * std::tgamma((ay + 1) * 0.5) *
                          std::tgamma((az + 1) * 0.5) / std::tgamma((ax + ay + az + 3) * 0.5);
            }
        }

        for (int m1 = 0; m1 < nSpherical; ++m1)
        {
            for (int m2 = 0; m2 < nSpherical; ++m2)
            {
                double value = 0.0;

                for (int a = 0; a < nCartesian; ++a)
                {
                    for (int b = 0; b < nCartesian; ++b)
                    {
                        value += qcx::integrals::internal::kSolidHarmonicG[l][m1][a] * w[a][b] *
                                 qcx::integrals::internal::kSolidHarmonicG[l][m2][b];
                    }
                }

                EXPECT_NEAR(value, m1 == m2 ? 1.0 : 0.0, 1e-13)
                    << "l=" << l << " m1=" << m1 << " m2=" << m2;
            }
        }
    }
}

TEST(MdTablesTest, AmplificationConstantsArePositive) {
    // The class constants must be positive and non-decreasing in the
    // total class (larger recurrence DAGs amplify no less).
    for (int lBra = 0; lBra <= 12; ++lBra)
    {
        for (int lKet = lBra; lKet <= 12; ++lKet)
        {
            const double c = qcx::integrals::internal::kClassAmplification[lBra][lKet];
            EXPECT_GT(c, 0.0) << "class (" << lBra << "," << lKet << ")";

            if (lKet + 1 <= 12)
            {
                EXPECT_LE(c, qcx::integrals::internal::kClassAmplification[lBra][lKet + 1]);
            }
        }
    }
}

} // namespace
