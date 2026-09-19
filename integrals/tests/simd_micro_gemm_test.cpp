// The AVX2 micro-GEMM tests (md_bra_simd.hpp): the SIMD kernels
// must reproduce the scalar MicroGemmAdd/MicroGemm (md_gemm.hpp) BYTE FOR
// BYTE on random micro shapes, and the runtime gate must reflect the actual
// AVX2+FMA support of the machine.
//
// EXACTNESS, not nearness. The scalar kernels are the ones every in-tree
// transform actually runs when the gate is closed, and the tier replaces
// them when it is open, so "1e-12 close" is not the contract - a run that
// takes the SIMD path must produce the same SCF trajectory as one that does
// not (the k = 1 byte pins depend on it). The tier is therefore
// contraction-neutral by construction (MulAdd, md_bra_simd.cpp) and this
// test compares raw bytes rather than a tolerance. The shapes below cover
// every branch of the two kernels: n = 1 (the single-column form), n in 2..3
// (the four-row panel, with m above and below one panel), n >= 4 (the
// n-vectorized branch, both with and without its column remainder), and a
// k beyond the small-n branches' broadcast table.
//
// ONE documented byte difference exists and cannot be closed: the assigning
// single-column form's l = 0 store, where the scalar kernel's `0 + product`
// preserves a zero's sign and the intrinsic form cannot (MSVC folds the add
// away). SimdDiffersFromScalarOnlyInTheSignOfZero is the test for it, and it
// fails if the difference ever becomes anything more than a zero's sign.
//
// The gate test is the regression pin for the defect this file's skip used
// to hide (2026-09-12): HasAvx2Fma() read the FMA flag from
// CPUID.(7,0):EBX bit 12 (RDT-M/PQM) instead of CPUID.01H:ECX bit 12, so on
// the reference machine it answered "no AVX2+FMA" - the only test of the
// tier SKIPPED there, and the dead tier went unnoticed.

#include "internal/md_bra_simd.hpp"
#include "internal/md_gemm.hpp"

#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <qcx/backend/cpu_features.hpp>
#include <random>
#include <string>
#include <vector>

#if QcxArchX86_64
// CPUID is x86-64 only: <cpuid.h> is a compile error outside x86 and
// <intrin.h> carries the intrinsic family that header belongs to. The two
// tests below that read it are excluded by the same architecture test the
// SIMD tier itself uses, so the exclusion moves with the tier rather than
// with whichever compiler happens to build this file.
#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif // QcxArchX86_64

namespace {

using qcx::backend::CpuHasAvx2;
using qcx::backend::CpuHasAvx2Fma;
using qcx::integrals::internal::HasAvx2Fma;
using qcx::integrals::internal::MicroGemm;
using qcx::integrals::internal::MicroGemmAdd;
using qcx::integrals::internal::MicroGemmAddSimd;
using qcx::integrals::internal::MicroGemmSimd;

std::vector<double> RandomMatrix(std::size_t count, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> values(count);

    for (double& value : values)
    {
        value = dist(rng);
    }

    return values;
}

/// The {m, n, k} micro shapes the transforms dispatch here, one per branch:
/// k = 1 and n = 1 (the single-column form), n = 2 and 3 with m below,
/// above and exactly one panel (the four-row panel and its scalar tail),
/// n = 4 and 6 (the n-vectorized branch, exact and with a column remainder),
/// and n = 3 at k = 20 (a shared dimension past the panel's broadcast table,
/// which must fall back to the general kernels' scalar tail).
const std::size_t kShapes[][3] = {
    {2, 1, 1},
    {5, 1, 1},
    {9, 1, 4},
    {12, 1, 10},
    {3, 2, 4},
    {4, 2, 10},
    {7, 2, 1},
    {2, 3, 5},
    {3, 3, 10},
    {8, 3, 6},
    {12, 3, 1},
    {2, 4, 4},
    {7, 5, 6},
    {13, 11, 7},
    {16, 16, 16},
    {5, 3, 20},
    {9, 2, 25},
};

TEST(SimdMicroGemmTest, SimdMatchesScalarWhenAvailable) {
    if (!HasAvx2Fma())
    {
        GTEST_SKIP() << "no AVX2+FMA on this machine";
    }

    std::mt19937_64 rng(20260817);

    for (const auto& shape : kShapes)
    {
        const std::size_t m = shape[0];
        const std::size_t n = shape[1];
        const std::size_t k = shape[2];
        const std::vector<double> a = RandomMatrix(m * k, rng);
        const std::vector<double> b = RandomMatrix(k * n, rng);
        const std::vector<double> seed = RandomMatrix(m * n, rng);
        const std::string label =
            " shape " + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

        // Accumulating form: several calls, so a first-call difference that
        // an exactly-representable term would mask cannot hide.
        std::vector<double> scalarAdd = seed;
        std::vector<double> simdAdd = seed;

        for (int call = 0; call < 8; ++call)
        {
            MicroGemmAdd(m, n, k, a.data(), b.data(), scalarAdd.data());
            MicroGemmAddSimd(m, n, k, a.data(), b.data(), simdAdd.data());
        }

        EXPECT_EQ(std::memcmp(scalarAdd.data(), simdAdd.data(), m * n * sizeof(double)), 0)
            << "MicroGemmAddSimd is not byte-identical to MicroGemmAdd" << label;

        // Assigning form.
        std::vector<double> scalarMul(m * n, 0.0);
        std::vector<double> simdMul(m * n, 0.0);
        MicroGemm(m, n, k, a.data(), b.data(), scalarMul.data());
        MicroGemmSimd(m, n, k, a.data(), b.data(), simdMul.data());

        EXPECT_EQ(std::memcmp(scalarMul.data(), simdMul.data(), m * n * sizeof(double)), 0)
            << "MicroGemmSimd is not byte-identical to MicroGemm" << label;
    }
}

// EXCLUDED ON A NON-x86-64 TARGET. There is no vector tier there for the
// scalar copy to differ from (MicroGemmAddSimd/MicroGemmSimd are the
// portable kernels themselves), so this test's assertion that the difference
// is real would fail by construction on the one path it is meant to measure.
#if QcxArchX86_64
// The one case random operands cannot reach: a product of exactly -0.0, and
// the tier's one documented byte difference. The scalar assigning kernel
// starts each cell from `T value = 0` and ADDS the first term, so -0.0 lands
// as +0.0 there; the tier's l = 0 store writes the product, so it lands as
// -0.0. The zero cannot be added back - MSVC folds
// `_mm256_add_pd(x, _mm256_setzero_pd())` into `x` (measured: the scalar
// `0.0 + x` is preserved, the vector form is not - see
// tools/bench/probes/zeroprobe.cpp, the tracked copy of the measurement).
// So this test pins what is TRUE instead of what would
// be nice: the two agree as VALUES everywhere, every differing byte is a cell
// where both are zero, and the accumulating form - which has no such store -
// is byte-identical even here.
TEST(SimdMicroGemmTest, SimdDiffersFromScalarOnlyInTheSignOfZero) {
    if (!HasAvx2Fma())
    {
        GTEST_SKIP() << "no AVX2+FMA on this machine";
    }

    const std::size_t shapes[][3] = {{6, 1, 1}, {6, 1, 3}, {5, 2, 2}, {7, 3, 2}, {4, 4, 2}};
    std::size_t signedZeroCells = 0;

    for (const auto& shape : shapes)
    {
        const std::size_t m = shape[0];
        const std::size_t n = shape[1];
        const std::size_t k = shape[2];
        std::vector<double> a(m * k);
        std::vector<double> b(k * n);

        // Exact zeros on both sides, with a negative factor opposite them, so
        // the products that are -0.0 are present by construction.
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a[i] = (i % 3 == 0) ? 0.0 : ((i % 3 == 1) ? -1.5 : 2.25);
        }

        for (std::size_t i = 0; i < b.size(); ++i)
        {
            b[i] = (i % 2 == 0) ? -1.0 : 0.5;
        }

        std::vector<double> scalarAdd(m * n, 0.0);
        std::vector<double> simdAdd(m * n, 0.0);

        for (int call = 0; call < 4; ++call)
        {
            MicroGemmAdd(m, n, k, a.data(), b.data(), scalarAdd.data());
            MicroGemmAddSimd(m, n, k, a.data(), b.data(), simdAdd.data());
        }

        const std::string label =
            " shape " + std::to_string(m) + "x" + std::to_string(n) + "x" + std::to_string(k);

        EXPECT_EQ(std::memcmp(scalarAdd.data(), simdAdd.data(), m * n * sizeof(double)), 0)
            << "accumulating form differs on signed-zero operands" << label;

        std::vector<double> scalarMul(m * n, 0.0);
        std::vector<double> simdMul(m * n, 0.0);
        MicroGemm(m, n, k, a.data(), b.data(), scalarMul.data());
        MicroGemmSimd(m, n, k, a.data(), b.data(), simdMul.data());

        for (std::size_t i = 0; i < m * n; ++i)
        {
            // The test asserts the SIMD path is bit-identical to the scalar one.
            // NOLINTNEXTLINE(bugprone-suspicious-memory-comparison)
            const bool sameBytes = std::memcmp(&scalarMul[i], &simdMul[i], sizeof(double)) == 0;

            if (sameBytes)
            {
                continue;
            }

            // A differing byte is only ever the SIGN OF A ZERO: both cells are
            // zero and the values still compare equal. Anything else - a
            // different magnitude, a different non-zero value - fails here,
            // and this is the assertion that keeps the documented residual
            // from growing into a real one.
            EXPECT_TRUE(scalarMul[i] == 0.0 && simdMul[i] == 0.0)
                << "assigning form differs by more than a zero's sign" << label;

            EXPECT_EQ(scalarMul[i], simdMul[i]) << "signed zeros compare unequal" << label;
            ++signedZeroCells;
        }
    }

    // The difference is real, is what the source documents, and never grew
    // beyond the cells that can produce it (the single-column assigning form
    // with a zero operand).
    EXPECT_GT(signedZeroCells, 0u)
        << "the signed-zero operands produced no -0.0 product - the case this test exists for "
           "is not being exercised";
}
#endif // QcxArchX86_64

//
// EXCLUDED ON A NON-x86-64 TARGET, by the same architecture test as the CPUID
// include above: there is no CPUID to read the two leaves from, and the gate
// it pins returns false there without ever reaching a feature bit.
#if QcxArchX86_64
// The gate's regression pin. The FMA feature is CPUID.01H:ECX bit 12; the bit
// this test exists for is CPUID.(EAX=7,ECX=0):EBX bit 12, RDT-M/PQM, which
// the gate used to read. On any machine where the two bits differ - and they
// differ on every CPU without the Xeon monitoring feature - the old gate
// answered "no FMA" and the tier above silently never ran.
TEST(SimdMicroGemmTest, GateReadsTheArchitecturalFmaBit) {
    int leaf1[4] = {};
    int leaf7[4] = {};

#ifdef _MSC_VER
    __cpuid(leaf1, 1);
    __cpuidex(leaf7, 7, 0);
#else
    unsigned int eax = 0;
    unsigned int ebx = 0;
    unsigned int ecx = 0;
    unsigned int edx = 0;
    __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    leaf1[2] = static_cast<int>(ecx);
    __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx);
    leaf7[1] = static_cast<int>(ebx);
#endif

    const bool architecturalFma = (leaf1[2] & (1u << 12)) != 0;
    const bool rdtMonitoringBit = (leaf7[1] & (1u << 12)) != 0;

    // The two bits are distinct features on a machine that has one and not
    // the other: this is the shape that hid the defect, so a machine that
    // cannot exhibit it is stated rather than silently passed.
    if (!architecturalFma || rdtMonitoringBit)
    {
        GTEST_SKIP() << "this machine does not separate the FMA bit (01H:ECX.12) from "
                        "the RDT-M/PQM bit (07H:EBX.12) - the wrong-leaf read cannot be "
                        "observed here";
    }

    // The CPU reports AVX2 and FMA; a false AVX2 verdict can still be correct
    // when the OS has not enabled the vector state, so that case is stated and
    // skipped rather than failed. It is the FMA half that this test pins.
    if (!CpuHasAvx2())
    {
        GTEST_SKIP() << "the CPU has the bits but the OS has not enabled the AVX state";
    }

    EXPECT_TRUE(HasAvx2Fma())
        << "the CPU reports FMA (01H:ECX.12) but the AVX2 tier's gate says no";
    EXPECT_TRUE(CpuHasAvx2Fma());
}

#endif // QcxArchX86_64
} // namespace
