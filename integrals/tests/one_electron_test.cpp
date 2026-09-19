// One-electron integral tests: the committed mpmath reference grid at 1e-10,
// structural sanity (symmetry, diagonal signs), and both rejection paths.
// The prep-footprint test at the end is the large-scale instrument (gated by
// QCX_FOOTPRINT_PREP_MEASURE): the shared pair build's realized pair-data
// residency at the 4,974-function C42H86/def2-QZVP point, measured against
// the driver's own ramp sequence.
#include "alkane_sto3g.hpp"
#include "fast_test_mode.hpp"
#include "h2_sto3g.hpp"
#include "h2_sto3g_fixture.hpp"
#include "h2o_sto3g.hpp"
#include "hf_sto3g.hpp"
#include "internal/footprint.hpp"
#include "internal/md_one_electron.hpp"
#include "qcx/basisset/basis_set.hpp"
#include "qcx/integrals/boys.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::integrals::test::LoadReferenceValues;
using qcx::integrals::test::ReferenceValue;
using qcx::testing::kPOrbitalBasis;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::MakeH2Sto3g;
using qcx::testing::MakeHeAtom;
using qcx::testing::MakeHfSto3g;
using qcx::testing::MakeHfSto3gBasis;
using qcx::testing::MakeSto3gBasis;

constexpr double kReferenceTolerance = 1e-10;
constexpr double kSymmetryTolerance = 1e-14;

// Runs the one-electron builder selected by kind ("S"/"T"/"V").
qcx::Result<CpuTensor2> Build(const qcx::molecule::Molecule& molecule,
                              const qcx::basisset::BasisSet& basisSet,
                              const std::string& kind) {
    if (kind == "S")
    {
        return qcx::integrals::BuildOverlapMatrix(molecule, basisSet);
    }

    if (kind == "T")
    {
        return qcx::integrals::BuildKineticMatrix(molecule, basisSet);
    }

    return qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);
}

TEST(OneElectronTest, MaskedNuclearAttractionRejectsDegenerateInputs) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // An empty center list would silently mean a zero matrix.
    const std::vector<std::size_t> empty;
    auto noCenters = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis, empty);
    ASSERT_FALSE(noCenters.has_value());
    EXPECT_EQ(noCenters.error().code, qcx::ErrorCode::kInvalidArgument);

    // Index 3 does not exist in the three-atom water molecule.
    const std::array<std::size_t, 1> outOfRange{3};
    auto badIndex = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis, outOfRange);
    ASSERT_FALSE(badIndex.has_value());
    EXPECT_EQ(badIndex.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(OneElectronTest, MaskedNuclearAttractionPartitionsTheCenters) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    // A partition of the three centers: the two subsets' matrices sum to
    // the full-basis matrix element-wise - the masking semantics pinned.
    const std::array<std::size_t, 2> oxygenAndHydrogen{0, 1};
    const std::array<std::size_t, 1> hydrogen{2};
    auto first = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis, oxygenAndHydrogen);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto second = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis, hydrogen);
    ASSERT_TRUE(second.has_value()) << second.error().message;
    auto full = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(full.has_value()) << full.error().message;

    const std::size_t n = full->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            EXPECT_NEAR((*first)(i, j) + (*second)(i, j), (*full)(i, j), 1e-12)
                << "(" << i << "," << j << ")";
        }
    }
}

TEST(OneElectronTest, MatchesMpmathReference) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    const std::vector<ReferenceValue> reference = LoadReferenceValues();
    ASSERT_GT(reference.size(), 0u);

    for (const char* kind : {"S", "T", "V"})
    {
        auto matrix = Build(*molecule, *basis, kind);
        ASSERT_TRUE(matrix.has_value()) << matrix.error().message;
        int checked = 0;

        for (const ReferenceValue& row : reference)
        {
            if (row.kind != kind)
            {
                continue;
            }

            EXPECT_NEAR((*matrix)(row.i, row.j), row.value, kReferenceTolerance)
                << kind << "(" << row.i << "," << row.j << ")";
            ++checked;
        }

        EXPECT_EQ(checked, 4) << kind;
    }
}

TEST(OneElectronTest, SymmetricWithSaneDiagonals) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR((*overlap)(i, j), (*overlap)(j, i), kSymmetryTolerance) << "S";
            EXPECT_NEAR((*kinetic)(i, j), (*kinetic)(j, i), kSymmetryTolerance) << "T";
            EXPECT_NEAR((*nuclear)(i, j), (*nuclear)(j, i), kSymmetryTolerance) << "V";
        }
    }

    // Normalized s functions: the overlap diagonal is 1 by construction
    // (the unit-norm renormalization is exact to ~1e-12).
    EXPECT_NEAR((*overlap)(0, 0), 1.0, 1e-10);
    EXPECT_NEAR((*overlap)(1, 1), 1.0, 1e-10);
    EXPECT_GT((*kinetic)(0, 0), 0.0);
    EXPECT_LT((*nuclear)(0, 0), 0.0);
}

TEST(OneElectronTest, PShellIsEvaluatedPositively) {
    // The s-only rejection is gone: the general-l MD 1e builders
    // evaluate p shells (a deliberate behavior change; the 2e s-only
    // BuildEriTensor still rejects them - see two_electron_test.cpp).
    auto basis = qcx::basisset::ParseNwchemText(kPOrbitalBasis);
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);
    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);
    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    // Six p functions (two atoms x px/py/pz); the overlap diagonal is 1 by
    // normalization, the matrices symmetric, the physical signs intact.
    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_NEAR((*overlap)(i, i), 1.0, 1e-10) << "S diagonal " << i;
        EXPECT_GT((*kinetic)(i, i), 0.0) << "T diagonal " << i;
        EXPECT_LT((*nuclear)(i, i), 0.0) << "V diagonal " << i;

        for (std::size_t j = 0; j < 6; ++j)
        {
            EXPECT_NEAR((*overlap)(i, j), (*overlap)(j, i), kSymmetryTolerance);
            EXPECT_NEAR((*kinetic)(i, j), (*kinetic)(j, i), kSymmetryTolerance);
            EXPECT_NEAR((*nuclear)(i, j), (*nuclear)(j, i), kSymmetryTolerance);
        }
    }

    // The p-s overlap between the atoms is finite and positive (same sign
    // convention as the s-s at this geometry).
    EXPECT_GT((*overlap)(0, 3), 0.0);
}

TEST(OneElectronTest, MissingElementIsInvalidArgument) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHeAtom();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    for (const char* kind : {"S", "T", "V"})
    {
        auto matrix = Build(*molecule, *basis, kind);
        ASSERT_FALSE(matrix.has_value()) << kind;
        EXPECT_EQ(matrix.error().code, qcx::ErrorCode::kInvalidArgument) << kind;
    }

    // The moment builders share BuildPairMatrix, so the missing-element
    // rejection applies to them too.
    auto dipole = qcx::integrals::BuildDipoleMatrix(*molecule, *basis);
    ASSERT_FALSE(dipole.has_value());
    EXPECT_EQ(dipole.error().code, qcx::ErrorCode::kInvalidArgument);
    auto quadrupole = qcx::integrals::BuildQuadrupoleMatrix(*molecule, *basis);
    ASSERT_FALSE(quadrupole.has_value());
    EXPECT_EQ(quadrupole.error().code, qcx::ErrorCode::kInvalidArgument);
}

TEST(OneElectronTest, DipoleAndQuadrupoleH2Sto3g) {
    // The dipole (r) and quadrupole (rr) integral matrices of H2/STO-3G at
    // R = 1.4 bohr, pinned against pyscf 2.14.0 (int1e_r comp=3, int1e_rr
    // comp=9 - the comp=6 layout DROPS zz - WSL 2026-08-25), origin at the
    // coordinate origin. The pyscf values are the absolute <u|r_k|v> /
    // <u|r_k r_l|v> integrals; with the default origin R = 0 the qcx
    // operators r_k - R_k coincide.
    //
    // Hand checks (stated provenance: the s-function moment formulas of
    // Helgaker Sec 9.5.1):
    //   - r_x(1,1) = 1.4: a normalized s function's r expectation is its
    //     center (the atom-1 coordinate).
    //   - r_x(0,1) = P_x S(0,1) = 0.7 * 0.6593182...: the pair midpoint
    //     (identical exponents on both H atoms) times the overlap.
    //   - rr_yy(0,0) = rr_zz(0,0) = rr_xx(0,0): the s function at the
    //     origin is per-axis symmetric.
    //   - the xy/xz/yz components and r_y/r_z vanish: the pair lies on
    //     the x-axis, so every P_y = P_z = 0.
    constexpr double kTolerance = 1e-8;
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto dipole = qcx::integrals::BuildDipoleMatrix(*molecule, *basis);
    ASSERT_TRUE(dipole.has_value()) << dipole.error().message;
    auto quadrupole = qcx::integrals::BuildQuadrupoleMatrix(*molecule, *basis);
    ASSERT_TRUE(quadrupole.has_value()) << quadrupole.error().message;

    const auto& rx = (*dipole)[0];
    const auto& ry = (*dipole)[1];
    const auto& rz = (*dipole)[2];
    EXPECT_NEAR(rx(0, 0), 0.0, kTolerance);
    EXPECT_NEAR(rx(0, 1), 0.4615227443, kTolerance);
    EXPECT_NEAR(rx(1, 1), 1.4, kTolerance);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(ry(i, j), 0.0, kTolerance) << "ry(" << i << "," << j << ")";
            EXPECT_NEAR(rz(i, j), 0.0, kTolerance) << "rz(" << i << "," << j << ")";
        }
    }

    const auto& qxx = (*quadrupole)[0];
    const auto& qxy = (*quadrupole)[1];
    const auto& qxz = (*quadrupole)[2];
    const auto& qyy = (*quadrupole)[3];
    const auto& qyz = (*quadrupole)[4];
    const auto& qzz = (*quadrupole)[5];
    EXPECT_NEAR(qxx(0, 0), 0.6495242361, kTolerance);
    EXPECT_NEAR(qxx(0, 1), 0.8766672260, kTolerance);
    EXPECT_NEAR(qxx(1, 1), 2.6095242361, kTolerance);
    EXPECT_NEAR(qyy(0, 0), 0.6495242361, kTolerance);
    EXPECT_NEAR(qyy(0, 1), 0.4901726804, kTolerance);
    EXPECT_NEAR(qyy(1, 1), 0.6495242361, kTolerance);
    EXPECT_NEAR(qzz(0, 0), 0.6495242361, kTolerance);
    EXPECT_NEAR(qzz(0, 1), 0.4901726804, kTolerance);
    EXPECT_NEAR(qzz(1, 1), 0.6495242361, kTolerance);

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < 2; ++j)
        {
            EXPECT_NEAR(qxy(i, j), 0.0, kTolerance) << "qxy(" << i << "," << j << ")";
            EXPECT_NEAR(qxz(i, j), 0.0, kTolerance) << "qxz(" << i << "," << j << ")";
            EXPECT_NEAR(qyz(i, j), 0.0, kTolerance) << "qyz(" << i << "," << j << ")";
        }
    }

    // Every moment matrix is symmetric (the shared builder mirrors the
    // canonical pair blocks transposed).
    for (std::size_t component = 0; component < 6; ++component)
    {
        for (std::size_t i = 0; i < 2; ++i)
        {
            for (std::size_t j = 0; j < 2; ++j)
            {
                EXPECT_NEAR((*quadrupole)[component](i, j),
                            (*quadrupole)[component](j, i),
                            kSymmetryTolerance);
            }
        }
    }
}

TEST(OneElectronTest, DipoleAndQuadrupoleOriginShift) {
    // The moment operators shift linearly with the reference point: for
    // R' = R + Delta,
    //   D_k(R') = D_k(R) - Delta_k S
    //   Q_kl(R') = Q_kl(R) - Delta_k D_l(R) - Delta_l D_k(R)
    //              + Delta_k Delta_l S
    // (algebraic identities of the integrals themselves), so the origin
    // parameter is cross-checked against the unshifted builders exactly.
    constexpr double kTolerance = 1e-12;
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);
    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;
    auto dipole = qcx::integrals::BuildDipoleMatrix(*molecule, *basis);
    ASSERT_TRUE(dipole.has_value()) << dipole.error().message;
    auto quadrupole = qcx::integrals::BuildQuadrupoleMatrix(*molecule, *basis);
    ASSERT_TRUE(quadrupole.has_value()) << quadrupole.error().message;

    const std::array<double, 3> shifted = {0.3, 0.2, -0.1};
    auto dipoleShifted = qcx::integrals::BuildDipoleMatrix(*molecule, *basis, shifted);
    ASSERT_TRUE(dipoleShifted.has_value()) << dipoleShifted.error().message;
    auto quadrupoleShifted = qcx::integrals::BuildQuadrupoleMatrix(*molecule, *basis, shifted);
    ASSERT_TRUE(quadrupoleShifted.has_value()) << quadrupoleShifted.error().message;

    const std::array<double, 3> delta = {0.3, 0.2, -0.1};

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        for (std::size_t i = 0; i < 2; ++i)
        {
            for (std::size_t j = 0; j < 2; ++j)
            {
                EXPECT_NEAR((*dipoleShifted)[axis](i, j),
                            (*dipole)[axis](i, j) - delta[axis] * (*overlap)(i, j),
                            kTolerance)
                    << "D_" << axis << "(" << i << "," << j << ")";
            }
        }
    }

    // The six components in the xx, xy, xz, yy, yz, zz order.
    const std::array<std::pair<int, int>, 6> components = {
        {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 2}}};

    for (std::size_t component = 0; component < components.size(); ++component)
    {
        const auto [k, l] = components[component];

        for (std::size_t i = 0; i < 2; ++i)
        {
            for (std::size_t j = 0; j < 2; ++j)
            {
                const double expected = (*quadrupole)[component](i, j) -
                                        delta[static_cast<std::size_t>(k)] *
                                            (*dipole)[static_cast<std::size_t>(l)](i, j) -
                                        delta[static_cast<std::size_t>(l)] *
                                            (*dipole)[static_cast<std::size_t>(k)](i, j) +
                                        delta[static_cast<std::size_t>(k)] *
                                            delta[static_cast<std::size_t>(l)] * (*overlap)(i, j);
                EXPECT_NEAR((*quadrupoleShifted)[component](i, j), expected, kTolerance)
                    << "Q_" << k << l << "(" << i << "," << j << ")";
            }
        }
    }
}

TEST(OneElectronTest, DipoleAndQuadrupoleHFSto3gPShell) {
    // Spot checks of the p-shell moment integrals (the E_{e_k} / E_{2e_k}
    // Hermite terms - the H2 s-only pins above cannot exercise them),
    // HF/STO-3G with F at the origin and H at (1.732500911, 0, 0), pinned
    // against pyscf 2.14.0 (WSL 2026-08-25).
    //
    // AO order: H 1s, F 1s, F 2s, F 2py, F 2pz, F 2px (indices 0..5). H
    // comes first because the (Z, x, y, z) atom renumbering sorts the
    // atoms by atomic number (shell_pairs.cpp); the p components follow the
    // kSolidHarmonicG row order (m ascending -l..+l: y, z, x). The mapping
    // to the pyscf order (F 1s, 2s, 2px, 2py, 2pz, H 1s) is
    // qcx -> pyscf: 0->5, 1->0, 2->1, 3->3, 4->4, 5->2.
    //
    // Hand checks (stated provenance: Helgaker Sec 9.5.1):
    //   - rx(0,0) = 1.732500911: a normalized s function's r expectation
    //     is its center (the H coordinate).
    //   - rx(5,5) = 0: <2px|x|2px> vanishes by parity (odd integrand).
    //   - rx(2,5): the same-atom <2s|x|2px> - a genuinely p-shell dipole
    //     (the E_{e_x} Hermite term with a first-order fold).
    //   - ry(3,0): the off-axis <2py|y|H 1s> (nonzero only for p, so it
    //     also proves the atom-0 shell really carries the p component).
    //   - qxx(5,5) = 0.6938969145 vs qyy(4,4) = 0.2312989715: <2px|x^2|2px>
    //     is 3x the perpendicular expectation - the E_{2e_x} term, and the
    //     index split proves the py/pz/px assignment (qyy(3,3) would equal
    //     this were 3 and 5 swapped).
    constexpr double kTolerance = 1e-8;
    auto basis = MakeHfSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeHfSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto dipole = qcx::integrals::BuildDipoleMatrix(*molecule, *basis);
    ASSERT_TRUE(dipole.has_value()) << dipole.error().message;
    auto quadrupole = qcx::integrals::BuildQuadrupoleMatrix(*molecule, *basis);
    ASSERT_TRUE(quadrupole.has_value()) << quadrupole.error().message;

    const auto& rx = (*dipole)[0];
    const auto& ry = (*dipole)[1];
    EXPECT_NEAR(rx(0, 0), 1.732500911, kTolerance);
    EXPECT_NEAR(rx(5, 5), 0.0, kTolerance);
    EXPECT_NEAR(rx(2, 5), 0.5657407566, kTolerance);
    EXPECT_NEAR(ry(3, 0), 0.2710146892, kTolerance);

    const auto& qxx = (*quadrupole)[0];
    const auto& qyy = (*quadrupole)[3];
    const auto& qyz = (*quadrupole)[4];
    EXPECT_NEAR(qxx(0, 0), 3.6510836428, kTolerance);
    EXPECT_NEAR(qxx(5, 5), 0.6938969145, kTolerance);
    EXPECT_NEAR(qxx(3, 3), 0.2312989715, kTolerance);
    EXPECT_NEAR(qyy(3, 3), 0.6938969145, kTolerance);
    EXPECT_NEAR(qyy(4, 4), 0.2312989715, kTolerance);
    EXPECT_NEAR(qyz(3, 4), 0.2312989715, kTolerance);
}

namespace {

constexpr double kPi = 3.14159265358979323846;

// The primitive normalization of the qcx convention
// ((2a/pi)^(3/4), ao_evaluator.cpp RadialNormalization for l = 0).
double PrimitiveNorm(double exponent) {
    return std::pow(2.0 * exponent / kPi, 0.75);
}

// The two-center s-Gaussian Coulomb potential of a unit point charge at p
// (Helgaker Sec 9.3): with P = (a A + b B)/(a + b),
//     <g_a(A)|1/|r - p||g_b(B)> = N_a N_b 2pi/(a + b) *
//         exp(-ab|A - B|^2/(a + b)) F0((a + b)|P - p|^2).
// The parser renormalizes spherical contractions to unit self-overlap
// under S(a, b) = (4ab)^(3/4)/(a + b)^(3/2) = N_a N_b (pi/(a + b))^(3/2),
// so the test renormalizes the stored coefficients the same way first.
double GaussianPotentialAt(const std::array<double, 3>& centerA,
                           const std::array<double, 3>& centerB,
                           const std::array<double, 3>& point,
                           double exponentA,
                           double exponentB) {
    const double alpha = exponentA + exponentB;
    const double abOverAlpha = exponentA * exponentB / alpha;
    double shift2 = 0.0;

    for (int k = 0; k < 3; ++k)
    {
        const double delta = centerA[k] - centerB[k];
        shift2 += delta * delta;
    }

    double point2 = 0.0;

    for (int k = 0; k < 3; ++k)
    {
        const double pK = (exponentA * centerA[k] + exponentB * centerB[k]) / alpha - point[k];
        point2 += pK * pK;
    }

    return PrimitiveNorm(exponentA) * PrimitiveNorm(exponentB) * (2.0 * kPi / alpha) *
           std::exp(-abOverAlpha * shift2) * qcx::integrals::BoysSingle(0, alpha * point2);
}

// The contracted matrix element <phi|1/|r - p||phi'> for the (renormalized)
// s contraction coefficients, centers A and B.
double ContractionPotentialAt(const std::vector<double>& coefficients,
                              const std::vector<double>& exponents,
                              const std::array<double, 3>& centerA,
                              const std::array<double, 3>& centerB,
                              const std::array<double, 3>& point) {
    double value = 0.0;

    for (std::size_t i = 0; i < coefficients.size(); ++i)
    {
        for (std::size_t j = 0; j < coefficients.size(); ++j)
        {
            value += coefficients[i] * coefficients[j] *
                     GaussianPotentialAt(centerA, centerB, point, exponents[i], exponents[j]);
        }
    }

    return value;
}

} // namespace

TEST(OneElectronTest, ElectronPotentialMatchesClosedForm) {
    // The electron potential V_elec(p) = -sum_uv D_uv <u|1/|r - p||v> pinned
    // against the closed-form s-Gaussian Coulomb integral (Helgaker Sec
    // 9.3). H2/STO-3G with the off-diagonal density [[1, 1/2], [1/2, 1]]
    // exercises both the diagonal-pair contraction (plain D) and the
    // off-diagonal (D_uv + D_vu) mirrored path. The atoms sit at x = +-0.7
    // bohr; the sample points are off the bond axis and at unequal distances
    // from the atoms, so no symmetry cancels an error.
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const qcx::basisset::ElementBasis* element = basis->Find(1);
    ASSERT_NE(element, nullptr);
    ASSERT_EQ(element->shells.size(), 1u);
    const qcx::basisset::Shell& shell = element->shells[0];
    ASSERT_EQ(shell.angularMomentum, 0);
    ASSERT_TRUE(shell.isSpherical);
    ASSERT_EQ(shell.coefficients.size(), 1u);
    const std::vector<double>& exponents = shell.exponents;
    const std::vector<double>& raw = shell.coefficients[0];
    ASSERT_EQ(exponents.size(), 3u);

    // Renormalize like the parser: unit contraction self-overlap under
    // S(a, b) = N_a N_b (pi/(a + b))^(3/2).
    double norm2 = 0.0;

    for (std::size_t i = 0; i < raw.size(); ++i)
    {
        for (std::size_t j = 0; j < raw.size(); ++j)
        {
            norm2 += raw[i] * raw[j] * PrimitiveNorm(exponents[i]) * PrimitiveNorm(exponents[j]) *
                     std::pow(kPi / (exponents[i] + exponents[j]), 1.5);
        }
    }

    std::vector<double> coefficients = raw;

    for (double& c : coefficients)
    {
        c /= std::sqrt(norm2);
    }

    const auto& coordinates = molecule->CoordinatesBohr();
    const std::array<double, 3> atomA = {coordinates(0, 0), coordinates(0, 1), coordinates(0, 2)};
    const std::array<double, 3> atomB = {coordinates(1, 0), coordinates(1, 1), coordinates(1, 2)};

    // The density [[1, 1/2], [1/2, 1]]; V_elec(p) = -(d_A + d_B + O) with
    // d_A = M_00, d_B = M_11, O = 2 * (1/2) * M_01 = M_01.
    const std::array<std::array<double, 3>, 3> points = {
        {{{3.0, 0.0, 0.0}}, {{-3.0, 0.0, 0.0}}, {{0.0, 0.0, 3.0}}}};
    auto pointTensor = CpuTensor2::Create({3, 3});
    ASSERT_TRUE(pointTensor.has_value()) << pointTensor.error().message;

    for (std::size_t p = 0; p < points.size(); ++p)
    {
        for (std::size_t k = 0; k < 3; ++k)
        {
            (*pointTensor)(p, k) = points[p][k];
        }
    }

    pointTensor->MarkHostDirty();
    Eigen::MatrixXd density = Eigen::MatrixXd::Identity(2, 2);
    density(0, 1) = 0.5;
    density(1, 0) = 0.5;
    auto densityTensor = qcx::testing::ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto values = qcx::integrals::BuildElectronPotentialAtPoints(
        *molecule, *basis, *densityTensor, *pointTensor);
    ASSERT_TRUE(values.has_value()) << values.error().message;
    ASSERT_EQ(values->size(), 3u);

    for (std::size_t p = 0; p < points.size(); ++p)
    {
        const double diagonal =
            ContractionPotentialAt(coefficients, exponents, atomA, atomA, points[p]) +
            ContractionPotentialAt(coefficients, exponents, atomB, atomB, points[p]);
        const double offDiagonal =
            ContractionPotentialAt(coefficients, exponents, atomA, atomB, points[p]);
        EXPECT_NEAR((*values)[p], -(diagonal + offDiagonal), 1e-10) << "point " << p;
    }
}

TEST(OneElectronTest, ElectronPotentialRejectsBadShapes) {
    auto basis = MakeSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2Sto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    Eigen::MatrixXd density = Eigen::MatrixXd::Identity(2, 2);
    auto densityTensor = qcx::testing::ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    auto points = CpuTensor2::Create({2, 2});
    ASSERT_TRUE(points.has_value()) << points.error().message;
    points->MarkHostDirty();
    auto wrongPoints =
        qcx::integrals::BuildElectronPotentialAtPoints(*molecule, *basis, *densityTensor, *points);
    ASSERT_FALSE(wrongPoints.has_value());
    EXPECT_EQ(wrongPoints.error().code, qcx::ErrorCode::kInvalidArgument);

    auto badDensity = CpuTensor2::Create({3, 3});
    ASSERT_TRUE(badDensity.has_value()) << badDensity.error().message;
    badDensity->MarkHostDirty();
    auto goodPoints = CpuTensor2::Create({1, 3});
    ASSERT_TRUE(goodPoints.has_value()) << goodPoints.error().message;
    goodPoints->MarkHostDirty();
    auto wrongDensity =
        qcx::integrals::BuildElectronPotentialAtPoints(*molecule, *basis, *badDensity, *goodPoints);
    ASSERT_FALSE(wrongDensity.has_value());
    EXPECT_EQ(wrongDensity.error().code, qcx::ErrorCode::kInvalidArgument);

    // The missing-basis-entry path shared with the matrix builders.
    auto helium = MakeHeAtom();
    ASSERT_TRUE(helium.has_value()) << helium.error().message;
    auto missing = qcx::integrals::BuildElectronPotentialAtPoints(
        *helium, *basis, *densityTensor, *goodPoints);
    ASSERT_FALSE(missing.has_value());
    EXPECT_EQ(missing.error().code, qcx::ErrorCode::kInvalidArgument);
}

// The wide-point regression (Release-only): the point-potential build's
// retained working set is Theta(Npoints), never Theta(Npairs x Npoints) - the
// former pair-parallel layout kept one row of Npoints doubles per pair (78
// pairs x 20K points x 8 B = 12.5 MB on this fixture; ~108 GiB at the c60
// CHELPG scale). The prefix contract pins the restructure bit-exactly: every
// point's value is a serial reduction over the same pair order, so the wide
// run's first points must equal the small run's values exactly.
TEST(OneElectronTest, ElectronPotentialWidePointSetMatchesPrefix) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "Release-only: the wide-point regression run";
    }

    const std::filesystem::path root(QcxBasisDataDir);
    auto basis = qcx::basisset::ParseNwchemDirectory((root / "def2-svp").string());
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    ASSERT_EQ(pairList->pairs.size(), 78u);
    const std::size_t n = pairList->functionCount;

    Eigen::MatrixXd density =
        Eigen::MatrixXd::Identity(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));
    auto densityTensor = qcx::testing::ToTensor(density);
    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    constexpr std::size_t kProbePoints = 128;
    constexpr std::size_t kWidePoints = 20000;

    const auto buildPoints = [](std::size_t count) {
        auto points = CpuTensor2::Create({count, 3});
        EXPECT_TRUE(points.has_value()) << points.error().message;

        for (std::size_t p = 0; p < count; ++p)
        {
            (*points)(p, 0) = 0.5 + 0.001 * static_cast<double>(p % 100);
            (*points)(p, 1) = -0.25 + 0.0017 * static_cast<double>(p % 61);
            (*points)(p, 2) = 0.75 - 0.0009 * static_cast<double>(p % 43);
        }

        points->MarkHostDirty();
        return std::move(*points);
    };

    auto smallPoints = buildPoints(kProbePoints);
    auto widePoints = buildPoints(kWidePoints);

    auto small = qcx::integrals::BuildElectronPotentialAtPoints(
        *molecule, *basis, *densityTensor, smallPoints);
    ASSERT_TRUE(small.has_value()) << small.error().message;
    ASSERT_EQ(small->size(), kProbePoints);

    auto wide = qcx::integrals::BuildElectronPotentialAtPoints(
        *molecule, *basis, *densityTensor, widePoints);
    ASSERT_TRUE(wide.has_value()) << wide.error().message;
    ASSERT_EQ(wide->size(), kWidePoints);

    for (std::size_t p = 0; p < kProbePoints; ++p)
    {
        EXPECT_EQ((*wide)[p], (*small)[p]) << "point " << p;
    }
}

TEST(OneElectronTest, TheChunkedPairBuildIsBitIdenticalToTheWholeStore) {
    // The shared pair build CHUNKS its pair data (one_electron.cpp
    // BuildPairMatrix): the geometry-only skeleton plus one kPairChunkBytes
    // chunk at a time, each chunk released before the next is built. On every
    // small fixture the whole store fits ONE chunk, so the multi-chunk path -
    // the chunk loop's per-chunk state, the release between chunks - is
    // unreachable there. C12H26/def2-QZVP is the smallest fixture whose store
    // exceeds the 512 MiB cap (0.86 GiB, two chunks), so it is the one that
    // exercises the loop as a loop.
    //
    // The reference is the pre-chunking build: the WHOLE store through
    // internal::BuildPairData (still the builders that genuinely need it) with
    // the same per-pair write into the same matrix cells. Exact equality, not
    // a tolerance: the chunked build is the same pair data in the same order
    // through the same builder function, so a chunk boundary must not move a
    // bit - the property ScreeningTest.ChunkedSweepIsBitIdenticalToTheSinglePass
    // pins for the sweep's own chunking.
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only (a 0.86 GiB pair store, twice)";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(12);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const std::array<int, 2> elements{6, 1};
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-qzvp", elements);

    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t storeBytes =
        qcx::integrals::internal::PairStoreBytes(*molecule, *basis, *pairList);

    // The fixture's own licence for this test: more than one chunk. A store
    // that fits one chunk silently degrades the test to the small fixtures'
    // coverage, which is exactly what it exists to add.
    ASSERT_GT(storeBytes, qcx::integrals::internal::kPairChunkBytes);

    auto chunked = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    ASSERT_TRUE(chunked.has_value()) << chunked.error().message;

    // The whole-store reference (the pre-chunking build).
    auto store = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);

    ASSERT_TRUE(store.has_value()) << store.error().message;
    ASSERT_EQ(store->size(), nPairs);

    const std::size_t n = pairList->functionCount;
    auto reference = CpuTensor2::Create({n, n});

    ASSERT_TRUE(reference.has_value()) << reference.error().message;

    for (std::size_t pairIdx = 0; pairIdx < nPairs; ++pairIdx)
    {
        const qcx::integrals::ShellPairIndex& pairIndex = pairList->pairs[pairIdx];
        const qcx::integrals::internal::MdPairData& pair = (*store)[pairIdx];
        const std::size_t oI = pairList->shells[pairIndex.i].functionOffset;
        const std::size_t oJ = pairList->shells[pairIndex.j].functionOffset;
        std::vector<double> block(pair.nFuncs);
        qcx::integrals::internal::BuildOverlapPair(pair, block.data());

        for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
        {
            for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
            {
                const double value = block[fa * pair.nFuncsB + fb];
                (*reference)(oI + fa, oJ + fb) = value;
                (*reference)(oJ + fb, oI + fa) = value;
            }
        }
    }

    ASSERT_EQ(chunked->Shape()[0], n);
    std::size_t moved = 0;

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            if ((*chunked)(i, j) != (*reference)(i, j))
            {
                ++moved;
                EXPECT_EQ((*chunked)(i, j), (*reference)(i, j)) << "(" << i << "," << j << ")";
            }
        }
    }

    // One line of evidence the reader can check without the diff: how many
    // elements the two agree on, exactly.
    std::printf("\n  chunked vs whole-store: %zu elements, %zu differ (store %.3f GiB, cap %.3f "
                "GiB)\n",
                n * n,
                moved,
                static_cast<double>(storeBytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(qcx::integrals::internal::kPairChunkBytes) /
                    (1024.0 * 1024.0 * 1024.0));
    std::fflush(stdout);
    EXPECT_EQ(moved, 0u);
}

TEST(OneElectronTest, SetupRampPeakBytesIsTheNeverUnderComposition) {
    // The setup admission's bound - the guard that survives the predictive
    // memory model's deletion, re-derived from the ENGINE's quantities: the
    // geometry-only skeleton, ONE pair-data chunk and the caller's live
    // matrices (the ramp, whose composition the lean member's envelope
    // charges as its runStoreBytes term), PLUS one SchwarzSweepBytes per
    // basis in effect (the screening pre-pass every budgeted builder runs in
    // front of its own budget decision). Pinned against the engine's own
    // formulas term by term, so a drift in the bound shows up here rather
    // than as a run that dies inside the setup.
    //
    // The sweep term is asserted as PRESENT, not as a larger number: the
    // bound equals ramp + sweep exactly, and the sweep is non-zero, so a
    // bound that dropped the term fails here instead of passing on
    // magnitude. That is the shape the ramp-only bound had.
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const std::size_t nPairs = pairList->pairs.size();
    const std::size_t n = pairList->functionCount;
    const std::size_t skeletonBytes = sizeof(qcx::integrals::internal::MdPairData) * nPairs;
    const std::size_t storePayloadBytes =
        qcx::integrals::internal::PairStoreBytes(*molecule, *basis, *pairList) - skeletonBytes;
    const std::size_t expectedRamp =
        skeletonBytes + std::min(qcx::integrals::internal::kPairChunkBytes, storePayloadBytes) +
        // The only int multiply is the literal 8 * 2; the size_t operands carry the width.
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result)
        8 * 2 * n * n;
    const std::size_t expectedSweep =
        qcx::integrals::internal::SchwarzSweepBytes(*molecule, *basis, *pairList);

    ASSERT_GT(expectedSweep, 0u);

    // No aux in effect: the orbital sweep alone rides the ramp.
    const auto noAux = qcx::integrals::SetupRampPeakBytes(*molecule, *basis, nullptr, 2);
    ASSERT_TRUE(noAux.has_value()) << noAux.error().message;
    EXPECT_EQ(static_cast<std::size_t>(*noAux), expectedRamp + expectedSweep);

    // The aux arm: a basis in effect adds exactly ITS own sweep - the term
    // the aux-based families (ri_j_link, ri_jk) allocate and the one the
    // measured death band needed, since the aux sweep is the larger of the
    // two on the fixture that measured it.
    const auto withAux = qcx::integrals::SetupRampPeakBytes(*molecule, *basis, &*basis, 2);
    ASSERT_TRUE(withAux.has_value()) << withAux.error().message;
    EXPECT_EQ(static_cast<std::size_t>(*withAux) - static_cast<std::size_t>(*noAux), expectedSweep);

    // The matrices are the caller's own concurrency, charged one n x n block
    // each: the difference between three live matrices and two is exactly
    // one matrix, which is what keeps the term never-under rather than
    // merely large.
    const auto three = qcx::integrals::SetupRampPeakBytes(*molecule, *basis, nullptr, 3);
    ASSERT_TRUE(three.has_value()) << three.error().message;
    EXPECT_EQ(static_cast<std::size_t>(*three) - static_cast<std::size_t>(*noAux), 8 * n * n);
}

TEST(OneElectronTest, PrintsThePrepFootprintAtThe4974FunctionTarget) {
    // The prep-footprint acceptance instrument (the 4,974-function target,
    // C42H86/def2-QZVP). The driver's shared setup ramp materializes the
    // overlap, kinetic and nuclear matrices through ONE shared build
    // (BuildPairMatrix) once per matrix, and each call used to build the
    // WHOLE contracted pair store - 9.95 GiB at this size, uncharged by every
    // footprint function and the second-largest term of the lean envelope.
    // The test drives the ramp in the driver's own order - the core
    // Hamiltonian (kinetic, nuclear) then the overlap - and holds all three
    // matrices, exactly as the driver does, so a 1 Hz working-set sampler over
    // this process measures the prep's realized residency rather than a model
    // of it. The Schwarz sweep (the builder Create's pre-pass: measured
    // ~1.46 GiB) runs last, so this instrument's series covers both terms the
    // shared build moved.
    if (std::getenv("QCX_FOOTPRINT_PREP_MEASURE") == nullptr)
    {
        GTEST_SKIP() << "set QCX_FOOTPRINT_PREP_MEASURE=1 to run the 4,974-function prep "
                        "measurement (minutes; ~10 GiB of pair data before the chunking)";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(42);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const std::array<int, 2> elements{6, 1};
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-qzvp", elements);

    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const std::size_t n = pairList->functionCount;
    const std::size_t nPairs = pairList->pairs.size();

    // The fixture's own count, pinned as the ladder instrument pins it: a
    // series measured on any other point is not this target's.
    ASSERT_EQ(n, 4974u);
    const std::size_t fullStoreBytes =
        qcx::integrals::internal::PairStoreBytes(*molecule, *basis, *pairList);
    const double kGiB = 1024.0 * 1024.0 * 1024.0;
    const auto started = std::chrono::steady_clock::now();
    const auto phase = [&started](const char* label) {
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::printf("  PHASE %8.1f s  %s\n", seconds, label);
        std::fflush(stdout);
    };

    std::printf("\n  the 1e prep footprint at the 4,974-function target (C42H86 / def2-QZVP)\n");
    std::printf("  n = %zu | canonical pairs %zu | the whole store %.4f GiB\n",
                n,
                nPairs,
                static_cast<double>(fullStoreBytes) / kGiB);

    // The same envelope the driver's lean leg reports as the run's modeled
    // peak (run_driver.cpp's lean branch): the model block and the sampled
    // series below are read side by side, never one for the other.
    auto envelope = qcx::integrals::EstimateLeanEnvelope(
        *molecule, *basis, qcx::integrals::LeanFockBuildOptions{});

    if (!envelope.has_value())
    {
        std::printf("  the lean envelope failed: %s\n", envelope.error().message.c_str());
    } else
    {
        std::printf("  EstimateLeanEnvelope total %.4f GiB\n", envelope->totalBytes / kGiB);
    }

    std::fflush(stdout);

    phase("ramp: kinetic");
    auto kinetic = qcx::integrals::BuildKineticMatrix(*molecule, *basis);

    ASSERT_TRUE(kinetic.has_value()) << kinetic.error().message;

    phase("ramp: nuclear attraction");
    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(*molecule, *basis);

    ASSERT_TRUE(nuclear.has_value()) << nuclear.error().message;

    phase("ramp: overlap");
    auto overlap = qcx::integrals::BuildOverlapMatrix(*molecule, *basis);

    ASSERT_TRUE(overlap.has_value()) << overlap.error().message;

    // The three matrices stay live to here: the driver holds S, T and V
    // together (BuildCoreHamiltonian and BuildOverlapMatrix), so the ramp's
    // resident set is the prep plus them, not the prep alone.
    EXPECT_EQ(overlap->Shape()[0], n);
    EXPECT_EQ(kinetic->Shape()[0], n);
    EXPECT_EQ(nuclear->Shape()[0], n);

    phase("Schwarz sweep (the builder Create's pre-pass)");
    auto schwarz = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);

    ASSERT_TRUE(schwarz.has_value()) << schwarz.error().message;
    EXPECT_EQ(schwarz->size(), nPairs);

    phase("done: the pair data is released, only S/T/V and the bounds are resident");
    std::printf("  the sweep's bounds: %zu doubles\n", schwarz->size());
    std::fflush(stdout);

    // Nothing above is asserted against a byte count: this test's deliverable
    // is the printed series (the sampler reads the process, the model is
    // printed for the side-by-side), and a byte pin here would be the model
    // dressed as a measurement.
}

// The point potentials' own pair-data chunking (one_electron.cpp
// BuildElectronPotentialAtPoints, the CHELPG/MK ESP path): the chunk loop
// builds one chunk at a time and every point accumulates ACROSS the chunk
// boundaries, so the property under test is the one that loop can break
// silently - each point must add the same terms in the same pair order the
// unchunked walk added, with no per-chunk rounding - and the fixture must
// therefore cross a chunk boundary.
//
// C12H26/def2-QZVP is the smallest alkane fixture whose store exceeds the
// 512 MiB cap (0.86 GiB; the licence is asserted below), so the loop runs as
// a LOOP here where every smaller fixture would leave the boundary untested.
// The density is synthetic and asymmetric: what is compared is the
// arithmetic, not the physics, and a synthetic one keeps the cost in the
// pair walk, which IS the ESP path's per-point cost. The reference is the
// pre-chunking walk, verbatim, over the whole store.
TEST(OneElectronTest, ElectronPotentialAtPointsIsBitIdenticalWithChunkedPairData) {
    if (qcx::testing::IsFastOnlyMode())
    {
        GTEST_SKIP() << "heavy numerical test: Release-only (a 0.86 GiB pair store per pass)";
    }

    auto molecule = qcx::testing::MakeAlkaneSto3g(12);

    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    const std::array<int, 2> elements{6, 1};
    auto basis = qcx::basisset::ParseNwchemDirectoryFiltered(
        std::string(QcxBasisDataDir) + "/def2-qzvp", elements);

    ASSERT_TRUE(basis.has_value()) << basis.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);

    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const std::size_t storeBytes =
        qcx::integrals::internal::PairStoreBytes(*molecule, *basis, *pairList);

    // The fixture's licence for this test: more than one chunk. A store that
    // fits one chunk degrades the run to the small fixtures' coverage, which
    // is exactly what this test exists to add.
    ASSERT_GT(storeBytes, qcx::integrals::internal::kPairChunkBytes);

    const std::size_t n = pairList->functionCount;
    Eigen::MatrixXd density =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            density(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
                0.001 * static_cast<double>((7 * i + 13 * j) % 17) - 0.005;
        }
    }

    auto densityTensor = qcx::testing::ToTensor(density);

    ASSERT_TRUE(densityTensor.has_value()) << densityTensor.error().message;

    // Two points from the molecule's own geometry: the first carbon's
    // position (the nuclear cusp region, the largest magnitudes) and a
    // displaced point.
    const auto& coordinates = molecule->CoordinatesBohr();
    constexpr std::size_t kPointCount = 2;
    auto points = CpuTensor2::Create({kPointCount, 3});

    ASSERT_TRUE(points.has_value()) << points.error().message;
    (*points)(0, 0) = coordinates(0, 0);
    (*points)(0, 1) = coordinates(0, 1);
    (*points)(0, 2) = coordinates(0, 2);
    (*points)(1, 0) = coordinates(0, 0) + 2.5;
    (*points)(1, 1) = coordinates(0, 1) - 1.25;
    (*points)(1, 2) = coordinates(0, 2) + 0.75;
    points->MarkHostDirty();

    auto chunked =
        qcx::integrals::BuildElectronPotentialAtPoints(*molecule, *basis, *densityTensor, *points);

    ASSERT_TRUE(chunked.has_value()) << chunked.error().message;
    ASSERT_EQ(chunked->size(), kPointCount);

    // The reference: the pre-chunking walk, verbatim - the WHOLE store in one
    // BuildPairData, then one serial accumulation per point in pair order.
    auto store = qcx::integrals::internal::BuildPairData(*molecule, *basis, *pairList);

    ASSERT_TRUE(store.has_value()) << store.error().message;
    ASSERT_EQ(store->size(), pairList->pairs.size());

    std::size_t widestPairFuncs = 0;

    for (const qcx::integrals::internal::MdPairData& pair : *store)
    {
        if (pair.nFuncs > widestPairFuncs)
        {
            widestPairFuncs = pair.nFuncs;
        }
    }

    const std::vector<double> charges(1, 1.0);
    std::vector<double> reference(kPointCount, 0.0);

    for (std::size_t p = 0; p < kPointCount; ++p)
    {
        std::vector<Eigen::Vector3d> center{
            Eigen::Vector3d((*points)(p, 0), (*points)(p, 1), (*points)(p, 2))};
        std::vector<double> block(widestPairFuncs);
        double value = 0.0;

        for (std::size_t pairIdx = 0; pairIdx < pairList->pairs.size(); ++pairIdx)
        {
            const qcx::integrals::ShellPairIndex& pairIndex = pairList->pairs[pairIdx];
            const qcx::integrals::internal::MdPairData& pair = (*store)[pairIdx];
            const std::size_t oI = pairList->shells[pairIndex.i].functionOffset;
            const std::size_t oJ = pairList->shells[pairIndex.j].functionOffset;
            const bool diagonal = pairIndex.i == pairIndex.j;
            qcx::integrals::internal::BuildNuclearPair(pair, charges, center, block.data());

            if (diagonal)
            {
                for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
                {
                    for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
                    {
                        value += density(static_cast<Eigen::Index>(oI + fa),
                                         static_cast<Eigen::Index>(oJ + fb)) *
                                 block[fa * pair.nFuncsB + fb];
                    }
                }
            } else
            {
                for (std::size_t fa = 0; fa < pair.nFuncsA; ++fa)
                {
                    for (std::size_t fb = 0; fb < pair.nFuncsB; ++fb)
                    {
                        value += (density(static_cast<Eigen::Index>(oI + fa),
                                          static_cast<Eigen::Index>(oJ + fb)) +
                                  density(static_cast<Eigen::Index>(oJ + fb),
                                          static_cast<Eigen::Index>(oI + fa))) *
                                 block[fa * pair.nFuncsB + fb];
                    }
                }
            }
        }

        reference[p] = value;
    }

    // One line of evidence the reader can check without the diff: how many
    // values the two agree on, exactly.
    std::size_t moved = 0;

    for (std::size_t p = 0; p < kPointCount; ++p)
    {
        if ((*chunked)[p] != reference[p])
        {
            ++moved;
            EXPECT_EQ((*chunked)[p], reference[p]) << "point " << p;
        }
    }

    std::printf("\n  ESP chunked vs pre-chunking walk: %zu points, %zu differ (store %.3f GiB, "
                "cap %.3f GiB)\n",
                kPointCount,
                moved,
                static_cast<double>(storeBytes) / (1024.0 * 1024.0 * 1024.0),
                static_cast<double>(qcx::integrals::internal::kPairChunkBytes) /
                    (1024.0 * 1024.0 * 1024.0));
    std::fflush(stdout);
    EXPECT_EQ(moved, 0u);
}

} // namespace
