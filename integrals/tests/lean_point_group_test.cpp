// The lean builder's Abelian point-group lane (the 2026-09-12 ruling, which
// supersedes the earlier "the symmetry reduction is skipped"): the reduction
// reaches the lean path as a CLASSIFICATION, not as a contraction
// (LeanFockBuildOptions::symmetryReduction), and the row walk drops the
// canonical cells whose ERI block the classification proves EXACTLY zero.
// The lane's second half is the ORBIT EXPANSION
// (LeanFockBuildOptions::symmetryOrbitExpansion): the walk visits one cell per
// symmetry orbit and the contraction expands the representative's block over
// the orbit's members - the petite-list orbit expansion, whose tests and
// their own pins are at the end of this file.
//
// What the tests pin, in order of load:
//   - the acceptance invariant: with the filter ENGAGED and DISENGAGED the
//     same fixture produces BYTE-IDENTICAL Fock matrices. Dropping an exact
//     zero cannot move a floating-point sum, so a byte difference here means
//     the filter dropped something that was not zero - a correctness bug,
//     not a tuning issue.
//   - the measured boundary of the rule the reduction cannot support: the
//     four-shell "the product of the four shell irreps must contain A1" test
//     is NOT sufficient - (O2s Opx | H1s H1s) on water/C2v has a b1 product
//     and is -0.10166208704320134 (pinned here against the machinery's own
//     path). The filter therefore drops a cell only when ONE group element
//     fixes all four of its shells function-for-function and the four signs
//     multiply to -1: otherwise the invariance relation links two non-zero
//     integrals instead of forcing a zero.
//   - the drop is real where the rule bites: the inversion fixture (a linear
//     H-O-H, whose O sits ON the inversion centre) drops exactly the cells
//     the rule proves zero, and nothing else.
//   - the k = 1 determinism contract survives: a filtered build is
//     byte-stable across calls and across fresh builders.
//   - the reduction validation: a malformed reduction is refused
//     (kInvalidArgument), never silently believed.
//   - the ORBIT EXPANSION's invariant: a per-pair map is not guaranteed to
//     reproduce a quartet-level reduction, so the live expanded walk's ERI
//     block count is measured against the machinery's own class-pair orbit
//     count on the fixture where the mask bites - 120 screened cells, 112
//     after the provable-zero drop, 68 ERI blocks, against the machinery's
//     76 orbits over the same 120 cells (its 76 - 8 dropped singletons). The
//     two pruning predicates are orbit-invariant, measured cell by cell
//     there, which is why the walk's screen-then-dedupe order cannot collapse
//     the reduction to the surviving cells.

#include "fast_test_mode.hpp"
#include "h2o_sto3g.hpp"
#include "internal/fock_screen.hpp"
#include "internal/lean_orbit_action.hpp"
#include "internal/md_batch.hpp"
#include "qcx/backend/tags.hpp"
#include "qcx/integrals/accuracy.hpp"
#include "qcx/integrals/eri_batch.hpp"
#include "qcx/integrals/fock_build.hpp"
#include "qcx/integrals/lean_fock_build.hpp"
#include "qcx/integrals/one_electron.hpp"
#include "qcx/integrals/screening.hpp"
#include "qcx/integrals/shell_pairs.hpp"
#include "qcx/integrals/symmetry_reduction.hpp"
#include "tensor_conversions.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using CpuTensor2 = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>;
using qcx::testing::IsFastOnlyMode;
using qcx::testing::MakeH2oSto3g;
using qcx::testing::MakeH2oSto3gBasis;
using qcx::testing::ToMatrix;
using qcx::testing::ToTensor;

// H = T + V from the one-electron engines (the builders' shared test
// convention, fock_build_test.cpp).
qcx::Result<CpuTensor2> BuildCoreHamiltonian(const qcx::molecule::Molecule& molecule,
                                             const qcx::basisset::BasisSet& basisSet) {
    auto kinetic = qcx::integrals::BuildKineticMatrix(molecule, basisSet);

    if (!kinetic.has_value())
    {
        return std::unexpected(kinetic.error());
    }

    auto nuclear = qcx::integrals::BuildNuclearAttractionMatrix(molecule, basisSet);

    if (!nuclear.has_value())
    {
        return std::unexpected(nuclear.error());
    }

    const std::size_t n = kinetic->Shape()[0];

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            (*kinetic)(i, j) += (*nuclear)(i, j);
        }
    }

    kinetic->MarkHostDirty();
    return std::move(*kinetic);
}

// The water C2v reduction over the real pair-list function order, duplicated
// from fock_build_test.cpp (the integrals module cannot include scf
// headers). Shell order [H1s, H2s, O1s, O2s, Op], function order
// [H1s, H2s, O1s, O2s, Opy, Opz, Opx]; with the molecule in the xy plane
// the elements are {I, sigma_z (z-flip), sigma_x (x-flip), C2 (about y)} -
// sigma_x and C2 swap the two H functions.
qcx::integrals::SymmetryReduction MakeWaterC2vReductionHFirst() {
    qcx::integrals::SymmetryReduction reduction;
    reduction.groupOrder = 4;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5, 6}, // I.
        {0, 1, 2, 3, 4, 5, 6}, // sigma_z: identity permutation.
        {1, 0, 2, 3, 4, 5, 6}, // sigma_x: swaps the two H functions.
        {1, 0, 2, 3, 4, 5, 6}, // C2: swaps the two H functions.
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1, 1}, // I.
        {1, 1, 1, 1, 1, -1, 1}, // sigma_z: p_z -> -p_z.
        {1, 1, 1, 1, 1, 1, -1}, // sigma_x: p_x -> -p_x.
        {1, 1, 1, 1, 1, -1, -1}, // C2: p_z and p_x flip.
    };
    return reduction;
}

// The linear H-O-H fixture: O at the origin, the H atoms at (0, 0, +-d).
// Its point group (D_infinity_h) contains the inversion about the O, and Ci
// is the subgroup the reduction realizes - the ONE shape in which the
// classification can prove a cell zero with a Cartesian shell basis: the
// inversion fixes every function of a shell CENTRED ON THE INVERSION CENTRE
// and its sign is (-1)^l on every function of that shell. A shell off the
// centre is permuted to its partner instead, which the rule refuses to
// treat as a zero (see the counterexample test).
qcx::Result<qcx::molecule::Molecule> MakeLinearHoh() {
    constexpr double kBondLength = 1.809067; // 0.9572 A, the water O-H.

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({3, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    (*coordinates)(0, 0) = 0.0;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 0.0;
    (*coordinates)(1, 1) = 0.0;
    (*coordinates)(1, 2) = kBondLength;
    (*coordinates)(2, 0) = 0.0;
    (*coordinates)(2, 1) = 0.0;
    (*coordinates)(2, 2) = -kBondLength;
    coordinates->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::vector<qcx::molecule::Atom>{{"O", 8, 0.0}, {"H", 1, 0.0}, {"H", 1, 0.0}},
        std::move(*coordinates),
        0,
        1);
}

// The Ci reduction of the linear H-O-H fixture over the same function order
// as water ([H1s, H2s, O1s, O2s, Opy, Opz, Opx]): the inversion swaps the
// two H functions, fixes the O functions and flips the three p signs.
qcx::integrals::SymmetryReduction MakeLinearHohInversionReduction() {
    qcx::integrals::SymmetryReduction reduction;
    reduction.groupOrder = 2;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5, 6}, // I.
        {1, 0, 2, 3, 4, 5, 6}, // i: H1 <-> H2, the O functions fixed.
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1, 1}, // I.
        {1, 1, 1, 1, -1, -1, -1}, // i: x, y, z all flip - the p signs.
    };
    return reduction;
}

// Bit equality of two matrices (the k = 1 byte contract).
bool BitIdentical(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
    {
        return false;
    }

    if (a.size() == 0)
    {
        return true;
    }

    return std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.size()) * sizeof(double)) ==
           0;
}

// A symmetric density with |D| <= 0.75 and a diagonal near 0.5 (the lean
// test file's PhysicalDensity).
// (n, seed) is the size-then-seed order of the random-density helper.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
Eigen::MatrixXd PhysicalDensity(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-0.25, 0.25);
    Eigen::MatrixXd d(static_cast<Eigen::Index>(n), static_cast<Eigen::Index>(n));

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            const double value = (i == j) ? 0.5 + dist(rng) : dist(rng);
            d(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = value;
            d(static_cast<Eigen::Index>(j), static_cast<Eigen::Index>(i)) = value;
        }
    }

    return d;
}

// The lean kTight k=1 options every byte pin uses.
qcx::integrals::LeanFockBuildOptions LeanSerialOptions() {
    qcx::integrals::LeanFockBuildOptions options;
    options.accuracy = qcx::integrals::AccuracyPreset::kTight;
    options.maxParallelChunks = 1;
    return options;
}

// The INDEPENDENT survivor walk of one fixture under one reduction: the
// canonical cells (rows r, kets q <= r) that clear the Schwarz cutoff, and
// of those the ones the classification proves exactly zero. The mask
// derivation here is the test's own (a second implementation of the
// production rule - agreement is a cross-check, not a tautology).
struct CellWalk {
    std::size_t survivors = 0; ///< The unfiltered surviving cell count.
    std::size_t dropped = 0; ///< Of those, the provably-zero cells (the mask rule).
    /// Of those, the cells the TEXTBOOK rule drops: the bra pair's and the ket
    /// pair's Abelian irrep labels differ (for Abelian groups the product of
    /// the four irreps contains the totally symmetric irrep iff the two pairs'
    /// labels agree). Derived here from the reduction's per-function signs
    /// alone, independently of the mask path - the equivalence is asserted,
    /// not assumed.
    std::size_t droppedByIrrepLabel = 0;
    /// Of those, the cells the textbook rule drops with the pair-mappable
    /// labels RESTRICTED to the elements that fix both pairs: the production
    /// rule's own set, reached from the characters instead of the masks.
    std::size_t droppedByFixedIrrepLabel = 0;
};

CellWalk WalkCells(const std::vector<double>& bounds,
                   const qcx::integrals::ShellPairList& pairList,
                   const qcx::integrals::SymmetryReduction* reduction,
                   qcx::integrals::AccuracyPreset accuracy) {
    const double cutoff =
        qcx::integrals::SchwarzThreshold(accuracy) * qcx::integrals::internal::kNeighborListSlack;
    const std::size_t nShells = pairList.shells.size();
    const std::size_t order = (reduction != nullptr) ? reduction->groupOrder : 1;

    // Per shell: the fixed-and-pure-sign masks (the test's own derivation).
    std::vector<std::uint32_t> shellPlus(nShells, 0u);
    std::vector<std::uint32_t> shellMinus(nShells, 0u);

    if (reduction != nullptr && order > 1)
    {
        for (std::size_t s = 0; s < nShells; ++s)
        {
            const qcx::integrals::ShellInfo& shell = pairList.shells[s];
            const std::size_t count = shell.angularMomentum + 1;

            for (std::size_t g = 0; g < order; ++g)
            {
                bool fixed = true;
                bool plus = true;
                bool minus = true;

                for (std::size_t k = 0; k < count; ++k)
                {
                    const std::size_t f = shell.functionOffset + k;
                    fixed = fixed && (reduction->permutation[g][f] == f);
                    plus = plus && (reduction->sign[g][f] == 1);
                    minus = minus && (reduction->sign[g][f] == -1);
                }

                if (fixed && plus)
                {
                    shellPlus[s] |= 1u << g;
                }

                if (fixed && minus)
                {
                    shellMinus[s] |= 1u << g;
                }
            }
        }
    }

    // The pair's Abelian irrep CHARACTER, derived independently of the mask
    // path: under element g the pair (i, j) is MAPPABLE when g maps each of its
    // two shells onto one of the two (to itself or to the other), and its label
    // is then the sign product s(mu)s(nu) - well defined only when every
    // function pair of the pair gives the same product. That is the textbook
    // "the pair transforms as the one-dimensional character chi_g", and
    // "Gamma(bra) != Gamma(ket)" is the textbook zero test for an Abelian
    // group: by that reading a pair whose two shells SWAP under g carries a
    // label too.
    std::vector<std::size_t> shellOfFunction(pairList.functionCount, 0);

    for (std::size_t s = 0; s < nShells; ++s)
    {
        const std::size_t count = pairList.shells[s].angularMomentum + 1;

        for (std::size_t k = 0; k < count; ++k)
        {
            shellOfFunction[pairList.shells[s].functionOffset + k] = s;
        }
    }

    const auto shellImage = [&](std::size_t s, std::size_t g) {
        return (reduction != nullptr && order > 1)
                   ? shellOfFunction[reduction->permutation[g][pairList.shells[s].functionOffset]]
                   : s;
    };

    const std::size_t nPairs = pairList.pairs.size();
    std::vector<std::uint32_t> pairLabelPlus(nPairs, 0u);
    std::vector<std::uint32_t> pairLabelMinus(nPairs, 0u);
    std::vector<std::uint32_t> pairFixedLabelPlus(nPairs, 0u);
    std::vector<std::uint32_t> pairFixedLabelMinus(nPairs, 0u);

    if (reduction != nullptr && order > 1)
    {
        for (std::size_t p = 0; p < nPairs; ++p)
        {
            const std::size_t i = pairList.pairs[p].i;
            const std::size_t j = pairList.pairs[p].j;
            const std::size_t countI = pairList.shells[i].angularMomentum + 1;
            const std::size_t countJ = pairList.shells[j].angularMomentum + 1;

            for (std::size_t g = 0; g < order; ++g)
            {
                const std::size_t imageI = shellImage(i, g);
                const std::size_t imageJ = shellImage(j, g);
                const bool mappable =
                    ((imageI == i && imageJ == j) || (imageI == j && imageJ == i));

                if (!mappable)
                {
                    continue;
                }

                const bool fixed = (imageI == i && imageJ == j);
                bool plus = true;
                bool minus = true;

                for (std::size_t a = 0; a < countI; ++a)
                {
                    for (std::size_t b = 0; b < countJ; ++b)
                    {
                        const int product =
                            reduction->sign[g][pairList.shells[i].functionOffset + a] *
                            reduction->sign[g][pairList.shells[j].functionOffset + b];
                        plus = plus && (product == 1);
                        minus = minus && (product == -1);
                    }
                }

                if (plus)
                {
                    pairLabelPlus[p] |= 1u << g;

                    if (fixed)
                    {
                        pairFixedLabelPlus[p] |= 1u << g;
                    }
                }

                if (minus)
                {
                    pairLabelMinus[p] |= 1u << g;

                    if (fixed)
                    {
                        pairFixedLabelMinus[p] |= 1u << g;
                    }
                }
            }
        }
    }

    // The pair's label under g from the character derivation above, in its two
    // readings: swap-permitting (pairLabel) and fixing-only (pairFixedLabel).
    const auto pairLabel = [&](std::size_t p, bool wantMinus) {
        if (wantMinus)
        {
            return pairLabelMinus[p];
        }

        return pairLabelPlus[p];
    };

    const auto pairFixedLabel = [&](std::size_t p, bool wantMinus) {
        if (wantMinus)
        {
            return pairFixedLabelMinus[p];
        }

        return pairFixedLabelPlus[p];
    };

    const auto pairMask = [&](std::size_t p, bool wantMinus) {
        const std::size_t i = pairList.pairs[p].i;
        const std::size_t j = pairList.pairs[p].j;

        if (wantMinus)
        {
            return (shellPlus[i] & shellMinus[j]) | (shellMinus[i] & shellPlus[j]);
        }

        return (shellPlus[i] & shellPlus[j]) | (shellMinus[i] & shellMinus[j]);
    };

    CellWalk walk;

    for (std::size_t r = 0; r < bounds.size(); ++r)
    {
        for (std::size_t q = 0; q <= r; ++q)
        {
            if (bounds[r] * bounds[q] < cutoff)
            {
                continue;
            }

            ++walk.survivors;

            if (reduction != nullptr && order > 1)
            {
                const std::uint32_t rowPlus = pairMask(r, false);
                const std::uint32_t rowMinus = pairMask(r, true);
                const std::uint32_t ketPlus = pairMask(q, false);
                const std::uint32_t ketMinus = pairMask(q, true);

                if (((rowPlus & ketMinus) | (rowMinus & ketPlus)) != 0u)
                {
                    ++walk.dropped;
                }

                // The textbook form, element-agnostic: the bra and ket pairs'
                // irrep labels differ ("the product of the four irreps does not
                // contain the totally symmetric irrep"). Read at pair
                // granularity it also covers pairs whose two shells SWAP under
                // the element - which the production rule refuses, because the
                // zero it forces rests on the ERI's own role-swap identity,
                // which the floating-point arithmetic does not deliver
                // bit-exactly. Both counts are reported: the FIXED one must
                // equal the mask rule's, and the broad one marks exactly where
                // the byte-identity boundary lies.
                if (((pairLabel(r, false) & pairLabel(q, true)) |
                     (pairLabel(r, true) & pairLabel(q, false))) != 0u)
                {
                    ++walk.droppedByIrrepLabel;
                }

                if (((pairFixedLabel(r, false) & pairFixedLabel(q, true)) |
                     (pairFixedLabel(r, true) & pairFixedLabel(q, false))) != 0u)
                {
                    ++walk.droppedByFixedIrrepLabel;
                }
            }
        }
    }

    return walk;
}

} // namespace

// The acceptance invariant on water/C2v: the filter changes no byte. The
// C2v reduction of water fixes no cell's four shells with an odd sign
// product (its p shell is sign-MIXED under every non-identity element, so
// no pair of shell pairs can be classified), so the drop count here is
// zero - the byte identity is then the guard that the engaged filter is
// inert, not a claim that it saved work.
TEST(LeanPointGroupTest, WaterC2vFilteredAndUnfilteredBuildsShareTheirBytes) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();
    const CellWalk walk =
        WalkCells(*bounds, *pairList, &reduction, qcx::integrals::AccuracyPreset::kTight);

    qcx::integrals::LeanFockBuildOptions plainOptions = LeanSerialOptions();
    auto plain =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    qcx::integrals::LeanFockBuildOptions filteredOptions = LeanSerialOptions();
    filteredOptions.symmetryReduction = &reduction;
    auto filtered =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, filteredOptions);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 20260912);

    qcx::integrals::FockBuildStats plainStats;
    auto plainFock = plain->BuildFock(*ToTensor(density), &plainStats);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;

    qcx::integrals::FockBuildStats filteredStats;
    auto filteredFock = filtered->BuildFock(*ToTensor(density), &filteredStats);
    ASSERT_TRUE(filteredFock.has_value()) << filteredFock.error().message;

    EXPECT_TRUE(BitIdentical(ToMatrix(*plainFock), ToMatrix(*filteredFock)))
        << "the filter moved a byte: it dropped something that was not zero";
    EXPECT_EQ(walk.dropped, 0u)
        << "water/C2v has no provably-zero cell - the byte identity above is the inert leg";
    EXPECT_EQ(walk.droppedByFixedIrrepLabel, walk.dropped)
        << "the mask rule IS the textbook irrep-mismatch rule, once the pair-mappable "
           "labels are restricted to the elements that fix both pairs";
    EXPECT_EQ(plainStats.fp64QuartetCount, walk.survivors);
    EXPECT_EQ(filteredStats.fp64QuartetCount, walk.survivors - walk.dropped)
        << "the filtered call must count the cells it actually computed";
}

// The inversion fixture (a linear H-O-H whose O sits on the inversion
// centre): the classification proves the odd-l-sum cells among the O's own
// shells zero, the row walk drops them, and the remaining bytes are
// identical to the unfiltered build. This is the filter's whole reach on a
// Cartesian s/p basis, and it is exactly the cell set the reduction's own
// invariances forbid.
TEST(LeanPointGroupTest, InversionCentreFixtureDropsItsProvablyZeroCells) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeLinearHoh();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeLinearHohInversionReduction();
    const CellWalk walk =
        WalkCells(*bounds, *pairList, &reduction, qcx::integrals::AccuracyPreset::kTight);

    qcx::integrals::LeanFockBuildOptions plainOptions = LeanSerialOptions();
    auto plain =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    qcx::integrals::LeanFockBuildOptions filteredOptions = LeanSerialOptions();
    filteredOptions.symmetryReduction = &reduction;
    auto filtered =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, filteredOptions);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 4242);

    qcx::integrals::FockBuildStats plainStats;
    auto plainFock = plain->BuildFock(*ToTensor(density), &plainStats);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;

    qcx::integrals::FockBuildStats filteredStats;
    auto filteredFock = filtered->BuildFock(*ToTensor(density), &filteredStats);
    ASSERT_TRUE(filteredFock.has_value()) << filteredFock.error().message;

    EXPECT_GT(walk.dropped, 0u) << "the inversion fixture must have provably-zero cells";
    EXPECT_TRUE(BitIdentical(ToMatrix(*plainFock), ToMatrix(*filteredFock)))
        << "the filter moved a byte: it dropped something that was not zero";
    EXPECT_EQ(plainStats.fp64QuartetCount, walk.survivors);
    EXPECT_EQ(filteredStats.fp64QuartetCount, walk.survivors - walk.dropped);
    EXPECT_GT(filteredStats.batchCount, 0u);

    // The dropped set is not "the cells the p shell touches": the O's own
    // shell pairs split by sign under the inversion, and the odd-l-sum cells
    // among them go. Asserting the exact count keeps a wider rule from
    // slipping in unnoticed. The O shells are O1s, O2s and Op (functions
    // 2..6 of the H-first order); the even pairs are (1s,1s), (1s,2s),
    // (2s,2s), (p,p) and the odd ones (1s,p), (2s,p), so the cells are
    // 4 even x 2 odd = 8.
    EXPECT_EQ(walk.dropped, 8u);
    EXPECT_GT(walk.droppedByIrrepLabel, walk.dropped)
        << "the swap-permitting reading is strictly wider - and it is not the rule this "
           "builder may use: its zeros rest on the ERI's role-swap identity, which the "
           "arithmetic does not deliver bit-exactly";
    EXPECT_EQ(walk.droppedByFixedIrrepLabel, walk.dropped)
        << "the mask rule IS the textbook irrep-mismatch rule, once the pair-mappable "
           "labels are restricted to the elements that fix both pairs";
}

// The measured boundary: the four-shell "product of the four shell irreps
// must contain the totally symmetric irrep" rule is NOT a zero test. On
// water/C2v the quartet (O2s Opx | H1s H1s) has the product b1 x a1 x a1 x
// a1 = b1 - the rule's drop - and the integral is -0.10166208704320134,
// pinned against the machinery's own path below. The reason is structural:
// the element that carries the sign (sigma_x) SWAPS the two H atoms, so the
// invariance relation links (O2s Opx | H1s H1s) to -(O2s Opx | H2s H2s)
// instead of forcing a zero. A filter built on that rule would corrupt the
// Fock; the production filter refuses exactly this case.
TEST(LeanPointGroupTest, TheFourShellIrrepProductRuleIsNotAZeroTest) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    // Function order [H1s, H2s, O1s, O2s, Opy, Opz, Opx].
    constexpr std::size_t kH1s = 0;
    constexpr std::size_t kH2s = 1;
    constexpr std::size_t kO2s = 3;
    constexpr std::size_t kOpx = 6;

    // rho = e_{H1s,H1s} with the J-only lane: F = H + 2 J(rho) turns the
    // returned matrix into the ERIs (m n | H1s H1s) at (F - H)/2.
    Eigen::MatrixXd rho = Eigen::MatrixXd::Zero(7, 7);
    rho(static_cast<Eigen::Index>(kH1s), static_cast<Eigen::Index>(kH1s)) = 1.0;
    Eigen::MatrixXd rhoOther = Eigen::MatrixXd::Zero(7, 7);
    rhoOther(static_cast<Eigen::Index>(kH2s), static_cast<Eigen::Index>(kH2s)) = 1.0;

    const Eigen::MatrixXd hCore = ToMatrix(*core);

    qcx::integrals::LeanFockBuildOptions leanOptions = LeanSerialOptions();
    leanOptions.buildCoulombOnly = true;
    auto lean =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, leanOptions);
    ASSERT_TRUE(lean.has_value()) << lean.error().message;

    qcx::integrals::FockBuildOptions machineryOptions;
    machineryOptions.accuracy = qcx::integrals::AccuracyPreset::kTight;
    machineryOptions.useDensityScreening = false;
    machineryOptions.usePerElementScreening = false;
    machineryOptions.useCertifiedMixedPrecision = false;
    machineryOptions.maxParallelChunks = 1;
    machineryOptions.buildCoulombOnly = true;
    auto machinery =
        qcx::integrals::DirectJkFockBuilder::Create(*molecule, *basis, *core, machineryOptions);
    ASSERT_TRUE(machinery.has_value()) << machinery.error().message;

    auto leanFirst = lean->BuildFock(*ToTensor(rho));
    ASSERT_TRUE(leanFirst.has_value()) << leanFirst.error().message;
    auto leanSecond = lean->BuildFock(*ToTensor(rhoOther));
    ASSERT_TRUE(leanSecond.has_value()) << leanSecond.error().message;
    auto machineryFirst = machinery->BuildFock(*ToTensor(rho));
    ASSERT_TRUE(machineryFirst.has_value()) << machineryFirst.error().message;

    const auto eri = [&hCore](const Eigen::MatrixXd& fock, std::size_t m, std::size_t n) {
        return (fock(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n)) -
                hCore(static_cast<Eigen::Index>(m), static_cast<Eigen::Index>(n))) /
               2.0;
    };

    const double value = eri(ToMatrix(*leanFirst), kO2s, kOpx);
    const double mirrored = eri(ToMatrix(*leanSecond), kO2s, kOpx);
    const double machineryValue = eri(ToMatrix(*machineryFirst), kO2s, kOpx);

    EXPECT_NEAR(value, -0.10166208704320134, 1e-12)
        << "the pinned (O2s Opx | H1s H1s) reference value";
    EXPECT_NEAR(value, machineryValue, 1e-12) << "the machinery must agree on the same ERI";
    EXPECT_NEAR(value + mirrored, 0.0, 1e-15)
        << "the sigma_x relation: (O2s Opx | H1s H1s) = -(O2s Opx | H2s H2s)";
    EXPECT_GT(std::abs(value), 1e-3) << "the 'forbidden' quartet is not small - it is not zero";

    // The rule's own predicate, read off the reduction: the four shells'
    // signs under sigma_x (element 2) multiply to -1, i.e. the product of
    // the four shell irreps does NOT contain A1 - the rule would drop this
    // quartet. The production filter refuses it because sigma_x does not fix
    // the H functions.
    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();
    const int product = reduction.sign[2][kO2s] * reduction.sign[2][kOpx] *
                        reduction.sign[2][kH1s] * reduction.sign[2][kH1s];
    EXPECT_EQ(product, -1) << "the reduction must classify this quartet as 'forbidden'";
    EXPECT_EQ(reduction.permutation[2][kH1s], kH2s)
        << "and the element must move the H atoms - which is why it is not a zero";
}

// The determinism contract survives the filter: a filtered k = 1 build is
// byte-stable across calls and across fresh builders (the lean pin).
TEST(LeanPointGroupTest, FilteredSerialPathPinsItsOwnBytesAcrossCallsAndBuilders) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeLinearHoh();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeLinearHohInversionReduction();

    qcx::integrals::LeanFockBuildOptions options = LeanSerialOptions();
    options.symmetryReduction = &reduction;
    auto first = qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(first.has_value()) << first.error().message;
    auto second = qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);
    ASSERT_TRUE(second.has_value()) << second.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 77);

    auto firstFock = first->BuildFock(*ToTensor(density));
    ASSERT_TRUE(firstFock.has_value()) << firstFock.error().message;
    auto repeatFock = first->BuildFock(*ToTensor(density));
    ASSERT_TRUE(repeatFock.has_value()) << repeatFock.error().message;
    auto secondFock = second->BuildFock(*ToTensor(density));
    ASSERT_TRUE(secondFock.has_value()) << secondFock.error().message;

    EXPECT_TRUE(BitIdentical(ToMatrix(*firstFock), ToMatrix(*repeatFock))) << "repeat call drifted";
    EXPECT_TRUE(BitIdentical(ToMatrix(*firstFock), ToMatrix(*secondFock)))
        << "a fresh builder drifted";

    // The threaded path walks the same filtered cells as the serial one
    // (the counts must agree) - the filter is per-cell, so the window
    // decomposition cannot change which cells are dropped.
    qcx::integrals::LeanFockBuildOptions threadedOptions = options;
    threadedOptions.maxParallelChunks = 4;
    auto threaded =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, threadedOptions);
    ASSERT_TRUE(threaded.has_value()) << threaded.error().message;

    qcx::integrals::FockBuildStats serialStats;
    auto serialFock = first->BuildFock(*ToTensor(density), &serialStats);
    ASSERT_TRUE(serialFock.has_value()) << serialFock.error().message;
    qcx::integrals::FockBuildStats threadedStats;
    auto threadedFock = threaded->BuildFock(*ToTensor(density), &threadedStats);
    ASSERT_TRUE(threadedFock.has_value()) << threadedFock.error().message;

    EXPECT_EQ(serialStats.fp64QuartetCount, threadedStats.fp64QuartetCount);
}

// The reduction validation: a malformed reduction is refused, never
// silently believed (a wrong classification would drop non-zero cells).
TEST(LeanPointGroupTest, MalformedReductionsAreRefused) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeLinearHoh();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const auto create = [&](const qcx::integrals::SymmetryReduction& reduction) {
        qcx::integrals::LeanFockBuildOptions options = LeanSerialOptions();
        options.symmetryReduction = &reduction;
        return qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, options);
    };

    // A row shorter than the function count.
    qcx::integrals::SymmetryReduction shortRow = MakeLinearHohInversionReduction();
    shortRow.permutation[1].pop_back();
    EXPECT_FALSE(create(shortRow).has_value());

    // A sign outside +-1.
    qcx::integrals::SymmetryReduction badSign = MakeLinearHohInversionReduction();
    badSign.sign[1][4] = 0;
    EXPECT_FALSE(create(badSign).has_value());

    // An index out of range.
    qcx::integrals::SymmetryReduction outOfRange = MakeLinearHohInversionReduction();
    outOfRange.permutation[1][0] = 99;
    EXPECT_FALSE(create(outOfRange).has_value());

    // A first element that is not the identity.
    qcx::integrals::SymmetryReduction noIdentity = MakeLinearHohInversionReduction();
    noIdentity.permutation[0][0] = 1;
    noIdentity.permutation[0][1] = 0;
    EXPECT_FALSE(create(noIdentity).has_value());

    // A group wider than the classification's 32 elements.
    qcx::integrals::SymmetryReduction tooWide = MakeLinearHohInversionReduction();
    tooWide.groupOrder = 40;
    EXPECT_FALSE(create(tooWide).has_value());

    // The trivial reduction (the default-constructed one) is NOT an error:
    // it is the unfiltered path, byte-for-byte.
    const qcx::integrals::SymmetryReduction trivial;
    auto plain = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());
    ASSERT_TRUE(plain.has_value()) << plain.error().message;
    auto trivialBuilder = create(trivial);
    ASSERT_TRUE(trivialBuilder.has_value()) << trivialBuilder.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 1234);
    auto plainFock = plain->BuildFock(*ToTensor(density));
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    auto trivialFock = trivialBuilder->BuildFock(*ToTensor(density));
    ASSERT_TRUE(trivialFock.has_value()) << trivialFock.error().message;

    EXPECT_TRUE(BitIdentical(ToMatrix(*plainFock), ToMatrix(*trivialFock)))
        << "a trivial reduction must be the unfiltered path exactly";
}

// ===========================================================================
// The petite-list orbit expansion: the ORBIT EXPANSION.
// With LeanFockBuildOptions::symmetryOrbitExpansion the walk visits one
// canonical cell per symmetry orbit (its surviving minimum) and the
// contraction expands that representative's block over the orbit's members.
// What the tests pin, in order of load:
//   - the BLOCK MAP itself, element by element, against the engine's own
//     block for the member's quartet - the frame composition (the two role
//     swaps, the within-pair flips, the whole-quartet role swap and the
//     function-level position map) is checked independently of any Fock
//     contraction, and a wrong axis map shows here as an O(1) difference;
//   - the WORK: the expanded build's fp64QuartetCount is the orbit count and
//     the plain build's is the covered member count - both cross-checked
//     against the machinery's OWN class decomposition
//     (BuildPairClasses + GenerateClassPairOrbits) at the same cutoff, so the
//     mechanism is verified against the family that already implements it;
//   - the DELTA against the plain build, as a ceiling with headroom: the
//     expansion replaces a member's evaluation with the representative's, so
//     the two differ by the engine's own ordering arithmetic and not at all
//     in exact arithmetic; the machinery
//     pins its own class path against the plain path the same way, at 1e-12,
//     fock_build_test.cpp:772). The delta is a CEILING, never an equality
//     pin: a bit-exact expansion would drive it to zero and leave the
//     assertion green - the counts leg is what a no-op mechanism cannot fake;
//   - the k = 1 determinism contract under the expansion (byte-identical
//     across calls and builders), and the windowed path's identical counts;
//   - the inerts and the refusals: the option with a trivial reduction is the
//     plain path byte for byte, the option without a reduction is refused,
//     and a reduction that splits a shell (no shell-closure) is refused
//     rather than believed.

namespace {

// The engine's own fp64 block for one ALREADY CANONICAL quartet - the
// independent reference the expansion is compared against. The batch
// machinery canonicalizes its request, and the request here is the form
// CanonicalTaskFormOf produces, so the assertion below is itself a check that
// the lean path's canonical form is the engine's.
std::vector<double> EngineBlockOf(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basisSet,
                                  const qcx::integrals::ShellQuartet& quartet) {
    auto batch = qcx::integrals::ComputeEriBatch(
        molecule, basisSet, std::vector<qcx::integrals::ShellQuartet>{quartet});

    if (!batch.has_value())
    {
        ADD_FAILURE() << batch.error().message;
        return {};
    }

    if (batch->computed.size() != 1 || batch->computed[0].i != quartet.i ||
        batch->computed[0].j != quartet.j || batch->computed[0].k != quartet.k ||
        batch->computed[0].l != quartet.l)
    {
        ADD_FAILURE() << "the lean canonical form is not the engine's canonical form";
        return {};
    }

    return batch->values;
}

// One canonical-role task form as a shell quartet (the form the assembler
// emits and the contraction reads).
qcx::integrals::ShellQuartet QuartetOfForm(
    const qcx::integrals::ShellPairList& pairList,
    const qcx::integrals::internal::MdCanonicalTaskForm& form) {
    const qcx::integrals::ShellPairIndex& bra = pairList.pairs[form.braPair];
    const qcx::integrals::ShellPairIndex& ket = pairList.pairs[form.ketPair];
    return qcx::integrals::ShellQuartet{bra.i, bra.j, ket.i, ket.j};
}

// The expansion's delta ceiling (absolute, in the ERI's own units). The
// mechanism is an expansion, so the expanded block and the engine's block are
// the same integral computed in two shell orders: they agree to the engine's
// own ordering arithmetic. The bound is the machinery's class-path precedent
// (1e-12, fock_build_test.cpp:772) and it is a CEILING - an improvement that
// makes the expansion bit-exact passes it unchanged. The measured maxima are
// reported in the assertions' messages.
constexpr double kOrbitExpansionDeltaCeiling = 1e-12;

// The measured-value formatter. std::to_string renders six decimals, so a
// last-bit delta - the exact quantity these pins exist to carry - prints as
// "0.000000", indistinguishable from an exact zero. Seventeen significant
// digits keep that distinction, which is the whole content of the delta leg.
std::string FormatMeasured(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

} // namespace

// The block map, element by element, for EVERY canonical cell of water/C2v
// and every generator: the expansion of the cell's block must reproduce the
// block the ENGINE computes for the member's quartet. This is the frame
// composition's own test - the two role swaps, the within-pair flips, the
// whole-quartet role swap and the function-level position map - and it needs
// no Fock contraction to exercise: a wrong axis map lands a value on the
// wrong element and shows as an O(1) difference, while a right one differs
// only by the engine's ordering arithmetic. The stabilizer case (a generator
// that maps the cell onto itself) is compared too: it is the block's own
// invariance under the group, which is the verdict the machinery's expansion
// relies on when it picks any element mapping a member.
TEST(LeanPointGroupTest, OrbitExpansionReproducesEveryMemberBlockTheEngineComputes) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();
    auto action = qcx::integrals::internal::LeanOrbitAction::Create(reduction, *pairList);
    ASSERT_TRUE(action.has_value()) << action.error().message;

    const std::size_t order = action->order;
    const std::size_t nPairs = pairList->pairs.size();
    double maxDelta = 0.0;
    double maxElement = 0.0;
    std::size_t memberChecks = 0;
    std::size_t stabilizerChecks = 0;
    std::size_t nonFinite = 0;
    std::vector<double> expanded;

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            const qcx::integrals::internal::MdCanonicalTaskForm repForm =
                qcx::integrals::internal::CanonicalTaskFormOf(*pairList, bra, ket);
            const std::vector<double> repBlock =
                EngineBlockOf(*molecule, *basis, QuartetOfForm(*pairList, repForm));

            if (repBlock.empty())
            {
                continue;
            }

            // The member set, seeded with the cell itself - the builder's own
            // dedup (a stabilizer maps the cell onto itself and must not be
            // contracted twice).
            std::array<std::size_t, qcx::integrals::internal::kSymmetryMaskBits> seenCells{};
            seenCells[0] = qcx::integrals::internal::LeanCellIndex(bra, ket);
            std::size_t seenCount = 1;

            for (std::size_t g = 1; g < order; ++g)
            {
                const std::size_t imageBra = action->pairImage[bra * order + g];
                const std::size_t imageKet = action->pairImage[ket * order + g];
                const std::size_t memberBra = std::max(imageBra, imageKet);
                const std::size_t memberKet = std::min(imageBra, imageKet);
                const std::size_t memberCell =
                    qcx::integrals::internal::LeanCellIndex(memberBra, memberKet);
                bool duplicate = false;

                for (std::size_t s = 0; s < seenCount; ++s)
                {
                    duplicate = duplicate || seenCells[s] == memberCell;
                }

                seenCells[seenCount] = memberCell;
                ++seenCount;

                if (duplicate && memberCell != qcx::integrals::internal::LeanCellIndex(bra, ket))
                {
                    continue;
                }

                const qcx::integrals::internal::MdCanonicalTaskForm memberForm =
                    qcx::integrals::internal::CanonicalTaskFormOf(*pairList, memberBra, memberKet);
                const std::vector<double> memberBlock =
                    EngineBlockOf(*molecule, *basis, QuartetOfForm(*pairList, memberForm));

                if (memberBlock.empty())
                {
                    continue;
                }

                expanded.resize(memberBlock.size());
                qcx::integrals::internal::LeanExpandOrbitMemberBlock(*action,
                                                                     *pairList,
                                                                     g,
                                                                     repForm.braPair,
                                                                     repForm.ketPair,
                                                                     memberForm.braPair,
                                                                     memberForm.ketPair,
                                                                     imageBra,
                                                                     imageKet,
                                                                     repBlock.data(),
                                                                     expanded.data());
                EXPECT_EQ(expanded.size(), memberBlock.size());

                for (std::size_t e = 0; e < expanded.size(); ++e)
                {
                    // NaN-PROPAGATING maxima, and non-finite counting: std::max
                    // and std::min SWALLOW a NaN (they return the other
                    // operand), so a NaN difference would leave a max-based
                    // metric reading zero while an inequality counter still
                    // fired - the false green this test grew the counters to
                    // catch. `!(x <= max)` propagates NaN into the metric.
                    const double delta = std::abs(expanded[e] - memberBlock[e]);

                    if (!(delta <= maxDelta))
                    {
                        maxDelta = delta;
                    }

                    if (!(std::abs(memberBlock[e]) <= maxElement))
                    {
                        maxElement = std::abs(memberBlock[e]);
                    }

                    if (!std::isfinite(expanded[e]) || !std::isfinite(memberBlock[e]))
                    {
                        ++nonFinite;
                    }
                }

                if (memberCell == qcx::integrals::internal::LeanCellIndex(bra, ket))
                {
                    ++stabilizerChecks;
                } else
                {
                    ++memberChecks;
                }
            }
        }
    }

    // Water/STO-3G C2v: 120 canonical cells, 76 orbits - the non-representative
    // expansions are the ones this test exists for, and no generator/vs the
    // fixture's four elements produces fewer than this.
    // The measured values are RECORDED, not merely asserted on: an assertion
    // speaks only when it fails, and the record wants the numbers either way
    // (gtest's XML output carries the properties).
    RecordProperty("max_block_delta", FormatMeasured(maxDelta));
    RecordProperty("max_block_element", FormatMeasured(maxElement));
    RecordProperty("member_checks", static_cast<int>(memberChecks));
    RecordProperty("stabilizer_checks", static_cast<int>(stabilizerChecks));
    RecordProperty("nonfinite_elements", static_cast<int>(nonFinite));

    EXPECT_GT(memberChecks, 40u) << "the fixture must exercise the expansion broadly";
    EXPECT_GT(stabilizerChecks, 10u)
        << "the fixture must exercise a cell's own invariance under a generator";
    EXPECT_EQ(nonFinite, 0u)
        << "a non-finite value reached a block: the expansion copied one, or the engine made one";
    EXPECT_LT(maxDelta, kOrbitExpansionDeltaCeiling)
        << "max |expanded - engine| = " << maxDelta << " against a block max of " << maxElement;
}

// The work, cross-checked against the machinery's own class decomposition.
// The machinery's BuildPairClasses + GenerateClassPairOrbits compute the same
// orbits the lean walk enumerates - at the same Schwarz cutoff, over the same
// pair list - so their orbit count must equal the expanded build's
// fp64QuartetCount and their member sum must equal the plain build's. That is
// the count leg: the work is a count, and the count is the family's
// own, not this test's model.
TEST(LeanPointGroupTest, OrbitExpansionCountsMatchTheMachineryClassDecomposition) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();

    qcx::integrals::LeanFockBuildOptions plainOptions = LeanSerialOptions();
    auto plain =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    qcx::integrals::LeanFockBuildOptions expandedOptions = LeanSerialOptions();
    expandedOptions.symmetryReduction = &reduction;
    expandedOptions.symmetryOrbitExpansion = true;
    auto expanded =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, expandedOptions);
    ASSERT_TRUE(expanded.has_value()) << expanded.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 20260912);
    qcx::integrals::FockBuildStats plainStats;
    auto plainFock = plain->BuildFock(*ToTensor(density), &plainStats);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    qcx::integrals::FockBuildStats expandedStats;
    auto expandedFock = expanded->BuildFock(*ToTensor(density), &expandedStats);
    ASSERT_TRUE(expandedFock.has_value()) << expandedFock.error().message;

    // The machinery's own decomposition, at the cutoff the lean walk screens
    // with (SchwarzThreshold(preset) x kNeighborListSlack - the same value
    // the neighbor-list path passes, fock_build.cpp).
    const double cutoff = qcx::integrals::SchwarzThreshold(plainOptions.accuracy) *
                          qcx::integrals::internal::kNeighborListSlack;
    auto table = qcx::integrals::BuildPairClasses(reduction, *pairList, *bounds, cutoff);
    ASSERT_TRUE(table.has_value()) << table.error().message;

    std::size_t orbitReps = 0;
    std::size_t memberQuartets = 0;

    for (std::size_t pair = 0; pair < table->classPairs.size(); ++pair)
    {
        qcx::integrals::GenerateClassPairOrbits(*table, reduction, *pairList, pair);
        orbitReps += table->classPairs[pair].orbits.size();

        for (const qcx::integrals::ClassOrbit& orbit : table->classPairs[pair].orbits)
        {
            memberQuartets += orbit.members.size();
        }
    }

    // The fixture's own numbers, for the record: water/STO-3G C2v at kTight
    // has 120 screened canonical cells covered by 76 ERI blocks, over
    // 66 kept class pairs (76
    // orbit representatives, 120 member quartets). The assertions below are
    // the relational form of the same statement - they hold on any fixture.
    // The mask's reach on this fixture is zero (the existing test above pins
    // walk.dropped == 0), so the plain build's count IS the surviving cell
    // set and the two equalities below compare like with like.
    RecordProperty("plain_blocks", static_cast<int>(plainStats.fp64QuartetCount));
    RecordProperty("expanded_blocks", static_cast<int>(expandedStats.fp64QuartetCount));
    RecordProperty("machinery_orbit_reps", static_cast<int>(orbitReps));
    RecordProperty("machinery_members", static_cast<int>(memberQuartets));

    EXPECT_EQ(memberQuartets, plainStats.fp64QuartetCount)
        << "the machinery's members must be exactly the plain walk's surviving cells";
    EXPECT_EQ(orbitReps, expandedStats.fp64QuartetCount)
        << "the expanded walk evaluates one block per orbit, and the orbit count is the "
           "machinery's";
    EXPECT_EQ(orbitReps, 76u);
    EXPECT_EQ(memberQuartets, 120u);
    EXPECT_LT(expandedStats.fp64QuartetCount, plainStats.fp64QuartetCount)
        << "the expansion must reduce the ERI blocks evaluated";
    // THE STRONG-FORM ARGUMENT, measured in the counters (the mechanism-value
    // audit, 2026-09-13). The orbit expansion removes ERI EVALUATIONS and
    // removes nothing at all from the ACCUMULATION: `ContractBlock` runs once
    // per assembled class task and once more per surviving member cell, so
    // the call's contract-block count is the plain walk's, cell for cell -
    // the member expansion REPLACES a member's own evaluation, it does not
    // remove its contraction. The audit had to reconstruct that invariance by
    // hand (8,811,481 either way on c8h18/def2-SVP, from a counter that did
    // not exist); this pin is it, read off two live builds. It is also why
    // the ERI wall falls while the contraction span does not.
    RecordProperty("plain_contract_block_calls", static_cast<int>(plainStats.contractBlockCalls));
    RecordProperty("expanded_contract_block_calls",
                   static_cast<int>(expandedStats.contractBlockCalls));
    EXPECT_GT(plainStats.contractBlockCalls, 0u)
        << "the accumulation counter must be live on the plain build";
    EXPECT_EQ(expandedStats.contractBlockCalls, plainStats.contractBlockCalls)
        << "the orbit expansion removes EVALUATIONS, not accumulations: the contract-block "
           "count must be invariant";
    EXPECT_GT(expandedStats.batchCount, 0u);
}

namespace {

// The linear H-O-H fixture's D2h subgroup - the group the production detector
// returns for it - over the same H-first function order the Ci fixture above
// uses ([H1s, H2s, O1s, O2s, Opy, Opz, Opx]). The permutation and sign rows
// are the detected reduction's own, carried verbatim (the integrals module
// cannot include the detection headers). The transcription is VALIDATED by
// the test below rather than trusted: the machinery's decomposition over this
// table must reproduce the fixture's recorded 120 member quartets and 76
// orbit representatives, so a mistyped row fails the anchor assertions there
// instead of quietly moving the measurement.
qcx::integrals::SymmetryReduction MakeLinearHohD2hReduction() {
    qcx::integrals::SymmetryReduction reduction;
    reduction.groupOrder = 8;
    reduction.isTrivial = false;
    reduction.permutation = {
        {0, 1, 2, 3, 4, 5, 6}, // E.
        {1, 0, 2, 3, 4, 5, 6},
        {1, 0, 2, 3, 4, 5, 6},
        {0, 1, 2, 3, 4, 5, 6},
        {1, 0, 2, 3, 4, 5, 6}, // i.
        {0, 1, 2, 3, 4, 5, 6},
        {0, 1, 2, 3, 4, 5, 6},
        {1, 0, 2, 3, 4, 5, 6},
    };
    reduction.sign = {
        {1, 1, 1, 1, 1, 1, 1}, // E.
        {1, 1, 1, 1, -1, -1, 1},
        {1, 1, 1, 1, 1, -1, -1},
        {1, 1, 1, 1, -1, 1, -1},
        {1, 1, 1, 1, -1, -1, -1}, // i: all three p signs flip.
        {1, 1, 1, 1, 1, 1, -1},
        {1, 1, 1, 1, -1, 1, 1},
        {1, 1, 1, 1, 1, -1, 1},
    };
    return reduction;
}

// One canonical pair's provable-zero classification - the test's own
// derivation of the production rule (lean_fock_build.cpp's shellPlus /
// shellMinus aggregation, which the record packs into the pair's two upper
// bytes). Bit g of plus = element g fixes BOTH of the pair's shells
// function-for-function and both carry a uniform +1 sign; bit g of minus is
// the same with a uniform -1. A second implementation, so the invariance
// measurement below is a cross-check rather than a restatement.
struct PairZeroMasks {
    std::uint32_t plus = 0u;
    std::uint32_t minus = 0u;
};

std::vector<PairZeroMasks> DerivePairZeroMasks(const qcx::integrals::SymmetryReduction& reduction,
                                               const qcx::integrals::ShellPairList& pairList) {
    const std::size_t order = reduction.groupOrder;
    const std::size_t nShells = pairList.shells.size();
    std::vector<std::uint32_t> shellPlus(nShells, 0u);
    std::vector<std::uint32_t> shellMinus(nShells, 0u);

    for (std::size_t s = 0; s < nShells; ++s)
    {
        const qcx::integrals::ShellInfo& shell = pairList.shells[s];
        const std::size_t count = qcx::integrals::ShellFunctionCount(shell);

        for (std::size_t g = 0; g < order; ++g)
        {
            bool fixed = true;
            bool plus = true;
            bool minus = true;

            for (std::size_t k = 0; k < count; ++k)
            {
                const std::size_t f = shell.functionOffset + k;
                fixed = fixed && (reduction.permutation[g][f] == f);
                plus = plus && (reduction.sign[g][f] == 1);
                minus = minus && (reduction.sign[g][f] == -1);
            }

            if (fixed && plus)
            {
                shellPlus[s] |= 1u << g;
            }

            if (fixed && minus)
            {
                shellMinus[s] |= 1u << g;
            }
        }
    }

    std::vector<PairZeroMasks> masks(pairList.pairs.size());

    for (std::size_t p = 0; p < pairList.pairs.size(); ++p)
    {
        const std::uint32_t plusI = shellPlus[pairList.pairs[p].i];
        const std::uint32_t plusJ = shellPlus[pairList.pairs[p].j];
        const std::uint32_t minusI = shellMinus[pairList.pairs[p].i];
        const std::uint32_t minusJ = shellMinus[pairList.pairs[p].j];
        masks[p].plus = (plusI & plusJ) | (minusI & minusJ);
        masks[p].minus = (plusI & minusJ) | (minusI & plusJ);
    }

    return masks;
}

// One reduction's measured decomposition of the inversion fixture's cell
// space: what the live lean walk evaluates (engaged and disengaged), what the
// machinery's own class-pair decomposition covers, and the orbit structure
// the test derives itself. The three are MEANT to agree - that agreement is
// the whole content of this test - so each is computed independently and the
// assertions below are equalities between them.
struct OrbitCounts {
    /// The walk's own cell count for this reduction: without a reduction this
    /// is the 8-fold-only count, with one it is the masked count (the walk is
    /// the plain path's until the expansion engages).
    std::size_t walk = 0;
    std::size_t masked = 0; ///< The reduction's provable-zero drop, expansion off.
    std::size_t expanded = 0; ///< The reduction plus the orbit expansion: the ERI blocks.
    std::size_t machineryMembers = 0; ///< The class decomposition's member quartets.
    std::size_t machineryOrbits = 0; ///< Its orbit representatives.
    std::size_t fullOrbits = 0; ///< The test's own BFS over the whole cell space.
    std::size_t survivingOrbits = 0; ///< The test's own BFS over the surviving cells.
    std::size_t survivingCells = 0; ///< The surviving cells the BFS covers.
    std::size_t nonUniform = 0; ///< Cells whose survival differs from an image's.
    std::size_t boundMismatch = 0; ///< Pairs whose Schwarz bound differs from an image's.
    std::size_t zeroMismatch = 0; ///< Cells whose provable-zero verdict differs from an image's.
    std::size_t dropped = 0; ///< The provable-zero cells (the mask's reach).
};

// Measures one reduction end to end. The survival predicate the test uses is
// the walk's own pair of rules - the provable-zero classification first, then
// the Schwarz cutoff - and the BFS below is a SECOND method for the orbit
// count (connected components over the image graph, which is how the
// machinery's BuildOrbits counts) alongside the walk's own minimum test.
OrbitCounts MeasureOrbitCounts(const qcx::molecule::Molecule& molecule,
                               const qcx::basisset::BasisSet& basisSet,
                               const CpuTensor2& core,
                               const qcx::integrals::ShellPairList& pairList,
                               const std::vector<double>& bounds,
                               const qcx::integrals::SymmetryReduction* reduction,
                               const Eigen::MatrixXd& density) {
    OrbitCounts counts;
    const double cutoff = qcx::integrals::SchwarzThreshold(qcx::integrals::AccuracyPreset::kTight) *
                          qcx::integrals::internal::kNeighborListSlack;
    const std::size_t nPairs = pairList.pairs.size();

    auto run = [&](bool expansion, std::size_t* out) {
        qcx::integrals::LeanFockBuildOptions options = LeanSerialOptions();
        options.symmetryReduction = reduction;
        options.symmetryOrbitExpansion = expansion;
        auto builder =
            qcx::integrals::LeanDirectFockBuilder::Create(molecule, basisSet, core, options);

        if (!builder.has_value())
        {
            ADD_FAILURE() << builder.error().message;
            return;
        }

        qcx::integrals::FockBuildStats stats;
        auto fock = builder->BuildFock(*ToTensor(density), &stats);

        if (!fock.has_value())
        {
            ADD_FAILURE() << fock.error().message;
            return;
        }

        *out = stats.fp64QuartetCount;
    };

    run(false, &counts.walk);

    if (reduction == nullptr)
    {
        return counts;
    }

    counts.masked = counts.walk;
    run(true, &counts.expanded);

    // The machinery's decomposition at the same cutoff the walk screens with.
    auto table = qcx::integrals::BuildPairClasses(*reduction, pairList, bounds, cutoff);

    if (!table.has_value())
    {
        ADD_FAILURE() << table.error().message;
        return counts;
    }

    for (std::size_t pair = 0; pair < table->classPairs.size(); ++pair)
    {
        qcx::integrals::GenerateClassPairOrbits(*table, *reduction, pairList, pair);
        counts.machineryOrbits += table->classPairs[pair].orbits.size();

        for (const qcx::integrals::ClassOrbit& orbit : table->classPairs[pair].orbits)
        {
            counts.machineryMembers += orbit.members.size();
        }
    }

    auto action = qcx::integrals::internal::LeanOrbitAction::Create(*reduction, pairList);

    if (!action.has_value())
    {
        ADD_FAILURE() << action.error().message;
        return counts;
    }

    const std::size_t order = action->order;
    const std::vector<PairZeroMasks> masks = DerivePairZeroMasks(*reduction, pairList);

    const auto provablyZero = [&](std::size_t bra, std::size_t ket) {
        return ((masks[bra].plus & masks[ket].minus) | (masks[bra].minus & masks[ket].plus)) != 0u;
    };
    const auto survives = [&](std::size_t bra, std::size_t ket) {
        return !provablyZero(bra, ket) && bounds[bra] * bounds[ket] >= cutoff;
    };
    // (bra, ket, g) is the row-then-column-then-group-element order of the pair-image table.
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    const auto image = [&](std::size_t bra, std::size_t ket, std::size_t g) {
        const std::size_t imageBra = action->pairImage[bra * order + g];
        const std::size_t imageKet = action->pairImage[ket * order + g];
        return std::pair{std::max(imageBra, imageKet), std::min(imageBra, imageKet)};
    };

    // (a) The orbit invariance of the two pruning predicates, MEASURED cell by
    // cell rather than assumed. Group invariance of the Schwarz bound and of
    // the provable-zero verdict is what makes the surviving set a union of
    // orbits, and that - not the order the walk happens to apply the two rules
    // in - is what makes "screen then dedupe" and "dedupe then screen" the
    // same count. A single mismatch here would mean a surviving orbit could
    // hold members the screen killed, and the reduction would then collapse
    // toward the surviving cells.
    for (std::size_t p = 0; p < nPairs; ++p)
    {
        for (std::size_t g = 1; g < order; ++g)
        {
            if (bounds[action->pairImage[p * order + g]] != bounds[p])
            {
                ++counts.boundMismatch;
            }
        }
    }

    for (std::size_t bra = 0; bra < nPairs; ++bra)
    {
        for (std::size_t ket = 0; ket <= bra; ++ket)
        {
            if (provablyZero(bra, ket))
            {
                ++counts.dropped;
            }

            for (std::size_t g = 1; g < order; ++g)
            {
                const auto [imageBra, imageKet] = image(bra, ket, g);

                if (provablyZero(bra, ket) != provablyZero(imageBra, imageKet))
                {
                    ++counts.zeroMismatch;
                }

                if (survives(bra, ket) != survives(imageBra, imageKet))
                {
                    ++counts.nonUniform;
                }
            }
        }
    }

    // (b) The test's own orbit decomposition: BFS over the image graph, which
    // is the method the machinery's BuildOrbits uses (connected components),
    // and the walk's own minimum test, which is what the lean path implements.
    // Both run over the same candidate set so the two counts are comparable.
    const auto countOrbits = [&](bool survivorsOnly) {
        std::vector<bool> assigned(nPairs * (nPairs + 1) / 2, false);
        std::size_t orbits = 0;
        std::size_t members = 0;

        for (std::size_t bra = 0; bra < nPairs; ++bra)
        {
            for (std::size_t ket = 0; ket <= bra; ++ket)
            {
                const std::size_t cell = qcx::integrals::internal::LeanCellIndex(bra, ket);

                if (assigned[cell] || (survivorsOnly && !survives(bra, ket)))
                {
                    continue;
                }

                ++orbits;
                std::vector<std::size_t> frontier{cell};
                assigned[cell] = true;
                std::size_t explored = 0;

                while (explored < frontier.size())
                {
                    const std::size_t current = frontier[explored++];
                    ++members;

                    // Invert LeanCellIndex: the row is the largest r with
                    // r(r+1)/2 <= current.
                    std::size_t row = 0;

                    while ((row + 1) * (row + 2) / 2 <= current)
                    {
                        ++row;
                    }

                    const std::size_t column = current - row * (row + 1) / 2;

                    for (std::size_t g = 1; g < order; ++g)
                    {
                        const auto [imageBra, imageKet] = image(row, column, g);

                        if (survivorsOnly && !survives(imageBra, imageKet))
                        {
                            continue;
                        }

                        const std::size_t imageCell =
                            qcx::integrals::internal::LeanCellIndex(imageBra, imageKet);

                        if (!assigned[imageCell])
                        {
                            assigned[imageCell] = true;
                            frontier.push_back(imageCell);
                        }
                    }
                }
            }
        }

        counts.survivingCells = survivorsOnly ? members : counts.survivingCells;
        return orbits;
    };

    counts.fullOrbits = countOrbits(false);
    counts.survivingOrbits = countOrbits(true);
    return counts;
}

} // namespace

// THE MEASUREMENT this file's orbit half turns on: does the lean walk's
// PER-PAIR orbit map capture the reduction the QUARTET-level machinery
// captures, or only pair equivalence? The same fixture the mask test uses (a
// linear H-O-H whose O sits on the inversion centre - the one shape where
// symmetry has both a zero to prove and an orbit to deduplicate), measured
// ENGAGED against DISENGAGED on the live builder, with the machinery's own
// class-pair decomposition beside it as the benchmark.
//
// Three counts decide it, on both the Ci(2) reduction the file's fixture
// carries and the D2h(8) reduction the production detector returns for the
// same geometry:
//   - the walk without a reduction (the 8-fold-only cell count),
//   - the walk with the reduction's mask only (the provable-zero drop),
//   - the walk with the orbit expansion (the ERI blocks actually evaluated),
// and each is compared with the machinery's member and orbit counts over the
// same screened set. The per-pair map's failure mode is exact and countable:
// its orbit count would sit strictly above the machinery's, because pair
// representatives need not correspond to a valid quartet orbit.
TEST(LeanPointGroupTest, OrbitExpansionCountsOnTheInversionFixtureAndThePruningCountsOnce) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeLinearHoh();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;
    auto bounds = qcx::integrals::ComputeSchwarzBounds(*molecule, *basis);
    ASSERT_TRUE(bounds.has_value()) << bounds.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 20260913);
    const qcx::integrals::SymmetryReduction ci = MakeLinearHohInversionReduction();
    const qcx::integrals::SymmetryReduction d2h = MakeLinearHohD2hReduction();

    const OrbitCounts ciCounts =
        MeasureOrbitCounts(*molecule, *basis, *core, *pairList, *bounds, &ci, density);
    const OrbitCounts d2hCounts =
        MeasureOrbitCounts(*molecule, *basis, *core, *pairList, *bounds, &d2h, density);
    const OrbitCounts plainCounts =
        MeasureOrbitCounts(*molecule, *basis, *core, *pairList, *bounds, nullptr, density);

    // The D2h(8) tables are the anchor: transcribed, so they are checked
    // against the fixture's recorded decomposition before anything leans on
    // them. 120 member quartets in 76 orbits - the numbers this
    // fixture's recorded decomposition carries.
    EXPECT_EQ(d2hCounts.machineryMembers, 120u);
    EXPECT_EQ(d2hCounts.machineryOrbits, 76u);

    // The masked walks agree with each other (the mask is the same drop under
    // both reductions), the machinery's member sum is the UNMASKED cell count
    // - BuildPairClasses carries the Schwarz screen, never the provable-zero
    // classification, which is a separate mechanism on the class path - and
    // the plain walk without a reduction is that same count, cell for cell.
    EXPECT_EQ(plainCounts.walk, 120u) << "the fixture's screened cell count";
    EXPECT_EQ(ciCounts.masked, d2hCounts.masked);
    EXPECT_EQ(d2hCounts.machineryMembers, plainCounts.walk);
    EXPECT_EQ(ciCounts.machineryMembers, plainCounts.walk);

    // The test's own BFS reproduces the machinery's orbit count on the whole
    // cell space - a second method agreeing with the benchmark, so the
    // surviving-set count below rests on a construction that is itself
    // checked. (It also validates the transcribed D2h table: a wrong row
    // changes these orbits.)
    EXPECT_EQ(d2hCounts.fullOrbits, d2hCounts.machineryOrbits);
    EXPECT_EQ(ciCounts.fullOrbits, ciCounts.machineryOrbits);

    // THE EQUALITY UNDER TEST: the live expanded walk evaluates exactly the
    // orbits a full decomposition finds among the SURVIVING cells. This is the
    // per-pair map against the quartet invariant, on the fixture where the
    // mask bites - if the per-pair map captured only pair equivalence, the
    // left side would be strictly larger.
    EXPECT_EQ(d2hCounts.expanded, d2hCounts.survivingOrbits);
    EXPECT_EQ(ciCounts.expanded, ciCounts.survivingOrbits);
    EXPECT_EQ(d2hCounts.masked, d2hCounts.survivingCells)
        << "the masked walk's cells are the surviving set";

    // The pruning predicates are orbit-invariant - measured, not assumed.
    EXPECT_EQ(d2hCounts.nonUniform, 0u)
        << "a surviving orbit holds a member the screen would have killed: the reduction then "
           "collapses to the surviving cells, and the counts above stop being orbit counts";
    EXPECT_EQ(d2hCounts.boundMismatch, 0u)
        << "the Schwarz bound is not group-invariant on this fixture, so the screen and the "
           "orbit reduction do not commute";
    EXPECT_EQ(d2hCounts.zeroMismatch, 0u)
        << "the provable-zero verdict is not group-invariant on this fixture";
    EXPECT_EQ(ciCounts.nonUniform, 0u);
    EXPECT_EQ(ciCounts.boundMismatch, 0u);
    EXPECT_EQ(ciCounts.zeroMismatch, 0u);

    // The mask's reach and the drop's shape: the dropped cells are whole
    // orbits, so the mask takes orbits off the reduction rather than shrinking
    // it - 76 - (the dropped singletons) is the expanded count, not
    // 76 x (a surviving fraction).
    EXPECT_EQ(d2hCounts.dropped, 8u) << "the odd-l-sum cells the classification proves zero";
    EXPECT_GT(d2hCounts.expanded, 0u);
    EXPECT_LT(d2hCounts.expanded, d2hCounts.masked);

    RecordProperty("plain_cells", static_cast<int>(plainCounts.walk));
    RecordProperty("ci_masked_cells", static_cast<int>(ciCounts.masked));
    RecordProperty("ci_expanded_blocks", static_cast<int>(ciCounts.expanded));
    RecordProperty("ci_machinery_orbits", static_cast<int>(ciCounts.machineryOrbits));
    RecordProperty("ci_full_orbits", static_cast<int>(ciCounts.fullOrbits));
    RecordProperty("ci_surviving_orbits", static_cast<int>(ciCounts.survivingOrbits));
    RecordProperty("d2h_masked_cells", static_cast<int>(d2hCounts.masked));
    RecordProperty("d2h_expanded_blocks", static_cast<int>(d2hCounts.expanded));
    RecordProperty("d2h_machinery_orbits", static_cast<int>(d2hCounts.machineryOrbits));
    RecordProperty("d2h_full_orbits", static_cast<int>(d2hCounts.fullOrbits));
    RecordProperty("d2h_surviving_orbits", static_cast<int>(d2hCounts.survivingOrbits));
    RecordProperty("d2h_dropped", static_cast<int>(d2hCounts.dropped));
    RecordProperty("d2h_surviving_cells", static_cast<int>(d2hCounts.survivingCells));
}

// The delta against the plain build, and the k = 1 determinism contract.
// The two builds accumulate the same surviving cells; the expansion's members
// carry the representative's values permuted and sign-flipped, so the Fock
// moves only by the engine's own ordering arithmetic. The assertion is a
// CEILING with headroom (a bit-exact expansion would pass it unchanged) - the
// count assertion above is the leg a no-op mechanism cannot fake.
TEST(LeanPointGroupTest, OrbitExpansionMatchesThePlainBuildToTheLastBits) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();

    qcx::integrals::LeanFockBuildOptions plainOptions = LeanSerialOptions();
    auto plain =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, plainOptions);
    ASSERT_TRUE(plain.has_value()) << plain.error().message;

    qcx::integrals::LeanFockBuildOptions expandedOptions = LeanSerialOptions();
    expandedOptions.symmetryReduction = &reduction;
    expandedOptions.symmetryOrbitExpansion = true;
    auto expanded =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, expandedOptions);
    ASSERT_TRUE(expanded.has_value()) << expanded.error().message;
    auto fresh =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, expandedOptions);
    ASSERT_TRUE(fresh.has_value()) << fresh.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 4242);
    auto plainFock = plain->BuildFock(*ToTensor(density));
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;
    auto expandedFock = expanded->BuildFock(*ToTensor(density));
    ASSERT_TRUE(expandedFock.has_value()) << expandedFock.error().message;
    auto secondFock = expanded->BuildFock(*ToTensor(density));
    ASSERT_TRUE(secondFock.has_value()) << secondFock.error().message;
    auto freshFock = fresh->BuildFock(*ToTensor(density));
    ASSERT_TRUE(freshFock.has_value()) << freshFock.error().message;

    const Eigen::MatrixXd plainMatrix = ToMatrix(*plainFock);
    const Eigen::MatrixXd expandedMatrix = ToMatrix(*expandedFock);
    double maxDelta = 0.0;
    double maxElement = 0.0;
    std::size_t moved = 0;

    std::size_t nonFinitePlain = 0;
    std::size_t nonFiniteExpanded = 0;

    for (Eigen::Index i = 0; i < plainMatrix.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < plainMatrix.cols(); ++j)
        {
            // NaN-propagating maxima and explicit non-finite counting: the
            // max-swallowing form below would report a zero delta over a
            // matrix full of NaN and let the ceiling assertion pass - the
            // false green this run actually met.
            const double expandedValue = expandedMatrix(i, j);
            const double plainValue = plainMatrix(i, j);
            const double delta = std::abs(expandedValue - plainValue);

            if (!(delta <= maxDelta))
            {
                maxDelta = delta;
            }

            if (!(std::abs(plainValue) <= maxElement))
            {
                maxElement = std::abs(plainValue);
            }

            if (!std::isfinite(plainValue))
            {
                ++nonFinitePlain;
            }

            if (!std::isfinite(expandedValue))
            {
                ++nonFiniteExpanded;
            }

            moved += (delta != 0.0) ? 1u : 0u;
        }
    }

    RecordProperty("max_fock_delta", FormatMeasured(maxDelta));
    RecordProperty("max_fock_element", FormatMeasured(maxElement));
    RecordProperty("moved_elements", static_cast<int>(moved));
    RecordProperty("nonfinite_plain", static_cast<int>(nonFinitePlain));
    RecordProperty("nonfinite_expanded", static_cast<int>(nonFiniteExpanded));

    EXPECT_EQ(nonFiniteExpanded, 0u)
        << "the expanded build delivered " << nonFiniteExpanded << " non-finite elements";
    EXPECT_EQ(nonFinitePlain, 0u) << "the plain build delivered " << nonFinitePlain
                                  << " non-finite elements";
    EXPECT_LT(maxDelta, kOrbitExpansionDeltaCeiling)
        << "max |expanded - plain| = " << maxDelta << " over a Fock max of " << maxElement << " ("
        << moved << " of " << plainMatrix.size() << " elements moved)";

    // The k = 1 pin: the expansion's own path is byte-stable across calls and
    // across fresh builders (the determinism contract the plain path has).
    EXPECT_TRUE(BitIdentical(expandedMatrix, ToMatrix(*secondFock)))
        << "a second call on one builder must be byte-identical";
    EXPECT_TRUE(BitIdentical(expandedMatrix, ToMatrix(*freshFock)))
        << "a fresh builder must be byte-identical";
}

// The windowed path keeps the mechanism's counts: the representative test is
// per cell and each window walks its own rows, so the window count is a
// scheduling choice and never a change of which blocks are evaluated. (The
// bytes of a k > 1 call are the windowed reduction's, not the k = 1 pin's -
// the existing lean contract - so this asserts the counts only.)
TEST(LeanPointGroupTest, OrbitExpansionWindowedRunKeepsTheRepresentativeCount) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();

    qcx::integrals::LeanFockBuildOptions serialOptions = LeanSerialOptions();
    serialOptions.symmetryReduction = &reduction;
    serialOptions.symmetryOrbitExpansion = true;
    auto serial =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, serialOptions);
    ASSERT_TRUE(serial.has_value()) << serial.error().message;

    qcx::integrals::LeanFockBuildOptions windowedOptions = serialOptions;
    windowedOptions.maxParallelChunks = 2;
    auto windowed =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, windowedOptions);
    ASSERT_TRUE(windowed.has_value()) << windowed.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 909);
    qcx::integrals::FockBuildStats serialStats;
    auto serialFock = serial->BuildFock(*ToTensor(density), &serialStats);
    ASSERT_TRUE(serialFock.has_value()) << serialFock.error().message;
    qcx::integrals::FockBuildStats windowedStats;
    auto windowedFock = windowed->BuildFock(*ToTensor(density), &windowedStats);
    ASSERT_TRUE(windowedFock.has_value()) << windowedFock.error().message;

    EXPECT_EQ(windowedStats.fp64QuartetCount, serialStats.fp64QuartetCount)
        << "two windows must evaluate the same orbit representatives as one";
    EXPECT_GT(serialStats.fp64QuartetCount, 0u);
}

// The two mechanisms composed on one build: the classification's provable-zero
// drop and the orbit expansion, on the inversion fixture (the one shape where
// the classification has reach at all - it drops 8 of 120 cells). What this
// pins: the walk runs both rules (the representative test's candidate set is
// the SURVIVING cells, so a provably-zero image never wins the minimum), the
// work still drops against the plain FILTERED build, and the Fock still moves
// only within the expansion's own delta. The dropped cells are whole orbits
// here (the zero property propagates through an orbit: A = 0 implies every
// image's block is its own negation), which is why the composed count is the
// filtered count's representatives and not something smaller.
TEST(LeanPointGroupTest, OrbitExpansionComposesWithTheZeroFilter) {
    if (IsFastOnlyMode())
    {
        GTEST_SKIP() << "fast mode";
    }

    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeLinearHoh();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    const qcx::integrals::SymmetryReduction reduction = MakeLinearHohInversionReduction();

    qcx::integrals::LeanFockBuildOptions filteredOptions = LeanSerialOptions();
    filteredOptions.symmetryReduction = &reduction;
    auto filtered =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, filteredOptions);
    ASSERT_TRUE(filtered.has_value()) << filtered.error().message;

    qcx::integrals::LeanFockBuildOptions bothOptions = filteredOptions;
    bothOptions.symmetryOrbitExpansion = true;
    auto both =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, bothOptions);
    ASSERT_TRUE(both.has_value()) << both.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 777);
    qcx::integrals::FockBuildStats filteredStats;
    auto filteredFock = filtered->BuildFock(*ToTensor(density), &filteredStats);
    ASSERT_TRUE(filteredFock.has_value()) << filteredFock.error().message;
    qcx::integrals::FockBuildStats bothStats;
    auto bothFock = both->BuildFock(*ToTensor(density), &bothStats);
    ASSERT_TRUE(bothFock.has_value()) << bothFock.error().message;

    // The filter's reach on this fixture is pinned above (8 cells dropped), so
    // the filtered count is strictly below the unfiltered 120 and the composed
    // count below the filtered one.
    EXPECT_LT(bothStats.fp64QuartetCount, filteredStats.fp64QuartetCount)
        << "the expansion must still drop the non-representatives under the filter";
    EXPECT_GT(bothStats.fp64QuartetCount, 0u);

    const Eigen::MatrixXd filteredMatrix = ToMatrix(*filteredFock);
    const Eigen::MatrixXd bothMatrix = ToMatrix(*bothFock);
    double maxDelta = 0.0;

    for (Eigen::Index i = 0; i < filteredMatrix.rows(); ++i)
    {
        for (Eigen::Index j = 0; j < filteredMatrix.cols(); ++j)
        {
            maxDelta = std::max(maxDelta, std::abs(bothMatrix(i, j) - filteredMatrix(i, j)));
        }
    }

    RecordProperty("filtered_blocks", static_cast<int>(filteredStats.fp64QuartetCount));
    RecordProperty("composed_blocks", static_cast<int>(bothStats.fp64QuartetCount));
    RecordProperty("max_composed_delta", FormatMeasured(maxDelta));

    EXPECT_LT(maxDelta, kOrbitExpansionDeltaCeiling)
        << "max |filtered+expanded - filtered| = " << maxDelta;
}

// The inerts and the refusals. A trivial (C1) reduction with the option on is
// the plain path: one orbit per cell, every cell its own representative, no
// member to expand, the same bytes. The option WITHOUT a reduction is refused
// (the reduction is what defines the orbits), and a reduction that maps one
// shell's functions onto more than one shell is refused by the action's own
// build - the closure the expansion's position arithmetic needs, which the
// classification alone does not (the same reduction is a valid classification
// input, so the two consumers' boundaries differ exactly there).
TEST(LeanPointGroupTest, OrbitExpansionInertsAndRefusals) {
    auto basis = MakeH2oSto3gBasis();
    ASSERT_TRUE(basis.has_value()) << basis.error().message;
    auto molecule = MakeH2oSto3g();
    ASSERT_TRUE(molecule.has_value()) << molecule.error().message;
    auto core = BuildCoreHamiltonian(*molecule, *basis);
    ASSERT_TRUE(core.has_value()) << core.error().message;

    auto pairList = qcx::integrals::BuildShellPairs(*molecule, *basis);
    ASSERT_TRUE(pairList.has_value()) << pairList.error().message;

    const qcx::integrals::SymmetryReduction trivial;
    const qcx::integrals::SymmetryReduction reduction = MakeWaterC2vReductionHFirst();

    // The option without a reduction is a caller error.
    qcx::integrals::LeanFockBuildOptions noReduction = LeanSerialOptions();
    noReduction.symmetryOrbitExpansion = true;
    auto refused =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, noReduction);
    EXPECT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().code, qcx::ErrorCode::kInvalidArgument);

    // The trivial reduction with the option on is the plain path, bit for bit.
    qcx::integrals::LeanFockBuildOptions trivialOptions = LeanSerialOptions();
    trivialOptions.symmetryReduction = &trivial;
    trivialOptions.symmetryOrbitExpansion = true;
    auto trivialBuilder =
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, trivialOptions);
    ASSERT_TRUE(trivialBuilder.has_value()) << trivialBuilder.error().message;

    auto plainBuilder = qcx::integrals::LeanDirectFockBuilder::Create(
        *molecule, *basis, *core, LeanSerialOptions());
    ASSERT_TRUE(plainBuilder.has_value()) << plainBuilder.error().message;

    const Eigen::MatrixXd density = PhysicalDensity(7, 31337);
    qcx::integrals::FockBuildStats trivialStats;
    auto trivialFock = trivialBuilder->BuildFock(*ToTensor(density), &trivialStats);
    ASSERT_TRUE(trivialFock.has_value()) << trivialFock.error().message;
    qcx::integrals::FockBuildStats plainStats;
    auto plainFock = plainBuilder->BuildFock(*ToTensor(density), &plainStats);
    ASSERT_TRUE(plainFock.has_value()) << plainFock.error().message;

    EXPECT_TRUE(BitIdentical(ToMatrix(*trivialFock), ToMatrix(*plainFock)))
        << "a trivial reduction has one orbit per cell: the expansion is the identity";
    EXPECT_EQ(trivialStats.fp64QuartetCount, plainStats.fp64QuartetCount);

    // A reduction that splits a shell: function 2 (O1s) and function 4 (Opy)
    // swap, so the O1s shell's functions land in two different shells. The
    // classification accepts the shape (its fixed-function test never needs
    // the closure), and the orbit action refuses it - the boundary between
    // the two consumers of one reduction, asserted on both sides.
    qcx::integrals::SymmetryReduction shellSplit = reduction;
    shellSplit.permutation[1][2] = 4;
    shellSplit.permutation[1][4] = 2;
    auto splitAction = qcx::integrals::internal::LeanOrbitAction::Create(shellSplit, *pairList);
    EXPECT_FALSE(splitAction.has_value());
    EXPECT_EQ(splitAction.error().code, qcx::ErrorCode::kUnimplemented);

    qcx::integrals::LeanFockBuildOptions splitMaskOptions = LeanSerialOptions();
    splitMaskOptions.symmetryReduction = &shellSplit;
    EXPECT_TRUE(
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, splitMaskOptions)
            .has_value())
        << "the classification alone must still accept the split-shell reduction";

    qcx::integrals::LeanFockBuildOptions splitOptions = LeanSerialOptions();
    splitOptions.symmetryReduction = &shellSplit;
    splitOptions.symmetryOrbitExpansion = true;
    EXPECT_FALSE(
        qcx::integrals::LeanDirectFockBuilder::Create(*molecule, *basis, *core, splitOptions)
            .has_value())
        << "the orbit action must refuse a reduction that splits a shell";
}
