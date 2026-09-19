// Full-group character tables: dimensionality sums,
// row/column orthogonality, class sanity, correlation-table consistency,
// and the Abelian-correspondence gate for the 8 Abelian groups.
//
// The tables are real-form: an irreducible complex-conjugate pair (C3's E,
// C5v's E1/E2, ...) is stored once as its 2-dim realification. Such rows
// carry the 2|G| orthogonality diagonal and count as dimension 2, so the
// standard identities take the per-row factor nu_i (1 for real-type rows,
// 2 for complex-pair rows), recovered from the diagonal row dot:
//   sum_c |C_c| chi_i(c) chi_j(c) = |G| delta_ij nu_i
//   sum_i dim_i chi_i(d) / nu_i = |G| delta_d0 (identity-column form)
//   sum_i dim_i^2 / nu_i = |G|
//   sum_i corr[i][a] dim_i / nu_i = |G|/|H|
#include "qcx/symmetry/character_tables.hpp"
#include "qcx/symmetry/full_group_tables.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <cmath>
#include <gtest/gtest.h>

namespace qcx::symmetry {
namespace {

// Every group with a finite character table; C-inf-v / D-inf-h have none.
constexpr std::array kFiniteGroups = {
    PointGroupName::kC1,  PointGroupName::kCi,  PointGroupName::kCs,  PointGroupName::kC2,
    PointGroupName::kC3,  PointGroupName::kC4,  PointGroupName::kC5,  PointGroupName::kC6,
    PointGroupName::kC7,  PointGroupName::kC8,  PointGroupName::kC2v, PointGroupName::kC3v,
    PointGroupName::kC4v, PointGroupName::kC5v, PointGroupName::kC6v, PointGroupName::kC7v,
    PointGroupName::kC8v, PointGroupName::kC2h, PointGroupName::kC3h, PointGroupName::kC4h,
    PointGroupName::kC5h, PointGroupName::kC6h, PointGroupName::kC7h, PointGroupName::kC8h,
    PointGroupName::kD2,  PointGroupName::kD3,  PointGroupName::kD4,  PointGroupName::kD5,
    PointGroupName::kD6,  PointGroupName::kD7,  PointGroupName::kD8,  PointGroupName::kD2h,
    PointGroupName::kD3h, PointGroupName::kD4h, PointGroupName::kD5h, PointGroupName::kD6h,
    PointGroupName::kD7h, PointGroupName::kD8h, PointGroupName::kD2d, PointGroupName::kD3d,
    PointGroupName::kD4d, PointGroupName::kD5d, PointGroupName::kD6d, PointGroupName::kD7d,
    PointGroupName::kD8d, PointGroupName::kS4,  PointGroupName::kS6,  PointGroupName::kS8,
    PointGroupName::kT,   PointGroupName::kTd,  PointGroupName::kTh,  PointGroupName::kO,
    PointGroupName::kOh,  PointGroupName::kI,   PointGroupName::kIh,
};

/// The ClassAxis -> axis letter map used by the runtime stage.
char AxisLetter(ClassAxis axis) {
    switch (axis)
    {
    case ClassAxis::kNone:
        return '\0';
    case ClassAxis::kPrincipal:
        return 'z';
    case ClassAxis::kOrbit0:
        return 'y';
    case ClassAxis::kOrbit1:
        return 'x';
    case ClassAxis::kOrbit2:
        return '?';
    }

    return '\0';
}

TEST(FullGroupTablesTest, EveryNameHasATable) {
    for (const PointGroupName group : kFiniteGroups)
    {
        SCOPED_TRACE(ToString(group));
        EXPECT_NE(FullGroupTableFor(group), nullptr);
    }

    // The linear-molecule groups have no finite character table.
    EXPECT_EQ(FullGroupTableFor(PointGroupName::kCInfV), nullptr);
    EXPECT_EQ(FullGroupTableFor(PointGroupName::kDInfH), nullptr);
}

TEST(FullGroupTablesTest, ClassSizesAndIdentityColumn) {
    for (const PointGroupName group : kFiniteGroups)
    {
        SCOPED_TRACE(ToString(group));
        const FullGroupTable& table = *FullGroupTableFor(group);
        EXPECT_GT(table.order, 0);
        EXPECT_GT(table.irrepCount, 0);
        EXPECT_GT(table.classCount, 0);
        EXPECT_EQ(table.classes[0].kind, OperationKind::kIdentity);
        EXPECT_EQ(table.classSizes[0], 1);

        int sizeSum = 0;
        double dimSqSum = 0.0;

        for (int c = 0; c < table.classCount; ++c)
        {
            sizeSum += table.classSizes[c];
        }

        for (int i = 0; i < table.irrepCount; ++i)
        {
            // The complex-pair rows (realifications) carry the 2|G|
            // orthogonality diagonal and count double in the real dim-sum.
            double rowDot = 0.0;

            for (int c = 0; c < table.classCount; ++c)
            {
                rowDot += table.classSizes[c] * table.characters[i][c] * table.characters[i][c];
            }

            const double nu = rowDot / table.order;
            ASSERT_TRUE(std::abs(nu - 1.0) < 1e-9 || std::abs(nu - 2.0) < 1e-9)
                << "row " << i << " has a " << nu << " orthogonality diagonal";
            dimSqSum += table.dimensions[i] * table.dimensions[i] / nu;
            // chi(E) = dim for every irrep.
            EXPECT_DOUBLE_EQ(table.characters[i][0], static_cast<double>(table.dimensions[i]));
            // Labels are distinct and non-empty.
            EXPECT_FALSE(table.irrepLabels[i].empty());

            for (int j = 0; j < i; ++j)
            {
                EXPECT_NE(table.irrepLabels[i], table.irrepLabels[j]);
            }
        }

        EXPECT_EQ(sizeSum, table.order);
        // sum_alpha dim_alpha^2 / nu_alpha = |G| (complex-orthonormal count).
        EXPECT_NEAR(dimSqSum, table.order, 1e-6);
    }
}

TEST(FullGroupTablesTest, RowsAndColumnsAreOrthogonal) {
    for (const PointGroupName group : kFiniteGroups)
    {
        SCOPED_TRACE(ToString(group));
        const FullGroupTable& table = *FullGroupTableFor(group);

        // nu_i from the diagonal row dot, before any other use.
        std::array<int, 16> nu{};

        for (int i = 0; i < table.irrepCount; ++i)
        {
            double rowDot = 0.0;

            for (int c = 0; c < table.classCount; ++c)
            {
                rowDot += table.classSizes[c] * table.characters[i][c] * table.characters[i][c];
            }

            const double nuD = rowDot / table.order;
            ASSERT_TRUE(std::abs(nuD - 1.0) < 1e-9 || std::abs(nuD - 2.0) < 1e-9)
                << "row " << i << " has a " << nuD << " orthogonality diagonal";
            nu[i] = static_cast<int>(std::lround(nuD));
        }

        // Rows: sum_c |C_c| chi_i(c) chi_j(c) = |G| delta_ij nu_i.
        for (int i = 0; i < table.irrepCount; ++i)
        {
            for (int j = 0; j < table.irrepCount; ++j)
            {
                double dot = 0.0;

                for (int c = 0; c < table.classCount; ++c)
                {
                    dot += table.classSizes[c] * table.characters[i][c] * table.characters[j][c];
                }

                EXPECT_NEAR(dot, (i == j) ? table.order * nu[i] : 0.0, 1e-9);
            }
        }

        // Identity-column completeness: sum_i dim_i chi_i(d) / nu_i = |G|
        // delta_d0. (The full column orthogonality does not survive the
        // realification: complex-pair rows carry (2 Re chi)^2 instead of the
        // pair's |chi|^2 + |chi|^2, so the general (c, d) column identity
        // only holds for the identity column, where chi_i(E) = dim_i.)
        for (int d = 0; d < table.classCount; ++d)
        {
            double dot = 0.0;

            for (int i = 0; i < table.irrepCount; ++i)
            {
                dot += table.dimensions[i] * table.characters[i][d] / nu[i];
            }

            EXPECT_NEAR(dot, (d == 0) ? table.order : 0.0, 1e-9);
        }
    }
}

TEST(FullGroupTablesTest, CorrelationIsAValidSubduction) {
    for (const PointGroupName group : kFiniteGroups)
    {
        SCOPED_TRACE(ToString(group));
        const FullGroupTable& table = *FullGroupTableFor(group);
        const CharacterTable& abelian = CharacterTableFor(table.abelian);

        // Row check: every irrep subduces to a combination whose total
        // dimension equals dim_Gamma (the realified pair rows subduce with
        // doubled multiplicities, so the unweighted sum still equals the
        // real dimension).
        for (int i = 0; i < table.irrepCount; ++i)
        {
            int subducedDim = 0;

            for (int a = 0; a < abelian.order; ++a)
            {
                subducedDim += table.correlation[i][a] * static_cast<int>(abelian.characters[a][0]);
            }

            EXPECT_EQ(subducedDim, table.dimensions[i]);
        }

        // Column check: the full regular representation restricted to the
        // subgroup is |G| / |H| copies of the Abelian regular rep, with the
        // complex-pair rows counted at half weight (nu_i, as above).
        std::array<int, 16> nu{};

        for (int i = 0; i < table.irrepCount; ++i)
        {
            double rowDot = 0.0;

            for (int c = 0; c < table.classCount; ++c)
            {
                rowDot += table.classSizes[c] * table.characters[i][c] * table.characters[i][c];
            }

            const double nuD = rowDot / table.order;
            ASSERT_TRUE(std::abs(nuD - 1.0) < 1e-9 || std::abs(nuD - 2.0) < 1e-9)
                << "row " << i << " has a " << nuD << " orthogonality diagonal";
            nu[i] = static_cast<int>(std::lround(nuD));
        }

        for (int a = 0; a < abelian.order; ++a)
        {
            double multiplicitySum = 0.0;

            for (int i = 0; i < table.irrepCount; ++i)
            {
                multiplicitySum +=
                    static_cast<double>(table.correlation[i][a] * table.dimensions[i]) / nu[i];
            }

            EXPECT_NEAR(multiplicitySum, static_cast<double>(table.order) / abelian.order, 1e-6);
        }
    }
}

TEST(FullGroupTablesTest, ClassKeysAreClassMinimal) {
    // The emitted class power is the class-minimal member power (min j over
    // the conjugacy class), the canonical key the runtime stage recomputes
    // from molecule-frame elements. The generator used to emit the arbitrary
    // class representative's power (D5d S10 classes (10,3),(10,9); Ih
    // (10,7),(10,9)), which no molecule-frame class can match. A class of
    // S10^k, S10^(n-k) members carries min(k, n-k); the Abelian groups
    // (C5h, C8h, S8) have singleton classes and keep every power.
    const auto powersOf = [](const FullGroupTable& table, OperationKind kind, int order) {
        std::vector<int> powers;

        for (int c = 0; c < table.classCount; ++c)
        {
            if (table.classes[c].kind == kind && table.classes[c].order == order)
            {
                powers.push_back(table.classes[c].power);
            }
        }

        std::sort(powers.begin(), powers.end());
        return powers;
    };

    const std::array kCases = {
        std::tuple{PointGroupName::kD4d, OperationKind::kImproper, 8, std::vector{1, 3}},
        std::tuple{PointGroupName::kD5d, OperationKind::kImproper, 10, std::vector{1, 3}},
        std::tuple{PointGroupName::kD5h, OperationKind::kImproper, 10, std::vector{2, 4}},
        std::tuple{PointGroupName::kIh, OperationKind::kImproper, 10, std::vector{1, 3}},
        std::tuple{PointGroupName::kS8, OperationKind::kImproper, 8, std::vector{1, 3, 5, 7}},
        std::tuple{PointGroupName::kC5h, OperationKind::kImproper, 10, std::vector{2, 4, 6, 8}},
        std::tuple{PointGroupName::kC8h, OperationKind::kImproper, 8, std::vector{1, 3, 5, 7}},
    };

    for (const auto& [group, kind, order, expected] : kCases)
    {
        SCOPED_TRACE(ToString(group));
        const FullGroupTable& table = *FullGroupTableFor(group);
        EXPECT_EQ(powersOf(table, kind, order), expected);
    }
}

TEST(FullGroupTablesTest, AbelianGroupsMatchCharacterTableFor) {
    // The 8 Abelian tables are shared: the full-group table for one of them
    // must be the same table in the extended shape - same classes (kind,
    // order, axis letter), same characters, same labels, identity
    // correlation.
    const std::array kAbelian = {
        std::pair{PointGroupName::kC1, PointGroup::kC1},
        std::pair{PointGroupName::kCi, PointGroup::kCi},
        std::pair{PointGroupName::kCs, PointGroup::kCs},
        std::pair{PointGroupName::kC2, PointGroup::kC2},
        std::pair{PointGroupName::kC2h, PointGroup::kC2h},
        std::pair{PointGroupName::kD2, PointGroup::kD2},
        std::pair{PointGroupName::kC2v, PointGroup::kC2v},
        std::pair{PointGroupName::kD2h, PointGroup::kD2h},
    };

    for (const auto& [name, abelian] : kAbelian)
    {
        SCOPED_TRACE(ToString(name));
        const FullGroupTable& table = *FullGroupTableFor(name);
        const CharacterTable& reference = CharacterTableFor(abelian);
        EXPECT_EQ(table.abelian, abelian);
        EXPECT_EQ(table.order, reference.order);
        EXPECT_EQ(table.classCount, reference.order);
        EXPECT_EQ(table.irrepCount, reference.order);

        // Classes: same (kind, order, axis) as the Abelian operations.
        for (int c = 0; c < reference.order; ++c)
        {
            EXPECT_EQ(table.classes[c].kind, reference.operations[c].kind);
            EXPECT_EQ(table.classes[c].order, reference.operations[c].order);
            EXPECT_EQ(AxisLetter(table.classes[c].axis), reference.operations[c].axis);
        }

        // Characters and labels: match row-wise by label (the full-group
        // rows are lexicographically sorted, the Abelian rows are not).
        for (int i = 0; i < table.irrepCount; ++i)
        {
            const std::string_view label = table.irrepLabels[i];
            int refRow = -1;

            for (int r = 0; r < reference.order; ++r)
            {
                if (reference.irrepLabels[r] == label)
                {
                    refRow = r;
                    break;
                }
            }

            ASSERT_GE(refRow, 0) << "no Abelian row labelled " << label;

            for (int c = 0; c < reference.order; ++c)
            {
                EXPECT_DOUBLE_EQ(table.characters[i][c],
                                 static_cast<double>(reference.characters[refRow][c]));
            }

            // Correlation: each full irrep maps to exactly its own row.
            for (int a = 0; a < reference.order; ++a)
            {
                EXPECT_EQ(table.correlation[i][a], (a == refRow) ? 1 : 0);
            }
        }
    }
}

} // namespace
} // namespace qcx::symmetry
