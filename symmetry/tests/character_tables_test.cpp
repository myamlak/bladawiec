// Character tables, name parsing, and the Abelian reduction table.
#include "qcx/symmetry/character_tables.hpp"
#include "qcx/symmetry/point_group.hpp"
#include "qcx/symmetry/point_group_name.hpp"

#include <gtest/gtest.h>

namespace qcx::symmetry {
namespace {

TEST(CharacterTablesTest, RowsAreOrthogonal) {
    // sum_op chi_i(op) * chi_j(op) = order * delta_ij - one property that
    // verifies all eight tables at once.

    for (const PointGroup group : {PointGroup::kC1,
                                   PointGroup::kCi,
                                   PointGroup::kC2,
                                   PointGroup::kCs,
                                   PointGroup::kC2h,
                                   PointGroup::kD2,
                                   PointGroup::kC2v,
                                   PointGroup::kD2h})
    {
        SCOPED_TRACE(ToString(group));
        const CharacterTable& table = CharacterTableFor(group);
        EXPECT_GE(table.order, 1);
        EXPECT_LE(table.order, 8);

        for (int i = 0; i < table.order; ++i)
        {
            for (int j = 0; j < table.order; ++j)
            {
                int dot = 0;

                for (int op = 0; op < table.order; ++op)
                {
                    dot += table.characters[i][op] * table.characters[j][op];
                }

                EXPECT_EQ(dot, (i == j) ? table.order : 0);
            }
        }
    }
}

TEST(CharacterTablesTest, D2hColumnsMatchOperations) {
    const CharacterTable& table = CharacterTableFor(PointGroup::kD2h);
    EXPECT_EQ(table.order, 8);
    EXPECT_EQ(table.irrepLabels[0], "Ag");
    EXPECT_EQ(table.irrepLabels[7], "B3u");
    int identityCount = 0;
    int inversionCount = 0;
    int c2Count = 0;
    int mirrorCount = 0;

    for (int op = 0; op < table.order; ++op)
    {
        switch (table.operations[op].kind)
        {
        case OperationKind::kIdentity:
            ++identityCount;
            break;
        case OperationKind::kInversion:
            ++inversionCount;
            break;
        case OperationKind::kRotation:
            ++c2Count;
            break;
        case OperationKind::kSigma:
            ++mirrorCount;
            break;
        case OperationKind::kImproper:
            break;
        }
    }

    EXPECT_EQ(identityCount, 1);
    EXPECT_EQ(inversionCount, 1);
    EXPECT_EQ(c2Count, 3);
    EXPECT_EQ(mirrorCount, 3);
    // Inversion is its own column and flips every gerade character.

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(table.characters[i][4], 1);
        EXPECT_EQ(table.characters[i + 4][4], -1);
    }
}

TEST(CharacterTablesTest, ComputationalGroupNamesRoundTrip) {
    for (const PointGroup group : {PointGroup::kC1,
                                   PointGroup::kCi,
                                   PointGroup::kC2,
                                   PointGroup::kCs,
                                   PointGroup::kC2h,
                                   PointGroup::kD2,
                                   PointGroup::kC2v,
                                   PointGroup::kD2h})
    {
        const auto* parsed = ParsePointGroupName(ToString(group));
        ASSERT_NE(parsed, nullptr);
        // C2h/C2v/D2h map 1:1; the general enum reuses the same names.
        EXPECT_EQ(ToString(*parsed), ToString(group));
    }
}

TEST(CharacterTablesTest, GeneralNamesParseAndReduce) {
    struct Expected {
        const char* name;
        PointGroup reduction;
    };

    const Expected kCases[] = {
        {"C1", PointGroup::kC1},   {"Ci", PointGroup::kCi},     {"Cs", PointGroup::kCs},
        {"C2", PointGroup::kC2},   {"C2v", PointGroup::kC2v},   {"C2h", PointGroup::kC2h},
        {"D2", PointGroup::kD2},   {"D2h", PointGroup::kD2h},   {"C3", PointGroup::kC1},
        {"C4", PointGroup::kC2},   {"C5", PointGroup::kC1},     {"C6", PointGroup::kC2},
        {"C7", PointGroup::kC1},   {"C8", PointGroup::kC2},     {"C3v", PointGroup::kCs},
        {"C4v", PointGroup::kC2v}, {"C5v", PointGroup::kCs},    {"C6v", PointGroup::kC2v},
        {"C7v", PointGroup::kCs},  {"C8v", PointGroup::kC2v},   {"C3h", PointGroup::kCs},
        {"C4h", PointGroup::kC2h}, {"C5h", PointGroup::kCs},    {"C6h", PointGroup::kC2h},
        {"C7h", PointGroup::kCs},  {"C8h", PointGroup::kC2h},   {"D3", PointGroup::kC2},
        {"D4", PointGroup::kD2},   {"D5", PointGroup::kC2},     {"D6", PointGroup::kD2},
        {"D7", PointGroup::kC2},   {"D8", PointGroup::kD2},     {"D3h", PointGroup::kC2v},
        {"D4h", PointGroup::kD2h}, {"D5h", PointGroup::kC2v},   {"D6h", PointGroup::kD2h},
        {"D7h", PointGroup::kC2v}, {"D8h", PointGroup::kD2h},   {"D2d", PointGroup::kD2},
        {"D3d", PointGroup::kC2h}, {"D4d", PointGroup::kD2},    {"D5d", PointGroup::kC2h},
        {"D6d", PointGroup::kD2},  {"D7d", PointGroup::kC2h},   {"D8d", PointGroup::kD2},
        {"S4", PointGroup::kC2},   {"S6", PointGroup::kCi},     {"S8", PointGroup::kC2},
        {"T", PointGroup::kD2},    {"Td", PointGroup::kD2h},    {"Th", PointGroup::kD2h},
        {"O", PointGroup::kD2},    {"Oh", PointGroup::kD2h},    {"I", PointGroup::kD2},
        {"Ih", PointGroup::kD2h},  {"Cinfv", PointGroup::kC2v}, {"Dinfh", PointGroup::kD2h},
    };

    for (const auto& testCase : kCases)
    {
        SCOPED_TRACE(testCase.name);
        const auto* parsed = ParsePointGroupName(testCase.name);
        ASSERT_NE(parsed, nullptr);
        EXPECT_EQ(LargestAbelianSubgroup(*parsed), testCase.reduction);
    }

    EXPECT_EQ(ParsePointGroupName("C42"), nullptr);
    EXPECT_EQ(ParsePointGroupName("c2v"), nullptr);
}

TEST(CharacterTablesTest, EveryNameParsesRoundTripsAndReduces) {
    // Exhaustive sweep over the full name table: parse, ToString round-trip,
    // reduction lands in the eight computational groups, and the reduction
    // is idempotent. Exact reduction values are pinned by the table above
    // (one representative per n-parity/family); this guards the plumbing
    // for all 55 names.
    for (const auto& [name, group] : detail::kPointGroupNames)
    {
        SCOPED_TRACE(name);
        const auto* parsed = ParsePointGroupName(name);
        ASSERT_NE(parsed, nullptr);
        EXPECT_EQ(*parsed, group);
        EXPECT_EQ(ToString(*parsed), name);
        const PointGroup reduced = LargestAbelianSubgroup(*parsed);
        const std::string_view reducedName = ToString(reduced);
        const auto* reducedGeneral = ParsePointGroupName(reducedName);
        ASSERT_NE(reducedGeneral, nullptr);
        EXPECT_EQ(LargestAbelianSubgroup(*reducedGeneral), reduced);
    }
}

TEST(CharacterTablesTest, UnicodeNamesParseBothWays) {
    // The table stores the Unicode symbols; the ASCII aliases must land on
    // the same groups, and both spellings round-trip through ToString.
    const auto* cInfVUnicode = ParsePointGroupName("C∞v");
    const auto* cInfVAscii = ParsePointGroupName("Cinfv");
    ASSERT_NE(cInfVUnicode, nullptr);
    ASSERT_NE(cInfVAscii, nullptr);
    EXPECT_EQ(*cInfVUnicode, PointGroupName::kCInfV);
    EXPECT_EQ(*cInfVAscii, PointGroupName::kCInfV);
    EXPECT_EQ(ToString(*cInfVUnicode), "C∞v");

    const auto* dInfHUnicode = ParsePointGroupName("D∞h");
    const auto* dInfHAscii = ParsePointGroupName("Dinfh");
    ASSERT_NE(dInfHUnicode, nullptr);
    ASSERT_NE(dInfHAscii, nullptr);
    EXPECT_EQ(*dInfHUnicode, PointGroupName::kDInfH);
    EXPECT_EQ(*dInfHAscii, PointGroupName::kDInfH);
    EXPECT_EQ(ToString(*dInfHUnicode), "D∞h");
}

} // namespace
} // namespace qcx::symmetry
