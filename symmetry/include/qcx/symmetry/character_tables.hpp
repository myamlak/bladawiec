#pragma once

#include "qcx/symmetry/point_group.hpp"

#include <array>
#include <string_view>

namespace qcx::symmetry {

/// Kind of a symmetry operation.
/// \ingroup qcx-symmetry
enum class OperationKind {
    kIdentity, ///< E.
    kInversion, ///< i.
    kRotation, ///< A proper Cn rotation; n = order.
    kSigma, ///< A mirror plane.
    kImproper, ///< An improper Sn rotation; n = order (S2n in Schoenflies).
};

/// Abstract operation in a character table column: kind plus the axis letter
/// it refers to. For rotations and improper rotations the axis is the
/// rotation axis; for a mirror it is the plane normal ('z' = the xy plane).
/// Identity and inversion use '\0'. Concrete axes are attached at detection
/// time.
/// \ingroup qcx-symmetry
struct OperationTemplate {
    OperationKind kind; ///< What the operation is.
    int order; ///< n of Cn/Sn; 1 for identity, inversion, and mirrors.
    char axis; ///< 'x', 'y', 'z', or '\0' when axis-free.
};

/// One character table: irreps, characters [irrep][operation] (entries beyond
/// order unused), and the ordered operations matching the column order.
/// \ingroup qcx-symmetry
struct CharacterTable {
    std::array<std::string_view, 8> irrepLabels; ///< Mulliken labels.
    std::array<std::array<int, 8>, 8> characters; ///< [irrep][operation].
    std::array<OperationTemplate, 8> operations; ///< Column order.
    int order; ///< Group order = number of irreps.
};

inline constexpr CharacterTable kC1Table{
    {{"A"}}, {{{1}}}, {{OperationTemplate{OperationKind::kIdentity, 1, '\0'}}}, 1};

inline constexpr CharacterTable kCiTable{{{"Ag", "Au"}},
                                         {{{1, 1}, {1, -1}}},
                                         {{OperationTemplate{OperationKind::kIdentity, 1, '\0'},
                                           OperationTemplate{OperationKind::kInversion, 1, '\0'}}},
                                         2};

inline constexpr CharacterTable kC2Table{{{"A", "B"}},
                                         {{{1, 1}, {1, -1}}},
                                         {{OperationTemplate{OperationKind::kIdentity, 1, '\0'},
                                           OperationTemplate{OperationKind::kRotation, 2, 'z'}}},
                                         2};

inline constexpr CharacterTable kCsTable{{{"A'", "A''"}},
                                         {{{1, 1}, {1, -1}}},
                                         {{OperationTemplate{OperationKind::kIdentity, 1, '\0'},
                                           OperationTemplate{OperationKind::kSigma, 1, 'z'}}},
                                         2};

inline constexpr CharacterTable kC2hTable{
    {{"Ag", "Bg", "Au", "Bu"}},
    {{{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}}},
    {{OperationTemplate{OperationKind::kIdentity, 1, '\0'},
      OperationTemplate{OperationKind::kRotation, 2, 'z'},
      OperationTemplate{OperationKind::kInversion, 1, '\0'},
      OperationTemplate{OperationKind::kSigma, 1, 'z'}}},
    4};

inline constexpr CharacterTable kD2Table{
    {{"A", "B1", "B2", "B3"}},
    {{{1, 1, 1, 1}, {1, 1, -1, -1}, {1, -1, 1, -1}, {1, -1, -1, 1}}},
    {{OperationTemplate{OperationKind::kIdentity, 1, '\0'},
      OperationTemplate{OperationKind::kRotation, 2, 'z'},
      OperationTemplate{OperationKind::kRotation, 2, 'y'},
      OperationTemplate{OperationKind::kRotation, 2, 'x'}}},
    4};

inline constexpr CharacterTable kC2vTable{
    {{"A1", "A2", "B1", "B2"}},
    {{{1, 1, 1, 1}, {1, 1, -1, -1}, {1, -1, 1, -1}, {1, -1, -1, 1}}},
    {{OperationTemplate{OperationKind::kIdentity, 1, '\0'},
      OperationTemplate{OperationKind::kRotation, 2, 'z'},
      OperationTemplate{OperationKind::kSigma, 1, 'y'},
      OperationTemplate{OperationKind::kSigma, 1, 'x'}}},
    4};

inline constexpr CharacterTable kD2hTable{{{"Ag", "B1g", "B2g", "B3g", "Au", "B1u", "B2u", "B3u"}},
                                          {{{1, 1, 1, 1, 1, 1, 1, 1},
                                            {1, 1, -1, -1, 1, 1, -1, -1},
                                            {1, -1, 1, -1, 1, -1, 1, -1},
                                            {1, -1, -1, 1, 1, -1, -1, 1},
                                            {1, 1, 1, 1, -1, -1, -1, -1},
                                            {1, 1, -1, -1, -1, -1, 1, 1},
                                            {1, -1, 1, -1, -1, 1, -1, 1},
                                            {1, -1, -1, 1, -1, 1, 1, -1}}},
                                          {{OperationTemplate{OperationKind::kIdentity, 1, '\0'},
                                            OperationTemplate{OperationKind::kRotation, 2, 'z'},
                                            OperationTemplate{OperationKind::kRotation, 2, 'y'},
                                            OperationTemplate{OperationKind::kRotation, 2, 'x'},
                                            OperationTemplate{OperationKind::kInversion, 1, '\0'},
                                            OperationTemplate{OperationKind::kSigma, 1, 'z'},
                                            OperationTemplate{OperationKind::kSigma, 1, 'y'},
                                            OperationTemplate{OperationKind::kSigma, 1, 'x'}}},
                                          8};

/// The character table of one of the eight Abelian computational groups.
constexpr const CharacterTable& CharacterTableFor(PointGroup group) noexcept {
    switch (group)
    {
    case PointGroup::kC1:
        return kC1Table;
    case PointGroup::kCi:
        return kCiTable;
    case PointGroup::kC2:
        return kC2Table;
    case PointGroup::kCs:
        return kCsTable;
    case PointGroup::kC2h:
        return kC2hTable;
    case PointGroup::kD2:
        return kD2Table;
    case PointGroup::kC2v:
        return kC2vTable;
    case PointGroup::kD2h:
        return kD2hTable;
    }

    return kC1Table; // Unreachable; silences -Wreturn-type.
}

} // namespace qcx::symmetry
