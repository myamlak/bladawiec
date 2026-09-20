#pragma once

#include "qcx/symmetry/point_group.hpp"

#include <array>
#include <string_view>
#include <utility>

namespace qcx::symmetry {

/// The full set of point groups the detector can name (~57, psi4-grade).
///
/// Only the eight Abelian groups (PointGroup) are used for computation;
/// everything else is detected in full and then reduced via
/// LargestAbelianSubgroup.
/// \ingroup qcx-symmetry
enum class PointGroupName {
    kC1, ///< No symmetry elements.
    kCi, ///< Inversion only.
    kCs, ///< One mirror plane.
    kC2, ///< One Cn axis, n = 2.
    kC3,
    kC4,
    kC5,
    kC6,
    kC7,
    kC8,
    kC2v, ///< Cn + n mirror planes containing it, n = 2.
    kC3v,
    kC4v,
    kC5v,
    kC6v,
    kC7v,
    kC8v,
    kC2h, ///< Cn + a perpendicular mirror, n = 2.
    kC3h,
    kC4h,
    kC5h,
    kC6h,
    kC7h,
    kC8h,
    kD2, ///< Cn + n perpendicular C2 axes, n = 2.
    kD3,
    kD4,
    kD5,
    kD6,
    kD7,
    kD8,
    kD2h, ///< Dn + a perpendicular mirror, n = 2.
    kD3h,
    kD4h,
    kD5h,
    kD6h,
    kD7h,
    kD8h,
    kD2d, ///< Dn + n diagonal mirrors, n = 2.
    kD3d,
    kD4d,
    kD5d,
    kD6d,
    kD7d,
    kD8d,
    kS4, ///< Improper S2n axis only, 2n = 4.
    kS6,
    kS8,
    kT, ///< Tetrahedral rotation group.
    kTd, ///< Full tetrahedral group.
    kTh, ///< Tetrahedral group with inversion.
    kO, ///< Octahedral rotation group.
    kOh, ///< Full octahedral group.
    kI, ///< Icosahedral rotation group.
    kIh, ///< Full icosahedral group.
    kCInfV, ///< Linear molecule with mirror planes (C infinity v).
    kDInfH, ///< Linear molecule with inversion (D infinity h).
};

/// Schoenflies symbol of the group ("C1".."D∞h"); ASCII "Cinfv"/"Dinfh"
/// aliases are accepted by Parse but never produced.
constexpr std::string_view ToString(PointGroupName group) noexcept {
    switch (group)
    {
    case PointGroupName::kC1:
        return "C1";
    case PointGroupName::kCi:
        return "Ci";
    case PointGroupName::kCs:
        return "Cs";
    case PointGroupName::kC2:
        return "C2";
    case PointGroupName::kC3:
        return "C3";
    case PointGroupName::kC4:
        return "C4";
    case PointGroupName::kC5:
        return "C5";
    case PointGroupName::kC6:
        return "C6";
    case PointGroupName::kC7:
        return "C7";
    case PointGroupName::kC8:
        return "C8";
    case PointGroupName::kC2v:
        return "C2v";
    case PointGroupName::kC3v:
        return "C3v";
    case PointGroupName::kC4v:
        return "C4v";
    case PointGroupName::kC5v:
        return "C5v";
    case PointGroupName::kC6v:
        return "C6v";
    case PointGroupName::kC7v:
        return "C7v";
    case PointGroupName::kC8v:
        return "C8v";
    case PointGroupName::kC2h:
        return "C2h";
    case PointGroupName::kC3h:
        return "C3h";
    case PointGroupName::kC4h:
        return "C4h";
    case PointGroupName::kC5h:
        return "C5h";
    case PointGroupName::kC6h:
        return "C6h";
    case PointGroupName::kC7h:
        return "C7h";
    case PointGroupName::kC8h:
        return "C8h";
    case PointGroupName::kD2:
        return "D2";
    case PointGroupName::kD3:
        return "D3";
    case PointGroupName::kD4:
        return "D4";
    case PointGroupName::kD5:
        return "D5";
    case PointGroupName::kD6:
        return "D6";
    case PointGroupName::kD7:
        return "D7";
    case PointGroupName::kD8:
        return "D8";
    case PointGroupName::kD2h:
        return "D2h";
    case PointGroupName::kD3h:
        return "D3h";
    case PointGroupName::kD4h:
        return "D4h";
    case PointGroupName::kD5h:
        return "D5h";
    case PointGroupName::kD6h:
        return "D6h";
    case PointGroupName::kD7h:
        return "D7h";
    case PointGroupName::kD8h:
        return "D8h";
    case PointGroupName::kD2d:
        return "D2d";
    case PointGroupName::kD3d:
        return "D3d";
    case PointGroupName::kD4d:
        return "D4d";
    case PointGroupName::kD5d:
        return "D5d";
    case PointGroupName::kD6d:
        return "D6d";
    case PointGroupName::kD7d:
        return "D7d";
    case PointGroupName::kD8d:
        return "D8d";
    case PointGroupName::kS4:
        return "S4";
    case PointGroupName::kS6:
        return "S6";
    case PointGroupName::kS8:
        return "S8";
    case PointGroupName::kT:
        return "T";
    case PointGroupName::kTd:
        return "Td";
    case PointGroupName::kTh:
        return "Th";
    case PointGroupName::kO:
        return "O";
    case PointGroupName::kOh:
        return "Oh";
    case PointGroupName::kI:
        return "I";
    case PointGroupName::kIh:
        return "Ih";
    case PointGroupName::kCInfV:
        return "C∞v";
    case PointGroupName::kDInfH:
        return "D∞h";
    }

    return "C1"; // Unreachable; silences -Wreturn-type.
}

namespace detail {

/// Symbol -> group, declaration order of PointGroupName.
inline constexpr auto kPointGroupNames = std::to_array<
    std::pair<std::string_view, PointGroupName>>({
    {"C1", PointGroupName::kC1},   {"Ci", PointGroupName::kCi},     {"Cs", PointGroupName::kCs},
    {"C2", PointGroupName::kC2},   {"C3", PointGroupName::kC3},     {"C4", PointGroupName::kC4},
    {"C5", PointGroupName::kC5},   {"C6", PointGroupName::kC6},     {"C7", PointGroupName::kC7},
    {"C8", PointGroupName::kC8},   {"C2v", PointGroupName::kC2v},   {"C3v", PointGroupName::kC3v},
    {"C4v", PointGroupName::kC4v}, {"C5v", PointGroupName::kC5v},   {"C6v", PointGroupName::kC6v},
    {"C7v", PointGroupName::kC7v}, {"C8v", PointGroupName::kC8v},   {"C2h", PointGroupName::kC2h},
    {"C3h", PointGroupName::kC3h}, {"C4h", PointGroupName::kC4h},   {"C5h", PointGroupName::kC5h},
    {"C6h", PointGroupName::kC6h}, {"C7h", PointGroupName::kC7h},   {"C8h", PointGroupName::kC8h},
    {"D2", PointGroupName::kD2},   {"D3", PointGroupName::kD3},     {"D4", PointGroupName::kD4},
    {"D5", PointGroupName::kD5},   {"D6", PointGroupName::kD6},     {"D7", PointGroupName::kD7},
    {"D8", PointGroupName::kD8},   {"D2h", PointGroupName::kD2h},   {"D3h", PointGroupName::kD3h},
    {"D4h", PointGroupName::kD4h}, {"D5h", PointGroupName::kD5h},   {"D6h", PointGroupName::kD6h},
    {"D7h", PointGroupName::kD7h}, {"D8h", PointGroupName::kD8h},   {"D2d", PointGroupName::kD2d},
    {"D3d", PointGroupName::kD3d}, {"D4d", PointGroupName::kD4d},   {"D5d", PointGroupName::kD5d},
    {"D6d", PointGroupName::kD6d}, {"D7d", PointGroupName::kD7d},   {"D8d", PointGroupName::kD8d},
    {"S4", PointGroupName::kS4},   {"S6", PointGroupName::kS6},     {"S8", PointGroupName::kS8},
    {"T", PointGroupName::kT},     {"Td", PointGroupName::kTd},     {"Th", PointGroupName::kTh},
    {"O", PointGroupName::kO},     {"Oh", PointGroupName::kOh},     {"I", PointGroupName::kI},
    {"Ih", PointGroupName::kIh},   {"C∞v", PointGroupName::kCInfV}, {"D∞h", PointGroupName::kDInfH},
});

} // namespace detail

/// Parses a Schoenflies symbol; exact case, "Cinfv"/"Dinfh" ASCII aliases
/// accepted. \returns nullptr for anything else.
constexpr const PointGroupName* ParsePointGroupName(std::string_view symbol) noexcept {
    if (symbol == "Cinfv")
    {
        symbol = "C∞v";
    } else if (symbol == "Dinfh")
    { symbol = "D∞h"; }

    for (const auto& [name, group] : detail::kPointGroupNames)
    {
        if (name == symbol)
        {
            return &group;
        }
    }

    return nullptr;
}

/// The largest Abelian subgroup of the group, for computation.
///
/// Verified against psi4's direct-detection semantics: Cn (n odd) -> C1,
/// Cn (n even) -> C2; Cnv (n odd) -> Cs, Cnv (n even) -> C2v; Cnh (n odd)
/// -> Cs, Cnh (n even) -> C2h; Dn (n odd) -> C2, Dn (n even) -> D2;
/// Dnh (n odd) -> C2v, Dnh (n even) -> D2h; Dnd (n odd) -> C2h, Dnd
/// (n even) -> D2; S4/S8 -> C2, S6 -> Ci; T/O/I -> D2, Td/Th/Oh/Ih -> D2h;
/// C∞v -> C2v, D∞h -> D2h.
constexpr PointGroup LargestAbelianSubgroup(PointGroupName group) noexcept {
    switch (group)
    {
    case PointGroupName::kC1:
        return PointGroup::kC1;
    case PointGroupName::kCi:
        return PointGroup::kCi;
    case PointGroupName::kCs:
        return PointGroup::kCs;
    case PointGroupName::kC2:
        return PointGroup::kC2;
    case PointGroupName::kC2v:
        return PointGroup::kC2v;
    case PointGroupName::kC2h:
        return PointGroup::kC2h;
    case PointGroupName::kD2:
        return PointGroup::kD2;
    case PointGroupName::kD2h:
        return PointGroup::kD2h;
    case PointGroupName::kC3: // Cn, n odd.
    case PointGroupName::kC5:
    case PointGroupName::kC7:
        return PointGroup::kC1;
    case PointGroupName::kC4: // Cn, n even.
    case PointGroupName::kC6:
    case PointGroupName::kC8:
        return PointGroup::kC2;
    case PointGroupName::kC3v: // Cnv, n odd.
    case PointGroupName::kC5v:
    case PointGroupName::kC7v:
        return PointGroup::kCs;
    case PointGroupName::kC4v: // Cnv, n even.
    case PointGroupName::kC6v:
    case PointGroupName::kC8v:
        return PointGroup::kC2v;
    case PointGroupName::kC3h: // Cnh, n odd.
    case PointGroupName::kC5h:
    case PointGroupName::kC7h:
        return PointGroup::kCs;
    case PointGroupName::kC4h: // Cnh, n even.
    case PointGroupName::kC6h:
    case PointGroupName::kC8h:
        return PointGroup::kC2h;
    case PointGroupName::kD3: // Dn, n odd.
    case PointGroupName::kD5:
    case PointGroupName::kD7:
        return PointGroup::kC2;
    case PointGroupName::kD4: // Dn, n even.
    case PointGroupName::kD6:
    case PointGroupName::kD8:
        return PointGroup::kD2;
    case PointGroupName::kD3h: // Dnh, n odd.
    case PointGroupName::kD5h:
    case PointGroupName::kD7h:
        return PointGroup::kC2v;
    case PointGroupName::kD4h: // Dnh, n even.
    case PointGroupName::kD6h:
    case PointGroupName::kD8h:
        return PointGroup::kD2h;
    case PointGroupName::kD3d: // Dnd, n odd.
    case PointGroupName::kD5d:
    case PointGroupName::kD7d:
        return PointGroup::kC2h;
    case PointGroupName::kD2d: // Dnd, n even.
    case PointGroupName::kD4d:
    case PointGroupName::kD6d:
    case PointGroupName::kD8d:
        return PointGroup::kD2;
    case PointGroupName::kS4:
        return PointGroup::kC2;
    case PointGroupName::kS6:
        return PointGroup::kCi;
    case PointGroupName::kS8:
        return PointGroup::kC2;
    case PointGroupName::kT:
        return PointGroup::kD2;
    case PointGroupName::kTd:
        return PointGroup::kD2h;
    case PointGroupName::kTh:
        return PointGroup::kD2h;
    case PointGroupName::kO:
        return PointGroup::kD2;
    case PointGroupName::kOh:
        return PointGroup::kD2h;
    case PointGroupName::kI:
        return PointGroup::kD2;
    case PointGroupName::kIh:
        return PointGroup::kD2h;
    case PointGroupName::kCInfV:
        return PointGroup::kC2v;
    case PointGroupName::kDInfH:
        return PointGroup::kD2h;
    }

    return PointGroup::kC1; // Unreachable; silences -Wreturn-type.
}

} // namespace qcx::symmetry
