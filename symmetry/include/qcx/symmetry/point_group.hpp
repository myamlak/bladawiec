#pragma once

#include <string_view>

namespace qcx::symmetry {

/// \defgroup qcx-symmetry Symmetry module
/// Point-group detection, character tables, and symmetry-adapted bases.

/// The eight Abelian point groups usable for irrep-blocked computation.
///
/// All non-Abelian groups are detected in full (see PointGroupName) and then
/// reduced to their largest Abelian subgroup before any computation - the
/// reduction table lives in point_group_name.hpp.
/// \ingroup qcx-symmetry
enum class PointGroup {
    kC1, ///< No symmetry.
    kCi, ///< Inversion only.
    kC2, ///< One C2 axis.
    kCs, ///< One mirror plane.
    kC2h, ///< C2 + the perpendicular mirror.
    kD2, ///< Three mutually perpendicular C2 axes.
    kC2v, ///< C2 + two mirror planes containing it.
    kD2h, ///< D2 + inversion (all three mirrors).
};

/// Standard symbol of the computational group ("C1".."D2h").
constexpr std::string_view ToString(PointGroup group) noexcept {
    switch (group)
    {
    case PointGroup::kC1:
        return "C1";
    case PointGroup::kCi:
        return "Ci";
    case PointGroup::kC2:
        return "C2";
    case PointGroup::kCs:
        return "Cs";
    case PointGroup::kC2h:
        return "C2h";
    case PointGroup::kD2:
        return "D2";
    case PointGroup::kC2v:
        return "C2v";
    case PointGroup::kD2h:
        return "D2h";
    }

    return "C1"; // Unreachable; silences -Wreturn-type.
}

} // namespace qcx::symmetry
