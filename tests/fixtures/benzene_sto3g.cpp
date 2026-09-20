#include "benzene_sto3g.hpp"

#include "h2_sto3g.hpp"
#include "qcx/memory/tensor.hpp"

#include <utility>
#include <vector>

namespace qcx::testing {

// The D6h radii in Bohr: 1.395 A and 2.480 A (the C-H distance
// r(CH) = 1.085 A measured from the carbon) converted by 1 A =
// 1.88972612463 Bohr. The spoke geometry puts H_k on the same ray as
// C_k, at 60-degree increments starting from the +x axis.
namespace {
constexpr double kCarbonRadius = 2.636168;
constexpr double kHydrogenRadius = 4.686530;

// The shared 12-row geometry (the six carbons on the R_C ring then the six
// hydrogens on the same spokes, both counterclockwise from +x) so the
// rotated variant cannot drift from the unrotated one - the pins pin both.
qcx::Result<qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>> MakeBenzeneCoordinates() {
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({12, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    // The six carbons on the R_C ring, counterclockwise from +x.
    (*coordinates)(0, 0) = 2.636168;
    (*coordinates)(0, 1) = 0.0;
    (*coordinates)(0, 2) = 0.0;
    (*coordinates)(1, 0) = 1.318084;
    (*coordinates)(1, 1) = 2.282988456;
    (*coordinates)(1, 2) = 0.0;
    (*coordinates)(2, 0) = -1.318084;
    (*coordinates)(2, 1) = 2.282988456;
    (*coordinates)(2, 2) = 0.0;
    (*coordinates)(3, 0) = -2.636168;
    (*coordinates)(3, 1) = 0.0;
    (*coordinates)(3, 2) = 0.0;
    (*coordinates)(4, 0) = -1.318084;
    (*coordinates)(4, 1) = -2.282988456;
    (*coordinates)(4, 2) = 0.0;
    (*coordinates)(5, 0) = 1.318084;
    (*coordinates)(5, 1) = -2.282988456;
    (*coordinates)(5, 2) = 0.0;

    // The six hydrogens on the same spokes, at R_H.
    (*coordinates)(6, 0) = 4.686530;
    (*coordinates)(6, 1) = 0.0;
    (*coordinates)(6, 2) = 0.0;
    (*coordinates)(7, 0) = 2.343265;
    (*coordinates)(7, 1) = 4.058654036;
    (*coordinates)(7, 2) = 0.0;
    (*coordinates)(8, 0) = -2.343265;
    (*coordinates)(8, 1) = 4.058654036;
    (*coordinates)(8, 2) = 0.0;
    (*coordinates)(9, 0) = -4.686530;
    (*coordinates)(9, 1) = 0.0;
    (*coordinates)(9, 2) = 0.0;
    (*coordinates)(10, 0) = -2.343265;
    (*coordinates)(10, 1) = -4.058654036;
    (*coordinates)(10, 2) = 0.0;
    (*coordinates)(11, 0) = 2.343265;
    (*coordinates)(11, 1) = -4.058654036;
    (*coordinates)(11, 2) = 0.0;
    coordinates->MarkHostDirty();

    return coordinates;
}

std::vector<qcx::molecule::Atom> BenzeneAtoms() {
    return {{"C", 6, 0.0},
            {"C", 6, 0.0},
            {"C", 6, 0.0},
            {"C", 6, 0.0},
            {"C", 6, 0.0},
            {"C", 6, 0.0},
            {"H", 1, 0.0},
            {"H", 1, 0.0},
            {"H", 1, 0.0},
            {"H", 1, 0.0},
            {"H", 1, 0.0},
            {"H", 1, 0.0}};
}
} // namespace

qcx::Result<qcx::basisset::BasisSet> MakeBenzeneSto3gBasis() {
    auto carbon = qcx::basisset::ParseNwchemText(kSto3gCarbon);

    if (!carbon.has_value())
    {
        return std::unexpected(carbon.error());
    }

    auto hydrogen = qcx::basisset::ParseNwchemText(kSto3gHydrogen);

    if (!hydrogen.has_value())
    {
        return std::unexpected(hydrogen.error());
    }

    auto merged = carbon->Merge(*hydrogen);

    if (!merged.has_value())
    {
        return std::unexpected(merged.error());
    }

    return std::move(*carbon);
}

qcx::Result<qcx::molecule::Molecule> MakeBenzeneSto3g() {
    auto coordinates = MakeBenzeneCoordinates();

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    return qcx::molecule::Molecule::Create(BenzeneAtoms(), std::move(*coordinates), 0, 1);
}

qcx::Result<qcx::molecule::Molecule> MakeRotatedBenzeneSto3g() {
    auto coordinates = MakeBenzeneCoordinates();

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    // 30 degrees in-plane: every in-plane C2 axis of the D6h fixture lands
    // 30 degrees off the global axes, so the D2h realization's in-plane C2
    // elements mix x and y in the global frame - only the C6-axis C2, the
    // inversion, and the molecular-plane mirror stay signed coordinate
    // permutations of the global frame (the C2h fallback of the symmetry
    // gate).
    constexpr double kCos30 = 0.8660254037844387;
    constexpr double kSin30 = 0.5;

    for (std::size_t row = 0; row < 12; ++row)
    {
        const double x = (*coordinates)(row, 0);
        const double y = (*coordinates)(row, 1);
        (*coordinates)(row, 0) = x * kCos30 - y * kSin30;
        (*coordinates)(row, 1) = x * kSin30 + y * kCos30;
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(BenzeneAtoms(), std::move(*coordinates), 0, 1);
}

} // namespace qcx::testing
