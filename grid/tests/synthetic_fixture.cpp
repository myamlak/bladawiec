#include "synthetic_fixture.hpp"

#include "qcx/backend/tags.hpp"
#include "qcx/memory/tensor.hpp"

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace qcx::grid {
namespace {

// The chain's fixed interatomic spacing, in Bohr (0.74 Angstrom - the H2
// bond length, so every neighbouring pair is a physically meaningful
// distance rather than a degenerate one).
constexpr double kChainSpacingBohr = 1.4;

// STO-3G hydrogen: one contracted s shell, so one basis function per atom.
// The same block the grid module's other AO fixtures parse.
constexpr std::string_view kSto3gHydrogen = R"(
BASIS "ao basis" SPHERICAL PRINT
H S
3.42525091 0.15432897
0.62391373 0.53532814
0.16885540 0.44463454
END
)";

} // namespace

qcx::Result<SyntheticSystem> MakeSyntheticChain(std::size_t atomCount) {
    if (atomCount == 0)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                          "synthetic chain needs at least one atom"});
    }

    // {atomCount, 3}: the shape Molecule::Create validates against the atom
    // list.  Rows are written in generation order; the molecule module
    // renumbers them canonically on construction.
    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomCount, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    std::vector<qcx::molecule::Atom> atoms;
    atoms.reserve(atomCount);

    for (std::size_t index = 0; index < atomCount; ++index)
    {
        (*coordinates)(index, 0) = static_cast<double>(index) * kChainSpacingBohr;
        (*coordinates)(index, 1) = 0.0;
        (*coordinates)(index, 2) = 0.0;
        // Mass 0 selects hydrogen's most-abundant isotope.
        atoms.push_back(qcx::molecule::Atom{"H", 1, 0.0});
    }

    coordinates->MarkHostDirty();

    // Neutral, singlet: 2 * atomCount electrons in closed shells.
    auto molecule =
        qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);

    if (!molecule.has_value())
    {
        return std::unexpected(molecule.error());
    }

    auto basis = qcx::basisset::ParseNwchemText(kSto3gHydrogen);

    if (!basis.has_value())
    {
        return std::unexpected(basis.error());
    }

    return SyntheticSystem{std::move(*molecule), std::move(*basis)};
}

} // namespace qcx::grid
