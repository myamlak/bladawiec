// The Molden-format F-file writer.
// Section layout and record formats follow the canonical molden manual
// page (molden_format.html) and pyscf's molden.py writer; the two encoded
// conventions (stored [GTO] coefficients as-is; the [MO] m-order ->
// 5D/7F-slot permutation) are documented on WriteMoldenFile. Validation
// runs first - every rejection happens before the output file opens.

#include "qcx/io/molden_export.hpp"

#include "qcx/basisset/basis_set.hpp"
#include "qcx/molecule/molecule.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace qcx::io {
namespace {

// The section headers, in file order.
constexpr std::string_view kMoldenFormatHeader = "[Molden Format]";
constexpr std::string_view kAtomsAuHeader = "[Atoms] (AU)";
constexpr std::string_view kFiveDKeyword = "[5D]";
constexpr std::string_view kSevenFKeyword = "[7F]";
constexpr std::string_view kGtoHeader = "[GTO]";
constexpr std::string_view kMoHeader = "[MO]";

// The [MO] record prefixes. The space after '=' is part of the format -
// consumers split on it.
constexpr std::string_view kSymRecord = "Sym= A";
constexpr std::string_view kEneRecord = "Ene= ";
constexpr std::string_view kSpinRecord = "Spin= ";
constexpr std::string_view kOccupRecord = "Occup= ";

// The format constants: [Atoms] (AU) Bohr coordinates at 12 decimals;
// the [MO] energies at ~15 significant digits, the [GTO]/[MO] coefficients
// at ~14. Free-format doubles throughout (fixed notation ends with the
// atoms section - the [GTO] numbers must not inherit its precision).
constexpr int kAtomCoordinateDecimals = 12;
constexpr int kEneSignificantDigits = 15;
constexpr int kCoefficientSignificantDigits = 14;

// The [GTO] shell label of one angular momentum (l = 0..6).
std::string_view ShellLabel(int angularMomentum) {
    static constexpr std::array<std::string_view, 7> kLabels = {"s", "p", "d", "f", "g", "h", "i"};

    return kLabels[static_cast<std::size_t>(angularMomentum)];
}

// The spherical m of one molden [MO] slot (1-based): the 5D/7F order
// (0, +1, -1, +2, -2, ...) - slot 1 -> m = 0, even slots -> +s/2, odd
// slots above 1 -> -(s-1)/2.
int MOfMoldenSlot(int slot) {
    if (slot == 1)
    {
        return 0;
    }

    return (slot % 2 == 0) ? slot / 2 : -(slot - 1) / 2;
}

} // namespace

qcx::Result<void> WriteMoldenFile(const qcx::molecule::Molecule& molecule,
                                  const qcx::basisset::BasisSet& basis,
                                  std::span<const MoldenMolecularOrbitals> moBlocks,
                                  std::string_view path) {
    // --- Validation pass: everything fails before the file opens. ------
    // The AO basis runs over atoms in molecule order, shells in basis-file
    // order, functions in the shell's m order; the [MO] rows permute that
    // order to the molden 5D/7F slots.
    std::vector<Eigen::Index> moldenRowToQcxIndex;
    std::vector<const qcx::basisset::ElementBasis*> elements;
    Eigen::Index qcxFunctionCount = 0;

    for (const auto& atom : molecule.Atoms())
    {
        const auto* element = basis.Find(atom.atomicNumber);

        if (element == nullptr)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the basis set has no element Z = " +
                                                  std::to_string(atom.atomicNumber)});
        }

        elements.push_back(element);

        for (const auto& shell : element->shells)
        {
            if (!shell.isSpherical)
            {
                return std::unexpected(qcx::Error{qcx::ErrorCode::kUnimplemented,
                                                  "cartesian shells cannot be exported "
                                                  "(spherical only)"});
            }

            for (int slot = 1; slot <= 2 * shell.angularMomentum + 1; ++slot)
            {
                moldenRowToQcxIndex.push_back(qcxFunctionCount + MOfMoldenSlot(slot) +
                                              shell.angularMomentum);
            }

            qcxFunctionCount += 2 * shell.angularMomentum + 1;
        }
    }

    for (const MoldenMolecularOrbitals& block : moBlocks)
    {
        if (block.coefficients.rows() != qcxFunctionCount ||
            block.coefficients.cols() != qcxFunctionCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "the MO coefficients must be n x n with n the AO count"});
        }

        if (block.energies.size() != qcxFunctionCount)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument, "the MO energies must have length n"});
        }

        if (block.occupations.size() != qcxFunctionCount)
        {
            return std::unexpected(qcx::Error{qcx::ErrorCode::kInvalidArgument,
                                              "the MO occupations must have length n"});
        }
    }

    std::ofstream file(std::string(path), std::ios::out | std::ios::trunc);

    if (!file)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                          "cannot write the molden file " + std::string(path)});
    }

    file << kMoldenFormatHeader << "\n";
    file << kAtomsAuHeader << "\n";

    // [Atoms] (AU): symbol, 1-based index, Z and the Bohr coordinates at
    // 12 decimals.
    const auto& coordinates = molecule.CoordinatesBohr();

    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        const auto& atom = molecule.Atoms()[atomIndex];
        file << atom.symbol << " " << atomIndex + 1 << " " << atom.atomicNumber << " " << std::fixed
             << std::setprecision(kAtomCoordinateDecimals) << coordinates(atomIndex, 0) << " "
             << coordinates(atomIndex, 1) << " " << coordinates(atomIndex, 2) << "\n";
    }

    // Spherical everywhere: the [5D]/[7F] keywords declare the
    // function order the [MO] rows below use.
    file << kFiveDKeyword << "\n";
    file << kSevenFKeyword << "\n";
    file << kGtoHeader << "\n";
    file << std::defaultfloat << std::setprecision(kCoefficientSignificantDigits);

    // [GTO]: per atom a "symbol 0" line (the atom index of the [Atoms]
    // block), per contraction a "label nPrims 1.00" line followed by the
    // "exponent coefficient" primitive rows, a blank line between atoms.
    // The STORED coefficients - the parse-time unit-norm scale is in
    // them, N_l(zeta) applies at evaluation only (ao_evaluator.hpp).
    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        const auto& atom = molecule.Atoms()[atomIndex];
        const auto* element = elements[atomIndex];
        file << atom.symbol << " 0\n";

        for (const auto& shell : element->shells)
        {
            for (const auto& coefficients : shell.coefficients)
            {
                file << ShellLabel(shell.angularMomentum) << " " << shell.exponents.size()
                     << " 1.00\n";

                for (std::size_t primitive = 0; primitive < shell.exponents.size(); ++primitive)
                {
                    file << shell.exponents[primitive] << " " << coefficients[primitive] << "\n";
                }
            }
        }

        file << "\n";
    }

    // [MO]: per orbital the Sym/Ene/Spin/Occup records, then all n
    // coefficient rows with 1-based indices in the molden 5D/7F order.
    file << kMoHeader << "\n";

    for (const MoldenMolecularOrbitals& block : moBlocks)
    {
        for (Eigen::Index orbital = 0; orbital < block.coefficients.cols(); ++orbital)
        {
            file << kSymRecord << "\n";
            file << kEneRecord << std::setprecision(kEneSignificantDigits)
                 << block.energies(orbital) << "\n";
            file << kSpinRecord << block.spin << "\n";
            file << kOccupRecord << block.occupations(orbital) << "\n";
            file << std::setprecision(kCoefficientSignificantDigits);

            for (std::size_t row = 0; row < moldenRowToQcxIndex.size(); ++row)
            {
                file << row + 1 << " " << block.coefficients(moldenRowToQcxIndex[row], orbital)
                     << "\n";
            }

            file << "\n";
        }
    }

    // The close flushes the stream: a mid-write failure (disk full, say)
    // surfaces here, not as a silently truncated file.
    file.close();

    if (!file)
    {
        return std::unexpected(qcx::Error{qcx::ErrorCode::kIOError,
                                          "cannot write the molden file " + std::string(path)});
    }

    return {};
}

} // namespace qcx::io
