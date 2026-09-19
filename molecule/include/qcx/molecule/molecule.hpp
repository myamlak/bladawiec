#pragma once

#include "qcx/backend/tags.hpp"
#include "qcx/error.hpp"
#include "qcx/memory/tensor.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace qcx::molecule {

/// Bohr per Angstrom: multiply lengths in Angstrom by this to get Bohr.
/// (The reciprocal of the 2018 CODATA Bohr radius in Angstrom.)
inline constexpr double kAngstromToBohr = 1.8897261246257702;

/// One atom: its element symbol, atomic number, and isotopic mass.
/// \ingroup qcx-molecule
struct Atom {
    std::string symbol; ///< IUPAC symbol, exact case (validated against the element table).
    int atomicNumber; ///< Atomic number Z.
    double isotopicMass; ///< Nuclear mass in u; 0 selects the most-abundant isotope.
};

/// Immutable, move-only collection of atoms with Bohr coordinates.
///
/// Construction validates against the periodic table and canonically
/// renumbers the atoms (by Z, then position); coordinates are stored in a
/// CPU Tensor<double, 2> with shape {atomCount, 3}.
/// \ingroup qcx-molecule
class Molecule {
public:
    /// Creates a molecule from atoms and Bohr coordinates.
    /// \param atoms Atoms in input order; renumbered canonically on success.
    /// \param coordinatesBohr Shape {atoms.size(), 3} coordinates in Bohr.
    /// \param charge Total electric charge (signed; at most the sum of Z).
    /// \param multiplicity Spin multiplicity 2S+1, at least 1.
    /// \returns The molecule, or an Error (kInvalidArgument for an empty atom
    /// list, a shape mismatch, an unknown element or symbol/Z mismatch, an
    /// invalid charge/multiplicity, or coincident atoms).
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- charge and
    // multiplicity are the standard QC pair; both are validated.
    static qcx::Result<Molecule> Create(
        std::vector<Atom> atoms,
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> coordinatesBohr,
        int charge,
        int multiplicity);

    /// Move construction (Molecule is movable-only).
    Molecule(Molecule&&) = default;
    /// Move assignment (Molecule is movable-only).
    /// \returns This molecule.
    Molecule& operator=(Molecule&&) = default;
    Molecule(const Molecule&) = delete;
    /// Copying is deleted: molecules are movable-only.
    Molecule& operator=(const Molecule&) = delete;
    ~Molecule() = default;

    /// Deep copy; the clone starts host-canonical with identical logical content.
    /// \returns The clone, or an Error when the underlying copy fails.
    qcx::Result<Molecule> Clone();

    /// Atoms in canonical order.
    /// \returns The atom list.
    const std::vector<Atom>& Atoms() const noexcept {
        return _atoms;
    }

    /// Bohr coordinates, shape {atomCount, 3}, rows matching Atoms().
    /// \returns The coordinate tensor.
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& CoordinatesBohr() const noexcept {
        return _coordinates;
    }

    /// Total electric charge.
    /// \returns The charge.
    int Charge() const noexcept {
        return _charge;
    }

    /// Spin multiplicity 2S+1.
    /// \returns The multiplicity.
    int Multiplicity() const noexcept {
        return _multiplicity;
    }

    /// Total number of electrons: the sum of atomic numbers minus the charge.
    /// \returns The electron count.
    int ElectronCount() const noexcept {
        int electrons = 0;

        for (const Atom& atom : _atoms)
        {
            electrons += atom.atomicNumber;
        }

        return electrons - _charge;
    }

    /// Number of atoms.
    /// \returns The atom count.
    std::size_t AtomCount() const noexcept {
        return _atoms.size();
    }

private:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- mirrors Create's signature.
    Molecule(std::vector<Atom> atoms,
             qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> coordinates,
             int charge,
             int multiplicity) :
        _atoms(std::move(atoms)), _coordinates(std::move(coordinates)), _charge(charge),
        _multiplicity(multiplicity) {}

    std::vector<Atom> _atoms;
    qcx::memory::Tensor<double, 2, qcx::backend::CpuTag> _coordinates;
    int _charge;
    int _multiplicity;
};

} // namespace qcx::molecule
