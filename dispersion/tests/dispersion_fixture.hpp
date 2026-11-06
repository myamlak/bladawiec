#pragma once

// Shared fixtures for the dispersion module's derivative tests: the D3/D4
// water fixture, the molecule construction from a coordinate vector, and the
// canonical-order accessors both derivative tests index by. Molecule::Create
// renumbers atoms canonically (by Z, then x, y, z), so a test that compares an
// analytic derivative against displaced geometries must run on the molecule's
// own order, not the fixture's input order.
//
// Test-only helper - not part of the public API.

#include "qcx/dispersion/dispersion.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace qcx::dispersion::testing {

struct FixtureAtom {
    const char* symbol;
    int atomicNumber;
    double x;
    double y;
    double z;
};

// excgrid's D3 fixture (external/excgrid/tests/d3_test.cpp): water, in Bohr.
inline constexpr FixtureAtom kWater[] = {
    {"O", 8, 0.0, 0.0, 0.0},
    {"H", 1, 1.4, 0.0, 0.9},
    {"H", 1, -1.4, 0.0, 0.9},
};

inline Eigen::VectorXd Flatten(std::span<const FixtureAtom> atoms) {
    Eigen::VectorXd coordinates(static_cast<Eigen::Index>(atoms.size() * 3));

    for (std::size_t atom = 0; atom < atoms.size(); ++atom)
    {
        coordinates(static_cast<Eigen::Index>(3 * atom)) = atoms[atom].x;
        coordinates(static_cast<Eigen::Index>(3 * atom + 1)) = atoms[atom].y;
        coordinates(static_cast<Eigen::Index>(3 * atom + 2)) = atoms[atom].z;
    }

    return coordinates;
}

inline qcx::Result<qcx::molecule::Molecule> MakeMolecule(std::span<const FixtureAtom> atoms,
                                                         const Eigen::VectorXd& coordinates,
                                                         int charge,
                                                         int multiplicity) {
    auto positions =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atoms.size(), 3});

    if (!positions.has_value())
    {
        return std::unexpected(positions.error());
    }

    std::vector<qcx::molecule::Atom> elements;
    elements.reserve(atoms.size());

    for (std::size_t i = 0; i < atoms.size(); ++i)
    {
        elements.push_back({atoms[i].symbol, atoms[i].atomicNumber, 0.0});
        (*positions)(i, 0) = coordinates(static_cast<Eigen::Index>(3 * i));
        (*positions)(i, 1) = coordinates(static_cast<Eigen::Index>(3 * i + 1));
        (*positions)(i, 2) = coordinates(static_cast<Eigen::Index>(3 * i + 2));
    }

    positions->MarkHostDirty();
    return qcx::molecule::Molecule::Create(
        std::move(elements), std::move(*positions), charge, multiplicity);
}

// The molecule's atoms and coordinates in the order it reports them.
inline std::vector<FixtureAtom> CanonicalAtoms(const qcx::molecule::Molecule& molecule) {
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinates =
        molecule.CoordinatesBohr();
    std::vector<FixtureAtom> atoms;
    atoms.reserve(molecule.AtomCount());

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        atoms.push_back({molecule.Atoms()[atom].symbol.c_str(),
                         molecule.Atoms()[atom].atomicNumber,
                         coordinates(atom, 0),
                         coordinates(atom, 1),
                         coordinates(atom, 2)});
    }

    return atoms;
}

inline Eigen::VectorXd CanonicalCoordinates(const qcx::molecule::Molecule& molecule) {
    const qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>& coordinates =
        molecule.CoordinatesBohr();
    Eigen::VectorXd flattened(static_cast<Eigen::Index>(molecule.AtomCount() * 3));

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        for (std::size_t direction = 0; direction < 3; ++direction)
        {
            flattened(static_cast<Eigen::Index>(3 * atom + direction)) =
                coordinates(atom, direction);
        }
    }

    return flattened;
}

inline DispersionOptions PresetOptions(DispersionForm form, std::string name) {
    DispersionOptions options;
    options.form = form;
    options.preset = std::move(name);
    return options;
}

} // namespace qcx::dispersion::testing
