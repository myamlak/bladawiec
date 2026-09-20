#pragma once

// Internal (non-installed) helpers shared by the integral engines: the
// s-function flattening over shells x contraction rows, the primitive
// normalization every BSE contraction coefficient carries, and the
// precomputed atom positions. Not part of the public API - see
// qcx/integrals/one_electron.hpp and qcx/integrals/two_electron.hpp.

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/molecule/molecule.hpp"

#include <Eigen/Core>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numbers>
#include <string>
#include <utility>
#include <vector>

namespace qcx::integrals::internal {

// One contracted s function: the (shell, contraction row) pair of one atom.
struct FlatSFunction {
    std::size_t atomIndex; // Index into Molecule::Atoms().
    // The owning s shell, borrowed (basisSet outlives). Reference semantics
    // with value semantics: never null, and the aggregate stays assignable,
    // which a plain reference member would not be.
    std::reference_wrapper<const qcx::basisset::Shell> shell;
    std::size_t contractionRow; // Row index in Shell::coefficients.
};

// Flattens every s-shell contraction row of \p basisSet into one entry per
// row, in molecule-canonical atom order.
//
// General contractions get one entry per row (an aug-cc-pVDZ H s shell has
// two), so flattening rows costs nothing and is already correct for them.
inline qcx::Result<std::vector<FlatSFunction>> FlattenSFunctions(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
    std::vector<FlatSFunction> functions;
    const auto& atoms = molecule.Atoms();

    for (std::size_t atomIndex = 0; atomIndex < atoms.size(); ++atomIndex)
    {
        const qcx::basisset::ElementBasis* elementBasis =
            basisSet.Find(atoms[atomIndex].atomicNumber);

        if (elementBasis == nullptr)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "basis set has no entry for " + atoms[atomIndex].symbol});
        }

        for (const qcx::basisset::Shell& shell : elementBasis->shells)
        {
            if (shell.angularMomentum != 0)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kUnimplemented,
                               "non-s shells are not implemented (angular momentum " +
                                   std::to_string(shell.angularMomentum) + ")"});
            }

            for (std::size_t row = 0; row < shell.coefficients.size(); ++row)
            {
                functions.push_back(FlatSFunction{atomIndex, shell, row});
            }
        }
    }

    return functions;
}

// (2a/pi)^(3/4): the primitive normalization every integral over s primitives
// carries on each contraction coefficient (BSE coefficients are unnormalized).
inline double PrimitiveNormalization(double exponent) noexcept {
    return std::pow(2.0 * exponent / std::numbers::pi, 0.75);
}

// Atom positions in Bohr, precomputed once per engine call so the O(n^4)
// loops read memory instead of the coordinate tensor.
inline std::vector<Eigen::Vector3d> AtomPositions(const qcx::molecule::Molecule& molecule) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<Eigen::Vector3d> positions;
    positions.reserve(molecule.AtomCount());

    for (std::size_t atomIndex = 0; atomIndex < molecule.AtomCount(); ++atomIndex)
    {
        positions.emplace_back(
            coordinates(atomIndex, 0), coordinates(atomIndex, 1), coordinates(atomIndex, 2));
    }

    return positions;
}

// One normalized primitive of one flattened s function.
struct PrimitiveData {
    double exponent; // Gaussian exponent a.
    double coefficient; // d = c * (2a/pi)^(3/4).
};

// The normalized primitives of one flattened s function.
struct SFunctionData {
    std::size_t atomIndex; // Index into Molecule::Atoms().
    std::vector<PrimitiveData> primitives; // In primitive order.
};

// Flattens \p basisSet and normalizes every primitive coefficient once, so
// the engines' inner loops multiply precomputed d values.
inline qcx::Result<std::vector<SFunctionData>> BuildSFunctionData(
    const qcx::molecule::Molecule& molecule, const qcx::basisset::BasisSet& basisSet) {
    auto flat = FlattenSFunctions(molecule, basisSet);

    if (!flat.has_value())
    {
        return std::unexpected(flat.error());
    }

    std::vector<SFunctionData> functions;
    functions.reserve(flat->size());

    for (const FlatSFunction& function : *flat)
    {
        SFunctionData data;
        data.atomIndex = function.atomIndex;
        const qcx::basisset::Shell& shell = function.shell.get();
        const std::vector<double>& coefficients = shell.coefficients[function.contractionRow];
        data.primitives.reserve(shell.exponents.size());

        for (std::size_t primitive = 0; primitive < shell.exponents.size(); ++primitive)
        {
            const double exponent = shell.exponents[primitive];
            data.primitives.push_back(PrimitiveData{
                exponent, coefficients[primitive] * PrimitiveNormalization(exponent)});
        }

        functions.push_back(std::move(data));
    }

    return functions;
}

} // namespace qcx::integrals::internal
