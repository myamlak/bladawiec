#pragma once

#include "qcx/basisset/basis_set.hpp"
#include "qcx/error.hpp"
#include "qcx/grid/ao_evaluator.hpp"
#include "qcx/molecule/molecule.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <vector>

namespace qcx::grid {

/// One AO shell's screening envelope: what a significance test needs to bound
/// the shell's contribution to the density at an arbitrary point.
///
/// The envelope is built from the BASIS, not from a density, so it is
/// density-independent and survives across SCF iterations; the density enters
/// only through the weight in ShellIsSignificant.
///
/// One envelope per EVALUATOR shell, which is one basis shell: a general
/// contraction keeps its coefficient rows together in the evaluator, and the
/// envelope's AO range covers all of them (the evaluator lays the rows out
/// contiguously from aoOffset). The mapping is by position, not by shell index.
///
/// The bound the test rests on: beyond the shell's radial maximum r* the AO
/// magnitudes decay at least as fast as exp(-z d^2), where z is the shell's
/// most diffuse primitive exponent and d is the distance past that maximum.
/// At r* itself r^l exp(-z r^2) is maximal, so the ratio of a value at r to the
/// shell's largest value is bounded by that exponential.
/// Public aggregate: the fields are the API (aggregate-struct exemption).
/// \ingroup qcx-grid
struct ShellEnvelope {
    std::size_t aoOffset = 0; ///< Index of the shell's first AO.
    std::size_t functionCount = 0; ///< AOs in the shell, all contraction rows.
    std::array<double, 3> center{}; ///< Shell center, Bohr.
    // Which atom carries the shell. The screening test does not need it; a
    // gradient does, because the AOs' motion reaches the coordinates of the atom
    // they sit on and of no other.
    std::size_t atomIndex = 0; ///< The molecule's atom the shell is centered on.
    double minExponent = 0.0; ///< Most diffuse primitive exponent.
    double extentRadius = 0.0; ///< Radius of the shell's radial maximum, Bohr.
};

/// Builds the screening envelopes for a molecule and basis.
///
/// The result is in the EVALUATOR's shell order - the molecule's canonical atom
/// order, each atom's basis shells in file order, each shell's contraction rows
/// in order - which is what ShellRanges() reports. A mismatch would silently
/// screen the wrong shells, so the builder takes the ranges and checks the
/// count against the basis rather than assuming the two agree.
/// \param molecule The molecule whose atoms center the shells.
/// \param basis The basis set; must contain every element of the molecule.
/// \param ranges The evaluator's shell ranges (AoEvaluator::ShellRanges()).
/// \returns The envelopes, or kInvalidArgument when the basis has no entry for
/// an element, when a shell has no positive exponent, or when the ranges do not
/// match the expanded shell count.
/// \ingroup qcx-grid
inline qcx::Result<std::vector<ShellEnvelope>> BuildShellEnvelopes(
    const qcx::molecule::Molecule& molecule,
    const qcx::basisset::BasisSet& basis,
    std::span<const ShellRange> ranges) {
    const auto& coordinates = molecule.CoordinatesBohr();
    std::vector<ShellEnvelope> envelopes;
    envelopes.reserve(ranges.size());
    std::size_t shellIndex = 0;

    for (std::size_t atom = 0; atom < molecule.AtomCount(); ++atom)
    {
        const qcx::basisset::ElementBasis* element =
            basis.Find(molecule.Atoms()[atom].atomicNumber);

        if (element == nullptr)
        {
            return std::unexpected(
                qcx::Error{qcx::ErrorCode::kInvalidArgument,
                           "BuildShellEnvelopes: no basis entry for element Z=" +
                               std::to_string(molecule.Atoms()[atom].atomicNumber)});
        }

        const std::array<double, 3> center = {
            coordinates(atom, 0), coordinates(atom, 1), coordinates(atom, 2)};

        for (const qcx::basisset::Shell& shell : element->shells)
        {
            double minExponent = shell.exponents.empty() ? 0.0 : shell.exponents[0];

            for (const double exponent : shell.exponents)
            {
                minExponent = std::min(minExponent, exponent);
            }

            if (minExponent <= 0.0)
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "BuildShellEnvelopes: shell has no positive exponent"});
            }

            // The radial maximum of r^l exp(-z r^2) sits at sqrt(l / 2z); the
            // shell is treated as significant out to there before the
            // exponential decay past it is allowed to argue for dropping it.
            const double angularMomentum = static_cast<double>(shell.angularMomentum);
            const double extentRadius = std::sqrt(angularMomentum / (2.0 * minExponent));

            if (shellIndex >= ranges.size())
            {
                return std::unexpected(
                    qcx::Error{qcx::ErrorCode::kInvalidArgument,
                               "BuildShellEnvelopes: fewer shell ranges than basis shells"});
            }

            envelopes.push_back(ShellEnvelope{ranges[shellIndex].aoOffset,
                                              ranges[shellIndex].functionCount,
                                              center,
                                              atom,
                                              minExponent,
                                              extentRadius});
            ++shellIndex;
        }
    }

    if (shellIndex != ranges.size())
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "BuildShellEnvelopes: more shell ranges than basis shells"});
    }

    return envelopes;
}

/// The decay factor of a shell at a point: one inside the shell's radial
/// extent, exp(-z d^2) beyond it.
/// \param shell The shell envelope.
/// \param point The grid point, Bohr.
/// \returns The factor, in [0, 1].
/// \ingroup qcx-grid
[[nodiscard]] inline double ShellDecayFactor(const ShellEnvelope& shell,
                                             const std::array<double, 3>& point) noexcept {
    double distanceSquared = 0.0;

    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        const double delta = point[axis] - shell.center[axis];
        distanceSquared += delta * delta;
    }

    const double distance = std::sqrt(distanceSquared);
    const double past = distance - shell.extentRadius;

    if (past <= 0.0)
    {
        return 1.0;
    }

    return std::exp(-shell.minExponent * past * past);
}

/// Whether a shell can be neglected at a point.
///
/// The test is DENSITY-WEIGHTED, not geometric: \p weight is the shell's share
/// of the density matrix (see ShellDensityWeights), so a distant shell is
/// dropped only when its contribution cannot matter, which is what keeps the
/// rule correct for diffuse functions and far-field coupling.  A pure distance
/// cutoff would drop exactly those shells and fail silently.
///
/// The weight is scaled by the decay factor and compared against \p tolerance:
/// a shell survives when `weight * decay > tolerance`, so a test that lands
/// exactly on the tolerance DROPS its shell.  The weight carries the density
/// but assumes unit-scale AO magnitudes (the basis module normalizes every
/// contraction to unit self-overlap), so the tolerance is a practical bound
/// rather than a rigorous one; the assembly gate verifies the coverage by
/// comparing screened against dense densities rather than trusting this
/// constant.
/// \param shell The shell envelope.
/// \param point The grid point, Bohr.
/// \param weight The shell's density weight.
/// \param tolerance The neglect threshold.
/// \returns True when the shell must be kept.
/// \ingroup qcx-grid
[[nodiscard]] inline bool ShellIsSignificant(const ShellEnvelope& shell,
                                             const std::array<double, 3>& point,
                                             double weight,
                                             double tolerance) noexcept {
    return weight * ShellDecayFactor(shell, point) > tolerance;
}

/// The per-shell density weights: how much of the density each shell carries.
///
/// Shell i's weight is the sum of the magnitudes of the density-matrix rows
/// that touch it, `sum over mu in i, nu of |D_mu nu|`, so a shell coupled to
/// large density elements stays significant far from its center while a shell
/// whose couplings are all negligible can be dropped near one.  One pass over
/// the density matrix, O(AOCount()^2), independent of the grid.
/// \param envelopes The shell envelopes.
/// \param density The spin density matrix in ROW-MAJOR order, AOCount() x
/// AOCount() with no padding.
/// \param aoCount The AO dimension.
/// \returns One weight per envelope, in the same order, or kInvalidArgument
/// when \p density does not hold exactly aoCount^2 entries.
/// \ingroup qcx-grid
[[nodiscard]] inline qcx::Result<std::vector<double>> ShellDensityWeights(
    std::span<const ShellEnvelope> envelopes,
    std::span<const double> density,
    std::size_t aoCount) {
    if (density.size() != aoCount * aoCount)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument,
                       "ShellDensityWeights: density must hold AOCount()^2 entries"});
    }

    std::vector<double> weights(envelopes.size(), 0.0);

    for (std::size_t shell = 0; shell < envelopes.size(); ++shell)
    {
        const ShellEnvelope& envelope = envelopes[shell];
        double weight = 0.0;

        for (std::size_t mu = 0; mu < envelope.functionCount; ++mu)
        {
            const std::size_t row = (envelope.aoOffset + mu) * aoCount;

            for (std::size_t nu = 0; nu < aoCount; ++nu)
            {
                weight += std::abs(density[row + nu]);
            }
        }

        weights[shell] = weight;
    }

    return weights;
}

} // namespace qcx::grid
