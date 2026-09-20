// Analytic construction of C60 and zigzag carbon nanotubes.
#include "large_molecules.hpp"

#include <array>
#include <cmath>
#include <vector>

namespace qcx::molecule::testing {

namespace {

// Shared Bohr-per-Angstrom factor from the molecule module (single definition
// repo-wide).
using qcx::molecule::kAngstromToBohr;
constexpr double kPi = 3.14159265358979323846;

qcx::Result<Molecule> CreateAllCarbon(std::size_t atomCount,
                                      const std::vector<std::array<double, 3>>& positionsBohr) {
    auto coords = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({atomCount, 3});

    if (!coords.has_value())
    {
        return std::unexpected(coords.error());
    }

    for (std::size_t i = 0; i < atomCount; ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coords)(i, d) = positionsBohr[i][d];
        }
    }

    return Molecule::Create(
        std::vector<Atom>(atomCount, Atom{"C", 6, 0.0}), std::move(*coords), 0, 1);
}

} // namespace

qcx::Result<Molecule> MakeBuckminsterfullerene() {
    // The 60-vertex truncated icosahedron [Kroto1985].
    constexpr double phi = 1.6180339887498948482;
    constexpr double scale = 0.7 * kAngstromToBohr; // unit edge 2 -> 1.40 Angstrom

    std::vector<std::array<double, 3>> vertices;
    const auto addEvenPermutations = [&](double a, double b, double c) {
        vertices.push_back({a * scale, b * scale, c * scale});
        vertices.push_back({c * scale, a * scale, b * scale});
        vertices.push_back({b * scale, c * scale, a * scale});
    };

    for (double signB : {1.0, -1.0})
    {
        for (double signC : {1.0, -1.0})
        {
            addEvenPermutations(0.0, signB * 1.0, signC * 3.0 * phi);
        }
    }

    for (double signA : {1.0, -1.0})
    {
        for (double signB : {1.0, -1.0})
        {
            for (double signC : {1.0, -1.0})
            {
                addEvenPermutations(signA * 1.0, signB * (2.0 + phi), signC * 2.0 * phi);
                addEvenPermutations(signA * phi, signB * 2.0, signC * (2.0 * phi + 1.0));
            }
        }
    }

    return CreateAllCarbon(vertices.size(), vertices);
}

qcx::Result<Molecule> MakeZigzagNanotube(std::size_t n, std::size_t cells) {
    // Graphene rolled along n*a1, the construction of [Saito1998].
    // Sites A(m, n2) = (m + n2/2)*sqrt(3)b, 3n2b/2 and B(m, n2) = A(m, n2) +
    // (sqrt(3)b/2, b/2); rolling maps the first coordinate to the tube angle
    // theta. Per cell (n2 = 2c, 2c+1) there are four rings of n atoms each: A(2c) at z = 3cb, B(2c)
    // at 3cb + b/2, A(2c+1) at 3cb + 3b/2, B(2c+1) at 3cb + 2b, with theta offsets 0, pi/n, pi/n,
    // 2pi/n.
    const double bondBohr = 1.42 * kAngstromToBohr;
    const double radius = static_cast<double>(n) * std::sqrt(3.0) * bondBohr / (2.0 * kPi);
    const std::size_t atomCount = 4 * n * cells;

    std::vector<std::array<double, 3>> positions;
    positions.reserve(atomCount);

    for (std::size_t c = 0; c < cells; ++c)
    {
        for (std::size_t m = 0; m < n; ++m)
        {
            const double baseTheta =
                2.0 * kPi * static_cast<double>(m + c) / static_cast<double>(n);
            const double offset = kPi / static_cast<double>(n);
            const auto push = [&](double theta, double z) {
                positions.push_back({radius * std::cos(theta), radius * std::sin(theta), z});
            };
            push(baseTheta, 3.0 * static_cast<double>(c) * bondBohr);
            push(baseTheta + offset, (3.0 * static_cast<double>(c) + 0.5) * bondBohr);
            push(baseTheta + offset, (3.0 * static_cast<double>(c) + 1.5) * bondBohr);
            push(baseTheta + 2.0 * offset, (3.0 * static_cast<double>(c) + 2.0) * bondBohr);
        }
    }

    return CreateAllCarbon(atomCount, positions);
}

} // namespace qcx::molecule::testing
