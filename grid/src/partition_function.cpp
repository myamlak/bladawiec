#include "qcx/grid/partition_function.hpp"

#include <cmath>
#include <limits>

namespace qcx::grid {

namespace {

// Bragg-Slater radii in Angstrom, scaled to Bohr on lookup; elements
// without a tabulated value fall back to a default (1.0 A) radius.
struct SlaterEntry {
    int atomicNumber;
    double radiusAngstrom;
};

constexpr std::array<SlaterEntry, 46> kSlaterRadii = {{
    {1, 0.35},  {2, 1.0},   {3, 1.45},  {4, 1.05},  {5, 0.85},  {6, 0.70},  {7, 0.65},  {8, 0.60},
    {9, 0.50},  {10, 1.0},  {11, 1.80}, {12, 1.50}, {13, 1.25}, {14, 1.10}, {15, 1.00}, {16, 1.00},
    {17, 1.00}, {18, 1.0},  {19, 2.20}, {20, 1.80}, {21, 1.60}, {22, 1.40}, {23, 1.35}, {24, 1.40},
    {25, 1.40}, {26, 1.40}, {27, 1.35}, {28, 1.35}, {29, 1.35}, {30, 1.35}, {31, 1.30}, {32, 1.25},
    {33, 1.15}, {34, 1.15}, {35, 1.15}, {36, 1.0},  {37, 2.35}, {38, 2.00}, {39, 1.80}, {40, 1.55},
    {41, 1.45}, {42, 1.45}, {43, 1.35}, {44, 1.30}, {45, 1.35}, {46, 1.40},
}};

constexpr double kSwitchWindow = 0.64; // SSF: |mu| <= a window.
constexpr double kDefaultRadiusA = 1.0; // Fallback for untabulated elements.
constexpr double kAngstromToBohr = 1.8897261246257702;

double RadiusOf(int atomicNumber) {
    for (const SlaterEntry& entry : kSlaterRadii)
    {
        if (entry.atomicNumber == atomicNumber)
        {
            return entry.radiusAngstrom * kAngstromToBohr;
        }
    }

    return kDefaultRadiusA * kAngstromToBohr;
}

// SSF switching function: s(mu) in [0, 1], s(mu) = 0 for mu >= a,
// s(mu) = 1 for mu <= -a, polynomial in between (z is the odd
// degree-7 polynomial z(t) = t(35 - t^2(35 - t^2(21 - 5 t^2)))/16).
double Switch(double mu) {
    if (mu >= kSwitchWindow)
    {
        return 0.0;
    }

    if (mu <= -kSwitchWindow)
    {
        return 1.0;
    }

    const double t = mu / kSwitchWindow;
    const double t2 = t * t;
    const double z = t * (35.0 - t2 * (35.0 - t2 * (21.0 - 5.0 * t2))) / 16.0;
    return 0.5 * (1.0 - z);
}

} // namespace

double BraggSlaterRadiusBohr(int atomicNumber) {
    return RadiusOf(atomicNumber);
}

std::vector<double> BeckeRawWeights(const std::array<double, 3>& pointBohr,
                                    const qcx::molecule::Molecule& molecule) {
    std::size_t ignored = 0;
    return BeckeRawWeights(pointBohr, molecule, ignored);
}

std::vector<double> BeckeRawWeights(const std::array<double, 3>& pointBohr,
                                    const qcx::molecule::Molecule& molecule,
                                    std::size_t& switchCount) {
    const std::vector<qcx::molecule::Atom>& atoms = molecule.Atoms();
    const auto& coordinates = molecule.CoordinatesBohr();
    const std::size_t atomCount = atoms.size();

    std::vector<double> distance(atomCount);

    for (std::size_t a = 0; a < atomCount; ++a)
    {
        const double dx = pointBohr[0] - coordinates(a, 0);
        const double dy = pointBohr[1] - coordinates(a, 1);
        const double dz = pointBohr[2] - coordinates(a, 2);
        distance[a] = std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    // Near-nucleus shortcut: within 0.5 (1 - a) of the nearest atom every
    // switch sits on the mu <= -a plateau, so the raw weight is 1 (the
    // standard Becke shortcut).  "Every switch" is exact only while the
    // nearest atom stands at >= 0.5 (1 + a) / a ~ 0.78 of its covalent-
    // radius sum R_ab - anomalously short bonds (e.g. He2) leave a small
    // SSF tail off the plateau; the molecular grid renormalizes the
    // weights over the partition of unity anyway. The condition compares
    // distance[a] against the nearest OTHER distance, so it can only fire
    // for the globally nearest atom - one O(N) argmin + second-min pass
    // replaces the per-atom O(N^2) scan (grid-3).
    std::size_t nearestAtom = 0;

    for (std::size_t b = 1; b < atomCount; ++b)
    {
        if (distance[b] < distance[nearestAtom])
        {
            nearestAtom = b;
        }
    }

    double secondNearest = std::numeric_limits<double>::infinity();

    for (std::size_t b = 0; b < atomCount; ++b)
    {
        if (b != nearestAtom && distance[b] < secondNearest)
        {
            secondNearest = distance[b];
        }
    }

    if (distance[nearestAtom] <= 0.5 * (1.0 - kSwitchWindow) * secondNearest)
    {
        std::vector<double> raw(atomCount, 0.0);
        raw[nearestAtom] = 1.0;
        return raw;
    }

    std::vector<double> raw(atomCount, 1.0);

    // The per-atom Bragg-Slater radii, hoisted out of the pair loop
    // (RadiusOf is a pure table scan; the values are identical).
    std::vector<double> radius(atomCount);

    for (std::size_t a = 0; a < atomCount; ++a)
    {
        radius[a] = RadiusOf(atoms[a].atomicNumber);
    }

    // Exact distance cutoff.  raw[a] is the product over b != a of
    // s((r_a - r_b)/R_ab), and the SSF switch is EXACTLY 0 on the mu >= a
    // plateau and EXACTLY 1 on the mu <= -a plateau (a = 0.64, the
    // partition's own decay scale).  So:
    //  * once (r_a - r_n)/R_an >= a for the NEAREST atom n, the factor
    //    (a, n) is exactly 0 and raw[a] = 0 exactly - the weight vanishes
    //    analytically, and the whole a-loop is skipped;
    //  * inside the loop, factors on the mu <= -a plateau are exactly 1
    //    (raw[a] * 1 == raw[a], so the multiply is skipped), and a factor
    //    on the mu >= a plateau zeroes the product (0 * x == 0, so the
    //    remaining factors are skipped too).
    // Each skipped factor is bit-identical to its unskipped evaluation
    // (the conditions reuse the exact mu expression the switch would see),
    // so the cutoff changes nothing numerically.  The atoms that survive
    // are those within r_n + a*R_an of the point - at most ~0.64 * 2 * 2.35
    // A ~ 5.7 bohr beyond the nearest atom for the largest tabulated
    // radius (Rb) - so the per-point pair loop never sees atoms beyond
    // the cutoff (the O(nAtoms^2) scan becomes O(k^2) over the atoms
    // inside it, the A57 painted corner).
    const double radiusNearest = radius[nearestAtom];

    for (std::size_t a = 0; a < atomCount; ++a)
    {
        // Never fires for a == nearestAtom (mu == 0), so no guard needed.
        const double muNearest =
            (distance[a] - distance[nearestAtom]) / (radius[a] + radiusNearest);

        if (muNearest >= kSwitchWindow)
        {
            raw[a] = 0.0;
            continue;
        }

        for (std::size_t b = 0; b < atomCount; ++b)
        {
            if (b == a)
            {
                continue;
            }

            // radius[] never holds 0 (untabulated elements fall back to
            // the 1.0 A default), so rAb > 0 here and no zero-divide guard
            // is needed.
            const double rAb = radius[a] + radius[b];
            const double mu = (distance[a] - distance[b]) / rAb;

            if (mu <= -kSwitchWindow)
            {
                continue; // s(mu) == 1 exactly: raw[a] * 1 == raw[a].
            }

            if (mu >= kSwitchWindow)
            {
                raw[a] = 0.0; // s(mu) == 0 exactly: the product stays 0.
                break;
            }

            ++switchCount;
            raw[a] *= Switch(mu);
        }
    }

    return raw;
}

std::vector<double> BeckePartitionWeights(const std::array<double, 3>& pointBohr,
                                          const qcx::molecule::Molecule& molecule) {
    std::vector<double> raw = BeckeRawWeights(pointBohr, molecule);
    double sum = 0.0;

    for (const double w : raw)
    {
        sum += w;
    }

    if (sum <= 0.0)
    {
        // Degenerate fallback (never expected with the SSF plateau):
        // share the point equally.
        for (double& w : raw)
        {
            w = 1.0 / static_cast<double>(raw.size());
        }

        return raw;
    }

    for (double& w : raw)
    {
        w /= sum;
    }

    return raw;
}

} // namespace qcx::grid
