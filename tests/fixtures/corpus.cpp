#include "corpus.hpp"

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <functional>
#include <numbers>
#include <utility>
#include <vector>

namespace qcx::symmetry::testing {
namespace {

// Shared Bohr-per-Angstrom factor from the molecule module (single definition
// repo-wide).
using qcx::molecule::kAngstromToBohr;

// Atom order does not matter: Molecule::Create canonically renumbers.
qcx::Result<qcx::molecule::Molecule> MakeMolecule(std::vector<qcx::molecule::Atom> atoms,
                                                  std::vector<std::vector<double>> rowsAngstrom) {
    auto coordinates =
        qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create({rowsAngstrom.size(), 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    for (std::size_t i = 0; i < rowsAngstrom.size(); ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            (*coordinates)(i, d) = rowsAngstrom[i][d] * kAngstromToBohr;
        }
    }

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

qcx::molecule::Atom Atom(const char* symbol, int atomicNumber) {
    return qcx::molecule::Atom{symbol, atomicNumber, 0.0};
}

} // namespace

qcx::Result<qcx::molecule::Molecule> MakeWater() {
    return MakeMolecule({Atom("O", 8), Atom("H", 1), Atom("H", 1)},
                        {{0.0, 0.0, 0.0}, {0.757, 0.586, 0.0}, {-0.757, 0.586, 0.0}});
}

qcx::Result<qcx::molecule::Molecule> MakeEthene() {
    return MakeMolecule(
        {Atom("C", 6), Atom("C", 6), Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1)},
        {{0.665, 0.0, 0.0},
         {-0.665, 0.0, 0.0},
         {1.235, 0.925, 0.0},
         {1.235, -0.925, 0.0},
         {-1.235, 0.925, 0.0},
         {-1.235, -0.925, 0.0}});
}

qcx::Result<qcx::molecule::Molecule> MakeBenzene() {
    std::vector<qcx::molecule::Atom> atoms;
    std::vector<std::vector<double>> rows;

    for (int k = 0; k < 6; ++k)
    {
        const double theta = static_cast<double>(k) * std::numbers::pi / 3.0;
        rows.push_back({1.396 * std::cos(theta), 1.396 * std::sin(theta), 0.0});
        atoms.push_back(Atom("C", 6));
    }

    for (int k = 0; k < 6; ++k)
    {
        const double theta = static_cast<double>(k) * std::numbers::pi / 3.0;
        rows.push_back({2.486 * std::cos(theta), 2.486 * std::sin(theta), 0.0});
        atoms.push_back(Atom("H", 1));
    }

    return MakeMolecule(std::move(atoms), std::move(rows));
}

qcx::Result<qcx::molecule::Molecule> MakeAmmonia() {
    // Full-precision trig: the C3 image must match to ~1e-16, while 4-digit
    // hand-typed coordinates land ~1e-4 off - right at the detector tolerance.
    std::vector<qcx::molecule::Atom> atoms = {Atom("N", 7)};
    std::vector<std::vector<double>> rows = {{0.0, 0.0, 0.0}};

    for (int k = 0; k < 3; ++k)
    {
        const double theta =
            static_cast<double>(k) * 2.0 * std::numbers::pi / 3.0 + std::numbers::pi / 6.0;
        rows.push_back({std::cos(theta), std::sin(theta), 0.4});
        atoms.push_back(Atom("H", 1));
    }

    return MakeMolecule(std::move(atoms), std::move(rows));
}

qcx::Result<qcx::molecule::Molecule> MakeMethane() {
    return MakeMolecule({Atom("C", 6), Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1)},
                        {{0.0, 0.0, 0.0},
                         {0.629, 0.629, 0.629},
                         {0.629, -0.629, -0.629},
                         {-0.629, 0.629, -0.629},
                         {-0.629, -0.629, 0.629}});
}

qcx::Result<qcx::molecule::Molecule> MakeSulfurHexafluoride() {
    return MakeMolecule({Atom("S", 16),
                         Atom("F", 9),
                         Atom("F", 9),
                         Atom("F", 9),
                         Atom("F", 9),
                         Atom("F", 9),
                         Atom("F", 9)},
                        {{0.0, 0.0, 0.0},
                         {1.56, 0.0, 0.0},
                         {-1.56, 0.0, 0.0},
                         {0.0, 1.56, 0.0},
                         {0.0, -1.56, 0.0},
                         {0.0, 0.0, 1.56},
                         {0.0, 0.0, -1.56}});
}

qcx::Result<qcx::molecule::Molecule> MakeCarbonDioxide() {
    return MakeMolecule({Atom("C", 6), Atom("O", 8), Atom("O", 8)},
                        {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.16}, {0.0, 0.0, -1.16}});
}

qcx::Result<qcx::molecule::Molecule> MakeHydrogenFluoride() {
    return MakeMolecule({Atom("F", 9), Atom("H", 1)}, {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.917}});
}

qcx::Result<qcx::molecule::Molecule> MakeTwistedHydrogenPeroxide() {
    // C2 axis = x: (x, y, z) -> (x, -y, -z) swaps the two halves.
    return MakeMolecule(
        {Atom("O", 8), Atom("O", 8), Atom("H", 1), Atom("H", 1)},
        {{0.0, 0.0, 0.73}, {0.0, 0.0, -0.73}, {0.2, 0.95, 0.8}, {0.2, -0.95, -0.8}});
}

qcx::Result<qcx::molecule::Molecule> MakeTransDiazene() {
    return MakeMolecule({Atom("N", 7), Atom("N", 7), Atom("H", 1), Atom("H", 1)},
                        {{0.0, 0.6, 0.0}, {0.0, -0.6, 0.0}, {-0.9, 1.05, 0.0}, {0.9, -1.05, 0.0}});
}

qcx::Result<qcx::molecule::Molecule> MakeAllene() {
    return MakeMolecule({Atom("C", 6),
                         Atom("C", 6),
                         Atom("C", 6),
                         Atom("H", 1),
                         Atom("H", 1),
                         Atom("H", 1),
                         Atom("H", 1)},
                        {{0.0, 0.0, 0.0},
                         {0.0, 0.0, 1.31},
                         {0.0, 0.0, -1.31},
                         {1.05, 0.0, 1.603},
                         {-1.05, 0.0, 1.603},
                         {0.0, 1.05, -1.603},
                         {0.0, -1.05, -1.603}});
}

qcx::Result<qcx::molecule::Molecule> MakeMesoDifluoroethane() {
    // Inversion through the origin pairs every atom: Ci.
    return MakeMolecule({Atom("C", 6),
                         Atom("C", 6),
                         Atom("F", 9),
                         Atom("F", 9),
                         Atom("Cl", 17),
                         Atom("Cl", 17),
                         Atom("H", 1),
                         Atom("H", 1)},
                        {{0.77, 0.0, 0.0},
                         {-0.77, 0.0, 0.0},
                         {1.45, 0.7, 0.75},
                         {-1.45, -0.7, -0.75},
                         {1.35, -0.6, -0.9},
                         {-1.35, 0.6, 0.9},
                         {1.1, -0.9, 0.6},
                         {-1.1, 0.9, -0.6}});
}

qcx::Result<qcx::molecule::Molecule> MakeHypofluorousAcid() {
    return MakeMolecule({Atom("O", 8), Atom("F", 9), Atom("H", 1)},
                        {{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}, {-0.2, 0.95, 0.0}});
}

qcx::Result<qcx::molecule::Molecule> MakeEclipsedEthane() {
    std::vector<qcx::molecule::Atom> atoms = {Atom("C", 6), Atom("C", 6)};
    std::vector<std::vector<double>> rows = {{0.0, 0.0, 0.76}, {0.0, 0.0, -0.76}};

    for (int k = 0; k < 3; ++k)
    {
        const double theta = static_cast<double>(k) * 2.0 * std::numbers::pi / 3.0;
        rows.push_back({std::cos(theta), std::sin(theta), 1.16});
        rows.push_back({std::cos(theta), std::sin(theta), -1.16});
        atoms.push_back(Atom("H", 1));
        atoms.push_back(Atom("H", 1));
    }

    return MakeMolecule(std::move(atoms), std::move(rows));
}

qcx::Result<qcx::molecule::Molecule> MakeStaggeredEthane() {
    std::vector<qcx::molecule::Atom> atoms = {Atom("C", 6), Atom("C", 6)};
    std::vector<std::vector<double>> rows = {{0.0, 0.0, 0.76}, {0.0, 0.0, -0.76}};

    for (int k = 0; k < 3; ++k)
    {
        const double theta = static_cast<double>(k) * 2.0 * std::numbers::pi / 3.0;
        rows.push_back({std::cos(theta), std::sin(theta), 1.16});
        rows.push_back({std::cos(theta + std::numbers::pi / 3.0),
                        std::sin(theta + std::numbers::pi / 3.0),
                        -1.16});
        atoms.push_back(Atom("H", 1));
        atoms.push_back(Atom("H", 1));
    }

    return MakeMolecule(std::move(atoms), std::move(rows));
}

qcx::Result<qcx::molecule::Molecule> MakeS8Crown() {
    std::vector<qcx::molecule::Atom> atoms;
    std::vector<std::vector<double>> rows;

    for (int k = 0; k < 8; ++k)
    {
        const double theta = static_cast<double>(k) * std::numbers::pi / 4.0;
        const double z = (k % 2 == 0) ? 0.6 : -0.6;
        rows.push_back({1.4 * std::cos(theta), 1.4 * std::sin(theta), z});
        atoms.push_back(Atom("S", 16));
    }

    return MakeMolecule(std::move(atoms), std::move(rows));
}

qcx::Result<qcx::molecule::Molecule> MakeGenericC1() {
    // Four atoms, non-planar, all distances distinct: genuinely C1. (Any
    // three-atom molecule is planar, so the in-plane mirror always exists
    // and the best it can do is Cs - a triatomic C1 test molecule is
    // therefore impossible.)
    constexpr double kHydrogenX = 1.0;
    constexpr double kFluorineX = -0.4;
    constexpr double kFluorineY = 0.9;
    constexpr double kFluorineZ = 0.2;
    constexpr double kChlorineX = -0.3;
    constexpr double kChlorineY = -0.5;
    constexpr double kChlorineZ = 1.2;
    return MakeMolecule({Atom("C", 6), Atom("H", 1), Atom("F", 9), Atom("Cl", 17)},
                        {{0.0, 0.0, 0.0},
                         {kHydrogenX, 0.0, 0.0},
                         {kFluorineX, kFluorineY, kFluorineZ},
                         {kChlorineX, kChlorineY, kChlorineZ}});
}

qcx::Result<qcx::molecule::Molecule> MakeD2Synthetic() {
    // The D2 orbit of a generic seed (a, b, c) with |a|, |b|, |c| distinct:
    // the three C2 axes are the coordinate axes and nothing else survives.
    constexpr double kSeedA = 1.0;
    constexpr double kSeedB = 0.7;
    constexpr double kSeedC = 0.4;
    return MakeMolecule({Atom("H", 1), Atom("H", 1), Atom("H", 1), Atom("H", 1)},
                        {{kSeedA, kSeedB, kSeedC},
                         {kSeedA, -kSeedB, -kSeedC},
                         {-kSeedA, kSeedB, -kSeedC},
                         {-kSeedA, -kSeedB, kSeedC}});
}

qcx::Result<qcx::molecule::Molecule> MakeD2dSynthetic() {
    // The D2d orbit pair of two seeds: four Cl atoms ON the C2' axes (the
    // xy diagonals at z = 0 - the atoms a C2' fixes) plus four H atoms in
    // the allene ring pattern (S4 about z maps the +z pair to the -z pair).
    // The S4 axis carries no atom, so the C2 on it fixes none while each
    // C2' fixes its two Cl atoms - the discriminating geometry for the
    // D2-family S4-axis principal rule (its fallback ranks the C2' above
    // the S4-axis C2 on the atom-fixed counts alone).
    return MakeMolecule({Atom("Cl", 17),
                         Atom("Cl", 17),
                         Atom("Cl", 17),
                         Atom("Cl", 17),
                         Atom("H", 1),
                         Atom("H", 1),
                         Atom("H", 1),
                         Atom("H", 1)},
                        {{1.0, 1.0, 0.0},
                         {-1.0, 1.0, 0.0},
                         {-1.0, -1.0, 0.0},
                         {1.0, -1.0, 0.0},
                         {1.4, 0.0, 0.8},
                         {-1.4, 0.0, 0.8},
                         {0.0, 1.4, -0.8},
                         {0.0, -1.4, -0.8}});
}

namespace {

// Closes a seed set under the generators until no new image appears (the
// finite orbit of the generated group). Points duplicate when the orbit
// closes; the dedup tolerance guards the accumulated trig error.
std::vector<std::vector<double>> CloseUnder(
    std::vector<std::vector<double>> points,
    const std::vector<std::function<std::array<double, 3>(const std::array<double, 3>&)>>&
        generators) {
    constexpr double kOrbitDedupTolerance = 1e-10;

    for (std::size_t head = 0; head < points.size(); ++head)
    {
        const std::array<double, 3> seed = {points[head][0], points[head][1], points[head][2]};

        for (const auto& generator : generators)
        {
            const auto image = generator(seed);
            bool seen = false;

            for (const auto& point : points)
            {
                if (std::abs(point[0] - image[0]) < kOrbitDedupTolerance &&
                    std::abs(point[1] - image[1]) < kOrbitDedupTolerance &&
                    std::abs(point[2] - image[2]) < kOrbitDedupTolerance)
                {
                    seen = true;
                    break;
                }
            }

            if (!seen)
            {
                points.push_back({image[0], image[1], image[2]});
            }
        }
    }

    return points;
}

// All-H atoms at the given orbit points.
qcx::Result<qcx::molecule::Molecule> MakeOrbitMolecule(
    const std::vector<std::vector<double>>& points) {
    std::vector<qcx::molecule::Atom> atoms;
    atoms.reserve(points.size());

    for (std::size_t i = 0; i < points.size(); ++i)
    {
        atoms.push_back(Atom("H", 1));
    }

    return MakeMolecule(std::move(atoms), points);
}

} // namespace

namespace {

// Which pure S2n group a synthetic fixture realizes; the point count and
// rotation step are derived from it, so a caller cannot pass them swapped
// (the int/double pair would compile silently either way - the shape
// bugprone-easily-swappable-parameters exists to catch).
enum class S2nKind : std::uint8_t { kS4, kS6, kS8 };

// Pure S2n synthetic: two same-element orbits at radii rA != rB, phases
// offset by half the rotation step, z alternating with k. A SINGLE orbit
// is closed under the containing Dnd group's extra C2' axes (on the orbit
// a C2' acts exactly like the S2n generator), so one orbit alone detects
// as D2d/D3d/D4d; the offset second orbit breaks those extra elements
// while S2n and its powers survive on both.
qcx::Result<qcx::molecule::Molecule> MakeS2nSynthetic(S2nKind kind) {
    constexpr double kInnerRadius = 1.0;
    constexpr double kOuterRadius = 1.6;
    constexpr double kPhaseDegrees = 35.0;
    constexpr double kZ = 0.4;
    const int pointCount = kind == S2nKind::kS4 ? 4 : (kind == S2nKind::kS6 ? 6 : 8);
    const double stepDegrees = kind == S2nKind::kS4 ? 90.0 : (kind == S2nKind::kS6 ? 60.0 : 45.0);
    std::vector<std::vector<double>> rows;

    for (int orbit = 0; orbit < 2; ++orbit)
    {
        const double radius = orbit == 0 ? kInnerRadius : kOuterRadius;
        const double phase0 = kPhaseDegrees + (orbit == 0 ? 0.0 : 0.5 * stepDegrees);

        for (int k = 0; k < pointCount; ++k)
        {
            const double theta =
                (phase0 + static_cast<double>(k) * stepDegrees) * std::numbers::pi / 180.0;
            const double z = (k % 2 == 0) ? kZ : -kZ;
            rows.push_back({radius * std::cos(theta), radius * std::sin(theta), z});
        }
    }

    return MakeOrbitMolecule(rows);
}

} // namespace

qcx::Result<qcx::molecule::Molecule> MakeS4Synthetic() {
    // S4 = C4 about z composed with sigmaH; orbits offset by half of 90 deg.
    return MakeS2nSynthetic(S2nKind::kS4);
}

qcx::Result<qcx::molecule::Molecule> MakeS6Synthetic() {
    // S6 = C6 about z composed with sigmaH; orbits offset by half of 60 deg.
    return MakeS2nSynthetic(S2nKind::kS6);
}

qcx::Result<qcx::molecule::Molecule> MakeS8Synthetic() {
    // S8 = C8 about z composed with sigmaH; orbits offset by half of 45 deg.
    return MakeS2nSynthetic(S2nKind::kS8);
}

qcx::Result<qcx::molecule::Molecule> MakeTSynthetic() {
    // T = <C2(z), C3((1,1,1)/sqrt(3))>: the 12-element orbit of a generic
    // seed (tetrahedron rotations), no mirrors, no inversion.
    const Eigen::Matrix3d c2z =
        Eigen::AngleAxisd(std::numbers::pi, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d c3 =
        Eigen::AngleAxisd(2.0 * std::numbers::pi / 3.0, Eigen::Vector3d(1.0, 1.0, 1.0).normalized())
            .toRotationMatrix();
    const auto apply = [](const Eigen::Matrix3d& m, const std::array<double, 3>& p) {
        const Eigen::Vector3d image = m * Eigen::Vector3d(p[0], p[1], p[2]);
        return std::array<double, 3>{image.x(), image.y(), image.z()};
    };
    return MakeOrbitMolecule(
        CloseUnder({{1.0, 0.7, 0.4}},
                   {[&apply, &c2z](const std::array<double, 3>& p) { return apply(c2z, p); },
                    [&apply, &c3](const std::array<double, 3>& p) { return apply(c3, p); }}));
}

qcx::Result<qcx::molecule::Molecule> MakeThSynthetic() {
    // The T orbit plus its inverted copy: Th (24 atoms), which contains the
    // sigmaH mirrors T x Ci inherits.
    const Eigen::Matrix3d c2z =
        Eigen::AngleAxisd(std::numbers::pi, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::Matrix3d c3 =
        Eigen::AngleAxisd(2.0 * std::numbers::pi / 3.0, Eigen::Vector3d(1.0, 1.0, 1.0).normalized())
            .toRotationMatrix();
    const auto apply = [](const Eigen::Matrix3d& m, const std::array<double, 3>& p) {
        const Eigen::Vector3d image = m * Eigen::Vector3d(p[0], p[1], p[2]);
        return std::array<double, 3>{image.x(), image.y(), image.z()};
    };
    std::vector<std::vector<double>> points =
        CloseUnder({{1.0, 0.7, 0.4}},
                   {[&apply, &c2z](const std::array<double, 3>& p) { return apply(c2z, p); },
                    [&apply, &c3](const std::array<double, 3>& p) { return apply(c3, p); }});

    // Append the inverted copy.
    const std::size_t tCount = points.size();

    for (std::size_t i = 0; i < tCount; ++i)
    {
        points.push_back({-points[i][0], -points[i][1], -points[i][2]});
    }

    return MakeOrbitMolecule(points);
}

} // namespace qcx::symmetry::testing
