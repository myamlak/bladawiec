#include "alkane_sto3g.hpp"

#include "h2_sto3g.hpp"
#include "qcx/error.hpp"
#include "qcx/memory/tensor.hpp"

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace qcx::testing {

namespace {
// Bohr per Angstrom (CODATA 2018).
constexpr double kBohrPerAngstrom = 1.889726125457828;
} // namespace

qcx::Result<qcx::basisset::BasisSet> MakeAlkaneSto3gBasis() {
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

    // Merge mutates *carbon in place and returns the error channel.
    auto mergeResult = carbon->Merge(*hydrogen);

    if (!mergeResult.has_value())
    {
        return std::unexpected(mergeResult.error());
    }

    return std::move(*carbon);
}

qcx::Result<qcx::molecule::Molecule> MakeAlkaneSto3g(std::size_t carbonCount) {
    if (carbonCount == 0)
    {
        return std::unexpected(
            qcx::Error{qcx::ErrorCode::kInvalidArgument, "an alkane needs at least one carbon"});
    }

    const std::size_t hydrogenCount = 2 * carbonCount + 2;
    std::vector<qcx::molecule::Atom> atoms;
    atoms.reserve(carbonCount + hydrogenCount);

    for (std::size_t i = 0; i < carbonCount; ++i)
    {
        atoms.emplace_back("C", 6, 0.0);
    }

    for (std::size_t i = 0; i < hydrogenCount; ++i)
    {
        atoms.emplace_back("H", 1, 0.0);
    }

    auto coordinates = qcx::memory::Tensor<double, 2, qcx::backend::CpuTag>::Create(
        {carbonCount + hydrogenCount, 3});

    if (!coordinates.has_value())
    {
        return std::unexpected(coordinates.error());
    }

    // The tetrahedral angle tau = arccos(-1/3) = 109.47 deg. The backbone
    // C-C-C angle IS tau, so successive C-C bonds turn by 180 - tau =
    // 70.53 deg: the bond vector components along the chain (x) and the
    // zigzag (y) are d*sqrt(2/3) and d/sqrt(3) (cos and sin of 35.26
    // deg). With the backbone at the tetrahedral angle, the internal
    // carbons' C-H directions are EXACTLY tetrahedral (see the internal
    // loop below) and the terminal fans are the exact tetrahedral 3-fan.
    const double kTetrahedral = std::acos(-1.0 / 3.0);
    const double kOutOfPlane = std::sqrt(2.0 / 3.0);
    const double kCC = 2.9066; // C-C 1.538 A, Bohr.
    const double kCH = 1.09 * kBohrPerAngstrom; // C-H 1.09 A, Bohr.
    const double stepX = kCC * kOutOfPlane;
    const double stepY = kCC / std::sqrt(3.0);

    // Carbons: the planar trans zigzag in the xy plane (odd carbons at
    // +stepY - the bond direction alternates +y, -y).
    for (std::size_t i = 0; i < carbonCount; ++i)
    {
        (*coordinates)(static_cast<Eigen::Index>(i), 0) = static_cast<double>(i) * stepX;
        (*coordinates)(static_cast<Eigen::Index>(i), 1) = (i % 2 == 1) ? stepY : 0.0;
        (*coordinates)(static_cast<Eigen::Index>(i), 2) = 0.0;
    }

    // The hydrogen rows follow the carbons, one placeHydrogen call each.
    std::size_t row = carbonCount;

    // (x, y, z) are the hydrogen's offset from its carbon in Bohr - one
    // position triple, always passed together.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    auto placeHydrogen = [&](std::size_t carbon, double x, double y, double z) {
        (*coordinates)(static_cast<Eigen::Index>(row), 0) =
            (*coordinates)(static_cast<Eigen::Index>(carbon), 0) + x;
        (*coordinates)(static_cast<Eigen::Index>(row), 1) =
            (*coordinates)(static_cast<Eigen::Index>(carbon), 1) + y;
        (*coordinates)(static_cast<Eigen::Index>(row), 2) = z;
        ++row;
    };

    if (carbonCount == 1)
    {
        // Methane: the four exact tetrahedral directions.
        const std::array<std::array<double, 3>, 4> directions = {
            {{{1.0, 1.0, 1.0}}, {{-1.0, -1.0, 1.0}}, {{-1.0, 1.0, -1.0}}, {{1.0, -1.0, -1.0}}}};
        const double scale = kCH / std::sqrt(3.0);

        for (const auto& direction : directions)
        {
            placeHydrogen(0, scale * direction[0], scale * direction[1], scale * direction[2]);
        }
    } else
    {
        // Terminal C_0: the three-H fan at the tetrahedral angle from the
        // first bond, spread at 120 deg around it in the plane
        // perpendicular to the bond (w1 in the xy plane, w2 = z).
        auto placeTerminalFan = [&](std::size_t carbon, double ux, double uy) {
            const double w1x = -uy;
            const double w1y = ux;

            for (std::size_t k = 0; k < 3; ++k)
            {
                const double phi = 2.0 * std::numbers::pi * static_cast<double>(k) / 3.0;
                placeHydrogen(carbon,
                              kCH * (std::cos(kTetrahedral) * ux +
                                     std::sin(kTetrahedral) * std::cos(phi) * w1x),
                              kCH * (std::cos(kTetrahedral) * uy +
                                     std::sin(kTetrahedral) * std::cos(phi) * w1y),
                              kCH * std::sin(kTetrahedral) * std::sin(phi));
            }
        };

        const double firstUx = stepX / kCC;
        const double firstUy = stepY / kCC;
        placeTerminalFan(0, firstUx, firstUy);

        // Internal carbons: the exact tetrahedral pair. With a and c the
        // unit bond directions from the carbon, h = -(a + c)/2 +/-
        // sqrt(2/3) * zhat satisfies h.a = h.c = cos(tau) and
        // h+.h- = cos(tau) - a perfect tetrahedron around the carbon
        // (possible because the backbone angle is tau, so |a + c|^2 =
        // 4/3). The two H's mirror across the backbone plane.
        for (std::size_t i = 1; i + 1 < carbonCount; ++i)
        {
            const double ax = ((*coordinates)(static_cast<Eigen::Index>(i - 1), 0) -
                               (*coordinates)(static_cast<Eigen::Index>(i), 0)) /
                              kCC;
            const double ay = ((*coordinates)(static_cast<Eigen::Index>(i - 1), 1) -
                               (*coordinates)(static_cast<Eigen::Index>(i), 1)) /
                              kCC;
            const double cx = ((*coordinates)(static_cast<Eigen::Index>(i + 1), 0) -
                               (*coordinates)(static_cast<Eigen::Index>(i), 0)) /
                              kCC;
            const double cy = ((*coordinates)(static_cast<Eigen::Index>(i + 1), 1) -
                               (*coordinates)(static_cast<Eigen::Index>(i), 1)) /
                              kCC;

            for (const double sign : {1.0, -1.0})
            {
                placeHydrogen(i,
                              kCH * (-0.5 * (ax + cx)),
                              kCH * (-0.5 * (ay + cy)),
                              kCH * sign * kOutOfPlane);
            }
        }

        // Terminal C_{n-1}: the three-H fan around the LAST bond.
        const std::size_t last = carbonCount - 1;
        const double lastUx = ((*coordinates)(static_cast<Eigen::Index>(last - 1), 0) -
                               (*coordinates)(static_cast<Eigen::Index>(last), 0)) /
                              kCC;
        const double lastUy = ((*coordinates)(static_cast<Eigen::Index>(last - 1), 1) -
                               (*coordinates)(static_cast<Eigen::Index>(last), 1)) /
                              kCC;
        placeTerminalFan(last, lastUx, lastUy);
    }

    coordinates->MarkHostDirty();

    return qcx::molecule::Molecule::Create(std::move(atoms), std::move(*coordinates), 0, 1);
}

} // namespace qcx::testing
