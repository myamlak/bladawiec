#pragma once

// Layout conventions of the matrix-form McMurchie-Davidson engine
// (the numerical scheme is derived in md_vrr.hpp).
//
// Index spaces, all zero-based:
//  - Cartesian components of a shell l: z-slowest (p = x,y,z; d = xx,xy,xz,
//    yy,yz,zz; ...) - kCartesianIndices.
//  - Spherical functions of a shell l: m ascending -l..+l - cos(m phi) for
//    m >= 0, sin(m phi) for m < 0 (the Racah-normalized real solid harmonics
//    of kSolidHarmonicG).
//  - 2D-Hermite index of (tx, ty), total n = tx + ty: n(n+1)/2 + tx
//    (Hermite2DIndex).
//  - 3D-Hermite sub-index within a fixed total n: entries (tx, ty, tz) with
//    tx + ty + tz = n ordered by ty then tz (tx determined) - SubIndex3.
//  - VRR slice T holds every [t]^(m) with |t| + m = T: tier n (|t| = n)
//    starts at kH2Prefix[n], within-tier layout per SubIndex3.
//
// Every count below is bounded by the generated tables (md_tables_gen.hpp).
// The layout consumers assert their ranges in Debug builds: the VRR group
// layout of ComputeEriClassImpl (md_vrr.hpp) checks that every task block
// lies inside its group's span, and the slice/prevSlice ping-pong buffers
// are std::array, bounds-checked under MSVC _STL_VERIFY.

#include "md_tables_gen.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>

namespace qcx::integrals::internal {

/// Number of Cartesian components of a shell of angular momentum l.
/// \param l Angular momentum, 0..kMaxShellL.
/// \returns (l+1)(l+2)/2.
inline constexpr int CartesianCount(int l) noexcept {
    return (l + 1) * (l + 2) / 2;
}

/// The unit normalization of one primitive of a spherical shell beyond the
/// The (2a/pi)^(3/4) radial factor: the solid-harmonic tables map to
/// angular-orthonormal harmonics for l >= 1 (l = 0 is the plain Gaussian,
/// already unit-normalized under that factor), so the radial solid-harmonic
/// restores the standard unit norm:
///   N_l = 2^(l+1) sqrt(pi a^l / (2l+1)!!),  N_0 = 1.
/// \param l Shell angular momentum, 0..kMaxShellL.
/// \param exponent The primitive exponent a.
/// \returns The multiplicative normalization.
inline double SolidNormalization(int l, double exponent) noexcept {
    if (l == 0)
    {
        return 1.0;
    }

    constexpr double kDoubleFactorial[7] = {1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0};
    return std::pow(2.0, l + 1) *
           std::sqrt(std::numbers::pi * std::pow(exponent, l) / kDoubleFactorial[l]);
}

/// Number of spherical functions of a shell of angular momentum l.
/// \param l Angular momentum, 0..kMaxShellL.
/// \returns 2l + 1.
inline constexpr int SphericalCount(int l) noexcept {
    return 2 * l + 1;
}

/// Number of functions of one shell: contraction rows x angular components.
/// \param angularMomentum Shell l, 0..kMaxShellL.
/// \param isSpherical Whether the shell is spherical (true) or Cartesian.
/// \param contractionRows Rows in Shell::coefficients.
/// \returns The function count of the shell.
inline constexpr std::size_t FunctionCount(int angularMomentum,
                                           bool isSpherical,
                                           std::size_t contractionRows) noexcept {
    return contractionRows * static_cast<std::size_t>(isSpherical
                                                          ? SphericalCount(angularMomentum)
                                                          : CartesianCount(angularMomentum));
}

/// Number of 2D-Hermite functions (tx, ty) with tx + ty <= total.
/// \param total Total angular momentum, 0..2*kMaxShellL.
/// \returns (total+1)(total+2)/2.
inline constexpr int Hermite2DCount(int total) noexcept {
    return (total + 1) * (total + 2) / 2;
}

/// Number of 3D-Hermite functions (tx, ty, tz) with tx + ty + tz <= total.
/// \param total Total angular momentum, 0..2*kMaxShellL.
/// \returns (total+1)(total+2)(total+3)/6.
inline constexpr int Hermite3DCount(int total) noexcept {
    return (total + 1) * (total + 2) * (total + 3) / 6;
}

/// Flat index of the 2D-Hermite pair (tx, ty): total-major, then tx.
/// \param tx Hermite order along x.
/// \param ty Hermite order along y.
/// \returns The index n(n+1)/2 + tx with n = tx + ty.
inline constexpr int Hermite2DIndex(int tx, int ty) noexcept {
    const int n = tx + ty;
    return n * (n + 1) / 2 + tx;
}

/// Sub-index of the 3D-Hermite triple within its fixed-total tier.
/// \param ty Hermite order along y.
/// \param tz Hermite order along z.
/// \param total tx + ty + tz (tx is implied).
/// \returns ty*(total+1) - ty*(ty-1)/2 + tz - the ty-major order used by
/// the VRR slice tiers.
inline constexpr int SubIndex3(int ty, int tz, int total) noexcept {
    return ty * (total + 1) - ty * (ty - 1) / 2 + tz;
}

/// Flat index of the 3D-Hermite triple (tx, ty, tz): total-major, then the
/// tier sub-index - the layout of the VRR slice buffers and the transform
/// columns.
/// \param tx Hermite order along x.
/// \param ty Hermite order along y.
/// \param tz Hermite order along z.
/// \returns kH2Prefix[n] + SubIndex3(ty, tz, n) with n = tx + ty + tz.
inline constexpr int Hermite3DIndex(int tx, int ty, int tz) noexcept {
    const int n = tx + ty + tz;
    return kH2Prefix[n] + SubIndex3(ty, tz, n);
}

/// Sanity: the generated tables match the closed forms above.
static_assert(kCartesianCount[6] == CartesianCount(6));
static_assert(kSphericalCount[6] == SphericalCount(6));
static_assert(kHermite2DCount[12] == Hermite2DCount(12));
static_assert(kHermite3DCount[12] == Hermite3DCount(12));

} // namespace qcx::integrals::internal
