// The effective-core-potential mathematics, pinned before the operator exists.
//
// The semi-local ECP operator is
//
//     V = V_loc(r_C) + sum_lm |Y_lm><Y_lm| [V_l(r_C) - V_loc(r_C)],
//
// with a local term V = sum_k c_k r_C^{n_k} exp(-zeta_k r_C^2), the radial
// coordinate measured from the ECP centre C. The ECP data of the vendored
// corpus is [Weigend2005], [Andrae1990], [Peterson2003]. Three pieces are
// independent of the operator and are pinned here with their own reference
// implementation:
//
//   1. the n = -2 radial base function G_0(T), which a Gaussian-basis ECP has
//      never needed in this tree;
//   2. the one- and three-centre local integrals, against direct numerical
//      quadrature of the original three-dimensional integral;
//   3. the solid-harmonic projector: its angular selection rule, its
//      completeness, and its invariance under a change of the spanning basis.
//
// A fourth cell pins the long-range tail convention on the data's own local
// terms; the operator-level version of that sum rule is a skipped cell at the
// end, because no ECP matrix builder exists yet. Until one does, the
// functions in this file are the reference oracle and the cells are the
// behaviour the builder must reproduce. When it lands, each cell must be
// repointed at it and the reference implementation deleted.
//
// Derivation of the radial base functions. For a local term
//
//     c |r - C|^n exp(-zeta |r - C|^2)
//
// against normalized s Gaussians exp(-a|r - A|^2), exp(-b|r - B|^2), the
// Gaussian product theorem gives p = a + b, P = (a A + b B) / p and
//
//     chi_A chi_B = N_A N_B exp(-a b |A - B|^2 / p) exp(-p |r - P|^2).
//
// Combining with the ECP Gaussian at C gives q = p + zeta,
// Q = (p P + zeta C) / q, W = p zeta |P - C|^2 / q and
//
//     integral = c N_A N_B exp(-a b |A-B|^2 / p) exp(-W) J_n(q, R),
//     J_n(q, R) = int exp(-q |r - Q|^2) |r - C|^n d^3r,  R = |Q - C|.
//
// The five closed forms, with T = q R^2 and the Boys function
// F_0(T) = int_0^1 exp(-T t^2) dt:
//
//     J_-2 = (2 pi^(3/2) / sqrt(q)) G_0(T)          <- the new function
//     J_-1 = (2 pi / q) F_0(T)                     <- the Boys base case
//     J_0  = (pi / q)^(3/2)
//     J_1  = (2 pi / q)(R^2 + 1/(2q)) F_0(T) + (pi / q^2) exp(-T)
//     J_2  = (pi / q)^(3/2) (R^2 + 3/(2q))
//
// G_0 is the n = -2 base function:
//
//     G_0(T) = int_0^1 exp(-T (1 - t^2)) dt = e^(-T) int_0^1 e^(T t^2) dt
//            = sqrt(pi) / (2 sqrt(T)) e^(-T) erfi(sqrt(T)),  G_0(0) = 1.
//
// Its series at the origin and its asymptotic series at large T are
//
//     G_0(T) = sum_k (-1)^k (2T)^k / (2k+1)!!
//            = 1 - 2T/3 + 4T^2/15 - 8T^3/105 + ...
//     G_0(T) ~ sum_k (2k-1)!! / (2^(k+1) T^(k+1))
//            = 1/(2T) + 1/(4T^2) + 3/(8T^3) + 15/(16T^4) + ...
//
// Relation to the Boys function: both are Laplace transforms over the unit
// interval,
//
//     F_N(T) = (1/2) int_0^1 s^(N-1/2) exp(-T s) ds,
//     G_N(T) = int_0^1 t^(2N) exp(-T(1-t^2)) dt
//            = (1/2) int_0^1 (1-s)^(N-1/2) exp(-T s) ds,
//
// so G_0 is the mirrored endpoint of F_0 and equals the Boys function
// continued to a negative argument, G_0(T) = exp(-T) F_0(-T). There is no
// algebraic identity at a positive argument - erf and erfi are different
// functions - which is why the n = -2 route needs a function of its own. The
// recurrence G_{N+1} = 1/(2T) - (2N+1)/(2T) G_N follows from one integration
// by parts; only G_0 is needed by the local channel, whose data terms all
// carry n = 2 in every vendored ECP family.
//
// The reference values in the table below were computed with mpmath 1.4.1 at
// 30-digit precision and rounded to double; they are the independent anchor
// for the series and asymptotic tests.

#include "qcx/integrals/boys.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <vector>

namespace {

using Vec3 = std::array<double, 3>;

constexpr double kPi = std::numbers::pi;

/// Gauss-Legendre nodes and weights on [-1, 1].
struct Quadrature {
    std::vector<double> nodes;
    std::vector<double> weights;
};

/// The nodal abscissae and weights of the n-point Gauss-Legendre rule.
/// \param n The number of nodes.
/// \returns The rule on [-1, 1].
Quadrature GaussLegendre(int n) {
    Quadrature rule;
    rule.nodes.resize(static_cast<std::size_t>(n));
    rule.weights.resize(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i)
    {
        // The Chebyshev-like initial guess; Newton converges quadratically.
        double z = std::cos(kPi * (static_cast<double>(i) + 0.75) / (static_cast<double>(n) + 0.5));

        for (int iteration = 0; iteration < 100; ++iteration)
        {
            double previous = 1.0;
            double current = z;

            for (int k = 2; k <= n; ++k)
            {
                const double next =
                    ((2.0 * k - 1.0) * z * current - (k - 1.0) * previous) / static_cast<double>(k);
                previous = current;
                current = next;
            }

            const double derivative =
                static_cast<double>(n) * (z * current - previous) / (z * z - 1.0);
            const double step = current / derivative;
            z -= step;

            if (std::abs(step) < 1e-15)
            {
                break;
            }
        }

        double previous = 1.0;
        double current = z;

        for (int k = 2; k <= n; ++k)
        {
            const double next =
                ((2.0 * k - 1.0) * z * current - (k - 1.0) * previous) / static_cast<double>(k);
            previous = current;
            current = next;
        }

        const double derivative = static_cast<double>(n) * (z * current - previous) / (z * z - 1.0);
        rule.nodes[static_cast<std::size_t>(i)] = z;
        rule.weights[static_cast<std::size_t>(i)] = 2.0 / ((1.0 - z * z) * derivative * derivative);
    }

    return rule;
}

/// The Boys function F_0(T) = int_0^1 exp(-T t^2) dt, through the module's
/// certified kernel.
/// \param t The argument, non-negative.
/// \returns F_0(t).
double BoysZero(double t) {
    return qcx::integrals::BoysSingle(0, t);
}

/// The n = -2 radial base function G_0(T) = int_0^1 exp(-T(1-t^2)) dt.
///
/// The series at the origin has alternating terms, so it is evaluated only
/// where it is well conditioned (T <= 64); above that the asymptotic series,
/// whose truncation error is below 1e-20 at the switch, is used.
/// \param t The argument, non-negative.
/// \returns G_0(t).
double RadialG0(double t) {
    if (t <= 0.0)
    {
        return 1.0;
    }

    if (t <= 64.0)
    {
        // G_0(T) = e^(-T) sum_k T^k / (k! (2k+1)).
        double sum = 0.0;
        double term = 1.0;

        for (int k = 0; k <= 400; ++k)
        {
            const double add = term / static_cast<double>(2 * k + 1);
            sum += add;

            if (std::abs(add) <= 1e-18 * std::abs(sum))
            {
                break;
            }

            term *= t / static_cast<double>(k + 1);
        }

        return std::exp(-t) * sum;
    }

    // G_0(T) ~ sum_k (2k-1)!! / (2^(k+1) T^(k+1)).
    double series = 0.0;
    double term = 1.0;

    for (int k = 0; k <= 32; ++k)
    {
        series += term;
        term *= static_cast<double>(2 * k + 1) / (2.0 * t);
    }

    return series / (2.0 * t);
}

/// The base radial integral J_n(q, R) of the three-centre reduction.
/// \param n The power of |r - C| in the local term, in {-2, ..., 2}.
/// \param q The combined Gaussian exponent.
/// \param r The distance |Q - C|.
/// \returns The closed form for the five integer powers the data can carry.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- n selects the family.
double RadialIntegralJn(int n, double q, double r) {
    const double t = q * r * r;

    switch (n)
    {
    case -2:
        return 2.0 * std::pow(kPi, 1.5) / std::sqrt(q) * RadialG0(t);
    case -1:
        return 2.0 * kPi / q * BoysZero(t);
    case 0:
        return std::pow(kPi / q, 1.5);
    case 1:
        return 2.0 * kPi / q * (r * r + 1.0 / (2.0 * q)) * BoysZero(t) +
               kPi / (q * q) * std::exp(-t);
    case 2:
        return std::pow(kPi / q, 1.5) * (r * r + 3.0 / (2.0 * q));
    default:
        return std::numeric_limits<double>::quiet_NaN();
    }
}

/// The (2a/pi)^(3/4) normalization of an s Gaussian.
/// \param a The exponent.
/// \returns The normalization constant.
double SGaussianNorm(double a) {
    return std::pow(2.0 * a / kPi, 0.75);
}

/// The one-centre local integral by direct radial quadrature:
/// N_a N_b 4 pi int_0^Rmax r^(n+2) exp(-(a+b+zeta) r^2) dr, with both s
/// Gaussians on the ECP centre. The quadrature is the independent reference;
/// the 256-node rule resolves the Gaussian to machine precision at the
/// exponents below.
/// \param a Exponent of the first s Gaussian.
/// \param b Exponent of the second s Gaussian.
/// \param zeta Exponent of the ECP Gaussian.
/// \param n The power of r in the local term.
/// \returns The integral, in atomic units.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- three exponents of one integral.
double DirectOneCentre(double a, double b, double zeta, int n) {
    const double q = a + b + zeta;
    const double rMax = 10.0 / std::sqrt(q);
    const Quadrature rule = GaussLegendre(256);
    double radial = 0.0;

    for (std::size_t i = 0; i < rule.nodes.size(); ++i)
    {
        const double r = 0.5 * (rule.nodes[i] + 1.0) * rMax;
        const double weight = 0.5 * rule.weights[i] * rMax;
        radial += weight * std::pow(r, n + 2) * std::exp(-q * r * r);
    }

    return SGaussianNorm(a) * SGaussianNorm(b) * 4.0 * kPi * radial;
}

/// The three-centre local integral by direct quadrature in spherical
/// coordinates about the ECP centre C:
/// int chi_A chi_B |r - C|^n exp(-zeta |r - C|^2) d^3r.
///
/// The volume element's r^2 and the local power combine into r^(n+2), so the
/// n = -2 integrand is analytic at the origin and the product rule converges
/// spectrally. 120 radial x 24 polar x 32 azimuthal nodes put the floor near
/// 1e-13 at the exponents below.
/// \param a Exponent of the s Gaussian at A.
/// \param b Exponent of the s Gaussian at B.
/// \param centreA First Gaussian centre, Bohr.
/// \param centreB Second Gaussian centre, Bohr.
/// \param ecpCentre ECP centre C, Bohr.
/// \param zeta Exponent of the ECP Gaussian.
/// \param n The power of |r - C| in the local term.
/// \returns The integral, in atomic units.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- geometry plus exponents.
double DirectThreeCentre(double a,
                         double b,
                         const Vec3& centreA,
                         const Vec3& centreB,
                         const Vec3& ecpCentre,
                         double zeta,
                         int n) {
    constexpr double kRadialMax = 9.0;
    constexpr int kRadialNodes = 120;
    constexpr int kPolarNodes = 24;
    constexpr int kAzimuthalNodes = 32;

    const Quadrature radial = GaussLegendre(kRadialNodes);
    const Quadrature polar = GaussLegendre(kPolarNodes);
    const double azimuthalWeight = 2.0 * kPi / static_cast<double>(kAzimuthalNodes);
    const Vec3 displacementA{
        ecpCentre[0] - centreA[0], ecpCentre[1] - centreA[1], ecpCentre[2] - centreA[2]};
    const Vec3 displacementB{
        ecpCentre[0] - centreB[0], ecpCentre[1] - centreB[1], ecpCentre[2] - centreB[2]};
    double total = 0.0;

    for (std::size_t ir = 0; ir < radial.nodes.size(); ++ir)
    {
        const double u = 0.5 * (radial.nodes[ir] + 1.0) * kRadialMax;
        const double radialWeight = 0.5 * radial.weights[ir] * kRadialMax;
        const double uFactor = std::pow(u, n + 2) * std::exp(-zeta * u * u);

        for (std::size_t it = 0; it < polar.nodes.size(); ++it)
        {
            const double ct = polar.nodes[it];
            const double st = std::sqrt(1.0 - ct * ct);
            const double polarWeight = radialWeight * polar.weights[it];

            for (int ip = 0; ip < kAzimuthalNodes; ++ip)
            {
                const double phi = 2.0 * kPi * (static_cast<double>(ip) + 0.5) /
                                   static_cast<double>(kAzimuthalNodes);
                const double ux = u * st * std::cos(phi);
                const double uy = u * st * std::sin(phi);
                const double uz = u * ct;
                const double distanceA = (ux + displacementA[0]) * (ux + displacementA[0]) +
                                         (uy + displacementA[1]) * (uy + displacementA[1]) +
                                         (uz + displacementA[2]) * (uz + displacementA[2]);
                const double distanceB = (ux + displacementB[0]) * (ux + displacementB[0]) +
                                         (uy + displacementB[1]) * (uy + displacementB[1]) +
                                         (uz + displacementB[2]) * (uz + displacementB[2]);
                total += polarWeight * azimuthalWeight * uFactor *
                         std::exp(-a * distanceA - b * distanceB);
            }
        }
    }

    return SGaussianNorm(a) * SGaussianNorm(b) * total;
}

/// The G_0 quadrature on [0, 1]: smooth for T <= 20, where the 128-node rule
/// is exact to machine precision.
/// \param t The argument.
/// \returns The quadrature value of G_0(t).
double G0ByUnitIntervalQuadrature(double t) {
    const Quadrature rule = GaussLegendre(128);
    double sum = 0.0;

    for (std::size_t i = 0; i < rule.nodes.size(); ++i)
    {
        const double s = 0.5 * (rule.nodes[i] + 1.0);
        sum += 0.5 * rule.weights[i] * std::exp(-t * (1.0 - s * s));
    }

    return sum;
}

/// The G_0 quadrature for large T, in the variable z = T(1 - t):
/// G_0(T) = T^(-1) int_0^T exp(-2z + z^2/T) dz. The integrand's peak sits at
/// z = 0 with a width of 1/2 independent of T, and the tail past z = 40 is
/// below exp(-48) for every T >= 40, so the 256-node rule on [0, 40] is exact
/// to about 1e-13.
/// \param t The argument, at least 40.
/// \returns The quadrature value of G_0(t).
double G0ByShiftedQuadrature(double t) {
    constexpr double kUpper = 40.0;
    const Quadrature rule = GaussLegendre(256);
    double sum = 0.0;

    for (std::size_t i = 0; i < rule.nodes.size(); ++i)
    {
        const double z = 0.5 * (rule.nodes[i] + 1.0) * kUpper;
        sum += 0.5 * rule.weights[i] * kUpper * std::exp(-2.0 * z + z * z / t);
    }

    return sum / t;
}

// ---- polynomial solid harmonics ----

/// One monomial c x^a y^b z^c of a solid-harmonic polynomial.
struct Monomial {
    int x; ///< Power of x.
    int y; ///< Power of y.
    int z; ///< Power of z.
    double coefficient; ///< The coefficient.
};

/// A polynomial as a list of monomials.
using Polynomial = std::vector<Monomial>;

Polynomial X() {
    return {{1, 0, 0, 1.0}};
}

Polynomial Y() {
    return {{0, 1, 0, 1.0}};
}

Polynomial Z() {
    return {{0, 0, 1, 1.0}};
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- the product commutes.
Polynomial Multiply(const Polynomial& left, const Polynomial& right) {
    Polynomial product;

    for (const Monomial& a : left)
    {
        for (const Monomial& b : right)
        {
            product.push_back(
                Monomial{a.x + b.x, a.y + b.y, a.z + b.z, a.coefficient * b.coefficient});
        }
    }

    return product;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- the sum commutes.
Polynomial Add(const Polynomial& left, const Polynomial& right) {
    Polynomial sum = left;
    sum.insert(sum.end(), right.begin(), right.end());
    return sum;
}

Polynomial Scale(const Polynomial& polynomial, double factor) {
    Polynomial scaled = polynomial;

    for (Monomial& monomial : scaled)
    {
        monomial.coefficient *= factor;
    }

    return scaled;
}

Polynomial operator+(const Polynomial& left, const Polynomial& right) {
    return Add(left, right);
}

Polynomial operator-(const Polynomial& left, const Polynomial& right) {
    return Add(left, Scale(right, -1.0));
}

Polynomial operator*(const Polynomial& left, const Polynomial& right) {
    return Multiply(left, right);
}

Polynomial operator*(double factor, const Polynomial& polynomial) {
    return Scale(polynomial, factor);
}

/// The real solid harmonics of degree l, as polynomials in (x, y, z) that
/// span the 2l+1 dimensional harmonic space of that degree. They are not
/// normalized: the projector below is built from the Gram matrix, so the
/// normalization and phase of each spanning function drop out, which is the
/// convention independence the tests pin.
/// \param l The degree, 0 through 4.
/// \returns The spanning set.
std::vector<Polynomial> SolidHarmonics(int l) {
    switch (l)
    {
    case 0:
        return {{{0, 0, 0, 1.0}}};
    case 1:
        return {X(), Y(), Z()};
    case 2:
        return {X() * Y(),
                Y() * Z(),
                X() * Z(),
                X() * X() - Y() * Y(),
                2.0 * Z() * Z() - X() * X() - Y() * Y()};
    case 3: {
        const Polynomial zSquared = Z() * Z();
        const Polynomial xSquared = X() * X();
        const Polynomial ySquared = Y() * Y();
        return {Z() * (2.0 * zSquared - 3.0 * xSquared - 3.0 * ySquared),
                X() * (4.0 * zSquared - xSquared - ySquared),
                Y() * (4.0 * zSquared - xSquared - ySquared),
                Z() * (xSquared - ySquared),
                X() * Y() * Z(),
                X() * (xSquared - 3.0 * ySquared),
                Y() * (3.0 * xSquared - ySquared)};
    }

    case 4: {
        const Polynomial zSquared = Z() * Z();
        const Polynomial xSquared = X() * X();
        const Polynomial ySquared = Y() * Y();
        const Polynomial radiusSquared = xSquared + ySquared + zSquared;
        const Polynomial sevenZSquaredMinusRadius = 7.0 * zSquared - radiusSquared;
        return {35.0 * zSquared * zSquared - 30.0 * zSquared * radiusSquared +
                    3.0 * radiusSquared * radiusSquared,
                X() * Z() * (7.0 * zSquared - 3.0 * radiusSquared),
                Y() * Z() * (7.0 * zSquared - 3.0 * radiusSquared),
                (xSquared - ySquared) * sevenZSquaredMinusRadius,
                X() * Y() * sevenZSquaredMinusRadius,
                X() * Z() * (xSquared - 3.0 * ySquared),
                Y() * Z() * (3.0 * xSquared - ySquared),
                xSquared * xSquared - 6.0 * xSquared * ySquared + ySquared * ySquared,
                X() * Y() * (xSquared - ySquared)};
    }

    default:
        return {};
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- (x, y, z) is a point.
double Evaluate(const Polynomial& polynomial, double x, double y, double z) {
    double value = 0.0;

    for (const Monomial& monomial : polynomial)
    {
        value += monomial.coefficient * std::pow(x, monomial.x) * std::pow(y, monomial.y) *
                 std::pow(z, monomial.z);
    }

    return value;
}

/// A quadrature on the unit sphere; the product rule is exact for the
/// polynomials of degree at most 8 used below (the azimuthal rule is exact
/// for trigonometric polynomials of degree 64).
struct AngularGrid {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> weight;
};

AngularGrid MakeAngularGrid() {
    constexpr int kPolarNodes = 48;
    constexpr int kAzimuthalNodes = 64;

    const Quadrature polar = GaussLegendre(kPolarNodes);
    const double azimuthalWeight = 2.0 * kPi / static_cast<double>(kAzimuthalNodes);
    AngularGrid grid;
    grid.x.reserve(polar.nodes.size() * kAzimuthalNodes);
    grid.y.reserve(polar.nodes.size() * kAzimuthalNodes);
    grid.z.reserve(polar.nodes.size() * kAzimuthalNodes);
    grid.weight.reserve(polar.nodes.size() * kAzimuthalNodes);

    for (std::size_t ip = 0; ip < polar.nodes.size(); ++ip)
    {
        const double ct = polar.nodes[ip];
        const double st = std::sqrt(1.0 - ct * ct);

        for (int ia = 0; ia < kAzimuthalNodes; ++ia)
        {
            const double phi =
                2.0 * kPi * (static_cast<double>(ia) + 0.5) / static_cast<double>(kAzimuthalNodes);
            grid.x.push_back(st * std::cos(phi));
            grid.y.push_back(st * std::sin(phi));
            grid.z.push_back(ct);
            grid.weight.push_back(polar.weights[ip] * azimuthalWeight);
        }
    }

    return grid;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- the product is symmetric.
double InnerProduct(const Polynomial& left, const Polynomial& right) {
    const AngularGrid grid = MakeAngularGrid();
    double sum = 0.0;

    for (std::size_t i = 0; i < grid.x.size(); ++i)
    {
        sum += grid.weight[i] * Evaluate(left, grid.x[i], grid.y[i], grid.z[i]) *
               Evaluate(right, grid.x[i], grid.y[i], grid.z[i]);
    }

    return sum;
}

/// The Gram matrix of a spanning set, inverted: the projector onto the span is
/// K(Omega, Omega') = q(Omega)^T G^-1 q(Omega').
/// \param basis The spanning set.
/// \returns The inverse Gram matrix.
Eigen::MatrixXd InverseGram(const std::vector<Polynomial>& basis) {
    const int size = static_cast<int>(basis.size());
    Eigen::MatrixXd gram(size, size);

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            gram(i, j) = InnerProduct(basis[static_cast<std::size_t>(i)],
                                      basis[static_cast<std::size_t>(j)]);
        }
    }

    return gram.inverse();
}

double ProjectorKernel(
    const std::vector<Polynomial>& basis,
    const Eigen::MatrixXd& inverseGram,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- the kernel is symmetric.
    const Vec3& left,
    const Vec3& right) {
    Eigen::VectorXd valuesLeft(static_cast<Eigen::Index>(basis.size()));
    Eigen::VectorXd valuesRight(static_cast<Eigen::Index>(basis.size()));

    for (std::size_t i = 0; i < basis.size(); ++i)
    {
        valuesLeft[static_cast<Eigen::Index>(i)] = Evaluate(basis[i], left[0], left[1], left[2]);
        valuesRight[static_cast<Eigen::Index>(i)] =
            Evaluate(basis[i], right[0], right[1], right[2]);
    }

    return valuesLeft.dot(inverseGram * valuesRight);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters) -- degree and argument.
double Legendre(int degree, double x) {
    if (degree == 0)
    {
        return 1.0;
    }

    double previous = 1.0;
    double current = x;

    for (int k = 2; k <= degree; ++k)
    {
        const double next =
            ((2.0 * k - 1.0) * x * current - (k - 1.0) * previous) / static_cast<double>(k);
        previous = current;
        current = next;
    }

    return current;
}

Vec3 Normalized(const Vec3& direction) {
    const double norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] +
                                  direction[2] * direction[2]);
    return {direction[0] / norm, direction[1] / norm, direction[2] / norm};
}

// ---- the Xe and W local channels, for the tail convention ----

/// One Gaussian term c r^n exp(-zeta r^2) of an ECP channel.
struct EcpTerm {
    int power; ///< The power n of r.
    double zeta; ///< The Gaussian exponent.
    double coefficient; ///< The coefficient.
};

/// The local channel of Xe (core 28) in def2-SVP: the `ul` block of
/// data/basis/def2-svp/Xe.nwchem, verbatim.
const std::array<EcpTerm, 4> kXenonLocal{{{2, 20.88155700, -23.08929500},
                                          {2, 20.78344300, -30.07447500},
                                          {2, 5.25338900, -0.28822700},
                                          {2, 5.36118800, -0.38692400}}};

/// The local channel of W (core 60) in def2-TZVP: the `ul` block of
/// data/basis/def2-tzvp/W.nwchem, verbatim.
const std::array<EcpTerm, 1> kTungstenLocal{{{2, 2.25888846, 14.15257947}}};

double GaussianPart(const std::array<EcpTerm, 4>& terms, double r) {
    double value = 0.0;

    for (const EcpTerm& term : terms)
    {
        value += term.coefficient * std::pow(r, term.power) * std::exp(-term.zeta * r * r);
    }

    return value;
}

double GaussianPart(const std::array<EcpTerm, 1>& terms, double r) {
    double value = 0.0;

    for (const EcpTerm& term : terms)
    {
        value += term.coefficient * std::pow(r, term.power) * std::exp(-term.zeta * r * r);
    }

    return value;
}

/// The n = -2 base function's table, computed with mpmath 1.4.1 at 30 digits
/// and rounded to double.
struct G0Reference {
    double t;
    double value;
};

const std::array<G0Reference, 12> kG0Table{{
    {1e-6, 0.9999993333336000},
    {1e-3, 0.99933359992382645},
    {0.1, 0.93592547232883118},
    {0.5, 0.72477845900707633},
    {1.0, 0.53807950691276842},
    {2.0, 0.31999403728270446},
    {5.0, 0.11570508894007744},
    {10.0, 0.053037580992901645},
    {50.0, 0.010103161564918599},
    {100.0, 0.0050253847187598528},
    {500.0, 0.0010010030151059555},
    {1000.0, 0.00050025037594079610},
}};

TEST(EcpRadialTest, G0MatchesTheIndependentReferenceTable) {
    // The anchor is the closed form evaluated outside the tree; every value
    // exercises one of the two evaluation routes (series below 64, asymptotic
    // above it).
    for (const G0Reference& reference : kG0Table)
    {
        SCOPED_TRACE(reference.t);
        EXPECT_NEAR(RadialG0(reference.t) / reference.value, 1.0, 1e-13);
    }
}

TEST(EcpRadialTest, G0SeriesNearTheOriginMatchesTheReferenceValues) {
    // G_0(T) = sum_k (-1)^k (2T)^k/(2k+1)!!; the coefficient formula is the
    // one the implementation must use near the origin. The compared values
    // are the external table, not the implementation's own output.
    for (std::size_t index = 0; index < 4; ++index)
    {
        const G0Reference& reference = kG0Table[index];
        const double t = reference.t;
        SCOPED_TRACE(t);
        double series = 0.0;
        double term = 1.0;

        for (int k = 0; k < 60; ++k)
        {
            series += term;
            term *= -2.0 * t / static_cast<double>(2 * k + 3);
        }

        EXPECT_NEAR(series / reference.value, 1.0, 1e-14);
    }
}

TEST(EcpRadialTest, G0LargeArgumentMatchesTheReferenceValues) {
    // G_0(T) ~ 1/(2T) sum_k (2k-1)!!/(2T)^k. The explicit asymptotic series
    // is compared with the external table, so the coefficient recurrence is
    // pinned independently of the implementation.
    for (std::size_t index = 8; index < kG0Table.size(); ++index)
    {
        const G0Reference& reference = kG0Table[index];
        const double t = reference.t;
        SCOPED_TRACE(t);
        double series = 0.0;
        double term = 1.0;

        for (int k = 0; k < 32; ++k)
        {
            series += term;
            term *= static_cast<double>(2 * k + 1) / (2.0 * t);
        }

        series /= 2.0 * t;
        EXPECT_NEAR(series / reference.value, 1.0, 1e-14);
    }
}

TEST(EcpRadialTest, G0MatchesIndependentQuadrature) {
    // Two quadratures, each on the interval where its integrand is smooth:
    // the unit-interval form below T = 20 and the z-shifted form above it.
    for (const double t : {0.01, 0.5, 2.0, 10.0})
    {
        SCOPED_TRACE(t);
        EXPECT_NEAR(RadialG0(t) / G0ByUnitIntervalQuadrature(t), 1.0, 1e-11);
    }

    for (const double t : {50.0, 200.0, 1000.0})
    {
        SCOPED_TRACE(t);
        EXPECT_NEAR(RadialG0(t) / G0ByShiftedQuadrature(t), 1.0, 1e-11);
    }
}

TEST(EcpRadialTest, JnBaseIntegralsAtCoincidentCentresMatchTheMoments) {
    // At R = 0 the base integrals are elementary Gaussian moments, derived
    // independently of the closed forms under test:
    //   J_-2 = 2 pi^(3/2)/sqrt(q),  J_-1 = 2 pi/q,  J_0 = (pi/q)^(3/2),
    //   J_1 = 2 pi/q^2,             J_2 = (pi/q)^(3/2) 3/(2q).
    // The n = -1 member is the Boys function at T = 0, F_0(0) = 1; the n = -2
    // member is G_0(0) = 1. This is the first check of the reduction's
    // normalization before the off-centre quadrature cells.
    for (const double q : {0.7, 2.3, 6.5})
    {
        SCOPED_TRACE(q);
        EXPECT_NEAR(RadialIntegralJn(-2, q, 0.0), 2.0 * std::pow(kPi, 1.5) / std::sqrt(q), 1e-14);
        EXPECT_NEAR(RadialIntegralJn(-1, q, 0.0), 2.0 * kPi / q, 1e-14);
        EXPECT_NEAR(RadialIntegralJn(0, q, 0.0), std::pow(kPi / q, 1.5), 1e-14);
        EXPECT_NEAR(RadialIntegralJn(1, q, 0.0), 2.0 * kPi / (q * q), 1e-14);
        EXPECT_NEAR(RadialIntegralJn(2, q, 0.0), std::pow(kPi / q, 1.5) * 3.0 / (2.0 * q), 1e-14);
    }
}

TEST(EcpLocalIntegralTest, OneCentreMatchesHighOrderQuadrature) {
    // The one-centre case C = A = B: the local integral is
    // N_a N_b J_n(a+b+zeta, 0). The 256-node radial quadrature is the
    // independent reference, held at 1e-10 relative.
    struct Case {
        double a;
        double b;
        double zeta;
    };

    const std::array<Case, 6> cases{{
        {0.5, 0.8, 0.0},
        {0.5, 0.8, 1.5},
        {1.2, 0.3, 0.0},
        {1.2, 0.3, 1.5},
        {2.0, 1.5, 0.0},
        {2.0, 1.5, 1.5},
    }};

    for (const Case& testCase : cases)
    {
        const double q = testCase.a + testCase.b + testCase.zeta;

        for (int n = -2; n <= 2; ++n)
        {
            SCOPED_TRACE(testCase.a);
            SCOPED_TRACE(testCase.b);
            SCOPED_TRACE(testCase.zeta);
            SCOPED_TRACE(n);

            const double reduced =
                SGaussianNorm(testCase.a) * SGaussianNorm(testCase.b) * RadialIntegralJn(n, q, 0.0);
            const double direct = DirectOneCentre(testCase.a, testCase.b, testCase.zeta, n);
            EXPECT_LT(std::abs(reduced - direct) / std::max(std::abs(direct), 1e-12), 1e-10);
        }
    }
}

TEST(EcpLocalIntegralTest, ThreeCentreMatchesDirectQuadrature) {
    // The full reduction: p = a+b, Q = (pP + zeta C)/(p+zeta), R = |Q-C|,
    // W = p zeta |P-C|^2/(p+zeta), and the prefactor of the Gaussian product
    // theorem. The direct spherical quadrature about C is the independent
    // reference for every n the ECP data can carry.
    const Vec3 centreA{0.0, 0.0, 0.0};
    const Vec3 centreB{1.3, 0.2, -0.4};
    const Vec3 ecpCentre{0.2, 0.6, 0.5};

    struct Case {
        double a;
        double b;
        double zeta;
    };

    const std::array<Case, 6> cases{{
        {0.5, 0.8, 0.4},
        {0.5, 0.8, 1.5},
        {1.2, 0.3, 0.4},
        {1.2, 0.3, 1.5},
        {2.0, 1.5, 0.4},
        {2.0, 1.5, 1.5},
    }};

    for (const Case& testCase : cases)
    {
        const double p = testCase.a + testCase.b;
        const Vec3 productCentre{(testCase.a * centreA[0] + testCase.b * centreB[0]) / p,
                                 (testCase.a * centreA[1] + testCase.b * centreB[1]) / p,
                                 (testCase.a * centreA[2] + testCase.b * centreB[2]) / p};
        const double q = p + testCase.zeta;
        const Vec3 weightedCentre{(p * productCentre[0] + testCase.zeta * ecpCentre[0]) / q,
                                  (p * productCentre[1] + testCase.zeta * ecpCentre[1]) / q,
                                  (p * productCentre[2] + testCase.zeta * ecpCentre[2]) / q};
        const double radius =
            std::sqrt((weightedCentre[0] - ecpCentre[0]) * (weightedCentre[0] - ecpCentre[0]) +
                      (weightedCentre[1] - ecpCentre[1]) * (weightedCentre[1] - ecpCentre[1]) +
                      (weightedCentre[2] - ecpCentre[2]) * (weightedCentre[2] - ecpCentre[2]));
        const double productCentreDistance =
            std::sqrt((productCentre[0] - ecpCentre[0]) * (productCentre[0] - ecpCentre[0]) +
                      (productCentre[1] - ecpCentre[1]) * (productCentre[1] - ecpCentre[1]) +
                      (productCentre[2] - ecpCentre[2]) * (productCentre[2] - ecpCentre[2]));
        const double separation = std::sqrt((centreA[0] - centreB[0]) * (centreA[0] - centreB[0]) +
                                            (centreA[1] - centreB[1]) * (centreA[1] - centreB[1]) +
                                            (centreA[2] - centreB[2]) * (centreA[2] - centreB[2]));
        const double prefactor =
            SGaussianNorm(testCase.a) * SGaussianNorm(testCase.b) *
            std::exp(-testCase.a * testCase.b * separation * separation / p) *
            std::exp(-p * testCase.zeta * productCentreDistance * productCentreDistance / q);

        for (int n = -2; n <= 2; ++n)
        {
            SCOPED_TRACE(testCase.a);
            SCOPED_TRACE(testCase.b);
            SCOPED_TRACE(testCase.zeta);
            SCOPED_TRACE(n);

            const double reduced = prefactor * RadialIntegralJn(n, q, radius);
            const double direct = DirectThreeCentre(
                testCase.a, testCase.b, centreA, centreB, ecpCentre, testCase.zeta, n);
            EXPECT_LT(std::abs(reduced - direct) / std::max(std::abs(direct), 1e-12), 1e-9);
        }
    }
}

TEST(EcpProjectorTest, AngularSelectionRuleZerosEveryProjectorAboveTheFunctionDegree) {
    // A function of angular momentum L placed at the ECP centre is a degree-L
    // solid harmonic times a radial factor. The projector P_l acts on the
    // angular part only, and a degree-l spherical harmonic is orthogonal to
    // every polynomial of degree below l; so the projector contribution
    // vanishes for l > L. The integrals below are those angular overlaps at
    // the 1e-11 floor of the 48 x 64 product rule.
    for (int degreeL = 0; degreeL <= 4; ++degreeL)
    {
        const std::vector<Polynomial> functions = SolidHarmonics(degreeL);

        for (int l = 0; l <= 4; ++l)
        {
            const std::vector<Polynomial> projectors = SolidHarmonics(l);
            double largest = 0.0;

            for (const Polynomial& function : functions)
            {
                for (const Polynomial& projector : projectors)
                {
                    largest = std::max(largest, std::abs(InnerProduct(projector, function)));
                }
            }

            SCOPED_TRACE(degreeL);
            SCOPED_TRACE(l);

            if (l > degreeL)
            {
                // The rule the operator must obey.
                EXPECT_LT(largest, 1e-11);
            } else if (l < degreeL)
            {
                // The same orthogonality in the other direction: a pure
                // degree-L solid harmonic has no lower-degree component.
                EXPECT_LT(largest, 1e-11);
            } else
            {
                // The degree is reached; the test is not trivially zero.
                EXPECT_GT(largest, 1e-3);
            }
        }
    }
}

TEST(EcpProjectorTest, CompletenessSumOfProjectorsIsTheIdentity) {
    // sum_m |Y_lm><Y_lm| is the identity on the degree-l harmonic space. In
    // the non-orthonormal spanning set its kernel is q(Omega)^T G^-1 q(Omega'),
    // which must reproduce q(Omega) for every spanning function - the
    // completeness identity at the 1e-12 the projector is required to hold.
    const AngularGrid grid = MakeAngularGrid();
    const std::array<Vec3, 5> directions{Normalized({1.0, 0.0, 0.0}),
                                         Normalized({0.0, 1.0, 0.0}),
                                         Normalized({1.0, 1.0, 0.0}),
                                         Normalized({1.0, 2.0, 3.0}),
                                         Normalized({-1.0, 0.5, 2.0})};

    for (int l = 0; l <= 4; ++l)
    {
        const std::vector<Polynomial> basis = SolidHarmonics(l);
        const Eigen::MatrixXd inverse = InverseGram(basis);

        for (const Vec3& direction : directions)
        {
            for (const Polynomial& function : basis)
            {
                double projected = 0.0;

                for (std::size_t i = 0; i < grid.x.size(); ++i)
                {
                    const Vec3 point{grid.x[i], grid.y[i], grid.z[i]};
                    projected += grid.weight[i] *
                                 ProjectorKernel(basis, inverse, direction, point) *
                                 Evaluate(function, grid.x[i], grid.y[i], grid.z[i]);
                }

                SCOPED_TRACE(l);
                const double expected =
                    Evaluate(function, direction[0], direction[1], direction[2]);
                EXPECT_NEAR(projected, expected, 1e-12);
            }
        }
    }
}

TEST(EcpProjectorTest, KernelAgreesWithTheLegendreAdditionTheorem) {
    // The completeness identity in closed form:
    // sum_m Y_lm(Omega) Y_lm(Omega') = (2l+1)/(4pi) P_l(cos gamma). Agreement
    // here says the spanning sets are the full harmonic spaces and the Gram
    // inverse carries the correct measure.
    const std::array<Vec3, 5> directions{Normalized({1.0, 0.0, 0.0}),
                                         Normalized({0.0, 1.0, 0.0}),
                                         Normalized({1.0, 1.0, 0.0}),
                                         Normalized({1.0, 2.0, 3.0}),
                                         Normalized({-1.0, 0.5, 2.0})};

    for (int l = 0; l <= 4; ++l)
    {
        const std::vector<Polynomial> basis = SolidHarmonics(l);
        const Eigen::MatrixXd inverse = InverseGram(basis);

        for (const Vec3& left : directions)
        {
            for (const Vec3& right : directions)
            {
                const double cosine = left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
                const double expected = (2.0 * l + 1.0) / (4.0 * kPi) * Legendre(l, cosine);
                SCOPED_TRACE(l);
                EXPECT_NEAR(ProjectorKernel(basis, inverse, left, right), expected, 1e-12);
            }
        }
    }
}

TEST(EcpProjectorTest, KernelIsInvariantUnderASpannedBasisChange) {
    // The projector is an operator, not a matrix: replacing the spanning set
    // by any nonsingular combination leaves the kernel unchanged. This is why
    // the spherical/Cartesian and phase conventions of a particular harmonic
    // basis cannot silently change an ECP projector block.
    const std::vector<Polynomial> standard = SolidHarmonics(1);
    const std::vector<Polynomial> rotated{X() + Y(), Y() + Z(), Z() + X()};
    const Eigen::MatrixXd standardInverse = InverseGram(standard);
    const Eigen::MatrixXd rotatedInverse = InverseGram(rotated);
    const std::array<Vec3, 4> directions{Normalized({1.0, 0.0, 0.0}),
                                         Normalized({1.0, 1.0, 1.0}),
                                         Normalized({0.3, -0.5, 0.8}),
                                         Normalized({-1.0, 2.0, -0.5})};

    for (const Vec3& left : directions)
    {
        for (const Vec3& right : directions)
        {
            EXPECT_NEAR(ProjectorKernel(standard, standardInverse, left, right),
                        ProjectorKernel(rotated, rotatedInverse, left, right),
                        1e-12);
        }
    }
}

TEST(EcpLocalPotentialTest, LocalTailGivesTheValenceChargeAtLongRange) {
    // The corpus stores only the Gaussian terms. The operator must add the
    // +N_core/r tail to the local channel and keep the nuclear attraction at
    // full Z, so that the total one-electron potential is
    //
    //     -Z/r + [N_core/r + sum_k c_k r^n exp(-zeta r^2)]
    //       -> -(Z - N_core)/r.
    //
    // Omitting the tail leaves -Z/r (the plausible wrong answer this cell
    // rejects); using a valence-Z nucleus with a full-Z tail would leave the
    // wrong sign. The terms are the ul blocks of the named corpus files.
    constexpr int kXenonZ = 54;
    constexpr int kXenonCore = 28;
    constexpr int kTungstenZ = 74;
    constexpr int kTungstenCore = 60;

    // The Gaussian part is present, not negligible, at short range.
    EXPECT_LT(GaussianPart(kXenonLocal, 1.0), -1e-3);
    EXPECT_GT(GaussianPart(kTungstenLocal, 1.0), 1e-1);

    for (const double r : {5.0, 10.0, 50.0})
    {
        const double xenonTotal =
            -static_cast<double>(kXenonZ) / r + kXenonCore / r + GaussianPart(kXenonLocal, r);
        const double tungstenTotal = -static_cast<double>(kTungstenZ) / r + kTungstenCore / r +
                                     GaussianPart(kTungstenLocal, r);

        SCOPED_TRACE(r);
        EXPECT_NEAR(r * xenonTotal, -static_cast<double>(kXenonZ - kXenonCore), 1e-10);
        EXPECT_NEAR(r * tungstenTotal, -static_cast<double>(kTungstenZ - kTungstenCore), 1e-10);
    }
}

TEST(EcpOperatorTest, LongRangePotentialSumRule) {
    // TODO(ECP): no ECP matrix builder exists yet, so neither side of the
    // comparison can be built today (the parsed EcpDefinition now carries
    // the radial terms). This cell is satisfied when a local-channel builder
    // exists whose matrix H_ecp carries the +N_core/r tail: for two s
    // Gaussians localized on a centre 20 Bohr from the ECP centre C, with the
    // Gaussian terms below 1e-20,
    //     H_ecp + V_nuc  ==  (Z - N_core)/Z * V_nuc
    // (V_nuc being the full-Z nuclear attraction) at 1e-10 relative, and
    //     H_ecp == -(N_core/Z) V_nuc
    // is the equivalent one-matrix form. The s-function-at-the-centre zero
    // test it pairs with is the projector selection rule above, which already
    // runs.
    GTEST_SKIP() << "ECP operator not implemented: no matrix builder";
}

} // namespace
