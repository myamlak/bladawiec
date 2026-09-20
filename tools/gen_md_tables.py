#!/usr/bin/env python3
"""Generate the constexpr tables of the matrix-form McMurchie-Davidson engine.

Emits integrals/src/internal/md_tables_gen.hpp (pure data tables) and
integrals/src/internal/md_dispatch_gen.hpp (the per-{L_bra, L_ket}-class
function-pointer table, gated on the QcxIntegralsLMax/QcxIntegralsF32 compile
definitions). Requires: `pip install mpmath`.

Every table is derived, not measured:

  - kSolidHarmonicG: the real solid-harmonic Cartesian transforms (Racah
    normalization; rows m ascending -l..+l - cos(m phi) for m >= 0, sin(m phi)
    for m < 0) from the Rodrigues expansion of P_l^m with exact rational
    Cartesian coefficients; the normalization constants at 30 digits (mpmath).
  - kParityMask: E_t^{(ij)} = 0 unless t = i + j (mod 2) per axis - the
    Hermite parity of the (x-A)^i (x-B)^j expansion.
  - kClassAmplification: the certified-mixed-precision amplification constants
    C_class. The VRR used by the engine is
        [t+e]^(m) = -2a(P-Q)_dir [t]^(m+1) - 2a t_dir [t-e]^(m+1)
    (derived in md_vrr.hpp). |d[t]^(0)| <= eps * A[t,0] * (2 sqrt a)^|t| *
    |seed_|t|| with A the product-sum over the recurrence DAG, and each
    geometric step 2a(P-Q)_dir K F_{m+1} is bounded by 2 sqrt(a) * kappa_m *
    K F_m with kappa_m = sup_{x>=0} sqrt(x) F_{m+1}(x) / F_m(x) (finite:
    both tails decay). kClassAmplification = 2 * max_{|t| <= L} A[t,0] -
    the factor 2 is the documented safety margin, and the per-class rounding
    allowance kClassRounding covers the GEMM/representation terms.
    The full per-quartet bound construction lives in md_vrr.hpp.
  - kMdClassTableF64/kMdClassTableF32: class dispatch. Rows above the
    configured Lmax (L_ket > 2*QcxIntegralsLMax) are nullptr by #if - no CI
    configuration ever links the full class matrix.

The committed headers must match this script's output byte for byte; the
regeneration ctest runs `--check`.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from math import comb, factorial

try:
    from mpmath import gamma, gammainc, mp, mpf, exp, pi, sqrt
except ImportError:
    if "--check" in sys.argv:
        # The regeneration ctest: CI lanes without the pip package skip.
        print("SKIP: mpmath is not installed (the --check mode requires it)")
        sys.exit(77)
    raise

mp.dps = 30  # reference error ~1e-25; 4 orders below the double floor 1e-14

MAX_SHELL_L = 6
MAX_TOTAL_L = 2 * MAX_SHELL_L  # 12: the largest per-class total L_bra + L_ket
MAX_VRR_L = 2 * MAX_TOTAL_L  # 24: the largest VRR order for two class totals
MAX_SEED_ORDER = MAX_VRR_L  # seeds F_m up to la+lb+lc+ld


def fmt(v):
    return f"{v:.17e}"


# ---------------------------------------------------------------------------
# Boys functions and the kappa_m constants
# ---------------------------------------------------------------------------

def boys_ref(n, x):
    """F_n(x) via the provably stable series (all terms positive)."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    s = mpf(0)
    term = mpf(1) / (n + mpf("0.5"))
    e = exp(-x) / 2
    for l in range(300):
        s += term
        if l > 20 and term < mpf("1e-28"):
            break
        term *= x / (n + l + mpf("1.5"))
    return e * s


def kappa_m(m):
    """sup_x sqrt(x) F_{m+1}(x) / F_m(x) over x >= 0, via a dense mpmath scan.

    Finite for every m: near 0 both sqrt(x) F_{m+1} and the ratio vanish
    (F_m(0) = 1/(2m+1)); for x -> inf, F_{m+1}/F_m ~ (m+1/2)/(2x), so
    sqrt(x) F_{m+1}/F_m ~ (m+1/2)/(2 sqrt(x)) -> 0. Each point uses the
    closed form F_m(x) = 1/2 x^(-(m+1/2)) gamma(m+1/2, x), so the ratio is
    gamma(m+3/2, x) / (sqrt(x) gamma(m+1/2, x)) - one gammainc pair.
    """
    a = m + mpf("0.5")
    best = mpf(0)
    x = mpf("1e-12")
    # Log-spaced scan 1e-12 .. 1e5 with ~1.3% relative resolution - plenty
    # for a bound constant that carries a 2x safety margin.
    while x < mpf("1e5"):
        value = gammainc(a + 1, 0, x) / (sqrt(x) * gammainc(a, 0, x))
        if value > best:
            best = value
        x *= mpf("1.0305")
    return best


# ---------------------------------------------------------------------------
# Layout conventions (mirrored in md_defs.hpp)
# ---------------------------------------------------------------------------

def hermite2d_count(l):
    return (l + 1) * (l + 2) // 2


def hermite3d_count(l):
    return (l + 1) * (l + 2) * (l + 3) // 6


def cartesian_components(l):
    """Cartesian (ix, iy, iz) per index, descending lexicographic - the
    standard z-slowest order: p = x, y, z; d = xx, xy, xz, yy, yz, zz; ..."""
    out = []
    for ix in range(l, -1, -1):
        for iy in range(l - ix, -1, -1):
            out.append((ix, iy, l - ix - iy))
    return out


# ---------------------------------------------------------------------------
# Real solid harmonics (Racah normalization, rows m = -l..+l: cos(m phi) for
# m >= 0, sin(m phi) for m < 0)
# ---------------------------------------------------------------------------

def angular_moment(a, b, c):
    """int x^a y^b z^c dOmega over the unit sphere: zero unless a, b, c are
    all even, else 2 Gamma((a+1)/2) Gamma((b+1)/2) Gamma((c+1)/2) /
    Gamma((a+b+c+3)/2)."""
    if a % 2 or b % 2 or c % 2:
        return mpf(0)
    return mpf(2) * mp.gamma(mpf(a + 1) / 2) * mp.gamma(mpf(b + 1) / 2) * \
        mp.gamma(mpf(c + 1) / 2) / mp.gamma(mpf(a + b + c + 3) / 2)


def verify_orthonormal(l, rows, carts):
    """G W G^T = I at the working precision (W = the angular monomial
    overlap) - the physics-defining property of the real spherical
    harmonics, independent of how G was constructed."""
    W = [[angular_moment(i[0] + j[0], i[1] + j[1], i[2] + j[2]) for j in carts] for i in carts]
    n = len(rows)
    for m1 in range(n):
        for m2 in range(n):
            value = sum(rows[m1][a] * W[a][b] * rows[m2][b]
                        for a in range(len(carts)) for b in range(len(carts)))
            target = mpf(1) if m1 == m2 else mpf(0)
            if abs(value - target) > mpf(10) ** -28:
                raise SystemExit(f"solid harmonics l={l} m1={m1} m2={m2}: "
                                 f"<Y|Y> = {mp.nstr(value, 30)}, expected {target}")


def solid_harmonic_table(l):
    """G[l][m][cart]: |lm> = sum_cart G[l][m][cart] x^ix y^iy z^iz, cart in
    z-slowest order. Exact rational coefficients x mpmath normalization."""
    carts = cartesian_components(l)
    nc = len(carts)

    def legendre_derivative_coeffs(l, m):
        """Coefficients c_k of d^m/du^m P_l(u) = sum_k c_k u^(l-m-2k)."""
        out = []
        for k in range((l - m) // 2 + 1):
            # P_l(u) = sum_k (-1)^k (2l-2k)! / (2^l k! (l-k)! (l-2k)!) u^(l-2k)
            pcoef = Fraction((-1) ** k * factorial(2 * l - 2 * k),
                             2 ** l * factorial(k) * factorial(l - k) * factorial(l - 2 * k))
            power = l - 2 * k - m
            if power < 0:
                continue
            # d^m/du^m u^(l-2k) = (l-2k)!/(l-2k-m)! u^(l-2k-m); the quotient
            # is exact because factorial(power) divides factorial(l-2k).
            out.append((power, pcoef * (factorial(l - 2 * k) // factorial(power))))
        return out

    def rho_power_coeffs(m, trig):
        """(x+iy)^m expanded: {(ix, iy): Fraction} for the real part (trig =
        "cos") or the imaginary part (trig = "sin")."""
        out = {}
        for j in range(m + 1):
            ix = m - j
            iy = j
            c = comb(m, j)
            if trig == "cos":
                # Re[(x+iy)^m]: even j terms with sign (-1)^(j/2)
                if j % 2 == 0:
                    out[(ix, iy)] = Fraction((-1) ** (j // 2) * c)
            else:
                # Im[(x+iy)^m]: odd j terms with sign (-1)^((j-1)/2)
                if j % 2 == 1:
                    out[(ix, iy)] = Fraction((-1) ** ((j - 1) // 2) * c)
        return out

    # r^l P_l^m(cos t) cos/sin(m phi) = sum over (power, ck) of
    #   ck z^power (x^2+y^2+z^2)^((l-power-m)/2) * (x+iy)^m trig
    # (u = z/r puts r^2 = x^2+y^2+z^2 into the leftover radial power, and
    # rho^m = (x+iy)^m carries the azimuthal part).
    rows = []
    # l = 0 is the identity: the basis convention is that a spherical s
    # function equals the Cartesian s function (both individually
    # normalized); the Racah fold of the real solid harmonics starts at
    # l = 1.
    if l == 0:
        return [[mpf(1)]]

    for m_signed in range(-l, l + 1):
        m = abs(m_signed)
        trig = "cos" if m_signed >= 0 else "sin"
        # normalization: sqrt((2l+1)(l-m)!/(2 pi (l+m)!)) for m > 0,
        # sqrt((2l+1)/(4 pi)) for m = 0.
        if m == 0:
            norm = sqrt(mpf(2 * l + 1) / (4 * pi))
        else:
            norm = sqrt(mpf(2 * l + 1) * factorial(l - m) / (2 * pi * factorial(l + m)))
        row = {cart: Fraction(0) for cart in range(nc)}
        for power, ck in legendre_derivative_coeffs(l, m):
            # z^power * rho^(l-power); l-power-m must be even and >= 0.
            rem = l - power - m
            if rem < 0 or rem % 2 != 0:
                continue
            nR2 = rem // 2
            base = rho_power_coeffs(m, trig)
            for (ix0, iy0), c0 in base.items():
                # Multiply by (x^2+y^2+z^2)^nR2: the trinomial
                # n!/(a!b!c!) x^{2a} y^{2b} z^{2c} over a+b+c = nR2.
                for a in range(nR2 + 1):
                    for b in range(nR2 - a + 1):
                        c = nR2 - a - b
                        ix = ix0 + 2 * a
                        iy = iy0 + 2 * b
                        iz = power + 2 * c
                        cxyz = Fraction(factorial(nR2), factorial(a) * factorial(b) * factorial(c))
                        cart = carts.index((ix, iy, iz))
                        row[cart] += ck * c0 * cxyz
        rows.append([norm * row[i] for i in range(nc)])
    verify_orthonormal(l, rows, carts)
    return [[mp.nstr(v, 30) for v in row] for row in rows]


# ---------------------------------------------------------------------------
# Certified-mixed-precision amplification constants
# ---------------------------------------------------------------------------

def class_amplification(l_total):
    """A[t,m] over the 3D Hermite index t with |t| + m <= l_total, via the
    recurrence DAG DP; C_class = 2 * max_{|t|<=l_total} A[t,0] (factor 2 =
    the documented safety margin). Direction choice per target: the first
    axis (x, then y, then z) with a nonzero component - identical to the
    kernel's choice in md_vrr.hpp."""
    kappa = [mpf(0)] + [kappa_m(m) for m in range(1, l_total + 1)]
    # key: (tx, ty, tz, m)
    a = {}
    for m in range(l_total + 1):
        a[(0, 0, 0, m)] = mpf(1)
    for m in range(l_total - 1, -1, -1):
        for tx in range(l_total - m + 1):
            for ty in range(l_total - m - tx + 1):
                for tz in range(l_total - m - tx - ty + 1):
                    if tx + ty + tz == 0:
                        continue
                    # first axis with a nonzero component
                    if tx > 0:
                        src1 = (tx - 1, ty, tz, m + 1)
                        src2 = (tx - 2, ty, tz, m + 1) if tx >= 2 else None
                        tdir = tx - 1
                    elif ty > 0:
                        src1 = (tx, ty - 1, tz, m + 1)
                        src2 = (tx, ty - 2, tz, m + 1) if ty >= 2 else None
                        tdir = ty - 1
                    else:
                        src1 = (tx, ty, tz - 1, m + 1)
                        src2 = (tx, ty, tz - 2, m + 1) if tz >= 2 else None
                        tdir = tz - 1
                    value = kappa[m + 1] * a[src1]
                    if src2 is not None:
                        value += kappa[m + 1] * tdir * a[src2]
                    a[(tx, ty, tz, m)] = value
    best = mpf(0)
    for tx in range(l_total + 1):
        for ty in range(l_total - tx + 1):
            for tz in range(l_total - tx - ty + 1):
                best = max(best, a[(tx, ty, tz, 0)])
    return 2 * best


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def write_tables_header(path):
    with open(path, "w", newline="\n") as f:
        f.write("// Generated by tools/gen_md_tables.py - DO NOT EDIT.\n")
        f.write("// Constexpr tables of the matrix-form MD engine: layout conventions\n")
        f.write("// (md_defs.hpp), the real solid-harmonic transforms, Hermite parity\n")
        f.write("// masks, and the certified-mixed-precision amplification constants.\n")
        f.write("// Validated by\n")
        f.write("// integrals/tests/md_tables_test.cpp (byte-identity regeneration).\n")
        f.write("#pragma once\n#include <array>\n#include <cstddef>\n\n")
        f.write("namespace qcx::integrals::internal {\n\n")
        f.write(f"inline constexpr int kMaxShellL = {MAX_SHELL_L};\n\n")
        f.write("inline constexpr auto kCartesianCount = std::to_array<int>({"
                + ", ".join(str(hermite2d_count(l)) for l in range(MAX_SHELL_L + 1)) + "});\n")
        f.write("inline constexpr auto kSphericalCount = std::to_array<int>({"
                + ", ".join(str(2 * l + 1) for l in range(MAX_SHELL_L + 1)) + "});\n")
        f.write("inline constexpr auto kHermite2DCount = std::to_array<int>({"
                + ", ".join(str(hermite2d_count(l)) for l in range(MAX_VRR_L + 1)) + "});\n")
        f.write("inline constexpr auto kHermite3DCount = std::to_array<int>({"
                + ", ".join(str(hermite3d_count(l)) for l in range(MAX_VRR_L + 1)) + "});\n\n")

        # kH2Prefix: sum_{n < N} H2Count(n) = N(N+1)(N+2)/6 (tier offset inside
        # a VRR slice).
        prefix = [str(n * (n + 1) * (n + 2) // 6) for n in range(MAX_VRR_L + 2)]
        f.write("inline constexpr auto kH2Prefix = std::to_array<int>({"
                + ", ".join(prefix) + "});\n\n")

        # Cartesian index map. The explicit std::array type (not to_array):
        # the member array's elements are themselves aggregates, which
        # to_array's element-type deduction cannot disambiguate.
        f.write("struct CartIndex { int ix, iy, iz; };\n")
        f.write("inline constexpr std::array<std::array<CartIndex, "
                + f"{hermite2d_count(MAX_SHELL_L)}" + ">, " + f"{MAX_SHELL_L + 1}"
                + "> kCartesianIndices = {{\n")
        for l in range(MAX_SHELL_L + 1):
            row = ", ".join(f"{{{ix}, {iy}, {iz}}}" for ix, iy, iz in cartesian_components(l))
            f.write("  {{" + row + "}},  // l=" + str(l) + "\n")
        f.write("}};\n\n")

        # Parity masks: bit t set iff t = i + j (mod 2), per (i, j) 1D pair.
        f.write("inline constexpr auto kParityMask = std::to_array<std::array<unsigned int, "
                + f"{MAX_SHELL_L + 1}" + ">>({\n")
        for i in range(MAX_SHELL_L + 1):
            row = []
            for j in range(MAX_SHELL_L + 1):
                mask = 0
                for t in range(i + j + 1):
                    if (t - i - j) % 2 == 0:
                        mask |= 1 << t
                row.append(str(mask))
            f.write(f"  {{{', '.join(row)}}},  // la={i}\n")
        f.write("});\n\n")

        # Solid harmonics. The explicit std::array type (not to_array): the
        # nested array<double,28> members make the to_array element type
        # ambiguous to brace elision (MSVC rejects it).
        f.write("// Real solid-harmonic transforms (Racah normalization; rows m = -l..+l,\n")
        f.write("// cos(m phi) for m >= 0, sin(m phi) for m < 0). Row = spherical function,\n")
        f.write("// column = Cartesian\n")
        f.write("// component (z-slowest). 30-digit mpmath generation.\n")
        f.write("inline constexpr std::array<std::array<std::array<double, "
                + f"{hermite2d_count(MAX_SHELL_L)}" + ">, " + f"{2 * MAX_SHELL_L + 1}" + ">, "
                + f"{MAX_SHELL_L + 1}" + "> kSolidHarmonicG = {{\n")
        for l in range(MAX_SHELL_L + 1):
            # Plain concatenation, not an f-string: in an f-string the {{
            # escape would render a single literal brace.
            f.write("  {{  // l=" + str(l) + "\n")
            for mrow in solid_harmonic_table(l):
                f.write("    {{" + ", ".join(fmt(float(mp.mpf(v))) for v in mrow) + "}},\n")
            f.write("  }},\n")
        f.write("}};\n\n")

        # The amplification constants, symmetric in (LBra, LKet).
        f.write("// Per-class amplification constants C_class: 2 * max over\n")
        f.write("// |t| <= L of the recurrence-DAG coefficient sum A[t,0] (see the\n")
        f.write("// generator docstring); L = LBra + LKet. Factor 2 = safety margin.\n")
        f.write("inline constexpr auto kClassAmplification = std::to_array<std::array<double, "
                + f"{MAX_TOTAL_L + 1}" + ">>({\n")
        amp = [class_amplification(l) for l in range(2 * MAX_TOTAL_L + 1)]
        for lbra in range(MAX_TOTAL_L + 1):
            row = ", ".join(fmt(float(amp[lbra + lket])) for lket in range(MAX_TOTAL_L + 1))
            f.write(f"  {{{row}}},  // L_bra={lbra}\n")
        f.write("});\n\n")

        # Per-class rounding allowance: ket GEMM inner dim + bra GEMM + final
        # representation, the (1 + G) factor of the certified bound.
        f.write("// Per-class transform-rounding allowance G: the ket GEMM\n")
        f.write("// inner dimension + the bra GEMM + the final representation.\n")
        f.write("inline constexpr auto kClassRounding = std::to_array<std::array<int, "
                + f"{MAX_TOTAL_L + 1}" + ">>({\n")
        for lbra in range(MAX_TOTAL_L + 1):
            row = ", ".join(str(hermite2d_count(lbra) + hermite2d_count(lket) + 2)
                            for lket in range(MAX_TOTAL_L + 1))
            f.write(f"  {{{row}}},  // L_bra={lbra}\n")
        f.write("});\n")
        f.write("\n}  // namespace qcx::integrals::internal\n")


def write_dispatch_header(path):
    with open(path, "w", newline="\n") as f:
        f.write("// Generated by tools/gen_md_tables.py - DO NOT EDIT.\n")
        f.write("// The per-{L_bra, L_ket}-class function-pointer table; entries whose\n")
        f.write("// class lies outside the instantiation matrix (the 2e triangle UNION\n")
        f.write("// the RI rectangle at the configured QcxIntegralsLMax) are nullptr -\n")
        f.write("// no CI configuration ever links the full class matrix.\n")
        f.write("#pragma once\n#include \"md_engine.hpp\"\n\n")
        f.write("namespace qcx::integrals::internal {\n\n")
        for lane in ("F64", "F32"):
            f.write(f"inline constexpr auto kMdClassTable{lane} = std::to_array<std::array<MdClassFn{lane}, "
                    + f"{MAX_TOTAL_L + 1}" + ">>({\n")
            for lbra in range(MAX_TOTAL_L + 1):
                # One brace per row: to_array<array<T, 13>> elements
                # initialize their member array directly (brace elision).
                f.write("  {  // L_bra=" + str(lbra) + "\n")
                for lket in range(MAX_TOTAL_L + 1):
                    # The 2e triangle (lBra <= lKet) UNION the RI rectangle
                    # (lKet <= the shell Lmax, lBra <= 2*Lmax): the 3c
                    # kernels use the orbital pair as the bra.
                    if lbra <= lket or lket <= MAX_TOTAL_L // 2:
                        # The cap must mirror the CMake instantiation matrix
                        # (integrals/CMakeLists.txt): the triangle needs
                        # L >= ceil(lKet/2); the rectangle additionally needs
                        # lKet <= L and lBra <= 2*L, so its cap is the larger
                        # of lKet and ceil(lBra/2). (The old ceil(max/2) cap
                        # emitted references to classes a reduced-Lmax build
                        # never instantiates - the CI Lmax=2 link error.)
                        if lbra <= lket:
                            cap = (lket + 1) // 2
                        else:
                            cap = max(lket, (lbra + 1) // 2)
                        if lane == "F32":
                            f.write("#if defined(QcxIntegralsF32) && (QcxIntegralsF32 == 1) && "
                                    f"(QcxIntegralsLMax >= {cap})\n")
                            f.write(f"    &ComputeEriClassF32<{lbra}, {lket}>,\n")
                            f.write("#else\n    nullptr,\n#endif\n")
                        else:
                            f.write(f"#if (QcxIntegralsLMax >= {cap})\n")
                            f.write(f"    &ComputeEriClass<{lbra}, {lket}>,\n")
                            f.write("#else\n    nullptr,\n#endif\n")
                    else:
                        f.write("    nullptr,  // beyond the 2e triangle and the RI rectangle\n")
                f.write("  },\n")
            f.write("});\n\n")
        f.write("}  // namespace qcx::integrals::internal\n")


def format_header(path):
    """Makes the emitted header clang-format-clean (the pre-commit hook blocks
    any staged .hpp that fails the dry-run gate) and LF (the repo norm).

    The path must live inside the repo tree: clang-format discovers the repo
    .clang-format relative to the file's directory, and a scratch copy outside
    the tree would be formatted with the LLVM default style instead.
    """
    clang_format = shutil.which("clang-format")
    if clang_format is None:
        default = (r"C:\Program Files\Microsoft Visual Studio\18\Community"
                   r"\VC\Tools\Llvm\x64\bin\clang-format.exe")
        if os.path.exists(default):
            clang_format = default
    if clang_format is not None:
        # Iterate to the fixpoint: clang-format is not idempotent on comments
        # trailing a `{` inside an initializer list (the dispatch header's
        # `{  // L_bra=n` rows move to their final line only on the second
        # pass - the 2026-08-22 finding). A single pass leaves the file
        # differing from a fresh --check AND failing the pre-commit dry-run.
        for _ in range(4):
            before = open(path, "rb").read()
            subprocess.run([clang_format, "-i", path], check=True)
            if open(path, "rb").read() == before:
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default="integrals/src/internal/md_tables_gen.hpp")
    parser.add_argument("--dispatch", default="integrals/src/internal/md_dispatch_gen.hpp")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed headers are byte-identical to a "
                             "fresh generation (exits nonzero on drift)")
    args = parser.parse_args()

    # Resolve relative paths against the repo root (the script's parent
    # directory): the regeneration ctest runs from the build directory.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.tables = args.tables if os.path.isabs(args.tables) else os.path.join(repo_root, args.tables)
    args.dispatch = (args.dispatch if os.path.isabs(args.dispatch)
                     else os.path.join(repo_root, args.dispatch))

    targets = [(args.tables, write_tables_header), (args.dispatch, write_dispatch_header)]
    if args.check:
        ok = True
        for path, writer in targets:
            # A uniquely named scratch copy next to the target (mkstemp): the
            # file must live inside the repo tree so clang-format discovers the
            # repo .clang-format (format_header), and the unique name keeps
            # concurrent --check runs (CI + local) and hard kills from
            # colliding on a fixed .check-tmp.hpp path.
            fd, tmp = tempfile.mkstemp(suffix=".hpp", prefix="md-gen-check-",
                                       dir=os.path.dirname(path))
            try:
                os.close(fd)
                writer(tmp)
                format_header(tmp)
                with open(tmp, "rb") as fg, open(path, "rb") as fc:
                    if fg.read() != fc.read():
                        ok = False
                        print(f"DRIFT: {path} differs from a fresh generation")
            finally:
                os.remove(tmp)
        return 0 if ok else 1

    for path, writer in targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        writer(path)
        format_header(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
