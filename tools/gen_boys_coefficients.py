#!/usr/bin/env python3
"""Generate the Boys-function Chebyshev coefficient tables and test reference data.

Reproduces integrals/src/boys_coefficients.hpp (double and float lanes) and
integrals/tests/data/boys_reference.csv from scratch, validating every fit
against a 30-digit mpmath reference. Requires: `pip install mpmath`.

Method:
  - Region A [0, x0): per-order piecewise Chebyshev fits evaluated by a split
    Clenshaw recurrence (even/odd, T_{2j}(t) = T_j(v), T_{2j+1}(t) = t*D_j(v),
    v = 2t^2 - 1). The fits are WEIGHTED: the downward batch recursion
    amplifies the seed error by up to x^n / prod_{j=0}^{n-1}(j + 1/2)
    (Vikhamar-Sandberg & Repisky, arXiv:2512.10059, eqs. 18-23), so the
    effective seed tolerance is tol / max(1, weight(x)).
  - Region B [x0, x1): one F0 fit + upward recursion.
  - Region C [x1, inf): asymptotic, no coefficients.
  - Boundaries: fixed kmax=32 values x0 = 11.899848152108484,
    x1 = 28.989337738820740 (the configuration validated end-to-end against
    the reference grid; the per-kmax table is printed for documentation).
  - Float lane: same structure, tolerance 1e-7, degree cap 10.

The DCT normalization is the standard one: c_0 gets 1/(deg+1), all other
coefficients - including the top one - get 2/(deg+1). An earlier normalization
bug (halving c_deg too) made fits diverge - see the design doc. The committed
header must match this script's output byte for byte for a given tolerance
configuration.
"""
import argparse
import os
import shutil
import subprocess
import sys
from functools import lru_cache

try:
    from mpmath import mp, mpf, cos, exp, pi
except ImportError:
    if "--check" in sys.argv:
        # The regeneration ctest: CI lanes without the pip package skip.
        print("SKIP: mpmath is not installed (the --check mode requires it)")
        sys.exit(77)
    raise

mp.dps = 30  # reference error ~1e-25; 4 orders below the double floor 1e-14

TOL_DOUBLE = mpf("5e-14")
TOL_FLOAT = mpf("1e-7")
X0 = mpf("11.899848152108484")
X1 = mpf("28.989337738820740")
MAX_ORDER = 32
MAX_DEG_DOUBLE = 18
MAX_DEG_FLOAT = 10


@lru_cache(maxsize=1 << 20)
def boys_ref(n, x):
    """F_n(x) via the provably stable series (V&S eq. 26): all terms positive."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    s = mpf(0)
    term = mpf(1) / (n + mpf("0.5"))
    e = exp(-x) / 2
    for l in range(300):
        s += term
        if l > 20 and term < mpf("1e-26"):
            break
        term *= x / (n + l + mpf("1.5"))
    return e * s


def grid(a, b, npts):
    return [a + (b - a) * i / npts for i in range(npts + 1)]


def cheb_coeffs(n, a, b, deg):
    """Chebyshev interpolation coefficients at the zeros of T_{deg+1}."""
    mid = (a + b) / 2
    half = (b - a) / 2
    nodes = [mid + half * cos(pi * (k + mpf("0.5")) / (deg + 1)) for k in range(deg + 1)]
    values = [boys_ref(n, x) for x in nodes]
    coeffs = []
    for j in range(deg + 1):
        s = mpf(0)
        for k in range(deg + 1):
            s += values[k] * cos(pi * j * (k + mpf("0.5")) / (deg + 1))
        factor = mpf(1) / (deg + 1) if j == 0 else mpf(2) / (deg + 1)
        coeffs.append(factor * s)
    return coeffs


def clenshaw_double(cs, x, a, b):
    """Split Clenshaw in IEEE double, algebraically identical to the C++ split
    Clenshaw (which uses FMA throughout)."""
    t = 2.0 * (x - a) / (b - a) - 1.0
    if len(cs) == 1:
        return cs[0]
    if len(cs) == 2:
        return cs[0] + t * cs[1]
    v = 2.0 * t * t - 1.0
    two_v = v + v
    # Even coefficients: cs[0], cs[2], ...; odd: cs[1], cs[3], ...
    even_cs = cs[0::2]
    odd_cs = cs[1::2]
    m_even = len(even_cs) - 1
    b1 = even_cs[m_even]
    b2 = 0.0
    for k in range(m_even - 1, 0, -1):
        b1, b2 = even_cs[k] + two_v * b1 - b2, b1
    even = even_cs[0] + v * b1 - b2
    m_odd = len(odd_cs) - 1
    if m_odd <= 0:
        return even + t * odd_cs[0]
    o1 = odd_cs[m_odd]
    o2 = 0.0
    for k in range(m_odd - 1, 0, -1):
        o1, o2 = odd_cs[k] + two_v * o1 - o2, o1
    odd = odd_cs[0] + (two_v - 1.0) * o1 - o2
    return even + t * odd


def seed_weight(n, x):
    """Downward-recursion amplification bound: max(1, x^n / prod(j+1/2))."""
    p = mpf(1)
    for j in range(n):
        p *= x / (j + mpf("0.5"))
    return max(mpf(1), p)


def fit_interval(n, a, b, tol, maxdeg, weighted):
    """Returns (deg, [float coeffs]) or None; weighted=True fits region-A seeds."""
    fmax = max(abs(boys_ref(n, x)) for x in grid(a, b, 8))
    if weighted:
        if max(fmax * seed_weight(n, x) for x in grid(a, b, 8)) < tol:
            return (0, [0.0])
    elif fmax < tol:
        return (0, [0.0])
    degs = (12, maxdeg) if maxdeg >= 12 else (maxdeg,)
    for deg in degs:
        cm = cheb_coeffs(n, a, b, deg)
        cd = [float(c) for c in cm]
        ok = True
        for x in grid(a, b, 8):
            err = abs(clenshaw_double(cd, x, a, b) - float(boys_ref(n, x)))
            budget = tol / seed_weight(n, x) * mpf("0.5") if weighted else tol
            if err > budget:
                ok = False
                break
        if ok:
            return (deg, cd)
    return None


def fit_order(n, region_b=False, f32=False):
    tol = TOL_FLOAT if f32 else TOL_DOUBLE
    maxdeg = MAX_DEG_FLOAT if f32 else MAX_DEG_DOUBLE
    if region_b:
        return fit_interval(0, X0, X1, tol, maxdeg, weighted=False)
    pieces = []
    stack = [(mpf(0), X0, 0)]
    while stack:
        a, b, depth = stack.pop()
        r = fit_interval(n, a, b, tol, maxdeg, weighted=True)
        if r is None:
            if depth > 40:
                raise RuntimeError(f"F{n}: fit never converged on [{a},{b}]")
            m = (a + b) / 2
            stack.append((m, b, depth + 1))
            stack.append((a, m, depth + 1))
        else:
            deg, cd = r
            pieces.append((float(a), float(b), deg, cd))
    pieces.sort(key=lambda p: p[0])
    return pieces


def fmt(v):
    return f"{v:.17e}"


def write_header(path, double_orders, float_orders, b_cheb, b_cheb_f32):
    with open(path, "w", newline="\n") as f:
        f.write("// Generated by tools/gen_boys_coefficients.py - DO NOT EDIT.\n")
        f.write("// Piecewise Chebyshev (split Clenshaw) fits of F_n(x), region A seeds\n")
        f.write("// weighted against downward-recursion amplification; validated against a\n")
        f.write("// 30-digit mpmath reference (definitive check: integrals/tests/boys_test.cpp).\n")
        f.write("#pragma once\n#include <array>\n#include <cstddef>\n\n")
        f.write("namespace qcx::integrals::detail {\n\n")
        f.write(f"inline constexpr int kMaxOrder = {MAX_ORDER};\n")
        f.write(f"inline constexpr double kX0 = {fmt(X0)};\n")
        f.write(f"inline constexpr double kX1 = {fmt(X1)};\n\n")

        all_coeffs = []
        meta = []
        for n in range(MAX_ORDER + 1):
            for (a, b, deg, cs) in double_orders[n]:
                meta.append((n, a, b, deg, len(all_coeffs)))
                all_coeffs.extend(fmt(c) for c in cs)
        f.write("inline constexpr auto kCoeffs = std::to_array<double>({\n")
        for i in range(0, len(all_coeffs), 6):
            f.write("  " + ", ".join(all_coeffs[i:i + 6]) + ",\n")
        f.write("});\n\n")
        f.write("struct OrderPiece { double a, b; int deg; int offset; };\n")
        f.write("inline constexpr auto kPieces = std::to_array<OrderPiece>({\n")
        for (n, a, b, deg, off) in meta:
            f.write(f"  {{{fmt(mpf(a))}, {fmt(mpf(b))}, {deg}, {off}}},  // F{n}\n")
        f.write("});\n")
        starts = []
        index = 0
        for n in range(MAX_ORDER + 1):
            starts.append(index)
            index += len(double_orders[n])
        starts.append(index)
        f.write("inline constexpr auto kPieceStart = std::to_array<int>({"
                + ", ".join(map(str, starts)) + "});\n")
        f.write("static_assert(std::size(kPieceStart) == kMaxOrder + 2,\n"
                "              \"piece-start table must cover kMaxOrder\");\n\n")
        deg, cs = b_cheb
        f.write("inline constexpr auto kBcoeffs = std::to_array<double>({"
                + ", ".join(fmt(c) for c in cs) + "});\n")
        f.write(f"inline constexpr int kBDeg = {deg};\n")
        f.write("\n}  // namespace qcx::integrals::detail\n")

        # Float lane.
        f.write("\nnamespace qcx::integrals::detail::f32 {\n\n")
        all_coeffs = []
        meta = []
        for n in range(MAX_ORDER + 1):
            for (a, b, deg, cs) in float_orders[n]:
                meta.append((n, a, b, deg, len(all_coeffs)))
                all_coeffs.extend(fmt(c) for c in cs)
        f.write("inline constexpr auto kCoeffs = std::to_array<float>({\n")
        for i in range(0, len(all_coeffs), 6):
            f.write("  " + ", ".join(all_coeffs[i:i + 6]) + ",\n")
        f.write("});\n\n")
        f.write("struct OrderPiece { float a, b; int deg; int offset; };\n")
        f.write("inline constexpr auto kPieces = std::to_array<OrderPiece>({\n")
        for (n, a, b, deg, off) in meta:
            f.write(f"  {{{fmt(mpf(a))}, {fmt(mpf(b))}, {deg}, {off}}},  // F{n}\n")
        f.write("});\n")
        starts = []
        index = 0
        for n in range(MAX_ORDER + 1):
            starts.append(index)
            index += len(float_orders[n])
        starts.append(index)
        f.write("inline constexpr auto kPieceStart = std::to_array<int>({"
                + ", ".join(map(str, starts)) + "});\n")
        f.write("static_assert(std::size(kPieceStart) == kMaxOrder + 2,\n"
                "              \"piece-start table must cover kMaxOrder\");\n\n")
        deg, cs = b_cheb_f32
        f.write("inline constexpr auto kBcoeffs = std::to_array<float>({"
                + ", ".join(fmt(c) for c in cs) + "});\n")
        f.write(f"inline constexpr int kBDeg = {deg};\n")
        f.write("\n}  // namespace qcx::integrals::detail::f32\n")


def write_reference(path):
    import csv
    # Dense over [0, 100]: the validated range of the committed grid. Beyond it
    # every F_n decays faster than the asymptotic tail (region C covers it, and
    # the 300-term series in boys_ref is only convergent for x <= ~250).
    xgrid = [mpf(0), mpf("1e-12")]
    xgrid += [mpf(10) ** (mpf(-8) + mpf(10) * i / 48) for i in range(49)]  # 1e-8..1e2
    xgrid += [X0, X1, mpf("0.05"), mpf("0.1"), mpf("1"), mpf("2"), mpf("4"), mpf("6"),
              mpf("8"), mpf("10"), mpf("13"), mpf("26.67"), mpf("29"), mpf("30"),
              mpf("31"), mpf("33"), mpf("40"), mpf("50"), mpf("100")]
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "x", "value"])
        for n in range(MAX_ORDER + 1):
            for x in xgrid:
                w.writerow([n, mp.nstr(x, 20), mp.nstr(boys_ref(n, x), 22)])
    print(f"wrote {path} ({len(xgrid) * (MAX_ORDER + 1)} rows)")


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
        subprocess.run([clang_format, "-i", path], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", default="integrals/src/boys_coefficients.hpp")
    parser.add_argument("--reference", default="integrals/tests/data/boys_reference.csv")
    parser.add_argument("--reference-only", action="store_true",
                        help="regenerate only the reference CSV (skip the fitting)")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed header/CSV are byte-identical to a "
                             "fresh generation (exits nonzero on drift)")
    args = parser.parse_args()

    if args.reference_only:
        os.makedirs(os.path.dirname(args.reference) or ".", exist_ok=True)
        write_reference(args.reference)
        return 0

    print("fitting double lane (weighted region-A seeds, tol 5e-14, deg<=18) ...")
    double_orders = []
    for n in range(MAX_ORDER + 1):
        pieces = fit_order(n)
        double_orders.append(pieces)
        print(f"  F{n:2d}: {len(pieces)} intervals, {sum(len(p[3]) for p in pieces)} coeffs")

    print("fitting float lane (weighted region-A seeds, tol 1e-7, deg<=10) ...")
    float_orders = []
    for n in range(MAX_ORDER + 1):
        pieces = fit_order(n, f32=True)
        float_orders.append(pieces)

    print("fitting region B seeds ...")
    b_cheb = fit_order(0, region_b=True)
    b_cheb_f32 = fit_order(0, region_b=True, f32=True)

    if args.check:
        import tempfile
        # The scratch header lives next to the committed one: clang-format
        # must discover the repo .clang-format, which it does relative to the
        # file's directory - a foreign temp dir would silently format the
        # scratch copy with the LLVM default style and every check would
        # false-positive on wrapping.
        tmp_header = args.header + ".check-tmp.hpp"
        tmp_reference = os.path.join(tempfile.gettempdir(), "boys_reference-check-tmp.csv")
        try:
            write_header(tmp_header, double_orders, float_orders, b_cheb, b_cheb_f32)
            format_header(tmp_header)
            write_reference(tmp_reference)
            ok = True
            for generated, committed in ((tmp_header, args.header),
                                         (tmp_reference, args.reference)):
                with open(generated, "rb") as fg, open(committed, "rb") as fc:
                    if fg.read() != fc.read():
                        ok = False
                        print(f"DRIFT: {committed} differs from a fresh generation")
            return 0 if ok else 1
        finally:
            for scratch in (tmp_header, tmp_reference):
                if os.path.exists(scratch):
                    os.remove(scratch)

    os.makedirs(os.path.dirname(args.header) or ".", exist_ok=True)
    write_header(args.header, double_orders, float_orders, b_cheb, b_cheb_f32)
    format_header(args.header)
    print(f"wrote {args.header}")

    os.makedirs(os.path.dirname(args.reference) or ".", exist_ok=True)
    write_reference(args.reference)

    # Documentation: per-kmax boundaries (the validated configuration uses the
    # fixed kmax = 32 values; the smaller values are the formula (V&S-style)
    # boundaries - the measured ones are recorded separately).
    print("per-kmax boundaries (documentation):")
    for kmax in (0, 4, 8, 12, 16, 24, 32):
        p = mpf(1)
        for n in range(kmax):
            p *= (n + mpf("0.5"))
        x0k = max(mpf(1), p ** (mpf(1) / max(kmax, 1)))
        print(f"  kmax={kmax:2d}: x0={mp.nstr(x0k, 8)}")


if __name__ == "__main__":
    sys.exit(main())
