#!/usr/bin/env python3
"""Generate the committed MD-engine reference grids (mpmath, 30 dps).

Implements the McMurchie-Davidson scheme independently of the C++ engine
(per-quartet recurrences at mpmath precision, dict-indexed Hermite spaces,
no shared code): the E coefficients from the 1D shift recurrences, the
seeds [0]^(m) = K F_m(a R_PQ^2) via the incomplete-gamma closed form, the
VRR [t+e]^(m) = -2a (P-Q)_dir [t]^(m+1) - 2a t_dir [t-e]^(m+1), the
Hermite->function transforms with the real solid-harmonic folds (same
convention as the engine's generated tables: Cartesian components
descending-lexicographic - p = x, y, z; d = xx, xy, xz, yy, yz, zz - and
Racah-normalized spherical functions, rows m = -l..+l (cos(m phi) rows for
m >= 0, sin(m phi) rows for m < 0) - l = 1 reads (y, z, x)).

Grids written:
  - integrals/tests/data/md_shellset_reference.csv: a synthetic two-center
    s/p shellset (STO-3G-like exponents, R = 1.4 bohr) - S/T/V blocks of
    every canonical pair (including a d shell on center A for the 1e spot
    checks) and the 2e blocks of every canonical quartet of the nine s/p
    pairs (classes (ss|ss), (ss|sp), (sp|sp), (ss|pp), (sp|pp), (pp|pp)).
  - integrals/tests/data/hf_sto3g_reference.csv and h2o_sto3g_reference.csv:
    the published STO-3G F/O/H contractions at the fixture geometries - S/T/V
    blocks per canonical pair and 2e blocks per canonical quartet.
  - integrals/tests/data/md_high_l_reference.csv: canonical-quartet spot
    checks of the f/g/h shells (l = 3..6 classes, fixed function-index
    pattern per quartet - spot checks, not full tensors), gated on max_l >= 3
    (the CI lanes cannot instantiate the f shell and skip the grid).
  - integrals/tests/data/md_3c_reference.csv: the (uv|P) blocks of every
    canonical H2O/STO-3G orbital pair against the tiny s/p auxiliary set
    (one function triple per task plus the corners of the (p,p) x O-p
    block), and the (P|Q) function metric elements - both through the
    phantom (P, s_0) construction of the RI engine.

The CSVs depend only on the published basis constants, the geometries, and
the recurrences above - not on the C++ implementation under test.
Requires: `pip install mpmath`.
"""
import argparse
import csv
import os
import sys
import tempfile
from fractions import Fraction
from math import comb, factorial

try:
    from mpmath import exp, gammainc, mp, mpf, pi, sqrt
except ImportError:
    if "--check" in sys.argv:
        # The regeneration ctest: CI lanes without the pip package skip.
        print("SKIP: mpmath is not installed (the --check mode requires it)")
        sys.exit(77)
    raise

mp.dps = 30  # reference error ~1e-25; 4 orders below the double floor 1e-14

# ---------------------------------------------------------------------------
# Conventions (identical to the engine's generated tables)
# ---------------------------------------------------------------------------

def cartesian_components(l):
    """Cartesian (ix, iy, iz) per index, descending lexicographic."""
    out = []
    for ix in range(l, -1, -1):
        for iy in range(l - ix, -1, -1):
            out.append((ix, iy, l - ix - iy))
    return out


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


def spherical_functions(l):
    """Real solid-harmonic Cartesian coefficients, Racah normalization,
    rows m = -l..+l (cos(m phi) for m >= 0, sin(m phi) for m < 0) -
    re-derived here, independently of tools/gen_md_tables.py (same
    convention)."""
    carts = cartesian_components(l)

    def legendre_derivative_coeffs(ld, m):
        out = []
        for k in range((ld - m) // 2 + 1):
            pcoef = Fraction((-1) ** k * factorial(2 * ld - 2 * k),
                             2 ** ld * factorial(k) * factorial(ld - k) * factorial(ld - 2 * k))
            power = ld - 2 * k - m
            if power >= 0:
                out.append((power, pcoef * (factorial(ld - 2 * k) // factorial(power))))
        return out

    rows = []
    # l = 0 is the identity: a spherical s function equals the Cartesian s
    # function (both individually normalized); the Racah fold starts at l=1.
    if l == 0:
        return [[mpf(1)]]

    for m_signed in range(-l, l + 1):
        m = abs(m_signed)
        trig = "cos" if m_signed >= 0 else "sin"
        if m == 0:
            norm = sqrt(mpf(2 * l + 1) / (4 * pi))
        else:
            norm = sqrt(mpf(2 * l + 1) * factorial(l - m) / (2 * pi * factorial(l + m)))
        row = {cart: Fraction(0) for cart in range(len(carts))}
        for power, ck in legendre_derivative_coeffs(l, m):
            rem = l - power - m
            if rem < 0 or rem % 2 != 0:
                continue
            for j in range(m + 1):
                ix = m - j
                iy = j
                c = comb(m, j)
                if trig == "cos" and j % 2 != 0:
                    continue
                if trig == "sin" and j % 2 == 0:
                    continue
                base = Fraction((-1) ** (j // 2) * c) if trig == "cos" else Fraction(
                    (-1) ** ((j - 1) // 2) * c)
                # Multiply by (x^2+y^2+z^2)^(rem/2): the trinomial
                # n!/(a!b!c!) x^{2a} y^{2b} z^{2c} over a+b+c = rem/2
                # (u = z/r puts r^2 = x^2+y^2+z^2 into the leftover
                # radial power).
                nR2 = rem // 2
                for a in range(nR2 + 1):
                    for b in range(nR2 - a + 1):
                        c = nR2 - a - b
                        ix2 = ix + 2 * a
                        iy2 = iy + 2 * b
                        iz = power + 2 * c
                        cart = carts.index((ix2, iy2, iz))
                        row[cart] += ck * base * Fraction(
                            factorial(nR2), factorial(a) * factorial(b) * factorial(c))
        rows.append([norm * row[i] for i in range(len(carts))])
    verify_orthonormal(l, rows, carts)
    return [[mp.mpf(v) for v in row] for row in rows]


# ---------------------------------------------------------------------------
# The MD recurrences at mpmath
# ---------------------------------------------------------------------------

def boys(n, x):
    """F_n(x) = 1/2 x^(-(n+1/2)) gamma(n+1/2, x) (exact limit at 0)."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    return mpf("0.5") * gammainc(n + mpf("0.5"), 0, x) / x ** (n + mpf("0.5"))


def e_shift(e, shift_value, p):
    """The 1D E-shift: E'_t = shift E_t + t E_{t-1}... expressed per old t:
    out[t] += shift e[t]; out[t-1] += t e[t]; out[t+1] += e[t]/(2p)."""
    out = {}
    for t, v in e.items():
        out[t] = out.get(t, 0) + shift_value * v
        if t >= 1:
            out[t - 1] = out.get(t - 1, 0) + t * v
        out[t + 1] = out.get(t + 1, 0) + v / (2 * p)
    return out


def e_table_1d(i_power, j_power, pa, pb, p):
    """E_t^{(i,j)} of the (x-A)^i (x-B)^j expansion."""
    e = {0: mpf(1)}
    for _ in range(i_power):
        e = e_shift(e, pa, p)
    for _ in range(j_power):
        e = e_shift(e, pb, p)
    return e


def pair_tables(la, lb, pa, pb, p):
    """The per-axis 1D E tables of one primitive pair: [axis][(ix, jx)] -> dict t."""
    out = []
    for axis in range(3):
        table = {}
        for ix in range(la + 1):
            for jx in range(lb + 1):
                table[(ix, jx)] = e_table_1d(ix, jx, pa[axis], pb[axis], p)
        out.append(table)
    return out


def fold_tables(la, lb, tables, spherical_a, spherical_b):
    """The folded transform T[(fa, fb)] -> dict of 3D-Hermite t -> value,
    per primitive pair (rows excluded; the contraction weights apply later).
    T[(fa,fb), (tx,ty,tz)] = sum over Cartesian powers of the per-axis
    products with the G folds."""
    ga = spherical_functions(la) if spherical_a else None
    gb = spherical_functions(lb) if spherical_b else None
    n_a = len(ga) if spherical_a else len(cartesian_components(la))
    n_b = len(gb) if spherical_b else len(cartesian_components(lb))
    out = {}
    for fa in range(n_a):
        for fb in range(n_b):
            row = {}
            for tx in range(la + lb + 1):
                for ty in range(la + lb - tx + 1):
                    for tz in range(la + lb - tx - ty + 1):
                        value = mpf(0)
                        for anga in range(len(cartesian_components(la))):
                            ca = cartesian_components(la)[anga]
                            wa = ga[fa][anga] if spherical_a else (mpf(1) if fa == anga else mpf(0))
                            for angb in range(len(cartesian_components(lb))):
                                cb = cartesian_components(lb)[angb]
                                wb = gb[fb][angb] if spherical_b else (mpf(1) if fb == angb else mpf(0))
                                value += (wa * wb *
                                          tables[0][(ca[0], cb[0])].get(tx, mpf(0)) *
                                          tables[1][(ca[1], cb[1])].get(ty, mpf(0)) *
                                          tables[2][(ca[2], cb[2])].get(tz, mpf(0)))
                        if value != 0:
                            row[(tx, ty, tz)] = value
            out[(fa, fb)] = row
    return out


def vrr_targets(seeds, l_total, alpha, dq):
    """[t]^(0) for |t| <= l_total via the slice recurrence (dicts).

    Slice `total` holds [t]^(total-|t|): [t]^(m) = -2a (P-Q)_dir [t-e_dir]^(m+1)
    - 2a t_dir [t-2e_dir]^(m+1) (from [t]^(m) = d_P^t [0]^(m); the t_dir term
    drops m by one unit only, so [t-2e_dir]^(m+1) has |t-2e| + m + 1 = total - 1:
    it lives in the PREVIOUS slice - reading the current slice's entry would
    use the wrong seed F_{total} (the p-shell regression)."""
    slices = [{(0, 0, 0): seeds[0]}]
    for total in range(1, l_total + 1):
        cur = {(0, 0, 0): seeds[total]}
        for n in range(1, total + 1):
            for ty in range(n + 1):
                for tz in range(n - ty + 1):
                    tx = n - ty - tz
                    value = mpf(0)
                    if tx >= 1:
                        value = -2 * alpha * dq[0] * cur.get((tx - 1, ty, tz), mpf(0))
                        if tx >= 2:
                            value += -2 * alpha * (tx - 1) * slices[total - 1].get(
                                (tx - 2, ty, tz), mpf(0))
                    elif ty >= 1:
                        value = -2 * alpha * dq[1] * cur.get((tx, ty - 1, tz), mpf(0))
                        if ty >= 2:
                            value += -2 * alpha * (ty - 1) * slices[total - 1].get(
                                (tx, ty - 2, tz), mpf(0))
                    else:
                        value = -2 * alpha * dq[2] * cur.get((tx, ty, tz - 1), mpf(0))
                        if tz >= 2:
                            value += -2 * alpha * (tz - 1) * slices[total - 1].get(
                                (tx, ty, tz - 2), mpf(0))
                    cur[(tx, ty, tz)] = value
        slices.append(cur)
    targets = {}
    for total in range(l_total + 1):
        for (tx, ty, tz), v in slices[total].items():
            if tx + ty + tz == total:
                targets[(tx, ty, tz)] = v
    return targets



def solid_normalization(l, a):
    """The unit normalization of one primitive of a spherical shell beyond
    the (2a/pi)^(3/4) factor - the mirror of SolidNormalization in
    md_defs.hpp: the G tables map to angular-orthonormal solid harmonics for
    l >= 1 (l = 0 is the plain Gaussian), so the radial solid-harmonic
    factor 2^(l+1) sqrt(pi a^l / (2l+1)!!) restores the unit norm."""
    if l == 0:
        return mpf(1)
    # (2l+1)!! = (2l+1)! / (2^l l!) - mpmath has no factorial2.
    double_factorial = mp.factorial(2 * l + 1) / (2 ** l * mp.factorial(l))
    return mpf(2) ** (l + 1) * mp.sqrt(mp.pi * a ** l / double_factorial)


def primitive_normalization(a):
    """(2a/pi)^(3/4), the engine's normalization convention."""
    return (2 * a / pi) ** mpf("0.75")


def contraction_weight(shell, row, exponent_idx):
    """One contraction weight d = c * (2a/pi)^(3/4) * solid_normalization,
    the engine's convention - with the unit-phantom exception: the exponent-0 s
    phantom of the RI construction (ri_engine.cpp phantomContractions,
    normalized = {{1.0}}) has weight 1 by convention, since its
    (2*0/pi)^(3/4) would vanish."""
    a = shell.exponents[exponent_idx]

    if a == 0:
        return mpf(1)

    return (shell.coefficients[row][exponent_idx] * primitive_normalization(a) *
            solid_normalization(shell.l, a))


def ri_block(shell_i, shell_j, aux_shell):
    """The 3-center (ij|P) function block: rows (fi, fj), cols fP. The aux
    shell P is the phantom pair (P, s_0): P crossed with a unit s phantom at
    the same center - the same construction the (P|Q) metric uses
    (ri_engine.cpp). The pair function is phi_P itself (p = zeta,
    E_cd = 1) and the full E-table fold with the ket-derivative sign rides
    the ket transform like any 2e ket pair. (The former "identity Hermite
    expansion" - a bare c_rho * G-fold with q = 2 zeta - was wrong for
    lP >= 1: with zero shifts the E recurrence still yields
    E(1, 0, 1) = 1/(2 zeta) etc.; the missing fold scaled and sign-flipped
    the l >= 1 columns of I, the 2026-08-22 RI-J finding.)"""
    phantom = ShellData(0, False, ["0.0"], [[1.0]], aux_shell.center)
    return eri_block(shell_i, shell_j, aux_shell, phantom)


def tiny_aux_shells():
    """The tiny s/p auxiliary set of the RI grid: per atom one s shell
    (uncontracted on H) and one p shell, matching the ri_engine_test
    fixture - which builds the aux pair list over the renumbered H2O
    molecule, so the order is H2, H1, O (the (Z, x, y, z) sort)."""
    o_s = (["2.5", "0.8"], [["0.3", "0.8"]])
    o_p = (["1.2"], [["1.0"]])
    h_s = (["1.0"], [["1.0"]])
    h_p = (["0.6"], [["1.0"]])
    origin = ("0.0", "0.0", "0.0")
    h1 = ("1.430428808474167", "1.107157044080814", "0.0")
    h2 = ("-1.430428808474167", "1.107157044080814", "0.0")
    return [
        ShellData(0, True, h_s[0], h_s[1], h2),
        ShellData(1, True, h_p[0], h_p[1], h2),
        ShellData(0, True, h_s[0], h_s[1], h1),
        ShellData(1, True, h_p[0], h_p[1], h1),
        ShellData(0, True, o_s[0], o_s[1], origin),
        ShellData(1, True, o_p[0], o_p[1], origin),
    ]


class ShellData:
    """One shell: l, spherical flag, exponent/coefficient lists, center.

    Every contraction is renormalized to unit norm, exactly like the
    parser (basis_set.cpp NormalizeContractions): the norm of the
    contracted function in the engine convention - the (2a/pi)^(3/4)
    primitive times solid_normalization - has the closed form
    S(a, b) = 2^l (4ab)^(3/4) (ab)^(l/2) / (a + b)^(l + 3/2) per
    primitive pair."""

    def __init__(self, l, spherical, exponents, coefficients, center):
        self.l = l
        self.spherical = spherical
        self.exponents = [mpf(str(e)) for e in exponents]
        self.coefficients = [[mpf(str(c)) for c in row] for row in coefficients]
        self.center = tuple(mpf(str(v)) for v in center)
        self.coefficients = [self._renormalize(row) for row in self.coefficients]

    def _renormalize(self, coefficients):
        # The exponent-0 unit phantom of the RI construction is already its
        # own weight; the norm formula below would degenerate to
        # 0^0.75 / 0^1.5 = nan for it.
        if any(a == 0 for a in self.exponents):
            return coefficients

        norm_squared = mpf(0)
        for k, a in enumerate(self.exponents):
            for l, b in enumerate(self.exponents):
                ab = a * b
                s = (mpf(2) ** self.l * (4 * ab) ** mpf("0.75") * ab ** (self.l / 2)
                     / (a + b) ** (self.l + mpf("1.5")))
                norm_squared += coefficients[k] * coefficients[l] * s
        if norm_squared <= 0:
            return coefficients
        scale = 1 / mp.sqrt(norm_squared)
        return [c * scale for c in coefficients]


def function_count(shell):
    if shell.spherical:
        return len(shell.coefficients) * (2 * shell.l + 1)
    return len(shell.coefficients) * len(cartesian_components(shell.l))


def eri_block(shell_i, shell_j, shell_k, shell_l):
    """The full (ij|kl) function block, rows (fi, fj), cols (fk, fl)."""
    la, lb = shell_i.l, shell_j.l
    lc, ld = shell_k.l, shell_l.l
    l_total = la + lb + lc + ld
    n_ij = function_count(shell_i) * function_count(shell_j)
    n_kl = function_count(shell_k) * function_count(shell_l)
    block = [[mpf(0) for _ in range(n_kl)] for _ in range(n_ij)]

    for a_idx, a in enumerate(shell_i.exponents):
        for b_idx, b in enumerate(shell_j.exponents):
            p = a + b
            pp = tuple((a * shell_i.center[d] + b * shell_j.center[d]) / p for d in range(3))
            pa = [pp[d] - shell_i.center[d] for d in range(3)]
            pb = [pp[d] - shell_j.center[d] for d in range(3)]
            rab2 = sum((shell_i.center[d] - shell_j.center[d]) ** 2 for d in range(3))
            e_ab = exp(-(a * b / p) * rab2) if rab2 != 0 else mpf(1)
            tables_ab = pair_tables(la, lb, pa, pb, p)
            t_ab = fold_tables(la, lb, tables_ab, shell_i.spherical, shell_j.spherical)

            for c_idx, c in enumerate(shell_k.exponents):
                for d_idx, d in enumerate(shell_l.exponents):
                    q = c + d
                    # The axis loop variable must not shadow the exponent d:
                    # qq = (c C + d D) / q with d the ket EXPONENT (the
                    # (0,0|0,0) regression - the shadowed form fed the axis
                    # index as the exponent weight).
                    qq = tuple((c * shell_k.center[axis] + d * shell_l.center[axis]) / q
                               for axis in range(3))
                    pc = [qq[axis] - shell_k.center[axis] for axis in range(3)]
                    pd = [qq[axis] - shell_l.center[axis] for axis in range(3)]
                    rcd2 = sum((shell_k.center[axis] - shell_l.center[axis]) ** 2
                               for axis in range(3))
                    e_cd = exp(-(c * d / q) * rcd2) if rcd2 != 0 else mpf(1)
                    tables_cd = pair_tables(lc, ld, pc, pd, q)
                    t_cd = fold_tables(lc, ld, tables_cd, shell_k.spherical, shell_l.spherical)

                    alpha = p * q / (p + q)
                    dq = [pp[axis] - qq[axis] for axis in range(3)]
                    x = alpha * sum(v * v for v in dq)
                    prefactor = (2 * pi ** mpf("2.5") / (p * q * sqrt(p + q))) * e_ab * e_cd
                    seeds = [prefactor * boys(m, x) for m in range(l_total + 1)]
                    targets = vrr_targets(seeds, l_total, alpha, dq)

                    for (fa, fb), row_ab in t_ab.items():
                        for (fc, fd), row_cd in t_cd.items():
                            value = mpf(0)
                            for (tx, ty, tz), v in targets.items():
                                # [p~+q~]^(0) with p~ from (ab), q~ from (cd)
                                for px in range(tx + 1):
                                    for py in range(ty + 1):
                                        for pz in range(tz + 1):
                                            qx, qy, qz = tx - px, ty - py, tz - pz
                                            bra = row_ab.get((px, py, pz), mpf(0))
                                            ket = row_cd.get((qx, qy, qz), mpf(0))
                                            if bra != 0 and ket != 0:
                                                # d_Q^u [0] = (-1)^|u| d_P^u [0]:
                                                # the ket Hermite derivatives
                                                # enter the [t+u]^(0) target with
                                                # the sign of the ket degree (the
                                                # odd-ket-degree regression,
                                                # 2026-08-18).
                                                value += (bra * ket * v *
                                                          ((-1) ** (qx + qy + qz)))
                            # Row indices follow the layout contract: bra
                            # rows (f_j * n_i + f_i), ket columns
                            # (f_l * n_k + f_k); the weights apply per
                            # contraction row pair.
                            ang_i = 2 * shell_i.l + 1 if shell_i.spherical else len(cartesian_components(shell_i.l))
                            ang_j = 2 * shell_j.l + 1 if shell_j.spherical else len(cartesian_components(shell_j.l))
                            ang_k = 2 * shell_k.l + 1 if shell_k.spherical else len(cartesian_components(shell_k.l))
                            ang_l = 2 * shell_l.l + 1 if shell_l.spherical else len(cartesian_components(shell_l.l))
                            n_i = len(shell_i.coefficients) * ang_i
                            n_k = len(shell_k.coefficients) * ang_k
                            for ri in range(len(shell_i.coefficients)):
                                for rj in range(len(shell_j.coefficients)):
                                    for rk in range(len(shell_k.coefficients)):
                                        for rl in range(len(shell_l.coefficients)):
                                            wi = contraction_weight(shell_i, ri, a_idx)
                                            wj = contraction_weight(shell_j, rj, b_idx)
                                            wk = contraction_weight(shell_k, rk, c_idx)
                                            wl = contraction_weight(shell_l, rl, d_idx)
                                            fi = ri * ang_i + fa
                                            fj = rj * ang_j + fb
                                            fk = rk * ang_k + fc
                                            fl = rl * ang_l + fd
                                            block[fj * n_i + fi][fl * n_k + fk] += wi * wj * wk * wl * value
    return block


def eri_spots(shell_i, shell_j, shell_k, shell_l, picks):
    """Only the requested (fi, fj, fk, fl) elements of the (ij|kl) block -
    the spot-grid form of eri_block. The per-element arithmetic is identical
    to eri_block (same primitive loops, same target loop, same contraction
    scatter), only the row/col product is restricted to the picks, which
    turns the (h,h|h,h) spot set from ~an hour of 30-digit mpmath (the full
    block is 11^4 elements) into seconds. Returns {block cell (row, col):
    value} with the same cell layout as eri_block."""
    la, lb = shell_i.l, shell_j.l
    lc, ld = shell_k.l, shell_l.l
    l_total = la + lb + lc + ld
    ang_i = 2 * shell_i.l + 1 if shell_i.spherical else len(cartesian_components(shell_i.l))
    ang_j = 2 * shell_j.l + 1 if shell_j.spherical else len(cartesian_components(shell_j.l))
    ang_k = 2 * shell_k.l + 1 if shell_k.spherical else len(cartesian_components(shell_k.l))
    ang_l = 2 * shell_l.l + 1 if shell_l.spherical else len(cartesian_components(shell_l.l))
    n_i = len(shell_i.coefficients) * ang_i
    n_k = len(shell_k.coefficients) * ang_k
    out = {}

    for a_idx, a in enumerate(shell_i.exponents):
        for b_idx, b in enumerate(shell_j.exponents):
            p = a + b
            pp = tuple((a * shell_i.center[d] + b * shell_j.center[d]) / p for d in range(3))
            pa = [pp[d] - shell_i.center[d] for d in range(3)]
            pb = [pp[d] - shell_j.center[d] for d in range(3)]
            rab2 = sum((shell_i.center[d] - shell_j.center[d]) ** 2 for d in range(3))
            e_ab = exp(-(a * b / p) * rab2) if rab2 != 0 else mpf(1)
            tables_ab = pair_tables(la, lb, pa, pb, p)
            t_ab = fold_tables(la, lb, tables_ab, shell_i.spherical, shell_j.spherical)

            for c_idx, c in enumerate(shell_k.exponents):
                for d_idx, d in enumerate(shell_l.exponents):
                    q = c + d
                    # The axis loop variable must not shadow the exponent d
                    # (the eri_block regression; same trap).
                    qq = tuple((c * shell_k.center[axis] + d * shell_l.center[axis]) / q
                               for axis in range(3))
                    pc = [qq[axis] - shell_k.center[axis] for axis in range(3)]
                    pd = [qq[axis] - shell_l.center[axis] for axis in range(3)]
                    rcd2 = sum((shell_k.center[axis] - shell_l.center[axis]) ** 2
                               for axis in range(3))
                    e_cd = exp(-(c * d / q) * rcd2) if rcd2 != 0 else mpf(1)
                    tables_cd = pair_tables(lc, ld, pc, pd, q)
                    t_cd = fold_tables(lc, ld, tables_cd, shell_k.spherical, shell_l.spherical)

                    alpha = p * q / (p + q)
                    dq = [pp[axis] - qq[axis] for axis in range(3)]
                    x = alpha * sum(v * v for v in dq)
                    prefactor = (2 * pi ** mpf("2.5") / (p * q * sqrt(p + q))) * e_ab * e_cd
                    seeds = [prefactor * boys(m, x) for m in range(l_total + 1)]
                    targets = vrr_targets(seeds, l_total, alpha, dq)

                    for (fa, fb, fc, fd) in picks:
                        row_ab = t_ab.get((fa, fb))
                        row_cd = t_cd.get((fc, fd))

                        if row_ab is None or row_cd is None:
                            continue

                        value = mpf(0)

                        for (tx, ty, tz), v in targets.items():
                            # [p~+q~]^(0) with p~ from (ab), q~ from (cd)
                            for px in range(tx + 1):
                                for py in range(ty + 1):
                                    for pz in range(tz + 1):
                                        qx, qy, qz = tx - px, ty - py, tz - pz
                                        bra = row_ab.get((px, py, pz), mpf(0))
                                        ket = row_cd.get((qx, qy, qz), mpf(0))
                                        if bra != 0 and ket != 0:
                                            # d_Q^u [0] = (-1)^|u| d_P^u [0]:
                                            # the ket Hermite derivatives enter
                                            # the [t+u]^(0) target with the sign
                                            # of the ket degree (the odd-ket-
                                            # degree regression, 2026-08-18).
                                            value += (bra * ket * v *
                                                      ((-1) ** (qx + qy + qz)))

                        for ri in range(len(shell_i.coefficients)):
                            for rj in range(len(shell_j.coefficients)):
                                for rk in range(len(shell_k.coefficients)):
                                    for rl in range(len(shell_l.coefficients)):
                                        wi = shell_i.coefficients[ri][a_idx] * primitive_normalization(a) * solid_normalization(shell_i.l, a)
                                        wj = shell_j.coefficients[rj][b_idx] * primitive_normalization(b) * solid_normalization(shell_j.l, b)
                                        wk = shell_k.coefficients[rk][c_idx] * primitive_normalization(c) * solid_normalization(shell_k.l, c)
                                        wl = shell_l.coefficients[rl][d_idx] * primitive_normalization(d) * solid_normalization(shell_l.l, d)
                                        fi = ri * ang_i + fa
                                        fj = rj * ang_j + fb
                                        fk = rk * ang_k + fc
                                        fl = rl * ang_l + fd
                                        cell = (fj * n_i + fi, fl * n_k + fk)
                                        out[cell] = out.get(cell, mpf(0)) + wi * wj * wk * wl * value
    return out


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def shellset_shells():
    """The synthetic two-center s/p(/d) shellset."""
    h_exponents = ["3.4252509140", "0.6239137298", "0.1688554040"]
    h_coefficients = [["0.1543289673", "0.5353281423", "0.4446345422"]]
    p_exponents = ["1.0", "0.3"]
    p_coefficients = [["1.0", "1.0"]]
    d_exponents = ["0.8"]
    d_coefficients = [["1.0"]]
    center_a = ("0.0", "0.0", "0.0")
    center_b = ("1.4", "0.0", "0.0")
    return [
        ShellData(0, True, h_exponents, h_coefficients, center_a),
        ShellData(1, True, p_exponents, p_coefficients, center_a),
        ShellData(2, True, d_exponents, d_coefficients, center_a),
        ShellData(0, True, h_exponents, h_coefficients, center_b),
        ShellData(1, True, p_exponents, p_coefficients, center_b),
    ]


def high_l_shells():
    """The high-l spot shells: single primitives, single contraction rows,
    spherical only - the molecule-canonical order the matching test
    fixture parses (atom A: f, h; atom B: g, i)."""
    f_exponents = ["0.6"]
    f_coefficients = [["1.0"]]
    g_exponents = ["0.5"]
    g_coefficients = [["1.0"]]
    h_exponents = ["0.4"]
    h_coefficients = [["1.0"]]
    i_exponents = ["0.3"]
    i_coefficients = [["1.0"]]
    center_a = ("0.0", "0.0", "0.0")
    center_b = ("1.4", "0.0", "0.0")
    return [
        ShellData(3, True, f_exponents, f_coefficients, center_a),
        ShellData(5, True, h_exponents, h_coefficients, center_a),
        ShellData(4, True, g_exponents, g_coefficients, center_b),
        ShellData(6, True, i_exponents, i_coefficients, center_b),
    ]


def sto3g_hf():
    """The fixture HF in the engine's molecule-canonical order: Molecule::Create
    renumbers atoms by (Z, x, y, z), so the molecule is [H, F] with
    H at 1.732500911056906 bohr and F at the origin - the shells follow
    that atom order. The F SP block splits into the 2s shell (the s
    column) and the 2p shell (the p column), one contraction row each."""
    f_s = (["166.6791340", "30.36081233", "8.216820672"],
           [["0.1543289673", "0.5353281423", "0.4446345422"]])
    f_sp = (["6.464803249", "1.502281245", "0.4885884864"],
            [["-0.09996722919", "0.3995128261", "0.7001154689"],
             ["0.1559162750", "0.6076837186", "0.3919573931"]])
    h_s = (["3.4252509140", "0.6239137298", "0.1688554040"],
           [["0.1543289673", "0.5353281423", "0.4446345422"]])
    origin = ("0.0", "0.0", "0.0")
    h_center = ("1.732500911056906", "0.0", "0.0")
    return [
        ShellData(0, True, h_s[0], h_s[1], h_center),
        ShellData(0, True, f_s[0], f_s[1], origin),
        ShellData(0, True, f_sp[0], [f_sp[1][0]], origin),
        ShellData(1, True, f_sp[0], [f_sp[1][1]], origin),
    ]


def sto3g_h2o():
    """The fixture H2O in the engine's molecule-canonical order: the (Z, x,
    y, z) renumbering puts H at x = -1.430428808474167 first, then
    H at x = +1.430428808474167 (both at y = 1.107157044080814), then O at
    the origin - the shells follow that atom order."""
    o_s = (["130.7093214", "23.80886605", "6.443608313"],
           [["0.1543289673", "0.5353281423", "0.4446345422"]])
    o_sp = (["5.033151319", "1.169596125", "0.3803889600"],
            [["-0.09996722919", "0.3995128261", "0.7001154689"],
             ["0.1559162750", "0.6076837186", "0.3919573931"]])
    h_s = (["3.4252509140", "0.6239137298", "0.1688554040"],
           [["0.1543289673", "0.5353281423", "0.4446345422"]])
    origin = ("0.0", "0.0", "0.0")
    h1 = ("1.430428808474167", "1.107157044080814", "0.0")
    h2 = ("-1.430428808474167", "1.107157044080814", "0.0")
    return [
        ShellData(0, True, h_s[0], h_s[1], h2),
        ShellData(0, True, h_s[0], h_s[1], h1),
        ShellData(0, True, o_s[0], o_s[1], origin),
        ShellData(0, True, o_sp[0], [o_sp[1][0]], origin),
        ShellData(1, True, o_sp[0], [o_sp[1][1]], origin),
    ]


def nuclear_targets(seeds, l_total, p, pc, pp):
    """The 1-center VRR [t+e]^(m) = -2p (P-C)_dir [t]^(m+1)
    - 2p t_dir [t-e]^(m+1) - the ERI recurrence with a -> p, (P-Q) ->
    (P-C). Returns [t]^(0) for |t| <= l_total."""
    dq = [pp[d] - pc[d] for d in range(3)]
    alpha = p  # the substitution of the ERI VRR
    return vrr_targets(seeds, l_total, alpha, dq)


def kinetic_fold(la, lb, tables, tables_raised, spherical_a, spherical_b, a):
    """The engine's kinetic per (fa, fb) - the mirror of BuildKineticPair in
    md_one_electron.hpp: per cartesian pair
      (3a + 2a(ix+iy+iz)) E0 - 2a^2 (E0^(ix+2) + E0^(iy+2) + E0^(iz+2))
        - 1/2 (ix(ix-1) E0^(ix-2) + iy(iy-1) E0^(iy-2) + iz(iz-1) E0^(iz-2))
    with the raised rows from the (la+2, lb) per-axis tables, then the G
    folds of both shells. (The 2a ix E0^(ix-1) cross terms fold into the E0
    coefficient - the expansion of (x-A)^ix is E0 itself; validated against
    pyscf on the HF T rows, 2026-08-18.)"""
    ga = spherical_functions(la) if spherical_a else None
    gb = spherical_functions(lb) if spherical_b else None
    n_a = len(ga) if spherical_a else len(cartesian_components(la))
    n_b = len(gb) if spherical_b else len(cartesian_components(lb))
    ex, ey, ez = tables_raised
    bx, by, bz = tables
    per_cart = {}

    for anga, ca in enumerate(cartesian_components(la)):
        for angb, cb in enumerate(cartesian_components(lb)):
            ix, iy, iz = ca
            jx, jy, jz = cb
            e0 = (ex[(ix, jx)].get(0, mpf(0)) * ey[(iy, jy)].get(0, mpf(0)) *
                  ez[(iz, jz)].get(0, mpf(0)))
            e2x = (ex[(ix + 2, jx)].get(0, mpf(0)) * ey[(iy, jy)].get(0, mpf(0)) *
                   ez[(iz, jz)].get(0, mpf(0)))
            e2y = (ex[(ix, jx)].get(0, mpf(0)) * ey[(iy + 2, jy)].get(0, mpf(0)) *
                   ez[(iz, jz)].get(0, mpf(0)))
            e2z = (ex[(ix, jx)].get(0, mpf(0)) * ey[(iy, jy)].get(0, mpf(0)) *
                   ez[(iz + 2, jz)].get(0, mpf(0)))
            e2mx = (bx[(ix - 2, jx)].get(0, mpf(0)) * by[(iy, jy)].get(0, mpf(0)) *
                    bz[(iz, jz)].get(0, mpf(0))) if ix >= 2 else mpf(0)
            e2my = (bx[(ix, jx)].get(0, mpf(0)) * by[(iy - 2, jy)].get(0, mpf(0)) *
                    bz[(iz, jz)].get(0, mpf(0))) if iy >= 2 else mpf(0)
            e2mz = (bx[(ix, jx)].get(0, mpf(0)) * by[(iy, jy)].get(0, mpf(0)) *
                    bz[(iz - 2, jz)].get(0, mpf(0))) if iz >= 2 else mpf(0)
            per_cart[(anga, angb)] = (
                (3 * a + 2 * a * (ix + iy + iz)) * e0
                - 2 * a * a * (e2x + e2y + e2z)
                - mpf("0.5") * (ix * (ix - 1) * e2mx + iy * (iy - 1) * e2my +
                                iz * (iz - 1) * e2mz))

    out = {}

    for fa in range(n_a):
        for fb in range(n_b):
            value = mpf(0)

            for anga in range(len(cartesian_components(la))):
                wa = ga[fa][anga] if spherical_a else (mpf(1) if fa == anga else mpf(0))

                for angb in range(len(cartesian_components(lb))):
                    wb = gb[fb][angb] if spherical_b else (mpf(1) if fb == angb else mpf(0))
                    value += wa * wb * per_cart[(anga, angb)]

            out[(fa, fb)] = value

    return out


def one_electron_block(shell_i, shell_j, kind, nuclei):
    """The S/T/V (i|j) function block at mpmath; V is evaluated over the
    nuclei list [(Z, center)] with early charge summation (the VRR runs per
    nucleus - its coefficients carry (P-C) - the charge sum folds into the
    [t]^(0) targets, and one bra contraction runs)."""
    la, lb = shell_i.l, shell_j.l
    l_total = la + lb
    n_i = function_count(shell_i)
    n_j = function_count(shell_j)
    block = [[mpf(0) for _ in range(n_j)] for _ in range(n_i)]
    for a_idx, a in enumerate(shell_i.exponents):
        for b_idx, b in enumerate(shell_j.exponents):
            p = a + b
            pp = tuple((a * shell_i.center[d] + b * shell_j.center[d]) / p for d in range(3))
            pa = [pp[d] - shell_i.center[d] for d in range(3)]
            pb = [pp[d] - shell_j.center[d] for d in range(3)]
            rab2 = sum((shell_i.center[d] - shell_j.center[d]) ** 2 for d in range(3))
            e_ab = exp(-(a * b / p) * rab2) if rab2 != 0 else mpf(1)
            tables = pair_tables(la, lb, pa, pb, p)
            folded = fold_tables(la, lb, tables, shell_i.spherical, shell_j.spherical)
            t_folded = (kinetic_fold(la, lb, tables, pair_tables(la + 2, lb, pa, pb, p),
                                     shell_i.spherical, shell_j.spherical, a)
                        if kind == "T" else None)
            if kind == "S":
                pref = (pi / p) ** mpf("1.5") * e_ab
            elif kind == "T":
                pref = (pi / p) ** mpf("1.5") * e_ab
            else:
                pref = (2 * pi / p) * e_ab
            for (fa, fb), row in folded.items():
                if kind == "S":
                    value = row.get((0, 0, 0), mpf(0))
                elif kind == "T":
                    value = t_folded.get((fa, fb), mpf(0))
                else:
                    g = {}
                    for charge, center in nuclei:
                        rpc2 = sum((pp[d] - center[d]) ** 2 for d in range(3))
                        seeds = [charge * boys(m, p * rpc2) for m in range(l_total + 1)]
                        targets = nuclear_targets(seeds, l_total, p, center, pp)
                        for t, v in targets.items():
                            g[t] = g.get(t, mpf(0)) + v
                    value = -sum(row.get(t, mpf(0)) * g.get(t, mpf(0))
                                 for t in set(row) | set(g))
                for ri in range(len(shell_i.coefficients)):
                    for rj in range(len(shell_j.coefficients)):
                        wi = shell_i.coefficients[ri][a_idx] * primitive_normalization(a) * solid_normalization(shell_i.l, a)
                        wj = shell_j.coefficients[rj][b_idx] * primitive_normalization(b) * solid_normalization(shell_j.l, b)
                        ang_i = 2 * shell_i.l + 1 if shell_i.spherical else len(cartesian_components(shell_i.l))
                        ang_j = 2 * shell_j.l + 1 if shell_j.spherical else len(cartesian_components(shell_j.l))
                        fi = ri * ang_i + fa
                        fj = rj * ang_j + fb
                        block[fi][fj] += wi * wj * pref * value
    return block


def build_all_rows(max_l=None):
    """Builds every grid row; returns {filename: rows}."""
    grids = {}
    rows = []
    # The shellset grid: 1e blocks (S/T/V) of every canonical pair and the
    # 2e blocks of every canonical quartet of the nine s/p pairs. The
    # shellset V is evaluated over the matching test molecule's nuclei:
    # H (Z=1) at the origin, He (Z=2) at (1.4, 0, 0) - the fixture the
    # eri_batch_test uses.
    shells = shellset_shells()
    nuclei = [(mpf(1), tuple(mpf(str(v)) for v in ("0.0", "0.0", "0.0"))),
              (mpf(2), tuple(mpf(str(v)) for v in ("1.4", "0.0", "0.0")))]
    for i in range(len(shells)):
        for j in range(i, len(shells)):
            block_s = one_electron_block(shells[i], shells[j], "S", nuclei)
            block_t = one_electron_block(shells[i], shells[j], "T", nuclei)
            block_v = one_electron_block(shells[i], shells[j], "V", nuclei)
            for fi in range(len(block_s)):
                for fj in range(len(block_s[0])):
                    rows.append(["S", "shellset", i, j, "", "", fi, fj, "", "",
                                 mp.nstr(block_s[fi][fj], 22)])
                    rows.append(["T", "shellset", i, j, "", "", fi, fj, "", "",
                                 mp.nstr(block_t[fi][fj], 22)])
                    rows.append(["V", "shellset", i, j, "", "", fi, fj, "", "",
                                 mp.nstr(block_v[fi][fj], 22)])

    # The 2e quartets of the nine s/p pairs (shells 0,1,3,4 - the d shell 2
    # is 1e-only here). The CSV quartet fields carry the ORIGINAL shell
    # indices (the d-shell gap is not collapsed - the consuming test maps
    # them straight through PairIndexOf).
    sp_shell_ids = [0, 1, 3, 4]
    sp_shells = [shells[i] for i in sp_shell_ids]
    n_sp = len(sp_shells)
    for a in range(n_sp):
        for b in range(a, n_sp):
            for c in range(n_sp):
                for d in range(c, n_sp):
                    if (a, b) < (c, d):
                        continue
                    block = eri_block(sp_shells[a], sp_shells[b], sp_shells[c], sp_shells[d])
                    for row_i in range(len(block)):
                        for col_i in range(len(block[0])):
                            rows.append(
                                ["ERI", "shellset", sp_shell_ids[a], sp_shell_ids[b],
                                 sp_shell_ids[c], sp_shell_ids[d], row_i, col_i, "", "",
                                 mp.nstr(block[row_i][col_i], 22)])
    grids["md_shellset_reference.csv"] = rows

    # The high-l spot grid: canonical quartets of the f/g/h/i shells (the
    # l = 3..6 classes, including (6,6)) with a fixed pattern of function
    # indices per quartet - spot checks, not full tensors (the full (i,i|i,i)
    # tensor alone is 13^4 rows). ERI rows only. Gated on max_l: builds that
    # cannot instantiate the f shell (Lmax < 3,
    # the CI lanes) skip the whole grid - the regeneration check runs on
    # every CI job and this section is the expensive one (2026-08-18).
    if max_l is None or max_l >= 3:
        hl_shells = high_l_shells()
        # Classes 6..10 (f/g/h): the l = 6-shell quartets (classes up to (12,12))
        # are deliberately out - a full i,i|i,i block is ~13M mpmath inner
        # operations (~30x the whole f/g/h set) and the regeneration check runs
        # on every CI job; the (6,6) shell class is validated by the unfolded
        # reference path and the mixed-precision sweep instead.
        hl_quartets = [(0, 0, 0, 0), (0, 1, 0, 1), (0, 2, 0, 2), (1, 1, 1, 1), (1, 2, 1, 2),
                       (2, 2, 2, 2)]
        spot_patterns = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
                         (1, 1, 1, 1), (2, 0, 0, 2), (1, 2, 3, 4), (0, 3, 1, 2), (2, 1, 0, 3),
                         (3, 1, 1, 2), (1, 1, 3, 3)]
        rows = []

        for a, b, c, d in hl_quartets:
            n_a = function_count(hl_shells[a])
            n_b = function_count(hl_shells[b])
            n_c = function_count(hl_shells[c])
            n_d = function_count(hl_shells[d])
            picks = list(spot_patterns)
            picks.append((n_a - 1, n_b - 1, n_c - 1, n_d - 1))
            picks = [(fi, fj, fk, fl) for (fi, fj, fk, fl) in picks
                     if fi < n_a and fj < n_b and fk < n_c and fl < n_d]
            cells = eri_spots(hl_shells[a], hl_shells[b], hl_shells[c], hl_shells[d], picks)

            for fi, fj, fk, fl in picks:
                row_i = fj * n_a + fi
                col_i = fl * n_c + fk
                rows.append(["ERI", "high-l", a, b, c, d, row_i, col_i, "", "",
                             mp.nstr(cells[(row_i, col_i)], 22)])

        grids["md_high_l_reference.csv"] = rows

    # The molecule grids: HF and H2O STO-3G (real nuclear charges at the
    # fixture geometries).
    hf_nuclei = [(mpf(9), (mpf(0), mpf(0), mpf(0))),
                 (mpf(1), (mpf("1.732500911056906"), mpf(0), mpf(0)))]
    h2o_nuclei = [(mpf(8), (mpf(0), mpf(0), mpf(0))),
                  (mpf(1), (mpf("1.430428808474167"), mpf("1.107157044080814"), mpf(0))),
                  (mpf(1), (mpf("-1.430428808474167"), mpf("1.107157044080814"), mpf(0)))]
    for name, make, nuclei in (("hf_sto3g", sto3g_hf, hf_nuclei),
                               ("h2o_sto3g", sto3g_h2o, h2o_nuclei)):
        mol_shells = make()
        rows = []
        for i in range(len(mol_shells)):
            for j in range(i, len(mol_shells)):
                block_s = one_electron_block(mol_shells[i], mol_shells[j], "S", nuclei)
                block_t = one_electron_block(mol_shells[i], mol_shells[j], "T", nuclei)
                block_v = one_electron_block(mol_shells[i], mol_shells[j], "V", nuclei)
                for fi in range(len(block_s)):
                    for fj in range(len(block_s[0])):
                        rows.append(["S", name, i, j, "", "", fi, fj, "", "",
                                     mp.nstr(block_s[fi][fj], 22)])
                        rows.append(["T", name, i, j, "", "", fi, fj, "", "",
                                     mp.nstr(block_t[fi][fj], 22)])
                        rows.append(["V", name, i, j, "", "", fi, fj, "", "",
                                     mp.nstr(block_v[fi][fj], 22)])
        for a in range(len(mol_shells)):
            for b in range(a, len(mol_shells)):
                for c in range(len(mol_shells)):
                    for d in range(c, len(mol_shells)):
                        if (a, b) < (c, d):
                            continue
                        block = eri_block(mol_shells[a], mol_shells[b], mol_shells[c],
                                          mol_shells[d])
                        for row_i in range(len(block)):
                            for col_i in range(len(block[0])):
                                rows.append(["ERI", name, a, b, c, d, row_i, col_i, "", "",
                                             mp.nstr(block[row_i][col_i], 22)])
        grids[f"{name}_reference.csv"] = rows

    # The 3c RI grid: the (uv|P) blocks of every canonical
    # H2O/STO-3G orbital pair against the tiny s/p auxiliary set above, one
    # function triple per task (plus the corners of the (p,p) x O-p block),
    # and the (P|Q) function metric elements - both through the phantom
    # (P, s_0) construction of ri_engine.cpp. Columns: kind ("R" = 3c with
    # shells i, j, p and within-shell functions ri, rj, rp; "M" = metric
    # with aux shells i, j and within-shell functions ri, rj).
    mol_shells = sto3g_h2o()
    aux_shells = tiny_aux_shells()
    rows = []

    for i in range(len(mol_shells)):
        for j in range(i, len(mol_shells)):
            for p_idx in range(len(aux_shells)):
                block = ri_block(mol_shells[i], mol_shells[j], aux_shells[p_idx])
                rows.append(["R", i, j, p_idx, 0, 0, 0, mp.nstr(block[0][0], 22)])

                # The corners of the (p,p) x O-p block: the O-p bra pair
                # (shell 4 = the molecule's only p shell) with the O-p aux
                # (index 5) so the corner points land on real p functions
                # (ri, rj, rp = 2). The earlier (s,s) x O-s choice collapsed
                # onto the 1-function bra and duplicated the base row.
                if i == 4 and j == 4 and p_idx == 5:
                    n_i = function_count(mol_shells[i])
                    n_p = len(block[0])

                    for (ri, rj, rp) in ((0, 0, n_p - 1), (n_i - 1, 0, 0),
                                         (n_i - 1, n_i - 1, n_p - 1)):
                        rows.append(["R", i, j, p_idx, ri, rj, rp,
                                     mp.nstr(block[rj * n_i + ri][rp], 22)])

    for i in range(len(aux_shells)):
        for j in range(i, len(aux_shells)):
            # The (P|Q) function metric: the phantom quartets
            # (P, s_0 | Q, s_0) - the engine's BuildAuxMetric construction.
            p_i = ShellData(0, False, ["0.0"], [[1.0]], aux_shells[i].center)
            p_j = ShellData(0, False, ["0.0"], [[1.0]], aux_shells[j].center)
            block = eri_block(aux_shells[i], p_i, aux_shells[j], p_j)
            n_i = function_count(aux_shells[i])
            n_j = function_count(aux_shells[j])

            for fi in range(n_i):
                for fj in range(n_j):
                    # With the phantom (one function) on both bra and ket
                    # sides, the block cell (fi, fj) is (P_i, s_0 | P_j, s_0).
                    rows.append(["M", i, j, 0, fi, fj, 0,
                                 mp.nstr(block[fi][fj], 22)])

    grids["md_3c_reference.csv"] = rows
    return grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="integrals/tests/data")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed CSVs are byte-identical to a "
                             "fresh generation (exits nonzero on drift)")
    parser.add_argument("--max-l", type=int, default=None,
                        help="skip grids whose shells exceed this l (the CI lanes "
                             "pass their QCX_INTEGRALS_LMAX)")
    args = parser.parse_args()

    # Resolve the relative outdir against the repo root (the script's
    # parent directory): the regeneration ctest runs from the build
    # directory.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.outdir = (args.outdir if os.path.isabs(args.outdir)
                   else os.path.join(repo_root, args.outdir))

    grids = build_all_rows(args.max_l)

    if args.check:
        ok = True
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in grids.items():
                fresh = os.path.join(tmp, name)
                write_csv(tmp, name, rows)
                committed = os.path.join(args.outdir, name)
                if not os.path.exists(committed):
                    print(f"DRIFT: {committed} is missing")
                    ok = False
                    continue
                with open(fresh, "rb") as ff, open(committed, "rb") as fc:
                    if ff.read() != fc.read():
                        print(f"DRIFT: {committed} differs from a fresh generation")
                        ok = False
        return 0 if ok else 1

    os.makedirs(args.outdir, exist_ok=True)
    for name, rows in grids.items():
        write_csv(args.outdir, name, rows)
    return 0


def write_csv(outdir, name, rows):
    path = os.path.join(outdir, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")

        if name == "md_3c_reference.csv":
            w.writerow(["kind", "i", "j", "p", "ri", "rj", "rp", "value"])
        else:
            w.writerow(["kind", "mol", "i", "j", "k", "l", "fi", "fj", "fk", "fl", "value"])

        for row in rows:
            w.writerow(row)
    print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    sys.exit(main())
