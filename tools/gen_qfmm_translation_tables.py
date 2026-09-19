#!/usr/bin/env python3
"""Generate the QFMM translation-operator tables ("generate and verify, don't
hand-derive": every table is derived here and each entry is checked before
emission; an earlier plan for this work was dropped in 397cd9e).

Emits integrals/src/internal/qfmm_tables_gen.hpp: the M2M, M2L and L2L
coefficient tables for multipole order L_mult <= kQfmmMaxLMult,
expressed in the monomial basis of degree <= 2 * kQfmmMaxLMult (84
entries at L = 3; the basis grows with the cap so the M2L numerator
degrees l_s + l_t <= 2 L and the C-table shift degrees l_t - l_s <= L
both fit at every cap). Requires: pip install mpmath. Structured
exactly like tools/gen_md_tables.py (30-digit mpmath, derive -> verify
-> emit, --check byte-identity); the solid-harmonic rows are imported
from gen_md_tables.solid_harmonic_table, so the tables agree with the
kSolidHarmonicG the MD engine folds.

Conventions (all locked against the physical checks in the verification
suite; the real solid harmonics R_lm are the codebase's G rows - Racah /
quantum-chemistry normalization, unit W-norm for l >= 1, R_00 = 1 with
W-norm N_0 = 4 pi):

  Moment packing:  idx = l^2 + (m + l), rows m = -l..+l - the
  (kQfmmMaxLMult + 1)^2 moments of order <= kQfmmMaxLMult.
  Monomial basis:  all (a, b, c) with a + b + c <= 2 * kQfmmMaxLMult,
  emitted as kQfmmMonomialExponents - the numerator degrees of the
  D-tilde table (l_source + l_target <= 2 L) and the C-table shift
  degrees (l_source - l_target <= L).
  C[lm, l'm'](d): the coefficient of R_lm(x) in R_{l'm'}(x + d) -
      C[lm,l'm'](d) = (1/N_l) sum_{|cart| = l} (W G^T)[cart,lm]
                      sum_{cart' >= cart} G[l'm'][cart'] binom(cart',cart)
                      d^{cart'-cart}
  (W = the angular monomial overlap; the projection is exact because
  R_{l'm'}(x + d) is harmonic in x, so its homogeneous parts are all
  harmonic. Nonzero only for l' <= l (a shifted source acquires only
  equal-or-higher harmonics; the same-l block is diagonal in m, with
  additional m-selection zeros at l' = l - 1), and an entry's monomials
  have d-degree = l - l' (mod 2) - the parity lives in the monomial
  degrees, not in a coefficient parity zero.)
  The emitted kQfmmM2M is the TRANSPOSE of C: kQfmmM2M[t][s][mono] = the
  coefficient of R_s(x) in R_t(x + d), so the M2M reads it directly:
      M_t(s + d) = sum_s sum_mono kQfmmM2M[t][s][mono] (-1)^|mono| d^mono M_s
  and the L2L reads it transposed (both directions documented in the
  header; the (-1)^|mono| is the C(-d) of the moment shift, the +d of the
  local shift).
  D-tilde[lm, l'm'](d): the local-expansion coefficient of R_lm(delta) in
  the potential of a Racah-moment source:
      V(d + delta) = sum_lm L_lm(d) R_lm(delta),
      L_lm(d) = sum_{l'm'} D-tilde[lm, l'm'](d) M_{l'm'},
      D-tilde[lm, l'm'](d) = f(l') (1/N_l) sum_{|cart| = l} (W G^T)[cart,lm]
                             (1/cart!) d^cart I_{l'm'}(d),
      I_{l'm'}(d) = R_{l'm'}(d) / |d|^(2l'+1),
      f(l') = (4 pi / (2 l' + 1)) / N_{l'}   (the addition-theorem factor
      1/|d - s| = sum_{l'm'} f(l') R_{l'm'}(s) R_{l'm'}(d) / |d|^(2l'+1);
      f(0) = 1 so the monopole column is the bare Coulomb term).
  The derivative is computed by the exact recursion
      d_j [c d^M |d|^p] = c M_j (M - e_j, p) + c p (M + e_j, p - 2)
  and put over the common denominator |d|^(2l'+1+2l): each term becomes
  c d^M (x^2 + y^2 + z^2)^(l - n) with n the number of p-branch picks - a
  homogeneous polynomial of degree l' + l <= 2 * QFMM_MAX_L, exactly the
  emitted basis.
  The emission carries the f-rescaling: kQfmmM2L[t][s][mono] = f-rescaled
  numerator coefficients, kQfmmM2LDenomPower[t][s] = 2 l' + 1 + 2 l, so
      L_t(d) = sum_s sum_mono kQfmmM2L[t][s][mono] d^mono M_s
               / |d|^kQfmmM2LDenomPower[t][s].

The verification suite (VERIFY BEFORE EMIT: if
any check fails, stop, report exactly which test failed and by how much,
and do not ship an unverified translation operator; emission is refused):

  V1  C definition:       sum_t kQfmmM2M[s][t](d) R_t(x) == R_s(x + d)
                          (the +d read), all s, random d, x.
  V2  M2M composition:    M2M(t, s, d1 + d2) == sum_s' M2M(t, s', d2)
                          M2M(s', s, d1) - shifts compose.
  V3  D-tilde vs d:       (a) the recursion's d^cart I_s(d) against nested
                          mp.diff of the closed form R_s(d)|d|^{-(2l'+1)}
                          (spot checks); (b) the Taylor-order pin:
                          |f(l') I_s(d + delta) - sum_t M2L(t,s,d) R_t(delta)|
                          scales as |delta|^4.
  V4  Cloud potential:    a charge cloud's potential via
                          L = M2L(d) M vs the direct sum q_j/|y - r_j| at
                          several points outside the cloud; the error at
                          every truncation order L must sit below the
                          (R/|d|)^(L+1) ladder (the per-order moment
                          coefficients of a RANDOM cloud are not monotone -
                          strict monotonicity is owned by the C++
                          acceptance suite on C12H26). The local
                          expansion at a nearby point (all targets) is
                          checked at a truncation-appropriate tolerance.
  V5  Two-step shifts:    REMOVED - the composition identities are
                          truncation-inexact by construction: the shifted
                          source's moments above L and the L2L shift's
                          local orders above L are dropped, so
                          sum_s' M2L(t, s', d2) M2M(s', s, d1) differs from
                          M2L(t, s, d1 + d2) by a tail that no geometry can
                          suppress. Four verification runs (2026-08-23)
                          pinned the deviation to the truncation structure
                          itself: V5a failed exactly where l_s = 3 (the
                          dropped source orders; the shifted kernel
                          R_s(y + d1)/|y + d1|^(2l_s+1) has moments at ALL
                          orders), V5b exactly where l_t = 3 (the dropped
                          local orders), at 1e-5..4e-3 against
                          max(|target|, largest |summand|) even at the
                          0.02 ratio. No finite tolerance makes the
                          composition check meaningful at the edge orders,
                          and no composition-level check can separate a
                          table defect from the truncation. The
                          verification role is replaced by V7 (exact, per
                          entry); the composition itself is not re-checked
                          here - the C++ acceptance suite runs the
                          real cascade.
  V7  M2L entries:        each emitted kQfmmM2L[t][s] against the
                          independent angular projection of the physical
                          moment-s field: the local-expansion coefficient
                          L_t = delta^{-l_t} <R_t, f(l') I_s(d + delta u)> /
                          <R_t, R_t> over the unit sphere, by 2D
                          Gauss-Legendre quadrature - exact up to
                          quadrature by the harmonic orthogonality (no
                          truncation at any order), and independent of the
                          emitted algebra (the d^cart I_s recursion, the
                          A/N projection, the f rescaling and the index
                          conventions all stand or fall together).
  V6  Moment shift:       moments of a charge cloud about s + d via the
                          emitted M2M (with the (-1)^|mono|) vs the direct
                          moments about s + d - the sign convention.

The committed header must match this script's output byte for byte; the
regeneration ctest runs `--check` (the gen_md_tables precedent).
"""
import argparse
import os
import random
import sys
import tempfile

# Import the pinned solid-harmonic machinery from the MD table generator:
# the same rows, in the same z-slowest order, at the same 30-digit precision
# - the emitted tables then agree with the kSolidHarmonicG the engine folds.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_md_tables import (  # noqa: E402
    angular_moment,
    cartesian_components,
    fmt,
    format_header,
    solid_harmonic_table,
)

try:
    from mpmath import mp, mpf, pi, sqrt, cos, sin  # noqa: F401
except ImportError:
    if "--check" in sys.argv:
        # The regeneration ctest: CI jobs without the pip package skip.
        print("SKIP: mpmath is not installed (the --check mode requires it)")
        sys.exit(77)
    raise

from math import comb, factorial  # noqa: E402

mp.dps = 30  # reference error ~1e-25 to 1e-30: 11+ orders below the double floor 1e-14, and ~4 orders under the D=50 worst V7 norm residual 2.96e-21

QFMM_MAX_L = 8  # kQfmmMaxLMult (8 keeps the emitted header small; L=10 measured ~30-36 MB)
MOMENT_COUNT = (QFMM_MAX_L + 1) ** 2  # 81 at the committed cap (16 at L = 3)
# The monomial basis of degree <= 2 * QFMM_MAX_L: all triples, a slowest
# (84 entries at L = 3, 969 at L = 8, 1771 at L = 10) - the M2L numerator
# degrees (l_s + l_t <= 2 L) and the M2M/L2L shift degrees (l_t - l_s <= L)
# both fit. This is the emission basis (kQfmmMonomialExponents), NOT the
# z-slowest Cartesian basis of the G tables - the two are distinct layouts
# and both are documented.
MONOS = [(a, b, c) for a in range(2 * QFMM_MAX_L + 1)
         for b in range(2 * QFMM_MAX_L + 1 - a)
         for c in range(2 * QFMM_MAX_L + 1 - a - b)]
MONO_INDEX = {mono: i for i, mono in enumerate(MONOS)}

PASS = 0
FAILS = []


def check(identity, lhs, rhs, tolerance):
    """One exact-identity check; on failure record the identity and the
    deviation."""
    global PASS
    diff = abs(lhs - rhs)
    scale = max(abs(rhs), mpf(1))
    if diff <= tolerance * scale:
        PASS += 1
    else:
        FAILS.append((identity, float(diff / scale), float(diff)))


def snap(v):
    """Exact zeros only: the structural zeros of the tables (the
    l_source > l_target entries of C - a shifted source never acquires
    lower harmonics) are emitted as exactly 0.0. The smallest genuine
    coefficient magnitude is ~0.315 (M2M) / ~0.79 (M2L), while the
    30-digit arithmetic leaves cancellation residues up to ~5e-15 (22 M2L
    entries at t = (3, +/-3); true value exactly 0, verified at 80
    digits); the 1e-12 cutoff zeros every residue and still sits 11+
    orders below the genuine floor."""
    if abs(v) < mpf("1e-12"):
        return mpf(0)
    return v


def pack(l, m):
    return l * l + m + l


def unpack(idx):
    l = int(idx ** 0.5)
    return l, idx - l * l - l


def binom3(cartp, cart):
    return comb(cartp[0], cart[0]) * comb(cartp[1], cart[1]) * comb(cartp[2], cart[2])


# ---------------------------------------------------------------------------
# The pinned solid-harmonic machinery (replicated from gen_md_tables.py so
# the verification and the emission agree with the C++ kSolidHarmonicG).
# ---------------------------------------------------------------------------

# G[l][m + l][cart]: the real solid harmonics, z-slowest Cartesian rows.
G = [[[mpf(v) for v in row] for row in solid_harmonic_table(l)] for l in range(QFMM_MAX_L + 1)]
CARTS = [cartesian_components(l) for l in range(QFMM_MAX_L + 1)]

# W[l][i][j] = the angular monomial overlap of degree-l monomials; the
# W-norms N[l] = <R_lm, R_lm>_W (row m = -l): N_0 = 4 pi (R_00 = 1), and
# N_l = 1 for l >= 1 (unit normalization - the generator's own
# verify_orthonormal pins G W G^T = I there).
W = [[[angular_moment(ci[0] + cj[0], ci[1] + cj[1], ci[2] + cj[2])
       for cj in CARTS[l]] for ci in CARTS[l]] for l in range(QFMM_MAX_L + 1)]
N = [sum(G[l][0][i] * W[l][i][j] * G[l][0][j]
         for i in range(len(CARTS[l])) for j in range(len(CARTS[l])))
     for l in range(QFMM_MAX_L + 1)]
# A[l][cart][m + l] = (W G^T)[cart, lm]: the dual coefficients such that the
# harmonic projection of the degree-l part of any polynomial p is
#   sum_lm [sum_cart A[l][cart][lm] p_cart] R_lm.
A = [[[sum(W[l][i][j] * G[l][t][j] for j in range(len(CARTS[l])))
       for t in range(2 * l + 1)] for i in range(len(CARTS[l]))]
     for l in range(QFMM_MAX_L + 1)]


def R_value(l, m, x):
    """R_lm(x) = sum_cart G[l][m + l][cart] x^cart (the z-slowest order); m
    is the angular index in [-l, l], the row is m + l."""
    carts = CARTS[l]
    row = G[l][m + l]
    return sum(row[i] * x[0] ** carts[i][0] * x[1] ** carts[i][1] * x[2] ** carts[i][2]
               for i in range(len(carts)))


def f_source(l):
    """The addition-theorem factor f(l') = (4 pi / (2 l' + 1)) / N_{l'}:
    f(0) = 1 (the monopole column is the bare Coulomb term)."""
    return (4 * pi / (2 * l + 1)) / N[l]


# ---------------------------------------------------------------------------
# The C table (M2M/L2L) and the D-tilde table (M2L)
# ---------------------------------------------------------------------------

def c_coeffs(row_l, row_m, col_l, col_m):
    """The coefficients of R_{row}(x) in R_{col}(x + d), as {mono: coeff}:
    C[row, col](d) = (1/N_row) sum_{|cart| = row_l} A[row][cart][row_m]
    sum_{cart' >= cart} G[col][col_m][cart'] binom(cart', cart) d^{cart'-cart}
    - the exact projection, homogeneous of degree col_l - row_l in d: zero
    for row_l > col_l, the row_l = col_l block diagonal in m, and the
    monomial d-degrees = col_l - row_l (mod 2) - no coefficient parity
    zeros."""
    acc = {}

    for cart_i, cart in enumerate(CARTS[row_l]):
        a_coeff = A[row_l][cart_i][row_m + row_l]

        for cart_j, cartp in enumerate(CARTS[col_l]):
            if cartp[0] < cart[0] or cartp[1] < cart[1] or cartp[2] < cart[2]:
                continue
            mono = (cartp[0] - cart[0], cartp[1] - cart[1], cartp[2] - cart[2])
            c = G[col_l][col_m + col_l][cart_j] * binom3(cartp, cart)
            acc[mono] = acc.get(mono, mpf(0)) + a_coeff * c

    inv_norm = 1 / N[row_l]
    return {mono: inv_norm * v for mono, v in acc.items()}


def d_terms(cart, l_s, m_s):
    """d^cart I_s(d) as the exact recursion sum of c d^M |d|^p terms:
    d_j [c d^M |d|^p] = c M_j (M - e_j, p) + c p (M + e_j, p - 2), seeded
    from I_s(d) = sum_cart' G[s][cart'] d^cart' |d|^{-(2 l_s + 1)}."""
    p0 = -(2 * l_s + 1)
    terms = [(G[l_s][m_s + l_s][j], CARTS[l_s][j], p0) for j in range(len(CARTS[l_s]))]

    for axis in range(3):
        for _ in range(cart[axis]):
            nxt = []

            for c, M, p in terms:
                if M[axis] > 0:
                    M2 = M[:axis] + (M[axis] - 1,) + M[axis + 1:]
                    nxt.append((c * M[axis], M2, p))
                M3 = M[:axis] + (M[axis] + 1,) + M[axis + 1:]
                nxt.append((c * p, M3, p - 2))

            terms = nxt

    return terms


def num_poly(cart, l_s, m_s, l_t):
    """The numerator of d^cart I_s(d) over the common denominator
    |d|^(2 l_s + 1 + 2 l_t): |d|^(2 l_s + 1 + 2 l_t) d^cart I_s(d), as
    {mono: coeff} - a homogeneous polynomial of degree l_s + l_t. With n
    the p-branch picks of the term (p = -(2 l_s + 1) - 2 n), each term
    becomes c d^M (x^2 + y^2 + z^2)^(l_t - n), expanded by the trinomial
    theorem into the monomial basis of degree <= 2 * QFMM_MAX_L."""
    p0 = -(2 * l_s + 1)
    poly = {}

    for c, M, p in d_terms(cart, l_s, m_s):
        n = (p0 - p) // 2
        rem = l_t - n

        for a in range(rem + 1):
            for b in range(rem - a + 1):
                cc = rem - a - b
                mono = (M[0] + 2 * a, M[1] + 2 * b, M[2] + 2 * cc)
                coeff = c * factorial(rem) / (factorial(a) * factorial(b) * factorial(cc))
                poly[mono] = poly.get(mono, mpf(0)) + coeff

    return poly


def build_tables():
    """The emitted tables: kQfmmM2M[t][s][mono] (the TRANSPOSE of C:
    coefficient of R_s(x) in R_t(x + d)) and kQfmmM2L[t][s][mono] (the
    f-rescaled D-tilde numerators)."""
    m2m = [[{} for _ in range(MOMENT_COUNT)] for _ in range(MOMENT_COUNT)]

    for t in range(MOMENT_COUNT):
        l_t, m_t = unpack(t)
        for s in range(MOMENT_COUNT):
            l_s, m_s = unpack(s)
            m2m[t][s] = c_coeffs(row_l=l_s, row_m=m_s, col_l=l_t, col_m=m_t)

    m2l = [[{} for _ in range(MOMENT_COUNT)] for _ in range(MOMENT_COUNT)]
    denom = [[0] * MOMENT_COUNT for _ in range(MOMENT_COUNT)]

    for t in range(MOMENT_COUNT):
        l_t, m_t = unpack(t)
        inv_norm = 1 / N[l_t]
        for s in range(MOMENT_COUNT):
            l_s, m_s = unpack(s)
            denom[t][s] = 2 * l_s + 1 + 2 * l_t
            acc = {}

            for cart_i, cart in enumerate(CARTS[l_t]):
                a_coeff = A[l_t][cart_i][m_t + l_t]

                if a_coeff == 0:
                    continue

                fact_inv = 1 / (factorial(cart[0]) * factorial(cart[1]) * factorial(cart[2]))

                for mono, coeff in num_poly(cart, l_s, m_s, l_t).items():
                    acc[mono] = acc.get(mono, mpf(0)) + a_coeff * fact_inv * coeff

            m2l[t][s] = {mono: f_source(l_s) * inv_norm * v for mono, v in acc.items()}

    return m2m, m2l, denom


# ---------------------------------------------------------------------------
# The verification suite (V1..V7; verify before emit)
# ---------------------------------------------------------------------------

def mono_values(d):
    """d^mono for every emission-basis monomial, at the working precision."""
    dx, dy, dz = d
    return [dx ** mono[0] * dy ** mono[1] * dz ** mono[2] for mono in MONOS]


def eval_m2m(m2m, t, s, d, sign):
    """sum_mono m2m[t][s][mono] sign^|mono| d^mono: sign = -1 is the M2M
    read (C(-d), the moment shift), sign = +1 the L2L read (C(+d))."""
    return sum(m2m[t][s].get(mono, mpf(0)) * (sign ** (mono[0] + mono[1] + mono[2]))
               * mv for mono, mv in zip(MONOS, mono_values(d)))


def eval_m2l(m2l, denom, t, s, d):
    """sum_mono m2l[t][s][mono] d^mono / |d|^denom[t][s]."""
    dx, dy, dz = d
    r2 = dx * dx + dy * dy + dz * dz
    num = sum(m2l[t][s].get(mono, mpf(0)) * mv
              for mono, mv in zip(MONOS, mono_values(d)))
    return num / (r2 ** (denom[t][s] / 2))


def verify(m2m, m2l, denom):
    """The full suite. Returns True iff every check passes; FAILS carries
    the exact failures (identity + deviation) otherwise."""
    rng = random.Random(20260823)
    rand_d = lambda: tuple(mpf(rng.uniform(-2.0, 2.0)) for _ in range(3))
    rand_x = lambda: tuple(mpf(rng.uniform(-0.4, 0.4)) for _ in range(3))

    # V1: the C definition through the emission: R_s(x + d) ==
    # sum_t kQfmmM2M[s][t](+d) R_t(x), all s, several (d, x).
    print("=== V1: the C definition (R_s(x + d) through the emitted M2M) ===")
    for l_s in range(QFMM_MAX_L + 1):
        for m_s in range(-l_s, l_s + 1):
            s = pack(l_s, m_s)
            for _ in range(3):
                d = rand_d()
                x = rand_x()
                target = R_value(l_s, m_s, (x[0] + d[0], x[1] + d[1], x[2] + d[2]))
                lhs = sum(eval_m2m(m2m, s, t, d, +1) * R_value(*unpack(t), x)
                          for t in range(MOMENT_COUNT))
                check(f"V1: s = ({l_s}, {m_s}), d = {d}, x = {x}", lhs, target, mpf("1e-22"))

    # V2: the M2M composition: M2M(t, s, d1 + d2) == sum_s' M2M(t, s', d2)
    # M2M(s', s, d1) - the shifts compose (C(-d2)C(-d1) = C(-d1-d2)).
    print("=== V2: the M2M composition ===")
    for _ in range(4):
        d1 = rand_d()
        d2 = rand_d()
        d12 = tuple(d1[i] + d2[i] for i in range(3))
        for t in range(MOMENT_COUNT):
            for s in range(MOMENT_COUNT):
                lhs = sum(eval_m2m(m2m, t, sp, d2, -1) * eval_m2m(m2m, sp, s, d1, -1)
                          for sp in range(MOMENT_COUNT))
                target = eval_m2m(m2m, t, s, d12, -1)
                check(f"V2: t = {unpack(t)}, s = {unpack(s)}", lhs, target, mpf("1e-22"))

    # V3a: the derivative recursion against nested mp.diff of the closed
    # form I_s(d) = R_s(d) |d|^{-(2 l_s + 1)} (spot checks - numerical
    # differentiation, so a looser tolerance than the exact identities).
    print("=== V3a: d^cart I_s(d) vs nested mp.diff ===")
    for (l_s, m_s, cart) in [(1, 0, (1, 0, 0)), (2, 1, (0, 1, 1)), (3, -2, (1, 0, 2))]:
        for _ in range(2):
            d = tuple(mpf(rng.uniform(1.5, 3.0)) for _ in range(3))

            def i_s(point):
                return R_value(l_s, m_s, point) / (point[0] ** 2 + point[1] ** 2 + point[2] ** 2) ** (l_s + mpf("0.5"))

            # The nested wrappers: the innermost differentiates the z slot,
            # the outermost the x slot (the mixed partials commute).
            g = i_s

            for axis in range(3):
                n = cart[axis]
                if n == 0:
                    continue
                prev = g

                def wrap(point, prev=prev, axis=axis, n=n):
                    def inner(v):
                        p = list(point)
                        p[axis] = v
                        return prev(tuple(p))
                    return mp.diff(inner, point[axis], n)

                g = wrap

            numeric = g(d)
            symbolic = sum(c * d[0] ** M[0] * d[1] ** M[1] * d[2] ** M[2]
                           * (d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** (p / 2)
                           for c, M, p in d_terms(cart, l_s, m_s))
            check(f"V3a: cart = {cart}, s = ({l_s}, {m_s}), d = {d}",
                  symbolic, numeric, mpf("1e-12"))

    # V3b: the Taylor-order pin: the local expansion of f(l') I_s(d + delta)
    # through the emitted M2L matches to |delta|^(QFMM_MAX_L + 1): the
    # residual at delta and 2 delta scales by 2^(QFMM_MAX_L + 1) (16 at
    # L = 3, 512 at L = 8). The stale |delta|^4 / ~16 phrasing was
    # L=3-era and failed 31/162 draws at L = 8 by up to 42x (measured
    # 2026-08-30).
    print("=== V3b: the M2L local expansion order ===")
    for l_s in range(QFMM_MAX_L + 1):
        for m_s in range(-l_s, l_s + 1):
            s = pack(l_s, m_s)
            for _ in range(2):
                d = rand_d()
                while sum(v * v for v in d) < mpf(2):
                    d = rand_d()
                direction = rand_x()
                while direction == (mpf(0), mpf(0), mpf(0)):
                    direction = rand_x()

                def residual(delta):
                    dl = tuple(d[i] + delta * direction[i] for i in range(3))
                    target = f_source(l_s) * R_value(l_s, m_s, dl) / (
                        dl[0] ** 2 + dl[1] ** 2 + dl[2] ** 2) ** (l_s + mpf("0.5"))
                    lhs = sum(eval_m2l(m2l, denom, t, s, d) * R_value(*unpack(t),
                               (delta * direction[0], delta * direction[1],
                                delta * direction[2]))
                              for t in range(MOMENT_COUNT))
                    return abs(lhs - target)

                r1 = residual(mpf("0.01"))
                r2 = residual(mpf("0.02"))
                ratio = 2 ** (QFMM_MAX_L + 1)
                check(f"V3b: s = ({l_s}, {m_s}): residual(2d)/residual(d) "
                      f"~ 2^(QFMM_MAX_L+1) = {ratio}",
                      r2, ratio * r1, mpf("0.5") * ratio * r1 + mpf("1e-20"))
                check(f"V3b: s = ({l_s}, {m_s}): residual(d) small",
                      r1, mpf(0), mpf("1e-6"))

    # V4: the literal check - a charge cloud's potential via
    # L = M2L(d) M vs the direct sum, at points outside the cloud, with the
    # truncation error bounded by (R/|d|)^(L + 1) and decreasing in L.
    print("=== V4: the charge-cloud potential ===")
    n_charges = 7
    R_cloud = mpf("0.5")
    center = rand_d()
    charges = []
    for _ in range(n_charges):
        offset = tuple(mpf(rng.uniform(-R_cloud, R_cloud)) for _ in range(3))
        q = mpf(rng.uniform(-1.0, 1.0))
        charges.append((tuple(center[i] + offset[i] for i in range(3)), q))

    moments = []
    for s in range(MOMENT_COUNT):
        l_s, m_s = unpack(s)
        moments.append(sum(q * R_value(l_s, m_s, tuple(r[i] - center[i] for i in range(3)))
                           for r, q in charges))

    # Points well outside the cloud (|d| >= 8 R_cloud).
    for _ in range(5):
        direction = rand_d()
        while direction == (mpf(0), mpf(0), mpf(0)):
            direction = rand_d()
        norm = (direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2) ** mpf("0.5")
        y = tuple(mpf(8) * direction[i] / norm + center[i] for i in range(3))
        d = tuple(y[i] - center[i] for i in range(3))
        direct = sum(q / ((y[0] - r[0]) ** 2 + (y[1] - r[1]) ** 2 + (y[2] - r[2]) ** 2) ** mpf("0.5")
                     for r, q in charges)

        for L in range(QFMM_MAX_L + 1):
            # The truncation is purely in the SOURCE order l_s <= L: at
            # delta = 0 only the l_t = 0 term survives (R_lm(0) = 0 for
            # l > 0), so the target truncation contributes nothing here.
            # The ladder bound (R/|d|)^(L+1) with a safety factor 4: the
            # per-order moment coefficients of a RANDOM cloud are not
            # monotone, so the order evidence is the ladder, not a strict
            # per-point decrease (monotonicity is the C++ acceptance
            # suite's C12H26 check).
            v_l = sum(eval_m2l(m2l, denom, 0, s, d) * moments[s]
                      for s in range(MOMENT_COUNT) if unpack(s)[0] <= L)
            err = abs(v_l - direct)
            check(f"V4: point {_}, L = {L}: |V_qfmm - V_direct|",
                  err, mpf(0), mpf("4") * (R_cloud / mpf(8)) ** (L + 1) + mpf("1e-25"))

        # The local expansion at a nearby delta (still outside the cloud):
        # the full sum_t L_t R_t(delta), truncation in both l_t and l_s.
        delta = tuple(mpf("0.25") * direction[i] / norm for i in range(3))
        y2 = tuple(y[i] + delta[i] for i in range(3))
        direct2 = sum(q / ((y2[0] - r[0]) ** 2 + (y2[1] - r[1]) ** 2 + (y2[2] - r[2]) ** 2) ** mpf("0.5")
                      for r, q in charges)
        L = QFMM_MAX_L
        local2 = sum(eval_m2l(m2l, denom, t, s, d) * R_value(*unpack(t), delta) * moments[s]
                     for t in range(MOMENT_COUNT) for s in range(MOMENT_COUNT)
                     if unpack(t)[0] <= L and unpack(s)[0] <= L)
        check(f"V4: point {_}: V(y + delta) via the full local expansion",
              local2, direct2, mpf("1e-2"))

    # V7: every emitted M2L entry against the independent angular
    # projection of the physical field. The moment-s field about the
    # source is Phi(x) = f(l_s) R_s(x - C_s) / |x - C_s|^(2 l_s + 1), and
    # its local expansion about the target obeys
    #   Phi(C_t + delta u) = sum_t' L_t' R_t'(delta u),  |u| = 1,
    # so by the exact orthogonality of the harmonics the entry is
    #   L_t = delta^{-l_t} <R_t, Phi(delta u)>_S2 / <R_t, R_t>_S2,
    # both inner products on the unit sphere by a FIXED-degree 50-point
    # Gauss-Legendre product rule (a D-point rule per axis integrates
    # algebraic degree 2D - 1 exactly, and the angular content of the
    # heaviest integrands at L = 8 is a trig polynomial of degree 16 =
    # 2 * QFMM_MAX_L in each angle, so the rule must clear that content;
    # the degree was re-tuned empirically on 2026-08-30 after the L=8
    # regen stalled on the L=3-era D = 25: the norm sweep reads the
    # highest <R_t, R_t>_S2 residual 0.315 at D = 25 (phi aliasing of
    # the degree-16 content, geometric in l), 8.7e-16 at D = 45, 5.3e-19
    # at D = 48 and 3.0e-21 at D = 50 - the committed degree, every row
    # landing ~5 orders under the 1e-15 gate; D = 60 reads the 1e-30
    # working-precision floor). Only the (delta/|d|)^k tail residual
    # remains beyond the quadrature error (negligible at delta/|d| <=
    # 0.025; the fixed rule avoids the adaptive ramp, ~40x faster). A
    # pointwise evaluation of the physical field that shares
    # no algebra with the emitted formula (the d^cart I_s recursion, the
    # A/N projection, the f rescaling and the index conventions all stand
    # or fall together), exact up to quadrature at ANY order: the
    # projection isolates the order-t coefficient, so there is no
    # truncation to bound. This replaces the V5 composition checks, which
    # were truncation-inexact by construction (see the module docstring).
    print("=== V7: the M2L entries as physical local coefficients (sphere quadrature) ===")

    def rand_unit():
        v = tuple(mpf(rng.uniform(-1.0, 1.0)) for _ in range(3))
        n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** mpf("0.5")

        if n == 0:
            return (mpf(1), mpf(0), mpf(0))

        return tuple(x / n for x in v)

    # Gauss-Legendre nodes/weights on [a, b] by Newton on the Legendre
    # polynomial (mpmath exposes no node API).
    def gl_nodes(a, b, n):
        nodes = []
        weights = []
        scale = (b - a) / 2
        shift = (a + b) / 2

        for k in range(n):
            x = cos(pi * (k + mpf("0.5")) / n)

            for _ in range(30):
                p = mp.legendre(n, x)
                p_prev = mp.legendre(n - 1, x)
                dp = n * (x * p - p_prev) / (x * x - 1)
                x -= p / dp

            p = mp.legendre(n, x)
            p_prev = mp.legendre(n - 1, x)
            dp = n * (x * p - p_prev) / (x * x - 1)
            nodes.append(shift + scale * x)
            weights.append(scale * 2 / ((1 - x * x) * dp * dp))

        return nodes, weights

    # 50 per axis (2500 nodes), re-tuned empirically 2026-08-30 for L = 8:
    # see the rule argument above; V7's cost scales with the node count.
    # D=50 is PINNED by the 1e-18 norm gate below (worst row 2.96e-21;
    # D=45 clears only a 1e-15 gate at 8.7e-16, ~1.1x margin - not the
    # pin). Any future degree change must re-pass that gate.
    gl_degree = 50
    theta_nodes, theta_weights = gl_nodes(0, pi, gl_degree)
    phi_nodes, phi_weights = gl_nodes(0, 2 * pi, gl_degree)
    # The spherical measure is sin(theta) dtheta dphi: fold the Jacobian
    # into the theta weights once (GL nodes are interior, sin > 0). Without
    # it the integral of 1 over the sphere comes out 2*pi^2, not 4*pi - the
    # V7 norm cross-check against N[l] caught exactly that on run 5.
    theta_weights = [w * mp.sin(t) for t, w in zip(theta_nodes, theta_weights)]

    # The (a, b, c, coeff) term lists of every harmonic row, and the
    # shared node data: the sphere points, their powers (degree <= 8) and
    # the product weights - built once, reused by all 6561 (t, s) pairs.
    g_terms = [[[(CARTS[l][i][0], CARTS[l][i][1], CARTS[l][i][2], G[l][row][i])
                 for i in range(len(CARTS[l]))]
                for row in range(2 * l + 1)]
               for l in range(QFMM_MAX_L + 1)]
    nodes = []
    w_theta = []
    w_phi = []
    u_pow = []

    for i in range(gl_degree):
        th = theta_nodes[i]
        st = sin(th)
        ct = cos(th)

        for j in range(gl_degree):
            ph = phi_nodes[j]
            sp = sin(ph)
            cp = cos(ph)
            nodes.append((st * cp, st * sp, ct))
            w_theta.append(theta_weights[i])
            w_phi.append(phi_weights[j])
            u_pow.append(((mpf(1), st * cp, (st * cp) ** 2, (st * cp) ** 3,
                           (st * cp) ** 4, (st * cp) ** 5, (st * cp) ** 6,
                           (st * cp) ** 7, (st * cp) ** 8),
                          (mpf(1), st * sp, (st * sp) ** 2, (st * sp) ** 3,
                           (st * sp) ** 4, (st * sp) ** 5, (st * sp) ** 6,
                           (st * sp) ** 7, (st * sp) ** 8),
                          (mpf(1), ct, ct ** 2, ct ** 3, ct ** 4, ct ** 5,
                           ct ** 6, ct ** 7, ct ** 8)))

    def r_at_node(l, row, powers):
        s = mpf(0)

        for (a, b, c, coeff) in g_terms[l][row]:
            s += coeff * powers[0][a] * powers[1][b] * powers[2][c]

        return s

    # R_t(u) at every node, shared by all (t, s) pairs.
    r_u = []

    for t in range(MOMENT_COUNT):
        l_t, m_t = unpack(t)
        r_u.append([r_at_node(l_t, m_t + l_t, u_pow[i]) for i in range(len(nodes))])

    # <R_t, R_t> on the sphere with the same rule. The norms also pin the
    # G rows' normalization: by construction <R_lm, R_lm>_S2 == N[l] (4 pi
    # for l = 0, unit W-norm for l >= 1), so the check below validates
    # both the table's normalization AND the quadrature nodes at once.
    r_norm = []

    for t in range(MOMENT_COUNT):
        l_t, m_t = unpack(t)
        r_norm.append(sum(w_theta[i] * w_phi[i] * r_u[t][i] ** 2 for i in range(len(nodes))))
        check(f"V7: <R_{l_t},{m_t}, R_{l_t},{m_t}>_S2 == N[{l_t}] "
              f"(D=50 pin: sweep worst 0.315 @ D=25, 8.7e-16 @ D=45, "
              f"5.3e-19 @ D=48, 2.96e-21 @ D=50, 1e-30 floor @ D=60)",
              r_norm[t], N[l_t],
              # The 1e-18 gate PINS the D=50 rule: D=45 clears only a
              # 1e-15 gate (8.7e-16, ~1.1x margin), D=50 lands every row
              # ~1.5 orders under 1e-18 (worst l=8 2.96e-21). Tightened
              # from the L=3-era 1e-15 - never loosened - and still
              # catches measure or normalization bugs, which fail at
              # O(1e-3..1).
              mpf("1e-18"))

    delta = mpf("0.05")

    # The two projection samples are drawn ONCE and shared by all (t, s)
    # pairs: the per-node field data (x/y/z, their degree-<=8 power
    # tuples and r2) depends only on the sample and the node index, so
    # building it inside the (t, s) loops would construct the same
    # expressions ~33M times at L = 8 (6561 pairs x 2 samples x 2500
    # nodes - 6561x redundant); hoisted here it is built once per sample
    # (2 x 2500) and the (t, s) loops only re-read it. Each inner sum
    # runs in the same node order over the same expressions, so for a
    # matched sample every projection is bit-identical to the inline
    # form. The samples are no longer 13122 independent draws: the two
    # shared draws verify every entry against the same directions, so
    # the per-entry verification power (2 random samples each) is
    # unchanged but the samples are correlated across entries (a benign
    # draw masks every entry at once, not one).
    for _ in range(2):
        u = rand_unit()
        d = tuple(mpf(rng.uniform(2.0, 4.0)) * u[i] for i in range(3))
        node_data = []

        for i in range(len(nodes)):
            x = d[0] + delta * nodes[i][0]
            y = d[1] + delta * nodes[i][1]
            z = d[2] + delta * nodes[i][2]
            px = (mpf(1), x, x * x, x * x * x, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8)
            py = (mpf(1), y, y * y, y * y * y, y ** 4, y ** 5, y ** 6, y ** 7, y ** 8)
            pz = (mpf(1), z, z * z, z * z * z, z ** 4, z ** 5, z ** 6, z ** 7, z ** 8)
            node_data.append((px, py, pz, x * x + y * y + z * z))

        for t in range(MOMENT_COUNT):
            l_t, m_t = unpack(t)

            for s in range(MOMENT_COUNT):
                l_s, m_s = unpack(s)
                factor = f_source(l_s) / (r_norm[t] * delta ** l_t)
                inner = mpf(0)

                for i in range(len(nodes)):
                    px, py, pz, r2 = node_data[i]
                    r_s = r_at_node(l_s, m_s + l_s, (px, py, pz))
                    inner += w_theta[i] * w_phi[i] * r_u[t][i] * r_s / r2 ** (l_s + mpf("0.5"))

                proj = factor * inner
                check(f"V7: t = {unpack(t)}, s = {unpack(s)}, d = {d}",
                      proj, eval_m2l(m2l, denom, t, s, d), mpf("1e-10"))

    # V6: the emitted M2M's moment shift against the direct moments about
    # the shifted center - the (-1)^|mono| sign convention end to end.
    print("=== V6: the moment shift vs the direct moments ===")
    for _ in range(3):
        s_center = rand_d()
        d = rand_d()
        for _ in range(2):
            offset = tuple(mpf(rng.uniform(-0.3, 0.3)) for _ in range(3))
            q = mpf(rng.uniform(-1.0, 1.0))
            charges2 = [(tuple(s_center[i] + offset[i] for i in range(3)), q)]
            direct_m = [sum(q * R_value(*unpack(t), tuple(r[i] - s_center[i] - d[i]
                                                          for i in range(3)))
                            for r, q in charges2)
                        for t in range(MOMENT_COUNT)]
            src_m = [sum(q * R_value(*unpack(s), tuple(r[i] - s_center[i] for i in range(3)))
                         for r, q in charges2)
                     for s in range(MOMENT_COUNT)]
            for t in range(MOMENT_COUNT):
                lhs = sum(eval_m2m(m2m, t, s, d, -1) * src_m[s] for s in range(MOMENT_COUNT))
                check(f"V6: t = {unpack(t)}", lhs, direct_m[t], mpf("1e-22"))

    return not FAILS


# ---------------------------------------------------------------------------
# The emission
# ---------------------------------------------------------------------------

def write_flat(f, name, dtype, total, items):
    """One flat constexpr std::array, wrapped at 8 entries per line.

    items: an iterable of formatted entry strings. The wrapping is
    deterministic (fixed 8-per-line) so --check byte-compares hold."""
    f.write(f"inline constexpr std::array<{dtype}, {total}> kQfmm{name} = {{{{ \n")
    line = []

    for it in items:
        line.append(it)

        if len(line) == 8:
            f.write("    " + ", ".join(line) + ",\n")
            line = []

    if line:
        f.write("    " + ", ".join(line) + ",\n")

    f.write("}};\n\n")


def write_run_table(f, name, table, run_comment):
    """One sparse (t, s) table: the flat values, the flat monomial indices
    (uint16; the monomial basis caps at 1771 entries at L = 10) and the
    per-(t, s) run offsets (uint32; offset[k] .. offset[k+1] = the run of
    (t, s) = (k / MOMENT_COUNT, k % MOMENT_COUNT)). Run entries are ordered
    by monomial index (ascending) - the deterministic order the values and
    monomial arrays share, so the arrays are index-aligned."""
    runs = []
    nz = 0

    for t in range(MOMENT_COUNT):
        for s in range(MOMENT_COUNT):
            run = [(mono_i, float(snap(table[t][s].get(mono, mpf(0)))))
                   for mono_i, mono in enumerate(MONOS)]
            run = [(m, v) for m, v in run if v != 0.0]
            runs.append(run)
            nz += len(run)

    offsets = [0]

    for run in runs:
        offsets.append(offsets[-1] + len(run))

    f.write("// The sparse " + name + " table: the dense [moments][moments][monomials]-"
            "class\n// emission at L = 8/10 would be tens of MB of source, so the "
            "coefficients\n// live in per-(t, s) runs of {monomial, value} pairs. "
            + run_comment + "\n")
    f.write(f"inline constexpr std::size_t kQfmm{name}Count = {nz};\n\n")
    f.write("// The flat values (index-aligned with kQfmm" + name + "Monomial).\n")
    write_flat(f, name + "Values", "double", nz,
               (fmt(v) for run in runs for _, v in run))
    f.write("// The monomial index of every value (the kQfmmMonomialExponents row).\n")
    write_flat(f, name + "Monomial", "std::uint16_t", nz,
               (str(m) for run in runs for m, _ in run))
    f.write("// The per-(t, s) run offsets; run (t, s) spans\n")
    f.write("// [kQfmm" + name + "RunOffset[t * kQfmmMomentCount + s],\n")
    f.write("//  kQfmm" + name + "RunOffset[... + 1]).\n")
    f.write(f"inline constexpr std::array<std::uint32_t, {MOMENT_COUNT * MOMENT_COUNT + 1}> "
            f"kQfmm{name}RunOffset = {{{{ \n")

    for k in range(0, len(offsets), 16):
        f.write("    " + ", ".join(str(o) for o in offsets[k:k + 16]) + ",\n")

    f.write("}};\n\n")


def write_header(path, m2m, m2l, denom):
    with open(path, "w", newline="\n", encoding="utf-8") as f:
        f.write("// Generated by tools/gen_qfmm_translation_tables.py - DO NOT EDIT.\n")
        f.write("// The QFMM translation-operator tables -\n")
        f.write("// generated and verified, not hand-derived.\n")
        f.write("// Moment packing: idx = l^2 + (m + l), rows m = -l..+l (MOMENT_COUNT =\n")
        f.write("// (kQfmmMaxLMult + 1)^2 moments). kQfmmMonomialExponents: the monomials\n")
        f.write("// of degree <= 2 * kQfmmMaxLMult (a slowest) - the C-table shift\n")
        f.write("// degrees (|s| - |t|) and the D-tilde numerator degrees (l_s + l_t).\n")
        f.write("// The M2M/M2L coefficients live in the SPARSE run layout: for every\n")
        f.write("// (t, s) the entries kQfmmM2MValues[k], k = the run\n")
        f.write("// kQfmmM2MRunOffset[t * kQfmmMomentCount + s] .. [...] + 1, hold the\n")
        f.write("// nonzero kQfmmM2M[t][s][kQfmmM2MMonomial[k]] coefficients. M2M is the\n")
        f.write("// TRANSPOSE of the C table: the coefficient of R_s(x) in R_t(x + d).\n")
        f.write("// The M2M reads it directly with the (-1)^|mono| of C(-d) (the moment\n")
        f.write("// shift M_t(s + d) = sum_s sum_mono kQfmmM2M[t][s][mono] (-1)^|mono|\n")
        f.write("// d^mono M_s); the L2L reads it transposed with +d (the local shift\n")
        f.write("// L_child = sum_s kQfmmM2M[s][t][mono] d^mono L_parent). kQfmmM2L\n")
        f.write("// carries the addition-theorem rescaling f(l') = (4 pi / (2 l' + 1)) /\n")
        f.write("// N_{l'} (f(0) = 1, so the monopole column is the bare Coulomb term):\n")
        f.write("// the M2L is L_t(d) = sum_s sum_mono kQfmmM2L[t][s][mono] d^mono M_s /\n")
        f.write("// |d|^kQfmmM2LDenomPower[t][s], with kQfmmM2LDenomPower[t][s] =\n")
        f.write("// 2 l_s + 1 + 2 l_t. kQfmmCartesianIndices / kQfmmSolidHarmonicG:\n")
        f.write("// the z-slowest Cartesian rows and the real-solid-harmonic\n")
        f.write("// transform coefficients to l = kQfmmMaxLMult (the moment path\n")
        f.write("// reads them up to l = lMult; the shell tables stop at kMaxShellL).\n")
        f.write("#pragma once\n#include <array>\n#include <cstddef>\n#include <cstdint>\n\n")
        f.write("namespace qcx::integrals::internal {\n\n")
        f.write(f"inline constexpr int kQfmmMaxLMult = {QFMM_MAX_L};\n\n")
        f.write(f"inline constexpr std::size_t kQfmmMomentCount = "
                f"{MOMENT_COUNT};\n\n")
        f.write(f"inline constexpr int kQfmmMonomialCount = {len(MONOS)};\n\n")
        f.write(f"inline constexpr std::array<std::array<int, 3>, {len(MONOS)}> "
                f"kQfmmMonomialExponents = {{{{ \n")
        for mono in MONOS:
            f.write(f"  {{{mono[0]}, {mono[1]}, {mono[2]}}},  // |mono| = {mono[0] + mono[1] + mono[2]}\n")
        f.write("}};\n\n")

        # The real solid-harmonic rows for the moment path's Cartesian ->
        # (l, m) contraction: the SAME pinned rows the MD engine's
        # kSolidHarmonicG comes from (gen_md_tables.solid_harmonic_table,
        # the z-slowest cartesian order, 30-digit mpmath), emitted to the
        # full QFMM cap. The shell tables stop at kMaxShellL = 6; the
        # moment path contracts moments up to l = lMult (the cap), so the
        # l > 6 rows exist only here. Rows l <= 6 are bit-identical to the
        # shell tables (pinned by QfmmLMultTest.
        # MomentSolidHarmonicRowsMatchTheShellTablesUpToShellL).
        carts = [cartesian_components(l) for l in range(QFMM_MAX_L + 1)]
        max_cart = max(len(c) for c in carts)
        f.write("// The z-slowest Cartesian (ix, iy, iz) rows per l (the same\n")
        f.write("// order the MD engine's kCartesianIndices uses, padded to the\n")
        f.write("// l = kQfmmMaxLMult row count).\n")
        f.write(f"inline constexpr std::array<std::array<std::array<int, 3>, {max_cart}>, "
                f"{QFMM_MAX_L + 1}> kQfmmCartesianIndices = {{{{ \n")
        # Full braces at every level ({{ per l block, {{...}} per row): the
        # md_tables_gen.hpp kCartesianIndices emission. Same C2078/C2131
        # rule as kQfmmSolidHarmonicG below: MSVC 19.51 rejects the
        # brace-elided THREE-level nested std::array as constexpr (the
        # two-level tables above are unaffected).
        for l in range(QFMM_MAX_L + 1):
            f.write(f"  {{{{  // l = {l}\n")
            for c in carts[l]:
                f.write(f"    {{{{ {c[0]}, {c[1]}, {c[2]} }}}},\n")
            f.write("  }},\n")
        f.write("}};\n\n")
        f.write("// The real solid-harmonic transform rows (Racah normalization,\n")
        f.write("// rows m = -l..+l, cos(m phi) for m >= 0, sin(m phi) for m < 0):\n")
        f.write("// the coefficients of R_lm in the z-slowest Cartesian monomials,\n")
        f.write("// from the pinned gen_md_tables.solid_harmonic_table rows (the\n")
        f.write("// same source as the MD engine's kSolidHarmonicG; 30-digit\n")
        f.write("// mpmath). The unused m / cart slots of the uniform shapes are\n")
        f.write("// exactly 0.0 (brace elision).\n")
        f.write(f"inline constexpr std::array<std::array<std::array<double, {max_cart}>, "
                f"{2 * QFMM_MAX_L + 1}>, {QFMM_MAX_L + 1}> kQfmmSolidHarmonicG = {{{{ \n")
        # Full braces at every level ({{ per l block, {{...}} per row): the
        # md_tables_gen.hpp kSolidHarmonicG emission. MSVC 19.51 rejects the
        # brace-elided variant of this shape with C2078/C2131 (the gen_md_tables
        # finding; the repro at the real 45 x 17 x 9 shape compiles only with
        # the full braces - verified with the pinned 19.51 toolset).
        for l in range(QFMM_MAX_L + 1):
            f.write(f"  {{{{  // l = {l}\n")
            for m in range(-l, l + 1):
                row = ", ".join(fmt(float(mp.mpf(v))) for v in G[l][m + l])
                f.write(f"    {{{{ {row} }}}},  // m = {m: d}\n")
            f.write("  }},\n")
        f.write("}};\n\n")

        # The explicit std::array type (not to_array): the nested
        # array<double, 84> members make to_array's element type ambiguous
        # to brace elision (the gen_md_tables finding, MSVC rejects it).
        write_run_table(f, "M2M", m2m,
                        "The M2M/L2L table (the TRANSPOSE of C; see above). Rows = "
                        "target (l, m), columns = source (l', m'); nonzero only for "
                        "l' <= l - the same-l block is diagonal in m, and an entry's "
                        "monomials have d-degree = l - l' (mod 2); the l' > l entries "
                        "are exact 0.0.")
        write_run_table(f, "M2L", m2l,
                        "The M2L numerators (f-rescaled D-tilde; see above), divided "
                        "by |d|^kQfmmM2LDenomPower[t][s]. Rows = target (l, m), "
                        "columns = source (l', m').")

        f.write("// The M2L denominators: 2 l_s + 1 + 2 l_t per (target, source).\n")
        f.write(f"inline constexpr std::array<std::array<int, {MOMENT_COUNT}>, "
                f"{MOMENT_COUNT}> kQfmmM2LDenomPower = {{{{ \n")
        for t in range(MOMENT_COUNT):
            l_t, m_t = unpack(t)
            f.write("  {{" + ", ".join(str(denom[t][s]) for s in range(MOMENT_COUNT))
                    + "}}," + "  // target (l, m) = ({}, {: d})\n".format(l_t, m_t))
        f.write("}};\n\n")
        f.write("}  // namespace qcx::integrals::internal\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--header", default="integrals/src/internal/qfmm_tables_gen.hpp")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed header is byte-identical to a "
                             "fresh generation (exits nonzero on drift)")
    args = parser.parse_args()

    # Resolve relative paths against the repo root (the script's parent
    # directory): the regeneration ctest runs from the build directory.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.header = args.header if os.path.isabs(args.header) else os.path.join(repo_root, args.header)

    m2m, m2l, denom = build_tables()

    if not verify(m2m, m2l, denom):
        print("\nVERIFICATION FAILED - %d check(s) failed; refusing to emit" % len(FAILS))
        for identity, rel, abs_ in FAILS:
            print("  FAIL %s: |lhs - rhs|/scale = %g (abs %g)" % (identity, rel, abs_))
        return 1

    print("ALL %d CHECKS PASSED" % PASS)

    if args.check:
        ok = True
        # A uniquely named scratch copy next to the target (mkstemp): the
        # file must live inside the repo tree so clang-format discovers the
        # repo .clang-format (format_header), and the unique name keeps
        # concurrent --check runs and hard kills from colliding.
        fd, tmp = tempfile.mkstemp(suffix=".hpp", prefix="qfmm-gen-check-",
                                   dir=os.path.dirname(args.header))
        try:
            os.close(fd)
            write_header(tmp, m2m, m2l, denom)
            format_header(tmp)
            with open(tmp, "rb") as fg, open(args.header, "rb") as fc:
                if fg.read() != fc.read():
                    ok = False
                    print(f"DRIFT: {args.header} differs from a fresh generation")
        finally:
            os.remove(tmp)
        return 0 if ok else 1

    os.makedirs(os.path.dirname(args.header), exist_ok=True)
    write_header(args.header, m2m, m2l, denom)
    format_header(args.header)
    print(f"wrote {args.header}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
