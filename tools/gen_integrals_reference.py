#!/usr/bin/env python3
"""Generate the committed H2/STO-3G integral reference grid.

Computes the closed-form s-s one-electron integrals (S, T, V) and the
s-s-s-s two-electron repulsion tensor ([Helgaker2000] - the citation keys
used in this tree are the CITATION.bib entries of the same name) for H2 in
STO-3G at R = 1.4 bohr with mpmath at 30 digits and writes
integrals/tests/data/h2_sto3g_reference.csv, consumed by
one_electron_test.cpp and two_electron_test.cpp at a 1e-10 tolerance.
Requires: `pip install mpmath`.

The primitive data are the published STO-3G H values (the same literal
strings as the NWChem fixture in tests/fixtures/h2_sto3g.hpp), so the CSV
depends only on those constants and the integral formulas - not on the C++
implementation under test.
"""
import argparse
import os
import sys
import tempfile

from mpmath import mp, mpf, erf, exp, pi, sqrt

mp.dps = 30  # reference error ~1e-25; 4 orders below the double floor 1e-14

EXPONENTS = [mpf("3.4252509140"), mpf("0.6239137298"), mpf("0.1688554040")]
COEFFICIENTS = [mpf("0.1543289673"), mpf("0.5353281423"), mpf("0.4446345422")]

R = mpf("1.4")
ATOMS = [(mpf(0), mpf(0), mpf(0)), (R, mpf(0), mpf(0))]  # (x, y, z), Bohr
CHARGES = [1, 1]
FUNCTION_CENTERS = [0, 1]  # two s functions: function 0 on atom 0, function 1 on atom 1


def f0(x):
    """F_0(x) = 1/2 sqrt(pi/x) erf(sqrt(x)), exact limit 1 at x = 0."""
    if x == 0:
        return mpf(1)
    return sqrt(pi) * erf(sqrt(x)) / (2 * sqrt(x))


def primitive_normalization(a):
    """(2a/pi)^(3/4), the normalization each BSE coefficient carries."""
    return (2 * a / pi) ** mpf("0.75")


def contraction_renormalize(exponents, coefficients):
    """Scales one contraction to unit norm in the engine convention - the
    same closed form as the parser (basis_set.cpp NormalizeContractions) and
    gen_md_reference.py ShellData._renormalize: the l = 0 primitive-pair
    overlap S(a, b) = (4ab)^(3/4) / (a + b)^(3/2) (the general form carries
    the 2^l (ab)^(l/2) factor). The raw BSE coefficients carry only the
    per-primitive (2a/pi)^(3/4) norm, so the raw contraction self-overlap is
    1 + 7e-11; the engine renormalizes it to exactly 1 at parse, and the
    grid must do the same or it drifts from the engine by ~1e-10 (the
    OneElectronTest V(0,0) finding)."""
    norm_squared = sum(
        coefficients[k] * coefficients[l] * (4 * exponents[k] * exponents[l]) ** mpf("0.75")
        / (exponents[k] + exponents[l]) ** mpf("1.5")
        for k in range(len(exponents)) for l in range(len(exponents)))
    scale = 1 / sqrt(norm_squared)
    return [c * scale for c in coefficients]


def distance2(u, v):
    return (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2 + (u[2] - v[2]) ** 2


def product_center(a, u, b, v):
    """The Gaussian-product center (a*u + b*v) / (a + b)."""
    return tuple((a * ui + b * vi) / (a + b) for ui, vi in zip(u, v))


def build():
    """Computes the reference rows as (kind, i, j, k, l, value) lists."""
    rows = []
    centers = [ATOMS[i] for i in FUNCTION_CENTERS]
    n = len(centers)
    renormalized = contraction_renormalize(EXPONENTS, COEFFICIENTS)
    normalized = [[c * primitive_normalization(a) for a, c in zip(EXPONENTS, renormalized)]]

    for _ in centers[1:]:
        normalized.append(normalized[0])

    for i in range(n):
        for j in range(n):
            s = mpf(0)
            t = mpf(0)
            v = mpf(0)
            for a, da in zip(EXPONENTS, normalized[i]):
                for b, db in zip(EXPONENTS, normalized[j]):
                    p = a + b
                    rab2 = distance2(centers[i], centers[j])
                    ab_over_p = a * b / p
                    overlap = (pi / p) ** mpf("1.5") * exp(-ab_over_p * rab2)
                    s += da * db * overlap
                    t += da * db * ab_over_p * (3 - 2 * ab_over_p * rab2) * overlap
                    center_p = product_center(a, centers[i], b, centers[j])
                    v -= da * db * (2 * pi / p) * exp(-ab_over_p * rab2) * sum(
                        charge * f0(p * distance2(center_p, c))
                        for charge, c in zip(CHARGES, ATOMS))
            rows.append(["S", i, j, "", "", s])
            rows.append(["T", i, j, "", "", t])
            rows.append(["V", i, j, "", "", v])

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    eri = mpf(0)
                    for a, da in zip(EXPONENTS, normalized[i]):
                        for b, db in zip(EXPONENTS, normalized[j]):
                            p = a + b
                            ab_over_p = a * b / p
                            rab2 = distance2(centers[i], centers[j])
                            cp = product_center(a, centers[i], b, centers[j])
                            for c, dc in zip(EXPONENTS, normalized[k]):
                                for d, dd in zip(EXPONENTS, normalized[l]):
                                    q = c + d
                                    alpha = p * q / (p + q)
                                    cd_over_q = c * d / q
                                    rcd2 = distance2(centers[k], centers[l])
                                    cq = product_center(c, centers[k], d, centers[l])
                                    rpq2 = distance2(cp, cq)
                                    eri += (da * db * dc * dd * (2 * pi ** mpf("2.5")) /
                                            (p * q * sqrt(p + q))) * \
                                        exp(-ab_over_p * rab2 - cd_over_q * rcd2) * \
                                        f0(alpha * rpq2)
                    rows.append(["ERI", i, j, k, l, eri])
    return rows


def write_reference(path):
    import csv
    rows = build()
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["kind", "i", "j", "k", "l", "value"])
        for kind, i, j, k, l, value in rows:
            w.writerow([kind, i, j, k, l, mp.nstr(value, 22)])
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="integrals/tests/data/h2_sto3g_reference.csv")
    parser.add_argument("--check", action="store_true",
                        help="verify the committed CSV is byte-identical to a fresh "
                             "generation (exits nonzero on drift)")
    args = parser.parse_args()

    # Resolve relative paths against the repo root (the script's parent
    # directory): the regeneration ctest runs from the build directory
    # (mirrors gen_md_tables.py / gen_md_reference.py).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isabs(args.reference):
        args.reference = os.path.join(repo_root, args.reference)

    if not args.check:
        os.makedirs(os.path.dirname(args.reference), exist_ok=True)
        write_reference(args.reference)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        fresh = os.path.join(tmp, "fresh.csv")
        write_reference(fresh)
        with open(fresh, "rb") as fresh_file, open(args.reference, "rb") as committed_file:
            if fresh_file.read() != committed_file.read():
                print(f"DRIFT: {args.reference} differs from a fresh generation")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
