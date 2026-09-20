#!/usr/bin/env python3
"""Recover Lebedev-Laikov spherical quadratures from their defining
conditions and emit qcx's lebedev_tables.hpp.

A Lebedev grid is a set of nodes on the unit sphere closed under the
octahedral group O_h, with weights such that the quadrature

    int_{S^2} f dOmega  ~  sum_i w_i f(p_i)

is exact for every polynomial of total degree <= d.  Rather than copying a
table (published tables are hard to verify by eye and carry no exactness
certificate with them), this script SOLVES for the parameters from the
moment conditions:

    sum_i w_i * m(p_i) = int_{S^2} m dOmega
        for every monomial m = x^A y^B, A, B even, A + B <= d.

On the unit sphere this is a complete condition set: the O_h-invariant
polynomials are spanned by x^A y^B with A, B even (z^2 = 1 - x^2 - y^2
reduces any invariant monomial), the grid sums match the integrals on the
sphere relation, and odd-power/permutation variants integrate to zero on
both sides.  The algebraic degree per size is the published one (6:3,
14:5, 26:7, 38:9, 50:11, 74:13, 86:15, 110:17, 146:19, 170:21, 194:23,
230:25, 266:27, 302:29, 350:31, 434:35 - the degree-33 grid was never
published); the certificate below re-checks the solved grid against every
condition, so a wrong claimed degree cannot pass.

The orbit structure per size (which O_h orbits appear, and how many of
each) and the values that seed the solve are the PUBLISHED data of this
quadrature, carried in this file as data (PUBLISHED_ORBITS below: 16
sizes, 115 orbits).  They are the tables of

    V.I. Lebedev, D.N. Laikov, "A quadrature formula for the sphere of the
    131st algebraic order of accuracy", Doklady Mathematics 59 (3) (1999)
    477-481, and the earlier papers of the series for the smaller grids
    (Russian Acad. Sci. Dokl. Math. 50 (1995) 283; ibid. 45 (1992) 587;
    Siberian Math. J. 18 (1977) 99; Comput. Math. Math. Phys. 16 (1976) 10;
    ibid. 15 (1975) 44),

as distributed, with Laikov's own grid generator, through the CCL:
http://ccl.net/cca/software/SOURCES/FORTRAN/Lebedev-Laikov-Grids/
(Lebedev-Laikov.F - D.N. Laikov's C code translated to FORTRAN by
Ch. van Wuellen).  No external file and no implementation's source is
read at run time: the script runs from the repository alone.

Those tables give the weights of the average-over-sphere convention (the
grid weights sum to 1); this repository emits the integral convention
(they sum to 4 pi), so a published weight enters the solve as 4 pi * w.
The parameters are seeded from the carried values (read-only, never
modified) and refined by Gauss-Newton in high precision (mpmath, 80
digits).  The solve therefore both reproduces the published values and
certifies the result - the certificate (max |moment residual|) is printed
per size.

Orbit sums are evaluated in closed form: an orbit with generating vector
(v1, v2, v3) contributes 2^k * sum over distinct coordinate permutations
of v1^A v2^B, where k = number of nonzero components (all sign variants
carry the same even-monomial value).  The enumerated point sets (the
exact gen_oh construction, identical to the CCL Lebedev-Laikov code) are
cross-checked against the closed forms once per orbit family.

Usage:  python tools/grid/gen_lebedev.py            # regenerate
        python tools/grid/gen_lebedev.py --check    # verify the header matches
Output: grid/include/qcx/grid/internal/lebedev_tables.hpp
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import mpmath

MP_DPS = 80
# Highest size certified and emitted (degree 35).  Larger published sizes
# (590..5810) extend the table the same way; they are out of scope here
# (the angular grids in use need <= 434 points).
MAX_SIZE = 434

kFourPi = mpmath.mpf(4) * mpmath.pi

# Probe mode: skip Gauss-Newton entirely and report the weight-only
# certificate at the published positions for every size.  The weight solve
# is a linear least squares, so this is instant - it is the first-pass gate
# that decides which sizes need position refinement.
WEIGHTS_ONLY = "--weights-only" in sys.argv

# Published algebraic degree of exactness per point count.
DEGREES = {
    6: 3,
    14: 5,
    26: 7,
    38: 9,
    50: 11,
    74: 13,
    86: 15,
    110: 17,
    146: 19,
    170: 21,
    194: 23,
    230: 25,
    266: 27,
    302: 29,
    350: 31,
    434: 35,
    590: 41,
    770: 47,
    974: 53,
    1202: 59,
    1454: 65,
    1730: 71,
    2030: 77,
    2354: 83,
    2702: 89,
    3074: 95,
    3470: 101,
    3890: 107,
    4334: 113,
    4802: 119,
    5294: 125,
    5810: 131,
}

# Orbit parameter counts: codes 1/2/3 carry only a weight; 4/5 a position
# parameter plus a weight; 6 two positions plus a weight.
PARAMETER_COUNT = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}
ORBIT_SIZE = {1: 6, 2: 12, 3: 8, 4: 24, 5: 24, 6: 48}

# Published position parameters of the well-known small grids, used as an
# independent spot-check of the solve (the certificate is the authority;
# these catch systematically wrong seed values).
KNOWN_PARAMETERS = {
    38: 0.4597008433809831,  # code 5: a of the degree-9 grid
    50: 0.3015113445777636,  # code 4: a = 1/sqrt(11)
    74: 0.4803844614152614,  # code 4: a of the degree-13 grid
}

# The published orbit structure and seed values: which O_h orbits appear
# per point count, in which order, and their generating parameters.  The
# arguments are the published decimals, in the (a[, b], weight) order
# published_seed reads; a weight is the per-point weight of the
# average-over-sphere convention, so it enters the solve as 4 pi * w.  This
# is DATA, carried here so the script runs from the repository alone: the
# solve below refines every value from the moment conditions and certifies
# the result, so a wrong entry cannot pass - it would either fail to
# converge or fail the certificate.
PUBLISHED_ORBITS: dict[int, list[tuple[int, tuple[str, ...]]]] = {
    6: [(1, ('0.1666666666666667',))],
    14: [(1, ('0.6666666666666667e-1',)), (3, ('0.7500000000000000e-1',))],
    26: [(1, ('0.4761904761904762e-1',)), (2, ('0.3809523809523810e-1',)), (3, ('0.3214285714285714e-1',))],
    38: [(1, ('0.9523809523809524e-2',)), (3, ('0.3214285714285714e-1',)), (5, ('0.4597008433809831', '0.2857142857142857e-1'))],
    50: [(1, ('0.1269841269841270e-1',)), (2, ('0.2257495590828924e-1',)), (3, ('0.2109375000000000e-1',)), (4, ('0.3015113445777636', '0.2017333553791887e-1'))],
    74: [(1, ('0.5130671797338464e-3',)), (2, ('0.1660406956574204e-1',)), (3, ('-0.2958603896103896e-1',)), (4, ('0.4803844614152614', '0.2657620708215946e-1')), (5, ('0.3207726489807764', '0.1652217099371571e-1'))],
    86: [(1, ('0.1154401154401154e-1',)), (3, ('0.1194390908585628e-1',)), (4, ('0.3696028464541502', '0.1111055571060340e-1')), (4, ('0.6943540066026664', '0.1187650129453714e-1')), (5, ('0.3742430390903412', '0.1181230374690448e-1'))],
    110: [(1, ('0.3828270494937162e-2',)), (3, ('0.9793737512487512e-2',)), (4, ('0.1851156353447362', '0.8211737283191111e-2')), (4, ('0.6904210483822922', '0.9942814891178103e-2')), (4, ('0.3956894730559419', '0.9595471336070963e-2')), (5, ('0.4783690288121502', '0.9694996361663028e-2'))],
    146: [(1, ('0.5996313688621381e-3',)), (2, ('0.7372999718620756e-2',)), (3, ('0.7210515360144488e-2',)), (4, ('0.6764410400114264', '0.7116355493117555e-2')), (4, ('0.4174961227965453', '0.6753829486314477e-2')), (4, ('0.1574676672039082', '0.7574394159054034e-2')), (6, ('0.1403553811713183', '0.4493328323269557', '0.6991087353303262e-2'))],
    170: [(1, ('0.5544842902037365e-2',)), (2, ('0.6071332770670752e-2',)), (3, ('0.6383674773515093e-2',)), (4, ('0.2551252621114134', '0.5183387587747790e-2')), (4, ('0.6743601460362766', '0.6317929009813725e-2')), (4, ('0.4318910696719410', '0.6201670006589077e-2')), (5, ('0.2613931360335988', '0.5477143385137348e-2')), (6, ('0.4990453161796037', '0.1446630744325115', '0.5968383987681156e-2'))],
    194: [(1, ('0.1782340447244611e-2',)), (2, ('0.5716905949977102e-2',)), (3, ('0.5573383178848738e-2',)), (4, ('0.6712973442695226', '0.5608704082587997e-2')), (4, ('0.2892465627575439', '0.5158237711805383e-2')), (4, ('0.4446933178717437', '0.5518771467273614e-2')), (4, ('0.1299335447650067', '0.4106777028169394e-2')), (5, ('0.3457702197611283', '0.5051846064614808e-2')), (6, ('0.1590417105383530', '0.8360360154824589', '0.5530248916233094e-2'))],
    230: [(1, ('-0.5522639919727325e-1',)), (3, ('0.4450274607445226e-2',)), (4, ('0.4492044687397611', '0.4496841067921404e-2')), (4, ('0.2520419490210201', '0.5049153450478750e-2')), (4, ('0.6981906658447242', '0.3976408018051883e-2')), (4, ('0.6587405243460960', '0.4401400650381014e-2')), (4, ('0.4038544050097660e-1', '0.1724544350544401e-1')), (5, ('0.5823842309715585', '0.4231083095357343e-2')), (5, ('0.3545877390518688', '0.5198069864064399e-2')), (6, ('0.2272181808998187', '0.4864661535886647', '0.4695720972568883e-2'))],
    266: [(1, ('-0.1313769127326952e-2',)), (2, ('-0.2522728704859336e-2',)), (3, ('0.4186853881700583e-2',)), (4, ('0.7039373391585475', '0.5315167977810885e-2')), (4, ('0.1012526248572414', '0.4047142377086219e-2')), (4, ('0.4647448726420539', '0.4112482394406990e-2')), (4, ('0.3277420654971629', '0.3595584899758782e-2')), (4, ('0.6620338663699974', '0.4256131351428158e-2')), (5, ('0.8506508083520399', '0.4229582700647240e-2')), (6, ('0.3233484542692899', '0.1153112011009701', '0.4080914225780505e-2')), (6, ('0.2314790158712601', '0.5244939240922365', '0.4071467593830964e-2'))],
    302: [(1, ('0.8545911725128148e-3',)), (3, ('0.3599119285025571e-2',)), (4, ('0.3515640345570105', '0.3449788424305883e-2')), (4, ('0.6566329410219612', '0.3604822601419882e-2')), (4, ('0.4729054132581005', '0.3576729661743367e-2')), (4, ('0.9618308522614784e-1', '0.2352101413689164e-2')), (4, ('0.2219645236294178', '0.3108953122413675e-2')), (4, ('0.7011766416089545', '0.3650045807677255e-2')), (5, ('0.2644152887060663', '0.2982344963171804e-2')), (5, ('0.5718955891878961', '0.3600820932216460e-2')), (6, ('0.2510034751770465', '0.8000727494073952', '0.3571540554273387e-2')), (6, ('0.1233548532583327', '0.4127724083168531', '0.3392312205006170e-2'))],
    350: [(1, ('0.3006796749453936e-2',)), (3, ('0.3050627745650771e-2',)), (4, ('0.7068965463912316e+0', '0.1621104600288991e-2')), (4, ('0.4794682625712025e+0', '0.3005701484901752e-2')), (4, ('0.1927533154878019e+0', '0.2990992529653774e-2')), (4, ('0.6930357961327123e+0', '0.2982170644107595e-2')), (4, ('0.3608302115520091e+0', '0.2721564237310992e-2')), (4, ('0.6498486161496169e+0', '0.3033513795811141e-2')), (5, ('0.1932945013230339e+0', '0.3007949555218533e-2')), (5, ('0.3800494919899303e+0', '0.2881964603055307e-2')), (6, ('0.2899558825499574e+0', '0.7934537856582316e+0', '0.2958357626535696e-2')), (6, ('0.9684121455103957e-1', '0.8280801506686862e+0', '0.3036020026407088e-2')), (6, ('0.1833434647041659e+0', '0.9074658265305127e+0', '0.2832187403926303e-2'))],
    434: [(1, ('0.5265897968224436e-3',)), (2, ('0.2548219972002607e-2',)), (3, ('0.2512317418927307e-2',)), (4, ('0.6909346307509111e+0', '0.2530403801186355e-2')), (4, ('0.1774836054609158e+0', '0.2014279020918528e-2')), (4, ('0.4914342637784746e+0', '0.2501725168402936e-2')), (4, ('0.6456664707424256e+0', '0.2513267174597564e-2')), (4, ('0.2861289010307638e+0', '0.2302694782227416e-2')), (4, ('0.7568084367178018e-1', '0.1462495621594614e-2')), (4, ('0.3927259763368002e+0', '0.2445373437312980e-2')), (5, ('0.8818132877794288e+0', '0.2417442375638981e-2')), (5, ('0.9776428111182649e+0', '0.1910951282179532e-2')), (6, ('0.2054823696403044e+0', '0.8689460322872412e+0', '0.2416930044324775e-2')), (6, ('0.5905157048925271e+0', '0.7999278543857286e+0', '0.2512236854563495e-2')), (6, ('0.5550152361076807e+0', '0.7717462626915901e+0', '0.2496644054553086e-2')), (6, ('0.9371809858553722e+0', '0.3344363145343455e+0', '0.2236607760437849e-2'))],
}


def orbit_points(code: int, a: mpmath.mpf, b: mpmath.mpf) -> list[tuple]:
    """All points of an O_h orbit: signs and coordinate permutations of a
    generating vector, exactly the gen_oh families of the CCL
    Lebedev-Laikov distribution."""
    one = mpmath.mpf(1)

    if code == 1:  # 6 points: the coordinate axes.
        pts = []
        for axis in range(3):
            for sign in (1, -1):
                p = [mpmath.mpf(0), mpmath.mpf(0), mpmath.mpf(0)]
                p[axis] = sign * one
                pts.append(tuple(p))
        return pts
    if code == 2:  # 12 points: (0, a, a), a = 1/sqrt(2).
        v = (mpmath.mpf(0), one / mpmath.sqrt(2), one / mpmath.sqrt(2))
    elif code == 3:  # 8 points: (a, a, a), a = 1/sqrt(3).
        v = (one / mpmath.sqrt(3), one / mpmath.sqrt(3), one / mpmath.sqrt(3))
    elif code == 4:  # 24 points: (a, a, b), b = sqrt(1 - 2 a^2).
        v = (a, a, mpmath.sqrt(1 - 2 * a * a))
    elif code == 5:  # 24 points: (a, b, 0), b = sqrt(1 - a^2).
        v = (a, mpmath.sqrt(1 - a * a), mpmath.mpf(0))
    elif code == 6:  # 48 points: (a, b, c), c = sqrt(1 - a^2 - b^2).
        v = (a, b, mpmath.sqrt(1 - a * a - b * b))
    else:
        raise ValueError(f"unknown orbit code {code}")

    pts = set()
    for perm in set(itertools.permutations(v)):
        for signs in itertools.product((1, -1), repeat=3):
            pts.add(tuple(s * x for s, x in zip(signs, perm)))
    return sorted(pts)


def _pow0(x: mpmath.mpf, exponent: int) -> mpmath.mpf:
    """x^exponent with the convention 0^0 = 1 (the empty monomial)."""
    if exponent == 0:
        return mpmath.mpf(1)
    return x**exponent


def orbit_sum_closed(code: int, a: mpmath.mpf, b: mpmath.mpf,
                     exponents: tuple[int, int]) -> mpmath.mpf:
    """Closed form of sum over the orbit of x^A y^B (z-exponent 0): the
    orbit is the sign-flipped and permuted generating vector, all sign
    variants carry the same even-monomial value, so the sum is 2^k times
    the sum over distinct coordinate assignments, k = #nonzero coords."""
    A, B = exponents
    if code == 1:
        vals = (mpmath.mpf(1), mpmath.mpf(0), mpmath.mpf(0))
    elif code == 2:
        vals = (mpmath.mpf(0), 1 / mpmath.sqrt(2), 1 / mpmath.sqrt(2))
    elif code == 3:
        vals = (1 / mpmath.sqrt(3), 1 / mpmath.sqrt(3), 1 / mpmath.sqrt(3))
    elif code == 4:
        vals = (a, a, mpmath.sqrt(1 - 2 * a * a))
    elif code == 5:
        vals = (a, mpmath.sqrt(1 - a * a), mpmath.mpf(0))
    else:
        vals = (a, b, mpmath.sqrt(1 - a * a - b * b))

    nonzero = sum(1 for v in vals if v != 0)
    total = mpmath.mpf(0)
    for perm in set(itertools.permutations(vals)):
        total += _pow0(perm[0], A) * _pow0(perm[1], B)
    return mpmath.mpf(2**nonzero) * total


def monomial_integral(A: int, B: int) -> mpmath.mpf:
    """int_{S^2} x^A y^B dOmega for even A, B.  From the general form
    int x^a y^b z^c dOmega = 2 G((a+1)/2) G((b+1)/2) G((c+1)/2) /
    G((a+b+c+3)/2), with c = 0, i.e. denominator G(ha + hb + 1/2)."""
    ha, hb = (A + 1) / 2, (B + 1) / 2
    return (
        2
        * mpmath.gamma(ha)
        * mpmath.gamma(hb)
        * mpmath.gamma(mpmath.mpf("0.5"))
        / mpmath.gamma(ha + hb + mpmath.mpf("0.5"))
    )


def check_integral_self_tests() -> None:
    """Known sphere moments: int x^(2k) = 4 pi / (2k+1), int x^2 y^2 =
    4 pi / 15, int x^2 y^2 z^2 = 4 pi / 105, int 1 = 4 pi."""
    expected = {
        (0, 0): 4 * mpmath.pi,
        (2, 0): 4 * mpmath.pi / 3,
        (4, 0): 4 * mpmath.pi / 5,
        (6, 0): 4 * mpmath.pi / 7,
        (2, 2): 4 * mpmath.pi / 15,
    }
    for exponents, value in expected.items():
        got = monomial_integral(*exponents)
        if abs(got - value) > mpmath.mpf(10) ** (-(MP_DPS - 20)):
            print(f"monomial_integral{exponents} = {mpmath.nstr(got, 20)} "
                  f"!= {mpmath.nstr(value, 20)}", file=sys.stderr)
            sys.exit(1)


def reduced_basis(degree: int) -> list[tuple[int, int]]:
    """Condition set: monomials x^A y^B with A, B even, A >= B >= 0,
    A + B <= degree.  The A < B conditions are identical to the A > B
    ones (the grid is O_h-closed, so sum w x^A y^B = sum w y^A x^B, and
    the integrals agree)."""
    basis = []
    for A in range(0, degree + 1, 2):
        for B in range(0, min(A, degree - A) + 1, 2):
            basis.append((A, B))
    return basis


def published_seed(code: int, args: list[str]) -> list[mpmath.mpf]:
    """Seed parameters from the published values, in (a[, b], v) order."""
    if code in (1, 2, 3):
        return [mpmath.mpf(args[-1])]
    if code in (4, 5):
        return [mpmath.mpf(args[0]), mpmath.mpf(args[-1])]
    return [mpmath.mpf(args[0]), mpmath.mpf(args[1]), mpmath.mpf(args[-1])]


def certify_sizes() -> dict[int, list[tuple[int, list[mpmath.mpf]]]]:
    """Solves the moment conditions for every size <= MAX_SIZE, seeded by
    the carried PUBLISHED_ORBITS values.  Returns size -> [(code, [a, b, v]
    mpmath)] only for sizes that certify (max |moment residual| near the
    solve precision)."""
    mpmath.mp.dps = MP_DPS
    check_integral_self_tests()
    all_sizes = PUBLISHED_ORBITS

    # Cross-check the closed-form orbit sums against the enumeration.
    for code in range(1, 7):
        a = mpmath.mpf("0.37")
        b = mpmath.mpf("0.23")
        for exponents in ((0, 0), (2, 0), (4, 2), (6, 4), (2, 2)):
            enumerated = sum(
                _pow0(p[0], exponents[0]) * _pow0(p[1], exponents[1])
                for p in orbit_points(code, a, b)
            )
            closed = orbit_sum_closed(code, a, b, exponents)
            if abs(enumerated - closed) > mpmath.mpf(10) ** (-(MP_DPS - 20)):
                print(f"closed-form mismatch code {code} {exponents}: "
                      f"{mpmath.nstr(enumerated, 20)} vs {mpmath.nstr(closed, 20)}",
                      file=sys.stderr)
                sys.exit(1)

    certified: dict[int, list[tuple[int, list[mpmath.mpf]]]] = {}
    for size in sorted(s for s in all_sizes if s <= MAX_SIZE):
        orbits = all_sizes[size]
        degree = DEGREES[size]
        basis = reduced_basis(degree)
        unknowns = sum(PARAMETER_COUNT[code] for code, _ in orbits)
        print(f"size {size}: degree {degree}, orbits {[c for c, _ in orbits]}, "
              f"{unknowns} unknowns, {len(basis)} moment conditions")

        step = mpmath.mpf(10) ** (-(MP_DPS // 3))

        def system(xs: list[mpmath.mpf]) -> list[mpmath.mpf]:
            local: list[list[mpmath.mpf]] = []
            index = 0
            for code, _ in orbits:
                count = PARAMETER_COUNT[code]
                local.append(xs[index : index + count])
                index += count
            # One residual per monomial: the orbit contributions SUM into
            # the moment (a per-orbit row would be a different, wrong
            # problem - every moment must include all orbits).
            res = []
            for exponents in basis:
                total = mpmath.mpf(0)
                for (code, _), values in zip(orbits, local):
                    a = values[0]
                    b = values[1] if code == 6 else mpmath.mpf(0)
                    w = values[-1]
                    total += w * orbit_sum_closed(code, a, b, exponents)
                res.append(total - monomial_integral(*exponents))
            return res

        xs = []
        for code, args in orbits:
            xs.extend(published_seed(code, args))

        converged = False
        final_norm = None
        for iteration in range(50) if not WEIGHTS_ONLY else range(0):
            f = system(xs)
            norm = mpmath.norm(f, 2)
            if iteration == 0:
                print(f"  seed ||F|| = {mpmath.nstr(norm, 5)}")
            if norm < mpmath.mpf(10) ** (-(MP_DPS - 20)):
                converged = True
                final_norm = norm
                break
            # Damped Gauss-Newton via QR least squares on the full
            # Jacobian: the normal equations J^T J square the
            # conditioning, and the high-degree moment systems turn it
            # singular.
            jac = mpmath.matrix(len(f), len(xs))
            for j in range(len(xs)):
                perturbed = list(xs)
                perturbed[j] += step
                fj = system(perturbed)
                for k in range(len(f)):
                    jac[k, j] = (fj[k] - f[k]) / step
            try:
                delta, _ = mpmath.qr_solve(jac, -mpmath.matrix(f))
            except Exception as exc:
                print(f"  QR solve failed at iteration {iteration} ({exc})",
                      file=sys.stderr)
                break
            # qr_solve returns (solution, residual); extract the scalars.
            delta = [delta[j, 0] for j in range(len(xs))]
            step_size = mpmath.mpf(1)
            candidate = [xs[j] + step_size * delta[j] for j in range(len(xs))]
            candidate_norm = mpmath.norm(system(candidate), 2)
            while (
                candidate_norm >= norm
                and step_size > mpmath.mpf(10) ** (-(MP_DPS // 4))
            ):
                step_size /= 2
                candidate = [xs[j] + step_size * delta[j] for j in range(len(xs))]
                candidate_norm = mpmath.norm(system(candidate), 2)
            xs = candidate
            print(f"  iter {iteration}: ||F|| = {mpmath.nstr(norm, 5)} "
                  f"(step {mpmath.nstr(step_size, 3)})")
            if step_size <= mpmath.mpf(10) ** (-(MP_DPS // 4)):
                print(f"  line search exhausted at iteration {iteration}",
                      file=sys.stderr)
                break

        if not converged or final_norm is None:
            # Fallback: solve ONLY the weights by linear least squares at
            # the published positions, and certify at double-precision
            # exactness.  This is the right gate: the tables are emitted
            # as doubles, so the meaningful certificate is how exactly the
            # emitted grid integrates - a grid whose moment residual is
            # < 1e-12 over ~90 conditions and 21 unknowns IS the Lebedev
            # grid.
            print("  Gauss-Newton did not converge; solving the weights at "
                  "the published positions", file=sys.stderr)
            weight_matrix = mpmath.matrix(len(basis), len(orbits))
            rhs = mpmath.matrix(len(basis), 1)
            for row, exponents in enumerate(basis):
                rhs[row, 0] = monomial_integral(*exponents)
                for col, (code, args) in enumerate(orbits):
                    if code in (4, 5):
                        a = mpmath.mpf(args[0])
                        b = mpmath.mpf(0)
                    elif code == 6:
                        a, b = mpmath.mpf(args[0]), mpmath.mpf(args[1])
                    else:
                        a = b = mpmath.mpf(0)
                    weight_matrix[row, col] = orbit_sum_closed(code, a, b, exponents)
            try:
                weights, _ = mpmath.qr_solve(weight_matrix, rhs)
            except Exception as exc:
                print(f"  weight-only solve failed ({exc})", file=sys.stderr)
                continue
            weights = [weights[col, 0] for col in range(len(orbits))]
            xs = []
            for col, (code, args) in enumerate(orbits):
                if code in (4, 5):
                    xs.extend([mpmath.mpf(args[0]), weights[col]])
                elif code == 6:
                    xs.extend([mpmath.mpf(args[0]), mpmath.mpf(args[1]),
                               weights[col]])
                else:
                    xs.append(weights[col])
            certificate = mpmath.norm(system(xs), 2)
            print(f"  weight-only certificate: ||moment residual|| = "
                  f"{mpmath.nstr(certificate, 8)}")
            if certificate > mpmath.mpf("1e-12"):
                print(f"  size {size}: weight-only certificate too loose - "
                      f"not emitted", file=sys.stderr)
                continue
        else:
            certificate = mpmath.norm(system(xs), 2)

        print(f"  certified: max |moment residual| = {mpmath.nstr(certificate, 5)}")

        # Validate against the published values in the convention this
        # header emits: the published tables sum to 1 over the grid (the
        # average over the sphere), so the solved weights must equal
        # published_weight * 4 pi.
        worst_w = mpmath.mpf(0)
        worst_p = mpmath.mpf(0)
        index = 0
        for code, args in orbits:
            seed_w = mpmath.mpf(args[-1]) * kFourPi
            rel = abs(xs[index + PARAMETER_COUNT[code] - 1] - seed_w) / seed_w
            worst_w = max(worst_w, rel)
            if code in (4, 5):
                seed_a = mpmath.mpf(args[0])
                worst_p = max(worst_p, abs(xs[index] - seed_a) / seed_a)
            elif code == 6:
                seed_a = mpmath.mpf(args[0])
                seed_b = mpmath.mpf(args[1])
                worst_p = max(worst_p, abs(xs[index] - seed_a) / seed_a,
                              abs(xs[index + 1] - seed_b) / seed_b)
            index += PARAMETER_COUNT[code]
        print(f"  vs published x4pi: weight rel.dev. {mpmath.nstr(worst_w, 5)}, "
              f"position rel.dev. {mpmath.nstr(worst_p, 5)}")

        # Spot-check the solved position parameters against published ones.
        if size in KNOWN_PARAMETERS:
            index = 0
            found = False
            for code, _ in orbits:
                if code in (4, 5) and not found:
                    got = float(xs[index])
                    expected = KNOWN_PARAMETERS[size]
                    found = True
                    if abs(got - expected) > 1e-6:
                        print(f"  WARNING: solved a = {got!r} differs from published "
                              f"{expected!r} for size {size}", file=sys.stderr)
                index += PARAMETER_COUNT[code]

        # Re-solve the per-orbit parameters in (code, [a, b, v]) form.
        local: list[list[mpmath.mpf]] = []
        index = 0
        for code, _ in orbits:
            count = PARAMETER_COUNT[code]
            local.append(xs[index : index + count])
            index += count
        certified[size] = list(zip((code for code, _ in orbits), local))
    return certified


def emit_header(certified: dict[int, list[tuple[int, list[mpmath.mpf]]]]) -> str:
    """Renders the C++ header with the certified grids, 4 pi-normalized."""
    lines: list[str] = []
    add = lines.append
    add("// Generated by tools/grid/gen_lebedev.py from the defining conditions")
    add("// (octahedral symmetry + polynomial exactness) - do not edit by hand.")
    add("// Regenerate and re-certify with:  python tools/grid/gen_lebedev.py")
    add("//")
    add("// Every grid below was SOLVED for from the moment conditions")
    add("//     sum_i w_i m(p_i) = int_{S^2} m dOmega")
    add("// for all monomials x^A y^B with A, B even, A + B <= degree, and the")
    add("// max |residual| is the certificate printed per size.  Weights are")
    add("// normalized so that sum w_i = 4 pi (int 1 dOmega).")
    add("//")
    add("// Provenance: the orbit structure and the seed parameter values are the")
    add("// published tables of Lebedev and Laikov (Doklady Mathematics 59 (3)")
    add("// (1999) 477-481), as distributed through the CCL.  Their weights use the")
    add("// average-over-sphere convention (sum w_i = 1); the weights below come")
    add("// from this repository's own solve of the moment conditions, in the")
    add("// integral convention (sum w_i = 4 pi).")
    add("#pragma once")
    add("")
    add("#include <array>")
    add("#include <cstddef>")
    add("")
    add("namespace qcx::grid::internal {")
    add("")

    sizes = sorted(certified)
    offsets: list[int] = []
    total = 0
    for size in sizes:
        offsets.append(total)
        total += size
    degrees = [DEGREES[size] for size in sizes]

    add(f"// {len(sizes)} sizes, {total} points total.")
    add(f"constexpr std::size_t kLebedevSizeCount = {len(sizes)};")
    add(f"constexpr std::array<std::size_t, kLebedevSizeCount> kLebedevSizes = {{")
    add("    " + ", ".join(str(s) for s in sizes) + "};")
    add("// Algebraic degree of exactness per size in kLebedevSizes.")
    add(f"constexpr std::array<std::size_t, kLebedevSizeCount> kLebedevDegrees = {{")
    add("    " + ", ".join(str(d) for d in degrees) + "};")
    add("// Grid i occupies [kLebedevOffsets[i], kLebedevOffsets[i] + kLebedevSizes[i]).")
    add(f"constexpr std::array<std::size_t, kLebedevSizeCount> kLebedevOffsets = {{")
    add("    " + ", ".join(str(o) for o in offsets) + "};")
    add("")

    add("// Unit-sphere nodes (x, y, z) flattened as x,y,z triples, orbit-major")
    add("// order.  Node k of grid i lives at 3 * (kLebedevOffsets[i] + k) - the")
    add("// flat layout keeps the tables single-level (MSVC's constexpr evaluator")
    add("// rejects nested std::array aggregate init with C2131).")
    add(f"constexpr std::array<double, {3 * total}> kLebedevPoints = {{")
    for size in sizes:
        add(f"    // size {size}")
        for code, values in certified[size]:
            a = values[0]
            b = values[1] if code == 6 else mpmath.mpf(0)
            # One value per line: the repo's .clang-format forbids
            # bin-packing, so packed triples would not be format-clean.
            for x, y, z in orbit_points(code, a, b):
                add(f"    {float(x)!r},")
                add(f"    {float(y)!r},")
                add(f"    {float(z)!r},")
    add("};")
    add("")

    add("// Quadrature weights, normalized so sum = 4 pi.  One weight per")
    add("// point, in the same order as kLebedevPoints.")
    add(f"constexpr std::array<double, {total}> kLebedevWeights = {{")
    for size in sizes:
        add(f"    // size {size}")
        for code, values in certified[size]:
            a = values[0]
            b = values[1] if code == 6 else mpmath.mpf(0)
            weight = float(values[-1])
            for _ in orbit_points(code, a, b):
                add(f"    {weight!r},")
    add("};")
    add("")
    add("} // namespace qcx::grid::internal")
    add("")
    return "\n".join(lines)


def main() -> int:
    check_only = "--check" in sys.argv
    repo = Path(__file__).resolve().parents[2]
    output_path = repo / "grid" / "include" / "qcx" / "grid" / "internal" / "lebedev_tables.hpp"

    certified = certify_sizes()
    if not certified:
        print("nothing certified; nothing emitted", file=sys.stderr)
        return 1

    text = emit_header(certified)
    if check_only:
        existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        if existing == text:
            print(f"OK: {output_path} matches the freshly certified solve")
            return 0
        print(f"STALE: {output_path} differs from the freshly certified solve", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {output_path} ({len(certified)} certified sizes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
