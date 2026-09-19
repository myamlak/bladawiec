#!/usr/bin/env python3
"""Generate symmetry/include/qcx/symmetry/full_group_tables.hpp.

Full-group (non-Abelian) character tables + full-group -> Abelian correlation
data, constexpr, for every PointGroupName except C-inf-v / D-inf-h (no finite
character table exists for the linear groups; the runtime stage labels those
via |lambda| projection over the recorded finite subset instead).

Method (all conventions mirror the runtime point-group detection machinery so
the runtime stage can recompute the same class keys from the molecule-frame
elements):
  - Closure: BFS over hand-specified generators (standard orientation, the
    principal axis along z), dedup at 1e-8, expected group order asserted.
  - Classification of every element: kind (determinant + trace), order
    (minimal power g^k = E), axis (the +1 eigenvector for rotations and
    improper rotations, the -1 eigenvector for mirrors; sign-canonicalized),
    power j = round(theta*order/2*pi) with theta the minimal angle from the
    trace (arccos), so a class is described by its minimal angle: C5 -> j=1,
    C5^2 -> j=2, S10 -> j=1, S10^3 -> j=3. This is the (kind, order, j, axis)
    class key the runtime recomputes. The emitted key is canonical: j is the
    class-minimal member power (min j over the conjugacy class - a class
    mixes conjugate powers, and the representative's own power is
    BFS-discovery dependent), and the class axis is the normalized
    member-axis minimum, so the principal/on-axis slot rules are
    representative-independent.
  - Conjugacy classes; per-class descriptor (kind, order, j, axis slot) with
    the slot rule: the principal axis = the class of the maximal-order proper
    rotation (ties: lexicographically smallest canonicalized axis); sigma and
    improper classes on the principal axis are kPrincipal; all other classes
    are kOrbitN ranked by lexicographically smallest member axis among
    same-(kind, order, j) classes.
  - Characters via the class-sum algebra: S_i = class sum, structure
    constants from S_i S_j = sum_k c_ijk S_k, generic central combination
    M = sum s_i S_i (seeded RNG, the s_i are random but fixed), the eigenspaces
    of M are the isotypic components, mu_{Gamma,i} = v^T S_i v (v in the
    component, |v| = 1; S_i is central, so it acts as the scalar
    |C_i| chi_i / dim there), dim^2 = |G| / sum_i mu_i^2/|C_i| (the
    complex-type rows carry the 2|G| orthogonality, so the naive square root
    is dim/sqrt(2) there and the integer-rounding check corrects it),
    chi_i = mu_i * dim / |C_i|, rounded to 12 dp, then verified
    (orthogonality with the complex-type 2|G| diagonal, sum dim^2 = |G| +
    (1/2) sum of complex-type dim^2, label uniqueness).
  - Labels: per-group specs (dimension + distinguishing character values)
    written in the standard Mulliken convention (the B1 = x^2-y^2-like and
    E1/E2 assignment conventions of the classic tables). The generator
    verifies every spec matches exactly one computed irrep and vice versa;
    the psi4 label cross-check is the empirical arbiter.
  - Correlation: subduction to the realizable Abelian subgroup H
    (LargestAbelianSubgroup, with the documented exception Td -> D2: D2h is
    not a subgroup of Td), realized canonically as a subgroup of G; H's
    characters are computed with the same class-sum machinery and matched to
    the existing 8 abelian tables' row order (character_tables.hpp), so the
    correlation columns index the runtime Abelian block order directly;
    n_alpha = (1/|H|) sum_h chi_Gamma(class_G(h)) * chi_alpha(h).

Usage:
  python tools/gen_full_group_tables.py            # regenerate the header
  python tools/gen_full_group_tables.py --check    # verify the committed header
                                                    # matches a fresh generation
"""
import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# 3x3 geometry
# ---------------------------------------------------------------------------

I3 = np.eye(3)
INV = -I3
SIGMA_H = np.diag([1.0, 1.0, -1.0])          # xy plane (normal z)
SIGMA_XZ = np.diag([1.0, -1.0, 1.0])         # xz plane (normal y)
C2_X = np.diag([1.0, -1.0, -1.0])            # C2 about x
PHI = (1.0 + np.sqrt(5.0)) / 2.0


def rot(axis, theta):
    """Rotation about a (not necessarily normalized) axis by theta."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


def mirror(normal):
    """Reflection in the plane with the given normal."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return I3 - 2.0 * np.outer(n, n)


def rot_z(theta):
    return rot((0.0, 0.0, 1.0), theta)


def s2n_z(n):
    """Improper rotation S_2n about z: R(z, 2pi/2n) . sigma_h."""
    return rot_z(np.pi / n) @ SIGMA_H


# ---------------------------------------------------------------------------
# Group construction and classification
# ---------------------------------------------------------------------------

def closure(generators, expected, name):
    """BFS closure over the generators, dedup at 1e-8, order asserted."""
    elems = [I3]
    changed = True
    while changed:
        changed = False
        for a in list(elems):
            for g in generators:
                p = a @ g
                if not any(np.max(np.abs(p - e)) < 1e-8 for e in elems):
                    elems.append(p)
                    changed = True
    if len(elems) != expected:
        raise SystemExit(f"{name}: closure order {len(elems)} != expected {expected}")
    return elems


def axis_of(g, target, tol=1e-6):
    """The canonicalized eigenvector for the target eigenvalue.

    Axial-vector method (works for non-symmetric rotation matrices, where
    eigh is invalid): the skew part of a proper rotation R(n, theta) has
    axial vector 2*sin(theta)*n. Mirrors pass -g (a C2 about the normal).
    The eigh fallback covers theta = pi (C2, where the skew part vanishes)
    and mirrors, which are symmetric so eigh is exact there.
    """
    m = g if target > 0.0 else -g
    axial = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    if np.linalg.norm(axial) > 1e-6:
        a = axial
    else:
        evals, evecs = np.linalg.eigh(g)
        i = int(np.argmin(np.abs(evals - target)))
        a = evecs[:, i].real
    nz = np.nonzero(np.abs(a) > 1e-8)[0]
    if a[nz[0]] < 0.0:
        a = -a
    return a


def canonical_axis(a):
    # Snap sub-tolerance components to exact zero first: a tiny residue in a
    # nominally-zero slot (6e-17 in the x slot of a y axis) would otherwise
    # survive and poison the lexicographic axis_key ordering used for class
    # sorting and the H realization (D2 triple -> scrambled correlation
    # columns).
    a = np.where(np.abs(a) > 1e-8, a, 0.0)
    nz = np.nonzero(np.abs(a) > 1e-8)[0]
    if nz.size == 0:
        return np.zeros(3)
    # Normalize to unit length: the axial-vector magnitude varies per element
    # (2|sin theta|), so a raw axis would rank the same physical axis
    # differently for different members of one conjugacy class (the D5d S10
    # classes: (0,0,1.902) vs (0,0,1.176) for the same z) and scramble the
    # principal tie-break and the on-principal-axis slot test. The unit
    # vector is the representative-independent class key.
    a = a / np.linalg.norm(a)
    if a[nz[0]] < 0.0:
        a = -a
    return a


def axis_key(a):
    """Sortable key of a canonicalized axis (lexicographic)."""
    return tuple(float(x) for x in a)


def same_axis(a, b, tol=1e-6):
    return np.max(np.abs(a - b)) < tol


def classify(g, tol=1e-8):
    """(kind, order, j, axis) of a 3x3 element.

    kind: 'identity' | 'inversion' | 'rotation' | 'sigma' | 'improper'
    order: minimal power g^k = E; mirrors use 1 (the OperationTemplate
        convention), identity/inversion 1.
    j: round(theta * order / 2pi), theta the minimal angle from the trace
        (so C5 and C5^2 get j = 1 and j = 2, S10 and S10^3 j = 1 and j = 3).
    axis: canonicalized direction (first nonzero component positive); None
        for identity/inversion.
    """
    if np.max(np.abs(g - I3)) < tol:
        return ("identity", 1, 1, None)
    if np.max(np.abs(g + I3)) < tol:
        return ("inversion", 1, 1, None)
    det = np.linalg.det(g)
    tr = np.trace(g)
    h = g.copy()
    order = None
    for k in range(1, 25):
        if np.max(np.abs(h - I3)) < 1e-8:
            order = k
            break
        h = h @ g
    assert order is not None, "element order not found"
    if det > 0.0:
        kind = "rotation"
    elif abs(tr - 1.0) < tol:
        return ("sigma", 1, 1, axis_of(g, -1.0))
    else:
        kind = "improper"
    # j = the power of the ambient generator, theta the rotation angle about
    # the canonicalized axis in (-pi, pi]. The bare trace cannot tell theta
    # from 2pi - theta (C4 vs C4^3, C8^5 vs C8^3, ...), so measure the angle
    # from the action on a perpendicular test vector instead.
    axis = axis_of(g, 1.0)
    ref = np.array([1.0, 0.0, 0.0])
    if abs(axis.dot(ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    v = ref - (ref.dot(axis) / axis.dot(axis)) * axis
    v = v / np.linalg.norm(v)
    u = g @ v
    theta = np.arctan2(u.dot(np.cross(axis, v)), u.dot(v))
    j = int(round(theta * order / (2.0 * np.pi))) % order
    if j == 0:
        j = order
    return (kind, order, j, axis)


def conjugacy_classes(elems, tol=1e-8):
    """Partition of element indices into conjugacy classes."""
    n = len(elems)
    pairs = [(x, np.linalg.inv(x)) for x in elems]
    unassigned = set(range(n))
    classes = []
    while unassigned:
        i = min(unassigned)
        g = elems[i]
        cls = []
        for j in sorted(unassigned):
            h = elems[j]
            if any(np.max(np.abs(x @ g @ xi - h)) < tol for x, xi in pairs):
                cls.append(j)
        for j in cls:
            unassigned.discard(j)
        classes.append(cls)
    return classes


def class_of_element(elems, raw_classes, tol=1e-8):
    """class index of every element, given the index-list classes."""
    out = []
    for g in elems:
        for k, cls in enumerate(raw_classes):
            if any(np.max(np.abs(g - elems[i])) < tol for i in cls):
                out.append(k)
                break
        else:
            raise AssertionError("element matched no class")
    return out


# ---------------------------------------------------------------------------
# Classes with slots (the (kind, order, j, slot) table keys)
# ---------------------------------------------------------------------------

KIND_PRIORITY = {"identity": 0, "rotation": 1, "inversion": 2, "improper": 3, "sigma": 4}
SLOT_PRIORITY = {"P": 0, "O0": 1, "O1": 2, "O2": 3, "N": 4}


def assign_slots(class_infos):
    """Attach the axis slot ('P'/'O0'/'O1'/'O2'/'N') to each class.

    Principal rule: the maximal-order proper-rotation class wins; ties are
    broken by the lexicographically smallest canonicalized axis. When no
    proper rotation exists (Cs), a lone sigma class is principal. Sigma and
    improper classes on the principal axis are principal too; everything
    else is an orbit ranked among same-(kind, order, j) classes by min axis.
    """
    rot = [c for c in class_infos if c[0] == "rotation"]
    sigmas = [c for c in class_infos if c[0] == "sigma"]
    principal_class = None
    if rot:
        max_order = max(c[1] for c in rot)
        # The tie-break is DETERMINISTIC: the class-minimal power first,
        # then the lex-min axis. Classes of the same axis (C5 and C5^2 in
        # the order-5 groups) differ in power, so the smaller power wins
        # regardless of the axes' last-bit float noise (the Ih table used
        # to emit C5^2 as the principal class by the raw float-noise min()).
        # The principal class is the class of the principal generator's
        # first power (j minimal); the axis comparison only separates
        # equal-power classes of different axes (the D2 family).
        principal_class = min((c for c in rot if c[1] == max_order),
                              key=lambda c: (c[2], axis_key(c[4])))
        principal_axis = principal_class[4]
    elif len(sigmas) == 1:
        principal_axis = sigmas[0][4]
    else:
        principal_axis = None

    keyed = {}
    for orig, c in enumerate(class_infos):
        kind, order, j, _, min_axis = c
        if kind in ("identity", "inversion"):
            slot = "N"
        elif kind in ("sigma", "improper") and principal_axis is not None and \
                same_axis(min_axis, principal_axis):
            slot = "P"
        elif kind == "rotation" and c is principal_class:
            slot = "P"
        else:
            slot = "N"
        keyed.setdefault((kind, order, j), []).append((orig, c, slot))

    # Preserve the input order; the caller applies the table sort.
    out = [None] * len(class_infos)
    for key in sorted(keyed, key=lambda k: (KIND_PRIORITY[k[0]], k[1], k[2])):
        members = sorted(keyed[key], key=lambda cs: axis_key(cs[1][4]))
        orbit_rank = 0
        for orig, c, slot in members:
            if slot == "N" and c[0] not in ("identity", "inversion"):
                # Orbit ranks skip the principal member of the group.
                slot = f"O{orbit_rank}"
                orbit_rank += 1
            out[orig] = (c[0], c[1], c[2], slot, c[4])
    return out


def sort_classes(class_infos):
    """Table column order: (kind, order, j, slot, min axis)."""
    return sorted(class_infos,
                  key=lambda c: (KIND_PRIORITY[c[0]], c[1], c[2],
                                 SLOT_PRIORITY[c[3]], axis_key(c[4])))


# ---------------------------------------------------------------------------
# Characters via the class-sum algebra
# ---------------------------------------------------------------------------

def characters(elems, cls_of, class_reps, class_sizes, name):
    """(dims, chars[irrep][class], complex_row_flags), rounded to 12 dp.

    Runs on the REGULAR representation (dimension |G|): the natural 3-dim
    representation misses irreps absent from the coordinate action (Ci's Ag,
    C2v's A2, ...), while the regular representation contains every irrep
    with multiplicity dim, so the isotypic decomposition is always complete.
    The class sums act on the isotypic component of irrep Gamma as the
    scalar |C_i| chi_i / dim regardless of the representation, so the
    character extraction is representation-independent.
    """
    n = len(elems)
    n_classes = len(class_reps)

    # Left-action permutation matrices of the regular representation.
    perm = np.zeros((n, n, n))
    for i, g in enumerate(elems):
        for j, e in enumerate(elems):
            p = g @ e
            k = next(k for k in range(n) if np.max(np.abs(p - elems[k])) < 1e-8)
            perm[i, k, j] = 1.0

    sums = []
    for c in range(n_classes):
        s = sum((perm[i] for i in range(n) if cls_of[i] == c), np.zeros((n, n)))
        sums.append(s)

    # Structure constants: S_i S_j = sum_k c_ijk S_k.
    counts = np.zeros((n_classes, n_classes, n_classes))
    for i in range(n):
        for j in range(n):
            p = elems[i] @ elems[j]
            m = next(m for m in range(n) if np.max(np.abs(p - elems[m])) < 1e-6)
            k = cls_of[m]
            counts[cls_of[i], cls_of[j], k] += 1.0
    structure = counts / np.array(class_sizes)[None, None, :]

    # Generic central combination: distinct eigenvalues => eigenspaces are
    # the isotypic components. The class sums are central, so S_i acts as the
    # scalar |C_i| chi_i / dim on each component.
    groups = None
    evecs = None
    for seed in range(100):
        rng = np.random.default_rng(12345 + seed)
        coeffs = rng.uniform(0.3, 1.0, size=n_classes)
        combo = sum(coeffs[i] * sums[i] for i in range(n_classes))
        evals, vecs = np.linalg.eig(combo)
        evals = evals.real
        groups = []
        used = [False] * n
        for k in range(n):
            if used[k]:
                continue
            members = [k]
            used[k] = True
            for l in range(k + 1, n):
                if abs(evals[l] - evals[k]) < 1e-6 * max(1.0, abs(evals[k])):
                    members.append(l)
                    used[l] = True
            groups.append(members)
        distinct = True
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                if abs(evals[g1[0]] - evals[g2[0]]) < 1e-5 * max(1.0, abs(evals[g1[0]])):
                    distinct = False
                    break
            if not distinct:
                break
        if distinct:
            evecs = vecs
            break
    else:
        raise AssertionError(f"{name}: no generic combination found")

    dims = []
    chars = []
    for members in groups:
        v = evecs[:, members[0]].real
        v = v / np.linalg.norm(v)
        mu = np.array([float(v @ sums[i] @ v) for i in range(n_classes)])
        d = np.sqrt(n / np.sum(mu * mu / np.array(class_sizes)))
        r = round(d)
        dim = int(r) if abs(d - r) < 1e-6 else int(round(d * np.sqrt(2.0)))
        chi = mu * dim / np.array(class_sizes)
        dims.append(dim)
        chars.append(np.round(chi, 12))

    # Verification: orthogonality (complex-type rows have the 2|G| diagonal),
    # dimension sums, E-column = dim.
    for a in range(len(chars)):
        for b in range(len(chars)):
            s = sum(class_sizes[c] * chars[a][c] * chars[b][c] for c in range(n_classes))
            if a == b:
                assert abs(s - n) < 1e-6 * (1 + n) or abs(s - 2 * n) < 1e-6 * (1 + n), \
                    f"{name}: row {a} self-sum {s} != |G| or 2|G|"
            else:
                assert abs(s) < 1e-6 * (1 + n), f"{name}: rows {a},{b} not orthogonal ({s})"
    complex_rows = [a for a in range(len(chars))
                    if abs(sum(class_sizes[c] * chars[a][c] ** 2
                               for c in range(n_classes)) - 2 * n) < 1e-6 * (1 + n)]
    total = sum(d * d for d in dims)
    assert abs(total - n - 0.5 * sum(dims[a] * dims[a] for a in complex_rows)) < \
        1e-6 * (1 + n), f"{name}: dim sum {total} inconsistent"
    for a in range(len(chars)):
        assert abs(chars[a][0] - dims[a]) < 1e-6, f"{name}: chi(E) != dim row {a}"
    return dims, chars, complex_rows


# ---------------------------------------------------------------------------
# Label specs (standard Mulliken conventions; verified against the computed
# characters - a mismatch fails the generation loudly)
# ---------------------------------------------------------------------------

def key_of(cls):
    """(kind, order, j, slot) of a class tuple."""
    return (cls[0], cls[1], cls[2], cls[3])


def e2(m, cls):
    """2cos(2*pi*m*j/order) on a rotation/improper class; chi(E) = 2 on the
    identity class; 0 elsewhere."""
    kind, order, j, slot = cls[:4]
    if kind == "identity":
        return 2.0
    if kind in ("rotation", "improper"):
        return 2.0 * np.cos(2.0 * np.pi * m * j / order)
    return 0.0


def b_parity(cls, n_ambient):
    """(-1)^k on the C_n^k classes (k = power of the ambient generator)."""
    kind, order, j, slot = cls[:4]
    if kind == "rotation":
        return (-1.0) ** (j * n_ambient // order)
    return 1.0


def make_specs(group, classes, reps, maps, elems, class_members):
    """Per-group specs: {label: (dim, {(kind, order, j, slot): value})}.

    maps: {'sigma_h': {class_index: class_index}, 'i': {...}} - the class
    permutation induced by multiplication with sigma_h / inversion
    (improper/reflection classes are the images of the rotation classes).
    """
    idx = {key_of(c): i for i, c in enumerate(classes)}
    rot_classes = [i for i, c in enumerate(classes) if c[0] == "rotation"]
    c2_classes = [i for i in rot_classes if classes[i][1] == 2]
    sigma_h_map = maps.get("sigma_h", {})
    i_map = maps.get("i", {})

    def rot_part(i):
        return sigma_h_map.get(i, i_map.get(i, i))

    def c2_on_principal():
        """The order-2 class whose axis is the principal (max-order) axis.

        The P slot goes to the maximal-order rotation class, so for n >= 4
        the principal-axis C2 (Cn/2) is not in the P slot.
        """
        p = next(i for i in rot_classes if classes[i][3] == "P")
        pa = classes[p][4] / np.linalg.norm(classes[p][4])
        return next(i for i in c2_classes
                    if np.max(np.abs(classes[i][4] / np.linalg.norm(classes[i][4])
                                     - pa)) < 1e-6)

    def c2prime_class():
        """The class containing the C2(x) generator element."""
        return next(i for i in range(len(classes))
                    if classes[i][0] == "rotation" and classes[i][1] == 2
                    and any(np.max(np.abs(elems[m] - C2_X)) < 1e-6
                            for m in class_members[i]))

    p_rot = next((i for i in rot_classes if classes[i][3] == "P"), None)
    pa = None
    if p_rot is not None:
        pa = classes[p_rot][4] / np.linalg.norm(classes[p_rot][4])

    def on_principal(i):
        """Class whose (normalized) axis is the principal rotation axis.

        The P slot test is unreliable for sigma_h: rotation/improper axes
        encode the rotation angle (C4 -> (0,0,2)), while a mirror's axis is
        its unit normal, so sigma_h keeps the P slot only when the principal
        rotation is a C2 (D2h) - otherwise it lands in an orbit slot.
        """
        if pa is None:
            return False
        a = classes[i][4] / np.linalg.norm(classes[i][4])
        return np.max(np.abs(a - pa)) < 1e-6

    specs = {}
    all_classes = list(range(len(classes)))
    e_key = key_of(classes[0])

    def all_one():
        return {key_of(classes[i]): 1.0 for i in all_classes}

    if group == "C1":
        specs["A"] = (1, all_one())
    elif group == "Cs":
        s = [i for i, c in enumerate(classes) if c[0] == "sigma"][0]
        specs["A'"] = (1, {e_key: 1.0, key_of(classes[s]): 1.0})
        specs["A''"] = (1, {e_key: 1.0, key_of(classes[s]): -1.0})
    elif group == "Ci":
        inv = [i for i, c in enumerate(classes) if c[0] == "inversion"][0]
        specs["Ag"] = (1, {e_key: 1.0, key_of(classes[inv]): 1.0})
        specs["Au"] = (1, {e_key: 1.0, key_of(classes[inv]): -1.0})
    elif group == "D2":
        z = [i for i in c2_classes if classes[i][3] == "P"][0]
        y = [i for i in c2_classes if classes[i][3] == "O0"][0]
        x = [i for i in c2_classes if classes[i][3] == "O1"][0]
        specs["A"] = (1, all_one())
        specs["B1"] = (1, {e_key: 1.0, key_of(classes[z]): 1.0,
                           key_of(classes[y]): -1.0, key_of(classes[x]): -1.0})
        specs["B2"] = (1, {e_key: 1.0, key_of(classes[z]): -1.0,
                           key_of(classes[y]): 1.0, key_of(classes[x]): -1.0})
        specs["B3"] = (1, {e_key: 1.0, key_of(classes[z]): -1.0,
                           key_of(classes[y]): -1.0, key_of(classes[x]): 1.0})
    elif group == "D2h":
        z = [i for i in c2_classes if classes[i][3] == "P"][0]
        y = [i for i in c2_classes if classes[i][3] == "O0"][0]
        x = [i for i in c2_classes if classes[i][3] == "O1"][0]
        sig_p = [i for i, c in enumerate(classes) if c[0] == "sigma" and c[3] == "P"][0]
        sig_y = [i for i, c in enumerate(classes) if c[0] == "sigma" and c[3] == "O0"][0]
        sig_x = [i for i, c in enumerate(classes) if c[0] == "sigma" and c[3] == "O1"][0]
        inv = [i for i, c in enumerate(classes) if c[0] == "inversion"][0]

        def d2h_row(axis_signs, g):
            """axis_signs: (z, y, x) on the rotations; u flips i/sigma."""
            spec = {e_key: 1.0}
            spec[key_of(classes[z])] = axis_signs[0]
            spec[key_of(classes[y])] = axis_signs[1]
            spec[key_of(classes[x])] = axis_signs[2]
            spec[key_of(classes[inv])] = g
            spec[key_of(classes[sig_p])] = g * axis_signs[0]
            spec[key_of(classes[sig_y])] = g * axis_signs[1]
            spec[key_of(classes[sig_x])] = g * axis_signs[2]
            return spec

        specs["Ag"] = (1, d2h_row((1, 1, 1), 1.0))
        specs["B1g"] = (1, d2h_row((1, -1, -1), 1.0))
        specs["B2g"] = (1, d2h_row((-1, 1, -1), 1.0))
        specs["B3g"] = (1, d2h_row((-1, -1, 1), 1.0))
        specs["Au"] = (1, d2h_row((1, 1, 1), -1.0))
        specs["B1u"] = (1, d2h_row((1, -1, -1), -1.0))
        specs["B2u"] = (1, d2h_row((-1, 1, -1), -1.0))
        specs["B3u"] = (1, d2h_row((-1, -1, 1), -1.0))
    elif group.startswith("C") and "v" not in group and "h" not in group:
        n = int(group[1:])
        specs["A"] = (1, all_one())
        if n % 2 == 0:
            specs["B"] = (1, {key_of(classes[i]): b_parity(classes[i], n)
                              for i in rot_classes})
        for m in range(1, (n - 1) // 2 + 1):
            specs[f"E{m if n >= 5 else ''}"] = (
                2, {key_of(classes[i]): e2(m, classes[i]) for i in all_classes})
    elif "v" in group:
        n = int(group[1:-1])
        sigma_classes = [i for i, c in enumerate(classes) if c[0] == "sigma"]
        sigmas_sorted = sorted(sigma_classes, key=lambda i: axis_key(classes[i][4]))
        specs["A1"] = (1, all_one())
        specs["A2"] = (1, {key_of(classes[i]): (1.0 if classes[i][0] == "rotation" else -1.0)
                           for i in all_classes if classes[i][0] != "identity"})
        if n % 2 == 0:
            # B1: sigma_v +1 / sigma_d -1; B2 the reverse (sigma_v = the class
            # of the planes through the C2(x) generator axis, the smaller
            # min-axis, matching the kC2v table's (sigma_y, sigma_x) order).
            for label, sgn0, sgn1 in (("B1", 1.0, -1.0), ("B2", -1.0, 1.0)):
                spec = {}
                for i in all_classes:
                    if classes[i][0] == "rotation":
                        spec[key_of(classes[i])] = b_parity(classes[i], n)
                    elif classes[i][0] == "sigma":
                        spec[key_of(classes[i])] = sgn0 if i == sigmas_sorted[0] else sgn1
                specs[label] = (1, spec)
        for m in range(1, (n - 1) // 2 + 1):
            specs[f"E{m if n >= 5 else ''}"] = (
                2, {key_of(classes[i]): e2(m, classes[i]) for i in all_classes})
    elif "h" in group and group[0] == "C":
        n = int(group[1:-1])
        impropers = [i for i, c in enumerate(classes) if c[0] == "improper"]
        if n % 2 == 1:
            specs["A'"] = (1, all_one())
            specs["A''"] = (1, {key_of(classes[i]): (1.0 if classes[i][0] == "rotation" else -1.0)
                                for i in all_classes if classes[i][0] != "identity"})
            for m in range(1, (n - 1) // 2 + 1):
                label = f"E{m if n >= 5 else ''}"
                for sgn, suffix in ((1.0, "'"), (-1.0, "''")):
                    spec = {}
                    for i in all_classes:
                        kind = classes[i][0]
                        if kind == "rotation":
                            spec[key_of(classes[i])] = e2(m, classes[i])
                        elif kind == "sigma":
                            spec[key_of(classes[i])] = 2.0 * sgn
                        elif kind == "improper":
                            spec[key_of(classes[i])] = sgn * e2(m, classes[rot_part(i)])
                    specs[label + suffix] = (2, spec)
        else:
            c2 = c2_on_principal()
            for label, b_like, g in (("Ag", False, 1.0), ("Bg", True, 1.0),
                                     ("Au", False, -1.0), ("Bu", True, -1.0)):
                spec = {}
                for i in all_classes:
                    kind = classes[i][0]
                    if kind == "rotation":
                        spec[key_of(classes[i])] = b_parity(classes[i], n) if b_like else 1.0
                    elif kind == "inversion":
                        spec[key_of(classes[i])] = g
                    elif kind == "sigma":
                        val = g * (b_parity(classes[c2], n) if b_like else 1.0)
                        spec[key_of(classes[i])] = val
                    elif kind == "improper":
                        val = g * (b_parity(classes[c2], n) if b_like else 1.0)
                        val *= b_parity(classes[rot_part(i)], n) if b_like else 1.0
                        spec[key_of(classes[i])] = val
                specs[label] = (1, spec)
            for m in range(1, n // 2):
                for sgn, suffix in ((1.0, "g"), (-1.0, "u")):
                    spec = {}
                    for i in all_classes:
                        kind = classes[i][0]
                        if kind == "rotation":
                            spec[key_of(classes[i])] = e2(m, classes[i])
                        elif kind == "inversion":
                            spec[key_of(classes[i])] = 2.0 * sgn
                        elif kind == "sigma":
                            # sigma_h = C2z * i; both act as scalars on the
                            # E_m pair: chi = sgn * e2(m, C2) = 2*sgn*(-1)^m.
                            spec[key_of(classes[i])] = sgn * e2(m, classes[c2])
                        elif kind == "improper":
                            # S_2n^k = sigma_h * C_n^k: the sigma_h part acts
                            # on the E_m pair as sgn*(-1)^m (the g/u parity
                            # alternates with m on the z-carrying pairs).
                            spec[key_of(classes[i])] = \
                                sgn * (-1.0) ** m * e2(m, classes[rot_part(i)])
                    specs[f"E{m if n >= 6 else ''}{suffix}"] = (2, spec)
    elif group.startswith("D") and "h" not in group and "d" not in group:
        n = int(group[1:])
        c2prime = c2prime_class()
        try:
            c2p = c2_on_principal()
        except StopIteration:
            c2p = None
        specs["A1"] = (1, all_one())
        # A2: +1 on the principal-axis rotations (C2z = Cn/2 included), -1 on
        # the perpendicular C2' classes (odd n: every C2 is perpendicular).
        specs["A2"] = (1, {key_of(classes[i]):
                           (1.0 if classes[i][1] != 2 or i == c2p else -1.0)
                           for i in all_classes})
        if n % 2 == 0:
            for label, sgn in (("B1", 1.0), ("B2", -1.0)):
                spec = {}
                for i in rot_classes:
                    if classes[i][1] != 2 or i == c2p:
                        spec[key_of(classes[i])] = b_parity(classes[i], n)
                    else:
                        spec[key_of(classes[i])] = sgn if i == c2prime else -sgn
                specs[label] = (1, spec)
        for m in range(1, (n - 1) // 2 + 1):
            spec = {}
            for i in all_classes:
                if classes[i][0] == "rotation" and classes[i][1] == 2 \
                        and i != c2p:
                    spec[key_of(classes[i])] = 0.0
                else:
                    spec[key_of(classes[i])] = e2(m, classes[i])
            specs[f"E{m if n >= 5 else ''}"] = (2, spec)
    elif "h" in group and group[0] in ("C", "D"):
        n = int(group[1:-1])
        c2prime = c2prime_class()
        c2prime_sigma = sigma_h_map.get(c2prime)
        if n % 2 == 1:
            specs["A1'"] = (1, all_one())
            # Dnh-odd = Dn x Cs: A2' = A2 (x) A' has chi(S) = +1, chi(sigma_v)
            # = -1 (the sigma_v class is the (C2', sigma_h) product class);
            # A2'' = A2 (x) A'' flips both. sigma_h is the sigma class on the
            # principal axis (on_principal; its slot is not P once the
            # principal rotation exceeds C2).
            specs["A2'"] = (1, {key_of(classes[i]):
                                (1.0 if classes[i][0] == "identity"
                                 or (classes[i][0] == "rotation"
                                     and classes[i][1] != 2)
                                 else -1.0 if classes[i][0] == "rotation" else
                                 1.0 if classes[i][0] == "improper" else
                                 1.0 if on_principal(i) else -1.0)
                                for i in all_classes})
            specs["A1''"] = (1, {key_of(classes[i]): (-1.0 if classes[i][0] in
                                                      ("sigma", "improper") else 1.0)
                                 for i in all_classes})
            specs["A2''"] = (1, {key_of(classes[i]):
                                 (1.0 if classes[i][0] == "identity"
                                  or (classes[i][0] == "rotation"
                                      and classes[i][1] != 2) else
                                  -1.0 if classes[i][0] == "rotation" else
                                  -1.0 if classes[i][0] == "improper" else
                                  -1.0 if on_principal(i) else 1.0)
                                 for i in all_classes})
            for m in range(1, (n - 1) // 2 + 1):
                label = f"E{m if n >= 5 else ''}"
                for sgn, suffix in ((1.0, "'"), (-1.0, "''")):
                    spec = {}
                    for i in all_classes:
                        kind = classes[i][0]
                        if kind == "rotation":
                            if classes[i][1] == 2:
                                spec[key_of(classes[i])] = 0.0
                            else:
                                spec[key_of(classes[i])] = e2(m, classes[i])
                        elif kind == "sigma":
                            spec[key_of(classes[i])] = 2.0 * sgn \
                                if on_principal(i) else 0.0
                        elif kind == "improper":
                            # The E pair rotates by the S-class's own angle
                            # (sigma_h flips z only); the sigma_h scale is sgn.
                            spec[key_of(classes[i])] = sgn * e2(m, classes[i])
                    specs[label + suffix] = (2, spec)
        else:
            c2 = c2_on_principal()
            for label, b_like, c2p_sig, g in (("A1g", False, 1.0, 1.0),
                                              ("A2g", False, -1.0, 1.0),
                                              ("B1g", True, 1.0, 1.0),
                                              ("B2g", True, -1.0, 1.0),
                                              ("A1u", False, 1.0, -1.0),
                                              ("A2u", False, -1.0, -1.0),
                                              ("B1u", True, 1.0, -1.0),
                                              ("B2u", True, -1.0, -1.0)):
                spec = {}
                for i in all_classes:
                    kind = classes[i][0]
                    if kind == "rotation":
                        if classes[i][1] == 2:
                            # The principal-axis C2 (C2z = Cn/2, an orbit slot
                            # for n >= 4) takes b_parity / +1 like the C_n
                            # class (c2 here is the class INDEX; c2p_sig is the
                            # per-label perpendicular sign). Perpendicular
                            # classes: C2'(x) pairs with sigma_v at +c2p_sig for
                            # every row; the C2''/sigma_d classes pair at
                            # +c2p_sig on the A rows (verified on D4h: A2g
                            # chi(C2'') = -1, chi(sigma_d) = -1) and -c2p_sig
                            # on the B rows.
                            if classes[i][3] == "P" or i == c2:
                                spec[key_of(classes[i])] = \
                                    b_parity(classes[i], n) if b_like else 1.0
                            else:
                                spec[key_of(classes[i])] = \
                                    c2p_sig if i == c2prime else \
                                    (c2p_sig if label[0] == "A" else -c2p_sig)
                        else:
                            spec[key_of(classes[i])] = \
                                b_parity(classes[i], n) if b_like else 1.0
                    elif kind == "inversion":
                        spec[key_of(classes[i])] = g
                    elif kind == "sigma":
                        base = g * (b_parity(classes[c2], n) if b_like else 1.0)
                        if on_principal(i):
                            spec[key_of(classes[i])] = base
                        else:
                            spec[key_of(classes[i])] = base * \
                                (c2p_sig if i == c2prime_sigma
                                 else (c2p_sig if label[0] == "A" else -c2p_sig))
                    elif kind == "improper":
                        base = g * (b_parity(classes[c2], n) if b_like else 1.0)
                        spec[key_of(classes[i])] = base * \
                            (b_parity(classes[rot_part(i)], n) if b_like else 1.0)
                specs[label] = (1, spec)
            for m in range(1, n // 2):
                for sgn, suffix in ((1.0, "g"), (-1.0, "u")):
                    spec = {}
                    for i in all_classes:
                        kind = classes[i][0]
                        if kind == "rotation":
                            # Zero only the PERPENDICULAR C2 classes; the
                            # principal-axis C2z keeps e2(m, C2z) = 2*(-1)^m.
                            if classes[i][1] == 2 and i != c2:
                                spec[key_of(classes[i])] = 0.0
                            else:
                                spec[key_of(classes[i])] = e2(m, classes[i])
                        elif kind == "inversion":
                            spec[key_of(classes[i])] = 2.0 * sgn
                        elif kind == "sigma":
                            # sigma_h class: chi = sgn * e2(m, C2) (see the
                            # Cnh-even branch); sigma_v/sigma_d: 0.
                            if on_principal(i):
                                spec[key_of(classes[i])] = sgn * e2(m, classes[c2])
                            else:
                                spec[key_of(classes[i])] = 0.0
                        elif kind == "improper":
                            # S_2n^k = sigma_h * C_n^k: sigma_h acts on the
                            # E_m pair as sgn*(-1)^m (Cnh-even branch note).
                            spec[key_of(classes[i])] = \
                                sgn * (-1.0) ** m * e2(m, classes[rot_part(i)])
                    specs[f"E{m if n >= 6 else ''}{suffix}"] = (2, spec)
    elif "d" in group and group[0] == "D":
        n = int(group[1:-1])
        c2prime = c2prime_class()
        sigma_idx = [i for i, c in enumerate(classes) if c[0] == "sigma"][0]
        impropers = [i for i, c in enumerate(classes) if c[0] == "improper"]
        if n % 2 == 1:
            for label, c2p, g in (("A1g", 1.0, 1.0), ("A2g", -1.0, 1.0),
                                  ("A1u", 1.0, -1.0), ("A2u", -1.0, -1.0)):
                spec = {}
                for i in all_classes:
                    kind = classes[i][0]
                    if kind == "rotation":
                        spec[key_of(classes[i])] = 1.0 if classes[i][1] != 2 else c2p
                    elif kind == "inversion":
                        spec[key_of(classes[i])] = g
                    elif kind == "sigma":
                        spec[key_of(classes[i])] = g * c2p
                    elif kind == "improper":
                        spec[key_of(classes[i])] = g
                specs[label] = (1, spec)
            for m in range(1, (n - 1) // 2 + 1):
                label = f"E{m if n >= 5 else ''}"
                for sgn, suffix in ((1.0, "g"), (-1.0, "u")):
                    spec = {}
                    for i in all_classes:
                        kind = classes[i][0]
                        if kind == "rotation":
                            if classes[i][1] == 2:
                                spec[key_of(classes[i])] = 0.0
                            else:
                                spec[key_of(classes[i])] = e2(m, classes[i])
                        elif kind == "inversion":
                            spec[key_of(classes[i])] = 2.0 * sgn
                        elif kind == "sigma":
                            spec[key_of(classes[i])] = 0.0
                        elif kind == "improper":
                            # E pairs are z-carrying: the reflection scale
                            # alternates with m (D5d: E1g = -e2, E2g = +e2) on
                            # top of the g/u sign; chi = sgn * (-1)^m * e2(m,
                            # S-class), the S-class angle directly.
                            spec[key_of(classes[i])] = \
                                sgn * (-1.0) ** m * e2(m, classes[i])
                    specs[label + suffix] = (2, spec)
        else:
            two_n = 2 * n
            c2p = c2_on_principal()
            for label, c2p_sig, sgn_sig in (("A1", 1.0, 1.0), ("A2", -1.0, -1.0),
                                            ("B1", 1.0, -1.0), ("B2", -1.0, 1.0)):
                spec = {}
                for i in all_classes:
                    kind = classes[i][0]
                    if kind == "rotation":
                        if classes[i][1] == 2:
                            # Principal-axis C2 (D4d: an orbit slot): +1 on
                            # every 1-dim row (C2z = S_2n^n); the
                            # perpendicular C2' classes pair via c2p_sig.
                            spec[key_of(classes[i])] = \
                                1.0 if (classes[i][3] == "P" or i == c2p) else c2p_sig
                        else:
                            spec[key_of(classes[i])] = 1.0
                    elif kind == "improper":
                        # chi(S_2n^k) = (-1)^k on B1/B2 (k = j * 2n/order),
                        # +1 on A1/A2 (verified on D2d/D4d probes).
                        _, order, j, _, _ = classes[i]
                        k = j * two_n // order
                        spec[key_of(classes[i])] = (-1.0) ** k if label[0] == "B" else 1.0
                    elif kind == "sigma":
                        spec[key_of(classes[i])] = sgn_sig
                specs[label] = (1, spec)
            for m in range(1, n):
                spec = {}
                for i in all_classes:
                    kind = classes[i][0]
                    if kind in ("rotation", "improper"):
                        if kind == "rotation" and classes[i][1] == 2 and i != c2p:
                            spec[key_of(classes[i])] = 0.0
                        else:
                            # z-free pairs: chi(E_m, S_2n^k) = e2(m, S-class)
                            # (verified on the D2d/D4d probes).
                            spec[key_of(classes[i])] = e2(m, classes[i])
                    elif kind == "identity":
                        spec[key_of(classes[i])] = 2.0
                    else:
                        spec[key_of(classes[i])] = 0.0
                specs[f"E{m}"] = (2, spec)
    elif group in ("S4", "S6", "S8"):
        impropers = [i for i, c in enumerate(classes) if c[0] == "improper"]
        c2 = [i for i in rot_classes if classes[i][1] == 2]
        c3 = [i for i in rot_classes if classes[i][1] == 3]
        inv = [i for i, c in enumerate(classes) if c[0] == "inversion"]
        if group == "S4":
            specs["A"] = (1, all_one())
            specs["B"] = (1, {e_key: 1.0, key_of(classes[impropers[0]]): -1.0,
                              key_of(classes[c2[0]]): 1.0})
            specs["E"] = (2, {e_key: 2.0, key_of(classes[impropers[0]]): 0.0,
                              key_of(classes[c2[0]]): -2.0})
        elif group == "S6":
            specs["Ag"] = (1, all_one())
            specs["Eg"] = (2, {e_key: 2.0, key_of(classes[c3[0]]): -1.0,
                               key_of(classes[inv[0]]): 2.0,
                               key_of(classes[impropers[0]]): -1.0})
            specs["Au"] = (1, {e_key: 1.0, key_of(classes[c3[0]]): 1.0,
                               key_of(classes[inv[0]]): -1.0,
                               key_of(classes[impropers[0]]): -1.0})
            specs["Eu"] = (2, {e_key: 2.0, key_of(classes[c3[0]]): -1.0,
                               key_of(classes[inv[0]]): -2.0,
                               key_of(classes[impropers[0]]): 1.0})
        else:
            specs["A"] = (1, all_one())
            b = {e_key: 1.0}
            for i in all_classes:
                kind, order, j, _, _ = classes[i]
                if kind == "improper":
                    b[key_of(classes[i])] = (-1.0) ** (j * 8 // order)
                elif kind == "rotation":
                    b[key_of(classes[i])] = 1.0
            specs["B"] = (1, b)
            for m in (1, 2, 3):
                specs[f"E{m}"] = (2, {key_of(classes[i]): e2(m, classes[i])
                                      for i in all_classes})
    elif group == "T":
        c3 = [i for i in rot_classes if classes[i][1] == 3]
        c2 = [i for i in rot_classes if classes[i][1] == 2]
        specs["A"] = (1, all_one())
        specs["E"] = (2, {e_key: 2.0, key_of(classes[c3[0]]): -1.0,
                          key_of(classes[c3[1]]): -1.0, key_of(classes[c2[0]]): 2.0})
        specs["T"] = (3, {e_key: 3.0, key_of(classes[c3[0]]): 0.0,
                          key_of(classes[c3[1]]): 0.0, key_of(classes[c2[0]]): -1.0})
    elif group == "Td":
        c3 = [i for i in rot_classes if classes[i][1] == 3][0]
        c2 = [i for i in rot_classes if classes[i][1] == 2][0]
        s4 = [i for i, c in enumerate(classes) if c[0] == "improper"][0]
        sg = [i for i, c in enumerate(classes) if c[0] == "sigma"][0]
        specs["A1"] = (1, all_one())
        specs["A2"] = (1, {e_key: 1.0, key_of(classes[c3]): 1.0,
                           key_of(classes[c2]): 1.0, key_of(classes[s4]): -1.0,
                           key_of(classes[sg]): -1.0})
        specs["E"] = (2, {e_key: 2.0, key_of(classes[c3]): -1.0,
                          key_of(classes[c2]): 2.0, key_of(classes[s4]): 0.0,
                          key_of(classes[sg]): 0.0})
        specs["T1"] = (3, {e_key: 3.0, key_of(classes[c3]): 0.0,
                           key_of(classes[c2]): -1.0, key_of(classes[s4]): 1.0,
                           key_of(classes[sg]): -1.0})
        specs["T2"] = (3, {e_key: 3.0, key_of(classes[c3]): 0.0,
                           key_of(classes[c2]): -1.0, key_of(classes[s4]): -1.0,
                           key_of(classes[sg]): 1.0})
    elif group == "Th":
        c3 = [i for i in rot_classes if classes[i][1] == 3]
        c2 = [i for i in rot_classes if classes[i][1] == 2][0]
        s6 = [i for i, c in enumerate(classes) if c[0] == "improper"]
        inv = [i for i, c in enumerate(classes) if c[0] == "inversion"][0]
        sg = [i for i, c in enumerate(classes) if c[0] == "sigma"][0]
        specs["Ag"] = (1, all_one())
        specs["Eg"] = (2, {e_key: 2.0, key_of(classes[c3[0]]): -1.0,
                           key_of(classes[c3[1]]): -1.0, key_of(classes[c2]): 2.0,
                           key_of(classes[inv]): 2.0, key_of(classes[s6[0]]): -1.0,
                           key_of(classes[s6[1]]): -1.0, key_of(classes[sg]): 2.0})
        specs["Tg"] = (3, {e_key: 3.0, key_of(classes[c3[0]]): 0.0,
                           key_of(classes[c3[1]]): 0.0, key_of(classes[c2]): -1.0,
                           key_of(classes[inv]): 3.0, key_of(classes[s6[0]]): 0.0,
                           key_of(classes[s6[1]]): 0.0, key_of(classes[sg]): -1.0})
        specs["Au"] = (1, {key_of(classes[i]): (1.0 if classes[i][0] == "rotation" else -1.0)
                           for i in all_classes if classes[i][0] != "identity"})
        specs["Eu"] = (2, {e_key: 2.0, key_of(classes[c3[0]]): -1.0,
                           key_of(classes[c3[1]]): -1.0, key_of(classes[c2]): 2.0,
                           key_of(classes[inv]): -2.0, key_of(classes[s6[0]]): 1.0,
                           key_of(classes[s6[1]]): 1.0, key_of(classes[sg]): -2.0})
        specs["Tu"] = (3, {e_key: 3.0, key_of(classes[c3[0]]): 0.0,
                           key_of(classes[c3[1]]): 0.0, key_of(classes[c2]): -1.0,
                           key_of(classes[inv]): -3.0, key_of(classes[s6[0]]): 0.0,
                           key_of(classes[s6[1]]): 0.0, key_of(classes[sg]): 1.0})
    elif group == "O":
        c3 = [i for i in rot_classes if classes[i][1] == 3][0]
        c4 = [i for i in rot_classes if classes[i][1] == 4][0]
        # The coordinate-axis C2 (C4^2) is an orbit slot in O (the C4 class
        # holds P); the diagonal C2' classes are the other orbit. The
        # coordinate class ranks O0 (its min axis (0,0,1) beats any diagonal
        # one lexicographically).
        c2z = [i for i in c2_classes if classes[i][3] == "O0"][0]
        c2p = [i for i in c2_classes if classes[i][3] != "O0"][0]
        rows = {
            "A1": (1.0, 1.0, 1.0, 1.0, 1.0),
            "A2": (1.0, 1.0, 1.0, -1.0, -1.0),
            "E": (2.0, -1.0, 2.0, 0.0, 0.0),
            "T1": (3.0, 0.0, -1.0, 1.0, -1.0),
            "T2": (3.0, 0.0, -1.0, -1.0, 1.0),
        }
        order = [0, c3, c2z, c4, c2p]
        for label, row in rows.items():
            specs[label] = (int(row[0]), {key_of(classes[order[k]]): row[k]
                                          for k in range(len(order))})
    elif group == "Oh":
        c3 = [i for i in rot_classes if classes[i][1] == 3][0]
        c4 = [i for i in rot_classes if classes[i][1] == 4][0]
        # The coordinate-axis C2 (C4^2) is an orbit slot in Oh (the C4 and S4
        # classes hold P); the diagonal C2' classes are the other orbit.
        # sigma_h (normal on the C4 axis) holds P once the axes are
        # normalized, so the sigma classes are picked by axis, not slot.
        c2z = [i for i in c2_classes if same_axis(classes[i][4], classes[c4][4])][0]
        c2p = [i for i in c2_classes if not same_axis(classes[i][4], classes[c4][4])][0]
        s4 = [i for i, c in enumerate(classes) if c[0] == "improper" and c[1] == 4][0]
        s6 = [i for i, c in enumerate(classes) if c[0] == "improper" and c[1] == 6][0]
        inv = [i for i, c in enumerate(classes) if c[0] == "inversion"][0]
        sg_p = [i for i, c in enumerate(classes)
                if c[0] == "sigma" and same_axis(c[4], classes[c4][4])][0]
        sg_o = [i for i, c in enumerate(classes)
                if c[0] == "sigma" and not same_axis(c[4], classes[c4][4])][0]
        rows = {
            "A1g": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "A2g": (1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0),
            "Eg": (2.0, -1.0, 0.0, 0.0, 2.0, 2.0, 0.0, -1.0, 2.0, 0.0),
            "T1g": (3.0, 0.0, -1.0, 1.0, -1.0, 3.0, 1.0, 0.0, -1.0, -1.0),
            "T2g": (3.0, 0.0, 1.0, -1.0, -1.0, 3.0, -1.0, 0.0, -1.0, 1.0),
            "A1u": (1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            "A2u": (1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0),
            "Eu": (2.0, -1.0, 0.0, 0.0, 2.0, -2.0, 0.0, 1.0, -2.0, 0.0),
            "T1u": (3.0, 0.0, -1.0, 1.0, -1.0, -3.0, -1.0, 0.0, 1.0, 1.0),
            "T2u": (3.0, 0.0, 1.0, -1.0, -1.0, -3.0, 1.0, 0.0, 1.0, -1.0),
        }
        order = [0, c3, c2p, c4, c2z, inv, s4, s6, sg_p, sg_o]
        for label, row in rows.items():
            specs[label] = (int(row[0]), {key_of(classes[order[k]]): row[k]
                                          for k in range(len(order))})
    elif group == "I":
        c5a = [i for i in rot_classes if classes[i][1] == 5 and classes[i][2] == 1][0]
        c5b = [i for i in rot_classes if classes[i][1] == 5 and classes[i][2] == 2][0]
        c3 = [i for i in rot_classes if classes[i][1] == 3][0]
        c2 = [i for i in rot_classes if classes[i][1] == 2][0]
        rows = {
            "A": (1.0, 1.0, 1.0, 1.0, 1.0),
            "T1": (3.0, PHI, 1.0 - PHI, 0.0, -1.0),
            "T2": (3.0, 1.0 - PHI, PHI, 0.0, -1.0),
            "G": (4.0, -1.0, -1.0, 1.0, 0.0),
            "H": (5.0, 0.0, 0.0, -1.0, 1.0),
        }
        order = [0, c5a, c5b, c3, c2]
        for label, row in rows.items():
            specs[label] = (int(row[0]), {key_of(classes[order[k]]): row[k]
                                          for k in range(len(order))})
    elif group == "Ih":
        c5a = [i for i in rot_classes if classes[i][1] == 5 and classes[i][2] == 1][0]
        c5b = [i for i in rot_classes if classes[i][1] == 5 and classes[i][2] == 2][0]
        c3 = [i for i in rot_classes if classes[i][1] == 3][0]
        c2 = [i for i in rot_classes if classes[i][1] == 2][0]
        # The two S10 classes (canonical class-min powers j = 1 and 3);
        # sort by j so the first S10 slot is the S10^1 class (36-degree
        # impropers) and the second the S10^3 class (108 degrees), the
        # published table's 12S10 and 12S10^3 columns.
        s10 = [i for i, c in enumerate(classes)
               if c[0] == "improper" and c[1] == 10]
        s10a, s10b = sorted(s10, key=lambda i: classes[i][2])
        s6 = [i for i, c in enumerate(classes)
              if c[0] == "improper" and c[1] == 6][0]
        inv = [i for i, c in enumerate(classes) if c[0] == "inversion"][0]
        sg = [i for i, c in enumerate(classes) if c[0] == "sigma"][0]
        rows = {
            "Ag": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "T1g": (3.0, PHI, 1.0 - PHI, 0.0, -1.0, 3.0, 1.0 - PHI, PHI, 0.0, -1.0),
            "T2g": (3.0, 1.0 - PHI, PHI, 0.0, -1.0, 3.0, PHI, 1.0 - PHI, 0.0, -1.0),
            "Gg": (4.0, -1.0, -1.0, 1.0, 0.0, 4.0, -1.0, -1.0, 1.0, 0.0),
            "Hg": (5.0, 0.0, 0.0, -1.0, 1.0, 5.0, 0.0, 0.0, -1.0, 1.0),
            "Au": (1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
            "T1u": (3.0, PHI, 1.0 - PHI, 0.0, -1.0, -3.0, -(1.0 - PHI), -PHI, 0.0, 1.0),
            "T2u": (3.0, 1.0 - PHI, PHI, 0.0, -1.0, -3.0, -PHI, -(1.0 - PHI), 0.0, 1.0),
            "Gu": (4.0, -1.0, -1.0, 1.0, 0.0, -4.0, 1.0, 1.0, -1.0, 0.0),
            "Hu": (5.0, 0.0, 0.0, -1.0, 1.0, -5.0, 0.0, 0.0, 1.0, -1.0),
        }
        order = [0, c5a, c5b, c3, c2, inv, s10a, s10b, s6, sg]
        for label, row in rows.items():
            specs[label] = (int(row[0]), {key_of(classes[order[k]]): row[k]
                                          for k in range(len(order))})
    else:
        raise AssertionError(f"no spec builder for {group}")
    return specs


# ---------------------------------------------------------------------------
# Correlation: subduction to the realizable Abelian subgroup
# ---------------------------------------------------------------------------

ABELIAN_TARGET = {
    "C1": "C1", "Ci": "Ci", "Cs": "Cs",
    "C2": "C2", "C3": "C1", "C4": "C2", "C5": "C1", "C6": "C2", "C7": "C1", "C8": "C2",
    "C2v": "C2v", "C3v": "Cs", "C4v": "C2v", "C5v": "Cs", "C6v": "C2v",
    "C7v": "Cs", "C8v": "C2v",
    "C2h": "C2h", "C3h": "Cs", "C4h": "C2h", "C5h": "Cs", "C6h": "C2h",
    "C7h": "Cs", "C8h": "C2h",
    "D2": "D2", "D3": "C2", "D4": "D2", "D5": "C2", "D6": "D2", "D7": "C2", "D8": "D2",
    "D2h": "D2h", "D3h": "C2v", "D4h": "D2h", "D5h": "C2v", "D6h": "D2h",
    "D7h": "C2v", "D8h": "D2h",
    "D2d": "D2", "D3d": "C2h", "D4d": "D2", "D5d": "C2h", "D6d": "D2",
    "D7d": "C2h", "D8d": "D2",
    "S4": "C2", "S6": "Ci", "S8": "C2",
    "T": "D2", "Td": "D2", "Th": "D2h", "O": "D2", "Oh": "D2h",
    "I": "D2", "Ih": "D2h",
}

# The existing Abelian tables' rows: label -> characters over the canonical
# element order (E, principal, then orbits; see character_tables.hpp).
ABELIAN_TABLES = {
    "C1": {"A": (1.0,)},
    "Cs": {"A'": (1.0, 1.0), "A''": (1.0, -1.0)},
    "Ci": {"Ag": (1.0, 1.0), "Au": (1.0, -1.0)},
    "C2": {"A": (1.0, 1.0), "B": (1.0, -1.0)},
    "C2v": {"A1": (1.0, 1.0, 1.0, 1.0), "A2": (1.0, 1.0, -1.0, -1.0),
            "B1": (1.0, -1.0, 1.0, -1.0), "B2": (1.0, -1.0, -1.0, 1.0)},
    "C2h": {"Ag": (1.0, 1.0, 1.0, 1.0), "Bg": (1.0, -1.0, 1.0, -1.0),
            "Au": (1.0, 1.0, -1.0, -1.0), "Bu": (1.0, -1.0, -1.0, 1.0)},
    "D2": {"A": (1.0, 1.0, 1.0, 1.0), "B1": (1.0, 1.0, -1.0, -1.0),
           "B2": (1.0, -1.0, 1.0, -1.0), "B3": (1.0, -1.0, -1.0, 1.0)},
    "D2h": {"Ag": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            "B1g": (1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0),
            "B2g": (1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0),
            "B3g": (1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0),
            "Au": (1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0),
            "B1u": (1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0),
            "B2u": (1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0),
            "B3u": (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0)},
}

CXX_ABELIAN = {"C1": "kC1", "Cs": "kCs", "Ci": "kCi", "C2": "kC2", "C2v": "kC2v",
               "C2h": "kC2h", "D2": "kD2", "D2h": "kD2h"}


def realize_h(group, elems, raw_classes, class_infos):
    """The canonical H <= G realization: elements in the Abelian tables' order."""
    target = ABELIAN_TARGET[group]
    n = len(elems)
    cls_of = class_of_element(elems, raw_classes)
    kind_of = [c[0] for c in class_infos]

    def rep_of_class(c):
        return next(elems[i] for i in range(n) if cls_of[i] == c)

    def classes_of_kind(kind):
        return [c for c in range(len(class_infos)) if kind_of[c] == kind]

    rot2 = [c for c in classes_of_kind("rotation") if class_infos[c][1] == 2]
    sigmas = sorted(classes_of_kind("sigma"), key=lambda c: axis_key(class_infos[c][4]))
    inversion = classes_of_kind("inversion")

    if target == "C1":
        h = [I3]
    elif target == "Cs":
        h = [I3, rep_of_class(sigmas[0])]
    elif target == "Ci":
        h = [I3, rep_of_class(inversion[0])]
    elif target == "C2":
        h = [I3, rep_of_class(rot2[0])]
    elif target == "C2v":
        # The closing triple (E, C2, m0, m1) with m1 = C2*m0: the two mirrors
        # are the planes 90 deg apart (sigma(xz), sigma(yz)); for C4v both lie
        # in the sigma_v class, for C6v they straddle the sigma_v/sigma_d
        # classes, so class-wise pairing does not close. Canonical pick: the
        # triple whose sorted canonical axes are lexicographically smallest
        # (the axis-aligned one; ties with the D2 triple convention).
        mirrors_all = [e for e in elems if classify(e)[0] == "sigma"]
        c2_all = [e for e in elems
                  if classify(e)[0] == "rotation" and classify(e)[1] == 2]
        best = None
        for r in c2_all:
            for m0 in mirrors_all:
                m1 = r @ m0
                if not any(np.max(np.abs(m1 - e)) < 1e-8 for e in mirrors_all):
                    continue
                if np.max(np.abs(m1 - m0)) < 1e-8:
                    continue
                if np.max(np.abs(m0 @ m1 - r)) > 1e-8:
                    continue
                if np.max(np.abs(m1 @ m0 - r)) > 1e-8:
                    continue
                axes = tuple(sorted(axis_key(classify(x)[3])
                                    for x in (r, m0, m1)))
                if best is None or axes < best[0]:
                    best = (axes, (r, m0, m1))
        assert best is not None, f"{group}: no C2v subgroup"
        r, m0, m1 = best[1]
        # Table column order (E, C2, sigma_y, sigma_x): the plane through the
        # C2 axis carries the smaller canonical axis first.
        if axis_key(classify(m1)[3]) < axis_key(classify(m0)[3]):
            m0, m1 = m1, m0
        h = [I3, r, m0, m1]
    elif target == "C2h":
        r = rep_of_class(rot2[0])
        inv = rep_of_class(inversion[0])
        h = [I3, r, inv, inv @ r]
    elif target in ("D2", "D2h"):
        # The canonical triple of pairwise-commuting order-2 rotations: the
        # triple whose sorted canonicalized axes are lexicographically
        # smallest; the axes then order the slots (z, y, x).
        order2 = [e for e in elems
                  if classify(e)[0] == "rotation" and classify(e)[1] == 2]
        best = None
        for a in order2:
            for b in order2:
                if np.max(np.abs(a - b)) < 1e-8:
                    continue
                c = a @ b
                if np.max(np.abs(c @ c - I3)) > 1e-8 or classify(c)[0] != "rotation":
                    continue
                if np.max(np.abs(b @ a - c)) > 1e-8:
                    continue
                axes = sorted((axis_key(canonical_axis(axis_of(x, 1.0))) for x in (a, b, c)))
                if best is None or axes < best[0]:
                    best = (axes, (a, b, c))
        assert best is not None, f"{group}: no D2 triple"
        triple = sorted(best[1], key=lambda x: axis_key(canonical_axis(axis_of(x, 1.0))))
        h = [I3] + list(triple)
        if target == "D2h":
            inv = rep_of_class(inversion[0])
            h = [I3] + list(triple) + [inv] + [inv @ r for r in triple]
    else:
        raise AssertionError(f"no H realization for {group} -> {target}")

    # Verify: subset of G, closed, Abelian, correct order.
    for g in h:
        assert any(np.max(np.abs(g - e)) < 1e-8 for e in elems), f"{group}: H not subset"
    for a in h:
        for b in h:
            p = a @ b
            assert any(np.max(np.abs(p - e)) < 1e-8 for e in h), f"{group}: H not closed"
            assert np.max(np.abs(a @ b - b @ a)) < 1e-8, f"{group}: H not Abelian"
    expected = {"C1": 1, "Cs": 2, "Ci": 2, "C2": 2, "C2v": 4, "C2h": 4,
                "D2": 4, "D2h": 8}[target]
    assert len(h) == expected, f"{group}: |H| = {len(h)} != {expected}"
    return target, h


def correlation(elems, class_of, dims, chars, target, h_elems):
    """corr[Gamma][alpha]: n_alpha = (1/|H|) sum_h chi_G(class(h)) chi_alpha(h)."""
    n = len(elems)
    h_idx = [next(i for i in range(n) if np.max(np.abs(g - elems[i])) < 1e-8)
             for g in h_elems]
    h_class = [class_of[i] for i in h_idx]
    # H's own characters via the same class-sum machinery (H is Abelian, so
    # every element is its own class).
    h_dims, h_chars, _ = characters(h_elems, list(range(len(h_elems))),
                                    h_elems, [1] * len(h_elems), f"H({target})")
    h_table = ABELIAN_TABLES[target]
    n_h = len(h_elems)
    h_row_order = []
    used = set()
    for label, row in h_table.items():
        assert len(row) == n_h
        match = [a for a in range(len(h_dims))
                 if h_dims[a] == 1 and a not in used and
                 all(abs(h_chars[a][k] - row[k]) < 1e-6 for k in range(n_h))]
        assert len(match) == 1, f"H({target}): row {label} matched {len(match)}"
        h_row_order.append(match[0])
        used.add(match[0])
    assert len(used) == len(h_dims), f"H({target}): rows not covered"

    corr = np.zeros((len(dims), n_h))
    for a in range(len(dims)):
        for alpha, hrow in enumerate(h_row_order):
            s = sum(h_chars[hrow][k] * chars[a][h_class[k]] for k in range(n_h))
            corr[a, alpha] = round(s / n_h)
    for a in range(len(dims)):
        assert abs(sum(corr[a]) - dims[a]) < 1e-6, \
            f"corr dim mismatch {target}: {sum(corr[a])} vs {dims[a]}"
        for alpha in range(n_h):
            assert abs(corr[a, alpha] - round(corr[a, alpha])) < 1e-6
    return corr.astype(int)


# ---------------------------------------------------------------------------
# Group definitions and the full pipeline
# ---------------------------------------------------------------------------

def group_generators(group):
    """Hand-specified generators, standard orientation (principal axis z)."""
    # Named groups first: the endswith("v"/"h"/"d") patterns below would
    # otherwise swallow Td/Th/Oh/Ih (and "T" parses as n = 0 in the C/D/S
    # convention).
    named = {
        "C1": [],
        "Ci": [INV],
        "Cs": [SIGMA_H],
        "S4": [s2n_z(2)],
        "S6": [s2n_z(3)],
        "S8": [s2n_z(4)],
        "T": [rot((1.0, 1.0, 1.0), 2.0 * np.pi / 3.0), C2_X],
        "Td": [rot((1.0, 1.0, 1.0), 2.0 * np.pi / 3.0), s2n_z(2)],
        "Th": [rot((1.0, 1.0, 1.0), 2.0 * np.pi / 3.0), C2_X, INV],
        "O": [rot((1.0, 1.0, 1.0), 2.0 * np.pi / 3.0), rot_z(np.pi / 2.0)],
        "Oh": [rot((1.0, 1.0, 1.0), 2.0 * np.pi / 3.0),
               rot_z(np.pi / 2.0), INV],
        "I": [rot((0.0, 1.0, PHI), 2.0 * np.pi / 5.0),
              rot((0.0, 1.0 + 2.0 * PHI, PHI), 2.0 * np.pi / 3.0)],
        "Ih": [rot((0.0, 1.0, PHI), 2.0 * np.pi / 5.0),
               rot((0.0, 1.0 + 2.0 * PHI, PHI), 2.0 * np.pi / 3.0), INV],
    }
    if group in named:
        return named[group]
    if group[1:].isdigit():
        n = int(group[1:])
    elif group.endswith(("v", "h", "d")):
        n = int(group[1:-1])
    else:
        raise AssertionError(f"no generators for {group}")
    if group[0] == "C" and "v" not in group and "h" not in group:
        return [rot_z(2.0 * np.pi / n)]
    if group.endswith("v"):
        return [rot_z(2.0 * np.pi / n), SIGMA_XZ]
    if group.endswith("h") and group[0] == "C":
        return [rot_z(2.0 * np.pi / n), SIGMA_H]
    if group[0] == "D" and "h" not in group and "d" not in group:
        return [rot_z(2.0 * np.pi / n), C2_X]
    if group.endswith("h"):
        return [rot_z(2.0 * np.pi / n), C2_X, SIGMA_H]
    if group.endswith("d"):
        if n == 2:
            return [s2n_z(2), C2_X]
        sigma_d = mirror((np.sin(np.pi / (2.0 * n)),
                          -np.cos(np.pi / (2.0 * n)), 0.0))
        return [rot_z(2.0 * np.pi / n), C2_X, sigma_d]
    raise AssertionError(f"no generators for {group}")


def expected_order(group):
    named = {"C1": 1, "Ci": 2, "Cs": 2, "S4": 4, "S6": 6, "S8": 8,
             "T": 12, "Td": 24, "Th": 24, "O": 24, "Oh": 48,
             "I": 60, "Ih": 120}
    if group in named:
        return named[group]
    if group[0] == "C" and "v" not in group and "h" not in group:
        return int(group[1:])
    if group.endswith("v") or group.endswith("h"):
        n = int(group[1:-1])
        return 2 * n if group[0] == "C" else (4 * n if "h" in group else 2 * n)
    if group.endswith("d"):
        return 4 * int(group[1:-1])
    if group.startswith("D"):
        return 2 * int(group[1:])
    raise AssertionError(f"no order for {group}")


def build_group(group):
    """Full pipeline for one group; returns the emission data."""
    generators = group_generators(group)
    elems = closure(generators, expected_order(group), group)
    raw = conjugacy_classes(elems)

    # Descriptors in raw class order, then slots, then the table sort.
    desc_raw = []
    for cls in raw:
        kinds = {classify(elems[i])[0] for i in cls}
        orders = {classify(elems[i])[1] for i in cls}
        assert len(kinds) == 1 and len(orders) == 1, \
            f"{group}: mixed class {cls}"
        # The class key power: the class-minimal member power (min j over
        # the conjugacy class). A class mixes conjugate powers by
        # construction (C3 with C3^2, C4 with C4^3, S10 with S10^9), so the
        # rep's own power is discovery-order dependent (D5d S10 classes
        # emitted (10,3),(10,9); D4d (8,3),(8,7); Ih (10,7),(10,9)) and would
        # not match the key the runtime recomputes from the molecule-frame
        # elements. The class-min power is the canonical key.
        kind, order, _rep_j, axis = classify(elems[cls[0]])
        j = min(classify(elems[i])[2] for i in cls)
        axes = [canonical_axis(axis) if axis is not None else np.zeros(3)
                for axis in [classify(elems[i])[3] for i in cls]]
        min_axis = min(axes, key=axis_key)
        desc_raw.append([kind, order, j, None, min_axis])
    desc = assign_slots(desc_raw)
    order_perm = sorted(range(len(raw)),
                        key=lambda c: (KIND_PRIORITY[desc[c][0]], desc[c][1], desc[c][2],
                                       SLOT_PRIORITY[desc[c][3]], axis_key(desc[c][4])))
    class_infos = [desc[c] for c in order_perm]
    raw_sorted = [raw[c] for c in order_perm]
    class_reps = [elems[cls[0]] for cls in raw_sorted]
    cls_of = class_of_element(elems, raw_sorted)
    class_sizes = [sum(1 for k in cls_of if k == c) for c in range(len(class_infos))]
    assert sum(class_sizes) == len(elems)

    # Class maps induced by sigma_h / inversion multiplication. A product may
    # fall outside the group (D5h: sigma_h * S10 = a C10 rotation); the odd-Dnh
    # and Dnd E-row formulas use the direct S-class e2, so a missing entry
    # falls back to the class itself (rot_part = identity).
    maps = {}
    for name, g in (("sigma_h", SIGMA_H), ("i", INV)):
        if any(np.max(np.abs(g - e)) < 1e-8 for e in elems):
            m = {}
            for c in range(len(class_infos)):
                p = g @ class_reps[c]
                e = next((e for e in range(len(elems))
                          if np.max(np.abs(p - elems[e])) < 1e-8), None)
                if e is not None:
                    m[c] = cls_of[e]
            maps[name] = m

    dims, chars, complex_rows = characters(elems, cls_of, class_reps, class_sizes,
                                           group)

    specs = make_specs(group, class_infos, class_reps, maps, elems, raw_sorted)
    irrep_assignment = {}
    for label, (dim, values) in specs.items():
        matches = []
        for a in range(len(dims)):
            if dims[a] != dim:
                continue
            if all(abs(chars[a][next(i for i in range(len(class_infos))
                                      if key_of(class_infos[i]) == k)] - v) < 1e-6
                   for k, v in values.items()):
                matches.append(a)
        assert len(matches) == 1, f"{group}: spec {label} matched {len(matches)}"
        irrep_assignment[matches[0]] = label
    assert set(irrep_assignment) == set(range(len(dims))), f"{group}: unmatched irreps"

    target, h_elems = realize_h(group, elems, raw_sorted, class_infos)
    corr = correlation(elems, cls_of, dims, chars, target, h_elems)
    assert corr.shape[0] == len(dims) and corr.shape[1] <= 8

    # Canonical emission order: rows sorted by their (lexicographic) label,
    # matching the Abelian tables' row order (A1 < A2 < B1 < B2, ...).
    labels = [irrep_assignment[a] for a in range(len(dims))]
    perm = sorted(range(len(labels)), key=lambda a: labels[a])
    labels = [labels[a] for a in perm]
    dims = [dims[a] for a in perm]
    chars = [chars[a] for a in perm]
    corr = corr[perm, :]
    return {
        "group": group, "target": target, "order": len(elems),
        "labels": labels, "dims": dims, "chars": chars,
        "classes": class_infos, "class_sizes": class_sizes,
        "corr": corr,
    }


# ---------------------------------------------------------------------------
# C++ emission
# ---------------------------------------------------------------------------

KIND_CXX = {"identity": "OperationKind::kIdentity", "rotation": "OperationKind::kRotation",
            "sigma": "OperationKind::kSigma", "inversion": "OperationKind::kInversion",
            "improper": "OperationKind::kImproper"}
SLOT_CXX = {"N": "ClassAxis::kNone", "P": "ClassAxis::kPrincipal",
            "O0": "ClassAxis::kOrbit0", "O1": "ClassAxis::kOrbit1",
            "O2": "ClassAxis::kOrbit2"}
GROUP_CXX = {"C1": "kC1", "Ci": "kCi", "Cs": "kCs", "C2": "kC2", "C3": "kC3",
             "C4": "kC4", "C5": "kC5", "C6": "kC6", "C7": "kC7", "C8": "kC8",
             "C2v": "kC2v", "C3v": "kC3v", "C4v": "kC4v", "C5v": "kC5v",
             "C6v": "kC6v", "C7v": "kC7v", "C8v": "kC8v",
             "C2h": "kC2h", "C3h": "kC3h", "C4h": "kC4h", "C5h": "kC5h",
             "C6h": "kC6h", "C7h": "kC7h", "C8h": "kC8h",
             "D2": "kD2", "D3": "kD3", "D4": "kD4", "D5": "kD5", "D6": "kD6",
             "D7": "kD7", "D8": "kD8",
             "D2h": "kD2h", "D3h": "kD3h", "D4h": "kD4h", "D5h": "kD5h",
             "D6h": "kD6h", "D7h": "kD7h", "D8h": "kD8h",
             "D2d": "kD2d", "D3d": "kD3d", "D4d": "kD4d", "D5d": "kD5d",
             "D6d": "kD6d", "D7d": "kD7d", "D8d": "kD8d",
             "S4": "kS4", "S6": "kS6", "S8": "kS8",
             "T": "kT", "Td": "kTd", "Th": "kTh", "O": "kO", "Oh": "kOh",
             "I": "kI", "Ih": "kIh"}

# Emission order = PointGroupName declaration order.
EMISSION_ORDER = ["C1", "Ci", "Cs", "C2", "C3", "C4", "C5", "C6", "C7", "C8",
                  "C2v", "C3v", "C4v", "C5v", "C6v", "C7v", "C8v",
                  "C2h", "C3h", "C4h", "C5h", "C6h", "C7h", "C8h",
                  "D2", "D3", "D4", "D5", "D6", "D7", "D8",
                  "D2h", "D3h", "D4h", "D5h", "D6h", "D7h", "D8h",
                  "D2d", "D3d", "D4d", "D5d", "D6d", "D7d", "D8d",
                  "S4", "S6", "S8", "T", "Td", "Th", "O", "Oh", "I", "Ih"]


def fmt_char(x):
    return f"{x:.12f}"


def pad(xs, width, zero="0"):
    out = ", ".join(str(x) for x in xs)
    if len(xs) < width:
        out += ", " + ", ".join(zero for _ in range(width - len(xs)))
    return out


def pad_labels(labels, width):
    return pad([f'"{label}"' for label in labels], width, zero='""')


def emit_header(tables):
    lines = []
    a = lines.append
    a("#pragma once")
    a("")
    a("// Generated by tools/gen_full_group_tables.py - do not edit by hand.")
    a("// Regeneration check: python tools/gen_full_group_tables.py --check")
    a("//")
    a("// Full-group (non-Abelian) character tables with irrep dimensions and")
    a("// full-group -> Abelian correlation data: the class keys, characters,")
    a("// and subduction the a-posteriori labeling stage needs after an SCF on")
    a("// the Abelian reduction. The 8 Abelian tables stay in")
    a("// character_tables.hpp; this header extends their shape with dimensions,")
    a("// class sizes, and the class descriptors the runtime stage recomputes")
    a("// from the molecule-frame elements (kind, order, power j, axis slot),")
    a("// with j the class-minimal member power and the slots assigned on the")
    a("// normalized member-axis minimum. C-inf-v and D-inf-h have no finite")
    a("// character table and are NOT emitted (the runtime labels them via")
    a("// |lambda| projection instead).")
    a("")
    a('#include "qcx/symmetry/character_tables.hpp"')
    a('#include "qcx/symmetry/point_group_name.hpp"')
    a("")
    a("#include <array>")
    a("#include <string_view>")
    a("")
    a("namespace qcx::symmetry {")
    a("")
    a("/// Axis slot of a conjugacy class in the canonical orientation.")
    a("/// kPrincipal = the class of the maximal-order proper rotation (ties:")
    a("/// the class-minimal power first, then the lexicographically smallest")
    a("/// canonical axis), plus the sigma/improper classes on that axis;")
    a("/// kOrbitN = the N-th orbit among same-(kind, order, power) classes")
    a("/// ranked by lexicographically smallest member axis; kNone for")
    a("/// identity/inversion.")
    a("enum class ClassAxis { kNone, kPrincipal, kOrbit0, kOrbit1, kOrbit2 };")
    a("")
    a("/// One conjugacy class of the full group: the (kind, order, power, axis)")
    a("/// key the runtime stage recomputes from the molecule-frame elements.")
    a("/// power j discriminates same-(kind, order) classes with different")
    a("/// angles (C5 j=1 vs C5^2 j=2; S10 j=1 vs S10^3 j=3).")
    a("struct OperationClass {")
    a("    OperationKind kind; ///< What the operation is.")
    a("    int order; ///< n of Cn/Sn (1 for identity, inversion, mirrors).")
    a("    int power; ///< j: minimal-angle power (see the header comment).")
    a("    ClassAxis axis; ///< Axis slot in the canonical orientation.")
    a("};")
    a("")
    a("/// One full-group table: irreps with dimensions, characters")
    a("/// [irrep][class], class descriptors and sizes, and the subduction to")
    a("/// the realizable Abelian subgroup (abelian) whose rows are the")
    a("/// existing 8 Abelian tables' rows in their order - so the correlation")
    a("/// columns index the runtime Abelian block order directly.")
    a("struct FullGroupTable {")
    a("    PointGroupName group; ///< The full group itself.")
    a("    PointGroup abelian; ///< Realizable Abelian subgroup (Td -> D2).")
    a("    int order; ///< |G|.")
    a("    int irrepCount; ///< Number of irreps.")
    a("    int classCount; ///< Number of conjugacy classes.")
    a("    std::array<std::string_view, 16> irrepLabels; ///< Mulliken labels.")
    a("    std::array<int, 16> dimensions; ///< Real irrep dimensions.")
    a("    std::array<std::array<double, 16>, 16> characters; ///< [irrep][class].")
    a("    std::array<OperationClass, 16> classes; ///< Class descriptors.")
    a("    std::array<int, 16> classSizes; ///< |C| per class.")
    a("    std::array<std::array<int, 8>, 16> correlation; ///< [irrep][abelian irrep].")
    a("};")
    a("")
    for group in EMISSION_ORDER:
        t = tables[group]
        # The table name drops the leading 'k' of GROUP_CXX (kC1 -> C1) so
        # the kFull prefix reads kFullC1Table, not kFullkC1Table.
        a(f"inline constexpr FullGroupTable kFull{GROUP_CXX[group][1:]}Table{{")
        a(f"    PointGroupName::{GROUP_CXX[group]},")
        a(f"    PointGroup::{CXX_ABELIAN[t['target']]},")
        a(f"    {t['order']},")
        a(f"    {len(t['labels'])}, {len(t['classes'])}, {{ {pad_labels(t['labels'], 16)} }},")
        a(f"    {{ {pad(t['dims'], 16)} }},")
        # std::array is a struct with one member: brace-enclosed elements
        # would consume the whole member (C2078), so the row lists need the
        # double-brace form, as in character_tables.hpp.
        a("    {{")
        for row in t["chars"]:
            a(f"        {{ {pad([fmt_char(v) for v in row], 16)} }},")
        a("    }},")
        a("    {{")
        for c in t["classes"]:
            a(f"        OperationClass{{{KIND_CXX[c[0]]}, {c[1]}, {c[2]}, "
              f"{SLOT_CXX[c[3]]}}},")
        a("    }},")
        a(f"    {{ {pad(t['class_sizes'], 16)} }},")
        a("    {{")
        for row in t["corr"]:
            a(f"        {{ {pad([int(v) for v in row], 8)} }},")
        a("    }},")
        a("};")
        a("")
    a("/// The full-group table for a detected group; nullptr for C-inf-v /")
    a("/// D-inf-h (no finite character table - the linear-molecule lambda")
    a("/// path in the labeling stage handles those).")
    a("constexpr const FullGroupTable* FullGroupTableFor(PointGroupName group) noexcept {")
    a("    switch (group)")
    a("    {")
    for group in EMISSION_ORDER:
        a(f"    case PointGroupName::{GROUP_CXX[group]}:")
        a(f"        return &kFull{GROUP_CXX[group][1:]}Table;")
    a("    case PointGroupName::kCInfV:")
    a("    case PointGroupName::kDInfH:")
    a("        return nullptr;")
    a("    }")
    a("")
    a("    return nullptr; // Unreachable; silences -Wreturn-type.")
    a("}")
    a("")
    a("} // namespace qcx::symmetry")
    a("")
    return "\n".join(lines)


def clang_format(path):
    """Run clang-format on the emitted header (local-only regeneration tool).

    Repository C++ must be clang-format-clean, so the generated header must be
    emitted in the repo's canonical format.
    """
    import shutil
    import subprocess

    exe = shutil.which("clang-format")
    if exe is None:
        vs = (r"C:\Program Files\Microsoft Visual Studio\18\Community" +
              r"\VC\Tools\Llvm\x64\bin\clang-format.exe")
        exe = vs if os.path.exists(vs) else None
    if exe is not None:
        subprocess.run([exe, "-i", path], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the committed header matches a fresh generation")
    args = parser.parse_args()

    tables = {}
    for group in EMISSION_ORDER:
        t = build_group(group)
        tables[group] = t
        print(f"{group:4s} |G|={t['order']:3d} |C|={len(t['classes']):2d} "
              f"irreps={len(t['labels']):2d} target={t['target']:3s} "
              f"labels={','.join(t['labels'])}")

    header = emit_header(tables) + "\n"
    out_path = "symmetry/include/qcx/symmetry/full_group_tables.hpp"
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(header)
    clang_format(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        formatted = f.read()
    if args.check:
        with open(out_path, "r", encoding="utf-8") as f:
            committed = f.read()
        if committed != formatted:
            print("MISMATCH: the committed header differs from a fresh generation.")
            sys.exit(1)
        print("OK: committed header matches a fresh generation.")
    else:
        os.replace(tmp_path, out_path)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
