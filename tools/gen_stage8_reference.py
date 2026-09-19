#!/usr/bin/env python3
"""Generate the disk-integral engine reference grids: the
pyscf/libcint ERI classes of H2O/cc-pVDZ that the storage tests persist and
serve (tools/gen_stage8_reference.py -> storage/tests/data). The referee
is PySCF [Pyscf2018] - an independently
developed engine, no shared code with the C++ pipeline.

Each committed CSV storage/tests/data/h2o_ccpvdz_class_{lBra}_{lKet}.csv
holds one class of the canonical quartets of the H2O/cc-pVDZ shell set
(classes (0,0), (0,1), (0,2), (1,1)). A row is `eri,i,j,k,l,f,value` with
(i,j,k,l) the canonical quartet (i <= j, k <= l, pair (i,j) >= (k,l),
L_bra <= L_ket), f the EriBlockIndex flat offset of the value in the
quartet's block, and value the libcint double. Rows are quartet-major in
the engine's canonical order (lKet, lBra, ketPair, braRowPairs, braPair)
with f ascending inside each block, so the CSV lines up 1:1 with the
store's served order.

The pyscf molecule is the fake-element-per-shell construction: one fake
element per shell at the shell's center, the NWChem file's contraction
columns kept as one general-contraction shell (libcint and the qcx parser
agree on the shell structure; both normalize contracted functions to unit
norm identically), spherical (cart=False), unit Bohr. The l = 1 row
mapping is the established sweep convention (tools/eri_pyscf_sweep.py,
validated at 4.4e-15): the qcx m = -1,0,+1 order is (y,z,x), libcint's is
(py,pz,px) - qcx row g maps to pyscf row [1,2,0][g]; the l >= 2 rows are
identical.

Requires pyscf (runs in WSL - pyscf is not installed on Windows). The
--check mode (the QCX_REFERENCE_REGENERATION_CHECK ctest, default OFF)
regenerates fresh and compares byte-for-byte with the committed CSVs, and
prints SKIP and exits 77 when pyscf is missing - the established
regeneration-check convention (gen_md_reference.py).
"""
import argparse
import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# The H2O/cc-pVDZ corpus, parsed with the qcx parser semantics
# ---------------------------------------------------------------------------

ANGULAR_LABELS = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}

# The qcx m = -1,0,+1 = (y,z,x) -> libcint (py,pz,px) row map (l = 1 only).
MAP1 = [1, 2, 0]

# H2O at the experimental geometry (same as the MakeH2oCcpvdz fixture).
O_CENTER = (0.0, 0.0, 0.0)
H_D = 1.430428808474167
H_H = 1.107157044080814


def parse_nwchem(path):
    """The NWChem element file -> [(l, [(exp, [coeffs...]), ...]), ...] with
    the qcx parser semantics: one shell per angular block, contractionCount
    = the block's column count."""
    shells = []
    current = None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1] in ANGULAR_LABELS:
                current = []
                shells.append((ANGULAR_LABELS[parts[1]], current))
                continue
            if not parts:
                continue
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue
            if current is not None and len(values) >= 2:
                current.append((values[0], values[1:]))
    return shells


def shell_count(shell):
    """The (2l+1) x contractionCount functions of one shell."""
    l, prims = shell
    return (2 * l + 1) * (1 if not prims else len(prims[0][1]))


# ---------------------------------------------------------------------------
# The pyscf molecule
# ---------------------------------------------------------------------------


def build_pyscf(shells_by_atom, centers):
    """One fake element per shell; returns (mol, per-shell function offsets)
    with the shells in the qcx order (atoms canonical, element shells in
    file order)."""
    from pyscf import gto

    atoms = []
    basis = {}
    offsets = []
    count = 0
    for ia, (shells, center) in enumerate(zip(shells_by_atom, centers)):
        for l, prims in shells:
            sym = "X%d_%d" % (ia, l)
            atoms.append("%s %.10f %.10f %.10f" % (sym, center[0], center[1], center[2]))
            basis[sym] = [[l] + [[exp] + list(cs) for exp, cs in prims]]
            offsets.append(count)
            count += shell_count((l, prims))
    mol = gto.M(atom=atoms, basis=basis, cart=False, unit="Bohr", verbose=0)
    mol.build()
    return mol, offsets


def pyscf_index(g, l, nctr):
    """The libcint function index of the qcx function index g within a
    shell (both contraction-major then m ascending; only the l = 1 rows
    differ: qcx (y,z,x) -> libcint (py,pz,px))."""
    n_ang = 2 * l + 1
    row = g // n_ang
    m = g % n_ang
    return row * n_ang + (MAP1[m] if l == 1 else m)


def pair_index(i, j, n):
    """The upper-triangle pair index (PairIndexOf of the C++ pair list)."""
    return i * (2 * n - i + 1) // 2 + (j - i)


def eri_block_index(fi, fj, fk, fl, n_i, n_j, n_k, n_l):
    """The EriBlockIndex flat offset: rows (fj*n_i+fi), cols (fl*n_k+fk)."""
    return (fj * n_i + fi) * (n_k * n_l) + (fl * n_k + fk)


def build_rows(int2e, offsets, shell_l, shell_ctr, class_key):
    """The (tag, i, j, k, l, f, value) rows of one class, in the engine's
    canonical quartet order with f ascending inside each block. Returns
    (rows, quartet_count)."""
    l_bra, l_ket = class_key
    n = len(shell_l)
    quartets = []
    for i in range(n):
        for j in range(i, n):
            for k in range(n):
                for l in range(k, n):
                    if pair_index(i, j, n) < pair_index(k, l, n):
                        continue
                    if shell_l[i] + shell_l[j] != l_bra or shell_l[k] + shell_l[l] != l_ket:
                        continue
                    quartets.append((i, j, k, l))

    def canon_key(q):
        i, j, k, l = q
        return (shell_l[k] + shell_l[l], shell_l[i] + shell_l[j],
                pair_index(k, l, n), shell_ctr[i] * shell_ctr[j],
                pair_index(i, j, n))

    quartets.sort(key=canon_key)
    rows = []
    for i, j, k, l in quartets:
        n_i = (2 * shell_l[i] + 1) * shell_ctr[i]
        n_j = (2 * shell_l[j] + 1) * shell_ctr[j]
        n_k = (2 * shell_l[k] + 1) * shell_ctr[k]
        n_l = (2 * shell_l[l] + 1) * shell_ctr[l]
        oi, oj, ok, ol = offsets[i], offsets[j], offsets[k], offsets[l]
        for fi in range(n_i):
            for fj in range(n_j):
                for fk in range(n_k):
                    for fl in range(n_l):
                        value = float(int2e[oi + pyscf_index(fi, shell_l[i], shell_ctr[i]),
                                            oj + pyscf_index(fj, shell_l[j], shell_ctr[j]),
                                            ok + pyscf_index(fk, shell_l[k], shell_ctr[k]),
                                            ol + pyscf_index(fl, shell_l[l], shell_ctr[l])])
                        f = eri_block_index(fi, fj, fk, fl, n_i, n_j, n_k, n_l)
                        rows.append(("eri", i, j, k, l, f, value))
    return rows, len(quartets)


def generate():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    basis_dir = os.path.join(repo_root, "data", "basis", "cc-pvdz")
    o_shells = parse_nwchem(os.path.join(basis_dir, "O.nwchem"))
    h_shells = parse_nwchem(os.path.join(basis_dir, "H.nwchem"))
    # The qcx molecule canonical renumbering (molecule/src/molecule.cpp
    # Create) sorts atoms by (Z, x, y, z): the canonical H2O order is
    # [H(-d,h), H(+d,h), O(0,0,0)] - the pyscf fake-element atom list must
    # mirror it or the shell indices of the CSVs disagree with the C++
    # pair list.
    centers = [(-H_D, H_H, 0.0), (H_D, H_H, 0.0), O_CENTER]
    shells_by_atom = [h_shells, h_shells, o_shells]

    mol, offsets = build_pyscf(shells_by_atom, centers)
    int2e = mol.intor("int2e_sph")

    shell_l = []
    shell_ctr = []
    for shells in shells_by_atom:
        for l, prims in shells:
            shell_l.append(l)
            shell_ctr.append(1 if not prims else len(prims[0][1]))

    grids = {}
    for class_key in [(0, 0), (0, 1), (0, 2), (1, 1)]:
        rows, quartet_count = build_rows(int2e, offsets, shell_l, shell_ctr, class_key)
        name = "h2o_ccpvdz_class_%d_%d.csv" % class_key
        grids[name] = rows
        print("%s: %d quartets, %d elements" % (name, quartet_count, len(rows)))
    return grids


def write_csv(outdir, name, rows):
    path = os.path.join(outdir, name)
    with open(path, "w", newline="") as f:
        f.write("eri,i,j,k,l,f,value\n")
        for row in rows:
            f.write("%s,%d,%d,%d,%d,%d,%.17g\n" % row)
    print("wrote %s (%d rows)" % (path, len(rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="storage/tests/data")
    parser.add_argument("--check", action="store_true",
                        help="regenerate fresh and compare byte-for-byte "
                             "with the committed CSVs (exit 1 on drift, 77 "
                             "without pyscf)")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = args.outdir
    if not os.path.isabs(outdir):
        outdir = os.path.join(repo_root, outdir)

    try:
        import pyscf  # noqa: F401
    except ImportError:
        if args.check:
            print("SKIP: pyscf is not installed (the --check mode requires it)")
            sys.exit(77)
        raise

    grids = generate()

    if args.check:
        ok = True
        with tempfile.TemporaryDirectory() as tmp:
            for name, rows in grids.items():
                fresh = os.path.join(tmp, name)
                write_csv(tmp, name, rows)
                committed = os.path.join(outdir, name)
                if not os.path.exists(committed):
                    print("DRIFT: %s is missing" % committed)
                    ok = False
                    continue
                with open(fresh, "rb") as ff, open(committed, "rb") as fc:
                    if ff.read() != fc.read():
                        print("DRIFT: %s differs from a fresh generation" % committed)
                        ok = False
        return 0 if ok else 1

    os.makedirs(outdir, exist_ok=True)
    for name, rows in grids.items():
        write_csv(outdir, name, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
