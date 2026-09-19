# Vendored basis-set corpus

Basis data downloaded from the **Basis Set Exchange (BSE)** at
https://www.basissetexchange.org on 2026-08-16 using
`basis-set-exchange==0.12` (pinned in `tools/requirements-basis.txt`, fetched
by `tools/fetch_basis_data.py`). One NWChem-format file per element per
family; `manifest.json` carries per-file sha256 sums so the corpus is
verifiable offline (`python tools/fetch_basis_data.py --check`).

## Families

| Directory | BSE name |
|---|---|
| sto-3g | sto-3g |
| sto-6g | sto-6g |
| 3-21g | 3-21g |
| 6-31g-star | 6-31G* |
| 6-31g-dstar | 6-31G** |
| 6-311g-dstar | 6-311G** |
| def2-svp | def2-SVP |
| def2-tzvp | def2-TZVP |
| def2-qzvp | def2-QZVP |
| cc-pvdz | cc-pVDZ |
| cc-pvtz | cc-pVTZ |
| aug-cc-pvdz | aug-cc-pVDZ |
| pcseg-1 | pcseg-1 |
| def2-universal-jfit | def2-universal-jfit |
| def2-universal-jkfit | def2-universal-jkfit |
| cc-pvdz-rifit | cc-pVDZ-RIFIT |
| cc-pvtz-rifit | cc-pVTZ-RIFIT |
| aug-cc-pvdz-rifit | aug-cc-pVDZ-RIFIT |
| aug-cc-pvtz | aug-cc-pVTZ |
| aug-cc-pvtz-rifit | aug-cc-pVTZ-RIFIT |
| def2-tzvpd | def2-TZVPD |
| def2-tzvpd-rifit | def2-TZVPD-RIFIT |
| def2-tzvppd | def2-TZVPPD |
| def2-qzvppd-rifit | def2-QZVPPD-RIFIT |
| cc-pvtz-jkfit | cc-pVTZ-JKFIT |

The five auxiliary families were added 2026-08-16 for the RI-J/
RIJCOSX work. BSE 0.12 names the universal Weigend
J/JK sets `def2-universal-jfit`/`def2-universal-jkfit` - the sets commonly
called def2/J and def2/JK (the names `def2/J`/`def2/JK` themselves are not
in BSE 0.12's database).

The def2-QZVP family was added 2026-08-28 (the symmetry
benchmark at benzene/def2-QZVP, 522 AO) and completed to the full
H-Rn element range in the same session; `tools/fetch_basis_data.py`
FAMILIES carries it for future refreshes.

The six families in the last block above were added 2026-09-13,
each against a named gap in
`integrals/include/qcx/integrals/aux_basis.hpp` rather than as a mirror of BSE:

- **aug-cc-pvtz + aug-cc-pvtz-rifit** - the aug-cc family stopped at aug-cc-pVDZ,
  so a diffuse calculation could not go up a rung at all. The `-rifit` is the
  matched J fit the tier-1b arm already computes from the orbital name.
- **def2-tzvpd + def2-tzvpd-rifit** - the diffuse def2 case the three-tier rule
  could not close: it was handed the NON-diffuse `def2-universal-jfit` with no
  better option to offer. `def2-TZVPD-RIFIT` is the matched diffuse-capable fit;
  measured against BSE 0.12 it is **92 functions on oxygen against
  `def2-TZVP-RIFIT`'s 76**, the difference being its trailing `s1 p1 d1 f1`
  diffuse block, which is the property the diffuse region needs.
- **def2-qzvppd-rifit** - the largest vendored diffuse-capable fit, and the
  member `SelectAuxBasis`'s tier 3 was previously a label without. It serves the
  diffuse def2 spellings with no vendored match (def2-svpd, def2-tzvppd,
  def2-qzvpd), which were otherwise handed a non-diffuse fit.
- **def2-tzvppd** (orbital) - vendored WITHOUT its own `def2-TZVPPD-RIFIT`, and
  that omission is the point: it is the bundled diffuse basis with no matched
  fit, so it is the case that makes tier 3 reachable in a real run rather than
  an arm only a name test ever visits. A `def2-tzvppd` RI-J run resolves to
  `def2-qzvppd-rifit` - a larger diffuse-capable fit - instead of the
  non-diffuse universal one. A `def2-TZVPPD-RIFIT` could be vendored later to
  move this name up to tier 1; it is left out because the tier-3 member already
  covers it and the corpus is curated, not a mirror.
- **cc-pvtz-jkfit** - the cc family's JK fit, and the RI-JK/RI-K path's tier 1
  for that family. The cc family previously had only RI-J `-rifit` sets, so every
  `cc-*` JK request fell through to the universal `def2-universal-jkfit`.

**Sets searched for and ABSENT from BSE 0.12, so deliberately NOT vendored**
(a name was not invented to fill the hole): **`cc-pVDZ-JKFIT` does not exist under
any name**, and no `aug-cc-pV*Z-JKFIT` exists either. The JKFIT family in BSE 0.12
is exactly `cc-pVTZ-JKFIT`, `cc-pVQZ-JKFIT`, `cc-pV5Z-JKFIT`,
`def2-SV(P)-JKFIT`, `def2-universal-JKFIT` and the SARC2 sets (relativistic, out
of this corpus's families) - enumerated by name from `bse.get_all_basis_names()`,
776 names total. Two consequences the code states rather than hides: a `cc-pvdz`
JK request resolves to `cc-pvtz-jkfit`, the smallest vendored rung at or above its
own cardinal, and **no diffuse-capable JK fit is vendored at all**, so the diffuse
def2 JK path keeps the universal JK fit and the notice reports it.

The fetch was run with the pinned `basis-set-exchange==0.12` and the manifest
diff was checked: the six families were ADDED and **no existing family's file
changed** (0 changed of 18), which is the reproducibility check that says the
corpus is still the pinned version's output. `python tools/fetch_basis_data.py
--check` passes.

Element coverage varies by family (a family lacking an element simply has no
file for it); def2 entries for heavy elements include ECP sections. The
def2-SVP and def2-TZVP Xe/Rn blocks are byte-identical upstream (verified
against BSE 0.12) - not a corpus error.

**Not** in that byte-identical class, recorded because it was checked and the
guess was wrong: `def2-TZVPD-RIFIT` and `def2-TZVPPD-RIFIT` share a composition
at Z = 6/8/36 but **differ at Z = 1 (H) and Z = 16 (S)**, so they are distinct
upstream sets. Only the TZVPD one is vendored; the rule serves `def2-tzvppd` from
the tier-3 member, not from an assumed alias of the TZVPD fit.

## Provenance and license

- BSE data is distributed under the **BSD-3-Clause license**; the BSE
  citation is the `Pritchard2019` entry in `CITATION.bib`.
- Individual basis families carry their own original references (STO-3G:
  Hehre/Stewart/Pople; Pople 3-21G/6-31G*/6-311G**; def2: Weigend/Ahlrichs;
  cc-pVnZ: Dunning; aug-cc-pVDZ: Kendall/Dunning/Harrison; pcseg: Jensen;
  RI auxiliaries: Weigend universal J/JK-fit sets, Weigend cc-*RIFIT sets).
  The per-basis reference metadata lives in the BSE database and is
  reproduced on the BSE website per family; the fetched files are the data,
  unmodified apart from the wrapping header/END.
