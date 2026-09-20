#!/usr/bin/env python3
"""Fetch NWChem-format basis files from the Basis Set Exchange into data/basis/.

Network tool, run once (or to refresh): downloads one file per element per
family and writes data/basis/manifest.json with per-file sha256 sums, so the
committed corpus is verifiable offline. Run with `--check` to verify the
vendored files against the manifest without network access.

Every written file carries a comment credit block above its `BASIS` line: the
exchange's own provenance block for the family, then the published
reference(s) the exchange records for that element's data. Both are read out
of the pinned package rather than written by hand, so a credit cannot drift
from the numbers under it; the NWChem grammar is untouched because `#` starts
a comment line.

Pinned: basis-set-exchange==0.12 (tools/requirements-basis.txt).
Provenance and licensing live in data/basis/VERSIONS.md.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = REPO / "data" / "basis"
ELEMENTS_JSON = HERE / "periodic" / "elements_data.json"

# BSE basis name -> safe directory name ('*' is illegal in Windows paths).
FAMILIES = [
    ("sto-3g", "sto-3g"),
    ("sto-6g", "sto-6g"),
    ("3-21g", "3-21g"),
    ("6-31G*", "6-31g-star"),
    ("6-31G**", "6-31g-dstar"),
    ("6-311G**", "6-311g-dstar"),
    ("def2-SVP", "def2-svp"),
    ("def2-TZVP", "def2-tzvp"),
    ("def2-QZVP", "def2-qzvp"),
    ("cc-pVDZ", "cc-pvdz"),
    ("cc-pVTZ", "cc-pvtz"),
    ("aug-cc-pVDZ", "aug-cc-pvdz"),
    ("pcseg-1", "pcseg-1"),
    # RI auxiliary sets - the RI-J path needs them.
    # BSE 0.12 names the universal Weigend J/JK sets "def2-universal-jfit"/
    # "def2-universal-jkfit" (the sets commonly called def2/J and def2/JK).
    ("def2-universal-jfit", "def2-universal-jfit"),
    ("def2-universal-jkfit", "def2-universal-jkfit"),
    ("cc-pVDZ-RIFIT", "cc-pvdz-rifit"),
    ("cc-pVTZ-RIFIT", "cc-pvtz-rifit"),
    ("aug-cc-pVDZ-RIFIT", "aug-cc-pvdz-rifit"),
    # Diffuse/tier-3 and RI-K additions (2026-09-13).
    # Each one closes a named gap in SelectAuxBasis rather than mirroring BSE:
    # aug-cc-pVTZ is the aug-cc family's next rung (the family stopped at DZ,
    # so a diffuse calculation could not go up a rung at all); its -RIFIT is
    # the matched J fit the tier-1b arm already computes. def2-TZVPD is the
    # exact diffuse case the rule could not close, and def2-TZVPD-RIFIT is its
    # matched diffuse-capable fit (verified to carry the diffuse shell block
    # that def2-TZVP-RIFIT lacks). def2-QZVPPD-RIFIT is the largest vendored
    # diffuse-capable fit and is tier 3's member for the other diffuse def2
    # spellings (def2-SVPD/TZVPPD/QZVPD/QZVPPD) that have no vendored match.
    # cc-pVTZ-JKFIT is the cc family's JK fit - the RI-K/RI-JK path's tier 1.
    # NOT vendored, because BSE 0.12 has no such set under any name:
    # cc-pVDZ-JKFIT and every aug-cc-pV*Z-JKFIT (see VERSIONS.md).
    ("aug-cc-pVTZ", "aug-cc-pvtz"),
    ("aug-cc-pVTZ-RIFIT", "aug-cc-pvtz-rifit"),
    ("def2-TZVPD", "def2-tzvpd"),
    ("def2-TZVPD-RIFIT", "def2-tzvpd-rifit"),
    # The diffuse def2 base with NO vendored matched fit, on purpose: it is the
    # bundled case that makes tier 3 REACHABLE rather than a name-only arm, and
    # the one that shows the tier doing its job (a diffuse basis served by a
    # larger diffuse-capable fit instead of the non-diffuse universal one).
    ("def2-TZVPPD", "def2-tzvppd"),
    ("def2-QZVPPD-RIFIT", "def2-qzvppd-rifit"),
    ("cc-pVTZ-JKFIT", "cc-pvtz-jkfit"),
]

# BSE element coverage varies by family; elements absent from a family are
# skipped (the vendored corpus records what each family actually offers).
ELEMENTS = list(range(1, 119))


def symbol_map() -> dict[int, str]:
    data = json.loads(ELEMENTS_JSON.read_text(encoding="utf-8"))
    return {e["z"]: e["symbol"] for e in data["elements"]}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def credit_block(bseName: str, z: int) -> str:
    """The comment credit block written above one file's `BASIS` line.

    Two facts, both taken from the pinned exchange data: where the numbers
    came from (the exchange's provenance block for the family, which names
    the resource, its version, the basis set and the revision the tables were
    imported from) and the published reference or references the exchange
    records for this element's data. References are per element because the
    exchange records them that way - a family assembled from several papers
    cites the paper behind each element's block.
    """
    import basis_set_exchange as bse
    from basis_set_exchange.references import reference_text

    lines = []

    for line in bse.get_basis(bseName, elements=[str(z)], fmt="nwchem", header=True).splitlines():
        if line.startswith("BASIS"):
            break
        lines.append(line)

    while lines and not lines[-1].strip():
        lines.pop()

    rendered = []
    seen = set()

    for group in bse.get_references(bseName, elements=[str(z)]):
        for info in group["reference_info"]:
            for key, reference in info["reference_data"]:
                if key in seen:
                    continue
                seen.add(key)
                rendered.append(reference_text(key, reference))

    if rendered:
        lines.append("# Published reference(s) for this element's data:")
        for text in rendered:
            for line in text.splitlines():
                lines.append(("# " + line).rstrip())

    return "\n".join(lines) + "\n"


def fetch_family(bseName: str, dirName: str, symbols: dict[int, str]) -> dict[str, str]:
    """Downloads one file per supported element; returns {symbol: sha256}."""
    import basis_set_exchange as bse

    hashes = {}
    skipped = 0
    for z in ELEMENTS:
        symbol = symbols[z]
        try:
            block = bse.get_basis(bseName, elements=[str(z)], fmt="nwchem", header=False)
        except KeyError:
            # Element not covered by this family (BSE raises KeyError for
            # genuinely absent elements).
            skipped += 1
            continue
        if not block:
            skipped += 1
            continue
        # BSE's nwchem writer emits the BASIS header and END despite
        # header=False; strip both and wrap exactly once.
        lines = block.strip().splitlines()
        if lines and lines[0].startswith("BASIS"):
            lines = lines[1:]
        body = "\n".join(lines).strip()
        if not body.endswith("END"):
            body += "\nEND"
        text = credit_block(bseName, z) + f'BASIS "ao basis" SPHERICAL PRINT\n{body}\n'
        directory = OUT / dirName
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{symbol}.nwchem"
        target.write_text(text, encoding="utf-8", newline="\n")
        hashes[symbol] = sha256_of(target)
        print(f"  {bseName}: {symbol} ({len(block.splitlines())} lines)")
    if not hashes:
        # A family that fetches zero files is a wrong BSE name (this is how
        # the bogus def2/J, def2/JK names were caught) - the corpus must
        # never silently record an empty family.
        raise SystemExit(f"ERROR: {bseName}: 0 files fetched - wrong BSE name?")
    print(f"  {bseName}: {len(hashes)} files, {skipped} elements skipped")
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify data/basis against manifest.json (offline)")
    args = parser.parse_args()

    manifest_path = OUT / "manifest.json"
    if args.check:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bad = 0
        for family, hashes in manifest.items():
            for symbol, expected in hashes.items():
                path = OUT / family / f"{symbol}.nwchem"
                if not path.exists() or sha256_of(path) != expected:
                    print(f"MISMATCH: {family}/{symbol}")
                    bad += 1
        if bad:
            print(f"{bad} files differ from the manifest")
            return 1
        print("data/basis matches manifest.json")
        return 0

    symbols = symbol_map()
    manifest = {}
    for bseName, dirName in FAMILIES:
        print(f"fetching {bseName} -> {dirName}")
        manifest[dirName] = fetch_family(bseName, dirName, symbols)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
