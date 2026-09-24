#!/usr/bin/env python3
"""Runs one fixed calculation at several accuracies and records what each run
produced, one file per run.

The fixture is the repository's pinned water/STO-3G RHF input; the only thing
that changes between rungs is the accuracy word in its [method] block, so a
difference between two records is attributable to that word or to nothing.

    sweep.py --qcx <qcx.exe> [--out DIR] [--ladder kLoose,kNormal,kTight]
    sweep.py --qcx <qcx.exe> --control 3      # same rung, three times

--control is the gate: it runs the SAME accuracy repeatedly and compares the
records. Until that comes back identical, a difference seen across the ladder
is not attributable to the accuracy change.

The comparison is over the NUMERIC record only. A run's result JSON also
carries wall-clock and host-throughput fields, which are measurements of the
machine and not of the calculation, so they differ between two runs of the
same input by construction. Both digests are reported: the raw one over the
whole stdout, so a reader can see that the difference is confined to those
fields, and the numeric one, which is what the gate asserts on.
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
FIXTURE = HERE / "fixture-water-sto3g-rhf.toml"

# Result blocks that measure the host rather than the calculation.
TIMING_ROOT = "timings_ms"
TIMING_SECTION = ("resources_resolved", "compute_profile")


def rung_input(fixture: pathlib.Path, accuracy: str, path: pathlib.Path, thread_cap: int = 1) -> None:
    """The fixture with its accuracy word and its thread cap replaced. Errors if
    the fixture no longer carries exactly one of each, rather than writing a rung
    whose accuracy is not the one it is labelled with."""
    text = fixture.read_text()

    for needle in ('accuracy = "', "thread_cap = "):
        count = text.count(needle)

        if count != 1:
            raise SystemExit(f"fixture carries {count} of {needle!r}, expected 1")

    head, tail = text.split('accuracy = "', 1)
    _, rest = tail.split('"', 1)
    text = head + 'accuracy = "' + accuracy + '"' + rest

    head, tail = text.split("thread_cap = ", 1)
    _, rest = tail.split("\n", 1)
    path.write_text(head + f"thread_cap = {thread_cap}" + "\n" + rest)


def numeric_record(result: dict) -> str:
    """The result with its host-measurement blocks removed, canonically
    serialised. Two runs of the same input agree on this and on nothing more."""
    trimmed = json.loads(json.dumps(result))
    trimmed.pop(TIMING_ROOT, None)
    section = trimmed.get(TIMING_SECTION[0])

    if isinstance(section, dict):
        section.pop(TIMING_SECTION[1], None)

    return json.dumps(trimmed, sort_keys=True)


def differing_leaves(runs: list) -> list:
    """Every leaf path, and its values, that is not the same across the runs."""

    def leaves(node, prefix=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaves(value, f"{prefix}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from leaves(value, f"{prefix}/[{index}]")
        else:
            yield prefix, node

    base = dict(leaves(runs[0]))
    paths = []

    for path in sorted(base):
        values = [dict(leaves(run)).get(path) for run in runs[1:]]

        if any(value != base[path] for value in values):
            paths.append((path, [base[path]] + values))

    return paths


def record(qcx: pathlib.Path, toml: pathlib.Path, outdir: pathlib.Path, tag: str) -> dict:
    """One run. stdout is the result JSON, stderr the refusal; both are kept."""
    proc = subprocess.run([str(qcx), "run", str(toml)], capture_output=True, text=True)

    (outdir / f"{tag}.json").write_text(proc.stdout, newline="")
    (outdir / f"{tag}.stderr").write_text(proc.stderr, newline="")

    row = {
        "tag": tag,
        # Read back from the rung file, so the manifest records the accuracy the
        # run was actually given and not the one the caller meant to give it.
        "accuracy": toml.read_text().split('accuracy = "')[1].split('"')[0],
        "exit": proc.returncode,
        "stdout_sha256": hashlib.sha256(proc.stdout.encode()).hexdigest(),
        "numeric_sha256": "",
        # Stage 1: the SCF field.
        "converged": None,
        "iterations": None,
        "total_energy_hartree": None,
        "electronic_energy_hartree": None,
        "energy_delta_hartree": None,
        "rms_density_delta": None,
        # Stage 2: post-SCF properties off the converged density.
        "dipole_z": None,
        "populations_present": None,
        # Stage 3: derivative stages. None of these is reachable on this fixture;
        # the columns exist so a run that grows one is visible in the manifest
        # rather than silently absent.
        "gradient_present": None,
        "response_present": None,
        "hessian_present": None,
    }

    if proc.returncode == 0 and proc.stdout.strip():
        result = json.loads(proc.stdout)
        row["numeric_sha256"] = hashlib.sha256(numeric_record(result).encode()).hexdigest()
        row["converged"] = result.get("converged")
        row["iterations"] = result.get("iterations")
        # repr() of the parsed double round-trips, so a comparison of two
        # records compares the numbers and not the writer's digit count.
        for key in ("total_energy_hartree", "electronic_energy_hartree"):
            row[key] = repr(result.get(key))

        row["energy_delta_hartree"] = repr(result.get("energy_delta_hartree"))
        row["rms_density_delta"] = repr(result.get("rms_density_delta"))
        moments = result.get("moments") or {}
        dipole = moments.get("dipole")
        row["dipole_z"] = repr(dipole[2]) if dipole else None
        row["populations_present"] = result.get("populations") is not None
        properties = result.get("properties") or {}
        row["gradient_present"] = properties.get("xc_gradient") is not None
        row["response_present"] = result.get("response") is not None
        row["hessian_present"] = result.get("hessian") is not None

    return row


def write_manifest(outdir: pathlib.Path, rows: list) -> None:
    fields = [
        "tag",
        "accuracy",
        "exit",
        "converged",
        "iterations",
        "total_energy_hartree",
        "electronic_energy_hartree",
        "energy_delta_hartree",
        "rms_density_delta",
        "dipole_z",
        "populations_present",
        "gradient_present",
        "response_present",
        "hessian_present",
        "numeric_sha256",
        "stdout_sha256",
    ]
    lines = [",".join(fields)]

    for row in rows:
        lines.append(",".join(str(row[f]) for f in fields))

    (outdir / "manifest.csv").write_text("\n".join(lines) + "\n", newline="")


def run_ladder(qcx: pathlib.Path, outdir: pathlib.Path, ladder: list, fixture: pathlib.Path) -> int:
    rows = []

    for accuracy in ladder:
        toml = outdir / f"{accuracy}.toml"
        rung_input(fixture, accuracy, toml)
        rows.append(record(qcx, toml, outdir, accuracy))

    write_manifest(outdir, rows)

    print(f"runs: {len(rows)}")
    for row in rows:
        print(
            f"  {row['tag']:10s} exit={row['exit']} converged={row['converged']} "
            f"iterations={row['iterations']} energy={row['total_energy_hartree']}"
        )
    return 0


def run_control(
    qcx: pathlib.Path, outdir: pathlib.Path, repeats: int, fixture: pathlib.Path
) -> int:
    """The gate: identical input, identical runs, identical numeric records.

    Run twice, at two thread caps. The pinned leg is the one the sweep runs at:
    the engine's parallel combine order is unspecified, so bit-equality is a
    single-threaded statement and the pinned leg is what can be asserted. The
    unpinned leg measures what that costs - how far the reported numbers move
    between two runs of the SAME input - because that movement is the floor a
    sweep's differences have to clear to mean anything.
    """
    failed = False

    for cap in (1, 0):
        toml = outdir / f"control-c{cap}.toml"
        rung_input(fixture, "kTight", toml, thread_cap=cap)
        rows = [record(qcx, toml, outdir, f"control-c{cap}-{i}") for i in range(repeats)]
        raw = sorted({row["stdout_sha256"] for row in rows})
        numeric = sorted({row["numeric_sha256"] for row in rows})
        energies = [float(row["total_energy_hartree"]) for row in rows]
        spread = max(energies) - min(energies)
        label = "pinned (thread_cap = 1)" if cap == 1 else "unpinned (default team)"

        print(f"control {label}: {repeats} runs")
        for index, row in enumerate(rows):
            print(f"  run {index}: energy={row['total_energy_hartree']}")

        print(f"  distinct whole-stdout records: {len(raw)} of {repeats}")
        print(f"  distinct numeric records:      {len(numeric)} of {repeats}")

        if len(numeric) == 1 and len(raw) != 1:
            runs = [json.loads((outdir / f"control-c{cap}-{i}.json").read_text()) for i in range(repeats)]
            print(f"  the whole-stdout difference is confined to {len(differing_leaves(runs))} leaves:")
            for path, values in differing_leaves(runs):
                print(f"    {path}")
                print(f"      {' '.join(str(v) for v in values)}")

        print(f"  energy spread across runs: {spread!r}")

        if cap == 1 and len(numeric) != 1:
            failed = True

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qcx", required=True, type=pathlib.Path)
    parser.add_argument("--out", type=pathlib.Path, default=HERE.parent.parent / "build" / "auxstage-runs")
    parser.add_argument("--fixture", type=pathlib.Path, default=FIXTURE)
    parser.add_argument("--ladder", default="kLoose,kNormal,kTight")
    parser.add_argument("--control", type=int, default=0)
    args = parser.parse_args()

    if not args.qcx.is_file():
        raise SystemExit(f"no such executable: {args.qcx}")

    if not args.fixture.is_file():
        raise SystemExit(f"no such fixture: {args.fixture}")

    if args.out.exists():
        shutil.rmtree(args.out)

    args.out.mkdir(parents=True)

    if args.control:
        return run_control(args.qcx, args.out, args.control, args.fixture)

    return run_ladder(args.qcx, args.out, args.ladder.split(","), args.fixture)


if __name__ == "__main__":
    sys.exit(main())
