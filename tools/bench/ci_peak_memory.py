#!/usr/bin/env python3
"""The CI peak-memory tripwire.

Runs one fixed qcx workload on the Linux CI runner (the gcc Release
leg) under /usr/bin/time -v and
compares the process peak RSS against the committed baseline
benchmarks/data/perf-guard/ci-peak-memory.json. ALERT-ONLY BY DESIGN:
every outcome path exits 0 - the workflow step must never fail the
job. A threshold trip past max(1.5 x baseline, baseline + 512 MiB)
emits a GitHub workflow warning annotation naming the workload and
the delta; thresholds go data-driven after ~12 unchanged rows.

Workload: the fixture manifest tools/bench/cases/c12h26_sto3g_rhf.toml
(38 atoms / 86 contracted STO-3G, direct RHF, core guess, kNormal -
sub-minute), converted to the sectioned run-input schema by the
harness's own qcx_runner.WriteRunToml and invoked as `qcx run <toml>`:
the same case manifest and binary contract the comparative harness
uses. (`qcx run` consumes the sectioned [molecule]/[basis]/... schema
of io/parse_input.cpp, not the flat case manifest - WriteRunToml is
the one conversion site, which is why this step runs the conversion
rather than naming the manifest to qcx directly.)

CI never commits: the measured peak is printed on a 'perf-guard
harvest' line for the lane to copy into the baseline file when an
accepted change moves the footprint (the committed-baseline
discipline of the committed reference pins).

Usage (from the repo root, as the workflow step does):

    python3 tools/bench/ci_peak_memory.py \
        --exe build/wsl-gcc-release/driver/qcx
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import tomllib
from typing import Optional

# The repo-root import (tools/ is a namespace package; scripts in
# tools/bench resolve the root themselves).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.bench.qcx_runner import WriteRunToml  # noqa: E402
from tools.check_binary_freshness import check_exe  # noqa: E402

# The first-cut alert rule: trip when the
# row exceeds max(1.5 x baseline peak, baseline + 512 MiB).
ALERT_MULTIPLIER = 1.5
ALERT_ADD_KIB = 512 * 1024

DEFAULT_CASE = "tools/bench/cases/c12h26_sto3g_rhf.toml"
DEFAULT_BASELINE = "benchmarks/data/perf-guard/ci-peak-memory.json"
DEFAULT_EXE = "build/wsl-gcc-release/driver/qcx"

# /usr/bin/time -v reports the process peak on stderr in kibibytes.
_TIME_PEAK_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def ParseGnuTimePeakKib(text: str) -> Optional[int]:
    """The 'Maximum resident set size (kbytes)' value of a /usr/bin/time
    -v stderr capture; None when the marker is absent."""
    match = _TIME_PEAK_PATTERN.search(text)
    return int(match.group(1)) if match else None


def TripThresholdKib(baseline_peak_kib: int) -> int:
    """The alert threshold: max(1.5 x baseline, baseline + 512 MiB)."""
    return max(int(ALERT_MULTIPLIER * baseline_peak_kib),
               baseline_peak_kib + ALERT_ADD_KIB)


def IsTrip(peak_kib: int, baseline_peak_kib: int) -> bool:
    """True when the measured peak trips the first-cut alert rule."""
    return peak_kib > TripThresholdKib(baseline_peak_kib)


def _WorkflowEscape(text: str) -> str:
    """Escape a GitHub workflow-command data section (% -> %25, CR and
    LF are illegal inside one annotation line)."""
    return (text.replace("%", "%25")
                .replace("\r", "%0D")
                .replace("\n", "%0A"))


def _WarningAnnotation(message: str) -> None:
    """A workflow warning annotation (job stays green - the tripwire
    never fails the job; the annotation is the alert channel)."""
    print(f"::warning title=peak-memory tripwire::{_WorkflowEscape(message)}")


def Main(argv: list) -> int:
    parser = argparse.ArgumentParser(
        description="CI peak-memory tripwire (alert-only; always "
                    "exits 0 so the workflow step never fails the job)")
    parser.add_argument("--exe", default=DEFAULT_EXE,
                        help="the built qcx binary (default: %(default)s)")
    parser.add_argument("--case", default=DEFAULT_CASE,
                        help="the case manifest to run (default: "
                             "%(default)s)")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE,
                        help="the committed baseline JSON (default: "
                             "%(default)s)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-run timeout in seconds (default: "
                             "%(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the baseline state "
                             "without launching any workload")
    args = parser.parse_args(argv)

    exe = pathlib.Path(args.exe)
    case_path = pathlib.Path(args.case)
    baseline_path = pathlib.Path(args.baseline)

    if not baseline_path.is_file():
        print(f"perf-guard: baseline {baseline_path} missing - the "
              f"tripwire is unarmed (no comparison possible); "
              f"check that benchmarks/data/perf-guard/ci-peak-memory.json "
              f"is committed")
        return 0

    with open(baseline_path, "rb") as f:
        baseline = json.load(f)
    base_kib = int(baseline["baseline_peak_kib"])
    history = baseline.get("history_rows") or []
    seed = bool(history and history[-1].get("seed"))

    if args.dry_run:
        print(f"perf-guard dry-run: workload {case_path.name} via "
              f"{exe} run (manifest -> run-input conversion, preset "
              f"from the manifest), baseline {baseline_path} "
              f"(baseline_peak_kib {base_kib}, "
              f"threshold {TripThresholdKib(base_kib)} KiB, "
              f"seed={seed}); no workload launched")
        return 0

    if not exe.is_file():
        print(f"perf-guard: qcx exe {exe} not found - tripwire "
              f"skipped this leg")
        return 0

    # A MISSING binary is a skip (there is nothing to measure); a STALE one is
    # not - it measures the wrong code and reports a number for it, which is
    # the tripwire's whole failure mode.
    _verdict = check_exe(exe)
    if _verdict is not None and _verdict.stale:
        print(f"perf-guard: qcx exe {exe} is STALE - {_verdict.newest[0]} is "
              f"{abs(_verdict.drift_seconds):.0f}s newer than the binary; refusing to "
              f"trip on old code (rebuild that target)")
        return 1

    try:
        with open(case_path, "rb") as f:
            manifest = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        _WarningAnnotation(f"case manifest {case_path} unreadable: "
                           f"{exc} - no memory row this push")
        return 0

    preset = manifest.get("method", {}).get("accuracy", "kNormal")

    try:
        with tempfile.TemporaryDirectory(prefix="perf_guard_") as tmp:
            run_toml = pathlib.Path(tmp) / f"{case_path.stem}.toml"
            WriteRunToml(manifest, preset, run_toml)
            try:
                proc = subprocess.run(
                    ["/usr/bin/time", "-v", str(exe), "run", str(run_toml)],
                    capture_output=True, text=True, timeout=args.timeout)
            except FileNotFoundError:
                _WarningAnnotation("/usr/bin/time missing - add the "
                                   "'time' package to the workflow apt "
                                   "install line")
                return 0
            except subprocess.TimeoutExpired:
                _WarningAnnotation(f"{case_path.stem} run exceeded "
                                   f"{args.timeout:.0f} s - no memory "
                                   f"row this push")
                return 0
    except Exception as exc:  # noqa: BLE001 - alert-only by design
        _WarningAnnotation(f"tripwire machinery failed: {exc}")
        return 0

    if proc.returncode != 0:
        stderr_tail = proc.stderr[-400:].strip()
        _WarningAnnotation(f"{case_path.stem} run exited "
                           f"{proc.returncode} - no memory row this "
                           f"push (stderr tail: {stderr_tail})")
        return 0

    peak_kib = ParseGnuTimePeakKib(proc.stderr)
    if peak_kib is None:
        _WarningAnnotation(f"no 'Maximum resident set size' marker in "
                           f"the /usr/bin/time -v output - is the "
                           f"invocation wrapped in GNU time?")
        return 0

    trip = IsTrip(peak_kib, base_kib)
    pct = 100.0 * (peak_kib - base_kib) / base_kib
    threshold = TripThresholdKib(base_kib)
    state = "TRIP" if trip else "OK"
    print(f"perf-guard peak-memory: {case_path.stem} peak RSS "
          f"{peak_kib} KiB vs baseline {base_kib} KiB "
          f"({pct:+.0f}%), threshold {threshold} KiB "
          f"(max(1.5x, +512 MiB)) - {state}")
    if trip:
        message = (f"{case_path.stem} peak RSS {peak_kib} KiB exceeds "
                   f"the committed baseline {base_kib} KiB by "
                   f"{pct:+.0f}% (alert at max(1.5x, +512 MiB) = "
                   f"{threshold} KiB)")
        if seed:
            message += ("; the baseline row is still the SEED "
                        "placeholder - replace it with a measured row "
                        "before judging the delta")
        _WarningAnnotation(message)
    if seed:
        print(f"perf-guard: baseline row is the SEED placeholder - "
              f"harvest the measured row below to arm the tight "
              f"comparison")
    print(f"perf-guard harvest: {case_path.stem} peak RSS {peak_kib} "
          f"KiB -> update baseline_peak_kib in {baseline_path} when "
          f"this becomes the accepted footprint (baseline and gate in the "
          f"same commit)")
    return 0


if __name__ == "__main__":
    sys.exit(Main(sys.argv[1:]))
