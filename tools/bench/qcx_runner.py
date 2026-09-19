#!/usr/bin/env python3
"""The qcx side of the comparative benchmark harness.

Writes the run-input TOML that `qcx run` accepts from a case manifest
(Angstrom coordinates, verbatim from the manifest - the single-copy
fairness guarantee), invokes the exe, parses the result JSON from
stdout (io/src/result_json.cpp), and measures wall (perf_counter) + child
CPU time (GetProcessTimes) around the subprocess.

Thread contract: OMP_NUM_THREADS is set in the child environment and
recorded as threads_set. The thread count the fp64 path actually gets
is capped at half the team by the size policy and is not observable
from outside the process, so the row carries threads_set plus a policy
note; the OMP_NUM_THREADS=1 point is the serial anchor that
sidesteps the question entirely.

The preset-to-SCF-tolerance mapping mirrors the pyscf conv_tol mapping:
kLoose 1e-8 / kNormal 1e-10 / kTight 1e-12, injected into the
[scf] block of the generated TOML so both sides converge to the same
gate.

Every child launches under the memory gate (tools/bench/memory_gate.py):
a job-object process-memory cap and a free-RAM floor - the 16 GiB user
rule (set after the 2026-08-29 machine kill). A refused
launch comes back as a refused row, never as an uncapped run.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from typing import Optional

# The repo-root import (tools/ is a namespace package; scripts in
# tools/bench resolve the root themselves).
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.bench import memory_gate, system  # noqa: E402

# The accuracy-preset names the run-input schema accepts (literal PascalCase).
PRESETS = ("kLoose", "kNormal", "kTight")

# The convergence-gate LADDER, per accuracy preset (owner ruling 2026-09-15):
# the DENSITY leg is always 10^2 looser than the ENERGY leg. This is the
# contract; a cell that needs a different gate must declare it at its own call
# site with its reason, and tools/check_gate_tolerances.py enforces the match.
#
# The history matters, because two earlier statements both pointed the wrong
# way. The 2026-08-29 table put the two legs EQUAL - 1e-8/1e-8,
# 1e-10/1e-10, 1e-12/1e-12 - which made "kLoose" tight on its binding leg and
# encoded 1e-10/1e-10, the pair the owner's later hard ruling banned outright.
# A first correction (2026-09-15) then flattened all three presets onto the
# operating default (1e-8/1e-6), which was safe but erased the ladder: the
# presets are supposed to differ. The owner's ruling that settles it is the
# three-row ladder below, restored as a LADDER rather than as equal legs.
#
# kNormal's pair is the operating default and is unchanged by any of this.
_PRESET_TOLERANCES = {
    "kTight": (1e-10, 1e-8),
    "kNormal": (1e-8, 1e-6),
    "kLoose": (1e-6, 1e-4),
}


def tolerances_for_preset(preset: str) -> tuple[float, float]:
    if preset not in _PRESET_TOLERANCES:
        raise ValueError(f"unknown accuracy preset {preset!r} (kLoose|kNormal|kTight)")
    return _PRESET_TOLERANCES[preset]


def _fmt_float(x: float) -> str:
    return repr(x)


def WriteRunToml(manifest: dict, preset: str, out_path: pathlib.Path,
                 fock_builder: Optional[str] = None,
                 max_iterations: Optional[int] = None) -> None:
    """Serialize the run-input TOML for one (case, preset) cell.

    Everything comes from the manifest verbatim (the single geometry
    copy); accuracy and the SCF tolerances are injected per preset.
    fock_builder overrides the manifest's builder for the extra-cell runs
    (--builders qfmm etc.); None means the manifest's own builder.
    max_iterations caps the SCF loop; None means the manifest's own value.
    A memory-reservation leg wants a SMALL cap - its peak is set by the
    workspace, allocated at Create, so iteration 1 already carries it and
    converging buys the measurement nothing.
    """
    scf = manifest.get("scf", {})
    energy_tol, density_tol = tolerances_for_preset(preset)
    builder = fock_builder or manifest["method"]["fock_builder"]
    lines = [
        "[molecule]",
        f"charge = {manifest.get('charge', 0)}",
        f"multiplicity = {manifest.get('multiplicity', 1)}",
        "atoms = " + _atoms_toml(manifest["atoms"]),
        "",
        "[basis]",
        f'orbital = "{manifest["basis"]["orbital"]}"',
    ]
    aux = manifest["basis"].get("aux")
    if aux:
        lines.append(f'aux = "{aux}"')
    lines += [
        "",
        "[method]",
        f'type = "{manifest["method"]["type"]}"',
        f'fock_builder = "{builder}"',
        f'accuracy = "{preset}"',
        "",
        "[scf]",
        f"max_iterations = {scf.get('max_iterations', 100) if max_iterations is None else max_iterations}",
        f"energy_tolerance = {_fmt_float(energy_tol)}",
        f"density_tolerance = {_fmt_float(density_tol)}",
        f"use_diis = {str(scf.get('use_diis', True)).lower()}",
        "",
        "[guess]",
        f'type = "{_guess_type(manifest)}"',
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _guess_type(manifest: dict) -> str:
    guess = manifest.get("guess") or {}
    return guess.get("type", "core")


def _atoms_toml(atoms: list) -> str:
    rows = []
    for row in atoms:
        parts = [f'"{row[0]}"'] + [_fmt_float(v) for v in row[1:]]
        rows.append("[" + ", ".join(parts) + "]")
    return "[" + ", ".join(rows) + "]"


def ChildCpuTimesMs(proc: subprocess.Popen) -> Optional[tuple[float, float]]:
    """(user_ms, kernel_ms) of a finished child process on Windows, via
    GetProcessTimes. None when unavailable (the row then reports cpu_ms
    as None)."""
    try:
        import ctypes
        from ctypes import wintypes

        class _FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        creation = _FileTime()
        exit_ = _FileTime()
        kernel = _FileTime()
        user = _FileTime()
        handle = getattr(proc, "_handle", None)
        if handle is None:
            return None
        ok = ctypes.windll.kernel32.GetProcessTimes(
            handle, ctypes.byref(creation), ctypes.byref(exit_),
            ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            return None
        _100ns_per_ms = 10000.0
        to_ms = lambda ft: ((ft.high << 32) | ft.low) / _100ns_per_ms
        return (to_ms(user), to_ms(kernel))
    except Exception:
        return None


def TimeoutRow(timeout_s: float) -> dict:
    """The outcome row of the per-run timeout kill: an explicit,
    distinguishable error marker (timed_out + error, converged false)
    following this file's error-record conventions - a kill is never
    a bare non-converged verdict. The c24h50_def2svp nightly failure
    was the 1800-s kill at ~iteration 5 reported as 'not converged'
    (the c24h50-admission-convergence diagnostic, c4b32dd)."""
    return {"side": "qcx", "exit_code": -1, "converged": False,
            "timed_out": True,
            "error": f"timed out after {timeout_s:.0f} s (per-run "
                     f"timeout kill - not an SCF verdict)"}


def RunQcx(exe: pathlib.Path,
           toml_path: pathlib.Path,
           threads: int,
           timeout_s: float = 3600.0,
           memory_cap_gib: float = memory_gate.DefaultCapGiB,
           free_floor_gib: float = memory_gate.DefaultFreeFloorGiB) -> dict:
    """Run `qcx run <toml>` and return the outcome row.

    Returns a dict with side/exit/converged/iterations/energy/...
    threads_effective records threads_set (the fp64 thread cap is not
    observable from outside; see the module docstring). The memory
    gate: the child launches only when the host has at least
    free_floor_gib free and runs under a job object with a hard
    memory_cap_gib process limit - fail-closed, a refused launch comes
    back as a refused row, never as an uncapped run.
    """
    free_gib = system.FreeRamGiB()
    if not memory_gate.CheckFreeRam(free_gib, free_floor_gib):
        return {"side": "qcx", "exit_code": -2, "converged": False,
                "refused": "free_ram_below_floor",
                "free_ram_gib": round(free_gib, 1),
                "error": f"free RAM {free_gib:.1f} GiB below the "
                         f"{free_floor_gib:.1f} GiB floor - refusing to "
                         f"launch (memory gate)"}
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    wall_start = time.perf_counter()
    proc = subprocess.Popen([str(exe), "run", str(toml_path)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            env=env)
    try:
        memory_gate.ApplyToChild(proc, memory_cap_gib)
    except RuntimeError as exc:
        return {"side": "qcx", "exit_code": -2, "converged": False,
                "refused": "memory_cap_failed",
                "error": f"memory cap could not be applied - {exc} "
                         f"(refusing to run uncapped)"}
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        return TimeoutRow(timeout_s)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    user_ms, kernel_ms = ChildCpuTimesMs(proc) or (None, None)
    cpu_ms = (user_ms + kernel_ms) if (user_ms is not None) else None
    # The memory-guard layers read the job-object peak after child exit
    # (the same job object the memory cap runs in - the cap and the
    # read-out share the handle). Fail-open: None when unavailable.
    peak_rss_kib = memory_gate.JobPeakMemoryKib(proc)

    row = {
        "side": "qcx",
        "exit_code": proc.returncode,
        "wall_ms": wall_ms,
        "cpu_ms": cpu_ms,
        "peak_rss_kib": peak_rss_kib,
        "memory_cap_gib": memory_cap_gib,
        "threads_set": threads,
        "threads_effective": threads,
        "thread_note": "threads_set=OMP_NUM_THREADS; the fp64 Fock "
                       "path is capped at half the team (not observable "
                       "from outside); OMP_NUM_THREADS=1 is the serial "
                       "anchor",
        "stderr_tail": stderr.decode(errors="replace")[-2000:],
    }

    if proc.returncode != 0:
        row["converged"] = False
        row["error"] = f"qcx exit {proc.returncode}"
        return row

    try:
        result = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        row["converged"] = False
        row["error"] = f"result JSON parse failed: {exc}"
        return row

    row["converged"] = bool(result.get("converged", False))
    row["iterations"] = result.get("iterations")
    row["total_energy_hartree"] = result.get("total_energy_hartree")
    row["spin_squared"] = result.get("spin_squared")
    timings = result.get("timings_ms") or {}
    row["driver_total_ms"] = timings.get("total")
    row["driver_scf_loop_ms"] = timings.get("scf_loop")
    row["schema_version"] = result.get("schema_version")
    return row
