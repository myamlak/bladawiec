#!/usr/bin/env python3
"""The link-mode wall-time comparison: one hot call, base revision vs head.

WHAT THIS INSTRUMENT ANSWERS
  The runtime-effectiveness rule says a change may not cost per-call time. This
  script measures the same hot call in two builds on one machine and reports the
  ratio with a canary, so a regression is a number with a denominator rather
  than an opinion. The hot call is the direct Fock builder on a fixed fixture
  (C6H14/STO-3G, kLoose, diagonal density), built by the
  `qcx-bench-fock-hot` target; the instrument works in the static build and in
  the shared build (`QCX_BUILD_SHARED=ON`), because the link-mode axis is
  exactly one of the questions it exists to price.

THE PROTOCOL (taken from the tree's timing tools, not invented here)
  * A fixed-work canary runs immediately before and after every measured call:
    a dependent xorshift chain with no memory and no floating point, so its
    cost cannot move with the work being measured. The SESSION FLOOR is the
    fastest canary reading of the whole run; every other reading is reported as
    100 * (reading - floor) / floor. A canary that widens means the machine
    moved, and the run says so instead of blaming the code.
  * WARM-UP, twice. The driver discards its first `--warmup` calls (page
    faults, table construction). Each arm is also launched once per round and
    that launch is discarded - the warm-up launch pays the loader and the OS
    file cache, which is precisely the term a static and a shared executable
    do not share. Only the second launch of each arm is measured.
  * ABBA ORDER. Rounds alternate base->head and head->base, so a monotonic
    drift over the run lands on both arms equally.
  * PAIRED INTERLEAVED STATISTIC. The per-round quantity is the ratio
    min(head calls) / min(base calls), and the reported ratio is the MEDIAN of
    those per-round ratios, with a seeded percentile bootstrap interval over
    the rounds. It is never a difference of separately measured medians. The
    minimum call time is the robust figure the Fock acceptance benchmark
    documents; the warm-up calls are discarded before it is taken, so the
    mean would carry nothing the minimum does not - it is not used.
  * The ratio is reported with its whole sample (every per-round ratio) and
    with the interval. A wide interval is reported as wide; the script does not
    re-run a cell until it looks tight. One interval rule binds: a point
    estimate that passes but whose interval reaches past the fail bound is
    FLAGGED, because such a run cannot tell a pass from a fail.

THE PRE-REGISTERED BANDS
  Ratio (head/base, upper side - the rule gates slowdown):
      <= 1.02            PASS
      (1.02, 1.05]       FLAGGED   -> the run is PROVISIONAL, repeat it in a
                                     quiet window before drawing a conclusion
      > 1.05             FAIL
  Canary peak excess over the session floor:
      <= 5%              quiet
      (5%, 10%]          wide -> the run is PROVISIONAL
      > 10%              the machine moved -> the run is PROVISIONAL and its
                         ratio is not used; repeat it in a quiet window
  Above 10% the canary overrides a ratio FAIL: the run is PROVISIONAL, because
  the ratio it would fail on is not admissible. A canary in the 5-10% band
  does not override the ratio - the run keeps its ratio verdict and carries
  the canary beside it.
  A checksum over the Fock matrix must agree between the arms to 1e-10
  relative, or the premise "both arms did the same work" is broken and the run
  FAILS whatever the timings say.

THE QUIET GATE
  Before any measurement the script applies the tree's timed-limb rule - no
  other process above 5% of one core - with `system.BusiestProcesses` on
  Windows and a /proc census on Linux. An unmeasurable census is recorded as
  unmeasurable (PROVISIONAL), never as quiet. The canary then covers the run
  itself.

THE MEMORY GATE
  Every child launches under the tree's 16 GiB rule: a free-RAM floor plus a
  hard process cap (a job object on Windows, RLIMIT_AS elsewhere). A refused
  launch stops the run; an uncapped benchmark never runs.

EXIT CODES
  0 PASS           both bands clear, checksum agrees, window quiet
  1 FAIL           ratio > 1.05 with an admissible canary, or the checksums
                   disagree
  2 PROVISIONAL    the canary is above 5% (above 10% the ratio is not used),
                   the ratio is in the flag band, or the quiet gate could not
                   prove the window quiet
  3 ERROR          setup failed (missing revision, executable, build, preset)

Usage
    # what would run, building nothing and launching nothing
    python tools/bench/link_mode_comparison.py --dry-run --base-rev origin/main

    # compare two already-built executables (either link mode)
    python tools/bench/link_mode_comparison.py \
        --base-exe build/base/benchmarks/Release/qcx-bench-fock-hot.exe \
        --head-exe build/head/benchmarks/Release/qcx-bench-fock-hot.exe

    # build both revisions from git, shared mode, and compare
    python tools/bench/link_mode_comparison.py \
        --base-rev origin/main --head-rev HEAD --link-mode shared \
        --record build/link-mode-shared.json

  The build mode creates detached git worktrees under build/link-mode-arms/,
  runs `cmake --preset` there with `-DQCX_BUILD_BENCHMARKS=ON` (the driver is
  a benchmark target and the WSL presets do not build that group by default),
  and builds only the one target. Both revisions must already contain
  `benchmarks/fock_hot_call.cpp`: a base from before the driver landed cannot
  build it, and setup says so and exits 3. `--reuse-trees` keeps the
  worktrees for a later run (an incremental build then rebuilds only what
  moved); without it they are removed at the end.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tools.bench import memory_gate, system  # noqa: E402

# The driver target and the accuracy preset it measures. The target owns the
# fixture; this script only launches it.
DriverTarget = "qcx-bench-fock-hot"

# The pre-registered bands. Fixed on purpose: a band that moves after the
# numbers arrive is not a pre-registration.
RatioPassBound = 1.02
RatioFailBound = 1.05
CanaryPassPercent = 5.0
CanaryProvisionalPercent = 10.0
ChecksumTolerance = 1e-10

# The measurement shape. Ten rounds is this instrument's default for a paired
# comparison; five is the floor below which an interval says nothing the reader
# can act on.
DefaultRounds = 10
MinimumRounds = 5
DefaultIterations = 21
DefaultWarmup = 3
DefaultSeed = 20260912
DefaultResamples = 10000

# The quiet gate's ceiling: "no other process above 5% CPU", normalized to one
# core. Stated once, here, so every probe compares against the same number.
QuietCeiling = 0.05


def NowUtc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def Sha256Of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def Median(values: list) -> float:
    """The median, interpolating the two middles of an even sample - the same
    convention as the tree's timing statistics."""
    return float(statistics.median(values))


def Percentile(values: list, fraction: float) -> float:
    """The linear-interpolated percentile (numpy's default), stated so a
    reader can reproduce the number without this file."""
    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def BootstrapMedianCi(values: list, seed: int, resamples: int) -> dict:
    """The percentile bootstrap interval of the median over `values`. For a
    paired comparison the caller passes one value per ROUND, which is what
    makes this a bootstrap over blocks. The seed and the resample count are
    returned with the interval, so the record can name how it was produced."""
    if len(values) < 2:
        return {"low": None, "high": None, "method": "percentile_bootstrap",
                "seed": seed, "resamples": 0,
                "reason": "fewer than two rounds - no interval exists"}

    generator = random.Random(seed)
    medians = []

    for _ in range(resamples):
        sample = [values[generator.randrange(len(values))]
                  for _ in range(len(values))]
        medians.append(Median(sample))

    return {"low": Percentile(medians, 0.025),
            "high": Percentile(medians, 0.975),
            "method": "percentile_bootstrap", "seed": seed,
            "resamples": resamples}


def GeometricMean(values: list) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


# --- the quiet gate ----------------------------------------------------------


def AncestorPids() -> set:
    """The measuring tree's own process chain. The gate subtracts its
    apparatus instead of guessing at it by name."""
    pids = set()
    pid = os.getpid()

    while pid > 0:
        pids.add(pid)

        try:
            text = pathlib.Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            break

        close = text.rfind(")")

        if close < 0:
            break

        try:
            pid = int(text[close + 2:].split()[1])
        except (IndexError, ValueError):
            break

    return pids


def ProcCpuCensus() -> dict | None:
    """One /proc reading of every process's CPU ticks and parent."""
    census = {}

    try:
        entries = list(pathlib.Path("/proc").iterdir())
    except OSError:
        return None

    for entry in entries:
        if not entry.name.isdigit():
            continue

        try:
            text = (entry / "stat").read_text()
        except OSError:
            continue

        close = text.rfind(")")

        if close < 0:
            continue

        fields = text[close + 2:].split()

        try:
            # After the comm field: index 0 is state (field 3), so ppid is 1,
            # utime 11 and stime 12 (fields 4, 14 and 15).
            census[int(entry.name)] = {
                "name": text[text.find("(") + 1:close],
                "ppid": int(fields[1]),
                "cpu_ticks": int(fields[11]) + int(fields[12])}
        except (IndexError, ValueError):
            continue

    return census


def PosixBusiestProcesses(window_seconds: float = 1.0,
                          limit: int = 3) -> list | None:
    """The /proc counterpart of system.BusiestProcesses: the busiest processes
    by CPU over the window, `cpu_fraction` normalized to one core. None when
    the census cannot run; an empty list when it ran and saw nothing busy."""
    first = ProcCpuCensus()

    if first is None:
        return None

    started = time.perf_counter()
    time.sleep(max(0.05, window_seconds))
    elapsed = time.perf_counter() - started
    second = ProcCpuCensus()

    if second is None:
        return None

    try:
        ticks = os.sysconf("SC_CLK_TCK")
    except (AttributeError, ValueError, OSError):
        return None

    budget = ticks * elapsed

    if budget <= 0.0:
        return None

    rows = []

    for pid, entry in second.items():
        before = first.get(pid)

        if before is None:
            continue

        fraction = (entry["cpu_ticks"] - before["cpu_ticks"]) / budget

        if fraction <= 0.0:
            continue

        rows.append({"name": entry["name"], "pid": pid,
                     "cpu_fraction": round(fraction, 4)})

    rows.sort(key=lambda row: row["cpu_fraction"], reverse=True)
    return rows[:limit]


def PosixFreeRamGiB() -> float | None:
    try:
        return (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
                / (2 ** 30))
    except (AttributeError, ValueError, OSError):
        return None


def FreeRamGiB() -> float | None:
    return system.FreeRamGiB() if os.name == "nt" else PosixFreeRamGiB()


def QuietGate(mode: str, window_seconds: float) -> dict:
    """The timed-limb rule at the start of the run. `mode` is auto, windows,
    posix or none; auto resolves by the host and `none` records that the check
    did not run rather than that the machine was quiet."""
    if mode == "none":
        return {"verdict": "skipped", "reason": "the gate was disabled"}

    use_windows = mode == "windows" or (mode == "auto" and os.name == "nt")
    rows = (system.BusiestProcesses(window_seconds) if use_windows
            else PosixBusiestProcesses(window_seconds))

    if rows is None:
        return {"verdict": "unmeasurable", "rows": None,
                "reason": "the process census could not run"}

    apparatus = AncestorPids()
    offenders = [row for row in rows
                 if row["cpu_fraction"] > QuietCeiling
                 and row["pid"] not in apparatus]

    if offenders:
        return {"verdict": "busy", "rows": rows, "offenders": offenders,
                "reason": "a process above the 5% ceiling"}

    return {"verdict": "quiet", "rows": rows, "offenders": []}


# --- arm resolution ----------------------------------------------------------


class SetupError(RuntimeError):
    """A run that cannot start or cannot continue. It is an ERROR (exit 3),
    never a measurement with a hole in it."""


def DefaultPreset(link_mode: str) -> str:
    """The release configuration per host, in the requested link mode. The
    shared presets exist beside their static twins; a comparison never mixes
    Debug and Release."""
    if os.name == "nt":
        return "windows-msvc-shared" if link_mode == "shared" else "windows-msvc"

    return ("wsl-gcc-shared-release" if link_mode == "shared"
            else "wsl-gcc-release")


def DefaultConfig() -> str:
    return "Release" if os.name == "nt" else ""


def DefaultBinaryDir(repo: pathlib.Path, preset: str) -> pathlib.Path:
    return repo / "build" / preset


def CheckPresetMatchesMode(preset: str, link_mode: str) -> None:
    """A shared configure goes through a `-shared` preset and a static one
    does not. Crossing them is a setup error, not a measurement."""
    is_shared_preset = preset.endswith("-shared") or "-shared-" in preset

    if link_mode == "shared" and not is_shared_preset:
        raise SetupError(
            f"preset {preset} is a static configuration; use a -shared preset "
            f"(for example {DefaultPreset('shared')}) for --link-mode shared")
    if link_mode == "static" and is_shared_preset:
        raise SetupError(
            f"preset {preset} is a shared configuration; use --link-mode "
            f"shared or a static preset")


def RunChecked(argv: list, cwd: pathlib.Path | None = None,
               timeout: float | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(argv, cwd=str(cwd) if cwd else None,
                            capture_output=True, text=True, timeout=timeout)

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SetupError(f"{' '.join(argv)} exited {result.returncode}: {detail}")

    return result


def VerifyRevision(repo: pathlib.Path, rev: str) -> str:
    result = RunChecked(["git", "-C", str(repo), "rev-parse", "--verify",
                         rev + "^{commit}"])
    return result.stdout.strip()


def CreateArmTree(repo: pathlib.Path, tree: pathlib.Path, rev: str) -> None:
    if tree.exists():
        raise SetupError(
            f"{tree} already exists; remove it (git worktree remove --force "
            f"{tree}) or pass --reuse-trees")

    RunChecked(["git", "-C", str(repo), "worktree", "add", "--detach",
                str(tree), rev])
    RunChecked(["git", "-C", str(tree), "submodule", "update", "--init",
                "--recursive"])


def RemoveArmTree(repo: pathlib.Path, tree: pathlib.Path) -> None:
    try:
        RunChecked(["git", "-C", str(repo), "worktree", "remove", "--force",
                    str(tree)])
    except SetupError as exc:
        print(f"warning: could not remove the arm tree {tree}: {exc}",
              file=sys.stderr)


def LocateExe(binary_dir: pathlib.Path, config: str) -> pathlib.Path | None:
    """The driver executable the build produced, by name. A multi-config
    generator puts it under the configuration directory and a single-config
    one may nest it differently; searching by name works for both."""
    name = DriverTarget + (".exe" if os.name == "nt" else "")
    candidates = [path for path in binary_dir.rglob(name) if path.is_file()]

    if not candidates:
        return None

    # The newest beats a stale copy a previous configuration left behind.
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def FreshnessOf(exe: pathlib.Path, binary_dir: pathlib.Path,
                config: str) -> dict:
    """The freshness verdict when this tree can produce one. A binary the
    tree did not build (a private copy, another tree) has no verdict and is
    recorded as unknown, never as fresh."""
    try:
        from tools import check_binary_freshness
        verdict = check_binary_freshness.check_exe(
            exe, tree=binary_dir, config=config or "Release")
    except Exception as exc:  # the index is a convenience, not the premise
        return {"stale": None, "reason": f"freshness check unavailable: {exc}"}

    if verdict is None:
        return {"stale": None,
                "reason": "the tree has no project output for this executable"}

    return verdict.as_record()


def DetectLinkMode(exe: pathlib.Path) -> str | None:
    """What the executable actually links, not what the caller asked for.
    Windows: the first-party DLL sitting beside the executable - the composed
    qcx.dll, or a per-module qcx-*.dll from a base revision that predates the
    composed shape. Elsewhere: ldd, where libqcx.so is both spellings' prefix.
    None when the question cannot be answered on this host."""
    if os.name == "nt":
        composed = (exe.parent / "qcx.dll").is_file()
        per_module = any(path.is_file() for path in exe.parent.glob("qcx-*.dll"))
        return "shared" if (composed or per_module) else "static"

    try:
        result = subprocess.run(["ldd", str(exe)], capture_output=True,
                                text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None

    return "shared" if "libqcx" in result.stdout else "static"


def ArmRecord(label: str, arm: dict) -> dict:
    exe = arm.get("exe")
    record = {"label": label, "rev": arm.get("rev"),
              "tree": str(arm["tree"]) if arm.get("tree") else None,
              "preset": arm.get("preset"), "config": arm.get("config"),
              "exe": str(exe) if exe else None}

    if exe and exe.is_file():
        record["sha256"] = Sha256Of(exe)
        record["mtime_utc"] = datetime.datetime.fromtimestamp(
            exe.stat().st_mtime, datetime.timezone.utc).isoformat(timespec="seconds")
        record["link_mode_detected"] = DetectLinkMode(exe)
        if arm.get("binary_dir"):
            record["freshness"] = FreshnessOf(exe, arm["binary_dir"],
                                              arm.get("config") or "")

    return record


# --- launching the driver ----------------------------------------------------


def DriverArgv(arm: dict, args) -> list:
    # A plan prints the shape of the launch before the executable exists; the
    # real run never gets here with a missing path.
    exe = str(arm["exe"]) if arm.get("exe") else f"<{DriverTarget}>"
    argv = [exe, "--iterations", str(args.iterations),
            "--warmup", str(args.warmup), "--json"]

    if args.parallel:
        argv.append("--parallel")

    return argv


def _PosixLimit(cap_gib: float):
    """The RLIMIT_AS cap for a POSIX child (the Windows job object has no
    counterpart here; the rule is the same and the mechanism is the host's)."""
    def apply():
        import resource
        cap = int(cap_gib * (2 ** 30))
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    return apply


def LaunchDriver(arm: dict, argv: list, cap_gib: float, free_floor_gib: float,
                 timeout: float) -> dict:
    """One child under the memory gate, with its process wall time. Raises
    SetupError when the launch is refused - never runs uncapped."""
    free_gib = FreeRamGiB()

    if not memory_gate.CheckFreeRam(free_gib, free_floor_gib):
        raise SetupError(
            f"free RAM {free_gib:.2f} GiB is below the {free_floor_gib:.1f} GiB "
            f"floor; refusing to launch")

    started = time.perf_counter()

    if os.name == "nt":
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        try:
            memory_gate.ApplyToChild(proc, cap_gib)
        except RuntimeError as exc:
            raise SetupError(str(exc)) from exc

        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise SetupError(f"the driver did not finish within {timeout:.0f}s")

        wall_ms = (time.perf_counter() - started) * 1000.0
        exit_code = proc.returncode
        peak_kib = memory_gate.JobPeakMemoryKib(proc)
    else:
        try:
            result = subprocess.run(argv, capture_output=True, text=True,
                                    timeout=timeout, preexec_fn=_PosixLimit(cap_gib))
        except subprocess.TimeoutExpired as exc:
            raise SetupError(
                f"the driver did not finish within {timeout:.0f}s") from exc

        wall_ms = (time.perf_counter() - started) * 1000.0
        stdout, stderr = result.stdout, result.stderr
        exit_code = result.returncode
        peak_kib = None

    if exit_code != 0:
        detail = (stderr or stdout or "").strip()
        raise SetupError(f"{arm['label']} driver exited {exit_code}: {detail}")

    record = ParseDriverJson(stdout, arm["label"])

    return {"record": record, "wall_ms": wall_ms, "peak_kib": peak_kib,
            "stderr": stderr}


def ParseDriverJson(stdout: str, label: str) -> dict:
    """The driver's one JSON line. Parsing is strict: a missing or malformed
    record is a setup error, because a run without its canary is not a
    measurement this script can report."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()

        if not line.startswith("{"):
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SetupError(f"{label}: the driver's JSON is malformed: {exc}")

        if record.get("schema") != "qcx-fock-hot/1":
            raise SetupError(
                f"{label}: unexpected driver schema {record.get('schema')!r}")

        for key in ("per_call_ms", "canary_before_ms", "canary_after_ms",
                    "fock_sum"):
            if key not in record:
                raise SetupError(f"{label}: the driver's record has no {key!r}")

        calls = len(record["per_call_ms"])

        if calls == 0:
            raise SetupError(f"{label}: the driver's record holds no call")
        if (len(record["canary_before_ms"]) != calls
                or len(record["canary_after_ms"]) != calls):
            raise SetupError(
                f"{label}: the driver's canary arrays do not match its "
                f"{calls} calls")

        return record

    raise SetupError(f"{label}: the driver printed no JSON record")


# --- the statistics ----------------------------------------------------------


def CanaryVerdict(readings: list) -> dict:
    """The session floor, the peak, and the band. Every reading is one canary
    run; the floor is the fastest of the session, so a machine that moved
    anywhere in the run shows up in the peak."""
    if not readings:
        return {"verdict": "unmeasurable", "floor_ms": None, "peak_ms": None,
                "peak_percent": None, "readings": 0}

    floor_ms = min(readings)
    peak_ms = max(readings)

    if floor_ms <= 0.0:
        return {"verdict": "unmeasurable", "floor_ms": floor_ms,
                "peak_ms": peak_ms, "peak_percent": None,
                "readings": len(readings)}

    peak_percent = 100.0 * (peak_ms - floor_ms) / floor_ms

    if peak_percent <= CanaryPassPercent:
        verdict = "quiet"
    elif peak_percent <= CanaryProvisionalPercent:
        verdict = "provisional"
    else:
        verdict = "abort"

    return {"verdict": verdict, "floor_ms": floor_ms, "peak_ms": peak_ms,
            "peak_percent": peak_percent, "readings": len(readings),
            "pass_percent": CanaryPassPercent,
            "provisional_percent": CanaryProvisionalPercent}


def RatioVerdict(ratio: float, ci: dict) -> dict:
    """The pre-registered band on the ratio.

    The point estimate decides the band, with one interval rule: a run whose
    point estimate passes but whose interval reaches past the fail bound is
    not evidence of a pass - it cannot tell a pass from a fail - so it is
    reported as a flag (PROVISIONAL at the combined stage) rather than as a
    pass. That is the pre-registered reading of "repeat it in a quiet
    window"."""
    if ratio <= RatioPassBound:
        verdict = "pass"

        if ci.get("high") is not None and ci["high"] > RatioFailBound:
            verdict = "flag"
    elif ratio <= RatioFailBound:
        verdict = "flag"
    else:
        verdict = "fail"

    return {"verdict": verdict, "ratio": ratio, "pass_bound": RatioPassBound,
            "fail_bound": RatioFailBound,
            "ci_low": ci.get("low"), "ci_high": ci.get("high"),
            "interval_crosses_fail_bound": bool(
                ci.get("high") is not None and ci["high"] > RatioFailBound)}


def CombinedVerdict(canary: dict, ratio: dict, checksum: dict,
                    quiet: dict, freshness: dict | None = None) -> dict:
    reasons = []

    # A binary older than the tree that produced it is not this revision's
    # binary: the timing belongs to whatever revision is in the file. The
    # freshness verdict is the tree's own index, so a stale arm refuses the
    # run outright rather than being averaged into it.
    stale = [label for label, record in (freshness or {}).items()
             if record and record.get("stale") is True]

    if stale:
        reasons.append(
            f"the {' and '.join(stale)} executable is older than its build "
            f"tree (stale); a stale binary proves nothing about this revision")
        return {"status": "FAIL", "exit_code": 1, "reasons": reasons}

    # An arm whose freshness cannot be judged (a private copy, a tree with no
    # index entry) is not stale, but it is also not proved; the run is
    # provisional rather than silently trusted.
    unproved = [label for label, record in (freshness or {}).items()
                if record is not None and record.get("stale") is None]

    if unproved:
        reasons.append(
            f"the {' and '.join(unproved)} executable's freshness could not "
            f"be judged ({', '.join(str((freshness[label] or {}).get('reason')) for label in unproved)})")

    # The checksum does not depend on the clock: a disagreement means the arms
    # did different work, and no timing from such a run means anything.
    if checksum.get("agree") is False:
        reasons.append(
            f"the arms computed different Fock matrices (relative deviation "
            f"{checksum.get('max_relative_deviation')})")
        return {"status": "FAIL", "exit_code": 1, "reasons": reasons}

    # A super-wide canary makes the ratio inadmissible, so a ratio FAIL in the
    # same run is not evidence about the code: the run needs a quiet window
    # first.
    if canary["verdict"] == "abort":
        reasons.append(
            f"canary peak {canary['peak_percent']:.1f}% exceeds "
            f"{CanaryProvisionalPercent:.0f}% - the machine moved, so the "
            f"ratio is not admissible")
        return {"status": "PROVISIONAL", "exit_code": 2, "reasons": reasons}

    if ratio["verdict"] == "fail":
        reasons.append(
            f"ratio {ratio['ratio']:.4f} exceeds {RatioFailBound}")
        return {"status": "FAIL", "exit_code": 1, "reasons": reasons}

    if canary["verdict"] == "provisional":
        reasons.append(
            f"canary peak {canary['peak_percent']:.1f}% is above "
            f"{CanaryPassPercent:.0f}%")
    if ratio["verdict"] == "flag":
        if ratio.get("interval_crosses_fail_bound") and ratio["ratio"] <= RatioPassBound:
            reasons.append(
                f"ratio {ratio['ratio']:.4f} passes but its interval reaches "
                f"{ratio['ci_high']:.4f}, past the fail bound {RatioFailBound}")
        else:
            reasons.append(
                f"ratio {ratio['ratio']:.4f} is above {RatioPassBound} but "
                f"inside the fail bound {RatioFailBound}")
    if canary["verdict"] == "unmeasurable":
        reasons.append("the canary produced no reading")
    if quiet.get("verdict") in ("busy", "unmeasurable"):
        reasons.append(f"the quiet gate says {quiet['verdict']}")
    if reasons:
        return {"status": "PROVISIONAL", "exit_code": 2, "reasons": reasons}

    return {"status": "PASS", "exit_code": 0, "reasons": []}


# --- the run -----------------------------------------------------------------

def RunRounds(arms: dict, args) -> dict:
    rounds = []

    for index in range(args.rounds):
        order = ["base", "head"] if index % 2 == 0 else ["head", "base"]
        row = {"round": index + 1, "order": order, "arms": {},
               "warmup_wall_ms": {}}

        for label in order:
            arm = arms[label]
            argv = DriverArgv(arm, args)
            # The warm-up launch is discarded on purpose: it pays the loader,
            # the OS file cache and the frequency ramp, and those are not the
            # call.
            warm = LaunchDriver(arm, argv, args.memory_cap_gib,
                                args.free_floor_gib, args.timeout)
            measured = LaunchDriver(arm, argv, args.memory_cap_gib,
                                    args.free_floor_gib, args.timeout)
            record = measured["record"]
            row["warmup_wall_ms"][label] = warm["wall_ms"]
            row["arms"][label] = {
                "min_ms": min(record["per_call_ms"]),
                "mean_ms": sum(record["per_call_ms"]) / len(record["per_call_ms"]),
                "per_call_ms": record["per_call_ms"],
                "canary_ms": record["canary_before_ms"] + record["canary_after_ms"],
                "process_wall_ms": measured["wall_ms"],
                "peak_kib": measured["peak_kib"],
                "fock_sum": record["fock_sum"],
            }

        row["ratio"] = (row["arms"]["head"]["min_ms"]
                        / row["arms"]["base"]["min_ms"])
        rounds.append(row)

    return {"rounds": rounds}


def ChecksumSummary(rounds: list) -> dict:
    """Both arms must produce the same Fock matrix: the comparison's premise
    is that they did the same work. The record carries one sum per arm, so it
    catches a differing matrix rather than proving two matrices equal; a
    disagreement is enough to void the run, which is what the premise needs."""
    values = [row["arms"][label]["fock_sum"] for row in rounds
              for label in ("base", "head")]
    reference = values[0]
    scale = max(abs(reference), 1e-300)
    deviation = max(abs(value - reference) / scale for value in values)

    return {"reference": reference, "max_relative_deviation": deviation,
            "tolerance": ChecksumTolerance,
            "agree": deviation <= ChecksumTolerance}


def Summarize(rounds: list, args) -> dict:
    ratios = [row["ratio"] for row in rounds]
    ci = BootstrapMedianCi(ratios, args.seed, args.resamples)
    ratio = {"per_round": ratios, "median": Median(ratios),
             "geometric_mean": GeometricMean(ratios),
             "min": min(ratios), "max": max(ratios),
             "method": "median of the paired per-round ratios",
             "interval": ci}
    ratio.update(RatioVerdict(ratio["median"], ci))
    readings = [value for row in rounds
                for label in ("base", "head")
                for value in row["arms"][label]["canary_ms"]]
    return {"ratio": ratio, "canary": CanaryVerdict(readings)}


def HumanReport(arms: dict, quiet: dict, summary: dict, checksum: dict,
                verdict: dict, args, stream=None) -> None:
    out = sys.stdout if stream is None else stream

    def emit(text: str) -> None:
        print(text, file=out)

    emit(f"link-mode comparison | {args.link_mode} | "
         f"base {arms['base'].get('rev') or arms['base']['exe']} | "
         f"head {arms['head'].get('rev') or arms['head']['exe']}")
    emit(f"rounds {args.rounds} | calls {args.iterations}, warm-up "
         f"{args.warmup} calls + one discarded launch per arm per round | "
         f"{'parallel' if args.parallel else 'serial'}")

    gate = quiet.get("verdict")

    if gate == "quiet":
        top = ", ".join(f"{row['name']}:{row['pid']} "
                        f"{100.0 * row['cpu_fraction']:.1f}%"
                        for row in quiet["rows"][:3])
        emit(f"quiet pre-gate: quiet ({top})")
    elif gate == "busy":
        names = ", ".join(f"{row['name']}:{row['pid']} "
                          f"{100.0 * row['cpu_fraction']:.1f}%"
                          for row in quiet["offenders"])
        emit(f"quiet pre-gate: BUSY ({names})")
    else:
        emit(f"quiet pre-gate: {gate} ({quiet.get('reason')})")

    canary = summary["canary"]

    if canary["peak_percent"] is None:
        emit("canary: no reading")
    else:
        emit(f"canary: floor {canary['floor_ms']:.3f} ms, peak "
             f"{canary['peak_ms']:.3f} ms, peak +{canary['peak_percent']:.1f}% "
             f"({canary['readings']} readings; band <= {CanaryPassPercent:.0f}% "
             f"quiet, <= {CanaryProvisionalPercent:.0f}% wide, above "
             f"{CanaryProvisionalPercent:.0f}% inadmissible)")

    ratio = summary["ratio"]
    ci = ratio["interval"]
    emit("per-round ratios (head/base): "
         + " ".join(f"{value:.4f}" for value in ratio["per_round"]))
    low = "n/a" if ci.get("low") is None else f"{ci['low']:.4f}"
    high = "n/a" if ci.get("high") is None else f"{ci['high']:.4f}"
    emit(f"ratio: median {ratio['median']:.4f}, range "
         f"[{ratio['min']:.4f}, {ratio['max']:.4f}], 95% CI [{low}, {high}] "
         f"({ci['method']}, seed {ci['seed']}, {ci['resamples']} resamples)")
    emit(f"checksum: sum {checksum['reference']:.12g}, max relative "
         f"deviation {checksum['max_relative_deviation']:.3g} "
         f"(tolerance {checksum['tolerance']:.0e}, "
         f"{'agree' if checksum['agree'] else 'DISAGREE'})")
    emit(f"verdict: {verdict['status']} (exit {verdict['exit_code']})")

    for reason in verdict["reasons"]:
        emit(f"  - {reason}")


# --- the dry run -------------------------------------------------------------


def PrintPlan(arms: dict, args, repo: pathlib.Path) -> None:
    print(f"link-mode comparison dry-run | {args.link_mode} | preset "
          f"{args.preset} | config {args.config or '(single-config)'}")
    print(f"repo {repo}")
    print("nothing is built, launched or created by a dry run")

    for label in ("base", "head"):
        arm = arms[label]
        print(f"\n{label}: rev {arm.get('rev') or '(prebuilt)'}")
        if arm.get("tree"):
            print(f"  worktree {arm['tree']}")
            print(f"  configure: cmake --preset {arm['preset']} "
                  f"-DQCX_BUILD_BENCHMARKS=ON"
                  + (f" -DQCX_INTEGRALS_LMAX={args.lmax}" if args.lmax else "")
                  + f"   (cwd {arm['tree']})")
            print(f"  build: cmake --build {arm['binary_dir']} --target "
                  f"{DriverTarget}"
                  + (f" --config {args.config}" if args.config else "")
                  + f" --parallel {args.jobs or '(generator default)'}")
        exe = arm.get("exe")
        if exe:
            print(f"  driver: {exe}")
            if exe.is_file():
                print(f"  detected link mode: {DetectLinkMode(exe)}")
            else:
                print("  NOTE: the executable does not exist")
        elif arm.get("binary_dir"):
            print(f"  driver: located under {arm['binary_dir']} after the build")
        print(f"  argv:   {' '.join(DriverArgv(arm, args))}")

    print(f"\nrounds {args.rounds}, ABBA order alternating base->head and "
          f"head->base; each arm is launched once (discarded) then measured")
    print(f"memory gate: free-RAM floor {args.free_floor_gib:.1f} GiB, "
          f"process cap {args.memory_cap_gib:.1f} GiB")
    print(f"quiet gate: {args.quiet_gate} (ceiling "
          f"{100.0 * QuietCeiling:.0f}% of one core)")
    print(f"bands: ratio <= {RatioPassBound} pass, > {RatioFailBound} fail; "
          f"canary <= {CanaryPassPercent:.0f}% quiet, > "
          f"{CanaryProvisionalPercent:.0f}% inadmissible")
    print(f"record: {args.record or '(stdout only)'}")


# --- main --------------------------------------------------------------------


def ResolveArgs(argv: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the Fock hot call between a base and a head "
                    "revision on one machine, with a canary and a "
                    "pre-registered band.")
    parser.add_argument("--base-rev", default=None,
                        help="the base revision (built from git in build mode)")
    parser.add_argument("--head-rev", default="HEAD",
                        help="the head revision (default: %(default)s)")
    parser.add_argument("--base-exe", default=None,
                        help="a prebuilt base executable; skips its build")
    parser.add_argument("--head-exe", default=None,
                        help="a prebuilt head executable; skips its build")
    parser.add_argument("--link-mode", choices=("static", "shared"),
                        default="static",
                        help="which link mode both arms use (default: "
                             "%(default)s)")
    parser.add_argument("--preset", default=None,
                        help="the configure preset for build mode (default: "
                             "the host's release preset for the link mode)")
    parser.add_argument("--config", default=None,
                        help="the build configuration (default: Release on a "
                             "multi-config generator, none elsewhere)")
    parser.add_argument("--lmax", type=int, default=None,
                        help="QCX_INTEGRALS_LMAX for both arms (default: the "
                             "preset's). Both arms always share it")
    parser.add_argument("--jobs", type=int, default=None,
                        help="parallel build jobs (default: the generator's)")
    parser.add_argument("--rounds", type=int, default=DefaultRounds,
                        help="measured rounds (default: %(default)s, minimum "
                             f"{MinimumRounds})")
    parser.add_argument("--iterations", type=int, default=DefaultIterations,
                        help="calls per measured launch (default: %(default)s)")
    parser.add_argument("--warmup", type=int, default=DefaultWarmup,
                        help="calls the driver discards per launch (default: "
                             "%(default)s)")
    parser.add_argument("--parallel", action="store_true",
                        help="measure the parallel Fock lane (default: serial)")
    parser.add_argument("--seed", type=int, default=DefaultSeed,
                        help="bootstrap seed (default: %(default)s)")
    parser.add_argument("--resamples", type=int, default=DefaultResamples,
                        help="bootstrap resamples (default: %(default)s)")
    parser.add_argument("--arm-root", default=None,
                        help="where build-mode worktrees live (default: "
                             "build/link-mode-arms under the repo)")
    parser.add_argument("--reuse-trees", action="store_true",
                        help="keep the worktrees for a later incremental run")
    parser.add_argument("--memory-cap-gib", type=float,
                        default=memory_gate.DefaultCapGiB,
                        help="the child's hard memory cap (default: %(default)s)")
    parser.add_argument("--free-floor-gib", type=float,
                        default=memory_gate.DefaultFreeFloorGiB,
                        help="the free-RAM floor for a launch (default: "
                             "%(default)s)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="per-launch timeout in seconds (default: "
                             "%(default)s)")
    parser.add_argument("--quiet-gate", choices=("auto", "windows", "posix",
                                                 "none"), default="auto",
                        help="the pre-run quiet probe (default: %(default)s)")
    parser.add_argument("--quiet-window", type=float, default=1.0,
                        help="the quiet probe's sampling window in seconds "
                             "(default: %(default)s)")
    parser.add_argument("--record", default=None,
                        help="write the JSON run record here")
    parser.add_argument("--json", action="store_true",
                        help="print the JSON record to stdout and the human "
                             "lines to stderr")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan without building or launching "
                             "anything")

    args = parser.parse_args(argv)

    if args.rounds < MinimumRounds:
        parser.error(f"--rounds must be at least {MinimumRounds}: an interval "
                     f"from fewer rounds does not support a 1.05x claim")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must not be negative")

    if not args.dry_run and not (args.base_exe or args.base_rev):
        parser.error("give --base-rev (build mode) or --base-exe (prebuilt)")

    return args


def ResolveArms(args, repo: pathlib.Path) -> dict:
    """Each arm is either a prebuilt executable or a revision to build. A
    prebuilt executable wins over a revision: it is the caller's answer to
    "which binary", and a run must not quietly build a different one."""
    arms = {}

    for label in ("base", "head"):
        arm = {"label": label, "preset": args.preset, "config": args.config}
        exe_arg = getattr(args, f"{label}_exe")

        if exe_arg:
            arm["exe"] = pathlib.Path(exe_arg)
        elif getattr(args, f"{label}_rev"):
            arm["rev"] = getattr(args, f"{label}_rev")
            arm["tree"] = args.arm_root / f"{label}-{args.preset}"
            arm["binary_dir"] = DefaultBinaryDir(arm["tree"], args.preset)

        arms[label] = arm

    return arms


def Main(argv: list) -> int:
    args = ResolveArgs(argv)
    repo = pathlib.Path(__file__).resolve().parents[2]
    args.preset = args.preset or DefaultPreset(args.link_mode)
    args.config = DefaultConfig() if args.config is None else args.config
    args.arm_root = (pathlib.Path(args.arm_root) if args.arm_root
                     else repo / "build" / "link-mode-arms")

    try:
        CheckPresetMatchesMode(args.preset, args.link_mode)
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    arms = ResolveArms(args, repo)

    if args.dry_run:
        for arm in arms.values():
            if arm.get("exe") is None and arm.get("binary_dir"):
                arm["exe"] = LocateExe(arm["binary_dir"], args.config)

        PrintPlan(arms, args, repo)
        return 0

    created_trees = []

    try:
        for label in ("base", "head"):
            arm = arms[label]

            if arm.get("exe"):
                if not arm["exe"].is_file():
                    raise SetupError(
                        f"the {label} executable {arm['exe']} does not exist")
                continue

            if not arm.get("rev"):
                raise SetupError(
                    f"{label} has neither an executable nor a revision")

            arm["commit"] = VerifyRevision(repo, arm["rev"])

        for label in ("base", "head"):
            arm = arms[label]

            if arm.get("exe"):
                continue

            if args.reuse_trees and arm["tree"].exists():
                # A reused tree is only this arm if it is AT the revision
                # asked for. A head tree left at the base revision would
                # otherwise produce ratio 1.0 and a false pass.
                reused = subprocess.run(
                    ["git", "-C", str(arm["tree"]), "rev-parse", "HEAD"],
                    capture_output=True, text=True)

                head = reused.stdout.strip() if reused.returncode == 0 else ""

                if head != arm["commit"]:
                    raise SetupError(
                        f"the reused {label} tree {arm['tree']} is at "
                        f"{head or 'an unreadable revision'}, not the "
                        f"requested {arm['commit']} ({arm['rev']}): drop "
                        f"--reuse-trees or remove the tree")
            else:
                CreateArmTree(repo, arm["tree"], arm["rev"])
                created_trees.append(arm["tree"])

            # The driver is a benchmark target; the WSL presets do not build
            # that group by default, so the arm tree asks for it explicitly.
            configure = ["cmake", "--preset", arm["preset"],
                         "-DQCX_BUILD_BENCHMARKS=ON"]

            if args.lmax:
                configure.append(f"-DQCX_INTEGRALS_LMAX={args.lmax}")

            RunChecked(configure, cwd=arm["tree"])

            build = ["cmake", "--build", str(arm["binary_dir"]),
                     "--target", DriverTarget]

            if args.config:
                build += ["--config", args.config]
            if args.jobs:
                build += ["--parallel", str(args.jobs)]

            print(f"building {label} ({arm['rev']}, {arm['preset']}) ...",
                  file=sys.stderr)
            RunChecked(build, cwd=arm["tree"])

            exe = LocateExe(arm["binary_dir"], args.config)

            if exe is None:
                raise SetupError(
                    f"the {label} build produced no {DriverTarget} under "
                    f"{arm['binary_dir']}")

            arm["exe"] = exe

        # The link-mode cross-check is on the executable, not on the flag: a
        # preset that says shared but yields a static binary is a build
        # problem this run must not average over.
        for label in ("base", "head"):
            detected = DetectLinkMode(arms[label]["exe"])

            if detected and detected != args.link_mode:
                raise SetupError(
                    f"the {label} executable {arms[label]['exe']} is a "
                    f"{detected} build, but --link-mode {args.link_mode} was "
                    f"requested")

        quiet = QuietGate(args.quiet_gate, args.quiet_window)
        covariates_before = system.MachineCovariates()
        measured = RunRounds(arms, args)
        covariates_after = system.MachineCovariates()
        rounds = measured["rounds"]
        summary = Summarize(rounds, args)
        checksum = ChecksumSummary(rounds)
        arm_records = {label: ArmRecord(label, arms[label])
                       for label in ("base", "head")}
        freshness = {label: record.get("freshness")
                     for label, record in arm_records.items()}
        verdict = CombinedVerdict(summary["canary"], summary["ratio"],
                                  checksum, quiet, freshness)

        record = {
            "schema": "qcx-link-mode-comparison/1",
            "created_utc": NowUtc(),
            "link_mode_requested": args.link_mode,
            "protocol": {
                "fixture": "C6H14/STO-3G", "preset": "kLoose",
                "parallel": args.parallel, "rounds": args.rounds,
                "iterations": args.iterations, "warmup": args.warmup,
                "warmup_launch_per_arm_per_round": True,
                "order": "ABBA, alternating base->head and head->base",
                "statistic": "median of paired per-round min-call ratios",
                "seed": args.seed, "resamples": args.resamples,
                "ratio_pass_bound": RatioPassBound,
                "ratio_fail_bound": RatioFailBound,
                "canary_pass_percent": CanaryPassPercent,
                "canary_provisional_percent": CanaryProvisionalPercent,
                "checksum_tolerance": ChecksumTolerance,
                "memory_cap_gib": args.memory_cap_gib,
                "free_floor_gib": args.free_floor_gib,
                "quiet_gate": args.quiet_gate, "quiet_ceiling": QuietCeiling,
                "lmax": args.lmax,
            },
            "machine": system.MachineState(),
            "covariates_before": covariates_before,
            "covariates_after": covariates_after,
            "arms": arm_records,
            "quiet": quiet,
            "rounds": rounds,
            "summary": summary,
            "checksum": checksum,
            "verdict": verdict,
        }

        if args.record:
            path = pathlib.Path(args.record)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")

        if args.json:
            print(json.dumps(record, indent=2))
            HumanReport(arms, quiet, summary, checksum, verdict, args,
                        stream=sys.stderr)
        else:
            HumanReport(arms, quiet, summary, checksum, verdict, args)

        if args.record:
            print(f"record: {args.record}",
                  file=sys.stderr if args.json else sys.stdout)
        return verdict["exit_code"]
    except SetupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    finally:
        if not args.reuse_trees:
            for tree in created_trees:
                RemoveArmTree(repo, tree)


if __name__ == "__main__":
    sys.exit(Main(sys.argv[1:]))
