#!/usr/bin/env python3
"""A convergence gate must be a rung of the ladder, or be declared with its reason.

WHY THIS EXISTS, measured. ONE ladder is the ruled set for convergence gates:
three presets, each an (energy, density) pair whose density leg is 10^2 looser
than its energy leg. The benchmark side kept a table of its own, which set the
two legs EQUAL per preset, as its mapping for a fortnight after the ruling that
retired it - so every benchmark staged through `WriteRunToml` ran at a gate
nobody had ruled, and it surfaced only because someone READ a staged run TOML of
a live 4,974-BF measurement (2026-09-15).

This file quotes no rung value. It reads them, prints them, and compares
against them, so there is nothing here to go stale when the ladder is re-ruled -
`--all` prints the ladder, and so does every failure.

The mechanism of the failure is the point, and it is not carelessness. The old
table was a considered decision; a LATER ruling replaced it; the replaced copy
lived in a file nothing compared against the ruling. A sweep alone repeats that:
the next ruling will have the same reach and the same silence. This check is the
comparison that was missing, and it compares against the ladder's own table -
READ by import from tools/bench/qcx_runner.py, never retyped here, on the same
one-fact-one-home rule the DAG/Doxyfile guard follows.

THE RULE, exactly. A gate site is COMPLIANT when the legs it binds are a RUNG of
the ladder, and VIOLATING when they are off the ladder and undeclared. Read that
sentence again before re-deriving the rule from this file's history, because the
repo carries two wrong readings of it already:

  - "tight is bad" - false. The tight preset is a rung; a site AT that rung is
    an intended configuration.
  - "tighter than the operating default must be declared" - false, and it was
    this file's FIRST contract. It flagged the ladder itself, so the ladder could
    not land. The default is one rung of three, not the rule.
  - "equal legs are fine" - false. An equal-legs pair (1e-10/1e-10 is what the
    table it retired encoded) is never a rung, because every rung's density leg
    is 10^2 looser than its energy leg. An off-ladder site means exactly that.

A leg set to 0 (or negative) DISABLES it - the walk runs on to the cap - so a
disabled leg is a WILDCARD here: it takes part in no rung match, either way. That
is the loosest a leg can be, and reporting it as off-ladder would be a false
positive that invites deleting this check.

WHAT IT CHECKS. A "gate site" is a tolerance VALUE bound to a tolerance NAME, in
one of two shapes:

  A. a binding   - `energyTolerance|densityTolerance|energy_tolerance|density_tolerance`
                   assigned a numeric literal (a C++ option field, a TOML [scf] key,
                   as in `"...\ndensity_tolerance = 1e-10"`). The name must stand
                   alone: `kFixtureEnergyTolerance` is an assertion band, not a
                   gate, and does not match.
  B. a table entry - a `"key": <float>` or `"key": (<float>, <float>)` line inside a
                   mapping assigned to a name containing `tol` (`_PRESET_TOLERANCES`,
                   `_PRESET_CONV_TOL`). That is the shape the 2026-09-15 defect had.

An energy leg immediately followed by a density leg - in the file's own site
order - is ONE gate, matched as a pair; a lone leg is matched against its own leg
of each rung. A pyscf-style `conv_tol` table is lone ENERGY legs by construction:
pyscf's conv_tol is its energy leg (see tools/bench/pyscf_runner.py).

Declarations live in one home, `tools/gate_sites_allowlist.txt`, one line per
off-ladder LEG: `path :: field :: value`. The reason does NOT live there - it
lives in the code, in the comment the declaration is checked against (see
below), so there is one home per fact.

SCOPE, stated so this is not read as proving more than it does. It covers those
two shapes over first-party TEST, BENCHMARK and TOOLS paths. Its costs, named
rather than discovered later:

  - a gate set by an expression or a variable, configured at runtime, or living in
    production source (`scf/include/**` declares the default itself, which this
    reads rather than scans) is SILENT here;
  - a tight gate held in a file-local `k`-constant is OUT OF SHAPE and silent
    (four_cell's `kDiagnosticEnergyTolerance` is the live example);
  - a leg NAMED IN PROSE is not a site: `//`, `#` or `*` comment lines are skipped,
    because the shipped comment `// a bare integer (energy_tolerance = 1) is
    unambiguous` (io/src/parse_input.cpp) is not a gate, and the old
    tighter-than-the-default rule only escaped flagging it because 1.0 is loose;
  - a BARE INTEGER LEG is not a site either - shape A binds only a
    FLOATING-POINT literal (one carrying `.` or an exponent). That is the same
    exclusion as the prose one above, applied to the same phrase where it stands
    in a fixture rather than a comment: io/tests/parse_input_test.cpp feeds
    `"[scf]\nenergy_tolerance = 1\n"` to the PARSER to assert that an integer is
    accepted where a double is expected. A convergence tolerance is a real
    number; no run is configured at `1`. Shape B's table entries are NOT
    narrowed this way - only shape A is, which is where that fixture lives;
  - a REGISTERED NON-GATE is not a site: the shapes above are a shape test, and a
    shape cannot tell a run's configuration from an operand of the predicate
    under test. `options.energyTolerance = 1e-10` (a declared real gate,
    scf/tests/robustness_uhf_test.cpp:469) and `twoWay.energyTolerance = 1e-6`
    (the RobustGate truth table's input, :487) are syntactically identical. The
    second is named in `tools/gate_sites_not_gates.txt`, with its reason in a
    comment beside it, and is skipped here. That register is not an exemption
    from the ladder - it is a statement that the value configures no run;
  - a gate whose two legs sit apart in the file, with another gate site between
    them, is judged leg by leg rather than as a pair: each leg must then be a rung
    value of its own leg, which is the weaker of the two judgements.

A gate outside these shapes is silent here; that is a narrower claim than "no
off-ladder gate exists", and it is the honest one.

Exit 0 when every gate site is a rung of the ladder or declared and carries an
adjacent comment, 1 otherwise.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = ROOT / "tools" / "gate_sites_allowlist.txt"

#: The OTHER register: sites the two shapes below match, which are NOT convergence
#: gates at all. A separate list, not lines in the allowlist, because the
#: allowlist's meaning is "the owner signed off on this off-ladder TOLERANCE" -
#: putting a predicate operand on it would say something false about the value.
#: Same `path :: field :: value` key, same 25-line reason requirement, so a
#: registration whose reason is deleted fails exactly as a declaration does.
#:
#: This is a SCOPE narrowing, not an exemption from the ladder. The line between
#: the two was measured on 2026-09-16 rather than assumed: a real gate and a
#: truth-table operand can be syntactically identical (`options.energyTolerance =
#: 1e-10` against `twoWay.energyTolerance = 1e-6`), so no shape rule can separate
#: them and the separation has to be stated per site by a reader.
NOT_GATES = ROOT / "tools" / "gate_sites_not_gates.txt"

#: The ladder's ONE home. Its table is read by import, never retyped: a second
#: hand-typed copy of a value whose home is elsewhere is the defect class this
#: check answers (a000a864), and in the checker it would be the same mistake.
LADDER_HOME = ROOT / "tools" / "bench" / "qcx_runner.py"
LADDER_VARIABLE = "_PRESET_TOLERANCES"

#: How much looser the density leg is than the energy leg, on every rung - the
#: rule the ladder is built on. Checked against the table rather than assumed,
#: because the defect it replaced was setting the two legs EQUAL.
DENSITY_LOOSENESS = 100.0

#: The operating default's ONE home per option set. Both are read and must agree,
#: and the default must BE a rung of the ladder: a default that differs between
#: the option structs, or that has left the ladder, is a second way the contract
#: drifts - which is what this check exists for.
DEFAULT_HOMES = (
    ROOT / "scf" / "include" / "qcx" / "scf" / "rhf.hpp",
    ROOT / "scf" / "include" / "qcx" / "scf" / "uhf.hpp",
)

#: Which leg a lone value bound to this name configures. `energy_tolerance` and
#: `density_tolerance` name their leg; a bare float in a `tol` table is a single
#: convergence tolerance and follows pyscf's own reading of it, the energy leg
#: (see tools/bench/pyscf_runner.py).
ENERGY_FIELDS = ("energyTolerance", "energy_tolerance")
DENSITY_FIELDS = ("densityTolerance", "density_tolerance")

#: How far above a site a comment may sit and still be "its" reason. The kept
#: sites in scf/tests/** carry 10-15 line comment blocks with a statement or two
#: between the comment and the assignment, so a strictly-adjacent rule would
#: flag the very sites the packet holds up as the model.
REASON_WINDOW = 25

#: The roots a gate site may live in. First-party only: the vendored trees carry
#: their own tolerances and the fleet does not own them.
SKIP_ROOTS = ("third_party/", "external/", "build/", ".git/")
SCAN_SUFFIXES = (".cpp", ".hpp", ".h", ".py", ".toml")

#: The two files whose CONTENT IS the banned spelling, so scanning them is a
#: category error rather than a hole: this checker's docstring names the shape it
#: must catch (`"...\ndensity_tolerance = 1e-10"`), and its test's regression
#: fixtures ARE the pre-fix tables verbatim - that literalness is the whole force
#: of the regression, so "fixing" these would gut the test rather than the defect.
#: The exemption is pinned narrow by the test's own case: an off-ladder gate in
#: any OTHER file of this directory still fires.
SELF = ("tools/check_gate_tolerances.py", "tools/check_gate_tolerances_test.py")

_BINDING = re.compile(r"([A-Za-z_]\w*)\s*=\s*([0-9][0-9eE.+-]*)")
_COMMENT = re.compile(r"^\s*(//|#|/\*|\*)")
_TABLE_OPEN = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*\{\s*$")
_ENTRY_SCALAR = re.compile(r"""^\s*["']([^"']+)["']\s*:\s*([0-9][0-9eE.+-]*)\s*,?\s*$""")
_ENTRY_PAIR = re.compile(
    r"""^\s*["']([^"']+)["']\s*:\s*\(\s*([0-9]+[0-9eE.+-]*)\s*,\s*([0-9]+[0-9eE.+-]*)\s*\)"""
)
_CPP_DEFAULT = r"double\s+{}\s*=\s*([0-9][0-9eE.+-]*)"

_LADDER_CACHE: dict[str, dict[str, tuple[float, float]]] = {}


class Site:
    """One tolerance value bound to a tolerance name, parsed."""

    def __init__(self, path: str, line: int, field: str, value: float) -> None:
        self.path = path
        self.line = line
        self.field = field
        self.value = value

    @property
    def leg(self) -> str:
        return _leg(self.field)

    @property
    def key(self) -> str:
        return f"{self.path} :: {self.field} :: {_fmt(self.value)}"


def _fmt(value: float) -> str:
    """A tolerance spelled one way, so a declaration and a site compare equal."""
    return f"{value:.0e}"


def _is_float_literal(text: str) -> bool:
    """Whether a bound literal is a real number rather than a bare integer.

    Shape A binds a tolerance; `1` is not one, and the parser fixture that writes
    it (`energy_tolerance = 1`) is asserting integer ACCEPTANCE, not configuring a
    run. A literal carrying `.` or an exponent is a real; `1.` and `1e0` and
    `1.0e0` all pass, so nothing a reader would call a tolerance is lost.
    """
    return "." in text or "e" in text or "E" in text


def _rel(path: Path) -> str:
    """A path as the tree spells it, in a message that survives either root."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def ladder() -> dict[str, tuple[float, float]]:
    """The convergence-gate ladder, READ from its one home - never retyped here.

    Every entry of the home's table is a rung. Three properties of the table are
    checked rather than trusted, because each is a way the contract can rot while
    still importing cleanly: the legs must be positive, the density leg must be
    `DENSITY_LOOSENESS` times the energy leg (the defect this replaced was
    making them equal), and no two presets may carry the same pair (the first
    "correction" flattened all three onto one pair, which erased the ladder).
    """
    key = str(LADDER_HOME)
    if key in _LADDER_CACHE:
        return _LADDER_CACHE[key]
    if not LADDER_HOME.exists():
        raise SystemExit(f"gate-tolerance check: {_rel(LADDER_HOME)} is missing, so the"
                         " ladder has no readable home")
    spec = importlib.util.spec_from_file_location("_qcx_ladder_home", LADDER_HOME)
    if spec is None or spec.loader is None:
        raise SystemExit(f"gate-tolerance check: {_rel(LADDER_HOME)} cannot be imported")
    home = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(home)
    except Exception as error:  # reported, never swallowed: a silent home is the defect
        raise SystemExit(f"gate-tolerance check: importing {_rel(LADDER_HOME)} to read"
                         f" {LADDER_VARIABLE} failed: {error!r}")
    table = getattr(home, LADDER_VARIABLE, None)
    if not isinstance(table, dict) or not table:
        raise SystemExit(f"gate-tolerance check: {_rel(LADDER_HOME)} has no non-empty"
                         f" {LADDER_VARIABLE} table; the ladder moved or was renamed,"
                         " and this check will not guess it")

    rungs: dict[str, tuple[float, float]] = {}
    for preset, pair in table.items():
        if not isinstance(preset, str) or not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise SystemExit(f"gate-tolerance check: {LADDER_VARIABLE}[{preset!r}] is not a"
                             " (energy, density) pair")
        energy, density = float(pair[0]), float(pair[1])
        if energy <= 0.0 or density <= 0.0:
            raise SystemExit(f"gate-tolerance check: {LADDER_VARIABLE}[{preset}] carries a"
                             " non-positive leg; a disabled leg is not a rung")
        if not math.isclose(density, energy * DENSITY_LOOSENESS, rel_tol=1e-9):
            raise SystemExit(
                f"gate-tolerance check: {LADDER_VARIABLE}[{preset}] is {energy!r}/{density!r}"
                f" - the density leg must be {DENSITY_LOOSENESS:.0f}x the energy leg."
                " Equal legs are the defect the ladder replaced, and a"
                " table that encodes it again is what this check exists to catch."
            )
        rungs[preset] = (energy, density)
    if len(set(rungs.values())) != len(rungs):
        raise SystemExit(f"gate-tolerance check: two rungs of {LADDER_VARIABLE} carry the"
                         " same (energy, density) pair; the presets are supposed to differ.")
    _LADDER_CACHE[key] = rungs
    return rungs


def operating_default() -> dict[str, float]:
    """The operating default, READ from its homes and checked onto the ladder.

    A second hand-typed copy of a value whose home is elsewhere is the defect
    class this check answers (a000a864), so the default is never retyped here -
    and the two option structs must agree on it, or "the default" is two facts.
    The default is itself a gate, so it must be a rung: the ruling makes it the
    kNormal rung, and a default that has left the ladder is a policy decision,
    not a local one, so it is reported rather than quietly re-baselined.
    """
    found: dict[str, float] = {}
    for path in DEFAULT_HOMES:
        if not path.exists():
            raise SystemExit(f"gate-tolerance check: {_rel(path)} is missing, so the"
                             " operating default has no readable home")
        text = path.read_text(encoding="utf-8")
        for leg, name in (("energy", "energyTolerance"), ("density", "densityTolerance")):
            match = re.search(_CPP_DEFAULT.format(name), text)
            if match is None:
                raise SystemExit(f"gate-tolerance check: no `double {name} = <value>` in"
                                 f" {_rel(path)}; the default moved or was renamed, and"
                                 " this check will not guess it")
            value = float(match.group(1))
            if leg in found and found[leg] != value:
                raise SystemExit(
                    f"gate-tolerance check: the {leg} default disagrees across its homes"
                    f" ({found[leg]!r} vs {value!r} in {_rel(path)}). One operating"
                    " default, or none."
                )
            found[leg] = value
    rungs = ladder()
    pair = (found["energy"], found["density"])
    if pair not in rungs.values():
        raise SystemExit(
            f"gate-tolerance check: the operating default {_fmt(pair[0])}/{_fmt(pair[1])} is"
            f" not a rung of the ladder ({', '.join(_rung_label(n, r) for n, r in rungs.items())})"
            f" read from {_rel(LADDER_HOME)}. Either the default has left the ladder - an"
            " ruling on the tolerances, not a local edit - or the ladder is stale."
        )
    return found


def _keys(path: Path) -> set[str]:
    """The `path :: field :: value` keys of one register, comments stripped."""
    if not path.exists():
        return set()
    keys = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            keys.add(line)
    return keys


def declared() -> set[str]:
    """The declared off-ladder sites, from their one home."""
    return _keys(ALLOWLIST)


def not_gate_keys() -> set[str]:
    """The registered non-gate sites, from their own register.

    These match the shapes below and configure no run. Read through the same key
    format and held to the same reason-comment rule as a declaration, so a
    registration whose reason is deleted fails rather than silently widening the
    checker's blind spot.
    """
    return _keys(NOT_GATES)


def sites_in(path: str, text: str) -> list[Site]:
    """Every gate site the two shapes describe, in file order."""
    found: list[Site] = []
    lines = text.splitlines()
    table_depth = 0
    table_name = ""

    for number, line in enumerate(lines, start=1):
        opened = _TABLE_OPEN.match(line)
        if opened is not None and table_depth == 0 and re.search(r"tol", opened.group(1), re.I):
            table_name = opened.group(1)
            table_depth = 1
            continue

        if table_depth:
            scalar = _ENTRY_SCALAR.match(line)
            pair = _ENTRY_PAIR.match(line)
            if scalar is not None:
                found.append(Site(path, number, f"{table_name}[{scalar.group(1)}].energy",
                                  float(scalar.group(2))))
            elif pair is not None:
                found.append(Site(path, number, f"{table_name}[{pair.group(1)}].energy",
                                  float(pair.group(2))))
                found.append(Site(path, number, f"{table_name}[{pair.group(1)}].density",
                                  float(pair.group(3))))
            table_depth += line.count("{") - line.count("}")
            if table_depth <= 0:
                table_depth = 0
                table_name = ""
            continue

        if _COMMENT.match(line):
            # A tolerance NAMED IN PROSE is not a gate: io/src/parse_input.cpp's
            # `// a bare integer (energy_tolerance = 1) is unambiguous` is a
            # comment about the parser's number handling, not a convergence gate.
            # The retired tighter-than-the-default rule never had to draw this
            # line (1.0 is loose); the ladder rule does, so it is drawn here.
            continue

        for match in _BINDING.finditer(line):
            # Shape A binds a FLOATING-POINT literal only. A bare integer is not
            # a convergence tolerance, and the fixture that proves the parser
            # accepts one (io/tests/parse_input_test.cpp, `energy_tolerance = 1`)
            # is the case this excludes - the same phrase the prose rule above
            # already skips in its shipped-comment form.
            if not _is_float_literal(match.group(2)):
                continue
            named = _tolerance_name(match.group(1), line, match.start(1))
            if named is not None:
                found.append(Site(path, number, _canonical(*named), float(match.group(2))))

    return found


def gates(sites: list[Site]) -> list[list[Site]]:
    """The sites grouped into gates: an energy leg then a density leg is one gate.

    Site order is the file's own order, so a comment or an unrelated statement
    between the two legs does not split the gate. A gate whose legs sit apart,
    with another GATE SITE between them, is split and judged leg by leg - the
    weaker judgement, and it is stated in the docstring rather than hidden.
    """
    grouped: list[list[Site]] = []
    index = 0
    while index < len(sites):
        first = sites[index]
        second = sites[index + 1] if index + 1 < len(sites) else None
        if (first.leg == "energy" and second is not None and second.leg == "density"):
            grouped.append([first, second])
            index += 2
        else:
            grouped.append([first])
            index += 1
    return grouped


def _rung_label(preset: str, rung: tuple[float, float]) -> str:
    return f"{preset} {_fmt(rung[0])}/{_fmt(rung[1])}"


def _ladder_label(rungs: dict[str, tuple[float, float]]) -> str:
    return ", ".join(_rung_label(preset, rung) for preset, rung in rungs.items())


def _gate_label(gate: list[Site]) -> str:
    energy = next((site.value for site in gate if site.leg == "energy"), None)
    density = next((site.value for site in gate if site.leg == "density"), None)
    if energy is not None and density is not None:
        return f"{_fmt(energy)}/{_fmt(density)}"
    if energy is not None:
        return f"energy {_fmt(energy)}"
    return f"density {_fmt(density)}"


def on_ladder(gate: list[Site], rungs: dict[str, tuple[float, float]]) -> bool:
    """Whether some rung matches every leg this gate actually binds.

    A disabled leg (<= 0) binds nothing: it takes part in no rung match either
    way, so a gate with one disabled leg is judged on the leg it does bind.
    """
    binds = {site.leg: site.value for site in gate if site.value > 0.0}
    if not binds:
        return True
    return any(
        all(binds[leg] == (rung[0] if leg == "energy" else rung[1]) for leg in binds)
        for rung in rungs.values()
    )


def has_adjacent_comment(text: str, site: Site) -> bool:
    """A comment within the window above the site.

    This does NOT judge whether the comment is a good reason - only a reader can.
    It fails when the reason is GONE, which is the other way a kept gate rots.
    """
    lines = text.splitlines()
    start = max(0, site.line - 1 - REASON_WINDOW)
    return any(_COMMENT.match(line) for line in lines[start:site.line - 1])


def _leg(field: str) -> str:
    for name in DENSITY_FIELDS:
        if field.endswith(name) or f"].density" in field or name in field:
            return "density"
    return "energy"


def _tolerance_name(name: str, line: str, start: int) -> tuple[str, str] | None:
    """The (leg, spelling) a bound identifier names, or None when it names no gate.

    The identifier must BE the field name, not merely end with it. That line is
    drawn on measured cases, because both sides of it are real:

    - `energyTolerance` / `energy_tolerance` ARE gates (a C++ option field and
      the TOML [scf] key). A fixture that writes both keys on one physical C++
      line spells the second `"...\\ndensity_tolerance = 1e-10"`, so a leading
      `n` left by the escape is stripped before the compare.
    - `kEnergyTolerance = 1e-12` (storage/tests/scf_checkpoint_test.cpp:47) is
      NOT: it is an EXPECT_NEAR assertion BAND, its only use. `kFixture...`,
      `kOperating...` and `kTrajectoryDensityBand` are bands for the same
      reason. A suffix rule flagged the checkpoint constant as a tight gate -
      a false positive whose only repair would be a declaration saying "this is
      not a gate", which is not what the allowlist means.

    The cost of that line, stated rather than discovered later: an off-ladder
    gate held in a file-local `k`-constant is OUT OF SHAPE and silent here.
    four_cell's `kDiagnosticEnergyTolerance = 1e-12` is the live example - a
    real gate, with a 20-line in-line reason, that this check does not cover.
    """
    if start > 0 and line[start - 1] == "\\" and name.startswith("n"):
        name = name[1:]
    flat = name.lower().replace("_", "")
    if flat == "energytolerance":
        return "energy", name
    if flat == "densitytolerance":
        return "density", name
    return None


def _canonical(leg: str, spelling: str) -> str:
    """The leg's field name in the file's own spelling, so a declaration key is
    stable and readable rather than the raw match."""
    if "_" in spelling:
        return f"{leg}_tolerance"
    return f"{leg}Tolerance"


def violations(sources: dict[str, str], rungs: dict[str, tuple[float, float]],
               enforce: dict[str, set[int]] | None = None
               ) -> tuple[list[str], int, int, int, list[str]]:
    """Undeclared off-ladder legs, the on-ladder count, the off-ladder count, the
    declared count, and the off-ladder legs this run is NOT enforcing.

    `enforce` is the set of line numbers the commit ADDS, per path. When it is
    given, only those lines can fail the run and the rest are returned as
    pre-existing. The blast radius is then the offender's commit rather than the
    committing author's - an unrelated hunk in a file that already carried
    an off-ladder gate is not blocked by a condition it did not create, and the
    introduction of one is still caught at the moment it is introduced. `None`
    means enforce everywhere, which is the audit reading (--all).
    """
    declared_keys = declared()
    not_gate = not_gate_keys()
    errors: list[str] = []
    preexisting: list[str] = []
    on_ladder_count = 0
    off_ladder = 0
    declared_and_seen = 0

    for path in sorted(sources):
        if (not path.endswith(SCAN_SUFFIXES) or path.startswith(SKIP_ROOTS)
                or path in SELF):
            continue
        text = sources[path]
        for gate in gates(sites_in(path, text)):
            if on_ladder(gate, rungs):
                on_ladder_count += len([s for s in gate if s.value > 0.0])
                continue
            for site in gate:
                # A disabled leg is a wildcard, not the tightest reading of one.
                if site.value <= 0.0:
                    continue
                # A registered non-gate is skipped BY SCOPE, before any ladder
                # judgement: it is not an off-ladder gate but not a gate at all,
                # so it must not inflate the off-ladder count either. Its reason
                # is still required - a registration nobody explained is how this
                # register would become a hole instead of a narrowing.
                if site.key in not_gate:
                    if not has_adjacent_comment(text, site):
                        errors.append(
                            f"{site.path}:{site.line}: {site.field} = {_fmt(site.value)} is"
                            " registered as NOT A GATE but its in-line reason is gone - no"
                            f" comment in the {REASON_WINDOW} lines above it. Restore the"
                            " reason, or delete the registration and let the site be judged."
                        )
                    continue
                off_ladder += 1
                if site.key in declared_keys:
                    declared_and_seen += 1
                    if not has_adjacent_comment(text, site):
                        errors.append(
                            f"{site.path}:{site.line}: {site.field} = {_fmt(site.value)} is"
                            " DECLARED but its in-line reason is gone - no comment in the"
                            f" {REASON_WINDOW} lines above it. Restore the reason, or move the"
                            " gate onto the ladder and drop the declaration."
                        )
                    continue
                message = (f"{site.path}:{site.line}: {site.field} = {_fmt(site.value)}: the"
                           f" gate {_gate_label(gate)} is off the ladder"
                           f" ({_ladder_label(rungs)}) and this leg is not declared")
                if enforce is not None and site.line not in enforce.get(site.path, set()):
                    preexisting.append(message)
                    continue
                errors.append(message)

    return errors, on_ladder_count, off_ladder, declared_and_seen, preexisting


def added_lines() -> dict[str, set[int]]:
    """The line numbers this commit ADDS, per path, in new-file numbering."""
    # The encoding is EXPLICIT, and errors are REPLACED, on every git decode below.
    # git's output is UTF-8 and the interpreter's default here is the system ANSI
    # codepage (cp1250), which leaves 0x81/0x83/0x88/0x90/0x98 undefined - and a
    # UTF-8 continuation byte can be any of those, so ONE character in any diff
    # (U+2248's trailing 0x88 did it, in a row another author had written) raised
    # UnicodeDecodeError inside the reader thread and left the decoded diff None:
    # the check then CRASHED on an unrelated commit instead of reporting. Same
    # reading as from_tree()'s file reads. Measured 2026-09-15.
    diff = subprocess.run(
        ["git", "diff", "--cached", "-U0", "--diff-filter=ACMR", "--no-color"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout
    added: dict[str, set[int]] = {}
    current = ""
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current = line[len("+++ b/"):]
            added.setdefault(current, set())
        elif line.startswith("@@") and current:
            match = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if match is not None:
                start = int(match.group(1))
                count = int(match.group(2) or 1)
                added[current].update(range(start, start + count))
    return added


def from_index() -> dict[str, str]:
    """The staged blobs: the commit's own content, so one author's in-flight gate
    does not block everyone else (the lesson of 2026-09-13)."""
    listing = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout.split()
    sources = {}
    for name in listing:
        blob = subprocess.run(["git", "show", f":{name}"], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", errors="replace")
        if blob.returncode == 0:
            sources[name] = blob.stdout
    return sources


def from_tree() -> dict[str, str]:
    """Every tracked first-party file - the negative control's reading."""
    listing = subprocess.run(["git", "ls-files"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", errors="replace", check=True).stdout.split()
    sources = {}
    for name in listing:
        if name.endswith(SCAN_SUFFIXES) and not name.startswith(SKIP_ROOTS):
            path = ROOT / name
            if path.is_file():
                sources[name] = path.read_text(encoding="utf-8", errors="replace")
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="scan the whole tracked tree, not just the staged paths")
    args = parser.parse_args()

    rungs = ladder()
    defaults = operating_default()
    sources = from_tree() if args.all else from_index()
    enforce = None if args.all else added_lines()
    errors, on_ladder_count, off_ladder, seen, preexisting = violations(sources, rungs, enforce)

    if errors:
        print("gate-tolerance check failed - a convergence gate must be a rung of the")
        print("ladder, or be declared with its reason. The ladder (energy/density per")
        print(f"preset, every rung's density leg 10^2 looser than its energy leg, read from"
              f" {_rel(LADDER_HOME)}):")
        for preset, rung in rungs.items():
            print(f"  {_rung_label(preset, rung)}")
        print("An off-ladder gate is a configuration nobody ruled - equal legs such as")
        print("1e-10/1e-10 are the shape it takes. Keep one only when a measured property")
        print("the ladder cannot assert depends on it: write WHY in a comment above the")
        print("site, then add `path :: field :: value` to tools/gate_sites_allowlist.txt.")
        for error in errors:
            print(f"  {error}")
        return 1

    scope = "the tracked tree" if args.all else "the lines this commit adds"
    print(f"gate-tolerance OK over {scope}: {on_ladder_count} gate leg(s) on the ladder,"
          f" {off_ladder} off it and declared ({seen} declared, {len(not_gate_keys())} leg(s)"
          f" registered as NOT A GATE, {REASON_WINDOW}-line comment"
          f" window); ladder {_ladder_label(rungs)} read from {_rel(LADDER_HOME)}; operating"
          f" default energy {_fmt(defaults['energy'])} / density {_fmt(defaults['density'])}"
          f" - a rung - read from {len(DEFAULT_HOMES)} homes")
    if preexisting:
        # Reported rather than enforced: these predate this commit, and failing on
        # them would block an author for a condition they did not create (the
        # lesson of 2026-09-13). Visible either way - silence is how the last one
        # lasted a fortnight.
        print(f"  {len(preexisting)} pre-existing off-ladder leg(s) in the files this"
              " commit touches, NOT failed here:")
        for note in preexisting:
            print(f"    {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
