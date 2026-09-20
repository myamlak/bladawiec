#!/usr/bin/env python3
"""The docs gate must scan every module it claims to gate, and must still be able to fail.

WHY THIS EXISTS, measured. `qcx-docs` is the authoritative docs gate
("zero warnings, EXTRACT_ALL must stay NO"), and its coverage is exactly the
Doxyfile INPUT list. On 2026-09-15 that list named 13 of the 14 DAG modules:
`scf/include` was absent, so the gate reported the same zero warnings whether scf
was documented or not - a whole module outside the gate with the gate green over
it (3908e583). The Doxyfile comment states the rule; nothing enforced it, and the
second, hand-typed copy of a list whose home is tools/check_dag.py is what
drifted. This check is that enforcement.

WHAT IT CHECKS (all static - no Doxygen run, no build, no test binary):

1. INPUT covers the DAG. Every module in tools/check_dag.py's ORDER that owns an
   `include/` directory must appear in INPUT as `<module>/include`. The list is
   IMPORTED from check_dag.py, never retyped here, because a third copy would be
   a third thing to drift. MISSING and EXTRA are reported as different mistakes,
   because they are different: a module absent from INPUT is an ungated module; an
   entry INPUT names that is not a DAG module is a stale entry, a typo, or an
   addition the Doxyfile's own comment forbids (excgrid is the deliberate
   exception, and it owns an include/ directory - it is the one entry that must
   NOT be there).
2. Every entry exists under the repo root. An entry naming a path that is not
   there covers nothing, whatever Doxygen makes of it. Each entry is reported
   once, under its primary class - an entry already named as EXTRA or MALFORMED
   is not repeated as ABSENT.
3. The gate can still fail: WARN_AS_ERROR must be declared and must not be NO.
   With NO, Doxygen prints the warnings and exits 0, so the target succeeds over
   a red tree - the 2026-08-24 defect that setting was adopted to close.

The gate's failure MODE is printed with the pass rather than checked, so the
verdict travels with what it is worth. WARN_AS_ERROR=YES stops Doxygen at the
FIRST warning, so a red run names one warning per run and never the full list
(measured 2026-09-15: two warnings were red in HEAD and the gate showed one; a
WARN_AS_ERROR=NO pass was needed to see the second). A GREEN run is still zero
warnings - the abort exits non-zero, so nothing is hidden from the verdict, only
from the diagnosis. FAIL_ON_WARNINGS, which Doxygen documents as reporting every
warning and then exiting non-zero, is accepted here but not adopted: the mode
change needs a real qcx-docs run to verify, and this check never builds.

SCOPE, stated so this is not read as proving more than it does: it proves the
gate's INPUT covers the DAG and that the paths are real. It does not prove the
headers parse without warnings - that is the qcx-docs target itself.

Exit 0 when the gate's coverage and mode are sound, 1 otherwise.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOXYFILE = ROOT / "Doxyfile"
DAG_CHECKER = Path(__file__).resolve().parent / "check_dag.py"

INCLUDE = "include"
INPUT_TAG = re.compile(r"\s*INPUT\s*=\s*(.*)$")
MODE_TAG = re.compile(r"^\s*WARN_AS_ERROR\s*=\s*(\S+)", re.MULTILINE)

#: The WARN_AS_ERROR values that keep the gate able to fail a build. YES is the
#: one in force; the two FAIL_ON_WARNINGS modes are Doxygen's own (read from its
#: manual, not measured here - see the module docstring).
FATAL_MODES = ("YES", "FAIL_ON_WARNINGS", "FAIL_ON_WARNINGS_PRINT")

#: What the printed verdict is worth under each mode, so a green run carries the
#: strength of its own claim. The FAIL_ON_WARNINGS lines are read from the
#: Doxygen manual; the YES line is measured (see the module docstring).
MODE_NOTES = {
    "YES": "a red run names the first warning only, not the full list (measured 2026-09-15)",
    "FAIL_ON_WARNINGS": "every warning is reported (read from the Doxygen manual)",
    "FAIL_ON_WARNINGS_PRINT": "every warning is reported (read from the Doxygen manual)",
}


def dag_order() -> list[str]:
    """The DAG module list, read from its one home: tools/check_dag.py's ORDER."""
    spec = importlib.util.spec_from_file_location("check_dag", DAG_CHECKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"not importable: {DAG_CHECKER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [str(name) for name in module.ORDER]


def input_entries(text: str) -> list[str]:
    """Every path the Doxyfile's INPUT tag names, in file order.

    Doxygen accumulates a repeated tag and splits list values on whitespace or
    commas, so all three forms are read: the checked-in file happens to use one
    aligned line, and a reformat must not turn this check into a scan of nothing.
    """
    entries: list[str] = []
    for line in text.splitlines():
        match = INPUT_TAG.match(line)
        if match is None:
            continue
        value = match.group(1).split("#")[0]
        entries.extend(token for token in re.split(r"[,\s]+", value) if token)
    return entries


def gate_mode(text: str) -> str:
    """The declared WARN_AS_ERROR value, or 'absent' when the tag is not there."""
    match = MODE_TAG.search(text)
    return match.group(1) if match else "absent"


def mode_errors(text: str) -> list[str]:
    """WARN_AS_ERROR must be declared and must be a mode that fails a build."""
    mode = gate_mode(text)
    if mode == "absent":
        return [
            "WARN_AS_ERROR is not declared in the Doxyfile, so the gate's failure"
            " mode is whatever Doxygen defaults to. Declare it (YES) - a gate whose"
            " ability to fail is inherited rather than stated is a gate nobody"
            " checks."
        ]
    if mode.upper() == "NO":
        return [
            "WARN_AS_ERROR = NO: Doxygen prints the warnings and exits 0, so"
            " qcx-docs succeeds over a red tree - the 2026-08-24 defect this"
            " setting was adopted to close. Use a mode that fails the build."
        ]
    if mode.upper() not in FATAL_MODES:
        return [
            f"WARN_AS_ERROR = {mode} is not a Doxygen mode"
            f" ({', '.join(FATAL_MODES)}). A typo here disables the gate quietly."
        ]
    return []


def coverage_errors() -> list[str]:
    """Every reason the docs gate cannot be trusted to have covered the DAG."""
    try:
        order = dag_order()
    except Exception as exc:  # reported, never raised - an unreadable list is a finding
        return [f"cannot read the module list from {DAG_CHECKER.name}: {exc}"]

    if not DOXYFILE.exists():
        return [f"{DOXYFILE.name} is missing"]

    text = DOXYFILE.read_text(encoding="utf-8")
    entries = input_entries(text)
    if not entries:
        return [
            "INPUT names nothing: with no INPUT the gate scans no file at all and"
            " reports zero warnings over the whole repository. Restore the module"
            " list (see the comment above the tag in Doxyfile)."
        ]

    modules = set(order)
    owning = [name for name in order if (ROOT / name / INCLUDE).is_dir()]
    named = [entry for entry in entries if entry.endswith(f"/{INCLUDE}")]
    named_modules = {entry[: -len(INCLUDE) - 1] for entry in named}
    mis_shaped: dict[str, str] = {}
    for entry in entries:
        name = entry.split("/")[0]
        if name in modules and not entry.endswith(f"/{INCLUDE}"):
            mis_shaped.setdefault(name, entry)

    errors: list[str] = []
    reported: set[str] = set()

    for module in owning:
        if module in named_modules or module in mis_shaped:
            continue
        errors.append(
            f"MISSING: {module}/{INCLUDE} is not in the Doxyfile INPUT, so the gate"
            f" reports the same zero warnings whether {module} is documented or not"
            f" (the scf hole, 3908e583). Add {module}/{INCLUDE} to INPUT."
        )

    for entry in named:
        module = entry[: -len(INCLUDE) - 1]
        if module in modules:
            continue
        reported.add(entry)
        errors.append(
            f"EXTRA: INPUT names {entry}, and {module} is not a module in"
            f" {DAG_CHECKER.name} (a stale entry after a rename, or a typo). Remove"
            f" it - and note excgrid is the deliberate exception, so its own"
            f" include/ directory must stay out of INPUT."
        )

    for module, entry in sorted(mis_shaped.items()):
        reported.add(entry)
        errors.append(
            f"MALFORMED: INPUT names {entry} for the module {module}. A module is"
            f" gated by its public headers, so under this INPUT {module}'s headers"
            f" are outside the gate. The entry must be {module}/{INCLUDE}."
        )

    # One entry is reported once, under its primary class: an entry already named
    # as EXTRA or MALFORMED does not need a second line saying its path is missing.
    for entry in entries:
        if entry in reported or (ROOT / entry).exists():
            continue
        errors.append(
            f"ABSENT: INPUT names {entry}, which does not exist under the repo"
            f" root, so the entry covers nothing. Fix the path or drop it."
        )

    errors.extend(mode_errors(text))
    return errors


def main() -> int:
    errors = coverage_errors()
    if errors:
        print("docs-gate check failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    mode = gate_mode(DOXYFILE.read_text(encoding="utf-8"))
    note = MODE_NOTES.get(mode.upper(), "mode not recognised")
    print(f"docs-gate OK: {len(dag_order())} DAG modules in the Doxyfile INPUT; "
          f"WARN_AS_ERROR={mode} - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
