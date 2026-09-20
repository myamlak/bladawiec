#!/usr/bin/env python3
"""The blank-line rule checker: reports C++
lines missing the mandated blank line before a multi-line control statement
or after its closing brace. The inverse of tools/add_control_blank_lines.py
(the 2026-08-16 migration sweep) - the same exemptions, so the checker flags
exactly what the sweep would have inserted: compact else/catch/do-while
chains (incl. the do-while `} while (...);` continuation), comment lines
attached to the statement's block (a comment directly above a control
statement or below its close), the first statement of a block,
closing-brace continuations (); / }, / };), nested closes, and blank/EOF
neighbors. `if constexpr/consteval/constinit` count as control statements.

Invoked on every commit: .githooks/pre-commit runs it via
tools/check_style_guards.py (G1) over the staged C++ files; the inverse
sweep is tools/add_control_blank_lines.py (the 2026-08-16 migration tool).
"""

import re
import sys
from pathlib import Path

CONTROL = re.compile(
    r"^(\s*)(if|for|while|switch|catch|do)"
    r"(\s*((constexpr|consteval|constinit)(\s*\(.*)?|\(.*))?$")
CLOSE = re.compile(r"^(\s*)\}\s*$")

VIOLATION_BEFORE = "missing blank line before control statement"
VIOLATION_AFTER = "missing blank line after multi-line control statement"


def violations(text: str) -> list:
    """[(lineNumber1, description)] - the blank lines the sweep would add."""
    lines = text.split("\n")
    found = []

    for i, line in enumerate(lines):
        prev = lines[i - 1] if i >= 1 else ""
        nxt = lines[i + 1] if i + 1 < len(lines) else ""

        if CONTROL.match(line) and not (
            not prev
            or prev.strip() == ""
            or prev.strip().endswith("{")
            or re.match(r"^\s*else\b", prev)
            or re.match(r"^\s*\}\s*else\b", prev)
            or re.match(r"^\s*\}\s*$", prev)
            or re.match(r"^\s*//", prev)
            or re.match(r"^\s*#", prev)
            or re.match(r"^\s*(else|case|default)\b", line)
        ):
            found.append((i + 1, VIOLATION_BEFORE))

        close_match = CLOSE.match(line)
        if close_match and close_match.group(1):
            tail = nxt.strip()
            # Compact chain continuation: else / else-if, catch, and the
            # do-while `while (...);` on the following line (CLOSE requires
            # the bare `}` line, so `} while (...);` never reaches here).
            chain = (re.match(r"^\s*else\b", nxt) or re.match(r"^\s*catch\b", nxt)
                     or re.match(r"^\s*while\b", nxt))
            if not chain and not (
                not nxt
                or tail == ""
                or tail == "}"
                or re.match(r"^[);,}]", tail)
                or re.match(r"^\s*//", nxt)
                or re.match(r"^\s*#", nxt)
                or re.match(r"^\s*$", nxt)
            ):
                found.append((i + 1, VIOLATION_AFTER))

    return found


def main() -> int:
    bad = 0

    for name in sys.argv[1:]:
        path = Path(name)
        text = path.read_text(encoding="utf-8")
        for line_number, description in violations(text):
            print(f"{path}:{line_number}: {description}")
            bad += 1

    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
