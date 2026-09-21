#!/usr/bin/env python3
"""Enforce the module DAG: a module may not reference one that comes after it in ORDER.

Each module's CMakeLists.txt is scanned for `qcx-<module>` references; a reference
from a module to one that comes *after* it in the ordering is a violation.

ORDER below is the module list's ONLY home: tools/check_docs_gate.py imports it as
the docs gate's expected coverage, so it is read, never copied. The copy that used
to sit on this line had already drifted - it named 12 of the 14 modules (grid and
storage were missing).
"""

import re
import sys
from pathlib import Path

ORDER = ["core", "backend", "memory", "linalg", "molecule", "symmetry", "basisset", "grid", "integrals", "scf", "response", "storage", "io", "properties", "driver"]
RANK = {m: i for i, m in enumerate(ORDER)}


def deps_of(module: str) -> list[str]:
    cml = Path(module) / "CMakeLists.txt"
    if not cml.exists():
        return []
    text = cml.read_text()
    return [m for m in ORDER if re.search(rf"qcx-{m}\b", text) and m != module]


def main() -> int:
    violations = []
    for module in ORDER:
        for dep in deps_of(module):
            if RANK[dep] > RANK[module]:
                violations.append(f"{module} -> {dep} violates DAG order")

    if violations:
        print("DAG violations found:")
        for v in violations:
            print(f"  {v}")
        return 1
    print("DAG OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
