#!/usr/bin/env python
"""Workspace-grant cohesion guard.

WHY THIS EXISTS, measured. The deleted predictive memory model supplied ONE
quantity to two consumers: its base term was both the pre-gate admission floor
and the reservation that turned a memory cap into a workspace grant. Every
budget site read `WorkspaceBudget::Create(cap - modeledBase)`. The deletion of
2026-09-17 re-derived the admission half as `CheckPreGateSetupAdmission` and
dropped the reservation half: all eight sites became `Create(cap)`. The effect
was measured on 2026-09-18 - the cheaper rungs became unreachable through the
driver at EVERY cap (RiFullFockRung::kBlocked needs the grant under 51,829,880 B
while the arm admits only cap >= 238,481,196 B on `ri_jk_propane`; the RI-J disk
rung needs it under 9,895,696 B while the process needs ~48 MB of cap), and two
driver tests went red.

The repair is one source consumed by eight: `integrals::SetupAdmissionReserveBytes`
(or the driver's `WorkspaceGrantBytes(cap, reserve)` built from it) at every
budget site. That rule lived only in a comment once, and a comment is what the
deletion was able to ignore. This check is the instruction to the machinery: a
budget site that stops consuming the shared grant fails here, in the same
commit, rather than in a driver suite much later.

WHAT IT CHECKS (mechanical, no execution, no build):
  1. Every `WorkspaceBudget::Create(` in driver/src/run_driver.cpp takes its
     argument from `WorkspaceGrantBytes(`. A site that goes back to
     `memoryCapGiB * kGiB` - or to any other locally-computed expression - fails.
  2. The number of such sites is the COUNT below. A ninth budget site added
     without a decision fails, so the count cannot drift silently; raise COUNT
     in the commit that adds one, with its consuming expression in (1).

Scope: `driver/src/run_driver.cpp` only - it is the sole builder of the driver's
workspace budgets (verified 2026-09-18: `grep -rn "WorkspaceBudget::Create" driver/`
returns this file alone).

Reports each offending site with its line, and exits 1 on any finding.
"""

import re
import sys
from pathlib import Path

TARGET = Path("driver/src/run_driver.cpp")

# The number of budget sites the tree is expected to carry. See (2) above.
COUNT = 8

# The one expression a budget site may be handed: the shared grant.
REQUIRED = "WorkspaceGrantBytes("


def main() -> int:
    if not TARGET.exists():
        print(f"workspace-grant: {TARGET} is missing")
        return 1

    lines = TARGET.read_text(encoding="utf-8", errors="replace").split("\n")
    findings = []
    sites = 0

    for index, line in enumerate(lines):
        if "WorkspaceBudget::Create(" not in line:
            continue

        # A line that names the call in prose is not a call site: this file's
        # own comments quote the pattern (and the deleted form of it) to explain
        # the repair, and a guard that counted those would fail on its own
        # documentation.
        if line.find("//") != -1 and line.find("//") < line.find("WorkspaceBudget::Create("):
            continue

        sites += 1
        # The argument may open on this line and close on the next one (the
        # 12-space sites wrap). Read the argument's first non-empty line.
        window = " ".join(part.strip() for part in lines[index : index + 3])
        if REQUIRED not in window:
            findings.append(
                (
                    index + 1,
                    REQUIRED,
                    line.strip() + " " + " ".join(p.strip() for p in lines[index + 1 : index + 3]),
                )
            )

    if sites != COUNT:
        print(
            f"workspace-grant: {TARGET} carries {sites} budget site(s), the guard "
            f"expects {COUNT}."
        )
        print(
            "  A site was added or removed. Every site must consume the shared "
            "grant - raise COUNT in tools/check_workspace_grant.py in the commit "
            "that adds one, and make it read WorkspaceGrantBytes(cap, reserve)."
        )
        return 1

    if findings:
        print(f"workspace-grant: {len(findings)} budget site(s) do not consume the shared grant:")
        for line_no, required, seen in findings:
            print(f"  {TARGET}:{line_no}: expected an argument from {required}")
            print(f"      saw: {seen[:150]}")
        print(
            "  A budget site must be granted the cap MINUS the run's reserve "
            "(WorkspaceGrantBytes), so the pre-gate arm's floor and the engine's "
            "rung condition are one quantity. A site that computes its own grant "
            "is the drift this guard exists to stop."
        )
        return 1

    print(f"workspace-grant OK: {COUNT} budget site(s), all consuming WorkspaceGrantBytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
