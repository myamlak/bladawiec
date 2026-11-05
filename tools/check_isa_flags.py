#!/usr/bin/env python3
"""Check the instruction-set tier a configured build actually compiles at.

Reads the compile database every configure writes (CMAKE_EXPORT_COMPILE_COMMANDS
is ON), so the check needs no build: it proves the flags the QCX_ISA option
promises are the flags the compiler was handed.

  --isa baseline: no first-party C++ command carries /arch:AVX2 or
      -mavx2/-mfma/-mf16c (nor an -march= spelling that implies them), and
      every GCC/Clang command keeps the -ffp-contract=off pin. MSVC keeps its
      default /fp:precise, so /fp:fast is refused there.
  --isa avx2: every first-party C++ command carries the full tier:
      /arch:AVX2 on MSVC, -mavx2 -mfma -mf16c -ffp-contract=off on GCC/Clang.

First-party means the module directories in tools/check_dag.py's ORDER, plus
tests/, benchmarks/ and external/boys/ (the Boys sources compiled into
qcx-integrals). third_party/ and external/excgrid/ are vendored and are added
before the tier is stated, so they are out of scope.

Usage:
  python tools/check_isa_flags.py --isa baseline \
      --compile-commands build/windows-msvc-baseline/compile_commands.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from check_dag import ORDER

TIER_FLAGS_GNU = ("-mavx2", "-mfma", "-mf16c")
TIER_FLAG_MSVC = "/arch:avx2"
PIN_FLAG_GNU = "-ffp-contract=off"
# Spellings that would silently enable the AVX2 tier at "baseline".
IMPLIED_TIER_FLAGS = ("-march=native", "-march=x86-64-v3", "-march=x86-64-v4", "-mavx512f")
FORBIDDEN_BASELINE = TIER_FLAGS_GNU + IMPLIED_TIER_FLAGS + (TIER_FLAG_MSVC, "/fp:fast")

FIRST_PARTY_PREFIXES = tuple(ORDER) + ("tests", "benchmarks", "external/boys")


def compiler_of(command: str) -> str:
    """Return 'msvc' or 'gnu' from the command's own compiler spelling."""
    lowered = command.lower()
    if "cl.exe" in lowered or "clang-cl" in lowered:
        return "msvc"
    return "gnu"


def tokens_of(command: str) -> list[str]:
    """Split a compile command into flag tokens, quotes stripped."""
    return [token.strip('"') for token in command.split()]


def first_party_source(entry: dict, source_root: Path) -> str | None:
    """Return the source-root-relative posix path when the entry is first-party."""
    raw = Path(entry["file"])
    if not raw.is_absolute():
        raw = Path(entry["directory"]) / raw
    try:
        relative = raw.resolve().relative_to(source_root)
    except ValueError:
        return None
    posix = relative.as_posix()
    if not any(posix == prefix or posix.startswith(prefix + "/")
               for prefix in FIRST_PARTY_PREFIXES):
        return None
    if raw.suffix == ".cu":
        return None
    return posix


def check_baseline(path: str, command: str, compiler: str) -> list[str]:
    tokens = tokens_of(command)
    findings = []
    for flag in FORBIDDEN_BASELINE:
        # CUDA host flags arrive as -Xcompiler=..., so test the whole token.
        if any(flag in token.lower() for token in tokens):
            findings.append(f"{path}: baseline carries {flag}")
    if compiler == "gnu" and PIN_FLAG_GNU not in tokens:
        findings.append(f"{path}: baseline is missing {PIN_FLAG_GNU}")
    return findings


def check_avx2(path: str, command: str, compiler: str) -> list[str]:
    tokens = tokens_of(command)
    findings = []
    if compiler == "msvc":
        if TIER_FLAG_MSVC not in (token.lower() for token in tokens):
            findings.append(f"{path}: avx2 is missing {TIER_FLAG_MSVC}")
    else:
        for flag in TIER_FLAGS_GNU:
            if flag not in tokens:
                findings.append(f"{path}: avx2 is missing {flag}")
        if PIN_FLAG_GNU not in tokens:
            findings.append(f"{path}: avx2 is missing {PIN_FLAG_GNU}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--isa", required=True, choices=("avx2", "baseline"))
    parser.add_argument("--compile-commands", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    arguments = parser.parse_args()

    if not arguments.compile_commands.is_file():
        print(f"no compile database at {arguments.compile_commands} - configure first")
        return 2

    entries = json.loads(arguments.compile_commands.read_text(encoding="utf-8"))
    source_root = arguments.source_dir.resolve()
    check = check_baseline if arguments.isa == "baseline" else check_avx2

    findings: list[str] = []
    checked = 0
    for entry in entries:
        path = first_party_source(entry, source_root)
        if path is None:
            continue
        command = entry.get("command") or " ".join(entry.get("arguments", []))
        findings.extend(check(path, command, compiler_of(command)))
        checked += 1

    if checked == 0:
        print(f"no first-party C++ commands in {arguments.compile_commands} - wrong database?")
        return 2

    if findings:
        print(f"QCX_ISA={arguments.isa}: {len(findings)} finding(s) in {checked} commands")
        for finding in findings:
            print(f"  {finding}")
        return 1

    print(f"QCX_ISA={arguments.isa}: {checked} first-party C++ commands carry the expected flags")
    return 0


if __name__ == "__main__":
    sys.exit(main())
