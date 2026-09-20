#!/usr/bin/env python
r"""The builder vocabulary, checked against every place that spells it.

WHY THIS EXISTS, measured. One fact - which builder kinds exist, which words spell
them and which of them the driver actually runs - lives in five places, and until this
check nothing held them against each other:

  1. the ENUM        `io/include/qcx/io/run_input.hpp`, `enum class BuilderKind`;
  2. the PARSER      `io/src/parse_input.cpp`, `ParseBuilder`'s accepted words;
  3. the DRIVER      `driver/src/run_driver.cpp`, its resolution of a kind;
  4. the REGISTRY    `integrals/include/qcx/integrals/aux_basis.hpp`,
                     `FockBuilderKind`, the kind the integrals aux rule is asked for;
  5. the RECORD      `io/src/result_json.cpp`, `selection.builder`, which writes
                     `ToString(builder)` - so the record's word IS the enum's.

THE LIVE CASE, and it is the reason the check is shaped this way rather than as a
word-list comparison. At the revision this landed on, `kRiJk`'s own doc comment read
*"nothing wires this kind, and the driver refuses the request by name"* while the tree
carried four working arms for it - a `case` in `driver/src/memory_model.cpp` (that file
has since been deleted; the reading is a historical one and the check no longer reads it), the
`AuxRuleKind` mapping, and an `riJkBuilder` handle that the run flow reads counters
through. Prose and code disagreed at the same commit, one enumerator away from a
neighbouring comment that HAD been repaired by hand. That is machine-detectable, and it
is this check's first finding: a claim on an enumerator is checked against the driver,
not read.

SCOPE, stated as the check's own boundary. Leg 5 asserts ONE direction - a kind whose
comment claims it is unwired, while the driver carries wiring evidence for it, fails by
name. The reverse (a kind with no evidence and no such claim) is NOT asserted: the
evidence table below is per-kind and a route it does not name is not proof of absence,
so asserting the reverse would fire on every kind whose wiring this table has not been
taught. The table is declared here and is itself reviewable; `gpu_split`'s empty entry
is the registered no-wiring precedent the enum's own comment states.

Exit 0 when every leg holds, 1 otherwise, naming the kind, the word and the file.

Run: python tools/check_vocabulary_contract.py [--quiet]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENUM_HPP = ROOT / "io" / "include" / "qcx" / "io" / "run_input.hpp"
PARSE_CPP = ROOT / "io" / "src" / "parse_input.cpp"
DRIVER_CPP = ROOT / "driver" / "src" / "run_driver.cpp"
REGISTRY_HPP = ROOT / "integrals" / "include" / "qcx" / "integrals" / "aux_basis.hpp"
RECORD_CPP = ROOT / "io" / "src" / "result_json.cpp"

#: The driver symbols that evidence EXECUTION wiring for a kind - a builder handle the
#: run flow reads, or the kind's entry in the one mapping from the io vocabulary to the
#: integrals aux rule. A kind whose entry is empty is not asserted to be unwired (see
#: SCOPE in the module docstring); `kGpuSplit` is the registered case, and the enum's
#: own comment says why: it exists so the vocabulary can name a candidate, with
#: execution staying on the wired kinds.
WIRING_EVIDENCE = {
    "kDirect": ("modeInfo", "exchangeModeInfo"),
    "kRiJLink": ("riJkModeInfo",),
    "kRiJk": ("riJkBuilder", "riJkModeInfo"),
    "kQfmm": ("qfmmModel",),
    "kGpu": (),
    "kGpuSplit": (),
}

#: Phrases on an enumerator that ASSERT it is not wired. Registered rather than
#: inferred: each is a claim about the driver, and each is checked against the driver.
UNWIRED_CLAIMS = (
    "nothing wires this kind",
    "refuses the request by name",
    "no builder to execute",
)

#: The one accepted word that is an ALIAS - it resolves to a kind that also has a
#: family word, so it is an input spelling and never a canonical output. The enum
#: documents it (kDirect's note, and ToString's).
ALIASES = {"in_memory": "kDirect"}

#: Kinds with no `[method]` word at all, and the reason. Totality is asserted against
#: this registration rather than against silence, so a kind that loses its word is a
#: finding instead of a smaller table (the `gpu_split` no-word precedent).
NO_WORD = {"kGpuSplit": "no [method] word by design: a heuristic pick, never a user request"}


class SourceError(Exception):
    """A surface could not be read, so the contract cannot be evaluated."""


def read(path: Path) -> str:
    if not path.exists():
        raise SourceError(f"{path.relative_to(ROOT)} is missing")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise SourceError(f"{path.relative_to(ROOT)} is empty")
    return text


def enum_kinds() -> list[str]:
    """The BuilderKind enumerators, in declaration order."""
    text = read(ENUM_HPP)
    body = re.search(r"enum class BuilderKind\s*\{(.*?)\n\};", text, re.DOTALL)
    if body is None:
        raise SourceError("run_input.hpp has no `enum class BuilderKind` block")
    kinds = re.findall(r"^\s+(k[A-Za-z0-9]+)\s*,", body.group(1), re.MULTILINE)
    if not kinds:
        raise SourceError("the BuilderKind block declares no enumerators")
    return kinds


def enum_comments() -> dict[str, str]:
    """Each enumerator with the doc comment that precedes or trails it.

    A trailing `///<` note and the block comment above the enumerator are one
    claim to a reader, so they are joined: the kRiJk sentence that this check
    exists for is split across the two.
    """
    text = read(ENUM_HPP)
    body = re.search(r"enum class BuilderKind\s*\{(.*?)\n\};", text, re.DOTALL)
    if body is None:
        raise SourceError("run_input.hpp has no `enum class BuilderKind` block")
    comments: dict[str, str] = {}
    pending: list[str] = []
    for line in body.group(1).splitlines():
        stripped = line.strip()
        match = re.match(r"^(k[A-Za-z0-9]+)\s*,", stripped)
        if match is not None:
            comments[match.group(1)] = " ".join(pending + [stripped])
            pending = []
            continue
        if stripped.startswith("///"):
            pending.append(stripped.lstrip("/").strip())
        elif stripped:
            pending = []
    return comments


def parser_words() -> dict[str, str]:
    """{accepted word: kind} from ParseBuilder, and its ALIAS spellings."""
    text = read(PARSE_CPP)
    start = text.find("ParseBuilder")
    if start < 0:
        raise SourceError("parse_input.cpp has no ParseBuilder")
    body = text[start:]
    words: dict[str, str] = {}
    pattern = re.compile(
        r'if \(name == "([A-Za-z_]+)"\)\s*\{\s*return BuilderKind::(k[A-Za-z0-9]+);',
        re.DOTALL,
    )
    for match in pattern.finditer(body):
        words[match.group(1)] = match.group(2)
    if not words:
        raise SourceError("ParseBuilder yielded no accepted words")
    return words


def canonical_words() -> dict[str, str]:
    """{kind: word} from ToString, the inverse the record and refusals print."""
    text = read(PARSE_CPP)
    body = re.search(r"std::string_view ToString\(BuilderKind builder\).*?\n\}", text, re.DOTALL)
    if body is None:
        raise SourceError("parse_input.cpp has no ToString(BuilderKind)")
    words = {}
    for match in re.finditer(
        r"case BuilderKind::(k[A-Za-z0-9]+):\s*return \"([A-Za-z_]+)\";", body.group(0)
    ):
        words[match.group(1)] = match.group(2)
    if not words:
        raise SourceError("ToString(BuilderKind) yielded no words")
    return words


def registry_kinds() -> list[str]:
    """The integrals-side kinds the aux rule is asked for."""
    body = re.search(r"enum class FockBuilderKind\s*\{(.*?)\n\};", read(REGISTRY_HPP), re.DOTALL)
    if body is None:
        raise SourceError("aux_basis.hpp has no `enum class FockBuilderKind` block")
    return re.findall(r"^\s+(k[A-Za-z0-9]+)\s*,", body.group(1), re.MULTILINE)


def record_uses_to_string() -> bool:
    """Whether the run record's builder field is written through ToString."""
    return bool(re.search(r'"builder",\s*std::string\(ToString\(', read(RECORD_CPP)))


def driver_wiring(kind: str) -> list[str]:
    """The wiring-evidence symbols for `kind` that the driver actually carries."""
    text = read(DRIVER_CPP)
    return [symbol for symbol in WIRING_EVIDENCE.get(kind, ()) if symbol in text]


def vocabulary_errors() -> list[str]:
    """Legs 1-4: totality and injectivity, both directions, by name."""
    errors: list[str] = []
    kinds = enum_kinds()
    words = parser_words()
    canonical = canonical_words()
    registry = registry_kinds()

    # Leg 1a - every kind has exactly one canonical word, or a registered reason.
    for kind in kinds:
        if kind in canonical:
            continue
        if kind in NO_WORD:
            continue
        errors.append(
            f"{kind}: the enum declares it and ToString spells no word for it, so no record "
            "and no refusal can name it. Add the word, or register it in NO_WORD with the "
            "reason it has none."
        )

    # Leg 1b - canonical words are INJECTIVE: two kinds sharing one word makes a
    # record unable to say which ran, and a reader unable to ask for one of them.
    seen: dict[str, str] = {}
    for kind, word in sorted(canonical.items()):
        if word in seen:
            errors.append(
                f"{seen[word]} and {kind} both spell {word!r} in ToString: the word no longer "
                "identifies one kind, so a record naming it states neither."
            )
        seen[word] = kind

    # Leg 1c - accepted words are INJECTIVE too, and every one resolves.
    accepted: dict[str, str] = {}
    for word, kind in sorted(words.items()):
        if word in accepted and accepted[word] != kind:
            errors.append(f"ParseBuilder accepts {word!r} for two kinds: {accepted[word]} and {kind}")
        accepted[word] = kind
        if kind not in kinds:
            errors.append(f"ParseBuilder returns {kind}, which the enum does not declare")

    # Leg 2 - TOTALITY of the request surface: every kind is reachable by some
    # word, or registered as deliberately not reachable.
    reachable = set(accepted.values())
    for kind in kinds:
        if kind in reachable or kind in NO_WORD:
            continue
        errors.append(
            f"{kind}: no [method] word reaches it and NO_WORD does not register it, so the kind "
            "exists with no way to ask for it and no stated reason."
        )

    # Leg 3 - the record's word must be one the parser accepts, or the record names
    # a state the input could not have written (the two-vocabularies defect).
    if not record_uses_to_string():
        errors.append(
            f"{RECORD_CPP.relative_to(ROOT)}: the record's `selection.builder` field no longer "
            "reads ToString(builder), so the record's vocabulary is no longer the enum's and "
            "this check cannot tie it to one."
        )
    # A NO_WORD kind is the exception the registration exists for, and the assertion
    # INVERTS for it rather than being skipped: the record must be able to name a kind
    # the input cannot request (gpu_split's whole purpose), so its word must NOT be
    # accepted. Skipping it would let a no-word kind quietly grow a request word, which
    # is the vocabulary change the registration is meant to make visible.
    for kind, word in sorted(canonical.items()):
        if kind in NO_WORD:
            if word in accepted:
                errors.append(
                    f"{kind}: ToString prints {word!r} and ParseBuilder ACCEPTS it, but the kind "
                    f"is registered as having no [method] word ({NO_WORD[kind]}). Either the word "
                    "became a request and the registration is stale, or it was accepted by mistake."
                )
            continue
        if word not in accepted:
            errors.append(
                f"{kind}: ToString prints {word!r}, which ParseBuilder does not accept, so the "
                "record and the input spell the same run two ways."
            )

    # Leg 4 - the registry the aux rule is asked for must have an entry for every
    # kind the driver routes through it, or the aux fit is chosen by fallback.
    if not registry:
        errors.append("the integrals registry declares no FockBuilderKind enumerators")
    if "kDefault" not in registry:
        errors.append(
            "the integrals registry has no kDefault: the aux rule's fallback answer is missing, "
            "and every kind without its own entry would take a default nothing declares."
        )
    return errors


def strip_quotations(comment: str) -> str:
    r"""The comment with every `"..."` span removed.

    The repository's own convention for a superseded reading is to QUOTE it -
    *"nothing wires this kind" (a superseded reading)* - and a phrase scan that cannot tell a
    recorded supersession from a live assertion reds the comment that records
    the fix. That is not hypothetical: it happened on this check's first
    correction, when the repaired `kRiJk` note was reported as making the claim
    it exists to retract. An assertion here is the UNQUOTED text.
    """
    return re.sub(r"\"[^\"]*\"", " ", comment)


def comment_claims_errors() -> list[str]:
    """Leg 5: an enumerator's unwired claim, against the driver's wiring evidence."""
    errors: list[str] = []
    comments = enum_comments()
    for kind in enum_kinds():
        comment = strip_quotations(comments.get(kind, ""))
        claims = [phrase for phrase in UNWIRED_CLAIMS if phrase in comment]
        if not claims:
            continue
        evidence = driver_wiring(kind)
        if evidence:
            errors.append(
                f"{kind}: its doc comment claims it is unwired ({claims[0]!r}) while "
                f"{DRIVER_CPP.relative_to(ROOT)} carries wiring evidence for it "
                f"({', '.join(evidence)}). The comment is the defect - a reader who believes it "
                "concludes the kind cannot run."
            )
    return errors


def main(argv: list[str]) -> int:
    try:
        errors = vocabulary_errors() + comment_claims_errors()
    except SourceError as error:
        print(f"vocabulary-contract: {error}", file=sys.stderr)
        return 1

    if errors:
        print("vocabulary-contract check failed:")
        for error in errors:
            print(f"  {error}")
        return 1

    if "--quiet" not in argv:
        kinds = enum_kinds()
        words = canonical_words()
        print(
            f"vocabulary OK: {len(kinds)} BuilderKind(s) - "
            + ", ".join(f"{k}={words.get(k, '<no word>')}" for k in kinds)
            + f"; {len(parser_words())} accepted word(s); the record writes ToString; "
            f"{len(registry_kinds())} integrals registry kind(s); no unwired claim contradicts "
            "the driver"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
