#!/usr/bin/env python3
"""Internal-linkage guard: a function DECLARED in a public header must not be
DEFINED with internal linkage.

WHY THIS EXISTS. A static library does not link-check. A function declared in
`<module>/include/qcx/...` and called from another translation unit, but
defined inside an anonymous namespace in a `<module>/src/*.cpp`, compiles
cleanly in every TU that declares or defines it - and fails only when an
EXECUTABLE links. The author builds a library, sees green, commits, and
the unresolved external surfaces in whoever links next. The defect is
therefore structurally guaranteed to be found by someone other than its
author, which is the failure mode this guard removes.

The instance that motivated it (2026-09-13): `qcx::integrals::SymmetrizeDensity`,
declared in integrals/include/qcx/integrals/symmetry_reduction.hpp and defined
inside the anonymous namespace of integrals/src/symmetry_reduction.cpp, called
cross-TU from integrals/src/fock_build.cpp. Every executable linking
qcx-integrals died with LNK2019.

WHAT IT FLAGS - the intersection of two independently-computed facts:
  1. a FREE function declared at namespace scope in a public header
     (`<module>/include/qcx/**.{hpp,h}`), and
  2. a function of the SAME NAME AND ARITY defined with internal linkage in a
     `.cpp` of the SAME MODULE - inside an anonymous namespace, or `static`.

WHAT IT DELIBERATELY DOES NOT FLAG, each for a stated reason:
  * A name defined with internal linkage and NOT declared in a public header.
    That is the ordinary translation-unit-local helper, and it is the
    overwhelming majority of anonymous-namespace content.
  * A name whose declared arity differs from the internal definition's. An
    overload set is the case static text cannot resolve, and a guess here is a
    false positive; the case is left to the link guard (tools/link_guard.py),
    which reads the real symbol table.
  * A name that ALSO has an external-linkage definition anywhere in the
    committed tree's `src/`. The linker resolves that symbol, so no LNK2019
    follows. Suppressing on the whole tree rather than on the module alone is
    the conservative direction: it can cost a report in a name coincidence, and
    it buys the absence of noise that would get the guard switched off.
  * A declaration carrying `static` in the header - its linkage is already
    internal, so the header promises no external symbol.
  * Class members, function-pointer variables, control statements, macro
    invocations, and anything else whose reading is not certain. Undecidable
    is not flagged.

SCOPE, stated so the guard is not read as proving more than it does. This is a
TEXT-LEVEL scan over the COMMITTED VIEW of the tree - the index, which under a
path-limited commit is HEAD plus the pathspec - and NEVER the working tree.
That is deliberate: a working-tree-wide check lets one author's work-in-progress
block every other commit, whereas this one describes only what is being
committed and so
cannot be tripped by a foreign uncommitted edit.

Its blind spots follow from being static text: it cannot see a defect that
exists only after preprocessing, one in generated code, a declaration injected
by a macro, or a symbol that is declared and simply DEFINED NOWHERE (that
class has no internal-linkage definition to find, and belongs to the link
guard). It also cannot see a cross-module pair, where a header of module A is
satisfied by an anonymous definition in module B - the same defect, rarer, and
likewise the link guard's.

Self-test: tools/check_internal_linkage_test.py
Standalone: python tools/check_internal_linkage.py [--all] [--rev REV]
"""

from __future__ import annotations

import argparse
import bisect
import re
import subprocess
import sys

# Words that can stand before a '(' in a head that is NOT a function
# declarator. Without this the scanner reads `if (x) {` as a function named
# `if` taking one argument.
KEYWORDS = frozenset({
    "alignas", "alignof", "and", "asm", "auto", "bool", "break", "case",
    "catch", "char", "char8_t", "char16_t", "char32_t", "class", "co_await",
    "co_return", "co_yield", "concept", "const", "consteval", "constexpr",
    "constinit", "const_cast", "continue", "decltype", "default", "delete",
    "do", "double", "dynamic_cast", "else", "enum", "explicit", "export",
    "extern", "false", "float", "for", "friend", "goto", "if", "inline",
    "int", "long", "mutable", "namespace", "new", "noexcept", "not",
    "nullptr", "operator", "or", "private", "protected", "public", "register",
    "reinterpret_cast", "requires", "return", "short", "signed", "sizeof",
    "static", "static_assert", "static_cast", "struct", "switch", "template",
    "this", "throw", "true", "try", "typedef", "typeid", "typename", "union",
    "unsigned", "using", "virtual", "void", "volatile", "wchar_t", "while",
    "xor",
})

# Type keywords that make a bare (whitespace-free) parameter plausible.
TYPE_KEYWORDS = frozenset({
    "bool", "char", "char8_t", "char16_t", "char32_t", "double", "float",
    "int", "long", "short", "signed", "size_t", "unsigned", "void", "wchar_t",
})

MODIFIERS = re.compile(
    r"^(?:(?:inline|constexpr|consteval|constinit|extern|static|friend|"
    r"virtual|explicit|mutable)\s+)+"
)
GENERIC_HEAD = re.compile(r"^template\s*<[^<>]*(?:<[^<>]*>[^<>]*)*>\s*")
TYPE_HEAD = re.compile(r"^\s*(?:class|struct|union|enum)\b")
NAMESPACE_HEAD = re.compile(r"^\s*(?:inline\s+)?namespace\b\s*([A-Za-z_][\w:]*)?")
EXTERN_C_HEAD = re.compile(r'^\s*extern\s*""')
STATIC_HEAD = re.compile(r"^static\s")

# What a definition head may carry after its closing ')': cv-qualifiers, an
# exception specification, a trailing return type, attributes, `= default`.
TAIL = re.compile(
    r"^\s*(?:const\b|volatile\b|noexcept\b[^;{}]*|override\b|final\b|"
    r"mutable\b|\[\[[^\]]*\]\]|->\s*[^;{}]*|=\s*(?:default|delete|0))\s*$"
)

# `bool operator==(...)`: the declarator is the symbol run after `operator`.
OPERATOR_TAIL = re.compile(
    r"operator\s*([+\-*/%^&|~!=<>\[\]()]+|new\[\]|delete\[\]|new|delete|"
    r"co_await)\s*$"
)

COMMENT = re.compile(r"//[^\n]*")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
RAW_STRING = re.compile(r'[rR]"([A-Za-z0-9_/]{0,16})\(.*?\)\1"', re.S)
STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
CHAR = re.compile(r"'(?:[^'\\]|\\.)*'")
PREPROC_LINE = re.compile(r"^[ \t]*#[^\n]*$", re.M)

HEADER_PATH = re.compile(r"^([a-z][a-z0-9]*)/include/qcx/.*\.(?:hpp|h)$")
SOURCE_PATH = re.compile(r"^([a-z][a-z0-9]*)/src/.*\.cpp$")

# Statement boundaries. Scanned with finditer (C speed) rather than a
# per-character Python loop: the guard reads every tracked source, and a
# character loop over ~5 MB of C++ is seconds where this is milliseconds.
BOUNDARY = re.compile(r"[{};]")


def strip_noncode(text: str) -> str:
    """Remove comments, literals and preprocessor lines, NEWLINE-PRESERVING.

    Reported lines come from the stripped text, so every substitution keeps the
    newline count: a block comment spanning lines collapses to its own newlines
    rather than vanishing and shifting every later line.
    """
    text = COMMENT.sub("", text)
    text = BLOCK_COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    text = RAW_STRING.sub(lambda m: '""' + "\n" * m.group(0).count("\n"), text)
    text = STRING.sub('""', text)
    text = CHAR.sub("''", text)
    return PREPROC_LINE.sub("", text)


def classify_head(head: str) -> str:
    """Which kind of brace frame a `... {` head opens.

    `anon_ns` is the one that matters: it is the marker of internal linkage.
    """
    if EXTERN_C_HEAD.match(head):
        return "transparent"
    namespace = NAMESPACE_HEAD.match(head)
    if namespace:
        return "namespace" if namespace.group(1) else "anon_ns"
    if TYPE_HEAD.match(head.strip()) or TYPE_HEAD.match(head):
        return "type"
    return "other"


def scan_heads(code: str):
    """Yield (head, stack_kinds, terminator, head_offset).

    A `head` is the text between two statement boundaries, so a signature
    broken across lines arrives whole. `stack_kinds` is the tuple of enclosing
    brace-frame kinds, outermost first, at the moment the head closes.
    """
    stack: list[str] = []
    start = 0
    for match in BOUNDARY.finditer(code):
        index = match.start()
        ch = match.group(0)
        if ch == "{":
            yield code[start:index], tuple(stack), "{", start
            stack.append(classify_head(code[start:index]))
            start = index + 1

        elif ch == "}":
            if stack:
                stack.pop()
            start = index + 1

        else:
            yield code[start:index], tuple(stack), ";", start
            start = index + 1


def namespace_level(stack) -> bool:
    """True when only namespace-ish frames enclose this point.

    A `type` frame is class scope (members are not free functions); an `other`
    frame is a function body, a braced initializer or a lambda.
    """
    return not any(kind in ("type", "other") for kind in stack)


def _matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1

        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _callee_name(text: str, open_index: int):
    """(name, name_start) for the declarator whose parameter list opens here.

    A name preceded by `::` is a QUALIFIED name - a member definition or a
    namespace-qualified one - and is not a free function, so it is rejected
    rather than reduced to its last component.
    """
    end = open_index
    while end > 0 and text[end - 1].isspace():
        end -= 1
    start = end
    while start > 0 and (text[start - 1].isalnum() or text[start - 1] == "_"):
        start -= 1
    if start < end:
        if start >= 2 and text[start - 2:start] == "::":
            return None, 0
        return text[start:end], start
    operator = OPERATOR_TAIL.search(text[:end])
    if operator:
        return "operator" + operator.group(1), operator.start()
    return None, 0


def _arity(params: str) -> int:
    params = params.strip()
    if not params or params == "void":
        return 0
    depth = 0
    commas = 0
    for ch in params:
        if ch in "(<[":
            depth += 1

        elif ch in ")>]":
            depth -= 1

        elif ch == "," and depth == 0:
            commas += 1
    return commas + 1


def _split_params(params: str) -> list:
    parts = []
    depth = 0
    current = ""
    for ch in params:
        if ch in "(<[":
            depth += 1

        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    return [part.strip() for part in parts]


def _plausible_params(params: str) -> bool:
    """Whether a parameter list can be a parameter list at all.

    `Foo x(1);` is direct-initialization, not a declaration, and is already
    rejected by the literal test. The residual ambiguity is `Foo x(y);` against
    `Foo Bar(Y);` - a lowercase bare identifier in parameter position reads as
    an expression, so a list made ONLY of those is not read as a declaration.
    This is an undecidable case being declined, not a case being decided.
    """
    for part in _split_params(params):
        if not part:
            continue
        if len(part.split()) > 1:
            continue
        if any(ch in part for ch in "*&<>[]::=..."):
            continue
        if part in TYPE_KEYWORDS or part.upper() == part:
            continue
        return False
    return True


def head_signature(head: str):
    """(name, arity, is_static) when the head declares or defines a free
    function, else None.

    The name is the declarator immediately before the parameter list's `(`.
    SOMETHING must precede it: at namespace scope a head with no return type is
    a constructor, and constructors live in class scope, excluded already.
    """
    text = strip_noncode(head).strip()
    if not text:
        return None
    is_static = bool(STATIC_HEAD.match(text))
    body = GENERIC_HEAD.sub("", MODIFIERS.sub("", text, count=1))
    paren = body.find("(")
    if paren < 0:
        return None
    name, name_start = _callee_name(body, paren)
    if not name or name in KEYWORDS:
        return None
    if not body[:name_start].strip():
        return None
    close = _matching_paren(body, paren)
    if close < 0:
        return None
    tail = body[close + 1:]
    if tail.strip() and not TAIL.match(tail):
        return None
    params = body[paren + 1:close]
    if _split_params(params) and _split_params(params)[0][:1].isdigit():
        return None
    if not _plausible_params(params):
        return None
    return name, _arity(params), is_static


def declared_free_functions(text: str) -> dict:
    """{name: set(arity)} for free functions declared at namespace scope.

    `static` declarations are skipped: their linkage is already internal, so
    the header promises no external symbol for anyone to fail to find.
    """
    found: dict[str, set] = {}
    for head, stack, terminator, _ in scan_heads(strip_noncode(text)):
        if terminator != ";" or not namespace_level(stack) or "(" not in head:
            continue
        signature = head_signature(head)
        if signature is None:
            continue
        name, arity, is_static = signature
        if is_static:
            continue
        found.setdefault(name, set()).add(arity)
    return found


def definitions(text: str):
    """Yield (name, arity, line, is_internal) for every function definition.

    A definition is a `... {` head at namespace level that is not a class,
    enum, namespace or `extern "C"` block. `is_internal` is the fact the guard
    turns on: an enclosing anonymous namespace, or a `static` head.
    """
    code = strip_noncode(text)
    newlines = [index for index, ch in enumerate(code) if ch == "\n"]
    for head, stack, terminator, offset in scan_heads(code):
        if terminator != "{" or "(" not in head:
            continue
        if classify_head(head) != "other" or not namespace_level(stack):
            continue
        signature = head_signature(head)
        if signature is None:
            continue
        name, arity, is_static = signature
        yield (
            name,
            arity,
            bisect.bisect_right(newlines, offset) + 1,
            "anon_ns" in stack or is_static,
        )


def find_violations(files: dict) -> list:
    """The guard, over {path: text} of every tracked public header and source.

    Returns [(source_path, line, name, arity, module)]. The header that
    satisfies a source's declaration must belong to the SOURCE'S OWN MODULE:
    that is the relationship the defect has, and the relationship the DAG makes
    legal.
    """
    headers_by_module: dict[str, dict] = {}
    for path, text in files.items():
        matched = HEADER_PATH.match(path)
        if not matched:
            continue
        declarations = headers_by_module.setdefault(matched.group(1), {})
        for name, arities in declared_free_functions(text).items():
            # UNION, never overwrite: an overload set split across two headers
            # of the module would otherwise keep only the last file's arities
            # and the guard would decline a defect it had already seen.
            declarations.setdefault(name, set()).update(arities)

    externally_defined = set()
    internal_by_module: dict[str, list] = {}
    for path, text in files.items():
        matched = SOURCE_PATH.match(path)
        if not matched:
            continue
        for name, arity, line, is_internal in definitions(text):
            if is_internal:
                internal_by_module.setdefault(matched.group(1), []).append(
                    (path, line, name, arity)
                )
            else:
                externally_defined.add((name, arity))

    violations = []
    for module, entries in internal_by_module.items():
        declared = headers_by_module.get(module, {})
        for path, line, name, arity in entries:
            arities = declared.get(name)
            if arities is None or arity not in arities:
                continue
            if (name, arity) in externally_defined:
                continue
            violations.append((path, line, name, arity, module))
    return sorted(violations)


def format_violation(violation) -> str:
    path, line, name, arity, module = violation
    return (
        f"{path}:{line}: {name} ({arity} arg) is DECLARED in "
        f"{module}/include/qcx/ but DEFINED with internal linkage - a caller in "
        f"another translation unit fails to link (LNK2019)"
    )


def _read_blobs(shas) -> dict:
    """Read many blobs in ONE git process."""
    shas = [sha for sha in shas if sha]
    if not shas:
        return {}
    payload = "".join(sha + "\n" for sha in shas).encode()
    proc = subprocess.run(
        ["git", "cat-file", "--batch"], input=payload, capture_output=True
    )
    data = proc.stdout
    blobs = {}
    pos = 0
    while pos < len(data):
        newline = data.find(b"\n", pos)
        if newline < 0:
            break
        header = data[pos:newline].decode("utf-8", "replace").split()
        pos = newline + 1
        if len(header) < 3:
            break
        size = int(header[2])
        blobs[header[0]] = data[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1
    return blobs


def _entries_from(output: bytes, sha_field: int):
    """Parse `-z` ls-files / ls-tree output into (path, blob_sha).

    The two commands differ in where the object id sits - `ls-files -s` is
    `<mode> <object> <stage>`, `ls-tree -r` is `<mode> <type> <object>` - so
    the caller names the field. Reading the wrong one silently yields "blob"
    as a sha, an empty file set and a guard that reports nothing: a false
    negative that looks exactly like a clean tree.
    """
    entries = []
    for record in output.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        fields = meta.split()
        if len(fields) <= sha_field:
            continue
        entries.append((path.decode("utf-8", "replace"), fields[sha_field].decode()))
    return entries


def committed_view(rev=None) -> dict:
    """Every tracked public header and source, as COMMITTED.

    rev=None reads the index - and under a path-limited commit git hands the
    hook its own temp index built from HEAD plus the pathspec, so this is
    exactly the tree the commit would create. The working tree is never read.
    """
    if rev is None:
        listing = subprocess.run(
            ["git", "ls-files", "-s", "-z"], check=True, capture_output=True
        ).stdout
        sha_field = 1
    else:
        listing = subprocess.run(
            ["git", "ls-tree", "-r", "-z", rev], check=True, capture_output=True
        ).stdout
        sha_field = 2
    entries = [
        (path, sha)
        for path, sha in _entries_from(listing, sha_field)
        if HEADER_PATH.match(path) or SOURCE_PATH.match(path)
    ]
    blobs = _read_blobs([sha for _, sha in entries])
    return {path: blobs[sha] for path, sha in entries if sha in blobs}


def staged_touches_guarded_surface() -> bool:
    """Whether this commit could introduce the defect at all.

    The relation is between a declaration and a definition, and it can only
    change when the commit changes one of them - so a commit touching neither a
    public header nor a source is skipped without reading the tree.
    """
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRD", "-z"],
        check=True,
        capture_output=True,
    ).stdout
    for record in out.split(b"\0"):
        if not record:
            continue
        path = record.decode("utf-8", "replace")
        if HEADER_PATH.match(path) or SOURCE_PATH.match(path):
            return True
    return False


def linkage_violations() -> list:
    """The hook entry point: [] when the commit cannot introduce the defect.

    A failure to READ the tree is reported as a finding rather than raised. It
    is not a clean result, and the two must never look alike: the module that
    owns this hook has already paid once for a guard that inspected nothing and
    said so in the same words as a guard that inspected everything (the empty-index
    precondition). Fail closed, and say why.
    """
    try:
        if not staged_touches_guarded_surface():
            return []
        return [format_violation(v) for v in find_violations(committed_view())]
    except (subprocess.CalledProcessError, OSError) as error:
        return [
            f"check_internal_linkage: cannot read the committed view ({error}) - "
            f"refusing to report a clean result that was never computed"
        ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--all",
        action="store_true",
        help="scan regardless of what is staged (a manual or whole-tree run)",
    )
    parser.add_argument(
        "--rev",
        default=None,
        help="scan a revision instead of the index (implies --all)",
    )
    args = parser.parse_args(argv)

    if args.rev is None and not args.all and not staged_touches_guarded_surface():
        print("check_internal_linkage: nothing staged on the guarded surface")
        return 0

    violations = find_violations(committed_view(args.rev))
    for violation in violations:
        print(format_violation(violation))
    where = "rev " + args.rev if args.rev else "committed view"
    print(f"check_internal_linkage: {len(violations)} finding(s) over the {where}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
