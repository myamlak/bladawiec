#!/usr/bin/env python3
"""Binary-freshness gate: is a built executable older than the sources in it?

THE DEFECT CLASS
Four incidents in one day (2026-09-12) shared one shape: a verdict was read
from a binary that was not built from the sources the verdict was attributed
to. A test "failed" on a sibling binary predating the fix; a test row's verdict
came from a binary linked before the change under test; a run used a
qcx-driver-tests.exe linked at 04:00 while its source landed at 10:19. Nothing
in the tree noticed any of them. In a build tree carrying twenty-odd targets
across five configurations and five trees, an mtime is the only cheap evidence
that a binary and a source belong to the same revision.

WHAT THIS PROVES, AND WHAT IT DOES NOT
It computes each executable's real source closure - the translation units the
build system itself compiles into it, plus the headers of every module in its
link closure - and fails when the newest of those is newer than the
executable. That direction is sound: a source edited after the binary was
written ALWAYS trips it.

It does not prove the binary matches its sources. It cannot: nothing records
what revision a binary was built from. A build that claims to relink and does
not (a silently skipped step, an interrupted build, a binary copied over the
built one) leaves the executable newer than its sources and this gate is blind
to it. It is also blind to a source edited with `touch` and no content change,
which trips it falsely - one rebuild clears that, and the report names the
offending file so the false positive is visible in a second. It reads only the
Visual Studio generator trees (the ones carrying .vcxproj): build/wsl-gcc and
build/wsl-clang are covered by CI building and testing in one job, and a
Ninja/Makefile tree needs the `link.txt` equivalent this script does not read.

WHY NOT A COMMIT HOOK
A stale binary is the NORMAL state at commit time - you commit the source,
then build. A hook that failed there would fire on every legitimate commit and
be disabled within a day. The gate belongs where a verdict is READ: the
benchmark runners' fire-time conditions, the full verification run, and the
close of a work session. It is
also importable (`from tools.check_binary_freshness import check_exe`) so a
runner can adopt it without shelling out; tools/bench/fire-window/run_ladder_sweep.py
already imports tools.bench that way. That runner carries its own narrower
version of this gate - five hardcoded driver wiring sources, one target - and
is unaffected by this script.

OBJECTS: THE SECOND DIMENSION (`--objects`)
An executable verdict is one answer about one artifact, and it is coarse in two
ways that matter when a tree drifts. It names a single newest source out of a
closure that measured 467 sources for one driver test - so the source you care
about can be in the closure, older than the binary, and still never appear in
the report, which is exactly how the 2026-09-13 CUDA incident read. And it
compares a binary the LINKER wrote, while the drift it is meant to catch was
found in an object the COMPILER wrote: `integrals/src/eri_cuda.cpp` sat two
hours and six minutes newer than the only object compiling it. `--objects`
compares each translation unit against its own object file, by the mapping the
.vcxproj itself carries (CMake writes <ObjectFileName> when the object's name
is not the item type's default, and that default applies when it is not). The
pair it names - this source, this object - is the pair an incident names. It
also covers the 334 static libraries of a tree whose executables number 13: an
object row needs no executable to exist.

Both item types are swept, and they differ in both halves of that mapping. A
ClCompile include is absolute (319 of 319 in that tree) and its object is
$(IntDir)<stem>.obj; a CudaCompile include is relative (6 of 6) and its object
mirrors the source's path from the module directory with the extension kept -
integrals/src/eri_cuda.cu is $(IntDir)src/eri_cuda.cu.obj, verified against the
object on disk. There is no second candidate rule: when an object is not at that
path the row says so and is counted as cannot-state, because a path chosen
because it happens to exist is a coin flip presented as a fact. How much that
leaves unstated is in the counts, not implied - of the 6 CUDA units of
build/windows-msvc-cuda on 2026-09-13, 2 resolve (eri_cuda.cu and
cuda_backend.cu) and 4 report no object there, and one of those four does have
an object under a different name (`eri_cuda_fock.obj`) which was checked by hand
and is NEWER than its source in both configurations, so the understatement hides
no drift today. Six units is a small number; the reason to carry the second rule
anyway is the shape of the miss - a unit the sweep does not read is a unit it
cannot count as unknown, and the report would then be silent about the CUDA half
of a tree whose CUDA half is the whole point of it.

Measured on 2026-09-13, `--objects` on the day-to-day tree at Release: 298
fresh, 1 stale, 7 not built - a tree that is genuinely current. On
build/windows-msvc-cuda at Release: 146 fresh, 59 stale, 120 not built, and at
Debug 166 fresh, 100 stale, 59 not built. The dimension is not noise; it is the
drift the executable rows understate.

REPORT MODE (`--report`), AND WHAT WOULD MAKE THE CUDA TREE A GATE
The gate's exit code is the right answer where a verdict is READ FROM that tree
(the verification protocol reads build/windows-msvc). build/windows-msvc-cuda is
read for a different purpose: to learn whether the CUDA translation units are
still compiled by anything at all, WITHOUT paying for a CUDA build. No CUDA-off
gate reaches either CUDA target - qcx-md-cuda-class-*.vcxproj references
ZERO_CHECK and never the md target, so the dependency does not run in reverse -
and a source landed there on 2026-09-12 that no gate had compiled since. That is
the incident this reporter exists for, and it is a reporter because a gate over
this tree would be red on every run until the 101 GB tree is rebuilt: 13 of 13
Release executables and 41 of 44 Debug ones were stale on 2026-09-13. A check
that over-triggers gets switched off - the same reason tools/check_doc_cites.py
is a reporter. The counts are printed on every run either way.
WHAT WOULD MAKE IT A GATE, in the same terms: the drift reaches zero. Either the
coordinator rebuilds the tree to a state where its artifacts postdate their
sources, or every remaining row is a source change that provably cannot reach
that target, recorded - `--record` exists for the one-binary case. Until then
the honest question a reader asks of a stale row is which stale path could move
the behaviour under test, not whether the number is zero.
`--report` never returns 1. It still returns 2 - "this tree cannot be read at
all" is a third answer about the instrument, not a verdict about freshness, and
collapsing it into a pass is the failure the exit-code design was paid for to
avoid.

PROVENANCE RECORDS (--record / --verify-record)
Where the mtime question cannot be asked - a private copy taken so a relink
cannot swap the binary under a measurement - the honest answer is a content
hash, not a revision. --record writes <exe>.provenance.json carrying the
sha256, the build time, and the newest source at record time; --verify-record
re-reads it and fails if the bytes changed or the recorded state was stale.
That record is what a run manifest CAN support; it is not a revision, and the
note it writes says so.

Usage:
    python tools/check_binary_freshness.py
    python tools/check_binary_freshness.py --tree build/windows-msvc-l
    python tools/check_binary_freshness.py --exe build/windows-msvc/driver/Release/qcx.exe
    python tools/check_binary_freshness.py --tree build/windows-msvc-cuda --config Debug --objects
    python tools/check_binary_freshness.py --tree build/windows-msvc-cuda --report --objects
    python tools/check_binary_freshness.py --record <exe> [--allow-stale]
    python tools/check_binary_freshness.py --verify-record <exe>

Exit codes: 0 clean, 1 stale (or a failed verification), 2 usage / no tree.
--report returns 0 where it would have returned 1; 2 stays 2.
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TREE = REPO_ROOT / "build" / "windows-msvc"
DEFAULT_CONFIG = "Release"

# Headers are not listed in a .vcxproj, so they are taken from the module
# directories the closure names, scanned whole. That OVER-APPROXIMATES inside a
# module: a header under <module>/tests is not compiled into the module's
# library, so editing one and rebuilding only the module's tests will flag
# every target linking that module. The cost is one extra rebuild; the
# alternative - guessing which headers a target really reads - risks the false
# negative this gate exists to prevent. Stated here so the report is read for
# what it is.
HEADER_SUFFIXES = (".hpp", ".hxx", ".h", ".inl")

# CMake's meta-targets (ZERO_CHECK, ALL_BUILD, INSTALL, RUN_TESTS) compile
# nothing and sit at the repository root. They are referenced by nearly every
# project, so scanning their module directory would scan the WHOLE repository
# for every executable - an over-approximation wide enough to be useless.
META_TARGET_TYPES = ("Utility",)

# MSBuild's own placeholders, as CMake writes them into a .vcxproj. $(IntDir) is
# the only one this script resolves (to <project dir>/<Configuration>/), which is
# why an object template not rooted at it is reported rather than guessed at.
INT_DIR = "$(IntDir)"

# Report mode prints at most this many stale rows per dimension. A gate names
# every offender - a caller cannot act on a count alone - but a reporter that
# prints 41 five-line blocks on every run stops being read, and --json carries
# the whole list for the caller that wants it.
REPORT_LIST_CAP = 12

PROVENANCE_NOTE = (
    "Binary provenance for a measurement. Records what CAN be known at record time: the "
    "measured bytes (sha256), when the executable was written, and the newest source in its "
    "compiled closure at that moment. It is NOT a revision: nothing in the build records which "
    "git revision a binary came from, so a run manifest must not claim one."
)


def local_name(tag):
    """The XML local name, so a namespaced element matches too."""
    return tag.rsplit("}", 1)[-1]


def config_of_condition(condition):
    """'Release' out of a CMake-emitted Condition, or '' when unparsable.

    CMake writes Condition="'$(Configuration)|$(Platform)'=='Release|x64'",
    so the configuration is the token before the '|'.
    """
    if not condition or "==" not in condition:
        return ""
    value = condition.split("==", 1)[1].strip().strip("'\"")
    return value.split("|", 1)[0].strip()


def repo_path(raw):
    """A path as written in a .vcxproj, as a Path, with backslashes folded.

    MSBuild writes absolute Windows paths with backslashes. Folding them keeps
    the value usable under a POSIX interpreter as well as on the native one.
    """
    return Path(raw.replace("\\", "/"))


def display_path(path):
    """A path for a report: repository-relative when it is inside, absolute
    otherwise. A private copy taken for a measurement can live anywhere, and
    an unguarded relative_to would abort the very report that describes it."""
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def path_key(path):
    """A case-folded absolute identity for a path, with no filesystem call.

    Path.resolve() on Windows is nt._getfinalpathname - a syscall each time -
    and the sweep resolves tens of thousands of paths walking link closures,
    which was two thirds of its runtime. os.path.abspath is string arithmetic
    and agrees with the resolved path for everything inside a build tree.
    """
    return os.path.normcase(os.path.abspath(str(path)))


class Project:
    """One .vcxproj: its own translation units, its link references, its
    per-configuration output name."""

    __slots__ = ("path", "module_dir", "tus", "refs", "ref_keys", "target_names",
                 "config_type", "object_templates", "cuda_tus", "cuda_templates")

    def __init__(self, path):
        self.path = path
        self.module_dir = path.parent
        self.tus = []
        self.refs = []
        self.ref_keys = []
        self.target_names = {}
        self.config_type = ""
        # Translation unit -> the ObjectFileName template the project states for
        # it ("" when it states none and the default name applies, None when the
        # stated templates disagree per configuration and neither can be used).
        self.object_templates = {}
        # The CUDA translation units, which are a separate item type
        # (<CudaCompile>) with their own naming rules - see cuda_units().
        self.cuda_tus = []
        self.cuda_templates = {}

    @property
    def int_dir(self):
        """Where the compiler writes this project's objects at `config`.

        CMake names the intermediate directory after the target, and the project
        file is named after the target too (OUTPUT_NAME renames the .exe, never
        the project file), so the stem IS the intermediate directory's name.
        """
        return self.module_dir / (self.path.stem + ".dir")

    def object_path(self, config, template, default_relative):
        """The object file the build writes for one translation unit, or None.

        `template` is what the project states for that unit ("" when it states
        none, so the item type's default applies; None when what it states
        cannot be used), and `default_relative` is that default, relative to
        $(IntDir). The two item types default differently, and both were
        measured on build/windows-msvc-cuda on 2026-09-13:
          - the CL task names an object <stem>.obj;
          - the CUDA task mirrors the source's path from the module's SOURCE
            directory and keeps the extension, so integrals/src/eri_cuda.cu is
            $(IntDir)src/eri_cuda.cu.obj - not $(IntDir)eri_cuda.obj.
        CMake states the CL name explicitly whenever it is not that default
        (<ObjectFileName>$(IntDir)src/x.cpp.obj</ObjectFileName>, seen for
        eri_cuda.cpp), and a template this script cannot resolve is reported as
        cannot-state rather than guessed at: a guessed path is a false verdict
        in whichever direction the guess happens to go.
        """
        if template is None:
            return None
        relative = default_relative
        if template:
            if not template.startswith(INT_DIR):
                return None
            relative = template[len(INT_DIR):]
            if not relative or relative.endswith(("/", "\\")):
                relative = default_relative
        return self.int_dir / config / relative

    def output_name(self, config):
        """The executable's stem, or the project stem as the fallback.

        CMake materializes `set_target_properties(... OUTPUT_NAME "qcx")` as a
        per-configuration <TargetName>, so qcx-driver-run.vcxproj is qcx.exe -
        the project stem is NOT the output name and must not be assumed.
        """
        if config in self.target_names:
            return self.target_names[config]
        if self.target_names:
            return sorted(self.target_names.values())[0]
        return self.path.stem

    def output_exe(self, config):
        return self.module_dir / config / (self.output_name(config) + ".exe")


def resolve_include(raw, project_path):
    """A .vcxproj Include as an absolute Path.

    MSBuild reads a relative Include against the project file's own directory,
    and CMake writes the two item types differently: 319 of 319 ClCompile
    includes are absolute and 6 of 6 CudaCompile includes are relative
    (`..\\..\\..\\integrals\\src\\eri_cuda.cu`), measured 2026-09-13. Folding
    backslashes is not enough for the relative half - an unresolved path would
    be tested against the process's working directory and silently disappear.
    """
    text = raw.replace("\\", "/")
    if os.path.isabs(text):
        return Path(text)
    return Path(os.path.normpath(os.path.join(str(project_path.parent), text)))


def object_template(element):
    """The ObjectFileName a <ClCompile> entry states for its own source.

    "" when the entry states none (the CL default <stem>.obj then applies), the
    template when it states one, and None when it states several that disagree -
    CMake conditions some of them per configuration, and a template read out of
    the wrong configuration's element is a path this script would then assert.
    """
    stated = [(kid.text or "").strip() for kid in element
              if local_name(kid.tag) == "ObjectFileName"]
    stated = [text for text in stated if text]
    if not stated:
        return ""
    if len(set(stated)) > 1:
        return None
    return stated[0]


def load_project(path):
    """Parse one .vcxproj, or return None when it is not usable."""
    project = Project(path)
    try:
        root = ElementTree.parse(str(path)).getroot()
    except (ElementTree.ParseError, OSError):
        return None
    for element in root.iter():
        tag = local_name(element.tag)
        if tag == "ClCompile":
            include = element.get("Include")
            if include:
                unit = repo_path(include)
                project.tus.append(unit)
                project.object_templates[unit] = object_template(element)
        elif tag == "CudaCompile":
            include = element.get("Include")
            if include:
                unit = resolve_include(include, path)
                project.cuda_tus.append(unit)
                project.cuda_templates[unit] = object_template(element)
        elif tag == "ProjectReference":
            include = element.get("Include")
            if include:
                project.refs.append(repo_path(include))
        elif tag == "TargetName":
            name = (element.text or "").strip()
            if name:
                project.target_names[config_of_condition(element.get("Condition"))] = name
        elif tag == "ConfigurationType":
            project.config_type = (element.text or "").strip()
    # A ProjectReference is repeated once per configuration, so the same target
    # arrives several times; the closure walk only ever needs the identity.
    project.ref_keys = sorted({path_key(ref) for ref in project.refs})
    return project


def index_tree(tree):
    """Every .vcxproj under a build tree, keyed by path identity."""
    index = {}
    if not tree.is_dir():
        return index
    for path in tree.rglob("*.vcxproj"):
        project = load_project(path)
        if project is not None:
            index[path_key(path)] = project
    return index


def closure_of(project, index):
    """The project plus every project reachable through ProjectReference."""
    seen = {}
    pending = [project]
    while pending:
        current = pending.pop()
        key = path_key(current.path)
        if key in seen:
            continue
        seen[key] = current
        for ref in current.ref_keys:
            if ref in index and ref not in seen:
                pending.append(index[ref])
    return list(seen.values())


def module_source_dir(project, tree, tree_key):
    """Where the module's sources live, mapped from the build tree to the
    repository by the same relative path (build/<tree>/<x> -> <x>)."""
    if not path_key(project.module_dir).startswith(tree_key):
        return None
    relative = os.path.relpath(str(project.module_dir), str(tree))
    if relative in (".", ""):
        return None
    return REPO_ROOT / relative.replace("\\", "/")


class Freshness:
    """The per-run caches: a project's own translation units, a project's
    closure, and a module's newest header - each computed once however many
    executables ask for it. Without the caches the sweep re-walks every
    closure for every executable, which is where the runtime goes."""

    def __init__(self, tree):
        self.tree = tree
        self.tree_key = path_key(tree)
        self._module_headers = {}
        self._units = {}
        self._cuda_units = {}
        self._closures = {}

    def closure(self, project, index):
        key = path_key(project.path)
        if key not in self._closures:
            self._closures[key] = closure_of(project, index)
        return self._closures[key]

    def translated_units(self, project):
        """The project's own TUs, ignoring anything inside a build tree (the
        generated cmake_pch.cxx is rewritten at configure time and would
        otherwise read as a source that is always newer). The paths in a
        .vcxproj are already absolute, so the build-tree test is a string
        prefix on the normalized path - resolving each one costs a syscall
        and is the difference between a fast gate and a slow one."""
        key = path_key(project.path)
        if key in self._units:
            return self._units[key]
        units = []
        for tu in project.tus:
            if not tu.is_absolute():
                continue
            if path_key(tu).startswith(self.tree_key):
                continue
            if tu.is_file():
                units.append(tu)
        self._units[key] = units
        return units

    def cuda_units(self, project):
        """The project's CUDA sources, filtered as the CL units are.

        They get their own accessor rather than a relaxation of the absolute-path
        rule in translated_units: the CL filter exists because an unresolvable
        path must not be asserted, and a CUDA include is RELATIVE by construction
        (6 of 6 measured), so the two need opposite treatment of the same input.
        A generated CUDA file inside the build tree is skipped exactly as a
        generated pch is.
        """
        key = path_key(project.path)
        if key in self._cuda_units:
            return self._cuda_units[key]
        units = []
        for unit in project.cuda_tus:
            if path_key(unit).startswith(self.tree_key):
                continue
            if unit.is_file():
                units.append(unit)
        self._cuda_units[key] = units
        return units

    def units(self, project):
        """(unit, stated template, default object name relative to $(IntDir)).

        The default is computed here because the CUDA one needs the module's
        SOURCE directory, which only the tree mapping knows: the object mirrors
        the source's path from there, not from the build directory.
        """
        rows = [(unit, project.object_templates.get(unit), unit.stem + ".obj")
                for unit in self.translated_units(project)]
        if project.cuda_tus:
            module_dir = module_source_dir(project, self.tree, self.tree_key)
            for unit in self.cuda_units(project):
                default = unit.name + ".obj"
                if module_dir is not None:
                    try:
                        relative = os.path.relpath(str(unit), str(module_dir))
                    except ValueError:  # a different drive: no relative path
                        relative = None
                    if relative is not None:
                        default = relative.replace("\\", "/") + ".obj"
                rows.append((unit, project.cuda_templates.get(unit), default))
        return rows

    def newest_header(self, project):
        """The newest header anywhere under the project's module directory."""
        module_dir = module_source_dir(project, self.tree, self.tree_key)
        if module_dir is None or not module_dir.is_dir():
            return None
        key = os.path.normcase(str(module_dir))
        if key in self._module_headers:
            return self._module_headers[key]
        newest = None
        for path in module_dir.rglob("*"):
            if path.suffix.lower() not in HEADER_SUFFIXES:
                continue
            try:
                stamp = path.stat().st_mtime_ns
            except OSError:
                continue
            if newest is None or stamp > newest[1]:
                newest = (path, stamp)
        self._module_headers[key] = newest
        return newest


def newest_source(exe, project, index, freshness, config):
    """The newest source compiled into `exe`, with its timestamp.

    Returns (path, mtime_ns, source_count) or None when the closure holds no
    readable source at all (which is itself worth reporting).
    """
    newest = None
    count = 0
    for member in freshness.closure(project, index):
        if member.config_type in META_TARGET_TYPES:
            continue
        for unit in freshness.translated_units(member):
            try:
                stamp = unit.stat().st_mtime_ns
            except OSError:
                continue
            count += 1
            if newest is None or stamp > newest[1]:
                newest = (unit, stamp)
        header = freshness.newest_header(member)
        if header is not None:
            count += 1
            if newest is None or header[1] > newest[1]:
                newest = header
    if newest is None:
        return None
    return (newest[0], newest[1], count)


class Verdict:
    __slots__ = ("exe", "target", "newest", "count", "exe_mtime_ns")

    def __init__(self, exe, target, newest, count, exe_mtime_ns):
        self.exe = exe
        self.target = target
        self.newest = newest
        self.count = count
        self.exe_mtime_ns = exe_mtime_ns

    @property
    def stale(self):
        return self.newest is not None and self.newest[1] > self.exe_mtime_ns

    @property
    def drift_seconds(self):
        """Executable time minus newest source time: negative when stale."""
        if self.newest is None:
            return None
        return (self.exe_mtime_ns - self.newest[1]) / 1e9

    def as_record(self):
        return {
            "exe": display_path(self.exe),
            "target": self.target,
            "stale": self.stale,
            "exe_mtime_utc": iso_utc(self.exe_mtime_ns),
            "newest_source": display_path(self.newest[0]) if self.newest else None,
            "newest_source_mtime_utc": iso_utc(self.newest[1]) if self.newest else None,
            "source_drift_seconds": round(self.drift_seconds, 3)
                                    if self.drift_seconds is not None else None,
            "source_count": self.count,
        }


class UnitVerdict:
    """One translation unit against the object file the compiler writes for it.

    The executable verdict answers "does any source beat this binary" over a
    whole closure; this answers "does THIS source beat ITS object", which is the
    pair an incident names - `integrals/src/eri_cuda.cpp` two hours newer than
    the only object compiling it - and one an executable row cannot name,
    because it reports the single newest source in a closure of hundreds.
    """

    __slots__ = ("target", "source", "obj", "source_mtime_ns", "obj_mtime_ns")

    def __init__(self, target, source, obj, source_mtime_ns, obj_mtime_ns):
        self.target = target
        self.source = source
        self.obj = obj
        self.source_mtime_ns = source_mtime_ns
        self.obj_mtime_ns = obj_mtime_ns

    @property
    def stale(self):
        return self.obj_mtime_ns is not None and self.source_mtime_ns > self.obj_mtime_ns

    @property
    def built(self):
        """The object is on disk, so the pair has a verdict at all."""
        return self.obj_mtime_ns is not None

    @property
    def drift_seconds(self):
        """Object time minus source time: negative when the object is behind."""
        if self.obj_mtime_ns is None:
            return None
        return (self.obj_mtime_ns - self.source_mtime_ns) / 1e9

    def as_record(self):
        return {
            "target": self.target,
            "source": display_path(self.source),
            "object": display_path(self.obj) if self.obj is not None else None,
            "stale": self.stale,
            "source_mtime_utc": iso_utc(self.source_mtime_ns),
            "object_mtime_utc": iso_utc(self.obj_mtime_ns) if self.built else None,
            "object_drift_seconds": round(self.drift_seconds, 3)
                                    if self.drift_seconds is not None else None,
            "cannot_state": "object path not stated" if self.obj is None
                            else (None if self.built else "not built at this config"),
        }


def iso_utc(mtime_ns):
    return datetime.fromtimestamp(mtime_ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def human_age(seconds):
    sign = "-" if seconds < 0 else "+"
    seconds = abs(seconds)
    if seconds < 90:
        return f"{sign}{seconds:.0f}s"
    if seconds < 5400:
        return f"{sign}{seconds / 60:.0f}m"
    if seconds < 172800:
        return f"{sign}{seconds / 3600:.1f}h"
    return f"{sign}{seconds / 86400:.1f}d"


def check_project(project, index, freshness, config):
    """The verdict for one project's executable, or None when not built."""
    exe = project.output_exe(config)
    if not exe.is_file():
        return None
    try:
        exe_mtime = exe.stat().st_mtime_ns
    except OSError:
        return None
    found = newest_source(exe, project, index, freshness, config)
    if found is None:
        return Verdict(exe, project.output_name(config), None, 0, exe_mtime)
    return Verdict(exe, project.output_name(config), (found[0], found[1]), found[2], exe_mtime)


def application_projects(tree, config, index):
    """Every Application target under the tree, in a stable order."""
    return sorted(
        (p for p in index.values()
         if p.config_type == "Application" and p.output_exe(config).is_file()),
        key=lambda p: str(p.output_exe(config)).lower())


def object_verdicts(freshness, index, config):
    """Every translation unit of every project, against its own object file.

    It reuses Freshness.units, so "this project's sources" is the same set the
    executable verdicts walk past (existing, outside the build tree): the
    generated pch is rewritten at configure time and would otherwise read as a
    source that is always newer than its object. Both item types are swept - a
    ClCompile and a CudaCompile differ in how their object is named, not in
    whether an object that is older than its source is a defect.

    Utility projects are skipped, as they are everywhere else - they compile
    nothing, so they have no object to compare.
    """
    rows = []
    for project in sorted(index.values(), key=lambda p: str(p.path).lower()):
        if project.config_type in META_TARGET_TYPES:
            continue
        for unit, template, default in freshness.units(project):
            try:
                source_mtime = unit.stat().st_mtime_ns
            except OSError:
                continue
            obj = project.object_path(config, template, default)
            obj_mtime = None
            if obj is not None:
                try:
                    obj_mtime = obj.stat().st_mtime_ns
                except OSError:
                    obj_mtime = None
            rows.append(UnitVerdict(project.path.stem, unit, obj, source_mtime, obj_mtime))
    return rows


def report_stale_unit(verdict):
    drift = verdict.drift_seconds
    print(f"binary-freshness: STALE  {display_path(verdict.obj)}")
    print(f"binary-freshness:        {verdict.target}: {display_path(verdict.source)}")
    print(f"binary-freshness:        object {iso_utc(verdict.obj_mtime_ns)}  "
          f"source {iso_utc(verdict.source_mtime_ns)}  ({human_age(drift)})")
    print("binary-freshness:        no build has compiled this source since it changed")


def report_capped(rows, describe, cap):
    """Print at most `cap` rows, then say how many were withheld.

    Report mode only: a gate names every offender (a caller cannot act on a
    count alone), while a reporter that prints 41 five-line blocks every run
    stops being read - and --json carries the whole list for a caller that
    wants it.
    """
    for row in rows[:cap]:
        describe(row)
    if len(rows) > cap:
        print(f"binary-freshness:        ... and {len(rows) - cap} more "
              "(read them with --json)")


def report_dimension(label, fresh, stale, cannot, detail):
    print(f"binary-freshness:   {label:12s} {fresh:6d} fresh  {stale:5d} STALE  "
          f"{cannot:5d} cannot-state   ({detail})")


def project_for_exe(exe, index, config):
    """The Application project whose output is `exe`, or None."""
    wanted = path_key(exe)
    for project in index.values():
        if project.config_type != "Application":
            continue
        if path_key(project.output_exe(config)) == wanted:
            return project
    return None


def check_exe(exe, tree=None, config=None, index=None, freshness=None):
    """The freshness verdict for ONE executable, for a caller that already has it.

    This is the adoption surface the module docstring names: a benchmark runner
    that has located its binary calls this instead of shelling out to the CLI.
    Pass `index` and `freshness` from a previous call to avoid re-walking the
    tree and re-stat-ing its sources once per binary - both are caches for one
    (tree, config) pair and neither is invalidated by this function.

    Returns a Verdict (whose `.stale` answers the question, and whose `.newest`
    and `.drift_seconds` name the offending source for the message), or None
    when `exe` is not an Application output of `tree` at `config` - a private
    copy, a hand-built binary, or a path belonging to another tree. None is
    deliberately not an error: the caller's own existence check owns that case,
    and a runner must not fail because it was pointed at a binary this tree did
    not produce. **Callers that need to tell "fresh" from "not my target" must
    test for None first**, because a fresh binary returns a Verdict that is
    simply not stale.
    """
    tree = DEFAULT_TREE if tree is None else Path(tree)
    config = DEFAULT_CONFIG if config is None else config
    index = index_tree(tree) if index is None else index
    freshness = Freshness(tree) if freshness is None else freshness
    project = project_for_exe(exe, index, config)
    if project is None:
        return None
    return check_project(project, index, freshness, config)


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_path(exe):
    return exe.parent / (exe.name + ".provenance.json")


def record(exe, verdict, allow_stale, copied_from=None, closure_note=None):
    # copied_from is {"path", "sha256"} of the binary a private copy came from.
    """Write the provenance sidecar; 1 when the binary is stale and the caller
    has not accepted that explicitly (recording a stale binary is legitimate,
    but it must be a deliberate act that the record then states).

    `verdict` may be None - a private copy taken so a relink cannot swap the
    binary under a measurement is not a build target and has no closure to
    compute. The record then carries the bytes and states that the closure is
    unknown, rather than inventing one."""
    payload = {
        "note": PROVENANCE_NOTE,
        "recorded_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256_of(exe),
        "size_bytes": exe.stat().st_size,
        "file_mtime_utc": iso_utc(exe.stat().st_mtime_ns),
        "verdict": verdict.as_record() if verdict is not None else None,
        "copied_from": copied_from,
        "closure_note": closure_note,
    }
    destination = provenance_path(exe)
    try:
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8", newline="\n")
    except OSError as error:
        print(f"binary-freshness: cannot write {destination}: {error}", file=sys.stderr)
        return 2
    print(f"binary-freshness: recorded {display_path(destination)}")
    print(f"binary-freshness:   sha256 {payload['sha256']}")
    if copied_from:
        print(f"binary-freshness:   copied from {copied_from['path']} "
              f"(sha256 {copied_from['sha256'][:12]})")
        if copied_from["sha256"] != payload["sha256"]:
            print("binary-freshness: WARNING the copy and that binary have ALREADY "
                  "diverged - the recorded closure does not describe these bytes")
    if verdict is None:
        print(f"binary-freshness:   closure unknown - {closure_note}")
        return 0
    if verdict.stale:
        report_stale(verdict)
        if not allow_stale:
            print("binary-freshness: refusing to bless a stale binary silently - "
                  "rebuild it, or re-run with --allow-stale to record the drift")
            return 1
        print("binary-freshness: stale binary recorded under --allow-stale; the "
              "record states the drift above")
    return 0


def verify_record(exe):
    """Re-read a sidecar against the binary on disk now."""
    source = provenance_path(exe)
    if not source.is_file():
        print(f"binary-freshness: no provenance record at {source}", file=sys.stderr)
        return 2
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        print(f"binary-freshness: {source} is unreadable: {error}", file=sys.stderr)
        return 2
    recorded = payload.get("sha256")
    current = sha256_of(exe)
    if recorded != current:
        print(f"binary-freshness: RECORD MISMATCH {exe}")
        print(f"binary-freshness:   recorded {recorded}")
        print(f"binary-freshness:   present  {current}")
        print("binary-freshness: the measured bytes are NOT the bytes on disk - the "
              "verdict taken from this path does not belong to that binary")
        return 1
    verdict = payload.get("verdict") or {}
    if verdict.get("stale"):
        print(f"binary-freshness: record was taken on a STALE binary {exe} "
              f"(newest source {verdict.get('newest_source')}, "
              f"drift {verdict.get('source_drift_seconds')}s)")
        return 1
    print(f"binary-freshness OK: {exe.name} matches its provenance record "
          f"(sha256 {current[:12]}, built {verdict.get('exe_mtime_utc')})")
    return 0


def explain(exe, tree, config):
    """Print why a verdict came out as it did: the closure, and the newest
    source. A gate whose reasoning cannot be audited is a gate that gets
    argued with instead of believed."""
    index = index_tree(tree)
    project = project_for_exe(exe, index, config)
    if project is None:
        print(f"binary-freshness: {exe} is not an Application target of {tree} at "
              f"{config}", file=sys.stderr)
        return 2
    freshness = Freshness(tree)
    print(f"binary-freshness: {display_path(exe)}")
    print(f"binary-freshness:   built {iso_utc(exe.stat().st_mtime_ns)}")
    members = sorted(freshness.closure(project, index),
                     key=lambda p: str(p.module_dir).lower())
    print(f"binary-freshness:   closure ({len(members)} project(s)):")
    for member in members:
        units = len(freshness.translated_units(member))
        module_dir = module_source_dir(member, tree, freshness.tree_key)
        module = display_path(module_dir) if module_dir else "(no module directory)"
        marker = " [meta target, not scanned]" if member.config_type in META_TARGET_TYPES else ""
        print(f"binary-freshness:     {member.path.name:34s} {member.config_type:14s} "
              f"{units:4d} TU  {module}{marker}")
    found = newest_source(exe, project, index, freshness, config)
    if found is None:
        print("binary-freshness:   no readable source in the closure")
        return 2
    print(f"binary-freshness:   newest source {display_path(found[0])} "
          f"{iso_utc(found[1])} over {found[2]} source(s)")
    verdict = Verdict(exe, project.output_name(config), (found[0], found[1]), found[2],
                      exe.stat().st_mtime_ns)
    if verdict.stale:
        report_stale(verdict)
    else:
        print(f"binary-freshness:   FRESH ({human_age(verdict.drift_seconds)} after the "
              "newest source)")
    return 1 if verdict.stale else 0


def report_stale(verdict):
    drift = verdict.drift_seconds
    print(f"binary-freshness: STALE  {display_path(verdict.exe)}")
    print(f"binary-freshness:        built        {iso_utc(verdict.exe_mtime_ns)}")
    print(f"binary-freshness:        newer source {display_path(verdict.newest[0])}")
    print(f"binary-freshness:                     {iso_utc(verdict.newest[1])}"
          f"  ({human_age(drift)})")
    print("binary-freshness:        rebuild this target, or record why that source "
          "change cannot reach it")


def check(tree, config, wanted, as_json, report=False, objects=False):
    """Sweep a tree (or the named executables) and report.

    `objects` adds the second dimension: every translation unit against the
    object file the compiler writes for it (see UnitVerdict). It widens the
    sweep, not the gate - a stale object fails the run only because --objects
    asked for it, and no adopted caller passes it.

    `report` makes the run unfailable: exit 1 becomes 0 and the counts are the
    finding. Exit 2 stays 2 - "this tree cannot be read at all" is the third
    answer, about the instrument rather than about freshness, and it is not a
    pass (see the module docstring).
    """
    index = index_tree(tree)
    if not index:
        print(f"binary-freshness: no .vcxproj under {tree} - nothing to check "
              "(a Ninja/Makefile tree is out of this gate's scope)", file=sys.stderr)
        return 2
    freshness = Freshness(tree)
    verdicts = []
    if wanted:
        for exe in wanted:
            project = project_for_exe(exe, index, config)
            if project is None:
                print(f"binary-freshness: {exe} is not an Application target of {tree} "
                      f"at {config} - cannot state its closure", file=sys.stderr)
                return 2
            verdict = check_project(project, index, freshness, config)
            if verdict is not None:
                verdicts.append(verdict)
    else:
        for project in application_projects(tree, config, index):
            verdict = check_project(project, index, freshness, config)
            if verdict is not None:
                verdicts.append(verdict)

    applications = [p for p in index.values() if p.config_type == "Application"]
    units = object_verdicts(freshness, index, config) if objects else []

    stale = [v for v in verdicts if v.stale]
    # An executable whose closure holds no readable source is not fresh: nothing
    # was compared, so the verdict is "cannot state" (the limitation the
    # self-test pins). It is counted here and never folded into the fresh count.
    empty = [v for v in verdicts if v.newest is None]
    stale_units = [u for u in units if u.stale]
    built_units = [u for u in units if u.built]
    unstated = [u for u in units if u.obj is None]

    if as_json:
        payload = {
            "tree": display_path(tree),
            "config": config,
            "checked": len(verdicts),
            "stale": len(stale),
            "fresh": len(verdicts) - len(stale) - len(empty),
            "cannot_state": len(empty),
            "application_targets": len(applications),
            "not_built": len(applications) - len(verdicts),
            "verdicts": [v.as_record() for v in verdicts],
        }
        if objects:
            payload["objects"] = {
                "checked": len(units),
                "fresh": len(built_units) - len(stale_units),
                "stale": len(stale_units),
                "cannot_state": len(units) - len(built_units),
                "not_built": len(units) - len(built_units) - len(unstated),
                "object_path_unstated": len(unstated),
                "rows": [u.as_record() for u in units],
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif report:
        print(f"binary-freshness REPORT (this run cannot fail): "
              f"{display_path(tree)}, {config}")
        report_capped(stale, report_stale, REPORT_LIST_CAP)
        report_capped(stale_units, report_stale_unit, REPORT_LIST_CAP)
        if not stale and not stale_units:
            print("binary-freshness:   nothing in this tree is behind its sources")
        report_dimension(
            "executables", len(verdicts) - len(stale) - len(empty), len(stale), len(empty),
            f"{len(verdicts)} of {len(applications)} Application target(s) have an output "
            f"here, {len(applications) - len(verdicts)} do not")
        if objects:
            detail = f"{len(built_units)} of {len(units)} translation unit(s) have an object here"
            if unstated:
                detail += f", {len(unstated)} state an object path this script cannot resolve"
            report_dimension("objects", len(built_units) - len(stale_units),
                             len(stale_units), len(units) - len(built_units), detail)
        print("binary-freshness:   report mode: exit 0 whatever the counts above - the counts "
              "are the finding, and --json carries every row")
    elif not stale and not stale_units:
        tail = (f"; {len(built_units)} translation unit(s) newer than their object"
                if objects else "")
        print(f"binary-freshness OK: {len(verdicts)} executable(s) newer than every "
              f"source in their closure ({tree.name}, {config}){tail}")
    else:
        for verdict in stale:
            report_stale(verdict)
        for unit in stale_units:
            report_stale_unit(unit)
        tail = (f", and {len(stale_units)} translation unit(s) their own object"
                if objects else "")
        print(f"binary-freshness FAILED: {len(stale)} of {len(verdicts)} executable(s) "
              f"predate a source in their own closure ({tree.name}, {config}){tail}")
    return 0 if report else (1 if (stale or stale_units) else 0)


def main():
    parser = argparse.ArgumentParser(
        description="Report executables older than the sources compiled into them")
    parser.add_argument("--tree", default=str(DEFAULT_TREE),
                        help="build tree to sweep (default: build/windows-msvc)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="configuration directory (default: Release)")
    parser.add_argument("--exe", action="append", default=[],
                        help="check this executable instead of sweeping the tree")
    parser.add_argument("--record", metavar="EXE",
                        help="write <exe>.provenance.json for a measurement")
    parser.add_argument("--from", dest="from_exe", metavar="EXE",
                        help="with --record: the build target a private copy came from")
    parser.add_argument("--allow-stale", action="store_true",
                        help="with --record: record a stale binary, stating the drift")
    parser.add_argument("--verify-record", metavar="EXE",
                        help="re-check an executable against its provenance record")
    parser.add_argument("--objects", action="store_true",
                        help="also check every translation unit against the object file "
                             "the compiler writes for it")
    parser.add_argument("--report", action="store_true",
                        help="reporter posture: print the counts, never exit 1 "
                             "(exit 2 - the tree cannot be read - is unchanged)")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.add_argument("--explain", metavar="EXE",
                        help="print the closure behind one executable's verdict")
    args = parser.parse_args()

    if args.verify_record:
        return verify_record(Path(args.verify_record))

    tree = Path(args.tree)
    if not tree.is_absolute():
        tree = REPO_ROOT / tree

    if args.record:
        exe = Path(args.record)
        if not exe.is_absolute():
            exe = REPO_ROOT / exe
        if not exe.is_file():
            print(f"binary-freshness: no such executable: {exe}", file=sys.stderr)
            return 2
        index = index_tree(tree)
        freshness = Freshness(tree)
        # A private copy has no project of its own; --from names the binary it
        # was copied from, whose closure is the honest answer for the copy.
        subject = exe
        copied_from = None
        if args.from_exe:
            subject = Path(args.from_exe)
            if not subject.is_absolute():
                subject = REPO_ROOT / subject
            if subject.resolve() == exe.resolve():
                subject = exe
            elif subject.is_file():
                copied_from = {"path": display_path(subject),
                               "sha256": sha256_of(subject)}
        project = project_for_exe(subject, index, args.config)
        if project is None:
            return record(exe, None, args.allow_stale, copied_from,
                          closure_note=(f"{display_path(subject)} is not an Application "
                                        f"target of {display_path(tree)} at "
                                        f"{args.config}, so its source closure cannot be "
                                        "computed from this path"))
        verdict = check_project(project, index, freshness, args.config)
        if verdict is None:
            return 2
        return record(exe, verdict, args.allow_stale, copied_from)

    if args.explain:
        exe = Path(args.explain)
        if not exe.is_absolute():
            exe = REPO_ROOT / exe
        return explain(exe, tree, args.config)

    wanted = [Path(e) if Path(e).is_absolute() else REPO_ROOT / e for e in args.exe]
    return check(tree, args.config, wanted, args.json, args.report, args.objects)


if __name__ == "__main__":
    sys.exit(main())
