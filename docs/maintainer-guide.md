# Maintainer guide

This page is the working manual for the repository: how to build it, which tests a change
must run, how to add a module, what enforces the conventions, and what a reviewer will ask
for. The ground rules for a contribution — the module DAG, errors-as-values, do not weaken
a gate, implement from the published algorithm, state your evidence, regeneration is
local-only, small changes — are in `CONTRIBUTING.md` and are not repeated here.

## 1. Build

Requirements are CMake 3.27 or newer, a C++23 compiler, Git and the pinned submodules.
Boost and HDF5 come from a vcpkg tree, which the `windows-msvc` preset points at.

```powershell
git clone --recurse-submodules <repository-url>
# or, in an existing clone:
git submodule update --init

cmake --preset windows-msvc
cmake --build build/windows-msvc
```

The presets are the supported configurations, and they are the reason a local build
matches CI:

| Preset | What it builds |
| --- | --- |
| `windows-msvc` | the primary configuration (configure preset) |
| `windows-msvc-debug`, `windows-msvc-release` | the two build presets over it |
| `windows-msvc-asan` | AddressSanitizer, MSVC only |
| `windows-msvc-cuda` | the CUDA backend, local-only — CI never builds it |
| `windows-msvc-a3c`, `windows-msvc-pf` | the narrower local configurations |
| `wsl-gcc`, `wsl-clang` | the GCC and Clang parity configurations, Debug |
| `wsl-gcc-release`, `wsl-clang-release` | the same configurations in Release |

Run both parity configurations before asking for review: some diagnostics are
compiler-specific, and the compiler you did not build with is the only one that reports
them.

A multi-configuration generator is in use, so a module's test binary lands under its
configuration: `build/windows-msvc/<module>/Release/qcx-<module>-tests.exe`.

The options that matter when configuring:

| Option | Default | Effect |
| --- | --- | --- |
| `QCX_BUILD_TESTS` | `ON` | builds the module test targets |
| `QCX_ENABLE_IO` | `ON` | HDF5-backed storage and checkpointing |
| `QCX_ENABLE_CUDA` | `OFF` | the CUDA backend |
| `QCX_ASAN` | `OFF` | AddressSanitizer (MSVC only) |
| `QCX_BUILD_BENCHMARKS` | `OFF` | the scaling benchmarks, local-only |
| `QCX_ENABLE_LARGE_TESTS` | `OFF` | registers the long-running test targets (see §3) |
| `QCX_REFERENCE_REGENERATION_CHECK` | `OFF` | the local-only regeneration checks |
| `QCX_TEST_TIMEOUT_SECONDS` | `2400` | the per-test timeout |
| `QCX_INTEGRALS_LMAX` | `6` | the highest angular momentum the integral engine builds for |

## 2. Test selection

**Run the subset the change reaches, not the whole suite.** The module DAG makes that a
derivation rather than a guess: `core → backend → memory → linalg → molecule → symmetry →
basisset → grid → integrals → scf → storage → io → properties → driver`. A change in
module X can move X and every module to its right, and nothing to its left — so the test
binaries from X onward are the only ones that can tell you something.

```powershell
# the modules a change to integrals/ reaches
foreach ($m in 'integrals','scf','storage','io','properties','driver') {
    & "build/windows-msvc/$m/Release/qcx-$m-tests.exe"
}
```

Narrow further with the GoogleTest filter while iterating:

```powershell
& build/windows-msvc/integrals/Release/qcx-integrals-tests.exe --gtest_filter='*Schwarz*'
```

The full selection — what CI runs — is the whole `ctest` set:

```powershell
ctest --test-dir build/windows-msvc -C Release -LE large -j 8 --output-on-failure
```

The `-LE large` is not optional. `large` is the single exclusion mechanism for
long-running tests, and leaving it off puts the multi-minute cases back in a routine run.
Run the suite in Debug as well before opening a pull request.

## 3. The `large` category

A test that takes minutes rather than seconds carries the `large` label so that a routine
`ctest` never picks it up. There are two ways a test gets that label, and the difference
matters:

- **Preferred: put the source in `<module>/tests/large/`.** It is compiled into
  `qcx-<module>-large-tests`, a target that only exists when `QCX_ENABLE_LARGE_TESTS` is
  `ON`. That absence is the fail-safe: a routine `ctest` cannot see a test that was never
  registered, so forgetting a flag cannot pull one back in.
- **When the source must stay in the normal target** — because another change owns the
  file, or its long and short cases cannot be separated — name the cases in the module's
  `qcx_mark_large_cases(...)` manifest instead. The manifest is checked at ctest time
  against the names the test target actually registered, so a renamed case fails loudly
  rather than silently rejoining the routine selection.

Large tests are excluded by the gate and the label, **never by skipping themselves**: the
category exists to run full strength, so it applies no fast-test mode.

```powershell
cmake --preset windows-msvc -DQCX_ENABLE_LARGE_TESTS=ON
cmake --build build/windows-msvc
ctest --test-dir build/windows-msvc -C Release -L large -j 8 --output-on-failure
```

Release only — never Debug, and never in the per-push CI run.

## 4. What the commit hook enforces

Enable it once per clone:

```powershell
git config core.hooksPath .githooks
```

The hook inspects the **staged** state and refuses the commit on the first failure. In
order, it checks that the index can actually describe the commit, normalizes staged text
files to LF (correcting the staged copy only — your working tree is never touched),
verifies that staged C++ is already clang-format-clean, and then runs the repository's
guards:

| Check | Enforces |
| --- | --- |
| DAG | `tools/check_dag.py` — no module depends on one to its right |
| Docs-gate coverage | the Doxyfile `INPUT` list covers every module in the DAG, and the gate is still able to fail |
| Convergence gates | no convergence tolerance was loosened past its allowed floor |
| Style guards | the blank line before and after a multi-line control statement, and the naming and raw-array scans |
| Vocabulary contract | the builder kinds, the words that spell them and the driver's wiring agree |
| Workspace grant | every driver budget site is handed the shared workspace grant |

The docs-gate coverage check is worth understanding, because it exists to close a real
failure: the Doxyfile `INPUT` list *is* the gate's coverage, so a module missing from it
is a module the gate reports zero warnings about without ever reading it. The check takes
the module list from the DAG checker rather than keeping its own copy, and reports a
missing module and a stale entry as the two different mistakes they are.

The gate's own failure mode is blunt: it builds with `WARN_AS_ERROR`, which stops at the
first warning. A green run really is zero warnings, but a red run shows you one warning at
a time — so when you are fixing several, re-run rather than assuming the first was the
only one.

## 5. Adding a module

A module is a directory plus four registrations, and the registrations are what make it a
module rather than a folder.

1. **Decide its rank, and add it to the DAG's order.** The module list has one home — the
   order table in `tools/check_dag.py`, which the docs-gate coverage check reads rather
   than copies. Adding a module means editing that one list; do not create a second.
2. **Create the layout:** `<module>/include/qcx/<module>/` for public headers,
   `<module>/src/` for the implementation, `<module>/tests/` for the tests.
3. **Write `<module>/CMakeLists.txt`** so it creates `qcx-<module>`, and link only
   modules that come before it in the order. The DAG check reads this file, so an illegal
   edge is caught by the hook rather than by a reviewer.
4. **Add `add_subdirectory(<module>)` to the root CMakeLists**, in DAG order.
5. **Add the module's include directory to the Doxyfile `INPUT` list.** The docs-gate
   check fails without it, deliberately: a module outside `INPUT` is a module the
   documentation build never reads.
6. **Give every public declaration its Doxygen comment and its `\ingroup qcx-<module>`.**
   The group is what places a declaration on the module's page in the generated reference
   and what makes it addressable by a `\ref` from another page. A documented declaration
   that omits the group is compiled and documented but has nowhere to appear, which is how
   a public function ends up invisible in the reference.
7. **Set up the precompiled header.** A precompiled header is mandatory on every
   first-party target — `target_precompile_headers(<target> PRIVATE <header>...)`, with
   angle-bracketed entries only and no `REUSE_FROM`. A bare path instead of an
   angle-bracketed one fails with C1083, and a CUDA target gets a CUDA-safe list only.
8. **Register the tests** so they compile into `qcx-<module>-tests`, and put any case that
   runs for minutes into the `large` category (§3).
9. **Cite anything derived.** An algorithm, kernel or fitted constant lands with its
   published source recorded, and the comment in the code cites it by its key from
   `CITATION.bib` — never an internal document.
10. **Keep allocation in the memory module.** A new module uses the existing tensors,
    buffers and workspace budgets rather than allocating for itself.

## 6. Style, and what catches a slip

The rules are in `CONTRIBUTING.md`; this is the enforcement behind them, so you know what
will be caught before review rather than by review.

- **Naming** — PascalCase types and functions, camelCase variables, `_camelCase` members,
  kPascalCase constants and enumerators, short lowercase namespaces, `Qcx`-prefixed
  macros, and no snake_case anywhere. The hook scans for the violations. There is one
  deliberate exemption: a public aggregate struct's fields are its API, so they are plain
  camelCase with no member underscore.
- **Formatting** — `.clang-format` is not advisory. The hook refuses staged C++ that is
  not already clean, so run it on your change rather than editing the config to suit it.
- **Static analysis** — `tools/run_clang_tidy.sh` is the sweep the clang-tidy CI job
  runs, over the same translation units, against the `wsl-clang` compile database.
  `.clang-tidy` sets `WarningsAsErrors: '*'`, so a finding reddens that job rather than
  being printed and left in the log: run it before asking for review.
- **Control statements** — always braces on their own line, never a one-liner, with a
  blank line before and after a multi-line control statement. clang-format cannot express
  the blank-line rule, which is why the hook checks it separately.
- **Headers** — `#pragma once`, and a Doxygen comment on every public declaration with
  `\param`, `\return` and `\pre` where a precondition exists.
- **Raw arrays** — `std::array<double, 3>` for coordinates and centers, `std::span` for
  pointer-plus-count APIs, and a raw array only where an ABI mandates one.
- **Comments** — brief, self-contained, why-focused, and citing published work by its
  citation key rather than an internal document.
- **Line endings** — LF everywhere.

## 7. Review conventions

A change is reviewed against these questions, so answering them in the description is the
fastest route through review.

- **Does it need a DAG edge that does not exist?** If so the change is what is wrong, not
  the DAG. Say which module the code belongs in and why it is not the one it is in.
- **What is the evidence?** A new algorithm, kernel or fitted constant names its source
  and what verified it — a paper, a finite-difference check, an analytic derivation. An
  addition with no stated evidence will be asked for one.
- **Did anything get weaker?** A moved tolerance, a relaxed gate or an adjusted expected
  value is the thing a reviewer looks for first. If a number moved, report the movement
  and say whether it was intended; do not adjust an expectation to match new output.
- **Are the numbers still pinned where they matter?** A new code path lands with the test
  that distinguishes it from the old one, and a numerical expectation lives next to the
  reasoning that produced it rather than standing as an unexplained constant.
- **Is it one logical change?** One change per pull request, small enough to review.
- **Is the public surface still honest?** A new public declaration has its comment, its
  group, and — if it is a boundary — its entry in this guide and the reference. A change
  that breaks a documented signature, unit or domain is a major-version change.
- **Does the documentation still describe the code?** The reference is generated from the
  headers, so a stale claim in prose is a defect in its own right: if this guide or the API
  guide contradicts the headers, the headers win and the page needs fixing.

Exact bitwise output is not promised across releases — it depends on the compiler, the
flags and the configuration — so a review does not ask for byte-identical results, only
for the tolerances and the convergence behaviour to be what the pages above say they are.
