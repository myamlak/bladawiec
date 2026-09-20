# Contributing to qcx

Thanks for considering a contribution. Read this whole file before opening a
pull request — a few of the rules here are unusual, and they exist because
this is a numerical library whose numbers are its contract.

## What this project is

A C++23 quantum-chemistry framework: Gaussian basis sets, one- and
two-electron integral engines, symmetry detection, DFT integration grids,
self-consistent field solvers, checkpointed storage and derived properties.
It is organized as a module DAG — `core → backend → memory → linalg →
molecule → symmetry → basisset → grid → integrals → scf → storage → io →
properties → driver` — and the dependency direction is enforced, not merely
documented. See the [README](README.md) for the build and the module map.

## Ground rules

1. **The module DAG is the architecture.** A module may depend on the
   modules to its left, never on one to its right and never sideways.
   `tools/check_dag.py` enforces it and runs in CI and in the commit hook.
   If a change seems to need an edge that the DAG forbids, the change is
   what is wrong.
2. **Errors are values.** No exceptions. Fallible code returns
   `qcx::Result<T>` (a `std::expected`); fallible construction is a
   `static Result<T> Create(...)` factory rather than a throwing
   constructor. Consumers translate the code at their own boundary.
3. **Do not weaken a test, a tolerance, or a gate to make a change pass.**
   The reference values, the Doxygen gate and the module tests are the
   reasons the numbers can be trusted. If a change turns one red, the
   change is the suspect.
4. **Implement from the published algorithm.** An implementation is written
   from the algorithm as published — a paper, a monograph, a standard — and
   not from another program's source. Another project may confirm a number
   afterwards; it never supplies the expression.
5. **State your evidence.** If you add an algorithm, a kernel or a fitted
   constant, say in the pull request what verified it — the published paper
   it comes from, a finite-difference check, an analytic derivation. An
   addition with no stated evidence will be asked for one.
6. **Regeneration is local-only.** Anything generated and committed —
   coefficient tables, element data, vendored basis sets — is checked by a
   script that verifies the committed bytes against a fresh run
   (`python tools/fetch_basis_data.py --check` for the basis corpus, and
   the equivalent check for the table you touched). Never commit
   regenerated output without running its check, and never wire
   regeneration into CI.
7. **Small, reviewable changes.** One logical change per pull request.

## Build and test

Requirements: CMake 3.27 or newer, a C++23 compiler (MSVC, GCC or Clang),
Git, and the pinned submodules. Boost and HDF5 come from a vcpkg tree; see
the [README](README.md) for the preset that points at it.

```powershell
git clone --recurse-submodules <repository-url>
# or, in an existing clone:
git submodule update --init

cmake --preset windows-msvc
cmake --build build/windows-msvc
ctest --test-dir build/windows-msvc -C Release -LE large -j 8 --output-on-failure
```

Before you read a verdict from a suite run, check that the binaries are newer
than the sources they were built from — `python tools/check_binary_freshness.py`
fails on any executable that predates a source in its own closure. A run
against a stale executable is a false green: it proves nothing about the tree
you are proposing.

Please run the suite in Debug as well before opening a pull request, and
run the subset the change reaches rather than the whole suite while you
iterate: by the module DAG, a change in module X reaches X and every module
after it, so the binaries from X onward are the ones that can move. The full
`ctest` selection is what CI runs.

The AddressSanitizer preset (`windows-msvc-asan`) and the CUDA preset
(`windows-msvc-cuda`) are local-only; CI never builds the CUDA backend.

## Style rules

- **Naming:** PascalCase types and functions, camelCase variables,
  `_camelCase` private members, kPascalCase constants and enums, short
  lowercase namespaces, `Qcx`-prefixed macros. No snake_case anywhere —
  the commit hook scans for it.
- **Formatting:** [.clang-format](.clang-format) is not advisory — 4-space
  indent, 100 columns, attached braces, left-aligned pointers. Run it on
  your changes rather than editing the config to suit them. Control
  statements always take braces on their own line, never a one-liner, and a
  multi-line control statement gets a blank line before and after it.
- **Headers:** `#pragma once`; Doxygen `///` on every public declaration
  with `\param`, `\return` and `\pre` where a precondition exists. The
  documentation gate builds with `EXTRACT_ALL = NO` and fails on any
  warning.
- **Raw arrays:** 3-vector coordinates use `std::array<double, 3>`;
  pointer-plus-count APIs use `std::span`; raw arrays only where an ABI
  mandates them.
- **Comments:** brief, self-contained and why-focused. Cite published work
  by its citation key from [CITATION.bib](CITATION.bib) — never an internal
  document.
- **Line endings:** LF everywhere, per [.gitattributes](.gitattributes).

The commit hook rejects staged C++ that is not clang-format-clean and runs
the style guards, so a formatting slip is caught before it reaches review.

## Tests

- Module tests live in `<module>/tests/`. A change lands with the tests
  that pin its behaviour: a new kernel against its reference values and its
  boundary cases, a new code path against the case that distinguishes it
  from the old one.
- Numerical expectations belong in the test next to the reasoning that
  produced them, not as an unexplained constant.
- Tests must pass in Debug and Release and on the CI matrix.
- Do not adjust an expected value to match new output unless the change was
  intended to move that number; report the movement instead.

## What never goes in

- References to the development repository this code is cut from:
  internal paths, decision numbers, stage or track names, or notes about
  how the work was organized internally.
- Vendored code beyond the pinned submodules, or vendored data without its
  provenance recorded.
- Exceptions, or a fallible constructor where a `Create` factory belongs.
- Regenerated output without its check having been run.
- Third-party code or data without its license recorded in
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

BSD-3-Clause — see [LICENSE](LICENSE). By contributing, you agree that your
contribution is licensed under the same terms. Third-party components and
their licenses are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Versioning

The project follows Semantic Versioning. The version identifier lives in
`CMakeLists.txt`, the [Doxyfile](Doxyfile) and [CITATION.cff](CITATION.cff),
and the three are expected to agree.

- **MAJOR** — an incompatible public API change (removal, rename,
  signature, layout, namespace), or a change to a promised unit, domain or
  reference value.
- **MINOR** — additive public API, or an internal numerical change that
  leaves every documented signature, unit and domain intact.
- **PATCH** — no intended public-API or numerical change.

Public headers and their documented domains are stable within a major
version. Exact bitwise output is **not** promised across releases: it
depends on the compiler, the flags and the build configuration. Pin the
release tag if you need reproducibility.

## Getting help

Open an issue for questions, bug reports, or proposals before writing code.
The maintainer reviews and merges every change to `main`.
