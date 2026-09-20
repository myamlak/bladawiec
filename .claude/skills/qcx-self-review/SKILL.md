---
name: qcx-self-review
description: Deep self-review of code changes against qcx conventions (naming, Doxygen, Result-based errors, RAII, DAG, tests, device correctness) producing a prioritized remediation plan.
---

# /qcx-self-review

Perform a thorough, deep self-review of **all code added or changed on the
current branch** (`git diff <base>...HEAD` in a qcx checkout), then produce a
prioritized, actionable remediation plan. Apply maximum reasoning depth: treat
every finding as a real defect until you have verified that it is not.

The rules a change is held to live in the repository's own pages, and this file
is the checklist to work through them with rather than a second copy of them.
Read `CONTRIBUTING.md` (the ground rules, the style rules, what never goes in)
and `docs/maintainer-guide.md` (the build, the test selection and the `large`
category, what the commit hook enforces, how a module is added, and the
questions review asks) before starting, and apply them throughout. Where this
file and those two disagree, they win. The depth below is the part they do not
carry: what each criterion actually means when you are reading the code.

## 1 — Establish scope

Find the divergence point from the base branch and enumerate every changed
file. Then read the **complete content** of every changed file — headers,
implementations and tests. A partial read produces a false-clean report. Do
not skim.

## 2 — Build an inventory

Before touching any criterion, write out a complete list of **every function,
class, struct, type alias, enum, and lambda** in the changed code. Apply each
criterion below to every item on that list — the inventory exists to force
complete coverage.

## 3 — Systematic analysis

---

### A — Documentation coverage, and the gate that measures it

**Declarations in public headers (`*/include/`):** every function, class,
struct, enum, and type alias carries a Doxygen comment in **`///` line
comments** (never `/** */`), placed directly above the declaration, with **the
first sentence as the brief** — no `\brief` keyword, which would restate it.
Required commands:

- `\param <name>` — once per parameter, the name matching the signature exactly
- `\tparam <name>` — once per template parameter whose role is not obvious
- `\returns` — whenever the return type is not `void` and the brief does not
  already say what comes back
- `\pre` — where the declaration has a precondition
- `\ingroup qcx-<module>` — on entities in multi-header modules; the module's
  `\defgroup` must exist, and `docs/maintainer-guide.md` §5 says why the group
  is not optional
- trailing `///<` briefs on enumerators and data members

Comments live on the **declaration in the header**, not only on the
definition. Verify the text matches the current signature and behaviour: an
outdated comment is as bad as a missing one. A `.cpp` file carries plain `//`
implementation notes and never `///`.

**Implementation-only functions and internal helpers (`.cpp`):** every
function without a header declaration needs at minimum a one-line comment
saying what it does — including lambdas and anonymous-namespace helpers.

**Gate:** `cmake --build build/windows-msvc --target qcx-docs` must emit
**zero warnings** (`WARN_IF_UNDOCUMENTED`, `WARN_NO_PARAMDOC` and
`WARN_IF_DOC_ERROR` are the linter), and `EXTRACT_ALL` must stay `NO` — with
`YES`, Doxygen silently disables `WARN_IF_UNDOCUMENTED`. Run it and report its
output. The gate stops at the first warning (`WARN_AS_ERROR`), so when you are
fixing several, re-run rather than assuming the first was the only one.

**Flag:** missing `///` blocks, missing or mismatched `\param`/`\tparam`/
`\returns`, stale text, missing one-liners on internal functions, doc-build
warnings.

---

### B — Unit test coverage

For every new or modified public function and class:

1. Confirm a test exists that exercises its primary behaviour — module tests
   live in `<module>/tests/` (wired with `gtest_discover_tests`),
   cross-cutting tests in `tests/`.
2. Think through and flag untested edge cases: empty or zero-size input, a
   single element, boundary values, invalid input including the `Result` error
   paths (`Create(0)` → `kInvalidArgument`), the scenario that motivated the
   change, and concurrency where it is relevant.
3. Flag every deleted test explicitly — deletion requires a documented
   justification.
4. Aliases and thin wrappers still need a smoke test.
5. **Numerical code must be cross-validated against an independent
   implementation or an analytic derivation** where one exists, and self-
   consistency alone is not validation. Another project may confirm a number
   afterwards; it never supplies the expression —
   `CONTRIBUTING.md` states that rule and this is where it is checked.
6. Device-touching tests are host-side `.cpp` files — a `.cu` only when it
   contains kernels, because nvcc compiles host code as C++20 while the
   headers are C++23 — and they are gated by `QCX_ENABLE_CUDA`.
7. Compile-time constraints are probed with `std::invocable` — **never** with
   a requires-expression calling the constrained member (clang hard-errors,
   MSVC false-positives).

**Flag:** missing tests, insufficient edge or error-path coverage, unjustified
test deletion, self-consistency-only numerical tests, `.cu` tests that could
be host-side.

---

### C — File organization, module boundaries, and the DAG

Ask of every changed file: *"What is this file about, in one sentence?"* — an
"and" joining unrelated concepts means mixed responsibilities; flag it.

- The module DAG and the direction it may be linked in are in
  `docs/maintainer-guide.md`; a `target_link_libraries` to a later module is a
  violation, and `python tools/check_dag.py` must pass (the commit hook runs
  it, and it is what makes an illegal edge a build failure rather than a
  review finding).
- New modules start as real libraries (not `INTERFACE`) once they have
  content; a header-only template module keeps a placeholder `.cpp` so the
  target stays a real library.
- **Exclusive allocation site:** `new`/`cudaMalloc` and friends exist only in
  `memory/`. Tests may own memory directly through `std::unique_ptr` and a
  RAII deleter.
- Header-only template modules keep templates in headers, with `.cpp`
  translation units only for non-template code.
- Include order: project headers (`"..."`) first, then third-party and system
  (`<...>`), sorted within each group. Each include must be required by the
  declarations in that header — flag extras.

**Flag:** mixed-responsibility files, DAG violations, allocation outside
`memory/`, unnecessary includes, wrong-file placement.

---

### D — Function length and lambda extraction

Functions longer than ~60 lines almost always contain hidden
sub-responsibilities — identify natural split points and propose names.
Lambdas longer than ~10 lines are named functions in disguise — propose a name
and a signature, and check for captures that would be cleaner as a helper
class. Also flag multiple early returns combined with substantial later logic,
deeply nested conditionals, and more than one distinct algorithm in one body.

**Flag:** long functions (with proposed splits), complex lambdas (with
proposed names), entangled control flow.

---

### E — In-function readability and intent

- Non-trivial algorithm sections need a comment describing *what* they compute
  or *why* — never a restatement of the names on the line.
- Non-obvious decisions (formulas, thresholds, workarounds, trade-offs) need a
  "why" comment.
- Magic numbers and unnamed constants must be named, with the `k` prefix the
  naming rules require.
- Dense math (integrals, tensor index arithmetic) must carry a reference or a
  derivation.
- Misleading names are documentation defects even when the code is correct.

**Flag:** unexplained algorithms, missing rationale, magic numbers,
misleading names.

---

### F — Performance

Scan for unnecessary copies (missing `const&`, missing `std::move` on
returns), hoistable loop-invariant computation, needless O(n²) where
O(n log n) is straightforward, missing `reserve()` before a known-size
`push_back` sequence, redundant container lookups, and cache-unfriendly
strides in hot paths. For a significant algorithmic gain, suggest a benchmark
test.

Dispatch decisions (a vendor-BLAS threshold, a backend choice) belong at the
coarse-grained level only — never inside a hot inner loop — and a small
fixed-size block is Eigen's work, not BLAS's.

**Flag:** measurable performance issues, each with a concrete fix.

---

### G — Documentation proofreading — stale and dead references

Applies to **every comment in every changed file**, not just the newly added
ones:

- Every `\param`/`\tparam` name exactly matches the current signature;
  `\returns` matches what is actually returned today.
- Cross-references ("See `Foo::Bar()`", "delegates to `helper()`") must
  resolve; an entity renamed mid-task leaves a dead link behind.
- Comments describing behaviour, constraints, invariants or preconditions must
  still hold for the current implementation — a residency precondition on
  `HostView()` or `DeviceHandle()` that the code no longer honours is a
  finding, not a stale nicety.
- Flag leftover TODO/FIXME/HACK/NOTE that refer to completed work, and
  commented-out dead code.

**Flag:** mismatched doc tags, dead cross-references, behaviourally stale
comments, stale preconditions, leftover TODOs, dead code.

---

### DS — Device and backend correctness (highest severity when violated)

- **Residency discipline:** every device-backed buffer and tensor follows the
  state machine documented in `memory/include/qcx/memory/device_buffer.hpp` —
  `MarkHostDirty()`/`MarkDeviceDirty()` on writes, `SyncToHost()`/
  `SyncToDevice()` (their `Result` checked) before a read from the other side,
  and `HostView()`/`DeviceHandle()` only under their documented canonical-side
  precondition. `CurrentResidency()` is what answers which side is current.
- **No swallowed device errors:** every fallible device operation returns
  `qcx::Result` with `ErrorCode::kDeviceError` (the message embeds
  `cudaGetErrorString`). The one unreportable path — a destructor's
  `cudaFree` — logs through `qcx::log`. Never `(void)` a device error on a new
  path.
- **No raw `T*` to device memory:** device access goes through `DevicePtr` at
  the kernel boundary, and host code never dereferences a device pointer.
- **Fallible construction is a factory:** a constructor that can fail is
  prohibited — use `static Result<T> Create(...)`.
- **The CUDA boundary is C++20:** `std::expected` (C++23) never appears in a
  `.cu` translation unit. Device error text crosses the nvcc boundary as plain
  `std::string` (the driver's text, empty on success) through the bridge header
  `backend/src/cuda_backend_detail.hpp`, and only the C++23 layer converts it
  to `Result`/`kDeviceError`.
- **A backend-agnostic module stays backend-agnostic:** no host-only
  assumption leaking into code that must run on any backend.
- **Compiler divergence:** `/Zc:__cplusplus`, `/Zc:preprocessor` and `/FS` are
  CXX-scoped flags — a bare one breaks nvcc; OpenMP loops use signed indices
  (MSVC C3016).

**Flag:** residency-machine misuse, swallowed device errors, raw device
pointers dereferenced on host, throwing or failing constructors, C++23
leaking into a `.cu` translation unit.

---

### CS — Code style

Priority order: **repository guidelines first** (`CONTRIBUTING.md`,
`docs/maintainer-guide.md` §6, `.clang-format`, `.clang-tidy`) — they override
everything below; **C++ Core Guidelines second**, for what they leave open.

- **Naming and formatting** are stated in those two pages and enforced by the
  commit hook; check the file is already `.clang-format`-clean rather than
  expecting a reviewer to run it. clang-format cannot express the blank-line
  rule around a multi-line control statement, which is why the hook checks it
  separately.
- **No member may share a name with a type in scope** — it changes the type's
  meaning (a hard error under GCC's `-Wchanges-meaning`, which clang-tidy
  escalates; plain clang and MSVC accept it silently). `DeviceHandle()` rather
  than `devicePtr()` is the shape, and
  `memory/include/qcx/memory/device_buffer.hpp` writes the reason at the
  declaration.
- **Error handling:** `qcx::Result`/`std::expected` is the foundation — no
  exceptions, no out-parameters carrying errors, no error code swallowed.
- **C++23 throughout**, except in `.cu` translation units.
- **Third-party dependencies** are pinned submodules under `third_party/` and
  `external/` (a release tag or a recorded SHA, never floating `main`) or come
  from vcpkg. A new vendored dependency needs a recorded reason, and a
  component or dataset lands with its licence in `THIRD_PARTY_NOTICES.md`.
- **Citations:** an algorithm, kernel or fitted constant that comes from a
  publication is cited in the comment by its key from `CITATION.bib` — the
  shape is `[Helgaker2000]` — and the entry exists. A number with no source
  recorded beside it is a finding, not a style preference.
- Core Guidelines spot checks: RAII for every resource `[R.1]`, ownership by
  smart pointer rather than a paired `new` `[R.11]`/`[R.23]`, raw pointers
  non-owning `[R.3]`, `noexcept` when it cannot throw `[F.6]`, comments
  explaining *why* rather than *what* `[NL.1]`, always initialize `[ES.20]`,
  `nullptr` rather than `0` `[ES.47]`.

**Flag:** every violation, with its source (the repository rule, or the Core
Guideline reference) and a concrete fix.

---

### CI — Continuous-integration and budget impact

- A changed build requirement must keep **BLAS-less builds green**: CI and the
  parity configurations configure without vcpkg, so the vendor-BLAS-gated path
  must compile clean without it.
- No new always-on CI jobs. Windows minutes bill at a higher rate and are
  gated on the Linux matrix, so a change that needs a new toolchain step is a
  red flag worth raising rather than adding.
- `tools/check_dag.py` must pass, and the commit hook must be able to run
  (`git config core.hooksPath .githooks`).
- The clang-tidy sweep must be able to parse every `*.cpp` it scans: a file
  that needs a local-only toolchain (the CUDA toolkit) is excluded from the
  tidy file list. The workflow states its own reading of this — see
  `.github/workflows/ci.yml`.
- New CUDA code stays behind `QCX_ENABLE_CUDA` (default `OFF`).

**Flag:** CI-breaking requirements, un-gated optional dependencies,
budget-affecting workflow changes.

---

## 4 — Produce the remediation plan

Output the plan in this structure. Include `file:line` for every finding.
Omit sections with no entries. The plan is the deliverable — be specific and
actionable.

```
# Self-Review Remediation Plan

## Critical — fix before commit/push
### [A1]: Short descriptive title
- **Where:** src/foo/bar.hpp:42
- **Issue:** What is wrong and why it matters
- **Fix:** Exact action to take

## Important — should fix
### [B1]: Short descriptive title
- **Where:** …
- **Issue:** …
- **Fix:** …

## Suggestions — nice to have
### [F1]: Short descriptive title
- **Where:** …
- **Issue:** …
- **Fix:** …

## Clean — no issues found
- [List each criterion letter or file that passed with no findings]
```

**Severity classification:**

| Severity | Criteria |
|---|---|
| **Critical** | Missing `///` Doxygen on any public API declaration; missing tests for new non-trivial logic; unjustified test deletion; DAG violation; allocation outside `memory/`; residency-machine misuse; swallowed device errors; raw device pointers dereferenced on host; failing constructors; `\param` mismatch with the current signature; behaviourally inaccurate comment on public API; doc-build warnings; CI-breaking requirements |
| **Important** | Missing one-liners on internal functions; complex unnamed lambdas; unexplained algorithms or missing "why" rationale; clear performance issues with a straightforward fix; dead cross-references; stale constraint or precondition notes; snake_case anywhere; `.cu` files that could be host-side `.cpp`; missing error-path tests; ownership hazards of the `[R.11]` kind |
| **Suggestion** | Optional benchmark tests; minor clarity refactors; leftover TODOs for completed work; commented-out dead code; `[NL.1]` comments that restate the code |

Report the plan as written. Apply fixes only when asked for them.
