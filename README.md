# bladawiec (qcx)

A quantum-chemistry framework for molecular electronic structure — integrals, SCF, and the
properties that follow. The methods it implements are the published ones, listed with their
sources in [`CITATION.bib`](CITATION.bib).

C++23, C++ Core Guidelines, native Windows/MSVC as the primary build target (Linux CI for parity).

## Module DAG

```
core -> backend -> memory -> linalg -> molecule -> symmetry -> basisset
```

Upward/sideways dependencies fail the build — enforced by `tools/check_dag.py`
(CI job + pre-commit hook).

## Dependencies

Vendored as pinned git submodules under `third_party/`:

| Library | Pin | Purpose |
|---|---|---|
| Eigen | 5.0.1 (official GitLab mirror) | linear algebra |
| spdlog | v1.17.0 | logging backend |
| GoogleTest | v1.17.0 | unit tests |
| stdexec | 2aaae5c (pinned commit) | std::execution (P2300) reference implementation |
| nlohmann/json | v3.12.0 | JSON I/O |
| toml++ | v3.4.0 | declarative input schema |
| Spectra | v1.2.0 | iterative eigenproblems (linalg seam) |
| amgcl | 1.5.0 | AMG sparse solver (linalg seam) |
| Google Benchmark | v1.9.5 | scaling benchmarks (local-only, `QCX_BUILD_BENCHMARKS`) |
| HighFive | v2.10.1 | HDF5 checkpoint/restart wrapper |

Boost (Boost.Graph) comes from vcpkg; the same classic-mode vcpkg tree supplies HDF5.
`QCX_ENABLE_IO` (HDF5) is ON by default and OFF in the CI Windows jobs.

## Building

Prerequisites: Visual Studio 2026 Community ("Desktop development with C++"), CMake ≥ 3.27,
Git, and the submodules:

```powershell
git clone --recurse-submodules git@github.com:myamlak/bladawiec.git
# or, in an existing clone:
git submodule update --init
```

Configure and build (native Windows):

```powershell
cmake --preset windows-msvc
cmake --build build/windows-msvc
ctest --test-dir build/windows-msvc --output-on-failure
```

AddressSanitizer build (`QCX_ASAN` option; MSVC-only — Clang/GCC parity builds run in WSL/CI):

```powershell
cmake --preset windows-msvc-asan
cmake --build build/windows-msvc-asan
ctest --test-dir build/windows-msvc-asan --output-on-failure
```

CUDA backend (`QCX_ENABLE_CUDA`; local-only — CI never builds it):

```powershell
cmake --preset windows-msvc-cuda
cmake --build build/windows-msvc-cuda
ctest --test-dir build/windows-msvc-cuda --output-on-failure
```

## Running a calculation

The build produces one command-line program, `qcx`:

```powershell
build\windows-msvc\driver\Release\qcx.exe run water.toml
```

A run reads one TOML file and writes to three places:

| Where | What |
|---|---|
| stdout | the result as a single JSON document |
| `<input>.out` | a human-readable report, written beside the input file |
| stderr | diagnostics, one `qcx: ` line per problem found |

So `qcx run water.toml` leaves `water.out` next to `water.toml`, and redirecting stdout still
yields the machine-readable document:

```powershell
build\windows-msvc\driver\Release\qcx.exe run water.toml > water.json
```

| Exit code | Meaning |
|---|---|
| 0 | success |
| 1 | the run failed, or the input was rejected — the problems are on stderr |
| 2 | the command line was malformed |

`qcx --help` lists the commands; `qcx run --help` prints the input schema.

## Input and output

The full input schema and the shape of the result live in two companion pages:

- **Input file reference** — `docs/input-guide.md`: every section and key, with a worked example.
- **Result reference** — `docs/output-guide.md`: the report file and the JSON document.

## Documentation

API documentation is generated with Doxygen (front page = this README):

```powershell
cmake --build build/windows-msvc --target qcx-docs
start build\doxygen\html\index.html
```

CI builds the docs on every push and uploads the HTML as the `qcx-api-docs` artifact
(GitHub Actions → run → Artifacts).

## Working in this repo

See `CONTRIBUTING.md` for the house rules (naming, formatting, error handling, tests)
and the versioning policy, and [`CITATION.bib`](CITATION.bib) for the published work the
algorithms, kernels and vendored data derive from.
