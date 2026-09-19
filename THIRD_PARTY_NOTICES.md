# Third-party notices

Repository license: BSD-3-Clause (`LICENSE`, `Copyright (c) 2026 Marcin
Makowski`). This file lists the license and the attribution obligations for
every third-party component that qcx compiles, links, or ships as vendored
data, so that the notices can travel with any redistribution of the source
or of a binary.

| Component | Version | Purpose | License | Full text |
|---|---|---|---|---|
| Eigen | 5.0.1 (pinned submodule) | dense linear algebra (memory, linalg, molecule, symmetry, grid) | MPL-2.0, with permissive BSD/Apache/MinPack parts | `third_party/eigen/LICENSE`, `third_party/eigen/COPYING.*` |
| spdlog | 1.17.0 (pinned submodule) | logging backend, built from source and linked into the binaries | MIT | `third_party/spdlog/LICENSE` |
| GoogleTest | 1.17.0 (pinned submodule) | unit-test framework (tests only) | BSD-3-Clause | `third_party/googletest/LICENSE` |
| stdexec | pinned submodule | P2300 `std::execution` reference implementation (backend scheduler) | Apache-2.0 with LLVM exception | `third_party/stdexec/LICENSE.txt` |
| nlohmann/json | 3.12.0 (pinned submodule) | JSON result output (io) | MIT | `third_party/json/LICENSE.MIT` |
| toml++ | 3.4.0 (pinned submodule) | TOML run-input schema (io) | MIT | `third_party/tomlplusplus/LICENSE` |
| Spectra | 1.2.0 (pinned submodule) | iterative eigensolvers (linalg) | MPL-2.0 | `third_party/spectra/LICENSE` |
| amgcl | 1.5.0 (pinned submodule) | algebraic-multigrid sparse solver (linalg), built in its Boost-free mode | MIT | `third_party/amgcl/LICENSE.md` |
| Google Benchmark | 1.9.5 (pinned submodule, optional build flag) | scaling benchmarks | Apache-2.0 | `third_party/benchmark/LICENSE` |
| HighFive | 2.10.1 (pinned submodule) | HDF5 C++ wrapper (storage) | BSL-1.0 | `third_party/highfive/LICENSE` |
| HDF5 | system or vcpkg package | checkpoint/restart file format (storage) | BSD-3-Clause (HDF Group / NCSA) | ships with the HDF5 package |
| Boost.Graph | system or vcpkg package | molecular connectivity graphs (molecule) | BSL-1.0 | ships with the Boost package |
| boys | v1.1.4 (pinned submodule; first-party) | Boys-function kernel, compiled into the integrals library | BSD-3-Clause | `external/boys/LICENSE` |
| excgrid | pinned submodule (first-party) | molecular block grids and exchange-correlation kernels (grid) | BSD-3-Clause | `external/excgrid/LICENSE` |

`boys` and `excgrid` are this project's own libraries, distributed under the
same BSD-3-Clause terms as qcx itself, and pinned as submodules so that the
exact kernel revision a build used is recorded. Every other row above is
third-party work.

The pinned submodules carry their own `LICENSE` files; their full texts are
not duplicated here. One file inside a submodule is under a license of a
different class and is called out below.

## Copyleft components: Eigen and Spectra (MPL-2.0)

Eigen and Spectra are the only weak-copyleft components in the tree. The
Mozilla Public License 2.0 is a *file-level* copyleft: it permits combining
MPL-covered files with files under other terms into a larger work, and
distributing that larger work under the other terms, provided the
MPL-covered files themselves stay under MPL-2.0 and their source — including
any modification of them — remains available.

qcx meets that obligation by not modifying them: both are consumed exactly
as pinned, Eigen through its include path and Spectra through its headers.
Their source is therefore the submodule tree itself, which is distributed
with the repository. Neither one imposes any condition on the licensing of
qcx's own sources, and neither conflicts with BSD-3-Clause.

Eigen additionally ships `COPYING.BSD` (Intel), `COPYING.APACHE` and
`COPYING.MINPACK` (University of Chicago) for the parts derived from those
projects, as `third_party/eigen/COPYING.README` describes. Those parts are
permissive and are compatible with BSD-3-Clause without further obligation.

## A GPL-3.0 file — not in this repository

**Nothing in this repository is GPL-licensed and no GPL code is distributed
with it.** `third_party/json` is a submodule: this project records a URL and a
commit, never the submodule's contents, so a plain `git clone` of this
repository leaves `third_party/json/` empty.

The file in question lives upstream, inside that submodule:
`third_party/json/tests/thirdparty/imapdl/` is **GPL-3.0-only** (declared in
nlohmann/json's own `.reuse/dep5`, full text in its
`LICENSES/GPL-3.0-only.txt`). It is a Python script invoked only by
nlohmann/json's own CI. It is noted here because it is the only strong-copyleft
code anywhere in the dependency tree.

It does not affect this project: only `third_party/json/include` is on any
include path, nlohmann/json's own test suite is never configured or built here,
and none of that code is compiled, linked or redistributed with a qcx binary.

The obligation it does place is on anyone who redistributes the JSON submodule
*directory* as a whole — a source archive that includes the submodule tree
carries a GPL-3.0-only file, which must stay under GPL-3.0 with its source
available. A binary release is unaffected. `git clone --recurse-submodules`
fetches that file from upstream under its own licence, not from this project.
Anyone trimming a source distribution can drop that directory and lose nothing
this project uses.

## Optional vendor BLAS

The linalg module can link one vendor BLAS: Intel oneMKL, AMD AOCL-BLAS, or
OpenBLAS, auto-detected in that order, or none — in which case the dense
path stays on Eigen. No vendor BLAS is vendored or distributed here; the
build finds whichever one is installed and links it.

- **OpenBLAS** — BSD-3-Clause (The OpenBLAS Project). Compatible with
  BSD-3-Clause.
- **Intel oneMKL** — proprietary, licensed under the Intel End User License
  Agreement for Developer Tools. A binary built against it carries Intel's
  own redistribution terms for the MKL runtime libraries, which are separate
  from this repository's license.
- **AMD AOCL-BLAS** — licensed by AMD; the license text ships with the AOCL
  package. Like MKL, a binary built against it carries the vendor's terms
  for the runtime libraries.

## Optional CUDA backend

The CUDA backend is a build option and is off by default. It requires the
NVIDIA CUDA toolkit, which is proprietary and distributed under NVIDIA's
own license terms; the toolkit is never vendored here and no CUDA artifact
is part of a default build.

## Vendored data

Basis sets under `data/basis/` are downloaded from the Basis Set Exchange
(https://www.basissetexchange.org) by `tools/fetch_basis_data.py` and
committed as data, one file per element per family, with per-file SHA-256
sums in `data/basis/manifest.json`. The Basis Set Exchange distributes this
data under the BSD-3-Clause license; the individual families carry their own
original references (Pople, Dunning, Weigend/Ahlrichs, Jensen and others),
which the BSE database records per family. The provenance of the corpus, the
families it covers and the pinned fetch version are documented in
`data/basis/VERSIONS.md`.

No other third-party data ships in the repository.

## Build-time-only tooling

The following are used to build, test, document or regenerate this project
and are not redistributed with it. Their licenses apply only if the tooling
itself is shipped:

- **CMake, Ninja, Visual Studio / MSVC, GCC, Clang, clang-format,
  clang-tidy, Doxygen, vcpkg** — development tools, licensed by their
  respective publishers.
- **Python packages** pinned in `tools/requirements-*.txt` — the basis-set
  pipeline (`basis-set-exchange`), the periodic-table generation
  (`periodictable`, `mendeleev`), and the verification oracles (`pyscf`,
  `numpy`, `scipy`). Installed from PyPI into the build or verification
  environment and never redistributed; each package's own license applies.
