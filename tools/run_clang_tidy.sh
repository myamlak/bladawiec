#!/bin/bash
# Local reproduction of the CI clang-tidy job (same find filter as the weekly
# job in .github/workflows/ci-windows-release.yml - tidy moved out of ci.yml
# into the weekly workflow, so ci.yml no longer carries a find filter).
#
# Scope: qcx's OWN sources. Every tree below carries .cpp files that are not
# ours to tidy, and walking them made the local gate disagree with the CI gate
# it mirrors:
#   scratch trees - directories holding whole COPIES of this repo's sources,
#               so walking them re-reports qcx's own files N times under a
#               scratch path - and points the fix at a copy that nobody ships.
#               They were 4185 of the 4524 walked files while
#               untracked/gitignored, so CI never saw them at all: the bug was
#               local-only and it made the local number meaningless.
#   external/ - the boys and excgrid SUBMODULES (older checkouts may also carry
#               the boys/excgrid remnants as plain dirs). A finding in a
#               submodule is fixed upstream in that repo, never here.
#   third_party/, build*/ - vendored deps and generated build trees.
cd "$(git rev-parse --show-toplevel)" || exit 1
mapfile -t files < <(find . \
    -path './build*' -prune -o \
    -path './third_party*' -prune -o \
    -path './external' -prune -o \
    -path './backend/tests/cuda_smoke_test.cpp' -prune -o \
    -path './memory/tests/device_buffer_cuda_test.cpp' -prune -o \
    -path './memory/tests/tensor_cuda_test.cpp' -prune -o \
    -path './integrals/tests/boys_cuda_test.cpp' -prune -o \
    -name '*.cpp' -print)
# run-clang-tidy.py (clang-tidy-21 has no -j itself): -j 0 parallelizes one
# process per core - the third-party-heavy TUs (amgcl) parse for
# ~30-50 s each and would otherwise serialize the whole run.
if [ ${#files[@]} -gt 0 ]; then
    echo "clang-tidy scope: ${#files[@]} translation units" >&2
    run-clang-tidy-21 -p build/wsl-clang -j 0 -quiet "${files[@]}"
else
    # An empty list must NOT fall through to a bare `-p <db>` run: that scans
    # every compile_commands.json entry, submodules included.
    echo "clang-tidy scope: no translation units found - nothing to check" >&2
fi
