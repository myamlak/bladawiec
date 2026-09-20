#!/bin/bash
# Local reproduction of the CI clang-tidy job in .github/workflows/ci.yml: the
# find filter below is duplicated there and the two must stay byte-identical.
#
# Scope: qcx's OWN sources. Every tree pruned below carries .cpp files that are
# not ours to tidy, and walking them made the local gate disagree with the CI
# gate it mirrors (2026-09-18):
#   the agent scratch tree - the git worktrees and run trees inside it hold
#               whole COPIES of this repo's sources, so walking them re-reports
#               qcx's own files N times under a scratch path - and points the
#               fix at a copy that nobody ships. They were 4185 of the 4524
#               walked files while untracked/gitignored, so CI never saw them
#               at all: the bug was local-only and it made the local number
#               meaningless.
#   external/ - the boys and excgrid SUBMODULES (older checkouts may also carry
#               the boys/excgrid remnants as plain dirs). A finding in a
#               submodule is fixed upstream in that repo, never here.
#   third_party/, build*/ - vendored deps and generated build trees.
#
# The find filter prunes the vendored TREES; our own TUs still include their
# headers, so the same two directories are excluded by name at the invocation
# (-exclude-header-filter) - a finding inside a vendored header is fixed
# upstream, never here.
cd "$(git rev-parse --show-toplevel)" || exit 1
mapfile -t files < <(find . \
    -path './build*' -prune -o \
    -path './third_party*' -prune -o \
    -path './.claude' -prune -o \
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
    # SINGLE dash on the header filter: run-clang-tidy.py registers every one
    # of its options with one leading dash, and argparse does not alias the
    # `--` spelling onto them - it exits 2 with "unrecognized arguments"
    # before running a single translation unit. The option is passed through
    # to clang-tidy verbatim, which is where it does the excluding.
    run-clang-tidy-21 -p build/wsl-clang -j 0 -quiet \
        -exclude-header-filter='(third_party|external)/.*' "${files[@]}"
else
    # An empty list must NOT fall through to a bare `-p <db>` run: that scans
    # every compile_commands.json entry, submodules included.
    echo "clang-tidy scope: no translation units found - nothing to check" >&2
fi
