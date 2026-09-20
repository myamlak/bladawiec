#!/usr/bin/env python3
"""Refuse a commit whose index cannot describe it.

An index kept outside the repository (`GIT_INDEX_FILE`) is a supported setup,
and this is the precondition that setup carries, because the failure it refuses
is silent:

  GIT_INDEX_FILE=<path that does not exist>
    git diff --cached --name-only --diff-filter=ACMR   ->  0 entries (exit 0)
    git diff --cached --name-only                      ->  the whole tree, all "D"

Against an empty index every tracked file is a DELETION, and the ACMR filter the
hook checks use drops deletions - so clang-format and the style guards inspect
nothing, print nothing and exit 0. Measured in a scratch repository: a bare
`git commit` in that state LANDS a tree with zero files (3 files changed, 3
deletions), i.e. one commit deleting every tracked file, with every guard
reporting OK. A guard that silently passes is worse than one that fails, so the
hook refuses here instead.

SCOPE (measured): the silent path needs a BARE commit. A path-limited commit
(`git commit <path>`) makes git export its own absolute temp index built from
HEAD plus the pathspec, so an outer index never reaches the hook, and a missing
private index fails loudly ("pathspec ... did not match any file(s) known to
git") before the hook runs. This check therefore passes on the path-limited path
by construction, and it is the bare-commit path it protects.

Escape hatch, stated rather than hidden: `git commit --no-verify` still skips
every hook check, this one included. The point is that the refusal is what a
mistake meets by default.
"""

import subprocess
import sys

REFUSAL = """pre-commit: REFUSED - the index this commit would use is EMPTY.

  resolved index : {resolved}
  GIT_INDEX_FILE : {env}
  index entries  : {index_count}
  files in HEAD  : {head_count}

An empty index cannot describe a commit: committing through it deletes every
tracked file. This is a measured failure, not a hypothetical one.

If GIT_INDEX_FILE names an index you keep beside the repository, check that the
path exists and carries the tree you expect:

  git read-tree HEAD                 (with GIT_INDEX_FILE exported)

Then re-run the commit. If the empty index is deliberate, `--no-verify`."""


def run(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def count_lines(args: list) -> int:
    done = run(args)
    if done.returncode != 0:
        return -1
    return sum(1 for line in done.stdout.splitlines() if line.strip())


def main() -> int:
    import os

    resolved = run(["git", "rev-parse", "--git-path", "index"])
    resolved_path = resolved.stdout.strip() if resolved.returncode == 0 else "<unknown>"

    # An unborn HEAD is the only case where an empty index is legitimate: there
    # is no tree yet to delete.
    head = run(["git", "rev-parse", "--verify", "HEAD"])
    if head.returncode != 0:
        return 0

    head_count = count_lines(["git", "ls-tree", "-r", "--name-only", "HEAD"])
    if head_count <= 0:
        return 0

    index_count = count_lines(["git", "ls-files"])
    if index_count < 0 or index_count > 0:
        return 0

    print(
        REFUSAL.format(
            resolved=resolved_path,
            env=os.environ.get("GIT_INDEX_FILE", "<unset>") or "<empty>",
            index_count=index_count,
            head_count=head_count,
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
