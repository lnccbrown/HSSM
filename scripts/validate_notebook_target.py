#!/usr/bin/env python3
"""Validate a repository-relative notebook selected for hosted execution."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class NotebookTargetError(ValueError):
    """Raised when a requested notebook is not a safe tracked docs target."""


def _validate_target_shape(target: str) -> pathlib.PurePosixPath:
    """Return a canonical docs path or raise ``NotebookTargetError``."""
    if not target:
        raise NotebookTargetError("the notebook path cannot be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in target):
        raise NotebookTargetError("control characters are not allowed")
    if target != target.strip():
        raise NotebookTargetError("leading or trailing whitespace is not allowed")
    if "\\" in target:
        raise NotebookTargetError("use repository-relative POSIX separators")

    path = pathlib.PurePosixPath(target)
    if path.is_absolute():
        raise NotebookTargetError("absolute paths are not allowed")
    if str(path) != target:
        raise NotebookTargetError("the path must be in canonical POSIX form")
    if len(path.parts) < 2 or path.parts[0] != "docs":
        raise NotebookTargetError("the path must be below docs/")
    if any(part in {".", ".."} for part in path.parts):
        raise NotebookTargetError("path traversal is not allowed")
    if path.parts[:2] == ("docs", "archive"):
        raise NotebookTargetError("archived notebooks are frozen and cannot be run")
    if ".ipynb_checkpoints" in path.parts:
        raise NotebookTargetError("notebook checkpoints cannot be run")
    if path.suffix != ".ipynb":
        raise NotebookTargetError("the target must be an .ipynb notebook")
    return path


def validate_notebook_target(
    target: str, repo_root: pathlib.Path = REPO_ROOT
) -> pathlib.Path:
    """Validate and return one tracked, non-archive documentation notebook."""
    path = _validate_target_shape(target)
    root = repo_root.resolve()
    docs_root = (root / "docs").resolve()
    candidate = (root / pathlib.Path(*path.parts)).resolve()

    if not candidate.is_relative_to(docs_root):
        raise NotebookTargetError("the resolved path escapes docs/")
    if not candidate.is_file():
        raise NotebookTargetError("the requested notebook does not exist")

    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", target],
        capture_output=True,
        check=False,
        text=True,
    )
    if tracked.returncode != 0:
        raise NotebookTargetError("the requested notebook is not tracked by git")
    return candidate


def main(argv: list[str] | None = None) -> int:
    """Validate one command-line target and report a concise failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="tracked repository-relative docs notebook")
    args = parser.parse_args(argv)

    try:
        validate_notebook_target(args.notebook)
    except NotebookTargetError as error:
        print(f"Invalid notebook target {args.notebook!r}: {error}", file=sys.stderr)
        return 2

    print(args.notebook)
    return 0


if __name__ == "__main__":
    sys.exit(main())
