#!/usr/bin/env python3
"""Reject machine-local paths embedded in documentation notebooks."""

import argparse
import json
import pathlib
import re
import sys

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
MACHINE_PATH = re.compile(
    r'(?:/Users|/home|/var/folders)/[^"\\\s]+'
    r'|[A-Za-z]:\\\\Users\\\\[^"\s]+'
)


def find_leaks(docs: pathlib.Path = DOCS) -> dict[str, list[str]]:
    """Return unique machine-local paths keyed by notebook path."""
    leaks: dict[str, list[str]] = {}
    for notebook in sorted(docs.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in notebook.parts:
            continue
        matches = sorted(set(MACHINE_PATH.findall(notebook.read_text())))
        if matches:
            leaks[str(notebook.relative_to(docs))] = matches
    return leaks


def main() -> int:
    """Report notebook path leaks; return 1 when any are present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args()
    leaks = find_leaks()

    if args.json:
        print(json.dumps({"leaks": leaks}, indent=2))
    elif leaks:
        for notebook, paths in leaks.items():
            for path in paths:
                print(f"LOCAL PATH: {notebook}: {path}")
        print("\nSanitize saved outputs before publishing the notebook.")
    else:
        checked = sum(1 for _ in DOCS.rglob("*.ipynb"))
        print(f"ok: no machine-local paths in {checked} notebooks")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
