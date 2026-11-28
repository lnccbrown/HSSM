#!/usr/bin/env python3
"""Reject machine-local paths embedded in documentation notebooks."""

import argparse
import json
import pathlib
import re
import sys
from collections.abc import Iterable
from typing import Any

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
IDENTIFYING_PATHS = (
    re.compile(r"(?:file://)?/(?:Users|home)/[^/\\\s\"'<>]+"),
    re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s\"'<>]+", re.IGNORECASE),
)
OUTPUT_LOCAL_PATHS = (
    re.compile(r"file://[^\s\"'<>]+", re.IGNORECASE),
    re.compile(r"/(?:private/)?(?:var/folders|tmp)(?:/[^/\\\s\"'<>]+)?"),
    re.compile(r"/(?:root|workspace|workspaces)(?:/[^/\\\s\"'<>]+)?"),
    re.compile(r"/mnt/[A-Za-z](?:/[^/\\\s\"'<>]+)?"),
    re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/][^\r\n\"'<>]+"),
    re.compile(
        r"(?<!\\)\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._-]+"
        r"(?:\\[^\r\n\"'<>]*)?"
    ),
)


def _strings(value: Any) -> Iterable[str]:
    """Yield strings recursively from a notebook value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _matches(texts: Iterable[str], patterns: tuple[re.Pattern[str], ...]) -> set[str]:
    """Return unique path fragments matched across a stream of strings."""
    return {
        match.group(0)
        for text in texts
        for pattern in patterns
        for match in pattern.finditer(text)
    }


def _output_strings(output: dict[str, Any]) -> Iterable[str]:
    """Yield human-readable output text without scanning binary MIME payloads."""
    for key in ("text", "traceback", "ename", "evalue"):
        yield from _strings(output.get(key))

    for mime_type, value in output.get("data", {}).items():
        if mime_type.startswith("text/") or mime_type in {
            "application/json",
            "application/javascript",
        }:
            yield from _strings(value)


def find_leaks(docs: pathlib.Path = DOCS) -> dict[str, list[str]]:
    """Return unique machine-local paths keyed by notebook path."""
    leaks: dict[str, list[str]] = {}
    for notebook in sorted(docs.rglob("*.ipynb")):
        if ".ipynb_checkpoints" in notebook.parts:
            continue
        content = json.loads(notebook.read_text())
        matches = _matches(_strings(content), IDENTIFYING_PATHS)
        output_text = (
            text
            for cell in content.get("cells", [])
            for output in cell.get("outputs", [])
            for text in _output_strings(output)
        )
        matches.update(_matches(output_text, OUTPUT_LOCAL_PATHS))
        if matches:
            leaks[str(notebook.relative_to(docs))] = sorted(matches)
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
