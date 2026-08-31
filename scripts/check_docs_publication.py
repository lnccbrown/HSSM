#!/usr/bin/env python3
"""Validate static publication artifacts generated from marimo notebooks.

Marimo ``.py`` files are the editable source of truth, but mkdocs-jupyter must
publish their output-bearing ``.ipynb`` exports.  This guard deliberately does
not require Jupyter execution counters: marimo exports can contain complete
outputs while leaving ``execution_count`` unset.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import pathlib
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MKDOCS = REPO_ROOT / "mkdocs.yml"
DOCS = REPO_ROOT / "docs"

_MARIMO_IMPORT = re.compile(r"^\s*(?:import\s+marimo\b|from\s+marimo\b)", re.MULTILINE)
_MARIMO_ELEMENT = re.compile(r"<marimo-[a-z0-9-]+\b", re.IGNORECASE)
_FULL_RUN_SOURCE = re.compile(
    r"^#\s*docs-require-full-run:\s*true\s*$", re.IGNORECASE | re.MULTILINE
)
_FULL_RUN_MARKER = "<!-- hssm-full-run-artifact: true -->"
_DETERMINISTIC_INIT_MARKER = "<!-- hssm-deterministic-init: true -->"
_DATA_IMAGE = re.compile(
    r"data:image/[a-z0-9.+-]+;base64,([a-z0-9+/=\r\n]+)", re.IGNORECASE
)
_TEXT_MIME_TYPES = {
    "application/javascript",
    "application/json",
    "text/html",
    "text/markdown",
    "text/plain",
}
_IMAGE_MIME_PREFIX = "image/"


def navigation_paths(mkdocs: pathlib.Path = MKDOCS) -> list[str]:
    """Return file targets from the top-level MkDocs ``nav`` block."""
    paths: list[str] = []
    in_nav = False
    for raw_line in mkdocs.read_text().splitlines():
        if raw_line == "nav:":
            in_nav = True
            continue
        if not in_nav:
            continue
        if raw_line and not raw_line[0].isspace() and not raw_line.startswith("#"):
            break

        item = raw_line.strip()
        if not item.startswith("- "):
            continue
        value = item[2:].strip()
        if ": " in value:
            value = value.rsplit(": ", maxsplit=1)[1]
        value = value.strip("'\"")
        if pathlib.PurePosixPath(value).suffix in {".ipynb", ".md", ".py"}:
            paths.append(value)
    return paths


def _is_marimo_source(path: pathlib.Path) -> bool:
    """Return whether a Python source imports marimo."""
    return path.is_file() and bool(_MARIMO_IMPORT.search(path.read_text()))


def paired_marimo_sources(repo_root: pathlib.Path = REPO_ROOT) -> list[pathlib.Path]:
    """Return marimo sources paired with nav-published ipynb artifacts."""
    docs = repo_root / "docs"
    return [
        source
        for target in navigation_paths(repo_root / "mkdocs.yml")
        if pathlib.PurePosixPath(target).suffix == ".ipynb"
        and _is_marimo_source(source := (docs / target).with_suffix(".py"))
    ]


def strict_source_issues(repo_root: pathlib.Path = REPO_ROOT) -> list[str]:
    """Run marimo's strict structural check on every published source pair."""
    sources = paired_marimo_sources(repo_root)
    if not sources:
        return []
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "check", "--strict", *sources],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return []
    detail = (result.stdout + result.stderr).strip() or "unknown marimo check failure"
    return [f"marimo check --strict failed:\n{detail}"]


def _strings(value: Any) -> Iterable[str]:
    """Yield strings recursively from a JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def _output_text(output: dict[str, Any]) -> str:
    """Return human-readable text from one notebook output."""
    values: list[Any] = [
        output.get("text"),
        output.get("traceback"),
        output.get("ename"),
        output.get("evalue"),
    ]
    data = output.get("data", {})
    if isinstance(data, dict):
        values.extend(value for mime, value in data.items() if mime in _TEXT_MIME_TYPES)
    return "\n".join(_strings(values))


def _is_meaningful_output(output: dict[str, Any]) -> bool:
    """Return whether an output contains non-empty reader-facing content."""
    output_type = output.get("output_type")
    if output_type == "error":
        return False
    if output_type not in {"display_data", "execute_result"}:
        return False
    if _output_text(output).strip():
        return True
    data = output.get("data", {})
    return isinstance(data, dict) and any(
        mime.startswith(_IMAGE_MIME_PREFIX) and any(_strings(value))
        for mime, value in data.items()
    )


def _image_payloads(output: dict[str, Any]) -> Iterable[str]:
    """Yield embedded image payloads from MIME data and HTML data URLs."""
    data = output.get("data", {})
    if not isinstance(data, dict):
        return
    for mime, value in data.items():
        if mime.startswith(_IMAGE_MIME_PREFIX):
            payload = "".join(_strings(value))
            if payload:
                yield payload
        elif mime == "text/html":
            text = html.unescape("".join(_strings(value)))
            yield from _DATA_IMAGE.findall(text)


def _artifact_issues(
    path: pathlib.Path,
    display_path: str,
    *,
    require_full_run: bool,
) -> list[str]:
    """Return publication issues for one generated notebook artifact."""
    try:
        notebook = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return [f"{display_path}: cannot read notebook: {error}"]
    if not isinstance(notebook, dict):
        return [f"{display_path}: notebook root is not a JSON object"]

    outputs = [
        output
        for cell in notebook.get("cells", [])
        if isinstance(cell, dict)
        for output in cell.get("outputs", [])
        if isinstance(output, dict)
    ]
    issues: list[str] = []
    if not any(_is_meaningful_output(output) for output in outputs):
        issues.append(
            f"{display_path}: generated marimo artifact has no meaningful output"
        )
    artifact_text = "\n".join(_output_text(output) for output in outputs)
    if require_full_run and _FULL_RUN_MARKER not in artifact_text:
        issues.append(
            f"{display_path}: sampling-heavy artifact lacks FULL_RUN publication marker"
        )
    if require_full_run and _DETERMINISTIC_INIT_MARKER not in artifact_text:
        issues.append(
            f"{display_path}: sampling-heavy artifact lacks deterministic-init marker"
        )

    for cell_number, cell in enumerate(notebook.get("cells", []), start=1):
        if not isinstance(cell, dict):
            continue
        image_hashes = [
            hashlib.sha256(payload.encode()).hexdigest()
            for output in cell.get("outputs", [])
            if isinstance(output, dict)
            for payload in _image_payloads(output)
        ]
        if len(image_hashes) != len(set(image_hashes)):
            issues.append(
                f"{display_path}: cell {cell_number} repeats an identical image output"
            )
        for output in cell.get("outputs", []):
            if not isinstance(output, dict):
                continue
            if output.get("output_type") == "error":
                name = output.get("ename", "error")
                value = output.get("evalue", "")
                issues.append(
                    (
                        f"{display_path}: cell {cell_number} retains {name}: {value}"
                    ).rstrip()
                )
            all_output_text = "\n".join(_strings(output))
            elements = sorted(
                set(_MARIMO_ELEMENT.findall(html.unescape(all_output_text)))
            )
            if elements:
                issues.append(
                    f"{display_path}: cell {cell_number} retains marimo UI markup: "
                    + ", ".join(elements)
                )
    return issues


def find_publication_issues(
    repo_root: pathlib.Path = REPO_ROOT,
) -> list[str]:
    """Return issues in nav-published marimo pages and their static artifacts."""
    docs = repo_root / "docs"
    issues: list[str] = []
    for target in navigation_paths(repo_root / "mkdocs.yml"):
        page = docs / target
        if page.suffix == ".py" and _is_marimo_source(page):
            issues.append(
                f"{target}: nav points at a marimo source; publish its output-bearing "
                ".ipynb export instead"
            )
            continue
        if page.suffix != ".ipynb":
            continue

        source = page.with_suffix(".py")
        if not _is_marimo_source(source):
            continue
        if not page.is_file():
            issues.append(f"{target}: generated marimo artifact is missing")
            continue
        issues.extend(
            _artifact_issues(
                page,
                target,
                require_full_run=bool(_FULL_RUN_SOURCE.search(source.read_text())),
            )
        )
    return issues


def main() -> int:
    """Report static marimo publication issues; return 1 when any are present."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args()
    issues = [*find_publication_issues(), *strict_source_issues()]
    if args.json:
        print(json.dumps({"issues": issues}, indent=2))
    elif issues:
        for issue in issues:
            print(f"PUBLICATION ERROR: {issue}")
    else:
        print("ok: nav-published marimo tutorials use clean output-bearing artifacts")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
