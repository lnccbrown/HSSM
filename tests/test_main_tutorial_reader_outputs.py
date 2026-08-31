"""Reader-facing output guards for the guided and scenic HSSM tutorials."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = (
    REPO_ROOT / "docs/tutorials/main_tutorial.ipynb",
    REPO_ROOT / "docs/tutorials/main_tutorial_scenic_route.ipynb",
)
MIN_STATIC_FIGURES = {
    "main_tutorial.ipynb": 2,
    "main_tutorial_scenic_route.ipynb": 30,
}
STATIC_IMAGE_MIMES = {"image/png", "image/jpeg", "image/svg+xml"}
MAX_TEXT_OUTPUT_CHARS = 8_000
MAX_AGGREGATE_TEXT_OUTPUT_CHARS = 45_000
MAX_TEXT_OUTPUT_COUNT = 120
IMAGE_DATA_URI = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+")
PLOT_OBJECT_REPR = re.compile(
    r"<(?:arviz_plots|matplotlib)\.[^>\n]* at 0x[0-9a-fA-F]+>"
    r"|(?:array\(\[.*)?<Axes(?:Subplot)?[^>]*>",
    re.DOTALL,
)
PROGRESS_MARKERS = (
    "\x1b[",
    "\r",
    "NUTS: [",
    "Output()<pre",
    "Sampling 1 chain",
    "Sampling 2 chains",
    "it/s]",
)


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text())


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(item) for item in value)
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _text_outputs(notebook: dict):
    for cell_index, cell in enumerate(notebook["cells"]):
        for output_index, output in enumerate(cell.get("outputs", [])):
            parts = [_as_text(output.get("text"))]
            for mime, value in output.get("data", {}).items():
                if mime.startswith("text/"):
                    parts.append(_as_text(value))
                elif mime == "application/json":
                    parts.append(_as_text(value))
            text = IMAGE_DATA_URI.sub("<embedded image>", "".join(parts))
            yield cell_index, output_index, text


def _is_sampling_call(expression: ast.Expr) -> bool:
    call = expression.value
    if not isinstance(call, ast.Call):
        return False
    if isinstance(call.func, ast.Attribute):
        return call.func.attr in {"sample", "sample_posterior_predictive"}
    if (
        isinstance(call.func, ast.Name)
        and call.func.id == "quiet_call"
        and call.args
        and isinstance(call.args[0], ast.Attribute)
    ):
        return call.args[0].attr in {"sample", "sample_posterior_predictive"}
    return False


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda path: path.stem)
def test_tutorial_outputs_are_compact_and_reader_safe(path: Path) -> None:
    """Reject tree widgets, sampler streams, and oversized prose-like output."""
    notebook = _load_notebook(path)
    errors = [
        cell_index
        for cell_index, cell in enumerate(notebook["cells"])
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == [], f"error outputs in cells {errors}"

    outputs = list(_text_outputs(notebook))

    tree_outputs = [cell for cell, _, text in outputs if "<xarray.DataTree" in text]
    assert tree_outputs == [], f"DataTree representations in cells {tree_outputs}"

    plot_object_reprs = []
    for cell_index, cell in enumerate(notebook["cells"]):
        for output_index, output in enumerate(cell.get("outputs", [])):
            data = output.get("data", {})
            if any(mime in STATIC_IMAGE_MIMES for mime in data):
                continue
            text = _as_text(output.get("text")) + "".join(
                _as_text(value)
                for mime, value in data.items()
                if mime.startswith("text/")
            )
            if PLOT_OBJECT_REPR.search(text):
                plot_object_reprs.append((cell_index, output_index, text[:120]))
    assert plot_object_reprs == [], (
        f"plot-object representations retained: {plot_object_reprs}"
    )

    progress_outputs = [
        (cell, marker)
        for cell, _, text in outputs
        for marker in PROGRESS_MARKERS
        if marker in text
    ]
    assert progress_outputs == [], f"sampler progress retained: {progress_outputs}"

    oversized = [
        (cell, output, len(text))
        for cell, output, text in outputs
        if len(text) > MAX_TEXT_OUTPUT_CHARS
    ]
    assert oversized == [], f"oversized text outputs: {oversized}"
    assert len(outputs) <= MAX_TEXT_OUTPUT_COUNT, (
        f"too many text outputs: {len(outputs)} > {MAX_TEXT_OUTPUT_COUNT}"
    )
    aggregate_chars = sum(len(text) for _, _, text in outputs)
    assert aggregate_chars <= MAX_AGGREGATE_TEXT_OUTPUT_CHARS, (
        "aggregate text output is too large: "
        f"{aggregate_chars} > {MAX_AGGREGATE_TEXT_OUTPUT_CHARS}"
    )


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda path: path.stem)
def test_committed_tutorial_artifacts_are_full_runs(path: Path) -> None:
    """Published outputs must never be replaced by a one-chain smoke artifact."""
    all_text = "\n".join(text for _, _, text in _text_outputs(_load_notebook(path)))
    assert "full (published outputs)" in all_text
    assert "<!-- hssm-full-run-artifact: true -->" in all_text
    assert "<!-- hssm-full-run-artifact: false -->" not in all_text


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda path: path.stem)
def test_sampling_results_are_assigned_before_display(path: Path) -> None:
    """Do not render a full DataTree by leaving a sampler call as an expression."""
    notebook = _load_notebook(path)
    bare_calls: list[int] = []
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        tree = ast.parse(_as_text(cell.get("source")))
        if any(
            _is_sampling_call(node) for node in tree.body if isinstance(node, ast.Expr)
        ):
            bare_calls.append(cell_index)

    assert bare_calls == [], f"bare sampling calls in cells {bare_calls}"


@pytest.mark.parametrize("path", TUTORIALS, ids=lambda path: path.stem)
def test_tutorials_retain_expected_static_figures(path: Path) -> None:
    """Keep the guided diagnostics and the scenic route's visual breadth."""
    notebook = _load_notebook(path)
    images = [
        _as_text(value)
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        for mime, value in output.get("data", {}).items()
        if mime in STATIC_IMAGE_MIMES
    ]
    image_digests = [hashlib.sha256(value.encode()).hexdigest() for value in images]
    duplicate_digests = sorted(
        digest for digest in set(image_digests) if image_digests.count(digest) > 1
    )

    assert len(images) >= MIN_STATIC_FIGURES[path.name]
    assert duplicate_digests == [], "duplicate static figure outputs retained"
