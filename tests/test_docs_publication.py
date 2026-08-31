"""Tests for static publication of marimo documentation notebooks."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from scripts.check_docs_publication import find_publication_issues, strict_source_issues


def _write_repo(
    root: pathlib.Path,
    *,
    nav_target: str,
    outputs: list[dict] | None = None,
    require_full_run: bool = False,
) -> None:
    """Create a tiny docs tree with one marimo source and optional artifact."""
    tutorials = root / "docs" / "tutorials"
    tutorials.mkdir(parents=True)
    (root / "mkdocs.yml").write_text(
        "site_name: test\nnav:\n  - Tutorial: "
        + nav_target
        + "\nplugins:\n  - search\n"
    )
    source = "import marimo\napp = marimo.App()\n"
    if require_full_run:
        source += "# docs-require-full-run: true\n"
    (tutorials / "demo.py").write_text(source)
    if nav_target.endswith(".ipynb"):
        (tutorials / "demo.ipynb").write_text(
            json.dumps(
                {
                    "cells": [
                        {
                            "cell_type": "code",
                            "execution_count": None,
                            "metadata": {},
                            "outputs": outputs or [],
                            "source": ["result"],
                        }
                    ],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            )
        )


def test_rejects_marimo_source_in_navigation(tmp_path):
    """A raw marimo Python page cannot represent its executed outputs."""
    _write_repo(tmp_path, nav_target="tutorials/demo.py")

    assert find_publication_issues(tmp_path) == [
        "tutorials/demo.py: nav points at a marimo source; publish its "
        "output-bearing .ipynb export instead"
    ]


def test_accepts_output_bearing_marimo_export_without_execution_count(tmp_path):
    """Marimo outputs are valid even when Jupyter counters are null."""
    _write_repo(
        tmp_path,
        nav_target="tutorials/demo.ipynb",
        outputs=[
            {
                "data": {"text/plain": ["compact result"]},
                "metadata": {},
                "output_type": "display_data",
            }
        ],
    )

    assert find_publication_issues(tmp_path) == []


def test_rejects_empty_error_and_marimo_ui_outputs(tmp_path):
    """Retained failures and custom elements are broken in static MkDocs pages."""
    _write_repo(
        tmp_path,
        nav_target="tutorials/demo.ipynb",
        outputs=[
            {
                "data": {
                    "application/vnd.marimo+json": (
                        "&lt;marimo-dropdown&gt;choose&lt;/marimo-dropdown&gt;"
                    ),
                    "text/html": "<marimo-slider>choose</marimo-slider>",
                },
                "metadata": {},
                "output_type": "display_data",
            },
            {
                "ename": "RuntimeError",
                "evalue": "failed",
                "output_type": "error",
                "traceback": ["traceback"],
            },
        ],
    )

    assert find_publication_issues(tmp_path) == [
        "tutorials/demo.ipynb: cell 1 retains marimo UI markup: "
        "<marimo-dropdown, <marimo-slider",
        "tutorials/demo.ipynb: cell 1 retains RuntimeError: failed",
    ]


def test_rejects_generated_marimo_artifact_without_outputs(tmp_path):
    """A generated artifact must show readers at least one computed result."""
    _write_repo(tmp_path, nav_target="tutorials/demo.ipynb")

    assert find_publication_issues(tmp_path) == [
        "tutorials/demo.ipynb: generated marimo artifact has no meaningful output"
    ]


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_stream_only_artifact_is_not_meaningful_output(tmp_path, stream_name):
    """Logs alone do not establish that a tutorial produced a computed result."""
    _write_repo(
        tmp_path,
        nav_target="tutorials/demo.ipynb",
        outputs=[
            {
                "name": stream_name,
                "output_type": "stream",
                "text": ["warning: backend unavailable\n"],
            }
        ],
    )

    assert find_publication_issues(tmp_path) == [
        "tutorials/demo.ipynb: generated marimo artifact has no meaningful output"
    ]


def test_sampling_heavy_artifact_requires_provenance_markers(tmp_path):
    """A quick or nondeterministic export cannot masquerade as a full artifact."""
    _write_repo(
        tmp_path,
        nav_target="tutorials/demo.ipynb",
        require_full_run=True,
        outputs=[
            {
                "data": {"text/markdown": "quick result"},
                "metadata": {},
                "output_type": "display_data",
            }
        ],
    )

    assert find_publication_issues(tmp_path) == [
        "tutorials/demo.ipynb: sampling-heavy artifact lacks FULL_RUN "
        "publication marker",
        "tutorials/demo.ipynb: sampling-heavy artifact lacks deterministic-init marker",
    ]

    notebook = tmp_path / "docs" / "tutorials" / "demo.ipynb"
    content = json.loads(notebook.read_text())
    content["cells"][0]["outputs"][0]["data"]["text/markdown"] += (
        "\n<!-- hssm-full-run-artifact: true -->"
    )
    notebook.write_text(json.dumps(content))
    assert find_publication_issues(tmp_path) == [
        "tutorials/demo.ipynb: sampling-heavy artifact lacks deterministic-init marker"
    ]

    content["cells"][0]["outputs"][0]["data"]["text/markdown"] += (
        "\n<!-- hssm-deterministic-init: true -->"
    )
    notebook.write_text(json.dumps(content))
    assert find_publication_issues(tmp_path) == []


def test_rejects_duplicate_image_outputs_in_one_cell(tmp_path):
    """Do not render a figure twice through displayhook and inline auto-flush."""
    repeated = {
        "data": {"image/png": "same-image", "text/plain": "<Figure>"},
        "metadata": {},
        "output_type": "display_data",
    }
    _write_repo(
        tmp_path,
        nav_target="tutorials/demo.ipynb",
        outputs=[repeated, repeated],
    )

    assert find_publication_issues(tmp_path) == [
        "tutorials/demo.ipynb: cell 1 repeats an identical image output"
    ]


def test_repository_marimo_publications_are_valid():
    """Keep the repository's checked-in publication artifacts healthy."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    assert find_publication_issues(repo_root) == []


@pytest.mark.skipif(
    importlib.util.find_spec("marimo") is None,
    reason="marimo is installed by the docs/notebook groups, not the core test group",
)
def test_repository_marimo_sources_pass_strict_check():
    """Catch broken reactive DAGs even while an older artifact remains healthy."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent

    assert strict_source_issues(repo_root) == []
