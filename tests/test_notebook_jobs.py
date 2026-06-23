"""Tests for notebook job inventory and reporting behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_notebook_jobs_module() -> object:
    """Load the notebook jobs script as a module for testing."""
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "notebook_jobs.py"
    spec = importlib.util.spec_from_file_location("notebook_jobs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_notebook(path: Path) -> None:
    """Write a minimal notebook for inventory tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """{
 "cells": [],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
""",
        encoding="utf-8",
    )


def write_manifest(path: Path) -> None:
    """Write a minimal manifest for inventory tests."""
    path.write_text(
        """default_timeout_seconds = 10000
default_kernel_name = "python3"
ignored_globs = ["**/*.nbconvert.ipynb"]

[[notebooks]]
path = "docs/tutorials/enabled.ipynb"
enabled = true

[[notebooks]]
path = "docs/tutorials/disabled.ipynb"
enabled = false
notes = "Intentionally disabled"
""",
        encoding="utf-8",
    )


def test_discover_inventory_ignores_generated_notebooks(tmp_path, monkeypatch):
    """Generated nbconvert artifacts should not require manifest entries."""
    module = load_notebook_jobs_module()
    monkeypatch.chdir(tmp_path)
    write_notebook(tmp_path / "docs" / "tutorials" / "enabled.ipynb")
    write_notebook(tmp_path / "docs" / "tutorials" / "disabled.ipynb")
    write_notebook(tmp_path / "docs" / "tutorials" / "enabled.nbconvert.ipynb")
    manifest_path = tmp_path / "notebooks.toml"
    write_manifest(manifest_path)

    manifest = module.load_manifest(manifest_path)
    inventory = module.discover_inventory(manifest, include_disabled=True)

    assert [entry["path"] for entry in inventory] == [
        "docs/tutorials/enabled.ipynb",
        "docs/tutorials/disabled.ipynb",
    ]


def test_initialize_run_scopes_inventory_to_selected_paths(tmp_path, monkeypatch):
    """Targeted runs should not mark untargeted notebooks as missing results."""
    module = load_notebook_jobs_module()
    monkeypatch.chdir(tmp_path)
    write_notebook(tmp_path / "docs" / "tutorials" / "enabled.ipynb")
    write_notebook(tmp_path / "docs" / "tutorials" / "disabled.ipynb")
    manifest_path = tmp_path / "notebooks.toml"
    write_manifest(manifest_path)

    manifest = module.load_manifest(manifest_path)
    run_dir = tmp_path / ".cache" / "single-disabled"
    inventory = module.initialize_run(
        manifest,
        run_dir,
        selected_paths={"docs/tutorials/disabled.ipynb"},
    )
    report = module.aggregate_run(run_dir)

    assert [entry["path"] for entry in inventory] == ["docs/tutorials/disabled.ipynb"]
    assert report["summary"] == {
        "total": 1,
        "passed": 0,
        "failed": 0,
        "timed_out": 0,
        "skipped": 1,
        "missing_result": 0,
        "needs_attention": 0,
    }
